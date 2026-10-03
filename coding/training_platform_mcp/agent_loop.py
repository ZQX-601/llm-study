"""显式单 Agent Runtime 教学实现。

模型（这里由确定性的 ``ScriptedPolicy`` 代替）只负责提出候选 Action；Runtime 负责
工具发现、参数校验、预算、状态迁移、幂等保护、轨迹记录和停止条件。真实 MCP 接入
只需实现 ``ToolBridge``，不需要修改 ``run_agent`` 的控制流。
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, TypeAlias


class TaskStatus(str, Enum):
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    NEEDS_HUMAN = "NEEDS_HUMAN"


class StopReason(str, Enum):
    COMPLETED = "COMPLETED"
    MAX_STEPS = "MAX_STEPS"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    NEEDS_HUMAN = "NEEDS_HUMAN"
    UNRECOVERABLE_FAILURE = "UNRECOVERABLE_FAILURE"
    REPEATED_ACTION = "REPEATED_ACTION"


@dataclass(frozen=True)
class ToolAction:
    """模型提出的候选工具调用；是否允许执行仍由 Runtime 决定。"""

    tool_name: str
    arguments: dict[str, object]
    rationale: str = ""


@dataclass(frozen=True)
class FinalAction:
    """模型认为任务已经完成时提出的候选最终回答。"""

    answer: str


Action: TypeAlias = ToolAction | FinalAction


@dataclass(frozen=True)
class ToolResult:
    """Tool Bridge 返回给 Runtime 的稳定结果合同。"""

    ok: bool
    data: dict[str, object] | None = None
    error_code: str | None = None
    message: str = ""
    retryable: bool = False
    result_unknown: bool = False


@dataclass(frozen=True)
class Observation:
    """一次候选 Action 经 Runtime 处理后形成的可审计观察。"""

    step: int
    tool_name: str
    ok: bool
    data: dict[str, object] | None = None
    error_code: str | None = None
    message: str = ""
    retryable: bool = False
    result_unknown: bool = False


@dataclass(frozen=True)
class TraceEvent:
    step: int
    phase: str
    detail: dict[str, object]


@dataclass
class TaskState:
    """Runtime 的权威任务状态，不能只存在于模型自然语言上下文中。"""

    goal: str
    budget_remaining: int = 8
    required_successful_tools: frozenset[str] = field(default_factory=frozenset)
    step: int = 0
    status: TaskStatus = TaskStatus.RUNNING
    stop_reason: StopReason | None = None
    final_answer: str | None = None
    observations: list[Observation] = field(default_factory=list)
    trace: list[TraceEvent] = field(default_factory=list)
    pending_idempotency_key: str | None = None
    action_counts: dict[str, int] = field(default_factory=dict)


ArgumentValidator = Callable[[Mapping[str, object]], str | None]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    validator: ArgumentValidator
    has_side_effect: bool = False


class Policy(Protocol):
    async def decide(self, state: TaskState, tools: Sequence[ToolSpec]) -> Action:
        """根据当前状态提出下一步候选动作。"""


class ToolBridge(Protocol):
    async def list_tools(self) -> Sequence[ToolSpec]:
        """动态返回当前身份和会话可用的最小工具集合。"""

    async def call_tool(self, name: str, arguments: Mapping[str, object]) -> ToolResult:
        """调用 MCP Tool 或等价 mock，并归一化结果。"""


class VerificationDecision(str, Enum):
    CONTINUE = "CONTINUE"
    COMPLETE = "COMPLETE"
    FAIL = "FAIL"
    NEEDS_HUMAN = "NEEDS_HUMAN"


class Verifier(Protocol):
    def verify_final(self, state: TaskState, action: FinalAction) -> VerificationDecision:
        """判断候选最终回答是否满足完成条件。"""

    def verify_observation(
        self,
        state: TaskState,
        action: ToolAction,
        observation: Observation,
    ) -> VerificationDecision:
        """根据工具观察决定继续、失败或转人工。"""


class RuleBasedVerifier:
    """教学用确定性 Verifier；生产版可替换为领域规则与独立模型。"""

    def verify_final(self, state: TaskState, action: FinalAction) -> VerificationDecision:
        if state.pending_idempotency_key is not None:
            return VerificationDecision.NEEDS_HUMAN
        if not action.answer.strip():
            return VerificationDecision.FAIL
        successful_tools = {
            observation.tool_name
            for observation in state.observations
            if observation.ok
        }
        if not state.required_successful_tools.issubset(successful_tools):
            # FinalAction 只是候选答案。缺少任务合同要求的工具证据时，让 Policy
            # 回到循环补证据，而不是把一段非空文本误判为任务完成。
            return VerificationDecision.CONTINUE
        return VerificationDecision.COMPLETE

    def verify_observation(
        self,
        state: TaskState,
        action: ToolAction,
        observation: Observation,
    ) -> VerificationDecision:
        if observation.result_unknown:
            return (
                VerificationDecision.CONTINUE
                if state.pending_idempotency_key
                else VerificationDecision.NEEDS_HUMAN
            )

        if not observation.ok:
            return (
                VerificationDecision.CONTINUE
                if observation.retryable
                else VerificationDecision.FAIL
            )

        if action.tool_name == "get_cancel_operation":
            remote_status = str((observation.data or {}).get("status", "UNKNOWN"))
            if remote_status == "SUCCEEDED":
                state.pending_idempotency_key = None
            elif remote_status == "FAILED_FINAL":
                state.pending_idempotency_key = None
                return VerificationDecision.FAIL

        return VerificationDecision.CONTINUE


def require_fields(*field_names: str) -> ArgumentValidator:
    """构造最小参数校验器；真实 MCP Bridge 可改用发现到的 JSON Schema。"""

    def validate(arguments: Mapping[str, object]) -> str | None:
        missing = [name for name in field_names if name not in arguments]
        if missing:
            return f"缺少必填参数：{', '.join(missing)}"
        return None

    return validate


def _action_signature(action: ToolAction) -> str:
    encoded = json.dumps(action.arguments, ensure_ascii=False, sort_keys=True, default=repr)
    return f"{action.tool_name}:{encoded}"


def _record(state: TaskState, phase: str, **detail: object) -> None:
    state.trace.append(TraceEvent(step=state.step, phase=phase, detail=dict(detail)))


def _stop(state: TaskState, status: TaskStatus, reason: StopReason) -> None:
    state.status = status
    state.stop_reason = reason
    _record(state, "stop", status=status.value, reason=reason.value)


def _validate_action(
    state: TaskState,
    action: ToolAction,
    tools_by_name: Mapping[str, ToolSpec],
) -> str | None:
    spec = tools_by_name.get(action.tool_name)
    if spec is None:
        return f"工具 {action.tool_name!r} 不在本轮允许列表中"

    if state.pending_idempotency_key is not None:
        if action.tool_name != "get_cancel_operation":
            return "写操作结果未知时，只允许按原幂等键查询权威操作状态"
        supplied_key = action.arguments.get("idempotency_key")
        if supplied_key != state.pending_idempotency_key:
            return "状态查询必须复用原 idempotency_key"

    return spec.validator(action.arguments)


async def run_agent(
    *,
    policy: Policy,
    bridge: ToolBridge,
    state: TaskState,
    verifier: Verifier | None = None,
    max_steps: int = 8,
    repeated_action_limit: int = 2,
) -> TaskState:
    """运行显式 Agent Loop，返回包含完整轨迹的最终 ``TaskState``。

    ``repeated_action_limit=2`` 表示同一调用最多执行两次，可覆盖一次瞬时失败重试；
    第三次相同调用会在执行前停止。状态查询同样受 ``max_steps`` 的总上限保护。
    """
    if max_steps < 1 or repeated_action_limit < 1:
        raise ValueError("max_steps 和 repeated_action_limit 必须大于 0")

    active_verifier = verifier or RuleBasedVerifier()
    tools = tuple(await bridge.list_tools())
    tools_by_name = {tool.name: tool for tool in tools}
    _record(state, "tool_discovery", tools=list(tools_by_name))

    while state.status == TaskStatus.RUNNING:
        if state.step >= max_steps:
            _stop(state, TaskStatus.FAILED, StopReason.MAX_STEPS)
            break

        action = await policy.decide(state, tools)
        state.step += 1

        if isinstance(action, FinalAction):
            _record(state, "model_decision", action="final", answer=action.answer)
            decision = active_verifier.verify_final(state, action)
            _record(state, "verification", decision=decision.value)
            if decision == VerificationDecision.COMPLETE:
                state.final_answer = action.answer
                _stop(state, TaskStatus.SUCCEEDED, StopReason.COMPLETED)
            elif decision == VerificationDecision.CONTINUE:
                continue
            elif decision == VerificationDecision.NEEDS_HUMAN:
                _stop(state, TaskStatus.NEEDS_HUMAN, StopReason.NEEDS_HUMAN)
            else:
                _stop(state, TaskStatus.FAILED, StopReason.UNRECOVERABLE_FAILURE)
            break

        _record(
            state,
            "model_decision",
            action="tool",
            tool_name=action.tool_name,
            arguments=dict(action.arguments),
            rationale=action.rationale,
        )

        signature = _action_signature(action)
        state.action_counts[signature] = state.action_counts.get(signature, 0) + 1
        if state.action_counts[signature] > repeated_action_limit:
            _stop(state, TaskStatus.FAILED, StopReason.REPEATED_ACTION)
            break

        validation_error = _validate_action(state, action, tools_by_name)
        if validation_error is not None:
            observation = Observation(
                step=state.step,
                tool_name=action.tool_name,
                ok=False,
                error_code="INVALID_ACTION",
                message=validation_error,
                retryable=True,
            )
            state.observations.append(observation)
            _record(state, "runtime_validation", accepted=False, error=validation_error)
            _record(state, "observation", error_code="INVALID_ACTION", retryable=True)
            continue

        _record(state, "runtime_validation", accepted=True)
        if state.budget_remaining <= 0:
            _stop(state, TaskStatus.FAILED, StopReason.BUDGET_EXHAUSTED)
            break

        state.budget_remaining -= 1
        _record(state, "tool_dispatch", tool_name=action.tool_name)
        try:
            result = await bridge.call_tool(action.tool_name, action.arguments)
        except TimeoutError:
            result = ToolResult(
                ok=False,
                error_code="TOOL_TIMEOUT",
                message="工具调用超时。",
                retryable=True,
            )

        if result.result_unknown:
            key = action.arguments.get("idempotency_key")
            state.pending_idempotency_key = key if isinstance(key, str) and key else None

        observation = Observation(
            step=state.step,
            tool_name=action.tool_name,
            ok=result.ok,
            data=result.data,
            error_code=result.error_code,
            message=result.message,
            retryable=result.retryable,
            result_unknown=result.result_unknown,
        )
        state.observations.append(observation)
        _record(
            state,
            "observation",
            ok=observation.ok,
            error_code=observation.error_code,
            retryable=observation.retryable,
            result_unknown=observation.result_unknown,
            data=observation.data or {},
        )

        decision = active_verifier.verify_observation(state, action, observation)
        _record(state, "verification", decision=decision.value)
        if decision == VerificationDecision.FAIL:
            _stop(state, TaskStatus.FAILED, StopReason.UNRECOVERABLE_FAILURE)
        elif decision == VerificationDecision.NEEDS_HUMAN:
            _stop(state, TaskStatus.NEEDS_HUMAN, StopReason.NEEDS_HUMAN)

    return state


ToolHandler = Callable[
    [Mapping[str, object]],
    ToolResult | Awaitable[ToolResult],
]


class MockToolBridge:
    """无需网络和 API Key 的确定性 Tool Bridge，便于验证 Runtime。"""

    def __init__(self, specs: Sequence[ToolSpec], handlers: Mapping[str, ToolHandler]) -> None:
        self._specs = tuple(specs)
        self._handlers = dict(handlers)
        self.call_history: list[tuple[str, dict[str, object]]] = []

    async def list_tools(self) -> Sequence[ToolSpec]:
        return self._specs

    async def call_tool(self, name: str, arguments: Mapping[str, object]) -> ToolResult:
        self.call_history.append((name, dict(arguments)))
        handler = self._handlers[name]
        result = handler(arguments)
        if inspect.isawaitable(result):
            return await result
        return result


class ScriptedPolicy:
    """按给定顺序产出 Action，用于把 Runtime 测试与模型随机性解耦。"""

    def __init__(self, actions: Sequence[Action]) -> None:
        self._actions = list(actions)
        self._index = 0

    async def decide(self, state: TaskState, tools: Sequence[ToolSpec]) -> Action:
        if self._index >= len(self._actions):
            return FinalAction("候选动作脚本已结束。")
        action = self._actions[self._index]
        self._index += 1
        return action

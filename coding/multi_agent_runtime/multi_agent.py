"""角色隔离、最小权限与 Handoff 合同的确定性多 Agent Runtime。

本实现不用真实模型或远程 MCP。不同 Agent 拥有独立 ``AgentSpec``、输入 Context 和
Tool 权限；Runtime 是全局状态、预算、Evidence 与路由规则的唯一权威。
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol


class AgentRole(str, Enum):
    COORDINATOR = "COORDINATOR"
    DIAGNOSTIC = "DIAGNOSTIC"
    REMEDIATION = "REMEDIATION"


class Sensitivity(str, Enum):
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    RESTRICTED = "RESTRICTED"


class TaskStatus(str, Enum):
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    NEEDS_HUMAN = "NEEDS_HUMAN"


class AgentResultStatus(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    NEEDS_HUMAN = "NEEDS_HUMAN"


class FanInDecision(str, Enum):
    MERGED = "MERGED"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True)
class AgentSpec:
    """Agent 的静态身份与能力清单，由 Runtime 配置而非 Prompt 自报。"""

    agent_id: str
    role: AgentRole
    allowed_tools: frozenset[str] = field(default_factory=frozenset)
    allowed_handoff_targets: frozenset[str] = field(default_factory=frozenset)
    accepted_sensitivities: frozenset[Sensitivity] = field(
        default_factory=lambda: frozenset({Sensitivity.PUBLIC})
    )


@dataclass(frozen=True)
class Evidence:
    key: str
    value: object
    source: str
    subject_id: str | None
    owner_agent_id: str
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    version: str | None = None


@dataclass(frozen=True)
class HandoffRequest:
    """源 Agent 提出的候选委派；Runtime 校验后才会交给目标 Agent。"""

    handoff_id: str
    source_agent_id: str
    target_agent_id: str
    objective: str
    subject_id: str
    evidence_keys: tuple[str, ...]
    requested_tools: frozenset[str]
    delegated_action_budget: int
    delegated_tool_budget: int
    expected_output_evidence_keys: tuple[str, ...] = ()
    parent_handoff_id: str | None = None
    depth: int = 1


@dataclass(frozen=True)
class HandoffInput:
    handoff_id: str
    objective: str
    subject_id: str
    evidence: tuple[Evidence, ...]
    allowed_tools: frozenset[str]


@dataclass(frozen=True)
class HandoffResult:
    handoff_id: str
    target_agent_id: str
    status: AgentResultStatus
    summary: str
    evidence_keys: tuple[str, ...]
    actions_used: int
    tool_calls_used: int


@dataclass(frozen=True)
class FanInResult:
    decision: FanInDecision
    handoff_ids: tuple[str, ...]
    evidence_keys: tuple[str, ...]
    conflict_keys: tuple[str, ...] = ()


@dataclass
class GlobalBudget:
    actions_remaining: int
    tool_calls_remaining: int
    handoffs_remaining: int


@dataclass(frozen=True)
class TraceEvent:
    sequence: int
    phase: str
    agent_id: str | None
    detail: dict[str, object]


@dataclass
class MultiAgentState:
    goal: str
    subject_id: str
    budget: GlobalBudget
    deadline_at: datetime | None = None
    status: TaskStatus = TaskStatus.RUNNING
    stop_reason: str | None = None
    final_answer: str | None = None
    evidence: dict[str, Evidence] = field(default_factory=dict)
    handoff_results: dict[str, HandoffResult] = field(default_factory=dict)
    handoff_signature_counts: dict[str, int] = field(default_factory=dict)
    trace: list[TraceEvent] = field(default_factory=list)


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    data: dict[str, object] | None = None
    error_code: str | None = None
    retryable: bool = False


ArgumentValidator = Callable[[Mapping[str, object]], str | None]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    validator: ArgumentValidator
    subject_field: str
    output_sensitivity: Sensitivity = Sensitivity.INTERNAL


ToolHandler = Callable[
    [Mapping[str, object]],
    ToolResult | Awaitable[ToolResult],
]


class MockToolBackend:
    def __init__(
        self,
        specs: Sequence[ToolSpec],
        handlers: Mapping[str, ToolHandler],
    ) -> None:
        self.specs = {spec.name: spec for spec in specs}
        self.handlers = dict(handlers)
        self.call_history: list[tuple[str, dict[str, object]]] = []

    async def call_tool(
        self,
        tool_name: str,
        arguments: Mapping[str, object],
    ) -> ToolResult:
        self.call_history.append((tool_name, dict(arguments)))
        result = self.handlers[tool_name](arguments)
        if inspect.isawaitable(result):
            return await result
        return result


def require_fields(*names: str) -> ArgumentValidator:
    def validate(arguments: Mapping[str, object]) -> str | None:
        missing = [name for name in names if name not in arguments]
        return f"缺少字段：{', '.join(missing)}" if missing else None

    return validate


class RuntimePolicyError(RuntimeError):
    """Handoff、权限、Evidence 或预算违反 Runtime 硬约束。"""


class SpecialistAgent(Protocol):
    spec: AgentSpec

    async def run(
        self,
        task: HandoffInput,
        session: AgentSession,
    ) -> HandoffResult:
        """在投影后的局部 Context 与委派预算内执行。"""


class CoordinatorAgent(Protocol):
    spec: AgentSpec

    def create_handoff(self, state: MultiAgentState) -> HandoffRequest:
        """提出候选 Handoff，不直接执行目标 Agent 或 Tool。"""

    def finalize(
        self,
        state: MultiAgentState,
        result: HandoffResult,
    ) -> str:
        """只使用 Runtime 已接纳的结果与 Evidence 生成最终答案。"""


class AgentSession:
    """目标 Agent 的受限执行视图；不能访问完整全局状态。"""

    def __init__(
        self,
        runtime: MultiAgentRuntime,
        state: MultiAgentState,
        spec: AgentSpec,
        request: HandoffRequest,
    ) -> None:
        self._runtime = runtime
        self._state = state
        self.spec = spec
        self.request = request
        self.actions_used = 0
        self.tool_calls_used = 0
        self._visible_evidence_keys = set(request.evidence_keys)

    def _consume_action(self, kind: str) -> None:
        if self.actions_used >= self.request.delegated_action_budget:
            raise RuntimePolicyError("DELEGATED_ACTION_BUDGET_EXHAUSTED")
        self._runtime._consume_global_action(self._state, self.spec.agent_id, kind)
        self.actions_used += 1

    async def call_tool(
        self,
        tool_name: str,
        arguments: Mapping[str, object],
        evidence_key: str,
    ) -> ToolResult:
        """每次 Tool 候选同时消耗一个 Agent Action 和一次 Tool 预算。"""

        self._consume_action("tool")
        if tool_name not in self.request.requested_tools:
            raise RuntimePolicyError("TOOL_NOT_DELEGATED")
        if tool_name not in self.spec.allowed_tools:
            raise RuntimePolicyError("TOOL_NOT_ALLOWED_FOR_AGENT")
        if self.tool_calls_used >= self.request.delegated_tool_budget:
            raise RuntimePolicyError("DELEGATED_TOOL_BUDGET_EXHAUSTED")
        if self._state.budget.tool_calls_remaining <= 0:
            raise RuntimePolicyError("GLOBAL_TOOL_BUDGET_EXHAUSTED")

        tool = self._runtime.tool_backend.specs.get(tool_name)
        if tool is None:
            raise RuntimePolicyError("UNKNOWN_TOOL")
        validation_error = tool.validator(arguments)
        if validation_error:
            raise RuntimePolicyError(f"INVALID_TOOL_ARGUMENTS: {validation_error}")
        if str(arguments.get(tool.subject_field, "")) != self.request.subject_id:
            raise RuntimePolicyError("TOOL_SUBJECT_MISMATCH")
        if evidence_key in self._state.evidence:
            raise RuntimePolicyError("EVIDENCE_KEY_ALREADY_EXISTS")

        self._state.budget.tool_calls_remaining -= 1
        self.tool_calls_used += 1
        self._runtime._record(
            self._state,
            "tool_dispatched",
            self.spec.agent_id,
            tool_name=tool_name,
            handoff_id=self.request.handoff_id,
        )
        result = await self._runtime.tool_backend.call_tool(tool_name, arguments)
        self._runtime._record(
            self._state,
            "observation",
            self.spec.agent_id,
            tool_name=tool_name,
            ok=result.ok,
            error_code=result.error_code,
        )
        if not result.ok:
            return result

        data = result.data or {}
        if str(data.get(tool.subject_field, "")) != self.request.subject_id:
            raise RuntimePolicyError("OBSERVATION_SUBJECT_MISMATCH")
        self._state.evidence[evidence_key] = Evidence(
            key=evidence_key,
            value=data,
            source=f"tool:{tool_name}",
            subject_id=self.request.subject_id,
            owner_agent_id=self.spec.agent_id,
            sensitivity=tool.output_sensitivity,
        )
        self._visible_evidence_keys.add(evidence_key)
        self._runtime._record(
            self._state,
            "evidence_accepted",
            self.spec.agent_id,
            evidence_key=evidence_key,
            source=f"tool:{tool_name}",
        )
        return result

    def complete(
        self,
        summary: str,
        evidence_keys: Sequence[str],
    ) -> HandoffResult:
        self._consume_action("complete")
        if not summary.strip():
            raise RuntimePolicyError("EMPTY_HANDOFF_SUMMARY")
        for key in evidence_keys:
            if key not in self._visible_evidence_keys:
                raise RuntimePolicyError(
                    "HANDOFF_RESULT_REFERENCES_UNSCOPED_EVIDENCE"
                )
            evidence = self._state.evidence.get(key)
            if evidence is None:
                raise RuntimePolicyError("HANDOFF_RESULT_REFERENCES_UNKNOWN_EVIDENCE")
            if evidence.subject_id not in (None, self.request.subject_id):
                raise RuntimePolicyError("HANDOFF_RESULT_SUBJECT_MISMATCH")
        return HandoffResult(
            self.request.handoff_id,
            self.spec.agent_id,
            AgentResultStatus.SUCCEEDED,
            summary,
            tuple(evidence_keys),
            self.actions_used,
            self.tool_calls_used,
        )


class MultiAgentRuntime:
    """全局路由、Context 投影、预算与 Tool dispatch 的权威控制面。"""

    def __init__(
        self,
        specs: Sequence[AgentSpec],
        specialists: Mapping[str, SpecialistAgent],
        tool_backend: MockToolBackend,
        repeated_handoff_limit: int = 2,
        max_handoff_depth: int = 4,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.specs = {spec.agent_id: spec for spec in specs}
        self.specialists = dict(specialists)
        self.tool_backend = tool_backend
        self.repeated_handoff_limit = repeated_handoff_limit
        self.max_handoff_depth = max_handoff_depth
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        if len(self.specs) != len(tuple(specs)):
            raise ValueError("agent_id 不能重复")
        if repeated_handoff_limit < 1:
            raise ValueError("repeated_handoff_limit 必须大于 0")
        if max_handoff_depth < 1:
            raise ValueError("max_handoff_depth 必须大于 0")

    @staticmethod
    def _handoff_signature(
        state: MultiAgentState,
        request: HandoffRequest,
    ) -> str:
        """忽略可随意更换的 handoff_id，对语义相同的委派生成稳定指纹。"""

        normalized_objective = " ".join(request.objective.lower().split())
        evidence_contract = []
        for key in sorted(request.evidence_keys):
            evidence = state.evidence.get(key)
            evidence_contract.append(
                (
                    key,
                    evidence.source if evidence else "MISSING",
                    evidence.subject_id if evidence else None,
                    evidence.version if evidence else None,
                )
            )
        return repr(
            (
                request.source_agent_id,
                request.target_agent_id,
                normalized_objective,
                request.subject_id,
                tuple(sorted(request.requested_tools)),
                tuple(sorted(request.expected_output_evidence_keys)),
                tuple(evidence_contract),
            )
        )

    @staticmethod
    def _record(
        state: MultiAgentState,
        phase: str,
        agent_id: str | None,
        **detail: object,
    ) -> None:
        state.trace.append(
            TraceEvent(len(state.trace) + 1, phase, agent_id, dict(detail))
        )

    def _consume_global_action(
        self,
        state: MultiAgentState,
        agent_id: str,
        kind: str,
    ) -> None:
        self._check_deadline(state)
        if state.budget.actions_remaining <= 0:
            raise RuntimePolicyError("GLOBAL_ACTION_BUDGET_EXHAUSTED")
        state.budget.actions_remaining -= 1
        self._record(state, "agent_action", agent_id, kind=kind)

    def _check_deadline(self, state: MultiAgentState) -> None:
        if state.deadline_at is None:
            return
        if state.deadline_at.tzinfo is None:
            raise RuntimePolicyError("DEADLINE_MUST_BE_TIMEZONE_AWARE")
        if self.clock() >= state.deadline_at:
            raise RuntimePolicyError("GLOBAL_DEADLINE_EXCEEDED")

    def _validate_handoff(
        self,
        state: MultiAgentState,
        request: HandoffRequest,
        return_action_reserve: int,
    ) -> tuple[AgentSpec, tuple[Evidence, ...]]:
        self._check_deadline(state)
        source = self.specs.get(request.source_agent_id)
        target = self.specs.get(request.target_agent_id)
        if source is None or target is None:
            raise RuntimePolicyError("UNKNOWN_AGENT")
        if request.depth < 1:
            raise RuntimePolicyError("INVALID_HANDOFF_DEPTH")
        if request.depth > self.max_handoff_depth:
            raise RuntimePolicyError("MAX_HANDOFF_DEPTH")
        if request.depth == 1 and request.parent_handoff_id is not None:
            raise RuntimePolicyError("ROOT_HANDOFF_CANNOT_HAVE_PARENT")
        if (
            request.depth > 1
            and request.parent_handoff_id not in state.handoff_results
        ):
            raise RuntimePolicyError("UNKNOWN_PARENT_HANDOFF")
        if source.agent_id == target.agent_id:
            raise RuntimePolicyError("SELF_HANDOFF_NOT_ALLOWED")
        if target.agent_id not in source.allowed_handoff_targets:
            raise RuntimePolicyError("HANDOFF_ROUTE_NOT_ALLOWED")
        if request.handoff_id in state.handoff_results:
            raise RuntimePolicyError("DUPLICATE_HANDOFF_ID")
        if not request.objective.strip():
            raise RuntimePolicyError("EMPTY_HANDOFF_OBJECTIVE")
        if request.subject_id != state.subject_id:
            raise RuntimePolicyError("HANDOFF_SUBJECT_MISMATCH")
        if not request.requested_tools.issubset(target.allowed_tools):
            raise RuntimePolicyError("HANDOFF_REQUESTS_EXCESS_TOOL_PERMISSION")
        if request.delegated_action_budget < 1 or request.delegated_tool_budget < 0:
            raise RuntimePolicyError("INVALID_DELEGATED_BUDGET")
        if request.delegated_action_budget > state.budget.actions_remaining:
            raise RuntimePolicyError("DELEGATED_ACTION_BUDGET_EXCEEDS_GLOBAL")
        if return_action_reserve < 0:
            raise RuntimePolicyError("INVALID_RETURN_ACTION_RESERVE")
        if (
            request.delegated_action_budget + return_action_reserve
            > state.budget.actions_remaining
        ):
            raise RuntimePolicyError(
                "DELEGATED_ACTION_BUDGET_BREAKS_RETURN_RESERVE"
            )
        if request.delegated_tool_budget > state.budget.tool_calls_remaining:
            raise RuntimePolicyError("DELEGATED_TOOL_BUDGET_EXCEEDS_GLOBAL")
        if state.budget.handoffs_remaining <= 0:
            raise RuntimePolicyError("GLOBAL_HANDOFF_BUDGET_EXHAUSTED")
        signature = self._handoff_signature(state, request)
        if (
            state.handoff_signature_counts.get(signature, 0)
            >= self.repeated_handoff_limit
        ):
            raise RuntimePolicyError("REPEATED_HANDOFF_ROUTE")

        projected: list[Evidence] = []
        for key in request.evidence_keys:
            evidence = state.evidence.get(key)
            if evidence is None:
                raise RuntimePolicyError("UNKNOWN_HANDOFF_EVIDENCE")
            if evidence.subject_id not in (None, request.subject_id):
                raise RuntimePolicyError("HANDOFF_EVIDENCE_SUBJECT_MISMATCH")
            if evidence.sensitivity not in target.accepted_sensitivities:
                raise RuntimePolicyError("HANDOFF_EVIDENCE_SENSITIVITY_DENIED")
            projected.append(evidence)
        return target, tuple(projected)

    async def execute_handoff(
        self,
        state: MultiAgentState,
        request: HandoffRequest,
        *,
        return_action_reserve: int = 0,
    ) -> HandoffResult:
        """校验 Handoff 后，仅投影显式引用的 Evidence 给目标 Agent。"""

        self._record(
            state,
            "handoff_proposed",
            request.source_agent_id,
            handoff_id=request.handoff_id,
            target_agent_id=request.target_agent_id,
        )
        try:
            target, projected = self._validate_handoff(
                state,
                request,
                return_action_reserve,
            )
        except RuntimePolicyError as exc:
            self._record(
                state,
                "handoff_rejected",
                request.source_agent_id,
                handoff_id=request.handoff_id,
                reason=str(exc),
            )
            raise

        agent = self.specialists.get(target.agent_id)
        if agent is None or agent.spec != target:
            raise RuntimePolicyError("TARGET_AGENT_IMPLEMENTATION_MISMATCH")
        signature = self._handoff_signature(state, request)
        state.handoff_signature_counts[signature] = (
            state.handoff_signature_counts.get(signature, 0) + 1
        )
        state.budget.handoffs_remaining -= 1
        self._record(
            state,
            "handoff_started",
            target.agent_id,
            handoff_id=request.handoff_id,
            projected_evidence_keys=list(request.evidence_keys),
            allowed_tools=sorted(request.requested_tools),
        )
        task = HandoffInput(
            request.handoff_id,
            request.objective,
            request.subject_id,
            projected,
            request.requested_tools,
        )
        session = AgentSession(self, state, target, request)
        result = await agent.run(task, session)
        if result.handoff_id != request.handoff_id:
            raise RuntimePolicyError("HANDOFF_RESULT_ID_MISMATCH")
        if result.target_agent_id != target.agent_id:
            raise RuntimePolicyError("HANDOFF_RESULT_AGENT_MISMATCH")
        if result.actions_used != session.actions_used:
            raise RuntimePolicyError("HANDOFF_RESULT_ACTION_COUNT_MISMATCH")
        if result.tool_calls_used != session.tool_calls_used:
            raise RuntimePolicyError("HANDOFF_RESULT_TOOL_COUNT_MISMATCH")
        for key in result.evidence_keys:
            if key not in state.evidence:
                raise RuntimePolicyError("HANDOFF_RESULT_REFERENCES_UNKNOWN_EVIDENCE")
        if (
            result.status is AgentResultStatus.SUCCEEDED
            and request.expected_output_evidence_keys
            and set(result.evidence_keys)
            != set(request.expected_output_evidence_keys)
        ):
            raise RuntimePolicyError("HANDOFF_RESULT_OUTPUT_CONTRACT_MISMATCH")
        state.handoff_results[request.handoff_id] = result
        self._record(
            state,
            "handoff_completed",
            target.agent_id,
            handoff_id=request.handoff_id,
            status=result.status.value,
            evidence_keys=list(result.evidence_keys),
        )
        return result

    async def execute_parallel_handoffs(
        self,
        state: MultiAgentState,
        requests: Sequence[HandoffRequest],
        *,
        return_action_reserve: int = 0,
    ) -> tuple[HandoffResult, ...]:
        """原子预检后并发执行独立 Handoff，共享同一全局预算。

        这里先按各分支声明的最大委派预算做保守预留；不能假设某分支“可能少用”就把
        同一份全局额度同时借给多个 Agent。
        """

        batch = tuple(requests)
        if len(batch) < 2:
            raise RuntimePolicyError("PARALLEL_BATCH_REQUIRES_MULTIPLE_HANDOFFS")
        handoff_ids = [request.handoff_id for request in batch]
        if len(handoff_ids) != len(set(handoff_ids)):
            raise RuntimePolicyError("PARALLEL_DUPLICATE_HANDOFF_ID")

        expected_keys = [
            key
            for request in batch
            for key in request.expected_output_evidence_keys
        ]
        if any(not request.expected_output_evidence_keys for request in batch):
            raise RuntimePolicyError("PARALLEL_OUTPUT_CONTRACT_REQUIRED")
        if len(expected_keys) != len(set(expected_keys)):
            raise RuntimePolicyError("PARALLEL_EVIDENCE_KEY_CONFLICT")
        if any(key in state.evidence for key in expected_keys):
            raise RuntimePolicyError("PARALLEL_OUTPUT_OVERWRITES_EXISTING_EVIDENCE")

        # 先完成所有静态校验，再启动任何 Agent，避免一半分支已产生副作用后才发现
        # 另一半路由或敏感级别非法。
        for request in batch:
            self._validate_handoff(state, request, 0)
        if (
            sum(request.delegated_action_budget for request in batch)
            + return_action_reserve
            > state.budget.actions_remaining
        ):
            raise RuntimePolicyError("PARALLEL_ACTION_BUDGET_OVERCOMMITTED")
        if (
            sum(request.delegated_tool_budget for request in batch)
            > state.budget.tool_calls_remaining
        ):
            raise RuntimePolicyError("PARALLEL_TOOL_BUDGET_OVERCOMMITTED")
        if len(batch) > state.budget.handoffs_remaining:
            raise RuntimePolicyError("PARALLEL_HANDOFF_BUDGET_OVERCOMMITTED")

        self._record(
            state,
            "fan_out_started",
            None,
            handoff_ids=handoff_ids,
            return_action_reserve=return_action_reserve,
        )
        results = await asyncio.gather(
            *(self.execute_handoff(state, request) for request in batch)
        )
        self._record(
            state,
            "fan_out_completed",
            None,
            handoff_ids=handoff_ids,
        )
        return tuple(results)

    def fan_in(
        self,
        state: MultiAgentState,
        requests: Sequence[HandoffRequest],
        results: Sequence[HandoffResult],
    ) -> FanInResult:
        """按声明合同确定性合并；不按 Agent 完成顺序覆盖结果。"""

        request_by_id = {request.handoff_id: request for request in requests}
        result_by_id = {result.handoff_id: result for result in results}
        if set(request_by_id) != set(result_by_id):
            raise RuntimePolicyError("FAN_IN_HANDOFF_SET_MISMATCH")

        merged_keys: list[str] = []
        conflicts: list[str] = []
        for handoff_id in sorted(request_by_id):
            request = request_by_id[handoff_id]
            result = result_by_id[handoff_id]
            if result.status is not AgentResultStatus.SUCCEEDED:
                conflicts.append(f"handoff:{handoff_id}")
                continue
            expected = set(request.expected_output_evidence_keys)
            actual = set(result.evidence_keys)
            if actual != expected:
                conflicts.append(f"contract:{handoff_id}")
                continue
            if any(key not in state.evidence for key in actual):
                conflicts.append(f"missing:{handoff_id}")
                continue
            merged_keys.extend(sorted(actual))

        decision = FanInDecision.CONFLICT if conflicts else FanInDecision.MERGED
        fan_in_result = FanInResult(
            decision=decision,
            handoff_ids=tuple(sorted(request_by_id)),
            evidence_keys=tuple(merged_keys),
            conflict_keys=tuple(conflicts),
        )
        self._record(
            state,
            "fan_in",
            None,
            decision=decision.value,
            evidence_keys=list(fan_in_result.evidence_keys),
            conflict_keys=list(fan_in_result.conflict_keys),
        )
        return fan_in_result

    async def run(
        self,
        state: MultiAgentState,
        coordinator: CoordinatorAgent,
    ) -> MultiAgentState:
        """运行一次 Coordinator → Specialist → Coordinator 的最小闭环。"""

        registered = self.specs.get(coordinator.spec.agent_id)
        if registered != coordinator.spec or coordinator.spec.role is not AgentRole.COORDINATOR:
            raise RuntimePolicyError("INVALID_COORDINATOR")
        try:
            self._consume_global_action(state, coordinator.spec.agent_id, "handoff")
            request = coordinator.create_handoff(state)
            # Specialist 完成后 Coordinator 还需要一个 finalize Action。该额度由
            # Runtime 预留，不能只靠 Coordinator Prompt 自觉少委派预算。
            result = await self.execute_handoff(
                state,
                request,
                return_action_reserve=1,
            )
            if result.status is AgentResultStatus.NEEDS_HUMAN:
                state.status = TaskStatus.NEEDS_HUMAN
                state.stop_reason = "SPECIALIST_NEEDS_HUMAN"
                return state
            if result.status is not AgentResultStatus.SUCCEEDED:
                state.status = TaskStatus.FAILED
                state.stop_reason = "SPECIALIST_FAILED"
                return state
            self._consume_global_action(state, coordinator.spec.agent_id, "finalize")
            answer = coordinator.finalize(state, result)
            if not answer.strip() or not result.evidence_keys:
                raise RuntimePolicyError("FINAL_ANSWER_WITHOUT_HANDOFF_EVIDENCE")
            state.final_answer = answer
            state.status = TaskStatus.SUCCEEDED
            self._record(
                state,
                "final_answer",
                coordinator.spec.agent_id,
                handoff_id=result.handoff_id,
            )
        except RuntimePolicyError as exc:
            state.status = TaskStatus.FAILED
            state.stop_reason = str(exc)
            self._record(state, "task_stopped", None, reason=str(exc))
        return state


class DiagnosticAgent:
    """独立诊断 Agent：拥有自己的身份、局部 Context 和只读 Tool。"""

    def __init__(self, spec: AgentSpec) -> None:
        self.spec = spec
        self.last_input: HandoffInput | None = None

    async def run(
        self,
        task: HandoffInput,
        session: AgentSession,
    ) -> HandoffResult:
        self.last_input = task
        status = await session.call_tool(
            "get_training_job",
            {"job_id": task.subject_id},
            "job_status",
        )
        if not status.ok:
            return HandoffResult(
                task.handoff_id,
                self.spec.agent_id,
                AgentResultStatus.FAILED,
                status.error_code or "状态查询失败",
                (),
                session.actions_used,
                session.tool_calls_used,
            )
        logs = await session.call_tool(
            "search_job_logs",
            {"job_id": task.subject_id, "query": "error"},
            "error_log",
        )
        if not logs.ok:
            return HandoffResult(
                task.handoff_id,
                self.spec.agent_id,
                AgentResultStatus.FAILED,
                logs.error_code or "日志查询失败",
                ("job_status",),
                session.actions_used,
                session.tool_calls_used,
            )
        return session.complete(
            "训练任务失败，日志显示 CUDA OOM。",
            ("job_status", "error_log"),
        )


class SingleToolAgent:
    """并发教学用 Specialist：一个独立只读 Tool 对应一个 Evidence 输出。"""

    def __init__(
        self,
        spec: AgentSpec,
        tool_name: str,
        evidence_key: str,
        arguments: Mapping[str, object],
        summary: str,
    ) -> None:
        self.spec = spec
        self.tool_name = tool_name
        self.evidence_key = evidence_key
        self.arguments = dict(arguments)
        self.summary = summary

    async def run(
        self,
        task: HandoffInput,
        session: AgentSession,
    ) -> HandoffResult:
        arguments = dict(self.arguments)
        arguments.setdefault("job_id", task.subject_id)
        result = await session.call_tool(
            self.tool_name,
            arguments,
            self.evidence_key,
        )
        if not result.ok:
            return HandoffResult(
                task.handoff_id,
                self.spec.agent_id,
                AgentResultStatus.FAILED,
                result.error_code or "Tool 调用失败",
                (),
                session.actions_used,
                session.tool_calls_used,
            )
        return session.complete(self.summary, (self.evidence_key,))


class ScriptedCoordinator:
    """确定性 Coordinator；只路由和汇总，不拥有诊断 Tool。"""

    def __init__(self, spec: AgentSpec) -> None:
        self.spec = spec

    def create_handoff(self, state: MultiAgentState) -> HandoffRequest:
        return HandoffRequest(
            handoff_id="handoff-diagnose-1",
            source_agent_id=self.spec.agent_id,
            target_agent_id="diagnostic-agent",
            objective=f"诊断训练任务 {state.subject_id} 失败原因",
            subject_id=state.subject_id,
            evidence_keys=(),
            requested_tools=frozenset({"get_training_job", "search_job_logs"}),
            delegated_action_budget=3,
            delegated_tool_budget=2,
        )

    def finalize(
        self,
        state: MultiAgentState,
        result: HandoffResult,
    ) -> str:
        job = state.evidence[result.evidence_keys[0]].value
        log = state.evidence[result.evidence_keys[1]].value
        job_data = job if isinstance(job, dict) else {}
        log_data = log if isinstance(log, dict) else {}
        return (
            f"任务 {job_data.get('job_id')} 状态为 {job_data.get('status')}；"
            f"诊断 Agent 返回 {log_data.get('error')}。"
        )


def build_demo_runtime() -> tuple[
    MultiAgentRuntime,
    ScriptedCoordinator,
    DiagnosticAgent,
    MockToolBackend,
]:
    coordinator_spec = AgentSpec(
        "coordinator-agent",
        AgentRole.COORDINATOR,
        allowed_handoff_targets=frozenset({"diagnostic-agent"}),
        accepted_sensitivities=frozenset(
            {Sensitivity.PUBLIC, Sensitivity.INTERNAL}
        ),
    )
    diagnostic_spec = AgentSpec(
        "diagnostic-agent",
        AgentRole.DIAGNOSTIC,
        allowed_tools=frozenset({"get_training_job", "search_job_logs"}),
        accepted_sensitivities=frozenset(
            {Sensitivity.PUBLIC, Sensitivity.INTERNAL}
        ),
    )
    diagnostic = DiagnosticAgent(diagnostic_spec)
    backend = MockToolBackend(
        (
            ToolSpec(
                "get_training_job",
                "查询训练任务状态",
                require_fields("job_id"),
                subject_field="job_id",
            ),
            ToolSpec(
                "search_job_logs",
                "查询训练日志",
                require_fields("job_id", "query"),
                subject_field="job_id",
                output_sensitivity=Sensitivity.INTERNAL,
            ),
        ),
        {
            "get_training_job": lambda args: ToolResult(
                True,
                {"job_id": args["job_id"], "status": "FAILED"},
            ),
            "search_job_logs": lambda args: ToolResult(
                True,
                {"job_id": args["job_id"], "error": "CUDA out of memory"},
            ),
        },
    )
    runtime = MultiAgentRuntime(
        (coordinator_spec, diagnostic_spec),
        {diagnostic_spec.agent_id: diagnostic},
        backend,
    )
    return runtime, ScriptedCoordinator(coordinator_spec), diagnostic, backend


def new_demo_state() -> MultiAgentState:
    return MultiAgentState(
        goal="诊断 train-42 失败原因",
        subject_id="train-42",
        budget=GlobalBudget(
            actions_remaining=5,
            tool_calls_remaining=2,
            handoffs_remaining=1,
        ),
    )

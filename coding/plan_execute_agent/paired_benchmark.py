"""ReAct 与 Plan-and-Execute 的确定性成对基准。

两种模式共享任务、工具语义、权限和整次任务预算；适配器只负责把同一 Tool 结果
转换为各 Runtime 的既有合同。这里不用单一总分，因为安全/正确性是硬门槛，Action、
Tool、Replan 和时延是通过硬门槛后才比较的效率指标。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from time import perf_counter_ns
from collections.abc import Mapping, Sequence

try:
    from .plan_execute import (
        CompleteStepAction,
        EvidencePlanVerifier,
        EvidenceRequirement,
        ExecutionPlan,
        MCPResult,
        MCPToolSpec,
        MockMCPClient,
        PlanRunState,
        PlanStatus,
        PlanStep,
        ScriptedPlanner,
        ScriptedStepExecutor,
        SkillRegistry,
        ToolAction as PlanToolAction,
        require_fields as require_mcp_fields,
        run_plan_execute,
    )
except ImportError:
    from plan_execute import (
        CompleteStepAction,
        EvidencePlanVerifier,
        EvidenceRequirement,
        ExecutionPlan,
        MCPResult,
        MCPToolSpec,
        MockMCPClient,
        PlanRunState,
        PlanStatus,
        PlanStep,
        ScriptedPlanner,
        ScriptedStepExecutor,
        SkillRegistry,
        ToolAction as PlanToolAction,
        require_fields as require_mcp_fields,
        run_plan_execute,
    )

try:
    from coding.training_platform_mcp.agent_loop import (
        FinalAction,
        MockToolBridge,
        ScriptedPolicy,
        TaskState,
        TaskStatus,
        ToolAction as ReActToolAction,
        ToolResult,
        ToolSpec,
        require_fields as require_tool_fields,
        run_agent,
    )
except ModuleNotFoundError:  # 支持从本目录直接执行脚本。
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from coding.training_platform_mcp.agent_loop import (
        FinalAction,
        MockToolBridge,
        ScriptedPolicy,
        TaskState,
        TaskStatus,
        ToolAction as ReActToolAction,
        ToolResult,
        ToolSpec,
        require_fields as require_tool_fields,
        run_agent,
    )


class BenchmarkMode(str, Enum):
    REACT = "react"
    PLAN_EXECUTE = "plan_execute"


class ScenarioKind(str, Enum):
    SIMPLE_QUERY = "simple_query"
    MULTI_STEP_DIAGNOSIS = "multi_step_diagnosis"
    RECOVERABLE_TIMEOUT = "recoverable_timeout"
    TARGET_CHANGE = "target_change"


@dataclass(frozen=True)
class BenchmarkBudget:
    """整次任务共享的资源上限，不随 Plan Step 数增长。"""

    max_actions: int = 8
    max_tool_calls: int = 4
    max_replans: int = 1
    repeated_action_limit: int = 2


@dataclass(frozen=True)
class ExpectedEvidence:
    tool_name: str
    subject_id: str


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    scenario: ScenarioKind
    goal: str
    expected_answer_terms: tuple[str, ...]
    required_evidence: tuple[ExpectedEvidence, ...]
    budget: BenchmarkBudget = field(default_factory=BenchmarkBudget)


@dataclass(frozen=True)
class ModeRunMetrics:
    mode: BenchmarkMode
    status: str
    stop_reason: str | None
    answer_correct: bool
    evidence_complete: bool
    trajectory_valid: bool
    safety_violation: bool
    task_success: bool
    actions: int
    tool_calls: int
    replans: int
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    monetary_cost: float | None = None


@dataclass(frozen=True)
class PairedResult:
    case: BenchmarkCase
    react: ModeRunMetrics
    plan_execute: ModeRunMetrics


@dataclass(frozen=True)
class CanonicalToolOutcome:
    ok: bool
    data: dict[str, object] | None = None
    error_code: str | None = None
    retryable: bool = False
    replan_reason: str | None = None


class OrderFulfillmentBackend:
    """两种 Runtime 共用的权威 Tool 语义；每次 mode run 获得独立同构实例。"""

    def __init__(self, scenario: ScenarioKind) -> None:
        self.scenario = scenario
        self.attempts: dict[tuple[str, str], int] = {}

    def call(
        self,
        tool_name: str,
        arguments: Mapping[str, object],
    ) -> CanonicalToolOutcome:
        order_id = str(arguments.get("order_id", ""))
        key = (tool_name, order_id)
        self.attempts[key] = self.attempts.get(key, 0) + 1

        if tool_name == "get_order":
            if self.scenario is ScenarioKind.SIMPLE_QUERY and order_id == "order-100":
                return self._order(order_id, "SHIPPED")
            if (
                self.scenario is ScenarioKind.MULTI_STEP_DIAGNOSIS
                and order_id == "order-200"
            ):
                return self._order(order_id, "DELAYED")
            if (
                self.scenario is ScenarioKind.RECOVERABLE_TIMEOUT
                and order_id == "order-300"
            ):
                if self.attempts[key] == 1:
                    return CanonicalToolOutcome(
                        False,
                        error_code="TOOL_TIMEOUT",
                        retryable=True,
                    )
                return self._order(order_id, "PROCESSING")
            if self.scenario is ScenarioKind.TARGET_CHANGE:
                if order_id == "order-400":
                    return CanonicalToolOutcome(
                        True,
                        {
                            "order_id": order_id,
                            "job_id": order_id,
                            "status": "REPLACED",
                            "replacement_order_id": "order-401",
                            "version": 1,
                        },
                        replan_reason="原订单已替换为 order-401",
                    )
                if order_id == "order-401":
                    return self._order(order_id, "DELIVERED")

        if (
            tool_name == "get_delivery_events"
            and self.scenario is ScenarioKind.MULTI_STEP_DIAGNOSIS
            and order_id == "order-200"
        ):
            return CanonicalToolOutcome(
                True,
                {
                    "order_id": order_id,
                    "job_id": order_id,
                    "delay_reason": "WEATHER",
                    "version": 1,
                },
            )
        return CanonicalToolOutcome(
            False,
            error_code="NOT_FOUND",
            retryable=False,
        )

    @staticmethod
    def _order(order_id: str, status: str) -> CanonicalToolOutcome:
        return CanonicalToolOutcome(
            True,
            {
                "order_id": order_id,
                # 当前 Plan Runtime 的 Evidence subject 从 job_id 取值；同时保留领域字段。
                "job_id": order_id,
                "status": status,
                "version": 1,
            },
        )


class BenchmarkFinalWriter:
    """只在全部目标 Evidence 已被 Runtime 验证后生成确定性答案。"""

    def __init__(self, case: BenchmarkCase) -> None:
        self.case = case

    def write(self, state: PlanRunState) -> str:
        present = {
            (
                evidence.source.removeprefix("mcp:"),
                evidence.subject_id,
            )
            for evidence in state.evidence.values()
        }
        required = {
            (item.tool_name, item.subject_id) for item in self.case.required_evidence
        }
        if not required.issubset(present):
            raise RuntimeError("Final Writer 缺少经过验证的订单 Evidence")
        return "；".join(self.case.expected_answer_terms)


def default_cases() -> tuple[BenchmarkCase, ...]:
    return (
        BenchmarkCase(
            "单工具查询",
            ScenarioKind.SIMPLE_QUERY,
            "查询 order-100 当前状态",
            ("order-100", "SHIPPED"),
            (ExpectedEvidence("get_order", "order-100"),),
        ),
        BenchmarkCase(
            "多步延迟诊断",
            ScenarioKind.MULTI_STEP_DIAGNOSIS,
            "诊断 order-200 延迟原因",
            ("order-200", "DELAYED", "WEATHER"),
            (
                ExpectedEvidence("get_order", "order-200"),
                ExpectedEvidence("get_delivery_events", "order-200"),
            ),
        ),
        BenchmarkCase(
            "可恢复超时",
            ScenarioKind.RECOVERABLE_TIMEOUT,
            "查询 order-300，瞬时超时后允许重试",
            ("order-300", "PROCESSING"),
            (ExpectedEvidence("get_order", "order-300"),),
        ),
        BenchmarkCase(
            "目标订单变化",
            ScenarioKind.TARGET_CHANGE,
            "查询 order-400 的有效履约目标",
            ("order-401", "DELIVERED"),
            (ExpectedEvidence("get_order", "order-401"),),
        ),
    )


def _react_actions(case: BenchmarkCase) -> list[ReActToolAction | FinalAction]:
    if case.scenario is ScenarioKind.SIMPLE_QUERY:
        tools = [ReActToolAction("get_order", {"order_id": "order-100"})]
    elif case.scenario is ScenarioKind.MULTI_STEP_DIAGNOSIS:
        tools = [
            ReActToolAction("get_order", {"order_id": "order-200"}),
            ReActToolAction(
                "get_delivery_events", {"order_id": "order-200"}
            ),
        ]
    elif case.scenario is ScenarioKind.RECOVERABLE_TIMEOUT:
        tools = [
            ReActToolAction("get_order", {"order_id": "order-300"}),
            ReActToolAction("get_order", {"order_id": "order-300"}),
        ]
    else:
        tools = [
            ReActToolAction("get_order", {"order_id": "order-400"}),
            ReActToolAction("get_order", {"order_id": "order-401"}),
        ]
    return [*tools, FinalAction("；".join(case.expected_answer_terms))]


def _plan_parts(
    case: BenchmarkCase,
) -> tuple[ExecutionPlan, Sequence[ExecutionPlan], dict[str, list[object]]]:
    def requirement(key: str, tool: str, subject: str) -> EvidenceRequirement:
        return EvidenceRequirement(
            key,
            frozenset({f"mcp:{tool}"}),
            subject_id=subject,
        )

    if case.scenario is ScenarioKind.SIMPLE_QUERY:
        step = PlanStep(
            "query-order-100",
            "查询订单状态",
            allowed_tools=frozenset({"get_order"}),
            required_evidence=frozenset({"order_status"}),
            evidence_requirements=(
                requirement("order_status", "get_order", "order-100"),
            ),
        )
        plan = ExecutionPlan("pair-simple", 1, case.goal, [step])
        actions = {
            step.step_id: [
                PlanToolAction("get_order", {"order_id": "order-100"}, "order_status"),
                CompleteStepAction("订单状态已验证"),
            ]
        }
        return plan, (), actions

    if case.scenario is ScenarioKind.MULTI_STEP_DIAGNOSIS:
        status_step = PlanStep(
            "query-order-200",
            "查询订单状态",
            allowed_tools=frozenset({"get_order"}),
            required_evidence=frozenset({"order_status"}),
            evidence_requirements=(
                requirement("order_status", "get_order", "order-200"),
            ),
        )
        event_step = PlanStep(
            "query-events-200",
            "查询履约事件",
            dependencies=(status_step.step_id,),
            allowed_tools=frozenset({"get_delivery_events"}),
            required_evidence=frozenset({"delivery_events"}),
            evidence_requirements=(
                requirement(
                    "delivery_events", "get_delivery_events", "order-200"
                ),
            ),
        )
        plan = ExecutionPlan(
            "pair-multi", 1, case.goal, [status_step, event_step]
        )
        actions = {
            status_step.step_id: [
                PlanToolAction("get_order", {"order_id": "order-200"}, "order_status"),
                CompleteStepAction("状态已验证"),
            ],
            event_step.step_id: [
                PlanToolAction(
                    "get_delivery_events",
                    {"order_id": "order-200"},
                    "delivery_events",
                ),
                CompleteStepAction("延迟原因已验证"),
            ],
        }
        return plan, (), actions

    if case.scenario is ScenarioKind.RECOVERABLE_TIMEOUT:
        step = PlanStep(
            "retry-order-300",
            "超时后重试订单查询",
            allowed_tools=frozenset({"get_order"}),
            required_evidence=frozenset({"order_status"}),
            evidence_requirements=(
                requirement("order_status", "get_order", "order-300"),
            ),
        )
        plan = ExecutionPlan("pair-timeout", 1, case.goal, [step])
        repeated = PlanToolAction(
            "get_order", {"order_id": "order-300"}, "order_status"
        )
        return plan, (), {
            step.step_id: [repeated, repeated, CompleteStepAction("重试成功")]
        }

    old_step = PlanStep(
        "query-order-400",
        "查询原订单",
        allowed_tools=frozenset({"get_order"}),
        required_evidence=frozenset({"old_order_status"}),
        evidence_requirements=(
            requirement("old_order_status", "get_order", "order-400"),
        ),
    )
    initial = ExecutionPlan("pair-target-change", 1, case.goal, [old_step])
    new_step = PlanStep(
        "query-order-401",
        "查询替换后的有效订单",
        allowed_tools=frozenset({"get_order"}),
        required_evidence=frozenset({"order_status"}),
        evidence_requirements=(
            requirement("order_status", "get_order", "order-401"),
        ),
    )
    revised = ExecutionPlan("ignored-by-runtime", 1, case.goal, [new_step])
    return initial, (revised,), {
        old_step.step_id: [
            PlanToolAction(
                "get_order", {"order_id": "order-400"}, "old_order_status"
            )
        ],
        new_step.step_id: [
            PlanToolAction("get_order", {"order_id": "order-401"}, "order_status"),
            CompleteStepAction("替换订单已验证"),
        ],
    }


def _answer_correct(case: BenchmarkCase, answer: str | None) -> bool:
    return bool(answer) and all(term in answer for term in case.expected_answer_terms)


def _required_pairs(case: BenchmarkCase) -> set[tuple[str, str]]:
    return {(item.tool_name, item.subject_id) for item in case.required_evidence}


async def _run_react(case: BenchmarkCase) -> ModeRunMetrics:
    backend = OrderFulfillmentBackend(case.scenario)

    def adapt(tool_name: str):
        def handler(arguments: Mapping[str, object]) -> ToolResult:
            outcome = backend.call(tool_name, arguments)
            return ToolResult(
                outcome.ok,
                outcome.data,
                outcome.error_code,
                retryable=outcome.retryable,
            )

        return handler

    specs = (
        ToolSpec("get_order", "查询订单", require_tool_fields("order_id")),
        ToolSpec(
            "get_delivery_events",
            "查询履约事件",
            require_tool_fields("order_id"),
        ),
    )
    bridge = MockToolBridge(
        specs,
        {spec.name: adapt(spec.name) for spec in specs},
    )
    state = TaskState(
        goal=case.goal,
        budget_remaining=case.budget.max_tool_calls,
        required_successful_tools=frozenset(
            item.tool_name for item in case.required_evidence
        ),
    )
    start = perf_counter_ns()
    state = await run_agent(
        policy=ScriptedPolicy(_react_actions(case)),
        bridge=bridge,
        state=state,
        max_steps=case.budget.max_actions,
        repeated_action_limit=case.budget.repeated_action_limit,
    )
    latency_ms = (perf_counter_ns() - start) / 1_000_000
    observed = {
        (observation.tool_name, str((observation.data or {}).get("order_id", "")))
        for observation in state.observations
        if observation.ok
    }
    phases = [event.phase for event in state.trace]
    actions = phases.count("model_decision")
    tool_calls = phases.count("tool_dispatch")
    evidence_complete = _required_pairs(case).issubset(observed)
    trajectory_valid = (
        actions <= case.budget.max_actions
        and tool_calls <= case.budget.max_tool_calls
    )
    answer_correct = _answer_correct(case, state.final_answer)
    safety_violation = False
    return ModeRunMetrics(
        BenchmarkMode.REACT,
        state.status.value,
        state.stop_reason.value if state.stop_reason else None,
        answer_correct,
        evidence_complete,
        trajectory_valid,
        safety_violation,
        state.status is TaskStatus.SUCCEEDED
        and answer_correct
        and evidence_complete
        and trajectory_valid
        and not safety_violation,
        actions,
        tool_calls,
        0,
        latency_ms,
    )


async def _run_plan_execute(case: BenchmarkCase) -> ModeRunMetrics:
    backend = OrderFulfillmentBackend(case.scenario)

    def adapt(tool_name: str):
        def handler(arguments: Mapping[str, object]) -> MCPResult:
            outcome = backend.call(tool_name, arguments)
            return MCPResult(
                outcome.ok,
                outcome.data,
                outcome.error_code,
                retryable=outcome.retryable,
                replan_reason=outcome.replan_reason,
            )

        return handler

    specs = (
        MCPToolSpec("get_order", "查询订单", require_mcp_fields("order_id")),
        MCPToolSpec(
            "get_delivery_events",
            "查询履约事件",
            require_mcp_fields("order_id"),
        ),
    )
    mcp = MockMCPClient(specs, {spec.name: adapt(spec.name) for spec in specs})
    initial, replans, actions = _plan_parts(case)
    start = perf_counter_ns()
    state = await run_plan_execute(
        goal=case.goal,
        planner=ScriptedPlanner(initial, replans),
        executor=ScriptedStepExecutor(actions),
        verifier=EvidencePlanVerifier(),
        final_writer=BenchmarkFinalWriter(case),
        skill_registry=SkillRegistry(Path(__file__).parent / "skills"),
        mcp_client=mcp,
        max_actions_per_step=case.budget.max_actions,
        max_tool_calls_per_step=case.budget.max_tool_calls,
        max_replans=case.budget.max_replans,
        max_total_actions=case.budget.max_actions,
        max_total_tool_calls=case.budget.max_tool_calls,
    )
    latency_ms = (perf_counter_ns() - start) / 1_000_000
    phases = [event.phase for event in state.trace]
    actions_count = phases.count("executor_decision")
    tool_calls = phases.count("mcp_tool_called")
    replans_count = phases.count("plan_revised")
    observed = {
        (evidence.source.removeprefix("mcp:"), evidence.subject_id or "")
        for evidence in state.evidence.values()
    }
    evidence_complete = _required_pairs(case).issubset(observed)
    trajectory_valid = (
        actions_count <= case.budget.max_actions
        and tool_calls <= case.budget.max_tool_calls
        and replans_count <= case.budget.max_replans
    )
    answer_correct = _answer_correct(case, state.final_answer)
    safety_violation = False
    status = state.plan.status if state.plan else PlanStatus.FAILED
    return ModeRunMetrics(
        BenchmarkMode.PLAN_EXECUTE,
        status.value,
        state.stop_reason,
        answer_correct,
        evidence_complete,
        trajectory_valid,
        safety_violation,
        status is PlanStatus.COMPLETED
        and answer_correct
        and evidence_complete
        and trajectory_valid
        and not safety_violation,
        actions_count,
        tool_calls,
        replans_count,
        latency_ms,
    )


async def run_paired_benchmark(
    cases: Sequence[BenchmarkCase] | None = None,
) -> tuple[PairedResult, ...]:
    selected = tuple(cases) if cases is not None else default_cases()
    if not selected:
        raise ValueError("成对基准至少需要一个 Case")
    results: list[PairedResult] = []
    for case in selected:
        react = await _run_react(case)
        plan_execute = await _run_plan_execute(case)
        results.append(PairedResult(case, react, plan_execute))
    return tuple(results)


def render_report(results: Sequence[PairedResult]) -> str:
    """报告将硬门槛与效率指标分列，缺失遥测保留为 ``None``。"""

    lines = [
        "case | mode | success | evidence | safe | actions | tools | replans | latency_ms | tokens | cost",
        "--- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---",
    ]
    for result in results:
        for metrics in (result.react, result.plan_execute):
            lines.append(
                " | ".join(
                    (
                        result.case.name,
                        metrics.mode.value,
                        str(metrics.task_success),
                        str(metrics.evidence_complete),
                        str(not metrics.safety_violation),
                        str(metrics.actions),
                        str(metrics.tool_calls),
                        str(metrics.replans),
                        f"{metrics.latency_ms:.3f}",
                        str(metrics.input_tokens),
                        str(metrics.monetary_cost),
                    )
                )
            )
    return "\n".join(lines)


async def main() -> None:
    print(render_report(await run_paired_benchmark()))


if __name__ == "__main__":
    asyncio.run(main())

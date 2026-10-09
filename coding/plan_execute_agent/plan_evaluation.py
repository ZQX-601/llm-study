"""Plan-and-Execute Trace 的最小离线评测适配器。

该模块不修改 Runtime 状态，只从 ``PlanRunState`` 重算硬门槛和轨迹指标。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

try:
    from .plan_execute import PlanRunState, PlanStatus
except ImportError:
    from plan_execute import PlanRunState, PlanStatus


AnswerValidator = Callable[[PlanRunState], bool]


@dataclass(frozen=True)
class PlanEvalCase:
    name: str
    expected_status: PlanStatus
    expected_stop_reason: str | None
    answer_validator: AnswerValidator
    forbidden_dispatched_tools: frozenset[str] = field(default_factory=frozenset)
    max_actions: int = 20
    max_tool_calls: int = 8
    max_replans: int = 1
    expect_answer_correct: bool = True
    expect_evidence_complete: bool = True
    expect_safety_violation: bool = False
    expect_trajectory_valid: bool = True


@dataclass(frozen=True)
class PlanEvalResult:
    case_name: str
    answer_correct: bool
    evidence_complete: bool
    safety_violation: bool
    trajectory_valid: bool
    task_success: bool
    case_passed: bool
    actions: int
    tool_calls: int
    completed_steps: int
    retry_decisions: int
    replans: int
    skill_activations: int
    resource_loads: int


def _evidence_complete(state: PlanRunState) -> bool:
    """独立重算计划所有 Step 的 Evidence 合同，不盲信 COMPLETED 标志。"""

    if state.plan is None:
        return False
    for step in state.plan.steps:
        if not step.required_evidence.issubset(state.evidence):
            return False
        if any(
            not requirement.matches(state.evidence.get(requirement.key))
            for requirement in step.evidence_requirements
        ):
            return False
    return True


def evaluate_plan_state(case: PlanEvalCase, state: PlanRunState) -> PlanEvalResult:
    """从结构化状态和 Trace 计算单条 Plan-and-Execute 结果。"""

    if case.max_actions < 0 or case.max_tool_calls < 0 or case.max_replans < 0:
        raise ValueError("评测预算不能为负数")

    phases = [event.phase for event in state.trace]
    dispatched_tools = [
        str(event.detail["tool_name"])
        for event in state.trace
        if event.phase == "mcp_tool_called" and "tool_name" in event.detail
    ]
    actions = phases.count("executor_decision")
    tool_calls = len(dispatched_tools)
    completed_steps = sum(
        1
        for event in state.trace
        if event.phase == "step_verified"
        and event.detail.get("decision") == "COMPLETE_STEP"
    )
    retry_decisions = sum(
        1
        for event in state.trace
        if event.phase == "step_verified"
        and event.detail.get("decision") == "RETRY_STEP"
    )
    replans = phases.count("plan_revised")
    answer_correct = case.answer_validator(state)
    evidence_complete = _evidence_complete(state)
    safety_violation = bool(
        case.forbidden_dispatched_tools.intersection(dispatched_tools)
    )
    trajectory_valid = (
        actions <= case.max_actions
        and tool_calls <= case.max_tool_calls
        and replans <= case.max_replans
    )
    status = state.plan.status if state.plan is not None else None
    task_success = (
        status is PlanStatus.COMPLETED
        and answer_correct
        and evidence_complete
        and not safety_violation
        and trajectory_valid
    )
    case_passed = (
        status is case.expected_status
        and state.stop_reason == case.expected_stop_reason
        and answer_correct is case.expect_answer_correct
        and evidence_complete is case.expect_evidence_complete
        and safety_violation is case.expect_safety_violation
        and trajectory_valid is case.expect_trajectory_valid
    )
    return PlanEvalResult(
        case_name=case.name,
        answer_correct=answer_correct,
        evidence_complete=evidence_complete,
        safety_violation=safety_violation,
        trajectory_valid=trajectory_valid,
        task_success=task_success,
        case_passed=case_passed,
        actions=actions,
        tool_calls=tool_calls,
        completed_steps=completed_steps,
        retry_decisions=retry_decisions,
        replans=replans,
        skill_activations=phases.count("skill_activated"),
        resource_loads=phases.count("skill_resource_loaded"),
    )

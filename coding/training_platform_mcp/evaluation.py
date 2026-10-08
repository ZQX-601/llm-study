"""单 Agent Runtime 的最小离线评测合同。

本模块刻意区分两个概念：

1. ``task_success`` 表示用户目标是否被正确、有证据且安全地完成；
2. ``case_passed`` 表示本次运行是否符合评测样例预先声明的预期。

例如安全回归样例期望 Runtime 把未确认写操作转为人工处理，此时任务本身没有完成，
但安全测试仍然可以通过。评测只消费 ``TaskState`` 与 Trace，不修改 Agent Runtime。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from math import ceil

from agent_loop import StopReason, TaskState, TaskStatus


AnswerValidator = Callable[[TaskState], bool]


class FailureLayer(str, Enum):
    """用于聚合诊断的粗粒度失败层；生产系统应结合领域 Trace 进一步细分。"""

    POLICY = "POLICY"
    RUNTIME = "RUNTIME"
    TOOL = "TOOL"
    EVIDENCE = "EVIDENCE"
    FINAL_ACTION = "FINAL_ACTION"
    SAFETY = "SAFETY"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class EvalCase:
    """一条确定性回归样例及其预期。

    ``required_successful_tools`` 描述完成结论所需的最小权威工具证据；
    ``forbidden_dispatched_tools`` 描述绝不能真正发送给 Tool Bridge 的动作。
    模型提出后被 Runtime 拦截不算安全违规，真正 dispatch 才算。
    """

    name: str
    expected_status: TaskStatus
    expected_stop_reason: StopReason | None
    answer_validator: AnswerValidator
    required_successful_tools: frozenset[str] = field(default_factory=frozenset)
    forbidden_dispatched_tools: frozenset[str] = field(default_factory=frozenset)
    max_steps: int = 8
    max_tool_calls: int = 8
    expect_answer_correct: bool = True
    expect_evidence_complete: bool = True
    expect_safety_violation: bool = False
    expect_trajectory_valid: bool = True


@dataclass(frozen=True)
class RunTelemetry:
    """一次运行的效率观测；缺失遥测必须用 ``None``，不能伪装成零成本。"""

    latency_ms: float
    input_tokens: int
    output_tokens: int
    monetary_cost: float

    def validate(self) -> None:
        values = (
            self.latency_ms,
            self.input_tokens,
            self.output_tokens,
            self.monetary_cost,
        )
        if any(value < 0 for value in values):
            raise ValueError("延迟、token 和费用不能为负数")


@dataclass(frozen=True)
class EvalResult:
    """单条样例的分层结果，保留诊断维度而不是压成一个加权总分。"""

    case_name: str
    answer_correct: bool
    evidence_complete: bool
    safety_violation: bool
    trajectory_valid: bool
    task_success: bool
    case_passed: bool
    status_matches: bool
    stop_reason_matches: bool
    steps: int
    tool_calls: int
    failure_layer: FailureLayer | None
    latency_ms: float | None
    total_tokens: int | None
    monetary_cost: float | None


@dataclass(frozen=True)
class EvalSummary:
    """一批样例的宏观指标；计数分母始终是 ``case_count``。"""

    case_count: int
    case_pass_rate: float
    task_success_rate: float
    answer_accuracy: float
    evidence_complete_rate: float
    safety_violation_rate: float
    trajectory_valid_rate: float
    average_steps: float
    average_tool_calls: float
    telemetry_coverage_rate: float
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    successful_latency_p50_ms: float | None
    successful_latency_p95_ms: float | None
    average_total_tokens: float | None
    average_monetary_cost: float | None
    failure_layer_counts: dict[str, int]


def _dispatched_tools(state: TaskState) -> list[str]:
    """从 Trace 读取真实 dispatch；模型只提出但被拦截的工具不计入。"""

    return [
        str(event.detail["tool_name"])
        for event in state.trace
        if event.phase == "tool_dispatch" and "tool_name" in event.detail
    ]


def _infer_failure_layer(
    *,
    state: TaskState,
    answer_correct: bool,
    evidence_complete: bool,
    safety_violation: bool,
    trajectory_valid: bool,
    status_matches: bool,
    stop_reason_matches: bool,
) -> FailureLayer | None:
    """按最早可观察证据给出教学用归因。

    这是确定性启发式，不冒充生产根因分析。真实系统还需记录可信身份、请求主体、
    工具版本和 Verifier 判定依据，才能把 Runtime 与后端 Tool 的责任继续拆开。
    """

    if safety_violation:
        return FailureLayer.SAFETY

    invalid_action_seen = any(
        observation.error_code == "INVALID_ACTION" for observation in state.observations
    )
    if invalid_action_seen:
        return FailureLayer.POLICY

    tool_failure_seen = any(
        not observation.ok and observation.error_code != "INVALID_ACTION"
        for observation in state.observations
    )
    if tool_failure_seen:
        return FailureLayer.TOOL

    if not evidence_complete:
        return FailureLayer.EVIDENCE
    if not answer_correct:
        return FailureLayer.FINAL_ACTION
    if not trajectory_valid or not status_matches or not stop_reason_matches:
        return FailureLayer.RUNTIME
    return None


def evaluate_state(
    case: EvalCase,
    state: TaskState,
    telemetry: RunTelemetry | None = None,
) -> EvalResult:
    """根据一条完成后的 ``TaskState`` 计算分层评测结果。"""

    if case.max_steps < 0 or case.max_tool_calls < 0:
        raise ValueError("max_steps 和 max_tool_calls 不能为负数")
    if telemetry is not None:
        telemetry.validate()

    dispatched_tools = _dispatched_tools(state)
    successful_tools = {
        observation.tool_name for observation in state.observations if observation.ok
    }

    answer_correct = case.answer_validator(state)
    evidence_complete = case.required_successful_tools.issubset(successful_tools)
    safety_violation = bool(
        case.forbidden_dispatched_tools.intersection(dispatched_tools)
    )
    trajectory_valid = (
        state.step <= case.max_steps and len(dispatched_tools) <= case.max_tool_calls
    )
    status_matches = state.status is case.expected_status
    stop_reason_matches = state.stop_reason is case.expected_stop_reason

    # 四个布尔变量已经是条件，不要写成 ``answer_correct = true``。轨迹合法性也是
    # 当前教学合同的硬门槛，不能因为最终文本碰巧正确就忽略失控循环或预算违规。
    task_success = (
        state.status is TaskStatus.SUCCEEDED
        and answer_correct
        and evidence_complete
        and not safety_violation
        and trajectory_valid
    )

    case_passed = (
        status_matches
        and stop_reason_matches
        and answer_correct is case.expect_answer_correct
        and evidence_complete is case.expect_evidence_complete
        and safety_violation is case.expect_safety_violation
        and trajectory_valid is case.expect_trajectory_valid
    )

    failure_layer = _infer_failure_layer(
        state=state,
        answer_correct=answer_correct,
        evidence_complete=evidence_complete,
        safety_violation=safety_violation,
        trajectory_valid=trajectory_valid,
        status_matches=status_matches,
        stop_reason_matches=stop_reason_matches,
    )
    return EvalResult(
        case_name=case.name,
        answer_correct=answer_correct,
        evidence_complete=evidence_complete,
        safety_violation=safety_violation,
        trajectory_valid=trajectory_valid,
        task_success=task_success,
        case_passed=case_passed,
        status_matches=status_matches,
        stop_reason_matches=stop_reason_matches,
        steps=state.step,
        tool_calls=len(dispatched_tools),
        failure_layer=failure_layer,
        latency_ms=telemetry.latency_ms if telemetry is not None else None,
        total_tokens=(
            telemetry.input_tokens + telemetry.output_tokens
            if telemetry is not None
            else None
        ),
        monetary_cost=telemetry.monetary_cost if telemetry is not None else None,
    )


def _nearest_rank(values: Iterable[float], percentile: float) -> float | None:
    """计算离线小样本也可复现的 nearest-rank 分位数。"""

    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(1, ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _mean_or_none(values: Iterable[float]) -> float | None:
    items = tuple(values)
    return sum(items) / len(items) if items else None


def aggregate_results(results: Iterable[EvalResult]) -> EvalSummary:
    """聚合一批结果；空集合直接报错，避免生成看似有效的 0 分报表。"""

    items = tuple(results)
    if not items:
        raise ValueError("至少需要一条 EvalResult 才能聚合")

    count = len(items)
    failure_counts = Counter(
        result.failure_layer.value
        for result in items
        if result.failure_layer is not None
    )

    def rate(predicate: Callable[[EvalResult], bool]) -> float:
        return sum(1 for result in items if predicate(result)) / count

    observed = tuple(result for result in items if result.latency_ms is not None)
    successful_observed = tuple(result for result in observed if result.task_success)

    return EvalSummary(
        case_count=count,
        case_pass_rate=rate(lambda result: result.case_passed),
        task_success_rate=rate(lambda result: result.task_success),
        answer_accuracy=rate(lambda result: result.answer_correct),
        evidence_complete_rate=rate(lambda result: result.evidence_complete),
        safety_violation_rate=rate(lambda result: result.safety_violation),
        trajectory_valid_rate=rate(lambda result: result.trajectory_valid),
        average_steps=sum(result.steps for result in items) / count,
        average_tool_calls=sum(result.tool_calls for result in items) / count,
        telemetry_coverage_rate=len(observed) / count,
        latency_p50_ms=_nearest_rank(
            (
                result.latency_ms
                for result in observed
                if result.latency_ms is not None
            ),
            0.50,
        ),
        latency_p95_ms=_nearest_rank(
            (
                result.latency_ms
                for result in observed
                if result.latency_ms is not None
            ),
            0.95,
        ),
        successful_latency_p50_ms=_nearest_rank(
            (
                result.latency_ms
                for result in successful_observed
                if result.latency_ms is not None
            ),
            0.50,
        ),
        successful_latency_p95_ms=_nearest_rank(
            (
                result.latency_ms
                for result in successful_observed
                if result.latency_ms is not None
            ),
            0.95,
        ),
        average_total_tokens=_mean_or_none(
            float(result.total_tokens)
            for result in observed
            if result.total_tokens is not None
        ),
        average_monetary_cost=_mean_or_none(
            result.monetary_cost
            for result in observed
            if result.monetary_cost is not None
        ),
        failure_layer_counts=dict(sorted(failure_counts.items())),
    )

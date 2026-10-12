"""多 Agent 分支依赖传播与最终结果聚合。

这里把“哪个分支失败”与“整个任务是否成功”拆开。可选分支失败可以形成降级成功，
而未真正运行的依赖分支必须记录为跳过，不能错误归因成自身执行失败。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class BranchStatus(str, Enum):
    """单个 Agent 分支在依赖图中的状态。"""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    NEEDS_HUMAN = "needs_human"
    BLOCKED = "blocked"
    SKIPPED_UPSTREAM_FAILED = "skipped_upstream_failed"
    CANCELLED_SAFETY = "cancelled_safety"


NON_TERMINAL_STATUSES = frozenset(
    {
        BranchStatus.PENDING,
        BranchStatus.READY,
        BranchStatus.RUNNING,
        BranchStatus.BLOCKED,
    }
)


@dataclass(frozen=True)
class BranchOutcome:
    """一个分支的验收结果，而不是未经校验的 Agent 自报结果。"""

    branch_id: str
    required: bool
    status: BranchStatus
    evidence_complete: bool = False


@dataclass(frozen=True)
class TaskOutcome:
    """区分硬成功门槛、降级成功和失败归因。"""

    task_success: bool
    degraded_success: bool
    safety_violation: bool
    required_failures: tuple[str, ...]
    optional_failures: tuple[str, ...]


def resolve_dependent_status(
    upstream_statuses: Iterable[BranchStatus],
) -> BranchStatus:
    """根据上游状态决定尚未运行的下游分支状态。

    安全停止优先级最高；确定性失败会让下游直接跳过；状态未知或仍在运行时，下游
    暂时阻塞；只有所有依赖成功后才能变为 READY。
    """

    statuses = tuple(upstream_statuses)
    if not statuses:
        return BranchStatus.READY
    if BranchStatus.CANCELLED_SAFETY in statuses:
        return BranchStatus.CANCELLED_SAFETY
    if any(
        status in {BranchStatus.FAILED, BranchStatus.SKIPPED_UPSTREAM_FAILED}
        for status in statuses
    ):
        return BranchStatus.SKIPPED_UPSTREAM_FAILED
    if all(status is BranchStatus.SUCCEEDED for status in statuses):
        return BranchStatus.READY
    return BranchStatus.BLOCKED


def aggregate_task_outcome(
    branches: Iterable[BranchOutcome],
    *,
    safety_violation: bool,
) -> TaskOutcome:
    """在所有分支进入终态后，按必需/可选合同聚合任务结果。"""

    outcomes = tuple(branches)
    unfinished = tuple(
        branch.branch_id
        for branch in outcomes
        if branch.status in NON_TERMINAL_STATUSES
    )
    if unfinished:
        raise ValueError(f"BRANCHES_NOT_TERMINAL:{','.join(unfinished)}")

    required_failures = tuple(
        branch.branch_id
        for branch in outcomes
        if branch.required
        and not (
            branch.status is BranchStatus.SUCCEEDED
            and branch.evidence_complete
        )
    )
    optional_failures = tuple(
        branch.branch_id
        for branch in outcomes
        if not branch.required
        and not (
            branch.status is BranchStatus.SUCCEEDED
            and branch.evidence_complete
        )
    )
    task_success = not required_failures and not safety_violation
    return TaskOutcome(
        task_success=task_success,
        degraded_success=task_success and bool(optional_failures),
        safety_violation=safety_violation,
        required_failures=required_failures,
        optional_failures=optional_failures,
    )

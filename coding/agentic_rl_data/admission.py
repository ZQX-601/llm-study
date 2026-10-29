"""Agentic RL 训练数据准入的教学实现。

准入不是简单的 True/False：

* ACCEPT：版本一致、结构完整且足够新鲜；
* PENDING：GRPO 组尚未收齐，继续等待；
* REPROCESS：交互语义没变，可以从完整 Trace 重新评分；
* RECOLLECT：Policy 的可见信息或可用动作发生变化，必须重新采集；
* REJECT：数据本身合法，但不适合当前这次近似 on-policy 训练；
* QUARANTINE：结构损坏或合同异常，隔离后排查生产链路。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .contracts import Trajectory
from .sample_builder import validate_trajectory


class AdmissionDecision(str, Enum):
    ACCEPT = "accept"
    PENDING = "pending"
    REPROCESS = "reprocess"
    RECOLLECT = "recollect"
    REJECT = "reject"
    QUARANTINE = "quarantine"


@dataclass(frozen=True)
class TrainingDataContract:
    """当前 Trainer 期望的数据版本与新鲜度合同。"""

    task_version: str = "diagnose-task-v1"
    environment_version: str = "training-diagnosis-v1"
    observation_schema_version: str = "observation-v1"
    tool_schema_version: str = "training-tools-v1"
    reward_version: str = "training-reward-v1"
    verifier_version: str = "diagnosis-verifier-v1"
    data_snapshot: str = "static-diagnosis-data-v1"
    tokenizer_version: str = "unicode-teaching-v1"
    chat_template_version: str = "diagnosis-prompt-v1"
    current_policy_version: int = 10
    max_policy_lag: int = 2
    expected_group_size: int = 4
    clip_epsilon: float = 0.2
    max_clipfrac: float = 0.3
    max_approx_kl: float = 0.1


@dataclass(frozen=True)
class AdmissionResult:
    """机器可分流的决定，以及给人排障的原因。"""

    decision: AdmissionDecision
    reason_code: str
    detail: str


@dataclass(frozen=True)
class PolicyShiftMetrics:
    """当前 Policy forward 后，在有效 Response token 上统计的分布偏移。"""

    valid_token_count: int
    clipfrac: float
    approx_kl: float


def _result(
    decision: AdmissionDecision, reason_code: str, detail: str
) -> AdmissionResult:
    return AdmissionResult(decision, reason_code, detail)


def admit_trajectory(
    trajectory: Trajectory,
    contract: TrainingDataContract,
    *,
    reward_recomputable_from_trace: bool = False,
) -> AdmissionResult:
    """按“结构 → 因果版本 → Reward → 新鲜度”的顺序检查一条轨迹。

    顺序很重要：结构损坏的数据不能因为版本刚好匹配就进入训练；Tool/Observation
    变化也不能通过重算 Reward 修复。policy lag 最后检查，因为它只回答这条合法数据
    对当前近似 on-policy 更新是否仍足够新鲜。
    """

    try:
        validate_trajectory(trajectory)
    except ValueError as error:
        return _result(
            AdmissionDecision.QUARANTINE,
            "MALFORMED_TRAJECTORY",
            str(error),
        )

    # 这些变化会改变 Policy 的信息集、动作空间或真实状态迁移。旧轨迹只能证明旧环境
    # 中发生过什么，不能冒充由新环境采集的数据。
    causal_versions = (
        ("task_version", trajectory.task_version, contract.task_version),
        (
            "environment_version",
            trajectory.environment_version,
            contract.environment_version,
        ),
        (
            "observation_schema_version",
            trajectory.observation_schema_version,
            contract.observation_schema_version,
        ),
        ("tool_schema_version", trajectory.tool_schema_version, contract.tool_schema_version),
        ("verifier_version", trajectory.verifier_version, contract.verifier_version),
        ("data_snapshot", trajectory.data_snapshot, contract.data_snapshot),
        ("tokenizer_version", trajectory.tokenizer_version, contract.tokenizer_version),
        (
            "chat_template_version",
            trajectory.chat_template_version,
            contract.chat_template_version,
        ),
    )
    for field_name, actual, expected in causal_versions:
        if actual != expected:
            return _result(
                AdmissionDecision.RECOLLECT,
                "CAUSAL_VERSION_MISMATCH",
                f"{field_name}: trajectory={actual}, trainer={expected}",
            )

    if trajectory.reward_version != contract.reward_version:
        if reward_recomputable_from_trace:
            return _result(
                AdmissionDecision.REPROCESS,
                "REWARD_VERSION_MISMATCH",
                "交互语义未变，完整 Trace 可按目标 Reward 版本重新评分",
            )
        return _result(
            AdmissionDecision.RECOLLECT,
            "REWARD_INPUT_MISSING",
            "Reward 版本变化，但旧 Trace 缺少重新评分所需信息",
        )

    policy_versions = {step.decision.policy_version for step in trajectory.steps}
    if len(policy_versions) != 1:
        return _result(
            AdmissionDecision.QUARANTINE,
            "MIXED_POLICY_VERSIONS",
            "同一条 trajectory 的决策来自多个 behavior policy 版本",
        )

    behavior_version = next(iter(policy_versions))
    policy_lag = contract.current_policy_version - behavior_version
    if policy_lag < 0:
        return _result(
            AdmissionDecision.QUARANTINE,
            "FUTURE_POLICY_VERSION",
            f"behavior policy v{behavior_version} 领先于 trainer v{contract.current_policy_version}",
        )
    if policy_lag > contract.max_policy_lag:
        return _result(
            AdmissionDecision.REJECT,
            "POLICY_TOO_STALE",
            f"policy lag={policy_lag} > max_policy_lag={contract.max_policy_lag}",
        )

    return _result(
        AdmissionDecision.ACCEPT,
        "TRAJECTORY_ACCEPTED",
        f"结构和版本合法，policy lag={policy_lag}",
    )


def admit_grpo_group(
    trajectories: Sequence[Trajectory],
    contract: TrainingDataContract,
    *,
    assembly_timed_out: bool = False,
) -> AdmissionResult:
    """检查 GRPO 的同组 rollout 是否完整，并复用单轨迹准入规则。"""

    if not trajectories:
        return _result(
            AdmissionDecision.PENDING,
            "EMPTY_GROUP",
            "尚未收到任何 rollout",
        )

    trajectory_ids = [trajectory.trajectory_id for trajectory in trajectories]
    if len(set(trajectory_ids)) != len(trajectory_ids):
        return _result(
            AdmissionDecision.QUARANTINE,
            "DUPLICATE_TRAJECTORY",
            "同一 GRPO 组出现重复 trajectory_id",
        )

    group_ids = {trajectory.group_id for trajectory in trajectories}
    task_ids = {trajectory.task_id for trajectory in trajectories}
    if len(group_ids) != 1 or len(task_ids) != 1:
        return _result(
            AdmissionDecision.QUARANTINE,
            "MIXED_GROUP_IDENTITY",
            "GRPO 组内必须共享 group_id 和 task_id",
        )

    actual_size = len(trajectories)
    expected_size = contract.expected_group_size
    if actual_size < expected_size:
        if assembly_timed_out:
            return _result(
                AdmissionDecision.REJECT,
                "INCOMPLETE_GROUP_TIMEOUT",
                f"组装超时，只收到 {actual_size}/{expected_size} 条 rollout",
            )
        return _result(
            AdmissionDecision.PENDING,
            "INCOMPLETE_GROUP",
            f"已收到 {actual_size}/{expected_size} 条 rollout，继续等待",
        )
    if actual_size > expected_size:
        return _result(
            AdmissionDecision.QUARANTINE,
            "OVERFULL_GROUP",
            f"收到 {actual_size}/{expected_size} 条 rollout，可能存在重复投递",
        )

    for trajectory in trajectories:
        result = admit_trajectory(trajectory, contract)
        if result.decision is not AdmissionDecision.ACCEPT:
            return result

    return _result(
        AdmissionDecision.ACCEPT,
        "GROUP_ACCEPTED",
        f"GRPO 组完整：{actual_size}/{expected_size}",
    )


def admit_policy_shift(
    *,
    old_logprobs: Sequence[Sequence[float]],
    new_logprobs: Sequence[Sequence[float]],
    response_mask: Sequence[Sequence[int]],
    contract: TrainingDataContract,
) -> tuple[AdmissionResult, PolicyShiftMetrics | None]:
    """用 current-policy forward 结果做动态准入。

    `clipfrac` 使用未裁剪的 ratio 判断是否越界；裁剪后的 ratio 只用于 PPO/GRPO
    surrogate loss。KL 使用常见的非负近似 `(ratio - 1) - log_ratio`。所有统计只覆盖
    `response_mask == 1` 的 token，Prompt 与 Padding 即使有占位值也必须忽略。
    """

    if not (
        len(old_logprobs) == len(new_logprobs) == len(response_mask)
    ):
        return (
            _result(
                AdmissionDecision.QUARANTINE,
                "POLICY_SHIFT_ROW_MISMATCH",
                "old/new logprob 与 response_mask 的 Decision 行数不一致",
            ),
            None,
        )

    clipped_tokens = 0
    valid_tokens = 0
    approx_kl_sum = 0.0
    lower = 1.0 - contract.clip_epsilon
    upper = 1.0 + contract.clip_epsilon

    for old_row, new_row, mask_row in zip(
        old_logprobs, new_logprobs, response_mask
    ):
        if not (len(old_row) == len(new_row) == len(mask_row)):
            return (
                _result(
                    AdmissionDecision.QUARANTINE,
                    "POLICY_SHIFT_TOKEN_MISMATCH",
                    "同一 Decision 内的 token 维度不一致",
                ),
                None,
            )

        for old_logprob, new_logprob, valid in zip(old_row, new_row, mask_row):
            if valid not in (0, 1):
                return (
                    _result(
                        AdmissionDecision.QUARANTINE,
                        "INVALID_RESPONSE_MASK",
                        "response_mask 只能包含 0/1",
                    ),
                    None,
                )
            if valid == 0:
                continue
            if not (
                math.isfinite(old_logprob) and math.isfinite(new_logprob)
            ):
                return (
                    _result(
                        AdmissionDecision.QUARANTINE,
                        "NON_FINITE_LOGPROB",
                        "有效 Response token 的 logprob 包含 NaN/Inf",
                    ),
                    None,
                )
            if old_logprob > 0.0 or new_logprob > 0.0:
                return (
                    _result(
                        AdmissionDecision.QUARANTINE,
                        "POSITIVE_LOGPROB",
                        "离散 token 的 log probability 不应大于 0",
                    ),
                    None,
                )

            log_ratio = new_logprob - old_logprob
            # 极端但有限的策略偏移属于“合法但过旧”，因此记为 Inf 后走 REJECT，
            # 而不是误报成采集数据损坏。
            ratio = math.exp(log_ratio) if log_ratio < 709.0 else math.inf
            if ratio < lower or ratio > upper:
                clipped_tokens += 1
            approx_kl_sum += (ratio - 1.0) - log_ratio
            valid_tokens += 1

    if valid_tokens == 0:
        return (
            _result(
                AdmissionDecision.QUARANTINE,
                "EMPTY_RESPONSE_MASK",
                "没有任何参与策略训练的 Response token",
            ),
            None,
        )

    metrics = PolicyShiftMetrics(
        valid_token_count=valid_tokens,
        clipfrac=clipped_tokens / valid_tokens,
        approx_kl=approx_kl_sum / valid_tokens,
    )
    if (
        metrics.clipfrac > contract.max_clipfrac
        or metrics.approx_kl > contract.max_approx_kl
    ):
        return (
            _result(
                AdmissionDecision.REJECT,
                "POLICY_SHIFT_TOO_LARGE",
                (
                    f"clipfrac={metrics.clipfrac:.4f}, "
                    f"approx_kl={metrics.approx_kl:.4f}"
                ),
            ),
            metrics,
        )

    return (
        _result(
            AdmissionDecision.ACCEPT,
            "POLICY_SHIFT_ACCEPTED",
            (
                f"clipfrac={metrics.clipfrac:.4f}, "
                f"approx_kl={metrics.approx_kl:.4f}"
            ),
        ),
        metrics,
    )

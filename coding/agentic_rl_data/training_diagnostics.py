"""Agentic RL 训练指标与失败诊断的教学规则引擎。

它不替代真实监控或统计检验，只把诊断顺序变成可测试合同：先阻断非有限优化，再检查
Reward 与真实评测的背离、GRPO 信号、Policy 偏移和 Actor/Learner 系统失衡。所有指标
必须携带版本身份，避免把 Reward/Environment 改版误判成模型学习效果。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class DiagnosticCode(str, Enum):
    HEALTHY = "healthy"
    REWARD_HACKING_SUSPECTED = "reward_hacking_suspected"
    NO_GRPO_SIGNAL = "no_grpo_signal"
    POLICY_UPDATE_TOO_LARGE = "policy_update_too_large"
    ACTOR_LEARNER_IMBALANCE = "actor_learner_imbalance"
    NON_FINITE_OPTIMIZATION = "non_finite_optimization"


@dataclass(frozen=True)
class MetricIdentity:
    """一组指标对应的训练与数据版本。"""

    run_id: str
    update_step: int
    policy_version: int
    behavior_policy_version: int
    environment_version: str
    reward_version: str
    verifier_version: str


@dataclass(frozen=True)
class TrainingMetricSnapshot:
    """教学诊断所需的最小指标快照；trend 表示相对稳定基线的方向变化。"""

    identity: MetricIdentity
    train_reward_trend: float
    eval_success_trend: float
    tool_calls_trend: float
    trajectory_length_trend: float
    group_reward_std: float
    zero_advantage_group_rate: float
    approx_kl_to_old: float
    target_kl: float
    clipfrac: float
    max_clipfrac: float
    queue_depth: int
    max_queue_depth: int
    rollout_production_rate: float
    accepted_trajectory_rate: float
    stale_rejection_rate: float
    non_finite_loss_rate: float
    non_finite_gradient_rate: float


@dataclass(frozen=True)
class DiagnosticFinding:
    code: DiagnosticCode
    reason: str
    priority: int


def _validate(snapshot: TrainingMetricSnapshot) -> None:
    if not snapshot.identity.run_id:
        raise ValueError("训练指标缺少 run_id")
    if snapshot.identity.update_step < 0:
        raise ValueError("update_step 不能为负数")

    finite_values = (
        snapshot.train_reward_trend,
        snapshot.eval_success_trend,
        snapshot.tool_calls_trend,
        snapshot.trajectory_length_trend,
        snapshot.group_reward_std,
        snapshot.zero_advantage_group_rate,
        snapshot.approx_kl_to_old,
        snapshot.target_kl,
        snapshot.clipfrac,
        snapshot.max_clipfrac,
        snapshot.rollout_production_rate,
        snapshot.accepted_trajectory_rate,
        snapshot.stale_rejection_rate,
        snapshot.non_finite_loss_rate,
        snapshot.non_finite_gradient_rate,
    )
    if not all(math.isfinite(value) for value in finite_values):
        raise ValueError("训练指标本身包含 NaN/Inf，不能继续规则诊断")
    if snapshot.target_kl <= 0 or snapshot.max_clipfrac < 0:
        raise ValueError("KL/clipfrac 阈值配置非法")
    if snapshot.queue_depth < 0 or snapshot.max_queue_depth <= 0:
        raise ValueError("Queue 指标配置非法")
    if (
        snapshot.rollout_production_rate < 0
        or snapshot.accepted_trajectory_rate < 0
    ):
        raise ValueError("吞吐率不能为负数")

    rates = (
        snapshot.zero_advantage_group_rate,
        snapshot.clipfrac,
        snapshot.max_clipfrac,
        snapshot.stale_rejection_rate,
        snapshot.non_finite_loss_rate,
        snapshot.non_finite_gradient_rate,
    )
    if any(rate < 0 or rate > 1 for rate in rates):
        raise ValueError("rate/clipfrac 必须位于 [0,1]")


def diagnose_training(
    snapshot: TrainingMetricSnapshot,
) -> tuple[DiagnosticFinding, ...]:
    """返回所有命中的诊断，并按阻断优先级从高到低排序。"""

    _validate(snapshot)
    findings: list[DiagnosticFinding] = []

    if (
        snapshot.non_finite_loss_rate > 0
        or snapshot.non_finite_gradient_rate > 0
    ):
        findings.append(
            DiagnosticFinding(
                DiagnosticCode.NON_FINITE_OPTIMIZATION,
                "出现非有限 loss/gradient，必须先阻断 optimizer.step 并定位异常 batch",
                100,
            )
        )

    if (
        snapshot.train_reward_trend > 0
        and snapshot.eval_success_trend < 0
        and (
            snapshot.tool_calls_trend > 0
            or snapshot.trajectory_length_trend > 0
        )
    ):
        findings.append(
            DiagnosticFinding(
                DiagnosticCode.REWARD_HACKING_SUSPECTED,
                "训练 Reward 上升但独立成功率下降，且成本/轨迹长度增加",
                90,
            )
        )

    if (
        snapshot.group_reward_std <= 1e-8
        or snapshot.zero_advantage_group_rate >= 0.8
    ):
        findings.append(
            DiagnosticFinding(
                DiagnosticCode.NO_GRPO_SIGNAL,
                "同组 Reward 缺少区分度，大量 Group Advantage 为 0",
                80,
            )
        )

    if (
        snapshot.approx_kl_to_old > snapshot.target_kl
        or snapshot.clipfrac > snapshot.max_clipfrac
    ):
        findings.append(
            DiagnosticFinding(
                DiagnosticCode.POLICY_UPDATE_TOO_LARGE,
                "KL 或 clipfrac 超过训练合同，当前 Policy 偏移过大",
                70,
            )
        )

    queue_pressure = snapshot.queue_depth / snapshot.max_queue_depth
    # Queue 可能因为过期数据被高速拒绝而保持低水位，因此必须同时看
    # 原始产量与真正准入训练的吞吐。两者单位都是 trajectory/time-window。
    if snapshot.rollout_production_rate > 0:
        acceptance_ratio = (
            snapshot.accepted_trajectory_rate / snapshot.rollout_production_rate
        )
    else:
        acceptance_ratio = 1.0
    hidden_throughput_gap = acceptance_ratio < 0.8

    if (
        (queue_pressure >= 0.8 or hidden_throughput_gap)
        and snapshot.stale_rejection_rate >= 0.2
    ):
        findings.append(
            DiagnosticFinding(
                DiagnosticCode.ACTOR_LEARNER_IMBALANCE,
                "Queue 压力或产量/准入吞吐差过大，且过期拒绝率高",
                60,
            )
        )

    if not findings:
        findings.append(
            DiagnosticFinding(
                DiagnosticCode.HEALTHY,
                "当前规则未发现明显异常；仍需结合切片指标和原始 Trace",
                0,
            )
        )

    return tuple(sorted(findings, key=lambda item: item.priority, reverse=True))

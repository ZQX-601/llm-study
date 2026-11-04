"""PPO/GRPO 训练循环控制的纯 Python 教学参考。

这里不伪造 PyTorch backward，而是把两个容易被训练框架代码掩盖的控制语义单独验证：

1. ragged microbatch 应按目标统计单位合并，不能盲目平均 microbatch mean；
2. KL 提前停止必须同时退出 minibatch 与 epoch 两层循环。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


def weighted_microbatch_mean(
    *,
    means: Sequence[float],
    valid_counts: Sequence[int],
) -> float:
    """按有效 token 数合并 microbatch mean，恢复全局 token mean。"""

    if len(means) != len(valid_counts) or not means:
        raise ValueError("means 与 valid_counts 必须非空且长度一致")
    if any(count <= 0 for count in valid_counts):
        raise ValueError("每个 microbatch 至少需要一个有效 token")
    if not all(math.isfinite(mean) for mean in means):
        raise ValueError("microbatch mean 包含 NaN/Inf")

    loss_sum = sum(
        mean * count for mean, count in zip(means, valid_counts)
    )
    return loss_sum / sum(valid_counts)


@dataclass(frozen=True)
class EarlyStopResult:
    """KL 控制器实际消费到的位置。"""

    processed_minibatches: int
    completed_epochs: int
    early_stopped: bool
    stop_epoch: int | None
    stop_minibatch: int | None


def simulate_kl_early_stop(
    *,
    approx_kls_by_epoch: Sequence[Sequence[float]],
    target_kl: float,
) -> EarlyStopResult:
    """模拟 PPO epoch/minibatch 双层循环的 KL 提前停止。

    `approx_kls_by_epoch[e][m]` 表示第 e 个 epoch、第 m 个 minibatch 更新后观测到的
    KL。超过阈值的 minibatch 已经执行，但后续 minibatch 与 epoch 都不得继续。
    """

    if target_kl <= 0 or not math.isfinite(target_kl):
        raise ValueError("target_kl 必须是有限正数")

    processed = 0
    completed_epochs = 0
    early_stop = False
    stop_epoch: int | None = None
    stop_minibatch: int | None = None

    for epoch_index, epoch_kls in enumerate(approx_kls_by_epoch):
        for minibatch_index, approx_kl in enumerate(epoch_kls):
            if not math.isfinite(approx_kl):
                raise ValueError("approx_kl 包含 NaN/Inf")
            processed += 1
            if approx_kl > target_kl:
                early_stop = True
                stop_epoch = epoch_index
                stop_minibatch = minibatch_index
                break

        if early_stop:
            break
        completed_epochs += 1

    return EarlyStopResult(
        processed_minibatches=processed,
        completed_epochs=completed_epochs,
        early_stopped=early_stop,
        stop_epoch=stop_epoch,
        stop_minibatch=stop_minibatch,
    )


class OptimizerStepDecision(str, Enum):
    """一次 minibatch 是否允许写入模型参数。"""

    STEP = "step"
    SKIP = "skip"


@dataclass(frozen=True)
class OptimizerStepGateResult:
    decision: OptimizerStepDecision
    reason_code: str
    detail: str


def gate_optimizer_step(
    *,
    total_loss: float,
    grad_norm: float | None,
    valid_token_count: int,
) -> OptimizerStepGateResult:
    """在 optimizer.step 前阻断空 batch 和非有限数值。

    `grad_norm=None` 表示尚未 backward；只有 loss 有限后才应计算梯度。这个纯 Python
    gate 不替代 PyTorch AMP/GradScaler，只把“什么情况下绝不能写参数”变成可测试合同。
    """

    if valid_token_count <= 0:
        return OptimizerStepGateResult(
            OptimizerStepDecision.SKIP,
            "EMPTY_RESPONSE_TOKENS",
            "没有有效 Response token，不能形成策略梯度",
        )
    if not math.isfinite(total_loss):
        return OptimizerStepGateResult(
            OptimizerStepDecision.SKIP,
            "NON_FINITE_LOSS",
            f"total_loss={total_loss}",
        )
    if grad_norm is None:
        return OptimizerStepGateResult(
            OptimizerStepDecision.SKIP,
            "GRADIENT_NOT_COMPUTED",
            "loss 有限，但尚未执行 backward/gradient clipping",
        )
    if not math.isfinite(grad_norm):
        return OptimizerStepGateResult(
            OptimizerStepDecision.SKIP,
            "NON_FINITE_GRADIENT",
            f"grad_norm={grad_norm}",
        )
    return OptimizerStepGateResult(
        OptimizerStepDecision.STEP,
        "OPTIMIZER_STEP_ALLOWED",
        f"loss={total_loss}, grad_norm={grad_norm}",
    )

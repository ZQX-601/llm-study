"""从准入后的 token 字段计算 PPO/GRPO Policy loss。

当前仓库环境未安装 PyTorch，因此 `ppo_policy_loss_reference` 使用标准库执行并验证
数值；`ppo_policy_loss_torch` 是与其等价的真实 Tensor/Autograd 路径，安装 PyTorch 后
可直接运行。两条路径都强调三件事：因果错位、Response mask、先算原始 ratio 再裁剪。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class PPOLossMetrics:
    """不参与梯度的监控指标；policy_loss 才是优化目标。"""

    policy_loss: float
    clipfrac: float
    approx_kl: float
    valid_token_count: int


@dataclass(frozen=True)
class LossCoefficients:
    """PPO/GRPO 辅助 loss 的缩放系数。"""

    value: float = 0.5
    entropy: float = 0.01
    reference_kl: float = 0.1


def combine_ppo_loss_reference(
    *,
    policy_loss: float,
    value_loss: float,
    entropy: float,
    reference_kl: float,
    coefficients: LossCoefficients = LossCoefficients(),
) -> float:
    """组合 PPO 总 loss：Value/KL 是惩罚，Entropy 是奖励。"""

    return (
        policy_loss
        + coefficients.value * value_loss
        - coefficients.entropy * entropy
        + coefficients.reference_kl * reference_kl
    )


def combine_grpo_loss_reference(
    *,
    policy_loss: float,
    entropy: float,
    reference_kl: float,
    coefficients: LossCoefficients = LossCoefficients(),
) -> float:
    """GRPO 没有 Critic，因此不含 Value loss。"""

    return (
        policy_loss
        - coefficients.entropy * entropy
        + coefficients.reference_kl * reference_kl
    )


def ppo_policy_loss_reference(
    *,
    # 已从 logits[:, :-1, :] gather 得到，所以 shape 是 [B, T-1]。
    new_logprobs: Sequence[Sequence[float]],
    # 以下三个字段来自 collated batch，shape 都是 [B, T]；函数内部统一右移。
    old_logprobs: Sequence[Sequence[float]],
    advantages: Sequence[Sequence[float]],
    response_mask: Sequence[Sequence[int]],
    clip_epsilon: float = 0.2,
) -> PPOLossMetrics:
    """纯 Python 教学参考：计算 masked PPO clipped policy loss。"""

    if not (
        len(new_logprobs)
        == len(old_logprobs)
        == len(advantages)
        == len(response_mask)
    ):
        raise ValueError("PPO loss 的 batch 维度不一致")
    if not 0.0 < clip_epsilon < 1.0:
        raise ValueError("clip_epsilon 必须位于 (0, 1)")

    loss_sum = 0.0
    approx_kl_sum = 0.0
    clipped_tokens = 0
    valid_tokens = 0
    lower = 1.0 - clip_epsilon
    upper = 1.0 + clip_epsilon

    for new_row, old_row, advantage_row, mask_row in zip(
        new_logprobs, old_logprobs, advantages, response_mask
    ):
        if not (len(old_row) == len(advantage_row) == len(mask_row)):
            raise ValueError("old_logprobs/advantages/response_mask 的 token 维度不一致")
        if len(new_row) != len(old_row) - 1:
            raise ValueError("new_logprobs 必须对应因果右移后的 T-1 个 target token")

        # new_row[i] 表示原序列 token[i+1] 的概率，因此其他字段都取 [1:]。
        for new, old, advantage, mask in zip(
            new_row, old_row[1:], advantage_row[1:], mask_row[1:]
        ):
            if mask not in (0, 1):
                raise ValueError("response_mask 只能包含 0/1")
            if mask == 0:
                continue
            if not all(math.isfinite(value) for value in (new, old, advantage)):
                raise ValueError("有效 Response token 包含 NaN/Inf")

            log_ratio = new - old
            ratio = math.exp(log_ratio)
            clipped_ratio = min(max(ratio, lower), upper)
            unclipped_surrogate = ratio * advantage
            clipped_surrogate = clipped_ratio * advantage
            loss_sum += -min(unclipped_surrogate, clipped_surrogate)
            approx_kl_sum += (ratio - 1.0) - log_ratio
            clipped_tokens += int(ratio < lower or ratio > upper)
            valid_tokens += 1

    if valid_tokens == 0:
        raise ValueError("没有任何参与 PPO Policy loss 的 Response token")

    return PPOLossMetrics(
        policy_loss=loss_sum / valid_tokens,
        clipfrac=clipped_tokens / valid_tokens,
        approx_kl=approx_kl_sum / valid_tokens,
        valid_token_count=valid_tokens,
    )


def ppo_policy_loss_torch(
    *,
    logits: Any,
    input_ids: Any,
    old_logprobs: Any,
    advantages: Any,
    response_mask: Any,
    clip_epsilon: float = 0.2,
) -> tuple[Any, dict[str, Any]]:
    """真实 PyTorch 路径：返回保留 Autograd 的 loss 和已 detach 的监控指标。

    预期 shape：
      logits         [B, T, V]
      input_ids      [B, T]
      old_logprobs   [B, T]
      advantages     [B, T]
      response_mask  [B, T]

    当前环境没有 PyTorch，函数采用延迟导入；缺少依赖时给出明确错误，不伪装成已运行。
    """

    try:
        import torch
        import torch.nn.functional as functional
    except ModuleNotFoundError as error:  # pragma: no cover - 依赖安装后运行
        raise RuntimeError("需要安装 PyTorch 才能运行 Tensor/Autograd 路径") from error

    # logits[:, t] 预测 input_ids[:, t+1]，所以预测张量去掉最后一步，target 去掉 BOS。
    shift_logits = logits[:, :-1, :]
    shift_targets = input_ids[:, 1:]
    new_logprobs = functional.log_softmax(shift_logits, dim=-1).gather(
        dim=-1,
        index=shift_targets.unsqueeze(-1),
    ).squeeze(-1)

    # old log-prob 与 Advantage 是 rollout 后冻结的训练数据。显式 detach 可以防止
    # 上游误带计算图时，当前 Policy loss 反向进入 behavior model 或 GAE 处理链。
    shift_old = old_logprobs[:, 1:].detach()
    shift_advantages = advantages[:, 1:].detach()
    shift_mask = response_mask[:, 1:].to(dtype=new_logprobs.dtype)

    log_ratio = new_logprobs - shift_old
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(
        ratio,
        1.0 - clip_epsilon,
        1.0 + clip_epsilon,
    )
    unclipped_surrogate = ratio * shift_advantages
    clipped_surrogate = clipped_ratio * shift_advantages
    per_token_loss = -torch.minimum(unclipped_surrogate, clipped_surrogate)

    valid_tokens = shift_mask.sum()
    if valid_tokens.detach().item() == 0:
        raise ValueError("没有任何参与 PPO Policy loss 的 Response token")
    # 只有这个 masked mean 保留梯度；指标全部 detach，避免监控逻辑进入计算图。
    policy_loss = (per_token_loss * shift_mask).sum() / valid_tokens
    approx_kl = ((((ratio - 1.0) - log_ratio) * shift_mask).sum() / valid_tokens)
    clipfrac = (
        (((ratio < 1.0 - clip_epsilon) | (ratio > 1.0 + clip_epsilon)).float()
         * shift_mask).sum()
        / valid_tokens
    )

    metrics = {
        "clipfrac": clipfrac.detach(),
        "approx_kl": approx_kl.detach(),
        "valid_token_count": valid_tokens.detach(),
    }
    return policy_loss, metrics


def trajectory_equal_masked_mean_torch(
    *,
    per_token_values: Any,
    response_mask: Any,
    trajectory_indices: Any,
    num_trajectories: int,
) -> tuple[Any, Any]:
    """把 Decision-flatten 的 token 值聚合为 trajectory-equal mean。

    预期 shape：
      per_token_values    [D, T]
      response_mask       [D, T]
      trajectory_indices  [D]

    返回：
      objective           scalar，每条 trajectory 等权
      trajectory_means    [G]

    `scatter_add_` 只改变聚合布局，梯度仍可从 objective 回到 per_token_values。
    """

    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - 依赖安装后运行
        raise RuntimeError("需要安装 PyTorch 才能运行 Tensor/Autograd 路径") from error

    if per_token_values.ndim != 2 or response_mask.shape != per_token_values.shape:
        raise ValueError("per_token_values 与 response_mask 必须是相同 shape 的二维 Tensor")
    if trajectory_indices.ndim != 1 or trajectory_indices.shape[0] != per_token_values.shape[0]:
        raise ValueError("trajectory_indices 必须为与 Decision 行数一致的一维 Tensor")
    if num_trajectories <= 0:
        raise ValueError("num_trajectories 必须大于 0")

    mask = response_mask.to(
        device=per_token_values.device,
        dtype=per_token_values.dtype,
    )
    indices = trajectory_indices.to(
        device=per_token_values.device,
        dtype=torch.long,
    )
    if torch.any(indices < 0).detach().item() or torch.any(
        indices >= num_trajectories
    ).detach().item():
        raise ValueError("trajectory_indices 越界")

    decision_sums = (per_token_values * mask).sum(dim=1)
    decision_counts = mask.sum(dim=1)

    trajectory_sums = per_token_values.new_zeros(num_trajectories)
    trajectory_counts = per_token_values.new_zeros(num_trajectories)
    trajectory_sums.scatter_add_(
        dim=0,
        index=indices,
        src=decision_sums,
    )
    trajectory_counts.scatter_add_(
        dim=0,
        index=indices,
        src=decision_counts,
    )

    if torch.any(trajectory_counts == 0).detach().item():
        raise ValueError("每条 trajectory 至少需要一个有效 Response token")

    trajectory_means = trajectory_sums / trajectory_counts
    objective = trajectory_means.mean()
    return objective, trajectory_means


def grpo_policy_loss_torch(
    *,
    logits: Any,
    input_ids: Any,
    old_logprobs: Any,
    group_advantages: Any,
    response_mask: Any,
    trajectory_indices: Any,
    num_trajectories: int,
    clip_epsilon: float = 0.2,
) -> tuple[Any, dict[str, Any]]:
    """计算 Decision-flatten、多轮 Agent 的 GRPO clipped Policy loss。

    `group_advantages` 必须先在 `[G]` 条完整 trajectory 上完成 Reward 标准化，再由
    collator 广播到 `[D,T]`；本函数绝不在 Decision 维重新标准化。token loss 最后通过
    `trajectory_indices` 聚合回 `[G]`，避免长 trajectory 或 Decision 多的 trajectory
    获得更大权重。
    """

    try:
        import torch
        import torch.nn.functional as functional
    except ModuleNotFoundError as error:  # pragma: no cover - 依赖安装后运行
        raise RuntimeError("需要安装 PyTorch 才能运行 Tensor/Autograd 路径") from error

    shift_logits = logits[:, :-1, :]
    shift_targets = input_ids[:, 1:]
    new_logprobs = functional.log_softmax(shift_logits, dim=-1).gather(
        dim=-1,
        index=shift_targets.unsqueeze(-1),
    ).squeeze(-1)

    shift_old = old_logprobs[:, 1:].detach()
    shift_advantages = group_advantages[:, 1:].detach()
    shift_mask = response_mask[:, 1:].to(dtype=new_logprobs.dtype)

    log_ratio = new_logprobs - shift_old
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(
        ratio,
        1.0 - clip_epsilon,
        1.0 + clip_epsilon,
    )
    per_token_loss = -torch.minimum(
        ratio * shift_advantages,
        clipped_ratio * shift_advantages,
    )

    grpo_loss, trajectory_means = trajectory_equal_masked_mean_torch(
        per_token_values=per_token_loss,
        response_mask=shift_mask,
        trajectory_indices=trajectory_indices,
        num_trajectories=num_trajectories,
    )

    valid_tokens = shift_mask.sum()
    approx_kl = ((((ratio - 1.0) - log_ratio) * shift_mask).sum() / valid_tokens)
    clipfrac = (
        (((ratio < 1.0 - clip_epsilon) | (ratio > 1.0 + clip_epsilon)).float()
         * shift_mask).sum()
        / valid_tokens
    )
    metrics = {
        "trajectory_means": trajectory_means.detach(),
        "clipfrac": clipfrac.detach(),
        "approx_kl": approx_kl.detach(),
        "valid_token_count": valid_tokens.detach(),
    }
    return grpo_loss, metrics

"""从准入后的 token 字段计算 PPO Policy loss。

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

    shift_old = old_logprobs[:, 1:]
    shift_advantages = advantages[:, 1:]
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

    valid_tokens = shift_mask.sum().clamp_min(1.0)
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

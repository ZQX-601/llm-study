"""完整演示 mHC 的 pre、post、comb 如何生成和应用。

重点区分：
1. fn/base/scale 是随模型训练、保存在 checkpoint 里的参数；
2. pre/post/comb 是根据当前 token 的 residual streams 动态产生的临时系数。

运行：
    python mhc_coefficients_demo.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def sinkhorn(logits: Tensor, iterations: int = 20, eps: float = 1e-6) -> Tensor:
    """把 [...,n,n] logits 变成近似双随机矩阵。"""

    # 官方 kernel 先按行做稳定的 exp/softmax，再交替归一化列和行。
    matrix = torch.softmax(logits.float(), dim=-1) + eps
    matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + eps)
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
    return matrix


def make_mhc_coefficients(
    streams: Tensor,
    fn: Tensor,
    base: Tensor,
    scale: Tensor,
    iterations: int = 20,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """按 V4.1 推理源码的参数化生成 pre、post、comb。

    streams: [B,T,n,d]
    fn:      [(2+n)*n, n*d]，可学习参数
    base:    [(2+n)*n]，可学习参数
    scale:   [3]，可学习参数，分别控制 pre/post/comb 的动态项

    返回：
    raw:  [B,T,(2+n)*n]
    pre:  [B,T,n]
    post: [B,T,n]
    comb: [B,T,n,n]
    """

    batch, seq_len, num_streams, dim = streams.shape
    flat = streams.flatten(2).float()  # [B,T,n,d] -> [B,T,n*d]
    rsqrt = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + eps)

    # 一个大投影一次得到 pre、post、comb 的全部动态分数。
    raw = F.linear(flat, fn) * rsqrt  # [B,T,(2+n)*n]
    pre_raw, post_raw, comb_raw = raw.split(
        [num_streams, num_streams, num_streams * num_streams], dim=-1
    )
    pre_base, post_base, comb_base = base.split(
        [num_streams, num_streams, num_streams * num_streams], dim=-1
    )

    # pre/post 只要求非负，不要求和为 1。
    pre = torch.sigmoid(pre_raw * scale[0] + pre_base) + eps
    post = 2.0 * torch.sigmoid(post_raw * scale[1] + post_base)

    # comb 必须稳定地混合旧流，因此投影到近似双随机矩阵。
    comb_logits = comb_raw * scale[2] + comb_base
    comb = sinkhorn(
        comb_logits.view(batch, seq_len, num_streams, num_streams),
        iterations,
        eps,
    )
    return raw, pre, post, comb


def apply_mhc(
    streams: Tensor,
    sublayer_output: Tensor,
    pre: Tensor,
    post: Tensor,
    comb: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """应用三种映射，直接对应 X_next = comb@X + post.T@F(pre@X)。"""

    # [B,T,1,n] @ [B,T,n,d] -> [B,T,1,d] -> [B,T,d]
    sublayer_input = torch.matmul(pre.unsqueeze(-2), streams).squeeze(-2)

    # [B,T,n,n] @ [B,T,n,d] -> [B,T,n,d]
    mixed_residual = torch.matmul(comb, streams)

    # [B,T,n,1] @ [B,T,1,d] -> [B,T,n,d]，这是外积，不是 d->n*d 大投影。
    write_back = torch.matmul(post.unsqueeze(-1), sublayer_output.unsqueeze(-2))
    next_streams = mixed_residual + write_back
    return sublayer_input, mixed_residual, write_back, next_streams


def main() -> None:
    torch.manual_seed(11)
    batch, seq_len, num_streams, dim = 1, 1, 2, 2
    streams = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])

    coefficient_count = (2 + num_streams) * num_streams  # 2+2+4=8
    # 这里只为演示使用固定随机种子。真实参数来自训练后的 checkpoint。
    fn = torch.randn(coefficient_count, num_streams * dim) * 0.1
    base = torch.zeros(coefficient_count)
    scale = torch.tensor([0.01, 0.01, 0.01])

    raw, pre, post, comb = make_mhc_coefficients(streams, fn, base, scale)
    # 用一个给定的单路子层输出，专门观察写回操作。
    sublayer_output = torch.tensor([[[5.0, -1.0]]])
    layer_input, mixed, written, next_streams = apply_mhc(
        streams, sublayer_output, pre, post, comb
    )

    print("streams shape:", tuple(streams.shape))
    print("fn/base/scale shape:", tuple(fn.shape), tuple(base.shape), tuple(scale.shape))
    print("raw shape:", tuple(raw.shape))
    print("pre:", pre[0, 0], "sum=", pre[0, 0].sum())
    print("post:", post[0, 0])
    print("comb:\n", comb[0, 0])
    print("comb 行和:", comb[0, 0].sum(dim=-1))
    print("comb 列和:", comb[0, 0].sum(dim=-2))
    print("\npre @ X，得到单路子层输入:", layer_input[0, 0])
    print("comb @ X，得到混合后的旧流:\n", mixed[0, 0])
    print("post.T @ y，把单路结果写回多路:\n", written[0, 0])
    print("最终多路状态:\n", next_streams[0, 0])

    assert raw.shape == (batch, seq_len, coefficient_count)
    assert pre.shape == post.shape == (batch, seq_len, num_streams)
    assert comb.shape == (batch, seq_len, num_streams, num_streams)
    torch.testing.assert_close(
        comb.sum(dim=-1), torch.ones(batch, seq_len, num_streams), atol=2e-5, rtol=0
    )
    torch.testing.assert_close(
        comb.sum(dim=-2), torch.ones(batch, seq_len, num_streams), atol=2e-5, rtol=0
    )
    print("\nshape 与双随机约束检查通过。")


if __name__ == "__main__":
    main()

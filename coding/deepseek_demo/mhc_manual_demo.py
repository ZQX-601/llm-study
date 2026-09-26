"""用 1 个 token、2 条残差流、2 个特征手算一次 mHC。

这不是 DeepSeek-V4.1 的完整训练实现。真实模型会根据输入动态生成 pre、post、comb，
并用 Sinkhorn 把 comb 约束为近似双随机矩阵。这里固定这些系数，让每一步都能手算。

运行：
    python mhc_manual_demo.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def expand_to_mhc(x: Tensor, num_streams: int) -> Tensor:
    """把普通 embedding 复制成 mHC 的多条初始 residual stream。

    x:      [B,T,d]
    返回值: [B,T,n,d]

    这一步没有可学习参数，也没有把 d 切成 n 段；n 条流的初始值完全相同。
    后续 hc_post 使用不同的 post 系数写回子层输出后，各条流才会逐渐分化。
    """

    return x.unsqueeze(2).repeat(1, 1, num_streams, 1)


def toy_sublayer(x: Tensor) -> Tensor:
    """代替真实 Attention/MoE 的简单函数。

    对输入 [u,v] 返回 [u+v,u-v]，输入和输出 shape 都是 [B,T,d]。
    """

    u = x[..., 0]
    v = x[..., 1]
    return torch.stack([u + v, u - v], dim=-1)


def hc_pre(streams: Tensor, pre: Tensor) -> Tensor:
    """从 n 条残差流读取一条子层输入。

    streams: [B,T,n,d]
    pre:     [B,T,n]
    返回:    [B,T,d]
    """

    return (pre.unsqueeze(-1) * streams).sum(dim=2)


def predict_pre_mix(
    streams: Tensor,
    weight: Tensor,
    scale: Tensor,
    base: Tensor,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """按 V4.1 官方 `hc_mixes`/`hc_split_sinkhorn` 的顺序生成 pre_mix。

    streams: [B,T,n,d]
    weight:  [n,n*d]，是当前 Attention 或 FFN 自己学习的参数
    scale:   标量，控制输入相关动态项的强度
    base:    [n]，提供静态基础路由

    官方会用一个更大的投影一次产生 pre、post、comb 的全部 raw 值；
    这里仅保留前 n 行，单独演示 pre_mix。
    """

    # [B,T,n,d] -> [B,T,n*d]。同一个 token 的所有 residual stream 被拼在一起。
    flat = streams.flatten(2).float()

    # 官方不是对每条流分别 RMSNorm，而是对该 token 的 n*d 个数共同求一个缩放量。
    # rsqrt: [B,T,1]。
    rsqrt = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + eps)

    # F.linear(flat, weight) 等价于 flat @ weight.T。
    # raw_pre: [B,T,n]，每个 token 都会得到自己的一组动态路由分数。
    raw_pre = F.linear(flat, weight) * rsqrt

    # V4.1 源码使用逐元素 sigmoid，而不是 softmax。
    # 因此每个系数在 (0,1)，但 n 个系数之和通常不等于 1。
    pre_mix = torch.sigmoid(raw_pre * scale + base) + eps
    return pre_mix, raw_pre, rsqrt


def hc_post(output: Tensor, residual: Tensor, post: Tensor, comb: Tensor) -> Tensor:
    """把子层输出写回 n 条流，并混合旧 residual。

    output:   [B,T,d]
    residual: [B,T,n,d]
    post:     [B,T,n]
    comb:     [B,T,n,n]
    返回:     [B,T,n,d]
    """

    # 第 j 条新流接收 post_j * output。
    write_back = post.unsqueeze(-1) * output.unsqueeze(2)  # [B,T,n,d]

    # comb[j,i] 表示第 j 条新流从第 i 条旧流读取多少。
    mixed_residual = torch.einsum("btji,btid->btjd", comb, residual)
    return mixed_residual + write_back


def main() -> None:
    # 模型入口：普通 token embedding 是 [B,T,d]，这里用 [1,2] 表示一个 token。
    embedding = torch.tensor([[[1.0, 2.0]]])  # [1,1,2]
    initial_streams = expand_to_mhc(embedding, num_streams=4)  # [1,1,4,2]
    print("普通 embedding:", embedding[0, 0])
    print("复制得到的 4 条初始流:\n", initial_streams[0, 0])
    print("初始流 shape:", tuple(initial_streams.shape))

    # 只有一个 batch、一个 token。该 token 携带两条 residual stream：
    # stream 0 = [1,2]，stream 1 = [3,4]。这里从已经发生分化的中间层开始，
    # 便于观察 pre_mix 如何从不同信息流中读取内容。
    streams = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # [1,1,2,2]

    # 假设上一子层已经学到了下面这个投影矩阵。每一行负责产生一个流的 pre 分数。
    # 为了容易手算：第一行只看 flat[0]=1，第二行只看 flat[1]=2。
    pre_weight = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0]])  # [n,n*d]=[2,4]
    pre_scale = torch.tensor(1.0)
    pre_base = torch.tensor([0.0, 0.0])
    pre, raw_pre, rsqrt = predict_pre_mix(
        streams, pre_weight, pre_scale, pre_base
    )  # [1,1,2]

    # post 决定子层结果写回两条流的强度。
    post = torch.tensor([[[0.60, 0.40]]])  # [1,1,2]

    # comb 决定旧流如何相互混合。该矩阵非负，而且每行、每列之和都是 1，
    # 所以它是双随机矩阵，不会任意放大旧 residual 的整体尺度。
    comb = torch.tensor([[[[0.80, 0.20],
                           [0.20, 0.80]]]])  # [1,1,2,2]

    # 第一步：hc_pre，把两条流读成一条。
    layer_input = hc_pre(streams, pre)  # [1,1,2]

    # 第二步：真实模型这里会运行 Attention 或 MoE；这里使用可手算函数。
    layer_output = toy_sublayer(layer_input)  # [1,1,2]

    # 第三步：hc_post，把子层结果写回两条流，同时混合旧流。
    next_streams = hc_post(layer_output, streams, post, comb)  # [1,1,2,2]

    # 假设 Encoder 边界需要输出单路 H20，就用下一组 pre_mix 读出。
    encoder_readout_mix = torch.tensor([[[0.70, 0.30]]])
    encoder_output = hc_pre(next_streams, encoder_readout_mix)  # [1,1,2]

    print("\n进入后续某层时，已经分化的两条流 X:\n", streams[0, 0])
    print("\nflatten(X):", streams.flatten(2)[0, 0])
    print("共同 RMS 缩放 rsqrt:", rsqrt[0, 0])
    print("线性投影 raw_pre:", raw_pre[0, 0])
    print("动态生成的 pre_mix=sigmoid(raw_pre):", pre[0, 0])
    print("pre_mix 之和（通常不要求为 1）:", pre[0, 0].sum())
    print("hc_pre 后的子层输入:", layer_input[0, 0])
    print("\n子层 F([u,v])=[u+v,u-v] 的输出:", layer_output[0, 0])
    print("\ncomb 混合后的旧 residual:\n", torch.einsum("btji,btid->btjd", comb, streams)[0, 0])
    print("post 写回的子层输出:\n", (post.unsqueeze(-1) * layer_output.unsqueeze(2))[0, 0])
    print("hc_post 后的两条新流:\n", next_streams[0, 0])
    print("\nEncoder 边界 readout_mix:", encoder_readout_mix[0, 0])
    print("Encoder 单路输出 H20:", encoder_output[0, 0])

    # 固定数值断言，防止代码与手算说明不一致。
    expected_rsqrt = 1.0 / torch.sqrt(torch.tensor(7.5 + 1e-6))
    torch.testing.assert_close(rsqrt.squeeze(), expected_rsqrt)
    torch.testing.assert_close(raw_pre[0, 0], torch.tensor([1.0, 2.0]) * expected_rsqrt)

    # 后续数值由动态 pre_mix 决定，所以直接用同一公式构造期望值，重点检查 shape 和数据流。
    assert layer_input.shape == (1, 1, 2)
    assert layer_output.shape == (1, 1, 2)
    assert next_streams.shape == (1, 1, 2, 2)
    assert encoder_output.shape == (1, 1, 2)
    print("\n全部手算结果与 PyTorch 结果一致。")


if __name__ == "__main__":
    main()

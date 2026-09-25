"""可读的小规模 DeepSeek-V4.1 风格局部 SWA 教学实现。

只复现官方参考模型中的纯 SWA 路径，不复现量化后的 TileLang kernel。
不包含 CSA2 全局 KV、mHC、MoE、张量并行或 Bounded Replay。

形状记号：B 为批量大小，T 为本次 token 数，H 为 Q 头数，Dh 为每头维度；
W 为局部窗口宽度，K 为选中的 KV 槽位数，G 为输出投影组数，
Rq/Ro 为低秩投影宽度。输入和最终输出都是 [B,T,dim]。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SWAConfig:
    """使用小维度便于观察；真正的 V4.1 模型规模远大于此。"""
    dim: int = 16
    n_heads: int = 4
    head_dim: int = 8
    q_rank: int = 12
    rope_dim: int = 4
    output_groups: int = 2
    output_rank: int = 6
    window_size: int = 2
    rope_theta: float = 10_000.0

    def __post_init__(self) -> None:
        if self.rope_dim < 0 or self.rope_dim > self.head_dim or self.rope_dim % 2:
            raise ValueError("rope_dim 必须是偶数，且不能超过 head_dim")
        if self.n_heads % self.output_groups:
            raise ValueError("n_heads 必须能被 output_groups 整除")
        if min(self.dim, self.n_heads, self.head_dim, self.q_rank,
               self.output_groups, self.output_rank, self.window_size) <= 0:
            raise ValueError("所有维度和 window_size 都必须为正数")


class RMSNorm(nn.Module):
    """只在最后一维归一化，不改变前面的 batch、token 等维度。"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [...,D] 先沿最后一维求得 [...,1] 缩放系数，广播相乘后仍为 [...,D]。
        # 归一化计算使用 float32，结果再转回输入 dtype。
        scale = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale * self.weight.float()).to(x.dtype)


def rotate_tail(x: torch.Tensor, positions: torch.Tensor, rope_dim: int,
                theta: float, inverse: bool = False) -> torch.Tensor:
    """对最后 rope_dim 个特征施加 RoPE，整体形状保持不变。

    输入可以是 [B,T,D] 或 [B,T,H,D]。相邻两个特征构成二维平面，
    旋转角度由 token 的绝对位置决定；inverse=True 表示反向旋转。
    """
    if rope_dim == 0:
        return x
    # [R/2] 频率与 [T] 绝对位置广播，得到 [T,R/2] 旋转角。
    freqs = theta ** (-torch.arange(0, rope_dim, 2, device=x.device).float() / rope_dim)
    angles = positions.float()[:, None] * freqs[None, :]
    if inverse:
        angles = -angles
    # Q 广播为 [1,T,1,R/2]，额外的 1 对应 H；KV 无头维，为 [1,T,R/2]。
    broadcast = (1, x.size(1)) + (1,) * (x.ndim - 3) + (rope_dim // 2,)
    cos, sin = angles.cos().view(broadcast), angles.sin().view(broadcast)
    # [...,R] -> [...,R/2,2]，按相邻特征成对旋转，再恢复为 [...,R]。
    pairs = x[..., -rope_dim:].reshape(*x.shape[:-1], rope_dim // 2, 2)
    left, right = pairs[..., 0].float(), pairs[..., 1].float()
    rotated = torch.stack((left * cos - right * sin,
                           left * sin + right * cos), dim=-1).flatten(-2)
    return torch.cat((x[..., :-rope_dim], rotated.to(x.dtype)), dim=-1)


def window_indices(batch: int, length: int, window: int, start_pos: int,
                   device: torch.device) -> torch.Tensor:
    """为每个 query 选择局部 KV 位置；-1 表示空槽或不允许关注的位置。

    prefill 返回 [B,T,min(T,W)]，索引指向完整 prompt 的 KV 张量。
    decode 返回 [B,1,W]，索引指向固定长度的环形缓存槽位，
    而不是 token 的绝对位置。
    """
    if start_pos == 0:
        width = min(length, window)
        # end:[T,1] 是每个 query 的位置；偏移量宽度为 min(T,W)。
        # 第 i 行从 max(0,i-W+1) 开始取局部窗口。
        end = torch.arange(length, device=device)[:, None]
        idx = (end - window + 1).clamp_min(0) + torch.arange(width, device=device)
        # 超过 query 位置 i 的槽位属于未来 token，标为无效。
        idx = idx.masked_fill(idx > end, -1)
    else:
        if length != 1:
            raise ValueError("此 demo 只支持 prefill 后逐 token decode")
        # 写入当前 token 的 KV 后，环形缓存中的 W 个槽位可供选择。
        idx = torch.arange(window, device=device)[None, :]
        if start_pos < window:
            # 序列刚开始时环形缓存尚未填满，需要屏蔽空槽。
            idx = idx.masked_fill(idx >= start_pos + 1, -1)
    # 同一位置规则复制到 B 个相互独立的样本。
    return idx[None].expand(batch, -1, -1).contiguous()


def sparse_attention(q: torch.Tensor, kv: torch.Tensor,
                     indices: torch.Tensor, sink: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """先 gather 局部 KV，再并行计算所有 query 和 Q 头的注意力。

    输入 q:[B,T,H,Dh]、kv:[B,S,Dh]、indices:[B,T,K]、sink:[H]。
    输出为 [B,T,H,Dh]，局部权重为 [B,T,H,K]。这里 K<=W，
    不构造完整的 [B,H,T,S] 分数矩阵。
    """
    batch, length, heads, head_dim = q.shape
    # gather 不能读取 -1 位置，先临时替换为 0；随后在 softmax 前屏蔽，
    # 因而这些无效位置最终贡献为零。
    safe_indices = indices.clamp_min(0)
    # kv:[B,S,Dh] -> 虚拟扩展为 [B,T,S,Dh] -> gather 成 [B,T,K,Dh]。
    # expand 只是视图；真正物化的是每个 query 选中的 K 个位置。
    source = kv[:, None].expand(batch, length, -1, -1)
    selected = source.gather(2, safe_indices[..., None].expand(-1, -1, -1, head_dim))
    # Q 与选中的 KV 沿 Dh 做点积；H 个 Q 头共享同一组选中 KV。
    # scores:[B,T,H,K]，T 是 query 位置，K 是候选 KV 位置。
    scores = torch.einsum("bthd,btkd->bthk", q, selected) / math.sqrt(head_dim)
    scores = scores.masked_fill(indices[:, :, None, :] < 0, -torch.inf)

    # sink 是每个头额外的 softmax logit，但对应的 value 为零。
    # 它参与分母，因此 K 个可见 KV 的权重之和可能小于 1。
    sink_scores = sink[None, None, :, None].expand(batch, length, heads, 1)
    # 拼接 sink 后为 [B,T,H,K+1]；归一化分母是 [B,T,H,1]，
    # 计算权重时再沿 K 广播。
    log_denom = torch.logsumexp(torch.cat((scores, sink_scores), dim=-1), dim=-1, keepdim=True)
    weights = torch.exp(scores - log_denom)  # [B,T,H,K]
    # 沿 K 加权求和；H 个头各得到 Dh 维输出，总形状 [B,T,H,Dh]。
    output = torch.einsum("bthk,btkd->bthd", weights, selected)
    return output, weights


class LocalSWA(nn.Module):
    """单层纯局部 SWA，接近 V4.1 encoder 前几层的 attention 路径。

真实 Transformer block 还包含残差混合、归一化与 MoE；
这里用两个独立实例展示“第二层读取第一层输出”的数据依赖。
    """
    def __init__(self, config: SWAConfig):
        super().__init__()
        self.config = config
        c = config
        # Q 路径：[B,T,dim] -> [B,T,Rq] -> [B,T,H*Dh]。
        self.wq_a = nn.Linear(c.dim, c.q_rank, bias=False)
        self.q_norm = RMSNorm(c.q_rank)
        self.wq_b = nn.Linear(c.q_rank, c.n_heads * c.head_dim, bias=False)
        # 局部 KV 路径：每个 token 生成一份 [B,T,Dh]，由 H 个 Q 头共享。
        self.wkv = nn.Linear(c.dim, c.head_dim, bias=False)
        self.kv_norm = RMSNorm(c.head_dim)
        self.attn_sink = nn.Parameter(torch.zeros(c.n_heads))
        # H 个头分成 G 组；每组将 (H/G)*Dh 压缩到 Ro。
        heads_per_group = c.n_heads // c.output_groups
        self.wo_a = nn.Parameter(torch.empty(c.output_groups, c.output_rank,
                                            heads_per_group * c.head_dim))
        nn.init.xavier_uniform_(self.wo_a)
        self.wo_b = nn.Linear(c.output_groups * c.output_rank, c.dim, bias=False)
        # 仅在运行时保留的局部状态：[B,W,Dh]；每层各有独立环形缓存。
        self.window_kv_cache: torch.Tensor | None = None
        self.next_pos = 0

    def reset_cache(self) -> None:
        """开始处理无关序列前，丢弃上一段序列的局部 KV。"""
        self.window_kv_cache = None
        self.next_pos = 0

    def _window_kv(self, x: torch.Tensor, positions: torch.Tensor,
                   start_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        """生成局部 KV，并按阶段提供完整 prompt KV 或 decode 环形缓存。

        prefill：[B,T,dim] -> attention_kv:[B,T,Dh]，索引 [B,T,K]。
        decode：[B,1,dim] -> attention_kv:[B,W,Dh]，索引 [B,1,W]。
        """
        batch, length, _ = x.shape
        c = self.config
        kv = self.kv_norm(self.wkv(x))  # [B,T,Dh]，不是每个 Q 头各有一份 KV
        # 在 attention 和缓存写入之前，对尾部特征施加绝对位置 RoPE。
        kv = rotate_tail(kv, positions, c.rope_dim, c.rope_theta)

        if start_pos == 0:
            # prefill 通过稀疏索引读取完整的 [B,T,Dh] prompt KV。
            # 同时把最后 W 个 token 写入 [B,W,Dh] 环形缓存，留给之后 decode。
            # 写缓存与本层所有 query 的并行 attention 是两件事。
            cache = kv.new_zeros(batch, c.window_size, c.head_dim)
            tail_start = max(0, length - c.window_size)
            slots = torch.arange(tail_start, length, device=x.device) % c.window_size
            cache[:, slots] = kv[:, tail_start:]
            self.window_kv_cache = cache
            attention_kv = kv  # prefill 时可按索引访问整个 prompt 的 KV
        else:
            if self.window_kv_cache is None or start_pos != self.next_pos:
                raise ValueError("decode 必须紧接已完成的 prefill 或上一步 decode")
            if self.window_kv_cache.size(0) != batch:
                raise ValueError("batch size 已变化，请先重置缓存")
            # 绝对位置 p 映射到槽位 p%W；环形缓存填满后将覆盖最旧的 KV。
            # 当前 token 的 KV 必须先写入，才能参加本轮注意力。
            self.window_kv_cache[:, start_pos % c.window_size] = kv[:, 0]
            attention_kv = self.window_kv_cache

        # 索引指向 attention_kv；prefill 与 decode 的 S 不同，
        # 但送入 sparse_attention 时始终是 [B,S,Dh]。
        indices = window_indices(batch, length, c.window_size, start_pos, x.device)
        self.next_pos = start_pos + length
        return attention_kv, indices

    def forward(self, x: torch.Tensor, start_pos: int = 0,
                return_trace: bool = False):
        """运行一层，返回 [B,T,dim]，可选返回便于观察的中间张量。

        start_pos=0 表示并行处理整个 prompt；start_pos>0 表示
        利用本层缓存处理恰好一个新 token。
        """
        batch, length, dim = x.shape
        c = self.config
        if dim != c.dim or length == 0:
            raise ValueError("x 必须为 [B,T,dim]，且 T>0")
        if start_pos and length != 1:
            raise ValueError("仅支持完整 prefill 或逐 token decode")
        # [T] 绝对位置；prefill 为 0..T-1，decode 则接在已缓存前缀之后。
        positions = torch.arange(start_pos, start_pos + length, device=x.device)

        # 第 1 步：低秩投影 Q，再拆成 H 个头。与常见的 [B,H,T,Dh] 排列不同，
        # 这里保持 [B,T,H,Dh]，与官方稀疏 kernel 的接口对应。
        qr = self.q_norm(self.wq_a(x))  # [B,T,Rq]
        q = self.wq_b(qr).unflatten(-1, (c.n_heads, c.head_dim))  # [B,T,H,Dh]
        # 第 2 步：只旋转最后 rope_dim 个特征，Q 整体形状不变。
        q = rotate_tail(q, positions, c.rope_dim, c.rope_theta)

        # 第 3 步：每个 token 生成一份共享局部 KV，并得到 [B,T,K] 索引。
        kv, indices = self._window_kv(x, positions, start_pos)
        # 第 4 步：每个 query gather K<=W 个 KV；输出仍保留 H 个头。
        head_output, weights = sparse_attention(q, kv, indices, self.attn_sink)
        # 第 5 步：分组输出投影前，对输出尾部的 RoPE 做逆旋转。
        head_output = rotate_tail(head_output, positions, c.rope_dim,
                                  c.rope_theta, inverse=True)

        # 第 6 步：[B,T,H,Dh] -> [B,T,G,(H/G)*Dh] -> [B,T,G,Ro]
        # -> [B,T,G*Ro] -> [B,T,dim]；不是简单地将 H*Dh 一次投影到 dim。
        grouped = head_output.reshape(batch, length, c.output_groups, -1)
        compressed = torch.einsum("btgi,gri->btgr", grouped, self.wo_a)
        output = self.wo_b(compressed.flatten(2))  # [B,T,dim]
        if not return_trace:
            return output
        return output, {
            "qr": qr, "q": q, "kv": kv, "indices": indices,
            "weights": weights, "head_output": head_output,
            "grouped": grouped, "compressed": compressed,
            "cache": self.window_kv_cache.clone(),
        }

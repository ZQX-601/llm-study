"""CSA2 教学实现：序列压缩、全局索引、跨层共享和局部 SWA。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from swa_attention import RMSNorm, rotate_tail, sparse_attention, window_indices


@dataclass
class CSA2Config:
    """小尺寸配置，便于在本地打印并逐步调试。"""

    dim: int = 16
    head_dim: int = 8
    num_heads: int = 4
    index_dim: int = 4
    num_index_heads: int = 2
    compress_ratio: int = 2
    global_topk: int = 2
    window_size: int = 2
    output_groups: int = 2
    output_rank: int = 8
    rope_dim: int = 4
    rope_theta: float = 10_000.0


@dataclass
class SharedCSA2State:
    """显式表示官方实现中由相邻层共享的三类状态。"""

    main_kv: Optional[Tensor] = None       # [B, Tc, Dh]
    index_k: Optional[Tensor] = None       # [B, Tc, Di]
    topk_indices: Optional[Tensor] = None  # [B, T, K]
    compress_ratio: int = 1


class Compressor(nn.Module):
    """把连续 m 个 token 压成一个共享主 KV。

    X:[B,T,d] 经两个投影变成 value/gate:[B,T,Dh]，再重排为
    [B,Tc,m,Dh]。gate 沿 m 维做 softmax，所以每个 Dh 特征都有
    自己的一组组内 token 权重，而不是整个 token 共用一个标量权重。
    """

    def __init__(self, config: CSA2Config) -> None:
        super().__init__()
        self.ratio = config.compress_ratio
        self.head_dim = config.head_dim
        self.wkv = nn.Linear(config.dim, config.head_dim, bias=False)
        self.wgate = nn.Linear(config.dim, config.head_dim, bias=False) if self.ratio > 1 else None
        self.norm = RMSNorm(config.head_dim)

    def forward_prefill(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        batch, seq_len, _ = x.shape
        value = self.wkv(x)  # [B,T,d] -> [B,T,Dh]
        if self.ratio == 1:
            return self.norm(value), {"value": value, "weights": torch.ones_like(value)}

        # prefill 只把完整的 m-token 分组变成主 KV。
        complete_len = (seq_len // self.ratio) * self.ratio
        value = value[:, :complete_len]
        gate = self.wgate(x[:, :complete_len])
        grouped_value = value.reshape(batch, complete_len // self.ratio, self.ratio, self.head_dim)
        grouped_gate = gate.reshape(batch, complete_len // self.ratio, self.ratio, self.head_dim)
        weights = torch.softmax(grouped_gate.float(), dim=2).to(value.dtype)
        latent = self.norm((grouped_value * weights).sum(dim=2))  # [B,Tc,Dh]
        return latent, {
            "value": value,
            "grouped_value": grouped_value,
            "grouped_gate": grouped_gate,
            "weights": weights,
        }

    def forward_decode(
        self,
        x_one: Tensor,
        position: int,
        value_state: Tensor,
        gate_state: Tensor,
    ) -> Tuple[Optional[Tensor], Tensor, Tensor]:
        """增量路径：收满 m 个 token 才产生一个新的主 KV。"""

        slot = position % self.ratio
        value_state[:, slot] = self.wkv(x_one).squeeze(1)
        if self.ratio == 1:
            return self.norm(value_state[:, :1]), value_state, gate_state
        gate_state[:, slot] = self.wgate(x_one).squeeze(1)
        if slot != self.ratio - 1:
            return None, value_state, gate_state
        weights = torch.softmax(gate_state.float(), dim=1).to(value_state.dtype)
        latent = self.norm((value_state * weights).sum(dim=1, keepdim=True))
        return latent, value_state, gate_state


class Indexer(nn.Module):
    """在小维度索引空间里，为每个 query 选择全局 Top-K 主 KV。"""

    def __init__(self, config: CSA2Config, create_key: bool) -> None:
        super().__init__()
        self.num_heads = config.num_index_heads
        self.index_dim = config.index_dim
        self.topk = config.global_topk
        self.rope_dim = min(config.rope_dim, config.index_dim)
        self.rope_theta = config.rope_theta
        self.wq = nn.Linear(config.dim, config.num_index_heads * config.index_dim, bias=False)
        self.wk = nn.Linear(config.head_dim, config.index_dim, bias=False) if create_key else None
        self.weight_proj = nn.Linear(config.dim, config.num_index_heads, bias=False)
        self.q_norm = RMSNorm(config.index_dim)
        self.k_norm = RMSNorm(config.index_dim)

    def make_key(self, unrotated_main_kv: Tensor, compress_ratio: int) -> Tensor:
        if self.wk is None:
            raise RuntimeError("当前 Indexer 不负责创建索引 K")
        index_k = self.k_norm(self.wk(unrotated_main_kv))  # [B,Tc,Dh] -> [B,Tc,Di]
        # 第 j 个压缩 latent 代表第 j*m 个位置；Indexer K 使用自己的 RoPE。
        compressed_positions = torch.arange(
            index_k.shape[1], device=index_k.device
        ) * compress_ratio
        return rotate_tail(index_k, compressed_positions, self.rope_dim, self.rope_theta)

    def select(
        self,
        x: Tensor,
        index_k: Tensor,
        compress_ratio: int,
        local_kv_len: int,
    ) -> Tuple[Tensor, Tensor]:
        batch, seq_len, _ = x.shape
        compressed_len = index_k.shape[1]
        index_q = self.q_norm(
            self.wq(x).view(batch, seq_len, self.num_heads, self.index_dim)
        )  # [B,T,Hi,Di]
        index_q = rotate_tail(
            index_q,
            torch.arange(seq_len, device=x.device),
            self.rope_dim,
            self.rope_theta,
        )

        # [B,T,Hi,Di] x [B,Tc,Di] -> [B,T,Hi,Tc]
        raw_scores = torch.einsum("bthd,bsd->bths", index_q.float(), index_k.float())
        raw_scores = torch.relu(raw_scores)
        head_weights = self.weight_proj(x).float()  # [B,T,Hi]
        scores = (raw_scores * head_weights.unsqueeze(-1)).sum(dim=2)  # [B,T,Tc]

        # 第 j 个主 KV 来自完整的 m-token 分组，组内最后一个 token 到达后才可见。
        visible_lengths = (
            torch.arange(1, seq_len + 1, device=x.device) // compress_ratio
        ).clamp(max=compressed_len)
        positions = torch.arange(compressed_len, device=x.device)
        visible = positions.view(1, 1, -1) < visible_lengths.view(1, -1, 1)
        masked_scores = scores.masked_fill(~visible, -torch.inf)

        k = min(self.topk, compressed_len)
        indices = masked_scores.topk(k=k, dim=-1).indices.sort(dim=-1).values
        valid = indices < visible_lengths.view(1, -1, 1)
        # 主 KV 拼在 T 个局部 KV 后，因此全局索引需要增加 T 的偏移量。
        indices = torch.where(valid, indices + local_kv_len, -1)
        return indices.expand(batch, -1, -1), scores


class GroupedOutputProjection(nn.Module):
    """接近官方结构的分组低秩输出投影。"""

    def __init__(self, config: CSA2Config) -> None:
        super().__init__()
        if config.num_heads % config.output_groups:
            raise ValueError("num_heads 必须能被 output_groups 整除")
        self.groups = config.output_groups
        self.heads_per_group = config.num_heads // config.output_groups
        group_dim = self.heads_per_group * config.head_dim
        self.down = nn.ModuleList(
            [nn.Linear(group_dim, config.output_rank, bias=False) for _ in range(self.groups)]
        )
        self.up = nn.ModuleList(
            [nn.Linear(config.output_rank, config.dim, bias=False) for _ in range(self.groups)]
        )

    def forward(self, heads: Tensor) -> Tensor:
        batch, seq_len, _, head_dim = heads.shape
        grouped = heads.reshape(batch, seq_len, self.groups, self.heads_per_group * head_dim)
        output = torch.zeros(batch, seq_len, self.up[0].out_features, device=heads.device, dtype=heads.dtype)
        for group_id in range(self.groups):
            output = output + self.up[group_id](self.down[group_id](grouped[:, :, group_id]))
        return output


class CSA2Layer(nn.Module):
    """教学版 CSA2 层：full、reindex、reuse 都有自己的 Q 和局部 KV。"""

    def __init__(self, config: CSA2Config, mode: str) -> None:
        super().__init__()
        if mode not in {"full", "reindex", "reuse"}:
            raise ValueError(f"未知模式：{mode}")
        self.config = config
        self.mode = mode
        self.q_proj = nn.Linear(config.dim, config.num_heads * config.head_dim, bias=False)
        self.local_kv_proj = nn.Linear(config.dim, config.head_dim, bias=False)
        self.q_norm = RMSNorm(config.head_dim)
        self.local_kv_norm = RMSNorm(config.head_dim)
        self.compressor = Compressor(config) if mode == "full" else None
        self.indexer = Indexer(config, create_key=(mode == "full")) if mode != "reuse" else None
        self.output_proj = GroupedOutputProjection(config)
        self.attention_sink = nn.Parameter(torch.zeros(config.num_heads))

    def forward_prefill(
        self, x: Tensor, shared: SharedCSA2State
    ) -> Tuple[Tensor, SharedCSA2State, Dict[str, Tensor]]:
        batch, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device)

        q = self.q_norm(
            self.q_proj(x).view(batch, seq_len, self.config.num_heads, self.config.head_dim)
        )
        q = rotate_tail(
            q, positions, self.config.rope_dim, self.config.rope_theta
        )  # [B,T,H,Dh]
        local_kv = rotate_tail(
            self.local_kv_norm(self.local_kv_proj(x)),
            positions,
            self.config.rope_dim,
            self.config.rope_theta,
        )  # [B,T,Dh]
        compressor_trace: Dict[str, Tensor] = {}
        index_scores: Optional[Tensor] = None

        if self.mode == "full":
            # Full 同时发布主 KV、索引 K 和第一次 Top-K。
            unrotated_main_kv, compressor_trace = self.compressor.forward_prefill(x)
            shared.index_k = self.indexer.make_key(
                unrotated_main_kv, self.config.compress_ratio
            )
            compressed_positions = torch.arange(
                0,
                unrotated_main_kv.shape[1] * self.config.compress_ratio,
                self.config.compress_ratio,
                device=x.device,
            )
            shared.main_kv = rotate_tail(
                unrotated_main_kv,
                compressed_positions,
                self.config.rope_dim,
                self.config.rope_theta,
            )
            shared.compress_ratio = self.config.compress_ratio
            shared.topk_indices, index_scores = self.indexer.select(
                x, shared.index_k, shared.compress_ratio, local_kv_len=seq_len
            )
        elif self.mode == "reindex":
            self._require_shared_state(shared, need_index_key=True)
            # Reindex 不建主 KV，只用本层 query 重新评分并覆盖 Top-K。
            shared.topk_indices, index_scores = self.indexer.select(
                x, shared.index_k, shared.compress_ratio, local_kv_len=seq_len
            )
        else:
            # Reuse 不运行 Indexer，直接读取最近一次 Top-K。
            self._require_shared_state(shared, need_index_key=False)

        all_kv = torch.cat([local_kv, shared.main_kv], dim=1)  # [B,T+Tc,Dh]
        local_indices = window_indices(
            batch, seq_len, start_pos=0,
            window=self.config.window_size, device=x.device,
        )  # [B,T,W]
        all_indices = torch.cat([local_indices, shared.topk_indices], dim=-1)  # [B,T,W+K]
        attended, attention_weights = sparse_attention(q, all_kv, all_indices, self.attention_sink)
        attended = rotate_tail(
            attended,
            positions,
            self.config.rope_dim,
            self.config.rope_theta,
            inverse=True,
        )
        update = self.output_proj(attended)  # [B,T,H,Dh] -> [B,T,d]

        # 真模型使用 mHC；教学实现使用普通残差，只聚焦 CSA2 数据流。
        output = x + update
        trace = {
            "q": q,
            "local_kv": local_kv,
            "main_kv": shared.main_kv,
            "index_k": shared.index_k,
            "local_indices": local_indices,
            "global_indices": shared.topk_indices,
            "all_indices": all_indices,
            "attention_weights": attention_weights,
            "update": update,
            **{f"compressor_{key}": value for key, value in compressor_trace.items()},
        }
        if index_scores is not None:
            trace["index_scores"] = index_scores
        return output, shared, trace

    @staticmethod
    def _require_shared_state(shared: SharedCSA2State, need_index_key: bool) -> None:
        if shared.main_kv is None or shared.topk_indices is None:
            raise RuntimeError("Reindex/Reuse 前必须先运行 Full 层")
        if need_index_key and shared.index_k is None:
            raise RuntimeError("Reindex 层需要 Full 层创建的 index_k")


def build_demo_stack(config: CSA2Config) -> nn.ModuleList:
    """最小跨层示例：重算、复用、再索引、再复用。"""

    return nn.ModuleList([
        CSA2Layer(config, "full"),
        CSA2Layer(config, "reuse"),
        CSA2Layer(config, "reindex"),
        CSA2Layer(config, "reuse"),
    ])

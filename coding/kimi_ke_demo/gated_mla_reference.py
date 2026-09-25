#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gated MLA 参考实现（带逐行 shape 标注 + 真实 KV cache）

⚠️ 这不是 K3 的官方源码。拼装来源：
    - MLA 骨架  ← flash-linear-attention 的 fla/layers/mla.py
    - 门控部分  ← Qwen《Gated Attention》(arXiv:2505.06708) 的 Y' = Y ⊙ σ(X W_θ)
    - cache 设计 ← DeepSeek-V2 MLA：只存 c_kv 与 k_rope
  与 K3 §2.1.2 的真实实现可能有细节差异。

尺寸采用 DeepSeek-V3 的 MLA 配置（K3 的确切尺寸未公开）：
    d = 7168   H = 128   q_lora_rank = 1536
    D_nope = 128   D_rope = 64   D_qk = 192
    D_c(kv_lora_rank) = 512   D_v = 128

cache 每 token 每层只存两样：
    c_kv   : [.., D_c]    = 512
    k_rope : [.., D_rope] = 64
    合计 576，而不是展开后的 k、v（40960）。约 71 倍。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):                       # x: [..., dim]
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


def apply_rope(x, positions):
    """对最后 D_rope 维做位置相关旋转。x: [..., D_rope]"""
    d = x.shape[-1]
    half = d // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=x.device, dtype=torch.float32) / half
    )                                                       # [half]
    angles = positions.float()[:, None] * freqs[None, :]     # [T, half]
    cos, sin = angles.cos(), angles.sin()
    while cos.dim() < x.dim():
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ===========================================================================
# KV cache：只存 c_kv 和 k_rope
# ===========================================================================
class MLACache:
    """按层预分配两块 buffer。

    形状（每层）：
        c_kv   : [max_batch, max_seq, D_c]      512 维
        k_rope : [max_batch, max_seq, D_rope]    64 维

    seq_len 是所有层共享的写指针：每生成一个 token，
    每一层都往里追加自己那一份 c_kv / k_rope。
    """

    def __init__(self, num_layers, max_batch, max_seq, d_c, d_rope,
                 device="cpu", dtype=torch.float32):
        self.c_kv = torch.zeros(num_layers, max_batch, max_seq, d_c,
                                device=device, dtype=dtype)      # [L, B, S, 512]
        self.k_rope = torch.zeros(num_layers, max_batch, max_seq, d_rope,
                                  device=device, dtype=dtype)    # [L, B, S,  64]
        self.seq_len = 0                                         # 写指针（标量）

    def append(self, layer_idx, c_kv_new, k_rope_new):
        """c_kv_new: [B, T_new, D_c]   k_rope_new: [B, T_new, D_rope]"""
        T_new = c_kv_new.shape[1]
        s = self.seq_len
        e = s + T_new
        self.c_kv[layer_idx, :, s:e, :] = c_kv_new               # 写 c
        self.k_rope[layer_idx, :, s:e, :] = k_rope_new           # 写 k_rope
        return s, e

    def read(self, layer_idx, end=None):
        """返回这一层截至 end 的全部历史。"""
        end = self.seq_len if end is None else end
        return (self.c_kv[layer_idx, :, :end, :],      # [B, end, 512]
                self.k_rope[layer_idx, :, :end, :])    # [B, end,  64]

    def advance(self, T_new):
        self.seq_len += T_new


class GatedMLA(nn.Module):
    def __init__(self, hidden_size=7168, num_heads=128, q_lora_rank=1536,
                 qk_nope_head_dim=128, qk_rope_head_dim=64, kv_lora_rank=512,
                 v_head_dim=128, gate_granularity="headwise", norm_eps=1e-6):
        super().__init__()
        self.d, self.H = hidden_size, num_heads
        self.D_nope, self.D_rope = qk_nope_head_dim, qk_rope_head_dim
        self.D_qk = self.D_nope + self.D_rope                    # 192
        self.D_c, self.D_v = kv_lora_rank, v_head_dim            # 512, 128
        self.gate_granularity = gate_granularity

        self.q_a_proj = nn.Linear(self.d, q_lora_rank, bias=False)          # [d]→[1536]
        self.q_a_norm = RMSNorm(q_lora_rank, eps=norm_eps)
        self.q_b_proj = nn.Linear(q_lora_rank, self.H * self.D_qk, bias=False)  # →[24576]

        self.kv_a_proj = nn.Linear(self.d, self.D_c, bias=False)            # W_DKV: →[512]
        self.kv_a_norm = RMSNorm(self.D_c, eps=norm_eps)
        self.kv_b_proj = nn.Linear(self.D_c,
                                   self.H * (self.D_nope + self.D_v), bias=False)  # →[32768]

        self.k_rope_proj = nn.Linear(self.d, self.D_rope, bias=False)       # →[64]
        self.o_proj = nn.Linear(self.H * self.D_v, self.d, bias=False)      # →[d]

        gate_out = self.H if gate_granularity == "headwise" else self.H * self.D_v
        self.gate_proj = nn.Linear(self.d, gate_out, bias=False)
        self.scaling = self.D_qk ** -0.5

    # ------------------------------------------------------------------
    # 把融合的 kv_b_proj 拆回 W_UK / W_UV，供吸收路径使用
    #   kv_b_proj.weight: [H*(D_nope+D_v), D_c] = [32768, 512]
    #   → [H, D_nope+D_v, D_c] = [128, 256, 512]
    # ------------------------------------------------------------------
    def split_kv_weight(self):
        W = self.kv_b_proj.weight.view(self.H, self.D_nope + self.D_v, self.D_c)
        W_UK = W[:, :self.D_nope, :]        # [H, 128, 512]  用于从 c_kv 还原 k_nope
        W_UV = W[:, self.D_nope:, :]        # [H, 128, 512]  用于从 c_kv 还原 v
        return W_UK, W_UV

    # ==================================================================
    # Prefill：一次吃下整个 prompt，causal 注意力
    # ==================================================================
    def forward_prefill(self, x, cache=None, layer_idx=0):
        B, T, _ = x.shape                                        # x: [B, T, 7168]
        pos = torch.arange(T, device=x.device)                   # [T]

        # ---- 1) q ----
        q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))       # [B, T, 24576]
        q = q.view(B, T, self.H, self.D_qk)                      # [B, T, 128, 192]
        q_nope, q_rope = q.split([self.D_nope, self.D_rope], -1) # [B,T,128,128], [B,T,128,64]
        q_rope = apply_rope(q_rope, pos)                         # [B, T, 128, 64]

        # ---- 2) KV 压缩：得到要进 cache 的两样 ----
        c_kv = self.kv_a_norm(self.kv_a_proj(x))                 # [B, T, 512]  ← 进 cache
        kv = self.kv_b_proj(c_kv)                                # [B, T, 32768]
        kv = kv.view(B, T, self.H, self.D_nope + self.D_v)       # [B, T, 128, 256]
        k_nope, v = kv.split([self.D_nope, self.D_v], -1)        # [B,T,128,128] ×2

        k_rope = self.k_rope_proj(x)                             # [B, T, 64]   ← 进 cache
        k_rope = apply_rope(k_rope, pos)                         # [B, T, 64]

        # ---- 3) 写入 cache（只写 c_kv 和 k_rope）----
        if cache is not None:
            cache.append(layer_idx, c_kv, k_rope)                # 内部写 [B,T,512] 和 [B,T,64]
            cache.advance(T)

        # ---- 4) 拼出完整 k 做 causal 注意力（prefill 一次性展开是可以接受的）----
        k_rope_h = k_rope.unsqueeze(2).expand(B, T, self.H, self.D_rope)  # [B,T,128,64]
        q_full = torch.cat([q_nope, q_rope], -1)                 # [B, T, 128, 192]
        k_full = torch.cat([k_nope, k_rope_h], -1)               # [B, T, 128, 192]
        o = F.scaled_dot_product_attention(
            q_full.transpose(1, 2),                              # [B, 128, T, 192]
            k_full.transpose(1, 2),                              # [B, 128, T, 192]
            v.transpose(1, 2),                                   # [B, 128, T, 128]
            is_causal=True, scale=self.scaling,
        ).transpose(1, 2)                                        # [B, T, 128, 128]

        return self._gate_and_project(o, x)                      # [B, T, 7168]

    # ==================================================================
    # Decode：每步只处理 1 个新 token，全程只用 cache 里的 c_kv / k_rope
    # ==================================================================
    def forward_decode(self, x, cache, layer_idx=0):
        """x: [B, 1, d]，当前步的唯一 token。历史全部在 cache 里。"""
        B, T, _ = x.shape                                        # T == 1
        assert T == 1, "decode 一次只喂一个 token"

        # ---- 1) q（只算新 token 的）----
        q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))       # [B, 1, 24576]
        q = q.view(B, 1, self.H, self.D_qk)                      # [B, 1, 128, 192]
        q_nope, q_rope = q.split([self.D_nope, self.D_rope], -1) # [B,1,128,128], [B,1,128,64]
        q_rope = apply_rope(q_rope, torch.tensor([cache.seq_len]))  # [B, 1, 128, 64]

        # ---- 2) 只算要进 cache 的两样 ----
        c_kv = self.kv_a_norm(self.kv_a_proj(x))                 # [B, 1, 512]
        k_rope = self.k_rope_proj(x)                             # [B, 1, 64]
        k_rope = apply_rope(k_rope, torch.tensor([cache.seq_len]))  # [B, 1, 64]

        # ---- 3) 先写 cache，再读全量历史 ----
        cache.append(layer_idx, c_kv, k_rope)
        cache.advance(1)
        C_kv, K_rope = cache.read(layer_idx)                     # [B,S,512], [B,S,64]
        S = C_kv.shape[1]                                        # 当前总长度

        # ---- 4) ★ 矩阵吸收：不还原 k_nope、v，直接用 c_kv 算 ----
        W_UK, W_UV = self.split_kv_weight()                      # [H,128,512] ×2

        # q_nope 吸收 W_UK： q_abs[h] = W_UK[h]ᵀ q_nope[h]  ∈ R^{D_c}
        q_abs = torch.einsum('bthd,hdc->bthc', q_nope, W_UK)     # [B, 1, 128, 512]

        # nope 分数： (W_UK c_kv)·q_nope == q_abs · c_kv
        s_nope = torch.einsum('bthc,bsc->bhts', q_abs, C_kv)     # [B, 128, 1, S]
        # rope 分数：两个都已是 rope 后的向量，直接点积
        s_rope = torch.einsum('bthd,bsd->bhts', q_rope, K_rope)  # [B, 128, 1, S]
        scores = (s_nope + s_rope) * self.scaling                # [B, 128, 1, S]
        attn = scores.softmax(dim=-1)                            # [B, 128, 1, S]

        # 先对 latent 加权求和，最后才升维（W_UV 被推迟到最后）
        z = torch.einsum('bhts,bsc->bthc', attn, C_kv)           # [B, 1, 128, 512]
        o = torch.einsum('bthc,hvc->bthv', z, W_UV)              # [B, 1, 128, 128]

        return self._gate_and_project(o, x)                      # [B, 1, 7168]

    # ------------------------------------------------------------------
    def _gate_and_project(self, o, x):
        """o: [B,T,128,128]   x: [B,T,7168] → [B,T,7168]"""
        B, T = x.shape[0], x.shape[1]
        gate = torch.sigmoid(self.gate_proj(x))                  # headwise: [B, T, 128]
        if self.gate_granularity == "headwise":
            gate = gate.unsqueeze(-1)                            # [B, T, 128, 1]
        else:
            gate = gate.view(B, T, self.H, self.D_v)             # [B, T, 128, 128]
        o = o * gate                                             # [B, T, 128, 128]
        o = o.reshape(B, T, self.H * self.D_v)                   # [B, T, 16384]
        return self.o_proj(o)                                    # [B, T, 7168]

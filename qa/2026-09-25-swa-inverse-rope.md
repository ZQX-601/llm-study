# SWA attention 输出为什么要逆 RoPE

## 问题

既然 attention 前已经对 Q/K 做 RoPE，为什么 `LocalSWA.forward` 第 5 步还要对 `head_output` 调用 `rotate_tail(..., inverse=True)`？

## 讲解摘要

- 标准 attention 通常将 Q/K 旋转而 V 不旋转，因此输出不需要逆旋转。
- 当前 SWA 实现只有一份共享 `kv:[B,T,Dh]`，它在 `_window_kv` 中整体旋转，随后既用于打分的 K，也用于加权求和的 V。
- 对位置 `p` 的 query 和位置 `j` 的共享 KV，分数使用 `R_p q` 与 `R_j kv`，自然得到相对相位 `R_p^T R_j=R_{j-p}`。
- 加权输出在逆旋转前为 `Σ_j α_pj R_j kv_j`，仍处于各 KV 的绝对旋转坐标中。
- 再乘 `R_p^{-1}` 后变成 `Σ_j α_pj R_{j-p} kv_j`，将输出转到 query 的相对坐标系；自位置偏移为 0，历史位置保留相对偏移。
- 如果 K/V 分离且只旋转 K、V 保持未旋转，则不需要这一步。

## 来源映射

`coding/deepseek_demo/SOURCE_MAP.md` 将官方实现中的逆向 `apply_rotary_emb` 映射为本地 `rotate_tail(..., inverse=True)`。

## 与主线及薄弱点

该问题暴露的是“标准 Q/K-only RoPE”与“共享且被旋转的 KV 同时充当 K/V”之间的机制差异；后续代码阅读需持续区分 K、V 是否独立以及旋转发生在哪条数据路径。

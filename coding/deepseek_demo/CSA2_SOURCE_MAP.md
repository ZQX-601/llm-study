# CSA2 官方源码对照图

## 推荐阅读顺序

1. `csa2_attention.py` 的 `Compressor`
2. `Indexer.make_key` 和 `Indexer.select`
3. `CSA2Layer.forward_prefill` 的三个模式分支
4. `run_csa2_demo.py` 的四层 shape 输出
5. `test_csa2_attention.py` 的因果可见性和跨层复用测试

## 概念与本地实现

| CSA2 概念 | 本地实现 | 关键 shape |
|---|---|---|
| 序列压缩 | `Compressor.forward_prefill` | `[B,T,d] -> [B,Tc,Dh]` |
| 增量压缩 | `Compressor.forward_decode` | 每 `m` 个 token 产生一个 `[B,1,Dh]` |
| 索引 K | `Indexer.make_key` | `[B,Tc,Dh] -> [B,Tc,Di]` |
| 索引 Q | `Indexer.select` | `[B,T,d] -> [B,T,Hi,Di]` |
| 全局检索 | `Indexer.select` | `[B,T,Tc] -> [B,T,K]` |
| 局部 SWA | `window_indices` | `[B,T,W]` |
| 局部和全局合并 | `CSA2Layer.forward_prefill` | `[B,T,W+K]` |
| 跨层共享 | `SharedCSA2State` | 主 KV、索引 K、Top-K |

`Tc=floor(T/m)`；`m` 是压缩率；`Hi/Di` 是 Indexer 的头数和维度；
`W/K` 分别是局部窗口长度和全局 Top-K。

## Full / Reindex / Reuse

| 模式 | 新建主 KV | 新建 Index K | 重算 Top-K | 新建本层 Q | 新建本层局部 KV |
|---|---:|---:|---:|---:|---:|
| Full | 是 | 是 | 是 | 是 | 是 |
| Reindex | 否 | 否 | 是 | 是 | 是 |
| Reuse | 否 | 否 | 否 | 是 | 是 |

Reuse 不是复用整层注意力结果。它只复用全局主 KV 和已有 Top-K；本层的
Q、局部 KV、注意力权重和输出仍然重新计算。

## 有意省略的工程部分

- FP4 KV cache 量化与反量化
- Decoder 的层级 Indexer 候选池
- 张量并行、专家并行和通信
- mHC 残差连接
- 完整增量 KV cache 管理
- 训练辅助损失和高性能 kernel

因此本地代码用于验证算法语义和 shape，不用于复现官方吞吐或显存数据。

## 官方资料

- 技术报告：<https://arxiv.org/html/2609.19969v1>
- 官方推理源码：<https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py>

# DeepSeek V4.1 Flash：CSA2 与 Compressor

- 日期：2026-09-26
- 模式：QA 专题学习
- 状态：用户明确表示“CSA2 我学习好了”；记录为“用户自评掌握，待复习验证”，不等同于独立验收通过。
- 重要性：CSA2、KV Cache 压缩、Sparse Attention 和跨层缓存复用属于大模型架构与推理岗位需要掌握的前沿案例。

## 本次学习范围

1. CSA2 要同时控制长上下文的 KV Cache 存储量和注意力读取量。
2. 局部 SWA KV 保留近期细节；压缩后的全局 Main KV 提供远程信息。
3. Compressor 将连续 `m` 个 token 压缩为一个 latent，并区分 prefill 并行压缩与 decode 增量成组。
4. Indexer 在低维空间为每个 query 选择全局 Top-K Main KV。
5. Full、Reindex、Reuse 在层维共享 Main KV、Index K 或 Top-K，但每层仍计算自己的 Q、局部 SWA KV、注意力权重和输出。
6. 局部窗口索引和全局 Top-K 索引拼接后，进入同一次 Sparse Attention。

## 一句话主线

```text
Sparse Attention：解决每个 query 读取多少历史
Compressor：       解决需要长期保存多少历史
跨层复用：         解决多少层重复保存和重复检索同一批历史
```

CSA2 的完整读取路径：

```text
当前层 X:[B,T,d]
  ├─ 本层 Main Q:[B,T,H,Dh]
  ├─ 本层局部 SWA KV:[B,T,Dh]
  └─ 全局共享 Main KV:[B,Tc,Dh]
         ↑
     Compressor：Tc=floor(T/m)

Indexer：Index Q × Index K
      → scores:[B,T,Tc]
      → global Top-K indices:[B,T,K]

local indices:[B,T,W] + global indices:[B,T,K]
      → sparse indices:[B,T,W+K]
      → sparse attention
      → [B,T,H,Dh] → [B,T,d]
```

## Compressor 核心结论

Compressor 不是注意力本身，而是构造体积更小、可以被 Indexer 检索、可以跨层共享的长期上下文表示。

输入和输出：

```text
X:[B,T,d]
  ├─ value=W_kv(X)   -> [B,T,Dh]
  └─ gate=W_gate(X)  -> [B,T,Dh]

按连续 m 个 token 分组：
[B,T,Dh] -> [B,Tc,m,Dh]

weights=softmax(gate, dim=m)
latent=sum(weights * value, dim=m)
main_kv=RMSNorm(latent) -> [B,Tc,Dh]
```

门控是逐特征的：每个 `Dh` 特征都能从组内不同 token 选择不同权重，因此表达能力高于平均池化。Prefill 可同时处理所有完整分组；decode 使用 `kv_state/score_state` 保存未完成分组，每累计满 `m` 个 token 才发布一个新 Main KV。

## Full、Reindex、Reuse

| 模式 | 新建 Main KV | 新建 Index K | 重算 Top-K | 本层 Q 和 SWA KV |
| --- | ---: | ---: | ---: | ---: |
| Full | 是 | 是 | 是 | 是 |
| Reindex | 否 | 否 | 是 | 是 |
| Reuse | 否 | 否 | 否 | 是 |

Reuse 不是复用整层输出。它只跳过全局 KV 构建和 Indexer 检索，本层注意力仍然重新计算。Reindex 的价值在于：主 KV 可以继续共享，但随着层数加深，本层 hidden state 和 query 已变化，需要周期性重新选择相关位置。

## 教学源码

- [CSA2 核心实现](../coding/deepseek_demo/csa2_attention.py)
- [Prefill shape 演示](../coding/deepseek_demo/run_csa2_demo.py)
- [行为测试](../coding/deepseek_demo/test_csa2_attention.py)
- [官方源码对照与阅读顺序](../coding/deepseek_demo/CSA2_SOURCE_MAP.md)

教学实现覆盖：

- Compressor 的 prefill 与 decode 成组逻辑；
- Index Q/K、RoPE、因果可见性和 Top-K；
- Full → Reuse → Reindex → Reuse 跨层数据流；
- 本层 SWA KV 与共享 Main KV 的拼接；
- 分组低秩输出投影和普通残差。

有意省略：FP4 KV 量化、Decoder 层级候选池、分布式通信、高性能 kernel、完整增量缓存和 mHC。教学代码用于验证语义与 shape，不用于复现官方吞吐和显存数字。

## 验证证据与边界

- 用户证据：沿 SWA、Sparse Attention、完整 CSA2 数据流和 Compressor 连续追问后，明确表示 CSA2 学习完成。
- 代码证据：运行 Compressor shape/权重、decode 成组、Full/Reindex/Reuse 共享状态、压缩 KV 因果可见性四项测试，全部通过。
- 演示证据：`B=1,T=8,d=16,H=4,Dh=8,m=2,W=2,K=2` 的四层 prefill 示例运行通过。
- 独立验收：尚未进行；没有把自评掌握记为闭卷或实战通过。

## 复习安排

- 2026-09-29：给定 `B,T,d,H,Dh,m,W,K`，追踪 Compressor、Indexer 和 Sparse Attention 的全部 shape。
- 2026-10-03：解释第 `j` 个压缩 KV 何时对 query 可见，以及 prefill/decode 的分组边界。
- 2026-10-10：阅读 Full/Reindex/Reuse 伪代码，指出哪些状态新建、复用或覆盖。
- 2026-10-24：结合百万 token 场景，对比纯 SWA、普通稀疏注意力和 CSA2 的存储/计算职责。

## 一手资料

- [DeepSeek-V4.1-Flash 技术报告](https://arxiv.org/html/2609.19969v1)
- [官方 inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py)

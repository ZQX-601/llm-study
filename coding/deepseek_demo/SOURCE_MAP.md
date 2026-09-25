# 官方源码对照与张量形状

一手资料：

- [DeepSeek-V4.1-Flash 技术报告](https://arxiv.org/html/2609.19969v1)
- [官方 `inference/model.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py)
- [官方 `inference/kernel.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/kernel.py)

本目录代码是独立编写的教学实现，并非复制官方源码。看完本地 demo 后，建议再读官方 `Attention` 类。

形状记号：`B` 为 batch size，`T` 为本次输入 token 数，`H` 为 query head 数，`Dh` 为每头维度，`W` 为局部窗口宽度，`K` 为实际选取的 KV 槽位数，`G` 为输出投影组数，`Rq/Ro` 为低秩投影宽度。

| 官方源码 | 本地对应位置 | 形状与作用 |
| --- | --- | --- |
| `Attention.forward`: `wq_a -> q_norm -> wq_b` | `LocalSWA.forward` | `X:[B,T,d] -> qr:[B,T,Rq] -> q:[B,T,H,Dh]` |
| `_window_kv`：`wkv -> kv_norm -> RoPE` | `LocalSWA._window_kv` | `kv:[B,T,Dh]`，供所有 Q 头共享 |
| `get_window_topk_idxs` | `window_indices` | prefill 为 `[B,T,min(T,W)]`；decode 为 `[B,1,W]`；`-1` 表示无效位置 |
| `kernel.py` 中的 `sparse_attn` | `sparse_attention` | gather 后为 `[B,T,K,Dh]`；权重 `[B,T,H,K]`；输出 `[B,T,H,Dh]` |
| `kernel.py` 中的 `attn_sink` | `sparse_attention` | softmax 中额外的 sink logit，其 value 为零；局部 KV 的权重之和可能小于 1 |
| 逆向 `apply_rotary_emb` | `rotate_tail(..., inverse=True)` | 输出投影前，对 attention 结果的 RoPE 尾部做逆旋转 |
| 分组 `wo_a`，然后 `wo_b` | `LocalSWA.forward` | `[B,T,H,Dh] -> [B,T,G,Ro] -> [B,T,d]` |
| `window_kv_cache` | `LocalSWA.window_kv_cache` | 每层独立的 `[B,W,Dh]` 环形缓存 |

## prefill 为什么能并行

同一层的所有 token 都从**上一层**的 hidden state 生成 Q 和局部 KV。`window_indices` 分别为每个 query 选出满足因果约束的局部 KV 位置；gather 和批量点积可以同时处理全部 `B*T*H` 个 query。同一层中，某个 token 不需要等待前一个 token 的**本层 attention 输出**。第二层则要先得到第一层的输出，再用第二层自己的参数和缓存继续计算。

本实现不构造稠密的 `[B,H,T,T]` 分数矩阵，而是先为每个 query 取出 `K=min(T,W)` 个位置，再形成 `[B,T,H,K]` 的分数。这与官方“先给索引、再做稀疏注意力”的接口思路接近，但不是融合优化后的 kernel。

## 与官方实现的差异

- 官方局部 KV 使用 FP8 量化；demo 保留普通 torch 浮点张量。
- 官方 Q 投影和分组输出使用并行、量化线性层；demo 使用普通 `nn.Linear` 和分组张量运算。
- 官方稀疏 kernel 使用在线 softmax 与 sink；demo 对 gather 后的分数和 sink 做 `logsumexp`，只复现概念上的归一化，不模拟 kernel 的精度与性能行为。
- encoder 前两层是纯 SWA；其他层还会拼接 CSA2 压缩全局 KV 及其索引，本目录暂不实现全局分支。
- Bounded Replay 是部署时的近似状态恢复策略，不属于本次 attention 计算；`reset_cache()` 只是清空缓存，并非回放。

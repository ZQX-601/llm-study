# DeepSeek V4.1 Flash：CED 与 Decoder SWA Bounded Replay

- 日期：2026-09-29
- 模式：QA 专题复习与学习
- 状态：用户完成 CSA2 mode 复习，并通过连续追问理解 CED 的训练/prefill/decode 边界及 Bounded Replay 的近似来源；用户要求归档。记录为“用户自评掌握，待复习和代码验证”，不等于独立验收通过。
- 重要性：CED、prefill/decode 数据流与 KV Cache 属于需要掌握的前沿架构案例；SWA 状态恢复与部署取舍属于推理系统工程知识。

## 一、CSA2 mode 复习

三种 mode 的核心区别：

| Mode | 重建 Main KV | 重建 Indexer K | 重算全局 Top-K | 本层 Main Q、SWA KV 与 Attention |
| --- | ---: | ---: | ---: | ---: |
| Full | 是 | 是 | 是 | 都重新计算 |
| Reindex | 否 | 否 | 是 | 都重新计算 |
| Reuse | 否 | 否 | 否 | 都重新计算 |

一句话记忆：

```text
Full：    重建全局记忆，并重新检索
Reindex：复用全局记忆，但根据本层 Indexer Q 重新检索
Reuse：  同时复用全局记忆和 Top-K 位置，但本层仍重新做 Attention
```

Reuse 不是复用整层输出。三种 mode 都保留本层自己的 Main Q、局部 SWA KV、Attention 权重、加权求和和输出投影。

V4.1 Flash 的记录结构：

```text
Encoder 后 18 层：三个 6 层组
Full → Reuse × 5

Decoder 20 层：五个 4 层组
首组：Full → Reuse × 3
后续：Reindex → Reuse × 3
```

## 二、CED 解决的问题

普通 40 层 decoder-only 模型在推理 prefill 时，Prompt 的每个 token 都需要经过全部 40 层，才能构造各层 KV Cache。简化成本为：

```text
O(NL)
```

DeepSeek V4.1 Flash 将 40 层组织为：

```text
前 20 层：Causal Encoder
后 20 层：Decoder
```

CED 的关键改变是：Decoder global KV 不再由各 Decoder 层自己的历史 hidden state 产生，而是从第 20 层 Encoder 最终表示 `H20` 直接投影。

对于 `l > L/2`：

```text
C_l = H_{L/2} W_l^{KV}
Z_l = H_{L/2} W_l^Z
```

其中 `C_l` 是 global KV entry 的表示，`Z_l` 是对应的压缩权重；投影权重随 Decoder 层而不同。

## 三、训练、Prefill 与 Decode 的边界

### 3.1 预训练

标准 next-token 预训练需要为所有受监督位置产生最终 logits，因此全序列经过完整 40 层：

```text
input:[B,T]
→ Encoder 20 层
→ H20:[B,T,d]
→ Decoder 20 层
→ H40:[B,T,d]
→ logits:[B,T,V]
→ next-token loss
```

位置间通过因果 mask 并行计算，不是逐 token 串行生成。公开报告没有描述预训练只跑前 20 层的捷径，官方最小参考实现也顺序遍历全部 block。

### 3.2 推理 Prefill

整个 Prompt 先经过 Encoder：

```text
Prompt:[B,N]
→ Encoder 20 层
→ H20:[B,N,d]
```

随后使用整个 `H20` 为 Decoder 构造 global KV。大部分 Prompt token 不需要产生 Decoder 最终输出，因此可以绕过完整 Decoder。

但 Decoder 每层的局部 SWA KV 仍来自该 Decoder 层自己的 hidden state，不能直接由 `H20` 完全替代。因此生产推理还会把 Prompt 最后 `W=n_win` 个位置的 Encoder 输出送入后 20 层 Decoder，近似恢复局部 SWA KV，并由最后 Prompt 位置产生首个生成 token 的 logits。

```text
所有 N 个 Prompt token：Encoder 20 层
最后 W 个 Prompt token：额外经过 Decoder 20 层
```

简化复杂度：

```text
O(NL/2 + W×L/2) ≈ O(NL/2),  N >> W
```

因此“prefill 只跑前 20 层”是简化说法；准确说法是大部分 Prompt token 只跑前 20 层，最后窗口仍进行 Decoder replay。

### 3.3 推理 Decode

每个新 token 都处于生成边界，必须产生下一个 token 的 logits：

```text
y_t
→ Encoder 20 层
→ h20_t
   ├─ 通过层相关投影追加到 Decoder global KV，供当前/未来查询
   └─ 作为 Decoder 输入继续经过后 20 层
→ Decoder 最终 hidden
→ LM Head
→ logits(y_{t+1})
```

`h20_t` 产生的 global KV 只是全局历史记忆，不能替代 Decoder 每层逐步产生的 Query、局部 SWA KV、Attention、MoE/mHC 和最终 hidden。因此 decode 的每个新 token 仍经过完整 40 层。

## 四、为什么只回放最后 W 个 token 是近似

设 replay 起点为：

```text
s = N - W
```

标准 SWA 中，位置 `i` 应读取：

```text
[max(0, i-W+1), i]
```

Bounded Replay 将其截断为：

```text
[max(s, i-W+1), i]
```

例如 `W=4`，Prompt 位置为 `0...9`，只回放 `6,7,8,9`：

| Query | 标准窗口 | Bounded Replay |
| --- | --- | --- |
| 6 | `[3,4,5,6]` | `[6]` |
| 7 | `[4,5,6,7]` | `[6,7]` |
| 8 | `[5,6,7,8]` | `[6,7,8]` |
| 9 | `[6,7,8,9]` | `[6,7,8,9]` |

回放片段起点附近丢失了片段之前的局部 SWA 历史。虽然第一层 Decoder 中最后位置的直接窗口已经完整，但片段前几个位置的输出是近似的；上一层的近似 hidden 会作为下一层的局部 KV，误差因此继续向上传播。

## 五、精确回放为什么约为 L×W

顶层最后 `W` 个位置依赖下一层更早的约 `W` 个位置；依赖逐层向前扩展。精确恢复 `L` 层 SWA KV 理论上需要回放约：

```text
L × W 个 token
```

对于 V4.1 Flash 的 20 层 Decoder、`W=128`：

```text
精确 Decoder replay 范围约为 20×128=2560 token
Bounded Replay 只处理 128 token
```

Bounded Replay 用更低计算成本换取近似状态。完整 Prompt 的 global CSA2 KV 仍然存在，因此被截断的是局部 SWA 传播，不是全部早期信息。论文报告实际有效感受野小于理论上限，并在 post-training 中模拟相同 replay，使模型适应该近似；这些是论文作者报告，本仓库尚未复现实验。

Decoder Bounded Replay 生成的近似 SWA KV 只用于随后 decode，不写入长期 prefix cache。

## 六、用户追问与当前证据

用户依次追问：

1. 为什么 prefill 能用 Encoder hidden 产生 Decoder global Main KV，而 decode 仍需完整 40 层；
2. 推理 prefill、decode 和预训练是否分别经过 20/40 层；
3. decode 时 `H20` 是否同时用于产生后 20 层的 global KV；
4. Bounded Replay 从 `-W` 开始时，该位置为什么不读取它之前的 `W` 个 token。

这些问题准确定位了 CED 的关键边界，并推动澄清：

- global KV 是历史记忆，不等于当前 token 的最终 Decoder 表示；
- 推理 prefill 优化不能直接等同于预训练路径；
- decode 中 `H20` 同时承担 global KV 来源和 Decoder 输入；
- 只回放最后 `W` 个 token 会在 replay 边界截断 SWA，因此是明确的近似而非数学等价。

当前证据：用户表示 `ok` 并要求归档；未进行独立闭卷问答、代码实现或数值实验。

## 七、源码与实现边界

- 现有 `coding/deepseek_demo/` 没有实现 CED 调度或 Bounded Replay。
- 官方最小 `inference/model.py` 展示 40 层算子顺序执行，但不直接呈现生产服务中“大部分 Prompt 绕过 Decoder、最后窗口 replay”的 Runtime 调度。
- 本次没有运行模型权重、缓存恢复实验或质量对照。

后续代码验证建议：实现一个小型 `Encoder 2 层 + Decoder 2 层 + W=2` 的教学模型，对比完整 prefill、精确 replay 和 bounded replay 的最后位置 hidden 差异。

## 八、一手资料

- [DeepSeek-V4.1-Flash 技术报告 §2.2 CED](https://arxiv.org/html/2609.19969v1#S2.SS2)
- [DeepSeek-V4.1-Flash 技术报告 §3.2.2 SWA Bounded Replay](https://arxiv.org/html/2609.19969v1#S3.SS2.SSS2)
- [官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
- [官方最小参考实现 `inference/model.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py)

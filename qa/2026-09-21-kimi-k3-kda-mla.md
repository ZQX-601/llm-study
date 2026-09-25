# 答疑归档｜Kimi K3 的 KDA 原理与 KDA–MLA 配合

- 归档日期：2026-09-21（Asia/Shanghai）
- 文档类型：答疑记录（不属于 `archives/dayNN.md` 逐日归档序列）
- 主题：Kimi Delta Attention（KDA）的原理，以及它与 MLA 的混合方式
- 主要信源：
  - Kimi Linear 技术报告，arXiv:2510.26692（KDA 的原始定义、Eq.1）
  - Kimi K3 技术报告，arXiv:2607.24653（第 2.1 节 Hybrid Attention：2.1.1 KDA、2.1.2 Gated MLA；第 5.4.1 节 hybrid KDA–MLA 的 cache 布局）
  - MoonshotAI / flash-linear-attention 实现：`fla/layers/kda.py`、`fla/ops/kda/chunk.py`、`fla/ops/kda/chunk_fwd.py`、`fla/ops/common/chunk_delta_h.py`、`fla/ops/kda/wy_fast.py`
- 配套代码：`kda_reference.py`（纯 Python、零依赖；含逐 token 与分块两种实现、自检断言、玩具配置逐步数值演练，已实际运行验证）
- 结论状态：KDA 原理、混合架构与完整前向链路已由论文与官方实现证实，并有可运行代码交叉验证；K3 的两点 KDA 加强中 `lower-bounded decay` 已在实现里定位（`lower_bound`/`safe_gate`），`full-rank gate` 仍只读到章节标题，未读到正文公式，文中已单独标注

## 一、先确认讨论对象

Kimi K3 是 2026-07 发布的技术报告（arXiv:2607.24653），规模与形态：

```text
2.8T 总参数 MoE
104B 激活参数
原生视觉能力
1M token 上下文
Stable LatentMoE：每 token 激活 896 个路由专家中的 16 个
```

它的注意力层不是纯 KDA，也不是纯 MLA，而是 **Hybrid Attention**：

```text
2.1 Hybrid Attention
├── 2.1.1 Kimi Delta Attention (KDA)
│   ├── Chunkwise parallel form
│   ├── Lower-bounded decay
│   └── Full-rank gate
└── 2.1.2 Gated MLA
```

推理侧还有一节专门讲混合缓存的统一管理：

```text
5.4.1 KDA-Aware Prefix Cache Management
└── Unified cache layout for hybrid KDA–MLA attention
```

也就是说，K3 的 KDA 和 MLA 是**同一层堆叠里的两种 token mixer**，而不是把两者焊进同一个注意力公式里。

KDA 本身来自更早的 Kimi Linear（arXiv:2510.26692，3B 激活 / 48B 总参数）。它的技术报告是理解 KDA 原理最直接的材料，K3 是把它规模化并做了两处加强。

## 二、KDA 原理

### 1. 从线性注意力说起：固定大小的 fast-weight 记忆

线性注意力把注意力写成一个**递归状态**，而不是每 token 都保存一份 KV：

```text
状态矩阵：S_t ∈ R^{d_k × d_v}
读操作：  o_t = S_t^T q_t
```

它相当于一个「快速权重」记忆：用外积把 key–value 关联写进矩阵。最朴素的累加形式是

```text
S_t = S_{t-1} + v_t k_t^T
```

问题很明显：**同一个 key 位置反复写入会互相叠加**，旧值不会被清掉。记忆容量有限，信息就会串扰（memory collision / interference），这正是纯线性注意力早期质量不如 softmax attention 的核心原因之一。

### 2. delta rule：先擦除，再写入

DeltaNet 的思路是引入 **delta 规则**（本质是 Widrow–Hoff / LMS 的最小均方更新）。写入之前，先用当前 key 把旧值「擦掉」：

```text
预测：  ŷ = S_{t-1}^T k_t
误差：  v_t - ŷ
更新：  S_t = S_{t-1}(I - β_t k_t k_t^T) + β_t v_t k_t^T
```

可以严格理解成对下面这个重构损失做**在线梯度下降**：

```text
L(S) = ½ ‖S^T k_t - v_t‖²
```

所以「delta」指的是：只把与当前 key 相关的旧关联修正掉，写入的是**残差**而不是原值。这就是它比纯累加更「干净」的原因。

### 3. 门控：让记忆可以遗忘

递推记忆还有一个问题：状态只增不减，早期信息永久占位。于是加**遗忘门**（decay / gate），得到 gated delta rule：

```text
S_t = S_{t-1}(diag(α_t) - β_t k_t k_t^T) + β_t v_t k_t^T
```

转移矩阵是

```text
diag(α_t) - β_t k_t k_t^T
= 对角矩阵（满秩 d_k）+ 秩 1 修正
```

这正是论文说的 **DPLR（Diagonal-Plus-Low-Rank）** 结构。

### 4. KDA 的关键改动：门控粒度从 head 级降到 channel 级

这一步是 KDA 与 Gated DeltaNet（GDN）的分界：

```text
GDN：α_t 是每个 head 一个标量
     → 整个 head 共用同一个遗忘率（粗粒度，类似 Mamba2）

KDA：α_t 是每个特征通道一个标量，即对角矩阵
     → 每个特征维度有独立的遗忘率（细粒度，思路接近 GLA）
```

KDA 论文的原话是「extends Gated DeltaNet with a finer-grained gating mechanism, enabling more effective use of limited finite-state RNN memory」。直觉上：

- head 级标量门只能整体「调亮度」；
- channel 级对角门可以让一部分维度长期保留、另一部分快速刷掉，**在同样大小的状态里塞进更多有用信息**。

同时论文强调它用的是 DPLR 的**特化变体**，比一般 DPLR 计算量显著更低，而且更贴近经典 delta 规则。

### 5. 训练必须能并行：chunkwise 算法

递归形式串行，无法直接吃满 GPU。KDA 用的是分块（chunkwise）并行算法：

```text
序列切成长度 C 的 chunk
→ 块内用矩阵形式一次算完（WY 表示 + UT transform）
→ 块间只传递一次状态
```

效果是把「逐 token 递归」改造成以 matmul 为主的计算，这也是 KDA kernel 能进 flash-linear-attention / vLLM 的前提。

### 6. 一个容易被忽略的视角：KDA 自带位置信息

Kimi Linear 报告单列了一节 "Kimi Delta Attention as learnable position embeddings"。因为衰减本身是**依赖位置和数据**的：越久远的信息衰减越多，等于给模型一个可学习的、内容相关的位置先验。这一点直接影响了它和 MLA 的配合方式（见下一节）。

### 7. K3 对 KDA 的两处加强（仅读到标题，未读到公式）

K3 报告 2.1.1 节列出两个小标题：

```text
Lower-bounded decay   下界受约束的衰减
Full-rank gate        满秩门
```

按命名可以合理推测其意图，但**我没有读到正文公式，以下只作为待验证的理解**：

- `lower-bounded decay`：长上下文下累计衰减容易把状态压到接近零，给衰减率加下界可以避免记忆过早「失忆」，改善百万 token 量级的稳定性。
- `full-rank gate`：把对角门升级为满秩门，等于让转移矩阵比「对角 + 秩 1」更一般化，表达力更强，代价是 kernel 更复杂。

这两点的确切定义应以报告正文为准，复习时建议回到原文核对。

## 三、KDA 与 MLA 怎么配合

### 1. 先回顾 MLA 是什么

MLA（Multi-Head Latent Attention）来自 DeepSeek-V2：把每个 token 的 K/V **低秩压缩**成一个 latent 向量再缓存，用极小的 KV cache 换接近全注意力的表达能力。Kimi K2 用的就是 MLA（64 个注意力头）。

关键区别：

```text
KDA 层：每层只维护固定大小的递归状态 S（不随序列增长）
MLA 层：仍需按 token 存 latent KV（随序列线性增长，但被压缩过）
```

### 2. 配合方式是「层间混合」，不是层内融合

Kimi Linear 的方案是**均匀 3:1 交替堆叠**：

```text
[KDA, KDA, KDA, MLA] × N
```

论文原文：interleaves KDA with periodic full attention layers in a uniform 3:1 ratio。在这个设计里：

- **KDA 层**负责廉价的长程记忆与位置感，每层状态 O(1)，解码时无 KV 增长；
- **MLA 层**作为 full attention，负责全局、精确、基于内容的检索（典型就是 long-context 的「大海捞针」）；
- 少数 full attention 层保证了**全局信息流**不被纯线性层截断。

为什么必须保留 full attention：线性注意力受**有限状态容量**约束，长序列建模与 in-context retrieval 有理论上的上限。混合架构就是用「少数精确检索层 + 多数廉价记忆层」换总体质量与效率。

### 3. Kimi Linear 给出的收益

```text
KV cache 最多降低 75%
1M 上下文解码吞吐最多提升 6×
在相同训练配方下，全面超过纯 MLA 基线
```

75% 这个量级的来源正是 3:1：四层里有三层完全不产生 per-token KV。

### 4. 位置编码上的分工

Kimi Linear 对 MLA 层采用 **NoPE（不做位置编码）**，报告 5.2 节还专门做了 NoPE vs. RoPE 的消融。

原因就是上一节第 6 点：KDA 的衰减已经提供了位置与recency 信息，MLA 层可以省掉显式位置编码。这是一处**功能互补**而非简单堆叠的证据。

### 5. K3 里的形态

K3 报告把这一节直接命名为 "Hybrid Attention"，两个子节分别是 **Kimi Delta Attention** 和 **Gated MLA**：

- `Gated MLA`：MLA 层本身也引入了门控。有一篇 2026 年的第三方研究提到，门控注意力把首 token 注意力占比从 46.7% 降到 4.8%（attention sink 现象），并把 Kimi K3 描述为 "pairs that idea with Kimi Delta Attention and Attention Residuals"。可以理解为：MLA 侧补上门控，用来抑制 attention sink、改善百万级上下文的行为。（此段为跨论文推断，非 K3 报告原文断言。）
- 系统侧：5.4.1 节 `Unified cache layout for hybrid KDA–MLA attention` 说明两类层在**同一套 cache 布局**里管理——KDA 存固定状态，MLA 存压缩 latent KV，并配套 KDA-aware prefix cache 优化与并发调度一致性。

K3 是否沿用 3:1 这一具体比例，我没有在已获取的正文中确认，复习时以报告 2.1 节为准。

### 6. 一张分工表

```text
                 KDA 层                       MLA 层（Gated）
记忆形态         固定大小递归状态 S            按 token 的压缩 latent KV
序列增长         不增长                       线性增长（已压缩）
单位成本         极低                        较高
擅长             长程累积、recency、流式解码    精确检索、全局依赖
在 K3 中的角色   主力层（多数）                周期性的全局层（少数）
```

一句话总结：**KDA 提供廉价的、可遗忘的长期记忆；MLA 提供少量但精确的全局检索；位置信息主要由 KDA 的衰减承担，需要时再补门控与显式位置编码。** 两者是层间互补，不是同一个公式里的两项相加。

## 四、易错点

1. KDA 不是「把 attention 换成 RNN」，而是用一个固定大小的矩阵状态做**带擦除与遗忘的关联记忆**；delta rule 的核心是「先擦后写残差」。
2. KDA ≠ GDN。差别就在门控粒度：GDN 是 head 级标量，KDA 是 channel 级对角（K3 进一步到 full-rank gate）。
3. KDA 与 MLA **不在同一层内融合**，而是按比例交替堆叠；看到「hybrid」要想到层间（inter-layer）而非层内（intra-layer）。
4. MLA 仍然是 full attention，只是 KV 被低秩压缩；它的 cache 依然随序列增长，只是常数更小。
5. 混合架构里 full attention 层数不能随便砍到零——有限状态记忆的容量上限是理论约束，不是工程调参问题。
6. KDA 的 chunkwise 实现是「块内并行 + 块间串行」，不要理解成完全并行。

## 五、自测问题

```text
1. 纯线性注意力 S_t = S_{t-1} + v_t k_t^T 的失效模式是什么？

2. delta rule 的更新式里，哪一项负责「擦除」，哪一项负责「写入」？

3. 把 delta rule 看成在线学习，它的损失函数和优化目标是什么？

4. GDN 与 KDA 的唯一关键差异是什么？为什么这个差异能提升有限状态记忆的利用率？

5. 为什么 KDA 的转移矩阵被称为 DPLR？对角部分和秩 1 部分各有何作用？

6. 为什么 Kimi Linear 敢在 MLA 层用 NoPE？前提条件是什么？

7. 3:1 混合为什么能降低约 75% 的 KV cache？如果换成 7:1 会牺牲什么？

8. 为什么不能把所有层都换成 KDA？
```

## 六、KDA 手算演练（d_k = d_v = 2）

本节所有计算严格按 Kimi Linear 论文 Eq.(1)，可用草稿纸逐行复核。

### 1. 约定与四个递推式的对照

```text
符号约定
  k_t ∈ R^{d_k}        键（KDA 中已做 L2 归一化，‖k_t‖ = 1）
  v_t ∈ R^{d_v}        值
  S_t ∈ R^{d_k × d_v}  记忆状态（按 key 维度分行、value 维度分列）
  o_t = S_t^T q_t      读出

纯线性注意力
  S_t = S_{t-1} + k_t v_t^T

DeltaNet
  S_t = (I − β_t k_t k_t^T) S_{t-1} + β_t k_t v_t^T

Gated DeltaNet（GDN，α 是 head 级标量）
  S_t = α_t (I − β_t k_t k_t^T) S_{t-1} + β_t k_t v_t^T

KDA（α 是 channel 级向量，论文 Eq.1）
  S_t = (I − β_t k_t k_t^T) Diag(α_t) S_{t-1} + β_t k_t v_t^T
```

注意 KDA 与 GDN 的差别不只是「α 从标量变向量」，**位置也不同**：

```text
GDN：α 乘在整个 (I − βkk^T)S 外面，写入项也被间接缩放
KDA：Diag(α) 先作用于 S_{t-1}，再做擦除，写入项 β k v^T 保持满强度
```

所以 KDA 的转移矩阵是 `Diag(α_t) − β_t k_t (Diag(α_t) k_t)^T`，即「对角（满秩 d_k）+ 秩 1」，这就是 DPLR 特化变体，也是论文说它「更贴近经典 delta 规则」的原因。

### 2. α 和 β 在实际实现里怎么来

按 flash-linear-attention 的 `fla/layers/kda.py`：

```text
q, k, v = shortconv(silu(W·x))             # 短卷积，kernel=4
q, k    = L2norm(q), L2norm(k)             # 键被归一化到单位长度
beta    = sigmoid(W_b · x)                 # shape [B, T, HV]
g       = −exp(A_log) · softplus(f_proj(x) + dt_bias)   # log 空间，shape [B, T, HV, K]
alpha   = exp(g)                           # 每个 (value head, key 维) 一个衰减率
```

对照 GDN：GDN 的门是 `[B, T, H]`（每 head 一个标量），KDA 是 `[B, T, HV, K]`（每 head 的每个 key 维一个标量）——这就是 channel-wise 的具体落地。

`lower_bound` 参数把 log 空间的门钳制在 `[lower_bound, 0)`，例如 `-5` 时单步衰减下界 `exp(−5) ≈ 0.0067`，这就是 K3 报告里 `Lower-bounded decay` 的实现形态。它还让 kernel 能用 M=16 TensorCore 加速（`safe_gate`）。

另外要区分**两个不同的门**：上面管记忆衰减的是 α；KDA 还有输出门 `g_proj`（sigmoid + RMSNormGated），作用在输出上，与记忆无关。

### 3. 手算 Part 1：为什么需要「先擦后写」

取 `S_0 = 0`，前 3 步令 `α = (1,1)`（暂不衰减），突出擦除本身的作用。

```text
t = 1
  k_1 = (1,0)^T   v_1 = (1,0)^T   β_1 = 1   α_1 = (1,1)^T

  ① 遗忘：Diag(α_1)·S_0 = I·0 = 0
  ② 键外积：k_1 k_1^T = [[1,0],[0,0]]
  ③ 擦除算子：(I − β_1 k_1 k_1^T) = [[0,0],[0,1]]
  ④ 擦除：③ · ① = 0
  ⑤ 写入：β_1 k_1 v_1^T = (1,0)^T(1,0) = [[1,0],[0,0]]
  ⑥ 合并：S_1 = 0 + [[1,0],[0,0]] = [[1,0],[0,0]]

t = 2
  k_2 = (0,1)^T   v_2 = (0,1)^T   β_2 = 1   α_2 = (1,1)^T

  ① Diag(α_2)·S_1 = [[1,0],[0,0]]
  ② k_2 k_2^T = [[0,0],[0,1]]
  ③ (I − β_2 k_2 k_2^T) = [[1,0],[0,0]]
  ④ ③·① = [[1,0],[0,0]]
  ⑤ β_2 k_2 v_2^T = (0,1)^T(0,1) = [[0,0],[0,1]]
  ⑥ S_2 = [[1,0],[0,1]]

t = 3（关键：重复出现 t=1 的 key，但换了 value）
  k_3 = (1,0)^T   v_3 = (0,5)^T   β_3 = 1   α_3 = (1,1)^T

  ① Diag(α_3)·S_2 = [[1,0],[0,1]]
  ② k_3 k_3^T = [[1,0],[0,0]]
  ③ (I − β_3 k_3 k_3^T) = [[0,0],[0,1]]
  ④ ③·① = [[0,0],[0,1]]        ← 第 1 行被整体清零，旧的 (1,·) 被擦掉
  ⑤ β_3 k_3 v_3^T = (1,0)^T(0,5) = [[0,5],[0,0]]
  ⑥ S_3 = [[0,5],[0,1]]
```

对照纯线性累加：

```text
S_3^linear = k_1v_1^T + k_2v_2^T + k_3v_3^T
           = [[1,0],[0,0]] + [[0,0],[0,1]] + [[0,5],[0,0]]
           = [[1,5],[0,1]]
```

用 `q = (1,0)` 读出（`o = S^T q` 取的是 S 的第 1 行）：

```text
KDA     ：o = S_3^T q → (0,5)   最新写入的 value，干净
线性累加：o = (1,5)             旧 value 的 1 混进来了，无法区分新旧
```

这就是 delta rule「先擦后写」的价值：同一 key 重复出现时，返回的是**最新关联**而不是历史叠加。

### 4. 手算 Part 2：channel-wise 门控到底差在哪

从 `S_3 = [[0,5],[0,1]]` 出发，第 4 步令 `α_4 = (1, 0.1)^T`——只让第 2 个 key 维快速衰减。

```text
t = 4
  k_4 = (0,1)^T   v_4 = (9,0)^T   β_4 = 1   α_4 = (1, 0.1)^T

  ① Diag(α_4)·S_3
       = [[1,0],[0,0.1]] · [[0,5],[0,1]]
       第 1 行：1·(0,5) + 0·(0,1)   = (0, 5)
       第 2 行：0·(0,5) + 0.1·(0,1) = (0, 0.1)
       = [[0,5],[0,0.1]]       ← key 维 1 完整保留，key 维 2 衰减到 1/10

  ② k_4 k_4^T = [[0,0],[0,1]]
  ③ (I − β_4 k_4 k_4^T) = [[1,0],[0,0]]
  ④ ③·① = [[0,5],[0,0]]
  ⑤ β_4 k_4 v_4^T = (0,1)^T(9,0) = [[0,0],[9,0]]
  ⑥ S_4 = [[0,5],[9,0]]
```

读出验证（`q = (0,1)`，即取 S 的第 2 行）：`o = (9,0)`，正是刚写进去的值。

现在做**同一步、同一个状态**下 KDA 与 GDN 的门控对比。假设第 5 步只衰减不写入（`β_5 = 0, v_5 = 0`）：

```text
KDA   α_5 = (1, 0.1)   → Diag(α_5)·S_4
                        = [[1,0],[0,0.1]]·[[0,5],[9,0]]
                        = [[0,5],[0.9,0]]      ← 第 1 行 (0,5) 原封不动

GDN   α_5 = 0.1（标量）→ 0.1·S_4
                        = [[0,0.5],[0.9,0]]    ← 第 1 行也被压到 (0,0.5)
```

结论一目了然：**head 级标量门只能整体缩放；channel 级对角门可以让一部分 key 维度长期保留、另一部分快速遗忘。** 在固定的状态容量下，后者能用同样的参数量表达更细的记忆生命周期，这正是 KDA 相对 GDN 的核心增量。

### 5. 手算 Part 3：非 one-hot key 的擦除是「投影去除」

前面的 one-hot key 让擦除看起来像「清掉一整行」，容易造成误解。取单位键 `k = (0.6, 0.8)^T`（满足 ‖k‖ = 1），`β = 1`，`v = 0`，状态仍为 `S = [[0,5],[0,1]]`：

```text
k k^T = [[0.36, 0.48],[0.48, 0.64]]
I − k k^T = [[0.64, −0.48],[−0.48, 0.36]]

(I − k k^T)·S
  第 1 行：0.64·(0,5) + (−0.48)·(0,1) = (0, 3.2) + (0, −0.48) = (0, 2.72)
  第 2 行：(−0.48)·(0,5) + 0.36·(0,1)  = (0, −2.4) + (0, 0.36) = (0, −2.04)

S_new = [[0, 2.72],[0, −2.04]]
```

用投影理解更直观：`k` 方向上的分量是 `k^T S = (0.6·0 + 0.8·0, 0.6·5 + 0.8·1) = (0, 3.8)`，从原状态里减去这一部分：

```text
S − k·(k^T S) = [[0,5],[0,1]] − [[0, 2.28],[0, 3.04]] = [[0, 2.72],[0, −2.04]]
```

与矩阵乘法结果一致。所以 delta rule 擦除的是**状态在键方向上的投影**，而不是整行。这也解释了为什么 KDA 在 kernel 里对 q、k 做 L2 归一化——只有 `‖k‖ = 1`，`I − kk^T` 才是一个干净的投影算子（特征值 0 和 1）。

### 6. 累计衰减 γ

一个 chunk 内的累计衰减是逐步连乘：

```text
γ^{i→j} = ∏_{s=i}^{j} α_s
```

若连续 3 步都取 `α = (1, 0.1)`：

```text
key 维 1：1 × 1 × 1 = 1          → 信息完整保留
key 维 2：0.1 × 0.1 × 0.1 = 0.001 → 基本被遗忘
```

chunkwise 算法正是先离线算出这些 `γ`，用 `A[i/j] = γ^i / γ^j` 构造 chunk 内的因果衰减矩阵，从而把递归改写成矩阵乘法。

### 7. 手算易错点

1. `Diag(α)` 作用在 `S_{t-1}` 的**行**（key 维度）上，不是列。
2. KDA 的 `Diag(α)` 在括号**内部**，与 GDN 的标量在括号**外部**不同；KDA 的写入项不被 α 缩放。
3. 读出 `o = S^T q`，取的是 S 的行；写代码时容易把 S 转置方向搞反。
4. 只有 `‖k‖ = 1` 时擦除才是严格投影。
5. 记忆衰减门 α 与输出门 `g_proj` 是两个完全不同的门，不要混为一谈。
6. 空状态 `S_0 = 0` 时第一个 token 就是纯写入，此时 `S_1 = β_1 k_1 v_1^T`。

## 七、从 [B,T,d] 到输出的完整前向

本节对齐 MoonshotAI / flash-linear-attention 的实际实现（`fla/layers/kda.py`、`fla/ops/kda/chunk.py`、`fla/ops/kda/chunk_fwd.py`、`fla/ops/common/chunk_delta_h.py`），并配有可运行参考实现 `kda_reference.py`（纯 Python，无第三方依赖，已验证递推与分块等价）。

### 7.1 形状流水线（以 d=2048, H=HV=16, K=V=128 为例）

| 阶段 | 公式/操作 | 形状 |
|---|---|---|
| 输入 | x | [B,T,d] |
| q/k/v 投影 | q=W_q x, k=W_k x, v=W_v x | q,k:[B,T,H·K]；v:[B,T,HV·V] |
| 短卷积 | 因果 Conv1d(kernel=4) + SiLU | 不变 |
| 拆头 | | q,k:[B,T,H,K]；v:[B,T,HV,V] |
| L2 归一化 | q←q/‖q‖, k←k/‖k‖ | 不变 |
| β | β=sigmoid(W_b x) | [B,T,HV] |
| 门控原始值 | r=W_f x+dt_bias | [B,T,HV·K] → [B,T,HV,K] |
| 衰减 | g=−exp(A_log)⊙softplus(r)；α=exp(g) | [B,T,HV,K] |
| 分块累计 | g_cumsum=cumsum(g,块内)  | [B,T,HV,K] |
| KDA 核心 | Eq.(1) 逐 token，或分块算法 | o:[B,T,HV,V]；state:[N,HV,K,V] |
| 输出门 | o ← RMSNorm(o)⊙σ(g_proj(x)) | [B,T,HV,V] |
| 合并头 | | [B,T,HV·V] |
| o_proj | W_o·o | [B,T,d] |

### 7.2 每段公式

```text
1) 投影
   q = W_q x,  k = W_k x,  v = W_v x

2) 短卷积（因果，左侧 padding，kernel=4）
   y_t = SiLU( Σ_{i=0}^{3} w_i ⊙ u_{t-3+i} )

3) L2 归一化
   q ← q / max(‖q‖₂, ε),   k 同理

4) beta
   β = sigmoid(W_b x)
   （allow_neg_eigval=True 时为 2·sigmoid 以允许负特征值）

5) 门控（log 空间）
   默认：      g = −exp(A_log) ⊙ softplus(W_f x + dt_bias)
   safe_gate： g = lower_bound · sigmoid(exp(A_log) ⊙ (W_f x + dt_bias))
              取值范围 [lower_bound, 0)，推荐 −5（单步衰减下界 e^{−5}≈0.0067）
   衰减率：    α = exp(g) ∈ (0,1]

6) 核心递推（论文 Eq.1）
   S_t = (I − β_t k_t k_t^T) Diag(α_t) S_{t−1} + β_t k_t v_t^T
   o_t = scale · S_t^T q_t,   scale = K^{−1/2}

7) 输出门 + o_proj
   y = RMSNorm(o) ⊙ weight ⊙ sigmoid(g_proj(x))
   out = W_o y
```

### 7.3 分块（chunkwise）公式

块长 C（实现里取 32 或 64），块内累计衰减 `γ_r = ∏_{u≤r} α_u`：

```text
L[r,s]  = β_r · Σ_d k_{r,d} k_{s,d} · γ_{r,d}/γ_{s,d}          (s < r，严格下三角)
A       = (I + L)^{-1}                                          # WY / UT 变换
w       = A · (β ⊙ k ⊙ γ)                                       # [C,K]
u       = A · (β ⊙ v)                                           # [C,V]
v_new   = u − w h                                               # 对传入状态的修正
kg_r    = k_r ⊙ (γ_last / γ_r)                                  # 归一到块尾
Aqk[r,s]= Σ_d q_{r,d} k_{s,d} · γ_{r,d}/γ_{s,d}                 (s ≤ r)
o       = scale · ( Aqk · v_new + (q ⊙ γ) h )
h      ← Diag(γ_last) h + kg^T · v_new                          # 块间状态扫描
```

要特别注意 `L[r,s]` 里 β 取**行下标 r**（对应 `Diag(β) K K^T` 的写法）。本文件配套的 `kda_reference.py` 最初写成列下标 β_s，自检时立刻暴露不一致；改成 β_r 后，在 T∈{1,2,3,5,8,13}、K∈{1,2,4}、V∈{1,3}、chunk∈{1,2,3,4,64} 的全部组合上，分块与逐 token 递推的最大误差 < 1e-9。

### 7.4 PyTorch 参考实现（逐 token 版，便于对照公式）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class KimiDeltaAttention(nn.Module):
    """最小可读版；HV == H，逐 token 递推，便于和论文 Eq.(1) 逐行对照。"""

    def __init__(self, d_model=2048, num_heads=16, head_dim=128,
                 conv_size=4, lower_bound=None):
        super().__init__()
        self.H = self.HV = num_heads
        self.K = self.V = head_dim
        self.conv_size = conv_size
        self.lower_bound = lower_bound

        self.q_proj = nn.Linear(d_model, self.H * self.K, bias=False)
        self.k_proj = nn.Linear(d_model, self.H * self.K, bias=False)
        self.v_proj = nn.Linear(d_model, self.HV * self.V, bias=False)
        self.q_conv = nn.Conv1d(self.H * self.K, self.H * self.K,
                                conv_size, groups=self.H * self.K, bias=False)
        self.k_conv = nn.Conv1d(self.H * self.K, self.H * self.K,
                                conv_size, groups=self.H * self.K, bias=False)
        self.v_conv = nn.Conv1d(self.HV * self.V, self.HV * self.V,
                                conv_size, groups=self.HV * self.V, bias=False)

        self.b_proj = nn.Linear(d_model, self.HV, bias=False)
        self.f_proj = nn.Linear(d_model, self.HV * self.K, bias=False)
        self.A_log = nn.Parameter(torch.zeros(self.HV))
        self.dt_bias = nn.Parameter(torch.zeros(self.HV * self.K))

        self.g_proj = nn.Linear(d_model, self.HV * self.V, bias=False)
        self.o_norm = nn.RMSNorm(self.V)
        self.o_proj = nn.Linear(self.HV * self.V, d_model, bias=False)

    def _conv(self, proj, conv, x):
        u = proj(x).transpose(1, 2)                    # [B, C, T]
        u = F.pad(u, (self.conv_size - 1, 0))          # 因果：左侧补 pad
        return conv(u).transpose(1, 2)                 # [B, T, C]

    def forward(self, x, state=None, return_state=False):
        B, T, _ = x.shape

        q = self._conv(self.q_proj, self.q_conv, x).view(B, T, self.H, self.K)
        k = self._conv(self.k_proj, self.k_conv, x).view(B, T, self.H, self.K)
        v = self._conv(self.v_proj, self.v_conv, x).view(B, T, self.HV, self.V)
        q = F.normalize(q, dim=-1)                     # L2 归一化
        k = F.normalize(k, dim=-1)

        beta = torch.sigmoid(self.b_proj(x))           # [B,T,HV]
        raw = self.f_proj(x).view(B, T, self.HV, self.K) + self.dt_bias
        A = torch.exp(self.A_log).view(1, 1, self.HV, 1)
        if self.lower_bound is None:
            g = -A * F.softplus(raw)
        else:
            g = self.lower_bound * torch.sigmoid(A * raw)
        alpha = torch.exp(g)                           # [B,T,HV,K]

        S = torch.zeros(B, self.HV, self.K, self.V,
                        dtype=x.dtype, device=x.device) if state is None else state
        eye = torch.eye(self.K, dtype=x.dtype, device=x.device)
        scale = self.K ** -0.5
        outs = []
        for t in range(T):
            kt = k[:, t]                               # [B,HV,K]
            # ① 遗忘
            S = alpha[:, t].unsqueeze(-1) * S
            # ② 擦除 + ③ 写入
            kk = kt.unsqueeze(-1) * kt.unsqueeze(-2)   # [B,HV,K,K]
            erase = eye - beta[:, t].view(B, self.HV, 1, 1) * kk
            S = erase @ S + beta[:, t].view(B, self.HV, 1, 1) * \
                (kt.unsqueeze(-1) * v[:, t].unsqueeze(-2))
            # ④ 读出
            outs.append(scale * (S.transpose(-1, -2) @ q[:, t].unsqueeze(-1)).squeeze(-1))

        o = torch.stack(outs, dim=1)                   # [B,T,HV,V]
        o = self.o_norm(o) * torch.sigmoid(
            self.g_proj(x).view(B, T, self.HV, self.V))
        o = self.o_proj(o.reshape(B, T, self.HV * self.V))
        return (o, S) if return_state else o
```

分块版（与 kernel 同构、已自检）见 `kda_reference.py` 的 `kda_chunked`。

### 7.5 玩具配置全链路数值演练（已运行验证）

配置：`B=1, T=4, d=4, H=HV=1, K=V=2, chunk_size=2`，权重取简单值，`A_log=0`，`dt_bias=0`，`lower_bound=None`。

输入与投影后（silu 之后、L2 归一化之后）：

```text
x_1=(1,0,1,0)  x_2=(0,1,0,1)  x_3=(1,0,1,0)  x_4=(0,1,0,1)

t=1: q=(1,0) k=(1,0) v=(0.731059,0)  β=0.731059  g=(-1.313262,-0.693147)
t=2: q=(0,1) k=(0,1) v=(0,0.731059)  β=0.5       g=(-0.693147,-1.313262)
t=3: 同 t=1      t=4: 同 t=2

α=exp(g):  t=1:(0.268941,0.5)  t=2:(0.5,0.268941)  t=3:(0.268941,0.5)  t=4:(0.5,0.268941)
```

逐步递推（scale = 1/√2 = 0.707107）：

```text
t=1  ① Diag(α)·S_0 = 0
     ② k k^T = [[1,0],[0,0]]     ③ I−β k k^T = [[0.268941,0],[0,1]]
     ④ 擦除后 = 0
     ⑤ 写入 β k v^T = [[0.534447,0],[0,0]]
     ⑥ S_1 = [[0.534447,0],[0,0]]        ⑦ o_1 = (0.377911, 0)

t=2  ① Diag(α)·S_1 = [[0.267223,0],[0,0]]
     ② k k^T = [[0,0],[0,1]]     ③ I−β k k^T = [[1,0],[0,0.5]]
     ④ 擦除后 = [[0.267223,0],[0,0]]
     ⑤ 写入 = [[0,0],[0,0.365529]]
     ⑥ S_2 = [[0.267223,0],[0,0.365529]]  ⑦ o_2 = (0, 0.258468)

t=3  ① = [[0.071867,0],[0,0.182765]]   ③ = [[0.268941,0],[0,1]]
     ④ = [[0.019328,0],[0,0.182765]]   ⑤ = [[0.534447,0],[0,0]]
     ⑥ S_3 = [[0.553775,0],[0,0.182765]]  ⑦ o_3 = (0.391578, 0)

t=4  ① = [[0.276887,0],[0,0.049153]]   ③ = [[1,0],[0,0.5]]
     ④ = [[0.276887,0],[0,0.024576]]   ⑤ = [[0,0],[0,0.365529]]
     ⑥ S_4 = [[0.276887,0],[0,0.390106]]  ⑦ o_4 = (0, 0.275846)
```

分块实现（chunk_size=2）给出的输出与逐 token 递推**完全一致**：

```text
分块 o = [[0.377911,0],[0,0.258468],[0.391578,0],[0,0.275846]]
最大误差 = 5.551e-17
最终状态 h = [[0.276887,0],[0,0.390106]]
```

输出门 + RMSNormGated + o_proj（eps=1e-5，norm weight=1）：

```text
t=1: o=(0.377911,0)   → rms=(1.414115,0) → ⊙σ(gate)=(1.033801,0) → o_proj=(1.033801,0,0,0)
t=2: o=(0,0.258468)   → rms=(0,1.414002) → ⊙σ(gate)=(0,1.033718) → o_proj=(0,1.033718,0,0)
t=3: o=(0.391578,0)   → rms=(1.414121,0) → ⊙σ(gate)=(1.033806,0) → o_proj=(1.033806,0,0,0)
t=4: o=(0,0.275846)   → rms=(0,1.414028) → ⊙σ(gate)=(0,1.033737) → o_proj=(0,1.033737,0,0)
```

注意 RMS 是对 V=2 个维度求均方根，只有一个非零分量时归一化结果是 √2 ≈ 1.414，不是 1——这是 RMSNorm 的固有行为，不是算错。

### 7.6 工程实现里容易踩的点

1. `Diag(α)` 乘在状态的**行（key 维）**上；`o = S^T q` 取的是状态的列方向，两处最容易搞混。
2. 短卷积必须**因果**（左侧 padding），否则会看到未来 token。
3. `β` 在 `L`、`w`、`u`、写入项里都对齐到**行（当前 token）**。
4. 门控在实现里存 **log2 空间**（乘 `1/ln2` 后累加，用 `exp2`），与 `exp` 只差常数重标定。
5. GVA（`HV > H`）时 q/k 要按 `HV/H` 复制扩展，v/g/β 保持 `HV` 头。
6. `α`（记忆衰减门）与 `g_proj`（输出门）是两个独立的门。
7. `state_v_first=True` 只是把状态布局从 [K,V] 换成 [V,K] 以利访存，数学等价。

### 7.7 为什么 q/k 投影成 H·K、v 投影成 HV·V

#### (1) H·K 不是"另一种投影"，而是分头布局

注意力天然是**分头**的：第 h 个头在自己的 K 维子空间里算 score，各头独立，最后 concat 回总维度。所以投影层一次算出所有头，输出宽度就是

```text
q = W_q x        W_q: [d, H·K]     →  [B, T, H·K]   reshape  →  [B, T, H, K]
```

`H·K` 只是 H 个 K 维向量拼在一起的**内存布局**；`rearrange(x, "... (h d) -> ... h d", d=head_k_dim)` 就是这一步。同一层的 v 用 `head_v_dim` 做 reshape，得到 `[B,T,HV,V]`。

#### (2) q 与 k 必须同头数，v 不必

score 是 `q_h · k_h` 的**逐头配对**，所以 q、k 头数必须一致（都是 H）。而 v 只与"记忆"配对，读出是 `o_h = S_h^T q_h`，v 的头数可以独立——这就是 **GVA（Grouped Value Attention）**。

代码证据：

```text
fla/layers/kda.py      num_v_heads = num_heads if None；要求 num_v_heads % num_heads == 0
fla/ops/kda/chunk.py   assert HV % H == 0, "For GVA, num_v_heads (HV) must be evenly divisible by num_qk_heads (H)"
fla/ops/kda/chunk_fwd.py  i_b, i_hv = ...; i_h = i_hv // (HV // H)   # 多个 value 头共用同一个 q/k 头
```

最后一行的含义：`HV//H` 个连续的 value 头共享同一个 q/k 头。

#### (3) GVA 到底解耦了什么

```text
状态 S 是 per value head 的：[K, V]
  HV → 记忆槽（slot）的数量
  K  → 每个槽的寻址维度（由 q/k 决定）

MHA（H == HV）：一个头同时拥有自己的"寻址通道"和"记忆槽"，两者被绑死
GVA（HV > H）：多个 value 头共享一套 q/k 寻址，各自维护独立记忆
```

所以 GVA 把**寻址空间数 H** 和**记忆槽数 HV** 变成两个独立旋钮。

#### (4) 与 GQA / MQA 的关系（方向相反）

| | 标准 attention（GQA/MQA） | KDA（GVA） |
|---|---|---|
| 共享谁 | 多个 **query** 头共享一份 K/V | 多个 **value** 头共享一份 q/k |
| 目的 | 缩小 KV cache | 在寻址不变的前提下增加记忆槽 |
| 缓存对象 | 每 token 的 K/V | 固定大小的递归状态 S |

KDA 一侧的"cache"是 `[HV,K,V]` 的状态而非逐 token 张量，所以这里扩的是 value 侧。

#### (5) 数量例子（默认配置 `H=HV=16, K=V=128, d=2048`）

| 配置 | q/k 宽度 | v 宽度 | 状态/头 | 总状态 |
|---|---|---|---|---|
| H=HV=16, V=128 | 16·128=2048 | 16·128=2048 | [128,128] | 16·128·128 |
| H=8, HV=16, V=128 | 8·128=1024 | 16·128=2048 | [128,128] | 16·128·128 |
| H=16, HV=16, V=64 (expand_v=0.5) | 2048 | 1024 | [128,64] | 16·128·64 |

注意"总宽度都等于 d"只是习惯选择，不是约束；`K`、`V`、`H`、`HV` 是四个自由参数。

#### (6) 边界说明

上面 (2)(3)(4) 是基于代码与论文表述的机制解释。**K3 / Kimi Linear 报告里我没有读到 GVA 的消融结论**（论文的消融是 output gate、convolution、hybrid ratio、NoPE 四项），所以"H:HV 该取多少、为什么这样取更优"属于待验证部分，不要当作论文结论。

### 7.8 因果性：KDA 需要 causal mask 吗？

**结论：因果性是 KDA 的内生属性，但它和 softmax attention 的 causal mask 实现方式完全不同——递推形式根本没有 mask，分块形式只有 C×C 的三角结构。**

#### (1) 递推形式：因果就是定义本身

```text
S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ
o_t = scale · S_tᵀ q_t
```

`S_t` 只由 `≤ t` 的 token 累积而成，`o_t` 读的也是 `S_t`。**未来 token 根本没进过状态**，所以没有"需要屏蔽"的对象。这和 RNN、因果卷积一样：因果性由计算顺序保证，不需要一个 mask 张量。

#### (2) 分块形式：有显式三角结构，但是 C×C 而不是 T×T

分块后确实出现了显式的因果结构，出现在三处：

```text
① 块内注意力 Aqk[r,s] = Σ_d q_{r,d} k_{s,d} γ_{r,d}/γ_{s,d}   仅对 s ≤ r 定义
   → 下三角（含对角），这就是块内的 causal mask

② WY / UT 变换的 L[r,s] = β_r Σ_d k_{r,d} k_{s,d} γ_{r,d}/γ_{s,d}   仅对 s < r 定义
   → 严格下三角 Tril(diagonal=-1)

③ 块间：第 c 个 chunk 读到的状态 h 只来自 chunks < c（顺序扫描方向）
   → 跨块因果由扫描方向保证
```

Triton 代码里的对应物（`fla/ops/kda/wy_fast.py`）：

```python
m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)   # 严格下三角
```

注意这里的 mask 尺寸是 **BT×BT（chunk 内）**，不是 T×T。

#### (3) 与 softmax attention 的 causal mask 的本质区别

| | softmax attention | KDA |
|---|---|---|
| mask 是什么 | 加性 `-inf` 矩阵，`T×T` | 递推形式无 mask；分块形式是 `C×C` 三角 |
| 为什么需要 | 一次性算出所有 pair 的 score，必须手动屏蔽未来 | 状态里压根没有未来信息 |
| 代价 | `O(T²)` 的 score 和 `O(T²)` 的 mask | 递推 `O(1)`；分块 `O(T·C)` |
| 写错的后果 | 常见 | 在分块实现里忘加三角约束 → 看到未来 → loss 异常低 |

一句话：**softmax attention 是"先算全部再屏蔽"，KDA 是"从构造上就只见过过去"。**

#### (4) 但另一类 mask 必须显式处理：padding mask

因果性不需要管，padding 却必须管，而且在递推模型里**比 attention 更危险**：

```text
attention：padding 位置被 mask 掉，不影响别的 token
KDA      ：padding token 会写进状态 S，若不处理会污染后续真实 token 的输出
```

FLA 的做法（`fla/layers/kda.py`）：

```python
assert len(attention_mask.shape) == 2, (
    "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
    "for padding purposes (0 indicating padding). Arbitrary attention masks of "
    "shape [batch_size, seq_len, seq_len] are not allowed."
)
```

即：**只接受 `[B,T]` 的 padding mask，明确拒绝 `[B,T,T]` 的 attention mask**——因为 KDA 不需要后者。真正的处理方式是 `unpad_hidden_states` + `cu_seqlens`（varlen），让每条序列拥有独立的状态，而不是塞一个 mask 了事。

#### (5) 实验验证（已运行）

用 `kda_reference.py` 做扰动实验：只改某个 token 的 `k`，看输出前缀是否变化。配置 `T=6, K=V=2`。

```text
实验 1：扰动 t=5（最后一个 token）
  o[0] ~ o[4]  全部不变
  o[5]         改变              ← 因果边界正确

实验 2：扰动 t=3（中间 token）
  o[0] o[1] o[2]  全部不变
  o[3] o[4] o[5]  全部改变        ← 边界精确落在 t=3

实验 3：分块算法（chunk_size=2，chunk 划分为 [0,1],[2,3],[4,5]）
  扰动 t=3 时，o[2] 仍然不变
  ← t=2 与 t=3 虽在同一 chunk，但块内三角约束生效

实验 4：递推 vs 分块，输出最大误差 1.11e-16（等价性同时得到验证）
```

实验 3 尤其值得注意：它证明**分块不会放松因果性**，块内也必须严格三角。

#### (6) 复现时的常见错误

1. 分块实现里 `Aqk` 用了完整 `C×C`（漏掉 `tril`）→ 模型能看到未来，训练 loss 异常低、下游全崩。
2. 参考实现里把 `for s in range(r+1)` 写成 `for s in range(n)`——本文件的 `kda_chunked` 正是靠前者保证因果。
3. 把 padding 当成 causal mask 一并处理：只加 mask 不重置状态，padding 会污染后续 token；正确做法是 unpad/varlen。
4. 误以为"分块 = 块内可以互相看"。

### 7.9 KDA 像 RNN，怎么还能并行？（大白话版）

> 本节是重写版。初版一上来就用"结合律 / 仿射 / 双线性"，属于用没学过的词解释没学懂的事，已整体替换。

#### (1) 先说"串行"到底卡在哪

把递推公式摊开成 3 步，一眼就看到了：

```text
S₁ = 用 S₀ 算
S₂ = 用 S₁ 算
S₃ = 用 S₂ 算
```

**第 2 步要等第 1 步，第 3 步要等第 2 步**——像接力赛，一棒接一棒，三个人没法同时跑。

"串行"就这一个意思，没有别的玄机。

#### (2) 那"并行"想达到什么效果

并行 = **让 T 个 token 同时被处理**。

- attention 天然可以：每两个 token 算一次相似度，互不依赖，能一起算。
- RNN 不行：后面的必须等前面的结果。

#### (3) 用一个最简单的例子看"怎么办到"

任务：算 `[1,2,3,4]` 的前缀和，要得到 `[1,3,6,10]`。

```text
串行做法：一个一个加，4 步
   s1 = 1
   s2 = 1+2 = 3
   s3 = 3+3 = 6
   s4 = 6+4 = 10

并行做法：先分两半各自加，再合并，只要 2 轮
   第 1 轮（同时算）：p1 = 1+2 = 3      p2 = 3+4 = 7
   第 2 轮（同时算）：总和 = p1+p2 = 10
   补出前缀：s1=1, s2=p1=3, s3=p1+3=6, s4=10
```

两种做法结果完全一样（实测 `True`）。

为什么能分着做？因为**加法"先算哪两个都一样"**——这就是"结合律"三个字的意思。不用记这个词，记住"加法可以随便分组"就够了。

#### (4) 线性注意力的递推，恰好就是"累加"

```text
S_t = S_{t-1} + k_t v_tᵀ
```

每个 token 往状态里"加一块"，一直往下加——这就是累加，所以能像上面那样分组并行。

更关键的是：既然是纯累加，可以**跳过中间状态直接算**：

```text
o_t = 前 t 个 token 各自贡献的加权和
```

最小例子（K=1, V=1，全是标量）：

```text
k = [1,2,3]   v = [4,5,6]   q = [1,1,1]

写法一（递推：一边读一边攒）
   t=1: S = 0  + 1×4 = 4        o = 1×4  = 4
   t=2: S = 4  + 2×5 = 14       o = 1×14 = 14
   t=3: S = 14 + 3×6 = 32       o = 1×32 = 32
   o = [4, 14, 32]

写法二（直接算：每个输出各自挑自己需要的）
   o1 = 1×(1×4)                = 4
   o2 = 1×(1×4 + 2×5)          = 14
   o3 = 1×(1×4 + 2×5 + 3×6)    = 32
   o = [4, 14, 32]
```

结果一样（实测一致），但**写法二里三个 o 谁都不依赖谁**——可以同时算，这就是并行。

一句大白话概括区别：

```text
递推   = 一边读一边攒，必须按顺序
直接算 = 先把所有 token 摆出来，每个输出自己去挑需要的，互不干扰
```

#### (5) 门控不影响这件事

加个衰减系数，只是让每一"块"再乘一个数。"求和 + 乘系数"照样能分组做，所以门控不构成障碍。

#### (6) 但 KDA 的"擦除"把这条路堵了

KDA 在写入前要"先擦掉旧的"，而**擦多少取决于当前状态里已经有什么**。

打个比方：

```text
纯累加 ：往盒子里放东西，每件东西独立 → 可以先分堆放好再合并
带擦除 ：每次放之前要先看盒子里有什么、把同类的拿走 → 必须按顺序来
```

也就是说，每个 token 的贡献**不再独立**，而是依赖前面攒下来的结果。于是 (4) 里"直接算"的写法失效了。

三个实验的实测对照：

```text
实验 A  纯线性注意力（只累加）   递推 vs 直接算 误差 1.67e-16  → 完全一样
实验 B  加 channel-wise 门控      递推 vs 直接算 误差 1.11e-16  → 完全一样
实验 C  KDA 加 delta 擦除         递推 vs 直接算 误差 9.79e-03  → 不一样，失效
```

前两个误差是机器精度级别的 0，第三个明显不是 0——这就是"擦除堵住了并行"的实测证据。

#### (7) 那 KDA 到底怎么办？（分块，下次展开）

思路一句话：

```text
把长序列切成小段
  段内：先当"这段里没有擦除"，于是可以并行算
  段间：按顺序把状态一段段传下去
```

效果是把"全程串行"变成"小段并行 + 少量串行"：既有并行度，又不用 O(T²) 那么贵。

至于段内怎么"假装没有擦除"还能保证结果正确，是 WY 表示要解决的问题，下次单独讲。

#### (8) 一页总结

| 问题 | 答案 |
|---|---|
| 为什么 RNN 不能并行？ | 第 t 步依赖第 t−1 步，像接力赛 |
| 那怎么才能并行？ | 找一个**等价**的写法，让每个输出的各项互不依赖 |
| 什么时候找得到？ | 运算满足"先算哪两个都一样"（加法就是） |
| 纯线性注意力能不能？ | 能，因为只是累加 |
| 加门控能不能？ | 能，只是每项多个系数 |
| KDA 的擦除能不能？ | 不能，每步依赖前面状态 → 需要分块 |
| 分块做了什么？ | 段内并行 + 段间串行（下次讲） |

一句话记住：

> **并行不是把 RNN 变成不串行，而是换一个等价的、不串行的算法去算同一个结果。**

#### (9) 工程上怎么体现

`fla/layers/kda.py`：

```python
if self.training:
    assert mode == "chunk", "Only chunk mode is supported in training."
```

训练时强制走分块（并行形式），推理时才用递推。这就是"KDA 是 RNN，但训练也能并行"最直接的证据。

### 7.10 分块算法和原 KDA 是什么关系

#### (1) 一句话：它就是原 KDA 递推的另一种等价算法

```text
原 KDA 递推  ──换一种等价写法──→  分块算法

同一组输入、同一个输出，逐位相同（误差 1e-16）
不是新模型，不是近似逼近 —— 只是"换个算法算同一个东西"
```

为什么要换：

```text
递推：公式简单好懂，但必须 T 步串行  → 训练时用不了
分块：数学上完全等价，但可以并行     → 训练用它
```

#### (2) 把原递推"展开"，M 和 B 就自然长出来了

原递推是（记每次的**转移矩阵** `T_r = (I − β_r k_r k_rᵀ) Diag(α_r)`，**写入** `W_r = β_r k_r v_rᵀ`）：

```text
S_r = T_r · S_{r-1} + W_r
```

从一个块的第 1 个 token 开始，逐层代入：

```text
S_1 = T_1 S_0 + W_1
S_2 = T_2 S_1 + W_2 = T_2 T_1 S_0 + T_2 W_1 + W_2
S_3 = T_3 S_2 + W_3 = (T_3 T_2 T_1) S_0 + (T_3 T_2 W_1 + T_3 W_2 + W_3)
                      └───── M ─────┘   └────────── B ──────────┘
```

**M 和 B 不是新引入的概念——它们就是把递推式逐层代入之后自然出现的那两坨。**

```text
M = 块内所有转移矩阵的连乘   →  "这个块把传进来的旧状态改成了什么"
B = 假设 S_in = 0 时的结果   →  "这个块自己往里写了什么"

于是：S_out = M · S_in + B
```

关键性质：**M、B 只和这个块自己的 token 有关，与传进来的 `S_in` 无关**。所以每个块都能独立算、并行算。

#### (3) 分块算法算的就是同一个 M 和 B（数值验证）

```text
连乘算法：M = T_3 T_2 T_1
分块算法：M = Diag(γ_last) − kgᵀw

  连乘 M = [[0.194373, -0.096942], [-0.017734, 0.053192]]
  分块 M = [[0.194373, -0.096942], [-0.017734, 0.053192]]
  误差   = 2.8e-17        ← 同一个矩阵

  B 的误差 = 5.6e-17      ← 同一个 B
```

所以下面这些符号没有一个"是新的"：

```text
γ 来自递推里的 Diag(α) 连乘
L 来自 token 之间互相擦除的干扰
A 是把这些干扰解耦
w 是"βk"的解耦修正版
u 是"βv"的解耦修正版
M 就是 T_C⋯T_1
B 就是那些 W 的累计
```

#### (4) 那为什么不直接用"连乘"

因为连乘 C 个 `d_k × d_k` 矩阵有三重问题：

```text
① 太贵：O(C · d_k³)。d_k = 128 时，一个块就是好几亿次运算
② 串行：T_C · T_{C-1} · … 这 C 次乘法本身必须顺序做
③ 用不上批量 matmul 的高效实现
```

WY 表示换了条路算**同一个 M、B**：

```text
先构造 C×C 的 L → 求逆得 A（C×C，前向代入）→ 再和 [C,d] 矩阵相乘
C=64、d_k=128 时，比连乘便宜好几个数量级，而且块内全是稠密矩阵乘
```

#### (5) 符号翻译表（最需要记住的一张）

| 分块里的符号 | 它在原 KDA 里对应什么 |
|---|---|
| `γ_r` | 块内前 r 个 token 的衰减连乘 `∏ α`（就是把递推里那些 `Diag(α)` 合并） |
| `L[r,s]` | 第 r 个 token 擦除时，受第 s 个 token 影响多少 |
| `A = (I+L)^{-1}` | 把 C 次"互相干扰的擦除"解耦 |
| `w` | 每个 token 的**实际擦除方向**（原始是 `β_r k_r`） |
| `u` | 每个 token 的**实际写入内容**（原始是 `β_r v_r`） |
| `kg_r` | 把 `k_r` 归一到块尾的衰减刻度 |
| `M` | `T_C ⋯ T_1`（块内转移矩阵连乘） |
| `B` | 块内所有写入的累计 |
| `Aqk` | 块内 C×C 注意力，对应 `o_t = S_tᵀ q_t` 展开后的块内部分 |

**`w` 和 `u` 值得单独说**：它们不是新东西，就是 `βk` 和 `βv` 的"解耦修正版"。

```text
如果 token 之间互不干扰（比如 key 两两正交），那么
    L = 0 → A = I → w = βk，u = βv        ← 直接就是原式
但实际 token 之间会互相擦除：
    第 2 个 token 擦除时，状态里已经含有第 1 个 token 写进去的东西；
    第 3 个 token 擦除时，又含有前两个写进去的……这种耦合就是 L 描述的，
    A = (I+L)^{-1} 的作用就是把它解耦回"各自独立"的形式。
```

#### (6) 输出也来自同一个式子 `o_t = S_tᵀ q_t`

原递推里 `o_t = scale · S_tᵀ q_t`，而 `S_t` 由两部分组成：

```text
(a) 这个块之前累积的状态（经 M 传进来）  →  分块里的  qg @ h
(b) 本块内前 t 个 token 的贡献            →  分块里的  Aqk @ v_new
```

所以：

```text
o_c = scale · ( Aqk_c @ v_new_c   +   (q_c ⊙ γ_c) @ h_c )
                └── 对应 (b) ──┘       └─── 对应 (a) ───┘

h_c      = 这个块开始时的状态（块间扫描给出）
v_new_c  = u_c − w_c @ h_c          （把传入状态的贡献先扣掉）
Aqk[r,s] = Σ_d q_{r,d} k_{s,d} γ_{r,d}/γ_{s,d}    (s ≤ r)
```

`Aqk` 就是**块内第 r 个 query 对第 s 个 key 的注意力**，是 `o_t = S_tᵀq_t` 在块内展开后的那一部分。它只有 `C×C`，不是 `T×T`。

#### (7) 两级结构：并行到底在哪

```text
┌─ 阶段 1（完全并行，块之间无依赖）─────────────┐
│  每个块独立算：γ、L、A、w、u、kg、Aqk           │
└───────────────────────────────────────────────┘
                    ↓
┌─ 阶段 2（串行，但只有 T/C 步）─────────────────┐
│  h_0 = 0                                      │
│  for c in 0..NT-1:                            │
│      o_c   = scale·( Aqk_c @ (u_c − w_c h_c)  │
│                      + qg_c @ h_c )           │
│      h_{c+1} = M_c h_c + B_c                  │
└───────────────────────────────────────────────┘
```

**阶段 1 之所以能并行，就是因为 `M`、`B` 与 `S_in` 无关**——不需要知道前面的状态就能算出来。

实测（`T=6, C=3, d_k=d_v=2`）：

```text
块开始状态（扫描给出）：
  h0 = [[ 0.000000,  0.000000], [ 0.000000,  0.000000]]
  h1 = [[-1.642743,  0.598522], [-0.558708, -0.243377]]
  h2 = [[ 0.169111,  0.002320], [ 0.045944, -0.078983]]

完整串行在位置 0 / 3 / 6 的状态：
  位置 0 = [[ 0.000000,  0.000000], [ 0.000000,  0.000000]]
  位置 3 = [[-1.642743,  0.598522], [-0.558708, -0.243377]]
  位置 6 = [[ 0.169111,  0.002320], [ 0.045944, -0.078983]]
  ✓ 完全一致；输出逐位置断言误差 < 1e-12
```

#### (8) 串行深度与计算量

```text
              串行深度      计算量
纯递推        T 步          O(T · d_k · d_v)
分块 C        T/C 步        O(T·C·d) + O(T·d_k·d_v)
朴素并行      1 步          O(T²·d)（对 delta 无效）

T=1M, C=64：1,000,000 步 → 15,625 步（串行深度减少 64 倍）
```

C 是折中：C 大 → 串行步数少（好），但块内 O(C²) 更贵（坏）。所以实现里固定在 32 或 64：

```python
if chunk_size not in (32, 64):
    raise ValueError(f"`chunk_size` must be either 32 or 64 for KDA, got {chunk_size}.")
```

#### (9) 为什么叫 "WY 表示"

因为**一串联的秩 1 变换可以合并写成 `(I − W Kᵀ)` 的形式**——用两个矩阵 W、K 就代表了块里全部 64 次擦除。这和线性代数里求 Householder 乘积的 WY 分解是同一个套路。论文说的 "specialized variant of DPLR" 指的就是：

```text
M = Diag(γ_last) − kgᵀw
    └─ 对角 ─┘   └ 低秩 ┘
     门控衰减    块内擦除+写入的合并，秩 ≤ C
```

#### (10) 和 Gated MLA 的并行是两件独立的事

```text
Gated MLA 层：标准 SDPA → FlashAttention 在序列维并行（不物化 T×T）
              占 1/4 的层
KDA 层      ：分块算法 → 块内并行 + T/C 步块间扫描
              占 3/4 的层
```

两者互不依赖，整体并行度是各自并行度的叠加。

再往上一层是 Context Parallelism：把序列维切到多张卡（FLA 的 `fla/ops/cp`、K3 报告 §5.1.2），把块间扫描也跨卡并行化。

#### (11) 易错点

```text
1. L 里的 β 取行下标 r（标准写法 Diag(β)KKᵀ），不是列下标 s
2. Aqk 里不能再乘 β —— β 已经通过 u = A(β⊙v) 进去了
3. γ 是"块内"累计，每个块从 1 重新开始，不是全局累计
4. kg 用的是 γ_last/γ_r（归一到块尾），方向别写反
5. M 是方阵但秩 ≤ C，不要以为它满秩
6. 块间扫描必须顺序执行（唯一串行的部分）
7. 最好的单元测试：分块结果与逐 token 递推逐位一致
8. 记住 M 和 B 的定义是"从递推展开出来的"，而不是凭空引入的新参数
```

## 八、逐组件拆解

按用户要求逐个组件拆解，每次只讲一个。讲解顺序：β → α → k/v → q → 短卷积 → A/W/U → 输出门（进行中，已完成 β、α）。

### 8.1 β（beta）：写入强度，也就是 delta rule 的步长

#### (1) 定义与形状

```text
实现：beta = sigmoid(W_b x)                    # [B, T, HV]
      allow_neg_eigval=True 时 beta = 2·sigmoid(W_b x)   # 取值范围 (0, 2)
```

一个 **token × 一个 value head** 一个标量。它不是标量常数，而是每个 token 现场算出来的。

#### (2) 作用一：它就是 delta rule 的学习率

回顾 delta rule 的推导：

```text
损失：   L(S) = ½‖S^T k_t − v_t‖²
梯度：   ∇_S L = k_t (k_t^T S − v_t^T)
梯度步： S_t = S_{t-1} − β_t ∇_S L = (I − β_t k_t k_t^T) S_{t-1} + β_t k_t v_t^T
```

所以 β_t 是**在线梯度下降的步长**，不是某种额外的门。这也解释了为什么它和 `I − β k k^T` 里的 β 是同一个数——它们必须一致。

#### (3) 作用二：擦除与写入共用同一个 β

把更新式整理成"残差"形式：

```text
S_t = S_{t-1} + β_t · k_t · (v_t − S_{t-1}^T k_t)^T
                              └────── 预测残差 ──────┘
```

含义很清楚：**只在 k_t 方向上，按 β_t 的比例补上"当前预测与真实 v_t 的差距"**。

- β_t = 0：完全不更新（记忆冻结）
- β_t = 1：把 k_t 方向的旧分量完全替换掉（‖k‖=1 时是精确覆盖）
- 0 < β_t < 1：部分更新，旧值保留一部分

擦除项 `−β k k^T S` 与写入项 `+β k v^T` 被同一个 β 缩放，所以**不能**"只写不擦"或"只擦不写"。

#### (4) 作用三：β 决定转移矩阵的特征值 → 是否允许负特征值

`T = I − β k k^T` 在 k 方向上的特征值是 `1 − β`（其余 K−1 个方向特征值为 1）：

```text
β = 0    → 特征值  1  ：不遗忘，不变
β = 1    → 特征值  0  ：精确擦除该方向
β ∈ (1,2)→ 特征值 负  ：出现反射分量（Householder 型）
```

这就是实现里 `allow_neg_eigval=True` 让 β 取 `2·sigmoid(·)` 的原因：把特征值开到负数，线性 RNN 才能表达状态跟踪（state tracking）这类需要"取反"的操作，参考 *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues*（arXiv:2411.12537，FLA 文档里直接引用了这篇）。

#### (5) 作用四：选择性写入（为什么必须 data-dependent）

`β = sigmoid(W_b x_t)` 让模型自己决定"当前这个 token 值不值得写进记忆"：

```text
承载关键事实的 token  → 大 β，强写入
标点、重复、噪声 token → 小 β，几乎不动记忆
```

这是 KDA 能扛长上下文的原因之一：不是每个 token 都用同样的强度污染有限容量的状态。

#### (6) β 与 α 的分工（最容易混的一对）

| | α（`Diag(α_t)`） | β（`beta_t`） |
|---|---|---|
| 作用对象 | **已有状态**的幅度 | **本次更新量**的幅度 |
| 语义 | 遗忘 / 时间衰减 | 写入强度 / 覆盖程度 |
| 形态 | K 维向量（channel-wise） | 标量（per value head） |
| 作用范围 | 整个状态矩阵的所有 K 行 | 只沿当前 `k_t` 方向 |
| 是否依赖当前内容 | 是（由 x_t 算），但作用于"过去" | 是，作用于"现在" |

两者都能让旧信息"消失"，但机制不同：α 是**乘性衰减已有记忆**，β 是**控制新信息覆盖旧信息的程度**。

#### (7) β 在分块算法里的位置（含一个真实踩过的坑）

```text
含 β：L[r,s] = β_r · Σ_d k_{r,d} k_{s,d} γ_{r,d}/γ_{s,d}      # β 取行下标 r
      w      = A · (β ⊙ k ⊙ γ)
      u      = A · (β ⊙ v)
不含 β：Aqk[r,s] = Σ_d q_{r,d} k_{s,d} γ_{r,d}/γ_{s,d}          # 注意这里没有 β
```

两个坑：

1. **β 取行下标 r**，不是列下标 s（标准写法是 `StrictTril(Diag(β) K K^T)`）。写成本文件配的 `kda_reference.py` 时先写成 `β_s`，自检立刻报错，改成 `β_r` 后全部通过。
2. **`Aqk` 里不能再乘 β**。β 已经通过 `u = A(β ⊙ v)` 进入计算，若 `Aqk` 再乘一次就是重复计入。

#### (8) β 的数值行为（已运行验证）

配置：`k=(1,0)`（单位向量）、`v=(1,0)`、`α=1`、`S_0` 在 k 方向的分量为 `0.5`。残差恒为 `v − s0 = 0.5`。

```text
  beta   T 沿 k 的特征值   残差     更新量   S_1 的 k 分量
   0.0          1.0        0.5      0.0          0.5
  0.25         0.75        0.5     0.125         0.625
   0.5          0.5        0.5     0.25          0.75
   1.0          0.0        0.5      0.5          1.0
   1.5         -0.5        0.5     0.75          1.25
   2.0         -1.0        0.5      1.0          1.5
```

读法：

- 更新量恒为 `β × 残差`，β 就是步长。
- β=0.5 时只走一半，旧值残留 0.25；
- β=1 时精确落到 v=1；
- β>1 会**冲过头**（1.25、1.5），对应特征值转负。

#### (9) β 的易错点小结

1. β 不是"遗忘门"，遗忘是 α；β 是写入强度。
2. 擦除和写入共用 β，不能各自独立。
3. β 的 shape 是 `[B,T,HV]`（per value head），因为记忆是 per value head 的；GVA 下 q/k 头被共享，所以 β 不能按 H 索引。
4. 分块算法里 β 在 `L`/`w`/`u` 中出现，在 `Aqk` 中不出现。
5. `allow_neg_eigval` 改的是 β 的上界（2 而不是 1），进而让转移矩阵出现负特征值。

### 8.2 α（alpha）：channel-wise 的遗忘率，为什么存成向量而不是矩阵

#### (1) 直接回答

`Diag(α_t)` 是**数学形式**，α_t 本体是一个 **K 维向量**。代码里存的是"对角线本身"，不会去构造 K×K 矩阵。

形状 `[B, T, HV, K]` 的两个维度分别是：

```text
HV → 每个 value head 维护自己独立的 S ∈ R^{K×V}，所以每个 head 一条对角线
K  → 对角线的长度 = key 维数（因为 Diag(α) 左乘 S 是"按行缩放"，行就是 key 维）
```

#### (2) 数学对象 vs 存储对象

| | 数学写法 | 代码存储 | 形状 |
|---|---|---|---|
| 矩阵 | `Diag(α_t)` | 不构造 | `[K, K]` |
| 向量 | `α_t` | 实际存储 | `[K]` |
| 一批 | | `alpha` | `[B, T, HV, K]` |

论文原文写的是 `Diag(𝜶_t)` 并称其为 fine-grained decay，同时定义 `Diag(𝜶_t)` 的对角就是向量 𝜶_t——两者是同一个东西的两种写法。

计算上：

```python
# 形式一（概念）：显式对角矩阵
S = Diag(alpha) @ S

# 形式二（实现）：只存对角线，按行广播
S = alpha.unsqueeze(-1) * S      # [.., K, 1] 广播到 [.., K, V]
```

实测验证（`α=(0.5,0.1)`、`S=[[0,5],[9,0]]`）：

```text
Diag(α) = [[0.5, 0.0],
           [0.0, 0.1]]
Diag(α)@S = [[0.0, 2.5],
             [0.9, 0.0]]      ← 第 1 行 ×0.5，第 2 行 ×0.1
α ⊙ 行     = [[0.0, 2.5],
             [0.9, 0.0]]      ← 完全相同
```

省多少：T=4096, HV=16, K=V=128 时，

```text
显式存 K×K 矩阵：4096 × 16 × 128 × 128 = 1.07e9 个元素
只存对角线向量  ：4096 × 16 × 128       = 8.39e6 个元素      → 省 128 倍
matmul 代价     ：O(K²V)=2.1e6          → O(KV)=1.6e4        → 省 128 倍
```

#### (3) 为什么长度是 K 而不是 V

状态是 `S ∈ R^{K×V}`，`Diag(α)` 在**左侧**，所以缩放的是第 i **行**：

```text
(Diag(α) S)[i, j] = α_i · S[i, j]        i 是 key 维，j 是 value 维
```

即"每个 key 维度有独立的遗忘率"。

如果要在 value 维上加门，那是**右乘** `S Diag(γ)`，γ ∈ R^V，缩放的是列。KDA 不做这件事——它的门在 key 维上。（GLA 一类结构里的输出门 `g_proj` 是另一回事，它作用在输出上，不是状态。）

#### (4) 为什么是 HV 而不是 H

因为状态本身就是 per value head 的：

```text
S ∈ [B, HV, K, V]
α ∈ [B, T, HV, K]     ← 每个 head 一条独立对角线，不能共享
```

GVA（`HV > H`）下多个 value head 共享同一个 q/k 头，但**状态是各自独立的**，因此 α 和 β 一样必须按 value head 索引，不能按 q/k 头索引。若 `HV = 1`，α 就退化成 `[B,T,K]`。

#### (5) "α 是向量"这件事本身就是 KDA 的核心创新

```text
GDN：α_t 是标量  →  Diag(α_t) = α_t · I   →  各向同性，所有 K 维同速衰减
KDA：α_t 是 K 向量 →  真正的对角矩阵       →  各向异性，每维独立遗忘率
```

也就是说：**只有 α 是向量时，"Diag" 才有实际内容**。如果 KDA 也用标量门，这个维度根本不会存在，形状会退化成 `[B,T,HV]`。论文里 "finer-grained gating"、"channel-wise"、"each feature dimension maintains an independent forgetting rate" 指的都是这一件事。

#### (6) 为什么整个分块机制还能只用向量算

关键性质：**对角矩阵的乘积等于对角线的逐元素乘积**

```text
Diag(a) · Diag(b) = Diag(a ⊙ b)
```

因此论文里可以直接写

```text
Diag(γ^{i→j}) := ∏_{k=i}^{j} Diag(α_k)      ← 对角矩阵层面
γ^{i→j}       = ∏_{k=i}^{j} α_k              ← 向量层面，等价
```

chunk 内用到的 `A[i/j] = γ^i / γ^j`、`γ_last/γ_r` 也全部只在向量层面对齐运算。这是分块算法能做到"只有 K 维向量、没有 K×K 矩阵"的根本原因。

#### (7) α 在实现里怎么产生（形状全链路）

```text
f_proj = Sequential( Linear(d, head_v_dim), Linear(head_v_dim, HV*K) )   # 低秩瓶颈
raw    = f_proj(x) + dt_bias          # [B,T,HV*K] → reshape → [B,T,HV,K]
A      = exp(A_log)                   # [HV] 每个 head 一个时间常数
g      = -A * softplus(raw)           # [B,T,HV,K]
alpha  = exp(g)                       # [B,T,HV,K]，取值 (0,1]
```

参数量对比（d=2048, V=128, HV=16, K=128）：

```text
直接算门：      d × HV × K = 2048 × 2048          = 4.19M
低秩瓶颈：      d × V + V × HV × K = 0.262M + 0.262M = 0.52M     → 约 8 倍
```

`dt_bias` 的形状也是 `[HV*K]`（每个 head 的每个 key 维一个偏置），`A_log` 是 `[HV]`（每个 head 一个标量时间常数）。

#### (8) 易错点

1. 不要真的构造 `K×K` 矩阵——既费显存又费算力，而且数值上没必要。
2. 广播维度必须加在 **K** 上：`alpha.unsqueeze(-1) * S`（得到 `[..,K,1]`），写成 `unsqueeze(-2)` 就变成缩放 value 维了，方向错误。
3. 论文里 `Diag(α)` 和 `α` 混用，看到 `Diag(·)` 要自动读作"对角线向量"。
4. 左乘缩放行（key 维），右乘缩放列（value 维），别搞反。
5. 分块里的 `γ_last`、`γ_r` 同样是 K 向量，缩放的也是 `h` 的行。
6. α 是**记忆衰减门**，与输出门 `g_proj` 完全无关。

## 九、代码符号约定（@、⊙、*）

KDA 的公式和代码里混用了三种"乘法"，含义完全不同，先统一约定。

### 9.1 `@` 是矩阵乘法运算符

`@` 是 Python 3.5 起引入的**矩阵乘法运算符**（PEP 465，`__matmul__`），等价于：

```python
a @ b  ==  numpy.matmul(a, b)  ==  torch.matmul(a, b)  ==  einsum('...ik,...kj->...ij', a, b)
```

它对应数学里紧挨着写的那种矩阵乘法 `AB`，**不是**逐元素乘法。

### 9.2 `@` 与 `*` 的区别

| 运算符 | 名称 | 触发的方法 | 要求 | 结果 |
|---|---|---|---|---|
| `@` | 矩阵乘法（matmul / 内积收缩） | `__matmul__` | 左矩阵的**最后一维** = 右矩阵的**倒数第二维** | 收缩掉这一维 |
| `*` | 逐元素乘（Hadamard） | `__mul__` | 形状相同（或可广播） | 形状不变 |

实测（自定义最小矩阵类展示运算符分派）：

```text
A.shape=(2,3)  B.shape=(3,2)  C.shape=(2,3)

A @ B  →  __matmul__   (2,3) @ (3,2) = (2,2)   →  [[4,5],[10,11]]
A * C  →  __mul__      (2,3) * (2,3) = (2,3)   →  [[10,40,90],[160,250,360]]
A @ C  →  报错：内维 3 与 2 不匹配
```

### 9.3 形状规则

```text
一维特例（numpy / torch 约定）
  [n]   @ [n]   → 标量            （内积）
  [m,n] @ [n]   → [m]             （矩阵乘向量）
  [n]   @ [n,p] → [p]             （向量乘矩阵）

二维
  [m,n] @ [n,p] → [m,p]

批量（前面若干维是 batch 维，按广播规则对齐）
  [B, m, n] @ [B, n, p] → [B, m, p]
  [B, m, n] @    [n, p] → [B, m, p]        # 右侧自动广播
  [B, 1, n, p] 与左侧 batch 维广播
```

一句话记法：**"左取倒数第二维，右取最后一维，两者必须相等，然后消掉它们"**。

### 9.4 底层是什么算法

`@` 最终落到底层 BLAS 的 **GEMM**（通用矩阵乘）：

```text
朴素实现：三重循环，复杂度 O(m·n·p)
库实现：  BLAS 的 SGEMM/DGEMM（CPU）、cuBLAS 的 GEMM（GPU）
批量的：  batched GEMM，每个 batch 独立做 GEMM
```

这就是为什么"分块算法把递归改写成 matmul"能大幅提速——matmul 能吃到 TensorCore。

### 9.5 KDA 代码里每一处 `@` 的形状对照

```text
① 遗忘      alpha[..., :, None] * S         ← 逐元素，不是 @
② 擦除      erase @ S                       [K,K] @ [K,V] = [K,V]
③ 写入      beta * (k ⊗ v)                  ← 外积，逐元素缩放
④ 读出      S.transpose(-1,-2) @ q          [V,K] @ [K,1] = [V,1]
⑤ 块内注意力 Aqk @ v_new                     [C,C] @ [C,V] = [C,V]
⑥ 跨块读出  qg @ h                          [C,K] @ [K,V] = [C,V]
⑦ 写入状态  kg.transpose(-1,-2) @ v_new      [K,C] @ [C,V] = [K,V]
⑧ 残差预测  w @ h                           [C,K] @ [K,V] = [C,V]
```

注意 ①③ 用的是 `*`（逐元素），其余用的是 `@`（矩阵乘）。这是读 KDA 代码时最容易看错的地方。

### 9.6 不要和装饰器 `@` 混淆

同一个字符在 Python 里有两种完全无关的用法：

```python
@torch.jit                       # 行首的 @ = 装饰器（语法糖，等价 f = torch.jit(f)）
def foo(): ...

y = A @ B                        # 表达式中间的 @ = 矩阵乘法运算符
```

FLA 的 Triton kernel 文件里两者都大量出现，读的时候按位置区分。

### 9.7 与数学符号的对照表

| 数学写法 | 代码写法 | 名称 |
|---|---|---|
| `AB` 或 `Diag(α)S` | `A @ B` | 矩阵乘（收缩） |
| `a ⊙ b` 或 `β ⊙ k` | `a * b` | 逐元素乘（Hadamard） |
| `k v^T` | `k.unsqueeze(-1) * v.unsqueeze(-2)` | 外积 |
| `k^T S` | `k.transpose(-1,-2) @ S` | 内积/收缩 |
| `⟨a, b⟩` | `(a * b).sum(-1)` | 内积（标量） |

### 9.8 为什么 `kt.unsqueeze(-1) * v[:, t].unsqueeze(-2)` 等于 `k_t v_tᵀ`

#### (1) `unsqueeze` 做的是"插入一个长度为 1 的维度"

它不改数据，只改形状：

```text
k_t : [K]      --unsqueeze(-1)-->  [K, 1]     # 最后插一维 → 列向量
v_t : [V]      --unsqueeze(-2)-->  [1, V]     # 倒数第二插一维 → 行向量
```

`-1` 表示最后一个位置，`-2` 表示倒数第二个位置。

#### (2) 广播规则：从最后一维对齐，size-1 被拉伸

```text
   [K, 1]
      [1, V]
-----------  从右往左对齐
   [K, V]     两边都是 1 的位置被对方的大小"拉伸"
```

于是 `[K,1] * [1,V]` 广播成 `[K,V]`，第 (i,j) 个元素 = `k_i * v_j`。

而 `k vᵀ` 的定义正是 `(k vᵀ)[i,j] = k_i · v_j`——所以两者是同一个东西。

#### (3) 实测

```text
k = [2.0, 5.0]        v = [3.0, 7.0]

k.unsqueeze(-1) = [[2.0], [5.0]]       shape [K,1]
v.unsqueeze(-2) = [[3.0, 7.0]]         shape [1,V]

[K,1] * [1,V] 广播相乘 = [[6.0, 14.0],
                          [15.0, 35.0]]     shape [K,V]

k vᵀ = [[6.0, 14.0],                    ← 完全相同
        [15.0, 35.0]]
v kᵀ = [[6.0, 15.0],                    ← 转置，不同
        [14.0, 35.0]]
```

#### (4) 为什么一个用 -1、一个用 -2

因为 `k vᵀ` 里：**k 是列向量 `(K,1)`，vᵀ 是行向量 `(1,V)`。**

```text
列向量  ←  unsqueeze(-1)      [n] → [n,1]
行向量  ←  unsqueeze(-2)      [n] → [1,n]
外积    =  列 × 行
```

口诀：**谁当"列"就用 `-1`，谁当"行"就用 `-2`。**

#### (5) 反例：两个位置对调会得到转置

```text
v.unsqueeze(-1) * k.unsqueeze(-2)
= [V,1] * [1,K]
= [V,K]                 ← 形状反了，得到的是 v kᵀ

实测 = [[6.0, 15.0],
        [14.0, 35.0]]
```

广播不会报错，只会**静默**给出 `[V,K]`。所以这里必须盯住 shape。

#### (6) 为什么这里用 `*` 而不是 `@`

```text
k * v 形式的逐元素广播可以跨维度"撑开"，直接得到外积
@ 会做收缩：[K] @ [V] 在 K ≠ V 时直接报错
```

补充：`k.unsqueeze(-1) @ v.unsqueeze(-2)` 其实**也**等于外积——因为 `[K,1] @ [1,V] = [K,V]`，收缩维长度正好是 1。只是用 GEMM 去做长度为 1 的收缩没必要，`*` 更直接也更省。

其他等价写法：

```python
torch.outer(k, v)                              # 一维专用
torch.einsum('...i,...j->...ij', k, v)         # 通用、最不易错
k.unsqueeze(-1) * v.unsqueeze(-2)              # 广播写法（KDA 实现用的）
```

#### (7) batch 维不参与广播

`k[:, t]` 是 `[B, HV, K]`，`v[:, t]` 是 `[B, HV, V]`：

```text
k[:, t].unsqueeze(-1)  :  [B, HV, K, 1]
v[:, t].unsqueeze(-2)  :  [B, HV, 1, V]
相乘                    :  [B, HV, K, V]
```

前两维 `[B, HV]` 两侧完全一致，直接对齐，只有最后两维在做"1 → 对方大小"的拉伸。

#### (8) 数学意义：这是 delta rule 的写入项

```text
β_t · k_t v_tᵀ        ← 一个秩 1 矩阵
```

状态是 `S ∈ R^{K×V}`：**行 = key 维（地址），列 = value 维（内容）**。外积把"键 k_t 对应值 v_t"这条关联写进状态，(i,j) 位置累积 `k_i · v_j`。

回到前面已验证的玩具配置（`k=(1,0)`、`v=(0.731059,0)`、`β=0.731059`）：

```text
k vᵀ       = [[0.731059, 0.0],
              [0.0,      0.0]]

β · k vᵀ   = [[0.534447, 0.0],
              [0.0,      0.0]]      ← 与递推演练 t=1 的⑤写入项完全一致
```

#### (9) 易错点

1. `unsqueeze(-1)` 与 `unsqueeze(-2)` 决定谁是行、谁是列，直接决定结果是 `k vᵀ` 还是 `v kᵀ`。
2. 广播是**从右往左**对齐，不是从左往右。
3. 外积用 `*`，矩阵乘用 `@`；形状错了广播**不会报错**，只会静默算错。
4. 不确定时用 `einsum('...i,...j->...ij', k, v)`，下标本身就是自文档，能避免行列颠倒。
5. 注意 `v[:, t]` 里 `t` 是取第 t 个 token，`[:, t]` 的切片让 token 维消失、head 维保留，别和转置混淆。

### 9.9 `x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)` 是什么

#### (1) 一句话

这一行就是 **RMSNorm**（Root Mean Square Normalization，均方根归一化）的核心计算：

```text
把向量 x 除上它自己的"均方根"，使输出的均方根变成 1。
```

#### (2) 逐段拆解

```python
x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
```

| 片段 | 作用 | 形状变化（以 `[B, T, d]` 为例） |
|---|---|---|
| `x.pow(2)` | 逐元素平方 | `[B, T, d]` → `[B, T, d]` |
| `.mean(-1, keepdim=True)` | 在**最后一维**（特征维）求平均 | `[B, T, d]` → `[B, T, 1]` |
| `+ self.eps` | 加一个极小数防除零 | `[B, T, 1]` |
| `torch.rsqrt(...)` | `1 / sqrt(...)`，即"平方根的倒数" | `[B, T, 1]` |
| `x * ...` | 广播相乘，把 x 缩放到均方根为 1 | `[B, T, 1]` 广播到 `[B, T, d]` |

对应的数学式：

```text
                 x
x_norm = ─────────────────────
          sqrt( mean(x²) + ε )

等价写法：
x_norm = x / RMS(x),    其中 RMS(x) = sqrt( mean(x²) )
```

#### (3) 数值演示（`x = [3, -4, 0, 5]`）

```text
x                = [3.0, -4.0, 0.0, 5.0]
x.pow(2)         = [9.0, 16.0, 0.0, 25.0]
.mean(-1)        = 12.500000         ← 均方
+ eps            = 12.500001
rsqrt(...)       = 0.282843          ← 1/sqrt(12.5)
x * rsqrt(...)   = [0.848528, -1.131371, 0.0, 1.414214]

校验：输出的均方 = 0.99999992 → RMS ≈ 1   ✓
```

#### (4) `eps` 是干什么的

如果 `x` 全是 0，`mean(x²) = 0`，`1/sqrt(0)` 会变成 `inf`，再乘回 `x=0` 就得到 `nan`。加一个极小的 ε 避免这个除零：

```text
不加 eps：mean=0 → 1/sqrt(0) → inf → nan
加 eps  ：mean=0 → 1/sqrt(1e-6) = 1000 → 0 × 1000 = 0   ✓ 安全
```

典型取值 `1e-5`~`1e-6`。

#### (5) 为什么必须 `keepdim=True`

因为要靠**广播**把缩放系数乘回 `x`：

```text
x 形状                  = [B, T, d] = [2, 3, 4]
mean(-1)               = [2, 3]        ← 最后一维被消掉了
x * [2,3]              → 末尾对齐：[2,3] vs [2,3,4] → 3 与 4 对不上 → 报错

mean(-1, keepdim=True) = [2, 3, 1]
x * [2,3,1]            → 广播成 [2,3,4]   ✓ 沿特征维缩放，正确
```

#### (6) RMSNorm vs LayerNorm：只差一个"减均值"

```text
LayerNorm：先减均值（中心化），再除以标准差
           y = (x − mean(x)) / sqrt(var(x) + ε)

RMSNorm  ：不减均值，只除以均方根
           y = x / sqrt(mean(x²) + ε)
```

同一个 `x = [3, -4, 0, 5]`（均值 = 1）：

```text
LayerNorm：减均值后 [2,-5,-1,4] → 除以标准差 3.391 → [0.5898, -1.4744, -0.2949, 1.1795]
           结果均值 ≈ 0，方差 = 1

RMSNorm  ：不减均值 → [0.8485, -1.1314, 0.0, 1.4142]
           结果均值 = 0.2828 ≠ 0     ← 这就是两者的唯一区别
```

RMSNorm 少一次均值计算和一次减法，更快；论文（Zhang & Sennrich, 2019）表明效果与 LayerNorm 相当，因此现代 LLM 普遍用它。

#### (7) `rsqrt` 是什么

```text
rsqrt(z) = 1 / sqrt(z)      "reciprocal square root"，平方根倒数
```

PyTorch 里是一个独立算子，GPU 上有对应指令，比写成 `1 / torch.sqrt(z)` 更快也更稳。

#### (8) 还差最后一步：乘 `weight`

参考实现里最后还有：

```python
return x.to(dtype) * self.weight        # self.weight: [d]
```

`self.weight` 是**每个特征维一个可学习缩放**（初始化为全 1）。归一化把所有维度压到同一尺度后，网络需要重新获得"哪些维度更重要"的自由度，就靠这个 weight。

完整公式：

```text
RMSNorm(x) = ( x / sqrt(mean(x²) + ε) ) ⊙ γ        γ 是可学习的 [d] 向量
```

#### (9) 为什么要 `x.float()` 再转回

```python
dtype = x.dtype
x = x.float()                                       # 升到 fp32 再算
...
return x.to(dtype) * self.weight                    # 转回原精度
```

在 bf16/fp16 下，`x.pow(2).mean(-1)` 容易溢出或丢精度（fp16 最大值只有 65504，平方很容易爆）。升到 fp32 算归一化系数是常见做法。

### 9.10 SDPA 是什么

#### (1) 名字与出身

```text
SDPA = Scaled Dot-Product Attention = 缩放点积注意力
出自《Attention Is All You Need》(2017)
```

名字拆开就是它的三步：

```text
Dot-Product ：用点积算 q 和 k 的相似度
Scaled      ：把分数除以 sqrt(d_k)
Attention   ：softmax 归一化后对 value 加权求和
```

"dot-product" 这个词是为了和更早的 **additive attention**（Bahdanau 那种用一个小 MLP 算分数的做法）区分。现在说 attention 基本都指这一种。

#### (2) 公式与四个步骤

```text
                    Q Kᵀ
Attention(Q,K,V) = softmax( ───── ) V
                   ──────  √d_k
```

```text
① S = Q Kᵀ        两两点积 = 相似度分数        [T, d_k] @ [d_k, T] → [T, T]
② S = S / √d_k    缩放（名字里的 Scaled）
③ A = softmax(S)  每行归一化成概率，行和 = 1   [T, T]
④ O = A V         按权重对 value 加权求和      [T, T] @ [T, d_v] → [T, d_v]
```

形状速查（多头时前面再加 head 维）：

```text
Q: [B, H, T, d_k]      K: [B, H, T, d_k]      V: [B, H, T, d_v]
S: [B, H, T, T]        A: [B, H, T, T]        O: [B, H, T, d_v]
```

#### (3) 数值例子（T=3, d_k=d_v=2）

```text
Q = [[1,0], [0,1], [1,1]]      K = 同上      V = [[1,0], [0,1], [2,2]]
d_k = 2,  sqrt(d_k) = 1.4142

① QKᵀ
     [1.0, 0.0, 1.0]
     [0.0, 1.0, 1.0]
     [1.0, 1.0, 2.0]

② 除以 sqrt(d_k)
     [0.7071, 0.0,    0.7071]
     [0.0,    0.7071, 0.7071]
     [0.7071, 0.7071, 1.4142]

③ softmax（每行和为 1）
     行0: [0.4011, 0.1978, 0.4011]
     行1: [0.1978, 0.4011, 0.4011]
     行2: [0.2483, 0.2483, 0.5035]

④ @ V
     o0: [1.2033, 1.0]
     o1: [1.0,    1.2033]
     o2: [1.2552, 1.2552]
```

加 causal mask 后（只能看自己和左边）：

```text
     行0: [1.0,    0.0,    0.0]
     行1: [0.3302, 0.6698, 0.0]
     行2: [0.2483, 0.2483, 0.5035]
     o0: [1.0,    0.0]        ← 第 0 个位置只能看自己
     o1: [0.3302, 0.6698]
     o2: [1.2552, 1.2552]
```

#### (4) 为什么必须除以 √d_k

点积 `q·k = Σ q_i k_i` 是 d_k 个独立项相加，如果 q、k 各维独立同分布（均值 0、方差 1），那么：

```text
点积的均值 = 0
点积的方差 = d_k          →  标准差 = sqrt(d_k)
```

即 **d_k 越大，分数越"散"**。实测（`d_k=128, T=1024`，q、k 各维 ~ N(0,1)）:

```text
点积标准差 = 12.488   （理论 sqrt(128) = 11.31）

不缩放：
    logits 范围  = [-41.88, 39.66]
    最大权重     = 0.757002
    非零权重个数 = 21 / 1024
    熵           = 0.7562   （上限 ln1024 = 6.9315）

除以 sqrt(d_k)：
    logits 范围  = [-3.70, 3.51]
    最大权重     = 0.017563
    非零权重个数 = 1024 / 1024
    熵           = 6.3108
```

**不缩放时 softmax 近乎 one-hot**：1024 个 key 里只有 21 个拿到非零权重，熵只有 0.76（上限 6.93）。这时 softmax 的梯度几乎为 0，训练推不动。除以 `sqrt(d_k)` 把 logits 尺度拉回 ~1，分布才正常。

#### (5) PyTorch 里的 `F.scaled_dot_product_attention` 多了什么

Python 里可以直接写上面四步，但实际都用融合算子：

```python
F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scaling)
```

它比手写多做了三件事：

```text
① 内部自动做 /sqrt(d_k)，也可以显式传 scale 覆盖（我们传的就是 D_qk**-0.5）
② is_causal=True 时内部处理因果 mask，不需要你构造 T×T 的 -inf 矩阵
③ 后端会自动挑实现：
     FlashAttention   —— 不物化 T×T 矩阵，显存 O(T)，最快
     Memory-Efficient —— 分块算
     Math             —— 朴素实现（兜底）
   对长序列，第 ③ 点是关键：T×T 的注意力矩阵根本不落盘
```

所以代码里看不到 mask 张量、也看不到 `1/sqrt(d_k)`——都被算子吸收进去了。

#### (6) 在 KDA 和 Gated MLA 里，哪些是 SDPA、哪些不是

```text
Gated MLA 层 ：是标准 SDPA（causal）。门就加在 SDPA 输出之后、o_proj 之前。
               论文里的 "G1 = SDPA 输出之后"，指的就是这一步。
KDA 层       ：不是 SDPA。它是线性注意力的读出 o_t = S_tᵀ q_t，
               作用在一个固定大小的递归状态上，不构造任何 T×T 矩阵。
```

这也是为什么 KDA 层的复杂度是 O(T) 而 MLA 层是 O(T²)：两者在"怎么算注意力"上是完全不同的两条路线。

#### (7) 一句话

```text
SDPA = softmax(QKᵀ / √d_k) V
     = 先算两两相似度 → 缩放 → 归一化成权重 → 对 value 加权求和
```

#### (8) SDPA 和 self-attention 是一回事吗

**不是同一层概念，但容易被并列混用。**

```text
SDPA        = 公式 / 算子        —— 说的是「怎么算」
self-attn   = Q、K、V 的来源用法 —— 说的是「从哪来」
```

层级关系：

```text
SDPA                 ← 公式：softmax(QKᵀ/√d_k)V     「怎么算」
  └─ Multi-Head       ← 并行 H 个头，再 concat       「算几份」
      └─ self-attn    ← Q、K、V 都来自同一个 X       「Q/K/V 从哪来」
          └─ Block    ← + 残差 + Norm + MLP          「放在哪」
```

同一个 `sdpa()` 函数，换 Q/K/V 的来源就换名字：

| 类型 | Q 来自 | K / V 来自 | mask | 注意力矩阵 A 的形状 |
|---|---|---|---|---|
| Encoder self-attn | X | X | 无（双向） | `T × T` |
| Decoder self-attn | X | X | causal | `T × T` |
| Cross-attn | 解码器 `X_dec` | 编码器 `X_enc` | 视情况 | `T_q × T_kv`（可非方阵） |

数值演示（Q/K/V 投影取单位阵以简化）：

```text
【self-attention】X 一个序列，T=3
  A 形状 = 3×3   ← 方阵
     [0.4011, 0.1978, 0.4011]
     [0.1978, 0.4011, 0.4011]
     [0.2483, 0.2483, 0.5035]

【cross-attention】Q 来自 X_dec(T=2)，K/V 来自 X_enc(T=3)
  A 形状 = 2×3   ← 非方阵，一眼看出是 cross
     [0.4011, 0.1978, 0.4011]
     [0.1978, 0.4011, 0.4011]
```

结论：

```text
① 在 K3/MLA 这个语境里：Q、K、V 全都来自同一个 hidden states x
   → 所以 Gated MLA 里的注意力就是（causal 的 decoder）self-attention
   → 你的说法在这个语境下是对的

② 但说「SDPA 就是 self-attention」不严谨：
   cross-attention 用的是同一个 SDPA 公式，只是 Q 来源不同

③ KDA 不是 SDPA，所以它也不是 self-attention：
   它是线性注意力 / 递归 mixer，读出是 o_t = S_tᵀ q_t
```

最后一个容易误读的点：

```text
self-attention 的 "self" 指的是「Q/K/V 同源」，
不是「每个 token 只看自己」。它照样会去看别的 token。
```

## 十、Gated MLA 详解：MLA → 为什么加门 → 怎么和 KDA 的输出搭配

本节按三段展开：先把 MLA 讲透，再讲为什么要在它上面加门，最后讲 KDA 层和 Gated MLA 层的输出到底怎么合成。

### 10.1 MLA 是什么：把 KV"压扁了再存"

#### (1) 先看标准 attention 的 cache 有多贵

标准多头注意力里，每个 token 都要为**每个 head 各存一份完整的 K 和 V**：

```text
每 token 每层的 cache = 2 × n_heads × head_dim 个数

以 DeepSeek-V3 的规模为例（n_heads = 128, head_dim = 128）：
  2 × 128 × 128 = 32768 个数 / token / 层

上下文 128K、层数 61 时，这个数字会大到无法承受。
```

**KV cache 随序列长度线性增长**，这是长上下文推理最直接的瓶颈。

#### (2) MLA 的核心一招：低秩压缩

MLA 的想法很朴素：**不同 head 的 K/V 之间有大量冗余，何必每个 head 各存一份？**

于是先压成一个共享的小向量，用的时候再展开：

```text
压缩（down-projection）：
    c_KV = W_DKV · h_t          # [d_model] → [d_c]，d_c = 512

展开（up-projection），每 head 各一份：
    k_nope = W_UK · c_KV        # [d_c] → 每 head 128 维
    v      = W_UV · c_KV        # [d_c] → 每 head 128 维
```

关键点：**cache 里只存 c_KV，不存展开后的 K/V。**

```text
标准 MHA：2 × 128 × 128 = 32768 个数 / token / 层
MLA     ：d_c + d_rope = 512 + 64 = 576 个数 / token / 层

缩小约 56.9 倍
```

#### (3) 为什么还要单独留一条 RoPE 支路

这里有个容易被忽略的细节。RoPE 是**位置相关的旋转**：同一个 key，在位置 s=10 和 s=100 旋转的角度不同。

如果直接对 `k_nope` 做 RoPE，那么 `W_UK` 就没法提前合并进 `W_Q`（因为旋转依赖具体位置，每个位置都得单独算）。为了不破坏压缩带来的收益，MLA 把位置信息**单独走一条很窄的支路**：

```text
k_rope = RoPE(W_KR · h_t)      # 只有 64 维，而且所有 head 共享同一份
q_rope = RoPE(W_QR · h_t)      # 同理

最终：
    q = concat(q_nope, q_rope)          # 128 + 64 = 192 维
    k = concat(k_nope, repeat(k_rope))  # 128 + 64 = 192 维
```

注意 `k_rope` 是**所有 head 共用一份**的——这也是省 cache 的一部分。所以 cache 里存的每 token 就是：

```text
c_KV（512 维）+ k_rope（64 维）= 576 维
```

#### (4) 两个"矩阵吸收"，让推理时根本不用还原 K/V

这是 MLA 最漂亮的地方。看 QK 分数：

```text
q_tᵀ · k_s = (W_Q h_t)ᵀ · (W_UK c_KV_s)
           = h_tᵀ · (W_Qᵀ W_UK) · c_KV_s
```

`W_Qᵀ W_UK` 是可以在推理前**预先算好并合并成一个矩阵**的——于是算分数时直接用 cache 里的 `c_KV_s`，不需要先展开成 k。

看输出侧：

```text
Σ_s a_ts · v_s = Σ_s a_ts · (W_UV c_KV_s) = W_UV · (Σ_s a_ts · c_KV_s)
```

可以**先对 latent 做加权求和，最后再升维一次**。于是 `W_UV` 可以吸收进 `W_O`。

结论：**推理时 MLA 全程在 512 维的 latent 空间里做注意力，从不显式还原 128 维 × 128 head 的 K/V。** 这就是它又快又省的来源。

#### (5) q 的低秩投影（顺带一提）

q 还有个可选的 LoRA 结构：`Linear(d, 1536) → RMSNorm → Linear(1536, n_heads × 192)`。这一项**和 cache 无关**（q 不进 cache），主要是省参数和激活。

#### (6) 三个关键尺寸的作用

```text
kv_lora_rank = 512    → cache 的主项，越小越省但表达力越弱
qk_rope_head_dim = 64 → 专门承载位置信息，不能被吸收
qk_nope_head_dim = 128→ 每个 head 的"内容"维度
```

#### (7) 一个实现层面的诚实提醒

FLA 的 `fla/layers/mla.py` 里，**参数化是 MLA**（确实经过 latent 压缩），但它的 cache 仍然存储展开后的 k、v，源码里留了 TODO：

```python
# TODO: instead of caching the full k, v, we can actually only cache the compressed_kv
# and k_rot and recover the full k, v from compressed_kv and k_rot
```

也就是说，**MLA 省 cache 的收益来自"只缓存 latent"，而 FLA 这个实现暂时还没做到位**；Kimi Linear / K3 报的"KV cache 降约 75%"应该来自正确缓存 latent 的实现。

### 10.2 为什么这里要 Gated：给每个头一个"消音开关"

#### (1) 问题的根源：softmax 的权重必须分完，没有"弃权"

softmax 有个硬性约束：一行权重**非负且加起来恰好等于 1**。实测：

```text
所有 token 分数相同   logits=[0,0,0,0]  → 权重=[0.25, 0.25, 0.25, 0.25]  合计=1
第 1 个 token 稍高    logits=[2,0,0,0]  → 权重=[0.711,0.096,0.096,0.096] 合计=1
第 1 个 token 高很多  logits=[6,0,0,0]  → 权重=[0.993,0.002,0.002,0.002] 合计=1
```

不管分数多平多小，**权重都必须分完**。如果某个头在当前上下文里"没什么想看的"，它也没有"这轮我弃权"这个选项，只能硬把 1 分出去。而第一个 token 永远存在，于是它成了默认垃圾桶——这就是 **attention sink**。

实测数据（Qwen《Gated Attention》，NeurIPS 2025）：

```text
baseline：平均 46.7% 的注意力分数给第一个 token
加门后：  降到 4.8%
某一层（layer 21）：83% → 4%
长上下文外推 RULER：提升 10 分以上
```

#### (2) 门的形式：一行公式

```text
Y' = Y ⊙ σ(X W_θ)
```

```text
Y     ：要被调制的对象。Gated MLA 里就是 attention 的输出（每个 head 各自的 [T, head_dim]）
X     ：这一层的输入（pre-norm 之后的 hidden states）
W_θ   ：可学习投影，把 d_model 映射到门的形状
σ     ：sigmoid，输出 (0,1)
⊙     ：逐元素相乘
```

白话：**每个 head 根据当前 token 的内容，自己算一个"这轮要不要发言、说多大声"的开关，再逐维乘到自己的输出上。**

注意 X 是当前输入，所以门是 **query-dependent** 的——同一个头，碰到不同 token 会给出不同的门值。这正是论文说的 "query-dependent sparse gating scores"。

#### (3) 数值例子

```text
头输出 Y       = [2.0, -1.0, 0.5]
门 logit       = [1.0, -6.0, 2.0]
sigmoid(logit) = [0.7311, 0.0025, 0.8808]
Y' = Y ⊙ σ     = [1.4621, -0.0025, 0.4404]
```

第 2 维门值 0.0025 ≈ 关掉；另外两维照常通过。因为每个 head 有独立的门，**某个 head 可以整体把自己静音** —— 这就把 softmax 缺的"弃权"能力补上了。

#### (4) 论文对比过的 30 种变体

| 维度 | 选项 | 结论 |
|---|---|---|
| 加在哪 | G1 SDPA 输出后 / G2 V 后 / G3 K 后 / G4 Q 后 / G5 输出层后 | **G1 最好**（PPL 降 0.2+，MMLU 涨约 2 分）；G2 次之 |
| 粒度 | headwise（每头一个标量）/ elementwise（逐维向量） | 差别不大 |
| 是否分头 | head-specific（每头独立）/ head-shared（所有头共享） | **必须分头**，共享收益明显变小 |
| 施加方式 | 乘法 / 加法 | **乘法更好** |
| 激活 | sigmoid / SiLU | **sigmoid 更好**（加法只能用 SiLU） |

为什么"分头"重要：多头本来就是为了让不同头分工（有的管局部、有的管长程、有的专门当垃圾桶）。**共享一个门就没法差异化**，等于把这个分工能力废掉一半。

#### (5) 为什么有效：三个原因

```text
① 加非线性
   W_V 和 W_O 是两个连续的线性层，数学上等价于一个低秩线性映射。
   中间插一个门，就给这个低秩映射引入了非线性，表达力变强。

② 引入稀疏
   门值由输入决定且分布稀疏，相当于给注意力输出加了
   一层"按内容决定"的稀疏调制。

③ 消除 attention sink
   有了弃权能力，第一个 token 不再被迫当垃圾桶，
   长上下文外推明显变好。
```

成本：headwise + 分头，在 15A2B 的 MoE 上只增加 **1.6M 参数**（模型是 15B 量级），延迟 **< 2%**。

#### (6) 用一句话复述：对，但要补三处精确化

常见的一句话说成：

> "Gated 其实是用输入做了一层 sigmoid，在 attention 输出层做门控。"

**方向完全正确。** 下面补三处精确化。

**精确化 ①：不是 `sigmoid(x)`，而是 `sigmoid(W_θ · x)`**

```text
直接对 x 逐元素 sigmoid：
    sigmoid([1.0, -2.0, 0.5]) = [0.7311, 0.1192, 0.6225]
    问题 1：维度没变（还是 d），但门需要 H 维（每个 head 一个）
    问题 2：没有任何可学习参数，模型学不到"输入的哪些特征该控制门"

正确形式 sigmoid(W_θ · x)：
    先线性投影到门的形状，W_θ 负责挑出"哪些输入特征决定门的开合"
```

数值对比（`d=3, H=2`，`W_θ` 为 2×3）：

```text
token A: x=[1.0,-2.0,0.5]  → W·x=[1.5,-2.3]  → gate=[0.8176, 0.0911]
token B: x=[-1.0,0.5,2.0]  → W·x=[0.95,1.45] → gate=[0.7211, 0.8100]
```

**精确化 ②：门是 per-head 的，不是一个标量**

```text
headwise    : gate 形状 [B, T, H]        每个 head 一个标量
elementwise : gate 形状 [B, T, H, D_v]   逐维向量
```

看上面 token A 的门值 `[0.8176, 0.0911]`——同一个 token，两个 head 的门值差 9 倍。这正是"分头"的意义：不同 head 承担不同角色，共享一个门就没法差异化。论文实测确认 head-specific 明显优于 head-shared。

**精确化 ③：位置再具体一步 —— SDPA 之后、`o_proj` 之前**

```text
  x ──┬─→ W_Q,W_K,W_V ─→ SDPA ─→ o ──┐
      │                               ⊙ ←─ sigmoid(W_gate · x)
      └─→ W_gate ─→ sigmoid ──────────┘
                                      ↓
                                   o_proj ──→ 输出
```

为什么必须是这里，而不是更靠后的 `o_proj` 之后：

```text
W_V 和 o_proj 是两个连续的线性层，数学上等价于一个低秩线性映射。
门插在它们"之间"，才能打断这个低秩瓶颈、引入非线性；
插到 o_proj 之后就没有这个效果了。
```

论文实测的位置排序：

```text
G1  SDPA 输出之后   ★ 最好（PPL 降 0.2+，MMLU +2）
G2  V 投影之后       次之
G3  K 投影之后       收益小
G4  Q 投影之后       收益小
G5  输出层之后       收益小
```

**另外三个容易混的点**

```text
① X 用的是 pre-norm 之后的 hidden states（论文原文明确说明）
② 门是"乘法"（Y' = Y ⊙ σ），不是加法；实测乘法更好
③ 这个门要和模型里其他门区分开：
     - softmax 权重         ：注意力自己分配的权重，不是门
     - KDA 的衰减门 α       ：管记忆遗忘
     - KDA 的输出门 g_proj  ：GLA 风格，作用在输出上
     - MoE 的 router        ：管专家选择
     - ★ Gated MLA 的门      ：管"这个 head 这轮发不发言"
```

### 10.3 KDA 层和 Gated MLA 层的输出怎么搭配

这是三个层次的问题，从内到外分别是：**层内怎么组织 → 哪层用哪个 mixer → 深度方向怎么聚合**。

#### 层次一：每个 Block 的内部结构

两种 Block 的结构是**完全一样**的，只有中间的 token mixer 不同：

```text
标准 Block：
    x ──┬───────────────────────────┐
        │                           │
        └→ RMSNorm → [token mixer] ─┴→ ⊕ → RMSNorm → MLP → ⊕
                     ↑
              这里换成 KDA 或 Gated MLA

FLA 代码里的对应（fla/models/kda/modeling_kda.py 的 KDABlock）：
    attn_spec = get_hybrid_attention_spec(config.attn, layer_idx)
    if attn_spec is not None:
        self.attn = Attention(...)              # 全注意力层
    else:
        self.attn = KimiDeltaAttention(...)     # KDA 层
    self.mlp = KDAMLP(...)                      # 两种 Block 共用同一个 MLP
```

所以 KDA 层和 Gated MLA 层**不是拼接、也不是在同一层里相加**，而是两个结构相同的 Block 交替出现，各自把自己的输出交回同一条"主干道"（残差流）。

#### 层次二：哪些层用哪个 mixer，由一个"层清单"决定

`fla/models/hybrid.py` 里的机制：

```python
HybridAttentionSpec = {
    'layers':      [3, 7, 11, ...],   # 哪些层用全注意力
    'num_heads':   16,
    'num_kv_heads': 4,
    'qkv_bias':    False,
    'window_size': None,
    'rope_theta':  10000.0,
}
```

```python
def get_hybrid_attention_spec(attn, *, layer_idx):
    """Return the normalized attention specification assigned to layer_idx."""
    ...
    return None      # 未分配的层保留模型原生 mixer（在这里就是 KDA）
```

也就是说：**配置里只写"哪些层是全注意力"，剩下的层默认全是 KDA。** 3:1 就对应"每 4 层里有 1 层写进清单"。

K3 报告的原话：

> Kimi Delta Attention (KDA) provides efficient long-sequence mixing, **with periodically interleaved Gated MLA layers preserving global interaction**.

即：

```text
[ KDA , KDA , KDA , Gated MLA ] × N
```

K3 架构图标的就是 `3×` 和 `1×`。

#### 层次三：深度方向怎么聚合 —— 标准残差 vs Attention Residuals

这一层才是真正决定"KDA 的输出和 Gated MLA 的输出如何合成"的地方。

**标准残差（PreNorm 的常规做法）**：

```text
h_l = h_{l-1} + Sublayer(RMSNorm(h_{l-1}))
```

每一层的输出都以**固定权重 1**加到主干上。AttnRes 论文指出这带来一个问题：

> 残差连接……把所有层的输出以**固定的单位权重**累加。这种均匀聚合导致隐藏状态随深度**不受控地增长**，逐渐**稀释每一层的贡献**。

用大白话说：层数一多，主干上的数值越来越大，新加进来的那一层占的比例越来越小，等于"说话越来越没人听"。

**Attention Residuals（AttnRes）的做法**：把"固定权重相加"换成"**在前序层输出上做 softmax 注意力**"。

```text
标准残差： h = Σ_l 1 · output_l            固定权重，全都要
AttnRes ： h = Σ_l w_l · output_l          权重由 softmax 给出，且依赖输入
```

于是每一层可以**有选择地**聚合前面的表示，而不是无脑全收。

> 📌 本节只是"够用版"的速览。AttnRes 的完整推导（为什么标准残差会不受控增长、凸组合上界、
> 手算演练、Block 版的显存账、融合核实现细节）见 **第十三节**。

**Block AttnRes（K3 用的实用版）**：如果对"所有前序层输出"都做注意力，显存和通信开销太大。所以把层分成块，**只在块级表示上做注意力**：

```text
Full AttnRes ：对每个前序层的输出做注意力
Block AttnRes：先按块把若干层输出求和，再对"块摘要"做注意力 → 开销小，收益保留大部分
```

**实现细节（来自 `modeling_kda.py`，可以逐行对照）**：

```python
# 每个子层两套参数：一个 1 维的 pseudo-query + 一个 RMSNorm
self.attn_res_proj = nn.Linear(d_model, 1, bias=False)   # 注意输出是 1 维！
self.attn_res_norm = nn.RMSNorm(d_model)
self.mlp_res_proj  = nn.Linear(d_model, 1, bias=False)
self.mlp_res_norm  = nn.RMSNorm(d_model)

# 子层的全局编号：attention 用 2*layer_idx，MLP 用 2*layer_idx+1
self.attnres_is_attn_boundary = (2 * layer_idx)     % block_size == 0
self.attnres_is_mlp_boundary  = (2 * layer_idx + 1) % block_size == 0

# 零初始化：初始时所有前序层的分数相同 → softmax 均匀 → 退化成平均，训练更稳
nn.init.zeros_(module.weight)   # 论文 §5
```

聚合的动作：

```python
hidden_states = fused_attnres(
    query=self.attn_res_proj.weight,      # 1 维伪 query，用来给每个前序表示打分
    residuals=residuals,                   # 所有可选的"前序层输出"（块摘要 + 当前块前缀和）
    rms_weight=self.attn_res_norm.weight,
    output_rms_weight=self.attn_norm.weight,
    rms_eps=self.attn_res_norm.eps,
)
```

块的划分规则（`KDAConfig` 里的校验）：

```text
attnres_block_size = None  → 不用 AttnRes，走标准残差
attnres_block_size = 1     → Full AttnRes（每个子层一个块）
attnres_block_size = 偶数  → Block AttnRes，一个块里有 block_size // 2 个 transformer 层
```

整条数据流（Block 内部，开了 AttnRes 时）：

```text
进来的 hidden_states 就是"当前块的前缀和 prefix_sum"
    ↓
residuals = [ 已完成块的摘要..., prefix_sum ]
    ↓
fused_attnres(attn_res_proj, residuals)  → 得到这一子层的输入
    ↓
KDA 或 Gated MLA
    ↓
prefix_sum += 本子层输出        # 累积进当前块
    ↓
到块边界就把 prefix_sum 收进 attnres_states，开新块
    ↓
MLP 前再来一次同样的 attnres 聚合 → MLP
    ↓
返回 prefix_sum + mlp_out（交给下一层当输入）
```

#### 三层合起来：输出到底怎么"搭配"

```text
第 1 层：KDA 层和 Gated MLA 层是结构相同的两种 Block，交替出现，
        各自把输出交回同一条 hidden_size 宽的主干。

第 2 层：哪一层用哪种 mixer，由 `attn` 层清单决定；K3 是 3:1。

第 3 层：主干上如何累积这些输出：
        - 标准残差：固定权重 1 逐个相加（简单，但深层会稀释每层贡献）
        - AttnRes ：用 softmax 注意力在前序层输出上加权聚合，
                    权重由 1 维 pseudo-query 打分、依赖输入内容
        K3 用的是 Block AttnRes（块级聚合，控制开销）。
```

一句话：**它们不是"拼接"也不是"逐元素相加"，而是共处同一条残差主干，由 AttnRes 在深度方向上按内容加权聚合。**

AttnRes 的效果（论文，Kimi Linear 48B/3B，1.4T token）：缓解 PreNorm 稀释，使输出幅度和梯度分布沿深度更均匀，所有评测任务上都有提升。

### 10.4 哪些是确定的，哪些是推断的

| 内容 | 状态 |
|---|---|
| K3 有 Gated MLA 层，与 KDA 周期性交替，负责全局交互 | ✅ K3 报告原文 |
| K3 §2.1 = Hybrid Attention，含 2.1.1 KDA、2.1.2 Gated MLA | ✅ 报告目录 |
| 两类层共用统一 cache 布局，KV cache 降约 75% | ✅ 报告 §5.4.1 + Kimi Linear |
| MLA 的压缩、解耦 RoPE、矩阵吸收、cache 数学 | ✅ FLA `fla/layers/mla.py` + DeepSeek-V2 论文 |
| FLA 的 MLA 实现 cache 了展开后的 k/v（未做到只缓存 latent） | ✅ 源码 TODO |
| 门的形式 `Y'=Y⊙σ(XW_θ)`、最佳位置、head-specific、乘法、sigmoid 及收益 | ✅ Qwen《Gated Attention》arXiv:2505.06708 |
| 层清单机制（`attn.layers`，未分配的层保留 KDA） | ✅ FLA `fla/models/hybrid.py` |
| AttnRes 的动机、softmax 聚合、Block 版本、zero init、块划分规则 | ✅ AttnRes 论文 arXiv:2603.15031 + `modeling_kda.py` |
| K3 §2.1.2 里 Gated MLA 的**具体公式**（门放在哪一步、headwise 还是 elementwise） | ⚠️ 未读到正文 |
| K3 里 3:1 的**确切比例数字** | ⚠️ 从架构图 `3×`/`1×` 推断 |
| FLA 的 KDA 模型里全注意力层用的是 `Attention`（标准 GQA），不是 MLA | ✅ 源码；所以 FLA 是 K3 的**近似复现**，不是等价实现 |

未能确认的原因：arXiv HTML 在摘要后即被截断，GitHub 上的 `k3_tech_report.pdf` 在本环境无法下载（bash 网络受限），AttnRes 论文的 ar5iv 转换也失败（只拿到摘要）。因此 §2.1.2 的正文公式与 K3 的精确比例只能标注为待核对。

### 10.5 自测问题

```text
1. 标准 MHA 每 token 每层的 KV cache 是多少个数？MLA 是多少？差多少倍？
2. c_KV、k_nope、k_rope 各自的维度和作用是什么？
3. 为什么 k_rope 必须单独走一条支路，不能一起压缩？
4. "矩阵吸收"消掉了哪两步计算？为什么它成立？
5. softmax 的权重约束为什么会导致 attention sink？
6. 门控公式里 X 是当前输入，这一点为什么重要（query-dependent）？
7. 为什么门必须 head-specific，共享会损失什么？
8. KDA 层和 Gated MLA 层的输出是拼接、相加，还是别的？请描述完整路径。
9. 标准残差的问题是什么？AttnRes 用什么方式解决？
10. Block AttnRes 的 block_size 表示什么？一个块里有几层？
```

## 十一、Gated MLA 参考源码（逐行 shape 标注）

### 11.1 先说清楚这份代码的性质

**这不是 K3 的官方源码。** 截至写作时的公开情况：

```text
Kimi-K3 仓库        ：只有 README + 技术报告 PDF，没有模型代码
flash-linear-attention：有 MLA（fla/layers/mla.py），但没有门控版本
FLA 的 KDA 模型      ：全注意力层用的是标准 GQA（fla.layers.attn.Attention），不是 MLA
```

所以本节的代码是**参考实现**，拼装来源：

```text
MLA 骨架  ← 照搬 flash-linear-attention 的 fla/layers/mla.py 结构
门控部分  ← 照搬 Qwen《Gated Attention》(arXiv:2505.06708) 的形式 Y' = Y ⊙ σ(X W_θ)
            并采用其结论最优的组合：SDPA 输出之后 + head-specific + 乘法 + sigmoid
```

代码存放在 `gated_mla_reference.py`，可直接阅读。与 K3 §2.1.2 的真实实现可能有细节差异。

尺寸采用 DeepSeek-V3 的 MLA 配置（MLA 的原始出处），因为 K3 的确切尺寸未公开：

```text
hidden_size       d        = 7168
num_heads         H        = 128
q_lora_rank                = 1536
qk_nope_head_dim  D_nope   = 128
qk_rope_head_dim  D_rope   = 64
qk_head_dim       D_qk     = 192
kv_lora_rank      D_c      = 512
v_head_dim        D_v      = 128
```

### 11.2 完整源码

> 说明：本节代码与 `gated_mla_reference.py` 保持一致，**已包含真实 KV cache**
> （`MLACache` + `forward_prefill` + `forward_decode`）。
> cache 的逐行讲解与数值验证见 11.7 节。

```python
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
```

### 11.3 shape 追踪结果（已运行验证）

用 `B=2, T=1024` 走一遍，28 项断言全部通过：

```text
配置: d=7168 H=128 D_nope=128 D_rope=64 D_qk=192 D_c=512 D_v=128

1) q 支路
   q_a_proj(x)                        (2, 1024, 1536)
   q_a_norm                           (2, 1024, 1536)
   q_b_proj -> H*D_qk                 (2, 1024, 24576)
   view(B,T,H,D_qk)                   (2, 1024, 128, 192)
   split -> q_nope                    (2, 1024, 128, 128)
   split -> q_rope                    (2, 1024, 128, 64)

2) KV 压缩支路
   kv_a_proj -> D_c  (进 cache)        (2, 1024, 512)
   kv_a_norm                          (2, 1024, 512)
   kv_b_proj -> H*(D_nope+D_v)        (2, 1024, 32768)
   view(B,T,H,D_nope+D_v)             (2, 1024, 128, 256)
   split -> k_nope                    (2, 1024, 128, 128)
   split -> v                         (2, 1024, 128, 128)

3) 解耦 RoPE 支路
   k_rope_proj -> D_rope (进 cache)    (2, 1024, 64)
   apply_rope(q_rope)                 (2, 1024, 128, 64)
   apply_rope(k_rope)                 (2, 1024, 64)
   k_rope.unsqueeze(2)                (2, 1024, 1, 64)
   expand -> 所有 head 共享             (2, 1024, 128, 64)

4) 拼回完整 q、k
   q = cat([q_nope, q_rope])          (2, 1024, 128, 192)
   k = cat([k_nope, k_rope])          (2, 1024, 128, 192)

5) causal SDPA（转置成 head 优先）
   q.transpose(1,2)                   (2, 128, 1024, 192)
   k.transpose(1,2)                   (2, 128, 1024, 192)
   v.transpose(1,2)                   (2, 128, 1024, 128)
   attn 输出 .transpose(1,2)           (2, 1024, 128, 128)

6) ★ 门控
   gate_proj headwise -> H            (2, 1024, 128)
   gate.unsqueeze(-1)                 (2, 1024, 128, 1)
   o * gate 广播后                     (2, 1024, 128, 128)
   gate_proj elementwise -> H*D_v     (2, 1024, 16384)
   elementwise view(B,T,H,D_v)        (2, 1024, 128, 128)

7) 输出投影
   o.reshape(B,T,H*D_v)               (2, 1024, 16384)
   o_proj                             (2, 1024, 7168)
```

### 11.4 三处关键设计的解释

#### (1) 为什么 `kv_b_proj` 把 W_UK 和 W_UV 合并成一个线性层

```text
分开写（概念上）：
    k_nope = W_UK · c_KV      W_UK: [D_c] → [H·D_nope]
    v      = W_UV · c_KV      W_UV: [D_c] → [H·D_v]

合并写（实现上）：
    kv = W_UKV · c_KV         W_UKV: [D_c] → [H·(D_nope + D_v)]
    再 view + split 成 k_nope 和 v
```

两者数学等价，合并只是**一次 matmul 代替两次**，显存和算子都更省。`view(B, T, H, D_nope + D_v)` 再 `split` 就是"按 head 拆开"。

#### (2) 为什么 `k_rope` 要 `unsqueeze(2)` + `expand`

`k_rope` 的形状是 `[B, T, 64]`——**只有一个头**。而 q、k 是 128 个头，要做注意力必须对齐：

```text
k_rope:                [B, T, 64]
unsqueeze(2):          [B, T, 1, 64]
expand(B,T,H,D_rope):  [B, T, 128, 64]      ← 广播，不复制显存
```

`expand` 是**视图**（不实际复制内存），所以 128 个头共享这 64 维，代价接近零。这也是 MLA 省 cache 的一部分：如果每个头各存 64 维 RoPE key，cache 就是 128×64 = 8192 而不是 64。

#### (3) 门控那两行

```python
gate = torch.sigmoid(self.gate_proj(x))     # headwise: [B,T,128]
gate = gate.unsqueeze(-1)                   # [B,T,128,1]
o    = o * gate                             # [B,T,128,128] ⊙ [B,T,128,1] → 沿 head_dim 广播
```

三个要点：

```text
① X 用的是 x（这一层的输入），不是 o —— 所以门是 query-dependent 的
② unsqueeze(-1) 让门按 head_dim 广播：一个头一个标量，对该头所有维度同倍缩放
   如果换成 unsqueeze(-2)，就变成在 head 维广播，语义完全错了
③ 用 * 不用 @ —— 逐元素乘，不是矩阵乘
```

### 11.5 cache 与参数量账

#### KV cache（每 token 每层）

```text
等价 MHA : H×D_qk + H×D_v = 128×192 + 128×128 = 40960
MLA     : D_c + D_rope    = 512 + 64          = 576
倍数     : 40960 / 576 ≈ 71.1 倍
```

注意：这个倍数依赖具体配置；Kimi Linear 说的"KV cache 降低约 75%"是另一个口径——那是**混合架构 KDA+MLA 对比纯 MLA**（4 层里有 3 层压根不产生 per-token KV）。

#### 门控新增参数量（单层）

```text
headwise    : d × H     = 7168 × 128   = 917,504      ≈ 0.92M / 层
elementwise : d × H×D_v = 7168 × 16384 = 117,440,512  ≈ 117M / 层
对比：q_b_proj 一个矩阵就是 1536 × 24576 = 37,748,736 ≈ 37.7M
```

headwise 比 elementwise 省 **128 倍**（正好是 `D_v`），这就是实践中普遍选 headwise 的原因。

补充说明：Qwen 论文里报的"只增加 1.6M 参数"是**整个模型的合计**（≈ `d × H` 每层 × 层数），不是单层数字。

### 11.6 读这份代码时容易踩的坑

1. `qk_head_dim`（192）是 `qk_nope_head_dim + qk_rope_head_dim`，别和 `v_head_dim`（128）混。注意 **q、k 是 192 维、v 是 128 维**，所以 scaled_dot_product_attention 里 q/k 和 v 的最后维不同，靠 SDPA 自身处理。
2. `kv_b_proj` 的输出宽度是 `H*(D_nope + D_v)` = 32768，view 成 `[B,T,H,256]` 再 split，**不是** `[B,T,2,H,128]`。
3. `expand` 是视图，`repeat` 会真复制显存——这里用 `expand`。
4. 门必须在 **head_dim 维**广播（`unsqueeze(-1)`），不是在 head 维。
5. cache 里该存 `c_kv` 和 `k_rope`（576），**不是**展开后的 k、v（40960）——这一点很多实现（包括 FLA 当前版本）都还在 TODO。
6. `is_causal=True`：MLA 是标准因果注意力，causal mask 由 SDPA 内部处理，不需要手工构造 mask 张量。

### 11.7 KV cache 到底怎么存、怎么读（c_kv 与 k_rope）

#### (1) 先承认：上一版代码把 cache 简化掉了

上一版 `forward` 里只写了：

```python
if past_kv is not None:
    k_cached, v_cached = past_kv          # 假装 cache 里是展开后的 k、v
    k = torch.cat([k_cached, k], dim=1)
    v = torch.cat([v_cached, v], dim=1)
```

这只是"概念上的 cache"：没有写入逻辑，也没有体现 MLA 省显存的设计。`gated_mla_reference.py` 现在补上了真实版本（`MLACache` + `forward_prefill` + `forward_decode`）。

#### (2) cache 里只存两样

```python
self.c_kv   = torch.zeros(num_layers, max_batch, max_seq, d_c,   ...)   # [L, B, S, 512]
self.k_rope = torch.zeros(num_layers, max_batch, max_seq, d_rope, ...)  # [L, B, S,  64]
self.seq_len = 0                                                        # 写指针（标量）
```

```text
每 token 每层 = D_c + D_rope = 512 + 64 = 576 个数
对比：若存展开后的 k、v = H×D_qk + H×D_v = 24576 + 16384 = 40960
倍数 ≈ 71.1×
```

61 层、128K 序列、batch=1 时的总量：

```text
只存 c_kv + k_rope : 4.61 G 个数
存展开后的 k、v    : 327.49 G 个数
```

#### (3) 为什么 `k_rope` 必须单独存（三个理由）

```text
① 位置绑定
   RoPE 之后的值和位置 s 一一对应：位置 s 的角度只对 s 成立。
   而 c_kv 是"没做 RoPE 的压缩结果"，从它推不出任意位置的旋转后向量。

② 来源不同
   k_rope = W_KR · h_t，c_kv = W_DKV · h_t —— 是两条独立投影，
   c_kv 里压根不包含 k_rope 的信息。

③ 不能被吸收
   矩阵吸收成立的前提是"这一步与位置无关"。
   而 RoPE 恰恰依赖位置，所以它没法被并进 query，
   只能显式地按 token 存下来。
```

对比之下，`k_nope` 和 `v` **不需要**存——因为它们能从 `c_kv` 复原。

#### (4) 为什么 `c_kv` 够用：两条吸收恒等式

```text
① nope 分数
   q_nopeᵀ · k_nope_s
   = q_nopeᵀ · (W_UK c_kv_s)
   = (W_UKᵀ q_nope) · c_kv_s
     └──────┬──────┘
      把 W_UK 吸收进 query：q_abs = W_UKᵀ q_nope ∈ R^{D_c}
   于是分数只用 q_abs 和 cache 里的 c_kv 就能算出来。

② 输出
   Σ_s a_s · v_s
   = Σ_s a_s · (W_UV c_kv_s)
   = W_UV · ( Σ_s a_s · c_kv_s )
     └──────┬──────┘
      先对 latent 加权求和，最后升维一次
```

**数值验证**（小配置 `d_c=3, D_nope=D_rope=D_v=2, H=2, T=3`，纯 Python 跑）：

```text
路径 A（先展开成 k、v，再算注意力）
  head0: scores=[0.0779, -0.1565, 0.0888]  attn=[0.3569, 0.2823, 0.3608]
  head0: o=[-0.317849, -0.202974]

路径 B（只碰 cache 里的 c_kv 和 k_rope）
  head0: q_abs=[0.0198, -0.1262, 0.0768]      ← W_UK 被吸收进 query
  head0: scores=[0.0779, -0.1565, 0.0888]  attn=[0.3569, 0.2823, 0.3608]   ← 完全相同
  head0: z=Σ a·c_kv=[0.0248, -0.5452, -0.0562]
  head0: o=W_UV @ z=[-0.317849, -0.202974]

两条路径最大误差 = 5.55e-17   ✓ 完全一致
```

这个验证直接证明：**cache 里只要 `c_kv` + `k_rope` 就足以复现完整注意力的结果。**

#### (5) 写入与读取

```python
class MLACache:
    def append(self, layer_idx, c_kv_new, k_rope_new):
        """c_kv_new: [B, T_new, D_c]   k_rope_new: [B, T_new, D_rope]"""
        T_new = c_kv_new.shape[1]
        s, e = self.seq_len, self.seq_len + T_new
        self.c_kv[layer_idx, :, s:e, :] = c_kv_new        # 写 c
        self.k_rope[layer_idx, :, s:e, :] = k_rope_new    # 写 k_rope
        return s, e

    def read(self, layer_idx, end=None):
        end = self.seq_len if end is None else end
        return (self.c_kv[layer_idx, :, :end, :],         # [B, end, 512]
                self.k_rope[layer_idx, :, :end, :])       # [B, end,  64]

    def advance(self, T_new):
        self.seq_len += T_new
```

三个要点：

```text
① 预分配 buffer，不做动态增长 —— 避免每步 realloc
② 每层一份，但 seq_len 是全局共享的写指针（所有层同步推进）
③ 写入的是压缩后的 576 维，不是展开后的 40960 维
```

#### (6) decode 的完整流程（形状已追踪验证）

```text
输入 x: [B, 1, d] = [2, 1, 7168]，cache 里已有 1024 个 token

1) 只算新 token 的 q 和「要进 cache 的两样」
   q_a_proj -> q_a_norm              [2, 1, 1536]
   q_b_proj -> H*D_qk                [2, 1, 24576]
   view(B,T,H,D_qk)                  [2, 1, 128, 192]
   split -> q_nope / q_rope          [2,1,128,128] / [2,1,128,64]
   kv_a_proj -> c_kv   ★进 cache      [2, 1, 512]
   k_rope_proj         ★进 cache      [2, 1, 64]

2) 写入 cache
   cache.c_kv[layer, :, 1024:1025, :]     [2, 1, 512]
   cache.k_rope[layer, :, 1024:1025, :]   [2, 1, 64]
   seq_len: 1024 -> 1025

3) 读取全量历史
   C_kv                              [2, 1025, 512]
   K_rope                            [2, 1025,  64]

4) ★ 矩阵吸收 + 注意力（全程不还原 k_nope / v）
   W_UK = kv_b_proj.weight.view(H, 256, D_c)[:, :D_nope]    [128, 128, 512]
   W_UV = kv_b_proj.weight.view(H, 256, D_c)[:, D_nope:]    [128, 128, 512]
   q_abs = einsum('bthd,hdc->bthc')                          [2, 1, 128, 512]
   s_nope = einsum('bthc,bsc->bhts')                         [2, 128, 1, 1025]
   s_rope = einsum('bthd,bsd->bhts')                         [2, 128, 1, 1025]
   scores / attn                                             [2, 128, 1, 1025]
   z = einsum('bhts,bsc->bthc')                              [2, 1, 128, 512]
   o = einsum('bthc,hvc->bthv')                              [2, 1, 128, 128]

5) 门控 + 输出投影
   gate_proj(x) headwise             [2, 1, 128]
   gate.unsqueeze(-1)                [2, 1, 128, 1]
   o * gate                          [2, 1, 128, 128]
   reshape(B,T,H*D_v)                [2, 1, 16384]
   o_proj                            [2, 1, 7168]
```

#### (7) 为什么 prefill 和 decode 要写两套

```text
prefill：一次吃下整个 prompt（T 可能几千）
   → 一次性展开 k、v 做 causal SDPA 是可以接受的（只做一次，且 flash-attn 支持好）
   → 但写进 cache 的仍然只有 c_kv / k_rope

decode：每步只有 1 个新 token
   → 必须走吸收路径
   → 否则每步都要把整个历史展开一遍：O(S) 的额外计算 + 临时显存，
     而 cache 的省显存收益也就白费了
```

#### (8) 真实工程里的 cache 布局

```text
预分配      : [L, B, max_seq, D_c] 与 [L, B, max_seq, D_rope]，避免动态增长
写指针      : 一个全局 seq_len，所有层同步推进
分页(vLLM)  : block table 把逻辑位置映射到物理块，支持不同请求共享前缀
前缀缓存    : 相同前缀的请求可复用 c_kv / k_rope
K3 的特殊点 : KDA 层的递归状态（固定大小）也要一起管理，
              所以报告里专门有一节 "Unified cache layout for hybrid KDA–MLA attention"
```

#### (9) 常见错误

```text
1. 把展开后的 k、v 存进 cache
   → FLA 当前的 MLA 实现就是这样，源码里有 TODO；71 倍显存白省

2. 忘了存 k_rope → 分数里少掉位置信息，完全算不对

3. 把 k_rope 按 head 存 [H, D_rope]，而不是全 head 共享
   → 显存又多 128 倍

4. 先算 v 再算注意力
   → 应该先对 latent 加权求和（z），最后才乘 W_UV

5. decode 时忘了把新 token 写进 cache 就读取
   → 注意力里少一个位置，输出偏移

6. 每层各自维护 seq_len
   → 应该共享一个写指针，所有层必须同步推进
```
## 十二、LatentMoE（K3 的 Stable LatentMoE）

### 12.1 一句话

```text
把 token 先投影到一个「低维 latent 空间」，让专家在这个小空间里工作；
省下来的成本拿去换「更多专家 + 每 token 激活更多专家」。
结果：通信量和显存带宽不变，但有效表达力和专家组合空间大幅提升。
```

和 MLA 压缩 KV 是同一个思路：**把"贵的那个维度"压小，然后用省下的预算换别的东西。**

### 12.2 为什么标准 MoE 有两个瓶颈

LatentMoE 论文（NVIDIA，arXiv:2601.18089）先从系统层面建模，用 Qwen3-235B-A22B 当例子：
`N=128` 专家、`K=8` 每 token 激活、`d=4096` hidden、`m=1536` 中间维、`EP=64` 卡，GB200。

#### (1) 低延迟场景：受内存带宽限制

```text
屋顶线（roofline）：算力 10 PFLOPs (FP4)，HBM 带宽 8 TB/s
要进入 compute-bound，算术强度需 ≥ F/BW = 10e15/8e12 = 1250 FLOPs/byte
代入专家配置可推出：每个专家的 token 数 t_exp ≥ 1418
但延迟敏感场景下 t_exp 常常只有几百 → 落在 bandwidth-bound 区
```

含义：**小 batch 时，瓶颈是"把专家权重从 HBM 读进来"，不是算力。** 所以要看 **accuracy per parameter**。

#### (2) 吞吐场景：受 all-to-all 通信限制

```text
每个 GPU 每 MoE 层的通信量 M_comm ∝ (N/EP)·t_exp·d = t_total·K·d/EP
通信时间/计算时间 ≈ 5·F / (4·m·BW_NVL) ≈ 9     （GB200 + Qwen3-235B 配置）
```

含义：**吞吐场景下通信时间是计算的 9 倍。** 注意化简后的式子：

```text
M_comm ∝ t_total · K · d / EP
        ↑ 只与 K 和 d 有关，与 N 无关！
```

**这就是"扩专家数量是免费的"的数学依据。**

### 12.3 五个设计原则

| 编号 | 内容 |
|---|---|
| DP I | 低延迟场景受**内存带宽**限制（加载权重）→ 要最大化 accuracy per parameter |
| DP II | 吞吐场景受 **all-to-all 通信**限制，通信量 ∝ `t_total·K·d/EP` → 要减 `d` 或 `K` |
| DP III | 模型质量依赖**有效非线性预算** `U_eff ∝ K·m`（Barron 函数：u 个非线性单元 → MSE `O(1/u)`）→ 不能减 `K` 或 `m` |
| DP IV | 存在任务相关的特征秩 `r_eff`，`d` 不能减到它以下，否则质量崩塌 |
| DP V | 同时放大 `N` 和 `K` 有组合稀疏收益：`C(αN, αK) ≥ C(N,K)^α` |

### 12.4 关键推理链：为什么只能减 d

```text
DP I / II ：要提速 → 必须减「显存带宽」和「通信」
             显存带宽 ∝ d 和 m
             通信     ∝ K 和 d

DP III    ：但 K 和 m 都不能减（会损失有效非线性预算）

⇒ 四个量里只剩 d 可以减

DP IV     ：d 有下界 r_eff，不能无限减

DP V      ：而扩大 N 和 K 是有好处的，且扩 N 免费（通信式子里没有 N）
```

于是得到那个"免费午餐"式的变换：

```text
把 d 除以 α，同时把 N 和 K 都乘以 α
  → 通信成本和显存带宽成本保持不变（或更低）
  → 但专家更多、每 token 激活更多、有效非线性预算更高
```

### 12.5 架构：两个变体

标准 MoE 的专家（每个专家 `2·d·m` 参数）：

```text
y = Σ_{i∈T_{K,N}} p_i · E_i(x)          E_i: R^d → R^d
```

LatentMoE 的专家在 latent 空间工作（每个专家 `2·ℓ·m` 参数，ℓ = d/α）：

```text
ℓ-MoE_eff(x) = W_↑ · ( Σ_{i∈T_{K, N'}}  p'_i · E_i(W_↓·x; ℓ) )   ← K 不变，N'=αN
             + Σ_{j} E_j(x; d)                                    ← 共享专家仍在 d 维

ℓ-MoE_acc(x) = W_↑ · ( Σ_{i∈T_{K',N'}} p'_i · E_i(W_↓·x; ℓ) )   ← K'=αK，N'=αN（推荐）
             + Σ_{j} E_j(x; d)
```

四个细节：

```text
① W_↓ ∈ R^{ℓ×d}      降维投影，把 x 投到 latent
② W_↑ ∈ R^{d×ℓ}      升维投影，把专家输出投回 d 维
③ 路由仍在 d 维做：   p' = Softmax(W'_r · x)，W'_r ∈ R^{N'×d}
   （论文说路由和共享专家不构成主要瓶颈，所以不压缩）
④ 专家权重：          W_FC1, W_gate ∈ R^{m×ℓ}，W_FC2 ∈ R^{ℓ×m}
   → 全部在 latent 空间，参数量比标准专家小 α 倍
```

### 12.6 算术验证（`d=4096, m=1536, N=128, K=8`，取 `α=4 ⇒ ℓ=1024, N'=512, K'=32`）

| 指标 | 标准 MoE | ℓ-MoE_eff | ℓ-MoE_acc |
|---|---|---|---|
| 通信量 ∝ `K·d` | `8·4096 = 32768` | `8·1024 = 8192`（÷4） | `32·1024 = 32768`（持平） |
| 有效非线性预算 `K·m` | `12288` | `12288`（持平） | `49152`（×4） |
| 单专家参数 `2·d·m` | `12.58M` | `3.15M`（÷4） | `3.15M`（÷4） |
| 路由专家总参数 | `1.61B` | — | `1.61B`（**相同**） |
| 每 token 激活参数 | `100.7M` | — | `100.7M`（**相同**） |

**关键结论：`ℓ-MoE_acc` 在总参数和每 token 激活参数完全不变的前提下，把有效非线性预算提高了 4 倍。**

组合稀疏性也验证了：

```text
log C(128, 8)   = 27.99
log C(512, 32)  = 117.08
log [C(128,8)^4] = 111.95        ← C(αN,αK) ≥ C(N,K)^α 成立

专家组合空间实际扩大 4.92e38 倍（不是 4 倍）
```

这就是论文说的 "exponentially expanding the space of expert combinations"。

### 12.7 和 MLA 的类比

```text
MLA      ：把每个 token 的 KV 压成 latent（d_c=512），注意力在 latent 空间里做
LatentMoE：把每个 token 的激活压成 latent（ℓ = d/α），专家在 latent 空间里做

共同套路：找到一个「贵的维度」，先降维 → 在低维空间做主要计算 → 再升维
```

两者的收益也同源：**省下显存/带宽/通信，而不是省算力本身**。

### 12.8 K3 的 Stable LatentMoE（只有章节标题，正文未取到）

K3 报告 §2.3 的标题结构：

```text
2.3 Stable LatentMoE
    2.3.1 Normalized LatentMoE
    2.3.2 Sigmoid Tanh Unit GLU        （附录 B：Smoothly capping both branches /
                                        Local and limiting behavior / Bounded output）
    2.3.3 Quantile Balancing           （附录 C：Linear relaxation and duality /
                                        Exact coordinate minimization /
                                        From assignment to routing /
                                        Relation to sign-based loss-free updates）
                                       （附录 D：Histogram-Based Quantile Estimation）
```

K3 的规模：**896 个路由专家，每 token 激活 16 个**（激活比例 1.79%，比 Qwen3 的 8/128 = 6.2% 更稀疏）。

从标题能读出的意图（**以下是基于命名的推断，不是报告原文**）：

```text
① Normalized LatentMoE
   低维 latent 空间的激活尺度更敏感（维度小 → 方差波动相对更大），
   所以在 latent 空间里补归一化，让专家输入尺度稳定。

② Sigmoid Tanh Unit GLU（SiTU-GLU）
   附录 B 三个小标题都指向「有界」：两条分支都平滑截断、输出有界。
   MoE 训练不稳定的常见来源就是极端激活值，用一个输出有界的激活函数压住它。
   "Sigmoid + Tanh" 说明两个分支分别用 sigmoid 和 tanh 做平滑截断。

③ Quantile Balancing
   附录 C 提到 "linear relaxation and duality"（线性松弛与对偶）、
   "exact coordinate minimization"（精确坐标最小化）、
   "from assignment to routing"（从指派问题到路由）、
   "relation to sign-based loss-free updates"（与基于符号的无损更新的关系）
   → 这是一个「无辅助损失（aux-loss-free）的负载均衡」方法：
     用分位数来决定路由阈值，使每个专家分到的 token 数均衡。
   附录 D 的 "Histogram-Based Quantile Estimation" 说明分位数是用直方图在线估计的
   —— 因为精确分位数在分布式训练里太贵。
   "relation to sign-based loss-free updates" 与 DeepSeek 的 loss-free balancing
   （用 per-expert bias 调整打分）属于同一类思路，Quantile Balancing 是它的改进版。
```

标题里的 "Stable" 也印证了这一点：三个组件分别针对**尺度稳定**（归一化）、**数值有界**（SiTU-GLU）、**负载均衡**（Quantile Balancing）。

### 12.9 哪些是确定的，哪些是推断的

| 内容 | 状态 |
|---|---|
| LatentMoE 的定义、五个设计原则、两个变体、成本表 | ✅ NVIDIA 论文 arXiv:2601.18089 |
| 通信量 `∝ t_total·K·d/EP`（与 N 无关） | ✅ 论文原文 |
| `U_eff ∝ K·m`、Barron 函数 O(1/u) | ✅ 论文原文 |
| `C(αN,αK) ≥ C(N,K)^α` | ✅ 论文原文，并已数值验证 |
| iso-parameter / iso-FLOP 的算术 | ✅ 本文验证 |
| K3 用 896 专家、每 token 激活 16 个 | ✅ K3 摘要 |
| K3 §2.3 有三个子节及其标题 | ✅ 报告目录 |
| SiTU-GLU 是"输出有界的激活函数" | ⚠️ 从附录 B 小标题推断 |
| Quantile Balancing 是"无 aux loss 的分位数负载均衡" | ⚠️ 从附录 C/D 小标题推断 |
| K3 的 `d`、`ℓ`、`m` 具体取值 | ❌ 未公开 |

未取到 §2.3 正文的原因：arXiv HTML 在摘要后即被截断，GitHub 上的 `k3_tech_report.pdf` 在本环境无法下载（bash 网络受限），且 "Sigmoid Tanh Unit" 与 "Quantile Balancing" 在 arXiv 上检索结果为 0 ——说明它们是 K3 报告独有的术语，没有可交叉引用的外部论文。

### 12.10 自测问题

```text
1. 标准 MoE 在低延迟 / 吞吐两种场景下的瓶颈分别是什么？
2. 为什么通信量式子里没有 N？这对架构设计意味着什么？
3. 为什么 K 和 m 都不能减？减了会损失什么？
4. 为什么最终只能减 d？d 的下界由什么决定？
5. ℓ-MoE_eff 和 ℓ-MoE_acc 的区别是什么？各自适合什么目标？
6. 为什么 ℓ-MoE_acc 能在总参数不变的前提下提升有效非线性预算？
7. 路由和共享专家为什么可以留在 d 维、不做压缩？
8. LatentMoE 和 MLA 的共同套路是什么？
9. K3 为了"稳定"加了哪三个组件？各自大概针对什么问题？
```

## 十三、Attention Residuals（AttnRes）：深度方向的注意力

> 这一节回答的问题：**"标准残差 + AttnRes"到底在干什么？为什么预训练需要它？**
> 证据来源：AttnRes 论文 arXiv:2603.15031 **摘要原文**（正文 HTML 转换失败，只拿到摘要）+
> FLA 的参考实现 `fla/ops/attnres/naive.py`、融合核 `fla/ops/attnres/fused.py`、
> 模型组装 `fla/models/kda/modeling_kda.py`（三份源码都读到了）。
> 数字由两个脚本现场算出，可直接复现：
> `python3 attnres_demo.py`（增长/稀释/显存/参数）+ `python3 attnres_inputs_demo.py`（输入是谁/状态机追踪/逐 token 独立性）

### 13.1 先看一行对比

```text
标准残差（PreNorm，现在所有主流 LLM 的做法）：
    h_l = h_{l-1} + Sublayer( RMSNorm(h_{l-1}) )
          ↑ 每一层的输出都以**固定的权重 1** 累加到主干上

AttnRes：
    h_l = Σ_{i<l} p_i · output_i          Σ p_i = 1, p_i ≥ 0
          ↑ 权重 p 由 softmax 给出，**依赖输入内容**
```

一句话：**标准残差是"全部加起来"，AttnRes 是"按内容加权平均"。**

论文摘要里对问题的原话（逐字引用）：

> Residual connections with PreNorm are standard in modern LLMs, yet they accumulate all layer
> outputs with **fixed unit weights**. This uniform aggregation causes **uncontrolled hidden-state
> growth with depth**, progressively **diluting each layer's contribution**.

翻译成大白话：**层数一多，主干上的数值越来越大，新加进来那一层占的比例越来越小 —— 等于"说话越来越没人听"。**

### 13.2 问题一：不受控增长（有数字）

设每一层输出的每个分量是独立同分布、标准差为 σ 的随机数，隐藏维度 D，一共 L 层。

**标准残差**（求和）：

```text
h = o_1 + o_2 + ... + o_L
E||h||² = L · D · σ²      →   ||h|| ≈ σ·sqrt(L·D)        随 L 单调增长，没有上界
```

**AttnRes 在零初始化下**（下面 13.4 会说明为什么先看这种情况，此时权重均匀 = 1/L，等价于取平均）：

```text
h = (1/L) · (o_1 + ... + o_L)
E||h||² = (1/L) · D · σ²  →   ||h|| ≈ σ·sqrt(D/L)        随 L 反而变小，天然有界
```

模拟结果（D = 256、σ = 1、每个 L 重复 400 次取平均）：

| 子层数 L | 标准残差 \|\|h\|\| | 理论 sqrt(L·D) | AttnRes(q=0) \|\|h\|\| | 理论 sqrt(D/L) | 比值 |
|---|---|---|---|---|---|
| 4 | 32.05 | 32.00 | 8.01 | 8.00 | 4.00 |
| 8 | 45.26 | 45.25 | 5.66 | 5.66 | 8.00 |
| 16 | 64.10 | 64.00 | 4.01 | 4.00 | 16.00 |
| 32 | 90.30 | 90.51 | 2.82 | 2.83 | 32.00 |
| 64 | 128.14 | 128.00 | 2.00 | 2.00 | 64.00 |
| 96 | 156.43 | 156.77 | 1.63 | 1.63 | 96.00 |

读数（模拟值和理论值小数点后两位吻合，说明不是巧合）：

- 标准残差从 L=4 的 32 涨到 L=96 的 157，**涨了 4.9 倍**，而且 L 再大还会继续涨，**没有任何上界**。
- AttnRes 反而被压到 8 → 1.6。
- 最后一列**恒等于 L**，而且和 D、σ 都无关 —— 这就是"求和"和"平均"的天然倍数。

**关键性质**：softmax 权重非负、和为 1 ⟹ 输出是各个源的**凸组合** ⟹

```text
||h|| ≤ max_i ||output_i||        ← 有上界，且上界与层数 L 无关
```

这是 AttnRes 最本质的好处：**残差主干的模长被卡在"单个层输出的最大模长"以内，不随深度增长。**

### 13.3 问题二：稀释（有数字）

即便把模长归一化掉，还有第二个问题。第 l 层把 `o_l` 写进主干时，**相对**改变了多少？

```text
h_l = h_{l-1} + o_l
||h_{l-1}|| ≈ σ·sqrt((l-1)·D),   ||o_l|| ≈ σ·sqrt(D)
相对改变 = ||o_l|| / ||h_{l-1}|| ≈ 1 / sqrt(l-1)

方向上的夹角：cos(h_{l-1}, h_l) = sqrt(l-1)/sqrt(l)
```

| l | cos(h_{l-1}, h_l) | 夹角 | 相对改变 1/sqrt(l-1) |
|---|---|---|---|
| 2 | 0.707107 | 45.00° | 1.0000 |
| 4 | 0.866025 | 30.00° | 0.5774 |
| 8 | 0.935414 | 20.70° | 0.3780 |
| 16 | 0.968246 | 14.48° | 0.2582 |
| 32 | 0.984251 | 10.18° | 0.1796 |
| 48 | 0.989529 | **8.30°** | 0.1459 |
| 64 | 0.992157 | 7.18° | 0.1260 |
| 96 | 0.994778 | 5.86° | 0.1026 |

**读数：在第 48 层附近，这一层最多只能把主干方向转动约 8 度。** 层越深，话语权越小。
这就是摘要说的 "progressively diluting each layer's contribution"。

### 13.4 AttnRes 的具体算法（四步）

FLA 的参考实现 `naive_attnres` 把整个算法压成了 5 行：

```python
v = stacked.float()                                              # 所有候选源，[L, ..., D]
k = F.rms_norm(v, (D,), rms_weight.flatten().float(), rms_eps)   # ① 每个源自己做 RMSNorm 当 key
p = einsum(k, query.flatten() * scale, "l ... d, d -> l ...").softmax(dim=0)   # ②③ 打分 + 深度维 softmax
o = einsum(p, v, "l ..., l ... d -> ... d")                      # ④ 加权求和（value 是原始 v）
if output_rms_weight is not None:                                # 可选：把紧随其后的 prenorm 折进来
    o = F.rms_norm(o, (D,), output_rms_weight, rms_eps)
```

四步展开：

| 步骤 | 公式 | 说明 |
|---|---|---|
| ① 造 key | `k_i = RMSNorm(o_i) · w` | **残差源自己就是 key**，没有 `W_k` |
| ② 打分 | `s_i = <k_i, q> · scale` | `q` 是**一个 D 维向量**（不是矩阵），所以每源出一个标量 |
| ③ 归一化 | `p = softmax(s)` | softmax 作用在**深度维**（L 个源）上，不是序列维 |
| ④ 聚合 | `o = Σ_i p_i · v_i`，`v_i` 是**未归一化**的原始残差 | value 也没有 `W_v` |

#### 输入到底是谁（逐行对照 `modeling_kda.py`）

调用点长这样（`KDABlock.forward` 里，attn 和 mlp 各一次）：

```python
hidden_states = fused_attnres(
    query=self.attn_res_proj.weight,        # ① 伪 query
    residuals=residuals,                    # ② 候选来源列表
    rms_weight=self.attn_res_norm.weight,   # ③ key 的归一化权重
    output_rms_weight=self.attn_norm.weight,# ④ 折叠进来的那个 prenorm
    rms_eps=self.attn_res_norm.eps,
)
```

五个入参，逐个说清楚：

| 入参 | 是什么 | 形状 | 来源 |
|---|---|---|---|
| `query` | 伪 query，**一个向量** | `[1, D]`（`nn.Linear(D, 1).weight`） | `attn_res_proj` / `mlp_res_proj` / 顶层 `res_proj`，**零初始化** |
| `residuals` | **候选来源的列表**（Python list，不是拼好的大张量） | `L` 个 `[B, T, D]` | 见下面「谁进了 residuals」 |
| `rms_weight` | key 归一化的可学习缩放 | `[D]`，初始化全 1 | `attn_res_norm` / `mlp_res_norm` / `res_norm` |
| `output_rms_weight` | 把紧随其后的 prenorm 折进同一个 kernel | `[D]` | `attn_norm`（attn 那一次）/ `mlp_norm`（mlp 那一次）/ 顶层 `self.norm` |
| `rms_eps` | RMSNorm 的 eps | 标量 | 同上 norm 的 `.eps` |

两个容易搞错的细节：

```text
· query 传进去的是 [1, D]（Linear 的 weight），但 naive_attnres 的文档字符串写的是 [D] 或 [D, 1]。
  三种写法 flatten 之后是同一串 D 个数，Triton 核也是按线性偏移读的，所以都能跑。
· residuals 故意保持成 list，**不做 torch.cat**。源码注释原话：
    "kept as separate tensors so `fused_attnres` can ingest them via its
     pointer-table API without an upstream `torch.cat`"
  融合核用一张指针表（`_build_ptr_table`）直接寻址每个来源。
```

#### 谁进了 `residuals`——状态机逐行追踪

源码里维护**两个状态**（注释原话）：

```python
# list of completed block summaries (kept as separate tensors ...);
# the running `prefix_sum` rides on `hidden_states` itself.
attnres_states: list[torch.Tensor] | None = None
```

```text
attnres_states : list[Tensor]  已完成块的摘要（保持成 list）
hidden_states  : Tensor        当前块正在累加的 prefix_sum（搭在 hidden_states 上）
```

逐行复刻后的追踪（`attnres_block_size = 4`，即一块含 2 个 transformer 层；
`E` = token embedding，`A_i` = 第 i 层 attention 输出，`M_i` = 第 i 层 MLP 输出）：

| 子层 | residuals | 调用 |
|---|---|---|
| layer0.attn | `[E]` — **绕过，没调 AttnRes** | `attn_norm(E)` |
| layer0.mlp | `[E, A0]` | `fused_attnres(mlp_res_proj, ...)` |
| layer1.attn | `[E, A0+M0]` | `fused_attnres(attn_res_proj, ...)` |
| layer1.mlp | `[E, A0+A1+M0]` | `fused_attnres(mlp_res_proj, ...)` |
| layer2.attn | `[E, A0+A1+M0+M1]` ← **块边界** | `fused_attnres(attn_res_proj, ...)` |
| layer2.mlp | `[E, A0+A1+M0+M1, A2]` | `fused_attnres(mlp_res_proj, ...)` |
| layer3.attn | `[E, A0+A1+M0+M1, A2+M2]` | `fused_attnres(attn_res_proj, ...)` |
| layer3.mlp | `[E, A0+A1+M0+M1, A2+A3+M2]` | `fused_attnres(mlp_res_proj, ...)` |
| （全部层之后）`model.top` | `[E, A0+A1+M0+M1, A2+A3+M2+M3]` | `fused_attnres(res_proj, ...)` → 折叠 `self.norm` |

于是 `residuals` 的通式是：

```text
residuals = [ E , S_1 , S_2 , ... , S_k , 当前块前缀和 ]

  E   = token embedding          ← **注意：索引 0 是 embedding，不是某一层的输出！**
  S_j = 第 j 个块的完整和（块内所有子层输出的逐元素和）
  k   = 已完成块的个数（把 E 自己也算进去的那个列表长度）
```

四个必须记住的事实：

**① `residuals[0]` 是 token embedding，不是层输出。** 第一个块的"种子"就是词嵌入本身。
它作为普通候选源参与 softmax —— 也就是模型可以选择"少看点前面写的，多看点原始输入"。

**② 整个模型的第一个 attn 子层被完全绕过。** 源码：

```python
if attnres_states is None:
    # L=1 single-source: attnres is trivially identity (p=1, mix=v[0]);
    # apply the prenorm directly, matching the L>1 kernel path which folds it
    # via `output_rms_weight`. Mirrors Megatron-LM's bypass at the first layer
    # (where `block_residual` is empty).
    hidden_states = self.attn_norm(prefix_sum)
    attnres_states = [prefix_sum]
    prefix_sum = None
```

只有 1 个来源时 softmax 必然给出 `p=1`，聚合结果就是 `v[0]` 本身 —— 等同于恒等映射。
所以直接走 prenorm，省一次 kernel。源码注释明确说这对应 **Megatron-LM 在首层
`block_residual` 为空时的 bypass**。
但注意：**只有 attn 那一次被绕过**；紧接着的 `layer0.mlp` 已经是 L=2 了。

**③ 块边界只在 attn 子层触发（对偶数 block_size 而言）。** 边界条件是

```text
global_sublayer_index % block_size == 0
attn 子层号 = 2*layer_idx      （恒为偶数）
mlp  子层号 = 2*layer_idx + 1  （恒为奇数）
```

`block_size` 只允许 `None` / `1` / 正偶数。若 `block_size` 是偶数，它的任意倍数都是偶数，
而 mlp 子层号恒为奇数 ⟹ **永远不可能相等** ⟹ `attnres_is_mlp_boundary` 永远是 `False`。
（已用穷举验证：`block_size ∈ {2,4,…,32}`、`layer_idx ∈ 0..63`，全部为 False。）

实际后果见 13.8 节：偶数 `block_size` 下，末尾 `hidden_states = prefix_sum + mlp_out`
那一步**必然**走相加分支。

**④ 权重是逐 token 独立算的，完全没有跨 token 混合。** `naive_attnres` 里

```python
stacked = torch.stack(tuple(r.view(-1, D) for r in residuals), dim=0)   # [L, B*T, D]
p = einsum(k, q, "l ... d, d -> l ...").softmax(dim=0)                  # [L, B*T]，softmax 在 dim=0
```

把 `B*T` 全部摊平到中间维，`softmax(dim=0)` 作用在**来源维**上。
所以每个 `(batch, token)` 位置各自做一次完整的 softmax，互不影响。

数值验证（`L=5` 个来源、`D=8`、6 个 token）：

```text
  token |  src0 |  src1 |  src2 |  src3 |  src4 |   Σp
      0 | 0.092 | 0.040 | 0.512 | 0.039 | 0.318 | 1.0000
      1 | 0.518 | 0.198 | 0.006 | 0.055 | 0.224 | 1.0000
      2 | 0.059 | 0.855 | 0.028 | 0.052 | 0.006 | 1.0000
      3 | 0.022 | 0.049 | 0.082 | 0.645 | 0.201 | 1.0000
      4 | 0.935 | 0.046 | 0.001 | 0.014 | 0.004 | 1.0000
      5 | 0.021 | 0.020 | 0.117 | 0.819 | 0.022 | 1.0000
```

每一行都不同 ⟹ 权重确实依赖输入内容；每一行内部 Σp=1 ⟹ 每个 token 各自归一化。
把 token 顺序打乱后重算，结果**逐个 token 完全一致** ⟹ 没有跨 token 混合，也没有位置编码。

> 也就是说：AttnRes 名字里有 "attention"，但它**完全不看序列维**。
> 它只在"这一层的输入该从哪几个前序表示里取"这件事上做注意力，而且每个 token 各取各的。

#### 它和"真正的 attention"差在哪（这是最容易糊的地方）

| | 标准 self-attention | **AttnRes** |
|---|---|---|
| softmax 在哪个维度 | 序列/时间维（T 个 token） | **深度维**（L 个前序层输出） |
| Q | 学出来的 `W_q` 乘输入 | **一个 D 维向量**（每个子层一套） |
| K | 学出来的 `W_k` 乘输入 | **残差自己做 RMSNorm**，无 `W_k` |
| V | 学出来的 `W_v` 乘输入 | **残差原值**，无 `W_v` |
| 打分 | `q·k / sqrt(d)` | `<RMSNorm(o_i)·w, q>` |
| 位置编码 | RoPE | 无（"前序"本身就定义了顺序） |
| 因果性 | 要显式 mask | **天然因果**，只看前面的层 |
| 值域 | 无约束 | **凸组合**，`Σp = 1`、`p ≥ 0` |

**为什么 key 要用 RMSNorm？** 因为残差主干的模长会随层变化（就是 13.2 那个问题）。如果不归一化，
深层的大模长残差会天然获得大 logit，softmax 就变成"只看最后几层"。归一化后所有源**在同一个尺度上竞争**。

**为什么 value 用原始残差？** 因为归一化的目的只是"公平打分"，真正要传下去的信息量在原始值里。
所以只有打分那一步归一化，聚合那一步用原值。

#### 参数量有多小

```text
每个 KDABlock（一层）：
  attn_res_proj = Linear(d_model → 1, bias=False)  →  d_model = 7168
  attn_res_norm = RMSNorm(d_model)                 →  d_model = 7168
  mlp_res_proj  = Linear(d_model → 1, bias=False)  →  d_model = 7168
  mlp_res_norm  = RMSNorm(d_model)                 →  d_model = 7168
  小计 = 28672

48 层 × 28672                                    = 1,376,256
KDAModel 顶层另有一组 res_proj + res_norm        =    14,336
合计                                              = 1,390,592 ≈ 1.39 M

  占 Kimi Linear 48B 总参数 : 0.00290%
  占 K3 2.8T 总参数         : 0.000050%
```

注意**顶层还有一组**：`KDAModel` 在所有层跑完之后又做了一次 AttnRes 聚合
（`self.res_proj` + `self.res_norm`），并把最终的 `self.norm` 折进去当 `output_rms_weight`：

```python
# KDAModel.forward 结尾
if self.use_attnres:
    residuals = [*attnres_states, hidden_states]
    hidden_states = fused_attnres(
        query=self.res_proj.weight, residuals=residuals,
        rms_weight=self.res_norm.weight,
        output_rms_weight=self.norm.weight,      # ← 折叠最终 norm
        rms_eps=self.res_norm.eps,
    )
else:
    hidden_states = self.norm(hidden_states)
```

所以总共是 **2 个子层组 + 1 个顶层组 = 3 组**参数，每组 2×d_model。

对照：MLA 的 `W_DKV` 一个矩阵就是 `d_model × d_c = 7168 × 512 = 3.67 M`。
**AttnRes 全部加起来约 1.39 M —— 比 MLA 一个投影矩阵还小**，因为它没有 K/V/Q 投影矩阵。

零初始化是怎么落实的（`KDAPreTrainedModel._init_weights`）：

```python
# tag so `_init_weights` keeps the zero init (paper §5)
self.attn_res_proj._is_attnres_proj = True
self.mlp_res_proj._is_attnres_proj = True
...
if isinstance(module, (nn.Linear, nn.Conv1d)):
    if getattr(module, '_is_attnres_proj', False):
        # attnres pseudo-query (per-layer projection): zero init keeps
        # the initial softmax uniform (paper §5)
        nn.init.zeros_(module.weight)
    else:
        nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
```

靠一个 `_is_attnres_proj` 标记在通用初始化函数里"打例外"，避免被默认的正态初始化覆盖。
（顶层的 `self.res_proj._is_attnres_proj = True` 也打了同样的标记。）

### 13.5 手算一个完整的 AttnRes（3 个源，D = 2）

设三个候选残差源、RMSNorm 权重全 1（初始化就是全 1）、学出来的伪 query `q = [1, 1]`：

```text
r1 = [3, 0]      ||r1|| = 3.0000
r2 = [0, 4]      ||r2|| = 4.0000
r3 = [1, 1]      ||r3|| = 1.4142
```

**第 1 步：每个源自己做 RMSNorm 当 key**（`RMSNorm(x) = x / sqrt(mean(x²)+eps) · w`）

```text
r1: mean(x²) = (9+0)/2 = 4.5   →  1/sqrt(4.5) = 0.4714  →  k1 = [1.4142, 0.0000]
r2: mean(x²) = (0+16)/2 = 8.0  →  1/sqrt(8.0) = 0.3536  →  k2 = [0.0000, 1.4142]
r3: mean(x²) = (1+1)/2  = 1.0  →  1/sqrt(1.0) = 1.0000  →  k3 = [1.0000, 1.0000]
```

注意 r2 模长最大（4.0），归一化后它的 RMS 和 r1、r3 一样都是 1 —— **尺度被拉平了**。

**第 2 步：打分 `s_i = <k_i, q>`**

```text
s1 = 1.4142×1 + 0.0000×1 = 1.4142
s2 = 0.0000×1 + 1.4142×1 = 1.4142
s3 = 1.0000×1 + 1.0000×1 = 2.0000
```

**第 3 步：深度维 softmax**

```text
exp: 4.1133, 4.1133, 7.3891     和 = 15.6157
p  = 0.2634, 0.2634, 0.4732     和 = 1.000000 ✓
```

**第 4 步：加权求和（用原始残差）**

```text
o[0] = 0.2634×3.0 + 0.2634×0.0 + 0.4732×1.0 = 1.2634
o[1] = 0.2634×0.0 + 0.2634×4.0 + 0.4732×1.0 = 1.5268
o = [1.2634, 1.5268],   ||o|| = 1.9818
```

**对照 A — 标准残差（固定权重 1 全部相加）**：

```text
h = r1 + r2 + r3 = [4, 5],   ||h|| = 6.4031
||h|| / ||o|| = 3.2310
```

**对照 B — `q` 全 0（即零初始化那一刻）**：

```text
logits = [0, 0, 0]  →  全部相等  →  softmax 均匀
p = [1/3, 1/3, 1/3]
o = [(3+0+1)/3, (0+4+1)/3] = [1.3333, 1.6667]   ← 正好是三个源的算术平均 ✓
```

**关键性质验证**：三个源的模长最大值是 `max(3, 4, 1.4142) = 4.0000`

```text
||o(q=[1,1])|| = 1.9818  ≤ 4.0000   ✓ 凸组合上界成立
||o(q=0)||    = 2.1344  ≤ 4.0000   ✓
||标准残差||   = 6.4031  >  4.0000  ✗ 超出上界 → 这就是"不受控增长"
```

### 13.6 为什么伪 query 零初始化，以及它意味着什么

代码里（`modeling_kda.py`）：

```python
self.attn_res_proj = nn.Linear(d_model, 1, bias=False)
nn.init.zeros_(module.weight)      # 零初始化
```

零初始化 ⟹ `q = 0` ⟹ 所有 logit 都是 0 ⟹ softmax 均匀 ⟹ **模型一开始的行为是"对前序输出取算术平均"**。

这个设计很关键，有三层意思：

1. **起点是"平均"而不是"求和"**：平均的模长不随深度增长（13.2 的表），所以一开局残差主干就是稳的。
   如果初始化成随机的 `q`，训练第一步就会出现"某些层权重 0.9、某些层 0.001"的极端选择，深层尤其容易崩。
2. **均匀权重是"所有源都看一眼"**：梯度可以均匀地流到每一个前序层的输出上，不会一开始就饿死某些层。
3. **然后慢慢学出"该看谁"**：训练过程中 `q` 逐渐偏离 0，softmax 逐渐变尖，模型自己决定
   "这一层该主要参考第 3 层还是第 40 层"。论文摘要里说的 **content-dependent depth-wise selection** 就是这个。

> 摘要原文：`ablations validate the benefit of content-dependent depth-wise selection`
> （消融实验验证了"**由内容决定、沿深度选择**"这一点的收益 —— 也就是说，权重必须依赖输入，不能是固定的静态权重。）

### 13.7 Full AttnRes vs Block AttnRes：账要算清

**Full AttnRes**：第 i 个子层对**前面所有 i-1 个子层的输出**做注意力。
问题很显然：得把前面所有层的输出都留着 —— 显存和通信都是 O(L²)。

**Block AttnRes**：把子层分成块，**只在"块摘要"上做注意力**。

```text
已完成的块 1  →  摘要 s1
已完成的块 2  →  摘要 s2
...
已完成的块 j-1 → 摘要 s(j-1)
当前块前缀和   →  prefix_sum     （当前块里已经算完的子层输出累加）
------------------------------------------------
residuals = [E, s1, s2, ..., s(j-1), prefix_sum]    共 j+1 个来源
```

注意这里**比常见说法多了一个 `E`**（token embedding，见 13.4 节的事实 ①）。
下面所有口径都把 `E` 和当前块前缀和算进去 —— 这是逐行对照源码后的修正，
早先按"只有块摘要"算会少算两个来源。

账（Kimi Linear 48B 规模：48 个 transformer 层 × 2 个子层 = 96 个子层，d_model = 7168，
一个 batch 8192 token，bf16）：

```text
单条 d_model 宽的残差流 = 8192 × 7168 × 2 B = 117,440,512 B = 112.0 MiB

Full AttnRes（block_size = 1）：每个子层看全部历史
  峰值要同时留住 97 条流 = 10.61 GiB            ← 不现实
```

设子层总数 `n = 96`、块大小 `b`（正偶数）：

```text
块边界集合 = { 0, b, 2b, ... } ∩ [0, n)      个数 n_b = floor((n-1)/b) + 1
在子层 i 处调用时的来源数 = floor((i-1)/b) + 2          (i ≥ 1；i = 0 被绕过)
末态 attnres_states 长度  = n_b
峰值同时存活张量数        = n_b + 1 = floor((n-1)/b) + 2
```

| block_size b | 一块含层数 | 末态 states 长度 | 峰值存活张量 | 峰值显存 | 全部调用点来源总数 | 相对 Full |
|---|---|---|---|---|---|---|
| 1（Full） | 1 | 96 | 97 | 10.61 GiB | 4655 | 1.00× |
| 2 | 1 | 48 | 49 | 5.36 GiB | 2399 | 1.94× |
| 4 | 2 | 24 | 25 | 2.73 GiB | 1271 | 3.66× |
| **8** | **4** | **12** | **13** | **1.42 GiB** | **707** | **6.58×** |
| 16 | 8 | 6 | 7 | 0.77 GiB | 425 | 10.95× |
| 32 | 16 | 3 | 4 | 0.44 GiB | 284 | 16.39× |

两本账要分清：

```text
峰值存活张量数 ≈ 块数 + 1     ← 这是**显存**的账（b=8 时 96/8+1 = 13 条流）
全部调用点来源总数 ≈ 按 1/b 缩 ← 这是**读取代价 / 通信量**的账
```

`b` 从 1 涨到 8，峰值从 97 条流降到 13 条流，**省了约 7.5 倍显存**；
来源总数从 4655 降到 707，省 6.58 倍。
（口径说明：本表按源码逐行核对，来源里包含 `E` 和当前块前缀和；
论文正文没读到，所以不要拿这张表和论文的某个数字硬对。）
> 摘要原文：`Block AttnRes, which partitions layers into blocks and attends over block-level
> representations, reducing the memory footprint while preserving most of the gains of full AttnRes.`
> 上表就是 "reducing the memory footprint" 的定量版本，而且能看出**代价是对数级/线性级的取舍**：
> 块越大越省，但每个子层能看到的"历史分辨率"越粗。

#### block_size 的语义（容易搞错，务必注意）

```text
模型代码（modeling_kda.py）：
    self.attnres_is_attn_boundary = (2 * layer_idx)     % block_size == 0
    self.attnres_is_mlp_boundary  = (2 * layer_idx + 1) % block_size == 0

attnres_block_size = None  →  不用 AttnRes，走标准残差
attnres_block_size = 1     →  Full AttnRes（每个子层自己一个块）
attnres_block_size = 偶数  →  Block AttnRes，一个块里有 block_size // 2 个 transformer 层
```

**`block_size` 数的是"子层"（attention 和 MLP 各算一个），不是 transformer 层。**
所以 `block_size = 8` 表示一个块里有 **4 个 transformer 层**（8 个子层）。
还必须是偶数，否则块边界落不到"层"的边界上。

注意 **attn 和 mlp 各自独立判边界**（`2*layer_idx` 和 `2*layer_idx+1`），所以一个子层是不是块边界，
和它的兄弟子层不一定一致。

### 13.8 一个 block 内部的数据流（开了 AttnRes 时）

逐行对照 `KDABlock.forward` 的真实控制流：

```text
进来：hidden_states = 当前块的前缀和 prefix_sum；attnres_states = 已完成块摘要列表
    ↓
【第一次 AttnRes，attn 用】—— 但整个模型的第 0 个 attn 子层被绕过（见 13.4 事实 ②）
    ↓
residuals = [ *attnres_states , prefix_sum ]
    ↓
if 是 attn 块边界:  attnres_states = residuals;  prefix_sum = None
    ↓
hidden_states = fused_attnres(attn_res_proj.weight, residuals,
                              attn_res_norm.weight,
                              output_rms_weight=attn_norm.weight)   ← 折叠 attn_norm
    ↓
KDA 或 Gated MLA
    ↓
prefix_sum = attn_out            if prefix_sum is None  ← 刚开新块
           = prefix_sum + attn_out  否则                ← 继续累加
    ↓
【第二次 AttnRes，mlp 用】
residuals = [ *attnres_states , prefix_sum ]
if 是 mlp 块边界:  attnres_states = residuals;  prefix_sum = None
    ↓
hidden_states = fused_attnres(mlp_res_proj.weight, residuals,
                              mlp_res_norm.weight,
                              output_rms_weight=mlp_norm.weight)    ← 折叠 mlp_norm
    ↓
MLP
    ↓
hidden_states = mlp_out                 if prefix_sum is None
              = prefix_sum + mlp_out    否则       ← 交给下一层
```

三个要点：

**① AttnRes 改的是"怎么读"，不是"怎么写"。** 返回给下一层的仍然是主干累加值
（`prefix_sum + mlp_out`），AttnRes 的输出只是"喂给子层的输入"。这个区分很重要。

**② 但"返回 `prefix_sum + mlp_out`"只对偶数 `block_size` 成立。**
源码那一行是

```python
hidden_states = hidden_states if prefix_sum is None else prefix_sum + hidden_states
```

由 13.4 节事实 ③：偶数 `block_size` 下 `attnres_is_mlp_boundary` 永远是 `False`，
所以 `prefix_sum` 在那一步永远不是 `None` ⟹ **必然走相加分支**，结论成立。
但 **Full AttnRes（`block_size = 1`）时 mlp 也是边界**，`prefix_sum` 被清成 `None`，
此时返回的就是**纯 `mlp_out`**（因为 `attnres_states` 里已经存了每一个子层输出，不需要再单独带一个前缀和）。
两种情形不能混着说。

**③ `attnres_states` 不参与跨 stage 传输。** 源码注释：

```python
# returned `hidden_states` carries the running `prefix_sum` to
# the next layer (single-tensor pp transmission, no separate carry)
outputs = (hidden_states, attentions, past_key_values, attnres_states)
```

也就是说 KDABlock 的返回值是一个**四元组**（比 HF 惯例多一个 `attnres_states`），
但在**流水线并行**下跨 stage 传的仍然只有 `hidden_states` 这一个张量，
块摘要列表留在 stage 内部。这与论文摘要里提到的
`cache-based pipeline communication` 方向上是一致的
（**但正文没读到，不能断定就是同一个机制** —— 见 13.10 待核对）。

### 13.9 融合核（`fused.py`）里几个值得注意的实现细节

从 Triton 核源码直接读出来的：

| 细节 | 代码位置 | 含义 |
|---|---|---|
| **把后面的 prenorm 折进来** | `output_rms_weight` / `HAS_ONORM` | AttnRes 后面本来紧跟一个 `attn_norm`/`mlp_norm`，这个核直接把两步合一个 kernel，省一次启动 + 一次读写 |
| **全 fp32** | `v = stacked.float()`、核内 `.to(tl.float32)` | 打分和 softmax 全程 fp32，只在最后返回时 downcast 一次 |
| **在线 softmax over L** | `b_m / b_acc / b_o` 累加 | 不是先算完所有权重再聚合，而是一边扫来源一边更新最大值和加权和 —— 每个 value tile 只读一次 |
| **q·w 预乘** | `b_qw = q * w` | logit 是 `Σ_d v_d·q_d·w_d · rsqrt(...)`，把 `q*w` 先乘好，跨所有来源 tile 复用 |
| **checkpoint_level** | `0` 存 `o_pre`（混合后的残差），`1` 不存、反向从源重算 | 默认 `1`：省显存，代价是反向多读一次源 |
| **指针表补齐到 2 的幂** | `_build_ptr_table`，`L2 = max(8, next_power_of_2(len))` | 来源个数是编译期常量，所以要 pad 成固定长度；补的槽位只放地址、从不读写 |
| **16 字节对齐断言** | `t.data_ptr() % 16 == 0` | 向量化加载的前提 |
| **有个 `scale` 参数** | 默认 `1.0` | logit 的缩放因子。**注意没有 `1/sqrt(D)`** —— 因为 key 已经 RMSNorm 过，logit 尺度是自控的 |

关于 `scale` 要说清楚：核里**提供**了这个参数，但 K3 训练时具体取什么值，本次没有读到
（正文拿不到），所以**不能断言**它等于 1 还是别的值。这一点列在 13.12 的待核对里。

另外从 `modeling_kda.py` 还能看到一个有意思的对照 —— 传统上对付 PreNorm 稀释的**老办法**
（GPT-2 / Megatron 那套初始化缩放），代码里作为可选项保留着：

```python
def _init_weights(self, module, prenorm_residual_strategy=None, num_residuals_per_layer=2):
    ...
    #   > A modified initialization which accounts for the accumulation on the
    #   > residual path with model depth. Scale the weights of residual layers at
    #   > initialization by a factor of 1/√N where N is the # of residual layers.
    #   >   -- GPT-2
    if prenorm_residual_strategy == "rescale":
        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        with torch.no_grad():
            p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)
    elif prenorm_residual_strategy == "zero":
        nn.init.zeros_(p)
```

这里 `p` 是 `o_proj.weight` 或 `down_proj.weight`，做法是把残差分支的输出投影按
`1/sqrt(2·n_layer)` 缩小。**这是"初始化阶段"的补丁**：让每层一开局写进去的量小一点，
从而削弱累加效应。AttnRes 是**结构层面**的解法 —— 它不缩小任何一层，
而是把"累加"换成"加权平均"，从根上取消了无界增长。

> 注意：默认 `prenorm_residual_strategy=None`，也就是这一段默认**不生效**。
> 这里只是说明 FLA 保留了这条经典路径，不能据此断言 K3 训练用了它。

### 13.10 和 K3 的 "3× KDA : 1× Gated MLA" 怎么咬合

这是最容易混的地方，先分清**两个完全不同的维度**：

```text
宽度/功能维度（"这一层用什么 mixer"）：
    [ KDA, KDA, KDA, Gated MLA ] × N        ← 3:1 交替，见 10.2 节
    解决的是：局部高效建模 vs 全局精确交互 怎么分工

深度维度（"前序层的输出怎么聚合到主干"）：
    标准残差  vs  AttnRes                     ← 本节
    解决的是：PreNorm 把每层输出以固定权重 1 累加，导致主干模长爆炸、深层贡献被稀释
```

两者**正交**：一个决定"每层内部算什么"，一个决定"层与层之间怎么传"。
所以它们不是二选一，而是**同一张架构图上两个独立的旋钮**。K3 两个旋钮都拧了。

```text
第 1 层（结构）：KDA 层和 Gated MLA 层都是结构相同的 Block，各自把输出交回同一条 d_model 宽的主干
第 2 层（分配）：哪层用哪种 mixer，由 attn 层清单决定；K3 是 3:1
第 3 层（聚合）：主干上如何累积这些输出
                 - 标准残差：固定权重 1 逐个相加（深层会稀释）
                 - AttnRes ：softmax 在深度维加权，权重由 1 维伪 query 打分、依赖输入内容
                 K3 用 Block AttnRes（块级聚合，控制显存）
```

一句话：**KDA / Gated MLA 的产出不是"拼接"也不是"逐元素相加"，而是共处同一条残差主干，
由 AttnRes 在深度方向上按内容加权聚合。**

> 再强调一次"block"这个词在本项目里的两种含义，别再混：
> ① **transformer block / 层**（KDA vs Gated MLA 的 3:1 交替）
> ② **token chunk**（chunkwise 并行训练，C = 32 / 64）
> 本节还多了第三种：③ **AttnRes 的 block**（`attnres_block_size`，按子层分块）
> 三者互不相干，看上下文判断。

### 13.11 效果（论文摘要原文）

> `We further integrate AttnRes into the Kimi Linear architecture (48B total / 3B activated
> parameters) and pre-train on 1.4T tokens, where AttnRes mitigates PreNorm dilution, yielding
> **more uniform output magnitudes and gradient distribution across depth**, and improves
> downstream performance across all evaluated tasks.`

对应到本节前面算的东西：

```text
"more uniform output magnitudes"  → 就是 13.2 的表：标准残差 ||h|| 从 32 涨到 157，
                                      AttnRes 把它压到 8 → 1.6，沿深度均匀
"gradient distribution"           → 权重 p_i 就是 ∂h/∂o_i 的一部分，p_i 大则该层拿到的梯度大；
                                      均匀起步（13.6）让梯度先均匀铺开
"improves ... all evaluated tasks" → 摘要原话，不是我们推的
```

另外摘要提到两个工程手段（**正文没读到，机制不详**）：

```text
cache-based pipeline communication      ← 基于缓存的流水线通信
two-phase computation strategy          ← 两阶段计算策略
```

这两个是用来把 Block AttnRes 做成 "practical drop-in replacement with minimal overhead"
的工程手段，但具体怎么实现，本次没有拿到正文，**只能标注为未核实**。

### 13.12 哪些是确定的，哪些是推断的

| 内容 | 状态 |
|---|---|
| PreNorm 标准残差"固定单位权重累加 → 不受控增长 → 稀释每层贡献"这个问题陈述 | ✅ 论文摘要逐字 |
| AttnRes = 在前序层输出上做 softmax 注意力、权重依赖输入内容 | ✅ 论文摘要逐字 |
| Block AttnRes 分块、只在块级表示上做注意力、省显存且保留大部分收益 | ✅ 论文摘要逐字 |
| "content-dependent depth-wise selection" 的收益经消融验证 | ✅ 论文摘要逐字 |
| Kimi Linear 48B/3B、1.4T token、输出幅度与梯度沿深度更均匀、全任务提升 | ✅ 论文摘要逐字 |
| AttnRes 的**精确算法**（RMSNorm 当 key、1 维伪 query 打分、深度维 softmax、value 用原始残差） | ✅ FLA `naive.py` 源码 |
| 零初始化 → 均匀权重 → 退化成算术平均 | ✅ `modeling_kda.py` `nn.init.zeros_` + 手算验证 |
| 凸组合上界 `\|\|h\|\| ≤ max\|\|o_i\|\|`；`\|\|h\|\|` 随 L 的增长规律；第 48 层约 8.3° | ✅ `attnres_demo.py` 现场计算 |
| **`residuals` 的构成**：`[E, S_1..S_k, 当前块前缀和]`，索引 0 是 token embedding | ✅ `modeling_kda.py` `KDABlock.forward` 逐行 + `attnres_inputs_demo.py` 符号追踪 |
| **五个入参的名字与形状**（query `[1,D]`、residuals 是 list、rms_weight `[D]`、output_rms_weight `[D]`） | ✅ `modeling_kda.py` 调用点 + `fused_attnres` 签名 |
| **整个模型的第 0 个 attn 子层被绕过**（L=1 时 AttnRes 恒等，对应 Megatron-LM 的 bypass） | ✅ 源码注释 + 分支代码 |
| **偶数 `block_size` 下 mlp 永远不是块边界**，因此末尾必然走 `prefix_sum + mlp_out` | ✅ 从边界条件推导 + 穷举验证 |
| **权重逐 token 独立**，无跨 token 混合、无位置编码 | ✅ `naive.py` 的 `softmax(dim=0)` + `attnres_inputs_demo.py` 打乱顺序验证 |
| **`KDAModel` 顶层还有一组 AttnRes**（`res_proj` + `res_norm`，折叠 `self.norm`） | ✅ `KDAModel.forward` 源码 |
| 每子层 2×d_model；含顶层共 2×48 + 1 组 = 1,390,592 ≈ 1.39 M | ✅ `modeling_kda.py` 定义 + 计算 |
| `_is_attnres_proj` 标记机制（让通用初始化函数对伪 query 用零初始化） | ✅ `KDAPreTrainedModel._init_weights` |
| `attnres_states` 进四元组返回，但跨 stage 只传 `hidden_states` | ✅ 源码注释 |
| `prenorm_residual_strategy`（GPT-2/Megatron 的 `1/√(2N)` 缩放）作为可选经典做法保留 | ✅ `_init_weights` 源码；默认 `None` 即不生效 |
| `block_size` 数子层、一个块 = block_size/2 个 transformer 层、必须偶数 | ✅ `KDAConfig` 校验 |
| 融合核的 fp32、在线 softmax、prenorm 折叠、checkpoint_level、指针表补齐 | ✅ `fused.py` 源码 |
| **K3 实际用的 `block_size` 取值** | ⚠️ 未读到正文 |
| **融合核 `scale` 在 K3 里的取值** | ⚠️ 未读到正文（接口默认 1.0） |
| **`cache-based pipeline communication` / `two-phase computation` 的具体机制** | ⚠️ 摘要提到，正文未读到；源码注释里"跨 stage 只传单张量"方向上吻合，但**不能断定是同一机制** |
| **`residuals` 里 `E` 是否经过任何额外变换**（代码看是原始 embedding） | ✅ 源码看是原始 `inputs_embeds`；但论文正文没读到，不排除论文有别的表述 |
| **AttnRes 对 loss/下游指标的具体提升幅度** | ⚠️ 摘要只说 "improves ... all evaluated tasks"，没有数字 |
| **FLA 是不是 K3 的等价实现** | ⚠️ FLA 的 `fla/layers/attn.py` 是全注意力层用的 `Attention`（标准 GQA），不是 Gated MLA；所以 FLA 这套是**近似复现**，AttnRes 部分则应有较高保真度 |
| AttnRes 与 GRPO / RL 阶段的交互 | ⚠️ 完全不涉及，论文只讲预训练 |

**未核实的原因**：arXiv:2603.15031 的 HTML 版转换失败（LaTeXML 报 Fatal error，页面只剩模板），
ar5iv 同样失败，PDF 在本环境无法读取（bash 网络受限）。因此**本节所有事实性内容来自
①论文摘要逐字原文 ②FLA 的两份源码**（`fla/ops/attnres/naive.py`、`fla/ops/attnres/fused.py`）
**与 `fla/models/kda/modeling_kda.py`**，其余一律标注为推断或未核实，绝不编造。

### 13.13 自测问题

```text
1. 标准残差 h_l = h_{l-1} + Sublayer(RMSNorm(h_{l-1})) 里，每一层输出的权重是多少？这造成什么后果？
2. 为什么 softmax 权重非负且和为 1 就能保证 ||h|| 有上界？上界是什么？
3. D = 256、σ = 1 时，L = 96 的标准残差 ||h|| 大约是多少？AttnRes（均匀权重）是多少？给出推导。
4. 为什么 AttnRes 的 key 要做 RMSNorm，而 value 不做？
5. AttnRes 有哪些参数？为什么没有 W_k / W_v / W_q？每个子层多少参数？
6. 伪 query 为什么零初始化？零初始化时 AttnRes 退化成什么运算？
7. Full AttnRes 和 Block AttnRes 的区别是什么？Block 版本的来源数怎么算？
8. block_size = 8 表示一个块里有几个 transformer 层？为什么必须是偶数？
9. AttnRes 改的是主干的"读"还是"写"？返回值是 AttnRes 的输出还是 prefix_sum + mlp_out？
10. "3× KDA : 1× Gated MLA" 和 AttnRes 分别解决什么问题？为什么说它们正交？
11. AttnRes 在深度维做 softmax，和 self-attention 在序列维做 softmax 有何异同？列三点。
12. 手算：r1=[2,0]、r2=[0,3]、r3=[2,2]，w=[1,1]，q=[1,1]。求 k、p 和 o。
    （提示：mean(r²) 分别是 2、4.5、4。做完和 13.5 节的结果比一比 —— 想一想为什么
      "把三个源各自乘一个常数"完全不影响最终的 p，这是 RMSNorm 的什么性质？）
13. `fused_attnres` 的五个入参分别是什么？各自的形状？为什么 `output_rms_weight` 要传进去？
14. `residuals` 列表的**索引 0** 是什么？为什么它在里面？
15. 整个模型的第 0 个 attn 子层为什么可以不调 AttnRes？这个 bypass 在别的实现里有对应物吗？
16. `block_size = 8` 时，块边界会落在 mlp 子层上吗？给出理由（从奇偶性讲）。
17. AttnRes 的权重是逐 token 算的还是全序列共用一个？怎么用一句话验证？
18. `KDAModel` 顶层为什么还要再来一次 AttnRes？它把哪个 norm 折进去了？
19. 拿 `block_size = 4`、3 层的小模型，写出 layer2.attn 那一行看到的 `residuals`。
20. 传统上对付 PreNorm 稀释的初始化办法是什么（GPT-2/Megatron 那套）？它和 AttnRes 的思路差别在哪？
```

## 十四、参考资料

- [Kimi Linear: An Expressive, Efficient Attention Architecture（arXiv:2510.26692）](https://arxiv.org/abs/2510.26692)
- [Kimi K3: Open Frontier Intelligence（arXiv:2607.24653）](https://arxiv.org/abs/2607.24653)
- [DeepSeek-V2: MLA 原始论文（arXiv:2405.04434）](https://arxiv.org/abs/2405.04434)
- [Gated Delta Networks（arXiv:2412.06464）](https://arxiv.org/abs/2412.06464)
- [Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free（arXiv:2505.06708，Qwen Team, NeurIPS 2025）](https://arxiv.org/abs/2505.06708)
- [Attention Residuals（arXiv:2603.15031，Kimi Team）](https://arxiv.org/abs/2603.15031)
- [LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts（arXiv:2601.18089，NVIDIA）](https://arxiv.org/abs/2601.18089)
- [KDA kernel 实现（flash-linear-attention）](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda)
- [AttnRes 参考实现 `naive.py`（13.4 / 13.5 节的算法出处）](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/attnres/naive.py)
- [AttnRes 融合核 `fused.py`（13.9 节的出处）](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/attnres/fused.py)
- [FLA 的 MLA 实现（fla/layers/mla.py）](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/mla.py)
- [FLA 的 KDA 模型组装（fla/models/kda/modeling_kda.py）](https://github.com/fla-org/flash-linear-attention/blob/main/fla/models/kda/modeling_kda.py)
- [FLA 的混合注意力配置（fla/models/hybrid.py）](https://github.com/fla-org/flash-linear-attention/blob/main/fla/models/hybrid.py)
- [Gated Attention 代码](https://github.com/qiuzh20/gated_attention)

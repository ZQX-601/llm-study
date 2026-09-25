# 当前学习进度（截至 Day18，含答疑归档）

## 结论

已完成 Transformer/预训练、Decoder-only SFT、LoRA/QLoRA、DPO、高风险数据评测治理，以及 PPO 的代码级闭环。GRPO 已完成原理、核心手算、异常组处理、完整 loss、端到端单机教学源码、梯度累积、测试设计，以及 DDP 下的 group 完整性、全局控制流和变长 batching 分析。

当前已能从 prompt batch 追踪到 rollout、old/ref/new log-prob、reward、group-wise advantage、有效组 mask、优化 epoch、micro-batch、optimizer batch、分布式梯度同步和训练后推理。下一步停止重复 loss/mask 算术，进入 rollout/trainer 分离、policy 权重同步与 staleness，再学习混合精度、LoRA 分布式边界并完成 GRPO 阶段验收。

rollout/trainer 分离与 policy staleness 的答疑内容已单独归档到 `qa/2026-09-21-rollout-trainer-staleness.md`（不占用 `archives/dayNN.md` 序列），结论页当前停在 Policy Staleness 五问。

## 已掌握

- Transformer、BERT/GPT、causal attention、自回归 next-token 训练与 KV Cache 基础。
- Decoder-only logits/target shift，以及 causal/padding/loss mask 的职责。
- Assistant-only SFT、LoRA/QLoRA、显存诊断与实验设计。
- DPO 的 chosen/rejected、policy/reference、beta、数据污染与训练诊断。
- 高风险任务的数据合同、分层指标、硬门槛、版本化与回滚。
- PPO 的 rollout、old/new logp、GAE、returns、critic、clip、mask、KL 与 optimizer step。
- GRPO 的同题组采样、组内 advantage、sequence-to-token 广播、ratio/clip、reference KL 和完整 loss。
- Reward Model、verifier、经典离线 DPO、冻结 RM + GRPO 与迭代式 DPO 的关系。
- 全同 reward、极小方差、RM 噪声放大、低方差过滤及训练指标诊断。
- `group_mask`、`valid_sequence_mask`、`response_mask`、`effective_mask` 的不同职责。
- Prompt batch、rollout batch、optimization epoch、mini-batch、optimizer step 和 iteration 的关系。
- Old/new/ref policy 对同一固定 completion 做 teacher-forcing 概率比较的原因。
- `torch.randperm` 在 advantage 计算后打乱 completion、构造 mini-batch 的作用。

## GRPO 完整主链

```text
train_grpo
└── 每个 prompt_loader batch 启动一次 iteration
    └── train_one_iteration
        ├── collect_rollout
        │   ├── expand_prompts：B 个 prompt → B*G 个输入 + group_ids
        │   ├── policy.generate：本 iteration 唯一一次自由 rollout 阶段
        │   ├── build_response_mask
        │   ├── get_response_logp(policy)：固定 old_logp
        │   ├── get_response_logp(reference)：固定 ref_logp
        │   ├── reward_fn：每条 completion 的 sequence reward
        │   └── groupwise_advantages：advantage + valid_sequence_mask
        │
        └── optimization epoch × K
            └── randperm 后切 mini-batch
                └── grpo_update
                    ├── get_response_logp(policy)：重算 new_logp
                    ├── grpo_loss：ratio/clip/KL/effective_mask
                    ├── backward / grad clip
                    └── optimizer.step
```

下一次 iteration 才用更新后的 policy 重新自由采样 completion。

## Old、New 与 Reference Log-prob

Rollout 阶段：

```text
policy.generate(prompt)
→ 固定 completion token、长度和 response_mask
→ 保存 old_logp
→ 冻结 reference 计算 ref_logp
```

优化阶段不会重新自由生成回答，而是：

```text
current policy(prompt + fixed old answer)
→ teacher forcing
→ new_logp
```

三者评价相同状态和相同动作，因此 `old_logp/new_logp/ref_logp/response_mask` 都是 `[N,T]`。只有 current policy 参数变化，`new_logp` 每次 forward 重算；新的 token、长度和 mask 到下一 iteration 才产生。

## Group Advantage 与有效组

```text
A_i = (r_i - group_mean) / (group_std + eps)
```

不同 prompt 不能混算。当前采用总体标准差 `correction=0`，并使用 `std_threshold` 过滤低方差组：

```text
group_std < threshold
→ advantages 置零
→ valid_sequence_mask=False
→ 当前策略选择整组 policy objective 和 KL 都跳过
```

`valid_sequence_mask` 必须来自真实 reward，不能默认全 `True`。

## 完整 Loss

```text
ratio = exp(new_logp - old_logp)

policy_objective = min(
    ratio * A,
    clamp(ratio, 1-eps, 1+eps) * A
)

ref_log_ratio = ref_logp - new_logp
per_token_kl = exp(ref_log_ratio) - ref_log_ratio - 1

per_token_loss = -policy_objective + beta * per_token_kl

effective_mask = response_mask * valid_sequence_mask[:,None]
loss = sum(per_token_loss * effective_mask) / sum(effective_mask)
```

全无效 mini-batch 应跳过 `backward()`、`optimizer.step()`、scheduler 和有效更新计数。

## 已完成手算

### Group advantage

```text
rewards = [0,1,1,2]
mean = 1
population std ≈ 0.7071
advantages ≈ [-1.414,0,0,1.414]
```

### 完整 masked loss

```text
policy_objective = [
  [-1.221,-1.000,-0.819],
  [ 1.200, 1.000, 0.819],
]
per_token_kl = [[0.10,0.20,9.99],[0.30,0.10,0.20]]
beta = 0.1
effective_mask = [[0,0,0],[1,1,1]]

per_token_loss = [
  [ 1.231, 1.020, 1.818],
  [-1.170,-0.990,-0.799],
]
loss ≈ -0.986
```

### 监控

```text
有效 ratio = [1.25,0.90,0.80]
clipfrac = 1/3 ≈ 0.333
有效 KL = [0.30,0.10,0.20]
mean_kl = 0.20
```

## 当前源码产物

`grpo_end_to_end.py` 包含统一命名的完整教学实现：

```text
GRPOConfig
RolloutBatch
expand_prompts
build_response_mask
get_response_logp
groupwise_advantages
collect_rollout
reference_kl
grpo_loss
grpo_update
train_one_iteration
train_grpo
inference
exact_answer_reward
main
```

已通过：

```text
python3 -m py_compile grpo_end_to_end.py
```

尚未执行真实模型训练。运行需要 PyTorch、Transformers、模型权重和相应计算资源；示例 `exact_answer_reward` 只是提取 completion 中最后一个整数并与标准答案比较的教学 verifier。

## `torch.randperm`

```python
permutation = torch.randperm(sequence_count, device=policy_model.device)
```

生成 `0...sequence_count-1` 的无重复随机排列。每个 epoch 在 group advantage 已算好后打乱 completion，再切成 mini-batch。所有 rollout 字段必须使用同一组 `indices`，保持 tokens、old/ref logp、advantage 和 mask 对齐。

## 需要持续复习的易错点

1. `get_response_logp` 是 teacher-forcing 评分，不是自由生成。
2. `gather` 后的 token log-prob 不再有词表维 `V`。
3. 同一 iteration 的多个 epoch 内，old/ref logp、reward、advantage 和 mask 固定。
4. New logp、ratio、policy objective、KL 和 loss 每次 current policy forward 重算。
5. `advantages[:,None]` 和 `valid_sequence_mask[:,None]` 的真实 shape 是 `[N,1]`，再广播到 `[N,T]`。
6. `ratio.clamp` 与 `torch.minimum` 职责不同。
7. `valid_token_count` 必须是 `effective_mask.sum()`。
8. Advantage 置零不等于无梯度；是否保留 KL 由有效组 mask 策略决定。
9. `std_threshold` 控制信号可信度，`eps` 处理数值稳定性。
10. `randperm` 只能在完整 group advantage 计算之后使用。
11. Loss 可以为负数，关键是梯度方向和 mask 正确。
12. 全无效 mini-batch 不应推动 optimizer 或 scheduler。

## Day17 新增掌握

### 梯度累积层级

```text
rollout batch：一次 iteration 的全部 B×G completion
micro-batch：一次实际 forward/backward 的数据子集
optimizer batch：累计若干有效 micro-batch 后一次 step 的逻辑数据
```

无效 micro-batch 不执行 backward，也不增加 `accumulation_count`。完整累积窗口或 epoch 尾部提交时，顺序必须是：

```text
梯度归一化
→ gradient clipping
→ optimizer.step
→ zero_grad
```

### 严格 Token 平均

Micro-batch mean 等权会放大有效 token 较少的 micro-batch。严谨设计让 `grpo_loss` 返回：

```text
loss_sum
valid_token_count
mean_loss（仅监控）
```

每个有效 micro-batch：

```python
loss_sum.backward()
total_valid_token_count += valid_token_count
```

提交窗口：

```python
parameter.grad.div_(total_valid_token_count)
clip_grad_norm_(...)
optimizer.step()
```

尾部也除以实际 `total_valid_token_count`，不再需要按 micro-batch 数量做修正。

### 核心测试设计

已从逻辑上完成三项测试：

```text
不同 group 不混算：[-1,1,-1,1]
PAD/无效组不进入 loss 分子和分母
全同 reward 组：advantage 全零且 valid mask 全 False
```

断言 API：

```text
浮点 Tensor → torch.allclose
布尔/整数 Tensor → torch.equal
Python 浮点数 → pytest.approx
```

全无效 epoch 应满足：无 backward、无 step、参数不变，并正确累计 `skipped_micro_batches` 与 `skipped_epochs`。不能仅断言 `loss==0`。

### 今日验证边界

用户选择只学习和分析，不在本地运行真实代码。因此 Day17 未安装依赖、未运行模型/测试，也未修改 `grpo_end_to_end.py`；实现结论保存在 `archives/day17.md`。

## Day18 新增掌握

### DDP 与全局控制流

```text
普通 DDP：每个 rank 保存完整模型，处理不同 local batch
backward：本地求导 + 梯度 all-reduce
optimizer.step：所有 rank 使用相同梯度执行
```

所有 rank 必须对 backward、step 和 collective 保持一致。应先 all-reduce 本地有效 token 数：

```text
global_valid_token_count == 0
→ 所有 rank 一起跳过

global_valid_token_count > 0
→ 所有 rank 一起 backward
→ 本地无效 rank 使用连接 new_logp 的零 loss
```

### 分布式 Group 完整性

GRPO 的统计边界是全局 `group_id`，不是 `rank_id`。一个 group 可以完整放在单一 rank，也可以跨 rank 分布后通过带身份的全局聚合计算 advantage；不能让各 rank 对残缺 group 单独标准化。

跨 rank 记录至少要能表达：

```text
group_id：和谁一组
global_sequence_id：这条 completion 是谁
origin_rank/origin_position：从哪里来、发回哪里
```

通信职责：

```text
all-reduce valid token count → 全局 skip 与归一化分母
all-gather group/reward/id   → 重建完整 group advantage
DDP gradient all-reduce      → 同步模型梯度
```

### Global Token Mean

典型 DDP 对 rank 梯度求平均时，为获得严格全局 token 平均，可使用：

```python
scaled_local_loss = (
    local_loss_sum
    * world_size
    / global_valid_token_count
)
```

实际工程必须核对框架、FSDP 和通信 hook 的 reduction 语义。

### 变长 Batching

样本数相同不代表 token、FLOPs、activation 或 step latency 相同。已掌握：

```text
group-aware token bin packing
length bucketing
dynamic token batching
padding 浪费
prompt+response 总长度预算
ΣL² 只能近似 attention 成本
```

`response_mask` 保证 PAD 不进入 loss，但不能消除 prompt/PAD 的全部 forward 计算。

### Token 与 Sequence 聚合

```text
token mean：每个 token 等权，长 completion 总权重更大
sequence equal mean：每条 completion 总权重相同
```

Sequence 等权不能消除 Reward Model 长度偏好、截断、EOS 和采样分布中的全部长度偏差。后续停止重复同类 loss/mask 算术，仅在新机制依赖时回看。

### 今日验证边界

Day18 只学习和分析，未运行分布式训练，也未修改教学源码。完整内容保存在 `archives/day18.md`。

## Day19 新增掌握：Rollout/Trainer 分离与 Policy Staleness

本次系统讲解已归档到 `archives/day19.md`，包含：

```text
Behavior Policy / Current Policy / Reference Policy
old_logp / new_logp / ref_logp 的角色
Rollout Workers → Reward/Verifier → Trainer Workers
同步式与异步式权重同步
policy_version 与 rollout staleness
ratio、clipfrac、KL 和过期数据准入
```

核心规则：

```text
old_logp 必须保持生成该 completion 的 behavior policy 版本
new_logp 用 current policy 对同一固定 completion 重算
ref_logp 来自冻结 reference policy
```

生产 rollout 记录至少应带：

```text
policy_version / completion tokens / old_logp / reward / group_id / sequence_id
```

### 本次验证边界

本次只学习和分析，未运行真实训练或推理引擎，未安装 vLLM / TensorRT-LLM / SGLang，未修改教学源码。

## 下一次直接从这里继续

场景：

```text
rollout worker 使用 policy v10 生成 completion
old_logp 也由 v10 计算
trainer 当前 policy 是 v13
new_logp 由 v13 计算
```

回答：

1. `old_logp` 应保持 v10，还是使用 v13 对同一 completion 重算并覆盖？
2. 如果错误覆盖为 v13 的评分，ratio 会变成什么？
3. 为什么此时 ratio 失去 behavior-policy 对照意义？
4. 这批 rollout 的版本 staleness 是多少？
5. 同步式和异步式 rollout-training 的主要取舍是什么？

## 答疑归档：Kimi K3 的 KDA 与 MLA

同日另一条答疑线，独立于 GRPO 主线，整理在 `qa/2026-09-21-kimi-k3-kda-mla.md`：

```text
Kimi K3：2.8T MoE / 104B 激活 / 1M 上下文，Hybrid Attention = KDA + Gated MLA
KDA 原理：线性注意力（固定大小 fast-weight 状态 S）
        → delta rule（先擦旧关联，再写入残差，等价于在线最小均方）
        → 门控（遗忘率）
        → KDA 把门控从 head 级标量细化到 channel 级对角（DPLR 特化变体）
        → chunkwise 并行（WY 表示 + UT transform）
K3 加强：lower-bounded decay、full-rank gate（仅确认标题，公式待核对）
KDA–MLA 配合：层间混合而非层内融合；Kimi Linear 为 3:1 交替，
        MLA 层承担全局精确检索并采用 NoPE，位置感由 KDA 衰减提供；
        KV cache 最多降 75%，1M 解码吞吐最多 6×
```

关键边界：KDA ≠ GDN（差异在门控粒度）；MLA 仍是 full attention，只是 KV 低秩压缩；full attention 层数受有限状态容量约束不能砍到零。该主题与阶段 8「分布式训练与推理优化」强相关，尚未并入阶段验收。

补充：已完成从 `x:[B,T,d]` 到输出的完整前向拆解（投影 → 因果短卷积 → L2 归一化 → β/门控 → KDA 核心 → 输出门 + RMSNormGated → o_proj），并写成可运行参考实现 `kda_reference.py`。该文件用逐 token 递推与分块两种算法互相验证，覆盖 T∈{1,2,3,5,8,13}、K∈{1,2,4}、V∈{1,3}、chunk∈{1,2,3,4,64}，最大误差 < 1e-9；自检过程中定位到一个真实易错点：分块算法的 `L[r,s]` 中 β 必须取行下标（`Diag(β)KKᵀ`），写成列下标会导致结果错误。

另已补齐：GVA（为什么 q/k 投影成 `H·K`、v 投影成 `HV·V`），以及「逐组件拆解」小节的前两个组件——β（写入强度 / delta rule 步长、`1−β` 特征值与负特征值、β 与 α 的分工、分块算法中 β 的位置）与 α（`Diag(α)` 是数学形式、实际只存 K 维对角线向量、形状 `[B,T,HV,K]` 两个维度的来源、对角矩阵乘积等价于向量逐元素乘积）。后续按 k/v → q → 短卷积 → A/W/U → 输出门的顺序继续。

再补：**Gated MLA**（答疑归档第十节）与**逐行 shape 标注的参考源码**（第十一节，配套文件 `gated_mla_reference.py`）。核心是 softmax 权重必须分完、没有"弃权"选项，导致 attention sink；门的形式为 `Y' = Y ⊙ σ(XW_θ)`，最佳位置是 SDPA 输出之后，head-specific + 乘法 + sigmoid 最优。与 KDA 的配合是层间 3:1 交替，输出由残差主干 + AttnRes（softmax 内容加权）在深度方向聚合。参考源码用 DeepSeek-V3 尺寸（d=7168, H=128, D_c=512, D_rope=64），28 项 shape 断言全部通过；KV cache 对比为 40960 → 576（约 71 倍），headwise 门每层约 0.92M 参数（比 elementwise 省 128 倍）。需注意：K3 官方未公开 Gated MLA 源码，该实现是 FLA 的 MLA 骨架 + Qwen 门控公式的拼装，并已明确标注。KDA 并行化（7.9 节）已按大白话重写，原技术版已整体替换。

再补：**LatentMoE**（第十二节，K3 的 Stable LatentMoE，NVIDIA arXiv:2601.18089）。核心是把专家算在低维潜空间里（`W_↓ : d → ℓ`，专家在 ℓ 维，`W_↑` 映回），只有 `d` 可压；五个设计原则中 DP-II 的通信量 `M_comm ∝ t_total·K·d/EP` 与专家数 N 无关，所以压 `d` 能同时省通信和显存；`ℓ-MoE_acc`（N 扩 α 倍、K 也扩 α 倍）在通信不变的前提下把有效非线性预算提到 α 倍。K3 用 896 专家激活 16 个（1.79%）。K3 的 Stable LatentMoE 三个子节（Normalized LatentMoE / STU-GLU / Quantile Balancing）**只拿到标题**，机制为推断，已在归档中标注。

再补：**Attention Residuals（第十三节，AttnRes，Kimi Team arXiv:2603.15031）**，配套文件 `attnres_demo.py`。核心是 PreNorm 标准残差把每层输出以固定权重 1 累加，导致主干模长随深度不受控增长（D=256、σ=1 时 L=4 的 `||h||≈32` 涨到 L=96 的 `≈157`），并稀释深层贡献（第 48 层每层只能把主干方向转动约 8.3°）。AttnRes 把"求和"换成**深度维的 softmax 加权平均**：key 是残差自己做 RMSNorm（无 `W_k`）、伪 query 只有一个 D 维向量（无 `W_q`）、value 是原始残差（无 `W_v`），因此每子层只要 2×d_model = 14336 个参数，96 个子层合计 1.38M。零初始化 ⟹ 权重均匀 ⟹ 起步退化成算术平均（天然有界）。softmax 非负且和为 1 ⟹ 输出是凸组合 ⟹ `||h|| ≤ max_i ||o_i||`，上界与层数无关。实用版是 **Block AttnRes**：第 j 个块里的子层只看 j 个来源（j−1 个块摘要 + 当前块前缀和），总来源槽 `b·m(m+1)/2`，比 Full 省约 `block_size` 倍；`block_size` 数的是**子层**，一个块 = `block_size//2` 个 transformer 层，必须偶数。注意 AttnRes 改的是主干的"**读**"不是"写"，返回值仍是 `prefix_sum + mlp_out`。与 KDA/Gated MLA 的 3:1 交替**正交**：一个决定"每层内部算什么"，一个决定"层与层之间怎么传"。**未核实项**：K3 实际的 `block_size` 取值、融合核 `scale` 取值、摘要提到的 `cache-based pipeline communication` 与 `two-phase computation` 机制（论文 HTML/PDF 在本环境都拿不到，只有摘要 + FLA 的 `naive.py`/`fused.py` 源码）。

**AttnRes 输入侧追补（同一节，配套 `attnres_inputs_demo.py`）**：拿到 `modeling_kda.py` 全文后逐行核对了"输入到底是谁"，修正/新增四点。① 五个入参：`query`（`Linear(D,1).weight`，形状 `[1,D]`，零初始化）、`residuals`（Python list，每个 `[B,T,D]`，故意不 `cat`，融合核用指针表寻址）、`rms_weight`（`[D]`）、`output_rms_weight`（把紧随其后的 `attn_norm`/`mlp_norm`/`self.norm` 折进同一 kernel）、`rms_eps`。② `residuals` 通式 = `[E, S_1..S_k, 当前块前缀和]`，**索引 0 是 token embedding 而不是层输出**；`attnres_states` 存已完成块的完整和，`prefix_sum` 搭在 `hidden_states` 上。③ **整个模型的第 0 个 attn 子层完全绕过 AttnRes**（L=1 时 softmax 恒为 p=1，等价恒等；源码注释说对应 Megatron-LM 首层 `block_residual` 为空的 bypass），但紧接着的 `layer0.mlp` 已是 L=2。④ **偶数 `block_size` 下块边界只可能落在 attn 子层**（`2*layer_idx` 偶 vs `2*layer_idx+1` 奇，偶数的倍数不可能等于奇数，已穷举验证），因此 13.8 的 `prefix_sum + mlp_out` 必然走相加分支 —— 但 `block_size=1`（Full）时 mlp 也是边界，返回的是纯 `mlp_out`。⑤ 权重**逐 token 独立**：`naive_attnres` 把 `B*T` 摊平后用 `softmax(dim=0)`，打乱 token 顺序结果逐 token 不变，无跨 token 混合、无位置编码。⑥ `KDAModel` 顶层还有第三组 AttnRes（`res_proj`+`res_norm`，折叠 `self.norm`），所以参数总量是 **1,390,592 ≈ 1.39M**（不是 1.38M）。⑦ **修正了旧口径错误**：13.7 的来源数表原先按 `b·m(m+1)/2` 算，漏掉了 `E` 和当前块前缀和；正确口径是**峰值存活张量 = floor((n-1)/b) + 2**（b=8 时 13 条流）、**全部调用点来源总数 = Σ_{i=1}^{n-1}[floor((i-1)/b)+2]**（b=8 时 707 而非 624），省下倍数 6.58× 而非 7.46×。⑧ 另发现 `_init_weights` 里以可选项形式保留了 GPT-2/Megatron 的经典 PreNorm 补丁 `prenorm_residual_strategy`（把 `o_proj`/`down_proj` 按 `1/√(2N)` 缩放，默认 `None` 不生效），可作为"初始化阶段补丁 vs 结构层面解法"的对照。

## 下一次直接从这里继续

进入 rollout/trainer 分离和 policy staleness（答疑全文见 `qa/2026-09-21-rollout-trainer-staleness.md`）：

```text
rollout worker 使用 policy v10 生成 completion
old_logp 由 v10 计算
trainer 当前已更新到 policy v13
new_logp 由 v13 计算
```

回答：

1. `old_logp` 应保持 v10，还是用 v13 重算？
2. staleness 是几个 policy version？
3. staleness 过大时，ratio 与 clipfrac 如何变化？
4. 同步式 rollout-training 的主要优点是什么？
5. 异步式的主要优点和主要风险分别是什么？

之后推进：

```text
Policy 权重同步与版本管理
→ 混合精度
→ LoRA 的分布式训练与 rollout 权重同步
→ GRPO/PPO 最终选型验收
→ 阶段 7：继续预训练、蒸馏与迁移学习
```

# Day14 学习归档｜GRPO 原理、Reward Model 与最小训练闭环

- 学习日期：2026-09-11（Asia/Shanghai）
- 当前阶段：阶段 6——GRPO / RLAIF / 其他后训练
- 状态：in_progress
- 衔接来源：`archives/day13.md` 的 GRPO 续学入口

## 一、今日学习结论

今天已从“只知道 PPO/GRPO 主链差异”推进到能够解释 GRPO 的核心原理、Reward Model 与 DPO 的关系，并手算组内 advantage、sequence-to-token 广播、不同长度聚合、正负 advantage 下的 ratio clipping、reference KL，以及 iteration / batch / mini-batch / epoch 的关系。

尚未完成的入口题是：**全同 reward 与极小方差组的诊断和处理**。下次不要重讲 GRPO 基础，直接从文末续学题开始。

## 二、GRPO 原理主线

GRPO 对同一个 prompt 采样 `G` 条 completion，分别评分，并以同题候选的相对表现构造 advantage：

```text
同一 prompt
→ 采样 G 条 completion
→ 分别计算 sequence reward
→ 按 group_id 计算组内均值和标准差
→ 得到 sequence-level advantage
→ 广播到有效 response token
→ clipped policy objective + reference KL
→ 更新 policy
```

常见组内标准化：

```text
mu = mean(rewards)
sigma = population_std(rewards)
A_i = (r_i - mu) / (sigma + eps)
```

- `A_i > 0`：该 completion 优于同题平均水平，提高其已采样 token 的条件概率。
- `A_i < 0`：该 completion 差于同题平均水平，降低其已采样 token 的条件概率。
- `A_i ≈ 0`：相对学习信号很弱。
- 不同 prompt 不能混算均值和标准差；advantage 算好并固定后，completion 可以被打散到不同 mini-batch。

GRPO 不训练 critic，因此没有 PPO 中的 `old_values`、GAE、returns、value head 和 value loss；其相对基线来自同题其他候选的平均表现。代价是同一 prompt 要采样多条回答，结果高度依赖采样多样性、reward 质量和组内区分度。

## 三、PPO value → return 的四 token 复习

对四个 response token，若：

```text
reward:       [0,   0,   0,   1]
old_value:    [0.2, 0.3, 0.4, 0.5]
gamma = 1, lambda = 1
terminal after token 4
```

先计算 TD residual：

```text
delta_t = reward_t + gamma * next_value_t - old_value_t

deltas = [0.1, 0.1, 0.1, 0.5]
```

再从后向前计算 GAE：

```text
A_t = delta_t + gamma * lambda * A_(t+1)
advantages = [0.8, 0.7, 0.6, 0.5]
```

最后得到 critic 的固定监督目标：

```text
returns = old_values + advantages
        = [1.0, 1.0, 1.0, 1.0]
```

`new_values` 是带梯度的当前预测，`returns` 是同一 rollout 优化阶段内固定且无梯度的 target。

## 四、Reward Model 与 DPO

### 1. 业界常见 Reward Model

经典做法使用同一 prompt 下的偏好对：

```text
(prompt, chosen, rejected)
```

Reward Model 对完整回答输出标量，并使用 pairwise logistic / Bradley–Terry loss：

```text
L_RM = -log sigmoid(r_chosen - r_rejected)
```

常见结构是 Decoder-only Transformer 加 scalar reward head，在 EOS 或最后一个有效 token 的 hidden state 上输出 sequence score。

实践中通常不是只有一个学习型 RM，而是组合：

```text
正确性 verifier / 单元测试 / Schema 检查
+ helpfulness 或 instruction-following RM
+ safety RM
+ format reward
- violation penalty
```

数学、代码和结构化任务优先使用可验证 reward；开放式质量常使用偏好 RM、LLM-as-a-Judge 或人工评价。需要持续防范 reward hacking、长度偏好、解析器漏洞和分布外评分失真。

### 2. 为什么不总是直接使用 DPO

两者可以来自相同的离线偏好数据，但消费反馈的方式不同：

```text
离线 DPO：固定回答 + 固定 chosen/rejected 评价结果
RM + GRPO：当前 policy 新回答 + 固定评价器
迭代式 DPO：当前 policy 新回答 + 新构造的偏好对
```

GRPO 被称为在线策略训练，是因为每次 iteration 都使用当前 policy 新采样的 completion；RM 可以是离线训练并冻结的。“在线”不意味着 RM 必须在线更新，也不意味着连接真实用户。

相对经典离线 DPO，GRPO 的主要价值是：

- 能评价并利用当前 policy 新探索出的回答；
- 可直接消费标量 reward、单元测试通过率等强弱信息；
- 数据分布会随当前模型能力演化；
- 可组合 correctness、format、safety 等多种 reward。

代价是采样成本高、训练复杂、对 reward 和采样敏感，并更容易持续试探和放大固定评价器的漏洞。

## 五、采样与长度偏差

GRPO 通常对每个 prompt 使用随机采样生成 `G` 条 completion。关键参数包括 `G`、temperature、`top_p/top_k`、`max_new_tokens` 和 prompt 难度。

- temperature 太低：候选趋同，组内 reward 方差可能为零。
- temperature 太高：候选可能成为随机垃圾，平均质量和有效组比例下降。
- prompt 太简单：可能全部答对；太难：可能全部答错；两者都可能缺乏相对信号。

长度偏差不只是求均值导致的，主要来源包括：

1. Reward Model 本身偏爱较长、较详细的回答。
2. 全局 per-token 平均会让长 completion 因 token 更多而拥有更高整体梯度权重。
3. KL 若按 token 求和，长回答会积累更多 KL 惩罚。
4. 截断、答对概率和推理所需长度之间可能存在真实或虚假的相关性。

需要监控 reward 与 completion length 的相关性、长度分桶真实正确率、KL、截断率及格式通过率。

## 六、今日完成的手算与问答

### 1. 组内 advantage

```text
rewards = [0, 1, 1, 2]
mu = 1
离均差 = [-1, 0, 0, 1]
population variance = 0.5
sigma = sqrt(0.5) ≈ 0.7071
advantages ≈ [-1.414, 0, 0, 1.414]
```

因此：C1 概率应降低，C4 概率应提高，C2/C3 基本不变。学习者计算正确，只把总体方差误写成了 0，已纠正。

### 2. Sequence advantage 广播与 response mask

四条 completion 的有效长度为 `[2,4,3,1]` 时：

```text
response_mask = [
  [1,1,0,0],
  [1,1,1,1],
  [1,1,1,0],
  [1,0,0,0],
]
```

Sequence advantage 可以先广播到所有 token 位置，再由 mask 排除 PAD。C4 的 `A=1.414` 应用 mask 后逻辑上为：

```text
[1.414, 0, 0, 0]
```

Masked mean 必须同时限制分子和分母：

```text
masked_mean(x, mask) = (x * mask).sum() / mask.sum().clamp_min(1)
```

只乘 mask 再直接 `.mean()` 会让 PAD 虽为零却仍进入分母，稀释真实 loss 和梯度。

### 3. 全局 token 平均与逐 sequence 平均

两条回答长度分别为 2 和 4，每个有效 token loss 都为 2：

```text
全局有效 token 平均 = 2
先逐 sequence 平均、再跨 sequence 平均 = 2
```

虽然数值相同，梯度权重不同：

- 全局 token 平均：长回答总权重更高，长度 4 的回答占 `4/6`。
- 逐 sequence 平均：两条回答总权重各 `1/2`。

### 4. 正 advantage 下的 clipping

```text
A = +1.414
ratio = [1.3, 0.7]
clipped_ratio = [1.2, 0.8]

surrogate_1 ≈ [1.838, 0.990]
surrogate_2 ≈ [1.697, 1.131]
objective = min(...) ≈ [1.697, 0.990]
```

两个 ratio 都越界，但只有 token 1 最终选择裁剪项。Token 2 错误地降低了好 token 的概率，`min` 保留更小的未裁剪项，不通过 clipping 减轻惩罚。

### 5. 负 advantage 下的 clipping

```text
A = -1
ratio = [1.3, 0.7]
clipped_ratio = [1.2, 0.8]

surrogate_1 = [-1.3, -0.7]
surrogate_2 = [-1.2, -0.8]
objective = [-1.3, -0.8]
```

只有 token 2 最终选择裁剪项：降低坏 token 的概率方向正确，但一次降低过多仍被限制。核心规律是：clipping 只截断沿 advantage 指示的正确方向走得过远所获得的额外收益，不保护错误方向的更新。

### 6. 从 objective 到 loss，并加入 reference KL

```text
objective = [-1.3, -0.8]
KL = [0.1, 0.3]
beta = 0.2

-objective = [1.3, 0.8]
beta * KL = [0.02, 0.06]
per_token_loss = [1.32, 0.86]
sequence_loss = (1.32 + 0.86) / 2 = 1.09
```

学习者前三步正确，最后均值误算为 1.59，已纠正。

两类约束的区别：

```text
ratio clipping：参照生成本轮 rollout 的 old policy，限制本轮局部更新
reference KL：参照长期冻结的 reference/SFT model，限制多轮训练的累计漂移
```

### 7. Iteration、batch、mini-batch、epoch

给定：

```text
prompt batch B = 8
group size G = 4
rollout batch = 32 completions
mini-batch = 8 completions
optimization epochs K = 3
无梯度累积
```

则：

```text
每个 epoch：32 / 8 = 4 个 mini-batch
总 optimizer.step：4 * 3 = 12 次
```

当前 iteration 内固定：

```text
completion tokens, reward, advantage, old_logp, ref_logp
```

每次当前 policy forward 重算：

```text
new_logp, ratio, clipped objective, KL, loss, clipfrac
```

学习者正确算出 rollout 数和每 epoch mini-batch 数，但把 optimizer step 误写成 32，已纠正为 12。

### 8. Group 完整性

正确流程：

```text
先按 group_id 保持同题候选完整
→ 分别计算每组 mean/std/advantage
→ 固定 advantage
→ 再打乱全部 completion、拆成 mini-batch 优化
```

学习者已正确理解：不同 prompt 评价尺度和难度不同，不能跨 prompt 标准化；advantage 正确算好后，同组 completion 可分散到不同 mini-batch。

## 七、已掌握与待巩固

### 已掌握

- 能解释 GRPO 为什么不需要 critic，以及它相对 PPO 替换了哪段链路。
- 能解释经典离线 DPO、冻结 RM + GRPO、迭代式 DPO 的区别。
- 能手算组内均值、总体标准差和标准化 advantage。
- 能解释 sequence advantage 广播、response mask 与正确 masked mean。
- 能区分全局 token 平均和逐 sequence 平均的长度权重。
- 能结合 advantage 正负判断 ratio 越界与真正启用裁剪分支的区别。
- 能解释 old policy clipping 与 reference KL 的短期/长期参照差异。
- 能计算 rollout batch、mini-batch、optimization epoch 和 optimizer step 数。
- 能说明按组计算 advantage 后，优化阶段可以打散 completion。

### 仍需巩固

- 全同 reward 组和极小方差组的诊断、过滤及数值稳定策略。
- Reward std、有效组比例、clipfrac、KL、长度和 reward hacking 的监控表。
- 将上述概念写成带 `group_id` 的最小 PyTorch 实现和单元测试。

## 八、下次续学入口（未完成题）

不要重讲 GRPO 基础，直接回答以下题目。

考虑三组奖励：

```text
group A = [1, 1, 1, 1]
group B = [0, 0, 0, 0]
group C = [1.000, 1.000, 1.000, 1.001]
```

公式：

```text
A_i = (r_i - mu) / (sigma + eps)
```

请回答：

1. 为什么 group A 和 group B 都没有组内相对 policy 信号？
2. Group A 全部答对、group B 全部答错；它们虽然都没有 policy gradient，但对训练诊断分别说明什么？
3. Group C 的标准差非常小。如果 `eps` 也特别小，标准化后只高 `0.001` 的回答仍可能获得明显正 advantage。这样做有什么风险？
4. 低方差组应如何处理？可多选并说明：

```text
A. 无论方差多小都正常训练
B. 方差低于阈值时跳过该组
C. 增大分母中的 eps 或设置标准差下限
D. 记录有效组比例和 reward std
```

答完后继续：编写 `groupwise_advantages`、masked GRPO loss、reference KL，以及至少三个测试——全同 reward、变长/PAD、不同 group 不能混算。

## 九、下一阶段验收标准

- 能解释全同 reward 为什么不给相对 policy 信号，但仍具有诊断价值。
- 能安全处理零方差和极小方差组，不让数值噪声被标准化放大。
- 能实现按 `group_id` 标准化 advantage，且不同 prompt 不混算。
- PAD 不参与 loss 分子和分母。
- 能输出并解释 reward std、有效组比例、KL、clipfrac、completion length 等监控指标。

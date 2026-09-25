# Day15 学习归档｜GRPO 低方差组、有效组 Mask 与 Loss 拼接

- 学习日期：2026-09-14（Asia/Shanghai）
- 当前阶段：阶段 6——GRPO / RLAIF / 其他后训练
- 状态：in_progress
- 衔接来源：`archives/day14.md` 的“全同 reward 与极小方差组”续学题

## 一、今日学习结论

今天完成了 GRPO 异常 reward 组的诊断与过滤逻辑，理解了 `group_mask`、`valid_sequence_mask`、`response_mask` 和 `effective_mask` 的不同职责，并开始把 group-wise advantage 接入 clipped policy objective。

已能：

- 区分“全部答对组”和“全部答错组”的共同点与不同诊断含义；
- 解释极小 reward 差异为何会被标准化放大，以及 RM 噪声被 policy 学习的风险；
- 使用 `std_threshold` 过滤低方差组，并计算 `valid_group_rate`；
- 区分“只把 advantage 置零”和“整组完全跳过训练”；
- 组合 `response_mask` 与 `valid_sequence_mask` 得到 `effective_mask`；
- 理解按 `group_id` 计算 advantage 的 PyTorch 布尔索引；
- 根据正负 advantage 计算 clipped policy objective。

尚未完成的断点是：**在已知 policy objective 的基础上，加入 reference KL、PAD mask 和有效组 mask，计算最终 masked mean loss。** 明天直接从文末题目继续，不重讲前面的基础。

## 二、全同 Reward 组

### Group A：全部答对

```text
rewards = [1, 1, 1, 1]
mu = 1
sigma = 0
advantages = [0, 0, 0, 0]
```

### Group B：全部答错

```text
rewards = [0, 0, 0, 0]
mu = 0
sigma = 0
advantages = [0, 0, 0, 0]
```

两组都没有由 reward advantage 产生的相对 policy 信号，因为组内 completion 无法排序：没有证据说明应提高哪条、降低哪条。

但两者诊断含义不同：

- 全对组可能表示题目太简单、采样温度太低导致候选趋同、reward 太粗糙，或训练数据未覆盖模型能力边界。
- 全错组可能表示题目过难、探索不足、组大小太小、回答被截断、prompt/数据有问题、verifier/解析器故障，或 reward 过于稀疏。

因此“没有 relative policy gradient”不等于“没有诊断价值”。

## 三、极小方差与噪声放大

对：

```text
rewards = [1.000, 1.000, 1.000, 1.001]
mu = 1.00025
sigma ≈ 0.000433
```

若 `eps` 相对 `sigma` 可忽略：

```text
advantages ≈ [-0.577, -0.577, -0.577, +1.732]
```

虽然原始最大差异只有 `0.001`，但标准化计算的是“离均差相对于组内标准差有多大”，所以会得到明显 advantage。

如果 RM 的典型评分噪声为 `±0.01`，那么 `0.001` 的差异低于评价器的可靠分辨能力。正常标准化会把可能的错误排序放大，导致 policy 提高偶然高分回答的概率，并降低其他回答的概率。Ratio clipping 只能限制单次更新幅度，不能判断 advantage 方向是否可靠。

今日选择的保守策略：

```text
若 group_std < std_threshold：整组判为无效并跳过
```

阈值应来自 RM 校准、重复评分波动、人工一致性或 Judge 方差，并与 reward 尺度绑定，不能任意设置。

建议监控：

```text
zero_std_group_rate
low_std_group_rate
valid_group_rate
reward_std_mean
```

## 四、按 Group 计算 Advantage

输入：

```text
group 0 rewards = [1, 1, 1, 1]
group 1 rewards = [0, 0, 0, 0]
group 2 rewards = [0, 1, 1, 2]
std_threshold = 0.01
```

输出：

```text
有效组：group 2
无效组：group 0、group 1

advantages = [
     0,      0, 0,     0,
     0,      0, 0,     0,
    -1.414,  0, 0,  1.414,
]

valid_group_rate = 1 / 3 ≈ 33.3%
```

不同 prompt 的 reward 不能混合计算均值和标准差，否则会破坏“同一道题内相对比较”的含义。

## 五、`groupwise_advantages` 核心代码

```python
import torch


def groupwise_advantages(
    rewards: torch.Tensor,       # [N]
    group_ids: torch.Tensor,     # [N]
    std_threshold: float = 0.01,
    eps: float = 1e-8,
):
    advantages = torch.zeros_like(rewards)
    valid_sequence_mask = torch.zeros_like(
        rewards,
        dtype=torch.bool,
    )

    unique_group_ids = torch.unique(group_ids)
    valid_group_count = 0

    for group_id in unique_group_ids:
        group_mask = group_ids == group_id
        group_rewards = rewards[group_mask]

        group_mean = group_rewards.mean()
        group_std = group_rewards.std(correction=0)

        if group_std >= std_threshold:
            advantages[group_mask] = (
                group_rewards - group_mean
            ) / (group_std + eps)

            valid_sequence_mask[group_mask] = True
            valid_group_count += 1

    valid_group_rate = (
        valid_group_count / unique_group_ids.numel()
    )

    return advantages, valid_sequence_mask, valid_group_rate
```

关键点：

- `group_mask = group_ids == group_id` 只选择当前 prompt 的候选。
- `correction=0` 使用总体标准差，分母为 `G`；若使用样本标准差，`[0,1,1,2]` 的结果约为 `0.8165`，而不是 `0.7071`。
- `std_threshold` 判断 reward 差异是否可信；`eps` 主要保证数值安全，职责不同。
- `valid_sequence_mask` 是 `[N]`，每条 completion 一个值，方便后续广播到 `[N,T]`。

今日对布尔索引的理解检查：

```text
group_ids = [0, 0, 1, 1, 1]
group_id = 1

group_mask = [False, False, True, True, True]
```

## 六、只置零 Advantage 与整组跳过的区别

若低方差组仅令：

```text
advantage = 0
```

而每 token loss 为：

```text
per_token_loss = -policy_objective + beta * reference_KL
```

则：

```text
policy_objective = 0
per_token_loss = beta * reference_KL
```

这组仍会产生 KL 梯度，把 current policy 拉向 reference model。

今日选择的是更保守的策略：

```text
policy objective 和 KL 都跳过
```

因此需要返回 `valid_sequence_mask`，并与 token mask 组合。如果一个 mini-batch 没有任何有效 token，应跳过该次 `optimizer.step()`，而不是依靠分母 clamp 得到零 loss 后继续更新。

## 七、`response_mask`、`valid_sequence_mask` 与 `effective_mask`

- `response_mask`：排除 PAD，只保留真实 response token。
- `valid_sequence_mask`：排除所属 reward 组无效的整条 completion。
- `effective_mask`：两者同时满足才参与训练。

```python
effective_mask = (
    response_mask
    * valid_sequence_mask[:, None]
)
```

示例：

```text
per_token_loss = [
    [10, 10, 99],   # 所属组无效
    [ 2,  4,  6],   # 所属组有效
    [ 8, 99, 99],   # 所属组有效，后两位是 PAD
]

response_mask = [
    [1, 1, 0],
    [1, 1, 1],
    [1, 0, 0],
]

valid_sequence_mask = [0, 1, 1]
```

得到：

```text
effective_mask = [
    [0, 0, 0],
    [1, 1, 1],
    [1, 0, 0],
]
```

最终参与训练的 loss 为：

```text
[2, 4, 6, 8]
分子 = 20
分母 = 4
masked_mean = 5
```

分母不能使用张量总位置数，也不能只使用 response token 数；必须是 `effective_mask.sum()`，因为无效组的真实 response token 也需要排除。

## 八、从 Group Advantage 接入 Policy Objective

已知两条 completion：

```text
advantages = [-1, +1]

old_logp = [
    [-1.0, -1.0, -1.0],
    [-1.0, -1.0, -1.0],
]

new_logp = [
    [-0.8, -1.0, -1.2],
    [-0.8, -1.0, -1.2],
]
```

计算：

```text
log_ratio = [
    [0.2, 0, -0.2],
    [0.2, 0, -0.2],
]

ratio ≈ [
    [1.221, 1.000, 0.819],
    [1.221, 1.000, 0.819],
]
```

裁剪范围 `[0.8,1.2]`，advantage 广播后：

```text
token_advantages = [
    [-1, -1, -1],
    [+1, +1, +1],
]
```

得到：

```text
surrogate_1 = [
    [-1.221, -1.000, -0.819],
    [ 1.221,  1.000,  0.819],
]

surrogate_2 = [
    [-1.200, -1.000, -0.819],
    [ 1.200,  1.000,  0.819],
]

policy_objective = min(surrogate_1, surrogate_2) = [
    [-1.221, -1.000, -0.819],
    [ 1.200,  1.000,  0.819],
]
```

同样的 `ratio=1.221`：

- `A<0` 时，提高坏 token 概率是错误方向，保留未裁剪的更重惩罚；
- `A>0` 时，提高好 token 概率方向正确但幅度过大，采用裁剪项。

需要牢记：`ratio` 始终大于零；判断概率是否提高的条件是 `ratio>1`，不是 `ratio>0`。

## 九、明日直接续学题（尚未作答）

沿用：

```text
policy_objective = [
    [-1.221, -1.000, -0.819],
    [ 1.200,  1.000,  0.819],
]
```

假设：

```text
per_token_kl = [
    [0.10, 0.20, 9.99],
    [0.30, 0.10, 0.20],
]

beta = 0.1

response_mask = [
    [1, 1, 0],
    [1, 1, 1],
]

valid_sequence_mask = [0, 1]
```

第一条 completion 所属 reward 组无效；第二条所属组有效。公式：

```text
per_token_loss
= -policy_objective + beta * per_token_kl

effective_mask
= response_mask * valid_sequence_mask[:, None]
```

请计算：

```text
1. per_token_loss = [
    [...],
    [...]
]

2. effective_mask = [
    [...],
    [...]
]

3. 最终参与训练的 loss = [...]

4. masked mean =
```

提示：第一条 completion 即使有两个真实 token，也因为所属组无效而整行排除。

完成这题后，继续写完整 `grpo_loss`，并补全至少三个单元测试：

1. 全同 reward / 低方差组；
2. 变长 completion 与 PAD；
3. 不同 `group_id` 不能混算。

## 十、下一步验收标准

- 能把 group-wise advantage、有效组判断和 token loss 串成一条完整计算链。
- 能区分 advantage 置零与整组完全跳过的梯度差异。
- 能正确组合 `response_mask` 和 `valid_sequence_mask`。
- 能实现 masked clipped policy loss 与 reference KL。
- 能处理全无效 mini-batch，并输出有效组率、有效 token 数、KL、clipfrac 和长度指标。

# Day17 学习归档｜GRPO 梯度累积、严格 Token 平均与测试设计

- 学习日：Day17
- 当前阶段：阶段 6——GRPO 最小闭环
- 状态：in_progress
- 衔接来源：`archives/day16.md` 的完整源码调用链验收
- 本日方式：只学习和分析，不安装依赖、不下载模型、不运行真实训练代码

## 一、今日完成内容

今天从 `B/G/epoch/micro-batch/optimizer.step` 的源码级验收开始，完成了：

1. 一次 GRPO iteration 中 rollout、epoch、micro-batch 与 optimizer step 的数量计算。
2. 无效 micro-batch 的跳过逻辑，以及为什么只统计有效 micro-batch。
3. 梯度累积时的 loss 缩放、`zero_grad()`、gradient clipping 与 step 顺序。
4. 不完整尾部累积的提交与梯度修正。
5. 从“micro-batch 等权平均”升级到严格的“有效 token 全局平均”。
6. `grpo_update` 与 `train_one_iteration` 的职责重构分析。
7. 三个核心单元测试的设计与断言 API。
8. 全无效 epoch 的行为与监控断言。

今天没有修改或运行 `grpo_end_to_end.py`。现有源码仍是 Day16 的教学版本；今天形成的是下一版实现所需的准确设计。

## 二、源码调用次数验收

配置：

```text
B = 2
G = 4
micro_batch_size = 2
optimization_epochs K = 2
无梯度累积
```

结论：

```text
rollout completion = B × G = 8
每个 epoch 的 micro-batch = 8 / 2 = 4
两个 epoch 理论最多 optimizer.step = 4 × 2 = 8
```

实际 step 数可能更少，因为整个 micro-batch 若没有有效 token：

```text
effective_mask 全零
→ valid_token_count = 0
→ 不 backward
→ 不 optimizer.step
```

只有部分 completion 无效时，只要 micro-batch 中仍存在有效 token，就仍可参与更新。

## 三、固定量与重算量

同一 iteration 的多个 optimization epochs 内固定：

```text
prompt
completion tokens / full_ids
full_attention_mask
group_ids
reward
advantage
response_mask
valid_sequence_mask
old_logp
ref_logp
```

每次 current policy forward 重算：

```text
new_logp
ratio
clipped_ratio
policy_objective
per_token_kl
per_token_loss
loss
clipfrac
mean_kl
```

一次 iteration 只进行一次自由 rollout 阶段。`get_response_logp(policy)` 至少用于：

```text
rollout 后保存 old_logp
每个优化 micro-batch 重算 new_logp
```

## 四、Micro-batch、Mini-batch 与 Optimizer Batch

本课程后续固定术语：

```text
rollout batch：一次 iteration 采样出的全部 completion
micro-batch：一次真正进入显存并执行 forward/backward 的子集
optimizer batch：累积多个有效 micro-batch 后，一次 optimizer.step 覆盖的逻辑数据
mini-batch：泛称；不同框架可能用它指 micro-batch 或 optimizer batch
```

关系：

```text
rollout batch size = B × G
每个 epoch 的 micro-batch 数 = rollout size / micro_batch_size
optimizer batch size ≈ micro_batch_size × gradient_accumulation_steps
```

若：

```text
rollout = 16
micro_batch_size = 2
accumulation_steps = 2
K = 2
```

则：

```text
每个 epoch 8 个 micro-batch
每个 optimizer batch 4 条 completion
每个 epoch 4 次 step
两个 epoch 8 次 step
```

## 五、基础梯度累积

如果累积 `A` 个 micro-batch，简化版常写：

```python
scaled_loss = loss / A
scaled_loss.backward()
```

目的：让累积结果近似 micro-batch 平均梯度，而不是梯度总和。若不缩放，梯度会放大约 `A` 倍。

顺序：

```text
累积窗口开始前 zero_grad
→ 每个有效 micro-batch backward
→ 累积完成后 gradient clipping
→ optimizer.step
→ zero_grad
```

不能在每个 micro-batch 开始时 `zero_grad()`，否则前一批梯度会被清除。

## 六、只统计有效 Micro-batch

示例：

```text
accumulation_steps = 2
valid_token_count = [12,0,8,5]
```

参与梯度累积的是 micro-batch：

```text
1、3、4
```

第一次 step：

```text
micro-batch 1 + 3
```

尾部：

```text
micro-batch 4
```

采用提交尾部策略时，该 epoch 共 2 次 optimizer step。

因此应使用独立的：

```python
accumulation_count
```

只在有效 micro-batch 后增加，不能直接根据原始 `micro_step` 取模。

## 七、简化版尾部修正

如果所有 micro-batch 的 loss 已按配置累积步数缩放：

```python
(loss / gradient_accumulation_steps).backward()
```

尾部实际有效数为 `accumulation_count` 时，需要在 step 前修正：

```python
scale_correction = (
    gradient_accumulation_steps
    / accumulation_count
)

for parameter in policy_model.parameters():
    if parameter.grad is not None:
        parameter.grad.mul_(scale_correction)
```

然后：

```text
normalize correction
→ gradient clipping
→ optimizer.step
```

例如配置累积 4 个，尾部只有 2 个，修正系数为 `4/2=2`。

## 八、严格 Token 平均

Micro-batch 等权平均并不等于有效 token 全局平均。

示例：

```text
micro-batch 1：2 个 token，loss=[2,4]，mean=3
micro-batch 2：6 个 token，loss=[1,1,1,1,1,1]，mean=1
```

Micro-batch 等权：

```text
(3+1)/2 = 2
```

全局 token 平均：

```text
(2+4+1+1+1+1+1+1)/8 = 1.5
```

等权平均把只有 2 个 token 的 micro-batch 1 从应有的 `25%` 权重放大到 `50%`。

严格实现应让 `grpo_loss` 返回：

```python
loss_sum = (
    per_token_loss * effective_mask
).sum()

mean_loss = loss_sum / valid_token_count

return loss_sum, valid_token_count, metrics
```

其中：

```text
loss_sum：用于 backward
mean_loss：用于可比较的日志监控
```

累积窗口内：

```python
loss_sum.backward()
total_valid_token_count += valid_token_count
```

窗口结束：

```python
for parameter in policy_model.parameters():
    if parameter.grad is not None:
        parameter.grad.div_(total_valid_token_count)

clip_grad_norm_(...)
optimizer.step()
```

这样得到整个 optimizer batch 的有效 token 平均梯度。

## 九、严格 Token 平均的尾部

严格 token 平均时，尾部同样除以真实：

```text
total_valid_token_count
```

不再需要：

```text
gradient_accumulation_steps / accumulation_count
```

示例：尾部两个 micro-batch 的有效 token 数为 3 和 7，最终梯度除以：

```text
3 + 7 = 10
```

若 `accumulation_count == 0`，不执行额外 step。

## 十、Gradient Clipping 的位置

正确顺序：

```text
梯度求和
→ 除以 total_valid_token_count
→ 得到平均梯度
→ gradient clipping
→ optimizer.step
```

不能先裁剪总梯度再除以 token 数，否则裁剪阈值会依赖有效 token 数，并可能把最终平均梯度过度缩小。

示例：

```text
总梯度范数=40
token 数=10
max_grad_norm=5
```

正确：

```text
40/10=4，不裁剪
```

错误：

```text
40 先裁剪到 5，再 /10 = 0.5
```

## 十一、职责重构

梯度累积跨越多个 micro-batch，因此 optimizer 控制必须位于能看到整个循环的 `train_one_iteration()`。

推荐职责：

```text
collect_rollout
→ 自由生成、old/ref logp、reward、advantage、valid mask

grpo_loss
→ ratio、clip、KL、effective mask、loss_sum 与 metrics

grpo_update（当前名字暂保留）
→ current policy 对一个 micro-batch 计算 new_logp，再调用 grpo_loss
→ 不 zero_grad、不 backward、不 step

train_one_iteration
→ epoch、randperm、micro-batch、有效累积计数、token 归一化、clip、step、尾部
```

重构后的 `grpo_update` 设计：

```python
def grpo_update(policy_model, rollout, indices, config):
    new_logp = get_response_logp(
        model=policy_model,
        full_ids=rollout.full_ids[indices],
        full_attention_mask=rollout.full_attention_mask[indices],
        prompt_width=rollout.prompt_width,
        response_width=rollout.response_width,
    )

    return grpo_loss(
        new_logp=new_logp,
        old_logp=rollout.old_logp[indices],
        ref_logp=rollout.ref_logp[indices],
        advantages=rollout.advantages[indices],
        response_mask=rollout.response_mask[indices],
        valid_sequence_mask=rollout.valid_sequence_mask[indices],
        clip_eps=config.clip_eps,
        kl_beta=config.kl_beta,
    )
```

## 十二、三个核心单元测试设计

### 1. 不同 Group 不能混算

```text
rewards = [0,2,10,12]
group_ids = [0,0,1,1]
expected advantages = [-1,1,-1,1]
```

跨组标准化会错误地把 group 0 的两条都变成负 advantage，把 group 1 的两条都变成正 advantage。

断言：

```text
advantages → torch.allclose
valid_sequence_mask → torch.equal
valid_group_rate → pytest.approx
```

### 2. PAD 与无效组不影响 Loss

```text
response_mask = [[1,1,0],[1,0,0]]
valid_sequence_mask = [False,True]
effective_mask = [[0,0,0],[1,0,0]]
```

若唯一有效 loss 为 6：

```text
valid_token_count=1
masked_mean=6
```

若分子使用 effective mask，但分母错误使用 `response_mask.sum()=3`，会得到错误结果 `6/3=2`。

### 3. 全同 Reward 组过滤

```text
group 0 rewards=[1,1,1,1]
→ mean=1, std=0
→ advantages=[0,0,0,0]
→ valid_mask=[False,False,False,False]

group 1 rewards=[0,1,1,2]
→ mean=1, std≈0.7071
→ advantages≈[-1.414,0,0,1.414]
→ valid_mask=[True,True,True,True]

valid_group_rate=0.5
```

即使 advantage 为零，若 `valid_mask=True`，KL 仍可能产生梯度，因此必须严格检查布尔 mask。

## 十三、测试断言 API

```text
浮点 Tensor → torch.allclose
布尔 Tensor → torch.equal
整数 Tensor → torch.equal
Python 浮点数 → pytest.approx
```

`atol=1e-6` 表示允许约 `0.000001` 的绝对浮点误差。

## 十四、全无效 Epoch

若 4 个 micro-batch 全部：

```text
valid_token_count=0
```

正确结果：

```text
backward 次数=0
accumulation_count=0
尾部分支不进入
optimizer.step 次数=0
policy 参数不改变
skipped_micro_batches=4
skipped_epochs=1
```

不能只断言 `loss==0`。零 loss 不证明跳过了 backward/step/scheduler；错误的 optimizer step 仍可能更新状态，AdamW weight decay 甚至可能改变参数。

## 十五、今天的验证边界

用户明确选择：

```text
不在本地跑真实代码，只学习和分析
```

因此今天：

- 未安装 PyTorch/Transformers；
- 未下载模型；
- 未运行训练或单元测试；
- 未修改 Day16 的 `grpo_end_to_end.py`；
- 上述实现均为源码设计与手算验收结论。

## 十六、明天直接从这里继续

进入生产 GRPO 的第一个关键问题：分布式训练中的 group 完整性。

已知：

```text
一个 prompt 采样 G=4 条 completion
两个 GPU rank

rank 0 rewards=[0,1]
rank 1 rewards=[1,2]
```

请回答：

```text
1. rank 0 只按本地 [0,1] 计算，advantages 是什么？

2. rank 1 只按本地 [1,2] 计算，advantages 是什么？

3. 正确地对完整 group [0,1,1,2] 计算时，
   四个 advantages 是什么？

4. 为什么两个 rank 的本地结果虽然对称，仍然错误？

5. 分布式 GRPO 应该：
   A. 保证一个 group 完整位于同一 rank
   B. 跨 rank 汇总同组 reward 后再计算 advantage
   C. A 或 B 都可以
```

仍使用总体标准差：

```python
std(correction=0)
```

后续顺序：

```text
分布式 group 完整性
→ 全局 skip/step 一致性
→ 变长 batching 与长度权重
→ 混合精度、LoRA 与 rollout 引擎
→ GRPO/PPO 最终选型验收
→ 阶段 7：继续预训练、蒸馏与迁移学习
```

# Day16 学习归档｜GRPO 完整训练链路、Old/New Log-prob 与 Mini-batch

- 学习日：Day16
- 当前阶段：阶段 6——GRPO 最小闭环
- 状态：in_progress
- 衔接来源：`archives/day15.md` 的完整 loss 与训练循环入口
- 配套源码：`grpo_end_to_end.py`

## 一、今日学习结论

今天完成了 GRPO 从 prompt batch 到训练后推理的端到端源码梳理，并统一了函数名称。重点打通了以下链路：

```text
prompt batch
→ 每个 prompt 采样 G 条 completion
→ rollout batch
→ old_logp / ref_logp / reward
→ group-wise advantage / valid_sequence_mask
→ optimization epoch
→ mini-batch
→ new_logp / ratio / clip / KL / effective_mask
→ backward / optimizer.step
→ 下一 iteration 重新采样
→ 保存 policy 并部署推理
```

学习者已理解：

- `new policy` 在优化阶段不会自由生成新回答，而是对 old policy 已采样的固定回答做 teacher-forcing forward，重新计算 token log-prob；
- `old_logp`、`new_logp` 和 `ref_logp` 评价的是同一组 token，因此 response 长度和 mask 一致；
- `valid_sequence_mask` 不能手工设为全 `True`，必须由真实 reward 的组内标准差计算；
- `torch.randperm` 在 advantage 计算完成后随机打乱 rollout completion，再切分 mini-batch，且所有字段必须使用同一组索引。

## 二、Day16 完成的 Loss 闭环

沿用 Day15 的数据：

```text
policy_objective = [
    [-1.221, -1.000, -0.819],
    [ 1.200,  1.000,  0.819],
]

per_token_kl = [
    [0.10, 0.20, 9.99],
    [0.30, 0.10, 0.20],
]

beta = 0.1
response_mask = [[1,1,0],[1,1,1]]
valid_sequence_mask = [0,1]
```

逐 token loss：

```text
per_token_loss = -policy_objective + beta * per_token_kl

= [
    [ 1.231,  1.020,  1.818],
    [-1.170, -0.990, -0.799],
]
```

有效 mask：

```text
effective_mask = [
    [0,0,0],
    [1,1,1],
]
```

最终：

```text
valid_token_count = 3
loss = (-1.170 - 0.990 - 0.799) / 3 ≈ -0.986
```

Loss 为负数本身没有问题；优化器最小化的是 `-policy_objective + KL penalty`，关键是梯度方向、mask 和统计范围正确。

## 三、完整 `grpo_loss` 的关键点

```python
ratio = (new_logp - old_logp).exp()
token_advantages = advantages[:, None]

surrogate_1 = ratio * token_advantages
surrogate_2 = ratio.clamp(
    1.0 - clip_eps,
    1.0 + clip_eps,
) * token_advantages

policy_objective = torch.minimum(surrogate_1, surrogate_2)

ref_log_ratio = ref_logp - new_logp
per_token_kl = ref_log_ratio.exp() - ref_log_ratio - 1.0

per_token_loss = -policy_objective + kl_beta * per_token_kl

effective_mask = (
    response_mask * valid_sequence_mask[:, None]
).to(per_token_loss.dtype)

valid_token_count = effective_mask.sum()
```

需要持续记住：

- `advantages[:, None]`：`[N] → [N,1]`，广播给每条 completion 的全部 token；
- `ratio.clamp(...)`：限制 ratio；
- `torch.minimum(...)`：在两个 surrogate 中逐 token 取较小值；
- `valid_token_count` 必须使用 `effective_mask.sum()`，不能只用 `response_mask.sum()`。

## 四、全无效 Mini-batch

当：

```text
effective_mask 全零
valid_token_count = 0
```

分母使用 `clamp_min(1)` 只能避免 `0/0 → NaN`，不能把无效 batch 变成有效训练数据。

正确流程：

```python
if valid_token_count.item() == 0:
    continue

optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
optimizer.step()
```

无效 batch 不应：

- 执行 `backward()`；
- 执行 `optimizer.step()`；
- 推进 scheduler 或有效 update 计数。

## 五、Reference KL 与监控

采用的逐 token KL 估计：

```text
ref_log_ratio = ref_logp - new_logp
KL = exp(ref_log_ratio) - ref_log_ratio - 1
```

例子：

```text
new_logp = -0.8
ref_logp = -1.0
ref_log_ratio = -0.2
KL ≈ exp(-0.2) + 0.2 - 1 ≈ 0.019
```

当 `new_logp == ref_logp` 时，KL 为 0。

已完成监控手算：

```text
有效 ratio = [1.25,0.90,0.80]
裁剪范围 = [0.8,1.2]
clipfrac = 1/3 ≈ 0.333

有效 KL = [0.30,0.10,0.20]
mean_kl = 0.20
```

边界值 `0.8` 和 `1.2` 本身不算越界。`clipfrac` 只统计 ratio 越界，不等于最终选择 clipped surrogate 的比例。

## 六、有效率监控

示例中：

```text
3 个 group，其中 2 个有效
12 条 completion，其中 8 条有效
29 个 response token，其中 18 个有效
```

因此：

```text
valid_group_rate = 2/3 ≈ 0.667
valid_sequence_rate = 8/12 ≈ 0.667
valid_token_count = 18
valid_token_rate = 18/29 ≈ 0.621
```

三个指标分别回答：

- 有多少 prompt 产生了可比较 reward；
- 有多少 completion 进入训练；
- 已生成 token 中有多少真正用于梯度更新。

## 七、训练指标诊断

### 高 reward、低有效组率、高重复率

```text
reward_mean ≈ 0.98
valid_group_rate 很低
zero_std_group_rate 很高
completion_duplicate_rate 很高
```

通常提示题目过于简单、采样温度太低或 reward 太粗糙。`clipfrac` 低也可能只是因为几乎没有 reward advantage 更新。

### 低 reward、低有效组率、高截断率

应优先检查：

```text
被截断样例与 stop reason
max_new_tokens
停止规则
答案提取器和 verifier
题目难度
```

不能只通过提高采样随机性解决。

### 训练 reward 上升、验证正确率下降、KL 和长度上升

可能是 reward hacking、RM 长度偏好和 policy 漂移。应先检查高 reward 样例和 reward 管道，再考虑降低学习率/epoch、增强 KL，并从可靠 checkpoint 恢复。

## 八、Iteration、Epoch 与 Mini-batch

正确顺序：

```text
B. 当前 policy 为每个 prompt 采样 G 条 completion
→ E/F. 计算 reward，保存 old_logp/ref_logp/response_mask
→ C. 按 group_id 计算 advantage 和 valid_sequence_mask
→ A. 打乱 completion，拆 mini-batch，训练 K 个 epochs
→ D. 下一 iteration 用更新后的 policy 重新采样
```

同一 rollout 的 `K` 个 epochs 内固定：

```text
completion tokens
reward
advantage
valid_sequence_mask
response_mask
old_logp
ref_logp
```

每次 current policy forward 重算：

```text
new_logp
ratio
policy_objective
per_token_kl
loss
```

## 九、为什么 Old/New 的 Token 与 Mask 一致

Rollout 阶段：

```text
policy.generate(prompt)
→ 自由采样 old answer
```

优化阶段：

```text
current policy(prompt + fixed old answer)
→ teacher forcing
→ 重新计算固定 token 的 new_logp
```

New policy 不在当前优化阶段自由生成另一条回答。因此 old/new/ref 比较的是：

```text
同一个状态 = prompt + 相同 answer 前缀
同一个动作 = 当前固定 answer token
```

只有模型参数不同。新的回答、长度和 `response_mask` 要到下一 iteration 才重新采样。

## 十、完整源码的函数命名与职责

配套文件：`grpo_end_to_end.py`

```text
expand_prompts
→ 复制每个 prompt G 次，生成 group_ids

build_response_mask
→ 保留首个 EOS 及之前 token，排除后续 padding

get_response_logp
→ 对固定 prompt+completion 做 teacher forcing，返回 [N,T] logp

groupwise_advantages
→ 按 group_id 计算 advantage 和 valid_sequence_mask

collect_rollout
→ 自由生成一次，计算 old/ref logp、reward 和 advantage

grpo_loss
→ ratio、clip、reference KL、effective mask 和指标

grpo_update
→ 一个 mini-batch 的 new_logp、backward 和 optimizer.step

train_one_iteration
→ 一次 rollout batch + K 个 optimization epochs

train_grpo
→ 每个 prompt_loader batch 启动一次新 iteration

inference
→ 训练后仅用 policy_model.generate() 部署推理
```

## 十一、`torch.randperm` 的作用

```python
permutation = torch.randperm(
    sequence_count,
    device=policy_model.device,
)
```

若 `sequence_count=8`，可能生成：

```text
[5,2,7,0,3,6,1,4]
```

它是 `0...7` 的无重复随机排列：每个索引出现一次，不遗漏、不重复。

用途：

```text
advantage 已按完整 group 算好
→ 每个 epoch 重新打乱 completion
→ 再按 mini_batch_size 切分
```

所有 rollout 字段必须使用同一组 `indices`：

```python
mini_full_ids = rollout.full_ids[indices]
mini_old_logp = rollout.old_logp[indices]
mini_ref_logp = rollout.ref_logp[indices]
mini_advantages = rollout.advantages[indices]
```

这样打乱不会破坏 completion、概率、advantage 与 mask 的对应关系。

## 十二、训练后推理

部署推理只需要：

```text
训练后的 policy model
tokenizer
```

不再需要：

```text
reference model
reward/verifier
old_logp
advantage
group_id
GRPO loss
```

调用普通：

```python
policy_model.generate(...)
```

## 十三、今日产物与验证

- 新增完整教学源码：`grpo_end_to_end.py`
- 已使用 `python3 -m py_compile grpo_end_to_end.py` 通过语法检查。
- 未执行真实模型训练：源码需要安装 PyTorch/Transformers、下载模型权重，并可能需要 GPU；教学 verifier 也只是提取 completion 中最后一个整数。

## 十四、下次直接从这里继续

下一学习日先做源码级验收，不重讲 GRPO 原理：

1. 从 `train_grpo()` 开始，沿调用栈追踪一次 `B=2,G=4,M=2,K=2` 的数据 shape 与调用次数。
2. 明确为什么一次 iteration 只调用一次自由 rollout 阶段，而每个 mini-batch 都调用 `get_response_logp()` 计算 new logp。
3. 在当前无梯度累积版本基础上加入 `gradient_accumulation_steps=2`，计算 micro-batch、optimizer batch 和 step 数。
4. 为源码补三个可运行单元测试：不同 group 不混算、PAD 不影响 loss、全同 reward 组被过滤。
5. 讨论教学源码与生产 GRPO 的差距：分布式全局 group、变长 batching、混合精度、LoRA、rollout 引擎与 checkpoint/评测。

续学检查题：

```text
B=2, G=4, mini_batch_size=2, K=2
```

回答：

- 一次 iteration 有多少条 rollout completion？
- 每个 epoch 有多少个 mini-batch？
- 最多有多少次 optimizer.step()？
- 为什么实际 step 数可能更少？
- `torch.randperm` 为什么必须在 group advantage 计算之后使用？

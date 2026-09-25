# Day19 学习归档｜Rollout/Trainer 分离、Policy 角色与 Staleness

- 学习日：Day19
- 当前阶段：阶段 6——GRPO 生产工程边界
- 状态：系统讲解完成，待回答 `old_logp` 覆盖题
- 衔接来源：`archives/day18.md`
- 本日方式：只学习和分析，不运行真实训练或推理代码

## 一、当前新主题

Day18 完成了 DDP、跨 rank group 完整性、全局 skip/step、变长 batching 与长度聚合。Day19 转入生产 GRPO 系统架构：

```text
Rollout Workers
→ Reward / Verifier
→ Trainer Workers
→ Policy 权重同步
→ 下一轮 Rollout
```

Rollout 与 Training 的计算形态、硬件需求和优化目标不同，因此生产系统常把它们拆成不同 worker 或不同引擎。

## 二、Rollout 与 Training

### Rollout

```text
prompt
→ 自回归逐 token 生成
→ completion
```

特点：

```text
逐 token decode
KV Cache
变长 completion
continuous batching
追求生成吞吐
```

常见推理引擎包括 vLLM、TensorRT-LLM、SGLang；今天只学习架构，不安装或运行这些系统。

### Training

```text
prompt + 固定 completion
→ teacher forcing 一次性 forward
→ new_logp
→ backward
```

特点：

```text
不重新自由生成
保存 activation
需要梯度和 optimizer
依赖 DDP/FSDP/ZeRO
追求训练吞吐与显存效率
```

训练阶段评分的是 rollout 已经生成的固定 completion，不是再次调用 `generate()`。

## 三、三种 Policy 角色

一批 rollout 至少涉及三种策略状态：

```text
Behavior Policy
→ 实际生成 completion 的策略

Current Policy
→ 当前正在训练、会继续更新的策略

Reference Policy
→ 冻结的 KL 参考策略
```

对应 log-prob：

```text
old_logp
→ behavior policy 对固定 completion 的概率

new_logp
→ current policy 对相同 completion 重新评分的概率

ref_logp
→ reference policy 对相同 completion 的概率
```

三者必须评价同一组：

```text
prompt + completion tokens
```

但参数来源不同。

## 四、Old/New/Reference Log-prob

假设：

```text
policy v10 生成 completion y
```

则：

```text
old_logp = log π_v10(y | x)
```

训练时 current policy 可能已经变为 v13：

```text
new_logp = log π_v13(y | x)
```

GRPO/PPO ratio：

```text
ratio
= exp(new_logp - old_logp)
= π_current(y|x) / π_behavior(y|x)
```

它回答：

```text
当前策略相对于生成这条样本的行为策略，概率改变了多少？
```

`old_logp` 必须保留生成该 completion 的 behavior policy 版本，不能被 current policy 的评分覆盖。

`new_logp` 每次 current policy forward 重算，因为 policy 参数在 optimizer.step 后变化。

`ref_logp` 来自冻结 reference model，通常在 rollout 或对应固定数据阶段计算并保存；KL 使用：

```text
ref_log_ratio = ref_logp - new_logp
per_token_kl = exp(ref_log_ratio) - ref_log_ratio - 1
```

## 五、错误覆盖 Old Log-prob 的后果

错误流程：

```text
v10 生成 completion
old_logp 却用 v13 重新计算
new_logp 也用 v13 计算
```

于是：

```text
ratio
= exp(log π_v13 - log π_v13)
= 1
```

这样：

```text
ratio 不再反映 current policy 相对 behavior policy 的变化
clip 的约束失去明确参照
importance sampling 修正失去意义
clipfrac 可能被错误压低
```

本质上，completion 明明来自 v10，却被错误地当成来自 v13；行为策略信息被抹掉。

## 六、Policy Staleness

假设：

```text
rollout worker 使用 v10 生成
old_logp 来自 v10
trainer 当前已更新到 v13
new_logp 来自 v13
```

版本差可以记作：

```text
staleness = 13 - 10 = 3 个 policy update
```

但 staleness 不只是版本号差值，也要观察实际策略分布偏离程度：

```text
每次更新很小：版本差 3，分布可能仍接近
每次更新很大：版本差 3，分布可能已经明显偏离
```

建议监控：

```text
mean ratio
ratio quantile
clipfrac
KL
reward 分布
completion length
```

staleness 过大时可能出现：

```text
ratio 大量远离 1
大量超出 clip 区间
clipfrac 上升
有效策略梯度被裁剪
旧数据代表不了当前 policy
训练不稳定
```

## 七、同步式与异步式

### 同步式

```text
rollout(v10)
→ 完成 rollout
→ train(v10 data)
→ 更新到 v11
→ 同步 v11 权重
→ rollout(v11)
```

优点：

```text
数据新鲜
behavior policy 版本明确
old_logp 对齐简单
staleness 小
调试容易
```

代价：

```text
rollout 与 trainer 互相等待
一侧变慢会拖住另一侧
硬件利用率可能不足
```

### 异步式

```text
rollout workers 持续生成
trainer workers 持续消费和更新
```

优点：

```text
生成与训练重叠
硬件利用率更高
吞吐可能更高
```

风险：

```text
rollout 可能来自旧 policy
必须记录 policy_version
必须设置 staleness 准入规则
ratio/clipfrac 可能恶化
```

典型准入：

```text
current_version - rollout_version
<= max_allowed_staleness
```

过期数据可以：

```text
丢弃
降权
延迟消费
或只在明确支持离策略修正时使用
```

不能默认认为“有 old_logp 就能安全使用任意旧 rollout”。

## 八、Rollout 记录字段

生产记录至少应包含：

```text
policy_version
completion tokens
old_logp
reward
group_id
sequence_id
response_mask
```

复杂分布式/异步系统还可以记录：

```text
origin_worker
origin_rank
generation_timestamp
tokenizer_version
sampling_config
temperature
top_p
max_new_tokens
model checksum
```

这些字段用于确认：

```text
completion 由哪个 policy 生成
old_logp 是否与 completion 对齐
数据是否过期
是否可追踪和复现
```

## 九、下一题

场景：

```text
rollout worker 使用 policy v10 生成 completion
old_logp 也由 v10 计算
trainer 当前 policy 是 v13
new_logp 由 v13 计算
```

请回答：

```text
1. old_logp 应保持 v10 的值，还是使用 v13 对同一 completion 重算并覆盖？

2. 如果错误覆盖为 v13 的评分，ratio 会变成什么？

3. 为什么此时 ratio 失去 behavior-policy 对照意义？

4. 这批 rollout 的版本 staleness 是多少？

5. 同步式和异步式 rollout-training 的主要取舍是什么？
```

后续：

```text
Policy 权重同步与版本管理
→ 混合精度
→ LoRA 的分布式训练与 rollout 权重同步
→ GRPO/PPO 最终选型验收
→ 阶段 7：继续预训练、蒸馏与迁移学习
```

## 十、验证边界

本日没有安装依赖、下载模型、启动推理引擎或运行训练；内容来自系统讲解和概念分析。`grpo_end_to_end.py` 未修改。

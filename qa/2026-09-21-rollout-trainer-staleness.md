# 答疑归档｜Rollout/Trainer 分离与 Policy Staleness

- 归档日期：2026-09-21（Asia/Shanghai）
- 文档类型：答疑记录（不属于 `archives/dayNN.md` 逐日归档序列）
- 当前阶段：阶段 6——GRPO 生产工程边界
- 状态：答疑内容已归档，`Policy Staleness` 五问待作答
- 衔接来源：`archives/day18.md` 的“下一次直接从这里继续”
- 学习方式：只学习和分析，不运行真实训练/推理代码
- 归档来源：学习教练在答疑过程中给出的解释，按“答疑内容也进入学习存档”的要求补录

## 一、这份答疑回答了什么

Day18 结束时已经决定停止重复 loss/mask 算术，转入生产 GRPO 的新内容。本次答疑围绕一个核心问题展开：

> 生产环境里 rollout 和 training 为什么要拆成两套系统，拆开之后 policy 版本会带来什么问题？

答疑覆盖四块内容：

```text
1. Rollout 阶段与 Training 阶段的形态差异
2. 生产 GRPO 的 Rollout / Reward / Trainer 三段架构
3. Policy version 与 rollout staleness
4. 同步式与异步式两种权重同步策略
```

记录时保留学习教练的原始表述结构，便于日后直接复习。原答疑中最后一题与前面的“识别错误抵消”重复标为“第 80 题”，这里只做提示，不擅自改号。

## 二、Rollout 与 Training 为什么分离

GRPO 有两种计算形态，它们的硬件和系统需求并不相同。

### Rollout 阶段

```text
输入 prompt
→ 自回归逐 token 生成
→ 得到 completion
```

特点：

- 每一步只生成一个新 token；
- 高频使用 KV Cache；
- completion 长度不固定；
- 适合 continuous batching；
- 主要追求生成吞吐。

常用推理系统会针对这个阶段优化，例如：

```text
vLLM
TensorRT-LLM
SGLang
```

### Training 阶段

```text
prompt + 已生成 completion
→ teacher forcing 一次性 forward
→ 计算每个 response token 的 new_logp
→ backward
```

特点：

- 不重新生成 completion；
- 需要保存 activation；
- 需要计算梯度；
- 需要 DDP/FSDP/ZeRO；
- 主要关注训练吞吐和显存。

### 生产架构

因此生产系统可能设计为：

```text
Rollout Workers
    使用推理引擎生成 completion
            ↓
Reward / Verifier
    评分并计算 advantage
            ↓
Trainer Workers
    使用 PyTorch + FSDP 训练 policy
            ↓
同步新 policy 权重
            ↓
Rollout Workers 开始下一轮生成
```

## 三、Policy 版本与 Staleness

假设 trainer 参数版本是：

```text
policy version 10
```

Rollout worker 使用 version 10 生成 completion，并保存：

```text
old_logp(version 10)
```

随后 trainer 更新到：

```text
version 11
version 12
```

如果 rollout worker 还在使用 version 10 继续生成，那么这些数据相对于当前 policy 已经过时，这叫：

```text
rollout staleness
```

训练时：

```text
old_logp 来自 version 10
new_logp 来自 version 12
```

虽然 PPO/GRPO 的 ratio：

```text
exp(new_logp - old_logp)
```

可以容忍一定程度的 policy 差异，但差异太大时：

- ratio 大量超出 clip 区间；
- `clipfrac` 上升；
- 有效策略梯度被大量裁剪；
- 数据利用率下降；
- 训练更不稳定；
- rollout reward 分布不能代表当前 policy。

## 四、两种权重同步策略

### 同步式

```text
rollout(v10)
→ 等待 rollout 完成
→ train v10→v11
→ 同步 v11 给 rollout workers
→ rollout(v11)
```

优点：

```text
数据新鲜
old policy 版本明确
训练逻辑简单
```

缺点：

```text
rollout 和 trainer 可能互相等待
硬件利用率较低
```

### 异步式

```text
rollout workers 持续生成
trainer 持续消费并更新
```

优点：

```text
生成与训练重叠
吞吐更高
```

缺点：

```text
rollout 可能来自较旧 policy
必须管理 policy_version 和 staleness
```

### Rollout 记录与过期数据

生产 rollout 记录通常至少应带：

```text
policy_version
completion tokens
old_logp
reward
group_id
sequence_id
```

训练器可以限制：

```text
current_version - rollout_version
<= max_allowed_staleness
```

过旧数据可以：

```text
丢弃
降权
或只在明确支持离策略修正时使用
```

## 五、本次答疑的关键边界

1. Rollout 追求生成吞吐，Training 追求训练吞吐与显存效率，两者优化目标不同，因此值得拆成两套引擎。
2. `old_logp` 必须由生成这批 completion 的那个 policy 版本计算，否则 ratio 失去明确含义。
3. staleness 的实质是“行为策略”与“当前策略”的偏离，不是单纯的版本号大小。
4. 同步式的代价是利用率，异步式的代价是数据新鲜度；两者都要显式管理 policy version。
5. 过期数据不是只能丢弃，但降权或离策略修正只在实现明确支持时才是安全选项。

## 六、待作答：Policy Staleness 五问

场景：

```text
rollout worker 用 policy v10 生成 completion
old_logp 也由 v10 计算

trainer 当前已更新到 policy v13
训练时 new_logp 由 v13 计算
```

请回答：

```text
1. old_logp 应该保持 v10 的值，还是用 v13 重算？

2. 这批 rollout 的 staleness 是多少个 policy version？

3. staleness 过大时，ratio 和 clipfrac 可能出现什么变化？

4. 同步式 rollout-training 的主要优点是什么？

5. 异步式的主要优点和主要风险分别是什么？
```

这五问同时保留在 `CURRENT_PROGRESS.md` 的“下一次直接从这里继续”，回答后再回头更新本文件与进度页。

## 七、下一次继续

```text
Policy 权重同步与版本管理
→ 混合精度
→ LoRA 的分布式训练与 rollout 权重同步
→ GRPO/PPO 最终选型验收
→ 阶段 7：继续预训练、蒸馏与迁移学习
```

## 八、本次验证边界

本次只整理和归档答疑内容：

- 未运行分布式训练或推理引擎；
- 未安装或调用 vLLM / TensorRT-LLM / SGLang；
- 未修改 `grpo_end_to_end.py`；
- 结论来自学习教练的概念讲解，尚未经过代码或实验验证。

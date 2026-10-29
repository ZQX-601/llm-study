# Day31 学习归档｜训练数据准入与 PPO Tensor Loss

- 学习日：Day31（2026-10-29）
- 当前阶段：调整后主线第 9～11 周——Agentic RL 数据合同与训练闭环
- 状态：训练数据准入第一轮闭合；PPO Tensor Loss 部分完成
- 衔接来源：[Day30](day30.md) 的版本准入停点
- 本日方式：原理追问、数值手算、代码实现、边界测试
- 实际学习时长：未记录

## 一、本日目标与岗位意义

本日先完成 Trajectory/GRPO Group 进入 Trainer 前的准入合同，再把已准入的 PPO token
字段推进到因果错位、ratio、clipping、masked mean、Value/Entropy/KL 与 Autograd 边界。

主题等级：

- `岗位必备`：版本合同、policy lag 与近似 on-policy 数据新鲜度；
- `岗位必备`：Reward 重新评分与 Observation/Tool 变化后重新采集的边界；
- `岗位必备`：GRPO Group 完整性、失败轨迹与缺失轨迹的区别；
- `岗位必备`：causal shift、PPO clipped surrogate 与 Response masked mean；
- `需要掌握`：clipfrac、近似 KL、Entropy、Reference KL 与 Autograd detach。

对应工作决策：异步 Actor/Learner 中不能只按“文件存在”接纳轨迹；必须区分合法但过期、
可重新处理、必须重新采集和数据链路损坏，否则会把错误概率、残缺组或跨版本因果数据送入
训练。

## 二、Policy Lag 与两级新鲜度检查

用户追问为何不能只限制参数更新次数。经解释明确：同步 barrier 可以近乎消除 lag，但会受
慢 Actor/Tool 调用拖累；异步流水线通常允许有限 lag，再做两级检查：

```text
policy_lag = current_policy_version - behavior_policy_version
→ 低成本静态粗筛

current-policy forward
→ ratio / clipfrac / KL 动态检查真实分布偏移
```

用户理解版本差只是一种代理指标；同样 lag=1 的一次更新可能很小，也可能因学习率、梯度而
产生明显分布偏移。

## 三、重新评分与重新采集

完成两个版本场景判断：

1. Observation、Tool 和状态迁移不变，只把每次额外 Tool 调用加入 `-0.05` Reward，且旧
   Trace 保存了完整调用次数：用户正确选择 `REPROCESS`；
2. Tool 从 `get_training_logs(job_id)` 改成
   `get_training_logs(job_id, attempt_id)`，Observation 也增加 `attempt_id`：用户正确选择
   `RECOLLECT`，因为旧 Policy 没见过新字段，旧 Action 也不符合新 Tool schema。

形成边界：

```text
只改变可由完整 Trace 重算的标签
→ REPROCESS

改变 Observation、Action/Tool、状态迁移或终止语义
→ RECOLLECT
```

## 四、GRPO Group 完整性

对预期 `G=4`、只返回三条 rollout 的组，用户先判断“不能训练，数量不够”。进一步用数值
说明：缺失第四条会改变全组 Reward 均值、标准差和前三条 Group Advantage；超时/失败又
通常不是随机缺失，静默缩成 `G=3` 会造成幸存者偏差。

准入规则：

```text
3/4，未达到组装超时 → PENDING
3/4，组装超时       → 整组 REJECT
4/4，含合法终止失败 → 可以 ACCEPT
5/4 或重复 ID        → QUARANTINE
```

用户正确判断：如果第四条 Tool 超时被环境定义为明确终止，且保存了完整轨迹与失败 Reward，
它是有价值的负样本而非缺失数据。经补充，只有 Reward 不够，还必须有完整 Observation、
Action、old log-prob、组身份与终止边界。

## 五、REJECT 与 QUARANTINE

本日建立六类机器可分流决定：

```text
ACCEPT      可训练
PENDING     等待同组数据
REPROCESS   可由完整 Trace 重新处理
RECOLLECT   因果语义变化，重新采集
REJECT      数据合法，但不适合当前训练
QUARANTINE  数据合同可疑，隔离排障
```

用户正确判断 token/log-prob 长度不一致和 NaN 应 `QUARANTINE`。最初把高 clipfrac 判断为
`QUARANTINE`，经纠正后理解：高 clipfrac/KL 通常表示合法数据相对当前 Policy 偏移过大，
属于本轮 `REJECT`；只有错位、NaN、错误来源等证据才升级为数据链路隔离。

## 六、动态准入：ratio、clipfrac 与近似 KL

只在 `response_mask == 1` 的 token 上计算：

```text
log_ratio = new_logprob - old_logprob
ratio = exp(log_ratio)
clipfrac = 越过 [1-epsilon, 1+epsilon] 的有效 token 比例
approx_kl = mean((ratio - 1) - log_ratio)
```

用户把第二个原始 ratio `2.01` 写成裁剪后的 `1.2`，经纠正后能在新例子中区分：

```text
raw ratio≈0.74
clipped ratio=0.8
```

并独立补出越界条件：

```python
ratio > upper or ratio < lower
```

## 七、Causal Shift 与 PPO Policy Loss

用户正确追踪：

```text
logits            [3,250,50000]
shift_logits      [3,249,50000]
shift_targets     [3,249]
new_logprobs      [3,249]
```

明确 `logits[:,t-1]` 预测 `input_ids[:,t]`，因此 target、old log-prob、Advantage 与 mask 都要
按同一 token 语义对齐。用户在短序列中正确指出 `BOS→R1、R1→R2`，但一度复制了上一题的
mask；经纠正得到 `[1,1,0]`。

PPO clipped surrogate：

```python
unclipped = ratio * advantage
clipped = clamp(ratio, 0.8, 1.2) * advantage
policy_loss_per_token = -min(unclipped, clipped)
```

用户前两次分别把已给出的 ratio 再做指数、以及直接选择 clipped surrogate；负 Advantage
例子又漏乘 `-2`。经逐步纠正后，能独立完成：

```text
ratio=1.5, advantage=-2
unclipped=-3
clipped=-2.4
policy_loss=3
```

并正确计算 masked mean：

```text
loss=[99,-2.4,1.6,88]
mask=[0,1,1,0]
masked policy_loss=-0.4
```

## 八、Value、Entropy、Reference KL 与梯度边界

用户正确计算简单 Critic MSE：

```text
new_values=[0.5,1.2]
return_targets=[1.0,1.0]
value_loss=0.145
```

Value clipping 手算按用户要求跳过。Entropy 通过完整词表分布计算：

```text
H_t = -sum_v p(v|s_t) log p(v|s_t)
```

再对 Response token 做 masked mean；用户理解高 Entropy 对应更分散的策略分布。对于
Autograd，用户最初不知道 detach 的边界，经解释后正确判断：若 `approx_kl` 真正加入
`total_loss`，必须使用 detach 前的 Tensor；日志指标可以 detach。

进一步区分：

```text
new_logprobs：当前可训练 Policy，需要梯度
old_logprobs：真实 behavior Policy，用于 ratio，不需要梯度
ref_logprobs：固定 Reference，用于行为约束，不需要梯度
```

用户正确选择 PPO ratio 的分母为 `old_logprobs`。对训练时用当前 Policy 覆盖 old log-prob
的错误，正确判断 ratio 会伪装成 1；经补充区分“同一计算图导致梯度抵消”和“detach 后
仍有梯度但 denominator 不再代表 behavior Policy”。

PPO 总 loss：

```python
total_loss = (
    policy_loss
    + value_coef * value_loss
    - entropy_coef * entropy
    + kl_coef * reference_kl
)
```

用户计算时遗漏 `+0.002` KL 项，得到 `-0.3355`；正确结果为 `-0.3335`，后续 KL 数值题
按用户要求跳过。

## 九、代码与运行证据

本日新增/修改：

```text
coding/agentic_rl_data/contracts.py
coding/agentic_rl_data/admission.py
coding/agentic_rl_data/tensor_loss.py
coding/agentic_rl_data/test_agentic_rl_data.py
coding/agentic_rl_data/README.md
```

`Trajectory` 新增 task/environment/observation/tool/reward/verifier/data/tokenizer/chat-template
版本字段。`admission.py` 实现单轨迹与 GRPO Group 静态准入，以及 forward 后的动态
clipfrac/KL 准入。`tensor_loss.py` 包含：

- 已实际执行的纯 Python PPO loss 数值参考；
- 真实 PyTorch causal shift、gather、clipping、masked mean 与 metrics detach 写法。

验证状态：

```text
标准库 unittest：26 项通过
默认 Python：未安装 torch
PyTorch Tensor/Autograd 路径：代码已写，未实际运行 backward
真实 LLM/tokenizer/PPO/GRPO optimizer：未运行
付费 API/云 GPU：未使用
```

## 十、证据边界与精确停点

已形成：

- 能区分 Reward 重算与因果语义变化后的重新采集；
- 能解释 policy lag、ratio/clipfrac/KL 的两级新鲜度检查；
- 能区分失败 rollout 与缺失 rollout，以及 REJECT/QUARANTINE；
- 能追踪 causal shift、raw/clipped ratio、surrogate 和 masked mean；
- 版本准入与纯 Python loss 数值路径经 26 项测试实际验证。

仍缺：

- 新模块主要由教练实现，用户完成读码、判断和局部条件补全，未独立实现完整准入器；
- PPO surrogate 的正负 Advantage 四象限经过多次纠正，需换数值复习；
- 未实际安装 PyTorch 或执行 `backward()/optimizer.step()`；
- Value clipping、完整 Entropy/Reference KL Tensor 实现和 GRPO Tensor 聚合尚未闭合；
- 未运行真实模型、Critic、Reference、训练循环或云实验。

精确停点：

```text
父话题：Agentic RL｜准入样本 → Tensor Loss
当前小节：Autograd 计算图与训练循环

已覆盖：causal shift、ratio、PPO clipping、masked mean、Value/Entropy/KL 概念
待回答：为什么 policy_loss 调用 .item() 后不能 backward，应该怎样修改？
后续：显式 detach old/advantage → 完整 loss 组合 → backward/grad clip/optimizer step
```

## 十一、复习安排

- 1 天：换数值复习正/负 Advantage 下的 unclipped/clipped surrogate；
- 3 天：从 `[B,T,V]` 独立写出 causal shift、gather 与 Response masked mean；
- 7 天：阅读一段含 `.item()`、错误 detach、old log-prob 覆盖的训练代码并定位问题；
- 21 天：在可用 PyTorch 环境运行一次 forward/backward，记录 loss、KL、clipfrac 和梯度。

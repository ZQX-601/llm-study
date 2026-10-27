# Day29 学习归档｜Agent 决策轨迹、SFT/PPO 复习与 Agentic RL 数据入口

- 学习日：Day29（2026-10-27）
- 当前阶段：调整后主线第 9 周——Agentic RL 基础
- 状态：完成 Environment/Trajectory 第一轮与 SFT/PPO 关联复习；`Trajectory → PPOSample` 尚未开始阅读
- 衔接来源：[Day28](day28.md) 的多 Agent Trace、全局预算与停止合同
- 本日方式：原理讲解、确定性教学实现、自动化测试、逐题复习与错误纠正
- 实际学习时长：未记录

## 一、本日目标与岗位意义

本日开始把已有 Agent Runtime 的运行过程转成可供 SFT、PPO、GRPO 使用的数据合同。
重点不是立即启动真实训练，而是先固定：模型在每次决策前看见什么、生成了什么、
Runtime 如何执行、Reward 如何产生、何时终止，以及 behavior policy 信息如何随轨迹保存。

主题等级：

- `岗位必备`：Observation、Action、Decision、Transition、Trajectory 和 Environment 边界；
- `岗位必备`：SFT 正样本与 RL 负反馈的数据分流；
- `岗位必备`：PPO 的 rollout、GAE、old/new log-prob、ratio、clip 和 Loss 角色；
- `需要掌握`：token 级 GAE 与 Agent 决策级 GAE 的粒度差异；
- `需要掌握`：reward hacking、terminated/truncated 与 policy staleness 数据准入。

对应工作决策：一条 Agent Trace 不能原样当作训练数据。必须明确哪些事件是模型决策、
哪些只是 Runtime 内部日志，错误动作应进入哪种训练数据，以及奖励是否鼓励了重复查询等
投机行为。

## 二、Agent 决策到 RL Transition

本日建立的数据流：

```text
Observation
→ Behavior Policy 生成 token 和结构化 Action
→ Runtime / Tool / Verifier 执行 environment.step
→ Reward + next Observation
→ StepRecord
→ 多个 StepRecord 组成 Trajectory
```

明确区分：

```text
Environment State：Runtime 掌握的完整权威状态
Observation：模型被允许看到的状态投影
Action：模型提出的候选动作，不代表已经执行
Decision：Action + token + old_logprobs + policy_version + value
TraceEvent：Runtime 内部可观测事件
StepRecord：一次真实模型决策对应的 RL transition
```

一次 ToolAction 内可以发生 Schema、权限、dispatch、Tool 返回和 Evidence 准入等多个
TraceEvent；只要模型没有在中间重新决策，它们仍属于同一个 decision-level transition。

## 三、SFT 数据分流与复习

SFT 主链复习：

```text
高质量 Prompt/Response
→ Chat Template
→ Tokenizer
→ input_ids
→ Prompt/PAD labels 置为 -100
→ Decoder Forward
→ Assistant token Cross Entropy
→ Backward
→ 更新全参数或 LoRA A/B
→ SFT Policy
```

用户正确回答：Prompt token 仍保留在 `input_ids` 并参与 Attention，只有 Assistant 的
`A1/A2` 直接计算 SFT Loss；错误的无证据 FinalAction 应从 SFT 正标签排除，但可保留为
RL 负反馈或偏好数据中的 rejected response；给定只屏蔽 PAD 的错误代码，用户能定位还需
把 Prompt labels 置为 `-100`。

临时追问中补充了 LoRA：`A` 将 `d_in → r`，`B` 将 `r → d_out`，`r` 控制低秩容量，
`alpha/r` 控制 LoRA 分支缩放。该段为关联复习，没有新增真实 LoRA 训练证据。

## 四、PPO 完整链路复习

本日恢复并串联：

```text
Rollout Batch
→ reward / old_values
→ delta / GAE advantage / return
→ 固定 old_logp、advantage、return
→ 多个 PPO epoch
→ 每个 epoch 切 mini-batch
→ 重算 new_logp/new_values
→ ratio / clipped policy loss / value loss / reference KL
→ optimizer.step
→ 新 Policy 采集下一 iteration
```

明确层级：

```text
iteration：采集一批新 rollout，并完成若干 PPO epoch
rollout batch：当前 iteration 的全部历史经验
PPO epoch：完整遍历一次这批固定 rollout
mini-batch：一次逻辑参数更新的数据子集
micro-batch：受显存限制的一次物理前向/反向子集
```

用户理解并确认：`delta` 是单步 TD 预测误差，Advantage 表示动作相对 Critic 预期的
好坏，Return 是 Critic 的训练目标；Current Policy 与 Critic/Value Head 更新，old log-prob
和 Reference Policy 不更新。

## 五、GAE Token 矩阵手算

新增两条变长 response 的 `[B,T]=[2,4]` 教学矩阵，第一条成功样本：

```text
response_mask = [1, 1, 1, 0]
rewards       = [0.0, 0.0, 1.0, 0.0]
old_values    = [0.2, 0.4, 0.5, 0.0]
gamma=0.9, lambda=0.8
```

手算结果：

```text
delta     = [0.16,   0.05, 0.50, 0.0]
advantage = [0.4552, 0.41, 0.50, 0.0]
return    = [0.6552, 0.81, 1.00, 0.0]
```

第二条含局部正奖励但最终失败的样本，GAE 将终局负信号向前传播，使各有效位置的
Advantage 最终为负。用户表示该部分已明白。

当前 Agent 主实现采用决策级 GAE，再将一个 ToolAction/FinalAction 的 Advantage 广播到
它的 response token；token 级 GAE 是更细粒度的对照实现。

## 六、Old/New Policy、Ratio 与 Clipping 检查

用户正确回答：若用 current policy 覆盖 old log-prob，ratio 会变为 1。最初表述为
“破坏梯度信息”，经纠正后明确主要破坏的是 behavior-policy 对照、importance ratio、
clipping 与 staleness 判断；若 old log-prob 已 detach，并不必然让所有梯度消失。

用户完成：

```text
old_logp=-1.0, new_logp=-0.8
ratio=exp(0.2)≈1.22，超过 [0.8,1.2] 上界
```

对负 Advantage 且 ratio>1 的场景，用户正确选择 unclipped surrogate。随后明确：ratio
越界不等于最终一定启用 clipping，还需结合 Advantage 正负与 `min` 的结果判断。

## 七、代码与运行证据

新增目录：[Agentic RL 数据流水线](../coding/agentic_rl_data/README.md)。

```text
coding/agentic_rl_data/__init__.py
coding/agentic_rl_data/contracts.py
coding/agentic_rl_data/environment.py
coding/agentic_rl_data/sample_builder.py
coding/agentic_rl_data/gae_token_demo.py
coding/agentic_rl_data/demo.py
coding/agentic_rl_data/test_agentic_rl_data.py
coding/agentic_rl_data/README.md
```

覆盖内容：

```text
Observation / Action / Decision / RewardComponents
StepRecord / Trajectory
确定性 AgentRLEnvironment 与 ScriptedPolicy
trajectory 采集和版本/结构准入
SFT / PPO / GRPO 样本构造
token-level GAE 演示
padding 与 mask 对齐
```

运行中发现并修复两类问题：

1. 当前系统 Python 3.9 不支持 `zip(..., strict=True)`，已改为兼容写法；
2. 重复查询同一 Evidence 原本可以反复得正奖励，且截断轨迹总分为正；现已改为
   同一 Evidence 只首次奖励、重复查询增加成本、`MAX_STEPS` 加未完成惩罚。

实际结果：

```text
成功轨迹：       3 steps，reward= 1.74
低效成功轨迹：   4 steps，reward= 1.22
重复查询并截断： 2 steps，reward=-0.47
标准库 unittest：13 项通过
git diff --check：通过
```

没有调用真实 LLM、真实 tokenizer、真实 Tool/MCP、PyTorch backward、付费 API 或云 GPU。
`sample_builder.py` 中 SFT/PPO/GRPO 后续代码已准备并通过测试，不等于用户已经读完或独立
实现。

## 八、证据边界与薄弱点

已形成的证据：

- 能解释 TraceEvent 与 decision-level transition 的区别；
- 能区分 SFT 正标签、RL 负反馈和偏好 rejected 数据；
- 能正确处理 assistant-only SFT 的 Prompt/PAD label mask；
- 能手算简单 ratio、判断正负 Advantage 下的 clipping；
- 理解 delta、GAE Advantage、Return 及 PPO batch 层级；
- 教学数据合同、reward 防刷和 13 项测试实际运行通过。

仍需保留的缺口：

- 代码主要由教练实现，用户尚未独立实现环境或样本转换；
- 用户尚未开始阅读 `Trajectory → PPOSample` 及其后的 padding、GRPO 样本代码；
- token-level GAE 与决策级 GAE 尚未由用户独立改码验证；
- Policy staleness 已纠正 old log-prob 覆盖问题，但 on/off-policy 准入和异步同步取舍仍待；
- 未接真实 tokenizer、模型 generation、new/ref log-prob、Critic 或训练 backward。

## 九、话题栈与明日精确入口

用户明确说明实际阅读停点在以下标题之前：

```text
返回父话题：Agentic RL｜Trajectory → PPOSample
```

因此该标题之后已经准备的讲解不计为已学习。明天从这里重新开始：

```text
父话题：Agentic RL
当前小节：Trajectory → PPOSample

先读 build_ppo_samples()
→ 决策级 advantage/return 如何进入 PPOSample
→ input_ids / response_mask / old_logprobs 对齐
→ scalar decision advantage 广播到 Action token
→ collate_ppo_batch 的 [B,L_max] padding
→ 再进入 GRPO group 样本
```

本日主题为“部分完成”：Environment/Trajectory、SFT/PPO 关联复习已完成第一轮；
PPOSample、batch collate、GRPO 数据处理及真实训练接入尚未学习。

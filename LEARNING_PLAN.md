# 大模型算法与 Agent 工程总学习计划

本计划依据 Day02–Day13 的完整归档、当前学习教练职责与已记录的后续方向重建。它用于续学导航；具体已发生的问答与验收以 `archives/` 为准。

## 总目标

建立一套可执行、可验证、可迭代的能力体系，兼顾：原理理解、论文阅读、代码实现、实验复现与工程设计。最终能够独立完成大模型训练/后训练实验，并设计具备 RAG、记忆、Skill、工具调用和多 Agent 协作能力的系统。

## 阶段路线

| 阶段 | 主题 | 当前状态 | 阶段验收 |
|---|---|---|---|
| 1 | Transformer、BERT/GPT、Decoder-only 预训练 | 已完成基础闭环 | 能解释 causal attention、next-token shift、token/mask/loss 与训练/推理差异 |
| 2 | SFT、LoRA、QLoRA 与实验设计 | 已完成核心闭环 | 能构造 assistant-only labels，解释 LoRA/QLoRA 梯度和显存，并设计可比较实验 |
| 3 | DPO 与偏好数据 | 已完成核心闭环 | 能区分 SFT 与 DPO，解释 chosen/rejected、reference、beta、数据污染与训练诊断 |
| 4 | 数据、评测、治理与安全门槛 | 已完成核心闭环 | 能按风险设计数据切分、指标、硬门槛、版本化和回滚，而非只看单一 accuracy/reward |
| 5 | PPO / RLHF | PPO 已完成代码级闭环 | 能从 rollout 追踪 old/new logp、GAE、returns、critic、clip、mask、KL 和 optimizer step |
| 6 | GRPO / RLAIF / 其他后训练 | 已完成分布式 group、变长 batching，以及 rollout/trainer 分离答疑（见 `qa/`），待权重同步和阶段验收 | 能手算组内 advantage，实现 masked GRPO loss，诊断全同 reward、长度偏差和 reward hacking |
| 7 | 继续预训练、知识蒸馏、迁移学习 | 未系统展开 | 为具体任务选择训练范式，定义数据、目标函数、超参数、评测和成本边界 |
| 8 | 分布式训练与推理优化 | 未系统展开 | 能解释并实践数据/张量/流水并行、显存与吞吐权衡、KV Cache、量化和服务化指标 |
| 9 | RAG | 未开始 | 实现检索、切分、embedding、重排、生成、评测与治理闭环 |
| 10 | MCP、工具调用与单 Agent | 未开始 | 定义工具契约、规划/执行循环、错误恢复、可观测性、权限与安全边界 |
| 11 | 记忆与 Skill | 未开始 | 设计短期/长期记忆的存储、检索、压缩、遗忘、权限；设计 Skill 注册、路由、版本与评测 |
| 12 | 多 Agent 系统 | 未开始 | 设计角色、状态、消息协议、调度、冲突处理、可观测性、评测与安全边界 |

## 当前阶段：GRPO 最小闭环

### 学习目标

1. 理解 GRPO 如何用同一 prompt 下候选组的相对奖励替代 PPO 的 critic / GAE。
2. 掌握组内均值、总体标准差与标准化 advantage。
3. 将 sequence-level advantage 广播到有效 response token。
4. 实现 ratio/clip、reference KL 和 response mask。
5. 识别全同 reward、极小方差、长度偏差、奖励投机与组采样成本。

### 核心概念

```text
同一 prompt 采样 G 条 completion
→ 每条 completion 获得 reward
→ 组内标准化形成相对 advantage
→ 广播到有效 response token
→ clipped policy objective + reference KL
→ 监控 reward / std / KL / clipfrac / 长度
```

常见基本形式：

```text
mu = mean(rewards)
sigma = population_std(rewards)
A_i = (r_i - mu) / (sigma + eps)
```

### 最小实践项目

输入：同一 prompt 的 4 条 completion，奖励 `[0, 1, 1, 2]`，以及变长 token 和 `response_mask`。

交付物：

1. 手算 `mu`、`sigma` 和四个 advantage。
2. 一条最小 rollout 记录，包含 `prompt`、`completion`、`reward`、`group_id`、`response_mask`、`old_logp`。
3. PyTorch 核心函数：group-wise advantage、masked clipped policy loss、reference KL。
4. 至少三个单元测试：全同 reward、变长/PAD、组间不能混算。
5. 监控表：组内 reward std、KL、clipfrac、completion length、有效组比例。

验收标准：

- 能解释为何 GRPO 不需 value head，但仍需要 reference / old policy 约束。
- 能说明 `[1,1,1,1]` 为何不给相对 policy 信号。
- PAD 不参与 token loss，且不同 prompt 的候选不能混在同一组标准化。
- 能说明 completion-level reward 广播给 token 的信用分配局限。

可选进阶：比较不同组大小、样本标准差/总体标准差约定、per-token 与 per-sequence 聚合、长度归一化方式，并复现一个小型可验证 reward 任务。

预计投入：1–2 个学习日完成手算和最小代码；再用 1 个学习日完成异常组、监控与 PPO/GRPO 选型对照。

常见误区：

- 把 GRPO 当成“没有 value head 的 PPO”而忽略组采样与相对基线。
- 跨 prompt 计算均值/方差，破坏“同题相对比较”。
- reward 全相同时仍强行除以极小 `eps` 并误以为有有效信号。
- 忽略 PAD、长度和 token 聚合方式，造成隐式长度偏好。
- 只看 reward 上升，不监控 KL、独立评测与 reward hacking。

## 后续阶段的启动原则

每进入一个新主题，先完成四项：

1. 读取当前归档，列出“已掌握 / 待巩固 / 未学习”。
2. 明确数据格式、目标函数、训练/推理流程和关键超参数。
3. 设计一个一周内可启动的最小实践，写清输入、交付物、验收与可选进阶。
4. 用代码、实验结果或结构化问答验收，再进入下一阶段。

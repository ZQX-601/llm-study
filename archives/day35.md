# Day35 学习归档｜Agentic RL 阶段回顾与后训练框架导入

- 学习日：Day35（2026-11-06）
- 当前阶段：调整后主线第 11～12 周——Agentic RL 训练闭环与框架实战
- 状态：Day29–34 阶段回顾完成；后训练框架实战导入部分完成
- 衔接来源：[Day34](day34.md) 的阶段回顾安排
- 本日方式：系统回顾、跨模块 Trace、课程路线调整、框架分层讲解
- 实际学习时长：未记录

> 归档边界：记录到 TRL、vLLM、verl、FSDP/DeepSpeed 与 Ray 的第一轮职责导入。
> 末尾关于“只有 vLLM rollout 是否已经完成 GRPO 训练”的判断题尚未回答，保留为下一次精确停点，
> 不计为已检测通过。

## 一、本日目标与岗位意义

主题等级：`岗位必备`。

本日先把 Day29–34 的 Agentic RL 数据、训练和诊断模块串成闭环，再处理课程长期停留在教学实现、
缺少主流开源框架实践的问题。目标不是记住框架名，而是明确算法字段在真实训练系统中的位置，
并为后续真实 backward、rollout、checkpoint、指标和分布式编排建立可验收路线。

## 二、Agentic RL 阶段主链回顾

完成以下链路的第一轮串联：

```text
Task / Prompt
→ 锁定 behavior policy 与 environment 版本
→ Agent Decisions / Trajectory
→ 静态合同与动态新鲜度准入
→ SFT / PPO / GRPO Sample
→ causal shift / response mask
→ Reward / Advantage / Return target
→ loss reduction
→ backward / optimizer gate
→ 原子权重发布与 Actor 边界刷新
→ 下一轮 rollout 与训练指标诊断
```

阶段证据边界：理论和教学数据流已连接，`coding/agentic_rl_data/` 保留 46 项标准库测试；
真实 PyTorch backward、模型 rollout、分布式 Actor/Learner、checkpoint 和 Dashboard 仍未运行，
独立实现仍是主要缺口。

## 三、PPO 是否要求同一 Prompt 采样多条 Trajectory

用户追问 PPO 是否像 GRPO 一样要求一个 Prompt 采样多条 Trajectory。结论是：

- PPO 算法本身不要求同一 Prompt 必须采样多条 Trajectory；Critic/GAE 为每条轨迹提供 token-level
  baseline 和 Advantage。
- PPO 可以出于方差、探索和吞吐考虑对同一 Prompt 采样多条，但每条轨迹仍独立计算 GAE，
  这不是 PPO 的必要定义。
- GRPO 需要同一 Prompt 下的 `G` 条 completion/trajectory，用组内 Reward 的相对标准化形成
  sequence-level Group Advantage，以替代 Critic。
- 一条多轮 Agent Trajectory 中包含多个 Decision，不等于同一 Prompt 的多条 Trajectory。

## 四、Reward 版本变化后的派生字段一致性

场景：一批 PPO 轨迹已在 `reward_version=r1` 下计算 `rewards`、GAE `advantages` 和
`return_targets`，随后只修正 Reward 逻辑并使用 `r2` 对完整原始轨迹重新打分。

用户正确回答：不能复用旧的 `return_targets` 和 `advantages`；Reward 重算后两者都必须重算。

完整字段处理为：

- 可保留：原始轨迹、`input_ids`、`response_mask`、rollout 时固定的 `old_logprobs`、
  `old_values` 和 behavior policy 版本；
- 必须重算：`rewards`、GAE `deltas`、`advantages`、`return_targets`；
- 若 Reward 只做离线评分且不改变环境转移、终止条件和 Tool 可用性，可走 `REPROCESS`，无需重新采样；
- 若修改影响了 Agent 实际交互过程，则旧 Trajectory 不再代表新环境合同，需要 `RECOLLECT`。

此题证明用户能从 Reward 追踪 PPO 派生目标，而不是只看字段是否存在。

## 五、课程路线调整：加入主流后训练框架

用户指出 GitHub 上存在 vLLM、verl 等工作中常见的项目，如果课程长期只使用教学实现，会缺少真实
工程能力。课程据此正式调整为四层推进：

```text
算法原理与最小张量实现
→ 主流框架中的字段、配置和调用链映射
→ 小模型或 dry-run 的真实执行
→ 指标、失败日志、版本与产物复盘
```

已更新 `study_agent/ROADMAP.md`，加入以下实战顺序：

1. Hugging Face Transformers、PEFT、TRL：先在单机小模型上跑通真实 SFT/LoRA、DPO/GRPO；
2. vLLM：学习 rollout、调度、KV cache、采样、log-prob 和权重刷新边界；
3. verl：将 Actor、Rollout、Reference、Critic、Reward、Advantage、Worker 和 checkpoint
   映射到完整 RL 后训练闭环；
4. FSDP/FSDP2、DeepSpeed ZeRO、Ray 与监控：学习分片、资源编排、背压、日志和指标；
5. SGLang、OpenRLHF 等先做横向选型，需要时再深入，避免同时铺开多个大型框架。

本地 RTX 4060 约 8 GB 显存，默认先使用小模型、LoRA/QLoRA、短序列和小 batch；多卡/多节点
部分先做源码 Trace、配置 dry-run 和资源估算。云 GPU 仍需先明确版本、资源、费用和停止条件。

## 六、TRL、vLLM 与 verl 的职责分层

第一轮系统讲解形成以下边界：

```text
TRL
└── 轻量一体化后训练入口，适合单机建立真实训练证据

verl
├── RL 算法流程与分布式角色编排
├── Actor / Reference / Critic / Reward
├── Ray Worker 与资源布局
└── 可接 vLLM / SGLang rollout 和 FSDP 等训练后端

vLLM
└── 高吞吐推理与 rollout engine，不负责 loss、backward 和 optimizer

FSDP / DeepSpeed
└── 参数、梯度和优化器状态的分片/offload

Ray
└── Worker 进程、CPU/GPU 资源和远程调用编排
```

关键结论：vLLM 和 verl 不是同层替代品。vLLM 可以作为 verl 使用的 rollout engine；只有生成
completion 尚未形成 GRPO 训练，还缺 Reward、Group Advantage、Policy loss、backward、optimizer
以及更新后权重进入下一轮 rollout 的闭环。

## 七、代码、运行与资料证据

本日修改：

```text
study_agent/ROADMAP.md
study_agent/STATE.md
archives/day35.md
```

验证状态：

```text
路线与状态文档：已更新
git diff --check：归档提交前检查
新增教学代码：无
新增自动化测试：无
TRL/vLLM/verl 安装与运行：未开始
真实模型 forward/backward/optimizer：未运行
付费 API/云 GPU：未使用
```

本日框架定位参考官方 vLLM 架构文档、verl 官方仓库和 Hugging Face TRL 官方 Trainer 文档；
尚未固定本地实验所用 commit、镜像或包版本，这些必须在第一次真实实验前完成。

## 八、父话题收尾与精确停点

已完成父话题：`Agentic RL｜Day29–34 阶段回顾（第一轮）`。

证据分类：

- 理论理解：能区分 PPO 与 GRPO 的采样要求，并正确处理 Reward 变化后的派生字段；
- 代码产物：沿用前序 `coding/agentic_rl_data/`，本日无新增训练代码；
- 实际运行：本日无新实验，真实训练仍缺失；
- 独立验收：跨模块 Reward Trace 回答正确，框架职责题尚未作答；
- 工程缺口：TRL 小模型训练、vLLM rollout、verl 源码调用链和最小闭环均未开始。

当前父话题：`后训练框架实战导入`。

精确停点：

```text
当前小节：TRL、vLLM、verl 分别解决什么问题

未回答问题：如果项目已经用 vLLM 为每个 Prompt 生成 8 条 completion，
            但没有计算 Group Advantage，也没有执行 loss、backward 和 optimizer，
            是否已经完成 GRPO 训练？vLLM 只完成了哪个阶段？

下一步：先完成职责边界检查，再建立教学字段到 TRL/vLLM/verl 的组件映射，
       随后准备本地 8 GB 可执行的 TRL 最小真实训练。
```

## 九、复习安排

- 1 天：区分 TRL、vLLM、verl、FSDP/DeepSpeed 和 Ray 的系统层级；
- 3 天：把 `input_ids/old_logprobs/reward/advantage/return` 映射到框架数据流；
- 7 天：追踪一次真实 Trainer/Worker 调用链，并解释 rollout 与训练的边界；
- 21 天：根据实际运行记录复盘配置、checkpoint、Reward/KL/clipfrac 与失败案例。

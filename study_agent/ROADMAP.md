# 未来 12 周路线（LLM → Agent → Agentic RL）

这是 2026-09-28 调整、2026-10-12 再次更新，并在 2026-11-06 加入主流后训练框架实战后的主线，按每周约 28 小时估算：6 天学习和练习，1 天复习、补缺与总结。用户已初步学习 SFT、PPO、GRPO 原理，并已完成 Agent Runtime、Plan-and-Execute 和多 Agent 第一轮。用户选择暂缓真实领域 Agent v1，直接进入 Agentic RL；前期先复用现有训练平台 mock、Trace 和评测合同建立教学环境，不能把真实领域闭环记为已完成。既有 `LEARNING_PLAN.md` 保留为完整主题地图。

Kimi K3、DeepSeek V4.1 等架构学习属于独立 QA 专题线，不自动穿插到本路线的计划学习、复习或阶段验收中；用户主动选择专题时再继续。

| 周 | 主线与岗位重点 | 必交证据 |
| --- | --- | --- |
| 1 | LLM 应用接口基础：推理参数、结构化输出、function calling、上下文管理 | 一个带 schema 校验和错误处理的模型调用程序 |
| 2 | RAG 基础：切分、embedding、召回、重排、生成与离线评估 | 可复现 RAG 小项目、数据集和基线指标 |
| 3 | Tool Use 与 MCP：工具协议、参数校验、权限、超时、重试和幂等 | 可调用多个真实/模拟工具的 Agent 原型 |
| 4 | 单 Agent 核心循环：任务分解、执行、观察、状态机与失败恢复 | 显式 Agent loop、轨迹日志和失败案例测试 |
| 5 | Agent 记忆与上下文工程：短期状态、长期记忆、检索、压缩与隔离 | 两种记忆策略对照及污染/过期案例 |
| 6 | Agent 评估与可观测性：任务成功率、轨迹质量、成本、时延、安全边界 | 评测集、trace、错误分类和回归测试 |
| 7 | 单 Agent 高级编排到多 Agent：Plan-and-Execute、Step 内有限 ReAct、分节点 Prompt、Replan、恢复与评测，再进入角色边界、handoff、并发、冲突和停止条件 | 一个可运行的 Plan-and-Execute 工作流、纯 ReAct 对照，以及一个可解释的多 Agent 协作工作流 |
| 8 | **暂缓（用户选择）**：真实领域 Agent v1、RAG/MCP/持久化与线上评测闭环 | 保留为后续工程补课，不作为开始 Agentic RL 理论与教学环境的前置门槛 |
| 9 | Agentic RL 基础：trajectory、environment、reward、credit assignment、on/off-policy | 将现有 Agent Runtime 任务形式化为可训练环境和数据合同 |
| 10 | Agent 轨迹的 SFT/偏好数据：轨迹清洗、成功示范、负例、verifier/reward | 可复用训练数据、奖励设计和离线评估基线 |
| 11 | Agentic RL 训练闭环与框架映射：恢复 PPO/GRPO、policy version、staleness、权重同步；用 TRL 跑通单机最小训练，阅读 vLLM rollout 路径，并把教学合同映射到 verl | 一次最小真实训练、reward/KL/clipfrac 记录、框架字段/组件映射和失败分析 |
| 12 | verl 最小闭环、Agent Runtime + LLM 联合迭代、消融、部署与面试级复盘 | 可复现配置与版本锁、训练/评测报告、架构图、成本与边界说明 |

## 后训练框架实战线

这条线从第 11 周开始贯穿后续课程。每个主题都按四层推进：

```text
算法原理与最小张量实现
→ 主流框架中的字段、配置和调用链映射
→ 小模型或 dry-run 的真实执行
→ 指标、失败日志、版本与产物复盘
```

| 层级 | 框架/组件 | 课程角色 | 最小验收证据 |
| --- | --- | --- | --- |
| 入门真实训练 | Hugging Face Transformers、PEFT、TRL | 在单机/小模型上跑通 SFT、LoRA、DPO/GRPO，理解 Dataset、Collator、Trainer、Reward function 与生成样本如何连接 | 固定版本和配置；一次真实 forward/backward/optimizer；loss 与评估记录；保存可加载的 adapter/checkpoint |
| Rollout/推理 | vLLM | 理解离线生成、连续批处理、调度、KV cache、采样参数、log-prob，以及 rollout engine 与训练权重同步的边界 | 用小模型生成多条 completion；保存 token/log-prob/采样配置；画出 request → scheduler → model executor → output 数据流 |
| RL 后训练主框架 | verl | 把已有的 Actor、Rollout、Reference、Critic、Reward、Advantage、DataProto、Worker/Ray、FSDP 和 checkpoint 知识映射到真实框架；完成 PPO/GRPO 最小闭环 | 锁定 commit/镜像与配置；先读通官方示例和关键调用链，再做 smoke test 或最小训练；保留日志、指标和失败分析 |
| 分布式训练基础 | PyTorch FSDP/FSDP2、DeepSpeed ZeRO | 理解参数、梯度、优化器状态的分片与 offload，能解释显存、通信、吞吐和 checkpoint 的取舍 | 用内存估算和配置对照解释为何选择某策略；有资源时执行最小多卡实验，不把静态配置当成已运行 |
| 编排与可观测性 | Ray、TensorBoard/W&B 或等价工具 | 理解 Actor/Learner 资源编排、故障与背压，并读取 reward、KL、clipfrac、吞吐、staleness 等真实曲线 | 一份资源拓扑、一次日志/曲线诊断、可追溯的 policy/reward/data/config 版本 |
| 横向选型 | SGLang、OpenRLHF 等 | 了解与 vLLM/verl 的边界和适用场景；只在目标岗位或项目需要时深入 | 基于同一需求做选型对比，不要求同时复现多个大型框架 |

框架学习不是从头通读仓库。默认按实际数据流选源码入口：配置/入口脚本 → Trainer 主循环
→ Worker/Engine → 数据协议 → loss/optimizer → checkpoint/指标；每次只追一条能运行或能验证的链路。
教学版 `coding/agentic_rl_data/` 继续保留，作用是给真实框架提供可核对的语义基线，而不是替代真实框架。

受本地 RTX 4060 约 8 GB 显存限制，先使用小模型、LoRA/QLoRA、短序列、小 batch、必要时 CPU offload
或仅做 rollout；verl 的多卡/多节点部分优先做源码 Trace、配置 dry-run 和资源估算。需要云 GPU 时，必须先明确
模型、框架 commit/镜像、卡型/卡数、预计时长、费用上限、停止条件和应保存的产物。

第 7 周按固定顺序推进，避免把同一 Runtime 内的规划/执行职责分离误认为多 Agent：

```text
ExecutionPlan / PlanStep 合同
→ Planner、Executor、Verifier、Final Writer 分节点 Prompt
→ create_plan → validate_plan → execute_step → verify_step
→ next / retry / replan / needs_human / fail
→ Step 内有限步 ReAct
→ plan_version、有界 Replan、恢复点与完整 Trace
→ 与纯 ReAct 做成功率、步骤、成本和时延对照
→ 再加入多 Agent 的角色权限、handoff、并发、冲突和全局停止条件
```

Plan-and-Execute 第一轮至少覆盖：正常多步骤执行、非法计划或工具、Step 缺证据、
Observation 使旧计划失效、Replan 超限、高风险操作转人工、中断后恢复，以及计划和步骤
Trace。真实 LLM Prompt 接入后置，先用确定性 Planner/Policy 验证 Runtime 合同。

## 每周节奏

- 每天默认约 2 小时原理/论文和约 2 小时代码/实验/检测；按当天任务微调。
- 每周可选择 1 篇与当前主线直接相关的论文或技术报告；不因 Kimi/DeepSeek 等独立 QA 专题打断主线。
- 周末回顾岗位必备知识和本周代码证据，更新 `STATE.md` 中的薄弱点及复习队列。
- 前 8 周优先使用 API、小模型或模拟环境完成 Agent 闭环；云计算主要留给第 10～12 周的训练实验。租用前明确模型、数据、预计时长、费用上限和验收指标；实验后保存结果并关闭资源。

## 阶段通过标准

- `岗位必备`：能解释原因、读懂关键代码、独立完成至少一个可复现实验或实现，并解释指标异常。
- `需要掌握`：能对比方法、定位常见错误，完成代码级练习或小实验。
- `前沿扩展`：能从一手资料说明动机、核心机制、代价和未核实部分；除非岗位需要，不要求复现整套大模型。

## RL 恢复入口与真实训练门槛

- 已有基础：SFT、PPO、GRPO 原理，old/new/ref log-prob，group advantage，mask、KL、clip，以及 rollout/trainer 和 staleness 的初步概念。
- 当前恢复内容：先学习 trajectory、environment、reward、credit assignment、policy version、staleness 和 on/off-policy 数据准入，并把现有确定性 Agent Trace 转成训练样本合同。
- 教学环境起点：复用现有训练平台 mock、Tool/Plan/Multi-Agent Runtime 和评测合同，因此可以立即开始 Agentic RL 原理与代码，不依赖付费资源。
- 真实训练门槛：在启动真实 GRPO/PPO 或云资源前，仍需稳定任务环境、数据版本、可复现 reward/verifier、轨迹日志、基线评测和明确失败类型；领域 Agent v1 的缺口届时必须补齐或以等价环境证据替代。框架实验必须锁定代码 commit/包版本、配置、模型与数据版本，不能只记录一条启动命令。
- 恢复后的目标：不是孤立复现算法，而是判断失败能否由数据、prompt、工具、检索或训练解决，再对适合训练的问题设计 SFT/奖励/Agentic RL 闭环。

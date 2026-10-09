# 未来 12 周路线（LLM → Agent → Agentic RL）

这是 2026-09-28 调整后的主线，按每周约 28 小时估算：6 天学习和练习，1 天复习、补缺与总结。用户已初步学习 SFT、PPO、GRPO 原理，因此暂缓孤立的 RL 工程练习，先建立 Agent 能力和领域任务，再回到 Agentic RL 完成真实训练闭环。既有 `LEARNING_PLAN.md` 保留为完整主题地图。

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
| 8 | 选择特定领域并构建 Agent v1：需求、数据、工具、知识、约束和验收 | 可运行领域 Agent、基线评测与失败样本库 |
| 9 | Agentic RL 基础：trajectory、environment、reward、credit assignment、on/off-policy | 将领域 Agent 形式化为可训练环境和数据合同 |
| 10 | 领域 Agent 的 SFT/偏好数据：轨迹清洗、成功示范、负例、verifier/reward | 可复用训练数据、奖励设计和离线评估基线 |
| 11 | Agentic RL 训练闭环：恢复 PPO/GRPO、policy version、staleness、权重同步 | 一次最小真实训练、reward/KL/clipfrac 记录和失败分析 |
| 12 | 领域 Agent + LLM 联合迭代、消融、部署与面试级复盘 | 项目 README、训练/评测报告、架构图、成本与边界说明 |

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

## RL 暂停点与恢复条件

- 已有基础：SFT、PPO、GRPO 原理，old/new/ref log-prob，group advantage，mask、KL、clip，以及 rollout/trainer 和 staleness 的初步概念。
- 暂缓内容：权重同步实现、真实 GRPO 训练、混合精度和 LoRA 分布式边界、RL 阶段验收。
- 恢复条件：领域 Agent v1 已经可运行，具备稳定评测集、轨迹日志、工具环境和明确失败类型。
- 恢复后的目标：不是孤立复现算法，而是判断领域失败能否由数据、prompt、工具、检索或训练解决，再对适合训练的问题设计 SFT/奖励/Agentic RL 闭环。

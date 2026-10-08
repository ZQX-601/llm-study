# Agentic LLM 与 Agent Runtime 的能力边界

- 开始及归档日期：2026-09-29；2026-10-08 补充 Agent 模式与分节点 Prompt
- 模式：QA 专题学习
- 原问题：ReAct、Plan、Reasoning、CoT 和工具调用等 Agent 能力是否正在逐渐内嵌到 LLM 中。
- 用户确认：已了解背景知识并明确表示明白。
- 当前证据：用户两次自评掌握；Day23–25 已有 Runtime、Memory 与 Evaluation 教学实现及 34 项测试，但真实 LLM Policy、分节点 System Prompt 和模式路由尚未实现或验收。

## 核心结论

Agent 的认知与决策能力正在逐渐被训练进 LLM，包括多步推理、规划、工具选择、参数生成、根据 observation 调整行为和判断任务是否完成。但完整 Agent 仍需要外部 Runtime 负责循环控制、状态、工具执行、权限、安全、幂等、超时、审计和评估。

```text
Agent System
= Agentic LLM
+ Tool Protocol
+ Runtime Loop
+ State / Memory
+ Tools / Environment
+ Permissions / Guardrails
+ Evaluation / Observability
```

## 概念边界

- CoT/Reasoning：解决模型如何进行多步推理，本身不等于 Agent。
- ReAct：推理、行动和 observation 交替；现代实现通常用内部推理加结构化 tool call，避免解析自然语言 `Thought/Action`。
- Plan-and-Execute：先形成高层计划，再逐项执行并按环境反馈重新规划；可在子任务内部继续使用 ReAct。
- Function Calling：模型提出结构化工具调用意图，应用程序负责校验和真实执行。
- Workflow：执行路径主要由代码预先定义；Agent：模型可以根据环境反馈动态决定过程和工具。

## 2026-10-08 补充：模式、Loop 与 System Prompt

用户追问：Agent 是否需要设计 ReAct、Plan-and-Execute 等模式，定义好 Loop，并使用不同
System Prompt 引导下一步行为。

补充结论：需要设计行为模式和节点 Prompt，但三者职责必须分开：

```text
Orchestration / Runtime Loop
→ 用代码定义状态、转移、预算、停止、权限、重试和错误恢复

Model Policy
→ 根据当前状态提出结构化 Action、Plan 或 FinalAction

Node Prompt
→ 约束模型在当前节点作为 Planner、Executor、Verifier 或 Final Writer 决策
```

不能只靠一份巨大 System Prompt 维持 Loop、安全和业务状态。Prompt 可以要求模型不调用
未提供工具、证据不足时继续查询，但 Runtime 仍必须强制执行工具白名单、Schema、权限、
确认、幂等、预算和停止条件。

### 模式选择

| 任务形态 | 推荐模式 |
| --- | --- |
| 简单问答和格式转换 | Direct LLM |
| 步骤固定、规则明确 | Deterministic Workflow |
| 下一步依赖最新 Observation | ReAct |
| 长任务、多阶段依赖 | Plan-and-Execute |
| 长计划内有动态子任务 | 外层 Plan-and-Execute + 内层有限步 ReAct |
| 高风险写操作 | Workflow + Runtime 硬校验 + 人工确认 |

ReAct 在每轮 Observation 后选择下一步；Plan-and-Execute 先产出带依赖、允许工具、所需
证据和完成条件的结构化计划，再由 Executor 逐步执行，必要时 Replan。混合模式可以用
外层计划管理全局依赖，在单个复杂 Step 内使用有预算上限的 ReAct。

### 分节点 Prompt

推荐拆分：

```text
Global Policy Prompt
├── Planner Prompt：只规划，不声称已经执行
├── ReAct/Executor Prompt：只处理当前步骤和允许工具
├── Verifier Prompt：检查证据与完成条件，不执行副作用动作
└── Final Answer Prompt：只引用已验证 Evidence
```

模型调用输入由稳定规则、当前节点 Prompt、Runtime 裁剪后的 TaskState、本轮 Tool Schema、
必要 Observation/Evidence 和输出 Schema 组成。外部文档与 Tool 输出始终作为不可信数据，
不能升级为 System 指令。

当前仓库的对应关系：

```text
run_agent             → Runtime Loop
TaskState             → 权威运行状态
Policy.decide         → 模型策略接口
ToolAction/FinalAction→ 结构化候选动作
RuleBasedVerifier     → 确定性完成条件
ScriptedPolicy        → 真实 LLM Policy 的教学替身
```

用户明确表示理解。该确认属于 QA 自评，不等于已独立设计模式路由、编写 Prompt 或接入
真实模型。

## 与主线和 Agentic RL 的关系

本专题是当前 LLM 应用接口与 Agent 主线的背景知识。后续领域 Agent 的一次执行会形成 `state → action → observation → ... → reward` 轨迹；Agentic RL 主要训练模型的决策策略，而不是替代外部工具环境和安全控制。

## 一手资料

- [ReAct 论文](https://arxiv.org/abs/2210.03629)
- [Toolformer 论文](https://arxiv.org/abs/2302.04761)
- [OpenAI Function Calling](https://developers.openai.com/api/docs/guides/function-calling)
- [OpenAI Reasoning Models](https://developers.openai.com/api/docs/guides/reasoning)
- [Anthropic：Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
- [OpenAI：A practical guide to building agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)

## 后续验证建议

已有最小 Agent loop 教学实现。后续在多 Agent/工作流阶段使用新领域需求，让用户独立选择
Direct、Workflow、ReAct、Plan-and-Execute 或混合模式，并设计 Planner/Executor/Verifier
合同与 Prompt；真实运行后再判定实践验收。

# Agentic LLM 与 Agent Runtime 的能力边界

- 开始及归档日期：2026-09-29
- 模式：QA 专题学习
- 原问题：ReAct、Plan、Reasoning、CoT 和工具调用等 Agent 能力是否正在逐渐内嵌到 LLM 中。
- 用户确认：已了解背景知识并明确表示明白。
- 当前证据：用户自评掌握；未进行独立问答、代码实现或运行验收。

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

## 与主线和 Agentic RL 的关系

本专题是当前 LLM 应用接口与 Agent 主线的背景知识。后续领域 Agent 的一次执行会形成 `state → action → observation → ... → reward` 轨迹；Agentic RL 主要训练模型的决策策略，而不是替代外部工具环境和安全控制。

## 一手资料

- [ReAct 论文](https://arxiv.org/abs/2210.03629)
- [Toolformer 论文](https://arxiv.org/abs/2302.04761)
- [OpenAI Function Calling](https://developers.openai.com/api/docs/guides/function-calling)
- [OpenAI Reasoning Models](https://developers.openai.com/api/docs/guides/reasoning)
- [Anthropic：Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)

## 后续验证建议

在实现最小 Agent loop 时，让用户从一段代码中区分 model policy、runtime loop、tool executor、state 和 guardrail；这是实践验证，不在本次 QA 中临时追加检测。

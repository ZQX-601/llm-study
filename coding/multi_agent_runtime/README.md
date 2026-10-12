# 多 Agent Runtime：角色、最小权限与 Handoff

本目录是 Day28 第一阶段教学实现，与 `plan_execute_agent/` 平级。它不接真实模型或远程
MCP，先用确定性组件验证多 Agent 的控制面合同。

## 为什么这才是多 Agent

Day26 的 Planner、Executor、Verifier、Final Writer 是同一个 Runtime 内的节点分工，
共享一个权威任务状态，不自动构成多个 Agent。本目录中的 Coordinator 和 Diagnostic：

- 有不同 `agent_id`、角色、Prompt 和 Tool 权限；
- Diagnostic 只收到 Runtime 投影后的局部 Context；
- Handoff 有显式来源、目标、目标主体、Evidence 引用和委派预算；
- Specialist 返回的是结构化结果和 Evidence 引用，不直接覆盖全局状态；
- Result 只能引用本次投影输入或该 Specialist 新产生的 Evidence，猜中其他全局 key
  也会被 Runtime 拒绝；
- Runtime 统一维护全局预算、Tool dispatch、Evidence 准入和完整 Trace。
- Runtime 在 Handoff 前为控制权返回后的 `finalize` 预留 Action，避免 Specialist 花完
  预算后任务无法收尾。

```text
User goal
  → Coordinator 提出 HandoffRequest
  → Runtime 校验 route / subject / Evidence / sensitivity / budget / tools
  → Context projection
  → Diagnostic Agent 在局部预算内调用只读 Tool
  → HandoffResult + Evidence refs
  → Runtime 校验并交回 Coordinator
  → Final answer
```

## 阅读顺序

1. `multi_agent.py`：先看 `AgentSpec`、`HandoffRequest`、`GlobalBudget` 和 Runtime；
2. `demo.py`：观察 Coordinator 与 Diagnostic 的独立 Trace；
3. `prompts.py`：查看各角色的 Prompt 边界；
4. `outcomes.py`：查看依赖失败传播、必需/可选分支与降级成功；
5. `test_multi_agent.py`、`test_outcomes.py`：查看 Runtime 防御与结果聚合验收。

## Handoff 最小合同

Handoff 不是把整段聊天复制给另一个模型，至少要显式声明：

```text
handoff_id
source_agent_id / target_agent_id
objective / subject_id
evidence_keys
requested_tools
delegated_action_budget / delegated_tool_budget
parent_handoff_id / depth
```

目标 Agent 的最终能力是下列集合的交集，而不是 Handoff 请求的并集：

```text
Runtime 注册能力 ∩ Agent 固有权限 ∩ 本次 Handoff requested_tools
```

## 并发 fan-out / fan-in

只有输入独立、无读后写依赖、无副作用冲突的分支才能并发。Runtime 在启动任何分支前
原子检查所有 Handoff、输出 Evidence key 和总预算：

```text
Σ delegated_action_budget + return reserve <= global action budget
Σ delegated_tool_budget                  <= global tool budget
branch count                             <= global handoff budget
```

每个并发请求必须声明 `expected_output_evidence_keys`，重复 key 会在 fan-out 前拒绝。
fan-in 按 `handoff_id` 确定性合并并重新核对输出合同，不使用“最后完成覆盖前者”。

## 循环与全局停止

`handoff_id` 只保证事件唯一，不能识别 Agent 换新 ID 后重复相同委派。Runtime 根据
来源/目标、规范化 objective、subject、Tool、输出合同以及输入 Evidence 的来源和版本
生成语义指纹。默认同一指纹最多执行两次，以允许一次受控重试；第三次在目标 Agent
启动前以 `REPEATED_HANDOFF_ROUTE` 停止。Handoff 还携带 `parent_handoff_id/depth`，
默认深度上限为 4；即使每条边都不重复，第 5 层也会以 `MAX_HANDOFF_DEPTH` 停止。
所有 Agent 同时共享全局 Action、Tool 和 Handoff 预算，因此局部循环最终也不能绕过
全局停止条件。`deadline_at` 是带时区的绝对时间，由 Runtime 在 Handoff、Agent Action
和 Tool dispatch 前检查；即使计数预算仍有剩余，过期后也以
`GLOBAL_DEADLINE_EXCEEDED` 停止。

## 当前边界

- 已实现：顺序 Handoff、只读 fan-out/fan-in、角色最小权限、Context 投影、敏感级别、
  全局/局部预算、输出合同、Tool/Evidence 主体校验、依赖状态传播、必需/可选分支聚合
  和 Trace；
- 尚未实现：语义冲突裁决、并发副作用、持久化恢复、
  真实 Agents SDK/MCP 和副作用 Agent；
- 当前 Coordinator/Diagnostic 都是确定性脚本，Prompt 只作为真实模型接入时的合同草案。

## 运行

```bash
UV_CACHE_DIR=/tmp/llm-agent-uv-cache \
UV_PYTHON_INSTALL_DIR=/tmp/llm-agent-uv-python \
uv run --offline --python 3.13 python coding/multi_agent_runtime/demo.py

UV_CACHE_DIR=/tmp/llm-agent-uv-cache \
UV_PYTHON_INSTALL_DIR=/tmp/llm-agent-uv-python \
uv run --offline --python 3.13 python -m unittest discover \
  -s coding/multi_agent_runtime -p 'test_*.py' -v
```

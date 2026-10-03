# Day23 学习归档｜显式单 Agent Loop、证据合同与失败恢复

- 学习日：Day23（2026-10-03）
- 当前阶段：调整后主线第 4 周——单 Agent 核心循环
- 状态：第一轮原理、代码阅读、教学实现与 mock 测试完成；独立从零实现和真实 MCP/LLM 集成尚未完成
- 衔接来源：[Day22](day22.md) 的训练平台 MCP 工具环境
- 本日方式：到期复习、原理讲解、逐题判断、代码补全、代码审查、教学实现和自动化测试
- 实际学习时长：未记录

## 一、本日目标与岗位意义

今天把 Day20–22 学过的模型调用、Runtime、Tool Calling 和 MCP 串成显式单 Agent
循环：

```text
用户目标
→ TaskState
→ Policy 生成候选 Action
→ Runtime 校验
→ Tool Bridge 调用 MCP/mock Tool
→ Observation
→ Verifier
→ 继续循环或写入 StopReason
```

主题等级：

- `岗位必备`：Policy 与 Runtime 的职责边界、显式状态、工具校验和停止条件；
- `岗位必备`：Observation、Trace、可恢复错误、写操作结果未知和幂等恢复；
- `岗位必备`：FinalAction 仍需证据验证，工具成功不等于用户目标完成；
- `需要掌握`：Tool Bridge 抽象、mock 与真实 MCP 的替换边界；
- `需要掌握`：证据引用、版本和事实/推断/缺口分离；
- `前沿扩展`：真实 LLM Policy、语义 Verifier、持久化 Runtime 和跨进程恢复本日未实现。

对应工作任务：构建一个可调试、可审计、不会把模型候选调用直接当作真实动作的 Agent
Runtime，并能从 Trace 判断错误发生在 Policy、Runtime、工具还是 Verifier。

## 二、到期复习：RAG 检索指标与多证据完整性

用户对三个 Query 的 Top-3 结果正确计算：

```text
Q1：Hit@3=1，Recall@3=1/2，Precision@3=1/3
Q2：Hit@3=0，Recall@3=0，  Precision@3=0
Q3：Hit@3=1，Recall@3=2/3，Precision@3=2/3

宏平均 Hit@3       = 2/3 ≈ 0.67
宏平均 Recall@3    = 7/18 ≈ 0.39
宏平均 Precision@3 = 1/3 ≈ 0.33
```

在“报销额度 + 超额审批规则”需要两份证据、Top-3 只命中额度文档的场景中，用户正确
判断：`Hit@3=1` 仍不代表证据完整，`Recall@3=1/2`，Agent 不能推测未召回的审批规则。

本轮复习通过概念与手算验收；尚未编写和运行检索指标函数。

## 三、单 Agent Loop 的组件与职责

### 3.1 TaskState

`TaskState` 是 Runtime 的权威状态，不是模型自行维护的聊天摘要。本日实现保存：

```text
goal / step / budget_remaining
observations / trace
status / stop_reason / final_answer
pending_idempotency_key
action_counts
required_successful_tools
```

用户正确指出：只把状态放进模型上下文会受截断、新旧状态混杂和注意力误判影响；
Policy 只选择 Action，而执行依赖 Runtime，所以 Runtime 必须维护独立结构化状态。还需
注意模型幻觉、Prompt Injection、并发和重启恢复，均不能依赖模型“记住状态”。

### 3.2 Policy 与 Action

Policy/LLM 只提出候选：

```text
ToolAction(tool_name, arguments, rationale)
FinalAction(answer)
```

模型产生 Action 不等于工具已执行。用户正确判断未知工具应先由 Runtime 阻止，不发送
给 MCP Server；Server 仍需保留不可绕过的最终鉴权。

### 3.3 Runtime、Tool Bridge 与 Observation

Runtime 依次负责：重复动作、允许工具、参数、幂等约束、预算、dispatch、状态更新和
停止条件。Tool Bridge 只暴露：

```text
list_tools() → MCP tools/list
call_tool()  → MCP tools/call
```

因此 `MockToolBridge` 可替换成真实 `MCPToolBridge`，而 `run_agent`、TaskState 和
Verifier 的控制合同保持不变。工具结果被归一化为 Observation 后写入 TaskState 和
Trace，供下一轮 Policy 与 Verifier 使用。

用户正确解释：非法 Action 也应生成 Observation，既为了保留完整 Trace，也为了让
Policy 在下一轮知道错误原因并纠正，而不是静默重复。

### 3.4 Verifier 与停止条件

Verifier 返回：

```text
CONTINUE / COMPLETE / FAIL / NEEDS_HUMAN
```

Runtime 写入明确 StopReason：

```text
COMPLETED
MAX_STEPS
BUDGET_EXHAUSTED
NEEDS_HUMAN
UNRECOVERABLE_FAILURE
REPEATED_ACTION
```

本日纠正了“工具调用成功就等于任务完成”的错误边界。成功 Observation 默认仍然
`CONTINUE`；只有 Policy 提出 FinalAction 且 Verifier 确认完成条件，任务才进入
`COMPLETED`。

## 四、失败恢复与状态迁移

### 4.1 非法 Action

```text
Policy 生成未知工具或非法参数
→ Runtime 在 dispatch 前拒绝
→ INVALID_ACTION Observation
→ Policy 可纠正
→ repeated_action / max_steps 防止无限循环
```

用户正确补全了结果未知状态下的安全校验：只允许
`get_cancel_operation`，且调用参数中的 key 必须等于 TaskState 保存的原 key。

### 4.2 可重试只读错误

```text
UPSTREAM_TIMEOUT + retryable=true
→ Observation
→ Verifier=CONTINUE
→ Policy 可重提查询
→ Runtime 用重复次数、max_steps 和预算限制重试
```

用户正确区分：`retryable=true` 只表示错误类型允许重试，不保证仍有预算。预算为 1
时，第一次工具调用把预算扣到 0；第二次合法 Action 会触发 `BUDGET_EXHAUSTED`，不再
dispatch，也不会生成新的工具 Observation。

### 4.3 RESULT_UNKNOWN 与原幂等键

正确链路：

```text
commit 超时
→ RESULT_UNKNOWN
→ Runtime 先保存 pending_idempotency_key
→ verify_observation 看到安全核对路径后返回 CONTINUE
→ Policy 调用 get_cancel_operation(原 key)
→ PROCESSING / UNKNOWN 有界查询
→ SUCCEEDED 清除 pending key
→ FinalAction
```

用户最初把 `PROCESSING`、`UNKNOWN` 也描述成再次记录 `RESULT_UNKNOWN`，并在
`SUCCEEDED` 后提到重新 commit；经纠正后明确：三者是权威查询返回的远端状态，确认
成功后绝不能再次 commit。若原 Action 根本没有 idempotency key，则不存在安全恢复
路径，Verifier 应返回 `NEEDS_HUMAN`。

用户还主动核对源码，指出 `verify_observation()` 在有 pending key 时确实返回
`CONTINUE`。进一步区分：

```text
verify_observation(RESULT_UNKNOWN + key) → CONTINUE
verify_final(pending key 尚未清除)       → NEEDS_HUMAN
```

Runtime 可以接收并记录候选 FinalAction，但不能认可它、写入 final answer 或宣布完成。

### 4.4 重复动作与最大步数

在 `repeated_action_limit=2`、`max_steps=8` 的场景中，用户最初把第三次相同调用的停止
原因判断为 `MAX_STEPS`；经解释后明确：第三次在 dispatch 前触发
`REPEATED_ACTION`，只消耗前两次工具预算。`MAX_STEPS` 是不同 Action 也不断循环时的
总兜底。

## 五、证据完整性与可审计 FinalAction

用户正确发现教学版 `verify_final()` 的真实缺口：若只检查“pending key 为空且答案非
空”，Policy 可以不调用工具就直接声称 OOM 并被判完成。

据此实现了最小完成证据合同：

```text
TaskState.required_successful_tools
→ 汇总成功 Observation 的 tool_name
→ 必需工具未成功时 FinalAction 返回 CONTINUE
→ 补齐权威证据后才能 COMPLETE
```

用户进一步指出，成功工具调用仍不足以证明答案正确；Verifier 还需检查：

- FinalAction 的错误结论是否与 Observation 的 `latest_error` 等字段一致；
- Policy 的结论由哪些真实证据支持；
- 证据是否完整、属于当前任务和当前版本；
- 是否区分已确认事实、推断和缺失信息。

可审计 FinalAction 后续应包含 `answer`、稳定的 `EvidenceRef`、`version`、
`confirmed_facts`、`inferences` 和 `missing_information`。用户正确说明：引用 step/event
而不复制整份 Observation 可减少 token 与无关信息，并区分安全重试产生的多次结果。
生产实现更适合使用稳定 `event_id`，避免 Trace 合并或重放后 step 变化。

## 六、任务分解与证据边界

本日实现采用反应式隐式分解：Policy 每轮根据最新 Observation 决定下一步；真实系统
也可把 `planned_steps` 显式写进状态并允许重规划。

在“用户说任务失败、权威状态却是 RUNNING 72%”的场景中，用户最初选择直接回答
“它没有失败”；经纠正后能够写出安全结论：当前只确认任务正在运行、错误为空，不能
据此否定历史失败；需要历史 run_id 或运行记录。该题验证了事实、推断与信息缺口的
区分。

## 七、代码产物

在 [训练平台 MCP 教学目录](../coding/training_platform_mcp/README.md) 中新增：

```text
agent_loop.py
  TaskStatus / StopReason
  ToolAction / FinalAction
  ToolResult / Observation / TraceEvent
  TaskState / ToolSpec
  Policy / ToolBridge / Verifier Protocol
  RuleBasedVerifier
  run_agent
  MockToolBridge / ScriptedPolicy

test_agent_loop.py
  单 Agent Loop 的正常、失败、安全和 Trace 测试
```

README 已补充 Agent Runtime 主链、真实 MCP 替换边界和完成证据合同。

实现过程中根据用户的代码审查新增 `required_successful_tools`，并修复一个旧测试预期：
缺参数的非法调用不会 dispatch，但 Policy 纠正参数后的合法调用应实际执行一次。

## 八、运行证据

在 Python 3.13.6 环境实际运行：

```powershell
python -m unittest discover -s . -p "test_*.py" -v
```

结果：13 项全部通过。其中 Agent Loop 新增 10 项：

```text
正常查询后 FinalAction
未知工具在 dispatch 前拒绝
缺参数后纠正并恢复
可重试超时恢复
RESULT_UNKNOWN 按原 key 查询成功
错误 key 被阻止
重复 Action 在第三次执行前停止
预算耗尽时不调用工具
Trace 包含完整阶段
无证据的 FinalAction 先 CONTINUE，补证据后 COMPLETE
```

原有 3 项 `retry_state_machine` 测试继续通过。没有调用付费模型，没有连接真实 MCP
Server 或训练平台，没有产生云资源费用。

## 九、当前能力证据与边界

- 理论理解：能解释显式 Agent Loop、各层职责、错误恢复和停止条件；
- 代码阅读：完成 `run_agent`、Verifier、幂等校验、预算与 Trace 的关键路径追踪；
- 局部代码能力：完成多处条件补全和状态推演，主动发现无证据 FinalAction 的缺口；
- 教学实现：完整 mock Runtime 与 13 项自动化测试已运行通过；
- 独立实现：尚未由用户从零实现完整 Runtime；
- 真实集成：尚未接真实 LLM Policy、MCP SDK、OAuth、持久化状态或训练平台。

因此今天可判定“单 Agent Loop 第一轮原理与教学实现完成”，不能判定“生产 Agent 或
独立实战通过”。

## 十、薄弱点与后续计划

- `RuleBasedVerifier` 只有最小工具证据合同，尚不能语义核对答案与字段；
- FinalAction 仍以自由文本为主，没有落地稳定 EvidenceRef/claim schema；
- `ScriptedPolicy` 不是实际 LLM Policy，没有验证真实模型的结构化 Action 稳定性；
- `MockToolBridge` 尚未替换成官方 MCP SDK 的动态 `tools/list`、`tools/call`；
- TaskState 未持久化，不能验证进程重启、并发和断点恢复；
- 缺少用户独立从零实现、改错和扩展测试的证据。

后续进入 Agent 记忆与上下文工程时，复用今天的 TaskState/Trace，明确区分短期运行
状态、模型上下文和长期记忆；真实 MCP 与语义 Verifier 作为单 Agent 工程缺口继续排队。

## 十一、复习安排

- 1 天后：从新领域 Trace 区分 Policy、Runtime、Tool、Verifier 的最早失败层；
- 3 天后：阅读一段错误 Agent Loop，定位先 dispatch 后校验、无限重试和提前完成；
- 7 天后：独立增加一种 StopReason 或完成证据合同并运行测试；
- Agent 评估阶段：实现结构化 claim/evidence 引用和领域 Verifier。

RAG 指标在本日手算和多证据场景通过，下一次使用实际 Top-K 输出编写并运行指标函数，
补代码证据。

## 十二、话题栈与下次入口

```text
主线：LLM → Agent → 领域 Agent → Agentic RL
已关闭父话题：单 Agent 核心循环第一轮
未解决追问：无
保留工程缺口：真实 LLM/MCP、持久化状态、语义 Verifier、独立从零实现
下一父话题：Agent 记忆与上下文工程
精确入口：区分短期 TaskState、模型 Context 与长期 Memory，分析何时写入、检索、过期和隔离
```

# 训练平台 MCP 教学示例

本目录演示如何把已有训练平台 REST API 包装为 MCP Server，并由 MCP Client
发现和使用 Tools、Resources 与 Prompts。代码尽量接近真实项目结构，但不会连接真实训练平台，也没有实现
生产 OAuth、持久化数据库和人工确认服务。

## 目录

```text
training_platform_mcp/
├── schemas.py          # Tool 输入、输出和领域对象
├── security.py         # 经验证身份、权限、确认和审计的接口边界
├── training_client.py  # 对已有训练平台 REST API 的适配层
├── server.py           # MCP Tools 与 Streamable HTTP 启动入口
├── client_demo.py      # MCP Client 的发现与直接调用示例
├── retry_state_machine.py       # Host 侧 RESULT_UNKNOWN 有界核对
├── test_retry_state_machine.py  # 不连接真实服务的状态机测试
├── agent_loop.py                # 显式单 Agent Runtime、Tool Bridge 与轨迹
├── test_agent_loop.py           # 正常链路、安全边界和失败恢复测试
├── memory_context.py            # 长期记忆生命周期、检索与 Context Builder
└── test_memory_context.py       # 过期、版本、隔离、污染与预算测试
```

## 设计主链

```text
Host/Agent
→ MCP Client
→ tools/list 或 tools/call
→ Training MCP Server
→ Schema、身份、权限、状态和版本校验
→ 现有训练平台 REST API
→ structuredContent / error
→ Host 将 Observation 交给模型分析
```

只读查询可以直接映射为 `get_training_job`。取消任务被拆为两步：

```text
prepare_cancel_training_job
→ 生成有期限、绑定用户和任务版本的 Proposal，不产生副作用
→ Host 展示 Proposal 并获取用户确认
→ commit_cancel_training_job
→ 验证确认、权限、版本与幂等键后才调用 POST /cancel
```

这避免把高风险 REST API 原样暴露给模型。

Server 还提供：

```text
固定 Resource：training://docs/error-codes
Resource Template：training://jobs/{job_id}/logs
Resource Template：training://jobs/{job_id}/config
Prompt：diagnose_training_job(job_id)
```

Resource 和 Prompt 在 Server 侧定义；Host 通过 `list_*` 发现，通过
`read_resource` 或 `get_prompt` 消费。Prompt 只返回消息，不会自行调用 Tool。

## 依赖与运行边界

参考依赖：

```text
mcp[cli] 2.x
pydantic 2.x
httpx
```

官方 SDK 可以从类型标注生成 Tool Schema，并处理 MCP 协议和 Transport；它不会
自动实现训练平台鉴权、租户隔离、业务状态机、幂等、确认和审计。

仅查看 MCP 工具定义时，可在补齐依赖后使用：

```powershell
mcp dev server.py
```

远程部署形态参考：

```powershell
mcp run server.py --transport streamable-http
```

本示例默认入口为 `http://127.0.0.1:8000/mcp`。在补齐以下适配前，不应把它当作
生产服务运行：

- 将 HTTP Bearer Token 验证结果绑定到 `security.bind_principal`；
- 将内存 ProposalStore 换成带事务和唯一约束的持久化存储；
- 接入真实确认服务，并让确认绑定 Proposal 摘要、用户和有效期；
- 配置训练平台 URL 与服务凭据；
- 增加 HTTPS、限流、脱敏、指标、Trace 和审计存储；
- 为超时后的 `RESULT_UNKNOWN` 增加按原幂等键查询操作状态的接口。

本目录现已提供该接口与 Host 状态机的教学实现；生产环境仍需由训练平台实现
幂等记录表、唯一约束、权威操作状态以及 `NOT_FOUND` 的明确语义合同。

## Agent 与 MCP Client 的区别

`client_demo.py` 直接指定工具名，因此只是 MCP Client，不是完整 Agent。完整 Agent
还需要：筛选本轮工具、把 Schema 交给模型、校验模型产生的 Tool Call、执行工具、
把结果作为 Observation 返回模型，并应用最大步数、成本和停止条件。

`agent_loop.py` 补齐了上述 Host/Runtime 控制流，但故意使用确定性的
`ScriptedPolicy` 和 `MockToolBridge`，不调用付费模型或真实训练平台：

```text
TaskState
→ Policy 提出 ToolAction / FinalAction
→ Runtime 检查最小工具集、参数、预算和幂等约束
→ ToolBridge 执行 MCP Tool 或 mock
→ Observation
→ Verifier 决定继续、完成、失败或转人工
→ 结构化 Trace 与明确 StopReason
```

`TaskState.required_successful_tools` 是最小完成证据合同。Verifier 只有在这些工具
至少成功产生过一次 Observation 后才接受 `FinalAction`；这可阻止 Policy 在尚未查询
权威状态时，仅凭一段非空文本宣布任务完成。生产系统还应继续检查返回字段、证据版本
和领域完整性，而不能把“工具调用成功”本身等同于答案正确。

实际运行教学测试：

```powershell
python -m unittest discover -s . -p "test_*.py" -v
```

其中写操作一旦返回 `RESULT_UNKNOWN`，Runtime 只允许调用
`get_cancel_operation` 并强制复用原 `idempotency_key`；模型不能换 key 或重新提交
写操作。`ToolBridge` 是真实 MCP 的替换边界：后续把 `list_tools` 与 `call_tool` 接到
官方 SDK 即可复用同一个 `run_agent`。

## Agent 记忆与 Context Builder

`memory_context.py` 把记忆链路拆为两个阶段：

```text
MemoryStore
→ MemoryWriteGate 区分不保存、TaskState、长期写入和需确认
→ 按 tenant/user/subject_key 保存版本链
→ 按最新版本、过期时间、敏感级别和置信度检索候选记忆
→ ContextBuilder 检查来源与 Prompt Injection
→ 与 goal、权威 TaskState、近期 Trace 一起按优先级和 token 预算组装 Context
```

`TaskState` 的优先级始终高于检索记忆。长期记忆只以“候选记忆”进入 Context，不能
覆盖 Runtime 的预算、确认状态、幂等键或权限结果。教学实现使用简单词项重合代替
embedding/BM25，但租户隔离、版本、过期、软删除和 Context 准入边界可直接映射到真实
存储系统。`GOAL` 和 `AUTHORITATIVE_TASK_STATE` 是 mandatory context；二者无法完整
放入预算时构建过程失败关闭，不能静默丢掉权威状态后继续请求模型。

对应测试覆盖：正确召回、过期记录、只使用最新版本、跨用户隔离、恶意记忆隔离，
预算不足时优先保留目标和权威 TaskState，以及早期偏好掉出滑动窗口后可由按需检索
重新进入 Context。

## Agent 评估与可观测性

`evaluation.py` 直接消费 `TaskState`、`Observation` 和 `TraceEvent`，将最终答案、证据、
安全、轨迹和效率指标分开计算：

```text
EvalCase + 完成后的 TaskState
→ answer_correct / evidence_complete
→ safety_violation / trajectory_valid
→ task_success / case_passed
→ failure_layer
→ EvalSummary
```

`task_success` 表示用户目标被正确、有证据且安全地完成；`case_passed` 表示实际行为符合
测试预期。安全回归样例可以期望 Runtime 拒绝未确认写操作，因此会出现
`task_success=False`、`case_passed=True`。禁止动作以真实 `tool_dispatch` 为准：模型提出
但被 Runtime 拦截不算已经执行。教学版 `failure_layer` 是基于现有 Trace 的确定性启发式，
不能替代记录可信身份、请求主体、版本和 Verifier 依据后的生产根因分析。

`test_evaluation.py` 覆盖正常完成、碰巧猜对但缺证据、禁止工具已 dispatch、轨迹超预算、
安全拒绝和批量指标聚合。效率指标与安全硬门槛分别报告，避免低时延或高答案相似度掩盖
越权行为。`RunTelemetry` 额外记录时延、输入/输出 token 和费用；汇总同时报告全部已观测
任务与成功任务的 p50/p95，并保留遥测覆盖率。未采集到的数据使用 `None`，不能误报成
零时延或零成本。

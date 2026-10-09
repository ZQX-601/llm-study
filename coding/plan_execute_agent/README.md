# Plan-and-Execute + Skill 渐进式加载 + MCP 教学实现

本目录与 `coding/training_platform_mcp/` 平级，使用训练任务诊断场景完整展示：

```text
Planner
→ Runtime 校验 ExecutionPlan
→ 选择当前 PlanStep
→ Skill metadata discovery
→ 激活相关 SKILL.md
→ MCP tools/list + tools/call 获得路由证据
→ 按 Observation 加载相关 reference
→ Step 内有限步 ReAct
→ Plan Verifier
→ next / retry / replan / needs_human / fail
→ Final Writer
```

## 阅读顺序

1. `demo.py`：先看训练任务诊断计划和每个 Step 的动作脚本。
2. `plan_execute.py`：再看数据合同、渐进式 Skill Registry、MCP Client 和 Runtime。
3. `prompts.py`：查看真实 LLM 接入时四个节点的 Prompt 边界。
4. `skills/training-diagnosis/`：查看 Skill 主说明和按需加载的 OOM reference。
5. `plan_evaluation.py`：查看如何从 Trace 重算成功、安全和预算指标。
6. `test_plan_execute.py` 与 `test_plan_evaluation.py`：查看正常、失败、重规划、
   人工确认、恢复和评测测试。

## 三层 Skill 加载

目录内放置两个 Skill，但正常 demo 只激活 `training-diagnosis`：

```text
启动发现：
metadata:report-format
metadata:training-diagnosis

Planner/Executor 选择诊断 Skill 后：
skill:training-diagnosis

执行到 OOM 知识步骤时：
resource:training-diagnosis:references/oom.md
```

因此 `report-format` 的 name/description 会参与路由，但其 SKILL.md 正文不会进入活跃
Context。`training-diagnosis` 的 OOM reference 也不会在激活 Skill 时自动加载，而是在当前
Step 明确需要时才加载。

通用诊断 Skill 可由用户的“训练故障诊断”目标直接路由；OOM 专用
reference 则必须等 MCP 日志提供 OOM Observation 后再加载，避免用预设原因
污染后续诊断。

## Runtime 与模型的职责

教学实现中的 `ScriptedPlanner` 和 `ScriptedStepExecutor` 是真实 LLM 的替身：

| 组件 | 职责 |
| --- | --- |
| Planner | 提出结构化计划 |
| Step Executor | 在当前 Step 内提出 Skill/Tool/完成动作 |
| Plan Verifier | 根据 Evidence 检查完成条件 |
| Final Writer | 只使用已验证 Evidence 生成结果 |
| Runtime | 校验计划、工具和 Skill 权限，维护状态、预算、Replan、恢复点和 Trace |
| MCP Client | 发现并调用外部能力，不负责总流程 |
| Skill | 提供稳定工作流与知识路由，不提供权威动态事实或执行权限 |

安全、权限、确认、资源路径和停止条件都由代码强制，不能只依赖 Prompt。

Step 内同时有 `max_actions_per_step` 和 `max_tool_calls_per_step`：前者防止模型决策空转，
后者独立约束真实的 MCP dispatch 次数。Runtime 还会对 Tool 名和规范化参数
生成 Action fingerprint，默认在第三次相同调用前以 `REPEATED_ACTION` 停止；只有显式
声明 `allows_polling` 的工具可以重复轮询，但仍受 Tool 总预算限制。

## Evidence 合同

`required_evidence` 只表示存在性；关键 Step 使用 `EvidenceRequirement` 同时约束
`key`、`allowed_sources`、`subject_id` 和 `required_version`。Runtime 在 Evidence 入库前
拦截来源不符的数据，Verifier 在 Step 完成前再检查一次。因此 Skill 中的 OOM
手册不能通过把 key 写成 `error_log` 来冒充 MCP 日志。

## 副作用恢复

对有副作用的 MCP Tool，Runtime 先持久化 `DispatchRecord`，再调用工具。若调用超时，
状态转为 `RESULT_UNKNOWN`；恢复时遇到相同 `idempotency_key` 会在再次 dispatch 之前
被拦截。当前教学版在此停到 `NEEDS_HUMAN`，生产版应再调用权威查询工具完成
`SUCCEEDED / FAILED / PROCESSING / UNKNOWN` 对账。

## 评测对接

`plan_evaluation.py` 不盲信 Plan 的 `COMPLETED` 标志，而是从 Evidence 合同和 Trace 重新
计算 `task_success`、`case_passed`、安全违规、轨迹合法性、Action、Tool call、Retry 和
Replan 次数。对比 ReAct 与 Plan-and-Execute 时必须使用同一批任务和可比的总预算。

Replan 时，Runtime 会强制维护 `plan_id`、原始 goal、递增的 `plan_version` 和
`replan_count`，不信任 Planner 返回的审计字段。旧 Evidence 会用新计划合同重新验证：
仍匹配的留在 active evidence，不匹配的移入 `archived_evidence` 并记录
`evidence_invalidated` Trace。

## 教学简化与生产差异

- 当前计划是顺序列表，只允许依赖已经出现的 Step；生产版可扩展为 DAG。
- Skill frontmatter 解析器只支持简单 `key: value`；生产版应采用完整规范解析与签名校验。
- MCP 使用内存 mock；真实实现应替换为官方 SDK 的 `tools/list` 和 `tools/call`。
- Planner/Executor 为确定性脚本；真实模型需要 Structured Output、Prompt 版本和回归评测。
- Evidence 保存在内存；生产版需要持久化、稳定 event ID、租户隔离和恢复事务。
- Final Writer 是确定性字符串；生产版还需 claim/evidence/version 级 Verifier。

## 运行

从本目录执行：

```bash
python demo.py
python -m unittest discover -s . -p 'test_*.py' -v
```

仓库当前测试环境使用 Python 3.13；若系统 `python3` 仍是 3.9，可使用已有 uv 临时环境：

```bash
UV_CACHE_DIR=/tmp/llm-agent-uv-cache \
UV_PYTHON_INSTALL_DIR=/tmp/llm-agent-uv-python \
uv run --offline --python 3.13 python demo.py
```

## Trace 阅读重点

正常运行应依次出现：

```text
skill_discovery
mcp_tools_listed
plan_created
step_started
executor_decision
skill_activated / skill_resource_loaded / mcp_tool_called
observation
step_verified
plan_completed
final_answer
```

发生计划失效时还会出现 `step_invalidated → plan_revised`；恢复执行时出现
`run_resumed`。这些事件让评测系统能够区分 Planner、Executor、Runtime、Skill、MCP
和 Verifier 的最早失败层。

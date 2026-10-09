# Day26 学习归档｜Plan-and-Execute、Skill 渐进式加载与 MCP

- 学习日：Day26（2026-10-09）
- 当前阶段：调整后主线第 7 周——单 Agent 高级编排
- 状态：Plan-and-Execute 第一轮原理、确定性实现、Trace 与自动化测试完成；真实 LLM/MCP、持久化和独立实现未完成
- 衔接来源：[Day25](day25.md) 的 Agent 评估、Trace 与失败归因
- 本日方式：到期复习、架构讲解、代码阅读、逐题判断、教练实现和自动化测试
- 实际学习时长：未记录

## 一、本日目标与岗位意义

今天将 Day23 的单循环 Runtime 升级为可观察、可重规划、可恢复的混合编排：

```text
外层：ExecutionPlan 控制长程结构
内层：当前 PlanStep 使用有预算上限的 ReAct
Runtime：持有权限、状态、预算、版本、恢复点和 Trace
```

主题等级：

- `岗位必备`：Planner、Executor、Verifier、Final Writer 和 Runtime 的责任边界；
- `岗位必备`：Evidence 合同、权限、幂等、副作用恢复和安全转人工；
- `需要掌握`：Skill 的 metadata → SKILL.md → supporting resource 渐进式加载；
- `需要掌握`：Action、Tool、Replan 预算和 ReAct/Plan-and-Execute 选型；
- `前沿扩展`：真实 LLM Structured Output、生产持久化、自动对账和多 Agent 本日未接入。

对应工作决策：短任务不为了“有计划”强行增加 Planner 开销；多步依赖、高风险、
可失效或需恢复的任务才使用 Plan-and-Execute，并且安全与停止条件由 Runtime 代码强制。

## 二、Day25 硬门槛复习

本日先复习 `task_success` 与 `case_passed`。用户在预期非法轨迹样例中最初同时判为
`false`；经纠正后明确：

```text
task_success = false  # 用户目标没有完成
case_passed = true    # Runtime 按安全样例预期拦截或转人工
```

末尾在“缺少确认的退款转人工”新场景中，用户正确判断：

```text
task_success = false
case_passed = true
safety_violation = false
```

危险动作只被模型提出、但在 dispatch 前被 Runtime 拦截，不算已发生安全违规。

## 三、Plan-and-Execute 主链与分节点责任

本日实现和运行的主链为：

```text
Skill metadata discovery
→ MCP tools/list
→ Planner.create_plan
→ Runtime.validate_plan
→ 选择当前 PlanStep
→ Step 内有限 ReAct
→ Verifier
→ next / retry / replan / needs_human / fail
→ Final Writer
```

四个模型节点和 Runtime 的边界：

| 组件 | 本日合同 |
| --- | --- |
| Planner | 只提出计划结构，不调用工具、不制造 Evidence |
| Step Executor | 只处理 Runtime 指定的当前 Step，提出 Skill/Tool/完成动作 |
| Plan Verifier | 根据结构化 Evidence 判断完成、重试、重规划或转人工 |
| Final Writer | 只使用已验证 Evidence 生成最终回答 |
| Runtime | 校验权限、参数、预算和版本，执行 MCP，持有恢复点和 Trace |

用户最终能完整解释：Planner 声明 `allowed_tools` 不等于执行；Executor 提出
`ToolAction` 后，仍须通过 Runtime 的权限、参数、确认和幂等检查才能 dispatch。

## 四、Skill 渐进式加载与 MCP 路由

本日增加的 Skill 知识点使用三层加载：

```text
启动发现：只读 name / description / path
选中 Skill：读取完整 SKILL.md
当前 Step 需要：按需读取 references / scripts / assets
```

用户正确指出一次加载全部 reference 会浪费 token 并污染 Context。进一步通过
OOM/磁盘故障场景明确：通用 `training-diagnosis` Skill 可由用户目标路由；专用
`references/oom.md` 必须在 MCP 日志确认 OOM 后才加载。

初版 demo 曾在日志返回前先加载 OOM reference。该顺序在教学中被发现并修正为：

```text
激活通用诊断 Skill
→ 查询 job_status
→ 查询 error_log
→ Observation 确认 OOM
→ 加载 references/oom.md
→ 生成报告
```

参考的官方一手资料：

- [OpenAI Skills 指南](https://developers.openai.com/api/docs/guides/tools-skills)
- [Skills and the OpenAI Agents SDK](https://developers.openai.com/blog/skills-agents-sdk)

## 五、Evidence 合同与纵深防御

原始的 `required_evidence={"error_log"}` 只检查 key，无法防止 Skill 手册冒充 MCP 日志。
本日增加：

```python
EvidenceRequirement(
    key="error_log",
    allowed_sources=frozenset({"mcp:search_job_logs"}),
    subject_id="train-42",
    required_version=None,
)
```

Verifier 现在同时检查 key、source、subject 和 version；Runtime 在 Evidence 入库前也会拦截
来源不符的写入。Skill 资源还有三道边界：

```text
required_skills    → 当前 Step 能否激活该 Skill
required_resources → 当前 Step 能否读取具体资源
Path.resolve + relative_to → 防止 ../ 和 symlink 逃逸 Skill 目录
```

用户能识别路径边界；在独立补写 `EvidenceRequirement.matches` 时曾将来源集合写成等值
比较，并忽略 `None` 通配语义，经纠正后能正确判断不限制版本的合同。

## 六、Retry、Replan 与版本所有权

核心区分：

```text
RETRY_STEP：路线仍有效，只是当前 Evidence 不完整或工具出现可恢复错误
REPLAN：新 Observation 使目标、主体、依赖或后续路线失效
```

用户最初把 `train-42` 被 `train-43` 替代判为 Retry；经纠正后能区分：同一
`job_id` 的瞬时超时可 Retry，后续步骤仍绑定旧主体则必须 Replan。

`plan_version`、`replan_count`、原始 goal 和 `plan_id` 由 Runtime 强制维护，不信任
Planner 返回的审计字段。Replan 后：

```text
仍满足新合同的 Evidence → 保留复用
不再匹配的 Evidence     → 移入 archived_evidence
Trace                         → 记录 evidence_invalidated 和原因
```

## 七、预算、重复动作与副作用恢复

Step 内有两个独立预算：

```text
max_actions_per_step    → 防模型决策空转和重复 CompleteStep
max_tool_calls_per_step → 限制真实 MCP dispatch
```

Runtime 对 Tool 名和规范化参数生成 fingerprint，默认在第三次相同调用前以
`REPEATED_ACTION` 停止；只有显式 `allows_polling` 的工具能重复轮询，但仍受 Tool 总预算限制。

对高风险副作用，必须先持久化 `DispatchRecord` 再调用 MCP：

```text
dispatch_intent_recorded
→ mcp_tool_called
→ 成功：SUCCEEDED
→ 超时：RESULT_UNKNOWN
→ 恢复时相同 idempotency_key 在再次 dispatch 前被拦截
```

用户正确指出 intent 必须在调用前持久化，恢复时先用原幂等键查询权威状态，
不能盲目使用新 key 重放副作用。当前教学版在 `RESULT_UNKNOWN` 停到 `NEEDS_HUMAN`；
自动查询和对账仍未实现。

## 八、ReAct 对照与评测适配

用户正确选择：

```text
单次状态查询          → 纯 ReAct
多步诊断/证据/恢复/确认 → Plan + Step 内有限 ReAct
```

本日新增 `plan_evaluation.py`，从 `PlanRunState` 和 Trace 重新计算：

```text
answer_correct / evidence_complete / safety_violation / trajectory_valid
task_success / case_passed
actions / tool_calls / completed_steps / retry_decisions / replans
skill_activations / resource_loads
```

用户能正确识别“ReAct 最多 3 次 Tool、Plan 每步 3 次且有 5 步”不是公平对照；
评测应在同一任务集、工具、模型、权限和可比总预算下比较成功、安全、成本和时延。
本日只完成合同和指标适配，没有运行两种模式的成对基准实验。

## 九、代码产物

新建 [Plan-and-Execute 教学目录](../coding/plan_execute_agent/README.md)：

```text
plan_execute.py
  ExecutionPlan / PlanStep / StepStatus / PlanStatus
  Planner / StepExecutor / PlanVerifier / FinalWriter
  SkillRegistry / MockMCPClient
  Evidence / EvidenceRequirement / ArchivedEvidence
  DispatchRecord / DispatchStatus
  run_plan_execute

demo.py
  训练任务诊断的确定性完整 Trace

prompts.py
  Planner / Executor / Verifier / Final Writer 节点 Prompt 边界

plan_evaluation.py
  Plan Trace 评测合同和指标提取

skills/
  training-diagnosis/SKILL.md
  training-diagnosis/references/oom.md
  report-format/SKILL.md

test_plan_execute.py / test_plan_evaluation.py
  正常、非法计划、缺证据、来源冒充、重规划、证据失效、
  Replan 超限、工具预算、重复动作、高风险转人工、RESULT_UNKNOWN 和恢复。
```

## 十、运行证据

使用临时 CPython 3.13 环境运行：

```bash
UV_CACHE_DIR=/tmp/llm-agent-uv-cache \
UV_PYTHON_INSTALL_DIR=/tmp/llm-agent-uv-python \
uv run --offline --python 3.13 python -m unittest discover \
  -s coding/plan_execute_agent -p 'test_*.py' -v
```

最终证据：

```text
Plan-and-Execute / Evaluation  18 项全部通过
原 training_platform_mcp       34 项全部通过
合计                         52 项全部通过
```

demo 实际运行得到 30 个 Trace event，确认日志 Observation 早于 OOM resource 加载。
没有调用付费模型、真实 MCP Server、数据库或云 GPU，没有产生云资源费用。

## 十一、用户检测证据与薄弱点

已能正确判断：

- 缺 Evidence 但路线有效时使用 Retry，主体或后续路线失效时 Replan；
- Tool 超时与 Verifier `RETRY_STEP` 属于不同层；
- Skill 渐进式加载、Observation 先于专用 reference 路由；
- 路径逃逸、工具预算、重复 Action、幂等意图和 Replan 字段的 Runtime 所有权；
- 简单查询用 ReAct，多步、可恢复或高风险任务使用 Plan + 有限 ReAct；
- 预算不同时不能直接比较两种模式的成功率。

需要继续验证：

- 最终高风险综合题中，用户最初判为 Retry、会 dispatch 且已安全违规；换退款场景后能正确得出
  `NEEDS_HUMAN`、无 dispatch、`safety_violation=false`；
- 独立代码题中直接访问了不存在的 `action.confirmation_id`/
  `action.idempotency_key`，并没有按 `spec.has_side_effect` 分支；经纠正后能判断只读工具应 `DISPATCH`；
- 本日新增实现主要由教练完成，用户完成代码阅读和局部判断，尚未独立从零实现或扩展 Runtime。

## 十二、复习安排

- 1 天后：换领域检查 Retry/Replan/NeedsHuman 及“提出危险动作”与“真实 dispatch”的区别；
- 3 天后：阅读一段错误 Plan Runtime，定位版本回退、旧 Evidence 误用和重复副作用；
- 7 天后：独立增加一种 Evidence 合同或自动对账路径并运行测试；
- 21 天后：用领域 Agent 真实任务比较 ReAct 与 Plan-and-Execute 的成功、成本和时延。

## 十三、话题栈与下次入口

```text
主线：LLM → Agent → 领域 Agent → Agentic RL
已关闭父话题：Plan-and-Execute 第一轮
未解决追问：无
保留工程缺口：真实 LLM/MCP、持久化 checkpoint、RESULT_UNKNOWN 自动对账、
              用户独立实现、成对 ReAct 对照实验
下一父话题：先用一个小型成对基准补齐 ReAct/Plan-and-Execute 实测，然后进入多 Agent
精确入口：设计相同任务、工具、权限和总预算，从 Trace 比较成功、Tool、Action、时延和成本
```

# Day25 学习归档｜Agent 评估、可观测性与失败归因

- 学习日：Day25（2026-10-08）
- 当前阶段：调整后主线第 6 周——Agent 评估与可观测性
- 状态：第一轮原理、情境判断、教学实现和自动化测试完成；独立实现与线上遥测接入尚未完成
- 衔接来源：[Day24](day24.md) 的 Agent Memory、Context Builder 与 Trace
- 本日方式：到期复习、指标设计、逐题判断、代码阅读、教练实现和自动化测试
- 实际学习时长：未记录

## 一、本日目标与岗位意义

今天建立 Agent 的最小评测与可观测性合同，重点不是把所有维度压成单一总分，而是回答：

```text
用户目标是否完成？
最终答案是否有完整证据？
执行轨迹是否符合预算和状态机？
是否发生越权、泄漏或绕过确认？
失败最早发生在 Policy、Runtime、Tool、Evidence 还是 FinalAction？
```

主题等级：

- `岗位必备`：任务成功、证据完整、安全硬门槛和轨迹质量；
- `岗位必备`：Trace、失败层归因、离线回归与线上监控边界；
- `需要掌握`：步骤、工具调用、token、费用和 p50/p95；
- `需要掌握`：`task_success` 与 `case_passed` 的区别；
- `前沿扩展`：LLM-as-a-Judge、真实线上采样和统计置信区间本日未实现。

对应工作决策：比较两个 Agent 版本时，先应用安全和正确性硬门槛，再优化成功率、成本和
时延；出现错误时沿 Trace 找最早失败层，而不是只看最终答案或最后抛错组件。

## 二、仓库同步与环境验证

本日先从 GitHub `github/main` 仅快进同步：

```text
a0c7ef1 → c88b3e0
```

同步后补齐 Day22–24 和 `coding/training_platform_mcp/`。系统 `python3` 实际为 3.9.6，
首次运行时因 `TypeAlias` 和 `X | None` 语法兼容性导致两个测试模块导入失败，只有 3 项
状态机测试运行。该问题属于解释器版本不匹配，不是教学代码逻辑失败。

随后使用 uv 在 `/tmp` 下载临时 CPython 3.13.15，运行：

```bash
UV_CACHE_DIR=/tmp/llm-agent-uv-cache \
UV_PYTHON_INSTALL_DIR=/tmp/llm-agent-uv-python \
uv run --python 3.13 python -m unittest discover \
  -s coding/training_platform_mcp -p 'test_*.py' -v
```

同步后的原有 25 项测试全部通过。

## 三、Memory/Context 与权威事实复习

在训练任务诊断场景中，用户正确选择：当前 Tool Observation 和当前用户已确认的最新偏好
可进入 Context；旧版本、跨用户记忆、外部 Prompt Injection 和过期动态事实应被过滤。

用户最初认为 Verifier 提供当前任务的权威事实；经退款新场景纠正后明确：

```text
Tool / External State → 提供带时间和版本的权威事实
Runtime               → 校验来源、结构、权限和状态
Verifier              → 判断最终结论是否被证据支持
```

`RUNNING + latest_error=null` 只能支持当前快照未返回错误，不能否定历史故障。

## 四、评估指标与硬门槛

本日将评估拆为六层：

| 层次 | 典型指标 |
| --- | --- |
| 任务结果 | task success、约束满足率 |
| 最终答案 | correctness、证据忠实度和完整性 |
| 轨迹 | 步数、重复动作、恢复成功率 |
| Tool/Runtime | 工具选择、参数、权限和状态迁移 |
| 安全 | 越权、跨用户泄漏、绕过确认和数据外发 |
| 效率 | token、费用、工具次数、p50/p95 时延 |

核心硬门槛：

```python
task_success = (
    state.status is TaskStatus.SUCCEEDED
    and answer_correct
    and evidence_complete
    and not safety_violation
    and trajectory_valid
)
```

用户正确选择安全违规为零、成功率仍超过门槛的 Agent B，而不是成功率略高但出现 3 次
安全违规的 Agent A。

### 4.1 本日纠正的安全边界

- 没有权威查询但碰巧猜对当前状态，默认属于证据与轨迹问题，不一定是安全违规；
- 绕过用户确认直接执行退款，即使金额正确且操作成功，仍是安全违规；
- 读取另一用户的工资，即使数值碰巧相同，仍是机密性违规；
- 未经授权把内部日志发送给第三方 MCP Server，属于跨边界数据外发，应由 Host/Runtime
  在 dispatch 前执行来源、敏感级别、目标信任、DLP 和确认检查。

用户在前两个安全场景中曾把安全范围理解得过窄，经第三方 MCP 数据外发新场景后能够
正确识别安全违规与 Runtime 控制点。

## 五、`task_success` 与 `case_passed`

两者不能混为一个指标：

```text
task_success：用户目标被正确、有证据且安全地完成
case_passed：实际行为符合该回归样例的预期
```

安全测试可以期望未确认写操作进入 `NEEDS_HUMAN`：任务没有完成，但安全回归通过。

用户两次补写 `task_success` 时使用了非 Python 语法 `answer_correct = true`，并遗漏
`trajectory_valid`。本日由教练在实现中写入正确表达式并用测试验证，因此该点记录为
“已解释并有教学代码证据，用户尚未独立补全通过”。

## 六、失败层归因

失败层应定位到最早产生错误的位置，不能把成功工作的防线判成故障源：

```text
Policy 生成 arguments={}
→ Runtime 正确拒绝缺少 job_id
→ 最早问题属于 Policy，而不是 Runtime
```

用户最初未直接回答该归因；在新的主体错配场景中，能够正确指出
`get_training_job(job_id="J42")` 返回 `J99` 的最早失败层为 Tool。

Tool Schema 版本变化场景还明确区分：如果 Policy 已收到 v2 仍生成旧字段，属于 Policy；
如果 Host 缓存 v1 或未把 v2 传给 Policy，则根因在 Host/Schema 传播。Trace 应记录
Schema version/hash，离线评测应保留版本化快照和破坏性变更合同测试。

## 七、代码产物

在 [训练平台 MCP 教学目录](../coding/training_platform_mcp/README.md) 新增：

```text
evaluation.py
  EvalCase / EvalResult / EvalSummary
  RunTelemetry
  FailureLayer
  evaluate_state
  aggregate_results

test_evaluation.py
  正常完成
  猜对但缺证据
  禁止工具真实 dispatch
  轨迹超预算
  安全拒绝与 case_passed
  指标聚合
  遥测覆盖、p50/p95、token 与费用
  非法负数遥测
```

教学实现由教练完成，用户参与指标判断和代码阅读，但没有独立从零实现评测模块。

## 八、遥测缺失与效率指标

缺失遥测必须使用 `None`，不能伪装成零时延或零费用。在 `100ms、200ms、1000ms` 三条
已观测记录和一条缺失记录中，用户正确指出把缺失写成 `0ms` 会把它当作正常返回并计入
统计。进一步得到：

```text
正确 coverage = 3/4 = 75%，p50 = 200ms
错误补 0      = 4/4 = 100%，p50 = 100ms
```

全部任务和成功任务的时延需要分别报告，避免快速失败样本让延迟指标虚假变好。

## 九、运行证据

最终在临时 CPython 3.13.15 环境运行全部测试：

```text
Agent Loop                 10 项
Agent Evaluation            9 项
Memory/Context             12 项
RESULT_UNKNOWN 状态机       3 项
合计                       34 项全部通过
```

没有调用付费模型、真实 MCP Server、向量数据库或线上监控系统，没有产生云资源费用。

## 十、当前证据与边界

- 理论理解：能区分答案、证据、任务、安全和效率指标；安全范围经多场景纠正；
- 代码阅读：能在主体错配 Trace 中定位 Tool 层，理解 Schema 版本传播；
- 教学实现：最小评测合同、失败归因和遥测聚合已实现，34 项测试通过；
- 独立实现：尚未独立写出完整 `task_success` 条件或扩展评测模块；
- 真实集成：没有真实 LLM judge、线上 trace backend、真实 token/cost 采集或告警。

因此可判定“Agent 评估与可观测性第一轮完成”，不能判定“独立评测平台或生产监控完成”。

## 十一、复习安排

- 1 天后：换领域区分 answer、evidence、task success、case passed 和 safety；
- 3 天后：根据一批实际结果手算遥测覆盖率、p50/p95，并检查缺失值处理；
- 7 天后：独立增加 Schema 版本一致性或一种 FailureLayer 评测与测试；
- 21 天后：结合领域 Agent 的真实轨迹建立回归集、错误分类和版本对照。

## 十二、话题栈与下次入口

```text
主线：LLM → Agent → 领域 Agent → Agentic RL
已关闭父话题：Agent 评估与可观测性第一轮
未解决追问：无
保留工程缺口：独立评测扩展、claim/evidence Verifier、线上遥测、真实 LLM/MCP
下一父话题：多 Agent 与工作流编排
精确入口：从单 Agent 不能稳定承担的角色边界、handoff、并发、冲突与停止条件开始
```

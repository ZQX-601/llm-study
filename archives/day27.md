# Day27 学习归档｜ReAct vs Plan-and-Execute 成对基准

- 学习日：Day27（2026-10-10）
- 当前阶段：调整后主线第 7 周——单 Agent 高级编排收尾
- 状态：公平成对基准、全局预算、RAG Top-K 指标、结果分析与逐题检测完成
- 衔接来源：[Day26](day26.md) 的 Plan-and-Execute 第一轮与公平对照缺口
- 本日方式：到期复习、教练实现、自动化测试、实际运行、结果分析和逐题检测
- 实际学习时长：未记录

## 一、本日目标与岗位意义

本日不直接进入多 Agent，先补齐第 7 周单 Agent 高级编排的实测证据：

```text
相同任务 + 相同 Tool 语义 + 相同权限 + 相同整次任务预算
→ 分别运行纯 ReAct 与 Plan-and-Execute
→ 分开比较硬门槛和效率指标
→ 给出按任务结构选模式的路由规则
```

主题等级：

- `岗位必备`：公平实验、总预算、成功/Evidence/安全硬门槛和 Trace 归因；
- `岗位必备`：Retry、Replan、NeedsHuman 与恢复点；
- `需要掌握`：RAG Hit/Recall/Precision 和多证据完整性；
- `需要掌握`：按任务复杂度、目标稳定性、风险和恢复要求选择 Agent 模式。

对应工作决策：不能因为 Plan-and-Execute 更结构化就默认用于所有任务，也不能因为
ReAct 更轻就让复杂任务只依赖模型上下文；先过正确性和安全硬门槛，再比较 Action、
Tool、时延和成本。

## 二、Plan 状态迁移复习

本日换成订单履约场景复习 Day26 状态迁移：

1. 高风险 `reroute_order` 带幂等键但缺少 `confirmation_id`：用户正确判断
   `PlanStatus.NEEDS_HUMAN`、Tool 未 dispatch、在 Runtime 确认门拦截；当前 Step
   为 `BLOCKED`，停止原因为 `CONFIRMATION_REQUIRED`。
2. `get_order(order-300)` 瞬时超时且目标、主体、合同均未变化：用户正确选择
   `RETRY_STEP`，并指出超过重试次数后不能继续无限重试。
3. 权威 Tool 返回 `order-400` 已被 `order-401` 替换：用户正确选择 `REPLAN`，并指出
   Evidence 应选择性失效。进一步明确：不直接删除旧 Evidence；匹配新合同的继续活跃，
   绑定旧主体的动态事实移入 `archived_evidence` 供审计。

## 三、公平成对基准设计

新增 [成对基准实现](../coding/plan_execute_agent/paired_benchmark.py)，核心类型为：

```text
BenchmarkCase
BenchmarkBudget
ModeRunMetrics
PairedResult
```

四类确定性订单任务：

```text
单工具查询
多步延迟诊断
可恢复超时
目标订单变化
```

两种模式共享 `OrderFulfillmentBackend` 的权威 Tool 语义；每次运行获得独立但同构的
后端实例，避免一方的重试计数污染另一方。ReAct 复用已有 `run_agent`，Plan 模式复用
Day26 的 `run_plan_execute`、确定性 Planner/Executor、Evidence Verifier 和 Final Writer。

默认公平总预算为：

```text
max_actions = 8
max_tool_calls = 4
max_replans = 1
repeated_action_limit = 2
```

## 四、全局预算修正

Day26 Runtime 只有每 Step 的 Action/Tool 上限，多 Step 任务可能隐式获得更多总预算。
本日为 `run_plan_execute` 增加：

```python
max_total_actions
max_total_tool_calls
```

Runtime 从完整 Trace 计算已经消耗的总量，因此新增 Step、Replan 或中断恢复都不能重置
预算。新增测试验证：第二个 Step 会在超过总 Action 或 Tool 上限时停止，Tool 上限检查
发生在真实 dispatch 之前。

## 五、实际成对结果

详细记录见 [paired_benchmark_results.md](../coding/plan_execute_agent/paired_benchmark_results.md)。
一次 Python 3.13 本地运行得到：

| 场景 | ReAct Action/Tool | Plan Action/Tool/Replan | 两边硬门槛 |
| --- | ---: | ---: | --- |
| 单工具查询 | 2 / 1 | 2 / 1 / 0 | 全部通过 |
| 多步延迟诊断 | 3 / 2 | 4 / 2 / 0 | 全部通过 |
| 可恢复超时 | 3 / 2 | 3 / 2 / 0 | 全部通过 |
| 目标订单变化 | 3 / 2 | 3 / 2 / 1 | 全部通过 |

硬门槛包括任务成功、Evidence 完整、轨迹合法和无安全违规。Token、输出 Token 和费用
未接入，保持 `None`，没有伪造为 0。时延是单次本地教学脚本值，只验证采集链路，
不作为真实性能显著性结论。

结果说明：

- 简单查询中两边 Action/Tool 相同，但 Plan 有建计划、校验和 Trace 管理的固定时延；
- 多步任务中 Plan 为显式 Step 完成验证多用一个 Action；
- 可恢复超时中两边都真实调用两次 Tool；
- 目标变化中两边都能完成，但只有 Plan 显式记录旧 Step 失效、Evidence 重验证和 Replan。

## 六、模式路由结论

本日形成的可执行路由规则：

```text
单步 + 低风险 + 目标稳定                 → 优先 ReAct
多步骤 + 有依赖/恢复点 + 目标可能变化     → 优先 Plan-and-Execute
局部探索或瞬时 Tool 失败                  → Step 内有限 ReAct
高风险副作用                              → 无论模式，都由 Runtime 做确认/权限/幂等
```

用户正确选择复杂订单任务使用 Plan-and-Execute + Step 内有限 ReAct，但最初解释为
“ReAct 做不到目标切换”。经纠正后明确：ReAct 也可能切换目标，Plan 的优势在于显式
Replan、Evidence 重验证、恢复点和失败归因，而不是能力上的绝对可达性。

恢复题中，用户最初只回答模式名称，没有列出结构化状态；讲解后能根据
`StepStatus.COMPLETED`、`current_step_index=1` 和仍有效的 Evidence 判断从 `step-2`
继续。生产恢复还需保存 Evidence 来源/主体/版本、DispatchRecord、幂等键和 Trace。

## 七、RAG Top-K 指标代码证据

新增 [RAG 指标目录](../coding/rag_metrics/README.md)，只接收检索后的文档 ID 排序，
不实现 BM25、Embedding 或向量库。实现并测试：

```text
Hit@K
Recall@K
Precision@K
EvidenceComplete@K
macro_average
```

本日约定 `Precision@K` 分母固定为请求的 `K`，少返回的槽位按未命中处理；
`required_evidence_groups` 支持一组内的替代证据，但每个必需组都必须在 Top-K 中被覆盖。

用户在首个手算中把 `EvidenceComplete@2` 写成 1，忽略了第二组证据只出现在第 3 位；
经纠正后在替代证据场景正确给出四项均为 1。代码检测中最初误认为集合 `&` 不是交集，
经说明后明确 `&` 等价于 `set.intersection`，真正错误是 Precision 分母写成
`len(top_k)`；随后正确算出少返回场景的 `Precision@4 = 0.25`。

## 八、代码与运行证据

本日新增或修改：

```text
coding/plan_execute_agent/paired_benchmark.py
coding/plan_execute_agent/test_paired_benchmark.py
coding/plan_execute_agent/paired_benchmark_results.md
coding/plan_execute_agent/plan_execute.py
coding/plan_execute_agent/test_plan_execute.py
coding/rag_metrics/metrics.py
coding/rag_metrics/test_metrics.py
coding/rag_metrics/README.md
```

使用仓库约定的离线 Python 3.13 环境实际运行：

```text
旧 training_platform_mcp 测试  34 项通过
Day26 原有 Plan 测试          18 项通过
Day27 新增基准/预算测试         7 项通过
Day27 新增 RAG 指标测试         5 项通过
合计                         64 项通过
```

即旧有 52 项保持通过，新增 12 项全部通过。没有调用真实 LLM、真实 MCP、付费 API、
数据库或云 GPU。DeepSeek 独立 QA 的 PyTorch 测试不属于本次验收范围。

## 九、当前证据边界与薄弱点

已形成的证据：

- 能在新领域正确判断 NeedsHuman、Retry、Replan 和恢复入口；
- 有实际运行的公平总预算、四组成对 Trace 与 RAG 指标代码；
- 能区分模式能力可达性与结构化可审计性的差异；
- 能在纠正后正确解释 Precision 分母和多证据完整性。

仍需继续验证：

- 本日代码仍主要由教练实现，用户尚未独立新增 Benchmark Case 或 Runtime gate；
- 基准是确定性 mock，未观测真实模型 Token、费用、时延分布和随机成功率；
- 未接真实 MCP、持久化 checkpoint、自动副作用对账或 claim/evidence Verifier；
- `&` 集合语义、恢复字段和“ReAct 也能切换目标”均在提示后纠正，应换场景复查。

## 十、复习安排

- 1 天后（2026-10-11）：给定新的成对结果，区分硬门槛、效率指标和缺失遥测；
- 3 天后（2026-10-13）：阅读错误恢复代码，检查 Step/Evidence/Dispatch 是否可安全复用；
- 7 天后（2026-10-17）：独立新增一个 Benchmark Case 或总预算 gate 测试；
- 21 天后（2026-10-31）：在领域 Agent 上使用真实调用遥测复测模式路由。

## 十一、话题栈与下次入口

```text
主线：LLM → Agent → 领域 Agent → Agentic RL
已关闭父话题：Plan-and-Execute 第一阶段及纯 ReAct 成对基准
未解决追问：无
保留工程缺口：真实 LLM/MCP、持久化、线上遥测、自动对账、用户独立扩展
下一父话题：多 Agent
精确入口：适用边界 → 角色与最小权限 → handoff 合同 → 并发与结果合并
          → 冲突处理 → 全局预算与停止条件
```

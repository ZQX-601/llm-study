# Day28 学习归档｜多 Agent Runtime、Handoff 与全局控制

- 学习日：Day28（2026-10-12）
- 当前阶段：调整后主线第 7 周——多 Agent 第一轮收尾
- 状态：角色边界、Handoff、并发合并、全局停止、失败传播与结果聚合完成第一轮
- 衔接来源：[Day27](day27.md) 的 Plan-and-Execute 第一阶段与多 Agent 入口
- 本日方式：原理讲解、确定性教学实现、自动化测试、逐题检测与错误纠正
- 实际学习时长：未记录

## 一、本日目标与岗位意义

本日区分同一 Runtime 内的 Planner/Executor 节点与真正的多 Agent，并实现一个可解释的
层级式协作 Runtime：Coordinator 不把完整上下文和全部权限交给 Specialist，而是通过
结构化 Handoff 委派最小任务，由 Runtime 统一控制权限、预算、证据、并发和停止条件。

主题等级：

- `岗位必备`：Agent 身份与最小权限、Handoff 合同、Context 投影和 Runtime 硬约束；
- `岗位必备`：全局预算、Deadline、循环/深度限制、安全失败与结果准入；
- `需要掌握`：fan-out/fan-in、失败隔离、依赖传播和必需/可选分支；
- `需要掌握`：迟到结果、幂等副作用和一次有效业务效果的工程边界。

对应工作决策：多 Agent 不是“多写几个 Prompt”，而是多个独立身份、权限、局部状态和
结果合同之间的受控协作。安全和一致性必须由 Runtime 硬代码保证，不能依赖模型自觉。

## 二、适用边界与组织模式

本日明确：实际系统中常见的是单 Agent + Tools，复杂任务再使用层级式多 Agent；
“一个 Coordinator 管理多个子 Agent”本身就是多 Agent。与完全对等的 peer-to-peer
相比，层级式结构更容易统一预算、权限、Trace、停止条件和最终结果。

只有任务确实存在以下特征时才值得拆分 Agent：

```text
不同角色需要不同权限或上下文
并行子任务能带来真实收益
子任务有独立验收合同
需要故障隔离、恢复或专门能力
```

如果只是同一状态机内的 Planner、Executor、Verifier 节点，不应仅因 Prompt 不同就称为
多个 Agent。

## 三、角色、Handoff 与 Context 投影

新增 [多 Agent 教学目录](../coding/multi_agent_runtime/README.md)，核心合同包括：

```text
AgentSpec
HandoffRequest / HandoffInput / HandoffResult
Evidence
GlobalBudget / MultiAgentState
MultiAgentRuntime
```

每个 Agent 有独立 `agent_id`、角色、允许的 Tool、可委派目标和可接收敏感级别。
Handoff 显式声明来源、目标、objective、subject、Evidence 引用、请求 Tool、局部预算、
输出 Evidence key、父 Handoff 和深度。Runtime 在目标 Agent 启动前原子校验：

```text
合法路由
requested_tools ⊆ target.allowed_tools
Evidence key / subject / sensitivity / version
Context projection
局部预算不能放大全局预算
输出合同不冲突
```

用户正确判断未授权 Tool 应在 Runtime 直接拒绝。对于一个 Handoff 同时引用合法
`job_status`、错误主体 Evidence 和受限 Evidence 的场景，用户最初倾向于只传合法项；
经纠正后明确当前合同采用原子 fail-closed：整个 Handoff 拒绝，不能静默改变 Agent
看到的任务语义。只引用合法 `job_status` 时才允许目标 Agent 运行。

## 四、Evidence 与结果准入

Specialist 不能直接覆盖全局状态，只能返回结构化 `HandoffResult` 和 Evidence 引用。
Runtime 只允许结果引用：

```text
本次 Handoff 投影给它的 Evidence
∪
该 Specialist 本次通过合法 Tool 新产生的 Evidence
```

即使 Agent 猜中了其他全局 Evidence key，也会以
`HANDOFF_RESULT_REFERENCES_UNKNOWN_EVIDENCE` 拒绝。用户正确判断伪造结果不能进入全局
状态。成功结果必须完整满足输出合同；失败结果可以没有预期产物，但必须保留失败 Trace，
不能伪造成成功。

## 五、全局预算、循环、深度与 Deadline

所有 Agent 共享同一份全局 Action、Tool、Handoff 预算，局部委派不能按 Agent 或 Step
重复获得总预算。Runtime 还为控制权返回后的 Coordinator `finalize` 预留 Action。

用户正确判断：当 Specialist 消耗预算导致 Coordinator 无法 finalize 时，任务不能成功；
预留应由 Runtime 强制，而不是写在容易被忽略或注入污染的 Prompt 中。

停止合同包括：

- 语义 Handoff 指纹限制，不能靠更换 `handoff_id` 绕过重复委派；
- `parent_handoff_id/depth` 与 `max_handoff_depth`；
- 带时区的绝对 `deadline_at`；
- Action、Tool 和 Handoff 全局计数。

用户正确指出：版本变化代表新状态，可以形成新委派；超过最大深度必须停止；Deadline
过期时，即使计数预算仍有余额也不能继续 dispatch。

## 六、并发 fan-out/fan-in 与失败隔离

只读、相互独立且无副作用冲突的分支可以并行。Runtime 在任何分支启动前对整个批次做
原子 preflight；若一个分支请求未授权 Tool，合法分支也不会抢先执行，且不扣执行预算。

fan-in 按 Handoff ID 确定性合并，不允许“最后完成覆盖前者”。同一个 claim 在相同版本
出现不同值时，应标记冲突，通过权威重查或人工裁决，不能任意择一。

对于部分失败：

- 成功分支的已验证 Evidence 保留；
- 只重试失败且可重试的独立分支；
- 未运行的依赖分支不能错误标成自身 `FAILED`；
- 上游状态未知时为 `BLOCKED`；上游确定失败且不再恢复时为
  `SKIPPED_UPSTREAM_FAILED`。

用户最初认为一个并发分支失败后应重跑全部分支，后续正确判断只重试 Metric Agent；
在依赖传播题中曾把未运行的 C 标成 `FAIL` 并删除兄弟分支 A 的 Evidence，经新场景纠正后
能正确保留独立 Evidence，并把下游标为 `SKIPPED_UPSTREAM_FAILED`。

## 七、安全取消、迟到结果与副作用幂等

安全失败采用 fail-fast：停止新的 Tool/Handoff dispatch，对正在运行的分支发出取消信号。
底层调用无法真正取消时，迟到结果只能进入审计 Trace，不能合并到当前 Evidence。

本日设计了结果身份：

```text
execution_epoch
dispatch_id
attempt_id
```

`pending/failed` 只是状态，不能区分旧调用的迟到结果和新重试结果；接收结果时必须核对
当前 epoch、活跃 dispatch/attempt 和分支状态。用户最初只提出 `pending` 与失败标记，
经纠正后能用 `execution_epoch` 判断新旧执行；进一步明确同一 epoch 内仍需更细粒度的
`dispatch_id/attempt_id`。

副作用 Tool 超时后不能直接生成新 ID 重试。正确流程是先按原 `idempotency_key` 查询
权威操作记录：已完成则复用结果，明确未执行才可用同一 key 重试，状态不确定则
`NEEDS_HUMAN`。Runtime 保存 key 本身不能保证 exactly-once；Tool 服务端还需要：

```text
idempotency_key 唯一约束
相同 key + 相同参数返回原结果
相同 key + 不同参数拒绝
业务变更与幂等账本处于同一原子事务
```

用户能识别重复扣减风险和人工介入边界；最初提出的“执行前查询”仍存在扣减后、记账前
崩溃窗口，经说明后正确补充并发锁/乐观锁。最终边界是数据库唯一约束或原子 CAS/事务，
分布式锁只能辅助，不能替代最后的一致性约束。

## 八、必需/可选分支与任务结果

新增 [结果聚合实现](../coding/multi_agent_runtime/outcomes.py)，把分支结果与全局任务结果
分开：

```python
task_success = (
    required_branches_succeeded
    and required_evidence_complete
    and not safety_violation
)
degraded_success = task_success and optional_branch_failed
```

可选增强分支失败时，核心任务仍可成功，但 Trace 和最终回答必须显式说明缺失产物，不能
引用不存在的 Evidence。若分支仍为 `RUNNING/BLOCKED`，不能提前结算成降级成功。

用户独立补全了 `task_success` 条件。对于可选 Agent 的未授权请求被 Runtime 成功拦截，
用户正确区分：`policy_violation_attempted=true`、`safety_violation=false`、
`task_success=true`、`case_passed=true`；最初遗漏 `degraded_success=true`，经解释后在非法
Tool 真正 dispatch 的新场景中正确判断 `safety_violation=true`、任务与测试均失败。

## 九、代码与运行证据

本日新增：

```text
coding/multi_agent_runtime/__init__.py
coding/multi_agent_runtime/multi_agent.py
coding/multi_agent_runtime/outcomes.py
coding/multi_agent_runtime/prompts.py
coding/multi_agent_runtime/demo.py
coding/multi_agent_runtime/test_multi_agent.py
coding/multi_agent_runtime/test_outcomes.py
coding/multi_agent_runtime/README.md
```

实际运行结果：

```text
training_platform_mcp   34 项通过
plan_execute_agent      25 项通过
rag_metrics              5 项通过
multi_agent_runtime     21 项通过
合计                    85 项通过
```

没有调用真实 LLM、真实 MCP、数据库、付费 API 或云 GPU。

## 十、当前证据边界与薄弱点

已形成的证据：

- 能解释层级式多 Agent 与单 Runtime 多节点的边界；
- 能在新场景判断权限、Context、预算、循环、Deadline、冲突和失败传播；
- 有可运行的确定性多 Agent Runtime、结果聚合和 21 项专项测试；
- 旧 64 项回归保持通过。

仍需保留的缺口：

- 代码仍主要由教练实现，用户尚未独立实现 Handoff Runtime；
- `dispatch_id/attempt_id/execution_epoch`、真实取消和迟到结果隔离目前是设计结论，尚未
  全部进入 Runtime；
- 语义冲突裁决、持久化 checkpoint、副作用 Agent 和幂等账本尚未实现；
- 未接真实 Agents SDK、MCP、线上遥测或真实模型；
- 领域 Agent v1 尚未构建，用户选择暂缓该阶段，不能将其记为完成。

## 十一、复习安排

- 1 天后：给定 Handoff Trace，检查权限、主体、版本和原子 preflight；
- 3 天后：阅读迟到结果代码，区分 epoch、dispatch 和 attempt；
- 7 天后：独立补一个冲突、取消或失败传播测试；
- 21 天后：在 Agentic RL 环境中检查多 Agent trajectory 的 credit assignment。

## 十二、话题栈与下次入口

用户明确调整路线：暂缓第 8 周真实领域 Agent v1，下一次直接进入 Agentic RL。为避免
把环境能力写成已完成，先复用现有训练平台 mock、Agent Runtime、Trace 和评测合同，
建立可训练环境与轨迹 Schema；真实 MCP/RAG/领域 Agent 仍保留为工程缺口。

```text
已关闭父话题：多 Agent 第一轮
下一父话题：Agentic RL
精确入口：trajectory/environment/reward/credit assignment
        → policy version 与 staleness
        → on-policy/off-policy 数据准入
        → 将现有 Agent Trace 转成训练样本合同
```

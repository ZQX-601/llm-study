# Day24 学习归档｜Agent 记忆、Context Builder 与长期记忆治理

- 学习日：Day24（2026-10-06）
- 当前阶段：调整后主线第 5 周——Agent 记忆与上下文工程
- 状态：第一轮原理、代码阅读、教学实现、逐题判断和自动化测试完成；真实向量检索、持久化数据库和独立从零实现尚未完成
- 衔接来源：[Day23](day23.md) 的 `TaskState`、Trace、Runtime 与完成证据合同
- 本日方式：到期复习、原理讲解、公开方案对照、代码审查、教学实现和自动化测试
- 实际学习时长：未记录

## 一、本日目标与岗位意义

本日从显式 Agent Loop 进入记忆与上下文工程，建立如下边界：

```text
TaskState：Runtime 保存的当前任务权威状态
Context：本轮临时提供给模型的有限信息视图
Long-term Memory：跨任务持久化、按需检索且仍需验证的候选信息
```

主题等级：

- `岗位必备`：状态、Context 与长期记忆的信任边界；
- `岗位必备`：写入门控、版本、过期、隔离、污染与 token 预算；
- `需要掌握`：滑动窗口、摘要和检索式记忆的组合；
- `需要掌握`：Memory 检索与 RAG 的共性和差异；
- `前沿扩展`：真实 embedding/BM25/reranker、持久化 Memory Store 和在线评测本日未实现。

对应工作任务：构建不会把临时状态永久化、不会跨用户泄漏、不会让陈旧或恶意记忆覆盖权威状态的 Agent Context Builder。

## 二、到期复习：高风险 commit 与 FinalAction 证据

### 2.1 Proposal、确认与 commit

在“删除长期闲置管理员”的新场景中，用户正确指出 Runtime 不能只检查参数合法和工具在列表中就执行高风险写操作，还需要权限、状态和用户确认。经补充后明确：原始目标确实包含删除，危险点不是目标偏离，而是用户尚未确认具体对象 `U17`。

正确链路：

```text
Policy 提出 DeleteProposal
→ Runtime 校验参数、权限和业务状态
→ Verifier 检查候选对象证据
→ Runtime 保存 AWAITING_CONFIRMATION
→ 用户确认绑定 proposal_id、对象、操作、参数和版本
→ commit 前重新检查权限、有效期与 record_version
→ Tool 服务端再次鉴权并执行
```

用户正确判断：用户确认 `U17@version=12` 后，若 commit 前版本变为 13，必须终止本次 commit，不能自动把旧确认迁移给新版本。这是 TOCTOU 边界。

### 2.2 FinalAction 的证据范围

在服务器状态场景中，用户最初认为 `RUNNING + latest_error=null` 可支持“没有发生故障”；经纠正后明确：当前观察只能支持“当前正在运行、当前查询未返回错误”，不能否定历史故障，也不能排除延迟、部分实例或依赖异常。两条 Observation 版本不一致时还需统一 snapshot/version。

最终证据结构应区分：

```text
confirmed_facts
inferences
missing_information
EvidenceRef
```

## 三、记忆与 Context 的核心边界

### 3.1 三层数据

- `TaskState` 保存当前目标、预算、确认状态、幂等键和执行进度，必须结构化且权威；
- `Context` 由 Goal、TaskState、近期 Trace、工具定义和候选记忆临时组装，受 token 预算限制；
- `Long-term Memory` 保存跨任务可能有用的偏好、经历和流程，但检索后仍需检查作用域、版本、时效、来源和权限。

用户正确完成酒店场景分类：当前城市、天数和“本次不要立即付款”属于 TaskState；“平时偏好安静的无烟房”适合作为用户确认的长期语义记忆。

### 3.2 三类长期记忆

```text
Episodic：某次具体经历
Semantic：相对稳定的事实或偏好
Procedural：可复用流程或操作规则
```

Procedural Memory 风险最高，外部网页或模型推断不能直接写成操作规则。

### 3.3 长期记忆与权威业务事实

用户正确判断：用户偏好的温度单位、常用出发城市可作为长期记忆；当天气温和航班余票必须在使用时查询权威系统。Memory 可以保存稳定引用和偏好，不能把过去的动态 Observation 当作当前真值。

## 四、长期记忆管理的主要难点

本日系统讨论了：

- Write Gate：判断不保存、仅 TaskState、长期写入或需要确认；
- 表示：结构化核心字段与自然语言内容并存；
- 作用域：organization、tenant、user、project、task 和 session 隔离；
- 版本与冲突：使用稳定 `subject_key`，默认只召回最新有效版本；
- 时间：TTL、时间衰减、来源版本和使用前刷新；
- 检索：相关性不等于任务效用，需要 query 改写、混合召回和重排；
- 污染：外部不可信文本不能成为 Procedural Memory；
- 摘要：保留原始 EvidenceRef，防止多轮摘要把推断变事实；
- 删除：权威记录、索引、缓存和摘要需要一致遗忘；
- 评估：写入、检索、准入、隔离和最终任务收益需分层衡量。

核心结论：长期记忆最难的不是 `query → embedding → top-k`，而是“该不该写、属于谁、何时失效、冲突相信谁、召回后能否进入 Context”。

## 五、OpenAI 与 Anthropic 公开方案对照

本日核对两家官方公开资料，不推断未公开的产品内部实现：

- OpenAI 将会话状态、Context 压缩和检索拆为 Conversations/Responses、Compaction、Vector Store/File Search；公开检索接口支持 attributes filter、查询改写、ranker、分数阈值和结果数量控制。应用仍负责用户隔离、版本、过期、冲突和准入。
- Anthropic Memory Tool 是客户端执行的持久文件操作接口，Claude 请求查看、创建、更新或删除 `/memories` 下的文件，实际存储由应用控制；可与 Context Editing/Compaction 组合。大量知识或记忆可采用 Contextual Retrieval：embedding + BM25 + 合并/去重 + reranker。

用户正确归纳：Memory Retrieval 与 RAG 的检索部分非常相似；主要差异在于 Memory 还需要动态写入、持续更新、用户隔离、冲突消解和持久化安全治理。

## 六、Context Engineering

本日对比四层策略：

```text
TaskState：当前权威状态
Sliding Window：最近原始消息和 Trace
Structured Summary：较早阶段的事实、决策和未解决事项
Retrieval：跨会话相关长期记忆
```

用户在预算情境中正确选择保留：

```text
当前 payment_status=AWAITING_CONFIRMATION
用户本轮“选择酒店但不要付款”
最新酒店查询结果
```

长期房型偏好按需检索；旧价格从活跃 Context 删除但原 Trace 可保留审计；无关餐厅聊天无需为了保留而摘要。

用户进一步正确判断：完整 TaskState 太大时，应由 Runtime 按字段白名单确定性裁剪，不能让 LLM 自由摘要权威状态。

## 七、代码产物

在 [训练平台 MCP 教学目录](../coding/training_platform_mcp/README.md) 新增：

```text
memory_context.py
  MemoryType / Sensitivity / SourceTrust
  MemoryCandidate / MemoryWriteGate / WriteDecision
  MemoryRecord / subject_key / 版本链
  InMemoryMemoryStore / RetrievalQuery / RetrievedMemory
  SlidingWindowHistoryBuilder
  ContextItem / ContextBundle / ContextBuilder
  mandatory context / ContextBudgetError
  Prompt Injection 与 Procedural Memory 来源准入

test_memory_context.py
  正确召回、过期、最新版本、跨用户隔离
  恶意记忆隔离、mandatory token 预算
  Write Gate 四类决策
  滑动窗口遗漏后由长期记忆找回
```

代码审查中发现并修复一个真实缺口：原实现只是给 Goal 和 TaskState 高优先级，预算极小时仍可能静默丢弃 TaskState。修正后两者为 mandatory context；无法放入时抛出 `ContextBudgetError`，失败关闭。

随后新增稳定 `subject_key` 和 `MemoryWriteGate`。用户正确判断“一次英文简历使用英文”只进入 TaskState，不应覆盖“默认中文”的长期偏好；只有“以后默认英文”才创建新版本。

用户还完成了无序版本选择的核心代码补全：

```python
if current is None or record.version > current.version:
    latest[key] = record
```

## 八、运行证据

在 Python 3.13.6 环境运行：

```powershell
python -m unittest discover -s coding\training_platform_mcp -p "test_*.py" -v
```

结果：25 项全部通过，其中：

- Agent Loop 测试 10 项；
- Memory/Context 测试 12 项；
- RESULT_UNKNOWN 状态机测试 3 项。

没有调用付费模型、真实 MCP Server、向量数据库或外部业务系统，没有产生云资源费用。

## 九、用户回答与纠正记录

- 正确判断版本变化后必须终止旧 commit；
- 初次把当前 `RUNNING + 无最新错误` 扩大为“没有发生故障”，经解释后能识别版本不一致和历史证据缺口；
- 正确区分过期、跨用户和恶意记忆分别在哪一层过滤；
- 正确选择 mandatory Context，并判断 TaskState 应确定性裁剪；
- 正确区分稳定偏好与动态业务事实；
- 在综合 Trace 中识别跨用户召回、旧版本进入 Context 和 commit 确认缺口；经补充明确最新版本收敛首先属于 Store/Retriever，Context Builder 保留防御性检查。

## 十、当前能力证据与边界

- 理论理解：能解释 TaskState、Context、长期记忆、RAG 和权威业务事实的边界；
- 代码阅读：能追踪 Write Gate、版本链、Retriever、Context Builder 和预算选择；
- 局部代码能力：完成版本选择条件，能定位 mandatory context 和跨层职责缺口；
- 教学实现：Memory/Context 模块及 12 项专项测试已实现，目录内 25 项测试全部通过；
- 独立实现：尚未由用户从零实现完整 Memory 模块；
- 真实集成：尚未接 embedding、BM25、reranker、持久化数据库或真实 LLM。

因此可判定“Agent 记忆与上下文工程第一轮完成”，不能判定“生产长期记忆系统或独立实战通过”。

## 十一、复习安排

- 1 天后：给定混合 Trace，区分 TaskState、短期 Context、长期记忆和权威业务查询；
- 3 天后：阅读错误 Memory Pipeline，定位跨用户、陈旧版本、污染和 mandatory 丢失；
- 7 天后：独立增加一种 WriteDecision、冲突处理或删除同步测试；
- Agent 评估阶段：建立 Write Precision/Recall、Retrieval Recall@K、stale/admission/leakage rate 与任务成功率对照。

## 十二、话题栈与下次入口

```text
主线：LLM → Agent → 领域 Agent → Agentic RL
已关闭父话题：Agent 记忆与上下文工程第一轮
未解决追问：无
保留工程缺口：真实混合检索、持久化、独立实现、在线评测与真实 LLM
下一父话题：Agent 评估与可观测性
精确入口：从任务成功率、轨迹质量、成本、时延和安全边界定义最小评测合同
```

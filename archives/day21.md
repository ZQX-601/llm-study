# Day21 学习归档｜RAG 基础、检索评估与 Agent 边界

- 学习日：Day21（2026-09-30）
- 当前阶段：调整后主线第 2 周——RAG 基础
- 状态：RAG 基础理论完成第一轮；工程实现、真实运行和独立验收尚未开始
- 衔接来源：[Day20](day20.md) 的 RAG/Tool/Runtime 职责边界停点
- 本日方式：系统讲解、架构分析和逐题工程情境判断；未接入模型、向量库或真实业务工具
- 实际学习时长：未记录

## 一、本日目标与岗位意义

今天建立领域 Agent 中 RAG 的完整认知模型，重点不是记住某个框架 API，而是能够判断：知识如何进入索引、如何召回和重排、如何组装成可引用 Context、如何与 Tool 和 Runtime 分工，以及回答错误时应该定位哪一层。

本日主题等级：

- `岗位必备`：Chunking、Sparse/Dense/Hybrid Retrieval、Reranker、Context Builder、检索评估和分层故障定位。
- `岗位必备`：RAG、Tool、Runtime、Skill、MCP 的职责和信任边界。
- `需要掌握`：Query Rewrite、Query Decomposition、多 Query、证据完整性检查和受控 Agentic RAG。
- `前沿扩展`：Graph RAG、多跳检索、多模态 RAG 等只做后续方向说明，本日未展开。

## 二、到期复习：权限、状态与审批

使用生产部署 Agent 场景复习 Day20：模型提出结构合法的生产部署请求，但任务处于 `WAITING_APPROVAL`，用户只有测试环境权限，且没有生产审批记录。

用户能够指出：

- 权限校验不通过；
- 当前状态不是允许部署的状态；
- 生产审批记录不存在；
- 因此 Runtime 不能执行工具调用。

该证据支持对 Schema、权限、状态和审批边界的工程判断；尚无代码实现。

## 三、RAG 主链与职责边界

### 3.1 基本定义

RAG 是 Retrieval-Augmented Generation：不通过修改模型权重记忆全部知识，而是在回答前检索外部证据并加入模型上下文。

```text
离线：文档 → 解析 → Chunk → Embedding/倒排统计 → Index

在线：Query → Query 处理 → Retriever → Reranker
     → Context Builder → LLM 生成 → 引用/结果校验
```

结合推荐/搜索经验，可将 Query 和 Chunk 类比为请求和候选 Item，将召回与重排类比为多路召回和精排；RAG 额外要求证据完整、可引用、有权限、版本有效，并能支持最终答案。

### 3.2 RAG、Tool 与 Runtime

- RAG：查询政策、手册、FAQ 等非结构化知识；
- Tool：查询当前订单、余额、库存、审批记录等权威业务事实；
- Runtime：执行权限、状态、金额、幂等和业务规则等确定性判断；
- LLM：理解用户、提出计划并根据证据解释结果。

用户在物流补偿场景中正确指出：RAG 提供规则，订单工具查询真实订单状态和金额，Runtime 综合规则与事实完成校验。补充更正：Runtime 不只是“调用 RAG 和工具”，还要把结构化政策与事实转成可复现的确定性判断。

## 四、Chunking 与文档解析

### 4.1 最小语义单元

Chunk 不能只追求固定长度。业务规则应尽量保留：

```text
适用对象 + 触发条件 + 结论 + 例外 + 上限
```

用户正确判断：将完整物流补偿规则保留在一个 Chunk，比把“会员订单”“延迟条件”“补偿金额”“禁止重复申请”机械拆成多个短 Chunk 更适合作为基线；后者会丢失条件与结论之间的关系。

需要区分：

- 一个 Chunk 通常不跨越不同章节；
- 一个很长的章节可以拆成多个 Chunk；
- 一条完整规则、FAQ 问答、表格行组或函数等语义原子不应轻易拆开。

### 4.2 自动切分与人工规则

实际工程采用“人工定义少量通用规则 + 程序批量执行 + 人工抽样评估”，不是人工逐篇切文档。

```text
文档解析器识别标题、段落、列表、表格
→ 建立结构化元素
→ Chunker 按硬边界和长度合并/拆分
→ 添加标题路径、版本、页码和权限 metadata
```

讨论了 Docling、Unstructured、LangChain/LlamaIndex 等工具的定位：工具负责识别常见结构并执行切分；业务方负责定义不能拆开的语义边界。FAQ 格式统一时可以用简单解析器，格式混乱时可由模型辅助抽取 Schema，但高风险结果仍需校验和抽样。

相关一手资料：

- [Docling Chunking](https://docling-project.github.io/docling/concepts/chunking/)
- [Unstructured Chunking Strategies](https://docs.unstructured.io/api-reference/partition/chunking)
- [LlamaIndex Semantic Splitter](https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/semantic_splitter/)

## 五、召回、重排与检索评估

### 5.1 Sparse、Dense 与 Hybrid Retrieval

- BM25/Sparse Retrieval 擅长订单号、错误码、版本号和精确业务术语；
- Dense Retrieval 使用 `query_embedding:[D]` 和 `chunk_embeddings:[N,D]` 计算语义相似度，擅长同义表达；
- Hybrid Retrieval 合并两路候选，可通过 RRF 等方法融合排名，避免直接比较不同尺度的原始分数。

用户正确判断：查询包含 `E1027`、`payment-service` 和 `v3.2.1` 时，应优先依赖 BM25 的精确匹配，Dense 只作为补充。

### 5.2 Retriever 与 Reranker

Retriever 强调在大量 Chunk 中快速提高 Recall；Cross-Encoder Reranker 对少量候选联合编码 Query 和 Chunk，提高最终 Precision。Reranker 只负责相关性排序，不能替代 Runtime 做业务判断。

首次回答中，用户将“普通会员退款规则”排在“高级会员物流补偿规则”之前，并认为 Reranker 可以直接决定补偿资格。经纠正后，用户在新的差旅报销场景中正确给出相关性排序，并明确最终报销结论由 Runtime 判定。

### 5.3 Hit、Recall 与 Precision

单条 Query：

```text
Hit@K      = Top-K 是否至少命中一个相关 Chunk，取值为 0 或 1
Recall@K   = 找回的相关 Chunk 数 / 全部相关 Chunk 数
Precision@K= 找回的相关 Chunk 数 / K
```

用户最初两次将“命中数量”或 `Precision` 当作 `Hit`；经不同场景纠正后，能够正确计算三个 Query 的 `Hit@3` 为 `1、0、1`，整体平均为 `2/3≈0.67`。

RAG 需要分层评估：文档解析、Chunk 完整性、Retriever Recall、Reranker 排序、Context 选择、答案正确性、忠实性、引用准确率和 Runtime 决策准确率不能混成一个指标。

## 六、Context Builder 与基于证据的生成

Retriever Top-K 不能直接全部拼入 Prompt。Context Builder 需要：

- 在检索前或最早阶段执行权限过滤；
- 按版本、生效时间和状态排除过期政策；
- 去重、合并父子 Chunk、补充标题和例外条款；
- 在 Token Budget 内选择完整证据；
- 为每个证据保留稳定 ID、来源、版本和章节；
- 将外部文档明确标记为数据，而不是指令；
- 证据不足或冲突时拒绝强行下结论。

在新旧物流规则场景中，用户正确选择 2026 年现行规则 B、C，排除相似度更高但已过期的 2024 规则 A。

推荐让 LLM 返回 `answer、evidence_ids、claims、missing_facts、proposed_action` 等结构化字段，但 Schema 合法仍不代表语义正确。用户最初认为事实不足时应“重新检索”；经补充后明确：缺知识才再次 RAG，缺订单、用户或权限等权威事实应调用 Tool/API 并由 Runtime 校验。

## 七、自然语言政策与可执行规则

高风险业务不能直接让 LLM依据自然语言规则决定退款或补偿。成熟系统通常维护双层表示：

```text
自然语言政策：供 RAG 检索、引用和解释
结构化规则：供 Runtime 确定性执行
```

离线规则编译流程可由模型辅助：

```text
政策文档 → 规则段落识别 → Schema 抽取草稿
→ 字段/操作符校验 → 边界与冲突测试 → 人工审核高风险规则
→ 按 policy_id + version 发布
```

运行时通过稳定的 `policy_id、version、section_id` 关联自然语言 Chunk 与结构化规则，而不是临时对两段文字做模糊匹配。数值、权限、状态和幂等属于硬规则；“情节严重”等模糊判断可由 LLM 提议，但应受金额上限和人工审批约束。

## 八、Query 处理与受控 Agentic RAG

用户自然语言常包含口语、指代和多个子问题，因此需要：

- Query Rewrite：映射为检索术语，但不能篡改用户事实；
- Query Decomposition：拆分触发条件、重复申请、金额上限等证据需求；
- Metadata Filter：提取地区、年份、会员类型、版本和权限；
- Multi-Query：使用不同表达提高 Recall；
- 证据自检：仅针对缺失证据定向再次检索。

用户在 `ORD-888` 高级会员重复补偿场景中正确拆出三类 RAG 查询，并提出查询订单详情、延迟天数、订单金额和补偿历史。补充项：会员身份、订单归属和当前用户权限同样不能只相信用户陈述。

受控 Agentic RAG 循环为：

```text
理解问题 → 制定检索计划 → 检索/过滤 → 检查证据完整性
  ├─ 缺规则：定向再次检索
  ├─ 缺业务事实：调用 Tool
  ├─ 证据冲突：停止并上报
  └─ 证据完整：交给 Runtime 判断
```

检索次数、最大 Query 数、每路 Top-K、总 Token 和总成本必须由 Runtime 限制。

## 九、分层故障定位与 Trace

RAG 错误需要沿以下层次定位：

```text
Parser → Chunker → Index → Query Processor → Retriever
→ Reranker → Context Builder → LLM → Runtime
```

用户在“正确 Chunk 已进入 Retriever Top-50，但被 Reranker 排除出 Top-5”的场景中首先定位到 Reranker，但同时提出修改 Chunker。经纠正后明确：证据已证明 Parser、Chunker、Index 和 Retriever 基本完成职责，应先检查 Reranker 输入、截断、模型与排序实现。随后在“PDF 原文存在、解析 Markdown 已丢失”的新场景中，用户正确定位 Parser，并指出下游暂时不应调整。

建议每次请求保留 Retrieval Trace：原始/改写 Query、filters、候选 Chunk、各阶段排名与分数、淘汰原因、最终证据、缺失证据、工具事实和 Runtime 决策。

## 十、RAG 与 Skill、Tool、MCP 的边界

严格分类：

| 内容 | 更准确的机制 | 是否属于经典 RAG |
| --- | --- | --- |
| 政策、手册、FAQ | Knowledge Retrieval | 是 |
| `SKILL.md` 与工作流说明 | Skill/Instruction Retrieval | 形式相似，通常不称经典 RAG |
| Tool/MCP 工具定义 | Tool Discovery/Tool Routing | 否 |
| 订单、库存等实时状态 | Tool Calling/State Retrieval | 否 |
| 编译后的硬规则 | Runtime | 否 |

MCP 是能力发现与调用协议，不等于 RAG；调用 `search_policy_documents` 可能实现 RAG，调用 `get_order` 是状态查询，调用 `issue_refund` 是有副作用的动作。

参考：[OpenAI Skills 文档](https://developers.openai.com/plugins/concepts/skills)、[OpenAI Tools 文档](https://developers.openai.com/api/docs/guides/tools)。

## 十一、代码与运行证据

- 本日未运行文档解析、Embedding、向量数据库、Hybrid Retrieval、Reranker 或模型 API。
- 曾准备一个标准库 BM25 教学草稿，但用户明确选择跳过 BM25 实现；该草稿未运行、未保留，不计为代码证据。
- 本日真实证据为原理讲解和工程情境问答，不等于可复现 RAG 项目或独立验收。

因此当前准确状态为：

```text
RAG 基础理论：完成第一轮
RAG 工程实现：尚未开始
RAG 独立验收：尚未通过
高级 RAG：留到领域项目按需学习
```

## 十二、薄弱点与后续计划

- `Hit@K` 曾与命中数量、Precision 混淆，已在新场景中纠正，仍需后续代码评估验证。
- Reranker 与 Runtime 的职责首次回答有误，已通过新场景纠正。
- 对“缺知识再检索、缺事实调工具”的路由已经理解，但没有实现。
- 没有任何 RAG 工程运行证据；按照当前路线，先继续 Tool/MCP 与 Agent Runtime，构建领域 Agent 时再接回真实 RAG 实操。

下一主线：Tool Calling、MCP、参数校验、权限、超时、重试和幂等，然后进入显式 Agent loop。

## 十三、复习安排

- 1 天后：给定一次失败 Trace，区分 Parser、Retriever、Reranker、Context Builder 和 LLM 故障。
- 3 天后：使用新的多证据场景计算 Hit、Recall、Precision，并判断证据是否完整。
- 7 天后：为一个领域 Agent 设计 RAG/Tool/Runtime 的输入输出合同。
- 领域 Agent v1 阶段：运行真实文档解析、检索、重排和评估，形成独立工程证据。

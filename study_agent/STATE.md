# 学习状态

更新日期：2026-10-06。此文件是以后每日续学的入口；已发生内容的详细证据见 `CURRENT_PROGRESS.md`、`archives/` 和 `qa/`。下列“已学习”不自动等于“独立实战通过”。

## 当前目标与资源

- 目标：约 3 个月内准备大模型算法岗位，覆盖预训练、微调、后训练与 Agent。
- 时间：每天至少 4 小时；原理和代码各约一半。
- 本地：NVIDIA GeForce RTX 4060，约 8 GB 显存；默认 Python 3.13.6 未安装 PyTorch（仅指检测时的默认环境）。
- 云计算：可接受每月 300–1000 元，尚未选服务商或创建训练环境。
- 使用方式：先通过仓库内的对话式教练学习，后期再扩展独立应用。

## 模式与归档确认

- “开始今天的学习计划吧”等进入计划学习：先读取本状态、12 周路线及最近 Day/QA 存档，结合到期复习推进主线；仅在用户明确说“将今天的学习内容存档”时创建下一个 `archives/dayNN.md`。
- 直接提问进入 QA：详细讲解原理、架构/数据流图、作用、源码和应用，允许连续追问；自然收尾时询问是否掌握。用户确认后才把具体主题正式存入 `qa/YYYY-MM-DD-具体主题.md`，并标为“用户自评掌握，待复习验证”。
- 已确认的计划学习归档进入主线复习；已确认的 QA 归档保留在独立专题待验收清单中，只有用户主动选择该专题时才复习，不自动插入主线。没有确认或检测证据时，不推定掌握。不会仅因经过一天或用户暂时没有回复就自动建档、定时推送。
- 2026-09-26 用户明确表示 DeepSeek V4.1 的 SWA 学习完成并要求存档；已聚合为 [SWA、稀疏注意力与整体架构](../qa/2026-09-26-deepseek-v4-1-swa-and-architecture.md)。尚未进行独立验收，后续按复习队列验证。
- 2026-09-26 用户明确表示 CSA2 学习完成；已归档为 [CSA2 与 Compressor](../qa/2026-09-26-deepseek-v4-1-csa2.md)。教学实现和四项行为测试已完成，独立验收仍待后续复习。
- 2026-09-26 用户明确表示 mHC 部分已经明白；已归档为 [mHC 与 Single-Pass mHC](../qa/2026-09-26-deepseek-v4-1-mhc.md)。包含两份可运行教学代码，独立验收待后续复习。
- 2026-09-29 用户确认已理解 ReAct、Plan-and-Execute、Reasoning、Tool Calling，以及模型决策能力与外部 Agent Runtime 的边界；已归档为 [Agentic LLM 与 Agent Runtime](../qa/2026-09-29-agentic-llm-and-agent-runtime.md)，待后续最小 Agent loop 实践验证。
- 2026-09-29 用户完成 CSA2 mode 复习，并通过连续追问理解 CED 的预训练/prefill/decode 边界、`H20` 的双重用途及 Decoder SWA Bounded Replay 的截断误差；已归档为 [CED 与 Decoder SWA Bounded Replay](../qa/2026-09-29-deepseek-v4-1-ced-and-bounded-replay.md)，待代码验证。

## 主线位置

- SFT、PPO、GRPO 的基本原理已完成初步学习；Day19 已讲 rollout/trainer 分离与 policy staleness，但检查题和工程闭环尚未完成。
- 2026-09-28 用户调整主线：暂缓权重同步、真实 GRPO 训练等 RL 工程实操，先学习 LLM 应用基础、RAG、工具调用、MCP、Agent 架构、记忆、评估和领域 Agent 落地。
- 2026-09-29 Day20 已完成 LLM 应用接口与 Agent Runtime 的第一轮概念学习：推理参数、Structured Output、Function Calling、多工具路由、Schema 设计、Context/State、Agent loop、失败恢复、幂等、Guardrail 和 Prompt Injection；RAG 仅完成概览导入，末尾职责边界题尚未回答。
- 2026-09-30 Day21 已完成 RAG 基础理论第一轮：文档解析与 Chunking、Sparse/Dense/Hybrid Retrieval、Reranker、检索指标、Context Builder、Query 处理、自然语言政策与结构化规则、分层故障定位，以及 RAG/Skill/Tool/MCP/Runtime 边界。用户选择跳过 BM25 实现；尚无真实 RAG 工程或运行证据。
- 2026-10-01 Day22 已完成 Tool Calling 与 MCP 第一轮：Tool Schema、动态发现、Host/Runtime/Client/Server、stdio/Streamable HTTP、Tools/Resources/Prompts、权限分层、超时/幂等/RESULT_UNKNOWN、Prompt Injection 和跨 Server 数据流。已产出训练平台 MCP 静态教学代码和 3 项 Host 状态机测试；未安装运行真实 MCP SDK 或连接训练平台。
- 2026-10-03 Day23 已完成单 Agent 核心循环第一轮：TaskState、Policy、ToolAction/FinalAction、Runtime 校验、Tool Bridge、Observation、Verifier、Trace、预算与停止条件。已产出可运行 mock Runtime；13 项测试通过。用户完成指标手算、代码追踪和局部补全，并发现“无证据 FinalAction 会被提前接受”的缺口；尚未独立从零实现或接入真实 LLM/MCP。
- 2026-10-06 Day24 已完成 Agent 记忆与上下文工程第一轮：TaskState/Context/Long-term Memory、Write Gate、版本/过期/隔离、滑动窗口/摘要/按需检索、污染防护和分层评估。已产出 Memory/Context 教学实现和 12 项专项测试；目录内共 25 项测试通过。尚未接真实 embedding/BM25/reranker、持久化数据库或真实 LLM。
- Agent 基础和领域项目完成后，再回到 Agentic RL：围绕真实 Agent 的轨迹、环境、奖励、信用分配和训练闭环，结合 SFT/PPO/GRPO 实现特定领域 Agent 与 LLM。
- Kimi K3、DeepSeek V4.1 等前沿架构属于独立 QA 专题线；除非用户主动选择专题复习或验收，不插入主线计划学习。

## 能力证据

| 主题 | 原理/问答 | 代码阅读或教学实现 | 独立实验 | 下一步 |
| --- | --- | --- | --- | --- |
| Transformer/预训练基础 | 已学习 | 有历史讲解 | 未记录真实预训练 | 小模型训练与数据流程复现 |
| SFT、LoRA、QLoRA、DPO | 已学习 | 有配置与原理分析 | 未记录可复现微调结果 | 建立数据、基线、微调和评估实验 |
| PPO/GRPO | 基本原理已初步掌握 | `grpo_end_to_end.py` 为教学实现 | 未运行真实模型训练 | Agent 项目完成后，以 Agentic RL 场景恢复实操 |
| 分布式训练与 rollout | Day18–19 已讲解 | 主要为设计与静态分析 | 未运行分布式训练 | 暂缓；后续随 Agentic RL 训练闭环恢复 |
| LLM 应用接口与 Agent Runtime | Day20–24 已完成接口、状态、工具边界、显式 Agent Loop、记忆与 Context 第一轮 | `agent_loop.py` 与 `memory_context.py` 覆盖 Runtime、Trace、Write Gate、Memory Store 和 Context Builder | 教学目录共 25 项测试通过；用户完成多次代码判断与局部补全，未独立从零实现 | 后续接真实 LLM/MCP、持久化与语义 Verifier |
| Kimi K3 架构 | KDA/MLA/LatentMoE/AttnRes 已专题学习 | 有参考演示代码，其中部分自检已运行 | 未训练 K3 | 核对官方一手资料，完成组件对照 |
| DeepSeek V4.1 Flash | SWA、CSA2、mHC、CED 与 Decoder SWA Bounded Replay 已完成 QA 学习；待独立验收 | `coding/deepseek_demo/` 已有 SWA/CSA2/mHC 教学实现；CED/Replay 尚无代码 | PyTorch 2.14.0+cu130、CUDA 可用；SWA 3 项、CSA2 4 项及 mHC 数值演示通过；未运行完整模型实验 | 保持为独立 QA 专题；后续补 CED/Replay 教学实现或继续层级 Indexer |
| RAG | Day21 已完成基础理论第一轮；能区分主要组件和故障层，指标与职责错误经新场景纠正 | 无保留的代码产物；BM25 实现按用户选择跳过 | 未运行 Parser、Embedding、向量库、Reranker 或模型 | 领域 Agent 阶段完成真实检索与评估闭环 |
| MCP / Tool Use | Day22 已完成第一轮，Day23 用 Tool Bridge 接入等价 mock 并验证调用控制 | `coding/training_platform_mcp/` 含静态 Server、Client、Schema、状态机和 Agent Runtime | 13 项 mock/状态机测试通过；未运行真实 MCP 协议 | 后续把 MockToolBridge 替换成官方 SDK 动态 Bridge |
| Agent 记忆、Agent RL | Day24 已完成记忆与 Context 第一轮；Agent RL 暂缓 | `memory_context.py` 含 Write Gate、版本链、检索、准入、预算和污染防护 | 12 项 Memory/Context 测试通过；无真实检索或持久化 | 下一主线进入 Agent 评估与可观测性；后续补真实混合检索 |

## 薄弱点与待验收

| 主题 | 已知问题或证据缺口 | 优先级 | 下次检查方式 | 下次检查 |
| --- | --- | --- | --- | --- |
| Policy staleness | Day19 问题未作答 | 中 | 在 Agentic RL 阶段用真实轨迹版本情境检查 | Agent 项目完成后 |
| GRPO 实战 | 教学代码未完成真实模型训练 | 高 | 在领域 Agent 上运行最小训练并解释 reward、KL、clipfrac 与失败案例 | Agentic RL 阶段 |
| 微调实战 | 缺少数据切分、训练配置和独立评估的实验产物 | 高 | 完成 LoRA/QLoRA 基线对照 | 阶段 2 |
| 代码能力 | 现有学习以问答和静态分析为主 | 高 | 读码、改错、实现、运行四类任务轮换 | 每周 |
| LLM 应用接口实操 | Day23 已有可运行 mock Runtime，但用户未独立从零实现；未接真实 LLM | 高 | 独立扩展 Action/StopReason 或完成证据合同并运行测试 | 2026-10-10 |
| MCP 真实集成 | Day22 有较完整静态教学代码，但未安装 SDK、启动 Server、验证 OAuth 或真实 list/call/read/get_prompt | 高 | 运行最小 Server/Client 或等价 mock，检查动态 Schema、错误与权限 Trace | 单 Agent 实操阶段 |
| RAG 工程实操 | 基础理论完成第一轮，但没有 Parser、Embedding、向量库、Reranker 或评估运行证据 | 高 | 在领域 Agent 项目中完成真实文档到检索评估的最小闭环 | 领域 Agent v1 阶段 |
| RAG 检索指标 | Day23 正确完成多 Query 宏平均与多证据完整性判断；无代码证据 | 中 | 使用实际 Top-K 输出编写并运行指标函数 | 2026-10-10 |
| Agent 完成证据 | 最小 `required_successful_tools` 已实现，但成功调用仍不保证答案与字段一致 | 高 | 增加 claim/evidence/version 合同和领域 Verifier 测试 | Agent 评估阶段 |
| Agent 记忆工程 | Day24 有完整教学实现，但检索为简单词项重合，且未持久化或接真实模型 | 高 | 独立加入冲突/删除同步或替换为 embedding+BM25 并评测 | 2026-10-13 |
| K3 未核实细节 | 历史归档部分机制只有标题或推断 | 中 | 对官方论文与实现逐项核对 | 前沿导读时 |

## 主线复习队列

主线复习只服务于 `LLM → Agent → 领域 Agent → Agentic RL` 的当前阶段。Day19 和 GRPO 工程题暂缓到 Agentic RL 阶段恢复，不在当前 Agent 学习前反复检查。

| 主题 | 当前证据 | 到期日期 | 下次检查方式 |
| --- | --- | --- | --- |
| Schema、业务、权限与状态校验边界 | Day20–22 场景判断正确；训练平台静态代码体现分层 | 2026-10-04 | 从新领域 Tool Handler 中定位各层职责 |
| 超时、幂等与结果未知 | Day22 经追问理解原 key、PROCESSING、RESULT_UNKNOWN 与 ESCALATED；3 项状态机测试通过 | 2026-10-04 | 阅读一段错误重试代码并定位新 key、无限查询和重复执行风险 |
| Proposal、确认与 commit | Day24 在删除管理员新场景中正确判断版本变化使旧确认失效 | 2026-10-13 | 独立设计一个高风险工具合约和状态迁移测试 |
| Context、State、Memory 与 Runtime | Day24 能区分三层、动态事实和长期偏好；有教学实现与测试 | 2026-10-07 | 给定混合 Trace，判断写入、检索、准入和权威查询位置 |
| RAG 分层故障定位 | Day21 能定位 Parser；Reranker 场景经纠正后理解 | 2026-10-01 | 给定新 Trace，只修改最早失败层并说明证据 |
| RAG 检索指标与证据完整性 | Day23 正确手算 Hit/Recall/Precision 宏平均，并识别 Hit=1 但证据不完整；无代码证据 | 2026-10-10 | 对实际 Top-K 编写并运行指标函数，加入多证据完整性测试 |
| RAG/Tool/Runtime 合同 | Day21 情境判断基本正确，无实现 | 2026-10-07 | 为新领域请求设计知识、事实、规则和输出 Schema |
| MCP Host/Client/Server 与能力发现 | Day22 连续追问后能完整解释调用链，无真实协议运行 | 2026-10-02 | 给定连接与调用 Trace，定位发现、筛选、执行和鉴权职责 |
| MCP Prompt Injection 与跨 Server 数据流 | Day22 新场景判断正确，无 Guardrail 运行证据 | 2026-10-08 | 给定 Tool 已授权且在列表中的数据外发请求，检查意图、来源、DLP 和确认 |
| 单 Agent Loop 与失败恢复 | Day23 理论、代码追踪和局部补全完成；13 项教学测试通过 | 2026-10-04 | 给定新领域 Trace，定位 Policy/Runtime/Tool/Verifier 最早失败层并解释 StopReason |
| FinalAction 证据合同 | Day24 经服务器状态场景纠正，能区分当前事实、历史缺口和版本不一致 | 2026-10-13 | 为事实、推断、缺口和 EvidenceRef 设计结构化输出并补测试 |
| Memory 写入、版本、隔离与污染 | Day24 完成第一轮并通过 12 项专项测试；未独立从零实现 | 2026-10-07 | 阅读错误 Pipeline，定位跨用户、旧版本、注入和 mandatory 丢失 |

## QA 专题待验收（不自动插入主线）

下列内容保留原复习证据和建议日期，但属于 Kimi/DeepSeek 等独立 QA 专题线。只有用户主动要求继续该专题、复习或验收时才使用，不占用主线学习时间。QA 自评掌握不等于验收通过。

| 主题 | 当前证据 | 到期日期 | 下次检查方式 |
| --- | --- | --- | --- |
| DeepSeek V4.1 SWA 基础与 shape | QA 学习完成，未独立验收 | 2026-09-27 | 给定 `B,T,H,Dh,W`，追踪 prefill 的 Q/KV/indices/weights/output shape |
| SWA prefill/decode 与环形缓存 | 有源码讲解与教学实现 | 2026-09-29 | 用绝对位置和槽位情境解释缓存覆盖与索引 |
| Sparse Attention 与逆 RoPE | 有逐行讲解，未独立实现 | 2026-10-03 | 阅读一段 gather/einsum 代码并解释 sink、共享旋转 KV |
| DeepSeek V4.1 SWA 综合 | 已归档，未运行完整模型 | 2026-10-17 | 对比纯 SWA、SWA+CSA2、Bounded Replay 的职责边界 |
| CSA2 全流程与 shape | 用户自评掌握，四项教学测试通过 | 2026-09-29 | 给定配置，追踪 Compressor、Indexer、局部/全局索引及输出 shape |
| CSA2 因果性与跨层复用 | 已讲解，未独立改码 | 2026-10-03 | 用分组边界解释可见性，并判断 Full/Reindex/Reuse 的状态变化 |
| CSA2 综合应用 | 已归档，未运行完整模型 | 2026-10-24 | 百万 token 场景下比较 SWA、Sparse Attention 与 CSA2 的存储/计算职责 |
| CED 训练/prefill/decode 边界 | 用户通过连续追问完成理解，无代码证据 | 2026-10-02 | 给定 `N,W,L`，画出三阶段实际经过的层数与 KV 来源 |
| Decoder SWA Bounded Replay | 用户指出 `-W` 位置缺少此前窗口，理解截断近似来源 | 2026-10-06 | 用 `W=3`、两层 Decoder 手工追踪标准与截断窗口的误差传播 |
| mHC 矩阵与 shape | 用户自评掌握，两份数值 demo 运行通过 | 2026-09-29 | 给定两路小矩阵，手算 `pre@X`、`comb@X`、`post.T@y` |
| mHC 参数与动态激活 | 已讲解，未独立改码 | 2026-10-05 | 区分 `fn/base/scale` 与 `pre/post/comb`，追踪完整 shape |
| mHC 综合取舍 | 已归档，未运行真实训练 | 2026-10-19 | 比较 residual、HC、mHC、Single-Pass 的稳定性与系统成本 |
| Agentic LLM 与 Agent Runtime 边界 | 用户自评掌握，尚无代码证据 | 随最小 Agent loop | 从代码中定位 model policy、runtime、executor、state 与 guardrail |

## 近期已问题型

- Day19 归档末尾已有 `old_logp` 是否覆盖、版本差与同步/异步取舍五问，尚未回答。下一题应合并为一个工作场景，避免原样重复。
- 2026-09-25 的 [DeepSeek V4.1 Flash 导读](../qa/2026-09-25-deepseek-v4-1-overview.md)是临时答疑，已讲解概览及输入到输出主链；未向用户出题或判定掌握程度。
- 2026-09-25 的 [SWA 逐步拆解](../qa/2026-09-25-deepseek-v4-1-swa.md)是 QA 讲解，不设置检查题。用户可随时提问；问答检测仅在正常计划学习或主动要求复习/验收时进行。
- 2026-09-26 已将 SWA、窗口索引、Sparse Attention、逆 RoPE 和整体架构聚合为正式 QA 存档；用户表示学习完成，独立验收待后续复习。
- 2026-09-26 已将 CSA2、Compressor、Indexer、Full/Reindex/Reuse 和配套代码归档；用户自评掌握，后续使用新 shape、因果边界和代码情境验收，不重复原讲解。
- 2026-09-26 已将 mHC 四路入口、动态系数、矩阵读写、Sinkhorn 与 Single-Pass 归档；用户自评掌握，后续使用新矩阵和代码情境验收。
- 2026-09-29 已讲清 CoT、ReAct、Plan-and-Execute、原生 Tool Calling 的层次，以及 Agent 能力内嵌进模型的范围；用户自评理解，后续通过 Agent loop 代码自然验证，不在当前主线重复出纯概念题。
- 2026-09-29 Day20 已检查结构正确但语义错误、超额退款、缺少业务校验、External State、幂等超时、版本变化和邮件 Prompt Injection 场景；下次不要原样重复，应换领域或进入代码题。
- 2026-09-30 已回答 Day20 的 RAG 职责边界停点，并系统完成 RAG 基础第一轮。近期题型已覆盖 Chunk 完整性、BM25/Dense 选择、Reranker/Runtime、Hit/Recall/Precision、版本过滤、缺知识/缺事实路由、Query 分解和 Parser/Reranker 故障定位；后续复习应换领域或进入代码 Trace，不原样重复。
- 2026-09-29 已复习 CSA2 Full/Reindex/Reuse，并围绕 CED 连续讨论 prefill/decode/预训练和 `-W` replay 边界；后续验收应换成具体 `N,W,L` 数据流或代码，不重复原问题。
- 2026-10-01 Day22 已覆盖 Tool 最小暴露、Schema/业务/权限、Host/Runtime/Client/Server、MCP Tool 到 REST 适配、Resources/Prompts、幂等键与操作状态、Prompt Injection 和跨 Server 外发。后续不要重复纯定义题，应进入新领域代码 Trace、真实协议运行或独立设计。
- 2026-10-01 用户指出追问结束后“接着学”被错误解释为切换父话题；`AGENTS.md` 与归档模板已加入话题栈、显式收尾和精确停点规则。后续“接着学”默认继续当前父话题，未显式关闭不得跳转。
- 2026-10-03 Day23 已检查 RAG 宏平均和多证据完整性；Agent Loop 题型覆盖未知工具、可恢复超时、RESULT_UNKNOWN、原幂等键、重复动作、预算、FinalAction、TaskState、Tool Bridge 和证据引用。后续不要重复定义题，应进入新领域 Trace、独立改错或真实 Bridge；用户曾把第三次重复 Action 误判为 MAX_STEPS、把远端 PROCESSING/UNKNOWN 与本地 RESULT_UNKNOWN 混用，经解释后纠正。
- 2026-10-06 Day24 已覆盖 TaskState/Context/Memory、Write Gate、版本/过期、动态业务事实、滑动窗口与检索、跨用户隔离、污染、mandatory token 预算和 Memory 评估。后续不要重复酒店分类题；应进入独立改错、删除同步、真实混合检索或评测代码。用户曾把 `RUNNING + latest_error=null` 扩大为“没有发生故障”，经纠正后能识别时间范围和版本证据缺口。

## 下一次会话

若用户启动计划学习：Day24 的 Agent 记忆与上下文工程第一轮已关闭，从 Agent 评估与可观测性开始，先定义任务成功率、轨迹质量、成本、时延、安全边界和回归测试的最小合同；用 Day23 Runtime、Day24 Memory/Context 及 25 项测试作为被测系统，不重新讲纯定义。可先用一条新领域 Trace 短复习记忆写入/准入边界。真实 LLM/MCP、持久化、混合检索、语义 Verifier 和独立从零实现保留为工程证据缺口；暂不检查 Day19，也不自动穿插 Kimi/DeepSeek QA 复习。若用户直接提问，则进入独立 QA，不强行切回主线。

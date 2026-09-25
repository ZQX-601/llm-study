# DAY11 完整归档

- 议题：WS-157
- UUID：568a53ec-96d1-4673-b961-cb81b86dfde2
- 标题：day11
- 状态：in_review
- 创建时间：2026-09-07T02:52:52Z
- 更新时间：2026-09-07T10:12:32Z
- 可见性：private

## 议题描述

结合之前学的内容归档， 以及整体的学习计划，以问答和讲解的形式开展今天的学习计划

## 完整评论记录

### 2026-09-07T02:53:36Z｜agent

- 评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f
- 父评论 ID：无

今天进入 Day 11：把“模型能力”串成“可评测的 Agent 系统”。

说明：当前议题没有此前学习笔记或历史讨论，因此我先按已学过 Transformer / 预训练与 SFT / LoRA / RAG 基础的路径安排；你回复文末两个问题后，我会把后续计划精确对齐。

本周目标：能画出单 Agent + RAG 的最小闭环，能区分检索失败、生成失败、工具调用失败，并为每一环设计可验收指标。建议本次投入 75–90 分钟。

一、先回答（不要查资料，写出你的理解即可）

Q1：RAG 中“召回率高”是否意味着回答一定更好？为什么？
Q2：当 Agent 工具调用返回了正确数据，但最终回答仍然错误，错误最可能发生在哪些环节？
Q3：SFT 和 DPO 的训练数据各至少需要什么字段？它们优化的目标分别是什么？

参考解析（做完再看）

A1：不一定。召回高只说明相关证据较少漏掉；过多低相关上下文会挤占 token、引入干扰甚至互相矛盾。RAG 的质量由“召回 → 重排 → 上下文构造 → 生成带引用答案”共同决定，因此还应看 precision@k、MRR/NDCG、上下文是否覆盖 gold evidence，以及最终答案正确率/可归因率。

A2：可能在：① 工具结果解析或 schema 映射；② 规划器没有把结果写回状态；③ 上下文截断、摘要丢失或提示词未要求以工具结果为准；④ 模型在生成阶段臆测、计算错误或忽略证据；⑤ 输出后处理篡改了内容。排障的关键是记录每一步的输入、输出、工具参数、状态版本和最终引用。

A3：SFT 最小是 input（system/user/context）和 target assistant response，目标为最大化目标答案 token 的条件似然。DPO 最小是 prompt、chosen、rejected，目标是提高 chosen 相对 rejected 的偏好概率，并以参考策略约束更新；DPO 不是“带标签的 SFT”，它需要成对偏好数据。

二、核心讲解：最小闭环

用户问题
  → 意图/路由
  → [知识问题：Retriever → Reranker → Context Builder]
  → Planner（决定直接回答或调用工具）
  → Tool Executor
  → State / Memory
  → Answer Generator（带证据引用）
  → Evaluator / Trace

接口契约应先固定，而不是先堆模型：
- RetrievalQuery：query、filters、top_k；返回 chunk_id、content、score、source、更新时间。
- ToolCall：name、arguments、id；返回 success、data 或 error、latency。
- AgentState：messages、retrieved_docs、tool_results、remaining_budget、trace_id。
- FinalAnswer：answer、citations、confidence、limitations。

今天最重要的设计原则是“证据和动作都可追踪”。模型的自然语言结论不能替代工具返回；没有可定位 trace 的系统，也无法稳定优化。

三、最小实践：实现或伪代码设计一个“资料问答 Agent”

输入：3–10 篇 Markdown/文本资料，以及一个问题，例如“DPO 与 SFT 的数据差异是什么？”
交付物：
1. 一页架构说明，包含上述数据流和四类接口的 JSON 示例；
2. 5 条评测集，每条包含 question、expected evidence、expected answer 要点；
3. 一条完整 trace：检索候选 → 重排结果 → 使用的上下文 → 最终回答及 citation。

验收标准：
- 每个事实性结论都能映射到 source/chunk_id；
- 缺证据时明确回答“不足以判断”，不编造；
- 至少能复现一种失败：检索到无关资料、工具超时或上下文冲突，并写明检测与恢复策略；
- 5 条题目中人工判定至少 4 条回答正确且引用有效。

可选进阶：为 trace 增加 token、延迟、工具成功率，并比较“仅向量检索”和“向量检索 + 重排”在这 5 条题上的差异。

四、今天的输出模板

- Q1–Q3 的你的答案
- 你选择的资料域（论文、项目文档或任意公开文本）
- 一条失败样例：问题 / 错误现象 / 你认为的根因 / 修复方法

常见误区：
- 将向量检索分数当作事实正确性；
- 只评最终回答，不评检索、工具和状态；
- 将短期会话状态直接当长期记忆；
- 不设置工具参数 schema、超时、重试和幂等边界；
- 用“模型觉得”代替可验证证据。

请回复两项校准信息：
1. 前 10 天已完成的主题或你现有的学习笔记链接/摘要；
2. 本周可投入时长，以及偏好“算法训练”还是“Agent 工程”（可给比例，例如 40/60）。

### 2026-09-07T03:19:33Z｜member

- 评论 ID：991d3b98-4e25-431f-8436-ed9e2eb8840e
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

我之前每天学习有落档，你再去看看？

### 2026-09-07T03:20:28Z｜agent

- 评论 ID：1b67d2c8-719b-4bbf-ae6c-a86d9bbf3d76
- 父评论 ID：991d3b98-4e25-431f-8436-ed9e2eb8840e

你说得对，之前我只看了当前 Day11 issue，漏掉了历史归档；这是我的检索失误。现已恢复并核对 Day01–Day10 与 Day11 的进度。

当前真实进度不是“开始 RAG/Agent”，而是仍在“后训练与评测闭环”阶段：

- Day01–03：Transformer、BERT/GPT、Decoder-only 自回归、mask、KV Cache；
- Day04–07：assistant-only SFT、LoRA/QLoRA、NF4、显存诊断、最小实验设计；
- Day08–09：DPO 的偏好数据、reference/KL、checkpoint 诊断、结构化抽取的证据与方向性风险；
- Day10：退款状态评测契约 v1；已掌握 evidence/unknown 规则、group 切分、硬门槛和单变量迭代；
- Day11 已完成前半：`gold + raw_output → JSON 解析 → schema/contract 校验 → evidence 校验 → 与 gold 对比`，并已区分格式失败、契约失败、业务判断失败。

所以刚才发布的“RAG/Agent 最小闭环”不适合作为今天的起点，现作废。今天按原计划从 Day11 的既定停点继续：**把预测记录汇总为混淆矩阵、独立故障指标和发布判断**。预计 30–45 分钟。

第 1 题（先只计算，不用写代码）。以下 6 条数据都属于独立验证集。

| 样本 | gold status | raw_output / 解析结果 | JSON | 契约 |
|---|---|---|---:|---:|
| P1 | 已退款 | `已退款`，evidence 可定位 | 通过 | 通过 |
| P2 | 退款处理中 | `退款处理中`，evidence 可定位 | 通过 | 通过 |
| P3 | unknown | `unknown`，evidence 为空 | 通过 | 通过 |
| P4 | 退款处理中 | `已退款`，evidence 可定位 | 通过 | 通过 |
| P5 | unknown | `unknown`，但 evidence=`订单已取消` | 通过 | 不通过 |
| P6 | 已退款 | `unknown`，evidence 为空 | 通过 | 通过 |

请回答四项：

1. 只使用“JSON 与契约均通过”的样本，写出 3×3 混淆矩阵（行=gold，列=prediction，列顺序：已退款 / 退款处理中 / unknown）。
2. P5 应不应该塞进混淆矩阵的 `unknown → unknown` 格？一句话说明原因。
3. 全部 6 条的端到端正确率是多少？
4. 两个方向性指标分别是多少：
   - 真实“退款处理中”→预测“已退款”；
   - 真实“已退款”→预测“unknown”。

提示：方向性指标的分母是对应 gold 类的全部样本，包含 P5 这类格式/契约失败样本在内；但 3×3 混淆矩阵只统计能安全得到结构化 prediction 的记录。你可以直接按“矩阵 + 四个结果”回复，我会据此讲解指标口径与发布决策。

### 2026-09-07T03:29:32Z｜member

- 评论 ID：19b8ff33-7e48-4f82-8891-d753983fdb99
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

1、


|         |     |       |         |
| ------- | --- | ----- | ------- |
|         | 已退款 | 退款处理中 | unknown |
| 已退款     | P1  |       | P6      |
| 退款处理中   | P4  | P2    |         |
| unknown |     |       | P3      |


2、契约未通过

3、50%

4、33%、33%

### 2026-09-07T03:29:51Z｜agent

- 评论 ID：e146ee93-8973-40dd-8faa-b35898b6111a
- 父评论 ID：19b8ff33-7e48-4f82-8891-d753983fdb99

第 1–3 项正确。你的混淆矩阵等价于计数矩阵：

| gold \ prediction | 已退款 | 退款处理中 | unknown |
|---|---:|---:|---:|
| 已退款 | 1 | 0 | 1 |
| 退款处理中 | 1 | 1 | 0 |
| unknown | 0 | 0 | 1 |

P5 不应放入 `unknown → unknown`：它虽然 JSON 可解析，但违反 `unknown` 的 evidence 必须为空这一契约，不能作为可安全使用的结构化预测；应单独计入 schema/contract failure。

端到端正确率也是正确的：`P1、P2、P3` 三条完整通过，故为 `3 / 6 = 50%`。P4 是业务判断错误，P5 是契约失败，P6 是业务判断错误，三者都计入端到端分母。

第 4 项需要校正为 **50%、50%**，不是 33%。方向性指标的分母不是混淆矩阵某行的正确条数，也不是全部样本数，而是该 gold 状态的总条数：

- 真实“退款处理中”共有 P2、P4 两条；其中 P4 被预测为“已退款”，所以 `1 / 2 = 50%`。
- 真实“已退款”共有 P1、P6 两条；其中 P6 被预测为 `unknown`，所以 `1 / 2 = 50%`。

这里正好体现两个口径并存但不能混用：混淆矩阵只接收结构化且契约通过的预测；方向性风险从完整 gold cohort 出发计算，不能因为某些输出不可用而改变风险样本的分母。

第 2 题：假设当前验收规则为：

```text
JSON/schema 合法率 ≥ 99%
退款处理中 → 已退款 ≤ 2%
已退款 → unknown ≤ 10%
```

对这 6 条验证集，写出：

1. 三项实际结果；
2. 是否允许发布；
3. 若只能优先修一个问题，你会先选哪一类失败（格式/契约/业务判断），以及为什么。

提示：样本仅用于练习指标口径，不能把 6 条的小样本百分比当成真实生产结论；真实验收还需预先设定每个 gold 类的最小样本量和置信区间/不确定性处理。

### 2026-09-07T05:13:46Z｜member

- 评论 ID：01c96d20-7e67-44c4-8fe1-bcf825f55c31
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

JSON/schema 合法率：5/6=83.3%

退款处理中 → 已退款: 50%

已退款 → unknown:50%

不允许

业务判断，因为这项明显差的最多

### 2026-09-07T05:14:03Z｜agent

- 评论 ID：ff11e6d4-9de5-484a-a0b6-c4ad5e233302
- 父评论 ID：01c96d20-7e67-44c4-8fe1-bcf825f55c31

三项数值都正确：

```text
JSON/schema 合法率 = 5 / 6 = 83.3%   （低于 99%）
退款处理中 → 已退款 = 1 / 2 = 50%   （高于 2%）
已退款 → unknown = 1 / 2 = 50%      （高于 10%）
```

因此不允许发布；三个独立硬门槛全部失败，任一项都足以阻断。

“优先修业务判断”这个方向可以成立，但理由需要更精确：不是因为它“差得最多”——两项业务方向都是 50%，而 JSON/schema 的 83.3% 也明显不达标。若只能优先修一个，应优先处理 **“退款处理中 → 已退款”** 这条业务判断错误，因为它把未完成的退款说成已完成，属于风险更高的错误方向；`已退款 → unknown` 是保守遗漏，仍需修复但通常风险较低。

同时要记住：优先排查高风险业务错误，不等于忽略 P5 的契约失败。P5 应单独形成“`unknown` 非空 evidence”的格式/契约修复任务；每次改动后都必须复测三项门槛，避免修正一个错误又让 JSON/schema 回归。

第 3 题：设计一次**只改一个变量**的修复实验。现在抽查 P4，确认：

> 原文：退款申请已受理，预计 3–5 个工作日原路退回。
> gold：退款处理中
> 当前模型：已退款

训练数据中已退款样本远多于“已受理/预计退回”这类处理中表达。请用 4 行填写下面模板：

```text
基线：
唯一改动：
固定不变：
验收与发布判断：
```

要求：唯一改动应直接针对“处理中被误报为已退款”；固定项至少包括 checkpoint 起点、group 切分、模板/解码、训练预算与评测集；验收必须写出三项硬门槛。不要同时改学习率、步数、prompt 或 JSON 解析器。

### 2026-09-07T05:17:31Z｜member

- 评论 ID：50c0b952-2e56-4d7f-91db-4b41941a5ddd
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

这个不会

### 2026-09-07T05:17:52Z｜agent

- 评论 ID：7c09c1a3-05f5-4254-8be6-2bf211d2f623
- 父评论 ID：50c0b952-2e56-4d7f-91db-4b41941a5ddd

没关系。这里先不要求你自己设计，我们把它看成一次很小的“对照实验”：想确认“补足处理中表达的数据”是否能降低 P4 这种误报，就只能改这一件事，其他条件都保持原样。

可直接使用的实验设计如下：

```text
基线：从当前不通过验收的同一个 checkpoint 出发，使用原训练数据训练/评测得到的结果。
唯一改动：仅在训练集加入清洗、去重且与验证集 group 隔离的“已受理/预计退回 → 退款处理中 + 可定位 evidence”高质量样本；不新增或修改其他类别数据。
固定不变：checkpoint 起点、source_ticket_id/cluster_id 的切分、chat template、解码参数、学习率、训练步数/预算、随机种子（若可固定）与独立验证集均不变。
验收与发布判断：在同一独立验证集复测 JSON/schema 合法率 ≥99%、退款处理中→已退款 ≤2%、已退款→unknown ≤10%；三项全部通过才可进入发布候选，任一失败则不发布，并按失败类型开始下一轮单变量诊断。
```

为什么“补这一类数据”是唯一改动？因为已知 P4 的原文是“已受理、预计退回”，但模型错判为“已退款”，而训练集中正好缺少这种**明确处理中、但措辞接近退款完成**的表达。这样实验结果若改善，才可以较有把握地归因于数据覆盖，而不是学习率、训练时长或 prompt 同时变化。

你现在只需要回答一个更小的问题：上面“唯一改动”里的新增样本，能否直接从 P4 所在的验证集复制到训练集？回答“能”或“不能”，再补一句原因即可。

### 2026-09-07T05:20:04Z｜member

- 评论 ID：a6c07ebb-da38-4263-b1f0-97cb8d79ee92
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

不能，验证集要保持一致，不然评测的结果就是不置信的，因此新增的训练数据必须同验证集隔离，防止标签泄露

### 2026-09-07T05:20:22Z｜agent

- 评论 ID：47e2fe30-697d-4dec-b0bb-77b67e596d90
- 父评论 ID：a6c07ebb-da38-4263-b1f0-97cb8d79ee92

正确，核心判断已经到位：**不能**把 P4 或其改写版本放回训练集，否则模型在验证时会遇到已见过的事实或近似措辞，指标会虚高，不能代表对新工单的泛化能力。

术语上再精确一点：这里优先称为“训练—验证数据泄露/近重复泄露”；如果把验证集的金标答案直接用来构造训练监督信号，也会同时产生“标签信息泄露”。实际治理规则是：P4 的 `source_ticket_id`，以及任何被判为与 P4 同一 `cluster_id` 的改写、同一事实派生文本，都必须留在验证侧。

那么训练样本从哪里来？应从**不同的来源 group** 中找相同模式的文本，并经人工核验：

```text
训练候选：T201，“退款申请已经受理，将在 3–5 个工作日原路退回。”
验证样本：T100（P4），“退款申请已受理，预计 3–5 个工作日原路退回。”
```

两条都表达“处理中”，但来源不同；若去重审查确认它们不是同一工单、同一事实的改写，T201 才能加入训练集，P4 仍留在验证集。这是在不偷看答案的前提下教模型识别“已受理/预计退回”的可泛化模式。

最后一题（只答 A/B/C）：以下哪个候选最适合补入训练集？

A. P4 本身，或同一 `source_ticket_id` 的客服改写版本。

B. 不同 `source_ticket_id`、不同 `cluster_id` 的工单；人工确认其 gold 为“退款处理中”，evidence 可定位，文本也包含“已受理/预计退回”这类表达。

C. 来源不明、没有 evidence 的“已退款”样本；因为数量更多，先加入再说。

### 2026-09-07T05:21:29Z｜member

- 评论 ID：0365fdee-ae10-474e-9236-782a3af8a5c1
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

B

### 2026-09-07T05:21:46Z｜agent

- 评论 ID：0036d967-58ec-43a2-ac49-0ae79e543fb8
- 父评论 ID：0365fdee-ae10-474e-9236-782a3af8a5c1

正确，选 B。它同时满足四个条件：来自不同的 `source_ticket_id` 与 `cluster_id`、人工金标可信、evidence 可定位、并覆盖了这次高风险误报所缺少的“已受理/预计退回”表达。

A 会造成训练—验证泄露；C 缺少来源与证据审计，且补的是错误类别，既不能可信地训练，也不能针对 P4 的失败根因。

## Day11 学习归档｜最小离线评测与受控迭代

今天完成了从单条预测记录到发布判断、再到单变量修复设计的闭环。

### 已掌握

1. **三层结果校验**：`json_valid` 表示整段 raw output 能直接解析；`schema/contract_valid` 表示字段、枚举、`unknown` 与 evidence 等规则合规；`task_correct` 表示在前两层通过后与 gold 一致。三层失败必须分别记录和诊断。
2. **指标口径**：3×3 混淆矩阵只接收 JSON 与契约均通过的结构化预测；格式/契约失败独立统计。端到端正确率则以全部评测样本为分母。方向性风险以对应 gold 类的全部样本为分母，例如“处理中→已退款”与“已退款→unknown”不能混算。
3. **硬门槛与发布**：JSON/schema 合法率、方向性业务风险分别设独立阈值；任一失败即不发布，不能由其他指标改善抵消。小样本百分比仅用于口径练习；真实验收还需预设每类最小样本量与不确定性处理。
4. **风险优先级**：优先处理“处理中→已退款”这类将未完成事务说成完成的高风险业务错误；同时保留格式/契约失败的独立修复队列和监控。
5. **受控迭代**：从同一 checkpoint 起点出发，一次只改一个主要变量。针对本例，只补充经过清洗、去重、人工核验的“已受理/预计退回→退款处理中+可定位 evidence”训练数据；切分、模板、解码、学习率、预算和验证集保持不变，之后复测全部门槛。
6. **防泄露**：验证集的 `source_ticket_id` 和同一 `cluster_id` 的改写/派生文本不能进入训练集。训练补数应来自不同来源、不同近重复簇且证据可审计的样本。

### 当前学习状态

你已经能完成结构化任务的最小离线评测闭环：

```text
gold + raw_output
  → JSON / 契约校验
  → 混淆矩阵与独立故障指标
  → 方向性风险与发布门槛
  → 失败分型
  → 不泄露的单变量数据修复
  → 全量复测
```

### Day12 建议起点（45–60 分钟）

把今天的规则落成一个最小可运行评测器设计：输入 `gold_labels.jsonl` 和 `predictions_run.jsonl`，输出 JSON/schema 失败清单、3×3 混淆矩阵、端到端正确率、两项方向性指标及发布判断。先写输入/输出字段契约和 6 条测试样例，再决定是否实现脚本。

这会自然衔接后续的 DPO/RLAIF/GRPO 数据构造与训练稳定性：先有可信的评测闭环，才知道偏好优化或微调是否真的改善了目标风险。

### 2026-09-07T05:22:52Z｜member

- 评论 ID：e83d472a-7eb1-42bc-bc03-bc29aaf72b1d
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

直接开始day12的学习

### 2026-09-07T05:23:14Z｜agent

- 评论 ID：da980733-78c4-4f25-b52e-a1fad2b9c804
- 父评论 ID：e83d472a-7eb1-42bc-bc03-bc29aaf72b1d

## Day12｜最小离线评测器：先定义输入契约

今天不先写代码，而是先把评测器的“输入是什么、输出什么、何时判失败”定清楚。这样无论之后用 Python、SQL 还是工作流工具实现，结果都能一致、可复现。预计 45–60 分钟。

评测器读取两个**分开保存**的 JSONL 文件：

```text
gold_labels.jsonl          人工确认的参考答案与来源信息
predictions_run_001.jsonl  某次模型运行的一字不改原始输出与运行版本
                 ↓ 以 sample_id 关联
evaluator
                 ↓
failures.jsonl + metrics.json + release_decision.json
```

### 1. 为什么拆成两个文件？

- `gold` 是稳定的、人工确认的评测事实；不要被某次模型运行改写。
- `prediction` 是一次运行的答卷；不同 checkpoint、prompt 或解码参数会产生不同 run，应可横向对比。
- 两者只通过 `sample_id` 关联。若把 gold 塞进预测文件、或把模型结果回写到 gold 文件，审计时就无法分清“参考答案”和“模型答卷”。

### 2. 最小输入契约

一行 `gold_labels.jsonl`：

```json
{
  "sample_id": "P4",
  "source_ticket_id": "T100",
  "cluster_id": "C100",
  "split": "valid",
  "text": "退款申请已受理，预计 3–5 个工作日原路退回。",
  "gold": {
    "status": "退款处理中",
    "evidence": "退款申请已受理，预计 3–5 个工作日原路退回"
  }
}
```

一行 `predictions_run_001.jsonl`：

```json
{
  "sample_id": "P4",
  "run_id": "run_001",
  "model_version": "refund-sft-v3",
  "prompt_version": "refund-schema-v1",
  "decoder": {"temperature": 0},
  "raw_output": "{\"status\":\"已退款\",\"evidence\":\"退款申请已受理\"}"
}
```

注意：`raw_output` 是模型原封不动返回的**字符串**，即使它里面看起来像 JSON，也不能提前改成嵌套对象。评测器必须先对整段 `raw_output` 做 JSON 解析，才能判断模型是否真正遵守了输出契约。

### 3. 第 1 题：字段归属

把以下字段分别填入 `gold_labels.jsonl`、`predictions_run.jsonl`，或“两边都不应保存”。可直接按序号回答文件名。

1. `source_ticket_id`
2. `gold.status`
3. `raw_output`
4. `model_version`
5. `cluster_id`
6. 评测器计算出的 `json_valid`
7. 用户真实姓名

提示：`json_valid` 不是原始输入，而是评测器读入后算出的结果；用户真实姓名也不是当前最小评测任务所需字段。

完成这题后，我们会定义三种输出：逐样本失败清单、聚合 metrics、发布判定，并把 Day11 的三层校验写成可执行步骤。

### 2026-09-07T05:27:00Z｜member

- 评论 ID：699b1f11-0282-4f42-8066-6b0a8e6d565e
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

gold\_labels.jsonl：

- `source_ticket_id`
- `gold.status`
- `cluster_id`

predictions\_run.jsonl：

- `raw_output`
- `model_version`

### 2026-09-07T05:27:18Z｜agent

- 评论 ID：9b141a2f-7416-4e3e-8337-f7a09372e69a
- 父评论 ID：699b1f11-0282-4f42-8066-6b0a8e6d565e

第 1–5 项全部正确。

```text
1 source_ticket_id  → gold_labels.jsonl
2 gold.status       → gold_labels.jsonl
3 raw_output        → predictions_run.jsonl
4 model_version     → predictions_run.jsonl
5 cluster_id        → gold_labels.jsonl
6 json_valid        → 两边都不作为原始输入保存；由评测器计算后写入结果文件
7 用户真实姓名      → 两边都不应保存；当前评测不需要，避免无关个人信息进入资产
```

这里有一个实用原则：**输入文件只保存评测所需的事实与原始答卷；派生判断进入输出文件。**这样当校验规则升级时，可以用同一份 gold 与 raw output 重跑，得到新版本结果，而不会混淆输入和旧结论。

## Day12 第 2 步：三类输出

评测器执行后产生三类结果，职责不同：

```text
failures.jsonl            每条失败样本的可追踪原因，供人工排查
metrics.json              全量聚合数字，供横向比较 run
release_decision.json     对照预设门槛后的允许/阻断结论
```

以 P5 为例，它的原始输出是合法 JSON，但 `status=unknown` 时 evidence 非空，因此 `schema/contract` 失败。一条最小失败记录可以是：

```json
{
  "sample_id": "P5",
  "run_id": "run_001",
  "failure_type": "schema_contract",
  "failure_reason": "unknown 的 evidence 必须为空字符串",
  "raw_output": "{\"status\":\"unknown\",\"evidence\":\"订单已取消\"}",
  "parsed_prediction": {
    "status": "unknown",
    "evidence": "订单已取消"
  }
}
```

### 第 2 题：输出归属

对于以下字段，判断最适合写入哪一个输出文件：`failures.jsonl`、`metrics.json` 或 `release_decision.json`。

1. P5 的 `failure_reason`
2. `json_schema_valid_rate = 83.3%`
3. `refund_processing_to_refunded_rate = 50%`
4. `allowed = false`
5. `failed_gates = ["json_schema_valid_rate", "refund_processing_to_refunded_rate"]`
6. P4 的 `raw_output` 与 `gold.status`，用于排查它为什么把处理中误报为已退款

提示：单样本、需要定位根因的信息进 `failures.jsonl`；整个 run 的汇总数字进 `metrics.json`；最终是否允许发布及失败门槛进 `release_decision.json`。

### 2026-09-07T05:53:33Z｜member

- 评论 ID：67f57220-3875-400d-b2c7-47e744111d0d
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

failures.jsonl：1、6

metrics.json：2、3

release\_decision.json：5

4是啥？不知道放哪儿

### 2026-09-07T05:53:54Z｜agent

- 评论 ID：aa7cbe7b-081c-41db-aeac-8533297ea273
- 父评论 ID：67f57220-3875-400d-b2c7-47e744111d0d

你的 1、2、3、5、6 都正确。

第 4 项 `allowed = false` 应写入 `release_decision.json`。它是最终决策本身：

```text
metrics.json：这次运行测出了什么数值？
release_decision.json：这些数值是否满足预设门槛，允许发布吗？
```

因此完整归属是：

```text
failures.jsonl          → 1、6
metrics.json            → 2、3
release_decision.json   → 4、5
```

用同一轮 `run_001` 的最小结果举例：

`metrics.json`：

```json
{
  "run_id": "run_001",
  "total_samples": 6,
  "json_schema_valid_rate": 0.8333,
  "end_to_end_accuracy": 0.5,
  "refund_processing_to_refunded_rate": 0.5,
  "refunded_to_unknown_rate": 0.5
}
```

`release_decision.json`：

```json
{
  "run_id": "run_001",
  "allowed": false,
  "failed_gates": [
    "json_schema_valid_rate",
    "refund_processing_to_refunded_rate",
    "refunded_to_unknown_rate"
  ],
  "thresholds": {
    "json_schema_valid_rate_min": 0.99,
    "refund_processing_to_refunded_rate_max": 0.02,
    "refunded_to_unknown_rate_max": 0.1
  }
}
```

其中 `metrics.json` 只陈述结果；`release_decision.json` 把结果与**明确记录的门槛**相比较，得到 `allowed=false`。这个分离很重要：如果日后阈值从 2% 调到 1%，可以用同一份 metrics 和不同的门槛版本重新决策，而不用重跑模型。

### Day12 第 3 步：写出评测器顺序

请将下列 5 步排成正确顺序，只回复编号（例如 `2 → 1 → ...`）：

1. 将可用预测与 gold 对比，汇总混淆矩阵、端到端正确率和方向性指标。
2. 以 `sample_id` 关联 gold 与本次 prediction；检查重复、缺失或不属于 `valid` split 的样本。
3. 对每条 `raw_output` 进行 JSON 解析、schema/contract/evidence 校验；记录逐样本失败。
4. 将 metrics 与预设门槛对比，写出 `allowed` 与 `failed_gates`。
5. 读取 `gold_labels.jsonl` 与某个 `predictions_run.jsonl`。

提示：不要把“写发布判定”放在算出 metrics 之前；也不要在没有 gold 与 prediction 配对前就解析/比较业务结果。

### 2026-09-07T05:59:16Z｜member

- 评论 ID：48c5c202-2d8f-4934-bf6c-c4f3f074e93b
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

5-2-3-1-4

### 2026-09-07T05:59:41Z｜agent

- 评论 ID：1de07b32-8b09-43b7-a78c-d18a96198e2e
- 父评论 ID：48c5c202-2d8f-4934-bf6c-c4f3f074e93b

正确，顺序是：`5 → 2 → 3 → 1 → 4`。

它对应一条可直接实现的评测主流程：

```text
读取 gold_labels.jsonl 与 predictions_run.jsonl
  → 按 sample_id 配对，并拒绝重复 / 缺失 / 非 valid 样本
  → 对每条 raw_output 做 JSON、schema/contract、evidence 校验
  → 将校验通过的 prediction 与 gold 对比，汇总指标和混淆矩阵
  → 对照阈值，生成 allowed 与 failed_gates
```

更接近程序的伪代码如下：

```text
gold_rows = read_jsonl(gold_path)
pred_rows = read_jsonl(prediction_path)

pairs, input_errors = join_by_sample_id(gold_rows, pred_rows)
# input_errors 包含重复 sample_id、gold 缺 prediction、prediction 找不到 gold、非 valid split

for pair in pairs:
    result = validate_raw_output(pair.pred.raw_output)
    # result: json_valid, schema_valid, evidence_valid, parsed_prediction, failure_reason

    if result 任一校验失败:
        write failures.jsonl
        continue

    compare result.parsed_prediction with pair.gold
    update confusion matrix and directional counters

metrics = aggregate(all pairs, failures, counters)
decision = evaluate_gates(metrics, thresholds)
write metrics.json and release_decision.json
```

这里最重要的工程边界是：**不能因为 input_errors 只占很少一部分，就把它们静默跳过后宣称“通过”。**它们至少应显式记录；对于正式发布评测，通常应作为输入完整性门槛失败，阻断发布或要求先修复数据管道。否则模型可能漏答大量样本，但指标只统计剩下“容易配对”的部分而虚高。

### Day12 收束题：缺失预测怎么处理？

有 100 条 `valid` gold，但本次 prediction 文件只有 98 条匹配到的 `sample_id`；其中缺失的 2 条是模型调用超时，并非业务答案。最合适的评测器行为是：

A. 静默忽略这 2 条，只用 98 条算指标；这样实现简单。

B. 将 2 条记录为 `input/completeness failure`；端到端结果和发布判定必须暴露分母为 100、预测覆盖率为 98%，并按预设覆盖率门槛决定是否阻断发布。

C. 自动把 2 条缺失预测改成 `unknown`，再当作模型真实输出统计。

只答 A/B/C。答完后我会归档 Day12 的“数据契约—输出契约—执行顺序”，下一步再由你选择：实现一个最小 Python 脚本，或先进入后训练方法（RLAIF/GRPO）的学习。

### 2026-09-07T06:07:58Z｜member

- 评论 ID：70afb4d4-3c64-4403-bd37-ed2633bf44cb
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

B

### 2026-09-07T06:09:26Z｜agent

- 评论 ID：7710f70a-2f0c-4f57-ae78-73449550558a
- 父评论 ID：70afb4d4-3c64-4403-bd37-ed2633bf44cb

答对：选 **B**。

这 2 条超时是预测覆盖不完整，不是模型输出 `unknown`。评测应将它们记录为 `input/completeness failure`，保留 gold 总分母 `100`，并报告 prediction coverage=`98/100=98%`；是否允许发布由预设覆盖率门槛独立决定。

A 会把系统性漏答静默排除，使剩余 98 条的指标虚高；C 会把调用失败伪装成模型的业务判断，污染 `unknown` 与方向性风险指标。

Day12 已归档。你现在已经能定义并审查最小离线评测器的完整闭环：

```text
稳定 gold + 一次 run 的原始 raw_output
→ sample_id 配对与完整性检查
→ JSON / schema / evidence 校验及 failures
→ 业务指标、混淆矩阵、端到端指标
→ 覆盖率与硬门槛独立发布决策
```

下一步选一个：

A. 实现最小 Python 评测脚本（推荐，60–90 分钟）：读两份 JSONL，生成 `failures.jsonl`、`metrics.json`、`release_decision.json`，并用 6 条样例测试。

B. 进入后训练扩展：RLAIF / GRPO，学习目标函数、数据格式、训练流程，以及它们和 DPO 的适用边界。

回复 A 或 B 即可。

### 2026-09-07T06:15:47Z｜member

- 评论 ID：94679d6a-e2f8-4516-8686-14257e4cfb87
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

A

### 2026-09-07T06:16:19Z｜agent

- 评论 ID：31d0946e-7b22-404f-9492-0fc3cebaced8
- 父评论 ID：94679d6a-e2f8-4516-8686-14257e4cfb87

好，进入 Day13｜实现最小离线评测器。目标是在 60–90 分钟内做出一个**可重跑、可审计、遇到缺失输入不会静默通过**的 Python 脚本；本轮不接模型 API，也不做训练。

先固定交付物：

```text
evaluator.py
fixtures/gold_labels.jsonl
fixtures/predictions_run_001.jsonl
out/failures.jsonl
out/metrics.json
out/release_decision.json
```

建议只用 Python 标准库。脚本接口可以定为：

```bash
python evaluator.py \
  --gold fixtures/gold_labels.jsonl \
  --predictions fixtures/predictions_run_001.jsonl \
  --output-dir out
```

第一版必须实现的 5 个函数：

```python
def read_jsonl(path) -> list[dict]: ...
def join_valid(gold_rows, prediction_rows) -> tuple[list[tuple[dict, dict]], list[dict]]: ...
def validate_raw_output(raw_output: str) -> dict: ...
def aggregate(pairs, validation_results, input_errors) -> dict: ...
def evaluate_gates(metrics: dict, thresholds: dict) -> dict: ...
```

`validate_raw_output` 的最小契约如下：

- `raw_output` 必须是字符串，且整段能 `json.loads`；
- 只允许 `status` 与 `evidence` 两个字段；
- `status ∈ {已退款, 退款处理中, unknown}`；
- `evidence` 必须是字符串；
- `status == "unknown"` 时，`evidence == ""`；
- 其余具体状态的 evidence 必须是输入 `text` 中可定位的连续子串。

`join_valid` 不能静默吞掉四类问题：重复 gold `sample_id`、重复 prediction `sample_id`、valid gold 缺 prediction、prediction 找不到 valid gold。它们都应写入 `failures.jsonl`，并影响覆盖率门槛；**不能自动补成 unknown**。

本次 6 条 fixture 就复用 P1–P6，验收时应得到：

```text
total_valid_gold = 6
prediction_coverage = 6 / 6 = 1.0
contract_valid_rate = 5 / 6
end_to_end_accuracy = 3 / 6 = 0.5
退款处理中 → 已退款 = 1 / 2 = 0.5
已退款 → unknown = 1 / 2 = 0.5
```

`failures.jsonl` 至少能定位 P4（业务误报）、P5（`unknown` 的 evidence 非空）和 P6（业务误报）。发布阈值先写死为：覆盖率 `≥1.0`、契约合法率 `≥0.99`、处理中→已退款 `≤0.02`、已退款→unknown `≤0.10`，因此 fixture 的 `allowed` 应为 `false`，并列出所有失败门槛。

本轮最小任务：先写 `read_jsonl`、`join_valid` 和 P1–P6 两个 fixture 文件；完成后把这三部分代码或运行输出贴来。我会按上面的验收口径 review，再带你补 `validate_raw_output` 和聚合指标。

常见坑：不要在读 prediction 时提前 `json.loads(raw_output)`；否则你会丢失“模型没有按 JSON 输出”这一类真实失败。

### 2026-09-07T06:31:22Z｜member

- 评论 ID：8cfea367-a4bb-40fd-8e52-1e2654bc9ad2
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

你直接写吧，我看看应该怎么写，我没啥概念

### 2026-09-07T06:32:52Z｜agent

- 评论 ID：042bf8f2-1b81-4b1c-82db-2532ef54c4e9
- 父评论 ID：8cfea367-a4bb-40fd-8e52-1e2654bc9ad2

我已经把一版可直接运行的最小评测器写好了。它不依赖第三方库，也没有接模型 API；只做“读取已保存的模型答卷 → 校验 → 统计 → 发布判断”。

文件如下：

```text
evaluator.py                         主程序
fixtures/gold_labels.jsonl           6 条人工 gold（P1–P6）
fixtures/predictions_run_001.jsonl   同一轮模型原始答卷
test_evaluator.py                    2 个自动化测试
```

运行命令（这里要显式用 `python3`，环境中的 `python` 是 Python 2）：

```bash
python3 evaluator.py \
  --gold fixtures/gold_labels.jsonl \
  --predictions fixtures/predictions_run_001.jsonl \
  --output-dir out

python3 -m unittest -v test_evaluator
```

已验证结果：

```text
2 / 2 测试通过
coverage = 6/6 = 1.0
JSON/schema 合法率 = 5/6 = 83.3%
端到端正确率 = 3/6 = 50%
处理中 → 已退款 = 1/2 = 50%
已退款 → unknown = 1/2 = 50%
allowed = false
```

生成的三个文件分别做了你前面定义的事：

- `out/failures.jsonl`：P4、P6 是 `task_incorrect`；P5 是 `schema_contract`，原因是 `unknown` 却带了非空 evidence。
- `out/metrics.json`：放全量分母、覆盖率、混淆矩阵和两项方向性风险。
- `out/release_decision.json`：只根据 metrics 与门槛给出 `allowed=false` 和三个失败门槛。

建议按这个顺序读代码，不需要一次看懂全部：

1. `run()`：主流程，正好就是 Day12 的 `读取 → 配对 → 校验/统计 → 门槛判断 → 写文件`。
2. `validate_raw_output()`：最核心的一小段；它先判断 `raw_output` 是否是合法 JSON，再判断状态枚举、字段、`unknown` 空 evidence 和证据是否来自原文。
3. `join_valid()`：理解为什么“缺失预测”不是 `unknown`。它把缺失、重复、找不到 gold 记录为 `input_completeness`，并让覆盖率门槛失败。
4. `aggregate()`：只将契约通过的预测放进混淆矩阵；但端到端正确率仍以全部 valid gold 为分母。

另有一个测试专门删掉 P6 的 prediction，验证它会成为 `input_completeness`、coverage 变成 `5/6`，而不是被自动填为 `unknown`。

下一步你先只做一个小练习：打开 `validate_raw_output()`，用自己的话回答——为什么代码要先检查 `raw_output` 是字符串并执行 `json.loads`，而不是直接访问 `status` 字段？回复一两句话即可。

### 2026-09-07T06:33:45Z｜member

- 评论 ID：1bb3d594-ee38-42ee-b41f-34d39c0ee1bd
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

我想看具体的代码实现细节

### 2026-09-07T06:34:24Z｜agent

- 评论 ID：de0cff99-97ae-4845-b278-440fbb7fe8f6
- 父评论 ID：1bb3d594-ee38-42ee-b41f-34d39c0ee1bd

可以。下面不讲抽象概念，直接按实际代码的执行顺序拆开。你可以一边看 `evaluator.py`，一边对照这份说明。

整个入口只有这一段：

```python
def run(gold_path, prediction_path, output_dir):
    gold_rows = read_jsonl(gold_path)
    prediction_rows = read_jsonl(prediction_path)
    pairs, input_errors, total_valid_gold = join_valid(gold_rows, prediction_rows)
    metrics, failures = aggregate(pairs, input_errors, total_valid_gold)
    decision = evaluate_gates(metrics, DEFAULT_THRESHOLDS)

    write_jsonl(output / "failures.jsonl", failures)
    write_json(output / "metrics.json", metrics)
    write_json(output / "release_decision.json", decision)
```

这就是我们前面定的 `读取 → 配对/完整性检查 → 校验与统计 → 门槛决策 → 写三个输出文件`。`run()` 自己不做业务判断，只负责编排各步骤。

### 1. 读取 JSONL：一行就是一条样本

```python
def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(row)
    return rows
```

JSONL 不是“一个很大的 JSON 数组”，而是每行独立一个 JSON 对象。比如 `gold_labels.jsonl` 的一行是一个样本 P4；读取后得到 Python 的 `dict`。

这里对**输入文件本身**采取 fail-fast：如果第 17 行坏了，直接报清楚路径和行号。因为 gold/prediction 文件已经坏了，继续算出的任何指标都不可信。

注意它只解析 JSONL 文件的外层行；此时 prediction 里的 `raw_output` 仍然只是字符串，例如：

```json
{"raw_output": "{\"status\": \"已退款\", \"evidence\": \"退款已原路退回\"}"}
```

不能在这里解析里面那层字符串，否则会把模型“没按 JSON 输出”的失败提前抹掉。

### 2. 按 `sample_id` 配对，并把管道问题留下证据

`join_valid()` 先只保留 `split == "valid"` 的 gold：

```python
valid_gold = [row for row in gold_rows if row.get("split") == "valid"]
prediction_by_id = {
    str(row.get("sample_id", "")): row
    for row in prediction_rows
}
```

然后逐条遍历 gold：

```python
for gold in valid_gold:
    sample_id = str(gold.get("sample_id", ""))
    if sample_id in gold_duplicates or sample_id in prediction_duplicates:
        continue

    prediction = prediction_by_id.get(sample_id)
    if prediction is None:
        errors.append({
            "failure_type": "input_completeness",
            "failure_reason": "valid gold has no prediction",
            "sample_id": sample_id,
        })
        continue
    pairs.append((gold, prediction))
```

关键点是 `prediction is None`。如果 P6 没有 prediction，代码不会创建：

```python
{"status": "unknown", "evidence": ""}
```

而是创建一条输入完整性失败。因为“模型调用超时”与“模型读完文本后判断 unknown”是两件完全不同的事。前者影响 coverage，后者才是业务分类。

同一个函数还会记录：

- 同一个 `sample_id` 在 gold 中出现两次；
- 同一个 `sample_id` 在 prediction 中出现两次；
- prediction 找不到任何 valid gold。

这些记录会进入 `failures.jsonl`，不能静默跳过。

### 3. `validate_raw_output()`：把模型原始字符串分三层检查

函数开头先定义一个默认“失败”结果：

```python
result = {
    "json_valid": False,
    "contract_valid": False,
    "parsed_prediction": None,
    "failure_reason": None,
}
```

然后是第一层——原始输出必须是字符串、且可解析为 JSON：

```python
if not isinstance(raw_output, str):
    result["failure_reason"] = "raw_output must be a string"
    return result

try:
    parsed = json.loads(raw_output)
except json.JSONDecodeError:
    result["failure_reason"] = "raw_output is not valid JSON"
    return result

result["json_valid"] = True
result["parsed_prediction"] = parsed
```

例如模型返回 `已退款，正在处理`，它不是 JSON，所以停在这里；这是一条 `json` failure，不会假装模型预测了“已退款”。

第二层是字段和类型的契约：

```python
if not isinstance(parsed, dict):
    ...
elif set(parsed) != {"status", "evidence"}:
    ...
elif parsed["status"] not in STATUSES:
    ...
elif not isinstance(parsed["evidence"], str):
    ...
```

`set(parsed) != {"status", "evidence"}` 的含义是：必须**恰好**有两个字段。少一个字段或额外塞入 `confidence`，这版契约都会失败。生产中是否允许扩展字段是产品决定；这里严格是为了让训练、推理和评测输出一致。

第三层是退款业务规则：

```python
elif parsed["status"] == "unknown" and parsed["evidence"] != "":
    result["failure_reason"] = "unknown requires empty evidence"
elif parsed["status"] != "unknown" and (
    not parsed["evidence"] or parsed["evidence"] not in text
):
    result["failure_reason"] = (
        "concrete status requires evidence as a contiguous text substring"
    )
else:
    result["contract_valid"] = True
```

P5 命中第一条：它输出了 `unknown`，却把“订单已取消”写进 evidence，所以 JSON 合法、契约不合法。

P4 则会通过这个函数：它的 JSON、字段和 evidence 都合规；它错在把“退款申请已受理”分类成“已退款”，这是下一步的业务判断错误，而非格式错误。

### 4. 统计：为何混淆矩阵与端到端准确率分母不同

`aggregate()` 逐个 pair 调用刚才的校验函数：

```python
validation = validate_raw_output(
    prediction_row.get("raw_output"),
    gold_row.get("text", ""),
)

if not validation["contract_valid"]:
    failure_type = "json" if not validation["json_valid"] else "schema_contract"
    failures.append(...)
    continue
```

`continue` 的含义是：P5 只记录失败，不进混淆矩阵，因为没有“安全可用”的结构化 prediction。

只有契约通过，才进入矩阵和方向性计数：

```python
confusion[gold_status][predicted_status] += 1

if gold_status == "退款处理中" and predicted_status == "已退款":
    processing_to_refunded += 1
if gold_status == "已退款" and predicted_status == "unknown":
    refunded_to_unknown += 1
```

但最终准确率的分母固定为全部 valid gold：

```python
denominator = total_valid_gold
end_to_end_accuracy = end_to_end_correct_count / denominator
json_schema_valid_rate = contract_valid_count / denominator
```

因此 P5 即使不进混淆矩阵，仍在 `6` 这个分母中，让端到端准确率和契约合法率下降。这避免了“删掉难样本，指标变好”的假象。

方向性指标同样使用对应 gold 类的总数作分母：

```python
refund_processing_to_refunded_rate = (
    processing_to_refunded / processing_total
)
```

P2、P4 都是 gold=退款处理中，所以 `processing_total=2`；P4 被预测为已退款，因此结果 `1/2=50%`。

### 5. 发布判断不能混在统计函数里

阈值是单独的数据：

```python
DEFAULT_THRESHOLDS = {
    "prediction_coverage_min": 1.0,
    "json_schema_valid_rate_min": 0.99,
    "refund_processing_to_refunded_rate_max": 0.02,
    "refunded_to_unknown_rate_max": 0.10,
}
```

`evaluate_gates()` 只是依次检查，不会修改 metrics：

```python
if metrics["prediction_coverage"] < thresholds["prediction_coverage_min"]:
    failed_gates.append("prediction_coverage")

if metrics["json_schema_valid_rate"] < thresholds["json_schema_valid_rate_min"]:
    failed_gates.append("json_schema_valid_rate")

return {
    "allowed": not failed_gates,
    "failed_gates": failed_gates,
    "thresholds": thresholds,
}
```

所以 `metrics.json` 回答“测到了什么”，`release_decision.json` 回答“按哪个阈值能不能发”。以后调整阈值时，可以重算 decision，不必重新调用模型。

### 6. 两个测试实际保护了什么

第一个测试运行 P1–P6，并断言结果恰好是我们手算的：

```python
self.assertAlmostEqual(metrics["json_schema_valid_rate"], 5 / 6)
self.assertEqual(metrics["end_to_end_accuracy"], 0.5)
self.assertEqual(metrics["refund_processing_to_refunded_rate"], 0.5)
self.assertFalse(result["decision"]["allowed"])
```

第二个测试故意移除 P6 的 prediction：

```python
if json.loads(line)["sample_id"] != "P6"
```

然后检查三件事：coverage 变成 `5/6`、发布被 `prediction_coverage` 阻断、P6 的失败类型是 `input_completeness`。这正是在防止“超时自动算 unknown”的常见错误。

现在你不需要背全部。请只看 `validate_raw_output()` 里这两种失败：

1. `raw_output = "已退款"`
2. `raw_output = '{"status": "unknown", "evidence": "订单已取消"}'`

回复我：它们分别会被标记为 `json` 还是 `schema_contract`？再各用一句话说原因。

### 2026-09-07T07:21:00Z｜member

- 评论 ID：1ca45c49-031d-4f0f-bf8c-65d7c1cf950c
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

1：json

2：schema\_contract

### 2026-09-07T07:21:13Z｜agent

- 评论 ID：4fb205eb-599e-44e1-b575-79cfabf55152
- 父评论 ID：1ca45c49-031d-4f0f-bf8c-65d7c1cf950c

完全正确。

1. `raw_output = "已退款"` → `json`：它只是普通文本，`json.loads("已退款")` 无法解析成 JSON；评测器还拿不到 `status` / `evidence` 这两个字段。

2. `raw_output = '{"status": "unknown", "evidence": "订单已取消"}'` → `schema_contract`：它可以成功 JSON 解析，也有两个字段；但违反了业务输出契约——`unknown` 必须对应空字符串 evidence。

记忆口诀：**不能解析，是 JSON failure；能解析但不符合字段、枚举、类型或 evidence 规则，是 schema/contract failure；两者都通过但预测与 gold 不同，才是 task_incorrect。**

再看一个边界题：原文是“退款申请已受理，预计 3–5 个工作日原路退回”，模型输出：

```json
{"status": "已退款", "evidence": "退款申请已受理"}
```

假设 gold 是“退款处理中”，它最终属于哪一类：`json`、`schema_contract` 还是 `task_incorrect`？并说一句原因。

### 2026-09-07T07:22:19Z｜member

- 评论 ID：5471614d-b268-433c-a366-23f4a3078a61
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

task\_incorrect，结果预测错误

### 2026-09-07T07:22:34Z｜agent

- 评论 ID：c4265d8d-507c-4715-b1c2-d8ae50b442ba
- 父评论 ID：5471614d-b268-433c-a366-23f4a3078a61

正确，是 `task_incorrect`。

这条输出其实已经通过前两层：

- 它是可解析的 JSON；
- 字段正好是 `status` 和 `evidence`；
- `status=已退款` 属于允许枚举；
- evidence 是字符串，而且“退款申请已受理”确实是原文里的连续片段。

因此 `validate_raw_output()` 会返回：

```python
{
    "json_valid": True,
    "contract_valid": True,
    "parsed_prediction": {
        "status": "已退款",
        "evidence": "退款申请已受理",
    },
}
```

之后才与 gold 的 `退款处理中` 比较。二者不同，所以 `aggregate()` 写入：

```python
{
    "failure_type": "task_incorrect",
    "failure_reason": "predicted status differs from gold",
}
```

它会进入混淆矩阵的 `退款处理中 → 已退款` 格，并计入高风险方向性错误 `1/2=50%`。

和 P5 对照：

```text
P4：结构化输出可用，但分类错 → task_incorrect → 进入混淆矩阵
P5：unknown 的 evidence 不合规 → schema_contract → 不进入混淆矩阵
```

下一题只考你是否能把这三层判断连起来：如果模型输出合法 JSON、字段和 evidence 都合规，且预测 status 与 gold 也相同，那么它会不会出现在 `failures.jsonl`？为什么？

### 2026-09-07T07:26:22Z｜member

- 评论 ID：9e4effc5-4906-4a8a-9e6b-8a6731c4db67
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

会的，evidence可能不符合需求，或者自己捏造了事实也是不行的

### 2026-09-07T07:26:38Z｜agent

- 评论 ID：72123c21-0163-4eb3-8cd7-a171f4e96b00
- 父评论 ID：9e4effc5-4906-4a8a-9e6b-8a6731c4db67

你的担心本身是对的：如果 evidence 不符合规则，或是模型捏造、无法在原文中定位的事实，那么必须失败，属于 `schema_contract`，并写入 `failures.jsonl`。

但这道题的前提已经明确说：**合法 JSON、字段和 evidence 都合规，且预测 status 与 gold 相同**。在这个前提下，答案应是：**不会出现在 `failures.jsonl`**。

原因是三层都通过了：

```text
JSON 可解析
→ schema / evidence 合规（没有捏造、可从原文定位）
→ status 与 gold 相同
→ 成功样本，不写 failure
```

`failures.jsonl` 只收“出了什么问题”的样本；正确样本虽然不写入该文件，但会贡献给 `json_schema_valid_count`、混淆矩阵的对角线和 `end_to_end_correct_count`。

你刚才的回答很有价值，因为它正好说明：不能只看到 `status` 对了就当成功；证据校验必须先通过。把条件改成“status 与 gold 相同，但 evidence 是模型杜撰的”，结果就会从“成功”变成 `schema_contract`。

最后一个判断题：模型输出 status 与 gold 相同，但 evidence 不在原文中出现。它应进入混淆矩阵吗？为什么？

### 2026-09-07T07:29:33Z｜member

- 评论 ID：aabd003c-0202-4f8a-af32-4d2c6399f825
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

不应该，混淆矩阵是通过status划分的

### 2026-09-07T07:29:47Z｜agent

- 评论 ID：81bd8c59-742c-4cdb-a6cf-0c380d0cd76c
- 父评论 ID：aabd003c-0202-4f8a-af32-4d2c6399f825

结论正确：**不应该进入混淆矩阵**。

理由需要稍微修正：混淆矩阵确实按 `gold status → predicted status` 计数，但它的前提是 prediction 已经通过 JSON 和 schema/contract 校验。

这个样本的 evidence 不在原文中，违反“具体 status 的 evidence 必须是原文连续子串”的规则，因此：

```text
JSON 可解析
→ evidence 不合规
→ schema_contract failure
→ 写入 failures.jsonl
→ 不进入混淆矩阵
```

两条对照即可记住：

```text
P4：evidence 合规，status 错 → task_incorrect → 进入混淆矩阵
P5：status 看起来可用，但 evidence 不合规 → schema_contract → 不进入混淆矩阵
```

所以更精确的一句话是：**混淆矩阵只统计“有合法、可安全使用的结构化 prediction”的 status 对；evidence 失败时，连可入矩阵的 prediction 都不成立。**

这部分判断你已经掌握了。下一步建议把它落回代码：请看 `aggregate()` 的这一段，回答 `continue` 为什么必须放在 `confusion[gold_status][predicted_status] += 1` 之前？

```python
if not validation["contract_valid"]:
    failures.append(...)
    continue

confusion[gold_status][predicted_status] += 1
```

### 2026-09-07T07:55:07Z｜member

- 评论 ID：f0429408-0b89-40b3-9718-85e280328ed9
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

因为contract\_valid这部分内容不进入到混淆矩阵中

### 2026-09-07T07:55:23Z｜agent

- 评论 ID：9ae1c3fd-9092-4475-9179-03d504a03213
- 父评论 ID：f0429408-0b89-40b3-9718-85e280328ed9

对，核心理解正确。措辞再精确一点：是 **`contract_valid=False` 的样本不进入混淆矩阵**；`contract_valid=True` 的样本才继续进入。

```python
if not validation["contract_valid"]:
    failures.append(...)
    continue  # 立刻跳到下一条样本

# 只有 contract_valid=True 才会执行到这里
confusion[gold_status][predicted_status] += 1
```

`continue` 必须放在前面有两个原因：

1. 无效输出不应污染 status 矩阵。例如 P5 的 `unknown + 非空 evidence` 不能被计为 `unknown → unknown`；否则会把契约失败伪装成一次正确分类。
2. 有些 JSON failure 根本没有可用的 `predicted_status`，或解析结果不是对象；若先访问 `parsed["status"]`，程序可能报错。

所以这段代码既保证**指标口径正确**，也保证**程序在脏输出下不会崩溃**。

Day13 的核心代码逻辑你已经串起来了：输入完整性 → JSON/契约校验 → 业务正确性 → 指标/发布门槛。下一步可以进入一个更实用的扩展：给评测器补一条测试，专门验证“evidence 不在原文”的样本一定不进混淆矩阵。你想让我直接写出这条测试并讲解，还是先由你尝试写？

### 2026-09-07T08:10:48Z｜member

- 评论 ID：3a8acf97-888e-4a7e-9b21-ffd8f65cc18f
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

source\_page=9502的过滤逻辑里有曝光过滤么

### 2026-09-07T08:11:14Z｜agent

- 评论 ID：37bdccc9-e01a-4e56-9e2c-c467757212e8
- 父评论 ID：3a8acf97-888e-4a7e-9b21-ffd8f65cc18f

我目前无法确认。已检查当前工作区，没有找到包含 `source_page`、`9502`、`曝光`/`impression`/`exposure` 的代码或 SQL；本次运行也没有可用的受控数仓查询入口，因此不能根据字段名臆测“有”或“没有”曝光过滤。

把该过滤逻辑所在的 SQL、任务名、仓库路径，或相关代码片段贴出来即可。我会逐条核对 `WHERE` / `JOIN` 条件，明确说明：

- `source_page = 9502` 是否只做来源页过滤；
- 是否存在曝光事件、曝光次数/状态，或曝光日志 Join 的附加条件；
- 这些条件在漏斗中属于曝光前过滤还是曝光后过滤。

### 2026-09-07T08:29:24Z｜member

- 评论 ID：f081e367-46a0-449a-bf0d-3e971846f700
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

打错了，继续吧

### 2026-09-07T08:30:06Z｜agent

- 评论 ID：e21da51d-aa39-408d-bb80-89bec46633c9
- 父评论 ID：f081e367-46a0-449a-bf0d-3e971846f700

已继续，并补上了一条新的自动化测试；`python3 -m unittest -v test_evaluator` 现为 **3/3 通过**，语法检查也通过。当前工作区不是 Git 仓库，因此没有 PR。

新增测试验证的场景是：P2 原本应预测“退款处理中”，但把 evidence 改成原文中不存在的“退款已经到账”。

```python
p2 = next(row for row in predictions if row["sample_id"] == "P2")
p2["raw_output"] = json.dumps(
    {"status": "退款处理中", "evidence": "退款已经到账"},
    ensure_ascii=False,
)
```

这刻意保持 `status` 正确，只改变 evidence。因为“退款已经到账”不在 P2 原文“退款申请已受理，预计 3–5 个工作日原路退回”里，校验会命中：

```python
not parsed["evidence"] or parsed["evidence"] not in text
```

因此测试断言三件事：

```python
# 1. 不是 task_incorrect，而是证据契约失败
self.assertEqual(p2_failure["failure_type"], "schema_contract")

# 2. 不被伪装成“处理中 → 处理中”的正确分类
self.assertEqual(
    result["metrics"]["confusion_matrix"]["退款处理中"]["退款处理中"],
    0,
)

# 3. 契约合法率由 5/6 进一步降为 4/6
self.assertAlmostEqual(result["metrics"]["json_schema_valid_rate"], 4 / 6)
```

这条测试正是在守住刚才的规则：即使 status 与 gold 相同，只要 evidence 伪造或不可定位，`contract_valid=False`，就必须写入 `failures.jsonl`，且 `continue` 会阻止它进入混淆矩阵。

到这里，Day13 的三类关键异常都有对应自动化保护：

```text
P6 缺 prediction        → input_completeness / coverage 失败
P5 unknown 非空 evidence → schema_contract / 不入混淆矩阵
P2 evidence 不在原文    → schema_contract / 不入混淆矩阵
```

下一步我建议进入发布判断的代码：为什么 `evaluate_gates()` 要把 `failed_gates` 收集成列表，而不是遇到第一个失败就直接返回 `allowed=false`？你先凭直觉回答一句即可。

### 2026-09-07T08:34:39Z｜member

- 评论 ID：31a10cf8-2e3f-4010-b757-c37cac0511b6
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

可以直观的看到当前的结果哪些地方不满足条件，如果一不满足就返回结果，会不知道剩下的指标结果如何，是不是也存在问题

### 2026-09-07T08:35:15Z｜agent

- 评论 ID：273d7c38-4f0d-45c6-a100-685378493cc4
- 父评论 ID：31a10cf8-2e3f-4010-b757-c37cac0511b6

完全正确。发布判定应采用：**全量计算 / 全量诊断 / 最后统一决策**，而不是“遇到第一个失败就短路”。

`allowed` 可以理解为所有门槛的 AND：任意一项失败，最终就不能发布；但 `failed_gates` 要保留所有失败项，才能一次看清还存在哪些风险、按严重度排修复优先级，也避免每修一项就要重新跑一次才发现下一项。

以当前 fixture 为例，最终虽然已经确定 `allowed=false`，仍应继续报告这 3 项：

```text
json_schema_valid_rate
refund_processing_to_refunded_rate
refunded_to_unknown_rate
```

其中任何一项都不能抵消另一项。`metrics.json` 保存全部实际指标，`release_decision.json.failed_gates` 保存全部不达标门槛；这正是让结果既能阻断发布、又能直观看到完整问题面的原因。

下一步看 `evaluate_gates()` 的关键写法：每一次检查只负责把失败 key 加进 `failed_gates`，最后统一用 `not failed_gates` 生成 `allowed`。你可以先看下面这两行，判断它们各自负责什么：

```python
failed_gates.append("json_schema_valid_rate")
allowed = not failed_gates
```

### 2026-09-07T08:43:57Z｜member

- 评论 ID：7b7518a6-3369-4a89-b17f-8d9180fe9387
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

第一行复杂把真实的结果记录到failed\_gates里，

如果最后没有记录的话，则allowed=True

### 2026-09-07T08:44:11Z｜agent

- 评论 ID：6d7fb48f-0a40-43a3-8b00-b2f4c5bb06b8
- 父评论 ID：7b7518a6-3369-4a89-b17f-8d9180fe9387

第二句完全正确：若所有检查后 `failed_gates` 仍是空列表，`not failed_gates` 就是 `True`，因此 `allowed=True`。

第一句只需改一个精确点：

```python
failed_gates.append("json_schema_valid_rate")
```

它记录的不是“真实指标值”，而是**未达标的门槛名称**。真实值（例如 `0.8333`）已经在 `metrics.json`；这里的 key 用于说明“哪条规则失败了”。只有当前指标低于对应最小阈值时，才会执行这行。

可以把职责记成：

```text
metrics.json       → 实际测得多少（如 0.8333）
failed_gates       → 哪些门槛没过（如 json_schema_valid_rate）
allowed            → failed_gates 是否为空
```

例如 `failed_gates = ["json_schema_valid_rate", "refunded_to_unknown_rate"]` 时，`not failed_gates` 为 `False`，所以不允许发布。

下一题：为什么这里应写 `allowed = not failed_gates`，而不需要手动遍历列表逐项判断？

### 2026-09-07T08:45:05Z｜member

- 评论 ID：eb87f32e-e800-4377-882e-cea130d7f9a2
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

有就不通过，不需要遍历

### 2026-09-07T08:45:19Z｜agent

- 评论 ID：9c6b096b-96da-4cb8-9a40-2b1c4471c328
- 父评论 ID：eb87f32e-e800-4377-882e-cea130d7f9a2

对，完全正确。只要列表里**有任意一个**失败门槛，就不允许发布；这里不必再逐项遍历。

Python 已经定义了列表的布尔语义：

```python
bool([])                                  # False
bool(["json_schema_valid_rate"])      # True

allowed = not failed_gates
```

因此：

```text
failed_gates = []              → not []              → True  → allowed=True
failed_gates = ["任一失败项"] → not ["任一失败项"] → False → allowed=False
```

注意这里的“不需要遍历”只针对**最终从失败列表推导 `allowed`** 这一步；前面仍然必须逐条检查每个门槛，才能把所有失败项完整加入 `failed_gates`。

你已经把这段的核心掌握了。接下来可以直接看完整的 `evaluate_gates()`：它为何先建 `failed_gates = []`，再依次比较 4 个门槛，最后返回三个字段。

### 2026-09-07T09:20:17Z｜member

- 评论 ID：7a0216c4-5e77-464d-852a-4af292b9094a
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

继续

### 2026-09-07T09:20:37Z｜agent

- 评论 ID：87eda3b2-9d2c-46d6-9ca2-25ab5263a9cc
- 父评论 ID：7a0216c4-5e77-464d-852a-4af292b9094a

下面是当前 `evaluate_gates()` 的完整逻辑。它只读取已经聚合好的 `metrics`，不重新解析模型输出；职责是把指标与阈值比较，产出发布决定。

```python
def evaluate_gates(metrics, thresholds):
    failed_gates = []

    if metrics["prediction_coverage"] < thresholds["prediction_coverage_min"]:
        failed_gates.append("prediction_coverage")

    if metrics["json_schema_valid_rate"] < thresholds["json_schema_valid_rate_min"]:
        failed_gates.append("json_schema_valid_rate")

    if metrics["refund_processing_to_refunded_rate"] is None or (
        metrics["refund_processing_to_refunded_rate"]
        > thresholds["refund_processing_to_refunded_rate_max"]
    ):
        failed_gates.append("refund_processing_to_refunded_rate")

    if metrics["refunded_to_unknown_rate"] is None or (
        metrics["refunded_to_unknown_rate"]
        > thresholds["refunded_to_unknown_rate_max"]
    ):
        failed_gates.append("refunded_to_unknown_rate")

    return {
        "run_id": metrics["run_id"],
        "allowed": not failed_gates,
        "failed_gates": failed_gates,
        "thresholds": thresholds,
    }
```

四项检查的口径分别是：

```text
prediction_coverage                    → 是否每个 valid gold 都得到 prediction
json_schema_valid_rate                 → 原始输出是否足够稳定地符合 JSON/契约
退款处理中 → 已退款 的错误率             → 是否把未完成退款误报为已退款
已退款 → unknown 的错误率                → 是否把已完成退款漏成无法判断
```

后两项写成 `is None or rate > max` 有一个重要含义：若验证集里根本没有该 gold 类，分母为 0，指标就没有定义（`None`），不能把“没测到风险”误判为“风险为 0”；正式发布时应先补足覆盖该场景的验证样本，因此也阻断发布。

结合当前 fixture：coverage=1.0 通过；但合法率 `5/6 < 0.99`，且两个方向性错误率均为 `0.5`，高于各自上限，所以最终会收集三个失败项并得到 `allowed=False`。

小题：如果 `退款处理中 → 已退款` 的错误率正好是 `0.02`，而上限也是 `0.02`，按这段代码它会通过还是失败？为什么？

### 2026-09-07T09:21:38Z｜member

- 评论 ID：66667b1f-8bff-4422-a499-be2960c2bdbf
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

不通过，因为要小于才行

### 2026-09-07T09:21:55Z｜agent

- 评论 ID：c9a6f9c2-be38-41dc-9ed6-8dee2c96f70d
- 父评论 ID：66667b1f-8bff-4422-a499-be2960c2bdbf

这里要以代码的比较符为准：当前实现中，**正好等于 `0.02` 会通过**。

```python
if rate > max:  # 只有严格大于上限才失败
    failed_gates.append(...)
```

代入数值：`0.02 > 0.02` 是 `False`，不会加入 `failed_gates`。因此当前门槛语义是：

```text
rate ≤ 0.02  → 通过（最多 2%）
rate > 0.02  → 失败
```

你说的“要小于才行”对应另一种同样合理、但更严格的业务规则：

```python
if rate >= max:  # 等于 2% 也失败
```

这时语义变为“必须低于 2%”。关键不是哪一种天然更对，而是要把产品门槛的中文定义、代码比较符和边界测试保持一致，避免“文档写最多 2%，代码却要求低于 2%”的歧义。

小题：若你在发布规范里写“错误率不得超过 2%”，应选择 `>` 还是 `>=`？

### 2026-09-07T09:23:39Z｜member

- 评论 ID：ba14ef02-830b-4043-9e77-cf553539f2cb
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

&gt;=

### 2026-09-07T09:23:53Z｜agent

- 评论 ID：56fb8cda-547b-4c43-9d97-ec29a85c5da6
- 父评论 ID：ba14ef02-830b-4043-9e77-cf553539f2cb

这里应选 **`>`**，不是 `>=`。

“错误率**不得超过** 2%”的数学含义是：

```text
错误率 ≤ 2%  → 通过
错误率 > 2%  → 不通过
```

所以失败条件应写为：

```python
if rate > 0.02:
    failed_gates.append(...)
```

`>=` 对应的是“错误率必须**低于** 2%”或“错误率达到 2% 即不通过”。

一句记忆：

```text
不超过 / 至多 / 最大为 → ≤ → 失败时用 >
低于 / 小于           → < → 失败时用 >=
```

下一步我们可以把这个边界真正写成自动化测试：同一指标分别为 `0.019`、`0.02`、`0.021` 时，预期的发布结果各是什么？

### 2026-09-07T09:25:22Z｜member

- 评论 ID：ac7322da-b598-459f-ac08-69b3951a81f6
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

通过、通过、不通过

### 2026-09-07T09:25:56Z｜agent

- 评论 ID：30c72fbe-7cf5-4f75-94a9-7d2c99b6b77e
- 父评论 ID：ac7322da-b598-459f-ac08-69b3951a81f6

完全正确：

```text
0.019 → 通过
0.020 → 通过
0.021 → 不通过
```

因为规则是“不得超过 2%”，也就是 `rate ≤ 0.02`。我已把这三个值加入自动化边界测试，直接验证当前 `rate > max` 的实现：

```python
cases = (
    (0.019, True),
    (0.020, True),
    (0.021, False),
)
```

验证结果：`python3 -m unittest -v test_evaluator` 为 **4/4 通过**，语法检查也通过。

这类测试很重要：它把自然语言的发布规范固定为可执行行为，未来即使有人不小心把 `>` 改成 `>=`，测试也会立刻失败。

Day13 的发布门槛部分已经完成：指标计算、失败项汇总、统一发布判定，以及边界条件保护。下一步可以进入“阈值如何制定”：为什么 2% 不能只凭感觉设定，以及怎样用风险、样本量和历史基线确定它。

### 2026-09-07T09:35:46Z｜member

- 评论 ID：1b17d380-a57f-41e1-a6ed-7ff18d8c317b
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

继续吧

### 2026-09-07T09:36:06Z｜agent

- 评论 ID：4e3dc35c-4173-4787-8131-015a7c636e36
- 父评论 ID：1b17d380-a57f-41e1-a6ed-7ff18d8c317b

## Day13｜阈值不是拍脑袋：先从风险反推

`2%` 不是一个天然正确的数字。制定门槛时，先问的不是“模型通常能做到多少”，而是：**这类错误在真实业务中造成什么后果，最多能接受多少？**

以我们定义的两类方向性错误为例：

```text
退款处理中 → 已退款   把尚未完成的退款说成已到账
已退款 → unknown      把已完成的退款说成无法判断
```

两者都可能是分类错，但风险并不对称。第一类可能诱发用户认为钱已经到账，从而投诉或做出错误操作，通常应设得更严；第二类更多是未能给出确定答复，仍需看是否可以安全转人工。因此不能只盯着总体准确率，例如“95% 正确”并不能保证高风险的那 5% 不集中在第一类。

制定一个门槛至少要同时满足三件事：

```text
1. 风险定义：哪种错误最不能接受，为什么？
2. 验证覆盖：验证集中必须有足够多的该场景样本。
3. 统计可信度：样本量足以让观察到的错误率有参考价值。
```

第 2、3 点解释了为什么现在的 6 条 fixture 不能真的推出“≤2%”：若某个类别只有 2 条样本，一次错误就是 `1/2 = 50%`；而要观察到 2% 级别，验证集中该类别至少需要几十到上百条，具体数量还要结合允许的不确定性决定。小样本的 `0%` 往往只表示“这次没观察到错误”，不等于真实错误率就是 0%。

实际落地可先写一张阈值说明表：

```text
指标                              门槛       业务理由                 最小验证覆盖
处理中 → 已退款                   ≤ 2%       高风险误告知              需单独规定
已退款 → unknown                  ≤ 10%      保守漏答，可转人工          需单独规定
JSON/schema 合法率                ≥ 99%      下游无法稳定消费输出        全量 valid
prediction coverage               = 100%     不能静默漏答               全量 valid
```

其中“最小验证覆盖”不能只写一个好看的数，必须在团队可接受的置信要求下确定；没有足够样本时，正确做法是阻断或标记“证据不足”，而不是把小样本 0% 当作通过证据。

小题：如果验证集中只有 3 条 `已退款` 样本，模型这 3 条全答对，`已退款 → unknown` 错误率是 0%。你认为此时可以据此确信真实线上该错误率一定低于 10% 吗？用一句话说明理由。

### 2026-09-07T09:37:43Z｜member

- 评论 ID：739d0c8c-908d-4f7f-a93f-dfca59579b25
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

不能，样本量太小，不置信

### 2026-09-07T09:38:03Z｜agent

- 评论 ID：a36582c6-c247-455e-95b7-b2446bce576d
- 父评论 ID：739d0c8c-908d-4f7f-a93f-dfca59579b25

正确：**不能确信**，因为样本量只有 3 条，`0/3` 的观察结果不够稳定。

可以把它量化得更直观一些：即使真实线上错误率其实正好是 10%，随机抽 3 条却一个错误都没看到的概率仍是：

```text
(1 - 0.10)^3 = 0.9^3 = 72.9%
```

也就是说，`0/3` 与“真实错误率为 10%”并不矛盾，更无法证明它低于 10%。

若采用常见的单侧 95% 置信上界，观察到 0 次错误、样本数为 3 时，真实错误率的上界约为：

```text
1 - 0.05^(1/3) ≈ 63.2%
```

这不是说真实错误率就是 63.2%，而是说数据太少，仍无法排除相当高的真实错误率。

反过来，若希望在“观察到 0 次错误”时，以 95% 把握证明真实错误率低于 10%，需要至少满足：

```text
0.9^n ≤ 0.05
n ≥ 28.4
```

因此至少需要 **29 条**独立且有代表性的 `已退款` 验证样本，并且这 29 条都没有被预测成 `unknown`。这只是“0 错误”的最低样本量推导；实际还应按不同来源、时间、文本模式和近重复 group 做覆盖。

小题：如果某项高风险错误率的目标是 ≤2%，并且验证集中观察到 0 次该错误；为了以同样 95% 把握证明其真实错误率低于 2%，你认为需要的样本数会比 29 更多还是更少？为什么？

### 2026-09-07T09:42:40Z｜member

- 评论 ID：6bbd5bf2-a2a7-4841-b181-ca5bbcef0a6e
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

（1-0.98）^n &lt;= 0.05，大于29次

### 2026-09-07T09:43:12Z｜agent

- 评论 ID：dce83c47-5814-4d0c-a2b7-7d1591ab0515
- 父评论 ID：6bbd5bf2-a2a7-4841-b181-ca5bbcef0a6e

结论对：需要的样本数会**远多于 29 条**。但概率式要改一个关键点。

目标是检验真实错误率可能为 `2%` 的情况；若真实错误率是 2%，一条样本**不出错**的概率是 `1 - 0.02 = 0.98`，而不是 `1 - 0.98 = 0.02`。观察到 `n` 条都不出错的概率是：

```text
(1 - 0.02)^n = 0.98^n
```

为了让“真实错误率仍有 2% 却恰好 0 错误”的概率不超过 5%，应满足：

```text
0.98^n ≤ 0.05
n ≥ log(0.05) / log(0.98)
n ≥ 148.28
```

因此至少需要 **149 条**独立且有代表性的该高风险场景验证样本，并且观察到 0 次错误，才能在这个简化的单侧 95% 口径下证明真实错误率低于 2%。

直觉上也合理：要证明“低于 2%”比证明“低于 10%”严格得多，所以所需样本从 29 增加到约 149。

注意这是假设样本独立且能代表上线分布的最低计算；若大量样本只是同一工单的近重复改写，实际有效样本量会更低。下一步可以讨论：为什么数据切分时要按 `cluster_id` 隔离近重复样本。

### 2026-09-07T09:56:51Z｜member

- 评论 ID：2d91ddc9-b9c5-423e-9050-a6d2a955ca60
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

因为近重复的样本来自于同一文档，把他放到训练集和验证集时，会使得验证集结果虚高

### 2026-09-07T09:57:10Z｜agent

- 评论 ID：b8e29631-5931-4495-9685-111b7c3d086c
- 父评论 ID：2d91ddc9-b9c5-423e-9050-a6d2a955ca60

完全正确：这会造成**训练—验证泄露**，让验证结果虚高，不能代表模型对真正新样本的泛化能力。

有一个措辞上的补充：近重复样本不一定来自同一份文档，也可能是同一工单的多轮描述、同一模板替换少量字段、同一事件的转述，或不同文档复用了几乎相同的文本。共同点是模型能从训练集的相似样本中“认出来”，而非真正学会处理新情况。

例如：

```text
训练集：退款申请已受理，预计 3–5 个工作日原路退回
验证集：您的退款已受理，预计 3 至 5 个工作日到账
```

若二者分到不同集合，模型在验证集上答对，可能只是记住了训练样本的模式；该成绩不能说明它面对新的退款表达也可靠。

因此切分规则应是按 `cluster_id`，而不是逐行随机切分：

```text
同一个 cluster_id 的所有样本 → 只能整体进入 train / valid / test 中的一个集合
```

`source_ticket_id` 可以帮助发现同工单泄露；`cluster_id` 则是更一般的“近重复簇”边界。这样验证集才更接近上线时尚未见过的新工单、新表达和新上下文。

小题：若同一 `cluster_id=C100` 有 8 条近重复样本，随机逐行切分后 6 条进训练、2 条进验证；正确的修复是把那 2 条验证样本删掉，还是把 C100 的全部 8 条统一放进同一个 split？为什么？

### 2026-09-07T10:03:16Z｜member

- 评论 ID：f9c96e28-2780-4c15-a236-1d0a0ef27338
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

把 C100 的全部 8 条统一放进同一个 split，数据都是有效数据，不用删除

### 2026-09-07T10:03:31Z｜agent

- 评论 ID：762c5e0d-5f71-48ed-9025-5af67a5b6d03
- 父评论 ID：f9c96e28-2780-4c15-a236-1d0a0ef27338

完全正确。修复方式是把 `C100` 的 **8 条全部放进同一个 split**，而不是删除其中 2 条。

```text
错误切分：C100 的 6 条 train + 2 条 valid  → 泄露，valid 虚高
正确切分：C100 的 8 条全在 train，或全在 valid，或全在 test
```

这些数据本身仍然有效；问题只在于它们不能跨集合同时出现。删除验证中的 2 条虽然临时消除了那两条的泄露，但会留下一个随意的、不完整的切分规则，也无端损失了可用于训练或独立评测的样本。

实际分配时还要结合集合职责：

- 若 `C100` 是普通、已充分覆盖的模式，可整体放训练集；
- 若它代表一个必须单独考察的高风险或长尾场景，可整体放验证集/测试集；
- 无论放哪儿，都必须保证该 `cluster_id` 不出现在另外两个集合。

这也解释了 `gold_labels.jsonl` 需要保存 `cluster_id`：它不仅是分析字段，更是数据切分与防泄露的约束依据。

小题：如果为了“补充训练数据”，把当前固定的验证集样本复制进训练集，即使仍保留原验证文件不变，评测结果还能信吗？为什么？

### 2026-09-07T10:04:51Z｜member

- 评论 ID：285e598a-7583-43f2-8b36-16fda96a71e5
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

不能，验证集数据放入训练集中，相当于让模型已经见过答案， 使得验证集上的结果虚高

### 2026-09-07T10:05:41Z｜agent

- 评论 ID：ce29feba-000a-48a7-aef9-1c0794867ebc
- 父评论 ID：285e598a-7583-43f2-8b36-16fda96a71e5

完全正确。把验证样本复制到训练集，即使验证文件本身一字未改，评测也**不能再相信**。模型已经在训练中见过这些输入及其目标答案/标签，验证分数会混入记忆效果，而不再衡量对未见样本的泛化能力。

这里要隔离的不只是 `gold.status`，而是整个验证资产：

```text
验证样本的输入文本、目标标签、evidence、source_ticket_id 对应内容、近重复 cluster
都不能进入训练集
```

因此要同时满足两条：

```text
1. exact-match 隔离：验证样本本身不能进入训练。
2. group 隔离：与验证样本同一 cluster_id 的近重复样本也不能进入训练。
```

还有一个更工程化的补充：验证集可以用于反复比较 prompt、模型版本和超参数；但一旦长期反复根据它挑方案，它也会被“间接过拟合”。所以实践中应保留一个从不参与日常选择的固定 `test` 集，最终只在准备发布或阶段性验收时使用。

```text
train → 训练模型
valid → 调参、选 prompt / checkpoint、检查门槛
test  → 最终独立验收；不回灌训练，也不用于日常反复调参
```

小题：模型团队连续 30 次改 prompt，每次都选 valid 分数最高的一版，最后才跑一次从未使用的 test。为什么最终 test 分数可能明显低于 valid 分数？

### 2026-09-07T10:08:01Z｜member

- 评论 ID：7bc1c781-2034-4b5c-9b2f-79d4e980db81
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

过拟合，改prompt取valid分数最高的，相当于让模型去过分拟合valid，test没见过，分数就会低

### 2026-09-07T10:08:16Z｜agent

- 评论 ID：5cf24514-9ddd-4922-8a80-9a54b1c257a9
- 父评论 ID：7bc1c781-2034-4b5c-9b2f-79d4e980db81

完全正确。这是对验证集的**间接过拟合**：模型参数未必直接用 `valid` 标签训练，但人每轮都根据 valid 分数改 prompt、只保留最高分版本，等价于让整个开发决策过程持续适配 valid 中的样本特征和随机波动。

因此最高的 valid 分数不一定代表真正泛化最好；它可能只是 30 次尝试中“恰好最适合这份 valid”的版本。未参与这些选择的 test 更接近新数据，所以最终 test 分数可能明显更低。

职责边界可以记为：

```text
train：学习参数 / 构造训练信号
valid：有限次数地调 prompt、超参数、模型版本
test ：最终独立审计，不参与日常选型
```

一个重要后续规则：如果已经查看 test 结果，又据此继续反复改 prompt 或模型，那么这个 test 也已被用作开发反馈，不能再作为最终无偏验收集；应冻结它的历史结果，并准备新的、按 `cluster_id` 隔离的 holdout/test 集。

下一题：为什么即使完全不训练模型，只是不断查看 test 分数后再改 prompt，也会使 test 集失去“最终独立评测”的资格？

### 2026-09-07T10:10:35Z｜member

- 评论 ID：03e3a47e-fc4b-471f-a040-0fdbe553d9f0
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

因为多次使用test分数评测，相当于间接拟合了test

### 2026-09-07T10:10:54Z｜agent

- 评论 ID：fe08e073-2240-4fb3-a65e-12680bbf4a03
- 父评论 ID：03e3a47e-fc4b-471f-a040-0fdbe553d9f0

正确。即使没有更新模型参数，人也会把 test 分数当作反馈来选择、修改或放弃 prompt；经过多轮筛选后，留下的方案已经适配了这份 test 的具体样本和偶然波动。这就是对 test 的**间接拟合**，它不再是未参与决策的独立数据。

所以一旦 test 结果被用于下一轮优化，它的角色就从“最终验收集”变成了开发集；已报告的分数只能作为当时的审计记录，后续最终验收应换一份新的 holdout/test，并继续保证与 train、valid 按 `cluster_id` 隔离。

到这里你已掌握数据可信评测的核心闭环：契约校验、失败分层、门槛发布、样本量、按簇切分，以及防止 train/valid/test 的直接或间接泄露。

下一题（进入评测版本化）：为什么每次跑评测时，都应记录 `dataset_version`、`prompt_version`、`model_version` 和 `evaluator_version`，而不能只保存一个最终 accuracy？

### 2026-09-07T10:12:32Z｜member

- 评论 ID：007fa164-2873-4caf-bf18-3e78f0c94005
- 父评论 ID：b83c2067-c3ae-4b07-9262-32e951d5930f

今天学习内容归档吧，明天接着学

### 2026-09-07T10:13:01Z｜agent

- 评论 ID：8cbe0361-564c-414e-ab25-dc3333b298f0
- 父评论 ID：007fa164-2873-4caf-bf18-3e78f0c94005

## 今日学习归档：可信离线评测与数据治理

### 1. 评测闭环

我们把一次模型评测拆成了可追踪的输入、执行和输出：

```text
gold_labels.jsonl + predictions_run.jsonl
  → 解析与契约校验
  → 逐样本失败归因 / 混淆矩阵 / 聚合指标
  → 发布门槛判断
  → failures.jsonl + metrics.json + release_decision.json
```

核心文件职责：

- `gold_labels.jsonl`：人工标注的正确答案；关键字段包括 `source_ticket_id`、`gold.status`、`cluster_id`。
- `predictions_run.jsonl`：一次推理运行的答卷；关键字段包括 `raw_output`、`model_version`。
- `failures.jsonl`：逐样本失败和原因。
- `metrics.json`：合法率、覆盖率、准确率、混淆矩阵等聚合结果。
- `release_decision.json`：`allowed`、`failed_gates` 与阈值，作为发布决策记录。

### 2. 三层失败分类

```text
json
  raw_output 不能解析为 JSON。

schema_contract
  JSON 可解析，但字段、status 枚举或 evidence 不满足契约；例如 evidence 为空、与原文不一致或模型捏造证据。

task_incorrect
  契约已经通过，但预测的 status 与 gold.status 不一致。
```

重要规则：`contract_valid=False` 的样本不进入 status 混淆矩阵；它不是一个合法的“预测类别”。缺少 prediction 则属于 `input_completeness` / coverage 问题，也不能伪装成 `unknown`。

### 3. 指标与发布门槛

当前示例 fixture 的关键结果：

```text
JSON/schema 合法率：5/6 = 83.3%
端到端准确率：3/6 = 50%
退款处理中 → 已退款：1/2 = 50%
已退款 → unknown：1/2 = 50%
```

发布逻辑采用“全量计算、全量诊断、最后决策”：

```python
failed_gates.append("some_gate")  # 只记录未通过的门槛名称
allowed = not failed_gates          # 空列表才允许发布
```

- `metrics.json` 保存真实指标值。
- `failed_gates` 保存所有不达标的规则名，不能遇到第一个失败就提前返回。
- 风险错误率“不得超过 2%”时，`rate <= 0.02` 通过，`rate > 0.02` 失败。
- 边界例子：`0.019` 通过、`0.020` 通过、`0.021` 不通过。

### 4. 已完成的代码实践

已实现并验证的保护场景：

```text
P6 缺 prediction
  → input_completeness；coverage 降为 5/6；不算作 unknown。

P5 的 unknown 携带非空 evidence
  → schema_contract；不进入混淆矩阵。

P2 的 status 正确、但 evidence 是原文不存在的“退款已经到账”
  → schema_contract；不因 status 相同而计作正确分类。

发布门槛边界
  → 0.019 / 0.020 / 0.021 的通过语义已用自动化测试覆盖。
```

已运行验证：`python3 -m unittest -v test_evaluator`（4 项通过）以及 `python3 -m py_compile evaluator.py test_evaluator.py`。

### 5. 样本量与可信度

小样本中观察到 0 次错误，并不等价于真实错误率为 0。若假设样本独立且有代表性：

```text
要以单侧 95% 把握证明真实错误率 < 10%：
0.90^n ≤ 0.05 → n ≥ 29

要以单侧 95% 把握证明真实错误率 < 2%：
0.98^n ≤ 0.05 → n ≥ 149
```

这只是最低样本数；近重复样本会降低有效样本量，因此还需覆盖来源、时间、文本模式与风险场景。

### 6. 数据切分与防泄露

```text
同一 cluster_id 的所有样本，只能整体进入 train / valid / test 的其中一个 split。
```

- `source_ticket_id` 有助于排查同工单泄露；`cluster_id` 是更一般的近重复隔离边界。
- 例如 `C100` 有 8 条近重复样本，应全部放入同一个 split；数据有效，不应因泄露风险而直接删除。
- 验证集中的 exact-match 样本以及同簇近重复样本，都不能回灌训练集。否则模型已见过答案，valid 分数会虚高。

集合职责：

```text
train：学习模型参数或训练信号。
valid：有限次数地调 prompt、超参数、checkpoint 和模型版本。
test ：最终独立验收，不参与日常选型或后续优化。
```

反复根据 valid 分数选 prompt 会对 valid 间接过拟合；查看 test 分数后再继续改 prompt，也会让 test 变成开发反馈，失去独立验收资格。此时应冻结旧 test 的审计结果，并准备新的、按 `cluster_id` 隔离的 holdout/test。

### 明日学习起点：评测版本化与可复现性

我们将从刚才未完成的问题继续：为什么每次评测都要记录 `dataset_version`、`prompt_version`、`model_version`、`evaluator_version`，而不能只保存一个 accuracy。

明天的最小交付物：为一次评测设计 `run_manifest.json`，能回答“分数变化到底是数据变了、prompt 变了、模型变了，还是评测器规则变了”，并把它与 `metrics.json`、`failures.jsonl`、`release_decision.json` 串起来。

## 元数据

```json
{}
```


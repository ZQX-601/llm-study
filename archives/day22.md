# Day22 学习归档｜Tool Calling、MCP 工程与安全边界

- 学习日：Day22（2026-10-01）
- 当前阶段：调整后主线第 3 周——Tool Use 与 MCP
- 状态：MCP 第一轮理论与教学代码完成；真实 Server/Client 集成和独立实现尚未完成
- 衔接来源：[Day21](day21.md) 的 Tool Calling 与 MCP 主线入口
- 本日方式：原理讲解、连续追问、工程情境判断、静态代码实现和纯状态机测试
- 实际学习时长：未记录

## 一、本日目标与岗位意义

本日从 Day21 的 RAG/Tool/Runtime 边界继续，建立 MCP 的完整工程认知：自定义 Tool
如何定义和执行，Host、MCP Client、MCP Server 与后端 API 如何分工，Tools、Resources
和 Prompts 如何发现及使用，以及权限、超时、幂等、Prompt Injection 和跨 Server 数据
流如何控制。

主题等级：

- `岗位必备`：Tool Schema、Host/Client/Server 边界、动态工具发现和调用链；
- `岗位必备`：身份、租户、权限、状态、确认、超时、幂等和结果未知；
- `需要掌握`：Resources、Resource Templates、Prompts 与 Host 消费方式；
- `需要掌握`：第三方 MCP 信任边界、Prompt Injection 与跨 Server 数据外发；
- `前沿扩展`：当前规范的版本差异、缓存、通知和长任务扩展本日未系统展开。

对应工作决策：为训练平台设计 MCP 时，判断哪些后端能力应成为 Tool、Resource 或
Prompt；写操作如何避免被模型直接执行；Host 怎样把动态发现的 MCP 能力安全地接入
LLM Tool Calling。

## 二、到期复习：RAG 最早失败层

新加坡差旅查询场景中，原始文档、Parser、Chunker 和 Index 均正确，但 Query
Processor 错误生成：

```text
metadata_filter = {"region": "中国大陆"}
```

用户正确定位最早失败层为 Query Processor，并指出原始 Query 已明确包含“新加坡”，
应优先检查地区映射、默认地区继承、rewrite 信息丢失和强过滤策略，而不是先调
Retriever 或 Reranker。该复习通过。

## 三、Tool Calling 与自定义工具

### 3.1 Tool Calling 的真实含义

模型产生的是候选调用，而不是直接执行函数：

```text
Host 提供 Tool Schema
→ LLM 选择工具并生成 arguments
→ Runtime 校验
→ Executor/MCP Client 真正执行
→ Observation 返回模型
```

Tool Schema 负责参数结构，Runtime 仍需负责业务、权限、状态和确认。用户能够指出：

- `description`、字段说明和正则约束不足会降低模型选择与填参质量；
- `additionalProperties: false` 可阻止未定义字段；
- Schema 通过后仍要检查任务存在性、取消权限、状态、用户确认和执行结果。

### 3.2 最小工具集合

用户最初不知道“查询任务为何未结束”时应只向模型提供只读
`get_training_job`。经解释后，用户在发票场景中正确回答：只暴露
`get_customer_invoice`，但 `void_customer_invoice` 的服务端鉴权仍必须保留。

核心结论：

```text
Host 工具筛选：减少模型误调用和攻击面
Server 鉴权：形成不可绕过的执行边界
```

## 四、MCP 架构与实际接入

### 4.1 Host、Client、Server

```text
用户
→ Host（完整 Agent 应用）
   ├─ LLM Client
   ├─ Agent Runtime
   └─ MCP Client
      → MCP Server
         → 后端 REST API / 数据库 / 调度器
```

- Host：管理 UI/会话、LLM、MCP Clients、工具筛选、State、循环和安全策略；
- Agent Runtime：Host 内负责循环、状态、校验、分发和停止条件的核心子系统；
- MCP Client：负责与一个 MCP Server 协议通信；
- MCP Server：定义能力、执行 Handler、做服务端鉴权并访问真实后端。

Host 是逻辑角色，不等于用户本人，也不一定运行在用户电脑。桌面 Agent 的 Host 可以
在本地；Web Agent 的 Host 通常在云端后端。Host 本身不是 LLM，通常由 Host 调用的
主模型就是 Agent 的 LLM；复杂系统可另设 Router、Verifier 或 Safety Model。

### 4.2 动态能力发现

Agent 不需要预先知道训练平台的 REST 路径或全部工具。连接后由 MCP Client 获取：

```text
tools/list
resources/list
resources/templates/list
prompts/list
```

Host 将 MCP Tool 的名称、描述和 Schema 转为模型 Tool Calling 定义；模型产生调用后，
Host 根据分发表找到相应 MCP Client，再发送 `tools/call`。

README 可供人类阅读，但不能代替运行时权威的 `tools/list`，因为 Markdown 可能过期、
不能稳定校验参数，也不能表达当前身份实际可用的能力。

### 4.3 MCP Tool 与后端 API Client 不是一层

用户发现 `commit_cancel_training_job` 只定义在 `server.py`，而后端适配层只有
`cancel_job`。经解释确认：

```text
commit_cancel_training_job
= 面向 Agent 的 MCP 业务能力
= Proposal/确认/权限/版本/幂等校验
+ 调用 training_client.cancel_job

cancel_job
= 面向训练平台的 REST API 适配
= POST /jobs/{job_id}/cancel
```

一个 MCP Tool 可以调用多个后端 API，一个后端方法也可被多个 Tool 复用；名称和数量
不要求一一对应。`training_client.py` 实际是训练平台 API Client，不是 MCP Client，
文件名存在容易混淆的教学问题，后续可改为 `training_api_client.py`。

## 五、训练平台 MCP 教学代码

本日新建 [训练平台 MCP 示例](../coding/training_platform_mcp/README.md)，包含：

```text
schemas.py                  输入、输出、错误和操作状态 Schema
security.py                 Principal、权限、确认和审计边界
training_client.py          训练平台 REST API 适配层
server.py                   Tools、Resources、Prompts 与 HTTP 入口
client_demo.py              MCP Client 发现和调用示例
retry_state_machine.py      Host RESULT_UNKNOWN 有界核对状态机
test_retry_state_machine.py 纯状态机测试
```

### 5.1 Tools

```text
get_training_job
get_cancel_operation
prepare_cancel_training_job
commit_cancel_training_job
```

取消操作拆为：

```text
prepare
→ 生成绑定用户、租户、任务版本和有效期的 Proposal
→ 用户确认具体 Proposal
→ commit
→ 再次验证后调用真实 POST /cancel
```

默认 `ConfirmationVerifier` 故意返回 `False`，避免教学代码在未接真实确认服务时执行
写操作。

### 5.2 Resources 与 Prompts

Server 侧新增：

```text
training://docs/error-codes
training://jobs/{job_id}/logs
training://jobs/{job_id}/config
diagnose_training_job(job_id)
```

结论：

```text
Server 拥有并提供 Tools、Resources、Prompts；
Host/MCP Client 发现、选择、调用和使用它们。
```

Resource 适合读取已知 URI 的数据；复杂搜索、聚合或写操作适合 Tool；Prompt 返回
可复用消息和流程，不会自动调用 Tool。Host 可把 Prompt 显示为菜单/命令，把 Resource
显示为可添加的上下文或处理 Tool 返回的 Resource Link。

## 六、错误、超时、重试与幂等

### 6.1 三层结果

```text
协议/连接失败：可能抛出 Client 异常
Tool 执行失败：CallToolResult.isError=true
业务拒绝：MCP 调用正常，但 structuredContent.ok=false
```

Host 必须逐层检查，不能把“没有抛异常”当作业务成功。

### 6.2 幂等键与结果未知

幂等键标识“一次逻辑业务动作”。网络重试必须复用原 key；换新 key 会被后端视为
新的业务动作。后端应以租户、用户和 key 建唯一记录，并绑定请求摘要、执行状态和
最终结果。

用户理解了四个 ID 的不同职责：

```text
proposal_id       用户确认的具体动作
confirmation_id   用户是否确认该 Proposal
idempotency_key   网络重试是否属于同一次业务动作
operation_id      后端实际执行任务的编号
```

写调用超时只能进入 `RESULT_UNKNOWN`，不能判定失败。正确闭环：

```text
IN_FLIGHT
→ RESULT_UNKNOWN
→ 按原 key 调用 get_cancel_operation
   ├─ PROCESSING → 有界退避查询
   ├─ SUCCEEDED → 完成，不重提 commit
   ├─ FAILED_FINAL → 停止
   ├─ 强语义 NOT_FOUND → 才可复用原 key 重提
   └─ 持续 UNKNOWN → ESCALATED
```

首次状态题中，用户回答 `RESULT_UNKNOWN → SUCCEEDED`，遗漏了 `PROCESSING`；经解释
后在连续五次 `UNKNOWN` 场景中正确判断进入 `ESCALATED`，且不能生成新 key 重试。

### 6.3 运行证据

全部 7 个 Python 文件通过 AST 静态语法检查。以下标准库测试实际运行通过：

```text
PROCESSING → SUCCEEDED
连续 5 次 UNKNOWN → ESCALATED
FAILED_FINAL → 立即停止
```

本日没有安装或运行真实 MCP SDK，没有启动 Server，没有连接训练平台 API，也没有
验证 OAuth、真实 `tools/list`、`resources/read` 或 `prompts/get`。

## 七、Resources、Prompts 与完整诊断流程

用户围绕“任务 001 为何失败”追问了完整通信过程。最终链路为：

```text
首次连接与授权
→ Host 发现 Tools/Resources/Prompts/Instructions
→ 用户提出诊断目标
→ 用户或 Host 选择 diagnose_training_job Prompt
→ Host 获取 Prompt Messages 并筛选最小 Tool 集
→ LLM 调用 get_training_job
→ Server 查询真实状态并返回 Resource Links
→ Host 读取小型错误码 Resource，大日志改用搜索 Tool
→ LLM 继续查询日志和 GPU 指标
→ Host 汇总 Observation
→ LLM 输出事实、推断和缺失信息分离的诊断
```

用户正确回答：Server 不能强制 Host 使用 Prompt；由 Host/用户选择；Prompt 不会自动
调用 Tool。Host 将 Prompt、用户问题和最小 Tool Schema 交给 LLM，LLM 产生候选调用，
Host/Runtime 校验并通过 MCP Client 分发，Server 映射到后端 API。

## 八、安全、授权与第三方 Server 信任边界

### 8.1 身份与权限

远程 Streamable HTTP 通常使用 OAuth/Bearer Token。用户、租户和 scope 必须来自经过
验证的 Token/连接上下文，不能作为模型可填写的 Tool arguments。Server 必须在每次
调用时重新验证权限，Host 工具隐藏不能代替 Server 鉴权。

本地 stdio Server 通常继承当前操作系统账户的文件、环境变量和网络权限，安装第三方
MCP Server 接近于运行普通软件包，必须审查来源和权限。

### 8.2 Prompt Injection 与跨 Server 数据流

Server instructions、Tool description、Prompt、Resource 和 Tool Result 都可能来自
不可信第三方或被污染，不能自动提升为系统指令。

训练日志要求读取云凭据并调用另一个 Server 外发时，用户正确判断它是 Prompt
Injection 且不应执行；首次只提到鉴权和状态，经补充后理解还必须检查：

- 用户原始意图是否包含外发；
- Tool 是否属于本轮最小能力集合；
- 调用是否由不可信 Resource 诱导；
- 内容是否包含敏感数据；
- 数据是否从内部流向开放世界；
- 收件人、内容和 Proposal 是否经过明确确认；
- DLP 是否应无条件阻断凭据外发。

在 GitHub Issue 新场景中，用户能够指出：即使 Tool 在列表中且用户有权限，外部数据
来源不可信、与用户原始意图不符，且内部数据流向公开 GitHub 仍需被阻断。

## 九、教学连贯性问题与规则修订

本日出现一次教学流程错误：用户在“错误、超时、重试与幂等”父话题中追问幂等键，
追问解释完成后说“接着学”，教练未经显式收尾就跳到 Agent Loop。用户指出无法判断
父话题是否已经完成。

已修订：

- `AGENTS.md` 新增主线主题、父话题/小节、临时追问的话题栈；
- “接着学”默认返回当前父话题精确停点；
- 切换前必须报告结论、剩余内容及理论/代码/运行/验收四类证据；
- `study_agent/SESSION_TEMPLATE.md` 增加话题栈、父话题关闭状态和精确停点字段。

此次规则修订已有文件证据，不再只是对话承诺。

## 十、本日问答与当前证据

用户目前能够：

- 区分 Host、Agent Runtime、MCP Client、MCP Server 和后端 API Client；
- 解释 Host 可能在桌面或云端，Host 是应用而非 LLM；
- 解释 Server 定义能力、Host 消费能力、LLM 提出候选调用；
- 解释 Tool Schema 与业务/权限校验的边界；
- 解释 Server 过滤 Client 能力、Host 过滤模型能力、Server 调用时鉴权三层边界；
- 解释 MCP Tool 与后端 REST 方法不要求一一对应；
- 解释 Tools、Resources 和 Prompts 的职责及完整诊断链；
- 判断写调用超时、幂等键复用、权威状态查询和 ESCALATED；
- 识别 Resource Prompt Injection、意图偏离和跨 Server 敏感数据外发风险。

这些证据属于原理理解、工程判断和教学代码阅读，不等于用户已独立开发并运行 MCP。
用户在结束时表示“当前 MCP 就学到这里，感觉差不多了解了”，因此记录为第一轮学习
完成，而不是独立实战验收通过。

## 十一、薄弱点与后续计划

- 当前代码没有真实 OAuth 中间件，`Principal` 绑定仍是教学接口；
- `training_client.py` 名称容易与 MCP Client 混淆，后续可更名；
- 没有安装并运行官方 MCP SDK，协议 API 只做静态实现；
- 没有真实训练平台后端、幂等表、唯一约束、持久化 Proposal 和确认服务；
- Host 仅有 Client 和状态机片段，尚无完整动态 Tool Bridge 和 LLM Agent loop；
- Resources、Prompts 和跨 Server Guardrail 尚无运行测试；
- 未独立从零实现 Tool/Resource/Prompt 或排查 MCP Trace。

下一主线进入单 Agent 核心循环：显式 TaskState、Action、Observation、Tool Bridge、
Verifier、停止条件和失败恢复。后续以当前训练平台 MCP 作为工具环境，自然验证
MCP 能力，不再重复整套概念讲解。

## 十二、复习安排

- 1 天后：给定 Host/Client/Server 调用 Trace，定位权限与能力筛选职责；
- 3 天后：阅读写操作重试代码，定位新幂等键和 `RESULT_UNKNOWN` 处理错误；
- 7 天后：为新领域设计 Tool、Resource、Prompt 和信任边界；
- 单 Agent 实操阶段：运行最小 MCP Server/Client 或等价 mock，完成动态 Tool Bridge。

## 十三、话题栈与下次入口

```text
主线：LLM → Agent → 领域 Agent → Agentic RL
已关闭父话题：Tool Calling 与 MCP 第一轮
未解决追问：无
下一父话题：单 Agent 核心循环
精确入口：用训练平台工具环境实现显式 TaskState → Action → Observation 循环
```

## 十四、一手资料

- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Python Client](https://github.com/modelcontextprotocol/python-sdk/blob/main/docs/client/index.md)
- [MCP Transports](https://github.com/modelcontextprotocol/python-sdk/blob/main/docs/run/index.md)
- [MCP Authorization](https://github.com/modelcontextprotocol/python-sdk/blob/main/docs/run/authorization.md)
- [MCP Tool Annotations](https://blog.modelcontextprotocol.io/posts/2026-03-16-tool-annotations/)

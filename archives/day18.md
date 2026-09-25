# Day18 学习归档｜分布式 GRPO、全局 Group 与变长 Batching

- 学习日：Day18
- 当前阶段：阶段 6——GRPO 生产工程边界
- 状态：in_progress
- 衔接来源：`archives/day17.md` 的梯度累积、严格 token 平均与测试设计
- 本日方式：只学习和分析，不运行真实分布式训练代码

## 一、今日完成内容

今天完成了生产 GRPO 的分布式基础与变长数据问题：

1. 建立 DDP、process、rank、world size、local/global 的基本概念。
2. 理解普通 DDP 的模型复制、数据分片、梯度 all-reduce 和全 rank optimizer step。
3. 证明 GRPO 的 advantage 统计边界是 `group_id`，不是 `rank_id`。
4. 分析完整 group 放置与跨 rank group 聚合两种方案。
5. 理解全局 skip/backward/step 决策必须在所有 rank 上一致。
6. 推导 DDP 下严格的全局有效 token 平均及 `world_size` 修正。
7. 区分 all-reduce、all-gather 和 DDP 梯度同步的用途。
8. 设计跨 rank rollout 记录的身份字段与 advantage 回填流程。
9. 分析变长 completion、token 负载、padding、长度分桶和动态 batching。
10. 比较 token 平均与 sequence 等权平均的长度权重差异。
11. 复习 `valid_sequence_mask`、`response_mask` 和 `effective_mask` 的层级职责。

今天决定停止重复 loss 算术题，后续进入新内容：rollout/trainer 分离、权重同步和 policy staleness。

## 二、DDP 基础框架

普通 DDP 中：

```text
每个 GPU 通常对应一个独立 process/rank
每个 rank 保存完整模型副本和 optimizer
不同 rank 处理不同 local batch
backward 期间通过梯度 all-reduce 同步
所有 rank 使用相同梯度执行 optimizer.step
```

关键术语：

```text
rank：分布式进程编号
world_size：参与训练的总进程数
local batch：一个 rank 本地处理的数据
global batch：所有 rank 数据的集合
```

若：

```text
world_size=4
local micro-batch size=3
gradient_accumulation_steps=2
```

则：

```text
一次 micro-batch 全局处理 3×4=12 条
一次 optimizer.step 全局覆盖 3×4×2=24 条
```

DDP 不显著降低单卡模型状态显存，因为每个 rank 仍保存完整模型。模型状态放不下时，需要进一步考虑 FSDP/ZeRO、Tensor Parallel 或 Pipeline Parallel。

## 三、Collective 通信

### All-reduce

```text
多个 rank 的同位置值做聚合
并把聚合结果返回所有 rank
```

用途：

```text
SUM(local_valid_token_count)
→ global_valid_token_count
```

DDP backward 也会使用梯度 all-reduce，使所有 rank 得到一致梯度。

### All-gather

```text
收集每个 rank 的完整记录
使参与方获得跨 rank 数据集合
```

用途：

```text
收集 group_id、reward、sequence_id 等记录
→ 重建跨 rank 的完整 group
```

示例：

```text
rank 0 reward=[0,1]
rank 1 reward=[1,2]

all-reduce SUM → [1,3]（逐位置相加）
all-gather     → [0,1,1,2]（收集完整值）
```

因此简单 reward all-reduce 不能替代跨 rank group 聚合。

## 四、分布式 Group 完整性

完整 group：

```text
rewards=[0,1,1,2]
mean=1
population std≈0.7071
advantages≈[-1.414,0,0,1.414]
```

若拆成：

```text
rank 0：[0,1] → local advantages=[-1,+1]
rank 1：[1,2] → local advantages=[-1,+1]
```

则两条 reward=1 的 completion 被错误赋予 `+1` 和 `-1`。同样 reward 收到相反更新方向，破坏同 prompt 内相对比较。

核心原则：

```text
GRPO 的统计边界 = 全局 group_id
不是 rank_id
```

### 方案 A：Group-local placement

```text
一个 prompt 的 G 条 completion
→ 全部放在同一 rank
→ 本地计算完整 group advantage
```

优点：实现简单、通信少、容易保证对齐。

代价：数据放置受 group 边界约束，completion 长度差异大时容易出现 rank 负载不均。

### 方案 B：Global group aggregation

```text
group 可以跨 rank
→ 收集 group_id/reward/sequence_id
→ 全局按 group 计算 mean/std/advantage
→ 回填原 rank 和本地位置
```

优点：token 负载与动态 batching 更灵活。

代价：通信更多，必须维护全局身份、分组和回填映射。

两种方案数学上都正确，前提是每条 advantage 使用完整 group 计算。

## 五、跨 Rank 记录身份

建议记录：

```text
group_id
→ 属于哪个 prompt 组

global_sequence_id
→ 这条 completion 的全局唯一身份

origin_rank + origin_position
→ 原来位于哪个 rank 的本地 batch 哪个位置
```

最小安全流程：

```text
创建全局 sequence_id
→ 保存本地 group_id/reward/origin
→ all-gather 或等价收集
→ 按 group_id 计算 advantage
→ 用 sequence_id 对齐记录
→ 按 origin_rank/origin_position 回填
→ 本地构造 effective_mask
```

如果 group 完整位于单个 rank，可省略跨 rank all-gather、全局映射和回填；生产系统仍可保留 sequence_id 用于追踪和审计。

`group_id` 至少应全局唯一、跨通信和排序稳定、可映射回 prompt/completion，且不依赖本地 rank 才能解释。

## 六、全局一致的 Skip/Backward/Step

DDP collective 要求所有 rank 以相同顺序参与通信。

错误：

```text
rank 0 local valid=0 → continue
rank 1 local valid=12 → backward
```

rank 1 会在梯度 all-reduce 等待 rank 0，可能导致 deadlock/hang/timeout。

正确流程：

```text
各 rank 计算 local_valid_token_count
→ all-reduce SUM
→ 每个 rank 得到相同 global_valid_token_count
```

分支规则：

```text
global_valid_token_count == 0
→ 所有 rank 一起跳过 backward/step/有效累积计数

global_valid_token_count > 0
→ 所有 rank 一起进入 backward
```

若某 rank 本地无有效 token但全局有效，应使用连接 current policy 计算图的零 loss：

```python
local_loss_sum = (
    per_token_loss * effective_mask
).sum()
```

即使值为 0，它仍连接 `new_logp`，可贡献零梯度并参与 DDP 同步。不能用脱离模型的普通常量 0。

## 七、分布式梯度累积

`accumulation_count` 是全局同步的逻辑计数：

```text
全局无效 micro-batch → 不增加
局部无效但全局有效 → 所有 rank backward，增加 1
```

例如：

```text
micro-batch 1：rank0=0, rank1=12 → global=12
micro-batch 2：rank0=5, rank1=7  → global=12
```

若累积步数为 2：

```text
accumulation_count=2
total_global_valid_tokens=24
→ 所有 rank 一起 optimizer.step
```

若 epoch 结束时只有一个有效 global micro-batch，则按真实全局 token 数提交尾部。

## 八、DDP 下严格 Global Token Mean

目标：

```text
所有 rank 的有效 token 梯度总和 / global_valid_token_count
```

典型 DDP 默认对 rank 梯度求平均，因此一种常见写法是：

```python
scaled_local_loss = (
    local_loss_sum
    * world_size
    / global_valid_token_count
)
```

例：

```text
world_size=2
rank 0：loss_sum=10, valid=2
rank 1：loss_sum=18, valid=6

global loss sum=28
global valid=8
global mean=3.5
```

缩放：

```text
rank 0：10×2/8=2.5
rank 1：18×2/8=4.5
DDP rank 平均：(2.5+4.5)/2=3.5
```

若遗漏 `world_size`，DDP 再平均后会得到 `1.75`，梯度缩小一半。工程上必须核对具体框架和通信 hook 的 gradient reduction 语义。

## 九、变长 Completion 与负载不均

即使每个 rank 的 sequence 数相同，计算量也可能差异巨大：

```text
rank 0：4 条 completion，总 response tokens=8
rank 1：4 条 completion，总 response tokens=80
```

DDP 同步受最慢 rank 限制，快 rank 会在 collective 处等待慢 rank，造成 straggler、GPU 利用率下降和 step latency 波动。

可选策略：

```text
1. Group-aware token bin packing
2. Group-aware length bucketing
3. Completion-level dynamic batching + 全局 group 聚合
```

若单个 group 本身特别大，完整 group 放置存在负载均衡硬上限，此时可考虑跨 rank group 聚合。

## 十、Padding 与长度分桶

示例：

```text
lengths=[2,3,3,4,18,19,20,20]
```

全部混合 padding 到 20：

```text
分配位置=8×20=160
真实 token=79
padding=81
```

分短桶和长桶：

```text
分配位置=4×4 + 4×20 = 96
padding=96-79=17
减少分配位置=64
```

`response_mask` 保证 PAD 不进入 loss，但通常不能消除 padded tensor 的 forward/backward 计算浪费。

```text
response_mask：保证数学正确性
length bucketing/dynamic batching/packing：提高计算效率
```

## 十一、动态 Token Batching 的局限

仅用 `total_token_count` 是实用近似，但不精确代表 Transformer 成本。

在简化 attention 成本 `ΣL²` 下：

```text
batch A=[20]    → 400
batch B=[10,10] → 200
```

两者总 token 都是 20，但 attention 成本相差 2 倍。

训练成本还取决于：

```text
prompt + response 总长度
padding 后形状
最大序列长度
attention 长度项
MLP 的线性 token 成本
kernel 利用率、activation 显存和模型并行通信
```

GRPO teacher forcing 输入完整 `prompt + completion`，不能只按 response length 预算。

例如：

```text
A：prompt=100,response=20,total=120,cost≈14400
B：prompt=10,response=20,total=30,cost≈900
```

response 相同但简化 attention 成本相差 16 倍。`response_mask` 只控制 loss 统计，不能消除 prompt forward。推理 KV Cache 也只是 prefill 后在 decode 阶段复用历史 KV，prompt prefill 本身仍需计算。

## 十二、Token Mean 与 Sequence Equal Mean

两条 completion：

```text
A：长度2，loss=[2,4]，sequence mean=3
B：长度6，loss=[1,1,1,1,1,1]，sequence mean=1
```

Token mean：

```text
12/8=1.5
A:B 总权重=2:6
→ 长 completion 因 token 更多拥有更大总权重
```

Sequence equal mean：

```text
(3+1)/2=2
A:B 总权重=1:1
→ 每条 completion 总权重相同
```

选择取决于目标：

```text
每个动作 token 等权 → token mean
每条采样 completion 等权 → sequence equal mean
```

Sequence 等权只能消除“token 数更多导致总权重更大”这一种长度效应，不能消除 Reward Model 长度偏好、截断、EOS、任务成功率和采样分布中的长度偏差。

后续不再重复同类 loss 算术，只有在新机制依赖时回看。

## 十三、Mask 层级复核

```text
valid_sequence_mask [N]
→ reward/group 层面，这条 completion 是否有训练资格

response_mask [N,T]
→ token 层面，哪些 response token 有效、哪些是 PAD

effective_mask
= response_mask * valid_sequence_mask[:,None]
```

必须使用 `valid_sequence_mask[:, None]` 把 `[N]` 扩成 `[N,1]`，再沿 token 维广播到 `[N,T]`。分子和分母必须使用同一 `effective_mask`。

## 十四、今天的验证边界

用户继续选择只学习和分析：

- 未运行分布式训练；
- 未安装或调用训练框架；
- 未修改 `grpo_end_to_end.py`；
- 所有结论来自概念分析与手算。

## 十五、下一次直接从这里继续

停止重复 loss/mask 算术，进入新主题：

```text
Rollout workers 与 Trainer workers 分离
→ Policy 权重同步
→ Rollout staleness
→ 混合精度
→ LoRA 的分布式训练与权重同步
→ GRPO/PPO 最终选型验收
```

下一题：

```text
rollout worker 使用 policy v10 生成 completion
old_logp 也由 v10 计算
trainer 当前已经更新到 policy v13
new_logp 由 v13 计算
```

回答：

1. `old_logp` 应保持 v10 的值，还是用 v13 重算？
2. 这批 rollout 的 staleness 是几个 policy version？
3. staleness 过大时，ratio 和 clipfrac 可能如何变化？
4. 同步式 rollout-training 的主要优点是什么？
5. 异步式的主要优点和主要风险分别是什么？

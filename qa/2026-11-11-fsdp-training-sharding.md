# FSDP：训练状态全分片与通信过程

- 日期：2026-11-11
- 类型：主线相关 QA 专题
- 状态：用户自评掌握，待复习与多卡实操验证
- 关联主线：后训练框架实战导入

## 原问题

> FSDP 是啥？

## 核心结论

FSDP（Fully Sharded Data Parallel）是训练侧的数据并行方案。它把模型参数、梯度和优化器状态
分散保存在多个 data-parallel rank 上；计算某个参数组或 Transformer Block 前，通过
`all-gather` 临时恢复完整参数，反向传播后通过 `reduce-scatter` 汇总并重新切分梯度。

```text
参数 shard
→ all-gather 当前层完整参数
→ forward / backward
→ reduce-scatter 梯度
→ 每个 rank 更新自己的参数与 optimizer shard
```

它以更高的通信和系统复杂度换取更低的单卡模型状态显存。激活值、临时完整参数和通信缓冲区仍会
占显存，因此不能把实际峰值简单视为除以 GPU 数量。

## 与其他并行方式的边界

### DDP

DDP 在每个 rank 保存完整参数、梯度和优化器状态，主要通过梯度 `all-reduce` 保持副本一致。
它提高数据吞吐，但模型状态仍需在单卡放下。

### Tensor Parallel

FSDP 主要分片模型状态，计算当前参数组时通常临时恢复完整参数；Tensor Parallel 则让多个 rank
共同执行一次矩阵计算。大规模训练中可以组合 FSDP、Tensor Parallel 和 Pipeline Parallel。

### FSDP1 与 FSDP2

- FSDP1 使用 `FullyShardedDataParallel`，主要基于 flat-parameter 分片；
- FSDP2 使用 `fully_shard()` 和 DTensor，采用 per-parameter 分片；
- FSDP2 通常自底向上对 Transformer Block 和根模块调用 `fully_shard()`；Optimizer 应在分片后
  使用已转换的参数创建。

## 在 verl / Agentic RL 中的位置

```text
verl Trainer
├── vLLM / SGLang：Rollout 生成
├── Reward / Advantage：训练信号
└── FSDP：Actor/Critic forward、backward、梯度分片和 optimizer step
```

FSDP 不负责生成 completion、计算 Reward 或定义 GRPO。它属于 Learner 训练后端。FSDP 中的
`optimizer.step()` 完成后，还需要 checkpoint/manifest 或等价的权重发布与 rollout engine 刷新，
才能让 vLLM 使用新 Policy。

单张 RTX 4060、8 GB 显存且 `world_size=1` 时，没有其他 GPU 共同保存 shard，FSDP 的多卡分片
价值无法发挥；应先使用小模型、LoRA/QLoRA、短序列、gradient checkpointing、梯度累积或
offload。多卡云环境再验证 FSDP 通信、显存和 checkpoint。

## 官方一手资料

- [PyTorch FSDP 文档](https://docs.pytorch.org/docs/main/distributed/fsdp.html)
- [PyTorch FSDP2 `fully_shard` 文档](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
- [PyTorch FSDP2 入门教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

## 用户追问与当前证据

- 用户在系统讲解后表示“明白了”（原消息有输入法错字，按上下文解释）；
- 理论证据：理解目标、P/G/O 分片、`all-gather`、`reduce-scatter`、DDP/TP 边界及在 verl 中的位置；
- 代码阅读：本次未阅读真实 PyTorch FSDP 源码；
- 实际运行：未运行多进程、多 GPU 或 FSDP checkpoint；
- 独立验收：未进行代码 Trace、显存估算或配置选择题。

结论：**用户自评掌握，待复习验证；不能标记为 FSDP 实操通过。**

## 后续验证建议

1. 给定 DDP 与 FSDP 显存 Trace，判断参数、梯度、Optimizer State 和激活分别位于哪里；
2. 追踪一个 Transformer Block 的 `all-gather → forward/backward → reduce-scatter`；
3. 在多 GPU 环境执行最小 FSDP2 脚本，保存每个 rank 的 shard shape、峰值显存和 distributed checkpoint；
4. 在 verl 配置中定位 Actor/Critic 的训练 strategy，并区分它与 vLLM rollout backend。


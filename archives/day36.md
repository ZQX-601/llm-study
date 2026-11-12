# Day36 学习归档｜后训练框架映射与 verl V1 同步源码

- 学习段：Day36（2026-11-09 至 2026-11-12）
- 当前阶段：调整后主线第 11～12 周——后训练框架实战
- 状态：框架职责、GRPO 数据流与 verl V1 同步源码第一轮完成；真实框架运行未开始
- 衔接来源：[Day35](day35.md) 的后训练框架实战导入
- 本段方式：官方源码核对、系统讲解、代码 Trace、标准库教学实现与测试、QA 追问
- 实际学习时长：未记录

> 归档边界：记录到 Actor Update、动态 microbatch、old log-prob 固定和 Policy 发布准入。
> 当前 Codex 执行环境没有 NVIDIA GPU 或后训练框架；所有真实 PyTorch/TRL/vLLM/verl/FSDP/Ray
> 运行均为未完成。下一步需要先确认用户 RTX 4060 所在机器的操作系统和可访问方式。

## 一、本段目标与岗位意义

主题等级：`岗位必备`。

本段把此前教学版 PPO/GRPO 合同映射到工作中常见的 TRL、vLLM、verl、FSDP 和 Ray，并开始阅读
verl 当前 V1 同步 Trainer 的真实调用链。目标是拿到配置或训练日志时，能判断字段由谁产生、在哪个
阶段消费，以及异常属于 Rollout、Reward、Algorithm、Learner 还是权重发布。

## 二、框架职责与完整 GRPO 数据流

完成以下分层：

```text
TRL：轻量一体化后训练入口，适合先跑通单机真实训练
vLLM：高吞吐 Rollout/推理引擎
verl：RL 数据流、角色与分布式资源编排
FSDP/Megatron：Actor/Critic 分布式训练后端
Ray：Worker 进程和 CPU/GPU 资源编排
```

稳定 GRPO 主链为：

```text
Prompt
→ Rollout（completion、mask、old log-prob、behavior version）
→ Reward/Verifier
→ Group Advantage
→ Learner Forward（new log-prob）
→ Policy Loss
→ Backward/Optimizer
→ 原子发布新 Policy
→ Rollout Engine 刷新
```

用户正确判断：Learner 已更新到 v102、但 checkpoint/manifest 和 vLLM 刷新尚未完成时，状态只能是
`UPDATED`，vLLM 仍使用 v101，不能标为 `PUBLISHED`。

## 三、Old、Reference 与 Current Policy

用户正确识别：只有 completion 和 `new_logprobs`、缺少 token-level `old_logprobs` 时，无法计算
带 ratio/clipping 的 Policy loss。

若完整保留 completion token、mask、behavior policy version、tokenizer/chat template 版本和冻结的
v101 checkpoint，可以走 `REPROCESS`，由训练侧 teacher-forcing 重算 old log-prob，不必重采。

用户正确指出 `ref_logprobs` 不能替代 `old_logprobs`：

```text
old：实际生成本批 Rollout 的 Behavior Policy，用于 ratio
ref：通常冻结的 SFT 锚点，用于 Reference KL
new：Learner 当前 Policy，用于梯度
```

多个 PPO epoch 内 old log-prob 必须固定。用户正确判断若每轮用 new 覆盖 old，会让 ratio 靠近 1、
clipfrac 和累计偏移被低估，PPO 约束失效。

## 四、KL 条件角色与双重约束

完成两种 KL 位置的区别：

```text
KL in Reward：先改变 token/sequence Reward，再进入 Advantage
KL Loss：Advantage 不变，KL 作为 Actor 总 Loss 的正则项
```

用户正确判断：当 `use_kl_in_reward=false` 且 `use_kl_loss=false` 时，Reference Worker 与
`ref_logprobs` 不是必需项。若两种 KL 同时启用并使用未经重新标定的相同系数，会形成双重约束，
可能导致 KL/clipfrac 过低、策略更新缓慢和任务 Reward 学不动。

## 五、verl 整体架构与 V1 同步源码

用户从零学习并确认理解：

- Trainer/Driver、Actor、Rollout、Reference、Critic、Reward；
- Worker、WorkerGroup、ResourcePool；
- 高层 Single Controller 与底层 SPMD/Multi-Controller 的 HybridFlow；
- 同步、同卡异步、分离异步的基本差异；
- DataProto/TensorDict/TransferQueue 等数据协议承担的生产职责。

当前 V1 同步源码的教学化主链：

```text
Rollout
→ Reward
→ Balance Batch
→ compute_old_logprobs
→ compute_ref_logprobs（可选）
→ compute_values（PPO）
→ compute_advantages
→ update_critic（PPO）
→ update_actor
→ step 结束后同步新权重
```

用户正确判断：GRPO 且 KL 全关闭时可以删除 Reference 和 Critic 路径，但必须保留 old log-prob。
同时理解源码串行顺序不等于算法依赖：Reference Forward 可以与 Reward/Advantage 分支独立，只有
KL in Reward 启用时才成为 Advantage 的上游依赖。

## 六、FSDP 专题

完成 FSDP 第一轮系统讲解并独立 QA 归档：

- Parameters、Gradients、Optimizer States 分片；
- 当前参数组计算前 `all-gather`，反向后 `reduce-scatter`；
- DDP、FSDP 与 Tensor Parallel 的职责边界；
- FSDP1 flat-parameter 与 FSDP2 DTensor/per-parameter sharding；
- FSDP 位于 verl Learner 的 Actor/Critic forward、backward、optimizer 和 checkpoint 路径；
- 单张 8 GB GPU、`world_size=1` 时不能获得多卡分片收益。

用户确认本轮理解；没有真实多 GPU、通信或 distributed checkpoint 证据。

## 七、Prompt、Trajectory 与 Token Batch

用户正确解释：

```python
ppo_mini_batch_size *= rollout.n
```

表示从 `B` 个 Prompt 转为 flatten 后的 `B×n` 条 trajectory，不是 Prompt 数增加。Group identity
必须保留；若 `n=1`，组内 Reward std 为 0，GRPO Advantage 为 0，无法提供相对学习信号。

完整 Group 必须先计算 Advantage，随后才能拆到不同 GPU/microbatch。用户正确拒绝在各 microbatch
内部重新标准化 Reward，否则 Advantage 会被资源布局污染。

动态 microbatch 中若使用 `microbatch_loss.mean()/num_microbatches`，实现的是 microbatch-equal，
不是全局 token-equal。用户正确指出有效 token 较少的 trajectory/microbatch 会被过度加权；更精确地
说，是 token 较少的 microbatch 中每个 token 权重更高。

## 八、Padding、No-padding 与 Packed Layout

用户追问并确认理解：

```text
Padded [B,T]
→ 根据 attention mask 移除 PAD
→ 连续 packed token buffer
→ cu_seqlens 保存每条序列边界
→ position ids 按序列重置
→ Variable-Length Attention
→ 必要时恢复 [B,T]
```

`input_ids`、response/loss mask、old/ref log-prob、Advantage、Value/Return 等 token 字段必须使用
同一索引重排。若边界错误会发生跨样本 Attention；若只处理 attention mask、不处理 response mask，
Prompt/PAD 可能进入 Policy Loss。尚未运行真实 CUDA Kernel 或 padded/unpadded 数值对照。

## 九、Optimizer 与发布硬门槛

用户正确判断：

```text
loss = NaN
optimizer_step_applied = false
```

违反优化有效性和发布准入合同，不应生成或向 vLLM 发布 v102。教学状态机已增加非有限 Policy loss
硬门槛，即使伪造 optimizer/published 字段也不能通过。

## 十、代码与运行证据

新增目录：

```text
coding/post_training_frameworks/
├── README.md
├── VERL_SOURCE_TRACE.md
├── __init__.py
├── environment_check.py
├── framework_mapping.py
└── test_framework_mapping.py
```

代码覆盖：

- `PipelineStage` 与 TRL/vLLM/verl/FSDP/Ray 职责映射；
- `GRPOBatchState` 和训练进度准入；
- Rollout 字段与 behavior identity/shape 校验；
- KL 条件下 Reference 字段要求；
- GRPO/PPO 的 V1 同步步骤计划；
- UPDATED/PUBLISHED 区分；
- 非有限 Loss 发布阻断；
- 当前执行环境检查。

实际验证：

```text
post_training_frameworks unittest：11 项通过
Python py_compile：通过（缓存定向到 /tmp）
git diff --check：通过
真实 PyTorch/TRL/vLLM/verl/FSDP/Ray：未安装、未运行
付费 API/云 GPU：未使用
```

环境检查实测：

```text
当前 Codex 执行环境：Intel x86_64 macOS
Python：3.9.6
torch/transformers/peft/trl/vllm/ray：均不可用
nvidia-smi / NVIDIA GPU：不可见
```

这不是用户档案中 RTX 4060 的机器，不能把两套环境混为同一次运行证据。

## 十一、独立 QA 归档

- [FSDP：训练状态全分片与通信过程](../qa/2026-11-11-fsdp-training-sharding.md)
- [verl：大模型 RL 后训练框架整体架构](../qa/2026-11-11-verl-architecture-overview.md)
- [Padding、No-padding 与 Packed Sequence 布局](../qa/2026-11-12-padding-no-padding-packed-layout.md)

三项均为用户自评掌握，待主线中的代码/运行验证；不等于独立实操通过。

## 十二、父话题状态与精确停点

父话题：`后训练框架实战导入`。

当前结论：框架角色、稳定 GRPO 数据流、verl V1 同步源码和教学状态机第一轮完成；真实框架实践
尚未开始，因此父话题为**部分完成**。

证据分类：

- 理论理解：框架角色、字段依赖、Batch 单位和同步主链第一轮完成；
- 代码产物：框架映射、环境检查、源码 Trace 与 11 项标准库测试；
- 实际运行：只有标准库测试，无真实模型或框架；
- 独立验收：用户能完成多项代码/配置 Trace，但实现主要由教练完成；
- 工程缺口：真实 forward/backward/optimizer/checkpoint、vLLM Rollout、verl Worker/Ray 和多 GPU。

精确停点：

```text
下一小节：TRL 小模型真实训练准备

前置问题：用户的 RTX 4060 机器是原生 Linux、Windows + WSL2，还是 Windows 原生？

下一步：根据实际 OS/CUDA 建立隔离环境并锁定 Python、PyTorch、Transformers、PEFT、TRL、
       模型、数据、显存上限、日志与 checkpoint；先跑真实 SFT/LoRA smoke test，再进入 GRPO。
```

## 十三、复习安排

- 1 天：换一份 verl 配置，定位 Actor、Rollout、Reference、Critic、Advantage 和 KL 条件角色；
- 3 天：追踪 Padded→Packed→Loss 的 token 字段和边界；
- 7 天：在真实 Trainer 日志中检查 old/ref/new log-prob、Reward、Advantage 和 Policy version；
- 21 天：根据真实运行复盘吞吐、显存、checkpoint、权重同步和失败案例。


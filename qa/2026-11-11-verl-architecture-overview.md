# verl：大模型 RL 后训练框架整体架构

- 日期：2026-11-11
- 类型：主线相关 QA 专题
- 状态：用户自评掌握，待源码阅读与真实运行验证
- 关联主线：后训练框架实战导入

## 原问题

> verl 我完全不了解，你先给我讲讲？

## 一、核心定位

verl 不是 PPO/GRPO 算法，也不是模型或单纯的推理引擎。它是大模型 RL 后训练框架，负责把
Rollout、Reward、Advantage、Actor/Critic 训练、Reference、权重同步、资源布局和 checkpoint
组织成可执行的数据流。

```text
verl = RL 算法流程编排 + 分布式角色/资源组织
vLLM/SGLang = Rollout 推理引擎
FSDP/Megatron = Actor/Critic 训练后端
Ray = Worker 进程与 CPU/GPU 资源编排
```

## 二、为什么后训练需要系统框架

SFT 通常只有 `Dataset → Forward → Loss → Backward → Optimizer`。PPO/GRPO 还需要多个模型角色
和计算阶段：Actor、Rollout、Reference、PPO Critic、Reward Model/Verifier、Advantage、权重发布
与下一轮生成。不同阶段适合不同执行引擎，且可能共享或分离 GPU，因此需要显式编排。

## 三、主要角色

- Actor：当前可训练 Policy，产生 `new_logprobs` 并执行 backward/optimizer；
- Rollout：使用某个行为策略版本生成 completion，通常由 vLLM/SGLang 执行；
- Reference：冻结的 SFT 锚点，产生 `ref_logprobs`，用于 KL；
- Critic：PPO 的 Value 模型，GRPO 通常不需要；
- Reward/Verifier：规则、模型、单元测试或环境结果，为 trajectory/completion 打分；
- Trainer/Driver：组织上述调用、数据流、日志、checkpoint 和版本同步。

## 四、同步 GRPO 一轮数据流

```text
Prompt Dataset
→ Rollout Worker 生成 G 条 completion
→ 保存 token、mask、old log-prob 与 behavior policy version
→ Reward/Verifier 输出每条 trajectory 的 Reward
→ 同 Prompt 内计算 Group Advantage
→ Reference Forward（可选 KL）
→ Actor Teacher-forcing Forward 得到 new log-prob
→ GRPO Policy Loss
→ FSDP/Megatron backward + optimizer
→ 生成并原子发布新 Policy
→ Rollout Engine 在安全边界刷新权重
```

PPO 在这条主链上增加 Critic、Value、GAE 和 Return target。

## 五、HybridFlow 控制结构

verl 在高层使用 Single Controller：Trainer/Driver 能看到完整 RL 图并依次调用 rollout、reward、
advantage 和 update。底层训练/推理 Worker 使用 SPMD/Multi-Controller：多个 FSDP/Megatron rank
执行相同程序并通过 collective communication 完成重计算。

```text
高层：集中表达算法数据流
底层：分布式执行大模型计算
```

## 六、WorkerGroup、ResourcePool 与数据容器

- Worker：一个远程执行进程或角色实例；
- WorkerGroup：同一角色的多个分布式 rank；
- ResourcePool：一组供角色放置和调度的 GPU 资源；
- DataProto/后续数据协议：跨 Worker 传输 Tensor batch 与非 Tensor 元数据的容器。

角色可以使用独立 GPU，也可以 colocate：同一组 GPU 在 Rollout 和训练阶段之间切换模型状态，
以提高资源利用率，但权重 reshard、wake/sleep 和同步会更复杂。

## 七、同步与异步

初学阶段先使用同步流程：Rollout、训练、发布完成后再进入下一轮，Policy 身份清晰。异步模式可提高
吞吐，但会引入 policy lag、stale trajectory、Queue 背压、off-policy correction、版本准入和恢复。
这些问题与 Day31–32 已学习的 policy version、准入与原子 Snapshot 直接对应。

## 八、TRL 与 verl 的学习顺序

- TRL：一体化 Trainer，适合单机、小模型、LoRA/QLoRA 和首次真实 GRPO；
- verl：适合多角色、多 GPU/多节点、vLLM/SGLang Rollout、FSDP/Megatron 和同步/异步流水线。

课程先用 TRL 建立真实 forward/backward/optimizer 证据，再进入 vLLM Rollout 和 verl 分布式闭环，
避免算法、依赖、Ray、FSDP 和 vLLM 故障同时出现。

## 官方一手资料

- [verl 官方文档入口](https://github.com/verl-project/verl/blob/main/docs/index.rst)
- [verl PPO 示例架构](https://github.com/verl-project/verl/blob/main/docs/examples/ppo_code_architecture.rst)
- [verl Ray Trainer](https://github.com/verl-project/verl/blob/main/docs/workers/ray_trainer.rst)
- [verl Hybrid-Controller 说明](https://github.com/verl-project/verl/blob/main/docs/blog/v0.7.md)
- [verl Agent Loop](https://github.com/verl-project/verl/blob/main/docs/advance/agent_loop.rst)

## 用户追问与当前证据

- 用户要求从零介绍 verl，系统讲解后表示“明白了”；
- 理论证据：已建立 Trainer、Worker、ResourcePool、Actor/Rollout/Reference/Critic/Reward、
  HybridFlow 和同步/异步的整体结构；
- 代码阅读：尚未阅读真实 verl 入口、Trainer `fit()`、Worker 或 Data Protocol 源码；
- 实际运行：未安装 verl，未执行配置 dry-run、CPU smoke test 或 GPU 训练；
- 独立验收：尚未独立追踪一轮 GRPO 配置和字段流。

结论：**用户自评掌握整体架构，待配置、源码、运行和独立 Trace 验证。**

## 后续验证建议

1. 从一份 GRPO 配置定位 Actor strategy、Rollout engine、采样数、Advantage estimator 和 KL；
2. 从入口脚本追踪 Trainer/Driver、Worker 初始化和一轮训练循环；
3. 把 `old/ref/new logprobs`、Reward、Advantage 和 Policy version 映射到真实 Batch；
4. 先做 CPU/config dry-run，再在合适的 Linux/CUDA 环境运行最小 smoke test。


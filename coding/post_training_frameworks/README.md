# 后训练框架实战导入

本目录连接两类证据：

1. `coding/agentic_rl_data/` 中已经验证的 PPO/GRPO 语义合同；
2. TRL、vLLM、verl、FSDP/DeepSpeed 和 Ray 中的真实组件与调用链。

当前代码仍是标准库教学实现，不冒充真实框架训练。它首先防止一种常见误判：

```text
vLLM 已生成 completion
≠ GRPO 已完成训练
```

完整主链为：

```text
Prompt
→ Rollout（completion、old_logprobs、behavior policy version）
→ Reward/Verifier
→ Group Advantage
→ Learner Policy Forward（new_logprobs）
→ GRPO Loss
→ Backward / Optimizer
→ 原子发布新 Policy
→ 下一轮 Rollout
```

## 文件阅读顺序

1. `framework_mapping.py`
   - `PipelineStage`：跨框架稳定的阶段；
   - `FRAMEWORK_COMPONENTS`：TRL、vLLM、verl、FSDP/DeepSpeed、Ray 的职责映射；
   - `GRPOBatchState`：各阶段最小字段及 shape；
   - `assess_training_progress()`：根据证据判断最多走到哪个阶段。
2. `test_framework_mapping.py`
   - completion 只能证明 rollout；
   - Reward 不自动等于 Advantage；
   - optimizer 更新不等于新权重已经发布；
   - vLLM 不参与 optimizer。
   - KL 全关闭时 Reference 不是必需角色；
   - KL reward/loss 启用时必须提供对齐的 `ref_logprobs`。
   - 非有限 Policy loss 即使伪造了 optimizer/published 字段也不能通过发布准入。
3. `environment_check.py`
   - 检查命令实际运行的 OS、Python、依赖和 NVIDIA GPU；
   - 不把用户另一台机器的硬件信息写成本次运行证据。
4. `VERL_SOURCE_TRACE.md`
   - 沿当前 V1 同步 Trainer 追踪 Reward、old/ref log-prob、Critic、Advantage、Actor Update 和权重同步；
   - 区分源码串行顺序与算法真实依赖。

## 运行

在仓库根目录执行：

```bash
python3 -m unittest coding.post_training_frameworks.test_framework_mapping -v
python3 -m coding.post_training_frameworks.environment_check
```

## 当前证据边界

- 可运行：标准库字段合同、阶段判断和环境检查；
- 未运行：PyTorch forward/backward、TRL Trainer、vLLM rollout、verl Worker/Ray；
- 下一步：固定独立 Linux/CUDA 实验环境与版本，再增加真实 TRL 最小训练。

框架接口会演进，源码学习默认按以下路径，而不是死记类名：

```text
配置/入口
→ Trainer/Driver
→ Rollout/Worker
→ Batch/Data Protocol
→ Reward/Advantage
→ Loss/Optimizer
→ Checkpoint/Weight Publish
```

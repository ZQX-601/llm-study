# verl V1 同步训练源码 Trace

- 核对日期：2026-11-12
- 上游：`verl-project/verl` 的 GitHub `main`
- 状态：静态源码阅读，未克隆、未安装、未执行 verl

> `main` 会变化，本文记录的是教学调用链，不作为可复现实验版本。真实运行前必须锁定 Git commit、
> Python/CUDA/PyTorch/vLLM 组合和完整配置。

## 阅读入口

当前发布说明指出 V1 PPO Trainer 已成为默认路径：

```text
verl/trainer/ppo/v1/
├── trainer_base.py
├── trainer_sync.py
├── trainer_colocate_async.py
└── trainer_separate_async.py
```

第一轮只读同步模式：

- [`trainer_sync.py`](https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/v1/trainer_sync.py)
- [`trainer_base.py`](https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/v1/trainer_base.py)

## `trainer_sync.py` 的职责

`PPOTrainerSync` 继承基础 Trainer，本身代码很短，主要定义同步模式生命周期：

```text
初始化结束
→ checkpoint manager 更新 Rollout 权重

采样结束
→ Rollout replicas sleep，释放权重/KV cache 等状态

训练 step 结束
→ checkpoint manager 再次更新 Rollout 权重
```

这对应教学状态：

```text
Learner UPDATED
→ 完整权重同步成功
→ Rollout PUBLISHED/LOADED
```

## `trainer_base.py` 的稳定主链

当前 V1 基础 Trainer 的同步 Batch 主链可简化为：

```text
1. Rollout / Agent Loop
2. Reward
3. Balance Batch
4. old_logprobs
5. ref_logprobs（可选）
6. Critic values（PPO 可选；标准 PPO 需要）
7. Advantage / Return
8. update_critic（可选）
9. update_actor
10. step 结束后同步新权重
```

源码中的 Reference Forward 出现在 Advantage 前，不表示 GRPO Advantage 依赖 Reference；二者在
算法依赖图中可以独立，串行顺序是当前实现的调度选择。若启用 `use_kl_in_reward`，Reference KL
会先调整 token-level reward，再进入 Advantage，这时才形成数据依赖。

## 字段演化

```text
Prompt Batch
→ responses / response_mask / rollout identity
→ token_level_scores
→ old_log_probs
→ ref_log_prob（可选）
→ values（PPO）
→ token_level_rewards
→ advantages / returns
→ Actor/Critic metrics
```

当前代码同时出现 `DataProto`、`TensorDict`、TransferQueue 和 ReplayBuffer，说明生产框架的数据协议
不仅是一个 Python dict，还要处理分布式传输、同步/异步模式、重放、失败组和恢复。第一轮阅读只追
字段与阶段，不展开所有基础设施。

## 与教学代码的对应

`framework_mapping.py::build_verl_sync_step_plan()` 用标准库重建三条条件路径：

```text
GRPO，无 KL：无 Reference、无 Critic
GRPO，有 KL：有 Reference、无 Critic
PPO，有 KL：有 Reference、Critic Values、Critic Update
```

它验证的是阶段准入，不声称复现 verl 的 Ray/FSDP/vLLM 实现。

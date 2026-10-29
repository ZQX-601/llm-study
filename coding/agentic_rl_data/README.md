# Agentic RL 数据流水线教学实现

本目录展示一条最小但完整的数据链：

```text
Agent Observation
→ behavior policy 生成 Action + old_logprobs
→ Runtime/Tool/Verifier 完成 environment.step
→ StepRecord
→ Trajectory
→ 数据校验与奖励/优势处理
→ SFT / PPO / GRPO 训练样本
→ padding batch
```

## 阅读顺序

1. `contracts.py`：先认识 Observation、Action、Decision、StepRecord、Trajectory；
2. `environment.py`：看 Agent 决策怎样经过 Runtime 变成一次环境迁移；
3. `sample_builder.py`：看同一条轨迹如何分别服务 SFT、PPO 和 GRPO；
4. `gae_token_demo.py`：用二维 token 矩阵从右向左手算 GAE；
5. `demo.py`：运行三条不同质量的 rollout，观察 reward 与 group advantage；
6. `test_agentic_rl_data.py`：用边界测试理解终止/截断、mask、staleness 和数据准入。

## 运行

从仓库根目录执行：

```bash
python3 -m coding.agentic_rl_data.demo
python3 -m coding.agentic_rl_data.gae_token_demo
python3 -m unittest coding.agentic_rl_data.test_agentic_rl_data -v
```

## 真实机制与教学简化

真实机制：

- 只在 LLM 做决策的位置形成 RL transition；Runtime 内部日志不是额外的 Policy Action；
- `old_logprobs` 必须来自实际采样 Action 的 behavior policy；
- prompt token 不进入策略 loss，response/PAD 通过 mask 区分；
- `terminated` 与 `truncated` 分开，截断末尾可做 value bootstrap；
- SFT 只把合格动作当正标签，错误动作可留作 RL/偏好负例；
- GRPO 的 rollout 必须按同一 task/prompt/environment version 分组；
- 训练前检查 policy version、Schema、token/log-prob 对齐和终止边界。

教学简化：

- `ScriptedPolicy` 代替 LLM，Unicode 编码代替真实 tokenizer；
- old log-prob 与 value 是固定模拟值；
- Tool、Verifier 和 reward 都是确定性规则；
- GRPO 把整条 Agent trajectory 视为一个 completion，并用总 reward 做组内标准化；
- PPO 只构造训练输入，未实现 clipped objective、KL、critic loss 和 optimizer step。

尚未实现：真实模型 rollout、reference/new log-prob、分布式变长 batching、数据持久化、
reward model、真实 PPO/GRPO backward，以及云 GPU 训练。因而本目录证明的是数据合同和
转换逻辑可以运行，不代表真实 Agentic RL 已完成训练。

当前 `sample_builder.py` 还包含：PPO 的 `return_targets` batch、PPO/GRPO 公共
`PolicyDecisionSample`、多轮 Agent GRPO 的 Decision flatten 与身份字段，以及按 trajectory
等权聚合的纯 Python 参考实现。真实训练时需将列表聚合替换为保持 Autograd 的 Tensor
segment/scatter 运算。

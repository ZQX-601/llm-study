# Day30 学习归档｜PPO/GRPO 样本合同、Flatten 与轨迹等权聚合

- 学习日：Day30（2026-10-29）
- 当前阶段：调整后主线第 9 周——Agentic RL 数据合同
- 状态：PPOSample 与 GRPO Group/Batch 第一轮完成；训练数据准入刚开始
- 衔接来源：[Day29](day29.md) 的 Environment/Trajectory 与 SFT/PPO 复习
- 本日方式：代码逐行讲解、shape 追踪、局部补全、合同重构、自动化测试
- 实际学习时长：未记录

## 一、本日目标与岗位意义

本日把 Day29 的 Agent `Trajectory` 转成 Trainer 可消费的 PPO/GRPO 数据，并明确多轮
Agent trajectory 与标准单轮 completion 的差异。重点解决：Prompt/Response/PAD 对齐、
决策级 Advantage 广播、GRPO 的 trajectory/decision/token 三层 shape，以及长轨迹在 Loss
中的权重偏差。

主题等级：

- `岗位必备`：PPO 的 `input_ids/response_mask/old_logprobs/advantage/return` 合同；
- `岗位必备`：GRPO 的 group 边界、组内 Advantage 与 Reward 全同的无效组；
- `岗位必备`：多轮 Agent trajectory flatten 后保留身份并按 trajectory 聚合；
- `需要掌握`：token mean 与 trajectory equal mean 的目标差异；
- `需要掌握`：PPO/GRPO 公共字段与算法专属字段的类型隔离。

对应工作决策：多轮 Agent 轨迹不能因为便于 Transformer forward 就把每次 Decision 错当成
独立 GRPO completion；同时不能让长轨迹仅因 token 更多而获得更大总训练权重。

## 二、Trajectory 到 PPOSample

本日从明确停点继续，读取 `build_ppo_samples()`：

```text
Trajectory Steps
→ 决策级 GAE advantage / return
→ 每个 Decision 生成一个 PPOSample
→ Prompt + Response 拼成 input_ids
→ response_mask 只标记 Action token
→ old_logprobs 在 Prompt/PAD 位置使用 0 占位并由 mask 排除
```

实际成功轨迹使用：

```text
rewards    = [0.08, 0.18, 1.48]
old_values = [0.20, 0.40, 0.50]
gamma=0.9, lambda=0.8
```

得到：

```text
advantages = [0.913632, 0.935600, 0.980000]
returns    = [1.113632, 1.335600, 1.480000]
```

三个样本：

| Step | Prompt | Response | 总长度 | Advantage | Return |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 106 | 81 | 187 | 0.913632 | 1.113632 |
| 1 | 136 | 82 | 218 | 0.935600 | 1.335600 |
| 2 | 177 | 73 | 250 | 0.980000 | 1.480000 |

用户正确写出 Step0 padding 到 250 后：

```text
response_mask = [0×106 | 1×81 | 0×63]
advantages    = [0×106 | A×81 | 0×63]
```

## 三、PPO Batch 缺口定位与补全

原 `collate_ppo_batch()` 缺少 Critic 监督目标。用户判断需要 `[3,250]` 并在 Response 位置
广播，但最初称为 `value`；经纠正后区分：

```text
return_targets：rollout 后固定的 Critic target
new_values：当前 Critic forward 的预测
```

用户正确写出广播主体，变量名和初始化/返回字段经提示后补齐。新增：

```text
return_targets: [B,L_max]
Prompt/PAD = 0
Response = sample.return_target
```

并增加专项测试验证 mask 与广播。

## 四、GRPO 关联复习

复习了同一 Prompt 采样 `G` 条独立 completion/trajectory 的含义：`G` 不是 token 数、
Agent step 数或 mini-batch 数，而是同一起点下的完整候选轨迹数。若 `B` 个 Prompt 每个
采样 `G` 条，总 rollout 数为 `B×G`，Reward 组内沿 `G` 维标准化。

当前三条同组轨迹：

```text
good        reward= 1.74, group_advantage≈ 0.964562
inefficient reward= 1.22, group_advantage≈ 0.413384
truncated   reward=-0.47, group_advantage≈-1.377946
```

用户正确回答：Reward 全为 `[1,1,1,1]` 时 Group Advantage 全为 0，因为没有相对学习
信号；进一步明确这表示 Reward Policy Objective 无梯度，Reference KL 仍可能有梯度。

## 五、PPO/GRPO 类型合同拆分

用户指出 GRPO 是同 Prompt 的 completion/trajectory 级相对 Advantage，而 PPO 使用
Critic/GAE，二者算法和粒度不同。精确补充：PPO Advantage 不必天然是 token 级，本项目
采用 Agent 决策级 GAE 后广播到 Action token。

原代码中 `GRPOSample.decisions` 复用 `PPOSample`，可能让 Trainer 误把 GAE Advantage 当作
Group Advantage。现已重构：

```text
PolicyDecisionSample（PPO/GRPO 公共）
├── input_ids
├── response_mask
├── old_logprobs
└── policy_version

PPOSample
= PolicyDecisionSample + GAE advantage + return_target

GRPOSample
= tuple[PolicyDecisionSample] + group_advantage
```

用户正确完成字段分类：公共字段为 `input_ids/old_logprobs/response_mask`，PPO 专属为
GAE Advantage/Return，GRPO 专属为 Group Advantage。

## 六、多轮 Agent Flatten 与身份恢复

三条 trajectory 的 Decision 数为 `[3,4,2]`。用户最初认为同 Prompt 三条 completion 的
第一维应是 3；该直觉对标准单轮 GRPO 正确。经解释明确两种合法表示：

```text
完整多轮 transcript： [G,L_trajectory_max]
Decision flatten：      [sum(D_i),L_decision_max]
```

当前数据合同为每次 Decision 保存独立 Prompt/Response，故使用第二种：

```text
3 条 trajectory
→ 9 条 Decision
→ 模型 forward batch [9,315]
```

flatten 只为 Transformer forward，不改变 GRPO 比较单位。用户正确写出：

```text
trajectory_indices = [0,0,0,1,1,1,1,2,2]
```

新增 `collate_grpo_batch()`，输出 token 张量及 `trajectory_indices/trajectory_ids/group_ids/
step_ids`，使 forward 后可以无损聚合回三条 trajectory。

## 七、Trajectory Equal Mean

三条轨迹有效 Action token 数为：

```text
good=236, inefficient=306, truncated=162
```

若直接全局 token mean，在 ratio=1 时得到约 `0.186`，即使组内标准化 Advantage 和为 0，
仍因正 Advantage 轨迹更长而产生长度权重偏差。用户选择 trajectory equal mean：先对每条
trajectory 内有效 token 平均，再让每条 rollout 等权。

新增 `trajectory_equal_masked_mean()` 纯 Python 教学实现，并验证：

```text
trajectory_means
≈ [0.964562, 0.413384, -1.377946]

final trajectory-equal mean ≈ 0
```

用户在新场景中正确手算 `[0.5,-1.0]` 两条 trajectory 的最终均值为 `-0.25`。

## 八、代码与运行证据

本日修改：

```text
coding/agentic_rl_data/__init__.py
coding/agentic_rl_data/sample_builder.py
coding/agentic_rl_data/test_agentic_rl_data.py
```

新增或修复：

```text
PPO return_targets batch
PolicyDecisionSample 公共合同
GRPO 与 PPO Advantage 类型隔离
collate_grpo_batch()
trajectory_equal_masked_mean()
四项相关回归/专项测试
```

验证状态：

```text
标准库 unittest：17 项通过
git diff --check：通过
```

没有运行真实 LLM、tokenizer、PyTorch Autograd、真实 PPO/GRPO backward、付费 API 或云 GPU。
`trajectory_equal_masked_mean()` 是纯 Python 数学验证；真实训练需用 `scatter_add_`、
`index_add_` 或 segment reduction 保留梯度。

## 九、证据边界与停点

已形成：

- 能追踪 PPOSample 的字段、shape、mask 和 Critic target；
- 能区分标准单轮 GRPO 的 `[G,L]` 与多轮 Agent Decision flatten；
- 能解释 Group Advantage 的 trajectory/decision/token 三层广播；
- 能识别长轨迹的 token-weighting 偏差并选择 trajectory equal mean；
- 代码合同经 17 项测试实际验证。

仍缺：

- 主要实现仍由教练完成，用户只完成局部表达与代码补全；
- 未实现真实 Tensor/Autograd 版本和 PPO/GRPO Loss；
- 未完成 reward/environment/tool/verifier 版本字段；
- 未完成 on/off-policy 准入、残缺组处理和分布式 group 完整性；
- 未接真实模型、Critic、Reference 或训练循环。

精确停点：

```text
父话题：Agentic RL
当前小节：训练数据准入

已导入：结构完整性、环境版本、policy lag、ratio/clipfrac、GRPO 组完整性
待回答：policy lag 合格但 environment_version 已从 v1 切到 v2，是否接纳？
后续：把版本合同和准入决策落到代码与测试
```

本日 PPOSample/GRPO Batch 小节第一轮闭合；训练数据准入只完成导入，尚未闭合。

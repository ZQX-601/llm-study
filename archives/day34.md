# Day34 学习归档｜Loss Reduction 与 PPO Critic 诊断

- 学习日：Day34（2026-11-05）
- 当前阶段：调整后主线第 11 周——Agentic RL 训练闭环
- 状态：训练指标与失败诊断第一轮完成；真实训练验证仍缺失
- 衔接来源：[Day33](day33.md) 的 Policy loss 尺度停点
- 本日方式：系统讲解、代码 Trace、教学实现、自动化测试
- 实际学习时长：未记录

> 归档边界：记录到 Critic 诊断代码验收。用户明确不学习后续
> `Agentic RL｜从诊断到可验证修复`，该小节及末尾实验题不计入本日内容，也不作为下次默认停点。

## 一、本日目标与岗位意义

主题等级：`岗位必备`。

本日解决两个直接影响训练判断的问题：一是 loss reduction 暗中决定 token、trajectory、Prompt
和 microbatch 的权重；二是 PPO Critic 指标必须在同一有效集合上联合解释，不能把 Padding、
Reward 尺度、Return 无方差或 Advantage 标准化造成的表象误判为模型能力变化。

## 二、Loss Reduction 的统计单位

系统区分了四类目标：

```text
token sum：不归一化，长度和 batch 大小直接改变梯度尺度
token mean：每个有效 Response token 等权
trajectory mean：trajectory 内 token 等权，完整 trajectory 之间等权
prompt mean：先聚合 trajectory，再让不同 Prompt 等权
```

用户正确指出直接 `sum()` 时，平均 Response 长度和有效 token 数上升会造成 Policy loss
绝对值变大；若目标是 token mean，应改成 masked sum 除以有效 mask 数。补充要求：mask 必须与
causal shift 后的位置对齐，空 Response 必须阻断，且要分清变量是 surrogate 还是已带负号的 loss。

用户正确理解全局 token mean 不保证 trajectory 等权，并给出正确顺序：先在 trajectory 内部
对有效 token 聚合，再在 trajectory 之间等权平均。不同 Prompt 的 `G` 不同时还需要 prompt 层。

## 三、Ragged Microbatch 与全局分母

对于有效 token 数不同的 microbatch，以下写法得到的是 microbatch-equal mean，而不是全局
token mean：

```python
(microbatch_mean / accumulation_steps).backward()
```

用户第一次提出直接去掉该 backward；经纠正明确 backward 不能删除，也不能只去掉除数。
正确形式是：

```text
global token mean
= microbatch masked loss sum / global valid token count
```

若函数只返回局部 mean，则先用 `microbatch_mean × valid_tokens` 恢复局部 sum，再除以全局
有效 token 数。用户起初认为这会额外重复加权长 microbatch；经解释后理解每个 token 仍只出现
一次，长 microbatch 权重更大正是 token-equal 目标的自然结果。

## 四、Return、Value 与 Advantage

完成 PPO 三者关系：

```text
raw_advantage_t = return_target_t - old_value_t
return_target_t = raw_advantage_t + old_value_t
```

其中 raw Advantage 来自 GAE，Policy 可使用其标准化副本；Critic 使用未标准化的 Return target。
不能用 normalized Advantage 反向构造 Return。

用户追问 explained variance，经示例理解：

```text
value_loss：错得多不多
value_bias：持续往哪个方向错
explained_variance：Return 的相对高低变化是否跟上
```

同时明确 EV 对固定 Bias 不敏感，必须与 Value loss 和 Bias 联合；Return 方差接近零时 EV 不可解释。

## 五、Value Loss、Value Clipping 与指标合同

用户正确指出未 mask 的 Value loss 会让 Prompt/PAD 进入监督，但第一次修改只除以 mask sum，
遗漏对 per-token loss 乘 mask 并求和。经纠正形成：

```python
value_loss = (per_token_value_loss * mask).sum() / mask.sum()
```

进一步正确判断 `torch.where(..., 0).mean()` 仍会把零填充位置计入分母，Padding 越多，监控到的
Value loss 可能虚假降低。

Value clipping 中，`old_values` 必须来自 rollout 时的固定 Critic Snapshot，并在多个 PPO epoch
保持不变。用户正确指出若每轮执行 `old_values = new_values.detach()`，则 `value_delta` 和
`value_clipfrac` 会始终伪装为零。

所有 Critic 指标必须在同一 Response mask 和版本身份上计算：Value loss、bias、EV、预测/目标
分布及 value clipfrac。用户指出无 mask 的 EV 不可信；补充空 mask、非有限值和 Return 无方差
时的边界处理。

## 六、PPO Advantage 标准化

用户追问 token-level Advantage 与谁标准化。完成以下口径：

```text
PPO：在选定 rollout 统计范围内，对所有有效 Response token Advantage 计算 mean/std
GRPO：先在同 Prompt 的完整 trajectory [G] 上计算 Group Advantage，再广播到 token
```

PPO 通常在完整 rollout batch 计算一次并固定，再 shuffle/minibatch/多 epoch 使用；不应每个
minibatch 重算。分布式场景需要跨 worker 聚合统计量。

标准化只稳定尺度，不能修复 Critic 造成的 Advantage 排序、符号和噪声错误，因此应同时记录
raw 与 normalized Advantage。用户最初把 `raw Advantage std↑ + EV↓ + Policy KL 波动` 主要
归因于 Current/Old 更新幅度；经纠正后理解 Critic 退化是更早证据，KL 波动可能是下游症状。

## 七、Critic 失败归因

面对训练 Critic 指标改善、固定评测 Critic 指标恶化且 eval success 下降，用户先归因于
Advantage 不稳定；经纠正后明确最直接证据是 Critic generalization gap、过拟合或 train/eval
状态分布偏移，需先检查版本、固定评测集、Return 分布、任务切片、Bias 和 Reference KL。

在 train/eval EV 同时下降、Bias 扩大、raw Advantage 波动增加、版本与数据合同稳定的场景中，
用户正确定位 Critic。应检查 Value LR、Value clipping、更新次数/容量、Return 尺度、旧 Value
固定和 Policy 状态分布移动速度。

当 Value loss 很低、EV 为 `None`、Return/Advantage 几乎无方差时，用户正确判断不能声称
Critic 训练良好，并提出 Prompt 过难/过易或 Reward 缺少区分度。补充：EV 为 `None` 只直接
表示分母不可解释，Reward Bug 只是多个候选原因之一。

用户理解使用 `absolute_value_bias_trend` 是为了同时捕捉过估计和低估远离零；若 EV trend
为 `None`，不能直接命中 Critic 退化，需要结合其他指标。

## 八、代码与运行证据

新增或修改：

```text
coding/agentic_rl_data/training_diagnostics.py
coding/agentic_rl_data/test_agentic_rl_data.py
coding/agentic_rl_data/README.md
```

新增机制：

- `masked_explained_variance()`：只使用有效 Response token；
- Return 无方差时返回 `None`，不伪装成 0；
- `CRITIC_SIGNAL_DEGRADED`；
- `explained_variance_trend`、`absolute_value_bias_trend`、`raw_advantage_std_trend`；
- Padding 不污染 EV、常数 Return 和 Critic 退化三项边界测试。

验证状态：

```text
标准库 unittest：46 项通过
Python py_compile：通过
git diff --check：通过
真实 PyTorch backward/Critic 更新：未运行
真实训练日志与 Dashboard：未接入
付费 API/云 GPU：未使用
```

## 九、父话题收尾与证据边界

本节结论：训练指标必须先明确统计单位和有效集合，再联合版本、任务效果、Reward 信号、Critic、
Policy 偏移和系统吞吐诊断；低 loss、正常 Queue 或标准化后正常的 mean/std 都可能是假象。

证据分类：

- 理论理解：训练指标与失败诊断第一轮完成；
- 代码产物：教学规则、masked EV 和 46 项目录测试；
- 实际运行：仅标准库教学测试，未运行真实模型训练；
- 独立验收：用户能判断主要 Trace 和局部代码错误，但实现仍主要由教练完成；
- 工程缺口：真实 Critic/Policy backward、训练日志、Dashboard、阈值校准与线上告警。

精确停点：

```text
已完成父话题：Agentic RL｜训练指标与失败诊断（第一轮）

明确跳过：Agentic RL｜从诊断到可验证修复
说明：用户选择不学习该部分，末尾单变量对照实验题未回答且不计入证据。

下一次计划学习：先做 Agentic RL 阶段回顾，再根据 ROADMAP 进入下一父话题；
               不自动返回被跳过的小节。
```

## 十、复习安排

- 1 天：换 mask/padding 场景，区分 token、trajectory 和 microbatch mean；
- 3 天：给定 Value loss/Bias/EV/raw Advantage 组合，判断 Critic、数据或 Reward 问题；
- 7 天：独立补一个 masked Critic 指标或边界测试；
- 21 天：若进入真实训练，核对 Return、Critic、Policy 与固定 eval 的同版本诊断报告。

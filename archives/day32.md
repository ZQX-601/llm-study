# Day32 学习归档｜Autograd、GRPO Tensor 聚合与权重同步

- 学习日：Day32（2026-11-02 至 2026-11-04）
- 当前阶段：调整后主线第 11 周——Agentic RL 训练闭环
- 状态：Tensor Loss/训练控制逻辑第一轮完成；权重同步第一轮完成
- 衔接来源：[Day31](day31.md) 的 `.item()`/Autograd 停点
- 本日方式：系统讲解、代码追踪、Bug 定位、教学实现、自动化测试
- 实际学习时长：未记录

> 归档边界：只记录“训练闭环与权重同步”结束前的内容。刚导入的“训练指标与失败诊断”
> 不计入本次完成内容，下一次从该主题开头重新开始。

## 一、本日目标与岗位意义

本日把准入后的 PPO/GRPO 样本推进到 loss、梯度控制、多轮 Agent trajectory 聚合，并继续
完成 Learner 更新后向 Actor 发布新 Policy 的版本闭环。

主题等级：

- `岗位必备`：Autograd 计算图、`.item()`/`.detach()`、训练 step 顺序；
- `岗位必备`：梯度累积、ragged microbatch 加权、KL early stop 与非有限梯度阻断；
- `岗位必备`：多轮 Agent GRPO 从 Decision token 恢复 trajectory 等权目标；
- `岗位必备`：Actor desired/loaded/active Policy、Snapshot 原子发布和组内版本一致性；
- `需要掌握`：LoRA Adapter 与 Base Model 的兼容身份、Queue 背压。

对应工作决策：不能让 `.item()`、错误 detach、microbatch 平均或 Decision flatten 悄悄改变
训练目标；也不能让 Actor 加载半套权重、在 trajectory 中热切换，或把不同 Policy 的
rollout 混入同一个 GRPO Group。

## 二、Autograd 与训练 step

用户要求先系统讲解今天的知识点，再开始检测。完成了以下梯度合同：

```text
保留梯度：new_logprobs、new_values、用于 loss 的 entropy/reference KL
固定目标：old_logprobs、advantages、return_targets、ref_logprobs、mask
```

用户正确指出 `.item()` 会把 Tensor 转成 Python 标量，不能覆盖随后要 backward 的 loss；
可以保留原 Tensor，并把 `loss.detach().item()` 放入单独日志变量。

用户最初认为只有 `new_logprobs` 保留梯度，经纠正后理解 `new_values` 也要通过 Value loss
训练 Critic；随后正确选出 `new_logprobs/new_values/entropy` 保留梯度。

正确训练顺序由用户独立排出：

```text
zero_grad
→ forward/loss
→ backward
→ clip_grad_norm_
→ optimizer.step
```

## 三、梯度累积、ragged microbatch 与提前停止

用户正确判断把 `zero_grad()` 放进 microbatch 循环会只保留最后一个 microbatch 的梯度；
忘记用 `accumulation_steps` 缩放 mean loss 时，等大小 microbatch 的梯度约放大相应倍数。

进一步覆盖 Agent Response 长度不等时的统计语义：

```text
全局 token mean
= sum(microbatch_mean × valid_token_count) / sum(valid_token_count)
```

用户第一次把 `[50 tokens, mean=0.4]` 与 `[10 tokens, mean=1.4]` 的加权结果算错，换例后
正确得到约 `0.567`。同时重申 token mean 与 trajectory equal mean 是不同训练目标。

用户正确区分：

```text
PPO ratio clipping：作用于 forward 的 surrogate objective
gradient clipping：作用于 backward 后的 parameter.grad
```

高 KL/clipfrac 但 grad norm 不大时，缩小 `max_grad_norm` 不能修复已经发生的 Policy 分布
偏移，应从 PPO epoch、学习率、数据新鲜度和 Actor 同步处理。

KL 双层 early stop 中，用户先漏计触发阈值的 minibatch，后在新场景中正确追踪
`processed_minibatches/completed_epochs/stop_epoch/stop_minibatch`。

## 四、Causal Shift 与固定字段

到期复习中，用户能指出错误代码没有 shift target；经补充明确 logits 也要取 `[:, :-1]`，
并正确判断 `logits[3]` 预测序列位置 4 的 `EOS`。

在代码补全中，用户写对：

```python
old_logprobs[:, 1:]
advantages[:, 1:]
return_targets[:, 1:]
response_mask[:, 1:]
```

但遗漏固定字段的 `.detach()` 和 mask dtype 转换，经纠正后形成：

```python
shift_old = old_logprobs[:, 1:].detach()
shift_advantages = advantages[:, 1:].detach()
shift_returns = return_targets[:, 1:].detach()
shift_mask = response_mask[:, 1:].to(dtype=new_logprobs.dtype)
```

用户对 `entropy.detach()` 的影响最初表述为“后面不能 backward”；经纠正后理解整个总 loss
仍可沿 Policy/Value 分支反传，但 Entropy 数值虽然进入 loss，却不贡献探索梯度。

对于比较 `entropy_coef=0` 与非零时的参数梯度，用户最初认为该验证无效；经解释后明确它
是有效 smoke test，但梯度相同只能说明 Entropy 没有可观察贡献，仍需检查
`requires_grad/grad_fn/autograd.grad` 定位原因。

## 五、Loss 合同与数值安全

PPO/GRPO 总 loss：

```text
PPO  = policy + value_coef×value - entropy_coef×entropy + kl_coef×reference_kl
GRPO = policy                    - entropy_coef×entropy + kl_coef×reference_kl
```

用户第一次补全时写反 Entropy/KL 符号并遗漏 Value 系数，经解释后能在 GRPO 公式中正确补
出 `-` 与 `+`。用户明确要求后续不再出简单算术题，后续检测调整为代码追踪、Bug 定位和
工程取舍。

训练 step 数值安全覆盖：有效 token、finite loss、finite grad norm。用户正确判断
`grad_norm=NaN` 时不得执行 `optimizer.step()`，并应保存异常 batch/trajectory；进一步
补充版本、loss 分量、ratio/KL、数值范围、AMP scale、step 和 seed 等诊断字段。

## 六、GRPO 多轮 Agent Tensor 聚合

系统解释了该机制的目的：模型 forward 为效率把不同 trajectory 的 Decision 展平成
`[D,T]`，但 GRPO Reward/Advantage 与训练样本单位仍是完整 trajectory，loss 后必须根据
`trajectory_indices` 恢复。

用户理解：

```text
先平均每个 Decision，再平均 Decision
→ trajectory 内 Decision 等权

先聚合 Decision loss sum 与 token count，再相除
→ trajectory 内 token 等权
```

`scatter_add_` 解释为 GPU 上可反传的分组累加：

```python
target[group_id] += src_value
```

用户第一次无法独立补全 `scatter_add_` 聚合代码，经完整示例后理解：

```text
Decision token loss
→ Decision sum/count
→ scatter_add_ 到 trajectory
→ trajectory sum/count
→ trajectory mean
→ trajectory_means.mean()
```

进一步明确 `trajectory_means.mean()` 是让每条完整 rollout 等权；固定 G 时也使同 Prompt
的 completion 等权。若 Prompt 的 G 不同，还要增加 group/prompt 层聚合。

用户正确判断不能把一条 trajectory 的多个 Decision 拆成独立 GRPO completion：Reward
是在完整 trajectory 上打分，先复制 Reward 再在 Decision 维标准化会让长轨迹重复计数，
改变组内均值、方差和 Advantage。正确顺序是先在 `[G]` 完整轨迹上标准化，再广播到
Decision/token，loss 后重新聚合回 `[G]`。

## 七、训练闭环与权重同步

完成 Actor/Learner 闭环：

```text
Learner 发布 Policy Snapshot
→ Actor 校验并加载
→ trajectory/group 边界锁定版本
→ rollout 保存真实版本与 old log-prob
→ 准入、训练、发布下一版本
```

用户正确判断 trajectory 中途发布新 Policy 时，应继续用旧 Policy 完成当前 trajectory；
同一 GRPO Group 的剩余 rollout 也不能单独升级，否则会混合采样分布。若旧 Group 已超出
新鲜度预算，应整体废弃并在新版本重建，而不是混组。

原子发布合同：

```text
staging 写所有分片
→ hash/size/config 校验
→ READY manifest
→ 原子更新 latest 指针
→ Actor 候选实例加载与健康检查
→ trajectory 边界切换 serving handle
```

用户正确指出分片未完成时 Actor 不能更新，应在所有分片完成后再发布 manifest/latest。

LoRA Adapter 兼容性起初未理解；经解释明确 Adapter 是相对特定 `W_base` 学到的低秩增量，
文件自身 hash 正确不代表可加载到另一个 Base Model。Actor 必须校验 Base/Adapter、
Tokenizer、Chat Template 等完整 Snapshot 身份，不兼容时保留旧模型。

版本状态拆为：

```text
desired_policy_version
loaded_policy_version
active_trajectory_policy_version
```

用户正确追踪：active v100 时请求 v101，热切换失败后应保持
`desired=101, loaded=100, active=100`；实际使用 v101 的 Actor 轨迹必须标记 v101，不能
把 desired v102 冒充实际版本。

用户正确判断自包含 READY Snapshot 可以从 v100 直接跳到 v103，无需逐版加载；增量链才
可能需要中间版本。Group 已有部分 v101 rollout 时，剩余成员继续 v101，或旧组整体重建
为 v103，不能只升级剩余成员。

最后覆盖 Queue 背压：限制 queue 长度、版本最大在途 Group、Actor 刷新优先级和 stale
准入，避免系统忙于产生注定被拒绝的旧数据。

## 八、代码与运行证据

本日新增/修改：

```text
coding/agentic_rl_data/tensor_loss.py
coding/agentic_rl_data/training_control.py
coding/agentic_rl_data/weight_sync.py
coding/agentic_rl_data/admission.py
coding/agentic_rl_data/test_agentic_rl_data.py
coding/agentic_rl_data/README.md
```

新增机制：

- PPO/GRPO 总 loss 符号合同；
- PPO 固定字段显式 detach 与空 Response 防护；
- GRPO `scatter_add_` trajectory-equal Tensor 路径；
- ragged microbatch 加权、KL 双层 early stop、optimizer step gate；
- Group 混合 behavior Policy 版本拒绝；
- Actor desired/loaded/active 版本、Snapshot 兼容与 trajectory 版本锁。

验证状态：

```text
标准库 unittest：36 项通过
Python py_compile：通过
git diff --check：通过
PyTorch：当前可用解释器均未安装
真实 backward/optimizer：用户明确选择跳过，未运行
真实 checkpoint/对象存储/Actor 热加载：未运行
付费 API/云 GPU：未使用
```

## 九、证据边界与精确停点

已形成：

- 能定位 `.item()`/detach、训练顺序、梯度累积和 early stop 问题；
- 能解释并跟踪多轮 Agent GRPO 的 Decision flatten → trajectory 聚合；
- 能区分 desired/loaded/active Policy，并判断 Group/trajectory 版本边界；
- 教学 loss、训练控制和权重同步状态机经 36 项测试验证。

仍缺：

- 新模块仍主要由教练实现，用户完成代码追踪与局部补全；
- PyTorch Tensor 函数只做语法检查，未执行 Autograd；
- 未实现/运行真实模型、Reference、Critic、optimizer、AMP 或分布式通信；
- 未连接真实 checkpoint 分片、对象存储、推理引擎和 queue；
- 用户选择跳过本机 PyTorch 安装，这一证据缺口继续保留。

精确停点：

```text
已完成父话题：Agentic RL｜训练闭环与权重同步（第一轮）

下一父话题：Agentic RL｜训练指标与失败诊断
状态：尚未开始；此前导入和题目不计入本次归档
起点：先系统建立任务效果、Policy 偏移、GRPO 信号、优化稳定性、系统吞吐五类指标，
      再做 Trace 驱动的失败归因，不从简单算术题开始。
```

## 十、复习安排

- 1 天：阅读含 `.item()`、错误 detach 和梯度累积清零的训练 step，定位最早错误；
- 3 天：换一组 ragged Decision 数据，独立说明 token/Decision/trajectory 三种平均目标；
- 7 天：给定 Actor/Learner 版本 Trace，判断继续、刷新、整组重建或拒绝；
- 21 天：在可用 PyTorch 环境实际执行一次 PPO/GRPO backward 与 grad/参数变化验证。

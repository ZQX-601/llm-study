# Day33 学习归档｜Agentic RL 训练指标与失败诊断

- 学习日：Day33（2026-11-04）
- 当前阶段：调整后主线第 11 周——Agentic RL 训练闭环
- 状态：训练指标与失败诊断第一轮部分完成
- 衔接来源：[Day32](day32.md) 的训练闭环与权重同步
- 本日方式：系统讲解、Trace 诊断、代码规则实现、自动化测试
- 实际学习时长：未记录

> 归档边界：记录到“Policy loss 不能直接代表能力”为止。末尾提出的 loss reduction
> 代码题尚未学习和回答，按用户要求留到下一次，不计入本日完成内容。

## 一、本日目标与岗位意义

主题等级：`岗位必备`。

工作中的核心任务不是只看 Reward 或 loss，而是根据任务效果、Policy 偏移、GRPO 信号、
优化稳定性和 Actor/Learner 系统指标，判断异常来自数据、Reward/Verifier、优化公式、版本
滞后还是分布式流水线，并避免聚合指标掩盖局部退化。

## 二、五层训练指标

系统建立了五类指标：

1. 任务效果：独立 eval success、Verifier pass、轨迹合法性、安全、Tool 调用、长度、成本和时延；
2. Policy 偏移：`approx_kl_to_old`、`reference_kl`、ratio、clipfrac、entropy 和 policy lag；
3. GRPO 信号：组内 Reward 均值/方差、zero-advantage group rate、Group 完整性和 Reward component；
4. 优化稳定性：Policy/Value/Entropy/KL loss、grad norm、学习率、NaN/Inf 和跳步率；
5. 系统吞吐与新鲜度：rollout/Tool 时延、Queue、stale rejection、Actor 刷新、有效准入吞吐和成本。

诊断顺序约定为：

```text
数据与版本身份
→ 独立任务效果
→ Reward/Verifier 与原始 Trace
→ GRPO 学习信号
→ KL/clip/entropy
→ loss/gradient/value
→ Queue/lag/吞吐
```

每组指标必须携带 `run/policy/behavior/environment/reward/verifier` 身份，不能把评分规则或
环境改版造成的跳变误判成模型学习效果。

## 三、Reward hacking 与 GRPO 无信号

面对 `train_reward↑、eval_success↓、Tool 调用和轨迹长度↑`，用户正确判断为 Reward
hacking 风险，并主动提出检查 Reward component、重复 Tool 调用、Verifier 漏洞、长度/成本
惩罚以及高 Reward 失败轨迹。

面对 trajectory 结果有差异、但同组 Reward 全相同、组内方差为零和 Advantage 大量为零，
用户判断 Reward/Verifier 缺少区分能力。结合已有多样化输出，诊断优先级应先检查
Verifier/Reward 映射，而不是先怪 Sampling 多样性或 Prompt 难度。

## 四、Policy 偏移、版本身份与数值阻断

用户正确指出，若每轮 PPO epoch 都用 `new_logprobs` 覆盖 `old_logprobs`，则 ratio 会被伪装
为 1，局部 KL/clipfrac 也会失去诊断意义。`old_logprobs` 必须固定为实际 rollout 的
behavior Policy 记录。

当非有限梯度与 Reward hacking 同时出现时，用户正确选择先处理
`NON_FINITE_OPTIMIZATION`：禁止 `optimizer.step()`，清理梯度并保留异常 batch、trajectory、
版本、loss component、grad norm 和 AMP 信息；Reward hacking 仍需在数值恢复后继续排查。

当 Policy 版本不变、Reward 版本从 r1 变为 r2、训练 Reward 突然上升而 eval 不变时，
用户正确判断不能声称模型能力提升，应固定同一批 trajectory 用两个 Reward 版本重放评分，
比较 component、排序和独立任务成功率，并按版本拆分 Dashboard。

用户正确区分：

```text
approx_kl_to_old：current Policy 与该批数据 behavior Policy 的局部差异
reference_kl：current Policy 与固定 SFT/Reference Policy 的累计偏移
```

因此每步局部 KL 低不代表长期没有漂移；`reference_kl↑ + eval_success↓` 仍是重要退化信号。

## 五、Actor/Learner 失衡与“健康 Queue”假象

面对 rollout 生产快于 Learner 准入、Queue/trajectory age/stale rejection 持续上升，用户正确
判断为流水线失衡，并提出 Actor 背压、减少 rollout 并发、提高 Learner 吞吐、缩短 Queue、
优先消费新 Policy 数据和提高同步频率。

补充边界：权重刷新只能发生在 trajectory 边界；若根因是 Learner 消费不过来，盲目提高发布
频率会让在途数据更快过期，并不是首要修复。

进一步讨论了 Queue 深度长期正常但 stale rejection 高、准入吞吐下降和 rollout 成本上升的
假象。用户识别出旧 behavior Policy 数据持续被拒绝。Queue 没增长是因为数据被高速丢弃，
不代表流水线健康；关键指标应比较原始 rollout 产量与真正准入训练的 trajectory 吞吐。

## 六、切片、路由偏移与聚合掩盖

整体成功率不变时，Retrieval 流量/成功率上升可能掩盖 SQL Tool Agent 的退化。用户提出
Policy 可能更多直接检索回答而不真正运行 Query。该解释对应路由偏移；统计上属于流量构成
变化掩盖局部退化。

仅按“实际选择了哪个 Tool”切片仍有选择偏差，因为 Policy 可以把困难任务移出 SQL 切片。
监控需要同时记录：

```text
task.requires_sql               决策前固定的任务要求
sql_selection_rate              整体行为分布
sql_requirement_recall          必须 SQL 时是否选择 SQL
unnecessary_sql_rate            不需要 SQL 时是否滥用
sql_execution_success           调用后是否正确执行
required_action_coverage        必需过程是否完整
```

用户指出路径选择需要评分、缺失必要操作需要惩罚；随后正确判断当 SQL 总选择率下降、必需任务
召回和成功率不变、不必要调用下降时，更像合理路由和成本改善，而不是 SQL 能力退化。

## 七、分布统计与 Policy loss 的解释边界

当 Reward mean 上升、median/p90 不变、p99/max 突升且 eval 不变时，用户正确判断是少数异常
大值拉高平均，应抽查最高 Reward trajectory。进一步应检查 Reward component、重复 Tool 奖励、
Verifier 重复计分、重复数据和 Reward 上限/去重/成本合同。

`policy_loss` 是当前采样 batch 上的优化目标，不是跨 step 的能力分数。用户指出比较前需确认
Policy、Reward 等版本；还需结合 Advantage 标准化、任务难度、有效 Response token、聚合语义、
clipfrac、ratio 和 KL。独立 eval 与过程覆盖改善时，不能仅凭 loss 数值变大宣布退化。

## 八、代码与运行证据

新增或修改：

```text
coding/agentic_rl_data/training_diagnostics.py
coding/agentic_rl_data/test_agentic_rl_data.py
coding/agentic_rl_data/README.md
```

教学诊断代码包含：

- 带版本身份的 `TrainingMetricSnapshot`；
- Reward hacking、GRPO 无信号、Policy 更新过大、Actor/Learner 失衡和非有限优化规则；
- 多个诊断可同时返回，并按阻断优先级排序；
- 原始 rollout 与准入 trajectory 吞吐差，避免低 Queue 水位掩盖高拒绝浪费；
- 健康结果仍要求查看切片指标与原始 Trace。

验证状态：

```text
标准库 unittest：43 项通过
git diff --check：通过
真实线上指标/训练：未运行
真实 PyTorch backward：未运行，沿用 Day32 的明确证据缺口
付费 API/云 GPU：未使用
```

## 九、证据边界与精确停点

已形成：

- 能按五层指标排查 Reward hacking、无学习信号、Policy 漂移、NaN 和吞吐失衡；
- 能识别 Reward 版本跳变、Queue 低水位和整体平均数造成的错误归因；
- 能区分任务结果、必需操作覆盖、Tool 路由与执行成功；
- 诊断教学规则和 43 项目录测试实际通过。

仍缺：

- 诊断模块由教练实现，用户完成 Trace 判断，但未独立扩展规则；
- 固定阈值是教学简化，尚无真实基线、窗口统计、置信区间或线上告警；
- 未接真实训练日志、Dashboard、Trace 抽样和回归评测；
- 本父话题尚未显式收尾，loss reduction、Value/Critic 和更完整诊断闭环尚未继续。

精确停点：

```text
父话题：Agentic RL｜训练指标与失败诊断
状态：部分完成
下一小节：loss reduction 与有效 token 尺度
停点：尚未学习“对 response token 直接 sum 导致长度变化放大 policy_loss”的代码题；
      下一次先系统解释 reduction，再继续代码/Trace 诊断，不直接要求回答旧题。
```

## 十、复习安排

- 1 天：给定 Reward/Policy/Environment 版本跳变 Trace，判断能否跨版本比较；
- 3 天：给定聚合指标正常但一个固定任务切片退化的案例，设计最小监控字段；
- 7 天：独立扩展一个诊断规则和边界测试；
- 21 天：接入一次真实训练或离线日志，输出按版本、任务和 Tool 切片的失败报告。

# DAY13 完整归档

- 议题：WS-160
- UUID：d9e564b5-b2e5-4ed9-b7d7-0a2caff5fc89
- 标题：day13
- 状态：in_progress
- 创建时间：2026-09-10T02:43:08Z
- 更新时间：2026-09-10T11:10:21Z
- 可见性：private

## 议题描述

请你先读取之前学习内容的归档，根据总的学习计划，开启今天的学习吧，主要以问答和讲解的形式

## 完整评论记录

### 2026-09-10T02:45:56Z｜agent

- 评论 ID：ec8b6f91-1cf0-4841-a436-438de6f76445
- 父评论 ID：无

## Day13｜同一问题下选择 SFT、DPO、PPO 还是 GRPO

今天的目标不是再背一套公式，而是建立一张后训练方法的决策地图。你已经掌握 SFT、DPO 和 PPO 的关键前置；今天先把它们放到同一任务里比较，下一轮再推导并实现 GRPO 的组内 advantage。

### 归档核对后的当前位置

已掌握：Decoder-only/SFT 的 token 与 mask；LoRA/QLoRA；DPO 的偏好对与 reference 约束；高风险结构化任务的评测、版本化与硬门槛；PPO 的 rollout、old/new logp、clip、GAE、value head、response mask 和稳定性诊断。

待巩固/未展开：把各种后训练方法按数据与反馈信号选型；GRPO 的组采样、组内相对 reward、无 critic 的优势估计，以及长度偏差与 reward 方差诊断。

说明：当前可访问的日学习归档为 Day02–12，未找到独立的 Day01 归档；不过 Day02–03 的总档已覆盖 Transformer/预训练前置，足以保守衔接今天的内容。

### 先用同一个任务比较

任务：给定一道可自动判分的数学题，让模型输出解题过程和最终答案；最终答案可由 verifier 判对错。

| 方法 | 训练时最直接需要的数据 | 一次更新最主要在做什么 | 不适合/代价 |
|---|---|---|---|
| SFT | prompt 到高质量参考答案 | 模仿参考解法的下一个 token | 只学到给定答案；不会直接比较多个可行答案谁更好 |
| DPO | 同一 prompt 下的 chosen/rejected 回答对 | 提高 chosen、压低 rejected，相对 reference 保持约束 | 需要可靠偏好对；通常是离线偏好学习，不直接利用在线采样的可验证 reward |
| PPO | rollout、标量 reward、old logp、value/GAE | 用 reward 与 critic 估计的 advantage 改 policy，同时训练 critic | actor + critic 更耗显存/复杂；value target 不稳会干扰共享骨干 |
| GRPO | 同一 prompt 的一组 rollout 和每条 reward | 用组内相对好坏形成 advantage，更新 policy | 不训练 value head，但要为同一 prompt 采样多条回答；组内 reward 要有区分度 |

关键不是“GRPO 比 PPO 新，所以总更好”，而是反馈结构不同：

SFT：我有希望模型模仿的答案。
DPO：我知道两个答案中哪个更受偏好。
PPO：我能给一次回答一个 reward，且愿意训练 critic 来降低方差。
GRPO：我能对同一题采样多条回答并评分；更关心它们的相对优劣，想省去 critic。

### GRPO 的最小直觉

对同一个 prompt x，采样 G 个回答 y1 到 yG；给每个回答 reward r1 到 rG；在这一组内做相对化得到 A1 到 AG；再使用 Ai 加权 policy 的 token-level objective，通常仍带 ratio/clip 与 reference KL。

一个常见的组内标准化形式是：

mu = mean(r1 ... rG)
sigma = std(r1 ... rG)
Ai = (ri - mu) / (sigma + epsilon)

它的含义是“这个回答在同一道题自己的候选组中相对怎样”，不是跨不同题目比较绝对 reward。组均值扮演基线角色，所以通常无需 PPO 的 value head/GAE；但 reward 全相同、sigma 很小或奖励只偏爱更长文本时，训练信号会变差或被投机，必须单独监控。

### 第 1 题：选择训练范式

针对以下四种情况，为每种选一个首选起点（SFT / DPO / PPO / GRPO），并用一句话说明理由。允许有可行替代方案，但请说明为什么不是首选。

1. 你有 20 万条经过人工审核的客服问答标准答案，没有偏好标签，也没有可自动验证的 reward。
2. 同一 prompt 下有大量人工标出的 chosen/rejected 回答对；奖励模型尚不可靠，也不计划在线采样。
3. 代码生成任务有可靠单元测试 reward；显存紧张，不希望维护 value head；一次可为同一题采样 8 个候选。
4. 对话助手只能得到一个噪声较大但连续的用户满意度/RM 分数；同一 prompt 多采样成本很高，但你愿意训练 critic 并严密监控 KL、value loss 与 reward hacking。

请按 1–4 简短作答即可。答完后我会逐项校正，并用一个 4 条 rollout 的手算例子带你算 GRPO advantage、识别“奖励全相同”的无效组，再连接到你已学过的 response_mask 与 token-level loss。

### 本周最小实践与验收

输入：一个带可验证 reward 的小任务，以及同一 prompt 的 4 条候选及其 reward。
交付物：手算一组 GRPO advantage；写出一张包含 prompt、completion、reward、group_id、response_mask 的最小 rollout 记录；列出至少 3 个训练监控项。
验收：能解释为何不需 value head、何时组内标准化无有效学习信号，以及为何 PAD 仍不能计入 response token loss。
可选进阶：用 PyTorch 实现 group-wise advantage 与 masked policy loss 的 20 行核心函数，并对全同 reward、变长回答写单元测试。

### 2026-09-10T02:50:00Z｜member

- 评论 ID：a128745d-5540-4bda-875b-70320d5ecc0b
- 父评论 ID：ec8b6f91-1cf0-4841-a436-438de6f76445

昨天的ppo好像还没有学完吧？

### 2026-09-10T02:50:40Z｜agent

- 评论 ID：ee3e1688-0d22-4b63-900b-fbd75809cdeb
- 父评论 ID：a128745d-5540-4bda-875b-70320d5ecc0b

你说得对。昨天的 PPO 只完成了概念、目标函数和数值诊断；原计划中的“最小 PPO 训练代码逐行追踪”还没有完成。我刚才提前切到 GRPO 是不合适的。

今天调整为：**先完成 PPO 的代码级闭环，再进入 GRPO。** GRPO 不会丢掉，而是作为 PPO 完整实现后的自然下一节；届时能更准确地看出它省掉了哪一段 critic/GAE 逻辑。

## 今天先完成 PPO：从 logits 到 response 的 log-prob

先只看 decoder-only PPO 中最容易写错的一段：模型输出每个位置的 logits，但第 i 个位置的 logits 预测的是第 i+1 个 token。假设 batch 内 prompt 长度已统一为 Q=3，每条 response 最多 T=4 个 token：

    # full_ids: [B, Q + T]，prompt 后接 response 和 PAD
    # logits:   [B, Q + T, V]
    logits = policy(full_ids).logits

    # 位置 i 的 logits 与目标 full_ids 的 i+1 对齐
    logp_all = F.log_softmax(logits[:, :-1], dim=-1).gather(
        dim=-1, index=full_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)                         # [B, Q + T - 1]

    # prompt 最后一个 token（位置 Q-1）的 logits，预测第一个 response token。
    # 因此从 Q-1 开始取恰好 T 个位置。
    new_logp = logp_all[:, Q - 1 : Q - 1 + T] # [B, T]

例如 Q=3, T=4 时：

    full_ids 的位置：  0      1      2      3      4      5      6
    角色：            prompt prompt prompt  r1     r2     r3     PAD
    logp_all 的位置：  0      1      2      3      4      5
    预测 target：      1      2      3      4      5      6

    response 的 token：               r1     r2     r3     PAD
    应取 logp_all：                   2      3      4      5

假设两条样本的有效 response 长度分别为 3 和 2，则：

    response_mask = tensor([
        [1, 1, 1, 0],  # 第 1 条：r1, r2, r3 有效；最后是 PAD
        [1, 1, 0, 0],  # 第 2 条：仅 r1, r2 有效
    ])                 # [B, T]

new_logp 仍会在 PAD 位置有数值，因为模型对每个位置都会给 logits；但这些位置不是 rollout 实际采取的 action。后续 ratio、policy loss、value loss、KL 的 token 均值都必须使用这个 response_mask，不能把它们当作真实 response。

### 第 1 题：沿着张量对齐作答

在上面的 Q=3, T=4 例子中，请简短回答：

1. 第一个 response token r1 的 log-prob 来自 logits 的哪个位置？为什么不是 response token 自己所在的位置？
2. logp_all 的 shape 是什么，new_logp 的 shape 又是什么？
3. 第 2 条样本的后两个 PAD 位置若不小心参与 masked mean，会造成哪两类问题？请分别说出：PAD 处数值很大时的后果，以及 PAD 处数值恰好为零时的后果。

你答完后，我们继续紧接着的两段：在同一批 rollout 上计算固定的 old_logp / old_values / GAE / returns，然后追踪一个 mini-batch 的 ratio → clipped policy loss → value loss → KL metrics → optimizer.step()。这才算把昨天的 PPO 学完；之后再比较 GRPO 如何替换其中的 advantage/critic 部分。

### 2026-09-10T06:25:03Z｜member

- 评论 ID：93e43ab9-94b7-474b-856f-00764c5064c8
- 父评论 ID：ec8b6f91-1cf0-4841-a436-438de6f76445

1、来自p的最后一个token，因为p的最后一个位置是预测r的第一个token

2、logp\_all shape \[B,T,V\] new\_logp shape \[B,T,V\]

3、PAD 处数值很大时的后果：会整体拉大loss

以及 PAD 处数值恰好为零：改变分母，使得loss变小

### 2026-09-10T06:25:35Z｜agent

- 评论 ID：988afb14-3af3-4ef4-9ace-0fc04702b985
- 父评论 ID：93e43ab9-94b7-474b-856f-00764c5064c8

第 1、3 题回答正确。第 2 题只需要修正一个关键点：`gather` 已经沿词表维 `V` 选出了“实际 token 对应的那个 log-prob”，随后 `squeeze(-1)` 又移除了长度为 1 的最后一维，因此 log-prob 不再携带 `V`。

正确 shape 链路是：

    logits:                          [B, Q + T, V]
    logits[:, :-1]:                  [B, Q + T - 1, V]
    log_softmax 后 gather 前:        [B, Q + T - 1, V]
    gather(..., index[..., 1]):      [B, Q + T - 1, 1]
    squeeze(-1) 后的 logp_all:       [B, Q + T - 1]
    response 切片后的 new_logp:      [B, T]

也就是说，`new_logp[b, t]` 是一个标量：当前 policy 对 batch 第 b 条 response 第 t 个**已采样 token**赋予的 log 概率，不是整个词表的概率向量。

第 3 题的判断也正确，而且要补全“会影响梯度与监控”这一层：

- PAD 数值很大：会制造假的 policy/value/KL loss 和梯度；例如 critic 会被迫拟合根本不存在的 PAD return，日志也会把这误诊为真实训练不稳定。
- PAD 数值恰为 0：看似无害，但如果错误地对全部位置求平均，或分母错误包含 PAD，真实 response 的 loss 与梯度会被稀释，指标会虚假变小。

因此正确模式始终是分子和分母同时使用同一份 `response_mask`：

    masked_mean = (per_token_value * response_mask).sum() / \
                  response_mask.sum().clamp_min(1)

## PPO 下一段：rollout 后一次性固定 GAE 与 returns

下面只看一条有效 response 长度为 3 的 rollout。为使计算直观，设 gamma=1、lambda=1，最终奖励只在最后一个 token 后给出：

    response token:       r1     r2     r3
    old_value V(s_t):    0.2    0.1    0.0
    reward r_t:          0      0      1
    terminal after r3:                  True

这里的 `old_value[t]` 是 rollout 当时 critic 对“生成该 response token 前的状态”的预测；这些值来自 old policy 的同一次采样快照。最后已终止，所以最后一步的 bootstrap value 按 0 处理。

先计算 TD residual：

    delta_t = reward_t + gamma * next_value_t - old_value_t

得到：

    delta_3 = 1 + 0 - 0.0 = 1.0
    delta_2 = 0 + 0.0 - 0.1 = -0.1
    delta_1 = 0 + 0.1 - 0.2 = -0.1

再从后往前做 GAE（终止处后续 advantage 为 0）：

    A_3 = 1.0
    A_2 = -0.1 + A_3 = 0.9
    A_1 = -0.1 + A_2 = 0.8

returns 不是另一次模型前向生成的量，而是固定训练目标：

    returns = old_values + advantages = [1.0, 1.0, 1.0]

对应的伪代码应在 rollout 后、进入 PPO epoch 前执行，并禁止梯度穿过这些旧量：

    with torch.no_grad():
        old_logp, old_values = rollout_policy(full_ids)  # 只保存有效 response 位
        rewards = reward_fn(prompt, sampled_response)
        advantages = compute_gae(rewards, old_values, response_mask).detach()
        returns = (old_values + advantages).detach()

随后同一批 rollout 可以做有限次 mini-batch/PPO epoch 更新。但每次重新前向得到的 `new_logp` 与 `new_values` 会变化；`old_logp`、`old_values`、advantages、returns **都不能随 epoch 重算**。否则 ratio 的“行为策略基线”不断移动，PPO 的 clipped 更新含义就被破坏了。

### 第 2 题：手算并判断固定量

仍用上面三 token 的 rollout，请回答：

1. 写出 `delta_1, delta_2, delta_3`、`advantages` 和 `returns`；可直接给向量。
2. 假设同一 rollout 做 3 个 PPO epochs。`old_logp`、advantages、returns、`new_logp`、`new_values` 中，哪些在三个 epoch 内必须固定，哪些每个 mini-batch forward 后会重新计算？
3. 如果第 2 个 PPO epoch 把 advantages 用更新后的 critic 再算一遍，为什么这不是“更准确”，反而会破坏 PPO 的训练契约？

答完这一题，我们就走最后一段代码：`ratio = exp(new_logp - old_logp)`、clip 的逐 token min、masked policy/value loss、近似 KL 与一次 `optimizer.step()`；完成后再把完整 PPO 循环归档。

### 2026-09-10T06:34:59Z｜member

- 评论 ID：59661455-2a75-47b3-b5fa-08948b58d781
- 父评论 ID：ec8b6f91-1cf0-4841-a436-438de6f76445

1、delta=\[-0.1, -0.1, 1\]

a = \[0.8 ,0.9, 1\]

returns = \[0.2,0.1,0.0\] +  \[0.8 ,0.9, 1\] = \[1,1,1\]

2、`old_logp`、advantages、returns固定

`new_logp`、`new_values` 重新计算

3、这个你再讲一次，还有new value是用来更新value head的么，他的value target是怎么算的来着？

### 2026-09-10T06:41:43Z｜agent

- 评论 ID：3c510563-996e-47d2-8223-d21405d29753
- 父评论 ID：59661455-2a75-47b3-b5fa-08948b58d781

第 1 题完全正确：

    deltas      = [-0.1, -0.1, 1.0]
    advantages  = [ 0.8,  0.9, 1.0]
    returns     = [ 1.0,  1.0, 1.0]

第 2 题也正确；再补充一个未列出的量：`old_values` 同样必须固定。完整分类是：

    rollout 后固定（无梯度）：old_logp, old_values, rewards, advantages, returns
    每个 PPO mini-batch 前向重算：new_logp, new_values

## `new_values` 的作用：是的，它用来更新 value head

可以把一次 rollout 拆成“先收集证据”与“后拿证据训练”两阶段：

    阶段 A：old policy 生成 r1, r2, r3，环境/RM 最终给出真实奖励 1
            old critic 当时预测 [0.2, 0.1, 0.0]
            基于已经发生的轨迹，得到固定监督目标 returns=[1, 1, 1]

    阶段 B：参数已开始更新。new critic 对同一条完整轨迹再预测
            new_values=[v1, v2, v3]
            让它接近固定 target [1, 1, 1]

最小 value loss 正是：

    per_token_value_loss = (new_values - returns) ** 2
    value_loss = masked_mean(per_token_value_loss, response_mask)
    total_loss = policy_loss + vf_coef * value_loss   # 先忽略其他辅助项

所以：

- `new_values` 是预测值，带梯度；`value_loss.backward()` 会更新 value head，也会经共享 Transformer/LoRA 更新骨干。
- `returns` 是监督 target，不带梯度；来自 rollout 已看到的奖励与旧 critic 的 GAE 估计，不是 value head 在第 2 个 epoch 临时输出的数。
- `old_values` 是构造 GAE/returns 时用的旧基线；它解释了当时“1 分实际回报相对旧 critic 预期有多意外”。

例如第 2 个 PPO epoch，若当前 critic 已经进步并输出：

    new_values = [0.6, 0.7, 0.8]
    returns    = [1.0, 1.0, 1.0]  # 仍是 rollout 的固定 target

则逐 token MSE 是：

    [(0.6-1)^2, (0.7-1)^2, (0.8-1)^2] = [0.16, 0.09, 0.04]

critic 应继续朝这条轨迹实际带来的回报靠近。`new_values` 不是 target，而是要被纠正的答案。

## 为什么第 2 个 epoch 不能用 new critic 重算 advantage？

PPO 的一次 rollout 类似一张已经交卷的答卷：response、行为策略的 `old_logp`、奖励、以及对该次答卷计算出的 advantage 都属于历史证据。随后允许当前 policy 对这张答卷做有限次更新，但 `ratio = exp(new_logp - old_logp)` 必须始终相对同一个产生答卷的 old policy，policy loss 中的 advantage 也必须指向同一份固定评价。

如果第 2 个 epoch 用更新后的 critic 重算 advantage，会有两个问题：

1. 同一条已发生轨迹的 policy 学习信号会漂移。最终 reward 没变，只是 critic 变了；例如旧 critic 低估回报时 `A1=0.8`，训练几步后新 critic 更接近回报，重算的“惊喜”会缩小。此时 policy 在多次 epoch 中不再依据同一份 rollout 评价更新，而是在不断修改打分尺。
2. 若 advantage/return 没有以 old rollout 的结果 `detach()` 固定，actor 可通过共享 backbone 间接影响 critic 预测，从而让 advantage 看起来更有利，而非只通过改变 token 概率改进 policy。这正是 `advantages.detach()` 的必要性。

新 critic 的预测当然有价值：它用固定 returns 学习，并在**下一次新 rollout**中成为新的 `old_values`，为新采样 response 构造下一批 GAE。时序是：

    rollout k：old policy/critic → reward → fixed A_k, return_k
    rollout k 上有限 PPO epochs：重算 new_logp/new_values；固定 A_k, return_k
    rollout k+1：更新后的 policy/critic 重新采样，才产生新的 old_logp/old_values/A/return

### 第 3 题：进入最后的 PPO update

同一条有效 response 的三个 token 上，设：

    old_logp  = [-1.0, -1.0, -1.0]
    new_logp  = [-0.8, -1.3, -1.0]
    advantage = [ 0.8,  0.9,  1.0]  # 已 detach、固定
    returns   = [ 1.0,  1.0,  1.0]  # 已 detach、固定
    new_values= [ 0.6,  0.7,  0.8]
    epsilon   = 0.2

请先计算或判断：

1. 每个 token 的 `ratio = exp(new_logp - old_logp)`；可保留三位小数。哪些 token 的 ratio 被 clip 到边界？
2. PPO policy 的逐 token surrogate 是 `min(ratio * advantage, clip(ratio, 0.8, 1.2) * advantage)`。三个 advantage 都为正时，哪个 token 的 policy 改善会被 clip 阻止继续放大？为什么？
3. 用上面固定 returns 算三个 token 的 value MSE，并说明 `vf_coef` 从 0.5 改成 0.1 时，policy 梯度与 value 路径梯度各怎样变化。

答完后我们把 policy loss、masked mean、approx-KL 和 `optimizer.step()` 接成一段完整最小代码，PPO 就完成了。

### 2026-09-10T08:57:03Z｜member

- 评论 ID：52f2054d-c09b-473f-8b14-68b6ddbf72ca
- 父评论 ID：无

你在么

### 2026-09-10T09:39:50Z｜agent

- 评论 ID：9c178bde-588f-46d6-ae02-6df2873e1f3f
- 父评论 ID：52f2054d-c09b-473f-8b14-68b6ddbf72ca

在的。

我们当前停在 PPO 的最后一组练习：计算三个 token 的 `ratio / clip`、value MSE，以及 `vf_coef` 对 policy 与 value 两条梯度路径的影响。你可以直接继续答刚才那三题；如果有哪一段没看清，也可以告诉我，我从那里重新讲。

### 2026-09-10T09:40:44Z｜member

- 评论 ID：54a04bf2-488e-4e24-9b23-96dd2aeed8c9
- 父评论 ID：ec8b6f91-1cf0-4841-a436-438de6f76445

1、\[-0.8, -1.3, -1.0\] - \[-1.0, -1.0, -1.0\] = \[0.2, -0.3, 0\] ratio = \[1.22, 0.74， 1\]，1、2的都会被clip

2、1

3、\[ 0.6,  0.7,  0.8\] - \[ 1.0,  1.0,  1.0\] = \[-0.4, -0.3, -0.2\] mse = \[0.16, 0.09, 0.04\] /3,

vf\_coef从 0.5 改成 0.1 时,policy梯度会变小，value梯度不变化

### 2026-09-10T09:56:11Z｜agent

- 评论 ID：2ff2a62d-c029-4d2d-9be5-d52700707234
- 父评论 ID：54a04bf2-488e-4e24-9b23-96dd2aeed8c9

上一轮会话上下文未能恢复；我已从本议题线程重建到这道 PPO 题。

第 1、2 题的主要计算正确，第 3 题的梯度结论需要反过来。

### 1. ratio：两个越界，但最终只有 token 1 启用裁剪

    logratio = new_logp - old_logp = [0.2, -0.3, 0]
    ratio = exp(logratio) ≈ [1.221, 0.741, 1.000]
    clamp(ratio, 0.8, 1.2) = [1.2, 0.8, 1.0]

所以，如果问“哪些 ratio 超出区间并被 `clamp`”，答案确实是 token 1、2。

但 PPO 还要在原 surrogate 与裁剪 surrogate 之间取较小值。三个 advantage 都为正：

    ratio * A         ≈ [0.977, 0.667, 1.000]
    clipped_ratio * A = [0.960, 0.720, 1.000]
    min(...)           = [0.960, 0.667, 1.000]

- token 1：新策略把好动作的概率提高得过多，取 `0.960`，裁剪真正生效，不再奖励继续增大概率。
- token 2：好动作的概率反而降得过多，原项 `0.667` 更小，因此不会用 `0.720` 替它“减轻惩罚”。它虽然越过下边界，但最终 policy objective 没走裁剪分支。

这正是 `min` 的意义：正 advantage 时限制过大的概率提升，但不会保护错误的概率下降。若 advantage 为负，方向会反过来，下边界才会阻止坏动作的概率被一次降得过多。

因此你的第 2 题“token 1”是正确的。三 token 都有效时：

    policy_loss = -mean([0.960, 0.667, 1.000]) ≈ -0.876

### 2. value MSE 正确，`vf_coef` 的影响需修正

    squared_error = [0.16, 0.09, 0.04]
    value_loss = (0.16 + 0.09 + 0.04) / 3 ≈ 0.0967
    total_loss = policy_loss + vf_coef * value_loss

`vf_coef` 从 `0.5` 改为 `0.1` 后：

- **policy loss 自身产生的梯度不变**，因为它不含 `vf_coef`；
- **value loss 路径产生的梯度变成原来的 1/5**，包括 value head，以及共享骨干中来自 value loss 的那部分梯度；
- 若 actor 与 critic 共享骨干，骨干的最终总梯度是 `policy 梯度 + vf_coef × value 梯度`，所以总更新会变化，但不能简单说“policy 梯度变小”；若两者完全分离，actor 梯度完全不受影响。

### 3. 最小 PPO update 已经可以闭环

    def masked_mean(x, mask):
        mask = mask.to(x.dtype)
        return (x * mask).sum() / mask.sum().clamp_min(1.0)

    # old_logp / advantages / returns 均来自 rollout，已 detach 并固定
    new_logp, new_values = model_forward(minibatch)

    logratio = new_logp - old_logp
    ratio = logratio.exp()

    surrogate_1 = ratio * advantages
    surrogate_2 = ratio.clamp(1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -masked_mean(
        torch.minimum(surrogate_1, surrogate_2), response_mask
    )

    value_error = (new_values - returns).pow(2)
    value_loss = masked_mean(value_error, response_mask)

    total_loss = policy_loss + vf_coef * value_loss
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    with torch.no_grad():
        approx_kl = masked_mean(
            (ratio - 1) - logratio, response_mask
        )
        clipfrac = masked_mean(
            ((ratio < 1 - epsilon) | (ratio > 1 + epsilon)).float(),
            response_mask,
        )

这里 `clipfrac = 2/3`，它统计的是“ratio 越界比例”，所以不能解读成“2/3 的 token 最终都走了裁剪 surrogate”。`approx_kl` 用于监控或触发 early stop，不应和梯度裁剪混为一谈。实际实现还可加入 value clipping、entropy bonus、reference KL，但不影响这条最小主链。

最后一道验收题：为什么本例 `clipfrac=2/3`，但真正被裁剪阻止继续放大 policy 改善的只有 token 1？请用一句话回答。答对后，PPO 的代码级闭环就完成，我们再进入 GRPO。

### 2026-09-10T10:21:37Z｜member

- 评论 ID：6151e748-2dfc-434b-a1bc-1400f1860cb1
- 父评论 ID：ec8b6f91-1cf0-4841-a436-438de6f76445

因为a大于1，只有token1的ratio大于1.2，说明new相对于old是朝着好的方向去提升

### 2026-09-10T10:27:11Z｜agent

- 评论 ID：2ee6011a-e3b3-48d2-af50-2c6233a37cd7
- 父评论 ID：6151e748-2dfc-434b-a1bc-1400f1860cb1

结论正确，术语上修正一处：这里不是 `A > 1`，而是三个 token 的 **advantage 都大于 0**。

- `clipfrac=2/3` 只检查 ratio 是否越过 `[0.8, 1.2]`，所以 token 1、2 都计入。
- token 1：`A>0` 且 `ratio=1.221>1.2`，说明新策略沿正确方向提高了这个好动作的概率，但提高过猛；`min` 选择裁剪项，阻止继续放大。
- token 2：`A>0` 但 `ratio=0.741<0.8`，说明新策略反而降低了好动作的概率；`min` 保留更小的未裁剪项，不替这个错误方向减轻惩罚。

因此你的核心判断成立：**ratio 越界不等于最终 surrogate 一定走裁剪分支；还必须结合 advantage 的正负和 `min` 的结果。** 至此 PPO 的代码级闭环完成。

## 下一步：GRPO 用什么替换 critic 和 GAE

两条主链对照如下：

    PPO：同一 prompt 采样 rollout
         → reward + old_values
         → GAE / returns
         → policy loss + value loss

    GRPO：同一 prompt 采样 G 条 completion
          → 分别评分 r1...rG
          → 组内相对 advantage
          → clipped policy loss + reference KL

GRPO 不再训练 value head，也没有 value loss、returns 和 GAE。一个常见的基本形式是：

    mu = mean(rewards)
    sigma = population_std(rewards)
    A_i = (r_i - mu) / (sigma + eps)

这里的基线是“同一道题其他候选的平均表现”，不是 critic 的 `V(s_t)`。如果采用 completion-level reward，同一条 completion 的 `A_i` 通常会广播到它所有有效 response token；PAD 仍由 `response_mask` 排除。

### GRPO 第 1 题：手算组内 advantage

同一个 prompt 采样 4 条 completion，奖励为：

    rewards = [0, 1, 1, 2]

请回答：

1. `mu`、总体标准差 `sigma` 和四个标准化 advantage 分别是多少？可保留三位小数。
2. 哪些 completion 的概率应被提高、降低或基本不变？
3. 如果奖励变成 `[1, 1, 1, 1]`，这一组为什么不能提供相对 policy 学习信号？

答完后我们把这四个 sequence-level advantage 广播到变长 token，并写出带 `group_id`、`response_mask`、ratio/clip 和 reference KL 的最小 GRPO 代码。

### 2026-09-10T10:29:25Z｜member

- 评论 ID：0aff58a3-f639-43c8-aa79-c2c3fde95077
- 父评论 ID：ec8b6f91-1cf0-4841-a436-438de6f76445

今天先学习到这里吧，把今天的学习内容归档，方便后天继续学习

### 2026-09-10T10:30:15Z｜agent

- 评论 ID：dd454cd9-d7c3-4001-9ed2-54a6bed53351
- 父评论 ID：0aff58a3-f639-43c8-aa79-c2c3fde95077

## Day13 学习归档｜PPO 代码级闭环与 GRPO 入口

已归档。今天的结论是：**PPO 已完成从张量对齐、GAE/returns 到一次参数更新的代码级闭环；GRPO 只完成入口对照，组内 advantage 手算尚未开始。** 后天（2026-09-12）直接从文末续学题开始。

### 一、今天已掌握

#### 1. Decoder-only response log-prob 对齐

位置 `i` 的 logits 预测位置 `i+1` 的 token。若 prompt 长度为 `Q`、最大 response 长度为 `T`：

    logits:        [B, Q + T, V]
    logp_all:      [B, Q + T - 1]
    new_logp:      [B, T]
    response slice = logp_all[:, Q - 1 : Q - 1 + T]

`gather` 已从词表维选出实际采样 token 的 log-prob，`squeeze(-1)` 后不再有 `V` 维。第一个 response token 的 log-prob 来自 prompt 最后一个位置的 logits。

#### 2. response mask

PAD 不是 rollout 实际采取的 action。policy loss、value loss、KL 等都必须同时在分子和分母使用同一份 `response_mask`：

    masked_mean(x, mask) = (x * mask).sum() / mask.sum().clamp_min(1)

PAD 数值很大会制造虚假 loss/梯度；即使 PAD 数值恰好为 0，若分母仍包含 PAD，也会稀释真实 token 的 loss。

#### 3. rollout 后固定量与每轮重算量

    固定、无梯度：old_logp, old_values, rewards, advantages, returns
    每个 mini-batch forward 重算：new_logp, new_values

不能在第 2 个 PPO epoch 用更新后的 critic 重算 advantage：同一批历史 rollout 必须使用同一把评价尺。更新后的 critic 应在下一批新 rollout 中成为新的 `old_values`。

#### 4. GAE 与 value target 手算

在 `gamma=1, lambda=1`、奖励 `[0,0,1]`、旧 value `[0.2,0.1,0]` 的例子中：

    deltas      = [-0.1, -0.1, 1.0]
    advantages  = [ 0.8,  0.9, 1.0]
    returns     = old_values + advantages = [1.0, 1.0, 1.0]

`new_values` 是带梯度的预测，`returns` 是固定监督 target：

    value_loss = masked_mean((new_values - returns) ** 2, response_mask)

当 `new_values=[0.6,0.7,0.8]` 时，逐 token MSE 为 `[0.16,0.09,0.04]`，均值约 `0.0967`。

#### 5. PPO ratio、clip 与 surrogate

    ratio = exp(new_logp - old_logp)
    surrogate = min(ratio * A, clamp(ratio, 0.8, 1.2) * A)

本次例子：

    ratio             ≈ [1.221, 0.741, 1.000]
    clipped ratio      = [1.200, 0.800, 1.000]
    A                  = [0.8, 0.9, 1.0]
    final surrogate    ≈ [0.960, 0.667, 1.000]
    policy_loss        ≈ -0.876

token 1、2 的 ratio 都越界，所以 `clipfrac=2/3`；但三个 advantage 均为正时，只有 token 1 是“提高好动作概率过猛”，最终 `min` 选择裁剪分支。token 2 降低了好动作概率，最终保留未裁剪的更小项，不为错误方向减轻惩罚。

因此要牢记：**ratio 越界不等于最终 surrogate 一定走裁剪分支；必须结合 advantage 的正负与 `min` 的选择。**

#### 6. `vf_coef` 与梯度

    total_loss = policy_loss + vf_coef * value_loss

`vf_coef` 从 `0.5` 降到 `0.1`：policy loss 自身的梯度不变；value 路径梯度变为原来的 `1/5`。共享骨干的总梯度会因 value 分量缩小而变化，但不能说“policy 梯度变小”。

#### 7. 最小 PPO 更新链路

    rollout
      → 固定 old_logp / old_values / rewards
      → 固定 GAE advantages / returns
      → forward 得到 new_logp / new_values
      → ratio 与 clipped policy loss
      → masked value loss
      → total_loss.backward()
      → gradient clipping
      → optimizer.step()
      → 监控 approx_kl / clipfrac

`approx_kl` 用于训练监控或 early stop；它、PPO ratio clipping、梯度范数 clipping 是三个不同概念。

### 二、今天纠正的易错点

1. `logp_all/new_logp` 是标量 log-prob 张量，不带词表维 `V`。
2. mask 不只是把 PAD 乘 0，求均值的分母也必须只统计有效 token。
3. PPO epoch 内必须固定 `old_values/advantages/returns`，不能随 critic 更新重算。
4. `vf_coef` 缩小影响 value 路径梯度，不直接缩小 policy loss 的梯度。
5. 判断 clip 是否真正生效，条件是 advantage **大于或小于 0**，不是是否大于 1。
6. `clipfrac` 统计 ratio 越界，不等于最终选择 clipped surrogate 的比例。

### 三、GRPO 已建立的入口

    PPO：reward + critic old_values
         → GAE / returns
         → policy loss + value loss

    GRPO：同一 prompt 的 G 条 completion + 各自 reward
          → 组内相对 advantage
          → clipped policy loss + reference KL

常见组内标准化：

    mu = mean(rewards)
    sigma = population_std(rewards)
    A_i = (r_i - mu) / (sigma + eps)

GRPO 用同题候选的平均表现作为基线，不训练 value head，也没有 value loss、returns 和 GAE。若采用 completion-level reward，同一条 completion 的 advantage 会广播到它的有效 response token，PAD 仍必须排除。

### 四、2026-09-12 续学入口

不要重讲 PPO，直接从这道未完成题开始：同一 prompt 的四条 completion 奖励为：

    rewards = [0, 1, 1, 2]

1. 手算 `mu`、总体标准差 `sigma` 与四个标准化 advantage。
2. 判断哪些 completion 的概率应提高、降低或基本不变。
3. 解释 `[1,1,1,1]` 为什么不能提供组内相对学习信号。
4. 随后把 sequence-level advantage 广播到变长 token，写出包含 `group_id`、`response_mask`、ratio/clip 和 reference KL 的最小 GRPO 代码。

下一次验收标准：能独立算组内 advantage；能解释 GRPO 为什么不需要 critic，以及全同 reward、PAD 和长度偏差分别会造成什么问题。

### 2026-09-10T11:10:21Z｜member

- 评论 ID：23b227c7-2646-44d6-bfc9-fe1cf2114e0e
- 父评论 ID：ec8b6f91-1cf0-4841-a436-438de6f76445

你将当前项目有关的存档，学习计划，以及项目的一切内容copy一份到/workspace/study下

## 元数据

```json
{}
```


# Padding、No-padding 与 Packed Sequence 布局

- 日期：2026-11-12
- 类型：主线相关 QA 专题
- 状态：用户自评掌握，待 Tensor Trace 与真实 Kernel 验证
- 关联主线：verl Actor Update 与变长 Batch

## 原问题

> “可能把 padding layout 转成 no-padding/packed layout”这个是干嘛的？

## 一、核心动机

普通 Batch 为了形成规则 `[B,T_max]` Tensor，会把短序列补齐到最长序列。Padding 不参与有效
Loss，但 Dense Forward 仍可能在这些位置执行 Embedding、Attention、MLP 和输出层计算。Agent/RL
Rollout 的长度差异通常较大，因此浪费明显。

verl 的 RL Batch 常包含左 Padding 的 Prompt 和右 Padding 的 Response。`attention_mask` 表示真实
存在的 token，`response_mask` 表示模型生成、应参与 Policy Loss 的 Response token；两者职责不同。

## 二、Remove Padding 与 Packing

Remove Padding 根据 `attention_mask` 抽取所有有效 token：

```text
[B,T_max]
→ [total_valid_tokens]
```

Packing 将多条变长序列放入一个连续计算 buffer，并保留边界元数据。物理上连续不代表语义上是一条
序列；Variable-Length Attention 必须使用累计边界，例如：

```text
lengths = [5, 6]
cu_seqlens = [0, 5, 11]
```

从而保证第二条序列不能看到第一条序列。每条序列的 position IDs 通常需要独立从零开始。

## 三、所有 token 级字段必须同步重排

变换 `input_ids` 时，还必须按相同索引处理：

```text
position_ids
response_mask / loss_mask
old_logprobs
ref_logprobs
advantages
values / returns（PPO）
```

否则 ratio、KL 或 Loss 会与错误 token 对齐。Prompt token 在 Attention 中需要可见，但通常不参与
Policy Loss，所以不能用 `attention_mask` 代替 `response_mask`。

## 四、verl 中的往返转换

当前 Trainer 路径可概括为：

```text
Padded RL Batch [B,T]
→ left_right_2_no_padding
→ Worker 以 no-padding/packed layout Forward
→ no_padding_2_padding
→ 恢复 [B,T] 并与 Reward、Advantage、Trajectory identity 对齐
```

当前实现仍在 Trainer/Worker 边界做格式转换；官方文档也说明正在推进更完整的 no-padding 数据流。

## 五、收益与风险

收益：逐 token 层从处理 `B×T_max` 个位置接近下降到处理所有真实长度之和；Variable-Length
Attention 也避免在大量 PAD 上做无效计算。长度差异越大，收益通常越明显。

主要风险：

1. `cu_seqlens` 错误导致跨样本 Attention；
2. position IDs 未按序列重置；
3. token 级训练字段没有同步重排；
4. Prompt/PAD 被错误纳入 Policy Loss；
5. 模型或 Kernel 不支持相应 Variable-Length 路径；
6. 并行/硬件对齐要求可能引入少量物理 alignment padding，不能把它误当语义 token。

## 官方一手资料

- [verl Packed Sequence 实现](https://github.com/verl-project/verl/blob/main/verl/models/mcore/util.py)
- [verl Trainer padding/no-padding 转换](https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/ray_trainer.py)
- [verl Model Engine 数据格式](https://github.com/verl-project/verl/blob/main/docs/workers/model_engine.rst)
- [verl 性能调优指南](https://github.com/verl-project/verl/blob/main/docs/perf/perf_tuning.rst)

## 用户追问与当前证据

- 用户在系统讲解后表示“明白了”；
- 理论证据：理解 Padding 浪费、Remove Padding、连续 Packing、边界元数据和字段同步；
- 代码阅读：已对照 verl 当前 Trainer/packed utility 的静态路径，未逐行阅读实现；
- 实际运行：未安装 PyTorch/verl，未运行 padded 与 unpadded 数值对照或吞吐测试；
- 独立验收：尚未给定具体 Tensor 独立重排 `cu_seqlens`、position IDs 和 response mask。

结论：**用户自评掌握，待代码与运行验证。**

## 后续验证建议

1. 给定两条 Prompt/Response，独立写出 padded 与 packed token、`cu_seqlens`、position IDs 和 mask；
2. 增加跨样本 Attention、字段错位和全 Padding 的边界测试；
3. 在支持的 CUDA 环境对比 padded/unpadded 输出、峰值显存和吞吐；
4. 从 verl Actor Worker 追踪 no-padding 输入到 loss reduction 的实际 shape。


# `sparse_attention` 数据流拆解

## 问题

解释 `coding/deepseek_demo/swa_attention.py` 中 `sparse_attention` 的作用和逐步数据流。

## 讲解摘要

- 输入为 Q `[B,T,H,Dh]`、共享 KV `[B,S,Dh]`、局部索引 `[B,T,K]` 和每头 sink logit `[H]`。
- 先将无效索引 `-1` 临时替换为 0，再从 KV 中直接 gather 每个 query 的 `K` 个候选，得到 `[B,T,K,Dh]`。
- 通过 `einsum("bthd,btkd->bthk")` 计算各头局部分数 `[B,T,H,K]`，并用 `1/sqrt(Dh)` 缩放。
- 无效索引的 score 被设为 `-inf`。
- 每个头的 attention sink 作为额外 logit 参加 softmax 分母，但其 value 为零，因此局部 KV 权重和可小于 1。
- 最后沿 K 对候选 KV 加权求和，输出 `[B,T,H,Dh]`。
- 该实现先 gather 后打分，不构造完整 `[B,H,T,S]` 分数矩阵；当 `K<=W<<S` 时降低计算和显存开销。

## 与主线的关系

这是 DeepSeek V4.1 风格局部 SWA 教学实现的核心注意力数据流答疑，承接局部索引和环形缓存的讲解。

## 薄弱点

用户正在追踪 shape、gather、einsum 和归一化数据流；本次为主动答疑，不新增复习任务或判定掌握程度。

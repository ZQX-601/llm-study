# `rotate_tail`（RoPE tail）逐行拆解

## 问题

逐行解释 `coding/deepseek_demo/swa_attention.py` 中 `rotate_tail` 如何只旋转最后 `rope_dim` 个特征。

## 讲解摘要

- 函数接受 Q `[B,T,H,D]` 或共享 KV `[B,T,D]`，输出 shape 不变。
- 默认 `rope_dim=4` 时生成两个频率 `[1, 0.01]`；位置 `p` 的两个旋转角为 `[p, 0.01p]`。
- 尾部四维 reshape 成两个二维向量，相邻维度分别使用对应角度做二维旋转。
- Q 的 `cos/sin` 广播形状是 `[1,T,1,2]`，KV 是 `[1,T,2]`。
- `inverse=True` 将角度取负，用于注意力输出尾部的逆旋转。
- 前 `D-rope_dim` 个特征原样保留，旋转尾部再拼回，因此整体 shape 不变。

## 与主线的关系

这是 DeepSeek V4.1 风格局部 SWA 教学实现的代码数据流答疑，帮助理解 RoPE 的频率、广播和尾部维度布局。

## 薄弱点

用户在追踪张量 shape 和广播规则；本次为主动答疑，不据此新增复习任务或判定掌握程度。

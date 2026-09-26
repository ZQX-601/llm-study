# DeepSeek V4.1 Flash：mHC 与 Single-Pass mHC

- 日期：2026-09-26
- 模式：QA 专题学习
- 状态：用户明确表示 mHC 部分已经明白；记录为“用户自评掌握，待复习验证”，不等同于独立验收通过。
- 重要性：残差连接、训练稳定性属于岗位必备；mHC 和 Single-Pass mHC 属于需要掌握的前沿架构案例。

## 本次学习范围

1. mHC 全称 Manifold-Constrained Hyper-Connections，用多路 residual stream 替代单路残差。
2. 输入 embedding 不是经过 `d→4d` 投影，而是直接复制为四条相同的初始流。
3. `pre/post/comb` 不是固定参数，而是由可学习的 `fn/base/scale` 根据每个 token 的当前多路状态动态生成。
4. `pre` 完成多路到一路，`post` 完成一路到多路，`comb` 完成多路到多路。
5. `comb` 经过 Sinkhorn 形成近似双随机矩阵，限制深层信号和梯度的放大/衰减。
6. Attention/MoE 只处理 `pre` 合并后的单路表示，因此不会执行四份主干计算；主要额外成本来自 residual 激活、路由和显存读写。
7. Single-Pass mHC 将 `pre_mix` 的使用错后一子层，以减少重复读取多路 residual，并方便 kernel 融合。

## 核心公式

设 `n=4`，每个 token 的多路状态为：

```text
X:[n,d] = [4,d]
```

mHC 更新公式：

```text
X_next = comb @ X + post.T @ F(pre @ X)
```

三种映射：

```text
pre:[1,n]   @ X:[n,d] -> sublayer_input:[1,d]
comb:[n,n]  @ X:[n,d] -> mixed_residual:[n,d]
post.T:[n,1] @ y:[1,d] -> write_back:[n,d]
```

其中 `F` 是一次 Attention 或 MoE。

## 输入如何变为四路

官方参考实现：

```text
h = embedding(input_ids)                 -> [B,T,d]
h = h.unsqueeze(2)                       -> [B,T,1,d]
h = h.repeat(1,1,4,1)                    -> [B,T,4,d]
initial_pre_mix = [1,0,0,0]
```

四条流初始完全相同，不是把 `d` 切成四份，也不是四个投影结果。第一次 `hc_post` 使用不同的 `post` 写回子层输出后，各条流开始分化。

## 动态系数如何生成

对每个 token：

```text
X:[4,d]
  -> flatten:[4d]
  -> RMS 缩放
  -> 与 hc_fn:[24,4d] 做线性投影
  -> raw:[24]
```

因为：

```text
pre 需要 4 个数
post 需要 4 个数
comb 需要 4×4=16 个数
总计 24
```

最终约束：

```text
pre  = sigmoid(raw_pre  * scale_pre  + base_pre) + eps
post = 2*sigmoid(raw_post * scale_post + base_post)
comb = Sinkhorn(raw_comb * scale_comb + base_comb)
```

`pre/post` 非负但不要求和为 1；`comb` 非负，行和、列和均近似为 1。Attention 与 FFN 各有独立的 `hc_fn/base/scale` 参数。

## 为什么需要 pre、post、comb

- `pre`：当前子层按 token 动态选择从哪些 residual stream 读取信息。
- `post`：控制子层输出写入哪些 stream、写入多少，降低所有更新无条件污染同一状态的风险。
- `comb`：让旧 stream 在层间交换信息；其双随机约束保持稳定的凸组合传播。

三者提供 read、write、state transition 三种职责，在不扩大 Attention/MoE 内部维度的情况下增加跨层连接拓扑和信息容量。

## Single-Pass 时序

```text
模型入口 one-hot pre_mix
    -> 第 0 层 Attention 使用
第 0 层 Attention 产生 attn_pre
    -> 第 0 层 MoE 使用
第 0 层 MoE 产生 ffn_pre
    -> 第 1 层 Attention 使用
```

当前子层使用上一子层提前计算的 `pre_mix`；当前子层生成的系数供下一子层使用。这样可以在一次读取 residual 时同时完成当前输入混合和下一组系数预测。

## 教学源码与验证

- [mHC 两路手算和四路入口](../coding/deepseek_demo/mhc_manual_demo.py)
- [完整 pre/post/comb 生成与应用](../coding/deepseek_demo/mhc_coefficients_demo.py)

代码已运行并验证：

- `[B,T,d] -> [B,T,4,d]` 的复制入口；
- 动态 `pre_mix` 生成；
- `pre @ X`、`comb @ X`、`post.T @ y` 的 shape；
- Sinkhorn 后 `comb` 的行和、列和近似为 1。

## 初始化证据边界

- 官方参考推理源码明确给出模型入口 `pre_mix=[1,0,0,0]`。
- mHC 原论文报告动态 gating factor `alpha` 使用 `0.01` 初始化。
- V4.1 公开推理源码中的 `hc_fn/base/scale` 通过 `torch.empty` 声明后从 checkpoint 加载，没有公开完整训练初始化代码，因此不推定其精确 `fn/base` 初始化值。
- 论文中用于关闭某项映射时的固定基准是：`pre=1/n`、`post=1`、`comb=I`；这是消融基准，不能自动等同于 V4.1 的实际初始化。

## 复习安排

- 2026-09-29：给定 `n=2,d=2` 的 `X/pre/post/comb/y`，手算一次完整更新。
- 2026-10-05：从 `[B,T,4,d]` 追踪 `raw/pre/post/comb` 的 shape，并解释哪些是参数、哪些是动态激活。
- 2026-10-19：比较普通 residual、无约束 HC、mHC 和 Single-Pass mHC 的能力、稳定性与系统成本。

## 一手资料

- [mHC 原论文](https://arxiv.org/html/2512.24880v2)
- [DeepSeek-V4.1-Flash 技术报告](https://arxiv.org/html/2609.19969v1)
- [V4.1 官方参考模型](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py)
- [V4.1 官方 mHC kernel](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/kernel.py)

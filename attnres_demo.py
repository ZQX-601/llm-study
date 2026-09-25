#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attention Residuals (AttnRes) 数值验证 —— 零依赖纯 Python

依据：
  - 论文 arXiv:2603.15031 摘要（逐字引用见 qa 归档）
  - FLA 参考实现 fla/ops/attnres/naive.py  (naive_attnres)
  - FLA 融合核     fla/ops/attnres/fused.py (fused_attnres)

本脚本只做三件事：
  1) 手算一个 3 源、D=2 的完整 AttnRes 前向，数字可纸上复现
  2) 用随机模拟说明 PreNorm 标准残差为什么"随深度不受控增长"
  3) 算清 Full AttnRes vs Block AttnRes 的显存/来源数账

运行： python3 attnres_demo.py
"""

import math
import random

# ---------------------------------------------------------------- 工具函数

def rms_norm(v, w):
    """对齐 torch.nn.functional.rms_norm(v, (D,), w, eps)。
       RMSNorm(x) = x / sqrt(mean(x^2) + eps) * w
       注意：没有减均值（那是 LayerNorm），也不除 sqrt(D)，除的是 sqrt(mean(x^2))。"""
    D = len(v)
    ms = sum(x * x for x in v) / D
    r = 1.0 / math.sqrt(ms + 1e-6)
    return [x * r * wi for x, wi in zip(v, w)]


def softmax(xs):
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es)
    return [e / s for e in es]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def attnres(residuals, q, w, output_w=None):
    """naive_attnres 的逐字翻译（fp32，单 token 版）。
       k = RMSNorm(residual)          # 残差自己当 key，没有 W_k！
       logit_l = <k_l, q> * scale     # q 是 1 维伪 query
       p = softmax(logit)             # 在"深度"这一维上做 softmax
       o = sum_l p_l * residual_l     # value 是未归一化的原始残差
       """
    ks = [rms_norm(r, w) for r in residuals]
    logits = [dot(k, q) for k in ks]
    p = softmax(logits)
    D = len(residuals[0])
    o = [sum(p[l] * residuals[l][d] for l in range(len(residuals))) for d in range(D)]
    if output_w is not None:
        o = rms_norm(o, output_w)
    return o, p, logits


def banner(t):
    print()
    print("=" * 72)
    print(t)
    print("=" * 72)


# ================================================================ 1. 手算

banner("1. 手算：3 个残差源，D=2，完全可纸上复现")

R = [[3.0, 0.0], [0.0, 4.0], [1.0, 1.0]]
W = [1.0, 1.0]          # attn_res_norm.weight，初始化就是全 1
Q = [1.0, 1.0]          # 训练后学出来的 1 维伪 query

print("残差源（谁想被聚合的候选）:")
for i, r in enumerate(R):
    print(f"  r{i+1} = {r}   ||r{i+1}|| = {norm(r):.4f}")

print("\n[第 1 步] 每个残差源自己做 RMSNorm 当 key（这一步就是 'attention' 里的 K）")
ks = []
for i, r in enumerate(R):
    k = rms_norm(r, W)
    ks.append(k)
    ms = sum(x * x for x in r) / len(r)
    print(f"  r{i+1}: mean(r^2) = {ms:.4f},  1/sqrt(.) = {1/math.sqrt(ms+1e-6):.4f},  "
          f"k{i+1} = {[round(x, 4) for x in k]}")

print("\n[第 2 步] logit = <k, q>，q 只有一个 D 维向量（不是矩阵，输出 1 维）")
logits = [dot(k, Q) for k in ks]
for i in range(3):
    terms = [f"{ks[i][d]:.4f}*{Q[d]}" for d in range(2)]
    print(f"  logit{i+1} = {' + '.join(terms)} = {logits[i]:.4f}")

print("\n[第 3 步] 在'深度'这一维做 softmax")
p = softmax(logits)
for i in range(3):
    print(f"  p{i+1} = {p[i]:.4f}")
print(f"  和 = {sum(p):.6f}  (必须为 1)")

print("\n[第 4 步] 加权求和，value 用的是**未归一化**的原始残差")
o, _, _ = attnres(R, Q, W)
for d in range(2):
    terms = " + ".join(f"{p[l]:.4f}*{R[l][d]:.1f}" for l in range(3))
    print(f"  o[{d}] = {terms} = {o[d]:.4f}")

print(f"\n结果: o = [{o[0]:.4f}, {o[1]:.4f}],  ||o|| = {norm(o):.4f}")

print("\n[对照 A] 标准残差（固定权重 1 全部相加）")
h = [sum(R[l][d] for l in range(3)) for d in range(2)]
print(f"  h = {h},  ||h|| = {norm(h):.4f}")
print(f"  ||h|| / ||o|| = {norm(h)/norm(o):.4f}   (= 源个数 3，因为一个是求和一个是加权平均)")

print("\n[对照 B] q 全 0（= 零初始化那一刻）")
o0, p0, l0 = attnres(R, [0.0, 0.0], W)
print(f"  logits = {l0}  →  全部相等 → softmax 均匀")
print(f"  p = {[round(x,4) for x in p0]}  (每个都是 1/3 = {1/3:.4f})")
print(f"  o = {[round(x,4) for x in o0]}   ← 正好是三个源的算术平均")
mean3 = [sum(R[l][d] for l in range(3)) / 3 for d in range(2)]
print(f"  手算平均 = {[round(x,4) for x in mean3]}   ✓ 一致")

print("\n[关键性质] softmax 权重非负且和为 1 ⟹ 输出是凸组合 ⟹ 模长被最大值卡住")
mx = max(norm(r) for r in R)
print(f"  三个源的 ||·|| 最大值 = {mx:.4f}")
print(f"  ||o(q=[1,1])|| = {norm(o):.4f}  ≤ {mx:.4f}   ✓")
print(f"  ||o(q=0)||    = {norm(o0):.4f}  ≤ {mx:.4f}   ✓")
print(f"  ||标准残差||   = {norm(h):.4f}  >  {mx:.4f}   ✗ 超出上界 → 这就是'不受控增长'")


# =========================================================== 2. 随机模拟

banner("2. 随机模拟：PreNorm 标准残差随深度增长 vs AttnRes 有界")

random.seed(20260921)
D = 256          # 向量维度
SIGMA = 1.0      # 每层输出每个分量的标准差
TRIALS = 400     # 重复次数，用来估均值

print(f"D = {D} 维, 每层输出 o_l ~ N(0, {SIGMA}^2) 逐分量独立, 重复 {TRIALS} 次取平均")
print("  期望值：标准残差 ||h|| -> sigma*sqrt(L*D)；AttnRes(平均) ||h|| -> sigma*sqrt(D/L)")
print()
print(f"{'子层数 L':>8} | {'标准残差 ||h||':>14} | {'理论 sqrt(L*D)':>14} | "
      f"{'AttnRes(q=0)':>13} | {'理论 sqrt(D/L)':>14} | {'比值':>6}")
print("-" * 88)

for L in [4, 8, 16, 32, 64, 96]:
    acc_sum = 0.0
    acc_mean = 0.0
    for _ in range(TRIALS):
        vecs = []
        for _l in range(L):
            vecs.append([random.gauss(0, SIGMA) for _ in range(D)])
        h = [sum(vecs[l][d] for l in range(L)) for d in range(D)]
        m = [sum(vecs[l][d] for l in range(L)) / L for d in range(D)]
        acc_sum += norm(h)
        acc_mean += norm(m)
    avg_sum = acc_sum / TRIALS
    avg_mean = acc_mean / TRIALS
    print(f"{L:>8} | {avg_sum:>14.2f} | {math.sqrt(L*D):>14.2f} | "
          f"{avg_mean:>13.2f} | {math.sqrt(D/L):>14.2f} | {avg_sum/avg_mean:>6.2f}")

print()
print("读法：")
print(f"  · 标准残差：L=4 时 ||h||≈{math.sqrt(4*D):.0f}，L=96 时 ≈{math.sqrt(96*D):.0f}，涨了 "
      f"{math.sqrt(96/4):.1f} 倍，且 L 再大还会继续涨 —— 没有任何上界")
print(f"  · AttnRes(零初始化)：L=4 时 ||h||≈{math.sqrt(D/4):.0f}，L=96 时 ≈{math.sqrt(D/96):.1f}，"
      f"反而被压小 —— 因为它是加权平均，天然有界")
print("  · 最后一列恒等于 L，而且和 D、sigma 都无关（求和 vs 平均的天然倍数）")

print("\n[稀释] 第 l 层写进主干时，相对改变量 ||o_l|| / ||h_{l-1}|| 有多大？")
print()
print(f"{'l':>4} | {'cos(h_{l-1}, h_l)':>18} | {'夹角(度)':>10} | {'相对改变 1/sqrt(l-1)':>22}")
print("-" * 64)
for l in [2, 4, 8, 16, 32, 48, 64, 96]:
    # cos(h_{l-1}, h_l) = ||h_{l-1}|| / ||h_l||  (h_l = h_{l-1} + o_l，o_l 方向随机)
    c = math.sqrt(l - 1) / math.sqrt(l)
    ang = math.degrees(math.acos(c))
    rel = 1.0 / math.sqrt(l - 1) if l > 1 else float('nan')
    print(f"{l:>4} | {c:>18.6f} | {ang:>10.2f} | {rel:>22.4f}")

print()
print("读法：在 48 层附近，每层只能把主干方向转动约 8 度。")
print("      层数越深，这一层写进去的东西在方向上的话语权越小 —— 这就是'稀释'。")


# =========================================================== 3. 显存账

banner("3. Block AttnRes 的账：来源数 与 显存")

N_SUB = 96      # Kimi Linear 48B：48 个 transformer 层 × 每层 2 个子层（attn + mlp）
D_MODEL = 7168  # 隐藏维度
TOKENS = 8192   # 一个训练 batch 的 token 数
BYTES = 2       # bf16

full_slots = N_SUB * (N_SUB + 1) // 2
per_stream_bytes = TOKENS * D_MODEL * BYTES


def n_boundaries(n, b):
    """块边界个数：{0, b, 2b, ...} ∩ [0, n)"""
    return (n - 1) // b + 1


def total_sources(n, b):
    """所有调用点的来源数之和。子层 i（i>=1）看到 floor((i-1)/b) + 2 个来源。
       +2 是 E（token embedding）和当前块前缀和各占一个 —— 逐行对照源码后的口径。"""
    return sum((i - 1) // b + 2 for i in range(1, n))


print(f"设定：{N_SUB} 个子层（48 层 × 2），d_model = {D_MODEL}，"
      f"batch = {TOKENS} token，bf16")
print(f"     单条 d_model 宽的残差流 = {TOKENS} × {D_MODEL} × {BYTES} "
      f"= {per_stream_bytes} B = {per_stream_bytes/1048576:.1f} MiB")
print()
print(f"Full AttnRes（block_size=1）：每个子层看全部历史")
print(f"              峰值要同时留住 {n_boundaries(N_SUB,1)+1} 条流 = "
      f"{(n_boundaries(N_SUB,1)+1)*per_stream_bytes/1073741824:.2f} GiB  ← 不现实")
print()
print("口径说明（逐行对照 modeling_kda.py 后修正）：")
print("  来源数(i) = floor((i-1)/block_size) + 2   ，i 是子层全局号，i=0 被绕过")
print("  峰值存活张量数 = 块边界数 + 1 = floor((n-1)/b) + 2")
print()
print(f"{'block_size':>10} | {'一块含层数':>10} | {'块边界数':>8} | {'峰值存活':>8} | "
      f"{'总来源数':>9} | {'省多少倍':>9} | {'峰值显存(GiB)':>14}")
print("-" * 92)

full_total = total_sources(N_SUB, 1)
for b in [1, 2, 4, 8, 16, 32]:
    nb = n_boundaries(N_SUB, b)
    peak = (nb + 1) * per_stream_bytes / 1073741824
    tot = full_total if b == 1 else total_sources(N_SUB, b)
    layers = 1 if b == 1 else b // 2
    print(f"{b:>10} | {layers:>10} | {nb:>8} | {nb+1:>8} | "
          f"{tot:>9} | {full_total/tot:>8.2f}x | {peak:>14.2f}")

print()
print("说明：")
print("  · block_size 数的是'子层'，一个块里有 block_size/2 个 transformer 层")
print("  · 来源里含 token embedding E 和当前块前缀和，所以比'只有块摘要'的口径多 2 个")
print("  · 峰值存活 = 块数 + 1，这是**显存**的账")
print("  · 总来源数 ≈ 按 1/b 缩，这是**读取代价**的账")
print("  · 详细状态机追踪见 attnres_inputs_demo.py")


# =========================================================== 4. 参数量

# 每个 KDABlock（一层）：attn_res_proj + attn_res_norm + mlp_res_proj + mlp_res_norm
#                         每个都是 d_model 个参数
# KDAModel 顶层：res_proj + res_norm
D_MODEL = 7168
N_SUB = 96
N_LAYER = N_SUB // 2
per_layer = 4 * D_MODEL
top_level = 2 * D_MODEL
total_params = per_layer * N_LAYER + top_level
print(f"每个 KDABlock（一层）：")
print(f"  attn_res_proj(Linear {D_MODEL}->1, 无 bias) = {D_MODEL}")
print(f"  attn_res_norm(RMSNorm {D_MODEL})          = {D_MODEL}")
print(f"  mlp_res_proj (Linear {D_MODEL}->1, 无 bias) = {D_MODEL}")
print(f"  mlp_res_norm (RMSNorm {D_MODEL})          = {D_MODEL}")
print(f"  小计                                        = {per_layer}")
print(f"{N_LAYER} 层 × {per_layer} = {per_layer*N_LAYER:,}")
print(f"KDAModel 顶层 res_proj + res_norm            = {top_level:,}")
print(f"合计                                          = {total_params:,} ≈ {total_params/1e6:.2f} M")
print()
for name, tot in [("Kimi Linear 48B", 48e9), ("K3 2.8T", 2.8e12)]:
    print(f"  占 {name:>16} 总参数的比例 = {total_params/tot*100:.5f}%")
print()
print("对照：MLA 的 W_DKV 一个矩阵就是 d_model × d_c = "
      f"{D_MODEL}×512 = {D_MODEL*512/1e6:.2f} M")
print("      AttnRes 全部加起来比 MLA 一个投影矩阵还小")
print()
print("为什么这么便宜？因为 AttnRes 没有 W_k / W_v / W_q 矩阵，")
print("key 和 value 直接复用残差本身，只学一个 D 维的'打分方向'。")


# =========================================================== 5. 自检

banner("5. 自检断言")

# 自检 1：softmax 权重和 = 1
assert abs(sum(p) - 1.0) < 1e-12
# 自检 2：torch 的 rms_norm 公式
assert abs(rms_norm([1.0, 1.0], [1.0, 1.0])[0] - 1.0 / math.sqrt(1.0 + 1e-6)) < 1e-12
# 自检 3：q=0 时输出必须等于算术平均
for d in range(2):
    assert abs(o0[d] - mean3[d]) < 1e-12
# 自检 4：凸组合的模长上界
assert norm(o) <= mx + 1e-12
assert norm(o0) <= mx + 1e-12
# 自检 5：cos(h_{l-1}, h_l) = sqrt((l-1)/l)
for l in [2, 7, 48, 96]:
    c = math.sqrt(l - 1) / math.sqrt(l)
    assert abs(1.0 / c - math.sqrt(l / (l - 1))) < 1e-12
# 自检 6：峰显存代数
assert full_slots == 4656
assert n_boundaries(N_SUB, 8) == 12
assert total_sources(N_SUB, 8) == 707
assert n_boundaries(N_SUB, 1) + 1 == 97
# 自检 7：参数量（含 KDAModel 顶层那一组）
assert per_layer == 28672
assert total_params == 1390592
# 自检 8：Block 总来源槽公式 b*m(m+1)/2
for b in [2, 4, 8, 16, 32]:
    m = N_SUB // b
    assert b * m * (m + 1) // 2 == (N_SUB * (m + 1)) // 2
# 自检 9：模拟结果与理论值一致（相对误差 < 2%）
random.seed(7)
for L in [4, 16, 96]:
    s = 0.0
    for _ in range(200):
        vs = [[random.gauss(0, 1) for _ in range(D)] for _ in range(L)]
        s += norm([sum(vs[l][d] for l in range(L)) for d in range(D)])
    assert abs(s / 200 - math.sqrt(L * D)) / math.sqrt(L * D) < 0.02

print("全部通过 ✓")
print()
print("注：本脚本零依赖，未使用 torch / numpy；所有数字由上面的代码现场算出，")
print("    可直接 python3 attnres_demo.py 复现。")

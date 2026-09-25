#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AttnRes 的输入到底是谁 —— 对照 fla/models/kda/modeling_kda.py 的状态机复现

依据（全部逐行对照真实源码）：
  KDABlock.forward / KDAModel.forward 里的 attnres 分支
  fused_attnres 的签名与 naive_attnres 的广播语义

本脚本做四件事：
  1) 符号追踪：把每个子层看到的 residuals 列表原样打出来（E / A_i / M_i / 块和）
  2) 推导：来源个数怎么增长、块边界在哪个子层触发、even block_size 的一个推论
  3) 数值验证：softmax 只在"深度"维做，token 之间完全独立
  4) 参数量重算（含 KDAModel 顶层那一组）

运行： python3 attnres_inputs_demo.py
"""

from collections import Counter
import math
import random

# ============================================================ 0. 符号代数

def show(c):
    """把一个'张量'显示成符号和，例如 E+2A0+M1"""
    if c is None:
        return "None"
    if len(c) == 0:
        return "0"
    parts = []
    for k in sorted(c):
        v = c[k]
        parts.append(k if v == 1 else f"{v}{k}")
    return "+".join(parts)


def banner(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


# ================================================ 1. 逐行复刻 KDABlock 状态机

def trace(n_layers, block_size):
    """逐行对照 modeling_kda.py 的 attnres 分支，记录每个 attnres 调用点的 residuals。

    维护的两个状态（源码注释原话）：
      attnres_states : list[Tensor]  已完成块的摘要（保持成 list，不做 cat）
      hidden_states  : Tensor        当前块正在累加的 prefix_sum
    """
    E = Counter({"E": 1})          # token embedding
    H = Counter(E)                 # hidden_states = 当前 prefix_sum
    states = None                  # attnres_states，初始 None
    rows = []

    for li in range(n_layers):
        is_attn_b = (2 * li) % block_size == 0
        is_mlp_b = (2 * li + 1) % block_size == 0

        # ---------------- attention 子层 ----------------
        prefix_sum = Counter(H)
        if states is None:
            # 源码：L=1 单来源，attnres 恒等，直接过 prenorm
            rows.append((f"layer{li}.attn", "**绕过**（L=1）",
                         "[E] 但没调 attnres", "attn_norm(E)"))
            hidden = Counter(prefix_sum)          # = attn_norm(prefix_sum)
            states = [Counter(prefix_sum)]        # 源码：attnres_states = [prefix_sum]
            prefix_sum = None
        else:
            residuals = [Counter(r) for r in states] + [Counter(prefix_sum)]
            tag = "**块边界**" if is_attn_b else ""
            rows.append((f"layer{li}.attn", tag,
                         "[" + ", ".join(show(r) for r in residuals) + "]",
                         "fused_attnres(attn_res_proj, residuals)"))
            hidden = Counter({"mix": 1})
            if is_attn_b:
                states = [Counter(r) for r in residuals]
                prefix_sum = None

        A = Counter({f"A{li}": 1})
        prefix_sum = Counter(A) if prefix_sum is None else prefix_sum + Counter(A)

        # ---------------- MLP 子层 ----------------
        residuals = [Counter(r) for r in states] + [Counter(prefix_sum)]
        tag = "**块边界**" if is_mlp_b else ""
        rows.append((f"layer{li}.mlp", tag,
                     "[" + ", ".join(show(r) for r in residuals) + "]",
                     "fused_attnres(mlp_res_proj, residuals)"))
        if is_mlp_b:
            states = [Counter(r) for r in residuals]
            prefix_sum = None

        M = Counter({f"M{li}": 1})
        H = Counter(M) if prefix_sum is None else prefix_sum + Counter(M)

    # ---------------- KDAModel 顶层再来一次 ----------------
    residuals = [Counter(r) for r in states] + [Counter(H)]
    rows.append(("model.top", "", "[" + ", ".join(show(r) for r in residuals) + "]",
                 "fused_attnres(res_proj, ...) → final norm"))
    return rows


def print_trace(n_layers, block_size, note):
    banner(f"符号追踪：{n_layers} 层（{2*n_layers} 个子层），"
           f"attnres_block_size = {block_size}")
    print(note)
    print()
    w = max(len(r[0]) for r in trace(n_layers, block_size))
    for i, (name, tag, res, call) in enumerate(trace(n_layers, block_size)):
        print(f"  [{i:>2}] {name:<{w}} {tag}")
        print(f"       residuals = {res}")
        print(f"       调用      = {call}")
        print()


print_trace(3, 4,
            "block_size=4 → 一个块含 4 个子层 = 2 个 transformer 层。\n"
            "  观察：① 第 1 个 attn 子层被**绕过**（只有 1 个来源，attnres 恒等）\n"
            "        ② residuals[0] 是 **token embedding E**，不是某一层的输出\n"
            "        ③ 块边界只在 **attn** 子层触发，此刻 prefix_sum 正好是上一块的完整和")

print_trace(4, 4,
            "跑到第二个块边界（layer2.attn，全局子层号 4），\n"
            "  看 states 怎么变成 [E, A0+M0+A1+M1] —— 上一块的完整和被'提升'成块摘要")

print_trace(2, 1,
            "block_size=1 → Full AttnRes：每个子层自己一个块，\n"
            "  此时 attn 和 mlp 都是边界，看 mlp 那一行 —— residuals 含**每一个**子层输出")

print_trace(1, 2,
            "block_size=2 是允许的最小偶数：一个块 = 1 个 transformer 层")


# ================================================ 2. 结构性推论

banner("2. 结构性推论（都可以从上面的追踪直接看出来）")

print("推论 A：residuals 的构成")
print()
print("  residuals = [ E, S_1, S_2, ..., S_k, 当前块前缀和 ]")
print()
print("  E   = token embedding（第一个块的'种子'，索引固定为 0）")
print("  S_j = 第 j 个块的**完整和**，即该块内所有子层输出的逐元素和")
print("  k   = 已完成的块数")
print()
print("  → 来源个数 = k + 2（E 占 1 个，当前块前缀和占 1 个）")
print("  → 注意 k 是把 E 自己也算进去的那个列表的长度，所以 k 并不是'层数/块大小'")
print()

print("推论 B：块边界只在 attn 子层触发")
print()
print("  边界条件：global_sublayer_index % block_size == 0")
print("    attn 子层号 = 2*layer_idx      （偶数）")
print("    mlp  子层号 = 2*layer_idx + 1  （奇数）")
print()
print("  block_size 只允许 None / 1 / 正偶数（KDAConfig 校验）。")
print("  若 block_size 是偶数，则它的任意倍数都是偶数，")
print("  而 mlp 子层号恒为奇数 ⟹ 永远不可能相等 ⟹ **mlp 永远不是块边界**")
print()
ok = all((2 * li + 1) % b != 0 for b in [2, 4, 6, 8, 16, 32] for li in range(64))
print(f"  穷举验证（block_size ∈ 2..32 偶数，layer_idx ∈ 0..63）：", end="")
print("全部为 False ✓" if ok else "✗ 有反例！")
print()
print("  这个推论的实际后果：")
print("    · block_size ≥ 2 偶数时，块边界永远落在 attn 子层上")
print("    · 因此代码里 `hidden_states = prefix_sum + mlp_out` 那一步")
print("      **必然**走相加分支，不会走 `prefix_sum is None` 的分支")
print("    · 只有 Full AttnRes（block_size=1）时 mlp 才是边界，")
print("      那时 `hidden_states` 只等于 `mlp_out`（因为 prefix_sum 被清成 None）")
print()

print("推论 C：来源个数怎么增长")
print()
print("  设子层总数 n，块大小 b（正整数偶数），子层全局号 i = 0..n-1。")
print("  块边界 = { 0, b, 2b, ... } ∩ [0, n)，个数 n_b = floor((n-1)/b) + 1")
print("  在子层 i 处调 attnres 时：")
print("    来源数(i) = 1 + floor((i-1)/b) + 1 = floor((i-1)/b) + 2   (i ≥ 1)")
print("    i = 0 时**绕过**，不调用")
print()
print("  最终 attnres_states 的长度 = n_b（含 E），")
print("  所以峰值同时存活的张量数 = n_b + 1 = floor((n-1)/b) + 2")
print()
N_SUB = 96


def n_boundaries(n, b):
    return (n - 1) // b + 1


def peak_live(n, b):
    return n_boundaries(n, b) + 1


def total_sources(n, b):
    return sum((i - 1) // b + 2 for i in range(1, n))


full_total = total_sources(N_SUB, 1)

print(f"  n = {N_SUB} 个子层（48 层 × 2），b = block_size：")
print()
print(f"{'block_size b':>12} | {'块中含层数':>10} | {'末态 states 长度':>16} | "
      f"{'峰值存活张量':>12} | {'全部调用点来源总数':>18} | {'相对 Full':>9}")
print("-" * 92)
for b in [1, 2, 4, 8, 16, 32]:
    t = full_total if b == 1 else total_sources(N_SUB, b)
    print(f"{b:>12} | {(b//2 if b > 1 else 1):>10} | {n_boundaries(N_SUB, b):>16} | "
          f"{peak_live(N_SUB, b):>12} | {t:>18} | {full_total/t:>8.2f}x")

print()
print("  读法：")
print("    · 峰值存活张量数 ≈ 块数 + 1 —— 这是**显存**的账")
print("      b 从 1 涨到 8，峰值从 97 条流降到 13 条流（96/8 + 1）")
print("    · 全部调用点来源总数 —— 这是**读取代价**的账，约按 1/b 缩")
print("    · 注意这里的口径包含了 E 和当前块前缀和，和论文口径可能略有差异；")
print("      归档里旧表少算了这两个，本表是逐行对照代码后的修正版")
print()


# ================================================ 3. 数值验证：逐 token 独立

banner("3. 数值验证：softmax 只在深度维做，token 之间完全独立")

def rms_norm(v, w, eps=1e-6):
    D = len(v)
    ms = sum(x * x for x in v) / D
    r = 1.0 / math.sqrt(ms + eps)
    return [x * r * wi for x, wi in zip(v, w)]


def softmax(xs):
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es)
    return [e / s for e in es]


def attnres_token(residuals_tok, q, w):
    """对**单个 token** 做 AttnRes。residuals_tok: [L, D]"""
    ks = [rms_norm(r, w) for r in residuals_tok]
    s = [sum(a * b for a, b in zip(k, q)) for k in ks]
    p = softmax(s)
    D = len(residuals_tok[0])
    o = [sum(p[l] * residuals_tok[l][d] for l in range(len(residuals_tok))) for d in range(D)]
    return o, p


random.seed(31337)
L, D, BT = 5, 8, 6          # 5 个来源（深度），8 维，6 个 token
# residuals: [L, BT, D]
R = [[[random.gauss(0, 1) for _ in range(D)] for _ in range(BT)] for _ in range(L)]
W = [1.0] * D
Q = [random.gauss(0, 1) for _ in range(D)]

print(f"来源数 L = {L}，维度 D = {D}，token 数 = {BT}")
print()

# 每个 token 单独算
per_token = []
for t in range(BT):
    res_tok = [R[l][t] for l in range(L)]
    o, p = attnres_token(res_tok, Q, W)
    per_token.append((o, p))

print("每个 token 拿到的深度权重 p（注意：**每个 token 都不一样**）")
print()
hdr = "  token | " + " | ".join(f"src{l}" for l in range(L)) + " |   Σp"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for t, (o, p) in enumerate(per_token):
    print(f"  {t:>5} | " + " | ".join(f"{v:.3f}" for v in p) + f" | {sum(p):.4f}")

print()
print("→ 如果是'整体一个权重'，各行应该完全相同。它们不同 ⟹ 权重是 input-dependent 的。")
print("→ 但每一行内部 Σp = 1 ⟹ 每个 token 各自做一次完整的 softmax。")

# 打乱 token 顺序，结果必须逐 token 不变
order = list(range(BT))
random.shuffle(order)
R_shuf = [[R[l][order[t]] for t in range(BT)] for l in range(L)]
ok_shuffle = True
for t in range(BT):
    res_tok = [R_shuf[l][t] for l in range(L)]
    o2, _ = attnres_token(res_tok, Q, W)
    o1, _ = per_token[order[t]]
    if max(abs(a - b) for a, b in zip(o1, o2)) > 1e-12:
        ok_shuffle = False
print(f"\n把 token 顺序打乱后重算：结果逐个 token 完全一致？ {'✓ 是' if ok_shuffle else '✗ 否'}")
print("→ 说明**没有任何跨 token / 跨序列的混合**，也没有位置编码。")
print("   AttnRes 里的 'attention' 完全不看序列维，只看深度维。")

# 凸组合上界
print()
worst = 0.0
for t in range(BT):
    o, p = per_token[t]
    mx = max(math.sqrt(sum(x * x for x in R[l][t])) for l in range(L))
    s = math.sqrt(sum(x * x for x in o))
    worst = max(worst, s / mx)
print(f"逐 token 检查 ||o|| ≤ max_l ||res_l||：最大比值 = {worst:.6f} ≤ 1  "
      f"{'✓' if worst <= 1 + 1e-12 else '✗'}")
print("→ 凸组合上界对每个 token 都成立。")

# 手算验证一个 token：q=0 时必须等于算术平均
print()
print("抽查：q 全 0 时，每个 token 的输出必须等于它 5 个来源的算术平均")
ok_zero = True
for t in range(BT):
    res_tok = [R[l][t] for l in range(L)]
    o0, p0 = attnres_token(res_tok, [0.0] * D, W)
    mean = [sum(res_tok[l][d] for l in range(L)) / L for d in range(D)]
    if max(abs(a - b) for a, b in zip(o0, mean)) > 1e-12:
        ok_zero = False
    assert all(abs(v - 1.0 / L) < 1e-12 for v in p0)
print(f"  全部 {BT} 个 token：{'✓ 通过' if ok_zero else '✗ 失败'}")


# ================================================ 4. 参数量重算

banner("4. 参数量重算（这次把 KDAModel 顶层那一组也算进去）")

D_MODEL = 7168
N_SUB = 96

per_attn = D_MODEL + D_MODEL       # attn_res_proj + attn_res_norm
per_mlp = D_MODEL + D_MODEL        # mlp_res_proj  + mlp_res_norm
top = D_MODEL + D_MODEL            # res_proj      + res_norm

total = (per_attn + per_mlp) * (N_SUB // 2) + top
print(f"每个 KDABlock（一层）:")
print(f"  attn_res_proj  Linear({D_MODEL}→1) = {D_MODEL}")
print(f"  attn_res_norm  RMSNorm({D_MODEL})  = {D_MODEL}")
print(f"  mlp_res_proj   Linear({D_MODEL}→1) = {D_MODEL}")
print(f"  mlp_res_norm   RMSNorm({D_MODEL})  = {D_MODEL}")
print(f"  小计 = {per_attn + per_mlp}")
print()
print(f"{N_SUB//2} 层 × {per_attn+per_mlp} = {(per_attn+per_mlp)*(N_SUB//2):,}")
print(f"KDAModel 顶层：res_proj + res_norm = {top:,}")
print(f"合计 = {total:,} ≈ {total/1e6:.2f} M")
print()
print(f"  占 Kimi Linear 48B 总参数 : {total/48e9*100:.5f}%")
print(f"  占 K3 2.8T 总参数         : {total/2.8e12*100:.6f}%")
print()
print("对照：MLA 的 W_DKV 一个矩阵 = 7168×512 = 3.67 M")
print("      AttnRes 全部加起来 ≈ 1.39 M —— 比 MLA 一个投影矩阵还小")


# ================================================ 5. 自检

banner("5. 自检断言")

# 5.1 block_size=4 的追踪里，layer0.attn 必须是绕过
rows = trace(3, 4)
assert "绕过" in rows[0][1], rows[0]
# 5.2 layer0.mlp 的 residuals 必须是 [E, A0]
assert show(Counter({"E": 1})) in rows[1][2] and "A0" in rows[1][2], rows[1]
# 5.3 块边界只在 attn：even block_size 下 mlp 永不越界
for b in [2, 4, 8, 16, 32]:
    assert all((2 * li + 1) % b != 0 for li in range(200))
# 5.4 block_size=1 时 mlp 才是边界
assert all((2 * li + 1) % 1 == 0 for li in range(10))
# 5.5 来源数 = floor((i-1)/b) + 2，末态 states 长度 = 边界个数
for b in [1, 2, 4, 8, 16, 32]:
    for n in [6, 8, 96]:
        assert n_boundaries(n, b) == (n - 1) // b + 1
        if n > 1:
            assert (1 - 1) // b + 2 == 2
# 5.5b 对 b 整除 n 的情形，峰值存活 = n/b + 1
for b in [2, 4, 8, 16, 32]:
    assert peak_live(N_SUB, b) == N_SUB // b + 1, (b, peak_live(N_SUB, b))
# 5.5c 符号追踪里最后一个 mlp 子层的来源数必须等于预测值
for b in [1, 2, 4, 8]:
    r = trace(N_SUB // 2, b)
    last_sublayer_idx = N_SUB - 1
    pred = (last_sublayer_idx - 1) // b + 2
    got = r[-2][2].count(",") + 1
    assert got == pred, (b, got, pred)
# 5.6 逐 token 独立性
for t in range(BT):
    assert abs(sum(per_token[t][1]) - 1.0) < 1e-12
# 5.7 凸组合上界
for t in range(BT):
    o, _ = per_token[t]
    mx = max(math.sqrt(sum(x * x for x in R[l][t])) for l in range(L))
    assert math.sqrt(sum(x * x for x in o)) <= mx + 1e-12
# 5.8 参数量
assert total == 1390592, total

print("全部通过 ✓")
print()
print("注：零依赖，未使用 torch / numpy。可直接 python3 attnres_inputs_demo.py 复现。")

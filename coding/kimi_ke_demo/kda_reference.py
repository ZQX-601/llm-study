#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KDA（Kimi Delta Attention）最小参考实现 —— 纯 Python，零第三方依赖。

对齐 Kimi Linear 论文 Eq.(1)：

    S_t = (I - beta_t * k_t @ k_t^T) @ Diag(alpha_t) @ S_{t-1} + beta_t * k_t @ v_t^T
    o_t = S_t^T @ q_t

其中 k_t ∈ R^K（单位向量）、v_t ∈ R^V、S_t ∈ R^{K x V}、alpha_t ∈ R^K（channel-wise 门）。

本文件包含两种等价算法：
  1. kda_recurrent  ：逐 token 递推，直接照抄公式，最容易看懂
  2. kda_chunked    ：分块并行（WY 表示 + 块间状态扫描），与 FLA kernel 同构

并用自检断言两者在任意 T / chunk_size 下数值一致。

用法：
    python3 kda_reference.py            # 自检 + 玩具配置逐步数值演练
    python3 kda_reference.py --selftest # 只跑自检
"""

from __future__ import annotations

import math
import random
import sys

# ----------------------------------------------------------------------------
# 基础线性代数（行优先 list[list[float]]）
# ----------------------------------------------------------------------------


def zeros(r: int, c: int):
    return [[0.0] * c for _ in range(r)]


def eye(n: int):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def matmul(a, b):
    """[m,n] @ [n,p] -> [m,p]"""
    m, n, p = len(a), len(b), len(b[0])
    out = zeros(m, p)
    for i in range(m):
        ai = a[i]
        oi = out[i]
        for k in range(n):
            aik = ai[k]
            if aik == 0.0:
                continue
            bk = b[k]
            for j in range(p):
                oi[j] += aik * bk[j]
    return out


def transpose(a):
    return [list(col) for col in zip(*a)]


def outer(u, v):
    """u[m] , v[p] -> [m,p]"""
    return [[ui * vj for vj in v] for ui in u]


def add(a, b):
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def sub(a, b):
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def scale_mat(a, s):
    return [[x * s for x in row] for row in a]


def diag_mul_left(d, a):
    """Diag(d) @ a"""
    return [[d[i] * a[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def solve_lower_unit(l_mat, b):
    """解 (I + L) X = B，L 严格下三角。前向代入，O(n^2 * p)，不需要求逆。"""
    n = len(l_mat)
    p = len(b[0])
    x = [[0.0] * p for _ in range(n)]
    for i in range(n):
        acc = list(b[i])
        li = l_mat[i]
        for k in range(i):
            lik = li[k]
            if lik == 0.0:
                continue
            xk = x[k]
            for j in range(p):
                acc[j] -= lik * xk[j]
        x[i] = acc
    return x


# ----------------------------------------------------------------------------
# 标量函数
# ----------------------------------------------------------------------------


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def softplus(x: float) -> float:
    # 数值稳定写法
    return math.log1p(math.exp(-abs(x))) + max(x, 0.0)


def silu(x: float) -> float:
    return x * sigmoid(x)


def l2norm(vec):
    n = math.sqrt(sum(x * x for x in vec))
    if n == 0.0:
        return list(vec)
    return [x / n for x in vec]


def vec_dot(a, b):
    return sum(x * y for x, y in zip(a, b))


# ----------------------------------------------------------------------------
# 1) 逐 token 递推（最直白，直接对应论文公式）
# ----------------------------------------------------------------------------


def kda_recurrent(q, k, v, beta, alpha, scale=1.0, S0=None, trace=False):
    """q,k: [T,K]  v: [T,V]  beta: [T]  alpha: [T,K]（每 token 的衰减率，已过 exp）

    返回 (o: [T,V], states: [T+1][K][V])
    """
    T = len(q)
    K, V = len(k[0]), len(v[0])
    S = zeros(K, V) if S0 is None else [row[:] for row in S0]
    states = [[row[:] for row in S]]
    outs = []
    for t in range(T):
        # ① 遗忘：Diag(alpha_t) @ S_{t-1}
        S_decayed = diag_mul_left(alpha[t], S)
        # ② 擦除：k_t k_t^T 与 (I - beta_t k_t k_t^T)
        kk = outer(k[t], k[t])
        erase_op = sub(eye(K), scale_mat(kk, beta[t]))
        # ③ 擦除后
        S_erased = matmul(erase_op, S_decayed)
        # ④ 写入：beta_t k_t v_t^T
        write = scale_mat(outer(k[t], v[t]), beta[t])
        # ⑤ 合并
        S = add(S_erased, write)
        states.append([row[:] for row in S])
        # ⑥ 读出：o_t = scale * S_t^T q_t，即 o[j] = scale * Σ_i S[i][j] * q[i]
        o_t = [scale * vec_dot([S[i][j] for i in range(K)], q[t]) for j in range(V)]
        outs.append(o_t)
        if trace:
            print(f"  t={t + 1}")
            print(f"    ① Diag(alpha)·S_prev = {fmt(S_decayed)}")
            print(f"    ② k k^T              = {fmt(kk)}")
            print(f"    ③ I - beta·k k^T     = {fmt(erase_op)}")
            print(f"    ④ 擦除后             = {fmt(S_erased)}")
            print(f"    ⑤ 写入 beta·k v^T    = {fmt(write)}")
            print(f"    ⑥ S_{t + 1}              = {fmt(S)}")
            print(f"    ⑦ o_{t + 1} = scale·S^T q = {fmt(outs[-1])}")
    return outs, states


# ----------------------------------------------------------------------------
# 2) 分块并行（与 FLA chunk_kda 同构）
# ----------------------------------------------------------------------------


def kda_chunked(q, k, v, beta, g, scale=1.0, C=64, S0=None):
    """g: [T,K] 为 log 空间的逐 token 门（未累加）。

    块内：
        gamma_r       = exp(Σ_{u<=r} g_u)            # 块内累计衰减
        ratio[r,s]    = gamma_r / gamma_s            # 论文里的 A[i/j]
        L[r,s]        = beta_s * (k_r · k_s) * ratio[r,s]      (s < r)
        A             = (I + L)^{-1}                  # WY / UT 变换
        w             = A @ (beta ⊙ k ⊙ gamma)
        u             = A @ (beta ⊙ v)
        v_new         = u - w @ h                     # 对传入状态的修正
        kg_s          = k_s ⊙ (gamma_last / gamma_s)
        Aqk[r,s]      = Σ_d q_{r,d} k_{s,d} ratio[r,s,d]   (s <= r)
        o             = Aqk @ v_new + (q ⊙ gamma) @ h
    块间：
        h <- Diag(gamma_last) @ h + kg^T @ v_new
    """
    T = len(q)
    K, V = len(k[0]), len(v[0])
    h = zeros(K, V) if S0 is None else [row[:] for row in S0]
    outs = []
    for start in range(0, T, C):
        end = min(start + C, T)
        n = end - start
        kc = k[start:end]
        qc = q[start:end]
        vc = v[start:end]
        bc = beta[start:end]
        gc = g[start:end]

        # 块内累计门（每个 chunk 从 0 重新开始）
        gcum = []
        acc = [0.0] * K
        for r in range(n):
            acc = [acc[d] + gc[r][d] for d in range(K)]
            gcum.append(list(acc))
        gamma = [[math.exp(x) for x in row] for row in gcum]           # [n,K]
        gamma_last = gamma[-1]                                          # [K]

        # ratio[r][s] = gamma_r / gamma_s  (逐 key 维)
        ratio = [[[gamma[r][d] / gamma[s][d] for d in range(K)] for s in range(n)] for r in range(n)]

        # L 严格下三角：L[r][s] = beta_r * Σ_d k_{r,d} k_{s,d} ratio[r][s][d]
        # 注意 beta 取行下标 r（对应 Diag(beta) K K^T 的写法），不是列下标 s
        L = zeros(n, n)
        for r in range(n):
            for s in range(r):
                L[r][s] = bc[r] * sum(kc[r][d] * kc[s][d] * ratio[r][s][d] for d in range(K))
        A = solve_lower_unit(L, eye(n))

        # w = A @ (beta ⊙ k ⊙ gamma) ; u = A @ (beta ⊙ v)
        kb = [[bc[r] * kc[r][d] * gamma[r][d] for d in range(K)] for r in range(n)]
        vb = [[bc[r] * vc[r][j] for j in range(V)] for r in range(n)]
        w = matmul(A, kb)                                              # [n,K]
        u = matmul(A, vb)                                              # [n,V]

        # v_new = u - w @ h   （h 是本 chunk 之前的状态）
        v_new = sub(u, matmul(w, h))                                   # [n,V]

        # 块内因果注意力 Aqk[r][s] = Σ_d q_r k_s ratio[r][s][d]  (s <= r)
        Aqk = zeros(n, n)
        for r in range(n):
            for s in range(r + 1):
                Aqk[r][s] = sum(qc[r][d] * kc[s][d] * ratio[r][s][d] for d in range(K))

        # o = scale * (Aqk @ v_new + (q ⊙ gamma) @ h)
        qg = [[qc[r][d] * gamma[r][d] for d in range(K)] for r in range(n)]
        o_intra = matmul(Aqk, v_new)
        o_inter = matmul(qg, h)
        o = scale_mat(add(o_intra, o_inter), scale)
        outs.extend(o)

        # 块间状态更新：h <- Diag(gamma_last) h + kg^T @ v_new
        kg = [[kc[s][d] * (gamma_last[d] / gamma[s][d]) for d in range(K)] for s in range(n)]
        h = add(diag_mul_left(gamma_last, h), matmul(transpose(kg), v_new))
    return outs, h


# ----------------------------------------------------------------------------
# 打印与自检
# ----------------------------------------------------------------------------


def fmt(m, nd=6):
    if m and isinstance(m[0], list):
        inner = ", ".join("[" + ", ".join(f"{x:.{nd}f}" for x in row) + "]" for row in m)
        return "[" + inner + "]"
    return "[" + ", ".join(f"{x:.{nd}f}" for x in m) + "]"


def max_abs_diff(a, b):
    if a and isinstance(a[0], list):
        return max(max_abs_diff(x, y) for x, y in zip(a, b))
    return max(abs(x - y) for x, y in zip(a, b))


def random_case(T, K, V, seed):
    rnd = random.Random(seed)
    q = [l2norm([rnd.gauss(0, 1) for _ in range(K)]) for _ in range(T)]
    k = [l2norm([rnd.gauss(0, 1) for _ in range(K)]) for _ in range(T)]
    v = [[rnd.gauss(0, 1) for _ in range(V)] for _ in range(T)]
    beta = [sigmoid(rnd.gauss(0, 1)) for _ in range(T)]
    g = [[-softplus(rnd.gauss(0, 1)) - 0.05 for _ in range(K)] for _ in range(T)]
    return q, k, v, beta, g


def selftest():
    print("== 自检：逐 token 递推 vs 分块，多组 T/K/V/chunk_size ==")
    ok = True
    seed = 0
    for T in (1, 2, 3, 5, 8, 13):
        for K in (1, 2, 4):
            for V in (1, 3):
                seed += 1
                q, k, v, beta, g = random_case(T, K, V, seed)
                alpha = [[math.exp(x) for x in row] for row in g]
                scale = K ** -0.5
                o_rec, _ = kda_recurrent(q, k, v, beta, alpha, scale=scale)
                for C in (1, 2, 3, 4, 64):
                    o_chk, _ = kda_chunked(q, k, v, beta, g, scale=scale, C=C)
                    d = max_abs_diff(o_rec, o_chk)
                    if d > 1e-9:
                        ok = False
                        print(f"  FAIL T={T} K={K} V={V} C={C} diff={d:.3e}")
    print("  结果：" + ("全部一致（最大误差 < 1e-9）" if ok else "存在不一致"))
    return ok


# ----------------------------------------------------------------------------
# 玩具配置：B=1, T=4, d=4, H=HV=1, K=V=2, chunk=2，全链路手算
# ----------------------------------------------------------------------------


def toy():
    print("\n== 玩具配置全链路数值演练 ==")
    print("d=4, H=HV=1, K=2, V=2, T=4, chunk_size=2")

    # --- 权重（取简单值，便于手算复核）---
    W_q = [[1, 0, 0, 0], [0, 1, 0, 0]]
    W_k = [[1, 0, 0, 0], [0, 1, 0, 0]]
    W_v = [[0, 0, 1, 0], [0, 0, 0, 1]]
    W_b = [[1, 0, 0, 0]]                      # -> beta logits, [1,4]
    W_f = [[1, 0, 0, 0], [0, 1, 0, 0]]        # -> gate raw,    [2,4]
    A_log = [0.0]                             # exp(A_log) = 1
    dt_bias = [0.0, 0.0]
    W_g1 = [[1, 0, 0, 0], [0, 1, 0, 0]]       # 输出门第一层 [2,4]
    W_g2 = [[1, 0], [0, 1]]                   # 输出门第二层 [2,2]
    norm_w = [1.0, 1.0]
    W_o = [[1, 0], [0, 1], [0, 0], [0, 0]]    # o_proj [4,2]

    x = [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
    ]
    T = len(x)

    def proj(w, vec):
        return [vec_dot(row, vec) for row in w]

    q, k, v, beta, g, gate = [], [], [], [], [], []
    print("\n-- 步骤 1~6：投影 / silu / L2 归一化 / beta / 门控 --")
    for t in range(T):
        q_raw = proj(W_q, x[t])
        k_raw = proj(W_k, x[t])
        v_raw = proj(W_v, x[t])
        q_t = l2norm([silu(z) for z in q_raw])
        k_t = l2norm([silu(z) for z in k_raw])
        v_t = [silu(z) for z in v_raw]
        b_logit = proj(W_b, x[t])[0]
        b_t = sigmoid(b_logit)
        raw_f = proj(W_f, x[t])
        g_t = [-math.exp(A_log[0]) * softplus(raw_f[d] + dt_bias[d]) for d in range(len(raw_f))]
        gate_t = [sigmoid(z) for z in proj(W_g2, proj(W_g1, x[t]))]
        q.append(q_t); k.append(k_t); v.append(v_t)
        beta.append(b_t); g.append(g_t); gate.append(gate_t)
        print(f"  t={t + 1}: q={fmt(q_t)} k={fmt(k_t)} v={fmt(v_t)} "
              f"beta={b_t:.6f} g={fmt(g_t)} sigmoid(gate)={fmt(gate_t)}")

    alpha = [[math.exp(z) for z in row] for row in g]
    print("\n  alpha = exp(g)：")
    for t in range(T):
        print(f"    t={t + 1}: {fmt(alpha[t])}")

    print("\n-- 步骤 7~9：KDA 递推（Eq.1），scale = 1/sqrt(K) = "
          f"{2 ** -0.5:.6f} --")
    o, _ = kda_recurrent(q, k, v, beta, alpha, scale=2 ** -0.5, trace=True)

    print("\n-- 分块实现交叉验证（chunk_size=2）--")
    o_chk, h_final = kda_chunked(q, k, v, beta, g, scale=2 ** -0.5, C=2)
    print(f"  分块输出 o = {fmt(o_chk)}")
    print(f"  与递推的最大误差 = {max_abs_diff(o, o_chk):.3e}")
    print(f"  最终状态 h = {fmt(h_final)}")

    print("\n-- 步骤 10~11：输出门 + RMSNormGated + o_proj --")
    y = []
    for t in range(T):
        m = sum(z * z for z in o[t]) / len(o[t])
        nz = [z / math.sqrt(m + 1e-5) for z in o[t]]
        gated = [nz[j] * norm_w[j] * gate[t][j] for j in range(len(nz))]
        y.append([vec_dot(row, gated) for row in W_o])
        print(f"  t={t + 1}: o={fmt(o[t])} -> rmsnorm={fmt(nz)} "
              f"-> gated={fmt(gated)} -> o_proj={fmt(y[t])}")
    return y


def main():
    only_self = "--selftest" in sys.argv
    ok = selftest()
    if not only_self:
        toy()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

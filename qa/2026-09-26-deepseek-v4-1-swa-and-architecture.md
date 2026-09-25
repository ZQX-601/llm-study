# DeepSeek V4.1 Flash：SWA、稀疏注意力与整体架构

- 开始日期：2026-09-25
- 归档日期：2026-09-26
- 模式：QA 专题学习
- 状态：用户明确表示“SWA 学完了”并要求存档；尚未经过独立问答或代码验收，因此记录为“学习完成，待复习验证”，不等同于已验收掌握。
- 重要性：SWA、KV Cache、prefill/decode 数据流属于岗位必备；CSA2、CED、Bounded Replay 属于需要掌握的前沿架构案例。

## 本次学习范围

1. 标准因果 SWA 的动机、窗口掩码、计算复杂度与多层有效感受野。
2. 使用 `4×4` 注意力矩阵手算 `W=2` 的 prefill，解释同层 token 并行和层间串行。
3. 从 `X:[B,T,d]` 出发，追踪低秩 Q、多 Q 头共享 KV、RoPE、局部索引、Sparse Attention、分组输出投影及 `[B,W,Dh]` 环形缓存的 shape。
4. `window_indices` 在 prefill 和 decode 中的索引语义，以及 `-1` 无效槽位的处理。
5. `sparse_attention` 的 gather、缩放点积、attention sink、softmax 与加权求和数据流。
6. 共享旋转 KV 同时承担 K/V 时，attention 输出为什么需要逆 RoPE。
7. DeepSeek-V4.1-Flash 的 20 层 causal encoder、20 层 decoder、SWA/CSA2、mHC、MoE、CED、Bounded Replay 与 DSpark 的整体位置关系。

## 核心架构结论

```text
文本/图像 embedding
        ↓
20 层 causal encoder
  ├─ 第 1～2 层：纯 SWA，窗口 128
  └─ 第 3～20 层：局部 SWA + 全局 CSA2（ratio=2）
        ↓
Encoder 最终表示 H20
  ├─ 为 decoder 投影全局 KV
  └─ 最近一个窗口进入 decoder SWA Bounded Replay
        ↓
20 层 decoder：局部 SWA + 全局 CSA2（ratio=1）
        ↓
合并 mHC 残差流 → RMSNorm → LM Head → 下一个 token
```

多数 CSA2 层中，数据不是“先得到 SWA 输出，再送给 CSA2”，而是先合并两类 KV 与索引，再执行一次稀疏注意力：

```text
局部 SWA KV + 压缩全局 KV       → 合并 KV 池
局部窗口索引 + 全局 Top-K 索引  → 合并索引
Q + 合并 KV 池 + 合并索引       → sparse_attn
```

`Full` 重新生成全局 KV 和索引；`Reindex` 复用全局 KV、重新选择索引；`Reuse` 同时复用全局 KV 和索引。所有 CSA2 层仍生成本层自己的 Q 和局部 SWA KV。

## 源码与详细笔记

- [SWA 主讲解](2026-09-25-deepseek-v4-1-swa.md)
- [窗口索引逐行拆解](2026-09-25-window-indices.md)
- [Sparse Attention 数据流](2026-09-25-sparse-attention.md)
- [RoPE tail 逐行拆解](2026-09-25-rope-tail-line-by-line.md)
- [Attention 输出逆 RoPE](2026-09-25-swa-inverse-rope.md)
- [DeepSeek V4.1 创新点与输入到输出](2026-09-25-deepseek-v4-1-overview.md)
- [教学代码与运行入口](../coding/deepseek_demo/README.md)
- [官方源码映射与 shape](../coding/deepseek_demo/SOURCE_MAP.md)
- [局部 SWA 教学实现](../coding/deepseek_demo/swa_attention.py)

代码实现覆盖低秩 Q、共享局部 KV、RoPE、局部 gather、attention sink、分组输出投影、两层 prefill 与单 token decode。它不包含 FP8/FP4 kernel、CSA2 全局分支、CED 调度、mHC、MoE 或 Bounded Replay，不能冒充完整模型复现。

## 一手资料

- [DeepSeek-V4.1-Flash 技术报告](https://arxiv.org/html/2609.19969v1)
- [官方 inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py)
- [官方 inference/kernel.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/kernel.py)

## 证据边界与复习安排

- 用户证据：能够持续沿 shape、窗口索引、Sparse Attention 和 RoPE 提出具体源码问题，并表示 SWA 学习完成。
- 静态代码证据：教学实现和源码映射已完成；Python 文件通过语法检查。
- 运行证据：归档前检测到 `torch 2.14.0+cu130` 且 `torch.cuda.is_available() == True`；运行 `python -m unittest discover -s coding/deepseek_demo -p "test_*.py" -v`，3 个测试全部通过。额外执行 `compileall` 时因 `__pycache__` 写入权限失败，但测试已成功导入并执行源码；随后使用只读 AST 检查确认 3 个 Python 文件语法正常。未运行完整 DeepSeek 模型。
- 独立验收：尚未进行，不标记“已验收掌握”。
- 建议复习：2026-09-27 检查局部窗口与 shape；2026-09-29 检查 prefill/decode 缓存；2026-10-03 用代码情境检查 Sparse Attention 与逆 RoPE；2026-10-17 做综合回顾。
- 下一步：学习 CSA2 的 Compressor、Indexer、Full/Reindex/Reuse，以及局部 SWA 与全局 Top-K 如何合并。

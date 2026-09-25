# DeepSeek V4.1 Flash：SWA 源码导读

本目录保存 2026-09-25 的 SWA 答疑代码。建议先读 `SOURCE_MAP.md`，再按其中的顺序阅读 `swa_attention.py`。代码复现的是**纯局部 SWA 路径**，尤其适合对照 encoder 前两层；它不是完整的 DeepSeek 模型，也不是高性能推理 kernel。

## 运行方法

使用已经安装 PyTorch 的 Python 环境：

```powershell
cd D:\study\llm-study\coding\deepseek_demo
python run_demo.py
python -m unittest test_swa_attention.py
```

2026-09-25 检查的默认 Python 环境没有安装 `torch`，因此目前还没有在该环境运行通过。不需要模型权重或云 GPU。

## 运行后会看到什么

- 两个参数独立的 attention 层；第二层读取第一层的输出。
- `B=2,T=4,d=16,H=4,Dh=8,W=2` 的 prefill，每个 query 对应的局部索引。
- 随后生成一个 token 时，每层如何使用自己的局部 KV 环形缓存。
- 单元测试会比较同一套权重下，单步 decode 与完整 prefill 最后一个位置的结果。

## 实现范围

已包含：低秩 Q、RMSNorm、共享局部 KV、RoPE 尾部旋转、局部索引 gather、attention sink、输出端逆 RoPE、分组低秩输出投影、每层独立的环形缓存。

未包含：FP8/FP4 量化、TileLang/FlashAttention kernel、CSA2 全局 KV、分层索引、CED 调度、Bounded Replay、mHC、MoE、张量并行。这里的 PyTorch gather 便于观察 shape，不能作为性能基准。

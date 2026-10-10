# RAG Top-K 最小指标

该目录只验证检索结果指标，不实现 BM25、Embedding 或向量库。

指标约定：

- `Hit@K`：Top-K 是否至少命中一篇相关文档；
- `Recall@K`：命中的相关文档数 / 全部相关文档数；
- `Precision@K`：命中的相关文档数 / K，少返回的槽位按未命中处理；
- `EvidenceComplete@K`：每个必需证据组是否至少被一篇文档覆盖。

运行：

```bash
python -m unittest discover -s coding/rag_metrics -p 'test_*.py' -v
```

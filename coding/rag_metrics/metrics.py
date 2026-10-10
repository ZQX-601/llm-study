"""不依赖检索器的最小 RAG Top-K 指标。

输入是检索后的文档 ID 排序，不实现 BM25、Embedding 或向量库。这样可以把
“检索算法是否工作”与“指标是否算对”拆开验证。
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True)
class QueryRetrievalResult:
    """一条查询的检索结果和人工标注。

    ``required_evidence_groups`` 的每组表示可替代证据。例如 ``({"a", "a2"},
    {"b"})`` 要求 Top-K 同时覆盖第一组中的至少一篇和第二组的 ``b``。
    未显式提供分组时，每个 relevant 文档都视为一项独立必需证据。
    """

    query_id: str
    retrieved_ids: tuple[str, ...]
    relevant_ids: frozenset[str]
    required_evidence_groups: tuple[frozenset[str], ...] = ()


@dataclass(frozen=True)
class RetrievalMetrics:
    hit_at_k: float
    recall_at_k: float
    precision_at_k: float
    evidence_complete_at_k: float


def _validate_case(case: QueryRetrievalResult, k: int) -> None:
    if k < 1:
        raise ValueError("k 必须大于 0")
    if not case.query_id:
        raise ValueError("query_id 不能为空")
    if not case.relevant_ids:
        raise ValueError("relevant_ids 不能为空，否则 Recall 没有定义")
    if len(case.retrieved_ids) != len(set(case.retrieved_ids)):
        raise ValueError("retrieved_ids 不能包含重复文档，否则会扭曲 Precision")
    if any(not group for group in case.required_evidence_groups):
        raise ValueError("Evidence 分组不能为空集合")


def evaluate_top_k(case: QueryRetrievalResult, k: int) -> RetrievalMetrics:
    """计算单条查询的 Hit/Recall/Precision/多证据完整性。

    Precision@K 的分母固定为请求的 ``k``；当系统少返回文档时，空缺位置按未命中
    处理。该约定可防止“只返回一个高置信结果”人为抬高 Precision@K。
    """

    _validate_case(case, k)
    top_k = set(case.retrieved_ids[:k])
    matched = top_k.intersection(case.relevant_ids)
    groups = case.required_evidence_groups or tuple(
        frozenset({doc_id}) for doc_id in sorted(case.relevant_ids)
    )
    evidence_complete = all(bool(top_k.intersection(group)) for group in groups)
    return RetrievalMetrics(
        hit_at_k=float(bool(matched)),
        recall_at_k=len(matched) / len(case.relevant_ids),
        precision_at_k=len(matched) / k,
        evidence_complete_at_k=float(evidence_complete),
    )


def macro_average(
    cases: Sequence[QueryRetrievalResult],
    k: int,
) -> RetrievalMetrics:
    """对查询等权平均，避免文档数多的查询支配整体结果。"""

    if not cases:
        raise ValueError("至少需要一条查询")
    rows = [evaluate_top_k(case, k) for case in cases]
    count = len(rows)
    return RetrievalMetrics(
        hit_at_k=sum(row.hit_at_k for row in rows) / count,
        recall_at_k=sum(row.recall_at_k for row in rows) / count,
        precision_at_k=sum(row.precision_at_k for row in rows) / count,
        evidence_complete_at_k=(
            sum(row.evidence_complete_at_k for row in rows) / count
        ),
    )

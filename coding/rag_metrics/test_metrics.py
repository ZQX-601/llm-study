"""RAG Top-K 指标测试。"""

import unittest

try:
    from .metrics import QueryRetrievalResult, evaluate_top_k, macro_average
except ImportError:
    from metrics import QueryRetrievalResult, evaluate_top_k, macro_average


class RetrievalMetricsTest(unittest.TestCase):
    def test_hit_does_not_mean_multi_evidence_complete(self) -> None:
        case = QueryRetrievalResult(
            query_id="q-order",
            retrieved_ids=("policy", "noise", "tracking"),
            relevant_ids=frozenset({"policy", "invoice"}),
            required_evidence_groups=(
                frozenset({"policy"}),
                frozenset({"invoice", "invoice-copy"}),
            ),
        )

        metrics = evaluate_top_k(case, k=3)

        self.assertEqual(metrics.hit_at_k, 1.0)
        self.assertEqual(metrics.recall_at_k, 0.5)
        self.assertAlmostEqual(metrics.precision_at_k, 1 / 3)
        self.assertEqual(metrics.evidence_complete_at_k, 0.0)

    def test_alternative_evidence_can_complete_a_group(self) -> None:
        case = QueryRetrievalResult(
            query_id="q-refund",
            retrieved_ids=("invoice-copy", "policy"),
            relevant_ids=frozenset({"invoice-copy", "policy"}),
            required_evidence_groups=(
                frozenset({"invoice", "invoice-copy"}),
                frozenset({"policy"}),
            ),
        )

        metrics = evaluate_top_k(case, k=2)

        self.assertEqual(metrics.recall_at_k, 1.0)
        self.assertEqual(metrics.precision_at_k, 1.0)
        self.assertEqual(metrics.evidence_complete_at_k, 1.0)

    def test_precision_penalizes_missing_slots(self) -> None:
        case = QueryRetrievalResult(
            query_id="q-short",
            retrieved_ids=("only-result",),
            relevant_ids=frozenset({"only-result"}),
        )
        self.assertEqual(evaluate_top_k(case, k=2).precision_at_k, 0.5)

    def test_macro_average_is_query_weighted(self) -> None:
        perfect = QueryRetrievalResult(
            "perfect", ("a",), frozenset({"a"})
        )
        miss = QueryRetrievalResult(
            "miss", ("x",), frozenset({"b"})
        )

        metrics = macro_average((perfect, miss), k=1)

        self.assertEqual(metrics.hit_at_k, 0.5)
        self.assertEqual(metrics.recall_at_k, 0.5)
        self.assertEqual(metrics.precision_at_k, 0.5)
        self.assertEqual(metrics.evidence_complete_at_k, 0.5)

    def test_rejects_invalid_labels_and_duplicates(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_top_k(
                QueryRetrievalResult("q", ("a",), frozenset()),
                k=1,
            )
        with self.assertRaises(ValueError):
            evaluate_top_k(
                QueryRetrievalResult("q", ("a", "a"), frozenset({"a"})),
                k=2,
            )
        with self.assertRaises(ValueError):
            evaluate_top_k(
                QueryRetrievalResult("q", ("a",), frozenset({"a"})),
                k=0,
            )


if __name__ == "__main__":
    unittest.main()

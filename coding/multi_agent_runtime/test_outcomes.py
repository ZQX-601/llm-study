"""分支依赖传播与任务结果聚合测试。"""

import unittest

try:
    from .outcomes import (
        BranchOutcome,
        BranchStatus,
        aggregate_task_outcome,
        resolve_dependent_status,
    )
except ImportError:
    from outcomes import (
        BranchOutcome,
        BranchStatus,
        aggregate_task_outcome,
        resolve_dependent_status,
    )


class OutcomeAggregationTest(unittest.TestCase):
    def test_optional_failure_is_degraded_success(self) -> None:
        result = aggregate_task_outcome(
            (
                BranchOutcome("diagnosis", True, BranchStatus.SUCCEEDED, True),
                BranchOutcome("trend", False, BranchStatus.FAILED, False),
            ),
            safety_violation=False,
        )

        self.assertTrue(result.task_success)
        self.assertTrue(result.degraded_success)
        self.assertEqual(result.optional_failures, ("trend",))

    def test_required_failure_fails_whole_task(self) -> None:
        result = aggregate_task_outcome(
            (
                BranchOutcome("status", True, BranchStatus.SUCCEEDED, True),
                BranchOutcome("inventory", True, BranchStatus.FAILED, False),
            ),
            safety_violation=False,
        )

        self.assertFalse(result.task_success)
        self.assertFalse(result.degraded_success)
        self.assertEqual(result.required_failures, ("inventory",))

    def test_actual_safety_violation_overrides_successful_branches(self) -> None:
        result = aggregate_task_outcome(
            (BranchOutcome("status", True, BranchStatus.SUCCEEDED, True),),
            safety_violation=True,
        )

        self.assertFalse(result.task_success)
        self.assertTrue(result.safety_violation)

    def test_dependency_failure_is_skipped_not_self_failure(self) -> None:
        self.assertIs(
            resolve_dependent_status(
                (BranchStatus.SUCCEEDED, BranchStatus.FAILED)
            ),
            BranchStatus.SKIPPED_UPSTREAM_FAILED,
        )
        self.assertIs(
            resolve_dependent_status(
                (BranchStatus.SUCCEEDED, BranchStatus.NEEDS_HUMAN)
            ),
            BranchStatus.BLOCKED,
        )

    def test_unfinished_branch_cannot_be_finalized(self) -> None:
        with self.assertRaisesRegex(ValueError, "BRANCHES_NOT_TERMINAL:metric"):
            aggregate_task_outcome(
                (BranchOutcome("metric", False, BranchStatus.RUNNING),),
                safety_violation=False,
            )


if __name__ == "__main__":
    unittest.main()

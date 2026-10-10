"""ReAct 与 Plan-and-Execute 成对基准测试。"""

import unittest

try:
    from .paired_benchmark import (
        ScenarioKind,
        default_cases,
        render_report,
        run_paired_benchmark,
    )
except ImportError:
    from paired_benchmark import (
        ScenarioKind,
        default_cases,
        render_report,
        run_paired_benchmark,
    )


class PairedBenchmarkTest(unittest.IsolatedAsyncioTestCase):
    async def test_four_cases_pass_hard_gates_under_same_budget(self) -> None:
        results = await run_paired_benchmark()

        self.assertEqual(len(results), 4)
        for result in results:
            self.assertTrue(result.react.task_success, result.case.name)
            self.assertTrue(result.plan_execute.task_success, result.case.name)
            self.assertLessEqual(
                result.react.actions, result.case.budget.max_actions
            )
            self.assertLessEqual(
                result.plan_execute.actions, result.case.budget.max_actions
            )
            self.assertLessEqual(
                result.react.tool_calls, result.case.budget.max_tool_calls
            )
            self.assertLessEqual(
                result.plan_execute.tool_calls,
                result.case.budget.max_tool_calls,
            )

    async def test_simple_query_exposes_plan_action_overhead(self) -> None:
        result = (await run_paired_benchmark(default_cases()[:1]))[0]
        self.assertLessEqual(result.react.actions, result.plan_execute.actions)
        self.assertEqual(result.react.tool_calls, result.plan_execute.tool_calls)

    async def test_timeout_retries_and_target_change_replans(self) -> None:
        results = await run_paired_benchmark()
        by_kind = {result.case.scenario: result for result in results}

        timeout = by_kind[ScenarioKind.RECOVERABLE_TIMEOUT]
        self.assertEqual(timeout.react.tool_calls, 2)
        self.assertEqual(timeout.plan_execute.tool_calls, 2)

        changed = by_kind[ScenarioKind.TARGET_CHANGE]
        self.assertEqual(changed.react.replans, 0)
        self.assertEqual(changed.plan_execute.replans, 1)
        self.assertEqual(changed.react.tool_calls, 2)
        self.assertEqual(changed.plan_execute.tool_calls, 2)

    async def test_deterministic_outcomes_exclude_wall_clock_latency(self) -> None:
        first = await run_paired_benchmark()
        second = await run_paired_benchmark()

        def stable_projection(results):
            return [
                (
                    item.case.name,
                    item.react.task_success,
                    item.react.actions,
                    item.react.tool_calls,
                    item.plan_execute.task_success,
                    item.plan_execute.actions,
                    item.plan_execute.tool_calls,
                    item.plan_execute.replans,
                )
                for item in results
            ]

        self.assertEqual(stable_projection(first), stable_projection(second))

    async def test_missing_telemetry_stays_missing(self) -> None:
        results = await run_paired_benchmark(default_cases()[:1])
        for metrics in (results[0].react, results[0].plan_execute):
            self.assertIsNone(metrics.input_tokens)
            self.assertIsNone(metrics.output_tokens)
            self.assertIsNone(metrics.monetary_cost)
            self.assertGreaterEqual(metrics.latency_ms, 0.0)
        report = render_report(results)
        self.assertIn("None", report)


if __name__ == "__main__":
    unittest.main()

"""Plan-and-Execute 评测适配器测试。"""

import unittest

try:
    from .plan_evaluation import PlanEvalCase, evaluate_plan_state
    from .plan_execute import (
        Evidence,
        EvidenceRequirement,
        ExecutionPlan,
        PlanRunState,
        PlanStatus,
        PlanStep,
        TraceEvent,
    )
except ImportError:
    from plan_evaluation import PlanEvalCase, evaluate_plan_state
    from plan_execute import (
        Evidence,
        EvidenceRequirement,
        ExecutionPlan,
        PlanRunState,
        PlanStatus,
        PlanStep,
        TraceEvent,
    )


def completed_state() -> PlanRunState:
    requirement = EvidenceRequirement(
        "job_status",
        frozenset({"mcp:get_training_job"}),
        subject_id="train-42",
    )
    plan = ExecutionPlan(
        "eval-plan",
        1,
        "查询任务状态",
        [
            PlanStep(
                "query",
                "查询任务",
                evidence_requirements=(requirement,),
            )
        ],
        current_step_index=1,
        status=PlanStatus.COMPLETED,
    )
    state = PlanRunState(goal=plan.goal, plan=plan, final_answer="train-42 已失败")
    state.evidence["job_status"] = Evidence(
        "job_status",
        {"job_id": "train-42", "status": "FAILED"},
        "mcp:get_training_job",
        version="7",
        subject_id="train-42",
    )
    state.trace.extend(
        [
            TraceEvent(1, "executor_decision", {"action": "ToolAction"}),
            TraceEvent(2, "mcp_tool_called", {"tool_name": "get_training_job"}),
            TraceEvent(3, "executor_decision", {"action": "CompleteStepAction"}),
            TraceEvent(4, "step_verified", {"decision": "COMPLETE_STEP"}),
        ]
    )
    return state


class PlanEvaluationTest(unittest.TestCase):
    def test_success_recomputes_evidence_and_trace_metrics(self) -> None:
        case = PlanEvalCase(
            "正常查询",
            PlanStatus.COMPLETED,
            None,
            lambda state: "失败" in (state.final_answer or ""),
            max_actions=2,
            max_tool_calls=1,
        )

        result = evaluate_plan_state(case, completed_state())

        self.assertTrue(result.task_success)
        self.assertTrue(result.case_passed)
        self.assertEqual(result.actions, 2)
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(result.completed_steps, 1)

    def test_forbidden_real_dispatch_is_safety_violation(self) -> None:
        state = completed_state()
        state.trace.append(
            TraceEvent(5, "mcp_tool_called", {"tool_name": "commit_cancel"})
        )
        case = PlanEvalCase(
            "禁止取消",
            PlanStatus.COMPLETED,
            None,
            lambda _: True,
            forbidden_dispatched_tools=frozenset({"commit_cancel"}),
            expect_safety_violation=True,
            max_tool_calls=2,
        )

        result = evaluate_plan_state(case, state)

        self.assertTrue(result.safety_violation)
        self.assertFalse(result.task_success)
        self.assertTrue(result.case_passed)

    def test_budget_mismatch_invalidates_trajectory(self) -> None:
        state = completed_state()
        state.trace.extend(
            TraceEvent(
                sequence=index + 5,
                phase="mcp_tool_called",
                detail={"tool_name": "get_training_job"},
            )
            for index in range(3)
        )
        case = PlanEvalCase(
            "Tool 超预算",
            PlanStatus.COMPLETED,
            None,
            lambda _: True,
            max_tool_calls=3,
            expect_trajectory_valid=False,
        )

        result = evaluate_plan_state(case, state)

        self.assertEqual(result.tool_calls, 4)
        self.assertFalse(result.trajectory_valid)
        self.assertFalse(result.task_success)
        self.assertTrue(result.case_passed)


if __name__ == "__main__":
    unittest.main()

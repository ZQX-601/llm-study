"""显式 Agent Loop 的正常链路、安全边界与失败恢复测试。"""

import unittest

from agent_loop import (
    FinalAction,
    MockToolBridge,
    ScriptedPolicy,
    StopReason,
    TaskState,
    TaskStatus,
    ToolAction,
    ToolResult,
    ToolSpec,
    require_fields,
    run_agent,
)


GET_JOB = ToolSpec(
    name="get_training_job",
    description="查询训练任务",
    validator=require_fields("job_id"),
)
COMMIT_CANCEL = ToolSpec(
    name="commit_cancel_training_job",
    description="提交已确认的取消操作",
    validator=require_fields("proposal_id", "confirmation_id", "idempotency_key"),
    has_side_effect=True,
)
GET_OPERATION = ToolSpec(
    name="get_cancel_operation",
    description="按原幂等键查询取消操作状态",
    validator=require_fields("idempotency_key"),
)


class AgentLoopTest(unittest.IsolatedAsyncioTestCase):
    async def test_normal_query_then_final_answer(self) -> None:
        bridge = MockToolBridge(
            [GET_JOB],
            {
                "get_training_job": lambda _: ToolResult(
                    ok=True,
                    data={"job_id": "train-2048", "state": "FAILED", "latest_error": "OOM"},
                )
            },
        )
        policy = ScriptedPolicy(
            [
                ToolAction("get_training_job", {"job_id": "train-2048"}),
                FinalAction("任务因 OOM 失败；这是权威状态返回的事实。"),
            ]
        )

        state = await run_agent(
            policy=policy,
            bridge=bridge,
            state=TaskState(
                "诊断任务",
                required_successful_tools=frozenset({"get_training_job"}),
            ),
        )

        self.assertEqual(state.status, TaskStatus.SUCCEEDED)
        self.assertEqual(state.stop_reason, StopReason.COMPLETED)
        self.assertEqual(len(state.observations), 1)
        self.assertEqual(bridge.call_history[0][0], "get_training_job")

    async def test_unknown_tool_is_rejected_before_dispatch_then_can_recover(self) -> None:
        bridge = MockToolBridge(
            [GET_JOB],
            {"get_training_job": lambda _: ToolResult(ok=True, data={"state": "RUNNING"})},
        )
        policy = ScriptedPolicy(
            [
                ToolAction("delete_everything", {}),
                ToolAction("get_training_job", {"job_id": "train-1"}),
                FinalAction("任务仍在运行。"),
            ]
        )

        state = await run_agent(policy=policy, bridge=bridge, state=TaskState("查询任务"))

        self.assertEqual(state.status, TaskStatus.SUCCEEDED)
        self.assertEqual(state.observations[0].error_code, "INVALID_ACTION")
        self.assertEqual(len(bridge.call_history), 1)

    async def test_missing_required_argument_is_rejected_by_runtime(self) -> None:
        bridge = MockToolBridge(
            [GET_JOB],
            {"get_training_job": lambda _: ToolResult(ok=True, data={"state": "RUNNING"})},
        )
        policy = ScriptedPolicy([
            ToolAction("get_training_job", {}),
            ToolAction("get_training_job", {"job_id": "train-1"}),
            FinalAction("参数错误已纠正，任务仍在运行。"),
        ])

        state = await run_agent(policy=policy, bridge=bridge, state=TaskState("查询任务"))

        self.assertEqual(state.status, TaskStatus.SUCCEEDED)
        self.assertIn("job_id", state.observations[0].message)
        self.assertEqual(
            bridge.call_history,
            [("get_training_job", {"job_id": "train-1"})],
        )

    async def test_premature_final_action_continues_until_required_evidence_exists(self) -> None:
        bridge = MockToolBridge(
            [GET_JOB],
            {"get_training_job": lambda _: ToolResult(ok=True, data={"state": "FAILED"})},
        )
        policy = ScriptedPolicy([
            FinalAction("没有证据，但我猜测任务因 OOM 失败。"),
            ToolAction("get_training_job", {"job_id": "train-2048"}),
            FinalAction("权威状态表明任务已经失败。"),
        ])

        state = await run_agent(
            policy=policy,
            bridge=bridge,
            state=TaskState(
                "诊断任务",
                required_successful_tools=frozenset({"get_training_job"}),
            ),
        )

        self.assertEqual(state.status, TaskStatus.SUCCEEDED)
        self.assertEqual(len(bridge.call_history), 1)
        final_decisions = [
            event.detail["decision"]
            for event in state.trace
            if event.phase == "verification" and "decision" in event.detail
        ]
        self.assertEqual(final_decisions[0], "CONTINUE")
        self.assertEqual(final_decisions[-1], "COMPLETE")

    async def test_retryable_failure_can_recover_once(self) -> None:
        attempts = 0

        def flaky_get_job(_: object) -> ToolResult:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return ToolResult(
                    ok=False,
                    error_code="UPSTREAM_TIMEOUT",
                    message="上游查询超时",
                    retryable=True,
                )
            return ToolResult(ok=True, data={"state": "RUNNING"})

        bridge = MockToolBridge([GET_JOB], {"get_training_job": flaky_get_job})
        repeated = ToolAction("get_training_job", {"job_id": "train-2"})
        policy = ScriptedPolicy([repeated, repeated, FinalAction("第二次查询成功。")])

        state = await run_agent(policy=policy, bridge=bridge, state=TaskState("查询任务"))

        self.assertEqual(state.status, TaskStatus.SUCCEEDED)
        self.assertEqual(attempts, 2)
        self.assertTrue(state.observations[0].retryable)
        self.assertTrue(state.observations[1].ok)

    async def test_result_unknown_forces_authoritative_query_with_original_key(self) -> None:
        original_key = "idem-1234567890123456"

        def commit(_: object) -> ToolResult:
            return ToolResult(
                ok=False,
                error_code="RESULT_UNKNOWN",
                message="响应丢失，执行结果未知",
                result_unknown=True,
            )

        bridge = MockToolBridge(
            [COMMIT_CANCEL, GET_OPERATION],
            {
                "commit_cancel_training_job": commit,
                "get_cancel_operation": lambda args: ToolResult(
                    ok=True,
                    data={"status": "SUCCEEDED", "idempotency_key": args["idempotency_key"]},
                ),
            },
        )
        policy = ScriptedPolicy(
            [
                ToolAction(
                    "commit_cancel_training_job",
                    {
                        "proposal_id": "proposal-1",
                        "confirmation_id": "confirmation-123456",
                        "idempotency_key": original_key,
                    },
                ),
                ToolAction("get_cancel_operation", {"idempotency_key": original_key}),
                FinalAction("权威操作记录表明取消已经成功。"),
            ]
        )

        state = await run_agent(policy=policy, bridge=bridge, state=TaskState("取消任务"))

        self.assertEqual(state.status, TaskStatus.SUCCEEDED)
        self.assertIsNone(state.pending_idempotency_key)
        self.assertEqual(
            [name for name, _ in bridge.call_history],
            ["commit_cancel_training_job", "get_cancel_operation"],
        )
        self.assertEqual(bridge.call_history[1][1]["idempotency_key"], original_key)

    async def test_result_unknown_blocks_new_commit_or_changed_key(self) -> None:
        original_key = "idem-1234567890123456"
        bridge = MockToolBridge(
            [COMMIT_CANCEL, GET_OPERATION],
            {
                "commit_cancel_training_job": lambda _: ToolResult(
                    ok=False,
                    error_code="RESULT_UNKNOWN",
                    result_unknown=True,
                ),
                "get_cancel_operation": lambda _: ToolResult(ok=True, data={"status": "UNKNOWN"}),
            },
        )
        policy = ScriptedPolicy(
            [
                ToolAction(
                    "commit_cancel_training_job",
                    {
                        "proposal_id": "proposal-1",
                        "confirmation_id": "confirmation-123456",
                        "idempotency_key": original_key,
                    },
                ),
                ToolAction("get_cancel_operation", {"idempotency_key": "different-key"}),
                FinalAction("无法确认结果。"),
            ]
        )

        state = await run_agent(policy=policy, bridge=bridge, state=TaskState("取消任务"))

        self.assertEqual(state.status, TaskStatus.NEEDS_HUMAN)
        self.assertIn("原 idempotency_key", state.observations[-1].message)
        self.assertEqual(len(bridge.call_history), 1)

    async def test_repeated_action_stops_before_third_dispatch(self) -> None:
        bridge = MockToolBridge(
            [GET_JOB],
            {
                "get_training_job": lambda _: ToolResult(
                    ok=False,
                    error_code="UPSTREAM_TIMEOUT",
                    retryable=True,
                )
            },
        )
        repeated = ToolAction("get_training_job", {"job_id": "train-3"})
        policy = ScriptedPolicy([repeated, repeated, repeated])

        state = await run_agent(policy=policy, bridge=bridge, state=TaskState("查询任务"))

        self.assertEqual(state.status, TaskStatus.FAILED)
        self.assertEqual(state.stop_reason, StopReason.REPEATED_ACTION)
        self.assertEqual(len(bridge.call_history), 2)

    async def test_budget_exhaustion_stops_before_dispatch(self) -> None:
        bridge = MockToolBridge(
            [GET_JOB],
            {"get_training_job": lambda _: ToolResult(ok=True, data={"state": "RUNNING"})},
        )
        policy = ScriptedPolicy([ToolAction("get_training_job", {"job_id": "train-4"})])

        state = await run_agent(
            policy=policy,
            bridge=bridge,
            state=TaskState("查询任务", budget_remaining=0),
        )

        self.assertEqual(state.stop_reason, StopReason.BUDGET_EXHAUSTED)
        self.assertEqual(bridge.call_history, [])

    async def test_trace_contains_all_runtime_phases(self) -> None:
        bridge = MockToolBridge(
            [GET_JOB],
            {"get_training_job": lambda _: ToolResult(ok=True, data={"state": "FAILED"})},
        )
        policy = ScriptedPolicy(
            [
                ToolAction("get_training_job", {"job_id": "train-5"}),
                FinalAction("任务失败。"),
            ]
        )

        state = await run_agent(policy=policy, bridge=bridge, state=TaskState("查询任务"))
        phases = {event.phase for event in state.trace}

        self.assertTrue(
            {
                "tool_discovery",
                "model_decision",
                "runtime_validation",
                "tool_dispatch",
                "observation",
                "verification",
                "stop",
            }.issubset(phases)
        )


if __name__ == "__main__":
    unittest.main()

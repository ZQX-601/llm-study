"""多 Agent 最小权限、Handoff、Context 投影与全局预算测试。"""

import unittest
from datetime import datetime, timedelta, timezone

try:
    from .multi_agent import (
        AgentResultStatus,
        AgentRole,
        AgentSpec,
        DiagnosticAgent,
        Evidence,
        FanInDecision,
        GlobalBudget,
        HandoffRequest,
        HandoffResult,
        MultiAgentRuntime,
        MultiAgentState,
        RuntimePolicyError,
        Sensitivity,
        SingleToolAgent,
        TaskStatus,
        build_demo_runtime,
        new_demo_state,
    )
except ImportError:
    from multi_agent import (
        AgentResultStatus,
        AgentRole,
        AgentSpec,
        DiagnosticAgent,
        Evidence,
        FanInDecision,
        GlobalBudget,
        HandoffRequest,
        HandoffResult,
        MultiAgentRuntime,
        MultiAgentState,
        RuntimePolicyError,
        Sensitivity,
        SingleToolAgent,
        TaskStatus,
        build_demo_runtime,
        new_demo_state,
    )


class RecordingAgent:
    def __init__(self, spec: AgentSpec) -> None:
        self.spec = spec
        self.seen_keys: tuple[str, ...] = ()
        self.called = False

    async def run(self, task, session) -> HandoffResult:
        self.called = True
        self.seen_keys = tuple(item.key for item in task.evidence)
        return session.complete("只处理投影证据", self.seen_keys)


class ForgingAgent:
    def __init__(self, spec: AgentSpec) -> None:
        self.spec = spec

    async def run(self, task, session) -> HandoffResult:
        return HandoffResult(
            task.handoff_id,
            self.spec.agent_id,
            AgentResultStatus.SUCCEEDED,
            "伪造完成",
            ("forged",),
            session.actions_used,
            session.tool_calls_used,
        )


class GuessingAgent:
    def __init__(self, spec: AgentSpec) -> None:
        self.spec = spec

    async def run(self, task, session) -> HandoffResult:
        return session.complete("尝试引用未投影证据", ("secret",))


class MultiAgentRuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_independent_read_agents_fan_out_and_fan_in(self) -> None:
        runtime, requests, state, backend = _parallel_fixture()

        results = await runtime.execute_parallel_handoffs(
            state,
            requests,
            return_action_reserve=1,
        )
        merged = runtime.fan_in(state, requests, results)

        self.assertEqual(merged.decision, FanInDecision.MERGED)
        self.assertEqual(
            set(merged.evidence_keys),
            {"parallel_job_status", "parallel_error_log"},
        )
        self.assertEqual(len(backend.call_history), 2)
        phases = [event.phase for event in state.trace]
        self.assertIn("fan_out_started", phases)
        self.assertIn("fan_out_completed", phases)
        self.assertIn("fan_in", phases)

    async def test_parallel_budget_is_preflighted_before_any_agent_runs(self) -> None:
        runtime, requests, state, backend = _parallel_fixture()
        state.budget.actions_remaining = 4

        with self.assertRaisesRegex(
            RuntimePolicyError, "PARALLEL_ACTION_BUDGET_OVERCOMMITTED"
        ):
            await runtime.execute_parallel_handoffs(
                state,
                requests,
                return_action_reserve=1,
            )

        self.assertEqual(backend.call_history, [])
        self.assertNotIn("fan_out_started", [event.phase for event in state.trace])

    async def test_parallel_duplicate_output_key_is_rejected(self) -> None:
        runtime, requests, state, backend = _parallel_fixture()
        conflicting = (
            requests[0],
            HandoffRequest(
                handoff_id=requests[1].handoff_id,
                source_agent_id=requests[1].source_agent_id,
                target_agent_id=requests[1].target_agent_id,
                objective=requests[1].objective,
                subject_id=requests[1].subject_id,
                evidence_keys=requests[1].evidence_keys,
                requested_tools=requests[1].requested_tools,
                delegated_action_budget=requests[1].delegated_action_budget,
                delegated_tool_budget=requests[1].delegated_tool_budget,
                expected_output_evidence_keys=("parallel_job_status",),
            ),
        )

        with self.assertRaisesRegex(
            RuntimePolicyError, "PARALLEL_EVIDENCE_KEY_CONFLICT"
        ):
            await runtime.execute_parallel_handoffs(state, conflicting)
        self.assertEqual(backend.call_history, [])

    async def test_fan_in_keeps_successful_branch_when_peer_fails(self) -> None:
        runtime, requests, state, backend = _parallel_fixture()
        try:
            from .multi_agent import ToolResult
        except ImportError:
            from multi_agent import ToolResult
        backend.handlers["search_job_logs"] = lambda _: ToolResult(
            False,
            error_code="TOOL_TIMEOUT",
            retryable=True,
        )

        results = await runtime.execute_parallel_handoffs(state, requests)
        merged = runtime.fan_in(state, requests, results)

        self.assertEqual(merged.decision, FanInDecision.CONFLICT)
        self.assertIn("parallel_job_status", state.evidence)
        self.assertNotIn("parallel_error_log", state.evidence)
        self.assertEqual(
            merged.evidence_keys,
            ("parallel_job_status",),
        )
        self.assertEqual(merged.conflict_keys, ("handoff:log-branch",))

    async def test_complete_handoff_has_separate_agents_and_trace(self) -> None:
        runtime, coordinator, diagnostic, backend = build_demo_runtime()
        state = await runtime.run(new_demo_state(), coordinator)

        self.assertEqual(state.status, TaskStatus.SUCCEEDED)
        self.assertIn("CUDA out of memory", state.final_answer or "")
        self.assertEqual(len(backend.call_history), 2)
        self.assertIsNotNone(diagnostic.last_input)
        self.assertEqual(diagnostic.last_input.evidence, ())
        phases = [event.phase for event in state.trace]
        self.assertIn("handoff_proposed", phases)
        self.assertIn("handoff_started", phases)
        self.assertIn("handoff_completed", phases)
        self.assertIn("final_answer", phases)
        self.assertEqual(state.budget.actions_remaining, 0)
        self.assertEqual(state.budget.tool_calls_remaining, 0)

    async def test_unauthorized_route_is_rejected_before_target_runs(self) -> None:
        runtime, _, diagnostic, backend = build_demo_runtime()
        state = new_demo_state()
        request = HandoffRequest(
            "bad-route",
            "diagnostic-agent",
            "coordinator-agent",
            "反向委派",
            "train-42",
            (),
            frozenset(),
            1,
            0,
        )

        with self.assertRaisesRegex(RuntimePolicyError, "HANDOFF_ROUTE_NOT_ALLOWED"):
            await runtime.execute_handoff(state, request)

        self.assertFalse(diagnostic.last_input)
        self.assertEqual(backend.call_history, [])
        self.assertEqual(state.trace[-1].phase, "handoff_rejected")

    async def test_handoff_cannot_expand_target_tool_permission(self) -> None:
        runtime, _, diagnostic, backend = build_demo_runtime()
        state = new_demo_state()
        request = HandoffRequest(
            "excess-tool",
            "coordinator-agent",
            "diagnostic-agent",
            "诊断后直接重启任务",
            "train-42",
            (),
            frozenset({"restart_training_job"}),
            1,
            1,
        )

        with self.assertRaisesRegex(
            RuntimePolicyError, "HANDOFF_REQUESTS_EXCESS_TOOL_PERMISSION"
        ):
            await runtime.execute_handoff(state, request)

        self.assertIsNone(diagnostic.last_input)
        self.assertEqual(backend.call_history, [])

    async def test_context_projection_excludes_unreferenced_secret(self) -> None:
        coordinator = AgentSpec(
            "coordinator",
            AgentRole.COORDINATOR,
            allowed_handoff_targets=frozenset({"reader"}),
        )
        reader_spec = AgentSpec(
            "reader",
            AgentRole.DIAGNOSTIC,
            accepted_sensitivities=frozenset({Sensitivity.INTERNAL}),
        )
        reader = RecordingAgent(reader_spec)
        runtime = MultiAgentRuntime((coordinator, reader_spec), {"reader": reader}, _empty_backend())
        state = MultiAgentState(
            "读取允许证据",
            "job-1",
            GlobalBudget(2, 0, 1),
            evidence={
                "allowed": Evidence(
                    "allowed", "状态", "user", "job-1", "coordinator"
                ),
                "secret": Evidence(
                    "secret",
                    "密钥",
                    "vault",
                    None,
                    "coordinator",
                    Sensitivity.RESTRICTED,
                ),
            },
        )
        request = HandoffRequest(
            "projection",
            "coordinator",
            "reader",
            "读取允许证据",
            "job-1",
            ("allowed",),
            frozenset(),
            1,
            0,
        )

        await runtime.execute_handoff(state, request)

        self.assertEqual(reader.seen_keys, ("allowed",))
        self.assertNotIn("secret", reader.seen_keys)

    async def test_restricted_evidence_requires_target_clearance(self) -> None:
        coordinator = AgentSpec(
            "coordinator",
            AgentRole.COORDINATOR,
            allowed_handoff_targets=frozenset({"reader"}),
        )
        reader_spec = AgentSpec(
            "reader",
            AgentRole.DIAGNOSTIC,
            accepted_sensitivities=frozenset({Sensitivity.INTERNAL}),
        )
        reader = RecordingAgent(reader_spec)
        runtime = MultiAgentRuntime((coordinator, reader_spec), {"reader": reader}, _empty_backend())
        state = MultiAgentState(
            "读取受限证据",
            "job-1",
            GlobalBudget(2, 0, 1),
            evidence={
                "secret": Evidence(
                    "secret",
                    "内部凭证",
                    "vault",
                    "job-1",
                    "coordinator",
                    Sensitivity.RESTRICTED,
                )
            },
        )
        request = HandoffRequest(
            "restricted",
            "coordinator",
            "reader",
            "读取受限证据",
            "job-1",
            ("secret",),
            frozenset(),
            1,
            0,
        )

        with self.assertRaisesRegex(
            RuntimePolicyError, "HANDOFF_EVIDENCE_SENSITIVITY_DENIED"
        ):
            await runtime.execute_handoff(state, request)
        self.assertFalse(reader.called)

    async def test_global_budget_cannot_be_multiplied_by_handoff(self) -> None:
        runtime, _, _, backend = build_demo_runtime()
        state = new_demo_state()
        state.budget.actions_remaining = 2
        request = HandoffRequest(
            "too-large",
            "coordinator-agent",
            "diagnostic-agent",
            "请求超出全局预算",
            "train-42",
            (),
            frozenset({"get_training_job", "search_job_logs"}),
            3,
            2,
        )

        with self.assertRaisesRegex(
            RuntimePolicyError, "DELEGATED_ACTION_BUDGET_EXCEEDS_GLOBAL"
        ):
            await runtime.execute_handoff(state, request)
        self.assertEqual(backend.call_history, [])

    async def test_runtime_reserves_coordinator_finalize_action(self) -> None:
        runtime, coordinator, diagnostic, backend = build_demo_runtime()
        state = new_demo_state()
        state.budget.actions_remaining = 4

        state = await runtime.run(state, coordinator)

        self.assertEqual(state.status, TaskStatus.FAILED)
        self.assertEqual(
            state.stop_reason,
            "DELEGATED_ACTION_BUDGET_BREAKS_RETURN_RESERVE",
        )
        self.assertIsNone(diagnostic.last_input)
        self.assertEqual(backend.call_history, [])
        self.assertNotIn(
            "handoff_started",
            [event.phase for event in state.trace],
        )

    async def test_forged_result_evidence_is_rejected(self) -> None:
        coordinator = AgentSpec(
            "coordinator",
            AgentRole.COORDINATOR,
            allowed_handoff_targets=frozenset({"forger"}),
        )
        forger_spec = AgentSpec("forger", AgentRole.DIAGNOSTIC)
        runtime = MultiAgentRuntime(
            (coordinator, forger_spec),
            {"forger": ForgingAgent(forger_spec)},
            _empty_backend(),
        )
        state = MultiAgentState(
            "验证伪造结果",
            "job-1",
            GlobalBudget(2, 0, 1),
        )
        request = HandoffRequest(
            "forged-result",
            "coordinator",
            "forger",
            "伪造 Evidence",
            "job-1",
            (),
            frozenset(),
            1,
            0,
        )

        with self.assertRaisesRegex(
            RuntimePolicyError, "HANDOFF_RESULT_REFERENCES_UNKNOWN_EVIDENCE"
        ):
            await runtime.execute_handoff(state, request)

    async def test_agent_cannot_reference_unprojected_global_evidence(self) -> None:
        coordinator = AgentSpec(
            "coordinator",
            AgentRole.COORDINATOR,
            allowed_handoff_targets=frozenset({"guesser"}),
        )
        guesser_spec = AgentSpec("guesser", AgentRole.DIAGNOSTIC)
        runtime = MultiAgentRuntime(
            (coordinator, guesser_spec),
            {"guesser": GuessingAgent(guesser_spec)},
            _empty_backend(),
        )
        state = MultiAgentState(
            "验证 Context 隔离",
            "job-1",
            GlobalBudget(1, 0, 1),
            evidence={
                "secret": Evidence(
                    "secret",
                    "虽然存在但未投影",
                    "vault",
                    "job-1",
                    "coordinator",
                    Sensitivity.INTERNAL,
                )
            },
        )
        request = HandoffRequest(
            "guess-secret",
            "coordinator",
            "guesser",
            "只处理公开输入",
            "job-1",
            (),
            frozenset(),
            1,
            0,
        )

        with self.assertRaisesRegex(
            RuntimePolicyError,
            "HANDOFF_RESULT_REFERENCES_UNSCOPED_EVIDENCE",
        ):
            await runtime.execute_handoff(state, request)

    async def test_new_handoff_ids_cannot_bypass_semantic_loop_guard(self) -> None:
        coordinator = AgentSpec(
            "coordinator",
            AgentRole.COORDINATOR,
            allowed_handoff_targets=frozenset({"reader"}),
        )
        reader_spec = AgentSpec(
            "reader",
            AgentRole.DIAGNOSTIC,
            accepted_sensitivities=frozenset({Sensitivity.INTERNAL}),
        )
        reader = RecordingAgent(reader_spec)
        runtime = MultiAgentRuntime(
            (coordinator, reader_spec),
            {"reader": reader},
            _empty_backend(),
            repeated_handoff_limit=2,
        )
        state = MultiAgentState(
            "检测委派循环",
            "job-1",
            GlobalBudget(3, 0, 3),
            evidence={
                "status": Evidence(
                    "status",
                    "FAILED",
                    "tool:get_job",
                    "job-1",
                    "coordinator",
                    Sensitivity.INTERNAL,
                    version="8",
                )
            },
        )

        def request(handoff_id: str) -> HandoffRequest:
            return HandoffRequest(
                handoff_id,
                "coordinator",
                "reader",
                "分析失败原因",
                "job-1",
                ("status",),
                frozenset(),
                1,
                0,
            )

        await runtime.execute_handoff(state, request("attempt-1"))
        await runtime.execute_handoff(state, request("attempt-2"))
        with self.assertRaisesRegex(RuntimePolicyError, "REPEATED_HANDOFF_ROUTE"):
            await runtime.execute_handoff(state, request("attempt-3"))

        self.assertEqual(len(state.handoff_results), 2)
        self.assertEqual(state.trace[-1].phase, "handoff_rejected")

    async def test_max_handoff_depth_stops_new_unique_route(self) -> None:
        runtime, _, diagnostic, backend = build_demo_runtime()
        state = new_demo_state()
        request = HandoffRequest(
            handoff_id="too-deep",
            source_agent_id="coordinator-agent",
            target_agent_id="diagnostic-agent",
            objective="一个从未出现过的新目标",
            subject_id="train-42",
            evidence_keys=(),
            requested_tools=frozenset({"get_training_job"}),
            delegated_action_budget=1,
            delegated_tool_budget=1,
            depth=5,
        )

        with self.assertRaisesRegex(RuntimePolicyError, "MAX_HANDOFF_DEPTH"):
            await runtime.execute_handoff(state, request)

        self.assertIsNone(diagnostic.last_input)
        self.assertEqual(backend.call_history, [])
        self.assertEqual(state.trace[-1].phase, "handoff_rejected")

    async def test_expired_global_deadline_blocks_handoff_with_budget_left(self) -> None:
        runtime, _, diagnostic, backend = build_demo_runtime()
        now = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)
        runtime.clock = lambda: now
        state = new_demo_state()
        state.deadline_at = now - timedelta(milliseconds=1)
        request = HandoffRequest(
            "after-deadline",
            "coordinator-agent",
            "diagnostic-agent",
            "查询状态",
            "train-42",
            (),
            frozenset({"get_training_job"}),
            1,
            1,
        )

        with self.assertRaisesRegex(RuntimePolicyError, "GLOBAL_DEADLINE_EXCEEDED"):
            await runtime.execute_handoff(state, request)

        self.assertEqual(state.budget.actions_remaining, 5)
        self.assertEqual(state.budget.tool_calls_remaining, 2)
        self.assertIsNone(diagnostic.last_input)
        self.assertEqual(backend.call_history, [])


def _empty_backend():
    try:
        from .multi_agent import MockToolBackend
    except ImportError:
        from multi_agent import MockToolBackend
    return MockToolBackend((), {})


def _parallel_fixture():
    try:
        from .multi_agent import MockToolBackend, ToolResult, ToolSpec, require_fields
    except ImportError:
        from multi_agent import MockToolBackend, ToolResult, ToolSpec, require_fields

    coordinator = AgentSpec(
        "coordinator",
        AgentRole.COORDINATOR,
        allowed_handoff_targets=frozenset({"status-agent", "log-agent"}),
    )
    status_spec = AgentSpec(
        "status-agent",
        AgentRole.DIAGNOSTIC,
        allowed_tools=frozenset({"get_training_job"}),
    )
    log_spec = AgentSpec(
        "log-agent",
        AgentRole.DIAGNOSTIC,
        allowed_tools=frozenset({"search_job_logs"}),
    )
    backend = MockToolBackend(
        (
            ToolSpec(
                "get_training_job",
                "查询任务状态",
                require_fields("job_id"),
                "job_id",
            ),
            ToolSpec(
                "search_job_logs",
                "查询日志",
                require_fields("job_id", "query"),
                "job_id",
            ),
        ),
        {
            "get_training_job": lambda args: ToolResult(
                True, {"job_id": args["job_id"], "status": "FAILED"}
            ),
            "search_job_logs": lambda args: ToolResult(
                True, {"job_id": args["job_id"], "error": "OOM"}
            ),
        },
    )
    status_agent = SingleToolAgent(
        status_spec,
        "get_training_job",
        "parallel_job_status",
        {},
        "状态查询完成",
    )
    log_agent = SingleToolAgent(
        log_spec,
        "search_job_logs",
        "parallel_error_log",
        {"query": "error"},
        "日志查询完成",
    )
    runtime = MultiAgentRuntime(
        (coordinator, status_spec, log_spec),
        {"status-agent": status_agent, "log-agent": log_agent},
        backend,
    )
    requests = (
        HandoffRequest(
            "status-branch",
            "coordinator",
            "status-agent",
            "查询状态",
            "train-42",
            (),
            frozenset({"get_training_job"}),
            2,
            1,
            ("parallel_job_status",),
        ),
        HandoffRequest(
            "log-branch",
            "coordinator",
            "log-agent",
            "查询错误日志",
            "train-42",
            (),
            frozenset({"search_job_logs"}),
            2,
            1,
            ("parallel_error_log",),
        ),
    )
    state = MultiAgentState(
        "并发诊断 train-42",
        "train-42",
        GlobalBudget(5, 2, 2),
    )
    return runtime, requests, state, backend


if __name__ == "__main__":
    unittest.main()

"""Plan-and-Execute、Skill 渐进式加载和 MCP 边界测试。"""

import unittest
from pathlib import Path

try:  # 从 ``coding`` 根目录发现测试时使用包导入。
    from .demo import build_plan
    from .plan_execute import (
        ActivateSkillAction,
        CompleteStepAction,
        DiagnosisFinalWriter,
        DispatchStatus,
        Evidence,
        EvidenceRequirement,
        EvidencePlanVerifier,
        ExecutionPlan,
        LoadSkillResourceAction,
        MCPResult,
        MCPToolSpec,
        MockMCPClient,
        PlanRunState,
        PlanStatus,
        PlanStep,
        ScriptedPlanner,
        ScriptedStepExecutor,
        SkillRegistry,
        StepStatus,
        ToolAction,
        require_fields,
        run_plan_execute,
    )
except ImportError:  # 从本目录单独运行测试时使用脚本导入。
    from demo import build_plan
    from plan_execute import (
        ActivateSkillAction,
        CompleteStepAction,
        DiagnosisFinalWriter,
        DispatchStatus,
        Evidence,
        EvidenceRequirement,
        EvidencePlanVerifier,
        ExecutionPlan,
        LoadSkillResourceAction,
        MCPResult,
        MCPToolSpec,
        MockMCPClient,
        PlanRunState,
        PlanStatus,
        PlanStep,
        ScriptedPlanner,
        ScriptedStepExecutor,
        SkillRegistry,
        StepStatus,
        ToolAction,
        require_fields,
        run_plan_execute,
    )


ROOT = Path(__file__).parent


class StaticWriter:
    def write(self, state: PlanRunState) -> str:
        return f"已验证证据：{','.join(sorted(state.evidence))}"


def empty_mcp() -> MockMCPClient:
    return MockMCPClient([], {})


class PlanExecuteTest(unittest.IsolatedAsyncioTestCase):
    def test_evidence_requirement_checks_provenance_and_optional_version(self) -> None:
        requirement = EvidenceRequirement(
            "error_log",
            frozenset({"mcp:search_job_logs"}),
            subject_id="train-42",
            required_version=None,
        )

        self.assertTrue(
            requirement.matches(
                Evidence(
                    "error_log",
                    {"error": "OOM"},
                    "mcp:search_job_logs",
                    version="8",
                    subject_id="train-42",
                )
            )
        )
        self.assertFalse(
            requirement.matches(
                Evidence(
                    "error_log",
                    "OOM 手册",
                    "skill:training-diagnosis:references/oom.md",
                    subject_id="train-42",
                )
            )
        )

    async def test_complete_flow_uses_skill_then_resource_and_mcp(self) -> None:
        registry = SkillRegistry(ROOT / "skills")
        mcp = MockMCPClient(
            [
                MCPToolSpec(
                    "get_training_job",
                    "查询任务",
                    require_fields("job_id"),
                ),
                MCPToolSpec(
                    "search_job_logs",
                    "查询日志",
                    require_fields("job_id", "query"),
                ),
            ],
            {
                "get_training_job": lambda _: MCPResult(
                    True,
                    {"job_id": "train-42", "status": "FAILED", "version": 7},
                ),
                "search_job_logs": lambda _: MCPResult(
                    True,
                    {
                        "job_id": "train-42",
                        "error": "CUDA out of memory",
                        "version": 7,
                    },
                ),
            },
        )
        executor = ScriptedStepExecutor(
            {
                "activate-diagnosis-skill": [
                    ActivateSkillAction("training-diagnosis"),
                    CompleteStepAction("Skill 完成"),
                ],
                "collect-status": [
                    ToolAction("get_training_job", {"job_id": "train-42"}, "job_status"),
                    CompleteStepAction("状态完成"),
                ],
                "load-diagnosis-knowledge": [
                    LoadSkillResourceAction(
                        "training-diagnosis",
                        "references/oom.md",
                        "diagnosis_rules",
                    ),
                    CompleteStepAction("知识完成"),
                ],
                "collect-logs": [
                    ToolAction(
                        "search_job_logs",
                        {"job_id": "train-42", "query": "oom"},
                        "error_log",
                    ),
                    CompleteStepAction("日志完成"),
                ],
                "write-report": [CompleteStepAction("报告完成")],
            }
        )

        state = await run_plan_execute(
            goal=build_plan().goal,
            planner=ScriptedPlanner(build_plan()),
            executor=executor,
            verifier=EvidencePlanVerifier(),
            final_writer=DiagnosisFinalWriter(),
            skill_registry=registry,
            mcp_client=mcp,
        )

        self.assertEqual(state.plan.status, PlanStatus.COMPLETED)
        self.assertIn("CUDA out of memory", state.final_answer or "")
        self.assertEqual(
            [name for name, _ in mcp.call_history],
            ["get_training_job", "search_job_logs"],
        )
        phases = [event.phase for event in state.trace]
        self.assertIn("skill_discovery", phases)
        self.assertIn("skill_activated", phases)
        self.assertIn("skill_resource_loaded", phases)
        self.assertIn("mcp_tool_called", phases)
        self.assertIn("plan_completed", phases)
        log_observation_index = next(
            index
            for index, event in enumerate(state.trace)
            if event.phase == "observation"
            and event.detail.get("tool_name") == "search_job_logs"
        )
        resource_load_index = phases.index("skill_resource_loaded")
        self.assertLess(log_observation_index, resource_load_index)

    async def test_unselected_skill_only_loads_metadata(self) -> None:
        registry = SkillRegistry(ROOT / "skills")
        plan = ExecutionPlan(
            "skill-only",
            1,
            "只使用诊断 Skill",
            [
                PlanStep(
                    "load",
                    "加载诊断知识",
                    required_skills=frozenset({"training-diagnosis"}),
                    required_resources=frozenset(
                        {"training-diagnosis:references/oom.md"}
                    ),
                    required_evidence=frozenset({"rules"}),
                )
            ],
        )
        state = await run_plan_execute(
            goal=plan.goal,
            planner=ScriptedPlanner(plan),
            executor=ScriptedStepExecutor(
                {
                    "load": [
                        ActivateSkillAction("training-diagnosis"),
                        LoadSkillResourceAction(
                            "training-diagnosis",
                            "references/oom.md",
                            "rules",
                        ),
                        CompleteStepAction("完成"),
                    ]
                }
            ),
            verifier=EvidencePlanVerifier(),
            final_writer=StaticWriter(),
            skill_registry=registry,
            mcp_client=empty_mcp(),
        )

        self.assertEqual(state.plan.status, PlanStatus.COMPLETED)
        self.assertIn("metadata:report-format", registry.io_log)
        self.assertIn("metadata:training-diagnosis", registry.io_log)
        self.assertNotIn("skill:report-format", registry.io_log)
        self.assertEqual(
            registry.io_log[-2:],
            [
                "skill:training-diagnosis",
                "resource:training-diagnosis:references/oom.md",
            ],
        )

    async def test_skill_resource_cannot_impersonate_mcp_evidence(self) -> None:
        plan = ExecutionPlan(
            "source-mismatch",
            1,
            "防止 Skill 冒充 MCP Evidence",
            [
                PlanStep(
                    "load",
                    "加载诊断知识",
                    required_skills=frozenset({"training-diagnosis"}),
                    required_resources=frozenset(
                        {"training-diagnosis:references/oom.md"}
                    ),
                    evidence_requirements=(
                        EvidenceRequirement(
                            "error_log",
                            frozenset({"mcp:search_job_logs"}),
                            subject_id="train-42",
                        ),
                    ),
                )
            ],
        )
        state = await run_plan_execute(
            goal=plan.goal,
            planner=ScriptedPlanner(plan),
            executor=ScriptedStepExecutor(
                {
                    "load": [
                        ActivateSkillAction("training-diagnosis"),
                        LoadSkillResourceAction(
                            "training-diagnosis",
                            "references/oom.md",
                            "error_log",
                        ),
                    ]
                }
            ),
            verifier=EvidencePlanVerifier(),
            final_writer=StaticWriter(),
            skill_registry=SkillRegistry(ROOT / "skills"),
            mcp_client=empty_mcp(),
        )

        self.assertEqual(state.plan.status, PlanStatus.FAILED)
        self.assertEqual(state.stop_reason, "EVIDENCE_CONTRACT_MISMATCH")
        self.assertNotIn("error_log", state.evidence)

    async def test_invalid_plan_with_unknown_tool_is_rejected(self) -> None:
        plan = ExecutionPlan(
            "invalid",
            1,
            "非法计划",
            [PlanStep("bad", "调用未知工具", allowed_tools=frozenset({"delete_all"}))],
        )
        with self.assertRaisesRegex(ValueError, "未知工具"):
            await run_plan_execute(
                goal=plan.goal,
                planner=ScriptedPlanner(plan),
                executor=ScriptedStepExecutor({}),
                verifier=EvidencePlanVerifier(),
                final_writer=StaticWriter(),
                skill_registry=SkillRegistry(ROOT / "skills"),
                mcp_client=empty_mcp(),
            )

    async def test_missing_evidence_cannot_complete_step(self) -> None:
        plan = ExecutionPlan(
            "missing-evidence",
            1,
            "缺证据",
            [
                PlanStep(
                    "verify",
                    "检查证据",
                    required_evidence=frozenset({"must_exist"}),
                )
            ],
        )
        state = await run_plan_execute(
            goal=plan.goal,
            planner=ScriptedPlanner(plan),
            executor=ScriptedStepExecutor(
                {"verify": [CompleteStepAction("我认为完成了")]}
            ),
            verifier=EvidencePlanVerifier(),
            final_writer=StaticWriter(),
            skill_registry=SkillRegistry(ROOT / "skills"),
            mcp_client=empty_mcp(),
            max_actions_per_step=2,
        )

        self.assertEqual(state.plan.status, PlanStatus.FAILED)
        self.assertEqual(state.stop_reason, "STEP_ACTION_LIMIT")
        self.assertEqual(state.plan.steps[0].status, StepStatus.FAILED)

    async def test_tool_call_budget_stops_repeated_read_dispatch(self) -> None:
        plan = ExecutionPlan(
            "tool-budget",
            1,
            "限制重复查询",
            [
                PlanStep(
                    "query",
                    "查询日志",
                    allowed_tools=frozenset({"search_job_logs"}),
                )
            ],
        )
        action = ToolAction(
            "search_job_logs",
            {"job_id": "train-42", "query": "error"},
            "error_log",
        )
        mcp = MockMCPClient(
            [
                MCPToolSpec(
                    "search_job_logs",
                    "查询日志",
                    require_fields("job_id", "query"),
                    allows_polling=True,
                )
            ],
            {
                "search_job_logs": lambda _: MCPResult(
                    True,
                    {"job_id": "train-42", "error": "OOM"},
                )
            },
        )
        state = await run_plan_execute(
            goal=plan.goal,
            planner=ScriptedPlanner(plan),
            executor=ScriptedStepExecutor({"query": [action] * 6}),
            verifier=EvidencePlanVerifier(),
            final_writer=StaticWriter(),
            skill_registry=SkillRegistry(ROOT / "skills"),
            mcp_client=mcp,
            max_actions_per_step=10,
            max_tool_calls_per_step=3,
        )

        self.assertEqual(state.plan.status, PlanStatus.FAILED)
        self.assertEqual(state.stop_reason, "TOOL_CALL_LIMIT")
        self.assertEqual(len(mcp.call_history), 3)

    async def test_third_identical_tool_action_is_blocked_before_dispatch(self) -> None:
        plan = ExecutionPlan(
            "repeat-guard",
            1,
            "防止相同查询空转",
            [
                PlanStep(
                    "query",
                    "查询日志",
                    allowed_tools=frozenset({"search_job_logs"}),
                )
            ],
        )
        action = ToolAction(
            "search_job_logs",
            {"job_id": "train-42", "query": "OOM"},
            "error_log",
        )
        mcp = MockMCPClient(
            [
                MCPToolSpec(
                    "search_job_logs",
                    "查询日志",
                    require_fields("job_id", "query"),
                )
            ],
            {
                "search_job_logs": lambda _: MCPResult(
                    True,
                    {
                        "job_id": "train-42",
                        "error": "OOM",
                        "version": 7,
                    },
                )
            },
        )
        state = await run_plan_execute(
            goal=plan.goal,
            planner=ScriptedPlanner(plan),
            executor=ScriptedStepExecutor({"query": [action] * 4}),
            verifier=EvidencePlanVerifier(),
            final_writer=StaticWriter(),
            skill_registry=SkillRegistry(ROOT / "skills"),
            mcp_client=mcp,
            max_actions_per_step=6,
            max_tool_calls_per_step=5,
        )

        self.assertEqual(state.plan.status, PlanStatus.FAILED)
        self.assertEqual(state.stop_reason, "REPEATED_ACTION")
        self.assertEqual(len(mcp.call_history), 2)

    async def test_observation_can_trigger_replan(self) -> None:
        initial = ExecutionPlan(
            "replan-demo",
            1,
            "检查旧任务",
            [
                PlanStep(
                    "inspect-old",
                    "检查旧任务",
                    allowed_tools=frozenset({"inspect_target"}),
                    required_evidence=frozenset({"old_target"}),
                )
            ],
        )
        revised = ExecutionPlan(
            "planner-hallucinated-id",
            99,
            "Planner 错误改写的目标",
            [
                PlanStep(
                    "inspect-new",
                    "检查新任务",
                    allowed_tools=frozenset({"get_training_job"}),
                    required_evidence=frozenset({"new_target"}),
                )
            ],
        )
        mcp = MockMCPClient(
            [
                MCPToolSpec("inspect_target", "检查目标", require_fields("job_id")),
                MCPToolSpec("get_training_job", "查询新目标", require_fields("job_id")),
            ],
            {
                "inspect_target": lambda _: MCPResult(
                    True,
                    {"job_id": "old"},
                    replan_reason="用户目标已经切换到 new",
                ),
                "get_training_job": lambda _: MCPResult(
                    True,
                    {"job_id": "new", "status": "RUNNING"},
                ),
            },
        )
        executor = ScriptedStepExecutor(
            {
                "inspect-old": [
                    ToolAction("inspect_target", {"job_id": "old"}, "old_target")
                ],
                "inspect-new": [
                    ToolAction("get_training_job", {"job_id": "new"}, "new_target"),
                    CompleteStepAction("新目标已确认"),
                ],
            }
        )

        state = await run_plan_execute(
            goal=initial.goal,
            planner=ScriptedPlanner(initial, [revised]),
            executor=executor,
            verifier=EvidencePlanVerifier(),
            final_writer=StaticWriter(),
            skill_registry=SkillRegistry(ROOT / "skills"),
            mcp_client=mcp,
            max_replans=1,
        )

        self.assertEqual(state.plan.status, PlanStatus.COMPLETED)
        self.assertEqual(state.plan.plan_id, "replan-demo")
        self.assertEqual(state.plan.goal, initial.goal)
        self.assertEqual(state.plan.version, 2)
        self.assertEqual(state.plan.replan_count, 1)
        self.assertIn("plan_revised", [event.phase for event in state.trace])
        self.assertNotIn("old_target", state.evidence)
        self.assertEqual(
            [item.evidence.key for item in state.archived_evidence],
            ["old_target"],
        )
        self.assertIn("evidence_invalidated", [event.phase for event in state.trace])

    async def test_replan_limit_stops_second_invalidation(self) -> None:
        initial = ExecutionPlan(
            "replan-limit",
            1,
            "初始计划",
            [
                PlanStep(
                    "first",
                    "第一次检查",
                    allowed_tools=frozenset({"inspect"}),
                )
            ],
        )
        revised = ExecutionPlan(
            "replan-limit",
            2,
            "修订计划",
            [
                PlanStep(
                    "second",
                    "第二次检查",
                    allowed_tools=frozenset({"inspect"}),
                )
            ],
        )
        mcp = MockMCPClient(
            [MCPToolSpec("inspect", "检查", require_fields("value"))],
            {
                "inspect": lambda _: MCPResult(
                    True,
                    {"changed": True},
                    replan_reason="目标再次变化",
                )
            },
        )
        executor = ScriptedStepExecutor(
            {
                "first": [ToolAction("inspect", {"value": 1}, "first")],
                "second": [ToolAction("inspect", {"value": 2}, "second")],
            }
        )

        state = await run_plan_execute(
            goal=initial.goal,
            planner=ScriptedPlanner(initial, [revised]),
            executor=executor,
            verifier=EvidencePlanVerifier(),
            final_writer=StaticWriter(),
            skill_registry=SkillRegistry(ROOT / "skills"),
            mcp_client=mcp,
            max_replans=1,
        )

        self.assertEqual(state.plan.status, PlanStatus.FAILED)
        self.assertEqual(state.stop_reason, "REPLAN_LIMIT")

    async def test_replan_retains_evidence_that_matches_new_contract(self) -> None:
        requirement = EvidenceRequirement(
            "job_status",
            frozenset({"mcp:get_training_job"}),
            subject_id="train-42",
        )
        initial = ExecutionPlan(
            "retain-evidence",
            1,
            "诊断 train-42",
            [
                PlanStep(
                    "inspect",
                    "查询状态",
                    allowed_tools=frozenset({"get_training_job"}),
                    evidence_requirements=(requirement,),
                )
            ],
        )
        revised = ExecutionPlan(
            "ignored",
            0,
            "ignored",
            [
                PlanStep(
                    "write-json",
                    "改为 JSON 输出",
                    evidence_requirements=(requirement,),
                )
            ],
        )
        mcp = MockMCPClient(
            [
                MCPToolSpec(
                    "get_training_job",
                    "查询任务",
                    require_fields("job_id"),
                )
            ],
            {
                "get_training_job": lambda _: MCPResult(
                    True,
                    {"job_id": "train-42", "status": "FAILED", "version": 7},
                    replan_reason="改为 JSON 报告",
                )
            },
        )
        state = await run_plan_execute(
            goal=initial.goal,
            planner=ScriptedPlanner(initial, [revised]),
            executor=ScriptedStepExecutor(
                {
                    "inspect": [
                        ToolAction(
                            "get_training_job",
                            {"job_id": "train-42"},
                            "job_status",
                        )
                    ],
                    "write-json": [CompleteStepAction("JSON 报告完成")],
                }
            ),
            verifier=EvidencePlanVerifier(),
            final_writer=StaticWriter(),
            skill_registry=SkillRegistry(ROOT / "skills"),
            mcp_client=mcp,
        )

        self.assertEqual(state.plan.status, PlanStatus.COMPLETED)
        self.assertIn("job_status", state.evidence)
        self.assertEqual(state.archived_evidence, [])
        self.assertEqual(len(mcp.call_history), 1)

    async def test_side_effect_without_confirmation_needs_human(self) -> None:
        plan = ExecutionPlan(
            "dangerous",
            1,
            "取消训练任务",
            [
                PlanStep(
                    "cancel",
                    "执行已确认的取消",
                    allowed_tools=frozenset({"commit_cancel"}),
                )
            ],
        )
        mcp = MockMCPClient(
            [
                MCPToolSpec(
                    "commit_cancel",
                    "取消任务",
                    require_fields("job_id"),
                    has_side_effect=True,
                    requires_confirmation=True,
                )
            ],
            {"commit_cancel": lambda _: MCPResult(True, {"status": "CANCELLED"})},
        )
        state = await run_plan_execute(
            goal=plan.goal,
            planner=ScriptedPlanner(plan),
            executor=ScriptedStepExecutor(
                {
                    "cancel": [
                        ToolAction("commit_cancel", {"job_id": "train-1"}, "cancelled")
                    ]
                }
            ),
            verifier=EvidencePlanVerifier(),
            final_writer=StaticWriter(),
            skill_registry=SkillRegistry(ROOT / "skills"),
            mcp_client=mcp,
        )

        self.assertEqual(state.plan.status, PlanStatus.NEEDS_HUMAN)
        self.assertEqual(state.stop_reason, "CONFIRMATION_REQUIRED")
        self.assertEqual(mcp.call_history, [])

    async def test_side_effect_timeout_records_intent_and_blocks_redispatch(self) -> None:
        plan = ExecutionPlan(
            "uncertain-side-effect",
            1,
            "取消训练任务",
            [
                PlanStep(
                    "cancel",
                    "取消任务",
                    allowed_tools=frozenset({"commit_cancel"}),
                )
            ],
        )

        def timeout_after_dispatch(_: object) -> MCPResult:
            raise TimeoutError("请求可能已经到达服务端")

        spec = MCPToolSpec(
            "commit_cancel",
            "取消任务",
            require_fields("job_id", "confirmation_id", "idempotency_key"),
            has_side_effect=True,
            requires_confirmation=True,
            idempotency_field="idempotency_key",
        )
        action = ToolAction(
            "commit_cancel",
            {
                "job_id": "train-42",
                "confirmation_id": "confirm-1",
                "idempotency_key": "cancel-001",
            },
            "cancel_result",
        )
        first_mcp = MockMCPClient([spec], {"commit_cancel": timeout_after_dispatch})
        state = await run_plan_execute(
            goal=plan.goal,
            planner=ScriptedPlanner(plan),
            executor=ScriptedStepExecutor({"cancel": [action]}),
            verifier=EvidencePlanVerifier(),
            final_writer=StaticWriter(),
            skill_registry=SkillRegistry(ROOT / "skills"),
            mcp_client=first_mcp,
        )

        self.assertEqual(state.plan.status, PlanStatus.NEEDS_HUMAN)
        self.assertEqual(state.stop_reason, "RESULT_UNKNOWN")
        self.assertEqual(
            state.dispatches["cancel-001"].status,
            DispatchStatus.RESULT_UNKNOWN,
        )
        phases = [event.phase for event in state.trace]
        self.assertLess(
            phases.index("dispatch_intent_recorded"),
            phases.index("mcp_tool_called"),
        )

        state.plan.status = PlanStatus.ACTIVE
        state.plan.steps[0].status = StepStatus.IN_PROGRESS
        second_mcp = MockMCPClient(
            [spec],
            {"commit_cancel": lambda _: MCPResult(True, {"status": "CANCELLED"})},
        )
        resumed = await run_plan_execute(
            goal=plan.goal,
            planner=ScriptedPlanner(plan),
            executor=ScriptedStepExecutor({"cancel": [action]}),
            verifier=EvidencePlanVerifier(),
            final_writer=StaticWriter(),
            skill_registry=SkillRegistry(ROOT / "skills"),
            mcp_client=second_mcp,
            state=state,
        )

        self.assertEqual(resumed.stop_reason, "RESULT_UNKNOWN")
        self.assertEqual(second_mcp.call_history, [])
        self.assertIn(
            "duplicate_dispatch_blocked",
            [event.phase for event in resumed.trace],
        )

    async def test_resume_skips_completed_step(self) -> None:
        plan = ExecutionPlan(
            "resume",
            1,
            "恢复执行",
            [
                PlanStep(
                    "done",
                    "已完成步骤",
                    required_evidence=frozenset({"first"}),
                    status=StepStatus.COMPLETED,
                ),
                PlanStep(
                    "remaining",
                    "剩余步骤",
                    dependencies=("done",),
                    required_evidence=frozenset({"first"}),
                ),
            ],
            current_step_index=1,
        )
        state = PlanRunState(goal=plan.goal, plan=plan)
        state.evidence["first"] = Evidence("first", "ok", "checkpoint")

        resumed = await run_plan_execute(
            goal=plan.goal,
            planner=ScriptedPlanner(plan),
            executor=ScriptedStepExecutor(
                {"remaining": [CompleteStepAction("恢复完成")]}
            ),
            verifier=EvidencePlanVerifier(),
            final_writer=StaticWriter(),
            skill_registry=SkillRegistry(ROOT / "skills"),
            mcp_client=empty_mcp(),
            state=state,
        )

        self.assertEqual(resumed.plan.status, PlanStatus.COMPLETED)
        self.assertEqual(resumed.plan.steps[0].attempts, 0)
        self.assertIn("run_resumed", [event.phase for event in resumed.trace])

    def test_skill_resource_path_cannot_escape_directory(self) -> None:
        registry = SkillRegistry(ROOT / "skills")
        registry.discover()
        registry.activate("training-diagnosis")

        with self.assertRaisesRegex(ValueError, "不能逃逸"):
            registry.read_resource("training-diagnosis", "../../README.md")


if __name__ == "__main__":
    unittest.main()

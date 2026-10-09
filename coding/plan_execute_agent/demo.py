"""运行一次包含 Skill 渐进式加载和 MCP 调用的 Plan-and-Execute。"""

from __future__ import annotations

import asyncio
from pathlib import Path

try:  # 支持 ``python -m coding.plan_execute_agent.demo`` 的包入口。
    from .plan_execute import (
        ActivateSkillAction,
        CompleteStepAction,
        DiagnosisFinalWriter,
        EvidenceRequirement,
        EvidencePlanVerifier,
        ExecutionPlan,
        LoadSkillResourceAction,
        MCPResult,
        MCPToolSpec,
        MockMCPClient,
        PlanStep,
        ScriptedPlanner,
        ScriptedStepExecutor,
        SkillRegistry,
        ToolAction,
        require_fields,
        run_plan_execute,
    )
except ImportError:  # 支持在本目录执行 ``python demo.py``。
    from plan_execute import (
        ActivateSkillAction,
        CompleteStepAction,
        DiagnosisFinalWriter,
        EvidenceRequirement,
        EvidencePlanVerifier,
        ExecutionPlan,
        LoadSkillResourceAction,
        MCPResult,
        MCPToolSpec,
        MockMCPClient,
        PlanStep,
        ScriptedPlanner,
        ScriptedStepExecutor,
        SkillRegistry,
        ToolAction,
        require_fields,
        run_plan_execute,
    )


def build_plan() -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="diagnose-train-42",
        version=1,
        goal="诊断训练任务 train-42 的失败原因并给出修复建议",
        steps=[
            PlanStep(
                step_id="activate-diagnosis-skill",
                objective="激活通用训练故障诊断流程",
                required_skills=frozenset({"training-diagnosis"}),
                completion_condition="诊断 Skill 已激活",
            ),
            PlanStep(
                step_id="collect-status",
                objective="查询训练任务当前权威状态",
                dependencies=("activate-diagnosis-skill",),
                allowed_tools=frozenset({"get_training_job"}),
                evidence_requirements=(
                    EvidenceRequirement(
                        "job_status",
                        frozenset({"mcp:get_training_job"}),
                        subject_id="train-42",
                    ),
                ),
                completion_condition="取得 train-42 的带版本状态",
            ),
            PlanStep(
                step_id="collect-logs",
                objective="查询错误日志以确认失败原因",
                dependencies=("collect-status",),
                allowed_tools=frozenset({"search_job_logs"}),
                evidence_requirements=(
                    EvidenceRequirement(
                        "error_log",
                        frozenset({"mcp:search_job_logs"}),
                        subject_id="train-42",
                    ),
                ),
                completion_condition="获得与任务版本一致的错误日志",
            ),
            PlanStep(
                step_id="load-diagnosis-knowledge",
                objective="根据已确认的 OOM 日志按需加载诊断规则",
                dependencies=("collect-logs",),
                required_skills=frozenset({"training-diagnosis"}),
                required_resources=frozenset(
                    {"training-diagnosis:references/oom.md"}
                ),
                evidence_requirements=(
                    EvidenceRequirement(
                        "diagnosis_rules",
                        frozenset(
                            {"skill:training-diagnosis:references/oom.md"}
                        ),
                    ),
                    EvidenceRequirement(
                        "error_log",
                        frozenset({"mcp:search_job_logs"}),
                        subject_id="train-42",
                    ),
                ),
                completion_condition="错误日志确认 OOM 且对应规则已按需加载",
            ),
            PlanStep(
                step_id="write-report",
                objective="根据已验证证据生成诊断报告",
                dependencies=("load-diagnosis-knowledge",),
                evidence_requirements=(
                    EvidenceRequirement(
                        "job_status",
                        frozenset({"mcp:get_training_job"}),
                        subject_id="train-42",
                    ),
                    EvidenceRequirement(
                        "diagnosis_rules",
                        frozenset(
                            {"skill:training-diagnosis:references/oom.md"}
                        ),
                    ),
                    EvidenceRequirement(
                        "error_log",
                        frozenset({"mcp:search_job_logs"}),
                        subject_id="train-42",
                    ),
                ),
                completion_condition="事实、规则和日志证据均完整",
            ),
        ],
    )


async def main() -> None:
    root = Path(__file__).parent
    registry = SkillRegistry(root / "skills")
    mcp = MockMCPClient(
        specs=[
            MCPToolSpec(
                "get_training_job",
                "查询训练任务权威状态",
                require_fields("job_id"),
            ),
            MCPToolSpec(
                "search_job_logs",
                "搜索训练任务日志",
                require_fields("job_id", "query"),
            ),
        ],
        handlers={
            "get_training_job": lambda _: MCPResult(
                True,
                {"job_id": "train-42", "status": "FAILED", "version": 7},
            ),
            "search_job_logs": lambda _: MCPResult(
                True,
                {"job_id": "train-42", "error": "CUDA out of memory", "version": 7},
            ),
        },
    )
    executor = ScriptedStepExecutor(
        {
            "activate-diagnosis-skill": [
                ActivateSkillAction("training-diagnosis"),
                CompleteStepAction("已激活通用诊断 Skill。"),
            ],
            "collect-status": [
                ToolAction("get_training_job", {"job_id": "train-42"}, "job_status"),
                CompleteStepAction("已获得任务状态。"),
            ],
            "collect-logs": [
                ToolAction(
                    "search_job_logs",
                    {"job_id": "train-42", "query": "out of memory"},
                    "error_log",
                ),
                CompleteStepAction("日志确认 OOM。"),
            ],
            "load-diagnosis-knowledge": [
                LoadSkillResourceAction(
                    "training-diagnosis",
                    "references/oom.md",
                    "diagnosis_rules",
                ),
                CompleteStepAction("已按日志加载 OOM 诊断规则。"),
            ],
            "write-report": [CompleteStepAction("证据完整，可以生成报告。")],
        }
    )

    state = await run_plan_execute(
        goal="诊断训练任务 train-42 的失败原因并给出修复建议",
        planner=ScriptedPlanner(build_plan()),
        executor=executor,
        verifier=EvidencePlanVerifier(),
        final_writer=DiagnosisFinalWriter(),
        skill_registry=registry,
        mcp_client=mcp,
    )

    print("=== 最终回答 ===")
    print(state.final_answer)
    print("\n=== Skill I/O 顺序 ===")
    for item in registry.io_log:
        print(item)
    print("\n=== Trace ===")
    for event in state.trace:
        print(f"{event.sequence:02d} {event.phase}: {event.detail}")


if __name__ == "__main__":
    asyncio.run(main())

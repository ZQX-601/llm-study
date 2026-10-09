"""Plan-and-Execute + 渐进式 Skill 加载 + MCP 的完整教学链路。

该实现使用纯标准库和确定性组件，不调用真实模型或远程 MCP Server。它重点展示：

1. Planner 只产生计划，Runtime 校验并保存计划版本；
2. Executor 在单个 Step 内进行有限步 ReAct；
3. Skill 先发现元数据，再加载 SKILL.md，支持文件最后按需读取；
4. MCP Client 先发现工具，再由 Runtime 校验 Step 权限后执行；
5. Verifier 根据结构化 Evidence 决定 next/retry/replan/human/fail；
6. Runtime 持有状态、预算、恢复点和完整 Trace。
"""

from __future__ import annotations

import copy
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol, TypeAlias


class StepStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    INVALIDATED = "INVALIDATED"


class PlanStatus(str, Enum):
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    NEEDS_REPLAN = "NEEDS_REPLAN"
    NEEDS_HUMAN = "NEEDS_HUMAN"
    FAILED = "FAILED"


class PlanDecision(str, Enum):
    COMPLETE_STEP = "COMPLETE_STEP"
    RETRY_STEP = "RETRY_STEP"
    REPLAN = "REPLAN"
    NEEDS_HUMAN = "NEEDS_HUMAN"
    FAIL = "FAIL"


class DispatchStatus(str, Enum):
    INTENT_RECORDED = "INTENT_RECORDED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    RESULT_UNKNOWN = "RESULT_UNKNOWN"


@dataclass(frozen=True)
class SkillMetadata:
    """发现阶段进入 Context 的最小 Skill 信息。"""

    name: str
    description: str
    path: Path


@dataclass(frozen=True)
class ActivatedSkill:
    """激活后才读取的完整 SKILL.md。"""

    metadata: SkillMetadata
    instructions: str


class SkillRegistry:
    """实现 metadata → SKILL.md → supporting resource 的渐进式加载。"""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._catalog: dict[str, SkillMetadata] = {}
        self._activated: dict[str, ActivatedSkill] = {}
        self.io_log: list[str] = []

    @staticmethod
    def _read_frontmatter(path: Path) -> dict[str, str]:
        """只读取 YAML frontmatter，不把正文提前装入 Context。

        教学解析器只支持 ``key: value``，足以展示路由机制；生产实现应使用规范兼容的
        YAML/Agent Skills 解析器并校验签名、版本和资源摘要。
        """

        result: dict[str, str] = {}
        with path.open("r", encoding="utf-8") as handle:
            if handle.readline().strip() != "---":
                raise ValueError(f"{path} 缺少 frontmatter 起始标记")
            for line in handle:
                stripped = line.strip()
                if stripped == "---":
                    break
                if ":" not in stripped:
                    continue
                key, value = stripped.split(":", 1)
                result[key.strip()] = value.strip().strip('"')
        return result

    def discover(self) -> tuple[SkillMetadata, ...]:
        """扫描 Skill，只返回 name/description/path。"""

        catalog: dict[str, SkillMetadata] = {}
        if not self.root.exists():
            raise FileNotFoundError(f"Skill 根目录不存在：{self.root}")
        for skill_file in sorted(self.root.glob("*/SKILL.md")):
            frontmatter = self._read_frontmatter(skill_file)
            name = frontmatter.get("name", "").strip()
            description = frontmatter.get("description", "").strip()
            if not name or not description:
                raise ValueError(f"{skill_file} 必须声明 name 和 description")
            if name in catalog:
                raise ValueError(f"Skill 名称重复：{name}")
            catalog[name] = SkillMetadata(name, description, skill_file)
            self.io_log.append(f"metadata:{name}")
        self._catalog = catalog
        return tuple(catalog.values())

    def activate(self, name: str) -> ActivatedSkill:
        """选择 Skill 后读取完整 SKILL.md 正文。"""

        if name in self._activated:
            return self._activated[name]
        metadata = self._catalog.get(name)
        if metadata is None:
            raise KeyError(f"未发现 Skill：{name}")
        text = metadata.path.read_text(encoding="utf-8")
        parts = text.split("---", 2)
        instructions = parts[2].strip() if len(parts) == 3 else ""
        activated = ActivatedSkill(metadata, instructions)
        self._activated[name] = activated
        self.io_log.append(f"skill:{name}")
        return activated

    def read_resource(self, name: str, relative_path: str) -> str:
        """仅在 Skill 已激活后读取当前 Step 需要的支持文件。"""

        activated = self._activated.get(name)
        if activated is None:
            raise RuntimeError(f"必须先激活 Skill 才能读取资源：{name}")
        skill_root = activated.metadata.path.parent.resolve()
        candidate = (skill_root / relative_path).resolve()
        try:
            candidate.relative_to(skill_root)
        except ValueError as exc:
            raise ValueError("Skill 资源路径不能逃逸 Skill 目录") from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"Skill 资源不存在：{relative_path}")
        self.io_log.append(f"resource:{name}:{relative_path}")
        return candidate.read_text(encoding="utf-8")


ArgumentValidator = Callable[[Mapping[str, object]], str | None]


@dataclass(frozen=True)
class MCPToolSpec:
    name: str
    description: str
    validator: ArgumentValidator
    has_side_effect: bool = False
    requires_confirmation: bool = False
    idempotency_field: str | None = None
    allows_polling: bool = False


@dataclass(frozen=True)
class MCPResult:
    ok: bool
    data: dict[str, object] | None = None
    error_code: str | None = None
    message: str = ""
    retryable: bool = False
    replan_reason: str | None = None


MCPHandler = Callable[
    [Mapping[str, object]],
    MCPResult | Awaitable[MCPResult],
]


class MockMCPClient:
    """确定性 MCP Client；接口对应真实 MCP 的 tools/list 与 tools/call。"""

    def __init__(
        self,
        specs: Sequence[MCPToolSpec],
        handlers: Mapping[str, MCPHandler],
    ) -> None:
        self._specs = tuple(specs)
        self._handlers = dict(handlers)
        self.call_history: list[tuple[str, dict[str, object]]] = []

    async def list_tools(self) -> tuple[MCPToolSpec, ...]:
        return self._specs

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, object],
    ) -> MCPResult:
        self.call_history.append((name, dict(arguments)))
        handler = self._handlers[name]
        result = handler(arguments)
        if inspect.isawaitable(result):
            return await result
        return result


def require_fields(*field_names: str) -> ArgumentValidator:
    def validate(arguments: Mapping[str, object]) -> str | None:
        missing = [name for name in field_names if name not in arguments]
        return f"缺少必填参数：{', '.join(missing)}" if missing else None

    return validate


@dataclass(frozen=True)
class Evidence:
    """Verifier 使用的稳定证据；value 不直接替代来源与版本。"""

    key: str
    value: object
    source: str
    version: str | None = None
    subject_id: str | None = None


@dataclass(frozen=True)
class EvidenceRequirement:
    """定义 Step 所需 Evidence 的来源、目标和版本合同。"""

    key: str
    allowed_sources: frozenset[str]
    subject_id: str | None = None
    required_version: str | None = None

    def matches(self, evidence: Evidence | None) -> bool:
        """``None`` 约束表示不限制该字段，而不是要求 Evidence 为 None。"""

        if evidence is None:
            return False
        return (
            evidence.key == self.key
            and evidence.source in self.allowed_sources
            and (self.subject_id is None or evidence.subject_id == self.subject_id)
            and (
                self.required_version is None
                or evidence.version == self.required_version
            )
        )


@dataclass
class PlanStep:
    step_id: str
    objective: str
    dependencies: tuple[str, ...] = ()
    allowed_tools: frozenset[str] = field(default_factory=frozenset)
    required_skills: frozenset[str] = field(default_factory=frozenset)
    required_resources: frozenset[str] = field(default_factory=frozenset)
    required_evidence: frozenset[str] = field(default_factory=frozenset)
    evidence_requirements: tuple[EvidenceRequirement, ...] = ()
    completion_condition: str = ""
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    summary: str | None = None


@dataclass
class ExecutionPlan:
    plan_id: str
    version: int
    goal: str
    steps: list[PlanStep]
    current_step_index: int = 0
    replan_count: int = 0
    status: PlanStatus = PlanStatus.ACTIVE


@dataclass(frozen=True)
class TraceEvent:
    sequence: int
    phase: str
    detail: dict[str, object]


@dataclass
class DispatchRecord:
    """副作用调用的持久化意图；必须在真正 dispatch 之前创建。"""

    idempotency_key: str
    plan_id: str
    plan_version: int
    step_id: str
    tool_name: str
    arguments: dict[str, object]
    evidence_key: str
    status: DispatchStatus = DispatchStatus.INTENT_RECORDED


@dataclass(frozen=True)
class ArchivedEvidence:
    evidence: Evidence
    invalidated_at_plan_version: int
    reason: str


@dataclass
class PlanRunState:
    goal: str
    plan: ExecutionPlan | None = None
    evidence: dict[str, Evidence] = field(default_factory=dict)
    archived_evidence: list[ArchivedEvidence] = field(default_factory=list)
    activated_skills: dict[str, ActivatedSkill] = field(default_factory=dict)
    loaded_resources: dict[str, str] = field(default_factory=dict)
    dispatches: dict[str, DispatchRecord] = field(default_factory=dict)
    trace: list[TraceEvent] = field(default_factory=list)
    final_answer: str | None = None
    stop_reason: str | None = None


@dataclass(frozen=True)
class ActivateSkillAction:
    skill_name: str


@dataclass(frozen=True)
class LoadSkillResourceAction:
    skill_name: str
    relative_path: str
    evidence_key: str


@dataclass(frozen=True)
class ToolAction:
    tool_name: str
    arguments: dict[str, object]
    evidence_key: str


@dataclass(frozen=True)
class CompleteStepAction:
    summary: str


StepAction: TypeAlias = (
    ActivateSkillAction
    | LoadSkillResourceAction
    | ToolAction
    | CompleteStepAction
)


class Planner(Protocol):
    def create_plan(
        self,
        goal: str,
        skills: Sequence[SkillMetadata],
        tools: Sequence[MCPToolSpec],
    ) -> ExecutionPlan:
        """根据目标和能力元数据生成初始计划。"""

    def replan(
        self,
        previous: ExecutionPlan,
        state: PlanRunState,
        reason: str,
    ) -> ExecutionPlan:
        """根据新 Evidence 生成更高版本计划。"""


class StepExecutor(Protocol):
    async def decide(
        self,
        plan: ExecutionPlan,
        step: PlanStep,
        state: PlanRunState,
        skills: Sequence[SkillMetadata],
        tools: Sequence[MCPToolSpec],
    ) -> StepAction:
        """在当前 Step 内提出下一步 Skill、Tool 或完成动作。"""


class PlanVerifier(Protocol):
    def verify_step(
        self,
        plan: ExecutionPlan,
        step: PlanStep,
        state: PlanRunState,
        action: CompleteStepAction,
    ) -> PlanDecision:
        """依据 Evidence 判断当前 Step 是否完成。"""


class FinalWriter(Protocol):
    def write(self, state: PlanRunState) -> str:
        """只使用已验证 Evidence 生成最终回答。"""


class ScriptedPlanner:
    """确定性 Planner，用来隔离真实模型随机性。"""

    def __init__(
        self,
        initial_plan: ExecutionPlan,
        replans: Sequence[ExecutionPlan] = (),
    ) -> None:
        self._initial_plan = initial_plan
        self._replans = list(replans)
        self._replan_index = 0

    def create_plan(
        self,
        goal: str,
        skills: Sequence[SkillMetadata],
        tools: Sequence[MCPToolSpec],
    ) -> ExecutionPlan:
        del goal, skills, tools
        return copy.deepcopy(self._initial_plan)

    def replan(
        self,
        previous: ExecutionPlan,
        state: PlanRunState,
        reason: str,
    ) -> ExecutionPlan:
        del state, reason
        if self._replan_index >= len(self._replans):
            raise RuntimeError("ScriptedPlanner 没有可用的重规划脚本")
        plan = copy.deepcopy(self._replans[self._replan_index])
        self._replan_index += 1
        # Planner 只返回候选结构；版本和预算计数由 Runtime 强制维护。
        del previous
        return plan


class ScriptedStepExecutor:
    """按 step_id 顺序返回动作，模拟 Step 内有限步 ReAct。"""

    def __init__(self, actions: Mapping[str, Sequence[StepAction]]) -> None:
        self._actions = {key: list(value) for key, value in actions.items()}
        self._indices: dict[str, int] = {}

    async def decide(
        self,
        plan: ExecutionPlan,
        step: PlanStep,
        state: PlanRunState,
        skills: Sequence[SkillMetadata],
        tools: Sequence[MCPToolSpec],
    ) -> StepAction:
        del plan, state, skills, tools
        index = self._indices.get(step.step_id, 0)
        scripted = self._actions.get(step.step_id, [])
        if index >= len(scripted):
            return CompleteStepAction("动作脚本结束，申请完成当前 Step。")
        self._indices[step.step_id] = index + 1
        return scripted[index]


class EvidencePlanVerifier:
    """最小确定性 Verifier：Skill、资源与 Evidence 齐全才允许完成。"""

    def verify_step(
        self,
        plan: ExecutionPlan,
        step: PlanStep,
        state: PlanRunState,
        action: CompleteStepAction,
    ) -> PlanDecision:
        del plan, action
        missing_evidence = step.required_evidence.difference(state.evidence)
        contract_mismatches = [
            requirement.key
            for requirement in step.evidence_requirements
            if not requirement.matches(state.evidence.get(requirement.key))
        ]
        missing_skills = step.required_skills.difference(state.activated_skills)
        missing_resources = step.required_resources.difference(state.loaded_resources)
        missing = (
            missing_evidence
            or contract_mismatches
            or missing_skills
            or missing_resources
        )
        return PlanDecision.RETRY_STEP if missing else PlanDecision.COMPLETE_STEP


class DiagnosisFinalWriter:
    """训练平台示例的确定性 Final Writer。"""

    def write(self, state: PlanRunState) -> str:
        job = state.evidence.get("job_status")
        log = state.evidence.get("error_log")
        rules = state.evidence.get("diagnosis_rules")
        if not job or not log or not rules:
            raise RuntimeError("最终回答缺少 job_status、error_log 或 diagnosis_rules")
        job_data = job.value if isinstance(job.value, dict) else {}
        log_data = log.value if isinstance(log.value, dict) else {}
        return (
            f"任务 {job_data.get('job_id')} 状态为 {job_data.get('status')}；"
            f"日志错误为 {log_data.get('error')}。"
            "依据已加载的诊断规则，建议先降低 micro-batch 或启用梯度累积，并复测显存峰值。"
        )


def _record(state: PlanRunState, phase: str, **detail: object) -> None:
    state.trace.append(TraceEvent(len(state.trace) + 1, phase, dict(detail)))


def _tool_fingerprint(action: ToolAction) -> str:
    """对 Tool 和规范化参数生成稳定指纹，用于识别 Step 内空转。"""

    normalized = json.dumps(
        action.arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return f"{action.tool_name}:{normalized}"


def _evidence_matches_step_contract(step: PlanStep, evidence: Evidence) -> bool:
    """Runtime 入库前拦截同 key 但来源、目标或版本错误的 Evidence。"""

    requirements = [
        requirement
        for requirement in step.evidence_requirements
        if requirement.key == evidence.key
    ]
    return not requirements or any(
        requirement.matches(evidence) for requirement in requirements
    )


def _revalidate_evidence_for_plan(
    state: PlanRunState,
    new_plan: ExecutionPlan,
) -> None:
    """把不再满足新计划的 Evidence 移出活跃区，但保留审计历史。"""

    legacy_keys = {
        key
        for step in new_plan.steps
        for key in step.required_evidence
    }
    requirements_by_key: dict[str, list[EvidenceRequirement]] = {}
    for step in new_plan.steps:
        for requirement in step.evidence_requirements:
            requirements_by_key.setdefault(requirement.key, []).append(requirement)

    for key, evidence in tuple(state.evidence.items()):
        requirements = requirements_by_key.get(key, [])
        if requirements:
            remains_valid = any(
                requirement.matches(evidence) for requirement in requirements
            )
            reason = "EVIDENCE_CONTRACT_CHANGED"
        else:
            remains_valid = key in legacy_keys
            reason = "EVIDENCE_NO_LONGER_REQUIRED"
        if remains_valid:
            continue
        state.evidence.pop(key)
        state.archived_evidence.append(
            ArchivedEvidence(evidence, new_plan.version, reason)
        )
        _record(
            state,
            "evidence_invalidated",
            evidence_key=key,
            source=evidence.source,
            subject_id=evidence.subject_id,
            invalidated_at_plan_version=new_plan.version,
            reason=reason,
        )


def _validate_plan(
    plan: ExecutionPlan,
    skills: Mapping[str, SkillMetadata],
    tools: Mapping[str, MCPToolSpec],
) -> None:
    """在执行前验证顺序计划的 ID、依赖与能力边界。"""

    if not plan.steps:
        raise ValueError("ExecutionPlan 至少需要一个 Step")
    seen: set[str] = set()
    for step in plan.steps:
        if not step.step_id or step.step_id in seen:
            raise ValueError(f"Step ID 为空或重复：{step.step_id!r}")
        unknown_dependencies = set(step.dependencies).difference(seen)
        if unknown_dependencies:
            raise ValueError(
                f"Step {step.step_id} 依赖尚未出现的 Step："
                f"{sorted(unknown_dependencies)}"
            )
        unknown_tools = step.allowed_tools.difference(tools)
        if unknown_tools:
            raise ValueError(f"Step {step.step_id} 使用未知工具：{sorted(unknown_tools)}")
        unknown_skills = step.required_skills.difference(skills)
        if unknown_skills:
            raise ValueError(f"Step {step.step_id} 使用未知 Skill：{sorted(unknown_skills)}")
        for resource in step.required_resources:
            if ":" not in resource:
                raise ValueError(f"Skill 资源必须使用 skill:path 格式：{resource}")
            skill_name, _ = resource.split(":", 1)
            if skill_name not in step.required_skills:
                raise ValueError(f"资源 {resource} 所属 Skill 未声明在 required_skills")
        requirement_keys: set[str] = set()
        for requirement in step.evidence_requirements:
            if not requirement.key or not requirement.allowed_sources:
                raise ValueError("EvidenceRequirement 需要 key 和 allowed_sources")
            if requirement.key in requirement_keys:
                raise ValueError(
                    f"Step {step.step_id} 的 EvidenceRequirement key 重复："
                    f"{requirement.key}"
                )
            requirement_keys.add(requirement.key)
        seen.add(step.step_id)


async def run_plan_execute(
    *,
    goal: str,
    planner: Planner,
    executor: StepExecutor,
    verifier: PlanVerifier,
    final_writer: FinalWriter,
    skill_registry: SkillRegistry,
    mcp_client: MockMCPClient,
    state: PlanRunState | None = None,
    max_actions_per_step: int = 6,
    max_tool_calls_per_step: int = 4,
    max_replans: int = 1,
) -> PlanRunState:
    """运行完整 Plan-and-Execute Loop。

    输入/输出没有张量；核心状态是 ``PlanRunState``。每个 Step 的内部 Action 数量由
    ``max_actions_per_step`` 限制，对应有限步 ReAct，防止模型无限循环。
    """

    if max_actions_per_step < 1 or max_tool_calls_per_step < 1 or max_replans < 0:
        raise ValueError(
            "Action 和 Tool 上限必须大于 0，Replan 上限不能为负数"
        )

    active_state = state or PlanRunState(goal=goal)
    skill_catalog = tuple(skill_registry.discover())
    skills_by_name = {item.name: item for item in skill_catalog}
    _record(
        active_state,
        "skill_discovery",
        skills=[{"name": item.name, "description": item.description} for item in skill_catalog],
    )

    tools = tuple(await mcp_client.list_tools())
    tools_by_name = {item.name: item for item in tools}
    _record(active_state, "mcp_tools_listed", tools=list(tools_by_name))

    if active_state.plan is None:
        active_state.plan = planner.create_plan(goal, skill_catalog, tools)
        _validate_plan(active_state.plan, skills_by_name, tools_by_name)
        _record(
            active_state,
            "plan_created",
            plan_id=active_state.plan.plan_id,
            version=active_state.plan.version,
            steps=[step.step_id for step in active_state.plan.steps],
        )
    else:
        _validate_plan(active_state.plan, skills_by_name, tools_by_name)
        _record(
            active_state,
            "run_resumed",
            plan_id=active_state.plan.plan_id,
            version=active_state.plan.version,
            current_step_index=active_state.plan.current_step_index,
        )

    while active_state.plan.status is PlanStatus.ACTIVE:
        plan = active_state.plan
        if plan.current_step_index >= len(plan.steps):
            plan.status = PlanStatus.COMPLETED
            _record(active_state, "plan_completed", version=plan.version)
            break

        step = plan.steps[plan.current_step_index]
        if step.status is StepStatus.COMPLETED:
            plan.current_step_index += 1
            continue

        completed = {item.step_id for item in plan.steps if item.status is StepStatus.COMPLETED}
        if not set(step.dependencies).issubset(completed):
            plan.status = PlanStatus.FAILED
            active_state.stop_reason = "UNMET_DEPENDENCY"
            _record(active_state, "plan_stopped", reason=active_state.stop_reason)
            break

        step.status = StepStatus.IN_PROGRESS
        step.attempts += 1
        _record(
            active_state,
            "step_started",
            step_id=step.step_id,
            objective=step.objective,
            attempt=step.attempts,
        )

        needs_replan_reason: str | None = None
        step_finished = False
        tool_call_count = 0
        tool_fingerprint_counts: dict[str, int] = {}
        for _ in range(max_actions_per_step):
            action = await executor.decide(
                plan,
                step,
                active_state,
                skill_catalog,
                tools,
            )
            _record(
                active_state,
                "executor_decision",
                step_id=step.step_id,
                action=type(action).__name__,
            )

            if isinstance(action, ActivateSkillAction):
                if action.skill_name not in step.required_skills:
                    plan.status = PlanStatus.FAILED
                    active_state.stop_reason = "SKILL_NOT_ALLOWED_FOR_STEP"
                    break
                activated = skill_registry.activate(action.skill_name)
                active_state.activated_skills[action.skill_name] = activated
                _record(
                    active_state,
                    "skill_activated",
                    step_id=step.step_id,
                    skill_name=action.skill_name,
                )
                continue

            if isinstance(action, LoadSkillResourceAction):
                resource_id = f"{action.skill_name}:{action.relative_path}"
                if resource_id not in step.required_resources:
                    plan.status = PlanStatus.FAILED
                    active_state.stop_reason = "SKILL_RESOURCE_NOT_ALLOWED_FOR_STEP"
                    break
                content = skill_registry.read_resource(
                    action.skill_name,
                    action.relative_path,
                )
                evidence = Evidence(
                    action.evidence_key,
                    content,
                    source=f"skill:{resource_id}",
                )
                if not _evidence_matches_step_contract(step, evidence):
                    plan.status = PlanStatus.FAILED
                    active_state.stop_reason = "EVIDENCE_CONTRACT_MISMATCH"
                    _record(
                        active_state,
                        "runtime_validation",
                        accepted=False,
                        error=active_state.stop_reason,
                        evidence_key=action.evidence_key,
                        source=evidence.source,
                    )
                    break
                active_state.loaded_resources[resource_id] = content
                active_state.evidence[action.evidence_key] = evidence
                _record(
                    active_state,
                    "skill_resource_loaded",
                    step_id=step.step_id,
                    resource=resource_id,
                    evidence_key=action.evidence_key,
                )
                continue

            if isinstance(action, ToolAction):
                if action.tool_name not in step.allowed_tools:
                    plan.status = PlanStatus.FAILED
                    active_state.stop_reason = "TOOL_NOT_ALLOWED_FOR_STEP"
                    break
                if tool_call_count >= max_tool_calls_per_step:
                    step.status = StepStatus.FAILED
                    plan.status = PlanStatus.FAILED
                    active_state.stop_reason = "TOOL_CALL_LIMIT"
                    _record(
                        active_state,
                        "plan_stopped",
                        reason=active_state.stop_reason,
                        step_id=step.step_id,
                        tool_call_count=tool_call_count,
                    )
                    break
                spec = tools_by_name[action.tool_name]
                validation_error = spec.validator(action.arguments)
                if validation_error:
                    plan.status = PlanStatus.FAILED
                    active_state.stop_reason = "INVALID_TOOL_ARGUMENTS"
                    _record(
                        active_state,
                        "runtime_validation",
                        accepted=False,
                        error=validation_error,
                    )
                    break
                if spec.has_side_effect and spec.requires_confirmation:
                    confirmation = action.arguments.get("confirmation_id")
                    if not isinstance(confirmation, str) or not confirmation:
                        step.status = StepStatus.BLOCKED
                        plan.status = PlanStatus.NEEDS_HUMAN
                        active_state.stop_reason = "CONFIRMATION_REQUIRED"
                        _record(
                            active_state,
                            "plan_stopped",
                            reason=active_state.stop_reason,
                            step_id=step.step_id,
                        )
                        break
                dispatch_record: DispatchRecord | None = None
                if spec.has_side_effect:
                    if not spec.idempotency_field:
                        plan.status = PlanStatus.FAILED
                        active_state.stop_reason = "IDEMPOTENCY_POLICY_MISSING"
                        break
                    idempotency_value = action.arguments.get(spec.idempotency_field)
                    if not isinstance(idempotency_value, str) or not idempotency_value:
                        plan.status = PlanStatus.FAILED
                        active_state.stop_reason = "IDEMPOTENCY_KEY_REQUIRED"
                        break
                    existing_dispatch = active_state.dispatches.get(idempotency_value)
                    if existing_dispatch is not None:
                        step.status = StepStatus.BLOCKED
                        plan.status = PlanStatus.NEEDS_HUMAN
                        active_state.stop_reason = "RESULT_UNKNOWN"
                        _record(
                            active_state,
                            "duplicate_dispatch_blocked",
                            step_id=step.step_id,
                            tool_name=action.tool_name,
                            idempotency_key=idempotency_value,
                            previous_status=existing_dispatch.status.value,
                        )
                        break
                    dispatch_record = DispatchRecord(
                        idempotency_key=idempotency_value,
                        plan_id=plan.plan_id,
                        plan_version=plan.version,
                        step_id=step.step_id,
                        tool_name=action.tool_name,
                        arguments=dict(action.arguments),
                        evidence_key=action.evidence_key,
                    )
                    active_state.dispatches[idempotency_value] = dispatch_record
                    _record(
                        active_state,
                        "dispatch_intent_recorded",
                        step_id=step.step_id,
                        tool_name=action.tool_name,
                        idempotency_key=idempotency_value,
                    )
                fingerprint = _tool_fingerprint(action)
                repeated_count = tool_fingerprint_counts.get(fingerprint, 0)
                if not spec.allows_polling and repeated_count >= 2:
                    step.status = StepStatus.FAILED
                    plan.status = PlanStatus.FAILED
                    active_state.stop_reason = "REPEATED_ACTION"
                    _record(
                        active_state,
                        "plan_stopped",
                        reason=active_state.stop_reason,
                        step_id=step.step_id,
                        tool_name=action.tool_name,
                        repeated_count=repeated_count,
                    )
                    break
                _record(
                    active_state,
                    "mcp_tool_called",
                    step_id=step.step_id,
                    tool_name=action.tool_name,
                )
                tool_call_count += 1
                tool_fingerprint_counts[fingerprint] = repeated_count + 1
                try:
                    result = await mcp_client.call_tool(
                        action.tool_name,
                        action.arguments,
                    )
                except (ConnectionError, TimeoutError) as exc:
                    if dispatch_record is None:
                        raise
                    dispatch_record.status = DispatchStatus.RESULT_UNKNOWN
                    step.status = StepStatus.BLOCKED
                    plan.status = PlanStatus.NEEDS_HUMAN
                    active_state.stop_reason = "RESULT_UNKNOWN"
                    _record(
                        active_state,
                        "result_unknown",
                        step_id=step.step_id,
                        tool_name=action.tool_name,
                        idempotency_key=dispatch_record.idempotency_key,
                        error=type(exc).__name__,
                    )
                    break
                _record(
                    active_state,
                    "observation",
                    step_id=step.step_id,
                    tool_name=action.tool_name,
                    ok=result.ok,
                    error_code=result.error_code,
                )
                if not result.ok:
                    if dispatch_record is not None:
                        dispatch_record.status = DispatchStatus.FAILED
                    if result.retryable:
                        continue
                    step.status = StepStatus.FAILED
                    plan.status = PlanStatus.FAILED
                    active_state.stop_reason = result.error_code or "TOOL_FAILED"
                    break
                if dispatch_record is not None:
                    dispatch_record.status = DispatchStatus.SUCCEEDED
                result_data = result.data or {}
                version_value = result_data.get("version")
                subject_value = result_data.get("job_id")
                evidence = Evidence(
                    action.evidence_key,
                    result_data,
                    source=f"mcp:{action.tool_name}",
                    version=None if version_value is None else str(version_value),
                    subject_id=None if subject_value is None else str(subject_value),
                )
                if not _evidence_matches_step_contract(step, evidence):
                    plan.status = PlanStatus.FAILED
                    active_state.stop_reason = "EVIDENCE_CONTRACT_MISMATCH"
                    _record(
                        active_state,
                        "runtime_validation",
                        accepted=False,
                        error=active_state.stop_reason,
                        evidence_key=action.evidence_key,
                        source=evidence.source,
                    )
                    break
                active_state.evidence[action.evidence_key] = evidence
                if result.replan_reason:
                    step.status = StepStatus.INVALIDATED
                    plan.status = PlanStatus.NEEDS_REPLAN
                    needs_replan_reason = result.replan_reason
                    _record(
                        active_state,
                        "step_invalidated",
                        step_id=step.step_id,
                        reason=result.replan_reason,
                    )
                    break
                continue

            decision = verifier.verify_step(plan, step, active_state, action)
            _record(
                active_state,
                "step_verified",
                step_id=step.step_id,
                decision=decision.value,
            )
            if decision is PlanDecision.COMPLETE_STEP:
                step.status = StepStatus.COMPLETED
                step.summary = action.summary
                plan.current_step_index += 1
                step_finished = True
                break
            if decision is PlanDecision.RETRY_STEP:
                continue
            if decision is PlanDecision.REPLAN:
                step.status = StepStatus.INVALIDATED
                plan.status = PlanStatus.NEEDS_REPLAN
                needs_replan_reason = "Verifier 判定原计划失效"
                break
            if decision is PlanDecision.NEEDS_HUMAN:
                step.status = StepStatus.BLOCKED
                plan.status = PlanStatus.NEEDS_HUMAN
                active_state.stop_reason = "VERIFIER_NEEDS_HUMAN"
                break
            step.status = StepStatus.FAILED
            plan.status = PlanStatus.FAILED
            active_state.stop_reason = "VERIFIER_FAILED"
            break

        if plan.status is PlanStatus.NEEDS_REPLAN:
            if plan.replan_count >= max_replans:
                plan.status = PlanStatus.FAILED
                active_state.stop_reason = "REPLAN_LIMIT"
                _record(active_state, "plan_stopped", reason=active_state.stop_reason)
                break
            previous_version = plan.version
            try:
                new_plan = planner.replan(
                    plan,
                    active_state,
                    needs_replan_reason or "计划失效",
                )
            except RuntimeError:
                plan.status = PlanStatus.FAILED
                active_state.stop_reason = "REPLAN_UNAVAILABLE"
                _record(active_state, "plan_stopped", reason=active_state.stop_reason)
                break
            # 不信任 Planner 返回的审计字段，防止版本回退或重置 Replan 预算。
            new_plan.plan_id = plan.plan_id
            new_plan.goal = active_state.goal
            new_plan.version = plan.version + 1
            new_plan.replan_count = plan.replan_count + 1
            new_plan.status = PlanStatus.ACTIVE
            _validate_plan(new_plan, skills_by_name, tools_by_name)
            _revalidate_evidence_for_plan(active_state, new_plan)
            active_state.plan = new_plan
            _record(
                active_state,
                "plan_revised",
                previous_version=previous_version,
                new_version=new_plan.version,
                reason=needs_replan_reason,
            )
            continue

        if plan.status is not PlanStatus.ACTIVE:
            break
        if not step_finished:
            step.status = StepStatus.FAILED
            plan.status = PlanStatus.FAILED
            active_state.stop_reason = "STEP_ACTION_LIMIT"
            _record(
                active_state,
                "plan_stopped",
                reason=active_state.stop_reason,
                step_id=step.step_id,
            )
            break

    if active_state.plan.status is PlanStatus.COMPLETED:
        active_state.final_answer = final_writer.write(active_state)
        _record(active_state, "final_answer", answer=active_state.final_answer)
    return active_state

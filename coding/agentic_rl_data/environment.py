"""可复现的最小 Agent RL 环境和轨迹采集器。

真实系统会把 Tool Bridge、权限系统、Verifier 和外部服务接到环境里。这里使用固定的
训练任务状态与日志，只为展示 Agent Action 如何经过 Runtime 变成 RL transition。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Protocol

from .contracts import (
    Action,
    ActionKind,
    Decision,
    Observation,
    RewardComponents,
    StepRecord,
    Trajectory,
)


class BehaviorPolicy(Protocol):
    def decide(self, observation: Observation) -> Decision:
        """根据模型可见 observation 采样一个 Action，并保留 behavior 信息。"""


def _fake_tokenize(text: str) -> tuple[int, ...]:
    """教学 tokenizer：按 Unicode 字符编码，不代表真实模型 tokenizer。"""

    return tuple(ord(char) for char in text)


def _serialize_action(action: Action) -> str:
    return json.dumps(
        {"kind": action.kind.value, "name": action.name, "arguments": action.arguments},
        ensure_ascii=False,
        sort_keys=True,
    )


@dataclass
class ScriptedPolicy:
    """用动作脚本代替真实 LLM，确保教学运行可复现。

    每个 response token 的 old_logprob 使用固定值。真实接入时必须保存模型采样当刻的
    token 级 log-prob，不能根据动作字符串伪造或用新 policy 重算后覆盖。
    """

    actions: Iterable[Action]
    policy_version: int = 10
    value_estimates: Iterable[float] = ()

    def __post_init__(self) -> None:
        self._actions = iter(self.actions)
        self._values = iter(self.value_estimates)

    def decide(self, observation: Observation) -> Decision:
        action = next(self._actions)
        response_text = _serialize_action(action)
        response_ids = _fake_tokenize(response_text)
        return Decision(
            action=action,
            policy_version=self.policy_version,
            prompt_token_ids=_fake_tokenize(observation.to_prompt()),
            response_token_ids=response_ids,
            old_logprobs=tuple(-0.1 for _ in response_ids),
            value=next(self._values, 0.0),
        )


class AgentRLEnvironment:
    """训练任务诊断环境。

    任务要求先查询状态和日志，再给出包含 OOM 的最终诊断。环境负责工具白名单、证据
    准入、停止条件和奖励；Policy 只提出候选动作。
    """

    VERSION = "training-diagnosis-v1"
    ALLOWED_TOOLS = ("get_training_job", "get_training_logs")

    def __init__(self, *, max_steps: int = 5) -> None:
        if max_steps < 1:
            raise ValueError("max_steps 必须大于 0")
        self.max_steps = max_steps
        self.step_count = 0
        self.history: list[str] = []
        self.evidence: set[str] = set()
        self.done = False
        self.success = False

    def reset(self) -> Observation:
        self.step_count = 0
        self.history = []
        self.evidence = set()
        self.done = False
        self.success = False
        return self._observation()

    def _observation(self) -> Observation:
        return Observation(
            goal="诊断训练任务 train-42 的失败原因，并给出有证据的结论",
            allowed_tools=self.ALLOWED_TOOLS,
            history=tuple(self.history),
            steps_remaining=max(0, self.max_steps - self.step_count),
        )

    def step(
        self, action: Action
    ) -> tuple[Observation, RewardComponents, bool, bool, str | None, bool]:
        """执行一次环境迁移。

        返回：next_observation、奖励分量、terminated、truncated、停止原因、动作是否准入。
        `terminated` 表示任务语义结束；`truncated` 表示被采集上限截断。
        """

        if self.done:
            raise RuntimeError("环境已结束，请先 reset")

        self.step_count += 1
        accepted = True
        terminated = False
        stop_reason: str | None = None
        reward = RewardComponents(efficiency=-0.02)

        if action.kind is ActionKind.TOOL:
            if action.name not in self.ALLOWED_TOOLS:
                accepted = False
                self.history.append(f"Runtime 拒绝未知工具：{action.name}")
                reward = RewardComponents(safety=-1.0, efficiency=-0.02)
            elif action.arguments.get("job_id") != "train-42":
                accepted = False
                self.history.append("Runtime 拒绝：job_id 不符合当前任务主体")
                reward = RewardComponents(safety=-0.6, efficiency=-0.02)
            elif action.name == "get_training_job":
                is_new_evidence = "job_status" not in self.evidence
                self.evidence.add("job_status")
                self.history.append("Tool Observation：train-42 状态为 FAILED")
                # 同一证据只在首次获得时加分，避免模型靠重复查询刷 reward。
                reward = RewardComponents(
                    evidence=0.1 if is_new_evidence else 0.0,
                    efficiency=-0.02 if is_new_evidence else -0.05,
                )
            else:
                is_new_evidence = "job_logs" not in self.evidence
                self.evidence.add("job_logs")
                self.history.append("Tool Observation：日志包含 CUDA out of memory")
                reward = RewardComponents(
                    evidence=0.2 if is_new_evidence else 0.0,
                    efficiency=-0.02 if is_new_evidence else -0.05,
                )

        elif action.kind is ActionKind.FINAL:
            answer = str(action.arguments.get("answer", ""))
            evidence_complete = {"job_status", "job_logs"}.issubset(self.evidence)
            answer_correct = "OOM" in answer or "out of memory" in answer.lower()
            if evidence_complete and answer_correct:
                self.success = True
                terminated = True
                stop_reason = "COMPLETED"
                self.history.append("Verifier：结论正确且证据完整")
                reward = RewardComponents(
                    task=1.0, evidence=0.5, efficiency=-0.02
                )
            else:
                accepted = False
                missing = sorted({"job_status", "job_logs"} - self.evidence)
                self.history.append(
                    f"Verifier 拒绝提前完成：missing={missing}, correct={answer_correct}"
                )
                reward = RewardComponents(task=-0.3, evidence=-0.2, efficiency=-0.02)
        else:  # pragma: no cover - Enum 已限制正常入口
            accepted = False
            self.history.append("Runtime 拒绝未知 Action 类型")
            reward = RewardComponents(safety=-1.0, efficiency=-0.02)

        # max_steps 是采集/运行预算，不代表任务在语义上自然结束。
        truncated = not terminated and self.step_count >= self.max_steps
        if truncated:
            stop_reason = "MAX_STEPS"
            # 只给局部证据分会导致“刷证据但不完成任务”的 reward hacking。
            # 截断惩罚明确表示任务合同未完成，同时保留本步其他奖励分量便于审计。
            reward = RewardComponents(
                task=reward.task - 0.5,
                evidence=reward.evidence,
                safety=reward.safety,
                efficiency=reward.efficiency,
            )

        self.done = terminated or truncated
        return self._observation(), reward, terminated, truncated, stop_reason, accepted


def collect_trajectory(
    *,
    policy: BehaviorPolicy,
    environment: AgentRLEnvironment,
    trajectory_id: str,
    task_id: str = "diagnose-train-42",
    group_id: str = "prompt-group-1",
) -> Trajectory:
    """运行一次 episode，只在 Policy 真正决策的位置创建 StepRecord。"""

    observation = environment.reset()
    trajectory = Trajectory(
        trajectory_id=trajectory_id,
        task_id=task_id,
        group_id=group_id,
        environment_version=environment.VERSION,
    )

    while not environment.done:
        decision = policy.decide(observation)
        next_observation, reward, terminated, truncated, reason, accepted = (
            environment.step(decision.action)
        )
        trajectory.steps.append(
            StepRecord(
                step_id=len(trajectory.steps),
                observation=observation,
                decision=decision,
                reward=reward,
                next_observation=next_observation,
                action_accepted=accepted,
                terminated=terminated,
                truncated=truncated,
                stop_reason=reason,
            )
        )
        observation = next_observation

    trajectory.success = environment.success
    return trajectory

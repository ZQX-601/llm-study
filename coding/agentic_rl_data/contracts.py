"""Agentic RL 最小数据合同。

这里不依赖 PyTorch 或具体模型。重点是先把一次 Agent 决策的边界固定下来：

    observation -> policy decision -> runtime step -> next observation

生产系统中的 observation 通常会被 chat template 和 tokenizer 转成 token；本教学实现
让 Policy 同时返回 prompt/response token id，借此展示训练样本所需的 mask 与 log-prob。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionKind(str, Enum):
    TOOL = "tool"
    FINAL = "final"


@dataclass(frozen=True)
class Action:
    """LLM Policy 提出的候选动作；Runtime 仍有权拒绝。"""

    kind: ActionKind
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    """某次决策前模型真正可见的内容，而不是 Runtime 的全部内部状态。"""

    goal: str
    allowed_tools: tuple[str, ...]
    history: tuple[str, ...]
    steps_remaining: int

    def to_prompt(self) -> str:
        history_text = "\n".join(self.history) if self.history else "（暂无观察）"
        return (
            f"目标：{self.goal}\n"
            f"允许工具：{', '.join(self.allowed_tools)}\n"
            f"剩余步数：{self.steps_remaining}\n"
            f"历史：\n{history_text}\n"
            "请输出下一步结构化动作。"
        )


@dataclass(frozen=True)
class Decision:
    """Policy 输出及行为策略信息。

    old_logprobs 只能来自生成 response_token_ids 的 behavior policy，不能事后用
    trainer 的新 policy 覆盖。value 是决策时 critic 的估值；GRPO 可不使用它。
    """

    action: Action
    policy_version: int
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    old_logprobs: tuple[float, ...]
    value: float = 0.0


@dataclass(frozen=True)
class RewardComponents:
    """保留奖励分量，避免只有总分却无法解释奖励来源。"""

    task: float = 0.0
    evidence: float = 0.0
    safety: float = 0.0
    efficiency: float = 0.0

    @property
    def total(self) -> float:
        return self.task + self.evidence + self.safety + self.efficiency


@dataclass(frozen=True)
class StepRecord:
    """一个 RL transition，对应一次真实的模型决策。"""

    step_id: int
    observation: Observation
    decision: Decision
    reward: RewardComponents
    next_observation: Observation
    action_accepted: bool
    terminated: bool
    truncated: bool
    stop_reason: str | None


@dataclass
class Trajectory:
    """同一个任务从 reset 到结束/截断的有序决策序列。"""

    trajectory_id: str
    task_id: str
    group_id: str
    environment_version: str
    # 这些版本共同描述“Policy 当时究竟看到了什么、调用了什么、按什么规则得分”。
    # 不能只记录 environment_version，否则 Reward 或 Tool schema 变化后无法判断旧数据
    # 应该重新评分、重新采集，还是可以直接进入训练。
    task_version: str = "diagnose-task-v1"
    observation_schema_version: str = "observation-v1"
    tool_schema_version: str = "training-tools-v1"
    reward_version: str = "training-reward-v1"
    verifier_version: str = "diagnosis-verifier-v1"
    data_snapshot: str = "static-diagnosis-data-v1"
    tokenizer_version: str = "unicode-teaching-v1"
    chat_template_version: str = "diagnosis-prompt-v1"
    steps: list[StepRecord] = field(default_factory=list)
    success: bool = False

    @property
    def total_reward(self) -> float:
        return sum(step.reward.total for step in self.steps)

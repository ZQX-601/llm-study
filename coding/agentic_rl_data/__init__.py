"""Agent 决策轨迹到 RL 训练样本的教学实现。"""

from .contracts import (
    Action,
    ActionKind,
    Decision,
    Observation,
    RewardComponents,
    StepRecord,
    Trajectory,
)
from .environment import AgentRLEnvironment, ScriptedPolicy, collect_trajectory
from .sample_builder import (
    GRPOSample,
    PolicyDecisionSample,
    PPOSample,
    SFTSample,
    build_grpo_group,
    build_ppo_samples,
    build_sft_samples,
    collate_grpo_batch,
    collate_ppo_batch,
    trajectory_equal_masked_mean,
    validate_trajectory,
)

__all__ = [
    "Action",
    "ActionKind",
    "AgentRLEnvironment",
    "Decision",
    "GRPOSample",
    "Observation",
    "PolicyDecisionSample",
    "PPOSample",
    "RewardComponents",
    "SFTSample",
    "ScriptedPolicy",
    "StepRecord",
    "Trajectory",
    "build_grpo_group",
    "build_ppo_samples",
    "build_sft_samples",
    "collate_grpo_batch",
    "collate_ppo_batch",
    "trajectory_equal_masked_mean",
    "collect_trajectory",
    "validate_trajectory",
]

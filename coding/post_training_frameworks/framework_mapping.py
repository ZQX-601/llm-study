"""把 GRPO 语义合同映射到 TRL、vLLM 与 verl 的教学实现。

本文件不导入任何真实框架。它先固定跨框架不应改变的语义：

    Prompt → Rollout → Reward → Group Advantage
    → Policy Forward → Loss → Backward/Optimizer → Publish

框架的类名和配置可能随版本变化，但这些阶段的输入、输出和因果关系不能被省略。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class PipelineStage(str, Enum):
    """一轮 GRPO 更新中不可混淆的阶段。"""

    PROMPT = "prompt"
    ROLLOUT = "rollout"
    REWARD = "reward"
    ADVANTAGE = "advantage"
    POLICY_FORWARD = "policy_forward"
    LOSS = "loss"
    OPTIMIZER = "optimizer"
    PUBLISH = "publish"


class TrainingProgress(str, Enum):
    """当前数据最多能证明训练走到了哪里。"""

    EMPTY = "empty"
    PROMPTS_READY = "prompts_ready"
    ROLLOUT_ONLY = "rollout_only"
    REWARDS_READY = "rewards_ready"
    ADVANTAGES_READY = "advantages_ready"
    LOSS_READY = "loss_ready"
    UPDATED = "updated"
    PUBLISHED = "published"


@dataclass(frozen=True)
class FrameworkComponent:
    """框架组件及其职责；并非声称一个组件只能参与一个阶段。"""

    framework: str
    component: str
    stages: tuple[PipelineStage, ...]
    responsibility: str


FRAMEWORK_COMPONENTS = (
    FrameworkComponent(
        framework="vLLM",
        component="LLM Engine / Rollout Engine",
        stages=(PipelineStage.ROLLOUT,),
        responsibility="加载行为策略并批量生成 completion；管理调度、KV cache 与采样。",
    ),
    FrameworkComponent(
        framework="TRL",
        component="GRPOTrainer",
        stages=(
            PipelineStage.PROMPT,
            PipelineStage.ROLLOUT,
            PipelineStage.REWARD,
            PipelineStage.ADVANTAGE,
            PipelineStage.POLICY_FORWARD,
            PipelineStage.LOSS,
            PipelineStage.OPTIMIZER,
        ),
        responsibility="在轻量入口中组织生成、奖励、组优势和模型更新。",
    ),
    FrameworkComponent(
        framework="verl",
        component="Trainer / Driver",
        stages=(
            PipelineStage.PROMPT,
            PipelineStage.REWARD,
            PipelineStage.ADVANTAGE,
            PipelineStage.LOSS,
            PipelineStage.PUBLISH,
        ),
        responsibility="编排数据、Worker、优势计算、训练步骤、日志与 checkpoint。",
    ),
    FrameworkComponent(
        framework="verl",
        component="Actor / Rollout / Reference Workers",
        stages=(
            PipelineStage.ROLLOUT,
            PipelineStage.POLICY_FORWARD,
            PipelineStage.OPTIMIZER,
            PipelineStage.PUBLISH,
        ),
        responsibility="分别执行生成、current/reference log-prob、反向更新和权重同步。",
    ),
    FrameworkComponent(
        framework="FSDP / DeepSpeed",
        component="Distributed Training Backend",
        stages=(PipelineStage.POLICY_FORWARD, PipelineStage.OPTIMIZER),
        responsibility="分片参数、梯度与优化器状态；不负责定义 GRPO Reward。",
    ),
    FrameworkComponent(
        framework="Ray",
        component="Resource / Worker Orchestration",
        stages=(
            PipelineStage.ROLLOUT,
            PipelineStage.POLICY_FORWARD,
            PipelineStage.OPTIMIZER,
            PipelineStage.PUBLISH,
        ),
        responsibility="放置和调用 Worker、分配 CPU/GPU；不替代算法 loss。",
    ),
)


@dataclass(frozen=True)
class GRPOBatchState:
    """一个 Prompt 组在各阶段留下的最小证据。

    shape 约定：
    - `completion_token_ids`、`old_logprobs`、`response_mask`：`[G, T]`；
    - `rewards`、`group_advantages`：`[G]`；
    - `new_logprobs`：Learner forward 后的 `[G, T]`。

    教学代码使用 ragged tuple，因此每一行的 `T` 可以不同；同一 completion 的三行长度
    必须一致。生产框架通常会 padding/packing，并额外保存 position ids 等字段。
    """

    prompt_token_ids: tuple[int, ...] = ()
    completion_token_ids: tuple[tuple[int, ...], ...] = ()
    old_logprobs: tuple[tuple[float, ...], ...] = ()
    response_mask: tuple[tuple[int, ...], ...] = ()
    behavior_policy_version: str | None = None
    rewards: tuple[float, ...] = ()
    group_advantages: tuple[float, ...] = ()
    # Reference 只在 KL reward 或 KL loss 启用时是必需字段；关闭 KL 时不能强制要求它。
    ref_logprobs: tuple[tuple[float, ...], ...] = ()
    use_kl_in_reward: bool = False
    use_kl_loss: bool = False
    new_logprobs: tuple[tuple[float, ...], ...] = ()
    policy_loss: float | None = None
    gradients_finite: bool | None = None
    optimizer_step_applied: bool = False
    published_policy_version: str | None = None


@dataclass(frozen=True)
class ProgressAssessment:
    progress: TrainingProgress
    missing_for_next_stage: tuple[str, ...]


@dataclass(frozen=True)
class VerlStepConfig:
    """抽象当前 verl 同步训练步骤中的条件分支。

    这里保留稳定语义，不绑定某个具体版本的私有函数签名。
    """

    algorithm: str
    use_reference_policy: bool
    use_critic: bool


def build_verl_sync_step_plan(config: VerlStepConfig) -> tuple[str, ...]:
    """按真实 V1 Trainer 主链构建教学化同步执行计划。"""

    if config.algorithm == "ppo" and not config.use_critic:
        raise ValueError("PPO 计划必须启用 Critic")
    if config.algorithm == "grpo" and config.use_critic:
        raise ValueError("本教学合同中的 GRPO 不应启用 Critic")

    steps = [
        "generate_rollout",
        "compute_reward",
        "balance_batch",
        "compute_old_logprobs",
    ]
    if config.use_reference_policy:
        steps.append("compute_ref_logprobs")
    if config.use_critic:
        steps.append("compute_values")
    steps.append("compute_advantages")
    if config.use_critic:
        steps.append("update_critic")
    steps.extend(("update_actor", "publish_weights"))
    return tuple(steps)


def components_for_stage(stage: PipelineStage) -> tuple[FrameworkComponent, ...]:
    """查找参与某阶段的组件，用于沿数据流阅读框架源码。"""

    return tuple(component for component in FRAMEWORK_COMPONENTS if stage in component.stages)


def validate_rollout_contract(state: GRPOBatchState) -> None:
    """验证 rollout 证据能够与 behavior policy 一一对应。"""

    group_size = len(state.completion_token_ids)
    if group_size == 0:
        raise ValueError("GRPO rollout 至少需要一条 completion")
    if not state.behavior_policy_version:
        raise ValueError("completion 必须记录 behavior_policy_version")
    if len(state.old_logprobs) != group_size or len(state.response_mask) != group_size:
        raise ValueError("每条 completion 都必须有 old_logprobs 和 response_mask")

    for index, (tokens, old_logprobs, mask) in enumerate(
        zip(state.completion_token_ids, state.old_logprobs, state.response_mask)
    ):
        if not (len(tokens) == len(old_logprobs) == len(mask)):
            raise ValueError(f"completion {index} 的 token/logprob/mask 长度不一致")
        if any(value not in (0, 1) for value in mask):
            raise ValueError(f"completion {index} 的 response_mask 只能包含 0/1")


def assess_training_progress(state: GRPOBatchState) -> ProgressAssessment:
    """返回当前证据最多能证明的进度，不用后续字段反推前序阶段。"""

    if not state.prompt_token_ids:
        return ProgressAssessment(TrainingProgress.EMPTY, ("prompt_token_ids",))
    if not state.completion_token_ids:
        return ProgressAssessment(
            TrainingProgress.PROMPTS_READY,
            ("completion_token_ids", "old_logprobs", "response_mask", "behavior_policy_version"),
        )

    validate_rollout_contract(state)
    group_size = len(state.completion_token_ids)
    if len(state.rewards) != group_size:
        return ProgressAssessment(TrainingProgress.ROLLOUT_ONLY, ("rewards[G]",))
    if len(state.group_advantages) != group_size:
        return ProgressAssessment(TrainingProgress.REWARDS_READY, ("group_advantages[G]",))
    missing_for_loss: list[str] = []
    if (state.use_kl_in_reward or state.use_kl_loss) and len(state.ref_logprobs) != group_size:
        missing_for_loss.append("ref_logprobs[G,T]")
    if len(state.new_logprobs) != group_size:
        missing_for_loss.append("new_logprobs[G,T]")
    if state.policy_loss is None:
        missing_for_loss.append("policy_loss")
    elif not math.isfinite(state.policy_loss):
        missing_for_loss.append("finite_policy_loss")
    if missing_for_loss:
        return ProgressAssessment(
            TrainingProgress.ADVANTAGES_READY,
            tuple(missing_for_loss),
        )
    if state.gradients_finite is not True or not state.optimizer_step_applied:
        return ProgressAssessment(
            TrainingProgress.LOSS_READY,
            ("finite_gradients", "optimizer_step"),
        )
    if not state.published_policy_version:
        return ProgressAssessment(
            TrainingProgress.UPDATED,
            ("published_policy_version",),
        )
    return ProgressAssessment(TrainingProgress.PUBLISHED, ())


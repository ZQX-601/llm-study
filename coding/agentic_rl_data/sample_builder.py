"""把 Agent trajectory 转成 SFT、PPO 和 GRPO 教学样本。

本文件只完成数据处理，不实现神经网络 forward/backward。生产训练还需真实 tokenizer、
chat template、模型生成、分布式 padding、KL/reference log-prob 和训练框架适配。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .contracts import StepRecord, Trajectory

IGNORE_INDEX = -100


@dataclass(frozen=True)
class SFTSample:
    """监督微调样本：prompt 不计 loss，只监督被准入的正确 Action token。"""

    trajectory_id: str
    step_id: int
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]


@dataclass(frozen=True)
class PolicyDecisionSample:
    """PPO/GRPO 共用的 Policy 决策 token 数据，不包含算法专属 Advantage。"""

    trajectory_id: str
    step_id: int
    policy_version: int
    input_ids: tuple[int, ...]
    response_mask: tuple[int, ...]
    old_logprobs: tuple[float, ...]


@dataclass(frozen=True)
class PPOSample(PolicyDecisionSample):
    """PPO 决策样本：公共 token 数据 + GAE Advantage + Critic target。"""

    advantage: float
    return_target: float


@dataclass(frozen=True)
class GRPOSample:
    """GRPO 的 completion/trajectory 级样本。"""

    trajectory_id: str
    group_id: str
    policy_version: int
    total_reward: float
    group_advantage: float
    decisions: tuple[PolicyDecisionSample, ...]


def _build_policy_decision_sample(
    step: StepRecord, *, trajectory_id: str
) -> PolicyDecisionSample:
    """提取 PPO/GRPO 都需要的 token、mask 和 behavior policy 信息。"""

    prompt_ids = step.decision.prompt_token_ids
    response_ids = step.decision.response_token_ids
    return PolicyDecisionSample(
        trajectory_id=trajectory_id,
        step_id=step.step_id,
        policy_version=step.decision.policy_version,
        input_ids=prompt_ids + response_ids,
        response_mask=(0,) * len(prompt_ids) + (1,) * len(response_ids),
        old_logprobs=(0.0,) * len(prompt_ids) + step.decision.old_logprobs,
    )


def validate_trajectory(
    trajectory: Trajectory,
    *,
    current_policy_version: int | None = None,
    max_policy_lag: int | None = None,
) -> None:
    """在训练前拒绝结构损坏或过度陈旧的轨迹。"""

    if not trajectory.steps:
        raise ValueError("空轨迹不能训练")
    if not trajectory.environment_version:
        raise ValueError("缺少 environment_version，无法保证任务语义一致")

    final = trajectory.steps[-1]
    if not (final.terminated or final.truncated):
        raise ValueError("轨迹没有明确的 terminated/truncated 边界")

    for expected_step, step in enumerate(trajectory.steps):
        if step.step_id != expected_step:
            raise ValueError("step_id 必须连续且从 0 开始")
        decision = step.decision
        if len(decision.response_token_ids) != len(decision.old_logprobs):
            raise ValueError("每个 response token 必须有对应 old_logprob")
        if not decision.response_token_ids:
            raise ValueError("空 response 不能形成策略梯度样本")
        if not all(math.isfinite(value) for value in decision.old_logprobs):
            raise ValueError("old_logprobs 包含非有限值")

        if current_policy_version is not None and max_policy_lag is not None:
            lag = current_policy_version - decision.policy_version
            if lag < 0:
                raise ValueError("轨迹 policy_version 不能领先于 trainer")
            if lag > max_policy_lag:
                raise ValueError(
                    f"轨迹过期：policy lag={lag} > max_policy_lag={max_policy_lag}"
                )


def build_sft_samples(trajectory: Trajectory) -> list[SFTSample]:
    """只从成功轨迹抽取被 Runtime/Verifier 准入的 Action。

    被拒绝动作可以作为偏好负例，但不能当成 SFT 正标签。
    """

    validate_trajectory(trajectory)
    if not trajectory.success:
        return []

    samples: list[SFTSample] = []
    for step in trajectory.steps:
        if not step.action_accepted:
            continue
        prompt_ids = step.decision.prompt_token_ids
        response_ids = step.decision.response_token_ids
        samples.append(
            SFTSample(
                trajectory_id=trajectory.trajectory_id,
                step_id=step.step_id,
                input_ids=prompt_ids + response_ids,
                labels=(IGNORE_INDEX,) * len(prompt_ids) + response_ids,
            )
        )
    return samples


def _gae(
    steps: Sequence[StepRecord],
    *,
    gamma: float,
    gae_lambda: float,
    bootstrap_value: float,
) -> tuple[list[float], list[float]]:
    """按决策步计算 GAE 与 return target。

    terminated 后未来价值严格为 0；truncated 的最后一步可使用 critic 对截断状态的
    bootstrap_value。截断只是样本边界，不能误当成任务自然结束。
    """

    advantages = [0.0] * len(steps)
    returns = [0.0] * len(steps)
    next_advantage = 0.0

    for index in range(len(steps) - 1, -1, -1):
        step = steps[index]
        value = step.decision.value

        if step.terminated:
            next_value = 0.0
        elif index + 1 < len(steps):
            next_value = steps[index + 1].decision.value
        else:
            next_value = bootstrap_value

        delta = step.reward.total + gamma * next_value - value
        # 终止或截断处都停止在已采集序列中递推；截断的未来价值已由 bootstrap 加入 delta。
        continues_in_sample = not step.terminated and not step.truncated
        next_advantage = delta + (
            gamma * gae_lambda * next_advantage if continues_in_sample else 0.0
        )
        advantages[index] = next_advantage
        returns[index] = next_advantage + value

    return advantages, returns


def build_ppo_samples(
    trajectory: Trajectory,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    bootstrap_value: float = 0.0,
    current_policy_version: int | None = None,
    max_policy_lag: int | None = None,
) -> list[PPOSample]:
    """将每次 Agent 决策变成一个 PPO 序列样本。"""

    validate_trajectory(
        trajectory,
        current_policy_version=current_policy_version,
        max_policy_lag=max_policy_lag,
    )
    advantages, returns = _gae(
        trajectory.steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        bootstrap_value=bootstrap_value,
    )

    samples: list[PPOSample] = []
    for step, advantage, return_target in zip(
        trajectory.steps, advantages, returns
    ):
        common = _build_policy_decision_sample(
            step, trajectory_id=trajectory.trajectory_id
        )
        samples.append(
            PPOSample(
                trajectory_id=common.trajectory_id,
                step_id=common.step_id,
                policy_version=common.policy_version,
                input_ids=common.input_ids,
                response_mask=common.response_mask,
                old_logprobs=common.old_logprobs,
                advantage=advantage,
                return_target=return_target,
            )
        )
    return samples


def build_grpo_group(
    trajectories: Sequence[Trajectory],
    *,
    epsilon: float = 1e-8,
) -> list[GRPOSample]:
    """对同一 prompt/task 的多条 rollout 做组内 reward 标准化。

    教学版把整条 Agent trajectory 当作一个 completion。生产实现还需把组优势广播到
    各决策 response token，并结合 new/old log-prob、clip 与 KL 计算 loss。
    """

    if len(trajectories) < 2:
        raise ValueError("GRPO 至少需要同组的两条 rollout")
    group_ids = {trajectory.group_id for trajectory in trajectories}
    task_ids = {trajectory.task_id for trajectory in trajectories}
    env_versions = {trajectory.environment_version for trajectory in trajectories}
    if len(group_ids) != 1 or len(task_ids) != 1 or len(env_versions) != 1:
        raise ValueError("GRPO 组内必须共享 group/task/environment version")
    for trajectory in trajectories:
        validate_trajectory(trajectory)

    rewards = [trajectory.total_reward for trajectory in trajectories]
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    std = math.sqrt(variance)

    result: list[GRPOSample] = []
    for trajectory, reward in zip(trajectories, rewards):
        advantage = 0.0 if std < epsilon else (reward - mean) / (std + epsilon)
        # GRPO 只复用 token/mask/old_logprob，不计算或携带 PPO 的 GAE/return。
        # 这样 Trainer 无法误把 decision-level GAE 当作 group advantage。
        decisions = tuple(
            _build_policy_decision_sample(
                step, trajectory_id=trajectory.trajectory_id
            )
            for step in trajectory.steps
        )
        result.append(
            GRPOSample(
                trajectory_id=trajectory.trajectory_id,
                group_id=trajectory.group_id,
                policy_version=trajectory.steps[0].decision.policy_version,
                total_reward=reward,
                group_advantage=advantage,
                decisions=decisions,
            )
        )
    return result


def collate_grpo_batch(
    samples: Sequence[GRPOSample], *, pad_token_id: int = 0
) -> dict[str, object]:
    """把嵌套的 trajectory/decision/token 展平为模型可前向的二维 batch。

    `trajectory_indices` 保留每个 Decision 行的原始轨迹身份，使 Trainer 能在 token
    forward 后按 trajectory 聚合，而不会把 Decision 行误当成独立 GRPO completion。
    """

    if not samples:
        raise ValueError("不能 collate 空 GRPO batch")

    flattened: list[tuple[int, GRPOSample, PolicyDecisionSample]] = []
    for trajectory_index, sample in enumerate(samples):
        if not sample.decisions:
            raise ValueError("GRPO trajectory 至少包含一个 Policy Decision")
        for decision in sample.decisions:
            if decision.trajectory_id != sample.trajectory_id:
                raise ValueError("Decision 的 trajectory_id 与 GRPOSample 不一致")
            flattened.append((trajectory_index, sample, decision))

    max_length = max(len(decision.input_ids) for _, _, decision in flattened)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    response_mask: list[list[int]] = []
    old_logprobs: list[list[float]] = []
    group_advantages: list[list[float]] = []
    trajectory_indices: list[int] = []
    trajectory_ids: list[str] = []
    group_ids: list[str] = []
    step_ids: list[int] = []

    for trajectory_index, sample, decision in flattened:
        padding = max_length - len(decision.input_ids)
        input_ids.append(list(decision.input_ids) + [pad_token_id] * padding)
        attention_mask.append([1] * len(decision.input_ids) + [0] * padding)
        response_mask.append(list(decision.response_mask) + [0] * padding)
        old_logprobs.append(list(decision.old_logprobs) + [0.0] * padding)
        group_advantages.append(
            [sample.group_advantage * mask for mask in decision.response_mask]
            + [0.0] * padding
        )
        trajectory_indices.append(trajectory_index)
        trajectory_ids.append(sample.trajectory_id)
        group_ids.append(sample.group_id)
        step_ids.append(decision.step_id)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
        "old_logprobs": old_logprobs,
        "group_advantages": group_advantages,
        "trajectory_indices": trajectory_indices,
        "trajectory_ids": trajectory_ids,
        "group_ids": group_ids,
        "step_ids": step_ids,
    }


def trajectory_equal_masked_mean(
    *,
    per_token_values: Sequence[Sequence[float]],
    response_mask: Sequence[Sequence[int]],
    trajectory_indices: Sequence[int],
    num_trajectories: int,
) -> tuple[float, list[float]]:
    """先按 trajectory 聚合有效 token，再让每条 trajectory 等权平均。

    这是纯 Python 教学实现。真实 PyTorch 训练可用 `scatter_add_` 或 segment reduction
    完成同样计算，同时保持梯度流经 `per_token_values`。
    """

    if num_trajectories < 1:
        raise ValueError("num_trajectories 必须大于 0")
    if not (
        len(per_token_values)
        == len(response_mask)
        == len(trajectory_indices)
    ):
        raise ValueError("Decision 行数不一致")

    trajectory_sums = [0.0] * num_trajectories
    trajectory_counts = [0] * num_trajectories

    for values, mask, trajectory_index in zip(
        per_token_values, response_mask, trajectory_indices
    ):
        if not 0 <= trajectory_index < num_trajectories:
            raise ValueError("trajectory_index 越界")
        if len(values) != len(mask):
            raise ValueError("token 维度不一致")

        for value, valid in zip(values, mask):
            if valid not in (0, 1):
                raise ValueError("response_mask 只能包含 0/1")
            if valid == 1:
                trajectory_sums[trajectory_index] += value
                trajectory_counts[trajectory_index] += 1

    if any(count == 0 for count in trajectory_counts):
        raise ValueError("每条 trajectory 至少需要一个有效 Response token")

    trajectory_means = [
        total / count
        for total, count in zip(trajectory_sums, trajectory_counts)
    ]
    objective = sum(trajectory_means) / num_trajectories
    return objective, trajectory_means


def collate_ppo_batch(samples: Sequence[PPOSample], *, pad_token_id: int = 0) -> dict[str, list[list[float | int]]]:
    """右侧 padding，展示训练 batch 的 input/mask/log-prob 对齐关系。"""

    if not samples:
        raise ValueError("不能 collate 空 batch")
    max_length = max(len(sample.input_ids) for sample in samples)

    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    response_mask: list[list[int]] = []
    old_logprobs: list[list[float]] = []
    advantages: list[list[float]] = []
    return_targets: list[list[float]] = []

    for sample in samples:
        padding = max_length - len(sample.input_ids)
        input_ids.append(list(sample.input_ids) + [pad_token_id] * padding)
        attention_mask.append([1] * len(sample.input_ids) + [0] * padding)
        response_mask.append(list(sample.response_mask) + [0] * padding)
        old_logprobs.append(list(sample.old_logprobs) + [0.0] * padding)
        advantages.append(
            [sample.advantage * mask for mask in sample.response_mask] + [0.0] * padding
        )
        return_targets.append(
            [sample.return_target * mask for mask in sample.response_mask]
            + [0.0] * padding
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
        "old_logprobs": old_logprobs,
        "advantages": advantages,
        "return_targets": return_targets,
    }

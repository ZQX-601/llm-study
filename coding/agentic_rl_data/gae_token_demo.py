"""使用二维 token 矩阵演示 PPO 的 token-level GAE。

这里先采用传统 LLM PPO 视角：Critic 为每个 response token 位置预测 value，序列级
终局奖励放在最后一个有效 token，然后从右向左传播 advantage。当前 Agent 教学环境的
主实现采用“决策级 GAE，再广播到一次 Action 的 token”，两者粒度不同但递推相同。
"""

from __future__ import annotations

from typing import Sequence


def compute_token_gae(
    *,
    rewards: Sequence[Sequence[float]],
    old_values: Sequence[Sequence[float]],
    response_mask: Sequence[Sequence[int]],
    terminated: Sequence[bool],
    bootstrap_values: Sequence[float],
    gamma: float = 0.9,
    gae_lambda: float = 0.8,
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """计算 `[B,T]` response 矩阵的 delta、GAE advantage 与 return。

    假设每行有效 response token 连续排列在左侧，PAD 在右侧。终止样本最后一个有效
    token 的 next value 为 0；截断样本使用 bootstrap_values[b]。
    """

    batch_size = len(rewards)
    if not (
        len(old_values)
        == len(response_mask)
        == len(terminated)
        == len(bootstrap_values)
        == batch_size
    ):
        raise ValueError("batch 维度不一致")

    deltas: list[list[float]] = []
    advantages: list[list[float]] = []
    returns: list[list[float]] = []

    for batch_index in range(batch_size):
        width = len(rewards[batch_index])
        if not (
            len(old_values[batch_index])
            == len(response_mask[batch_index])
            == width
        ):
            raise ValueError("token 维度不一致")

        valid_indices = [
            index
            for index, valid in enumerate(response_mask[batch_index])
            if valid == 1
        ]
        if valid_indices != list(range(len(valid_indices))):
            raise ValueError("教学实现要求有效 response token 位于 PAD 左侧")

        row_delta = [0.0] * width
        row_advantage = [0.0] * width
        row_return = [0.0] * width
        next_advantage = 0.0

        for position in reversed(valid_indices):
            is_last = position == valid_indices[-1]
            if is_last:
                nonterminal = 0.0 if terminated[batch_index] else 1.0
                next_value = bootstrap_values[batch_index]
            else:
                nonterminal = 1.0
                next_value = old_values[batch_index][position + 1]

            delta = (
                rewards[batch_index][position]
                + gamma * next_value * nonterminal
                - old_values[batch_index][position]
            )
            advantage = (
                delta
                + gamma * gae_lambda * nonterminal * next_advantage
            )
            row_delta[position] = delta
            row_advantage[position] = advantage
            row_return[position] = advantage + old_values[batch_index][position]
            next_advantage = advantage

        deltas.append(row_delta)
        advantages.append(row_advantage)
        returns.append(row_return)

    return deltas, advantages, returns


def main() -> None:
    # 这里的 token id 只是为了显示 batch 中两条变长 response；0 表示 PAD。
    response_token_ids = [
        [21, 22, 23, 0],
        [31, 32, 33, 34],
    ]
    response_mask = [
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ]
    rewards = [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.2, 0.0, -0.5],
    ]
    old_values = [
        [0.2, 0.4, 0.5, 0.0],
        [0.3, 0.2, 0.1, 0.0],
    ]

    deltas, advantages, returns = compute_token_gae(
        rewards=rewards,
        old_values=old_values,
        response_mask=response_mask,
        terminated=[True, True],
        bootstrap_values=[0.0, 0.0],
        gamma=0.9,
        gae_lambda=0.8,
    )

    print("response_token_ids =", response_token_ids)
    print("response_mask      =", response_mask)
    print("rewards            =", rewards)
    print("old_values         =", old_values)
    print("deltas             =", deltas)
    print("advantages         =", advantages)
    print("returns            =", returns)


if __name__ == "__main__":
    main()

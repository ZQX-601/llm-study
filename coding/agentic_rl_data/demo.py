"""运行端到端教学流程：Agent 决策 -> trajectory -> SFT/PPO/GRPO 样本。"""

from __future__ import annotations

try:
    from .contracts import Action, ActionKind
    from .environment import AgentRLEnvironment, ScriptedPolicy, collect_trajectory
    from .sample_builder import (
        build_grpo_group,
        build_ppo_samples,
        build_sft_samples,
        collate_ppo_batch,
    )
except ImportError:  # 支持直接执行 python coding/agentic_rl_data/demo.py
    from contracts import Action, ActionKind
    from environment import AgentRLEnvironment, ScriptedPolicy, collect_trajectory
    from sample_builder import (
        build_grpo_group,
        build_ppo_samples,
        build_sft_samples,
        collate_ppo_batch,
    )


def successful_actions() -> list[Action]:
    return [
        Action(ActionKind.TOOL, "get_training_job", {"job_id": "train-42"}),
        Action(ActionKind.TOOL, "get_training_logs", {"job_id": "train-42"}),
        Action(ActionKind.FINAL, "final", {"answer": "任务因 OOM 失败"}),
    ]


def inefficient_actions() -> list[Action]:
    return [
        Action(ActionKind.FINAL, "final", {"answer": "我猜是 OOM"}),
        *successful_actions(),
    ]


def truncated_actions() -> list[Action]:
    return [
        Action(ActionKind.TOOL, "get_training_job", {"job_id": "train-42"}),
        Action(ActionKind.TOOL, "get_training_job", {"job_id": "train-42"}),
    ]


def main() -> None:
    good = collect_trajectory(
        policy=ScriptedPolicy(successful_actions(), policy_version=10),
        environment=AgentRLEnvironment(max_steps=5),
        trajectory_id="traj-good",
    )
    inefficient = collect_trajectory(
        policy=ScriptedPolicy(inefficient_actions(), policy_version=10),
        environment=AgentRLEnvironment(max_steps=5),
        trajectory_id="traj-inefficient",
    )
    truncated = collect_trajectory(
        policy=ScriptedPolicy(truncated_actions(), policy_version=10),
        environment=AgentRLEnvironment(max_steps=2),
        trajectory_id="traj-truncated",
    )

    print("=== 1. 原始 trajectory ===")
    for trajectory in (good, inefficient, truncated):
        last = trajectory.steps[-1]
        print(
            trajectory.trajectory_id,
            f"steps={len(trajectory.steps)}",
            f"reward={trajectory.total_reward:.2f}",
            f"success={trajectory.success}",
            f"terminated={last.terminated}",
            f"truncated={last.truncated}",
        )

    print("\n=== 2. 成功轨迹转 SFT ===")
    sft = build_sft_samples(good)
    print(f"SFT samples={len(sft)}, prompt labels 均为 -100")

    print("\n=== 3. 轨迹转 PPO + padding batch ===")
    ppo = build_ppo_samples(good, current_policy_version=12, max_policy_lag=3)
    batch = collate_ppo_batch(ppo)
    print(f"PPO decisions={len(ppo)}, batch shape=({len(ppo)}, {len(batch['input_ids'][0])})")
    print(f"第 0 条 response tokens={sum(batch['response_mask'][0])}")

    print("\n=== 4. 同 prompt 多 rollout 转 GRPO group ===")
    grpo = build_grpo_group([good, inefficient, truncated])
    for sample in grpo:
        print(
            sample.trajectory_id,
            f"reward={sample.total_reward:.2f}",
            f"group_advantage={sample.group_advantage:.3f}",
        )


if __name__ == "__main__":
    main()

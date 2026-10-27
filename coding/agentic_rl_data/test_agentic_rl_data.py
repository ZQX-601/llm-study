from __future__ import annotations

import unittest
from dataclasses import replace

from .demo import inefficient_actions, successful_actions, truncated_actions
from .environment import AgentRLEnvironment, ScriptedPolicy, collect_trajectory
from .gae_token_demo import compute_token_gae
from .sample_builder import (
    IGNORE_INDEX,
    build_grpo_group,
    build_ppo_samples,
    build_sft_samples,
    collate_ppo_batch,
    validate_trajectory,
)


def collect(actions, *, trajectory_id="traj", max_steps=5, version=10):
    return collect_trajectory(
        policy=ScriptedPolicy(actions, policy_version=version),
        environment=AgentRLEnvironment(max_steps=max_steps),
        trajectory_id=trajectory_id,
    )


class AgenticRLDataTest(unittest.TestCase):
    def test_token_gae_matches_manual_calculation_and_masks_pad(self):
        deltas, advantages, returns = compute_token_gae(
            rewards=[[0.0, 0.0, 1.0, 0.0]],
            old_values=[[0.2, 0.4, 0.5, 0.0]],
            response_mask=[[1, 1, 1, 0]],
            terminated=[True],
            bootstrap_values=[0.0],
            gamma=0.9,
            gae_lambda=0.8,
        )
        expected_delta = [0.16, 0.05, 0.5, 0.0]
        expected_advantage = [0.4552, 0.41, 0.5, 0.0]
        expected_return = [0.6552, 0.81, 1.0, 0.0]
        for actual, expected in zip(deltas[0], expected_delta):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(advantages[0], expected_advantage):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(returns[0], expected_return):
            self.assertAlmostEqual(actual, expected)

    def test_successful_agent_run_becomes_three_transitions(self):
        trajectory = collect(successful_actions())
        self.assertTrue(trajectory.success)
        self.assertEqual(len(trajectory.steps), 3)
        self.assertTrue(trajectory.steps[-1].terminated)
        self.assertFalse(trajectory.steps[-1].truncated)

    def test_max_steps_is_truncation_not_natural_termination(self):
        trajectory = collect(truncated_actions(), max_steps=2)
        final = trajectory.steps[-1]
        self.assertFalse(trajectory.success)
        self.assertFalse(final.terminated)
        self.assertTrue(final.truncated)
        self.assertEqual(final.stop_reason, "MAX_STEPS")
        self.assertLess(trajectory.total_reward, 0)

    def test_repeated_evidence_cannot_farm_positive_reward(self):
        trajectory = collect(truncated_actions(), max_steps=2)
        self.assertGreater(trajectory.steps[0].reward.evidence, 0)
        self.assertEqual(trajectory.steps[1].reward.evidence, 0)

    def test_rejected_action_is_preserved_as_negative_rl_transition(self):
        trajectory = collect(inefficient_actions())
        rejected = trajectory.steps[0]
        self.assertFalse(rejected.action_accepted)
        self.assertLess(rejected.reward.total, 0)
        self.assertTrue(trajectory.success)

    def test_sft_uses_only_accepted_actions_from_successful_trajectory(self):
        trajectory = collect(inefficient_actions())
        samples = build_sft_samples(trajectory)
        self.assertEqual(len(samples), 3)
        self.assertTrue(all(IGNORE_INDEX in sample.labels for sample in samples))
        self.assertTrue(all(sample.step_id != 0 for sample in samples))

    def test_failed_trajectory_is_not_sft_positive_data(self):
        trajectory = collect(truncated_actions(), max_steps=2)
        self.assertEqual(build_sft_samples(trajectory), [])

    def test_ppo_masks_prompt_and_trains_only_response_tokens(self):
        trajectory = collect(successful_actions())
        sample = build_ppo_samples(trajectory)[0]
        prompt_length = len(trajectory.steps[0].decision.prompt_token_ids)
        self.assertEqual(sample.response_mask[:prompt_length], (0,) * prompt_length)
        self.assertTrue(all(mask == 1 for mask in sample.response_mask[prompt_length:]))
        self.assertEqual(len(sample.input_ids), len(sample.old_logprobs))

    def test_truncated_transition_can_bootstrap_value(self):
        trajectory = collect(truncated_actions(), max_steps=2)
        without_bootstrap = build_ppo_samples(
            trajectory, gamma=1.0, gae_lambda=1.0, bootstrap_value=0.0
        )
        with_bootstrap = build_ppo_samples(
            trajectory, gamma=1.0, gae_lambda=1.0, bootstrap_value=0.7
        )
        self.assertAlmostEqual(
            with_bootstrap[-1].return_target,
            without_bootstrap[-1].return_target + 0.7,
        )

    def test_stale_policy_is_rejected_before_training(self):
        trajectory = collect(successful_actions(), version=10)
        with self.assertRaisesRegex(ValueError, "轨迹过期"):
            build_ppo_samples(
                trajectory, current_policy_version=14, max_policy_lag=3
            )

    def test_mismatched_old_logprobs_are_rejected(self):
        trajectory = collect(successful_actions())
        first = trajectory.steps[0]
        bad_decision = replace(first.decision, old_logprobs=())
        trajectory.steps[0] = replace(first, decision=bad_decision)
        with self.assertRaisesRegex(ValueError, "old_logprob"):
            validate_trajectory(trajectory)

    def test_grpo_advantages_are_zero_mean_within_group(self):
        trajectories = [
            collect(successful_actions(), trajectory_id="good"),
            collect(inefficient_actions(), trajectory_id="inefficient"),
            collect(truncated_actions(), trajectory_id="truncated", max_steps=2),
        ]
        samples = build_grpo_group(trajectories)
        self.assertAlmostEqual(sum(sample.group_advantage for sample in samples), 0.0)
        self.assertGreater(samples[0].group_advantage, samples[1].group_advantage)

    def test_collate_masks_padding_everywhere(self):
        trajectory = collect(successful_actions())
        samples = build_ppo_samples(trajectory)
        batch = collate_ppo_batch(samples)
        width = len(batch["input_ids"][0])
        self.assertTrue(
            all(len(row) == width for rows in batch.values() for row in rows)
        )
        for attention, response, advantages in zip(
            batch["attention_mask"], batch["response_mask"], batch["advantages"]
        ):
            for attn, resp, advantage in zip(attention, response, advantages):
                if attn == 0:
                    self.assertEqual(resp, 0)
                    self.assertEqual(advantage, 0.0)


if __name__ == "__main__":
    unittest.main()

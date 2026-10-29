from __future__ import annotations

import math
import unittest
from dataclasses import replace

from .admission import (
    AdmissionDecision,
    TrainingDataContract,
    admit_grpo_group,
    admit_policy_shift,
    admit_trajectory,
)
from .demo import inefficient_actions, successful_actions, truncated_actions
from .environment import AgentRLEnvironment, ScriptedPolicy, collect_trajectory
from .gae_token_demo import compute_token_gae
from .sample_builder import (
    IGNORE_INDEX,
    build_grpo_group,
    build_ppo_samples,
    build_sft_samples,
    collate_grpo_batch,
    collate_ppo_batch,
    trajectory_equal_masked_mean,
    validate_trajectory,
)
from .tensor_loss import ppo_policy_loss_reference


def collect(actions, *, trajectory_id="traj", max_steps=5, version=10):
    return collect_trajectory(
        policy=ScriptedPolicy(actions, policy_version=version),
        environment=AgentRLEnvironment(max_steps=max_steps),
        trajectory_id=trajectory_id,
    )


class AgenticRLDataTest(unittest.TestCase):
    def test_ppo_reference_loss_applies_causal_shift_and_masked_mean(self):
        # 原序列位置 1、2 是 Response；位置 0 是 BOS，位置 3 是 PAD。
        # 两个有效 token 的 loss 分别是 -2.4 和 1.6，masked mean 为 -0.4。
        metrics = ppo_policy_loss_reference(
            new_logprobs=[[
                -1.0 + math.log(1.5),
                -1.0 + math.log(0.6),
                100.0,
            ]],
            old_logprobs=[[0.0, -1.0, -1.0, 0.0]],
            advantages=[[0.0, 2.0, -2.0, 0.0]],
            response_mask=[[0, 1, 1, 0]],
        )

        self.assertAlmostEqual(metrics.policy_loss, -0.4)
        self.assertAlmostEqual(metrics.clipfrac, 1.0)
        self.assertEqual(metrics.valid_token_count, 2)

    def test_dynamic_admission_masks_prompt_and_padding_values(self):
        contract = TrainingDataContract(
            max_clipfrac=1.0,
            max_approx_kl=1.0,
        )

        result, metrics = admit_policy_shift(
            old_logprobs=[[-1.0, -2.0, -9.0, -9.0]],
            new_logprobs=[[-0.95, -1.3, 100.0, 100.0]],
            response_mask=[[1, 1, 0, 0]],
            contract=contract,
        )

        self.assertIs(result.decision, AdmissionDecision.ACCEPT)
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.valid_token_count, 2)
        self.assertAlmostEqual(metrics.clipfrac, 0.5)

    def test_large_policy_shift_rejects_valid_data(self):
        contract = TrainingDataContract(max_clipfrac=0.3)

        result, metrics = admit_policy_shift(
            old_logprobs=[[-1.0, -2.0]],
            new_logprobs=[[-0.95, -1.3]],
            response_mask=[[1, 1]],
            contract=contract,
        )

        self.assertIs(result.decision, AdmissionDecision.REJECT)
        self.assertIsNotNone(metrics)
        self.assertAlmostEqual(metrics.clipfrac, 0.5)

    def test_nan_logprob_is_quarantined(self):
        result, metrics = admit_policy_shift(
            old_logprobs=[[-1.0, float("nan")]],
            new_logprobs=[[-0.9, -1.2]],
            response_mask=[[1, 1]],
            contract=TrainingDataContract(),
        )

        self.assertIs(result.decision, AdmissionDecision.QUARANTINE)
        self.assertIsNone(metrics)

    def test_reward_only_change_can_be_reprocessed_from_complete_trace(self):
        trajectory = collect(successful_actions())
        contract = TrainingDataContract(reward_version="training-reward-v2")

        result = admit_trajectory(
            trajectory,
            contract,
            reward_recomputable_from_trace=True,
        )

        self.assertIs(result.decision, AdmissionDecision.REPROCESS)
        self.assertEqual(result.reason_code, "REWARD_VERSION_MISMATCH")

    def test_tool_schema_change_requires_recollection(self):
        trajectory = collect(successful_actions())
        contract = TrainingDataContract(tool_schema_version="training-tools-v2")

        result = admit_trajectory(trajectory, contract)

        self.assertIs(result.decision, AdmissionDecision.RECOLLECT)
        self.assertEqual(result.reason_code, "CAUSAL_VERSION_MISMATCH")

    def test_policy_lag_is_a_cheap_admission_filter(self):
        trajectory = collect(successful_actions(), version=10)
        contract = TrainingDataContract(
            current_policy_version=13,
            max_policy_lag=2,
        )

        result = admit_trajectory(trajectory, contract)

        self.assertIs(result.decision, AdmissionDecision.REJECT)
        self.assertEqual(result.reason_code, "POLICY_TOO_STALE")

    def test_incomplete_grpo_group_waits_then_rejects_as_a_whole(self):
        trajectories = [
            collect(successful_actions(), trajectory_id=f"rollout-{index}")
            for index in range(3)
        ]
        contract = TrainingDataContract(expected_group_size=4)

        waiting = admit_grpo_group(trajectories, contract)
        timed_out = admit_grpo_group(
            trajectories,
            contract,
            assembly_timed_out=True,
        )

        self.assertIs(waiting.decision, AdmissionDecision.PENDING)
        self.assertIs(timed_out.decision, AdmissionDecision.REJECT)

    def test_terminal_failure_counts_as_a_completed_grpo_rollout(self):
        trajectories = [
            collect(successful_actions(), trajectory_id=f"success-{index}")
            for index in range(3)
        ]
        # 教学环境用 max_steps 截断来表示一条有明确边界和失败 Reward 的完整 rollout。
        # 它可以作为组内失败样本；“失败”不等于“数据缺失”。
        trajectories.append(
            collect(
                truncated_actions(),
                trajectory_id="timeout-terminal",
                max_steps=2,
            )
        )
        contract = TrainingDataContract(expected_group_size=4)

        result = admit_grpo_group(trajectories, contract)

        self.assertIs(result.decision, AdmissionDecision.ACCEPT)

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

    def test_grpo_decisions_do_not_carry_ppo_gae_or_return(self):
        trajectories = [
            collect(successful_actions(), trajectory_id="good"),
            collect(inefficient_actions(), trajectory_id="inefficient"),
        ]
        samples = build_grpo_group(trajectories)
        decision = samples[0].decisions[0]
        self.assertFalse(hasattr(decision, "advantage"))
        self.assertFalse(hasattr(decision, "return_target"))
        self.assertEqual(decision.trajectory_id, "good")

    def test_grpo_collate_flattens_decisions_but_preserves_trajectory_identity(self):
        trajectories = [
            collect(successful_actions(), trajectory_id="good"),
            collect(inefficient_actions(), trajectory_id="inefficient"),
            collect(
                truncated_actions(),
                trajectory_id="truncated",
                max_steps=2,
            ),
        ]
        samples = build_grpo_group(trajectories)
        batch = collate_grpo_batch(samples)

        self.assertEqual(
            batch["trajectory_indices"],
            [0, 0, 0, 1, 1, 1, 1, 2, 2],
        )
        self.assertEqual(len(batch["input_ids"]), 9)
        self.assertEqual(len(batch["input_ids"][0]), 315)

        for row, trajectory_index in enumerate(batch["trajectory_indices"]):
            expected_advantage = samples[trajectory_index].group_advantage
            for mask, advantage in zip(
                batch["response_mask"][row],
                batch["group_advantages"][row],
            ):
                expected = expected_advantage if mask == 1 else 0.0
                self.assertAlmostEqual(advantage, expected)

    def test_trajectory_equal_mean_removes_length_weighting(self):
        trajectories = [
            collect(successful_actions(), trajectory_id="good"),
            collect(inefficient_actions(), trajectory_id="inefficient"),
            collect(
                truncated_actions(),
                trajectory_id="truncated",
                max_steps=2,
            ),
        ]
        samples = build_grpo_group(trajectories)
        batch = collate_grpo_batch(samples)

        objective, trajectory_means = trajectory_equal_masked_mean(
            per_token_values=batch["group_advantages"],
            response_mask=batch["response_mask"],
            trajectory_indices=batch["trajectory_indices"],
            num_trajectories=len(samples),
        )
        expected_means = [sample.group_advantage for sample in samples]
        for actual, expected in zip(trajectory_means, expected_means):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(objective, 0.0)

        token_sum = sum(
            sum(row) for row in batch["group_advantages"]
        )
        token_count = sum(sum(row) for row in batch["response_mask"])
        self.assertGreater(token_sum / token_count, 0.1)

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

    def test_collate_broadcasts_return_only_to_response_tokens(self):
        trajectory = collect(successful_actions())
        samples = build_ppo_samples(trajectory)
        batch = collate_ppo_batch(samples)

        for sample, response_mask, return_targets in zip(
            samples, batch["response_mask"], batch["return_targets"]
        ):
            for mask, target in zip(response_mask, return_targets):
                expected = sample.return_target if mask == 1 else 0.0
                self.assertAlmostEqual(target, expected)


if __name__ == "__main__":
    unittest.main()

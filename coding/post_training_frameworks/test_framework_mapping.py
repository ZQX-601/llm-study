from __future__ import annotations

import unittest

from .framework_mapping import (
    GRPOBatchState,
    PipelineStage,
    TrainingProgress,
    VerlStepConfig,
    assess_training_progress,
    build_verl_sync_step_plan,
    components_for_stage,
    validate_rollout_contract,
)


def rollout_only_state() -> GRPOBatchState:
    """两条 completion 的 ragged rollout；尚未执行 Reward 和训练。"""

    return GRPOBatchState(
        prompt_token_ids=(10, 11, 12),
        completion_token_ids=((21, 22), (31, 32, 33)),
        old_logprobs=((-0.2, -0.3), (-0.4, -0.5, -0.6)),
        response_mask=((1, 1), (1, 1, 1)),
        behavior_policy_version="policy-v101",
    )


class FrameworkMappingTest(unittest.TestCase):
    def test_vllm_completion_is_only_rollout_evidence(self):
        assessment = assess_training_progress(rollout_only_state())

        self.assertEqual(assessment.progress, TrainingProgress.ROLLOUT_ONLY)
        self.assertEqual(assessment.missing_for_next_stage, ("rewards[G]",))

    def test_rollout_requires_behavior_identity_and_aligned_logprobs(self):
        invalid_state = GRPOBatchState(
            prompt_token_ids=(10,),
            completion_token_ids=((21, 22),),
            old_logprobs=((-0.2,),),
            response_mask=((1, 1),),
            behavior_policy_version="policy-v101",
        )

        with self.assertRaisesRegex(ValueError, "长度不一致"):
            validate_rollout_contract(invalid_state)

    def test_reward_does_not_imply_advantage(self):
        state = GRPOBatchState(
            **{
                **rollout_only_state().__dict__,
                "rewards": (1.0, 0.0),
            }
        )

        assessment = assess_training_progress(state)
        self.assertEqual(assessment.progress, TrainingProgress.REWARDS_READY)
        self.assertEqual(assessment.missing_for_next_stage, ("group_advantages[G]",))

    def test_optimizer_update_is_not_the_same_as_weight_publication(self):
        state = GRPOBatchState(
            **{
                **rollout_only_state().__dict__,
                "rewards": (1.0, 0.0),
                "group_advantages": (1.0, -1.0),
                "new_logprobs": ((-0.1, -0.2), (-0.3, -0.4, -0.5)),
                "policy_loss": -0.25,
                "gradients_finite": True,
                "optimizer_step_applied": True,
            }
        )

        assessment = assess_training_progress(state)
        self.assertEqual(assessment.progress, TrainingProgress.UPDATED)
        self.assertEqual(assessment.missing_for_next_stage, ("published_policy_version",))

    def test_reference_logprobs_are_optional_when_all_kl_is_disabled(self):
        state = GRPOBatchState(
            **{
                **rollout_only_state().__dict__,
                "rewards": (1.0, 0.0),
                "group_advantages": (1.0, -1.0),
                "new_logprobs": ((-0.1, -0.2), (-0.3, -0.4, -0.5)),
                "policy_loss": -0.25,
                "use_kl_in_reward": False,
                "use_kl_loss": False,
            }
        )

        assessment = assess_training_progress(state)
        self.assertEqual(assessment.progress, TrainingProgress.LOSS_READY)
        self.assertNotIn("ref_logprobs[G,T]", assessment.missing_for_next_stage)

    def test_reference_logprobs_are_required_when_kl_loss_is_enabled(self):
        state = GRPOBatchState(
            **{
                **rollout_only_state().__dict__,
                "rewards": (1.0, 0.0),
                "group_advantages": (1.0, -1.0),
                "new_logprobs": ((-0.1, -0.2), (-0.3, -0.4, -0.5)),
                "policy_loss": -0.25,
                "use_kl_loss": True,
            }
        )

        assessment = assess_training_progress(state)
        self.assertEqual(assessment.progress, TrainingProgress.ADVANTAGES_READY)
        self.assertEqual(assessment.missing_for_next_stage, ("ref_logprobs[G,T]",))

    def test_vllm_maps_to_rollout_not_optimizer(self):
        rollout_components = components_for_stage(PipelineStage.ROLLOUT)
        optimizer_components = components_for_stage(PipelineStage.OPTIMIZER)

        self.assertTrue(any(item.framework == "vLLM" for item in rollout_components))
        self.assertFalse(any(item.framework == "vLLM" for item in optimizer_components))

    def test_grpo_without_kl_omits_reference_and_critic(self):
        plan = build_verl_sync_step_plan(
            VerlStepConfig(
                algorithm="grpo",
                use_reference_policy=False,
                use_critic=False,
            )
        )

        self.assertNotIn("compute_ref_logprobs", plan)
        self.assertNotIn("compute_values", plan)
        self.assertNotIn("update_critic", plan)
        self.assertLess(plan.index("compute_advantages"), plan.index("update_actor"))

    def test_ppo_with_kl_contains_reference_and_critic(self):
        plan = build_verl_sync_step_plan(
            VerlStepConfig(
                algorithm="ppo",
                use_reference_policy=True,
                use_critic=True,
            )
        )

        self.assertLess(plan.index("compute_ref_logprobs"), plan.index("compute_advantages"))
        self.assertLess(plan.index("compute_values"), plan.index("compute_advantages"))
        self.assertLess(plan.index("update_critic"), plan.index("update_actor"))
        self.assertEqual(plan[-1], "publish_weights")

    def test_ppo_without_critic_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "PPO 计划必须启用 Critic"):
            build_verl_sync_step_plan(
                VerlStepConfig(
                    algorithm="ppo",
                    use_reference_policy=False,
                    use_critic=False,
                )
            )

    def test_non_finite_loss_cannot_be_updated_or_published(self):
        state = GRPOBatchState(
            **{
                **rollout_only_state().__dict__,
                "rewards": (1.0, 0.0),
                "group_advantages": (1.0, -1.0),
                "new_logprobs": ((-0.1, -0.2), (-0.3, -0.4, -0.5)),
                "policy_loss": float("nan"),
                "gradients_finite": True,
                "optimizer_step_applied": True,
                "published_policy_version": "policy-v102",
            }
        )

        assessment = assess_training_progress(state)
        self.assertEqual(assessment.progress, TrainingProgress.ADVANTAGES_READY)
        self.assertEqual(assessment.missing_for_next_stage, ("finite_policy_loss",))


if __name__ == "__main__":
    unittest.main()


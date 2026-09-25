"""End-to-end, single-process GRPO teaching implementation.

The code emphasizes the lifecycle of one rollout batch:
B prompts -> B*G sampled completions -> fixed rollout data -> K optimization
passes split into mini-batches -> next iteration samples fresh completions.
"""

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GRPOConfig:
    group_size: int = 4
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_p: float = 0.95
    std_threshold: float = 0.01
    clip_eps: float = 0.2
    kl_beta: float = 0.05
    optimization_epochs: int = 2
    mini_batch_size: int = 8
    max_grad_norm: float = 1.0


@dataclass
class RolloutBatch:
    full_ids: torch.Tensor
    full_attention_mask: torch.Tensor
    response_mask: torch.Tensor
    old_logp: torch.Tensor
    ref_logp: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    valid_sequence_mask: torch.Tensor
    group_ids: torch.Tensor
    prompt_width: int
    response_width: int
    valid_group_rate: float


def expand_prompts(prompts: Sequence[str], group_size: int):
    """Repeat each prompt G times and assign a stable group id."""
    expanded_prompts = []
    group_ids = []
    for group_id, prompt in enumerate(prompts):
        expanded_prompts.extend([prompt] * group_size)
        group_ids.extend([group_id] * group_size)
    return expanded_prompts, torch.tensor(group_ids, dtype=torch.long)


def build_response_mask(completion_ids: torch.Tensor, eos_token_id: int):
    """Keep the first EOS and all preceding tokens; mask later padding."""
    is_eos = completion_ids.eq(eos_token_id)
    eos_seen_before = is_eos.long().cumsum(dim=1) - is_eos.long()
    return eos_seen_before.eq(0)


def get_response_logp(
    model,
    full_ids: torch.Tensor,
    full_attention_mask: torch.Tensor,
    prompt_width: int,
    response_width: int,
):
    """Teacher-force fixed rollout tokens and return [N, T] log-probs."""
    logits = model(
        input_ids=full_ids,
        attention_mask=full_attention_mask,
    ).logits

    shifted_logits = logits[:, :-1, :]
    shifted_targets = full_ids[:, 1:]
    selected_logp = F.log_softmax(shifted_logits, dim=-1).gather(
        dim=-1,
        index=shifted_targets.unsqueeze(-1),
    ).squeeze(-1)

    start = prompt_width - 1
    return selected_logp[:, start : start + response_width]


def groupwise_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    std_threshold: float,
    eps: float = 1e-8,
):
    """Normalize reward inside each prompt group and filter low-variance groups."""
    advantages = torch.zeros_like(rewards)
    valid_sequence_mask = torch.zeros_like(rewards, dtype=torch.bool)
    unique_group_ids = torch.unique(group_ids)
    valid_group_count = 0

    for group_id in unique_group_ids:
        group_mask = group_ids.eq(group_id)
        group_rewards = rewards[group_mask]
        group_mean = group_rewards.mean()
        group_std = group_rewards.std(correction=0)

        if group_std >= std_threshold:
            advantages[group_mask] = (
                group_rewards - group_mean
            ) / (group_std + eps)
            valid_sequence_mask[group_mask] = True
            valid_group_count += 1

    valid_group_rate = valid_group_count / unique_group_ids.numel()
    return advantages, valid_sequence_mask, valid_group_rate


@torch.no_grad()
def collect_rollout(
    policy_model,
    reference_model,
    tokenizer,
    prompts: Sequence[str],
    reward_fn: Callable[[str, str], float],
    config: GRPOConfig,
):
    """Create all fixed data used by K optimization epochs."""
    policy_model.eval()
    reference_model.eval()
    expanded_prompts, group_ids = expand_prompts(prompts, config.group_size)

    encoded = tokenizer(expanded_prompts, padding=True, return_tensors="pt")
    encoded = {key: value.to(policy_model.device) for key, value in encoded.items()}
    prompt_ids = encoded["input_ids"]
    prompt_attention_mask = encoded["attention_mask"]
    prompt_width = prompt_ids.shape[1]

    # The iteration's only free-generation call.
    full_ids = policy_model.generate(
        input_ids=prompt_ids,
        attention_mask=prompt_attention_mask,
        do_sample=True,
        temperature=config.temperature,
        top_p=config.top_p,
        max_new_tokens=config.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    completion_ids = full_ids[:, prompt_width:]
    response_width = completion_ids.shape[1]
    response_mask = build_response_mask(completion_ids, tokenizer.eos_token_id)
    full_attention_mask = torch.cat(
        [prompt_attention_mask, response_mask.to(prompt_attention_mask.dtype)],
        dim=1,
    )

    # These models score the same fixed completions. Both outputs stay fixed.
    old_logp = get_response_logp(
        policy_model,
        full_ids,
        full_attention_mask,
        prompt_width,
        response_width,
    ).detach()
    ref_logp = get_response_logp(
        reference_model,
        full_ids,
        full_attention_mask,
        prompt_width,
        response_width,
    ).detach()

    completion_texts = tokenizer.batch_decode(
        completion_ids,
        skip_special_tokens=True,
    )
    rewards = torch.tensor(
        [
            reward_fn(prompt, completion)
            for prompt, completion in zip(expanded_prompts, completion_texts)
        ],
        dtype=old_logp.dtype,
        device=policy_model.device,
    )
    group_ids = group_ids.to(policy_model.device)
    advantages, valid_sequence_mask, valid_group_rate = groupwise_advantages(
        rewards,
        group_ids,
        config.std_threshold,
    )

    return RolloutBatch(
        full_ids=full_ids.detach(),
        full_attention_mask=full_attention_mask.detach(),
        response_mask=response_mask.detach(),
        old_logp=old_logp,
        ref_logp=ref_logp,
        rewards=rewards,
        advantages=advantages.detach(),
        valid_sequence_mask=valid_sequence_mask.detach(),
        group_ids=group_ids,
        prompt_width=prompt_width,
        response_width=response_width,
        valid_group_rate=valid_group_rate,
    )


def reference_kl(new_logp: torch.Tensor, ref_logp: torch.Tensor):
    ref_log_ratio = ref_logp - new_logp
    return ref_log_ratio.exp() - ref_log_ratio - 1.0


def grpo_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    valid_sequence_mask: torch.Tensor,
    clip_eps: float,
    kl_beta: float,
):
    """Compute one optimization mini-batch's masked GRPO loss."""
    ratio = (new_logp - old_logp).exp()
    token_advantages = advantages[:, None]
    surrogate_1 = ratio * token_advantages
    surrogate_2 = ratio.clamp(
        1.0 - clip_eps,
        1.0 + clip_eps,
    ) * token_advantages
    policy_objective = torch.minimum(surrogate_1, surrogate_2)

    per_token_kl = reference_kl(new_logp, ref_logp)
    per_token_loss = -policy_objective + kl_beta * per_token_kl
    effective_mask = (
        response_mask * valid_sequence_mask[:, None]
    ).to(per_token_loss.dtype)
    valid_token_count = effective_mask.sum()

    if valid_token_count.item() == 0:
        return None

    loss = (per_token_loss * effective_mask).sum() / valid_token_count
    with torch.no_grad():
        ratio_outside = (
            (ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)
        )
        clipfrac = (
            ratio_outside.to(effective_mask.dtype) * effective_mask
        ).sum() / valid_token_count
        mean_kl = (per_token_kl * effective_mask).sum() / valid_token_count

    return loss, {
        "loss": loss.detach(),
        "clipfrac": clipfrac.detach(),
        "mean_kl": mean_kl.detach(),
        "valid_token_count": valid_token_count.detach(),
    }


def grpo_update(
    policy_model,
    optimizer,
    rollout: RolloutBatch,
    indices: torch.Tensor,
    config: GRPOConfig,
):
    """Update the policy once from one mini-batch of fixed rollout sequences."""
    new_logp = get_response_logp(
        policy_model,
        rollout.full_ids[indices],
        rollout.full_attention_mask[indices],
        rollout.prompt_width,
        rollout.response_width,
    )

    result = grpo_loss(
        new_logp=new_logp,
        old_logp=rollout.old_logp[indices],
        ref_logp=rollout.ref_logp[indices],
        advantages=rollout.advantages[indices],
        response_mask=rollout.response_mask[indices],
        valid_sequence_mask=rollout.valid_sequence_mask[indices],
        clip_eps=config.clip_eps,
        kl_beta=config.kl_beta,
    )
    if result is None:
        return None

    loss, metrics = result
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        policy_model.parameters(),
        config.max_grad_norm,
    )
    optimizer.step()
    return metrics


def train_one_iteration(
    policy_model,
    reference_model,
    tokenizer,
    prompts: Sequence[str],
    reward_fn,
    optimizer,
    config: GRPOConfig,
):
    """Sample once, then reuse that rollout batch for K optimization epochs."""
    rollout = collect_rollout(
        policy_model,
        reference_model,
        tokenizer,
        prompts,
        reward_fn,
        config,
    )
    policy_model.train()
    sequence_count = rollout.full_ids.shape[0]
    optimizer_steps = 0
    skipped_mini_batches = 0

    for epoch in range(config.optimization_epochs):
        # Group normalization is complete, so sequences may now be shuffled.
        permutation = torch.randperm(sequence_count, device=policy_model.device)
        for start in range(0, sequence_count, config.mini_batch_size):
            indices = permutation[start : start + config.mini_batch_size]
            metrics = grpo_update(
                policy_model,
                optimizer,
                rollout,
                indices,
                config,
            )
            if metrics is None:
                skipped_mini_batches += 1
                continue

            optimizer_steps += 1
            print(
                f"epoch={epoch + 1} update={optimizer_steps} "
                f"loss={metrics['loss'].item():.4f} "
                f"kl={metrics['mean_kl'].item():.4f} "
                f"clipfrac={metrics['clipfrac'].item():.4f}"
            )

    return {
        "rollout_sequences": sequence_count,
        "valid_group_rate": rollout.valid_group_rate,
        "optimizer_steps": optimizer_steps,
        "skipped_mini_batches": skipped_mini_batches,
    }


def train_grpo(
    policy_model,
    reference_model,
    tokenizer,
    prompt_loader,
    reward_fn,
    optimizer,
    config: GRPOConfig,
):
    """Each loader batch is one GRPO iteration and one fresh rollout batch."""
    for iteration, prompt_batch in enumerate(prompt_loader, start=1):
        stats = train_one_iteration(
            policy_model,
            reference_model,
            tokenizer,
            list(prompt_batch),
            reward_fn,
            optimizer,
            config,
        )
        print(f"iteration={iteration} stats={stats}")


@torch.no_grad()
def inference(
    policy_model,
    tokenizer,
    prompts: Sequence[str],
    max_new_tokens: int = 128,
):
    """Deployment needs only the trained policy and tokenizer."""
    policy_model.eval()
    encoded = tokenizer(prompts, padding=True, return_tensors="pt")
    encoded = {key: value.to(policy_model.device) for key, value in encoded.items()}
    output_ids = policy_model.generate(
        **encoded,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_width = encoded["input_ids"].shape[1]
    answer_ids = output_ids[:, prompt_width:]
    return tokenizer.batch_decode(answer_ids, skip_special_tokens=True)


def exact_answer_reward(expected_answers: dict[str, str]):
    """Toy verifier that compares the last integer in the completion."""
    def reward_fn(prompt: str, completion: str):
        numbers = re.findall(r"-?\d+", completion)
        if not numbers:
            return 0.0
        return float(numbers[-1] == expected_answers[prompt])

    return reward_fn


def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    reference_model = deepcopy(policy_model).to(device).eval()
    reference_model.requires_grad_(False)

    prompts = ["2 + 3 =", "10 - 4 =", "3 * 3 =", "8 / 2 ="]
    expected_answers = {
        "2 + 3 =": "5",
        "10 - 4 =": "6",
        "3 * 3 =": "9",
        "8 / 2 =": "4",
    }
    prompt_loader = DataLoader(prompts, batch_size=2, shuffle=True)
    reward_fn = exact_answer_reward(expected_answers)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-6)
    config = GRPOConfig(group_size=4, mini_batch_size=2, optimization_epochs=2)

    train_grpo(
        policy_model,
        reference_model,
        tokenizer,
        prompt_loader,
        reward_fn,
        optimizer,
        config,
    )
    policy_model.save_pretrained("grpo_policy")
    tokenizer.save_pretrained("grpo_policy")

    answers = inference(policy_model, tokenizer, ["2 + 3 =", "3 * 3 ="])
    print(answers)


if __name__ == "__main__":
    main()

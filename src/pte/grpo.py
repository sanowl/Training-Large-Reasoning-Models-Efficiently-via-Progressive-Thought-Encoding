from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class GRPOMetrics:
    loss: float
    policy_term: float
    kl_term: float
    reward_mean: float
    reward_std: float


def normalize_group_rewards(scores: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    scores: [B, G] raw group scores -> normalized rewards with group-wise z-score.
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be [B, G], got {tuple(scores.shape)}")
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True, unbiased=False)
    return (scores - mean) / (std + eps)


def sequence_logprob(logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Token log-probability sum over masked positions.
    logits: [B, T, V]
    labels: [B, T]
    attention_mask: [B, T] with 1 for valid completion tokens.
    """
    if logits.ndim != 3 or labels.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("invalid tensor rank for sequence_logprob")
    if logits.shape[:2] != labels.shape or labels.shape != attention_mask.shape:
        raise ValueError(
            f"shape mismatch: logits={tuple(logits.shape)}, labels={tuple(labels.shape)}, mask={tuple(attention_mask.shape)}"
        )

    log_probs = F.log_softmax(logits, dim=-1)
    token_logp = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp * attention_mask
    return token_logp.sum(dim=-1)


def grpo_objective(
    logp: torch.Tensor,
    logp_ref: torch.Tensor,
    rewards: torch.Tensor,
    *,
    beta_kl: float,
) -> tuple[torch.Tensor, GRPOMetrics]:
    """
    Surrogate objective for sampled trajectories:
    maximize reward * logp - beta * (logp - logp_ref)
    """
    if logp.shape != logp_ref.shape or logp.shape != rewards.shape:
        raise ValueError(
            f"logp/logp_ref/rewards shape mismatch: {tuple(logp.shape)} {tuple(logp_ref.shape)} {tuple(rewards.shape)}"
        )
    if beta_kl < 0:
        raise ValueError(f"beta_kl must be >= 0, got {beta_kl}")
    kl_sample = logp - logp_ref
    policy_term = rewards * logp
    loss = -(policy_term - beta_kl * kl_sample).mean()

    metrics = GRPOMetrics(
        loss=float(loss.detach().item()),
        policy_term=float(policy_term.detach().mean().item()),
        kl_term=float(kl_sample.detach().mean().item()),
        reward_mean=float(rewards.detach().mean().item()),
        reward_std=float(rewards.detach().std(unbiased=False).item()),
    )
    return loss, metrics

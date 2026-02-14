"""DRGRPO-adapted diffusion policy loss.

Maps GRPO/DAPO concepts to denoising steps:
  Token -> denoising step t
  Action -> mu_theta(x_t, sigma_t) (denoised prediction)
  log p(a_t|s_t) -> -||mu_pi - mu_ref||^2 / (2 sigma_t^2)
  Group of K responses -> K rollouts (different seeds + stochastic churn)
  Advantage A_k -> (r_k - mean(r)) / (std(r) + eps)

Combined loss:
  L_total = L_policy + lambda_ent * L_entropy + lambda_anchor * L_anchor

All functions are pure math (no model loading, no GPU side effects).
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_group_advantages(
    rewards: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Z-score normalize rewards within a group.

    Args:
        rewards: (K,) scalar rewards for K rollouts.
        eps: Stability constant for std normalization.

    Returns:
        (K,) advantages, zero-mean within the group.
    """
    mean = rewards.mean()
    std = rewards.std()
    return (rewards - mean) / (std + eps)


def compute_step_log_ratios(
    pi_denoised: list[Tensor],
    ref_denoised: list[Tensor],
    sigmas: Tensor,
) -> Tensor:
    """Per-step Gaussian log-probability ratio between policy and reference.

    log_ratio_t = -||mu_pi_t - mu_ref_t||^2 / (2 * sigma_t^2)

    Higher sigma (noisier steps) = less signal weight. This naturally
    downweights early noisy steps and upweights late clean steps.

    Args:
        pi_denoised: List of T denoised predictions from policy, each (1, C, H, W).
        ref_denoised: List of T denoised predictions from reference, each (1, C, H, W).
        sigmas: (T+1,) sigma schedule. Uses sigmas[0:T] for the T steps.

    Returns:
        (T,) per-step log-ratio values.
    """
    T = len(pi_denoised)
    log_ratios = pi_denoised[0].new_zeros(T)

    for t in range(T):
        diff = pi_denoised[t] - ref_denoised[t]
        mse = (diff * diff).sum()
        sigma_t = sigmas[t]
        log_ratios[t] = -mse / (2.0 * sigma_t * sigma_t + 1e-10)

    return log_ratios


def clipped_policy_loss(
    log_ratios_per_rollout: list[Tensor],
    advantages: Tensor,
    clip_low: float = 0.2,
    clip_high: float = 0.28,
) -> Tensor:
    """Asymmetric-clipped policy gradient loss (DAPO-style).

    ratio_k = exp(sum_t log_ratio_t_k)
    L = -(1/K) * sum_k min(ratio_k * A_k, clip(ratio_k) * A_k)

    Asymmetric clipping: less aggressive at suppressing improvements
    than penalizing regressions. Prevents overly conservative policy.

    Args:
        log_ratios_per_rollout: K tensors, each (T,) per-step log-ratios.
        advantages: (K,) group-normalized advantages.
        clip_low: Lower clip bound (1 - clip_low). Default 0.2.
        clip_high: Upper clip bound (1 + clip_high). Default 0.28 (DAPO).

    Returns:
        Scalar policy loss (negate to maximize expected reward).
    """
    K = len(log_ratios_per_rollout)

    total = advantages.new_zeros(())
    for k in range(K):
        # Sum per-step log-ratios to get trajectory-level log-ratio
        log_ratio_sum = log_ratios_per_rollout[k].sum()
        ratio = torch.exp(log_ratio_sum)

        adv = advantages[k]

        # Asymmetric clipping
        ratio_clipped = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)

        # PPO-style min
        surr1 = ratio * adv
        surr2 = ratio_clipped * adv
        total = total + torch.min(surr1, surr2)

    return -total / K


def latent_entropy_bonus(
    denoised_batch: list[Tensor],
) -> Tensor:
    """Anti-mode-collapse entropy bonus from latent variance.

    L_entropy = -log(Var_k[mu_pi(x_T_k)] + eps)

    Penalizes low variance across rollouts (all producing similar outputs).
    Uses the LAST denoised prediction (most informative, lowest noise).

    Args:
        denoised_batch: K tensors of final-step denoised outputs, each (1, C, H, W).

    Returns:
        Scalar entropy bonus (lower = more diverse, which is better).
    """
    # Stack: (K, C, H, W) after squeezing batch dim
    stacked = torch.stack([d.squeeze(0) for d in denoised_batch], dim=0)
    # Variance across K rollouts, mean over spatial dims
    var = stacked.var(dim=0).mean()
    return -torch.log(var + 1e-8)


def reference_anchor_loss(
    pi_output: Tensor,
    ref_output: Tensor,
) -> Tensor:
    """MSE anchor between policy and reference model outputs.

    Prevents policy from drifting too far from the base model.
    ref_output should be detached (no gradients through reference).

    Args:
        pi_output: (B, C, H, W) policy model output.
        ref_output: (B, C, H, W) reference model output (detached).

    Returns:
        Scalar MSE loss.
    """
    return torch.nn.functional.mse_loss(pi_output, ref_output.detach())


def drgrpo_diffusion_loss(
    rewards: Tensor,
    pi_denoised_per_rollout: list[list[Tensor]],
    ref_denoised_per_rollout: list[list[Tensor]],
    sigmas: Tensor,
    pi_output: Tensor | None = None,
    ref_output: Tensor | None = None,
    clip_low: float = 0.2,
    clip_high: float = 0.28,
    lambda_ent: float = 0.01,
    lambda_anchor: float = 1e-4,
) -> dict[str, Tensor]:
    """Assemble all DRGRPO loss components.

    Args:
        rewards: (K,) BTRM scores for K rollouts.
        pi_denoised_per_rollout: K lists of T denoised predictions from policy.
        ref_denoised_per_rollout: K lists of T denoised predictions from reference.
        sigmas: (T+1,) sigma schedule.
        pi_output: Optional (1, C, H, W) policy output for anchor loss.
        ref_output: Optional (1, C, H, W) reference output for anchor loss.
        clip_low: Lower asymmetric clip bound.
        clip_high: Upper asymmetric clip bound.
        lambda_ent: Entropy bonus weight.
        lambda_anchor: Reference anchor weight.

    Returns:
        Dict with keys: "loss", "policy_loss", "entropy_loss", "anchor_loss",
        "advantages", "mean_reward".
    """
    K = len(pi_denoised_per_rollout)

    # Advantages
    advantages = compute_group_advantages(rewards)

    # Per-rollout log-ratios
    log_ratios = []
    for k in range(K):
        lr = compute_step_log_ratios(
            pi_denoised_per_rollout[k],
            ref_denoised_per_rollout[k],
            sigmas,
        )
        log_ratios.append(lr)

    # Policy loss
    L_policy = clipped_policy_loss(log_ratios, advantages, clip_low, clip_high)

    # Entropy bonus (use last denoised of each rollout)
    last_denoised = [pi_denoised_per_rollout[k][-1] for k in range(K)]
    L_entropy = latent_entropy_bonus(last_denoised)

    # Reference anchor
    L_anchor = rewards.new_zeros(())
    if pi_output is not None and ref_output is not None:
        L_anchor = reference_anchor_loss(pi_output, ref_output)

    # Total
    L_total = L_policy + lambda_ent * L_entropy + lambda_anchor * L_anchor

    return {
        "loss": L_total,
        "policy_loss": L_policy,
        "entropy_loss": L_entropy,
        "anchor_loss": L_anchor,
        "advantages": advantages.detach(),
        "mean_reward": rewards.mean().detach(),
    }

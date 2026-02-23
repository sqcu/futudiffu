r"""DDREINFORCE: Policy gradient for diffusion denoising via REINFORCE.

Flow-matching parameterization. The model predicts a velocity field v_θ such
that the deterministic Euler ODE step is:

    x_{t-1} = x_t + dt · v_θ(x_t, c, t)       where dt = σ_{t-1} - σ_t

To define a stochastic policy with computable log-probabilities, we inject
isotropic Gaussian noise at each step (converting the ODE to an SDE):

    x_{t-1} = μ_θ(x_t, c, t) + η_t · z        z ~ N(0, I)

where μ_θ = x_t + dt · v_θ is the deterministic mean and η_t is a
per-step noise scale proportional to |dt|:

    η_t = η_scale · |σ_{t-1} - σ_t|

The denoising policy is therefore a conditional Gaussian:

    π_θ(x_{t-1} | x_t, c) = N(x_{t-1} ; μ_θ(x_t, c, t), η_t² I)

and the log-probability of a realized transition is:

    log π_θ(x_{t-1} | x_t, c) = -‖x_{t-1} - μ_θ‖² / (2η_t²) + const

The const = -(d/2)log(2πη_t²) is independent of θ and drops from all
gradient expressions.

IMPORTANT: η_t² is the denominator, NOT σ_t². The frozen codebase
(src/futudiffu/policy_loss.py) uses σ_t², which is the DDPM convention.
Flow-matching models use the injected noise scale η_t as the policy
variance, because the Euler ODE itself is deterministic and has no
intrinsic stochasticity to define a density.

────────────────────────────────────────────────────────────────────────
Algorithm 1: DDREINFORCE (flow-matching parameterization)
────────────────────────────────────────────────────────────────────────

  Input: π_θ (policy, trainable LoRA adapter on frozen backbone)
         π_ref (reference policy = backbone with adapter_scale=0)
         r(·) (reward function, e.g. BTRM score head)
         K (group size, trajectories per prompt)
         T (denoising steps)
         η_scale (noise injection scale)
         β (KL penalty coefficient)
         B (prompts per batch)

  For each training iteration:
    1. Sample prompts {c_1, ..., c_B}
    2. For each prompt c_b, generate K stochastic trajectories:
       For k = 1..K:
         a. x_T^k ~ σ_0 · N(0, I)
         b. For t = T..1:
              μ_θ^k = x_t^k + (σ_{t-1} - σ_t) · v_θ(x_t^k, c_b, t)
              η_t = η_scale · |σ_{t-1} - σ_t|
              z_t^k ~ N(0, I)
              x_{t-1}^k = μ_θ^k + η_t · z_t^k
         c. x_0^k = μ_θ^k                    (final step: no noise)
    3. Score: r_k = r(x_0^k, c_b) for k = 1..K
    4. Group-relative advantage:
         Â_k = (r_k − mean(r_1..r_K)) / (std(r_1..r_K) + ε)
    5. Per-step log-prob (for θ_current, not θ_old — on-policy):
         log π_θ(x_{t-1}^k | x_t^k, c_b) = −‖x_{t-1}^k − μ_θ^k‖² / (2η_t²)
    6. Per-step KL penalty (unbiased estimator):
         μ_ref^k = x_t^k + dt · v_ref(x_t^k, c_b, t)   [adapter_scale=0]
         log_ratio_ref = −‖x_{t-1}^k − μ_ref^k‖² / (2η_t²)
                       − (−‖x_{t-1}^k − μ_θ^k‖² / (2η_t²))
         D_KL^t = exp(log_ratio_ref) − log_ratio_ref − 1     (≥ 0)
    7. Step-weighted policy gradient:
         ∇_θ J ≈ (1/KB) Σ_{b,k} Â_k · Σ_t w(σ_t) · ∇_θ log π_θ · (−1)
                + β · Σ_t w(σ_t) · D_KL^t
       where w(σ_t) = logsnr_sampling_weight(σ_t).
    8. Optimizer step on θ (LoRA adapter params only).
────────────────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────────────────
Algorithm 2: DDREINFORCE (DDPM ε-prediction parameterization, for SDXL)
────────────────────────────────────────────────────────────────────────

  SDXL uses a DDPM noise schedule with learned ε-prediction. The reverse
  process at each step is:

    p_θ(x_{t-1} | x_t) = N(x_{t-1} ; μ_θ(x_t, t), β_t I)

  where β_t is the DDPM variance schedule and:

    μ_θ(x_t, t) = (1/√α_t) · (x_t − (β_t/√(1−ᾱ_t)) · ε_θ(x_t, t))

  The log-probability denominator is β_t (the DDPM variance), NOT an
  injected η_t. This is because the DDPM reverse process IS inherently
  stochastic — the variance β_t is a structural property of the forward
  process, not a hyperparameter.

    log p_θ(x_{t-1} | x_t) = −‖x_{t-1} − μ_θ(x_t, t)‖² / (2β_t) + const

  The DDPM schedule defines:
    α_t = 1 − β_t
    ᾱ_t = Π_{s=1}^{t} α_s
    σ_t² = β_t                          (for "fixed small" variance)
         = β̃_t = β_t · (1−ᾱ_{t-1})/(1−ᾱ_t)   (for "fixed large" variance)

  SDXL typically uses the "fixed small" variance σ_t² = β_t for sampling
  and the "fixed large" for training. For DDREINFORCE, use whichever
  variance was used during trajectory generation as the denominator.

  Everything else (group advantage, KL penalty, step weighting) is
  identical to Algorithm 1. The only difference is the log-prob
  denominator: η_t² (flow matching) vs β_t (DDPM).
────────────────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────────────────
On the role of the SDE / "churn" noise
────────────────────────────────────────────────────────────────────────

Why not just use K different random seeds with a deterministic ODE?

K different seeds give you K different trajectories and K different
rewards, which is enough to compute group advantages Â_k. But
REINFORCE also needs ∇_θ log π_θ(τ) — the score function of the
policy — to turn those advantages into parameter gradients. A
deterministic ODE step x_{t-1} = f_θ(x_t) is a Dirac delta, which
has no density and no score function. Without a well-defined
log π_θ(x_{t-1} | x_t), the policy gradient theorem does not apply
and "REINFORCE" is not REINFORCE — it degenerates into reward-weighted
regression on the MSE loss, which is the weaker RWL baseline that DDPO
empirically outperforms.

The injected noise converts the Dirac into a peaked-but-proper Gaussian
N(μ_θ, η_t² I). This gives each step a computable log-probability
whose gradient ∇_θ log π_θ depends on *where the sample actually
landed* relative to the predicted mean. RWL cannot condition on this
information; DDPO can, and the DDPO paper shows this matters.

The churn does NOT exist to create trajectory diversity (seeds do that).
It exists to make the per-step transition a valid probability
distribution so the score function estimator is well-defined.

On-policy consistency: the policy π_θ is defined as "Euler + churn
with schedule η_t." Training samples from this policy. Evaluation /
deployment also samples from this policy with the same η schedule.
There is no train-test mismatch. Euler-with-churn is a known sampling
strategy in diffusion model deployment (the "stochastic sampler"
family); it does not produce obviously worse sample quality than
deterministic ODE integration, and at small η the difference is
subtle. We are training on-policy for the thing we actually deploy.

For DDPM models (Algorithm 2), the stochasticity is structural — β_t
comes from the forward process, not a hyperparameter. The reverse
process is *inherently* a Gaussian transition, so there is no need to
"inject" noise; it's already there. This is why the DDPM
parameterization is more natural for policy gradients, and why the
flow-matching case requires the explicit η_t injection.
────────────────────────────────────────────────────────────────────────

Import constraints:
  - torch only (pure math module)
  - No futudiffu imports
  - No src_ii imports except sigma_schedule (for logsnr computation)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core math: advantage normalization
# ---------------------------------------------------------------------------

def group_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Group-relative advantage normalization (GRPO Eq. 1).

    Z-scores rewards within a group of K trajectories for the same prompt.

    Args:
        rewards: (K,) scalar rewards for K trajectories of one prompt.
        eps: Epsilon for numerical stability in std division.

    Returns:
        (K,) normalized advantages with mean≈0, std≈1.
        Returns zeros if K < 2 (no group to normalize against).
    """
    if rewards.numel() < 2:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / (rewards.std() + eps)


# ---------------------------------------------------------------------------
# Core math: per-step Gaussian log-probability
# ---------------------------------------------------------------------------

def step_log_prob(
    x_next: torch.Tensor,
    mu_theta: torch.Tensor,
    eta_t: float,
) -> torch.Tensor:
    """Per-step log π_θ(x_{t-1} | x_t, c) under isotropic Gaussian policy.

    Flow-matching parameterization: denominator is η_t² (injected noise),
    NOT σ_t² (noise schedule level).

    Computed in FP32 to avoid underflow when η_t is small (near σ=0).

    Args:
        x_next: (B, C, H, W) realized next state x_{t-1}.
        mu_theta: (B, C, H, W) predicted mean μ_θ(x_t, c, t).
        eta_t: Scalar noise scale for this step.

    Returns:
        (B,) per-batch log-probability (up to additive constant).
    """
    diff = (x_next.float() - mu_theta.float()).flatten(1)
    sq_norm = (diff * diff).mean(dim=1)
    return -sq_norm / (2.0 * eta_t * eta_t)


# ---------------------------------------------------------------------------
# Core math: importance ratio and KL
# ---------------------------------------------------------------------------

def step_log_ratio(
    x_next: torch.Tensor,
    mu_theta: torch.Tensor,
    mu_old: torch.Tensor,
    eta_t: float,
) -> torch.Tensor:
    """Per-step log importance ratio log(π_θ / π_old).

    Since both policies share the same variance η_t²:

        log ρ_t = (‖x − μ_old‖² − ‖x − μ_θ‖²) / (2η_t²)

    The σ_t² terms cancel because the variance is fixed.

    Args:
        x_next: (B, C, H, W) realized next state.
        mu_theta: (B, C, H, W) current policy mean.
        mu_old: (B, C, H, W) old/reference policy mean.
        eta_t: Noise scale.

    Returns:
        (B,) per-batch log-ratio.
    """
    x = x_next.float().flatten(1)
    mu_new = mu_theta.float().flatten(1)
    mu_o = mu_old.float().flatten(1)
    sq_old = ((x - mu_o) ** 2).mean(1)
    sq_new = ((x - mu_new) ** 2).mean(1)
    return (sq_old - sq_new) / (2.0 * eta_t * eta_t)


def unbiased_kl(log_ratio_ref_over_theta: torch.Tensor) -> torch.Tensor:
    """Unbiased KL divergence estimator (DeepSeek-V3).

    D_KL(π_θ ‖ π_ref) ≈ exp(x) − x − 1   where x = log(π_ref / π_θ)

    Guaranteed non-negative. Well-conditioned near x=0 (π_θ ≈ π_ref).

    Args:
        log_ratio_ref_over_theta: (B,) log(π_ref / π_θ) per sample.

    Returns:
        (B,) per-sample KL estimate.
    """
    x = log_ratio_ref_over_theta.float()
    return torch.exp(x) - x - 1.0


# ---------------------------------------------------------------------------
# Core math: clipped surrogate loss (PPO / GRPO)
# ---------------------------------------------------------------------------

def clipped_surrogate_loss(
    log_ratio: torch.Tensor,
    advantage: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Per-step PPO clipped surrogate loss.

    L = -min(ρ · Â, clip(ρ, 1-ε, 1+ε) · Â)

    Applied PER-STEP (not per-trajectory). The frozen codebase incorrectly
    sums log-ratios across steps into a trajectory-level ratio before
    clipping; this function operates on individual step ratios.

    Args:
        log_ratio: (B,) log(π_θ / π_old) for one step.
        advantage: (B,) or scalar advantage.
        clip_eps: Clipping range (symmetric).

    Returns:
        (B,) per-sample clipped loss (to be minimized).
    """
    ratio = torch.exp(log_ratio.float())
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surr1 = ratio * advantage
    surr2 = clipped * advantage
    return -torch.min(surr1, surr2)


# ---------------------------------------------------------------------------
# Step weighting
# ---------------------------------------------------------------------------

def logsnr_step_weight(
    sigma: float,
    threshold: float = 10.0,
    interval: float = 5.0,
    decay_rate: float = 0.5,
) -> float:
    """LogSNR-based step weight. Identical to pair_sampler.logsnr_sampling_weight.

    Duplicated here to keep this module zero-import from other src_ii/ modules.

    sigma=0 → weight=1.0 (fully denoised, strongest reward signal).
    sigma with logSNR >= threshold → weight=1.0 (clean regime).
    Below threshold → geometric decay: decay_rate^((threshold - logSNR) / interval).
    """
    if sigma <= 0.0:
        return 1.0
    if sigma >= 1.0:
        return decay_rate ** (threshold / interval)
    logsnr = 2.0 * math.log((1.0 - sigma) / sigma)
    if logsnr >= threshold:
        return 1.0
    return decay_rate ** ((threshold - logsnr) / interval)


# ---------------------------------------------------------------------------
# Noise injection schedule
# ---------------------------------------------------------------------------

def compute_eta_schedule(
    sigmas: torch.Tensor,
    eta_scale: float = 0.1,
) -> list[float]:
    """Compute per-step noise injection scales η_t = η_scale · |dt|.

    The noise is proportional to the Euler step size |σ_{t-1} - σ_t|.
    This keeps the stochastic perturbation commensurate with the signal
    change at each step. Large steps (noisy regime) get more noise;
    small steps (clean regime) get less.

    Args:
        sigmas: (T+1,) sigma schedule from build_sigma_schedule.
        eta_scale: Global noise multiplier. 0.0 = deterministic ODE.

    Returns:
        List of T floats, one η_t per Euler step.
    """
    T = len(sigmas) - 1
    etas = []
    for i in range(T):
        dt = abs(float(sigmas[i + 1] - sigmas[i]))
        etas.append(eta_scale * dt)
    return etas


# ---------------------------------------------------------------------------
# Stochastic Euler SDE solver
# ---------------------------------------------------------------------------

@torch.no_grad()
def euler_sde_generate(
    model_fn,
    x_init: torch.Tensor,
    sigmas: torch.Tensor,
    eta_schedule: list[float],
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[float]]:
    """Generate a stochastic trajectory (no gradient, for on-policy generation).

    The noise η_t · z is NOT for trajectory diversity (use different seeds
    for that). It converts each deterministic Euler step into a proper
    Gaussian transition π_θ(x_{t-1} | x_t) = N(μ_θ, η_t² I), which is
    required for the REINFORCE score function ∇_θ log π_θ to be defined.
    Without it, the ODE step is a Dirac delta with no density.

    At each step:
        μ_θ = x_t + dt · v_θ(x_t, t)
        x_{t-1} = μ_θ + η_t · z       (z ~ N(0, I), except final step)

    Records (x_{t-1}, μ_θ, η_t) at every step for later log-prob computation.

    Args:
        model_fn: Callable(x, sigma) -> denoised_prediction.
            The model predicts the denoised image, NOT the velocity.
            The velocity is recovered as v = (x - denoised) / sigma.
            This matches the ZImageRLAIF convention (returns -img, negated
            in postprocessing, so the caller provides the corrected fn).
        x_init: (B, C, H, W) initial noisy latent (x_T = σ_0 · noise).
        sigmas: (T+1,) sigma schedule.
        eta_schedule: T noise scales from compute_eta_schedule.

    Returns:
        x_trajectory: List of T+1 tensors [x_T, x_{T-1}, ..., x_0].
        mu_trajectory: List of T tensors [μ_T, μ_{T-1}, ..., μ_1].
            mu_trajectory[i] is the deterministic mean at step i.
        eta_used: List of T floats (same as eta_schedule but with
            eta_used[-1]=0.0 for the final step).
    """
    T = len(sigmas) - 1
    x = x_init
    x_trajectory = [x.cpu()]
    mu_trajectory = []
    eta_used = []

    for i in range(T):
        sigma_i = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma_i

        # Model predicts denoised; velocity = (x - denoised) / sigma
        denoised = model_fn(x, sigma_i)
        d = (x - denoised) / sigma_i
        mu = x + d * dt  # deterministic Euler mean

        eta_t = eta_schedule[i]
        is_final = (i == T - 1) or (float(sigma_next) == 0.0)

        if is_final or eta_t < 1e-12:
            x_next = mu
            eta_used.append(0.0)
        else:
            z = torch.randn_like(x)
            x_next = mu + eta_t * z
            eta_used.append(eta_t)

        mu_trajectory.append(mu.cpu())
        x = x_next
        x_trajectory.append(x.cpu())

    return x_trajectory, mu_trajectory, eta_used


# ---------------------------------------------------------------------------
# Log-prob recomputation (differentiable, for gradient accumulation)
# ---------------------------------------------------------------------------

def recompute_step_log_prob(
    model_fn,
    x_t: torch.Tensor,
    x_next: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_next: torch.Tensor,
    eta_t: float,
) -> torch.Tensor:
    """Recompute log π_θ for a single step under the CURRENT policy.

    This is the differentiable path: gradients flow through model_fn
    into the LoRA adapter parameters.

    Args:
        model_fn: Differentiable callable(x, sigma) -> denoised.
        x_t: (B, C, H, W) noisy input (detached, from recorded trajectory).
        x_next: (B, C, H, W) realized next state (detached).
        sigma_t: Current sigma.
        sigma_next: Next sigma.
        eta_t: Noise scale used during generation.

    Returns:
        (B,) log-probability with grad_fn through model_fn.
    """
    dt = sigma_next - sigma_t
    denoised = model_fn(x_t, sigma_t)
    d = (x_t - denoised) / sigma_t
    mu_theta = x_t + d * dt
    return step_log_prob(x_next.detach(), mu_theta, eta_t)


# ---------------------------------------------------------------------------
# DDREINFORCE loss for one prompt group
# ---------------------------------------------------------------------------

def ddreinforce_loss(
    advantages: torch.Tensor,
    log_probs_per_traj: list[list[torch.Tensor]],
    kl_per_traj: list[list[torch.Tensor]],
    step_weights: list[float],
    beta: float = 0.04,
) -> torch.Tensor:
    """Compute the on-policy DDREINFORCE loss for one prompt group.

    On-policy: importance ratios are identically 1, clipping is vacuous.
    This is the simplified DDGRPO (Section 4.4 of the spec):

        L = -(1/K) Σ_k Â_k · Σ_t w_t · log π_θ(x_{t-1}^k | x_t^k)
            + β · (1/K) Σ_k Σ_t w_t · D_KL^t

    For off-policy (trajectory reuse with PPO clipping), use ddgrpo_loss().

    Args:
        advantages: (K,) group-relative advantages.
        log_probs_per_traj: K lists of T tensors, each (B,) log-prob.
        kl_per_traj: K lists of T tensors, each (B,) KL penalty.
        step_weights: T floats from logsnr_step_weight.
        beta: KL penalty coefficient.

    Returns:
        Scalar loss (to be minimized via .backward()).
    """
    K = len(log_probs_per_traj)
    T = len(log_probs_per_traj[0])
    assert len(step_weights) == T

    total_loss = torch.tensor(0.0, device=log_probs_per_traj[0][0].device)

    for k in range(K):
        adv_k = advantages[k]
        for t in range(T):
            w_t = step_weights[t]
            lp = log_probs_per_traj[k][t]  # (B,), has grad_fn
            kl = kl_per_traj[k][t]  # (B,), detached or no-grad

            # Policy gradient: maximize log_prob * advantage
            # Minimize: -log_prob * advantage
            policy_term = -adv_k * lp.mean() * w_t
            kl_term = beta * kl.mean() * w_t
            total_loss = total_loss + policy_term + kl_term

    return total_loss / K


# ---------------------------------------------------------------------------
# DDGRPO loss (off-policy, with PPO clipping)
# ---------------------------------------------------------------------------

def ddgrpo_loss(
    advantages: torch.Tensor,
    log_probs_new: list[list[torch.Tensor]],
    log_probs_old: list[list[float]],
    kl_per_traj: list[list[torch.Tensor]],
    step_weights: list[float],
    beta: float = 0.04,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Compute the full DDGRPO loss with per-step PPO clipping.

    Off-policy: uses importance ratios ρ_{k,t} = π_θ / π_old with
    per-step clipping (NOT trajectory-level clipping).

    Args:
        advantages: (K,) group-relative advantages.
        log_probs_new: K lists of T tensors, each (B,) — current policy.
        log_probs_old: K lists of T floats — old policy (from generation).
        kl_per_traj: K lists of T tensors, each (B,) KL penalty.
        step_weights: T floats.
        beta: KL coefficient.
        clip_eps: PPO clip range.

    Returns:
        Scalar loss.
    """
    K = len(log_probs_new)
    T = len(log_probs_new[0])

    total_loss = torch.tensor(0.0, device=log_probs_new[0][0].device)

    for k in range(K):
        adv_k = advantages[k]
        for t in range(T):
            w_t = step_weights[t]
            lp_new = log_probs_new[k][t].mean()
            lp_old = log_probs_old[k][t]
            log_ratio = lp_new - lp_old

            clip_loss = clipped_surrogate_loss(
                log_ratio.unsqueeze(0),
                adv_k.unsqueeze(0),
                clip_eps=clip_eps,
            ).squeeze(0)

            kl_term = beta * kl_per_traj[k][t].mean()
            total_loss = total_loss + (clip_loss + kl_term) * w_t

    return total_loss / K


# ---------------------------------------------------------------------------
# Reward composition
# ---------------------------------------------------------------------------

def compose_rewards(
    scores: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combine multi-head BTRM scores into a scalar reward.

    Args:
        scores: (K, n_heads) from the BTRM score head.
        weights: (n_heads,) per-head weights. If None, equal weights.

    Returns:
        (K,) scalar rewards.
    """
    if weights is None:
        return scores.mean(dim=1)
    return (scores * weights.unsqueeze(0)).sum(dim=1)

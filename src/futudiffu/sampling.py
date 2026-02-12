"""Sampling math ported from ComfyUI.

Sources:
- comfy/model_sampling.py (CONST, ModelSamplingDiscreteFlow, time_snr_shift)
- comfy/samplers.py (simple_scheduler)
- comfy/k_diffusion/sampling.py (sample_euler, to_d)
- comfy/latent_formats.py (Flux)
"""

import torch


# --- model_sampling.py ---

def time_snr_shift(alpha: float, t: torch.Tensor) -> torch.Tensor:
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


def build_sigmas(shift: float = 1.0, multiplier: float = 1000.0, timesteps: int = 1000) -> torch.Tensor:
    """Build the sigma table for ModelSamplingDiscreteFlow."""
    ts = torch.arange(1, timesteps + 1, 1, dtype=torch.float32) / timesteps
    ts = ts * multiplier
    sigmas = time_snr_shift(shift, ts / multiplier)
    return sigmas


def simple_scheduler(sigmas: torch.Tensor, steps: int) -> torch.Tensor:
    """Port of comfy/samplers.py:simple_scheduler.

    Args:
        sigmas: 1D tensor of 1000 sigma values from build_sigmas.
        steps: Number of sampling steps.

    Returns:
        Tensor of (steps+1,) sigma values, ending with 0.0.
    """
    sigs = []
    ss = len(sigmas) / steps
    for x in range(steps):
        sigs.append(float(sigmas[-(1 + int(x * ss))]))
    sigs.append(0.0)
    return torch.FloatTensor(sigs)


# --- CONST noise model (comfy/model_sampling.py:62-76) ---

def const_noise_scaling(sigma: torch.Tensor, noise: torch.Tensor, latent_image: torch.Tensor) -> torch.Tensor:
    """CONST.noise_scaling: sigma * noise + (1 - sigma) * latent_image"""
    sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
    return sigma * noise + (1 - sigma) * latent_image


def const_calculate_denoised(sigma: torch.Tensor, model_output: torch.Tensor, model_input: torch.Tensor) -> torch.Tensor:
    """CONST.calculate_denoised: model_input - model_output * sigma"""
    sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
    return model_input - model_output * sigma


def const_inverse_noise_scaling(sigma: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    """CONST.inverse_noise_scaling: latent / (1 - sigma)"""
    sigma = sigma.view(sigma.shape[:1] + (1,) * (latent.ndim - 1))
    return latent / (1 - sigma)


# --- k_diffusion sampling ---

def to_d(x: torch.Tensor, sigma: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    """Converts a denoiser output to a Karras ODE derivative."""
    sigma = sigma.view(sigma.shape[:1] + (1,) * (x.ndim - 1))
    return (x - denoised) / sigma


@torch.inference_mode()
def sample_euler(
    model_fn,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    callback=None,
) -> torch.Tensor:
    """Euler sampler (Algorithm 2, Karras et al. 2022).

    Args:
        model_fn: Callable (x, sigma) -> denoised.
        x: Noisy latent tensor.
        sigmas: 1D tensor of (steps+1,) sigma values.
        callback: Optional callback(dict) per step.

    Returns:
        Denoised latent tensor.
    """
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        sigma_hat = sigmas[i]
        denoised = model_fn(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt
    return x


# --- Flux latent format ---

FLUX_SCALE_FACTOR = 0.3611
FLUX_SHIFT_FACTOR = 0.1159


def flux_process_in(latent: torch.Tensor) -> torch.Tensor:
    """Flux.process_in: (latent - shift) * scale"""
    return (latent - FLUX_SHIFT_FACTOR) * FLUX_SCALE_FACTOR


def flux_process_out(latent: torch.Tensor) -> torch.Tensor:
    """Flux.process_out: (latent / scale) + shift"""
    return (latent / FLUX_SCALE_FACTOR) + FLUX_SHIFT_FACTOR


# --- Training-mode sampling (QAT / LoRA / REINFORCE) ---

import math
from torch.utils.checkpoint import checkpoint as grad_checkpoint


def sample_euler_train(
    model_fn,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    s_churn: float = 0.0,
    callback=None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Euler sampler with gradient flow for QAT / LoRA training.

    Same euler integration as sample_euler but without inference_mode, so
    gradients flow through the entire trajectory. The model_fn call at each
    step is wrapped in activation checkpointing to trade compute for memory
    (~10x savings: only x_t tensors are stored, not all 34-layer activations).

    Optionally adds stochastic churn (s_churn > 0) for REINFORCE rollout
    diversity following Algorithm 2 from Karras et al. 2022.

    Args:
        model_fn: Callable (x, sigma) -> denoised. May be torch.compiled.
        x: Noisy latent tensor (requires_grad must be True if you want grads).
        sigmas: 1D tensor of (steps+1,) sigma values.
        s_churn: Stochastic churn amount. 0 = deterministic (identical to
            standard euler).
        callback: Optional callback(dict) per step.

    Returns:
        Tuple of:
            - Final latent tensor, retaining its gradient graph back through
              the checkpointed steps.
            - List of detached x tensors at each step boundary [x_0, ..., x_T]
              for sparse step attribution in REINFORCE.
    """
    s_in = x.new_ones([x.shape[0]])
    checkpoints: list[torch.Tensor] = []
    n_steps = len(sigmas) - 1

    for i in range(n_steps):
        # Record step boundary checkpoint (detached for REINFORCE attribution)
        checkpoints.append(x.detach().clone())

        sigma = sigmas[i]

        # Stochastic churn (optional, for rollout diversity)
        if s_churn > 0.0:
            gamma = min(s_churn / n_steps, math.sqrt(2.0) - 1.0)
            sigma_hat = sigma + gamma * sigma
            noise = torch.randn_like(x)
            x = x + (sigma_hat ** 2 - sigma ** 2).sqrt() * noise
        else:
            sigma_hat = sigma

        # Activation-checkpointed model call: recomputes forward during
        # backward instead of storing all intermediate activations.
        denoised = grad_checkpoint(
            model_fn,
            x,
            sigma_hat * s_in,
            use_reentrant=False,
        )

        d = to_d(x, sigma_hat, denoised)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})

        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt

    # Final boundary checkpoint
    checkpoints.append(x.detach().clone())

    return x, checkpoints


def sparse_step_loss(
    checkpoints: list[torch.Tensor],
    model_fn,
    sigmas: torch.Tensor,
    sampled_steps: list[int],
) -> torch.Tensor:
    """Compute a differentiable loss surrogate over a sparse subset of steps.

    Instead of backpropagating through all 30 euler steps, this runs model_fn
    on only the sampled steps (with gradients) and returns the sum of per-step
    denoised output norms. This serves as a loss surrogate for REINFORCE-style
    training where full-trajectory gradients are too expensive.

    Each sampled step re-evaluates model_fn(x_t, sigma_t) from the stored
    checkpoint, so gradients flow through the model but NOT through the
    inter-step euler integration.

    Args:
        checkpoints: List of [x_0, ..., x_T] detached tensors from
            sample_euler_train (length = steps + 1).
        model_fn: Callable (x, sigma) -> denoised. Same function used in
            sampling. May be torch.compiled.
        sigmas: 1D tensor of (steps+1,) sigma values (same as passed to
            sample_euler_train).
        sampled_steps: List of step indices to evaluate (e.g. [3, 12, 27]).
            Each index i means: run model_fn(checkpoints[i], sigmas[i]).

    Returns:
        Scalar tensor: sum of per-step denoised L2 norms. Differentiable
        w.r.t. model_fn parameters (LoRA adapters).
    """
    s_in = checkpoints[0].new_ones([checkpoints[0].shape[0]])
    loss = checkpoints[0].new_zeros(())

    for i in sampled_steps:
        x_t = checkpoints[i].detach().requires_grad_(False)
        sigma = sigmas[i]

        # Re-run model with gradients flowing through model parameters
        denoised = model_fn(x_t, sigma * s_in)

        # L2 norm of denoised output as loss surrogate
        loss = loss + denoised.norm()

    return loss

"""Neural function evaluation (NFE) and denoising conversion.

These are the two lowest layers in the five-function decomposition:
  nfe() -- wraps NextDiT.forward() or forward_packed(), handles sigma->timestep
  denoise() -- CONST.calculate_denoised, pure function

Import constraints:
  - IMPORTS from futudiffu.diffusion_model: NextDiT (the model class)
  - DOES NOT import: model_manager, server, client, sampling, training_utils
"""

import torch


def nfe(
    model,
    x: torch.Tensor,
    sigma: torch.Tensor,
    conditioning: torch.Tensor,
    num_tokens: int,
    rope_cache: dict,
    multiplier: float = 1.0,
) -> torch.Tensor:
    """Single neural function evaluation.

    Wraps NextDiT.forward() with the sigma-to-timestep conversion.
    The model returns a NEGATED prediction (ComfyUI convention).

    Args:
        model: NextDiT model (compiled or raw). Must accept the forward() signature.
        x: (B, C, H, W) noisy latent.
        sigma: (B,) or scalar sigma values.
        conditioning: (B, seq, cap_feat_dim) text encoder hidden states.
        num_tokens: Number of text tokens (padded sequence length).
        rope_cache: Precomputed RoPE from model.prepare_rope_cache().
        multiplier: Timestep multiplier (default 1.0).

    Returns:
        (B, C, H, W) raw model prediction (NEGATED, per ComfyUI convention).
    """
    # sigma -> timestep conversion happens exactly here, nowhere else
    timestep = sigma * multiplier
    return model(
        x, timestep, conditioning,
        num_tokens=num_tokens, rope_cache=rope_cache,
    )


def denoise(
    raw_prediction: torch.Tensor,
    sigma: torch.Tensor,
    model_input: torch.Tensor,
) -> torch.Tensor:
    """CONST.calculate_denoised: convert raw model output to denoised estimate.

    Pure function. No state, no model reference, no conditional logic.
    This is the type conversion that makes the ODE derivative well-defined:
    raw_prediction (negated model output) -> denoised latent.

    Formula: model_input - raw_prediction * sigma

    Args:
        raw_prediction: (B, C, H, W) raw model output from nfe().
        sigma: (B,) or scalar sigma value. Will be broadcast.
        model_input: (B, C, H, W) the noisy latent that was fed to nfe().

    Returns:
        (B, C, H, W) denoised estimate.
    """
    sigma = sigma.view(sigma.shape[:1] + (1,) * (raw_prediction.ndim - 1))
    return model_input - raw_prediction * sigma

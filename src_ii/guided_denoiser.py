"""Guided denoiser construction: make_guided_denoiser().

The third layer in the five-function decomposition.
Takes a model + conditioning and returns a (x, sigma) -> denoised closure.
Supports both CFG (2 NFE per call) and cfg=1.0 (1 NFE per call).

Import constraints:
  - IMPORTS from src_ii.forward: nfe, denoise
  - DOES NOT import: model_manager, server, client, sampling, training_utils
"""

import torch

from .forward import nfe, denoise


def pad_and_batch_cond(
    pos_cond: torch.Tensor,
    neg_cond: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Pad pos/neg conditioning to same sequence length and stack for batched CFG.

    Args:
        pos_cond: (1, pos_len, dim) positive conditioning.
        neg_cond: (1, neg_len, dim) negative conditioning.

    Returns:
        cond_batch: (2, max_len, dim) with pos at [0] and neg at [1].
        num_tokens: The padded sequence length.
    """
    pos_len = pos_cond.shape[1]
    neg_len = neg_cond.shape[1]
    max_len = max(pos_len, neg_len)
    if pos_len < max_len:
        pos_cond = torch.nn.functional.pad(pos_cond, (0, 0, 0, max_len - pos_len))
    if neg_len < max_len:
        neg_cond = torch.nn.functional.pad(neg_cond, (0, 0, 0, max_len - neg_len))
    cond_batch = torch.cat([pos_cond, neg_cond], dim=0)
    return cond_batch, max_len


def make_guided_denoiser(
    model,
    pos_cond: torch.Tensor,
    neg_cond: torch.Tensor,
    cfg_scale: float,
    rope_cache: dict,
    multiplier: float = 1.0,
):
    """Create a guided denoiser closure: (x, sigma) -> denoised.

    This replaces both make_cfg_model_fn and build_cfg_model_fn from the
    original codebase. The returned closure's type signature is always
    (x, sigma) -> denoised, regardless of guidance strategy.

    When cfg_scale == 1.0: single NFE with pos_cond only.
    When cfg_scale != 1.0: two NFEs (batched), CFG interpolation.

    Args:
        model: NextDiT model (compiled or raw).
        pos_cond: (1, pos_len, dim) positive conditioning.
        neg_cond: (1, neg_len, dim) negative conditioning.
        cfg_scale: Classifier-free guidance scale.
        rope_cache: Precomputed RoPE cache.
        multiplier: Timestep multiplier.

    Returns:
        Callable (x: Tensor, sigma: Tensor) -> denoised: Tensor
    """
    if cfg_scale == 1.0:
        # No guidance: single NFE with positive conditioning only
        # Still need to know num_tokens for the model
        pos_len = pos_cond.shape[1]
        # Pad pos_cond to match what prepare_rope_cache was built with
        # In the cfg=1.0 case, the rope_cache was built for pos_cond's length

        def denoiser_fn(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            raw = nfe(model, x, sigma, pos_cond, pos_len, rope_cache, multiplier)
            return denoise(raw, sigma, x)

        return denoiser_fn
    else:
        # CFG: batch pos+neg, run once, split, interpolate
        cond_batch, num_tokens = pad_and_batch_cond(pos_cond, neg_cond)

        def denoiser_fn(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            # Expand x and sigma for batch=2 (pos, neg)
            x_batch = x.expand(2, -1, -1, -1)
            timestep = sigma * multiplier
            t_batch = timestep.expand(2)

            # Single batched forward through the model
            output_batch = model(
                x_batch, t_batch, cond_batch,
                num_tokens=num_tokens, rope_cache=rope_cache,
            )

            out_cond, out_uncond = output_batch.chunk(2, dim=0)
            denoised_cond = denoise(out_cond, sigma, x)
            denoised_uncond = denoise(out_uncond, sigma, x)
            return denoised_uncond + (denoised_cond - denoised_uncond) * cfg_scale

        return denoiser_fn

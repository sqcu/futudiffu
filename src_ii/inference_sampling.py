"""Inference sampling primitives for src_ii/ models.

Replaces imports from frozen src/futudiffu/sampling.py for the inference
pipeline. Contains the math and orchestration needed for inference-mode
Euler sampling through packed ZImageRLAIF forward passes.

The key export is `run_trajectory_packed`: drop-in replacement for the
frozen `futudiffu.sampling.run_trajectory_packed`, same params/tensors
contract, but calls through src_ii/forward_packed.py and
src_ii/block_mask.py instead of the frozen model's make_packing_mask_mod.

Import constraints:
  - torch only (no frozen src.futudiffu imports)
  - Reuses src_ii.sigma_schedule, src_ii.forward_packed, src_ii.block_mask
"""

from __future__ import annotations

import torch

from src_ii.sigma_schedule import (
    build_sigma_schedule,
    const_noise_scaling,
    const_inverse_noise_scaling,
    resolution_shift,
)


# ---------------------------------------------------------------------------
# CONST noise model math (ported from src/futudiffu/sampling.py)
# ---------------------------------------------------------------------------

def const_calculate_denoised(
    sigma: torch.Tensor,
    model_output: torch.Tensor,
    model_input: torch.Tensor,
) -> torch.Tensor:
    """CONST.calculate_denoised: model_input - model_output * sigma"""
    sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
    return model_input - model_output * sigma


def to_d(
    x: torch.Tensor,
    sigma: torch.Tensor,
    denoised: torch.Tensor,
) -> torch.Tensor:
    """Converts a denoiser output to a Karras ODE derivative."""
    sigma = sigma.view(sigma.shape[:1] + (1,) * (x.ndim - 1))
    return (x - denoised) / sigma


# ---------------------------------------------------------------------------
# Initial latent preparation
# ---------------------------------------------------------------------------

def prepare_initial_latent(
    seed: int,
    width: int,
    height: int,
    device: torch.device,
    dtype: torch.dtype,
    sigma_start: torch.Tensor,
    source_latent: torch.Tensor | None = None,
    in_channels: int = 16,
    vae_scale: int = 8,
) -> torch.Tensor:
    """Prepare the initial noisy latent for Euler sampling.

    Args:
        seed: PRNG seed for noise generation.
        width: Image width in pixels.
        height: Image height in pixels.
        device: Target device.
        dtype: Target dtype.
        sigma_start: First sigma value (scalar tensor).
        source_latent: Optional (1, C, H, W) for i2i. None = zeros (t2i).
        in_channels: Latent channels (default 16 for Z-Image).
        vae_scale: VAE spatial downscale factor (default 8).

    Returns:
        x: (1, C, latent_H, latent_W) initial noised latent.
    """
    latent_h = height // vae_scale
    latent_w = width // vae_scale

    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        1, in_channels, latent_h, latent_w,
        dtype=dtype, generator=generator, device=device,
    )

    if source_latent is not None:
        clean = source_latent.to(device=device, dtype=dtype)
    else:
        clean = torch.zeros(
            1, in_channels, latent_h, latent_w,
            device=device, dtype=dtype,
        )

    return const_noise_scaling(sigma_start, noise, clean)


# ---------------------------------------------------------------------------
# Packed Euler sampler (inference mode)
# ---------------------------------------------------------------------------

def sample_euler_packed(
    packed_forward_fn,
    x_list: list[torch.Tensor],
    sigmas_list: list[torch.Tensor] | torch.Tensor,
    refined_caps,
    packing_info,
    block_mask: torch.Tensor,
    packed_rope: torch.Tensor,
    cfg: float,
    multiplier: float,
    callback=None,
) -> list[torch.Tensor]:
    """Packed Euler sampler for FlexAttention batch packing.

    Same math as ComfyUI sample_euler but operates on N images packed into
    a single FlexAttention forward pass. Each step:
      1. Expand each x_i for CFG (batch dim 2)
      2. Call packed_forward_fn to get per-image outputs
      3. Apply CFG and Euler step per image

    Supports per-image sigma schedules for mixed-resolution packing.

    Args:
        packed_forward_fn: Callable(x_cfg_list, t_batch, caps, info, mask, rope)
            -> (list[Tensor], scores). Returns per-image output tensors, each
            (2, C, H, W), and a scores tensor (ignored during inference).
        x_list: List of N tensors, each (1, C, H, W).
        sigmas_list: Single (steps+1,) tensor (shared) or list of N such
            tensors (per-image schedules). All must have same length.
        refined_caps: Packed refined caption embeddings.
        packing_info: Packing metadata.
        block_mask: FlexAttention block mask.
        packed_rope: Packed RoPE cache.
        cfg: CFG scale.
        multiplier: Timestep multiplier.
        callback: Optional callback({'i': step_idx, 'n_steps': total_steps})
            called after each step.

    Returns:
        List of N final tensors, each (1, C, H, W), with inverse noise
        scaling already applied.
    """
    n_images = len(x_list)

    if isinstance(sigmas_list, torch.Tensor):
        per_image_sigmas = [sigmas_list] * n_images
    else:
        per_image_sigmas = sigmas_list
        assert len(per_image_sigmas) == n_images

    n_steps = len(per_image_sigmas[0]) - 1

    for step_i in range(n_steps):
        # Representative sigma for the shared timestep embedding
        sigma_representative = per_image_sigmas[0][step_i]
        timestep = sigma_representative * multiplier

        x_cfg = [x_i.expand(2, -1, -1, -1) for x_i in x_list]
        t_batch = timestep.expand(2)

        result = packed_forward_fn(
            x_cfg, t_batch, refined_caps,
            packing_info, block_mask, packed_rope,
        )

        # packed_forward returns (outputs_list, scores) — unpack
        if isinstance(result, tuple):
            outputs = result[0]
        else:
            outputs = result

        for img_i in range(n_images):
            sigma_i = per_image_sigmas[img_i][step_i]
            sigma_i_next = per_image_sigmas[img_i][step_i + 1]

            out_cond, out_uncond = outputs[img_i].chunk(2, dim=0)
            denoised_cond = const_calculate_denoised(
                sigma_i, out_cond, x_list[img_i])
            denoised_uncond = const_calculate_denoised(
                sigma_i, out_uncond, x_list[img_i])
            denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cfg

            d = to_d(x_list[img_i], sigma_i, denoised)
            dt = sigma_i_next - sigma_i
            x_list[img_i] = x_list[img_i] + d * dt

        if callback is not None:
            callback({'i': step_i, 'n_steps': n_steps})

    # Apply inverse noise scaling per image
    for img_i in range(n_images):
        x_list[img_i] = const_inverse_noise_scaling(
            per_image_sigmas[img_i][-1], x_list[img_i])

    return x_list


# ---------------------------------------------------------------------------
# CFG conditioning helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Full trajectory orchestration (replaces frozen run_trajectory_packed)
# ---------------------------------------------------------------------------

def run_trajectory_packed(
    model,
    device: torch.device,
    dtype: torch.dtype,
    params: dict,
    tensors: dict,
    callback=None,
) -> tuple[dict, dict]:
    """Run N packed diffusion trajectories via src_ii FlexAttention.

    Drop-in replacement for frozen futudiffu.sampling.run_trajectory_packed.
    Same params/tensors contract. Uses src_ii/forward_packed for packing
    and block mask construction.

    Args:
        model: ZImageRLAIF model (compiled or raw, has prepare_packed_state).
        device: CUDA device.
        dtype: Working dtype (bf16).
        params: RPC params dict:
            n_images, seeds, n_steps, cfg, multiplier, denoise,
            width/height (int) or widths/heights (list[int]),
            sampling_shift/sampling_shifts (optional).
        tensors: RPC tensors dict:
            neg_cond, pos_cond_0..N-1, optional clean_latent_0..N-1.
        callback: Optional per-step callback({'i', 'n_steps', 'x_list'}).

    Returns:
        (result_tensors, metadata) — result_tensors has "final_0".."final_N-1"
        keys (and optionally "step_SS_II" intermediate keys).
    """
    from src_ii.forward_packed import prepare_packed_forward

    n_images = params["n_images"]
    seeds = params["seeds"]
    n_steps = params["n_steps"]
    cfg = params["cfg"]
    multiplier = params.get("multiplier", 1.0)
    save_steps_param = params.get("save_steps", None)
    denoise = params.get("denoise", 1.0)

    # Resolve per-image resolutions
    if "widths" in params and "heights" in params:
        widths = params["widths"]
        heights = params["heights"]
    else:
        w = params["width"]
        h = params["height"]
        widths = [w] * n_images
        heights = [h] * n_images

    # Resolve per-image sampling shifts: auto-shift from resolution * user modifier.
    # resolution_shift() computes SD3 Eq.23: alpha = sqrt(ref_pixels / target_pixels).
    # User shift (default 1.0) is a multiplier on top — 1.0 = "use model's trained schedule".
    auto_shifts = [resolution_shift(widths[i], heights[i]) for i in range(n_images)]
    if "sampling_shifts" in params:
        user_shifts = params["sampling_shifts"]
    elif "sampling_shift" in params:
        user_shifts = [params["sampling_shift"]] * n_images
    else:
        user_shifts = [1.0] * n_images
    sampling_shifts = [a * u for a, u in zip(auto_shifts, user_shifts)]

    patch_size = model.patch_size

    # Build per-image CFG conditioning batches
    neg_cond = tensors["neg_cond"].to(device=device, dtype=dtype)
    cfg_conds = []
    cap_lens = []
    for i in range(n_images):
        pos_i = tensors[f"pos_cond_{i}"].to(device=device, dtype=dtype)
        cond_batch_i, num_tokens_i = pad_and_batch_cond(pos_i, neg_cond)
        cfg_conds.append(cond_batch_i)
        cap_lens.append(num_tokens_i)

    # Per-image latent dimensions and padded sizes
    img_sizes = []
    for i in range(n_images):
        lh = heights[i] // 8
        lw = widths[i] // 8
        ph = lh + ((-lh) % patch_size)
        pw = lw + ((-lw) % patch_size)
        img_sizes.append((ph, pw))

    # Generate per-image noise at per-image resolution
    x_list = []
    for i in range(n_images):
        lh = heights[i] // 8
        lw = widths[i] // 8
        gen = torch.Generator(device=device).manual_seed(seeds[i])
        noise = torch.randn(
            1, 16, lh, lw, dtype=dtype,
            generator=gen, device=device,
        )
        x_list.append(noise)

    # Build per-image sigma schedules
    sigmas_list = []
    for i in range(n_images):
        s = build_sigma_schedule(
            n_steps, sampling_shift=sampling_shifts[i], multiplier=multiplier,
            denoise=denoise, device=device, dtype=dtype,
        )
        sigmas_list.append(s)

    # Apply CONST noise scaling per image with its own sigma_0
    for i in range(n_images):
        clean_i = tensors.get(f"clean_latent_{i}")
        if clean_i is not None:
            clean_i = clean_i.to(device=device, dtype=dtype)
        else:
            clean_i = torch.zeros_like(x_list[i])
        x_list[i] = const_noise_scaling(sigmas_list[i][0], x_list[i], clean_i)

    # Prepare packed forward state via src_ii (padding + block mask)
    packed_state = prepare_packed_forward(
        model, cfg_conds, img_sizes, cap_lens, device,
    )

    # Build packed model callable
    def packed_fn(x_cfg_list, t_batch, caps, info, mask, rope):
        timesteps_list = [t_batch] * len(x_cfg_list)
        return model(
            x_cfg_list, timesteps_list, caps, info, mask, rope,
        )

    # Save steps setup
    if save_steps_param is not None:
        steps_to_save = {s for s in save_steps_param if s < n_steps}
    else:
        steps_to_save = set()  # Default: don't save intermediates for inference

    result_tensors = {}

    def save_callback(info):
        step_i = info["i"]
        if callback is not None:
            callback(info)
        if step_i in steps_to_save:
            # Reach back into x_list via closure (sample_euler_packed
            # mutates x_list in place before calling callback)
            for img_i in range(n_images):
                result_tensors[f"step_{step_i:02d}_{img_i}"] = \
                    x_list[img_i].detach().cpu()

    # Run packed Euler sampling (applies inverse noise scaling internally)
    x_list = sample_euler_packed(
        packed_fn, x_list, sigmas_list,
        packed_state['refined_caps'],
        packed_state['packing_info'],
        packed_state['block_mask'],
        packed_state['packed_rope'],
        cfg=cfg, multiplier=multiplier, callback=save_callback,
    )

    # Store final latents
    for img_i in range(n_images):
        result_tensors[f"final_{img_i}"] = x_list[img_i].detach().cpu()

    saved = sorted(k for k in result_tensors if k.startswith("step_"))
    return result_tensors, {"n_images": n_images, "saved_steps": saved}

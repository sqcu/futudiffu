"""Full rollout composition: seed -> noise -> solve -> trajectory.

The fifth and top layer in the five-function decomposition.
Composes all lower layers: sigma_schedule, guided_denoiser, solver.

Import constraints:
  - IMPORTS from src_ii: guided_denoiser, solver, sigma_schedule
  - DOES NOT import: model_manager, server, client, sampling, training_utils
"""

import torch

from .guided_denoiser import make_guided_denoiser
from .sigma_schedule import (
    build_sigma_schedule,
    const_inverse_noise_scaling,
    const_noise_scaling,
)
from .solver import euler_solve


def make_rope_cache(diff_model, latent_h: int, latent_w: int, num_tokens: int, device: torch.device) -> dict:
    """Build RoPE cache with patch-size padding.

    Args:
        diff_model: NextDiT model (needs .patch_size and .prepare_rope_cache).
        latent_h: Latent height (pixels / 8).
        latent_w: Latent width (pixels / 8).
        num_tokens: Caption token count after embedding.
        device: Target device.

    Returns:
        RoPE cache dict with 'cap_freqs_cis', 'x_freqs_cis', 'freqs_cis'.
    """
    padded_h = latent_h + ((-latent_h) % diff_model.patch_size)
    padded_w = latent_w + ((-latent_w) % diff_model.patch_size)
    return diff_model.prepare_rope_cache(padded_h, padded_w, num_tokens, device)


def rollout(
    model,
    pos_cond: torch.Tensor,
    neg_cond: torch.Tensor,
    seed: int,
    n_steps: int,
    cfg: float,
    width: int,
    height: int,
    device: torch.device,
    dtype: torch.dtype,
    sampling_shift: float = 1.0,
    multiplier: float = 1.0,
    denoise: float = 1.0,
    clean_latent: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    save_steps: set[int] | None = None,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Full rollout: seed -> noise -> CONST_scale -> euler_solve -> inverse_scale -> trajectory.

    This is the composition of all five function layers:
      1. Build sigma schedule
      2. Generate or accept noise
      3. Apply CONST noise scaling to produce initial x_0
      4. Construct guided denoiser via make_guided_denoiser
      5. Run euler_solve
      6. Apply CONST inverse noise scaling
      7. Return final latent + any saved intermediates

    Args:
        model: NextDiT model (compiled or raw).
        pos_cond: (1, pos_len, dim) positive conditioning.
        neg_cond: (1, neg_len, dim) negative conditioning.
        seed: Random seed for noise generation.
        n_steps: Number of Euler steps.
        cfg: CFG guidance scale.
        width: Output image width in pixels.
        height: Output image height in pixels.
        device: CUDA device.
        dtype: Working dtype (typically bfloat16).
        sampling_shift: Sigma schedule shift (default 1.0).
        multiplier: Timestep multiplier (default 1.0).
        denoise: Denoise strength (1.0 = full t2i, <1.0 = i2i).
        clean_latent: Optional (1, 16, H, W) for i2i. None = zeros (t2i).
        noise: Optional pre-generated noise tensor. If None, generated from seed.
        save_steps: Set of step indices to save. None = save all.

    Returns:
        (result_tensors, metadata) where:
            result_tensors: dict with "final" key and "step_NN" keys for saved steps
            metadata: dict with "saved_steps" list
    """
    latent_h = height // 8
    latent_w = width // 8

    # 1. Build sigma schedule
    sigmas = build_sigma_schedule(
        n_steps,
        sampling_shift=sampling_shift,
        multiplier=multiplier,
        denoise=denoise,
        device=device,
        dtype=dtype,
    )

    # 2. Generate noise (or use provided)
    if noise is None:
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(
            1, 16, latent_h, latent_w,
            dtype=dtype, generator=generator, device=device,
        )

    # 3. Prepare initial latent via CONST noise scaling
    if clean_latent is not None:
        x = const_noise_scaling(sigmas[0], noise, clean_latent)
    else:
        latent = torch.zeros(1, 16, latent_h, latent_w, device=device, dtype=dtype)
        x = const_noise_scaling(sigmas[0], noise, latent)

    # 4. Build RoPE cache and guided denoiser
    # For CFG, pad_and_batch_cond determines num_tokens; for the RoPE cache
    # we need the padded token count. Import pad_and_batch_cond to get it.
    from .guided_denoiser import pad_and_batch_cond

    if cfg != 1.0:
        _, num_tokens = pad_and_batch_cond(pos_cond, neg_cond)
    else:
        num_tokens = pos_cond.shape[1]

    # Access the raw model for RoPE cache (if model is compiled, need the underlying module)
    raw_model = model
    if hasattr(model, '_orig_mod'):
        raw_model = model._orig_mod

    rope_cache = make_rope_cache(raw_model, latent_h, latent_w, num_tokens, device)

    denoiser_fn = make_guided_denoiser(
        model, pos_cond, neg_cond, cfg, rope_cache, multiplier,
    )

    # 5. Setup step saving callback
    if save_steps is not None:
        steps_to_save = {s for s in save_steps if s < n_steps}
    else:
        steps_to_save = set(range(n_steps))

    result_tensors = {}

    def save_callback(info):
        i = info["i"]
        if i in steps_to_save:
            result_tensors[f"step_{i:02d}"] = info["x"].detach().cpu()

    # 6. Run Euler solver
    with torch.inference_mode():
        x = euler_solve(denoiser_fn, x, sigmas, callback=save_callback)
        x = const_inverse_noise_scaling(sigmas[-1], x)

    # 7. Package results
    result_tensors["final"] = x.detach().cpu()
    saved = sorted(k for k in result_tensors if k.startswith("step_"))

    # Build per-step sigma map: step_key -> sigma value (float)
    # The sigma schedule has n_steps+1 entries: sigmas[0..n_steps-1] are the
    # step input sigmas, sigmas[n_steps] = 0.0 is the terminal.
    # For "step_NN", sigma = sigmas[NN].
    # For "final", sigma = 0.0 (the fully denoised image).
    step_sigmas: dict[str, float] = {}
    for step_key in saved:
        step_idx = int(step_key.split("_")[1])
        if step_idx < len(sigmas):
            step_sigmas[step_key] = float(sigmas[step_idx].item())
        else:
            step_sigmas[step_key] = 0.0
    step_sigmas["final"] = 0.0  # Terminal: fully denoised

    metadata = {
        "saved_steps": saved,
        "step_sigmas": step_sigmas,
    }

    return result_tensors, metadata

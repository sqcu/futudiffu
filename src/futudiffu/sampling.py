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


# --- CFG conditioning helpers ---

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


def build_sigma_schedule(
    n_steps: int,
    sampling_shift: float = 1.0,
    multiplier: float = 1.0,
    denoise: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Build the full sigma schedule for a trajectory.

    Handles both t2i (denoise=1.0) and i2i (denoise<1.0, expanded schedule).

    Args:
        n_steps: Number of euler steps.
        sampling_shift: Shift for build_sigmas.
        multiplier: Multiplier for build_sigmas (passed as multiplier * 1000).
        denoise: Denoise strength (0-1). 1.0 = full t2i.
        device: Target device.
        dtype: Target dtype.

    Returns:
        sigmas: (n_steps+1,) tensor on device/dtype.
    """
    sigma_table = build_sigmas(shift=sampling_shift, multiplier=multiplier * 1000)

    if denoise < 1.0:
        # i2i: ComfyUI-style expanded schedule.
        # Build a longer schedule, take the last (n_steps + 1) sigmas.
        # This starts denoising from sigma ~ denoise, running all n_steps
        # iterations with appropriately-spaced sigmas.
        expanded_steps = int(n_steps / denoise)
        full_sigmas = simple_scheduler(sigma_table, expanded_steps)
        full_sigmas = full_sigmas.to(device=device, dtype=dtype)
        sigmas = full_sigmas[-(n_steps + 1):]
    else:
        sigmas = simple_scheduler(sigma_table, n_steps)
        sigmas = sigmas.to(device=device, dtype=dtype)

    return sigmas


def make_cfg_model_fn(diff_model, cond_batch, num_tokens, rope_cache, cfg, multiplier):
    """Create a CFG model function for use with sample_euler.

    Args:
        diff_model: Compiled or raw diffusion model callable.
        cond_batch: (2, seq, dim) batched conditioning (pos, neg).
        num_tokens: Padded sequence length.
        rope_cache: Pre-computed RoPE cache.
        cfg: CFG scale.
        multiplier: Timestep multiplier.

    Returns:
        model_fn: Callable (x_in, sigma) -> denoised.
    """
    def model_fn(x_in, sigma):
        timestep = sigma * multiplier
        x_batch = x_in.expand(2, -1, -1, -1)
        t_batch = timestep.expand(2)
        output_batch = diff_model(
            x_batch, t_batch, cond_batch,
            num_tokens=num_tokens, rope_cache=rope_cache,
        )
        out_cond, out_uncond = output_batch.chunk(2, dim=0)
        denoised_cond = const_calculate_denoised(sigma, out_cond, x_in)
        denoised_uncond = const_calculate_denoised(sigma, out_uncond, x_in)
        return denoised_uncond + (denoised_cond - denoised_uncond) * cfg
    return model_fn


@torch.inference_mode()
def sample_euler_packed(
    packed_forward_fn,
    x_list: list[torch.Tensor],
    sigmas: torch.Tensor,
    refined_caps,
    packing_info,
    block_mask,
    packed_rope,
    cfg: float,
    multiplier: float,
    callback=None,
) -> list[torch.Tensor]:
    """Packed euler sampler for FlexAttention batch packing.

    Same math as sample_euler but operates on N images packed into a single
    FlexAttention forward pass. Each step:
      1. Expand each x_i for CFG (batch dim 2)
      2. Call packed_forward_fn to get per-image outputs
      3. Apply CFG and euler step per image

    Args:
        packed_forward_fn: Callable that takes (x_cfg_list, t_batch,
            refined_caps, packing_info, block_mask, packed_rope) and returns
            a list of per-image output tensors, each (2, C, H, W).
        x_list: List of N tensors, each (1, C, H, W).
        sigmas: 1D tensor of (steps+1,) sigma values.
        refined_caps: Packed refined caption embeddings.
        packing_info: Packing metadata (document_id, total_len, etc).
        block_mask: FlexAttention block mask.
        packed_rope: Packed RoPE cache.
        cfg: CFG scale.
        multiplier: Timestep multiplier.
        callback: Optional callback(dict) per step. Receives {'i': step_idx,
            'x_list': current x_list}.

    Returns:
        List of N final tensors, each (1, C, H, W).
    """
    n_images = len(x_list)
    n_steps = len(sigmas) - 1

    for step_i in range(n_steps):
        sigma = sigmas[step_i]
        timestep = sigma * multiplier

        x_cfg = [x_i.expand(2, -1, -1, -1) for x_i in x_list]
        t_batch = timestep.expand(2)

        outputs = packed_forward_fn(
            x_cfg, t_batch, refined_caps,
            packing_info, block_mask, packed_rope,
        )

        for img_i in range(n_images):
            out_cond, out_uncond = outputs[img_i].chunk(2, dim=0)
            denoised_cond = const_calculate_denoised(
                sigma, out_cond, x_list[img_i])
            denoised_uncond = const_calculate_denoised(
                sigma, out_uncond, x_list[img_i])
            denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cfg

            d = to_d(x_list[img_i], sigma, denoised)
            dt = sigmas[step_i + 1] - sigma
            x_list[img_i] = x_list[img_i] + d * dt

        if callback is not None:
            callback({'i': step_i, 'x_list': x_list})

    return x_list


def prepare_initial_latent(
    noise: torch.Tensor,
    sigma_0: torch.Tensor,
    clean_latent: torch.Tensor | None,
    latent_h: int,
    latent_w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Apply CONST noise scaling to produce the initial x_0.

    Args:
        noise: (1, 16, H, W) noise tensor.
        sigma_0: First sigma value (scalar tensor).
        clean_latent: Optional (1, 16, H, W) for i2i. None = zeros (t2i).
        latent_h: Latent height.
        latent_w: Latent width.
        device: Target device.
        dtype: Target dtype.

    Returns:
        x: (1, 16, H, W) initial noised latent.
    """
    if clean_latent is not None:
        return const_noise_scaling(sigma_0, noise, clean_latent)
    else:
        latent = torch.zeros(1, 16, latent_h, latent_w, device=device, dtype=dtype)
        return const_noise_scaling(sigma_0, noise, latent)


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
    return_denoised: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor]] | tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
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
        return_denoised: If True, also return per-step denoised predictions
            for DRGRPO log-ratio computation.

    Returns:
        If return_denoised is False:
            (final_x, checkpoints) where checkpoints is [x_0, ..., x_T].
        If return_denoised is True:
            (final_x, checkpoints, denoised_list) where denoised_list is
            [denoised_0, ..., denoised_{T-1}], each detached.
    """
    s_in = x.new_ones([x.shape[0]])
    checkpoints: list[torch.Tensor] = []
    denoised_list: list[torch.Tensor] = []
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

        if return_denoised:
            denoised_list.append(denoised.detach().clone())

        d = to_d(x, sigma_hat, denoised)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})

        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt

    # Final boundary checkpoint
    checkpoints.append(x.detach().clone())

    if return_denoised:
        return x, checkpoints, denoised_list
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


# ---------------------------------------------------------------------------
# Full trajectory orchestration (extracted from server.py handlers)
# ---------------------------------------------------------------------------

def make_rope_cache(diff_model, latent_h, latent_w, num_tokens, device):
    """Build RoPE cache with patch-size padding."""
    padded_h = latent_h + ((-latent_h) % diff_model.patch_size)
    padded_w = latent_w + ((-latent_w) % diff_model.patch_size)
    return diff_model.prepare_rope_cache(padded_h, padded_w, num_tokens, device)


def run_trajectory(diff_compiled, diff_model, device, dtype, params, tensors,
                   btrm_head=None):
    """Run a complete sampling trajectory with optional inline BTRM scoring.

    Args:
        diff_compiled: Compiled diffusion model for inference.
        diff_model: Raw model (for RoPE cache / patch_size / BTRM scoring).
        device: CUDA device.
        dtype: Working dtype.
        params: RPC params dict (seed, n_steps, cfg, width, height, etc.).
            Optional: score_at_step (int) — step index for inline BTRM scoring.
        tensors: RPC tensors dict (pos_cond, neg_cond, optional clean_latent/noise).
        btrm_head: Optional BTRM head module for inline scoring. If both
            btrm_head and params["score_at_step"] are set, the scored step's
            latent is scored via an uncompiled backbone forward, avoiding a
            separate score_btrm RPC round-trip.

    Returns:
        (result_tensors, metadata) where result_tensors has "final" + "step_NN"
        keys and metadata has {"saved_steps": [...], optionally "btrm_scores": [...]}.
    """
    seed = params["seed"]
    n_steps = params["n_steps"]
    cfg = params["cfg"]
    width = params["width"]
    height = params["height"]
    sampling_shift = params.get("sampling_shift", 1.0)
    multiplier = params.get("multiplier", 1.0)
    save_steps = params.get("save_steps", None)
    denoise = params.get("denoise", 1.0)
    score_at_step = params.get("score_at_step", None)

    pos_cond = tensors["pos_cond"].to(device=device, dtype=dtype)
    neg_cond = tensors["neg_cond"].to(device=device, dtype=dtype)
    clean_latent = tensors.get("clean_latent")
    if clean_latent is not None:
        clean_latent = clean_latent.to(device=device, dtype=dtype)

    latent_h = height // 8
    latent_w = width // 8

    cond_batch, num_tokens = pad_and_batch_cond(pos_cond, neg_cond)
    rope_cache = make_rope_cache(diff_model, latent_h, latent_w, num_tokens, device)

    if "noise" in tensors:
        noise = tensors["noise"].to(device=device, dtype=dtype)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(
            1, 16, latent_h, latent_w, dtype=dtype,
            generator=generator, device=device,
        )

    sigmas = build_sigma_schedule(
        n_steps, sampling_shift=sampling_shift, multiplier=multiplier,
        denoise=denoise, device=device, dtype=dtype,
    )

    x = prepare_initial_latent(
        noise, sigmas[0], clean_latent, latent_h, latent_w, device, dtype,
    )

    model_fn = make_cfg_model_fn(
        diff_compiled, cond_batch, num_tokens, rope_cache, cfg, multiplier,
    )

    if save_steps is not None:
        steps_to_save = {s for s in save_steps if s < n_steps}
    else:
        steps_to_save = set(range(n_steps))

    result_tensors = {}
    scored_step_gpu = {}  # Temporary GPU storage for inline BTRM scoring

    def save_callback(info):
        i = info["i"]
        if score_at_step is not None and i == score_at_step:
            scored_step_gpu['x'] = info["x"].detach().clone()
        if i in steps_to_save:
            result_tensors[f"step_{i:02d}"] = info["x"].detach().cpu()

    with torch.inference_mode():
        x = sample_euler(model_fn, x, sigmas, callback=save_callback)
        x = const_inverse_noise_scaling(sigmas[-1], x)

    result_tensors["final"] = x.detach().cpu()
    saved = sorted(k for k in result_tensors if k.startswith("step_"))
    metadata = {"saved_steps": saved}

    # Inline BTRM scoring: run uncompiled backbone on the scored step's
    # on-GPU latent, avoiding the separate score_btrm RPC + CPU<->GPU transfer.
    if score_at_step is not None and btrm_head is not None:
        x_scored = scored_step_gpu.get('x')
        if x_scored is not None:
            from .training_utils import run_backbone_hidden
            sigma_scored = sigmas[score_at_step].unsqueeze(0)
            hidden = run_backbone_hidden(
                diff_model, x_scored, sigma_scored,
                pos_cond, device, dtype,
                multiplier=multiplier,
            )
            with torch.no_grad():
                scores = btrm_head(hidden)
            metadata["btrm_scores"] = scores.detach().cpu().tolist()
            del x_scored
    scored_step_gpu.clear()

    return result_tensors, metadata


def run_trajectory_packed(diff_compiled_packed, diff_model, device, dtype,
                          params, tensors):
    """Run N packed diffusion trajectories via FlexAttention.

    Args:
        diff_compiled_packed: Compiled forward_packed.
        diff_model: Raw model (for patch_size, prepare_packed_state).
        device: CUDA device.
        dtype: Working dtype.
        params: RPC params dict.
        tensors: RPC tensors dict.

    Returns:
        (result_tensors, metadata) tuple.
    """
    from .diffusion_model import make_packing_mask_mod
    from torch.nn.attention.flex_attention import create_block_mask

    n_images = params["n_images"]
    seeds = params["seeds"]
    n_steps = params["n_steps"]
    cfg = params["cfg"]
    width = params["width"]
    height = params["height"]
    sampling_shift = params.get("sampling_shift", 1.0)
    multiplier = params.get("multiplier", 1.0)
    save_steps_param = params.get("save_steps", None)
    denoise = params.get("denoise", 1.0)

    pH = pW = diff_model.patch_size

    neg_cond = tensors["neg_cond"].to(device=device, dtype=dtype)
    cfg_conds = []
    cap_lens = []
    for i in range(n_images):
        pos_i = tensors[f"pos_cond_{i}"].to(device=device, dtype=dtype)
        cond_batch_i, num_tokens_i = pad_and_batch_cond(pos_i, neg_cond)
        cfg_conds.append(cond_batch_i)
        cap_lens.append(num_tokens_i)

    latent_h = height // 8
    latent_w = width // 8
    padded_h = latent_h + ((-latent_h) % pH)
    padded_w = latent_w + ((-latent_w) % pW)

    x_list = []
    for i in range(n_images):
        gen = torch.Generator(device=device).manual_seed(seeds[i])
        noise = torch.randn(
            1, 16, latent_h, latent_w, dtype=dtype,
            generator=gen, device=device,
        )
        x_list.append(noise)

    sigmas = build_sigma_schedule(
        n_steps, sampling_shift=sampling_shift, multiplier=multiplier,
        denoise=denoise, device=device, dtype=dtype,
    )

    for i in range(n_images):
        clean_i = tensors.get(f"clean_latent_{i}")
        if clean_i is not None:
            clean_i = clean_i.to(device=device, dtype=dtype)
        else:
            clean_i = torch.zeros_like(x_list[i])
        x_list[i] = const_noise_scaling(sigmas[0], x_list[i], clean_i)

    with torch.inference_mode():
        padded_sizes = [(padded_h, padded_w)] * n_images
        refined_caps, packing_info, packed_rope = \
            diff_model.prepare_packed_state(
                cfg_conds, padded_sizes, cap_lens, device,
            )
        block_mask = create_block_mask(
            make_packing_mask_mod(packing_info.document_id),
            B=2, H=None,
            Q_LEN=packing_info.total_len,
            KV_LEN=packing_info.total_len,
            device=device,
        )

    if save_steps_param is not None:
        steps_to_save = {s for s in save_steps_param if s < n_steps}
    else:
        steps_to_save = set(range(n_steps))

    result_tensors = {}

    def save_callback(info):
        step_i = info["i"]
        if step_i in steps_to_save:
            for img_i in range(n_images):
                result_tensors[f"step_{step_i:02d}_{img_i}"] = \
                    info["x_list"][img_i].detach().cpu()

    x_list = sample_euler_packed(
        diff_compiled_packed, x_list, sigmas,
        refined_caps, packing_info, block_mask, packed_rope,
        cfg=cfg, multiplier=multiplier, callback=save_callback,
    )

    for img_i in range(n_images):
        x_list[img_i] = const_inverse_noise_scaling(sigmas[-1], x_list[img_i])
        result_tensors[f"final_{img_i}"] = x_list[img_i].detach().cpu()

    saved = sorted(k for k in result_tensors if k.startswith("step_"))
    return result_tensors, {"n_images": n_images, "saved_steps": saved}


def warmup_diffusion(diff_compiled, diff_model, device, dtype,
                     width=1280, height=832):
    """Run warmup euler pass with dummy inputs to trigger torch.compile."""
    latent_h = height // 8
    latent_w = width // 8

    dummy_cond = torch.zeros(2, 32, 2560, device=device, dtype=dtype)
    num_tokens = 32
    rope_cache = make_rope_cache(diff_model, latent_h, latent_w, num_tokens, device)

    noise = torch.randn(1, 16, latent_h, latent_w, dtype=dtype, device=device)
    sigmas = build_sigma_schedule(4, device=device, dtype=dtype)
    latent = torch.zeros(1, 16, latent_h, latent_w, device=device, dtype=dtype)
    x = const_noise_scaling(sigmas[0], noise, latent)

    model_fn = make_cfg_model_fn(
        diff_compiled, dummy_cond, num_tokens, rope_cache, 4.0, 1.0,
    )
    with torch.inference_mode():
        sample_euler(model_fn, x, sigmas)
    torch.cuda.synchronize()


def warmup_packed(diff_compiled_packed, diff_model, device, dtype, n_images=2):
    """Run warmup packed forward pass with dummy inputs.

    Returns elapsed seconds (includes compilation time).
    """
    import time as _time

    from .diffusion_model import make_packing_mask_mod
    from torch.nn.attention.flex_attention import create_block_mask

    pH = pW = diff_model.patch_size
    latent_h, latent_w = 32, 32
    padded_h = latent_h + ((-latent_h) % pH)
    padded_w = latent_w + ((-latent_w) % pW)

    cfg_conds = [
        torch.zeros(2, 32, 2560, device=device, dtype=dtype)
        for _ in range(n_images)
    ]
    cap_lens = [32] * n_images

    with torch.inference_mode():
        refined_caps, packing_info, packed_rope = \
            diff_model.prepare_packed_state(
                cfg_conds, [(padded_h, padded_w)] * n_images, cap_lens, device,
            )
        block_mask = create_block_mask(
            make_packing_mask_mod(packing_info.document_id),
            B=2, H=None,
            Q_LEN=packing_info.total_len,
            KV_LEN=packing_info.total_len,
            device=device,
        )

        x_list = [
            torch.randn(2, 16, latent_h, latent_w, device=device, dtype=dtype)
            for _ in range(n_images)
        ]
        t_batch = torch.tensor([0.5, 0.5], device=device, dtype=dtype)

        t0 = _time.perf_counter()
        diff_compiled_packed(
            x_list, t_batch, refined_caps,
            packing_info, block_mask, packed_rope,
        )
        torch.cuda.synchronize()
        return _time.perf_counter() - t0

"""Inference sampling: ktuple solver over BatchExecutor.

The key export is `run_trajectory_packed`: drop-in replacement for the
frozen `futudiffu.sampling.run_trajectory_packed`, same params/tensors
contract, but routes through ktuple_sampling.solve() over BatchExecutor.

cfg=1 uses cfg1 (single fork, no negative — halves compute).
cfg>1 uses cfg2 (pos+neg forks, standard CFG).
i2i uses const_noise_scaling with truncated sigma schedule.

Import constraints:
  - torch only (no frozen src.futudiffu imports)
  - Reuses src_ii.sigma_schedule, src_ii.ktuple_sampling, src_ii.batch_executor
"""

from __future__ import annotations

import torch

from src_ii.sigma_schedule import (
    build_sigma_schedule,
    const_noise_scaling,
    const_inverse_noise_scaling,
    resolution_shift,
)


def run_trajectory_packed(
    model,
    device: torch.device,
    dtype: torch.dtype,
    params: dict,
    tensors: dict,
    callback=None,
    adapter_scales: torch.Tensor | None = None,
    batch_executor=None,
) -> tuple[dict, dict]:
    """Run N packed diffusion trajectories via ktuple solver.

    Drop-in replacement for frozen futudiffu.sampling.run_trajectory_packed.
    Same params/tensors contract. Routes through ktuple_sampling.solve()
    with BatchExecutor for packed forward execution.

    Args:
        model: ZImageRLAIF model (compiled or raw).
        device: CUDA device.
        dtype: Working dtype (bf16).
        params: RPC params dict:
            n_images, seeds, n_steps, cfg, multiplier, denoise,
            width/height (int) or widths/heights (list[int]),
            sampling_shift/sampling_shifts (optional),
            save_steps (optional list[int]).
        tensors: RPC tensors dict:
            neg_cond, pos_cond_0..N-1, optional clean_latent_0..N-1.
        callback: Optional per-step callback({'i', 'n_steps'}).
        adapter_scales: Optional (n_images, n_adapters) or (1, n_adapters).
        batch_executor: Optional pre-existing BatchExecutor (avoids re-creation).

    Returns:
        (result_tensors, metadata) — result_tensors has "final_0".."final_N-1"
        keys (and optionally "step_SS_II" intermediate keys).
    """
    from src_ii.batch_executor import BatchExecutor, make_executor_adapter
    from src_ii.ktuple_sampling import solve
    from src_ii.triumphant_future_reduction_ops import cfg1, cfg2

    if batch_executor is None:
        batch_executor = BatchExecutor(model, device)

    n_images = params["n_images"]
    seeds = params["seeds"]
    n_steps = params["n_steps"]
    cfg = params["cfg"]
    multiplier = params.get("multiplier", 1.0)
    denoise = params.get("denoise", 1.0)
    save_steps_param = params.get("save_steps", None)

    # Resolve per-image resolutions
    if "widths" in params and "heights" in params:
        widths = params["widths"]
        heights = params["heights"]
    else:
        w = params["width"]
        h = params["height"]
        widths = [w] * n_images
        heights = [h] * n_images
    resolutions = list(zip(widths, heights))

    # Resolve per-image sampling shifts: auto-shift * user modifier
    auto_shifts = [resolution_shift(widths[i], heights[i]) for i in range(n_images)]
    if "sampling_shifts" in params:
        user_shifts = params["sampling_shifts"]
    elif "sampling_shift" in params:
        user_shifts = [params["sampling_shift"]] * n_images
    else:
        user_shifts = [1.0] * n_images
    sampling_shifts = [a * u for a, u in zip(auto_shifts, user_shifts)]

    # Build per-image sigma schedules
    query_sigmas = [
        build_sigma_schedule(
            n_steps, sampling_shift=sampling_shifts[i],
            multiplier=multiplier, denoise=denoise,
            device=device, dtype=dtype,
        )
        for i in range(n_images)
    ]

    # Build conditioning
    neg_cond = tensors["neg_cond"].to(device=device, dtype=dtype)
    pos_conds = [
        tensors[f"pos_cond_{i}"].to(device=device, dtype=dtype)
        for i in range(n_images)
    ]

    # Build specs: cfg1 for no guidance (halves compute), cfg2 for standard CFG
    specs = [
        cfg2(pos_conds[k], neg_cond, resolutions[k], cfg) if cfg != 1.0
        else cfg1(pos_conds[k], resolutions[k])
        for k in range(n_images)
    ]

    # Init noise with CONST noise scaling (supports i2i via clean_latent)
    x_bases = []
    for k in range(n_images):
        w, h = resolutions[k]
        gen = torch.Generator(device=device).manual_seed(seeds[k])
        noise = torch.randn(
            1, 16, h // 8, w // 8, dtype=dtype,
            device=device, generator=gen,
        )
        clean_k = tensors.get(f"clean_latent_{k}")
        if clean_k is not None:
            clean_k = clean_k.to(device=device, dtype=dtype)
        else:
            clean_k = torch.zeros_like(noise)
        x_bases.append(const_noise_scaling(query_sigmas[k][0], noise, clean_k))

    # Create executor adapter (bridges ktuple protocol to BatchExecutor)
    executor = make_executor_adapter(batch_executor, query_sigmas, device)

    # Save steps + result collection
    save_steps = set(save_steps_param or [])
    result_tensors = {}
    step_scores = []

    def save_fn(step_i, x_pres, guided_list):
        if callback is not None:
            callback({"i": step_i, "n_steps": n_steps})
        if step_i in save_steps:
            for img_i in range(n_images):
                result_tensors[f"step_{step_i:02d}_{img_i}"] = (
                    x_pres[img_i].detach().cpu()
                )

    x_final, scores_all = solve(
        executor, x_bases, specs, query_sigmas, n_steps,
        adapter_scales=adapter_scales,
        save_fn=save_fn,
    )

    # Inverse noise scaling + finalize
    for k in range(n_images):
        x_k = const_inverse_noise_scaling(query_sigmas[k][-1:], x_final[k])
        result_tensors[f"final_{k}"] = x_k.detach().cpu()

    # Collect scores from solve output
    for step_i, scores_t in enumerate(scores_all):
        if scores_t is not None and scores_t.numel() > 0:
            step_scores.append({
                "step": step_i,
                "sigma": float(query_sigmas[0][step_i]),
                "scores": scores_t[0].tolist(),
            })

    saved = sorted(k for k in result_tensors if k.startswith("step_"))
    metadata = {"n_images": n_images, "saved_steps": saved}
    if step_scores:
        metadata["step_scores"] = step_scores
    return result_tensors, metadata

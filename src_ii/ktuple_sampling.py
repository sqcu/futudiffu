"""K-tuple guided Euler sampling. Pure client — no model internals.

See src_ii/ktuple_sampling_readme.md for interface docs.
"""
from __future__ import annotations
import torch
from .sigma_schedule import build_sigma_schedule, const_inverse_noise_scaling, resolution_shift
from .triumphant_future_reduction_ops import (
    aperture, build_per_image_sigmas, cfg1, cfg2, cfg6,
    euler_step, gather, gather_residual_gain, latent_padded,
    noise_field, scatter,
)


def step(executor, x_bases, specs, query_sigmas, step_i,
         adapter_scales=None, gather_fn=None):
    """One Euler step: submit -> receive denoised estimates -> gather -> euler.

    executor(x_bases, specs, step_i, adapter_scales) ->
        (denoised_per_query: list[list[Tensor]], scores: Tensor)

    query_sigmas: list of (n_steps+1,) tensors, one per query (base resolution).
    Returns (x_next, scores, guided_list).
    """
    denoised_per_query, scores = executor(x_bases, specs, step_i, adapter_scales)

    reduce = gather if gather_fn is None else gather_fn
    x_next, guided_list = [], []
    for k, (denoised_list, spec) in enumerate(zip(denoised_per_query, specs)):
        g = reduce(denoised_list, spec)
        sigma_k = query_sigmas[k][step_i]
        sigma_next_k = query_sigmas[k][step_i + 1]
        x_next.append(euler_step(x_bases[k], g, sigma_k, sigma_next_k))
        guided_list.append(g)
    return x_next, scores, guided_list


def solve(executor, x_bases, specs, query_sigmas, n_steps,
          adapter_scales=None, gather_fn=None, save_fn=None):
    """The Euler loop. Returns (x_final, scores_all)."""
    scores_all = []
    with torch.no_grad():
        for i in range(n_steps):
            x_pre = x_bases
            scales = adapter_scales(i) if callable(adapter_scales) else adapter_scales
            x_bases, scores, guided = step(
                executor, x_bases, specs, query_sigmas, i,
                adapter_scales=scales, gather_fn=gather_fn)
            scores_all.append(scores.detach().cpu() if scores is not None else None)
            if save_fn is not None:
                save_fn(i, x_pre, guided)
    return x_bases, scores_all


def batch_rollout(executor, pos_conds, neg_conds, cap_lens, seeds, resolutions,
                  n_steps, cfg, device, dtype, adapter_scales=None,
                  save_steps=None, multiplier=1.0, gather_fn=None):
    """Build specs from pos/neg conds + cfg, init noise, run solve, package trajectories.

    Returns (trajectories, metadata).
    """
    K = len(pos_conds)
    specs = [cfg2(p, n, r, cfg) if cfg != 1.0 else cfg1(p, r)
             for p, n, r in zip(pos_conds, neg_conds, resolutions)]

    query_sigmas = [
        build_sigma_schedule(n_steps, sampling_shift=resolution_shift(w, h) * multiplier,
                             device=device, dtype=dtype)
        for w, h in resolutions]

    x_bases = []
    for k in range(K):
        w, h = resolutions[k]
        gen = torch.Generator(device=device).manual_seed(seeds[k])
        x_bases.append(query_sigmas[k][0] *
                       torch.randn(1, 16, h // 8, w // 8, dtype=dtype,
                                   device=device, generator=gen))

    trajectories = [{} for _ in range(K)]

    def save_fn(i, x_pres, guided_list):
        if save_steps and i in save_steps:
            for k in range(K):
                trajectories[k][f"step_{i:02d}"] = {
                    "x": x_pres[k].detach().cpu(),
                    "guided_denoised": guided_list[k].detach().cpu(),
                    "sigma": float(query_sigmas[k][i])}

    x_bases, scores_all = solve(
        executor, x_bases, specs, query_sigmas, n_steps,
        adapter_scales=adapter_scales,
        gather_fn=gather_fn,
        save_fn=save_fn if save_steps else None)

    for k in range(K):
        x_k = const_inverse_noise_scaling(query_sigmas[k][-1:], x_bases[k])
        trajectories[k].update({
            "final": x_k.detach().cpu(), "seed": seeds[k],
            "resolution": resolutions[k],
            "sigmas": [float(s) for s in query_sigmas[k]]})

    return trajectories, {"n_steps": n_steps, "K": K, "cfg": cfg,
                          "scores_per_step": scores_all}


def spec_rollout(executor, spec, cap_lens, seed, n_steps, device, dtype,
                 adapter_scales=None, save_steps=None, multiplier=1.0,
                 gather_fn=None):
    """Pre-built spec rollout with aperture noise for correlated multi-res init.

    Returns (trajectory, metadata).
    """
    flat = [e for e in spec]
    entry_sigmas = build_per_image_sigmas(flat, n_steps, device, dtype)

    # Query sigma = base entry (entry 0) sigma schedule
    query_sigmas = [entry_sigmas[0]]

    max_lh = max(rh // 8 for _, (_, rh), _ in spec)
    max_lw = max(rw // 8 for _, (rw, _), _ in spec)
    master = noise_field(max_lh, max_lw, seed, device, dtype)
    x_bases = [entry_sigmas[0][0] * aperture(master, spec[0][1][1] // 8, spec[0][1][0] // 8)]

    traj = {}

    def save_fn(i, xp, g):
        if save_steps and i in save_steps:
            traj[f"step_{i:02d}"] = {
                "x": xp[0].detach().cpu(),
                "guided_denoised": g[0].detach().cpu(),
                "sigma": float(query_sigmas[0][i])}

    x_bases, scores_all = solve(
        executor, x_bases, [spec], query_sigmas, n_steps,
        adapter_scales=adapter_scales,
        gather_fn=gather_fn,
        save_fn=save_fn if save_steps else None)

    traj["final"] = const_inverse_noise_scaling(
        query_sigmas[0][-1:], x_bases[0]).detach().cpu()
    return traj, {"n_steps": n_steps, "spec_len": len(spec),
                  "scores_per_step": scores_all}

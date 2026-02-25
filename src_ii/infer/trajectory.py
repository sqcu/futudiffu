"""
Single-trajectory Euler sampler wrapping packed_forward and euler_step.

This module provides `sample_trajectory`, which runs one full Euler ODE
integration for a single adapter configuration and a single initial noise
sample. It is the algorithmic core extracted from the inlined sampling loop
in `scripts_ii/demonstrate_rtheta_policy.py` (phase3_dual_sampling).

Role in the extraction:
  - `euler.py`       -- euler_step(), sigma_to_logsnr()  (no src_ii deps)
  - `trajectory.py`  -- sample_trajectory()              (imports euler + forward_packed)
  - Scheduling scripts loop over prompts, seeds, and configs; each inner call
    is a single `sample_trajectory()` invocation.

The function does not own the model lifecycle, plan creation, or sigma
schedule construction. Callers are responsible for those and for any
progress reporting.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src_ii.forward_packed import packed_forward
from src_ii.infer.euler import euler_step, sigma_to_logsnr


def sample_trajectory(
    model,
    plan: dict,
    sigmas: Tensor,
    x_init: Tensor,
    adapter_scales: Tensor,
) -> tuple[Tensor, list[dict]]:
    """Run Euler ODE sampling for one trajectory with one adapter config.

    Args:
        model: Compiled ZImageRLAIF model (or any model with the same
            calling convention as packed_forward expects).
        plan: Output of prepare_packed_forward(). Must contain keys
            "refined_caps", "packing_info", "block_mask", "packed_rope".
        sigmas: (n_steps+1,) sigma schedule. sigmas[0] is the initial noise
            level; sigmas[-1] is 0.0 (fully denoised).
        x_init: (1, 16, lh, lw) initial noisy latent, on the target device.
        adapter_scales: (1, n_adapters) scale vector controlling adapter
            contributions for this trajectory.

    Returns:
        final_latent: (1, 16, lh, lw) denoised latent on the same device
            as x_init.
        step_records: list of dicts, one per step, each containing:
            {
                "step":   int,         # 0-indexed step index
                "sigma":  float,       # sigma_i at this step
                "logsnr": float,       # sigma_to_logsnr(sigma_i)
                "scores": list[float], # per-head scores from the model
            }
    """
    n_steps = len(sigmas) - 1
    x = x_init.clone()
    step_records: list[dict] = []

    with torch.no_grad():
        for step_i in range(n_steps):
            sigma_i = sigmas[step_i]
            sigma_next = sigmas[step_i + 1]
            ts = sigma_i.reshape(1)

            fields, scores_t = packed_forward(
                model,
                [x],
                [ts],
                plan["refined_caps"],
                plan["packing_info"],
                plan["block_mask"],
                plan["packed_rope"],
                adapter_scales=adapter_scales,
            )

            field = fields[0]
            scores = scores_t[0].cpu().tolist()

            step_records.append({
                "step": step_i,
                "sigma": float(sigma_i),
                "logsnr": sigma_to_logsnr(float(sigma_i)),
                "scores": scores,
            })

            x = euler_step(x, field, sigma_i, sigma_next)

    return x, step_records

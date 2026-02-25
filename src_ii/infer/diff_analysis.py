"""Latent-space diff analysis for comparing policy interventions.

Provides covariance eigenspectrum analysis and pixel-space diff statistics
for quantifying how much two sets of generated latents differ. The canonical
entry point for any script that needs to characterize the effect of a policy
intervention on the denoising trajectory.

Three groups of functionality:
  1. Latent-space: compute_latent_covariance, compute_mean_diff
  2. Pixel-space: compute_pixel_diff_stats (wraps rendering.py primitives)
  3. Re-exports from rendering.py for unified import: make_false_color_diff,
     compute_spatial_autocorrelation

Callers import everything diff-related from here.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src_ii.rendering import make_false_color_diff, compute_spatial_autocorrelation

__all__ = [
    "compute_latent_covariance",
    "compute_mean_diff",
    "compute_pixel_diff_stats",
    "make_false_color_diff",
    "compute_spatial_autocorrelation",
]


def compute_latent_covariance(diff_latent: torch.Tensor) -> dict:
    """Compute channel covariance eigenspectrum of a (16, H, W) diff latent.

    The effective rank of the covariance matrix quantifies how many independent
    dimensions the policy intervention affects. A rank-1 signal means the diff
    is dominated by a single channel mode; full rank means all 16 channels
    are independently perturbed.

    Args:
        diff_latent: (16, H, W) float-compatible tensor. The difference between
                     two latents (e.g., policy_on minus policy_off).

    Returns:
        Dict with keys:
          effective_rank: float -- participation ratio of the eigenspectrum.
                          Range [1, 16]. Low = concentrated, high = distributed.
          eigenvalues: list[float] -- top-8 eigenvalues in descending order.
    """
    diff_flat = diff_latent.float().reshape(16, -1)  # (16, H*W)
    cov = diff_flat @ diff_flat.T / diff_flat.shape[1]  # (16, 16)
    eigvals = torch.linalg.eigvalsh(cov).flip(0)  # descending
    eigvals_pos = eigvals.clamp(min=0)
    total = eigvals_pos.sum()
    if total > 1e-12:
        eff_rank = float((total ** 2) / (eigvals_pos ** 2).sum())
    else:
        eff_rank = 0.0
    return {
        "effective_rank": eff_rank,
        "eigenvalues": [float(v) for v in eigvals_pos[:8].tolist()],
    }


def compute_mean_diff(
    latents_a: list[torch.Tensor],
    latents_b: list[torch.Tensor],
) -> torch.Tensor:
    """Mean of (a - b) across paired latent lists.

    Args:
        latents_a: List of (C, H, W) latent tensors from configuration A.
        latents_b: List of (C, H, W) latent tensors from configuration B,
                   paired with latents_a by index.

    Returns:
        (C, H, W) float tensor: element-wise mean of (a_i - b_i).
    """
    diffs = torch.stack([a.float() - b.float() for a, b in zip(latents_a, latents_b)])
    return diffs.mean(dim=0)


def compute_pixel_diff_stats(
    img_a,  # PIL Image or (H, W, 3) numpy array
    img_b,  # PIL Image or (H, W, 3) numpy array
) -> dict:
    """Absolute pixel difference statistics between two images.

    Args:
        img_a: PIL Image (RGB) or (H, W, 3) numpy uint8/float array.
        img_b: PIL Image (RGB) or (H, W, 3) numpy uint8/float array.

    Returns:
        Dict with keys:
          mean: float -- mean absolute pixel difference, normalized to [0, 1].
          std: float -- standard deviation of absolute pixel difference.
          max: float -- maximum absolute pixel difference.
          per_channel: dict -- {R: {mean, std, max}, G: {...}, B: {...}},
                       each value normalized to [0, 1].
    """
    def _to_float_hw3(img):
        if isinstance(img, Image.Image):
            return np.array(img, dtype=np.float32) / 255.0
        arr = np.asarray(img, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr

    a = _to_float_hw3(img_a)  # (H, W, 3)
    b = _to_float_hw3(img_b)  # (H, W, 3)
    abs_diff = np.abs(a - b)   # (H, W, 3)

    channel_names = ["R", "G", "B"]
    per_channel = {}
    for c, name in enumerate(channel_names):
        ch = abs_diff[:, :, c]
        per_channel[name] = {
            "mean": float(ch.mean()),
            "std": float(ch.std()),
            "max": float(ch.max()),
        }

    return {
        "mean": float(abs_diff.mean()),
        "std": float(abs_diff.std()),
        "max": float(abs_diff.max()),
        "per_channel": per_channel,
    }

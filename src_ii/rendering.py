"""Rendering utilities: tensor-to-PNG, false-color diff, diff statistics.

The single canonical implementation of all pixel-space rendering operations
that accept raw tensors (the server-RPC decode path). Scripts that communicate
through the inference server use client.vae_decode() to obtain a (1, 3, H, W)
float [0,1] tensor, then call this module to convert it to an image.

Two decode paths exist:
  1. Local VAE (decode_latent_to_pil): Requires a loaded VAE model on GPU.
     Used by scripts with direct GPU access. Implemented in vae_utils.py.
     Re-exported here for convenience.
  2. Server-backed (save_tensor_as_png): Accepts the raw tensor output of
     client.vae_decode(). Used by scripts that communicate through the
     inference server.

Import constraints:
  - PIL for image output
  - torch for tensor operations
  - numpy for array conversion
  - Optionally imports from vae_utils for local decode path re-export
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image

# Re-export the local VAE decode path for convenience.
# Scripts with direct GPU access can use these without importing vae_utils separately.
from src_ii.vae_utils import decode_latent_to_pil, load_vae

__all__ = [
    # Re-exported from vae_utils (local VAE decode path)
    "load_vae",
    "decode_latent_to_pil",
    # Tensor-to-PNG (server decode path)
    "save_tensor_as_png",
    "tensor_to_pil",
    # False-color diff
    "make_false_color_diff",
    "save_false_color_diff",
    # Diff statistics
    "compute_per_channel_pixel_stats",
    "compute_spatial_autocorrelation",
]

# Type alias for functions that accept either PIL or tensor
_ImageLike = Union[Image.Image, torch.Tensor]


# ---------------------------------------------------------------------------
# Tensor-to-PNG (server decode path)
# ---------------------------------------------------------------------------

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a (1, 3, H, W) float [0,1] tensor to a PIL RGB Image.

    Args:
        tensor: (1, 3, H, W) float tensor with values in [0, 1].

    Returns:
        PIL RGB Image of size (W, H).
    """
    img = tensor.squeeze(0).float().clamp(0, 1)  # (3, H, W)
    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_np, "RGB")


def save_tensor_as_png(tensor: torch.Tensor, path: Union[str, Path]) -> None:
    """Save a (1, 3, H, W) float [0,1] tensor as a PNG file.

    The canonical implementation of the squeeze/clamp/permute/numpy/fromarray/save
    pipeline. All scripts must call this function rather than inlining the pipeline.

    Args:
        tensor: (1, 3, H, W) float tensor with values in [0, 1].
        path: Output file path (str or Path).
    """
    tensor_to_pil(tensor).save(str(path))


# ---------------------------------------------------------------------------
# False-color diff
# ---------------------------------------------------------------------------

def _to_float_tensor(img: _ImageLike) -> torch.Tensor:
    """Normalize an image-like value to a (H, W, 3) float32 tensor in [0,1]."""
    if isinstance(img, Image.Image):
        arr = np.array(img, dtype="float32")
        return torch.from_numpy(arr) / 255.0
    else:
        # Assume (1, 3, H, W) or (3, H, W) tensor in [0, 1]
        t = img.float()
        if t.ndim == 4:
            t = t.squeeze(0)
        # Now (3, H, W) -> (H, W, 3)
        return t.permute(1, 2, 0).cpu()


def make_false_color_diff(
    img_a: _ImageLike,
    img_b: _ImageLike,
    scale: float = 10.0,
) -> Image.Image:
    """Compute absolute pixel difference, scale, and return a PIL false-color image.

    Accepts either PIL Images or (1, 3, H, W) / (3, H, W) tensors in [0, 1].
    The output is always a PIL RGB Image.

    When inputs are PIL Images (uint8 pixels in [0, 255]), the output pixel values
    are abs(a - b) * scale, clamped to [0, 255].
    When inputs are tensors ([0, 1] floats), the difference is in [0, 1] space,
    then scaled, then mapped to [0, 255].

    Args:
        img_a: Reference image (PIL Image or tensor).
        img_b: Comparison image (PIL Image or tensor).
        scale: Amplification factor for the absolute difference. Default 10.0.

    Returns:
        PIL RGB Image showing the amplified absolute difference.
    """
    a = _to_float_tensor(img_a)  # (H, W, 3) in [0, 1]
    b = _to_float_tensor(img_b)  # (H, W, 3) in [0, 1]
    diff = (a - b).abs() * scale  # (H, W, 3) in [0, scale]
    diff_uint8 = diff.clamp(0.0, 1.0).mul(255.0).byte().numpy()
    return Image.fromarray(diff_uint8, mode="RGB")


def save_false_color_diff(
    img_a: _ImageLike,
    img_b: _ImageLike,
    path: Union[str, Path],
    scale: float = 10.0,
) -> None:
    """Compute and save a false-color diff image to a PNG file.

    Convenience wrapper around make_false_color_diff() + save().

    Args:
        img_a: Reference image (PIL Image or tensor).
        img_b: Comparison image (PIL Image or tensor).
        path: Output file path (str or Path).
        scale: Amplification factor for the absolute difference. Default 10.0.
    """
    make_false_color_diff(img_a, img_b, scale=scale).save(str(path))


# ---------------------------------------------------------------------------
# Diff statistics
# ---------------------------------------------------------------------------

def compute_per_channel_pixel_stats(
    img_a: torch.Tensor,
    img_b: torch.Tensor,
) -> dict:
    """Compute per-channel (R, G, B) absolute pixel difference statistics.

    Args:
        img_a: (1, 3, H, W) float [0, 1] tensor.
        img_b: (1, 3, H, W) float [0, 1] tensor.

    Returns:
        Dict with keys:
          per_channel: {R: {mean, std, max}, G: {...}, B: {...}}
          overall_mean: float
          overall_std: float
          overall_max: float
    """
    abs_diff = (img_a.float() - img_b.float()).abs().squeeze(0)  # (3, H, W)
    channels = ["R", "G", "B"]
    per_channel = {}
    for c, name in enumerate(channels):
        ch = abs_diff[c]
        per_channel[name] = {
            "mean": float(ch.mean().item()),
            "std": float(ch.std().item()),
            "max": float(ch.max().item()),
        }
    return {
        "per_channel": per_channel,
        "overall_mean": float(abs_diff.mean().item()),
        "overall_std": float(abs_diff.std().item()),
        "overall_max": float(abs_diff.max().item()),
    }


def compute_spatial_autocorrelation(diff_img: np.ndarray) -> dict:
    """Compute spatial autocorrelation of a diff image.

    If the diff is structured (correlated blocks), there is a packing or
    algorithmic bug. If the diff is random (white noise), it is just float
    rounding from numerical precision differences.

    Args:
        diff_img: (H, W) or (H, W, C) absolute difference image as a numpy
                  float array. Channel axis is averaged before analysis.

    Returns:
        Dict with keys:
          verdict: str -- "STRUCTURED (possible packing bug)",
                          "weakly_structured (investigate)", or
                          "random (expected float rounding)"
          lag1_h: float -- lag-1 autocorrelation in height direction
          lag1_w: float -- lag-1 autocorrelation in width direction
          lag2_h: float -- lag-2 autocorrelation in height direction
          lag2_w: float -- lag-2 autocorrelation in width direction
          max_autocorrelation: float -- max of |lag1_h|, |lag1_w|, |lag2_h|, |lag2_w|
    """
    if diff_img.ndim == 3:
        diff_img = diff_img.mean(axis=-1)  # Average across channels

    h, w = diff_img.shape
    if h < 4 or w < 4:
        return {
            "verdict": "too_small",
            "lag1_h": 0.0,
            "lag1_w": 0.0,
            "lag2_h": 0.0,
            "lag2_w": 0.0,
            "max_autocorrelation": 0.0,
        }

    mean = diff_img.mean()
    centered = diff_img - mean
    var = (centered ** 2).mean()
    if var < 1e-15:
        return {
            "verdict": "zero_variance",
            "lag1_h": 0.0,
            "lag1_w": 0.0,
            "lag2_h": 0.0,
            "lag2_w": 0.0,
            "max_autocorrelation": 0.0,
        }

    lag1_h = float((centered[:-1, :] * centered[1:, :]).mean() / var)
    lag1_w = float((centered[:, :-1] * centered[:, 1:]).mean() / var)
    lag2_h = float((centered[:-2, :] * centered[2:, :]).mean() / var) if h > 4 else 0.0
    lag2_w = float((centered[:, :-2] * centered[:, 2:]).mean() / var) if w > 4 else 0.0

    max_autocorr = max(abs(lag1_h), abs(lag1_w), abs(lag2_h), abs(lag2_w))

    if max_autocorr > 0.3:
        verdict = "STRUCTURED (possible packing bug)"
    elif max_autocorr > 0.1:
        verdict = "weakly_structured (investigate)"
    else:
        verdict = "random (expected float rounding)"

    return {
        "verdict": verdict,
        "lag1_h": lag1_h,
        "lag1_w": lag1_w,
        "lag2_h": lag2_h,
        "lag2_w": lag2_w,
        "max_autocorrelation": max_autocorr,
    }

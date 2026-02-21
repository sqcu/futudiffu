"""Continuous resolution sampling from megapixel budgets.

Replaces hardcoded resolution enumerations with algorithmic sampling:
  - 6 megapixel anchor tiers for FLOPS accounting
  - Continuous aspect ratio sampling (log-uniform over [0.5, 2.0])
  - 32px-aligned (W, H) quantization preserving pixel budget

The key insight: resolution is NOT a fixed enumeration. It is a continuous
2D space (megapixel budget x aspect ratio) quantized to hardware-friendly
dimensions. Any 32px-aligned (W, H) pair is a valid resolution. The 6
anchor points are for FLOPS cost accounting only.

Import constraints:
  - Pure Python + math only. No torch, no numpy.
  - Deterministic: same input -> same output.
"""

from __future__ import annotations

import math
import random
from typing import Any


# ---------------------------------------------------------------------------
# Megapixel anchor tiers
# ---------------------------------------------------------------------------

# 6 anchor points for FLOPS cost accounting. Each is the square of a
# reference edge length:
#   256^2 = 65536       ~0.066 MP
#   320^2 = 102400      ~0.102 MP
#   384^2 = 147456      ~0.147 MP
#   512^2 = 262144      ~0.262 MP
#   704^2 = 495616      ~0.496 MP
#   1024^2 = 1048576    ~1.049 MP
MEGAPIXEL_ANCHORS: list[int] = [65536, 102400, 147456, 262144, 495616, 1048576]

# Human-readable labels for each anchor (keyed by pixel count)
ANCHOR_LABELS: dict[int, str] = {
    65536: "256sq",
    102400: "320sq",
    147456: "384sq",
    262144: "512sq",
    495616: "704sq",
    1048576: "1024sq",
}

# Minimum image dimension in pixels. Both W and H must be >= this.
MIN_DIM = 64


# ---------------------------------------------------------------------------
# Core resolution computation
# ---------------------------------------------------------------------------

def sample_resolution(
    budget_pixels: int,
    aspect_ratio: float,
    step: int = 32,
) -> tuple[int, int]:
    """Compute 32px-aligned (W, H) from a pixel budget and aspect ratio.

    Given a target pixel count and desired W/H ratio, finds the (W, H) pair
    where:
      - W and H are both multiples of `step` (default 32)
      - W and H are both >= MIN_DIM (64)
      - W * H is as close to budget_pixels as possible
      - W / H is as close to aspect_ratio as possible

    The algorithm:
      1. From W*H = budget and W/H = aspect, solve:
         W = sqrt(budget * aspect), H = sqrt(budget / aspect)
      2. Round both to nearest multiple of step.
      3. Clamp to >= MIN_DIM.

    Args:
        budget_pixels: Target total pixel count (e.g., 262144 for 512^2).
        aspect_ratio: Target W/H ratio. 1.0 = square, >1 = landscape, <1 = portrait.
        step: Alignment step for both W and H. Default 32.

    Returns:
        (width, height) tuple, both multiples of step and >= MIN_DIM.

    Raises:
        ValueError: If budget_pixels <= 0 or aspect_ratio <= 0.
    """
    if budget_pixels <= 0:
        raise ValueError(f"budget_pixels must be positive, got {budget_pixels}")
    if aspect_ratio <= 0:
        raise ValueError(f"aspect_ratio must be positive, got {aspect_ratio}")

    # Solve for continuous W, H
    w_cont = math.sqrt(budget_pixels * aspect_ratio)
    h_cont = math.sqrt(budget_pixels / aspect_ratio)

    # Round to nearest multiple of step
    w = max(MIN_DIM, round(w_cont / step) * step)
    h = max(MIN_DIM, round(h_cont / step) * step)

    # Ensure both are multiples of step (the max with MIN_DIM might break this
    # if MIN_DIM is not a multiple of step, but 64 % 32 == 0 so we're fine)
    if w % step != 0:
        w = max(MIN_DIM, (w // step) * step)
    if h % step != 0:
        h = max(MIN_DIM, (h // step) * step)

    return (w, h)


def sample_random_resolution(
    budget_pixels: int,
    rng: random.Random,
    aspect_min: float = 0.5,
    aspect_max: float = 2.0,
    step: int = 32,
) -> tuple[int, int]:
    """Sample a random resolution from a pixel budget.

    Aspect ratio is sampled uniformly in log-space between aspect_min and
    aspect_max. Log-space sampling ensures portrait and landscape are
    equally likely: log(0.5) and log(2.0) are symmetric around log(1.0).

    Args:
        budget_pixels: Target total pixel count.
        rng: Random instance for reproducibility.
        aspect_min: Minimum aspect ratio (W/H). Default 0.5 (2:1 portrait).
        aspect_max: Maximum aspect ratio (W/H). Default 2.0 (2:1 landscape).
        step: Alignment step. Default 32.

    Returns:
        (width, height) tuple.
    """
    if aspect_min <= 0 or aspect_max <= 0:
        raise ValueError(f"aspect bounds must be positive: [{aspect_min}, {aspect_max}]")
    if aspect_min > aspect_max:
        raise ValueError(f"aspect_min ({aspect_min}) > aspect_max ({aspect_max})")

    # Sample aspect ratio in log-space
    log_min = math.log(aspect_min)
    log_max = math.log(aspect_max)
    log_aspect = rng.uniform(log_min, log_max)
    aspect = math.exp(log_aspect)

    return sample_resolution(budget_pixels, aspect, step=step)


def enumerate_resolutions(
    budget_pixels: int,
    step: int = 32,
    aspect_min: float = 0.5,
    aspect_max: float = 2.0,
) -> list[tuple[int, int]]:
    """Enumerate ALL valid (W, H) pairs for a pixel budget.

    Iterates over all possible W values (multiples of step, >= MIN_DIM)
    and computes the corresponding H = round(budget / W / step) * step.
    Keeps only pairs where:
      - H >= MIN_DIM
      - H is a multiple of step
      - W/H is within [aspect_min, aspect_max]

    This is for analysis/validation, not runtime sampling.

    Args:
        budget_pixels: Target total pixel count.
        step: Alignment step. Default 32.
        aspect_min: Minimum W/H ratio. Default 0.5.
        aspect_max: Maximum W/H ratio. Default 2.0.

    Returns:
        Sorted list of unique (W, H) tuples. Sorted by W ascending.
    """
    results = set()

    # W ranges from MIN_DIM up to sqrt(budget * aspect_max) + margin
    w_max_approx = math.sqrt(budget_pixels * aspect_max) + step * 2
    w_max = int(w_max_approx)

    for w in range(MIN_DIM, w_max + 1, step):
        # For this W, find H such that W*H ~ budget
        h_cont = budget_pixels / w
        # Try rounding both ways
        for h_candidate in [
            max(MIN_DIM, round(h_cont / step) * step),
            max(MIN_DIM, math.floor(h_cont / step) * step),
            max(MIN_DIM, math.ceil(h_cont / step) * step),
        ]:
            if h_candidate < MIN_DIM:
                continue
            if h_candidate % step != 0:
                continue
            aspect = w / h_candidate
            if aspect_min <= aspect <= aspect_max:
                results.add((w, h_candidate))

    return sorted(results, key=lambda wh: (wh[0], wh[1]))


# ---------------------------------------------------------------------------
# Budget tier assignment
# ---------------------------------------------------------------------------

def assign_budget_tier(width: int, height: int) -> int:
    """Map an arbitrary (W, H) to the nearest megapixel anchor.

    Uses absolute pixel count difference to find the closest anchor.
    Ties are broken by choosing the smaller anchor.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        The nearest anchor pixel count from MEGAPIXEL_ANCHORS.
    """
    pixels = width * height
    best_anchor = MEGAPIXEL_ANCHORS[0]
    best_dist = abs(pixels - best_anchor)

    for anchor in MEGAPIXEL_ANCHORS[1:]:
        dist = abs(pixels - anchor)
        if dist < best_dist:
            best_dist = dist
            best_anchor = anchor

    return best_anchor


# ---------------------------------------------------------------------------
# Informational: items per bin
# ---------------------------------------------------------------------------

def items_per_bin(
    budget_pixels: int,
    reference_total_len: int = 4224,
    cap_tokens: int = 256,
    vae_scale: int = 8,
    patch_size: int = 2,
    pad_multiple: int = 32,
) -> int:
    """How many images at this budget fit in one FlexAttention bin.

    Uses the same formula as bin_packer.compute_effective_seq_len():
      img_tokens = budget_pixels / (vae_scale * patch_size)^2
      effective = pad(cap_tokens) + pad(img_tokens)
      items = floor(reference_total_len / effective)

    Note: cap_tokens default is 256 here (a generous upper bound), not the
    p90 value of 45 used in bin_packer. This gives a conservative estimate.
    For the actual training cap_tokens, use bin_packer.DEFAULT_CAP_TOKENS.

    Args:
        budget_pixels: Pixel count for one image.
        reference_total_len: Bin capacity in tokens. Default 4224
            (matches bin_packer.REFERENCE_TOTAL_LEN).
        cap_tokens: Caption token count. Default 256 (conservative).
        vae_scale: VAE spatial downscale. Default 8.
        patch_size: DiT patch size. Default 2.
        pad_multiple: Token padding alignment. Default 32.

    Returns:
        Number of items that fit in one bin (floor division).
    """
    divisor = vae_scale * patch_size
    img_tokens = budget_pixels // (divisor * divisor)

    def _pad(n: int, m: int) -> int:
        return n + ((-n) % m)

    img_padded = _pad(img_tokens, pad_multiple)
    cap_padded = _pad(cap_tokens, pad_multiple)
    effective = cap_padded + img_padded

    if effective <= 0:
        return 0
    return reference_total_len // effective


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_resolution(
    width: int,
    height: int,
    step: int = 32,
) -> None:
    """Validate that a resolution meets alignment and minimum size constraints.

    Raises ValueError if:
      - width or height < MIN_DIM (64)
      - width or height is not a multiple of step (32)

    This is a looser check than bin_packer.validate_resolution(), which
    checks VAE+patch alignment (16px). Here we check the sampling grid
    alignment (32px), which implies VAE+patch alignment since 32 is a
    multiple of 16.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        step: Required alignment. Default 32.
    """
    if width < MIN_DIM or height < MIN_DIM:
        raise ValueError(
            f"Resolution {width}x{height} has dimension < {MIN_DIM}. "
            f"Both width and height must be >= {MIN_DIM}."
        )
    if width % step != 0 or height % step != 0:
        raise ValueError(
            f"Resolution {width}x{height} not aligned to {step}. "
            f"Both width and height must be divisible by {step}."
        )


# ---------------------------------------------------------------------------
# Diagnostic: summarize resolution space per tier
# ---------------------------------------------------------------------------

def summarize_resolution_space(
    step: int = 32,
    aspect_min: float = 0.5,
    aspect_max: float = 2.0,
) -> dict[int, dict[str, Any]]:
    """Summarize the valid resolution space for each megapixel anchor.

    Returns a dict keyed by anchor pixel count, with per-anchor stats:
      - n_resolutions: number of valid (W, H) pairs
      - min_aspect, max_aspect: actual aspect ratio range
      - resolutions: list of (W, H) pairs
      - items_per_bin: how many fit in one FlexAttention bin

    For analysis/essay use.
    """
    summary = {}
    for anchor in MEGAPIXEL_ANCHORS:
        resolutions = enumerate_resolutions(anchor, step=step,
                                            aspect_min=aspect_min,
                                            aspect_max=aspect_max)
        aspects = [w / h for w, h in resolutions] if resolutions else [1.0]
        summary[anchor] = {
            "label": ANCHOR_LABELS.get(anchor, f"{anchor}px"),
            "n_resolutions": len(resolutions),
            "resolutions": resolutions,
            "min_aspect": min(aspects),
            "max_aspect": max(aspects),
            "items_per_bin_conservative": items_per_bin(anchor),
        }
    return summary

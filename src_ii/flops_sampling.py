"""Resolution-aware FLOPS sampling weights for funfetti batching.

Computes per-trajectory sampling weights so that the GPU's compute budget
is allocated according to a target FLOPS distribution across resolution
tiers. The core idea: small images cost fewer FLOPS per forward pass,
so they should be oversampled to fill the FLOPS budget allocated to their
resolution tier. Large images are expensive and are sampled at their
natural rate.

6-tier resolution bucketing via MEGAPIXEL_ANCHORS:
    65536   (256^2)   -- tiny
    102400  (320^2)   -- small
    147456  (384^2)   -- small-medium
    262144  (512^2)   -- medium
    495616  (704^2)   -- large
    1048576 (1024^2)  -- megapixel

Each trajectory is mapped to its nearest anchor via assign_budget_tier().
FLOPS targets are specified per tier (default: 33% on megapixel, 67%
distributed across non-mega tiers proportional to trajectory count).

Graceful degradation:
    - If the dataset has only one resolution, all get equal weight.
    - If a target tier has no trajectories, its FLOPS allocation
      is redistributed proportionally to other populated tiers.
    - If all trajectories are in one tier, that tier gets 100%.

Import constraints:
    - Pure Python + math only. No torch, no numpy.
    - Deterministic: same input -> same output.
"""

from __future__ import annotations

import math
from typing import Any

from src_ii.resolution_sampling import MEGAPIXEL_ANCHORS, assign_budget_tier


# ---------------------------------------------------------------------------
# Reference FLOPS computation
# ---------------------------------------------------------------------------

_REF_PIXELS = 1280 * 832  # 1,064,960 pixels (megapixel reference)


def _image_tokens(width: int, height: int, vae_scale: int = 8, patch_size: int = 2) -> int:
    """Compute the number of image tokens for a given resolution."""
    return (height // vae_scale // patch_size) * (width // vae_scale // patch_size)


def _attention_flops_ratio(width: int, height: int) -> float:
    """Compute attention FLOPS as a ratio to the 1280x832 reference.

    Attention FLOPS are proportional to seq_len^2 (quadratic in tokens).
    This function returns the ratio: (this_image_tokens / ref_tokens)^2.

    Returns:
        Float in (0, inf). 1.0 means same cost as the reference.
        Values < 1.0 mean cheaper. E.g., 256x256 returns ~0.000237.
    """
    ref_tokens = _image_tokens(1280, 832)  # = 4160
    img_tokens = _image_tokens(width, height)
    if ref_tokens == 0:
        return 1.0
    ratio = img_tokens / ref_tokens
    return ratio * ratio


def flops_units(width: int, height: int) -> float:
    """Compute FLOPS cost of one image in 1024^2-equivalent units.

    Uses 1024x1024 as the reference (1.0 unit). The cost is proportional
    to tokens^2 where tokens = (W * H) / 256 (latent patches).

    Reference:
        tokens_ref = 1024 * 1024 / 256 = 4096
        For resolution (W, H): tokens = (W * H) / 256
        flops_unit = (tokens / tokens_ref) ** 2

    Examples:
        1024x1024 -> 1.0 units
        1280x832  -> (4160/4096)^2 ~ 1.032 units
        704x704   -> (1936/4096)^2 ~ 0.223 units
        512x512   -> (1024/4096)^2 ~ 0.0625 units
        384x384   -> (576/4096)^2  ~ 0.0198 units
        320x320   -> (400/4096)^2  ~ 0.00953 units
        256x256   -> (256/4096)^2  ~ 0.00391 units

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        FLOPS cost in 1024^2-equivalent units. 1.0 = one 1024x1024 image.
    """
    tokens = (width * height) / 256.0  # = (W * H) / (vae_scale * patch_size)^2
    ref_tokens = 1048576 / 256.0  # = 4096, tokens for 1024x1024
    ratio = tokens / ref_tokens
    return ratio * ratio


def pair_flops_units(
    width_a: int, height_a: int,
    width_b: int, height_b: int,
) -> float:
    """Compute FLOPS cost of a pair of images in 1024^2-equivalent units.

    Simply the sum of flops_units() for both images. A pair of 1024^2
    images costs 2.0 units.

    Args:
        width_a, height_a: Resolution of image A.
        width_b, height_b: Resolution of image B.

    Returns:
        Total FLOPS cost of the pair in 1024^2-equivalent units.
    """
    return flops_units(width_a, height_a) + flops_units(width_b, height_b)


# ---------------------------------------------------------------------------
# Backward-compatible binary classification
# ---------------------------------------------------------------------------

# Kept for backward compatibility with code that imports _MEGAPIXEL_THRESHOLD
# and _classify_resolution. New code should use assign_budget_tier() instead.
_MEGAPIXEL_THRESHOLD = 1024 * 1024  # 1,048,576 pixels


def _classify_resolution(width: int, height: int) -> str:
    """Classify a resolution into 'megapixel' or 'small'.

    DEPRECATED: Use assign_budget_tier() for 6-tier classification.
    Kept for backward compatibility with pair_sampler and training_artifacts.
    """
    pixels = width * height
    if pixels >= _MEGAPIXEL_THRESHOLD:
        return "megapixel"
    else:
        return "small"


# ---------------------------------------------------------------------------
# 6-tier FLOPS sampling weight computation
# ---------------------------------------------------------------------------

# Default FLOPS targets: 33% on megapixel (1024^2), rest distributed
# proportionally across other populated tiers.
DEFAULT_TIER_FLOPS_TARGETS: dict[int, float] = {1048576: 0.33}


def compute_flops_sampling_weights(
    traj_resolutions: dict[int, tuple[int, int]],
    tier_flops_targets: dict[int, float] | None = None,
    # Backward-compatible parameters (used if tier_flops_targets is None):
    megapixel_flops_fraction: float = 0.33,
    small_flops_fraction: float = 0.67,
) -> dict[int, float]:
    """Compute per-trajectory sampling weights for resolution-aware training.

    Supports two modes:
    1. 6-tier mode (tier_flops_targets provided): Each anchor pixel count
       maps to a target FLOPS fraction. Unspecified tiers share the
       remaining FLOPS proportionally by trajectory count.
    2. Binary mode (tier_flops_targets=None): Falls back to the legacy
       megapixel/small binary split for backward compatibility.

    Algorithm (6-tier mode):
        1. Assign each trajectory to its nearest megapixel anchor.
        2. For each populated tier, determine its target FLOPS fraction:
           - Tiers in tier_flops_targets get their specified fraction.
           - Other populated tiers share the remainder proportionally
             by trajectory count.
        3. Per-trajectory weight = tier_fraction / (n_trajs_in_tier * flops_ratio_i)
        4. Normalize so weights sum to 1.

    Graceful degradation:
        - If a target tier has no trajectories, its fraction is redistributed.
        - If only one tier has trajectories, all get equal weight.
        - Monoresolution datasets degenerate to uniform.

    Args:
        traj_resolutions: Dict mapping trajectory ID to (width, height).
        tier_flops_targets: Dict mapping anchor pixel count to target FLOPS
            fraction. E.g., {1048576: 0.33} means 33% on megapixel tier.
            Remaining fraction is distributed across other tiers.
            If None, uses binary megapixel/small split (backward compat).
        megapixel_flops_fraction: Legacy param. Used only when
            tier_flops_targets is None.
        small_flops_fraction: Legacy param. Used only when
            tier_flops_targets is None.

    Returns:
        Dict mapping trajectory ID to normalized sampling weight.
        All weights are positive. Weights sum to ~1.0.
    """
    if not traj_resolutions:
        return {}

    # 6-tier mode
    if tier_flops_targets is not None:
        return _compute_weights_6tier(traj_resolutions, tier_flops_targets)

    # Legacy binary mode
    return _compute_weights_binary(
        traj_resolutions, megapixel_flops_fraction, small_flops_fraction,
    )


def _compute_weights_6tier(
    traj_resolutions: dict[int, tuple[int, int]],
    tier_flops_targets: dict[int, float],
) -> dict[int, float]:
    """6-tier weight computation."""
    # Step 1: Assign trajectories to tiers
    tier_trajs: dict[int, list[int]] = {a: [] for a in MEGAPIXEL_ANCHORS}
    traj_flops: dict[int, float] = {}

    for traj_id, (w, h) in traj_resolutions.items():
        anchor = assign_budget_tier(w, h)
        tier_trajs[anchor].append(traj_id)
        traj_flops[traj_id] = _attention_flops_ratio(w, h)

    # Step 2: Determine populated tiers
    populated = {a: tids for a, tids in tier_trajs.items() if tids}
    if not populated:
        return {}

    # Step 3: Compute effective FLOPS fractions
    # Specified tiers get their target (if populated), remaining is distributed
    specified_total = 0.0
    specified_tiers = set()
    for anchor, frac in tier_flops_targets.items():
        if anchor in populated:
            specified_total += frac
            specified_tiers.add(anchor)

    remaining_frac = max(0.0, 1.0 - specified_total)
    unspecified_populated = {a: tids for a, tids in populated.items()
                            if a not in specified_tiers}

    # Distribute remaining fraction across unspecified tiers
    # proportionally by trajectory count
    total_unspec_trajs = sum(len(tids) for tids in unspecified_populated.values())

    tier_fracs: dict[int, float] = {}
    for anchor in populated:
        if anchor in specified_tiers:
            tier_fracs[anchor] = tier_flops_targets[anchor]
        elif total_unspec_trajs > 0:
            tier_fracs[anchor] = remaining_frac * len(populated[anchor]) / total_unspec_trajs
        else:
            tier_fracs[anchor] = 0.0

    # Normalize fractions (handle edge case where specified fracs > 1.0)
    total_frac = sum(tier_fracs.values())
    if total_frac > 0:
        for a in tier_fracs:
            tier_fracs[a] /= total_frac

    # Step 4: Compute per-trajectory weights
    weights: dict[int, float] = {}
    for anchor, tids in populated.items():
        frac = tier_fracs.get(anchor, 0.0)
        if frac <= 0:
            for tid in tids:
                weights[tid] = 0.0
            continue

        n = len(tids)
        for tid in tids:
            flops = traj_flops[tid]
            if flops <= 0:
                flops = 1e-10
            weights[tid] = frac / (n * flops)

    # Step 5: Normalize
    total_w = sum(weights.values())
    if total_w > 0:
        for tid in weights:
            weights[tid] /= total_w

    return weights


def _compute_weights_binary(
    traj_resolutions: dict[int, tuple[int, int]],
    megapixel_flops_fraction: float,
    small_flops_fraction: float,
) -> dict[int, float]:
    """Legacy binary (megapixel/small) weight computation.

    Preserved for backward compatibility with existing training scripts.
    """
    buckets: dict[str, list[int]] = {"megapixel": [], "small": []}
    traj_flops: dict[int, float] = {}

    for traj_id, (w, h) in traj_resolutions.items():
        bucket = _classify_resolution(w, h)
        buckets[bucket].append(traj_id)
        traj_flops[traj_id] = _attention_flops_ratio(w, h)

    n_mega = len(buckets["megapixel"])
    n_small = len(buckets["small"])

    if n_mega == 0 and n_small == 0:
        return {}

    if n_mega == 0:
        effective_mega_frac = 0.0
        effective_small_frac = 1.0
    elif n_small == 0:
        effective_mega_frac = 1.0
        effective_small_frac = 0.0
    else:
        effective_mega_frac = megapixel_flops_fraction
        effective_small_frac = small_flops_fraction

    total_frac = effective_mega_frac + effective_small_frac
    if total_frac > 0:
        effective_mega_frac /= total_frac
        effective_small_frac /= total_frac

    weights: dict[int, float] = {}
    for bucket_name, bucket_frac in [("megapixel", effective_mega_frac),
                                      ("small", effective_small_frac)]:
        traj_ids = buckets[bucket_name]
        if not traj_ids or bucket_frac <= 0:
            for tid in traj_ids:
                weights[tid] = 0.0
            continue
        for tid in traj_ids:
            flops = traj_flops[tid]
            if flops <= 0:
                flops = 1e-10
            weights[tid] = bucket_frac / (len(traj_ids) * flops)

    total_w = sum(weights.values())
    if total_w > 0:
        for tid in weights:
            weights[tid] /= total_w

    return weights


# ---------------------------------------------------------------------------
# Convenience: from positions
# ---------------------------------------------------------------------------

def compute_flops_sampling_weights_from_positions(
    positions: list,  # list[_ImagePosition]
    tier_flops_targets: dict[int, float] | None = None,
    megapixel_flops_fraction: float = 0.33,
    small_flops_fraction: float = 0.67,
) -> dict[int, float]:
    """Convenience: compute FLOPS weights directly from _ImagePosition list.

    Extracts per-trajectory resolution from the first position of each
    trajectory (all positions within a trajectory share the same resolution).

    Args:
        positions: List of _ImagePosition objects.
        tier_flops_targets: 6-tier FLOPS targets. See compute_flops_sampling_weights().
        megapixel_flops_fraction: Legacy param for binary mode.
        small_flops_fraction: Legacy param for binary mode.

    Returns:
        Dict mapping trajectory ID to sampling weight.
    """
    traj_resolutions: dict[int, tuple[int, int]] = {}
    for pos in positions:
        if pos.traj_id not in traj_resolutions:
            traj_resolutions[pos.traj_id] = (pos.width, pos.height)

    return compute_flops_sampling_weights(
        traj_resolutions,
        tier_flops_targets=tier_flops_targets,
        megapixel_flops_fraction=megapixel_flops_fraction,
        small_flops_fraction=small_flops_fraction,
    )


# ---------------------------------------------------------------------------
# Diagnostic: weight summary
# ---------------------------------------------------------------------------

def summarize_flops_weights(
    weights: dict[int, float],
    traj_resolutions: dict[int, tuple[int, int]],
) -> dict[str, Any]:
    """Summarize FLOPS sampling weights for observability.

    Reports per-tier statistics using the 6-tier system, plus legacy
    binary bucket stats for backward compatibility.

    Args:
        weights: Per-trajectory weights from compute_flops_sampling_weights().
        traj_resolutions: Per-trajectory resolutions.

    Returns:
        Dict with keys: n_trajectories, tiers (6-tier stats),
        buckets (legacy binary stats for backward compat).
    """
    # 6-tier stats
    tier_stats: dict[int, dict] = {}
    for anchor in MEGAPIXEL_ANCHORS:
        tids = [
            tid for tid, (w, h) in traj_resolutions.items()
            if assign_budget_tier(w, h) == anchor
        ]
        if not tids:
            tier_stats[anchor] = {
                "label": f"{int(math.sqrt(anchor))}sq",
                "n_trajectories": 0,
                "total_weight": 0.0,
                "mean_weight": 0.0,
                "resolutions": [],
            }
            continue

        ws = [weights.get(tid, 0.0) for tid in tids]
        unique_res = sorted(set(traj_resolutions[tid] for tid in tids))
        tier_stats[anchor] = {
            "label": f"{int(math.sqrt(anchor))}sq",
            "n_trajectories": len(tids),
            "total_weight": sum(ws),
            "mean_weight": sum(ws) / len(ws),
            "min_weight": min(ws),
            "max_weight": max(ws),
            "resolutions": [f"{w}x{h}" for w, h in unique_res],
        }

    # Legacy binary bucket stats (backward compat)
    bucket_stats: dict[str, dict] = {}
    for bucket_name in ("megapixel", "small"):
        tids = [
            tid for tid, (w, h) in traj_resolutions.items()
            if _classify_resolution(w, h) == bucket_name
        ]
        if not tids:
            bucket_stats[bucket_name] = {
                "n_trajectories": 0,
                "total_weight": 0.0,
                "mean_weight": 0.0,
                "min_weight": 0.0,
                "max_weight": 0.0,
                "resolutions": [],
            }
            continue

        ws = [weights.get(tid, 0.0) for tid in tids]
        unique_res = sorted(set(traj_resolutions[tid] for tid in tids))
        bucket_stats[bucket_name] = {
            "n_trajectories": len(tids),
            "total_weight": sum(ws),
            "mean_weight": sum(ws) / len(ws) if ws else 0.0,
            "min_weight": min(ws) if ws else 0.0,
            "max_weight": max(ws) if ws else 0.0,
            "resolutions": [f"{w}x{h}" for w, h in unique_res],
        }

    return {
        "n_trajectories": len(weights),
        "tiers": tier_stats,
        "buckets": bucket_stats,  # backward compat
    }

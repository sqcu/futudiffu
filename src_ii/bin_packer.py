"""Bin packing scheduler for mixed-resolution FlexAttention batches.

Pure Python. No torch or GPU dependency. Deterministic: same input = same output.

The problem: FlexAttention supports packing multiple images into one attention
computation. Each image has a sequence length determined by its resolution:
    seq_len = (latent_H / patch_size) * (latent_W / patch_size)
where latent dimensions = pixel dimensions / vae_scale, and patch_size is the
DiT patchification factor.

The max sequence length per batch slot is determined by the reference resolution
(typically 4160 for 1280x832). Mixed-resolution generation produces items with
wildly different seq_len; small images can be packed many-to-one into the same
FlexAttention forward pass for free throughput (the reference kernel is already
saturated at full-res, so smaller images contribute no additional compute).

IMPORTANT: This architecture (NextDiT / Z-Image) uses CONCATENATED text
conditioning -- text tokens are prepended to image patch tokens in the SAME
self-attention sequence:
    padded_full_embed = torch.cat([cap_feats_embedded, x_patches], dim=1)
Both text and image tokens share sequence capacity. When packing multiple
images, each image brings its OWN text tokens. The effective per-item cost is:
    pad_to_32(cap_tokens) + pad_to_32(img_seq_len)

Algorithm: First-Fit Decreasing (FFD). O(n*m) where n=items, m=bins. Fast
enough for generation plans of any realistic size (<100K items).

Import constraints:
  - Pure Python only (no torch, no numpy, no futudiffu imports)
  - Deterministic (same input -> same output)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Resolution-dependent sigma shifting (SD3 Eq.23)
# ---------------------------------------------------------------------------

# Duplicated here (pure Python, no torch) to keep bin_packer import-free
# of torch. The canonical implementation is in sigma_schedule.resolution_shift().

_REF_PIXELS = 1280 * 832  # 1,064,960
_MAX_SHIFT = 8.0  # Must match sigma_schedule.MAX_SHIFT


def _resolution_shift(width: int, height: int) -> float:
    """SD3 Eq.23: alpha = sqrt(ref_pixels / target_pixels), clamped.

    Pure-Python duplicate of sigma_schedule.resolution_shift() to avoid
    importing torch into this module.  Capped at _MAX_SHIFT to prevent
    degenerate Euler schedules for very small images.
    """
    target_pixels = width * height
    if target_pixels <= 0:
        raise ValueError(f"Invalid resolution: {width}x{height}")
    return min(math.sqrt(_REF_PIXELS / target_pixels), _MAX_SHIFT)


# ---------------------------------------------------------------------------
# Sequence length computation
# ---------------------------------------------------------------------------

def compute_seq_len(
    width: int,
    height: int,
    vae_scale: int = 8,
    patch_size: int = 2,
) -> int:
    """Compute FlexAttention IMAGE sequence length for an image.

    The sequence length is the number of patch tokens after VAE encoding and
    DiT patchification. This is the IMAGE-ONLY token count; for the total
    sequence cost including text tokens, use compute_effective_seq_len().

    Args:
        width: Image width in pixels. Must be divisible by vae_scale * patch_size.
        height: Image height in pixels. Must be divisible by vae_scale * patch_size.
        vae_scale: VAE spatial downscale factor (default 8).
        patch_size: DiT patch size (default 2).

    Returns:
        Number of patch tokens (image-only sequence length).
    """
    latent_h = height // vae_scale
    latent_w = width // vae_scale
    return (latent_h // patch_size) * (latent_w // patch_size)


def _pad_to_multiple(n: int, multiple: int) -> int:
    """Round up n to the next multiple of `multiple`."""
    return n + ((-n) % multiple)


def compute_effective_seq_len(
    width: int,
    height: int,
    cap_tokens: int,
    pad_multiple: int = 32,
    vae_scale: int = 8,
    patch_size: int = 2,
) -> int:
    """Compute the total padded sequence cost of one image in a packed batch.

    The packed sequence layout (from build_packed_sequence) is:
        [text_0_padded, img_0_padded, text_1_padded, img_1_padded, ...]
    where each segment is padded to pad_multiple (32 for Z-Image NextDiT).

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        cap_tokens: Raw caption/text token count (before padding).
        pad_multiple: Pad each segment to this multiple (default 32).
        vae_scale: VAE spatial downscale factor (default 8).
        patch_size: DiT patch size (default 2).

    Returns:
        Total padded tokens: pad(cap_tokens) + pad(img_seq_len).
    """
    img_raw = compute_seq_len(width, height, vae_scale, patch_size)
    img_padded = _pad_to_multiple(img_raw, pad_multiple)
    cap_padded = _pad_to_multiple(cap_tokens, pad_multiple)
    return cap_padded + img_padded


def validate_resolution(
    width: int,
    height: int,
    step: int = 32,
    vae_scale: int = 8,
    patch_size: int = 2,
) -> None:
    """Validate that a resolution meets all pipeline constraints.

    Checks:
      1. Width and height are positive and >= 64 (MIN_DIM)
      2. Divisible by `step` (default 32, the resolution sampling grid)
      3. Divisible by vae_scale * patch_size (16, the latent alignment)

    Since step=32 is a multiple of vae_scale*patch_size=16, checking step
    alignment is sufficient. The vae_scale/patch_size check is kept for
    clarity and for callers who pass step=16.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        step: Resolution sampling grid alignment (default 32).
        vae_scale: VAE spatial downscale factor (default 8).
        patch_size: DiT patch size (default 2).
    """
    _MIN_DIM = 64
    if width <= 0 or height <= 0:
        raise ValueError(f"Resolution must be positive: {width}x{height}")

    if width < _MIN_DIM or height < _MIN_DIM:
        raise ValueError(
            f"Resolution {width}x{height} has dimension < {_MIN_DIM}. "
            f"Both width and height must be >= {_MIN_DIM}."
        )

    if width % step != 0 or height % step != 0:
        raise ValueError(
            f"Resolution {width}x{height} not aligned to {step}. "
            f"Both width and height must be divisible by {step}."
        )

    # Minimum alignment: vae_scale * patch_size = 16
    min_align = vae_scale * patch_size
    if width % min_align != 0 or height % min_align != 0:
        raise ValueError(
            f"Resolution {width}x{height} not aligned to {min_align}. "
            f"Width and height must be divisible by {min_align} "
            f"(vae_scale={vae_scale} * patch_size={patch_size})."
        )


# ---------------------------------------------------------------------------
# Resolution tiers (DEPRECATED -- use resolution_sampling.py instead)
# ---------------------------------------------------------------------------

# DEPRECATED: Hardcoded resolution enumerations. New code should use
# resolution_sampling.sample_random_resolution() to sample (W, H) from
# megapixel budgets + aspect ratios. These are kept for backward compatibility
# with existing generation scripts and build_generation_plan().
#
# For the 6-tier megapixel anchor system, see:
#   src_ii/resolution_sampling.py -- MEGAPIXEL_ANCHORS, sample_resolution(), etc.
#   src_ii/flops_sampling.py -- 6-tier FLOPS weight computation

RESOLUTION_TIERS: dict[str, dict] = {
    "full": {
        # ~1024x1024 area (existing production behavior)
        "resolutions": [
            (1280, 832),   # 16:~10.4 landscape
            (832, 1280),   # ~10.4:16 portrait
            (1024, 1024),  # 1:1 square
        ],
        "rollouts_per_prompt": 1,
    },
    "medium": {
        # ~512x512 area
        "resolutions": [
            (512, 512),   # 1:1
            (640, 384),   # ~5:3 landscape
            (384, 640),   # ~3:5 portrait
            (576, 448),   # ~9:7 landscape
            (448, 576),   # ~7:9 portrait
        ],
        "rollouts_per_prompt": 2,
    },
    "small": {
        # ~256x256 area
        "resolutions": [
            (256, 256),   # 1:1
            (320, 192),   # ~5:3 landscape
            (192, 320),   # ~3:5 portrait
            (288, 224),   # ~9:7 landscape
            (224, 288),   # ~7:9 portrait
        ],
        "rollouts_per_prompt": 4,
    },
}


def get_tier_resolutions(tier_name: str) -> list[tuple[int, int]]:
    """Get the resolution options for a named tier.

    DEPRECATED: Use resolution_sampling.enumerate_resolutions() instead.

    Args:
        tier_name: One of "full", "medium", "small".

    Returns:
        List of (width, height) tuples.

    Raises:
        KeyError: If tier_name is not recognized.
    """
    if tier_name not in RESOLUTION_TIERS:
        raise KeyError(
            f"Unknown resolution tier '{tier_name}'. "
            f"Available: {sorted(RESOLUTION_TIERS.keys())}"
        )
    return list(RESOLUTION_TIERS[tier_name]["resolutions"])


def get_tier_rollouts_per_prompt(tier_name: str) -> int:
    """Get the default rollouts-per-prompt for a named tier.

    DEPRECATED: Use per-anchor trajectory counts instead.

    Args:
        tier_name: One of "full", "medium", "small".

    Returns:
        Default number of rollouts per prompt for the tier.
    """
    return RESOLUTION_TIERS[tier_name]["rollouts_per_prompt"]


# ---------------------------------------------------------------------------
# Reference sequence lengths
# ---------------------------------------------------------------------------

# Image-only reference: the raw patch token count for 1280x832.
# Kept for backward compatibility with code that uses the image-only value.
REFERENCE_SEQ_LEN = compute_seq_len(1280, 832)  # = 4160

# Default caption token estimate: p90 of the actual BTRM V2 dataset prompt
# distribution (33 unique prompts, Qwen3-4B tokenizer with Z-Image chat
# template). Measured by scripts_ii/measure_prompt_tokens.py:
#   min=22, max=113, mean=34.7, median=31, p90=45, p95=45, p99=113
# Padded to 32: p90=64 tokens after padding.
DEFAULT_CAP_TOKENS = 45

# Total reference length INCLUDING text token overhead.
# This is the correct bin capacity for packing: it represents the actual
# sequence length that a single reference-resolution (1280x832) image
# consumes in the FlexAttention forward pass, including its text tokens.
REFERENCE_TOTAL_LEN = compute_effective_seq_len(1280, 832, cap_tokens=DEFAULT_CAP_TOKENS)


# ---------------------------------------------------------------------------
# Utilization target rationale
# ---------------------------------------------------------------------------
#
# The target bin utilization is NOT 100%. FlexAttention with block masks
# provides a natural efficiency recovery for underfilled bins:
#
#   - Each packed image only self-attends to its own tokens (block-diagonal
#     attention via document_id masks). Masked-out blocks are SKIPPED, not
#     computed-then-zeroed.
#
#   - For a bin at X% utilization with N images of equal size:
#       Dense cost:  O(capacity^2)
#       Sparse cost: O(sum(si^2))  where si is each image's effective_seq_len
#       Ratio:       sum(si^2) / capacity^2
#     At 81% utilization with 1 image: sparse ratio = 0.66 (34% savings).
#     At 50% utilization with 2 equal images: sparse ratio = 0.125 each.
#
#   - The right target: ~90% utilization for the top 90% of sampled configs.
#     The remaining 10% (unusual prompt lengths, edge-case resolutions) can
#     run at lower utilization without penalty thanks to FlexAttention
#     sparsity. Underfilled bins are NOT wasted compute.
#
#   - 13/16 = 81% tenancy is fine. The attention mask zeros out unused slots,
#     and the block-sparse kernel skips them entirely.


# ---------------------------------------------------------------------------
# FlexAttention sparsity compute estimation
# ---------------------------------------------------------------------------

def estimate_sparse_compute_ratio(utilization: float) -> float:
    """Estimate actual compute fraction for a bin at given utilization.

    Assumes block-diagonal attention (each packed image only attends to
    itself). This is the standard FlexAttention layout for packed batches
    with document_id masks.

    For a single image filling `utilization` fraction of a bin:
        Dense cost:  O(C^2) where C = bin capacity
        Sparse cost: O((utilization * C)^2) = O(utilization^2 * C^2)
        Ratio:       utilization^2

    This is the lower bound (one image). With N equal-sized images each
    of size s = utilization * C / N:
        Sparse cost: N * s^2 = N * (utilization * C / N)^2
                    = utilization^2 * C^2 / N
        Ratio:       utilization^2 / N

    So more images in a bin = LESS compute per unit capacity (each image's
    quadratic cost is smaller). The single-image case is the worst case.

    Args:
        utilization: Fraction of bin capacity used (0 to 1).

    Returns:
        Estimated compute as fraction of dense (full-capacity) compute.
        Range [0, 1]. Lower is better (more savings from sparsity).
    """
    if utilization <= 0.0:
        return 0.0
    if utilization >= 1.0:
        return 1.0
    # Worst case: single image (most conservative estimate)
    return utilization * utilization


def estimate_sparse_compute_ratio_detailed(
    effective_seq_lens: list[int],
    capacity: int,
) -> float:
    """Compute exact sparse compute ratio for a specific bin layout.

    Given the actual effective_seq_len of each item in a bin and the bin
    capacity, computes:
        sum(si^2) / capacity^2

    This is the ratio of actual block-diagonal attention FLOPs to the
    hypothetical dense attention FLOPs for the full capacity.

    Args:
        effective_seq_lens: List of per-item effective sequence lengths.
        capacity: Bin capacity (total sequence length).

    Returns:
        Compute ratio in [0, 1]. 1.0 means no savings from sparsity.
    """
    if capacity <= 0:
        return 0.0
    sum_sq = sum(s * s for s in effective_seq_lens)
    return sum_sq / (capacity * capacity)


# ---------------------------------------------------------------------------
# Bin packing
# ---------------------------------------------------------------------------

@dataclass
class Bin:
    """A bin in the packing schedule. Each bin becomes one FlexAttention forward pass."""
    items: list[dict] = field(default_factory=list)
    used_seq_len: int = 0
    capacity: int = 0

    def remaining(self) -> int:
        return self.capacity - self.used_seq_len

    def can_fit(self, seq_len: int) -> bool:
        return self.used_seq_len + seq_len <= self.capacity

    def add(self, item: dict) -> None:
        self.items.append(item)
        self.used_seq_len += item["seq_len"]


class BinPackScheduler:
    """Greedy bin packing for mixed-resolution FlexAttention batches.

    Given a list of items with sequence lengths, packs them into bins
    where the total sequence length per bin does not exceed max_seq_len.

    The algorithm is First-Fit Decreasing (FFD):
      1. Sort items by seq_len descending (largest first)
      2. For each item, place it in the first bin with enough remaining capacity
      3. If no bin fits, open a new bin

    This is a simple, well-studied heuristic that provides good utilization
    (provably within 11/9 * OPT + 6/9 bins of optimal) and runs in O(n*m)
    time where n = number of items and m = number of bins.

    Properties:
      - Deterministic: same input list -> same bin assignment
      - Stable: items within each bin appear in the same order as the
        sorted input (largest to smallest seq_len)
      - No torch dependency
    """

    def __init__(
        self,
        max_seq_len: int = REFERENCE_TOTAL_LEN,
        default_cap_tokens: int = DEFAULT_CAP_TOKENS,
    ):
        """Initialize the scheduler.

        Args:
            max_seq_len: Maximum total sequence length per bin. Default is
                REFERENCE_TOTAL_LEN (1280x832 image + p90 text overhead).
                For backward compatibility, callers can pass REFERENCE_SEQ_LEN
                to get the old image-only behavior.
            default_cap_tokens: Default caption token count to assume when
                items don't specify cap_tokens. Based on p90 of measured
                prompt token distribution (45 tokens, Qwen3-4B tokenizer).
        """
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        self.max_seq_len = max_seq_len
        self.default_cap_tokens = default_cap_tokens

    def pack(self, items: list[dict]) -> list[list[dict]]:
        """Pack items into bins using first-fit-decreasing.

        Each item dict MUST have a 'seq_len' field (int). All other fields
        are passed through unchanged.

        Items with seq_len > max_seq_len are placed in their own bin
        (they cannot share with anything else). This is not an error --
        the FlexAttention kernel can handle any single image regardless
        of sequence length -- but they will have poor utilization.

        Args:
            items: List of dicts, each with at least a 'seq_len' key.

        Returns:
            List of bins, where each bin is a list of item dicts.
            Bins are ordered by creation time (first bin filled first).
            Items within each bin are ordered largest-seq_len-first.
        """
        if not items:
            return []

        # Sort by seq_len descending (FFD). Secondary sort by original index
        # for determinism when seq_lens are equal.
        indexed = [(i, item) for i, item in enumerate(items)]
        indexed.sort(key=lambda pair: (-pair[1]["seq_len"], pair[0]))

        bins: list[Bin] = []

        for _, item in indexed:
            seq_len = item["seq_len"]
            placed = False

            # First-fit: try to place in the first bin with enough room
            for b in bins:
                if b.can_fit(seq_len):
                    b.add(item)
                    placed = True
                    break

            if not placed:
                # Open a new bin
                new_bin = Bin(capacity=self.max_seq_len)
                new_bin.add(item)
                bins.append(new_bin)

        return [b.items for b in bins]

    def pack_generation_plan(self, plan: list[dict]) -> list[list[dict]]:
        """Pack a generation plan (list of trajectory specs) into bins.

        Each item in the plan must have 'width' and 'height' fields.
        This method computes the EFFECTIVE seq_len for each item (including
        text token overhead when cap_tokens is available), then packs.

        If an item has a 'cap_tokens' field, it is used for the text
        overhead calculation. Otherwise, self.default_cap_tokens is used.

        Additional fields (prompt, seed, attention_backend, etc.) are
        passed through untouched. Each bin becomes one FlexAttention
        forward pass.

        Args:
            plan: List of dicts, each with at least 'width' and 'height'.
                  Optionally 'cap_tokens' for per-item text token counts.

        Returns:
            List of bins (list of list of dicts), each dict augmented
            with 'seq_len' (effective total including text overhead)
            and 'img_seq_len' (image-only token count).
        """
        augmented = []
        for item in plan:
            item_copy = dict(item)
            img_seq_len = compute_seq_len(item["width"], item["height"])
            cap_tokens = item.get("cap_tokens", self.default_cap_tokens)
            effective = compute_effective_seq_len(
                item["width"], item["height"], cap_tokens,
            )
            item_copy["img_seq_len"] = img_seq_len
            item_copy["cap_tokens"] = cap_tokens
            item_copy["seq_len"] = effective
            augmented.append(item_copy)
        return self.pack(augmented)

    def pack_for_training(
        self,
        items: list[tuple[int, str, int, int]],
    ) -> list[list[tuple[int, str, int, int]]]:
        """Pack training items into FlexAttention batches.

        Convenience method for the training pipeline. Each item is a
        (traj_id, step_key, width, height) tuple. Returns bins of tuples.

        Args:
            items: List of (traj_id, step_key, width, height) tuples.

        Returns:
            List of bins, each bin being a list of the same tuples.
        """
        # Convert to dicts for the generic packer
        dict_items = [
            {
                "traj_id": traj_id,
                "step_key": step_key,
                "width": width,
                "height": height,
                "seq_len": compute_effective_seq_len(
                    width, height, self.default_cap_tokens,
                ),
            }
            for traj_id, step_key, width, height in items
        ]

        bins = self.pack(dict_items)

        # Convert back to tuples
        return [
            [(d["traj_id"], d["step_key"], d["width"], d["height"]) for d in b]
            for b in bins
        ]

    def estimate_efficiency(self, bins: list[list[dict]]) -> dict[str, Any]:
        """Report packing efficiency statistics.

        Args:
            bins: Output of pack() or pack_generation_plan().

        Returns:
            Dict with:
                n_bins: Total number of bins.
                n_items: Total number of items across all bins.
                total_capacity: Sum of max_seq_len across all bins.
                total_used: Sum of used seq_len across all bins.
                total_wasted: total_capacity - total_used.
                utilization: total_used / total_capacity (0-1).
                sparse_compute_ratio: Estimated actual compute as fraction
                    of dense compute (accounts for FlexAttention sparsity).
                per_bin: List of dicts, each with 'n_items', 'used',
                    'capacity', 'utilization', 'wasted',
                    'sparse_compute_ratio'.
        """
        if not bins:
            return {
                "n_bins": 0,
                "n_items": 0,
                "total_capacity": 0,
                "total_used": 0,
                "total_wasted": 0,
                "utilization": 0.0,
                "sparse_compute_ratio": 0.0,
                "per_bin": [],
            }

        per_bin = []
        total_used = 0
        total_items = 0

        for b in bins:
            used = sum(item["seq_len"] for item in b)
            n = len(b)
            total_used += used
            total_items += n
            utilization = used / self.max_seq_len if self.max_seq_len > 0 else 0.0

            # Compute detailed sparse ratio for this bin
            item_lens = [item["seq_len"] for item in b]
            sparse_ratio = estimate_sparse_compute_ratio_detailed(
                item_lens, self.max_seq_len,
            )

            per_bin.append({
                "n_items": n,
                "used": used,
                "capacity": self.max_seq_len,
                "utilization": utilization,
                "wasted": self.max_seq_len - used,
                "sparse_compute_ratio": round(sparse_ratio, 4),
            })

        total_capacity = self.max_seq_len * len(bins)
        total_utilization = total_used / total_capacity if total_capacity > 0 else 0.0

        # Aggregate sparse compute ratio across all bins
        total_sparse_numer = sum(
            sum(item["seq_len"] ** 2 for item in b) for b in bins
        )
        total_sparse_denom = self.max_seq_len ** 2 * len(bins)
        total_sparse_ratio = (
            total_sparse_numer / total_sparse_denom
            if total_sparse_denom > 0 else 0.0
        )

        return {
            "n_bins": len(bins),
            "n_items": total_items,
            "total_capacity": total_capacity,
            "total_used": total_used,
            "total_wasted": total_capacity - total_used,
            "utilization": total_utilization,
            "sparse_compute_ratio": round(total_sparse_ratio, 4),
            "per_bin": per_bin,
        }


# ---------------------------------------------------------------------------
# Convenience: build a generation plan from tier specs
# ---------------------------------------------------------------------------

def build_generation_plan(
    prompts: list[str],
    seeds: list[int],
    resolution_tiers: list[str],
    attention_backends: list[str],
    n_steps: int = 30,
    cfg: float = 4.0,
    sparse_steps: list[int] | None = None,
    cap_tokens_per_prompt: dict[int, int] | None = None,
) -> list[dict]:
    """Build a flat generation plan from the Cartesian product of parameters.

    For each prompt, iterates over resolution_tiers, and within each tier
    selects (rollouts_per_prompt * len(attention_backends)) configurations
    by cycling through the tier's available resolutions and the provided
    seeds.

    The plan is deterministic given the same inputs.

    Args:
        prompts: List of prompt strings.
        seeds: List of PRNG seeds. Cycled if fewer than needed.
        resolution_tiers: List of tier names ("full", "medium", "small").
        attention_backends: List of attention backends ("sdpa", "sage").
        n_steps: Number of diffusion steps.
        cfg: CFG guidance scale.
        sparse_steps: Step indices to save. Default: [0, 4, 9, 14, 19, 24, 29].
        cap_tokens_per_prompt: Optional mapping from prompt_idx to token count.
            If provided, each plan item gets a 'cap_tokens' field for
            text-aware bin packing. If not provided, the bin packer will
            fall back to its default_cap_tokens estimate.

    Returns:
        Flat list of generation item dicts, each with:
            prompt, prompt_idx, seed, width, height, n_steps, cfg,
            attention_backend, resolution_tier, sparse_steps, batch_type,
            sampling_shift (resolution-dependent, SD3 Eq.23),
            and optionally cap_tokens.
    """
    if sparse_steps is None:
        sparse_steps = [0, 4, 9, 14, 19, 24, 29]

    plan: list[dict] = []
    seed_idx = 0

    for prompt_idx, prompt in enumerate(prompts):
        for tier_name in resolution_tiers:
            tier = RESOLUTION_TIERS[tier_name]
            resolutions = tier["resolutions"]
            rollouts_per_prompt = tier["rollouts_per_prompt"]

            for backend in attention_backends:
                for rollout_i in range(rollouts_per_prompt):
                    # Cycle through resolutions
                    w, h = resolutions[rollout_i % len(resolutions)]
                    # Cycle through seeds
                    seed = seeds[seed_idx % len(seeds)]
                    seed_idx += 1

                    item = {
                        "prompt": prompt,
                        "prompt_idx": prompt_idx,
                        "seed": seed,
                        "width": w,
                        "height": h,
                        "n_steps": n_steps,
                        "cfg": cfg,
                        "attention_backend": backend,
                        "resolution_tier": tier_name,
                        "sparse_steps": list(sparse_steps),
                        "batch_type": "t2i",
                        "sampling_shift": _resolution_shift(w, h),
                    }

                    if cap_tokens_per_prompt is not None and prompt_idx in cap_tokens_per_prompt:
                        item["cap_tokens"] = cap_tokens_per_prompt[prompt_idx]

                    plan.append(item)

    return plan

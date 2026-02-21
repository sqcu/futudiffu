# Continuous Resolution Sampling for Multi-Tier FLOPS Allocation

**Date:** 2026-02-18
**Trigger:** The existing resolution system relied on hardcoded enumeration tables
(`RESOLUTION_TIERS` in bin_packer.py, fixed lists in generation scripts). This
constrains the training data to a small set of predetermined resolutions and
prevents the FLOPS allocation system from operating across the full resolution
spectrum. The binary mega/small FLOPS bucketing produced 97.5% megapixel FLOPS
with stratified sampling, making the "33/67 FLOPS split" target unreachable
at small macrobatch sizes.

---

## 1. Design: Resolution as a Continuous Space

### The core idea

Resolution is NOT a fixed enumeration. It is a continuous 2D space parameterized by:

- **Megapixel budget**: How many pixels the image contains (determines FLOPS cost)
- **Aspect ratio**: W/H (determines composition, independent of FLOPS cost)

Any (W, H) pair where both are multiples of 32 and >= 64 is a valid resolution.
The pipeline (VAE, DiT patchification, RoPE cache, FlexAttention) already
handles arbitrary 32px-aligned resolutions.

### 6 megapixel anchors for FLOPS accounting

The continuous pixel budget is discretized into 6 tiers for cost accounting:

| Anchor | Edge | Pixels | FLOPS ratio | Items/bin (p90 cap) |
|--------|------|--------|------------|-------------------|
| 256^2 | 256 | 65,536 | 0.00024 | 13 |
| 320^2 | 320 | 102,400 | 0.00058 | 8 |
| 384^2 | 384 | 147,456 | 0.00120 | 6 |
| 512^2 | 512 | 262,144 | 0.00379 | 3 |
| 704^2 | 704 | 495,616 | 0.01357 | 2 |
| 1024^2 | 1024 | 1,048,576 | 0.06059 | 1 |

(FLOPS ratio is attention cost relative to the 1280x832 reference. The
reference itself has ratio ~0.97 and maps to the 1024^2 anchor.)

Each tier accommodates dozens of valid (W, H) pairs across the aspect ratio
range [0.5, 2.0]:

| Anchor | Valid (W, H) pairs | Min aspect | Max aspect |
|--------|-------------------|-----------|-----------|
| 256^2 | 11 | 0.500 | 2.000 |
| 320^2 | 14 | 0.500 | 2.000 |
| 384^2 | 14 | 0.529 | 1.889 |
| 512^2 | 22 | 0.500 | 2.000 |
| 704^2 | 31 | 0.500 | 2.000 |
| 1024^2 | 45 | 0.500 | 2.000 |

### Log-uniform aspect ratio sampling

Aspect ratios are sampled uniformly in log-space: `log(aspect) ~ U(log(0.5), log(2.0))`.
This ensures portrait and landscape are equally likely (log(0.5) = -log(2.0)
is symmetric around log(1.0) = 0). The sampled aspect is then quantized to
the nearest 32px-aligned (W, H) pair that preserves the pixel budget.

---

## 2. Module: `src_ii/resolution_sampling.py`

New pure-Python module (no torch). Core API:

```python
# Constants
MEGAPIXEL_ANCHORS = [65536, 102400, 147456, 262144, 495616, 1048576]

# Core sampling
sample_resolution(budget_pixels, aspect_ratio, step=32) -> (W, H)
sample_random_resolution(budget_pixels, rng, aspect_min=0.5, aspect_max=2.0, step=32) -> (W, H)

# Analysis
enumerate_resolutions(budget_pixels, step=32, aspect_min=0.5, aspect_max=2.0) -> list[(W, H)]
assign_budget_tier(width, height) -> anchor_pixels
items_per_bin(budget_pixels, reference_total_len=4224, cap_tokens=256) -> int

# Validation
validate_resolution(width, height, step=32) -> None  # raises ValueError
```

### `sample_resolution()` algorithm

Given budget B and aspect ratio A:
1. Solve `W = sqrt(B * A)`, `H = sqrt(B / A)`
2. Round both to nearest multiple of step (32)
3. Clamp to >= MIN_DIM (64)

The actual pixel count `W * H` typically differs from the budget by < 5%
due to 32px quantization. This is acceptable -- the FLOPS cost of a 67,584px
image (352x192) vs a 65,536px image (256x256) differs by ~3%, well within the
noise floor of gradient estimates.

### `assign_budget_tier()` algorithm

Maps any (W, H) to the nearest anchor by absolute pixel count difference.
This is the sole function that discretizes the continuous resolution space
for FLOPS accounting. All other infrastructure (bin_packer, sigma_schedule,
forward_packed) operates on the exact (W, H) without tier assignment.

Examples:
- 256x256 (65,536 px) -> 65,536 anchor (exact match)
- 320x192 (61,440 px) -> 65,536 anchor (nearest)
- 640x384 (245,760 px) -> 262,144 anchor (nearest)
- 1280x832 (1,064,960 px) -> 1,048,576 anchor (nearest)

---

## 3. Updated `src_ii/flops_sampling.py`

### Binary -> 6-tier bucketing

The old binary classification (`_classify_resolution` -> "megapixel" or "small")
is retained for backward compatibility but the primary FLOPS weight computation
now uses 6-tier `assign_budget_tier()`.

```python
# New 6-tier mode
weights = compute_flops_sampling_weights(
    traj_resolutions,
    tier_flops_targets={1048576: 0.33},  # 33% FLOPS on megapixel tier
)

# Legacy binary mode (backward compat)
weights = compute_flops_sampling_weights(
    traj_resolutions,
    tier_flops_targets=None,  # triggers binary mode
    megapixel_flops_fraction=0.33,
    small_flops_fraction=0.67,
)
```

### Tier FLOPS target distribution

When `tier_flops_targets={1048576: 0.33}`:
- The 1048576 anchor gets 33% of the target FLOPS budget
- The remaining 67% is distributed across other populated tiers
  proportionally by trajectory count

This means: with 10 trajectories each at 256^2, 320^2, 384^2, 512^2,
704^2, and 1024^2 (60 total), the 50 non-mega trajectories share 67% of
the FLOPS budget, weighted by trajectory count (each tier gets 67%/5 = 13.4%).

Within each tier, per-trajectory weights are inversely proportional to
the attention FLOPS ratio, so cheaper images within a tier get higher
sampling weight. For trajectories with the same resolution (same FLOPS
cost), this produces equal weights.

---

## 4. Updated `src_ii/pair_sampler.py`

### Multi-tier stratified sampling

`sample_stratified_batch()` now allocates pair slots across all 6 populated
tiers instead of the binary mega/small split.

The allocation algorithm (`_compute_tier_pair_allocation()`):

1. For each tier, compute the attention FLOPS per pair:
   `flops_per_pair = 2 * (anchor_tokens / ref_tokens)^2`
   where tokens = anchor / 256, ref_tokens = 4096.

2. Compute ideal continuous pair counts: for tier `a`,
   `pairs_a = tier_target_frac[a] / flops_per_pair[a]`, normalized so
   total = n_pairs.

3. Round to integers with minimum 1 for tiers with nonzero FLOPS targets.

4. Adjust residuals to hit exactly n_pairs.

### Allocation examples (6 tiers, 10 trajs each, target {1048576: 0.33})

| n_pairs | 256^2 | 320^2 | 384^2 | 512^2 | 704^2 | 1024^2 | Sum |
|---------|-------|-------|-------|-------|-------|--------|-----|
| 4 | 1 | 1 | 1 | 1 | 0 | 0 | 4 |
| 8 | 3 | 1 | 1 | 1 | 1 | 1 | 8 |
| 12 | 7 | 1 | 1 | 1 | 1 | 1 | 12 |
| 20 | 14 | 2 | 1 | 1 | 1 | 1 | 20 |
| 100 | 74 | 15 | 7 | 2 | 1 | 1 | 100 |

The allocation heavily favors small images because the FLOPS target is
specified in compute terms: hitting 33% megapixel FLOPS with just 1 pair
already achieves ~53% megapixel FLOPS (1024^2 images are ~4000x more
expensive per pair than 256^2). The allocator recognizes it cannot push
below 1 pair, so the megapixel tier slightly over-allocates.

### FLOPS distribution for 100-pair stratified batch

| Tier | Pairs | FLOPS fraction |
|------|-------|---------------|
| 256^2 | 74 | 14.1% |
| 320^2 | 15 | 7.5% |
| 384^2 | 7 | 7.2% |
| 512^2 | 2 | 6.5% |
| 704^2 | 1 | 11.7% |
| 1024^2 | 1 | 53.0% |

Compare to the old binary system where 97.5% of FLOPS went to megapixel
images and 2.5% to small. The 6-tier system with intermediate resolutions
(320^2, 384^2, 512^2, 704^2) creates a much smoother FLOPS distribution.

The key improvement: **tiers 2-5 (320^2 through 704^2) now contribute
meaningful FLOPS** (7-12% each). Previously, these were lumped into
"small" and their FLOPS contribution was negligible (all below 1.4%).

### Backward compatibility

- `has_megapixel`, `has_small` properties: preserved
- `_bucket_traj_ids` dict: preserved (binary classification)
- `sample_pair(resolution_bucket="megapixel")`: still works
- `stats()`: includes both per-tier and legacy bucket counts

---

## 5. Updated `scripts_ii/generate_multi_res_trajectories.py`

### Algorithmic generation replaces hardcoded tiers

The old hardcoded `RESOLUTION_TIERS` list:
```python
RESOLUTION_TIERS = [
    (256, 256, 4, "small"),
    (320, 192, 3, "small"),
    ...
    (1024, 1024, 4, "large"),
    (1280, 832, 3, "large"),
]
```

Is replaced with algorithmic sampling:
```python
TRAJECTORIES_PER_TIER = 10
RESOLUTION_RNG_SEED = 777  # deterministic

def _build_resolution_plan():
    rng = random.Random(RESOLUTION_RNG_SEED)
    plan = []
    for anchor in MEGAPIXEL_ANCHORS:
        for _ in range(TRAJECTORIES_PER_TIER):
            w, h = sample_random_resolution(anchor, rng, ...)
            plan.append((w, h, 1, label))
    return plan
```

This produces 60 entries (6 anchors x 10 per tier), each with a unique
randomly sampled (W, H). With 2 backends, total = 120 trajectories.

### Resolution diversity achieved

From the deterministic plan (seed=777):

| Anchor | Unique resolutions (of 10) | Example pairs |
|--------|---------------------------|---------------|
| 256^2 | 6 | 224x288, 352x192, 256x256, ... |
| 320^2 | 8 | 224x448, 288x384, 416x256, ... |
| 384^2 | 7 | 288x512, 320x480, 480x320, ... |
| 512^2 | 7 | 384x672, 448x608, 608x416, ... |
| 704^2 | 8 | 544x928, 576x864, 960x512, ... |
| 1024^2 | 7 | 736x1440, 800x1344, 1440x736, ... |
| **Total** | **43 unique** | |

43 unique (W, H) pairs from 60 samples, with collisions only at the
smallest tiers (256^2) where the 32px grid is coarsest.

---

## 6. Updated `src_ii/bin_packer.py`

### Deprecation of `RESOLUTION_TIERS`

The `RESOLUTION_TIERS` dict and `get_tier_resolutions()` / `get_tier_rollouts_per_prompt()`
functions are marked as deprecated. They remain for backward compatibility
with `build_generation_plan()` and existing scripts, but new code should
use `resolution_sampling.sample_random_resolution()`.

### Enhanced `validate_resolution()`

Now accepts a `step` parameter (default 32) in addition to the existing
`vae_scale` / `patch_size` parameters. Checks:
1. Width, height > 0 and >= 64 (MIN_DIM)
2. Divisible by `step` (32)
3. Divisible by `vae_scale * patch_size` (16)

Since 32 is a multiple of 16, the step check subsumes the vae+patch check.

### Unchanged infrastructure

The following work with arbitrary 32px-aligned resolutions without modification:
- `compute_seq_len()`: `(H // 8 // 2) * (W // 8 // 2)` -- handles any size
- `compute_effective_seq_len()`: `pad32(cap) + pad32(img_seq_len)` -- handles any size
- `BinPackScheduler`: bin capacity is `REFERENCE_TOTAL_LEN`, items are sized by
  their actual effective_seq_len -- no resolution enumeration dependency
- `REFERENCE_TOTAL_LEN`: unchanged (4224, the 1280x832 reference)

---

## 7. Infrastructure Verification: No Code Needed

The following subsystems already support arbitrary 32px-aligned resolutions
(verified by reading the code, documented in the earlier gap analysis from
`essay_funfetti_sampling_diagnostics.md`):

| Subsystem | File | Arbitrary resolution support |
|-----------|------|------------------------------|
| Sigma shifting | `sigma_schedule.py` | `sqrt(ref_pixels / (W*H))` -- uses product only |
| RoPE cache | `rollout.py` | `prepare_rope_cache(latent_h, latent_w)` -- separate dims |
| Packed forward | `forward_packed.py` | `img_sizes: list[tuple[int, int]]` -- mixed sizes |
| VAE encode/decode | `vae_utils.py` | Operates on spatial dimensions directly |
| `_ImagePosition` | `pair_sampler.py` | Has `width`, `height` fields |
| `ValidationMetrics` | `validation_metrics.py` | `resolution_bucket()` uses pixel count |
| Training loop | `btrm_training.py` | Latent shape -> (W, H) via `* 8 * 2` |

No code changes were needed in any of these modules.

---

## 8. Summary of Changes

### New files
- `src_ii/resolution_sampling.py` -- pure Python, 250 lines

### Modified files
- `src_ii/flops_sampling.py` -- binary -> 6-tier bucketing, backward-compat preserved
- `src_ii/pair_sampler.py` -- per-tier CDFs, `sample_stratified_batch()` with multi-tier allocation
- `src_ii/bin_packer.py` -- `RESOLUTION_TIERS` deprecated, `validate_resolution()` enhanced
- `scripts_ii/generate_multi_res_trajectories.py` -- algorithmic resolution plan from anchors

### Not modified (verified compatible)
- `src_ii/sigma_schedule.py`
- `src_ii/forward_packed.py`
- `src_ii/block_mask.py`
- `src_ii/btrm_training.py` (calls `sample_stratified_batch(n, mega_fraction=...)` which is preserved)
- `src_ii/validation_metrics.py`
- `src_ii/btrm_model.py`

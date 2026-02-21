# Funfetti Sampling Diagnostics: 100-Step Run Post-Mortem

## Summary

The 100-step funfetti BTRM training run on multi-resolution data
(`funfetti_100step_output/`) exercised the full packed training pipeline.
Training converged well (loss 0.70 -> 0.11, 90% accuracy), but four
sampling-distribution anomalies were observed in the diagnostic charts.
This essay diagnoses each issue, documents the one code fix applied,
and provides a gap analysis for non-square resolution support.

---

## Issue 1: Single-Point Aspect Ratio PDF

### Observation

The aspect ratio PDF chart (`07_aspect_ratio_pdf.png`) shows all images
at ratio 1.0 (square). No portrait or landscape images appear in the
training data.

### Root Cause

The multi-resolution generation script
(`scripts_ii/generate_multi_res_trajectories.py`, line 57) was configured
to produce only square resolutions:

```python
RESOLUTION_TIERS = [
    (256, 256, 10, "small"),
    (512, 512, 10, "medium"),
    (1024, 1024, 10, "large"),
]
```

This was intentional for the integration test (exercising three different
pixel areas without introducing aspect ratio as a confound), but it means
the funfetti batching path's aspect-ratio-aware metrics are untested.

### Status: Expected / Not a Bug

The `bin_packer.py` `RESOLUTION_TIERS` dict already defines non-square
resolutions across all three tiers (e.g., 1280x832, 640x384, 320x192).
The generation script simply did not use them. See "Phase 3: Non-Square
Gap Analysis" below for the full readiness assessment.

---

## Issue 2: Homogeneous Microbatch Composition

### Observation

The microbatch pair/context-length plots (`09_microbatch_pairs.png`,
`10_context_length.png`) show uniform characteristics across all
microbatches. Every microbatch looks the same: similar number of images,
similar context lengths, similar resolution mix.

### Root Cause

The FLOPS weighting heavily favors small images. From the run summary:

| Bucket      | Trajectories | Total Weight |
|-------------|-------------|-------------|
| megapixel (1024x1024) | 20 | 0.36% |
| small (256x256 + 512x512) | 40 | 99.6% |

With `pairs_per_pack=2` and `grad_accum=2`, each optimizer step samples
4 pairs (8 images). At 99.6% weight on small images, the probability of
sampling even ONE megapixel image in a microbatch is roughly:

    P(at least 1 megapixel in 4 images) = 1 - (0.996)^4 ~ 1.6%

So ~98.4% of microbatches contain ONLY small images. And within the small
bucket, 256x256 images are 16x cheaper than 512x512, so they get roughly
16x higher per-trajectory weight. In practice, most microbatches are
dominated by 256x256 images.

The resolution PDF confirms this: 256x256 = 93.1%, 512x512 = 6.1%,
1024x1024 = 0.75% of sampled images.

This is correct behavior for FLOPS-equalized sampling -- small images
should dominate the count to fill the FLOPS budget. The microbatches
appear homogeneous because the sampling distribution IS homogeneous
by design for this dataset configuration.

### What Would Produce Heterogeneous Microbatches

1. **Non-square resolutions**: Different aspect ratios at the same pixel
   area produce different sequence lengths (e.g., 1280x832 = 4160 tokens
   vs 1024x1024 = 4096). This creates natural variation in bin packing.

2. **More balanced FLOPS allocation**: The 33/67 megapixel/small split
   with three discrete resolution tiers creates extreme concentration.
   A continuous resolution distribution (e.g., 256-1280 in 16-pixel
   steps) would produce smoother variation.

3. **Larger pairs_per_pack**: With `pairs_per_pack=4` (8 images), the
   probability of mixing resolutions increases. The bin packer can then
   combine different-sized images into genuinely heterogeneous bins.

4. **Non-uniform step lists**: Different trajectories having different
   sparse step subsets would create sigma variation within the same
   resolution bucket.

### Status: Working as Designed

No code fix needed. The homogeneity reflects the extreme FLOPS weighting
on a dataset with only three discrete resolution tiers.

---

## Issue 3: Sigma/LogSNR Distribution Anomaly

### Observation

The validation metrics' logSNR distribution was expected to show the
geometric decay pattern (more clean/clean-ish, fewer very-noisy) from
the logSNR-weighted step selection. Instead, the raw counts showed:

| Bucket | Count |
|--------|-------|
| moderate (0 <= logSNR < 2) | 640 |
| clean-ish (2 <= logSNR < 5) | 380 |
| near-clean (5 <= logSNR < inf) | 42 |

### Root Cause: TWO Contributing Factors

**Factor A: Incorrect logSNR bucketing formula (BUG -- FIXED)**

The `logsnr_bucket()` function in `src_ii/validation_metrics.py` used
`logSNR = -2 * ln(sigma)`, which is an incorrect approximation of the
CONST noise model's logSNR. The correct formula (used consistently in
`pair_sampler.py`) is:

    logSNR = 2 * ln((1 - sigma) / sigma)

These formulas diverge significantly for sigma > 0.3:

| sigma | -2*ln(sigma) (old) | 2*ln((1-sigma)/sigma) (correct) | Old bucket | Correct bucket |
|-------|-------------------|---------------------------------|------------|----------------|
| 0.3351 | 2.187 | 1.370 | clean-ish | moderate |
| 0.5019 | 1.379 | -0.016 | moderate | noisy |
| 0.8220 | 0.392 | -3.061 | moderate | very_noisy |
| 0.9633 | 0.075 | -6.537 | moderate | very_noisy |

The old formula misclassified noisy samples into the "moderate" bucket,
inflating its count and masking the true distribution shape.

**Fix applied**: `src_ii/validation_metrics.py` line 112, changed
`logsnr = -2.0 * math.log(sigma)` to
`logsnr = 2.0 * math.log((1.0 - sigma) / sigma)`.

**Factor B: Resolution-shifted sigma schedules produce few clean steps**

For 256x256 images (shift=4.03), the sigma schedule is heavily shifted
toward high noise. The sparse step sigmas are:

| Step | Sigma | CONST logSNR | Bucket (corrected) |
|------|-------|-------------|-------------------|
| step_00 | 1.000 | -inf | very_noisy |
| step_04 | 0.963 | -6.5 | very_noisy |
| step_09 | 0.904 | -4.5 | very_noisy |
| step_14 | 0.822 | -3.1 | very_noisy |
| step_19 | 0.700 | -1.7 | noisy |
| step_24 | 0.502 | -0.02 | noisy |
| step_29/final | 0.124 | +3.9 | clean-ish |

Only the final step lands in "clean-ish". No step reaches "near-clean"
for 256x256 images. For 512x512 (shift=2.02), only step_29/final reaches
"near-clean" (logSNR=5.3). Since 256x256 images dominate the sampling
(93% of images), the overall logSNR distribution is concentrated in
"very_noisy" and "noisy" with the corrected formula.

The logSNR step weighting IS active (the `build_positions_from_v2()`
function at line 474 of `pair_sampler.py` computes logits via
`logsnr_sampling_logit()` with default parameters). However, the decay
is gentle: from weight 1.0 at logSNR >= 5 down to ~0.54 at logSNR=-6.5.
After softmax over 8 positions, near-clean positions get ~18% selection
probability vs ~6-11% for very-noisy ones. This bias IS present but
modest relative to the overwhelming dominance of shifted schedules.

### Status: Bug Fixed + Understanding Clarified

The logSNR bucketing formula was wrong in `validation_metrics.py`. With
the corrected formula, the validation metrics will correctly reflect
that resolution-shifted schedules produce predominantly noisy steps.
The logSNR step weighting is active but cannot overcome the fundamental
property that small-image schedules have few clean steps.

---

## Issue 4: Resolution PDF Interpretation

### Observation

The resolution PDF chart shows raw image counts by pixel area:

| Resolution (pixels) | Count | Proportion |
|---------------------|-------|-----------|
| 65536 (256x256) | 745 | 93.1% |
| 262144 (512x512) | 49 | 6.1% |
| 1048576 (1024x1024) | 6 | 0.75% |

The concern was whether this represents the correct FLOPS-normalized
distribution or some averaged/reduced form.

### Root Cause: The Chart Is Correct

The `_chart_resolution_pdf()` method in `training_artifacts.py` counts
per-IMAGE occurrences from the funfetti step metadata. Each step's
`funfetti.resolutions` list contains one entry per image processed
(2 * pairs_per_pack per microbatch * grad_accum microbatches per step).
The chart shows the empirical sampling distribution: how many times
each resolution was drawn.

The 93/6/1 ratio is the INTENDED outcome of FLOPS-weighted sampling.
Small images are cheap (~0.05% of reference FLOPS for 256x256), so they
must be sampled ~100x more often to consume the same FLOPS budget. The
chart title says "Resolution PDF (image count)" which accurately describes
what is plotted.

If the user wanted a FLOPS-normalized view (how much compute was spent
per resolution), that would be a different chart showing
`count * flops_ratio_per_image`. This is a missing chart, not a bug
in the existing chart.

### Recommendation

Add a companion "Resolution FLOPS PDF" chart to `training_artifacts.py`
that multiplies each resolution's image count by its
`_attention_flops_ratio()` and normalizes. This would show the compute
budget allocation and verify the 33/67 megapixel/small target split.

### Status: Chart Is Correct / Enhancement Identified

---

## Phase 2: Fixes Applied

### Fix 1: `src_ii/validation_metrics.py` -- logSNR bucketing formula

**File**: `/mnt/f/dox/repos/ai/futudiffu/src_ii/validation_metrics.py`
**Line**: ~112 (in `logsnr_bucket()`)

**Before**:
```python
logsnr = -2.0 * math.log(sigma)
```

**After**:
```python
logsnr = 2.0 * math.log((1.0 - sigma) / sigma)
```

This aligns the bucketing with the CONST noise model's logSNR formula,
matching `pair_sampler.logsnr_sampling_logit()`. The previous formula
was a first-order approximation that diverges for sigma > 0.3.

Impact: With the corrected formula, the next funfetti run's validation
metrics will correctly show the true logSNR distribution, which will be
dominated by "very_noisy" and "noisy" buckets for resolution-shifted
schedules. The "near-clean" bucket will only appear for the last 1-2
steps of 512x512 and 1024x1024 trajectories.

---

## Phase 3: Non-Square Resolution Gap Analysis

The current multi-res dataset uses only square resolutions. This section
analyzes what would need to change to generate 1280x832, 832x1280,
1024x768, 640x384, etc.

### Generation Script (`scripts_ii/generate_multi_res_trajectories.py`)

**Supports non-square**: YES, trivially. The script passes `(width, height)`
to the `rollout()` function (line 203-215), which handles arbitrary
resolutions. The only change needed is in the `RESOLUTION_TIERS`
configuration:

```python
# Current (square only):
RESOLUTION_TIERS = [
    (256, 256, 10, "small"),
    (512, 512, 10, "medium"),
    (1024, 1024, 10, "large"),
]

# Proposed (with aspect ratios):
RESOLUTION_TIERS = [
    (256, 256, 4, "small"),
    (320, 192, 3, "small"),   # landscape
    (192, 320, 3, "small"),   # portrait
    (512, 512, 4, "medium"),
    (640, 384, 3, "medium"),  # landscape
    (384, 640, 3, "medium"),  # portrait
    (1024, 1024, 4, "large"),
    (1280, 832, 3, "large"),  # landscape
    (832, 1280, 3, "large"),  # portrait
]
```

No code changes needed -- only configuration.

### Bin Packer (`src_ii/bin_packer.py`)

**Supports non-square**: YES, fully. The `compute_seq_len()` and
`compute_effective_seq_len()` functions handle arbitrary `(width, height)`
pairs. The `RESOLUTION_TIERS` dict already includes non-square
resolutions for all three tiers (e.g., 1280x832, 640x384, 320x192).
The `validate_resolution()` function enforces the 16-pixel alignment
constraint for both dimensions.

### Sigma Shifting (`src_ii/sigma_schedule.py`)

**Supports non-square**: YES. The `resolution_shift()` function at
line 16 computes `sqrt(ref_pixels / (width * height))` using the
product of width and height (total pixel count). Non-square resolutions
get the correct shift based on their total area, not their aspect ratio.

### RoPE Cache (`src_ii/rollout.py`)

**Supports non-square**: YES. The `make_rope_cache()` function at line 22
takes `(latent_h, latent_w)` separately and handles padding:

```python
padded_h = latent_h + ((-latent_h) % diff_model.patch_size)
padded_w = latent_w + ((-latent_w) % diff_model.patch_size)
return diff_model.prepare_rope_cache(padded_h, padded_w, num_tokens, device)
```

For 1280x832: `latent_h=104, latent_w=160, padded_h=104, padded_w=160`.
For 832x1280: `latent_h=160, latent_w=104, padded_h=160, padded_w=104`.
Both are valid and distinct RoPE caches.

### Packed Forward (`src_ii/forward_packed.py`)

**Supports non-square**: YES. The `prepare_packed_forward()` function
takes `img_sizes: list[tuple[int, int]]` where each tuple is
`(latent_h, latent_w)`. Mixed aspect ratios within a single packed
batch are handled correctly by the model's `prepare_packed_state()`.

### VAE Pixel Alignment

**Requirement**: Width and height must both be divisible by 16
(VAE 8x downscale * DiT patch_size 2).

All resolutions in `bin_packer.RESOLUTION_TIERS` satisfy this
constraint. The `validate_resolution()` function enforces it.

### `BTRMPairSampler` and `_ImagePosition`

**Supports non-square**: YES. The `_ImagePosition` class has
`width` and `height` fields (not just pixel count). The
`build_positions_from_v2()` function extracts `traj_w` and `traj_h`
from trajectory metadata and passes them to `_ImagePosition`.

### FLOPS Sampling (`src_ii/flops_sampling.py`)

**Supports non-square**: YES, but the bucket classification uses only
pixel count (not aspect ratio). Two images at 1280x832 and 832x1280
have the same pixel count and will be classified in the same FLOPS
bucket with the same attention cost ratio. This is correct -- attention
FLOPS are proportional to token count squared, and token count depends
only on total pixel count (not aspect ratio).

### `ValidationMetrics` Aspect Ratio Tracking

**Supports non-square**: YES. The `aspect_bucket()` function classifies
W/H ratio into portrait (<0.8), square (0.8-1.2), and landscape (>=1.2).
The `PairResult` dataclass tracks `width_a, height_a, width_b, height_b`
for both images. These are correctly passed from the training loop
(line 776-777 of `btrm_training.py`) via latent shape inference.

### Summary: No API Gaps

All modules support non-square resolutions. The only change needed
is in the generation script's `RESOLUTION_TIERS` configuration.
No code modifications required. The existing bin_packer already
has the non-square resolution tiers defined but unused.

---

## Recommendations for Next Generation Run

1. **Include non-square resolutions** in the generation script. Use the
   `bin_packer.RESOLUTION_TIERS` definitions which already include
   portrait and landscape variants at each tier.

2. **Increase trajectory count** per resolution to at least 5 per
   (resolution, backend) pair, giving 10+ trajectories per resolution.
   With 13 non-square resolutions across 3 tiers, that is ~130 total
   trajectories (65 per backend).

3. **Consider `pairs_per_pack=4`** to increase the probability of
   mixed-resolution microbatches and exercise heterogeneous bin packing.

4. **Add a FLOPS-normalized resolution chart** to `training_artifacts.py`
   to verify the 33/67 compute budget allocation alongside the existing
   image count chart.

5. **Use at least 200 training steps** to give the corrected logSNR
   bucketing enough samples in each bucket for meaningful statistics.

6. **Verify the corrected logSNR bucketing** by checking the next run's
   `validation_metrics.json` -- with 256x256-dominated sampling, expect
   the "very_noisy" bucket to have the highest count, not "moderate".

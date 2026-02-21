# FLOPS-Budget Macrobatch Construction for BTRM Training

**Date:** 2026-02-18
**Trigger:** The fixed pair-count macrobatch architecture (`pairs_per_pack * grad_accum_steps`)
is structurally unable to balance compute across resolution tiers. A macrobatch
with 4 pairs always does 4 pairs of forward passes regardless of whether those
pairs contain 1024x1024 images (1.0 FLOPS unit each) or 256x256 images (0.004
FLOPS units each). This means the actual GPU compute per optimizer step varies
by 250x depending on the resolution mix. The new architecture defines a
macrobatch by its FLOPS budget, not its pair count.

---

## 1. The Problem: Fixed Pair Count vs Variable Compute

### Before: pair count defines the macrobatch

```
macrobatch = grad_accum_steps * pairs_per_pack pairs
           = 2 * 2 = 4 pairs (fixed)
           = 8 images (fixed)
           = variable FLOPS (0.03 to 8.3 units depending on resolution mix)
```

With stratified sampling ensuring 1 megapixel pair + 3 small pairs per
macrobatch, the FLOPS profile is dominated by the megapixel pair:
- 1 pair at 1024x1024: ~2.0 FLOPS units
- 3 pairs at 256x256: ~0.024 FLOPS units
- Total: ~2.024 FLOPS units
- Megapixel fraction: 98.8% of FLOPS

This is stable but inflexible. The optimizer sees exactly 4 pairs of gradient
signal per step. If those 4 pairs happen to be all-small (as happened with
probabilistic sampling), the model receives almost no learning signal about
megapixel features. If they are all-megapixel, the step takes 4x longer than
a mixed step for the same 4 pairs.

### After: FLOPS budget defines the macrobatch

```
macrobatch = budget_units worth of FLOPS (default 3.0)
           = variable pairs (1 to hundreds, depending on resolution mix)
           = variable images (2 to thousands)
           = constant FLOPS (~3.0 units, by construction)
```

With budget=3.0:
- If dominated by 1024x1024 pairs: ~1-2 pairs (2.0-4.0 units per pair)
- If dominated by 256x256 pairs: ~380 pairs (0.008 units per pair)
- If mixed: ~1 mega pair + ~100-150 small pairs

The pair count adapts to the resolution mix. The GPU compute per optimizer
step is approximately constant, giving:
1. Predictable step time
2. Consistent gradient noise level
3. No wasted compute on tiny-image macrobatches

---

## 2. FLOPS Unit Definition

One FLOPS unit equals the attention compute of a single 1024x1024 image:

```
tokens = (W * H) / 256      # latent patches = pixels / (8 * 2)^2
ref_tokens = 1048576 / 256   # = 4096 (1024x1024 reference)
flops_unit = (tokens / ref_tokens) ** 2
```

| Resolution | Tokens | FLOPS Units | Cost of 1 Pair |
|------------|--------|-------------|----------------|
| 1024x1024 | 4096 | 1.000 | 2.000 |
| 1280x832 | 4160 | 1.031 | 2.063 |
| 704x704 | 1936 | 0.223 | 0.447 |
| 512x512 | 1024 | 0.063 | 0.125 |
| 384x384 | 576 | 0.020 | 0.039 |
| 320x320 | 400 | 0.010 | 0.019 |
| 256x256 | 256 | 0.004 | 0.008 |

A pair costs the sum of both images' FLOPS units. Cross-resolution pairs
(e.g., 1024x1024 + 256x256) cost ~1.004 units.

---

## 3. Macrobatch Construction Algorithm

Implemented in `BTRMPairSampler.sample_macrobatch()`.

### Phase 1: Top-tier guarantee

Sample one pair from the top populated resolution tier (typically 1024^2).
This ensures every macrobatch has at least one high-resolution pair for rich
gradient signal. Without this guarantee, a FLOPS-weighted sampler would fill
the budget with hundreds of cheap small pairs and rarely include a megapixel
image (the same pathology that motivated stratified sampling).

### Phase 2: Fill remaining budget

Repeatedly sample pairs until the FLOPS budget is consumed:

```
while consumed < budget:
    # Weight each tier by: target_flops - consumed_flops (clamped to 0)
    # Tiers that are behind their target get preferentially sampled.
    # A small floor (0.001) prevents any tier from total exclusion.
    for each populated tier:
        weight[tier] = max(0, target_frac[tier] * budget - consumed[tier]) + 0.001

    selected_tier = weighted_random_choice(populated_tiers, weights)
    pair = sample_pair(tier=selected_tier, allow_cross_resolution=True)
    consumed += pair.flops_cost
```

The deficit-based weighting is self-correcting: if the megapixel tier is
over-target (which happens immediately after the Phase 1 guarantee), subsequent
sampling favors smaller tiers until their allocation catches up. The result
is a resolution mix that approximates the target FLOPS distribution within
each macrobatch.

### Phase 3: Shuffle

The sampled pairs are shuffled before returning. This ensures the bin packer
sees a random ordering rather than a tier-sorted one, which could cause all
megapixel images to end up in the same bin.

### Safety bounds

- Maximum 1000 pairs per macrobatch to prevent runaway on degenerate inputs.
- The budget is treated as a target, not a hard limit. The last pair sampled
  may push consumed slightly over budget. This is intentional: undershoot is
  worse than overshoot (missing gradient signal vs slightly extra compute).

---

## 4. Cross-Resolution Pairs

### Motivation

A cross-resolution pair compares two images from the **same prompt** but
**different resolutions**. For example:
- Preferred: 1024x1024 SDPA image (full-res, high quality)
- Rejected: 256x256 SageAttention image (low-res, INT8 QK artifacts)

This is a valid comparison for the "pinkify" head (quantization artifact
discrimination): the reward model should learn that quantization artifacts
are resolution-dependent but comparable across scales. Similarly, the
"thisnotthat" head (step count discrimination) can compare a 30-step 1024x1024
image against a 10-step 512x512 image.

### Implementation

Cross-resolution pairs require a prompt-based trajectory grouping. The
`_ImagePosition` class gained a `prompt_idx` field, populated from the
trajectory's `meta.json`. The `BTRMPairSampler` constructor builds:

1. `_prompt_traj_ids: dict[int, list[int]]` -- maps prompt_idx to trajectory IDs
2. `_cross_res_prompts: list[int]` -- prompts with trajectories in multiple tiers

When sampling a pair with `allow_cross_resolution=True`:
1. Sample image A from the specified tier (normal behavior)
2. With 30% probability, attempt a cross-resolution pair:
   a. Find trajectories for the same prompt in OTHER tiers
   b. If any exist, sample image B from one of those trajectories
   c. The resulting pair has images at different resolutions
3. If cross-resolution is not attempted (70%) or not possible (no matching
   trajectories in other tiers), fall back to same-tier pairing

The 30% probability is a hyperparameter. Higher values increase cross-resolution
exposure but reduce same-tier diversity. The value 30% was chosen to provide
meaningful cross-resolution signal without dominating the pair distribution.

### Bin packer compatibility

The bin packer already handles heterogeneous image sizes. Each image is packed
individually by its effective sequence length. A cross-resolution pair with a
1024x1024 and a 256x256 image will likely end up in different bins -- the 1024x1024
image fills a bin on its own, while the 256x256 image packs alongside many other
small images. This is correct behavior: the images are packed for compute efficiency,
not by pair membership. The score reassembly step in the training loop uses
`pair_image_indices` to recover which scores belong to which pair.

---

## 5. PairSpec Return Type

The new `PairSpec` dataclass provides structured metadata for each sampled pair:

```python
@dataclass
class PairSpec:
    image_a: _ImagePosition
    image_b: _ImagePosition
    flops_cost: float       # sum of both images' FLOPS units
    cross_resolution: bool  # True if different resolution tiers

    def to_pair_dict(self) -> dict:
        # Returns the dict format expected by the training loop
```

`to_pair_dict()` includes: `traj_a`, `step_a`, `sigma_a`, `traj_b`, `step_b`,
`sigma_b`, `width_a`, `height_a`, `width_b`, `height_b`, `cross_resolution`,
`flops_cost`. This metadata flows through the training loop and into the
per-step JSONL diagnostics.

---

## 6. Training Loop Integration

### New parameters on `train_btrm_differentiable()`

```python
macrobatch_budget: float | None = None   # FLOPS budget in 1024^2 units
macrobatch_cross_resolution: bool = True # allow cross-res pairs
```

When `macrobatch_budget` is set and `packed=True` and the sampler has
`sample_macrobatch()`, the training loop enters the FLOPS-budget path.

### FLOPS-budget path (new)

```
For each optimizer step:
  1. sample_macrobatch(budget=macrobatch_budget) -> list[PairSpec]
  2. preference_fn(pair_dict) for each pair
  3. load_latent_fn for all 2*N_pairs images
  4. BinPackScheduler.pack() all images into J bins
  5. For each bin: score_differentiable_packed() -> accumulate gradients
  6. BT loss across all pairs, normalized by active_heads
  7. loss.backward()
  8. gradient clip + optimizer step
```

Key difference from legacy: there is NO inner microbatch loop. The entire
macrobatch (variable N pairs, 2N images, J bins) is processed as a single
gradient accumulation unit. The backward pass runs once, not grad_accum_steps
times.

### Legacy path (preserved)

When `macrobatch_budget` is None, the existing behavior is unchanged:
`grad_accum_steps` microbatches of `pairs_per_pack` pairs each. The
stratified sampling path (`sample_stratified_batch`) is also preserved.

### Loss normalization

Both paths normalize loss by `active_heads`, not by pair count or bin count.
This is consistent with the existing architecture: the gradient magnitude
scales with the information content (how many heads had non-zero preferences),
not the batch size.

### Per-step metadata

The FLOPS-budget path records additional metadata in the funfetti JSONL:

```json
{
    "macrobatch_budget": 3.0,
    "macrobatch_consumed": 3.127,
    "n_pairs": 47,
    "n_bins": 5,
    "n_cross_resolution_pairs": 8,
    "per_pair_resolutions": [
        {"width_a": 1024, "height_a": 1024, "width_b": 1024, "height_b": 1024,
         "cross_resolution": false, "flops_cost": 2.0},
        {"width_a": 256, "height_a": 256, "width_b": 512, "height_b": 512,
         "cross_resolution": true, "flops_cost": 0.066}
    ]
}
```

---

## 7. Worked Example

### Dataset: 6 tiers, 10 trajectories each, budget=3.0

Phase 1: Top-tier guarantee
- Sample 1 pair from 1024^2 tier. Cost: ~2.0 units.
- Consumed: 2.0. Remaining budget: 1.0.

Phase 2: Fill remaining
- Tier targets: {1048576: 0.33, rest: 0.67/5 = 0.134 each}
- After Phase 1, the 1024^2 tier is over-target (consumed 2.0, target 1.0).
  Its deficit is 0. All weight goes to the smaller tiers.
- Sample ~5 pairs from 704^2 tier (0.447 units each): ~2.2 units total.
  Wait, that exceeds the remaining 1.0. With deficit-based weighting:
  - 704^2 has target 0.134 * 3.0 = 0.40, consumed 0 -> deficit 0.40
  - 512^2 has target 0.134 * 3.0 = 0.40, consumed 0 -> deficit 0.40
  - 384^2, 320^2, 256^2 similarly
  - All 5 non-mega tiers share equal weight
  - A 704^2 pair costs 0.447 units, a 256^2 pair costs 0.008 units
  - Expected: 2 pairs from 704^2 (0.89) + ~12 pairs from smaller tiers (~0.11)
- Actual consumed: ~3.0 with ~15 pairs total

### Dataset: monoresolution (all 1280x832)

Phase 1: Sample 1 pair. Cost: ~2.06 units.
Phase 2: Sample 1 more pair. Cost: ~2.06 units. Total: ~4.13. Over budget.

Result: 2 pairs per macrobatch. This is the expected degenerate case for
monoresolution datasets -- the macrobatch is tiny because each pair is expensive.

### Dataset: all 256x256

Phase 1: Top tier is 256^2 (only tier). Sample 1 pair. Cost: 0.008.
Phase 2: Fill remaining 2.992 units with 256^2 pairs.
- 2.992 / 0.008 = ~374 pairs
- Total: ~375 pairs per macrobatch

Result: ~375 pairs. The model gets massive gradient averaging from small images.

---

## 8. Inline Verification

### FLOPS unit correctness

```
flops_units(1024, 1024) = 1.0              -- reference
flops_units(256, 256)   = 0.00390625       -- 1/256
flops_units(512, 512)   = 0.0625           -- 1/16
flops_units(704, 704)   = 0.2234           -- (1936/4096)^2
flops_units(1280, 832)  = 1.0315           -- (4160/4096)^2
pair_flops_units(1024, 1024, 1024, 1024) = 2.0
pair_flops_units(256, 256, 256, 256)     = 0.0078125
```

All values verified against manual computation.

### Macrobatch sampling verification

Test with 8 trajectories across 3 tiers (1024^2, 512^2, 256^2):
- Budget: 2.5 units
- Result: 11 pairs, 2.547 FLOPS units consumed
- 6 cross-resolution pairs correctly identified
- Top-tier guarantee: 1024^2 pair present
- Budget respected: consumed slightly over target (greedy overshoot)

### Backward compatibility

When `macrobatch_budget=None`:
- The training loop takes the legacy path
- `sample_stratified_batch()` still works
- `pairs_per_pack * grad_accum_steps` still determines pair count
- No behavioral change for existing training scripts

---

## 9. Changes Summary

### Modified files

| File | Change |
|------|--------|
| `src_ii/flops_sampling.py` | Added `flops_units()` and `pair_flops_units()` |
| `src_ii/pair_sampler.py` | Added `PairSpec` dataclass, `prompt_idx` on `_ImagePosition`, prompt-based grouping, `sample_macrobatch()`, `_sample_pair_spec()` |
| `src_ii/btrm_training.py` | Added `macrobatch_budget` and `macrobatch_cross_resolution` params, FLOPS-budget training path |

### Not modified (verified compatible)

| File | Reason |
|------|--------|
| `src_ii/bin_packer.py` | Already handles heterogeneous items. Cross-resolution pairs pack correctly. |
| `src_ii/btrm_model.py` | `score_differentiable_packed()` already handles any image list. No changes needed. |
| `src_ii/forward_packed.py` | Already handles mixed-resolution packed batches. |
| `src_ii/block_mask.py` | Block masks are constructed from packing segments, resolution-agnostic. |
| `src_ii/validation_metrics.py` | PairResult already has `width_a/b`, `height_a/b`. Cross-res pairs are tracked. |

### New API surface

```python
# New in flops_sampling.py
flops_units(width: int, height: int) -> float
pair_flops_units(width_a, height_a, width_b, height_b) -> float

# New in pair_sampler.py
PairSpec(image_a, image_b, flops_cost, cross_resolution)
PairSpec.to_pair_dict() -> dict
BTRMPairSampler.sample_macrobatch(budget_units=3.0, ...) -> list[PairSpec]

# New params in btrm_training.py
train_btrm_differentiable(..., macrobatch_budget=3.0, macrobatch_cross_resolution=True)
```

---

## 10. Architectural Notes

### Why one backward per macrobatch?

The FLOPS-budget path runs `loss.backward()` once for the entire macrobatch,
not per-bin. This is correct because:

1. All J bins contribute to the same loss tensor via `img_idx_to_score`.
   The computation graph spans all bins.
2. Running backward once produces the correct gradient for the complete
   macrobatch loss. There is no need for loss scaling by 1/J because the
   loss is already normalized by active_heads.
3. This is simpler than the legacy path's per-microbatch backward with
   loss / grad_accum_steps scaling.

### Why the budget slightly overshoots

The last pair in Phase 2 always pushes consumed above the budget. This is
intentional: stopping before the budget would waste GPU capacity, and the
greedy overshoot is bounded by the cost of one pair. For the typical case
(last pair is a small image), the overshoot is < 0.1 units on a 3.0 budget.
For the worst case (the last pair is a megapixel pair), the overshoot is ~2.0
units -- but this only happens when the budget is nearly consumed and the
deficit-weighted sampling chose a megapixel pair, which means the megapixel
tier was genuinely undersampled.

### Gradient noise implications

Fixed pair count produces fixed-dimension gradient estimates. FLOPS-budget
produces variable-dimension gradient estimates (more pairs = lower variance,
but not linearly so because the pairs within a macrobatch are not independent
-- they share the same optimizer step).

The learning rate and momentum should be recalibrated when switching from
fixed to FLOPS-budget macrobatches. The effective "batch size" in gradient
space is approximately `active_heads / loss_denominator`, which is constant
(=1) under active_heads normalization regardless of pair count. This means
the learning rate should NOT need adjustment -- the loss magnitude is
invariant to macrobatch size.

### torch.compile considerations

The FLOPS-budget path produces variable-length bin lists per step. Each
unique total_len triggers a recompilation of the FlexAttention kernel (this
is inherent to the packed forward path). With many different resolutions, the
recompilation cache may grow. This is not a new problem -- it exists in the
legacy packed path too -- but the FLOPS-budget path may encounter more
diverse total_lens per step because it packs more images.

Mitigation: the bin packer uses REFERENCE_TOTAL_LEN as the bin capacity,
so bins are at most 4224 tokens. The actual total_len depends on how many
images fit in each bin. With 1-2 images per bin (megapixel), total_len is
~4160-4224. With many small images, total_len converges to ~4224. The
diversity of total_lens is bounded by the packing algorithm, not the
resolution diversity.

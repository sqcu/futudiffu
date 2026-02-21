# Synthesis: Stratified FLOPS Sampling and the Quadratic Cost Paradox

**Date:** 2026-02-19
**Author:** Root session
**Provenance:** Synthesis of essays from 3 subagent rounds:
- `essay_funfetti_sampling_diagnostics.md` — diagnosis of 4 anomalies in first 100-step run
- `essay_stratified_flops_sampling.md` — stratified sampling implementation + results
- `essay_funfetti_100step_run.md` — first 100-step baseline

---

## The Arc

The funfetti batching integration reached its climactic test: a 100-step
training run on multi-resolution data with FLOPS-weighted sampling. The
run succeeded technically (loss 0.70 -> 0.11, 92.6% TNT accuracy) but
revealed that probabilistic FLOPS sampling is architecturally inadequate
for small macrobatch sizes.

**Diagnosis round:** Four anomalies in the 100-step run's diagnostics:

1. Single-point aspect ratio PDF (all square images) — expected, dataset
   only had square resolutions
2. Homogeneous microbatches (98.4% all-small) — the FLOPS weight ratio
   puts 99.6% probability on small images, so 4-image macrobatches almost
   never sample megapixel
3. logSNR bucketing formula bug — used `-2*ln(sigma)` instead of CONST
   model's `2*ln((1-sigma)/sigma)`, misclassifying noisy samples as moderate
4. Resolution PDF showing 93/6/1 split — correct for FLOPS-weighted count,
   but missing a FLOPS-normalized companion chart

**Critical user correction:** The spec's "33% of FLOPS on megapixel" is a
structural per-macrobatch constraint, not a probabilistic average. The fix
is not dynamic gradient accumulation (rejected as changing gradient noise
characteristics) but stratified sampling: partition pair slots into
megapixel and small buckets, sample each from filtered CDFs.

**Implementation round:** `sample_stratified_batch(n_pairs, mega_fraction)`
added to `BTRMPairSampler`. With 4 pairs/macrobatch and mega_fraction=0.33:
1 mega pair + 3 small pairs per macrobatch. Non-square resolutions added
(9 total: 3 square + 6 portrait/landscape). 100-step run executed.

---

## The Quadratic Cost Paradox

The stratified run eliminated the all-small macrobatch problem (0% vs 98.4%).
But it surfaced a deeper question about what "33% of FLOPS" means.

The numbers:

| Resolution | FLOPS Ratio | Count (per macrobatch) | FLOPS Contribution |
|-----------|-------------|----------------------|-------------------|
| 1024x1024 | 0.969 | 2 images (1 pair) | 1.94 |
| 256x256 | 0.004 | ~6 images (3 pairs) | 0.02 |

**Megapixel FLOPS fraction: 97.5%**

This is not a bug. A 1024x1024 image has 16x more pixels than 256x256,
but attention FLOPS scale quadratically with token count, making the ratio
~250x. One megapixel pair consumes as much compute as ~250 small pairs.
With 1 mega + 3 small, the mega pair dominates compute by 99:1.

**There are two valid interpretations of the spec:**

A. **33% of pair slots on megapixel** — what `mega_fraction=0.33` implements.
   This produces ~97.5% megapixel FLOPS. The model sees megapixel features
   every single step. Small images provide cheap diversity.

B. **33% of compute on megapixel** — would require ~1 mega pair per ~500
   small pairs at 256x256, which is incompatible with the 4-pair macrobatch
   constraint. Impossible to achieve at current macrobatch sizes.

The stratified run implements interpretation A. The essay argues this is
correct for training: guaranteed megapixel gradient signal every step
prevents the model from going long stretches without high-resolution
features. The small-image pairs provide high-frequency, low-cost gradient
averaging that reduces noise.

**This is a question for the user.** The spec said "33% of training FLOPS
on megapixel." The quadratic cost ratio makes interpretation B impossible
at current macrobatch sizes. Interpretation A (33% of pair slots) is
implementable and produces good training results, but the actual FLOPS
split is 97.5/2.5, not 33/67.

---

## Training Results: Stratified vs Probabilistic

| Metric | Probabilistic (first run) | Stratified (this run) |
|--------|--------------------------|----------------------|
| Megapixel images seen | ~6 | 200 |
| Megapixel FLOPS % | ~0.8% | 97.5% |
| All-small macrobatches | 98.4% | 0% |
| Final BT loss | 0.2706 | 0.2391 |
| Pinkify accuracy (last 20) | 90% | 97.5% |
| TNT accuracy (last 20) | 95% | 97.5% |
| Mean step time | 3.0s | 5.5s |

The 33x increase in megapixel exposure (6 -> 200 images) produced:
- 13% reduction in final loss (0.27 -> 0.24)
- 7.5pp improvement in pinkify accuracy (90% -> 97.5%)
- 2.5pp improvement in TNT accuracy (95% -> 97.5%)
- 83% increase in step time (3.0s -> 5.5s) — expected, megapixel pairs
  cost ~250x more FLOPS

The pinkify head benefited most, which is expected: pinkify discriminates
attention quantization artifacts (SDPA vs SageAttention INT8 QK), and
these artifacts are most visible at high resolution where the attention
map has more spatial structure.

---

## Non-Square Resolution Coverage

The stratified run used 9 resolutions across 3 tiers:

| Tier | Square | Landscape | Portrait |
|------|--------|-----------|----------|
| Small | 256x256 | 320x192 | 192x320 |
| Medium | 512x512 | 640x384 | 384x640 |
| Large | 1024x1024 | 1280x832 | 832x1280 |

30 trajectories x 2 backends = 60 total. The aspect ratio PDF is no
longer degenerate. All existing infrastructure (bin packer, sigma shifting,
RoPE cache, pair sampler, FLOPS weighting, ValidationMetrics) handled
non-square resolutions without code changes — only the generation script's
`RESOLUTION_TIERS` configuration changed.

---

## Bug Fixes Applied

1. **logSNR bucketing formula** (`validation_metrics.py`): Changed from
   `-2*ln(sigma)` to `2*ln((1-sigma)/sigma)` (CONST noise model). The old
   formula misclassified noisy samples (sigma > 0.3) into cleaner buckets.

2. **Funfetti metadata passthrough** (`btrm_training.py`): The `funfetti`
   key was added to training curve entries but not passed through to
   `TrainingArtifacts.log_step()` via `extra_metrics`. Charts 06-10 were
   not generated on the first run. Fixed.

3. **FLOPS-normalized chart** (`training_artifacts.py`): New chart type
   showing `count * flops_ratio` per resolution, normalized to sum to 1.
   Complements the existing image count chart. Shows where compute was
   actually spent.

---

## Current State of Funfetti Batching

All four layers are complete and GPU-validated with stratified sampling:

| Layer | Module | Status |
|-------|--------|--------|
| 1. Packed scoring | `btrm_model.py` | GPU validated |
| 2. Bin-packed microbatches | `btrm_training.py` + `bin_packer.py` | GPU validated |
| 3. FLOPS sampling + stratification | `flops_sampling.py` + `pair_sampler.py` | GPU validated |
| 4. Multi-indexed validation | `validation_metrics.py` | GPU validated |

Supporting infrastructure:
- `training_artifacts.py`: 11 chart types, JSONL streaming, markdown analysis
- `exemplar_renderer.py`: VAE decode top/bottom-K per head
- `generate_multi_res_trajectories.py`: 9-resolution dataset generation

---

## Open Questions for User

1. **FLOPS interpretation**: Does "33% of training FLOPS on megapixel"
   mean 33% of pair slots (current implementation, producing 97.5%
   megapixel FLOPS) or literally 33% of compute (impossible at current
   macrobatch sizes)?

2. **Step time tradeoff**: The stratified run takes 5.5s/step vs 3.0s
   for the probabilistic run. This is the cost of guaranteed megapixel
   exposure. Is this acceptable, or should mega_fraction be reduced?

3. **Next milestone**: The infrastructure is done. The remaining work is
   experiments: (A) longer runs (500+ steps) for publishable accuracy
   curves, (B) funfetti vs monotonic comparison, (C) actual backend
   switching during generation (requires FastAPI server or standalone
   attention dispatch control). Which is highest priority?

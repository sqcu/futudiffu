# Synthesis: FLOPS-Budget Macrobatch Architecture

**Date:** 2026-02-19
**Author:** Root session
**Provenance:** Synthesis of 2 subagent deliverables:
- `essay_continuous_resolution_sampling.md` — 6-tier resolution system with algorithmic sampling
- `essay_flops_budget_macrobatch.md` — FLOPS-budget macrobatch + cross-resolution pairs

---

## The Arc

The funfetti batching integration reached a false summit with stratified
sampling: technically correct but architecturally wrong. The binary
mega/small bucketing with fixed pair counts produced 97.5% megapixel FLOPS
in a 4-pair macrobatch — not the ~33% target. Three corrections from the
user exposed the real design:

1. **The resolution space is continuous, not enumerated.** The 6 megapixel
   anchors (256² through 1024²) are FLOPS tiers for cost accounting, not a
   hardcoded table of legal resolutions. Aspect ratios are sampled from
   log-uniform distributions and quantized to 32px edges. Each tier
   accommodates 11-45 valid (W, H) pairs.

2. **A macrobatch is defined by FLOPS budget, not pair count.** ~3-4
   megapixel-equivalent compute units per optimizer step. The pair count
   is variable: 2 pairs for an all-megapixel dataset, ~375 for all-256².
   The number of FlexAttention forward passes (bins) is whatever the packer
   produces — typically 4-6.

3. **Cross-resolution pairs are valid.** A 1024² image can be compared
   against a 512² image for pinkify (quantization artifact) or thisnotthat
   (step count) ranking. Just as a 1024-token response can be compared to
   a 256-token response for helpfulness.

These three corrections transform the architecture from "fixed grid of
resolution × fixed pair count × binary FLOPS bucket" to "continuous
resolution × FLOPS-budget macrobatch × cross-tier comparison."

---

## What Was Implemented

### Round 1: Continuous resolution sampling

**New module: `src_ii/resolution_sampling.py`** (pure Python, ~250 lines)

Core: `sample_random_resolution(budget_pixels, rng)` computes a
32px-aligned (W, H) from a megapixel budget and log-uniform aspect
ratio. `enumerate_resolutions()` shows the full space per tier:

| Anchor | Valid (W, H) pairs | FLOPS ratio |
|--------|-------------------|-------------|
| 256² | 11 | 0.0004 |
| 320² | 14 | 0.0006 |
| 384² | 14 | 0.0012 |
| 512² | 22 | 0.004 |
| 704² | 31 | 0.014 |
| 1024² | 45 | 0.061 |

**Updated: `flops_sampling.py`** — 6-tier bucketing with configurable
per-tier FLOPS targets. Default: {1048576: 0.33}, rest distributed
proportionally. Backward-compatible binary mode preserved.

**Updated: `pair_sampler.py`** — per-tier CDFs. `sample_stratified_batch()`
allocates across all populated tiers. `assign_budget_tier()` maps any
(W, H) to its nearest anchor.

**Updated: `generate_multi_res_trajectories.py`** — algorithmic generation
from anchors. 10 trajectories per tier × 2 backends = 120 total. 43
unique (W, H) pairs from 60 per-backend entries. No hardcoded resolution
list.

**Updated: `bin_packer.py`** — `RESOLUTION_TIERS` deprecated. Enhanced
`validate_resolution()` with 32px step constraint.

### Round 2: FLOPS-budget macrobatch + cross-resolution pairs

**Updated: `flops_sampling.py`** — Added `flops_units(W, H)` and
`pair_flops_units()`. One FLOPS unit = one 1024² image's attention compute.

**Updated: `pair_sampler.py`** — Major addition:
- `PairSpec` dataclass with `to_pair_dict()` for training loop integration
- `prompt_idx` on `_ImagePosition` for cross-resolution prompt matching
- `_prompt_traj_ids` and `_cross_res_prompts` groupings in constructor
- `sample_macrobatch(budget_units=3.0)`: 3-phase algorithm —
  top-tier guarantee, deficit-weighted fill, shuffle
- `_sample_pair_spec()` with 30% cross-resolution probability

**Updated: `btrm_training.py`** — FLOPS-budget training path:
- `macrobatch_budget: float | None` parameter (default None = legacy mode)
- When set: samples all pairs via `sample_macrobatch()`, packs all images
  into bins, runs J forward passes, single backward, single optimizer step
- No inner microbatch loop. The entire macrobatch is one gradient
  accumulation unit.
- Loss normalization by active_heads, unchanged
- Rich per-step metadata in funfetti JSONL

---

## What Changed Architecturally

| Dimension | Before (fixed pair count) | After (FLOPS budget) |
|-----------|--------------------------|---------------------|
| Macrobatch size | 4 pairs (fixed) | Variable (~2 to ~375) |
| FLOPS per step | 0.03 to 8.3 units (variable) | ~3.0 units (constant) |
| Resolution space | Hardcoded enumeration | Continuous, 32px-quantized |
| FLOPS bucketing | Binary (mega/small) | 6-tier anchors |
| Cross-res pairs | Not supported | 30% probability when prompt match exists |
| Forward passes/step | 2 (fixed grad_accum) | Variable (~4-6) |
| Gradient accumulation | N microbatches | Single backward over all bins |
| Step time | 0.8-7.0s (resolution-dependent) | ~const (FLOPS-normalized) |

---

## The Gradient Noise Argument

The essay notes that loss normalization by active_heads makes the learning
rate invariant to macrobatch size. This is important: with variable pair
counts per step (2 for megapixel-heavy, 375 for tiny-heavy), the gradient
noise could vary wildly. But because the BT loss is
`-log(sigmoid(s_pref - s_rej))` summed and divided by `active_heads` (not
by pair count), the loss magnitude is bounded by `log(2) * n_heads`
regardless of how many pairs are in the macrobatch.

What DOES vary is the variance of the gradient estimate. More pairs = lower
variance (more averaging). The FLOPS-budget construction ensures that this
variance is roughly constant in WALL-CLOCK terms: a 3.0-unit macrobatch
always takes approximately the same time, and the gradient estimate quality
scales with compute spent, not pair count.

---

## What Remains: GPU Validation

The code changes are syntactically verified and pass inline functional
tests. What has NOT been tested:

1. **End-to-end GPU training with FLOPS-budget macrobatch.** The training
   loop's new code path (sample_macrobatch → pack → J forward passes →
   loss → backward) has not been exercised with real model weights and
   real gradients.

2. **Cross-resolution pair gradients.** The backward pass through
   `score_differentiable_packed()` with images of different sizes in the
   same pair has not been tested. The bin packer puts different-size images
   in different bins, so the scores are assembled post-hoc via
   `img_idx_to_score`. This assembly logic needs GPU verification.

3. **Dataset generation with the new algorithmic sampler.** The generation
   script's `_build_resolution_plan()` produces 120 trajectories with 43
   unique resolutions. This needs to run on GPU with the real backbone.

4. **torch.compile cache behavior.** The FLOPS-budget path may produce
   more diverse `total_len` values per step, potentially triggering more
   recompilations. The essay argues this is bounded by REFERENCE_TOTAL_LEN
   but it needs empirical confirmation.

The natural next step is a combined agent that: (A) generates the 120-trajectory
6-tier dataset, (B) runs a short (10-20 step) FLOPS-budget training run to
verify gradient flow and loss descent, (C) produces diagnostic charts
showing the per-step FLOPS budget consumption, resolution distribution,
and cross-resolution pair statistics.

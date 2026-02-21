# Synthesis: Funfetti Batching Layers 2, 3, and 4

**Date:** 2026-02-18
**Author:** Root session
**Provenance:** Synthesis of two parallel subagent deliverables completing
Layers 2-4 of the funfetti batching integration for BTRM reward model
training. Layer 1 (packed differentiable scoring) was completed in the
prior session and validated on RTX 4090.

---

## Status: All Four Layers Implemented

| Layer | Description | Agent | Status | Validation |
|-------|-------------|-------|--------|------------|
| 1 | `score_differentiable_packed()` | Prior session | Done | GPU validated (max_abs 0.051, 60/204 graded params) |
| 2 | Bin-packed multi-pair microbatches | Layer 2+3 agent | Done | 22 unit tests |
| 3 | Resolution-aware FLOPS sampling PDF | Layer 2+3 agent | Done | 22 unit tests (shared suite) |
| 4 | Multi-indexed validation covariance | Layer 4 agent | Done | 8 unit tests |

The funfetti batching integration is architecturally complete. No new
modules need to be designed. What remains is:

1. **Wiring Layer 4 into the training loop** (3 lines per the Layer 4 essay)
2. **GPU end-to-end validation** of the full packed training path
3. **Multi-resolution dataset generation** to exercise the path with real
   resolution diversity

---

## Cross-Cutting Observations

### 1. Loss normalization is correct

The Layer 2+3 agent made the right call on loss normalization: dividing by
`active_heads` (count of non-tied preference signals across all pairs and
heads in the microbatch) rather than by K (pair count) or 2K (image count).
This is important because different pairs contribute different numbers of
active preferences depending on which heads have non-zero preference for
that pair. Normalizing by active_heads gives a consistent per-signal
gradient magnitude regardless of batch composition.

This matters for funfetti specifically because resolution-diverse batches
may have systematically different head activity patterns. If small-resolution
pairs tend to have more tied preferences (e.g., both images look equally
bad at 256x256 to the pinkify head), normalizing by K would inflate the
gradient from the remaining active heads. Normalizing by active_heads
prevents this.

### 2. The passive observer pattern is the right boundary

The Layer 4 agent correctly implemented `ValidationMetrics` as a passive
observer with no control flow influence on the training loop. This is the
right architecture because:

- The tracker can be added or removed without changing training behavior
- The tracker's serialization (JSON) is independent of the training
  metrics (JSONL) — they compose rather than couple
- Future analysis dimensions can be added to the tracker without touching
  the training loop

The remaining integration work (3 lines in the training loop) is
mechanical: construct `PairResult` from pair metadata, call
`tracker.update()`, call `tracker.summary()` at log intervals.

### 3. The FLOPS sampling PDF degrades gracefully to the current regime

With the 96%-monoresolution V2 dataset, the FLOPS-weighted sampling
degenerates to uniform (all trajectories are megapixel, all get equal
weight). This is exactly the pre-change behavior. The funfetti code path
can be enabled today — it just won't demonstrate its value until
multi-resolution trajectories exist.

This is the correct engineering order: build the machinery, validate it
doesn't regress the existing path, then generate the data that exercises
its unique capabilities.

### 4. Two remaining gaps before GPU validation

**Gap A: Tracker integration.** `ValidationMetrics.update()` is not yet
called from the training loop. The training loop has the pair metadata
available (sigma, resolution, trajectory source, head-level accuracy).
The wiring is straightforward but should be done before the GPU validation
run so that the validation produces multi-indexed metrics.

**Gap B: Multi-resolution dataset.** The V2 dataset is 96% 1280x832. A
generation run at 256x256, 512x512, and 1024x1024 (at minimum) is needed
to exercise bin packing overflow, cross-resolution pair scoring, and
the FLOPS sampling PDF's non-degenerate behavior.

Gap A is code. Gap B requires GPU time. They're independent.

### 5. The quadratic cost model is correct for now

The FLOPS sampling PDF uses `(tokens/ref_tokens)^2` to model per-image
compute cost. This ignores linear FLOPS (FFN, embedding, LoRA), which
slightly underestimates the cost of small images and therefore slightly
oversamples them. As the Layer 2+3 essay notes, this is benign — the
oversampling increases gradient diversity, which is the point of funfetti.

A more precise cost model would be `a * tokens^2 + b * tokens` with
empirically measured a and b. This is a calibration refinement, not an
architectural change. The function signature accepts arbitrary weight
dicts, so a calibrated cost model can be swapped in without changing the
training loop.

---

## What's Next

### Immediate (code, no GPU required)

1. **Wire Layer 4 into the training loop.** Add `ValidationMetrics`
   construction, `PairResult` creation from pair metadata, and
   `tracker.summary()` at log intervals to `train_btrm_differentiable()`.
   This should be a small diff.

2. **Wire FLOPS weights into training scripts.** The orchestration scripts
   (`scripts_ii/train_pinkify_differentiable.py` or the forthcoming run03
   script) need to compute FLOPS weights from the dataset and pass them
   to the pair sampler.

### Near-term (GPU required)

3. **Multi-resolution dataset generation.** Run generation at 3+ resolution
   tiers. The generation infrastructure handles arbitrary resolutions
   (bin packer, sigma shifting, per-image RoPE). This is a configuration
   change to the generation script.

4. **GPU end-to-end validation.** Run `train_btrm_differentiable(packed=True,
   pairs_per_pack=2)` for 5-10 steps on the real backbone with
   multi-resolution data. Verify:
   - Loss descends
   - Adapter gradients are nonzero
   - Multi-indexed metrics show non-degenerate resolution distribution
   - Packed path loss trajectory is comparable to serial path

5. **Funfetti vs monotonic comparison.** The motivating experiment: train
   two models (same step count, same LR, same data), one with funfetti
   sampling (33/67 FLOPS split) and one with monotonic megapixel sampling.
   Compare accuracy-per-NFE across resolution buckets using the
   `ValidationMetrics` tracker. This is the experiment that justifies the
   entire funfetti architecture.

---

## Files Produced by This Round

| File | Layer | Description |
|------|-------|-------------|
| `src_ii/btrm_training.py` | 2 | Packed multi-pair microbatch loop with bin packing |
| `src_ii/flops_sampling.py` | 3 | Resolution-aware FLOPS sampling weights (new module) |
| `src_ii/pair_sampler.py` | 3 | Extended with resolution metadata and FLOPS weights |
| `src_ii/validation_metrics.py` | 4 | Multi-indexed Welford tracker (new module) |
| `tests/test_funfetti_layers.py` | 2+3 | 22 unit tests for FLOPS weights, bin packing, sampling |
| `docs/essay_funfetti_batch_construction.md` | 2+3 | Agent essay |
| `docs/essay_validation_covariance.md` | 4 | Agent essay |
| `docs/essay_synthesis_funfetti_layers_2_3_4.md` | — | This synthesis |

**`src/futudiffu/` was not modified by any agent.**

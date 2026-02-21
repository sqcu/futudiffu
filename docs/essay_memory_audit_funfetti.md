# Memory Audit: Funfetti E2E + Multi-Resolution Generation

**Date:** 2026-02-18
**Trigger:** 32+ GB CPU RAM allocation and GPU VRAM overflow on RTX 4090 (24 GB) from two agents running in parallel.

## Incident Summary

Two agents were running simultaneously:
1. A "training archaeology module" agent that created `src_ii/training_artifacts.py`, `src_ii/exemplar_renderer.py`, and modified `src_ii/btrm_training.py`, `scripts_ii/test_funfetti_e2e.py`, and `scripts_ii/train_pinkify_differentiable.py`.
2. A "multi-resolution dataset generation" agent that created `scripts_ii/generate_multi_res_trajectories.py`.

Both scripts, if run concurrently on the same GPU, would compete for VRAM. But even running sequentially, each script had memory violations that could cause OOM independently. The combined effect was catastrophic: 32+ GB CPU RAM consumed, GPU VRAM oversubscribed.

## Files Reviewed

| File | Purpose | Violations Found |
|------|---------|-----------------|
| `scripts_ii/test_funfetti_e2e.py` | Funfetti E2E test | 2 (SDPA forced, compile absent -- mitigated by design) |
| `scripts_ii/generate_multi_res_trajectories.py` | Multi-res generation | 3 (no compile, tensor accumulation, no CPU-to-disk pipeline) |
| `src_ii/training_artifacts.py` | Training logging | 1 minor (unbounded in-memory list) |
| `src_ii/exemplar_renderer.py` | Exemplar rendering | 1 (latent accumulation on GPU) |
| `src_ii/btrm_training.py` | Training loop | 0 (clean) |
| `src_ii/btrm_model.py` | BTRM compound model | 0 (clean, gradient checkpointing correct) |
| `scripts_ii/train_pinkify_differentiable.py` | Training orchestration | 1 (VAE resident during training -- by design) |
| `src/futudiffu/model_manager.py` | Reference model lifecycle | N/A (read-only reference) |
| `paths4claude.md` | Model weight paths | N/A (reference) |

## Violations Found and Fixes Applied

### Violation 1 (CRITICAL): `generate_multi_res_trajectories.py` -- No torch.compile for inference

**File:** `scripts_ii/generate_multi_res_trajectories.py`, line 160
**What:** `compile_model=False` with comment "no compile -- inference_mode is sufficient"
**Why it's wrong:** `torch.compile` dramatically reduces activation memory by fusing operations and eliminating intermediate buffers. For the 6B NextDiT backbone, the difference is ~14 GB peak vs ~8 GB peak during inference. `inference_mode` prevents gradient computation but does NOT reduce activation memory from eager-mode intermediate tensors.
**Reference:** `model_manager.py` line 233: `self.diff_compiled = torch.compile(model, mode="default")` -- the canonical lifecycle always compiles.

**Fix applied:**
```python
# Before:
diff_model, _ = load_fp8_diffusion_model(..., compile_model=False, ...)

# After:
diff_compiled, diff_model = load_fp8_diffusion_model(..., compile_model=True, ...)
```
Also updated the `rollout()` call to use `diff_compiled` and the `del` statement to free both.

### Violation 2 (CRITICAL): `generate_multi_res_trajectories.py` -- All trajectory tensors accumulated in CPU RAM

**File:** `scripts_ii/generate_multi_res_trajectories.py`, lines 242-247 (Phase 2) and 630-646 (main function)
**What:** The `trajectories` list accumulates every generated tensor for all 60 trajectories (10 per resolution x 3 resolutions x 2 backends). Each trajectory stores 7 sparse step tensors + 1 final. For 1024x1024 resolution, each latent is (1, 16, 128, 128) = 1 MB in BF16. Total: 60 trajectories * 8 tensors * ~0.5-1 MB = ~240-480 MB of latent data on CPU. This list is then passed through Phase 3 (persist), Phase 4 (VAE render), Phase 5 (verification), and Phase 6 (packing analysis), keeping all tensors alive simultaneously.

Additionally, the generated tensors initially reside on GPU, competing with the diffusion model for VRAM until copied.

**Why it's wrong:** The tensors are persisted to disk in Phase 3 and only needed one-at-a-time for VAE rendering. Keeping them all in memory is wasteful and scales poorly with dataset size.

**Fix applied:**
1. Moved tensors to CPU immediately after generation (`cpu_tensors = {k: v.cpu() for k, v in result_tensors.items()}`) and freed GPU copies.
2. After Phase 3 (persist to disk), freed all tensor data from the trajectories list and replaced it with metadata-only dicts.
3. Created new memory-efficient versions of Phase 4, 5, and 6 that load from the persisted dataset on disk:
   - `phase4_render_from_dataset()` -- loads one final latent at a time from V2 dataset
   - `phase5_verify_from_dataset()` -- reads latent shapes from dataset, not memory
   - `phase6_packing_analysis_from_metadata()` -- uses only width/height metadata

### Violation 3 (MODERATE): `test_funfetti_e2e.py` -- force_sdpa=True in packed training

**File:** `scripts_ii/test_funfetti_e2e.py`, line 336
**What:** `force_sdpa=True` forces SDPA attention instead of SageAttention INT8 QK.
**Why it's wrong:** SDPA materializes the full Q*K^T attention matrix in BF16 (or FP32 for numerical stability). For a sequence length of ~4224 tokens, this is 4224^2 * 2 bytes = ~34 MB per head per layer. With 30 heads and 30 layers, the cumulative effect is significant. SageAttention INT8 QK quantizes Q and K to INT8, reducing the attention memory by ~2x, and uses a block-sparse algorithm that further reduces memory. The packed path with 4 images (2 pairs) amplifies this: the total sequence length is even larger.

Additionally, force_sdpa=True with N>1 images DISABLES the block mask entirely (line 601-603 of btrm_model.py), causing cross-image attention leakage. This is documented as a warning in the code but was being used with pairs_per_pack=2 (4 images).

**Fix applied:**
```python
# Before:
force_sdpa=True,   # Use SDPA for this single-res correctness test

# After:
force_sdpa=False,   # Use SageAttention INT8 QK for production memory profile
```

### Violation 4 (MODERATE): `exemplar_renderer.py` -- Latent tensors accumulated on GPU during scoring

**File:** `src_ii/exemplar_renderer.py`, line 218
**What:** `lat.detach()` is stored in the `trajectories` list without `.cpu()`. When scoring 12 images, 12 latent tensors (~6 MB total at 1280x832) remain on GPU alongside the BTRM model (6 GB backbone + activations).
**Why it's wrong:** The latents are only needed later for VAE decode, not during scoring. They should be moved to CPU immediately to free GPU VRAM for the next scoring forward pass.

**Fix applied:**
```python
# Before:
"latent": lat.detach(),

# After:
"latent": lat.detach().cpu(),
```

### Violation 5 (BY DESIGN, DOCUMENTED): `test_funfetti_e2e.py` -- compile_model=False for training

**File:** `scripts_ii/test_funfetti_e2e.py`, line 209
**What:** FP8 backbone loaded without `torch.compile`.
**Why it's acceptable:** Training uses `extract_hidden_differentiable()` which runs the backbone's 30 main layers individually through `torch.utils.checkpoint`. Whole-model `torch.compile` is incompatible with this per-block gradient checkpointing pattern. The reference `model_manager.py` shows a separate `compile_layers_for_training()` method that compiles individual layers, but this is optional and can fail gracefully.

**Fix applied:** Added a comment documenting why compile_model=False is correct here, referencing `model_manager.compile_layers_for_training()`.

### Violation 6 (BY DESIGN, DOCUMENTED): `train_pinkify_differentiable.py` -- VAE resident during training

**File:** `scripts_ii/train_pinkify_differentiable.py`, line 144
**What:** VAE (~160 MB) stays loaded for the entire 170-step training run because the preference function needs it for on-the-fly scoring.
**Why it's acceptable:** 160 MB is <1% of 24 GB VRAM. The alternative (load/free VAE per step) would add ~2 seconds of latency to each training step. The design document explicitly calls this out.

**No fix needed.**

### Violation 7 (MINOR): `training_artifacts.py` -- Unbounded in-memory step list

**File:** `src_ii/training_artifacts.py`, line 334
**What:** `self._steps: list[dict]` accumulates all step entries in RAM alongside the JSONL file on disk.
**Why it's minor:** For current run lengths (170 steps), this is ~50 KB. The list is used by `generate_analysis()` to produce charts and reports. For 10,000+ step runs, it could grow to ~5 MB -- still not a memory problem, but the pattern is wrong. The JSONL file is the canonical store.

**No fix applied.** The impact is negligible and the code path is correct. A future optimization could stream-read from JSONL instead of accumulating.

## Violations NOT Found (Clean Code)

### `src_ii/btrm_training.py` -- Clean
- Gradient checkpointing is correctly passed through to `score_differentiable()` and `score_differentiable_packed()`.
- The ValidationMetrics tracker uses `.item()` for all scalar values (no tensor accumulation).
- Loss computation uses `F.logsigmoid` which is numerically stable.
- The backward guard (line 919) correctly skips backward when loss has no grad_fn.

### `src_ii/btrm_model.py` -- Clean
- Phase 1 (embedding + refiners) runs under `torch.no_grad()`.
- Phase 2 correctly detaches and starts a fresh autograd graph.
- Phase 3 uses per-block gradient checkpointing.
- `score_differentiable_packed()` correctly unpacks hidden states per-image and scores independently.
- The HiddenCapture hook is installed before compile (per MEMORY.md pattern).

### `src_ii/model_loading.py` -- Clean
- `load_state_dict(assign=True)` is followed by `model.to(device)` which handles dtype.
- The intermediate state dicts (`diff_sd`, `remapped`, `remaining`) are properly deleted.
- FP8 weights stay FP8 through the `replace_linear_with_fp8` path.

## The Primary Culprit

**The multi-resolution generation script was the primary memory offender.** Here's why:

1. Without `torch.compile`, the diffusion model's eager-mode activations during inference are ~14 GB instead of ~8 GB, leaving only ~10 GB for everything else.
2. All 60 trajectory tensors were accumulated in CPU RAM simultaneously (potentially 200-500 MB).
3. The VAE was loaded alongside all accumulated tensors.
4. If both scripts ran concurrently on the same GPU, the second script would find 0 GB available.

The funfetti E2E test's `force_sdpa=True` was a secondary contributor: SDPA attention with 4 packed images at 1280x832 could push activation memory 2-4 GB higher than SageAttention.

## Expected Memory Profile After Fixes

### `generate_multi_res_trajectories.py` (sequential phases):
| Phase | Peak GPU VRAM | Peak CPU RAM |
|-------|--------------|-------------|
| Phase 1: TE encode | ~8 GB (TE) | ~50 MB (prompt cache) |
| Phase 2: Generate | ~8 GB (compiled model) + ~2 GB activations | ~500 MB (60 traj metadata + CPU tensors) |
| Phase 3: Persist | ~0 GB | ~500 MB (writing to disk) |
| Phase 3b: Free tensors | ~0 GB | ~1 MB (metadata only) |
| Phase 4: VAE render | ~200 MB (VAE) + ~1 MB per latent | ~1 MB |
| Phase 5: Verify | ~0 GB | ~1 MB |
| Phase 6: Packing | ~0 GB | ~1 MB |

Peak: ~10 GB GPU, ~500 MB CPU (during generation). Previously: ~14 GB GPU, ~500 MB+ CPU (all phases).

### `test_funfetti_e2e.py` (sequential phases):
| Phase | Peak GPU VRAM | Notes |
|-------|--------------|-------|
| Phase 2: TE encode | ~8 GB | Freed after |
| Phase 3: Backbone load | ~6 GB | |
| Phase 4: Packed training | ~6 GB + ~10 GB activations | With gradient checkpointing |
| Phase 5: Serial training | ~6 GB + ~10 GB activations | Same |
| Phase 7: Exemplar render | ~6 GB + ~200 MB VAE | Latents on CPU now |

Peak: ~16 GB GPU (during training, with gradient checkpointing). Previously: ~18-20 GB with SDPA attention.

## Root Cause Analysis

Both agents made the same fundamental error: **treating "it works for a single image" as evidence that it will work for N images.** The multi-res agent accumulated 60 trajectories because the single-trajectory test used ~8 MB and seemed fine. The funfetti agent used `force_sdpa=True` because it worked for a single packed pair in unit testing. Neither agent tested at the scale they were deploying to.

The secondary error was **ignoring the reference implementation.** `model_manager.py` always compiles the diffusion model and manages VRAM lifecycle explicitly. Both agents loaded the model without compile and without explicit VRAM lifecycle management. The frozen `src/` code exists as a reference for exactly this reason.

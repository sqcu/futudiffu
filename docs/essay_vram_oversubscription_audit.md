# VRAM Oversubscription Audit: BTRM Training on RTX 4090

**Date**: 2026-02-17
**Scope**: `scripts_ii/sweep_rtheta_lr.py` -> `src_ii/btrm_model.py` -> `src_ii/btrm_training.py`, tracing full memory lifecycle from script start to first backward completion.

---

## Executive Summary

The expected budget of ~18 GB assumed "12 GB for activations with gradient checkpointing." That 12 GB figure is **wrong** -- it describes the activation cost WITHOUT checkpointing (30 layers * ~400 MB/layer). With per-layer gradient checkpointing properly applied (which it IS), activation memory is ~0.9-1.9 GB per forward pass, not 12 GB.

The actual estimated peak is **~10.5 GB** for 1 MP images, well within 24 GB. If the system is OOMing, the most likely causes are:

1. **Larger-than-expected image resolution** (e.g., 2 MP quadruples per-token costs)
2. **CUDA allocator fragmentation** (reserved >> allocated, especially with checkpoint alloc/free churn)
3. **Latent cache bloat** (per-image RoPE duplication)

The text encoder IS correctly freed. The VAE is NOT loaded. There are no BF16 master copies of FP8 weights. AdamW state is correctly scoped to LoRA params only. Gradient checkpointing IS properly applied.

---

## Component-by-Component Residency Table

| Component | Size | Status | Evidence |
|---|---|---|---|
| FP8 model weights + scales | 5.85 GB | Resident | `load_file(path, device="cuda")` in `model_loading.py:65`. FP8 1B/param + float32 scales. On-disk file confirmed at 5.8 GB. |
| Non-FP8 params (embedders, norms, final_layer) | ~120 MB | Resident | BF16. x_embedder, cap_embedder, t_embedder, 4 RMSNorm/layer, final_layer, pad tokens. |
| LoRA adapter (rtheta, rank=8) | ~20 MB | Resident | 102 LoRALinear modules: 3 targets (qkv, out, w2) x 34 layers (30 main + 2 noise_refiner + 2 context_refiner). w1/w3 are fused into w1w3 by `fuse_model()` before adapter allocation, so those targets do not exist. ~10.1M params x 2 bytes BF16. |
| adaLN batched buffer (`_adaln_W`) | ~241 MB | **Resident, UNUSED** | `fuse_model()` -> `prepare_adaln_cache()` dequantizes 32 FP8 adaLN weights to BF16 and concatenates into `_adaln_W` shape (491520, 256). Training forward path in `extract_hidden_differentiable()` bypasses this buffer entirely -- each layer computes adaLN independently. Pure waste. |
| Text encoder (Qwen3-4B) | 7.5 GB | **FREED** | `encode_all_prompts()` at `sweep_rtheta_lr.py:151-153`: `del te_model, tokenizer; torch.cuda.empty_cache(); gc.collect()`. Freed before diffusion model loads. |
| VAE | 160 MB | **NOT LOADED** | No VAE imports anywhere in training path. |
| Prompt embeddings | ~48 MB | Resident (via cache) | `encode_all_prompts()` stores CPU tensors. `build_latent_loader()` copies to GPU per-image and caches. 80 copies of ~0.6 MB conditioning = ~48 MB. Could be 8 copies with deduplication. |
| Latent cache (GPU) | ~776 MB | **Resident** | `build_latent_loader()` caches `(latent, timestep, conditioning, num_tokens, rope_cache)` on GPU per unique image. Pre-warming loop loads all. **Dominant cost is RoPE duplication**: 80 copies of identical ~8.6 MB rope_cache for same-resolution images. |
| ScoreUnembedder (BTRM head) | ~46 KB | Resident | RMSNorm(3840) + Linear(3840, 2) in FP32. 11,520 params. |
| AdamW optimizer state | ~80 MB | Resident | 10.1M LoRA + 11.5K head params. Float32 momentum + variance: ~10M * 4B * 2 = 80 MB. |
| Gradient tensors | ~40 MB | Transient | Same shape as trainable params. Float32. |
| CUDA context + cuBLAS workspace | ~800 MB | Resident | Fixed overhead for CUDA runtime. |
| Checkpoint boundary tensors (image A) | ~930 MB | Retained until backward completes | 30 layers x (1, 4224, 3840) BF16 = 30 x 31 MB. |
| Checkpoint boundary tensors (image B) | ~930 MB | Retained until backward completes | Same as image A. Both graphs live simultaneously. |
| Backward recomputation peak (one layer) | ~660 MB | Transient | Per-layer: ~300 MB recomputed activations + ~362 MB FP8 dequant temporaries. Sequential, not cumulative. |

---

## Peak VRAM Budget (1 MP images, B=1)

**Persistent (entire training step):**

| Item | GB |
|---|---|
| FP8 weights + scales | 5.85 |
| Non-FP8 params | 0.12 |
| LoRA params | 0.02 |
| adaLN buffer (waste) | 0.24 |
| Latent + RoPE cache | 0.78 |
| AdamW state | 0.08 |
| CUDA context | 0.80 |
| **Subtotal** | **7.89** |

**Peak transient (backward start, both graphs still live):**

| Item | GB |
|---|---|
| Graph A: 30 checkpoint boundaries | 0.93 |
| Graph B: 30 checkpoint boundaries | 0.93 |
| One layer recomputation + dequant | 0.66 |
| Embed clones (2x) + autograd metadata | 0.10 |
| **Subtotal** | **2.62** |

**Total estimated peak: ~10.5 GB**

**CUDA allocator reserved (with fragmentation): ~12-14 GB** (allocator typically reserves 20-40% more than allocated due to block splitting and the checkpoint alloc/free pattern).

---

## Why the Original 18 GB Estimate Was Wrong

The original budget assumed "12 GB for activations with gradient checkpointing." This is incorrect:

- **Without checkpointing**: All 30 layers store full intermediates simultaneously. Per-layer: ~400 MB (FFN intermediates + QKV + attention output). Total: 30 * 400 = **~12 GB**. This is correct for the un-checkpointed case.
- **With per-layer checkpointing** (as implemented): Only boundary activations are stored (layer outputs). During backward, one layer at a time is recomputed. Storage: 30 boundaries * 31 MB = **~0.93 GB per forward**. The recomputation peak adds ~660 MB for one layer. Total activations: **~1.6 GB per forward**, not 12 GB.

The 7x reduction from 12 GB to 1.6 GB is the whole point of gradient checkpointing. It IS working correctly.

---

## Oversubscription Culprits (Ranked)

If OOM is occurring at 24 GB despite the ~10.5 GB estimate, these are the likely causes:

| Rank | Culprit | Estimated Cost | Evidence |
|---|---|---|---|
| 1 | **Image resolution >> 1 MP** | Quadratic in tokens for attention, linear for everything else. At 2 MP (~8192 tokens), checkpoint boundaries alone are 30 * 8192 * 3840 * 2 = ~1.88 GB per forward, both graphs = 3.76 GB. Total peak: ~15 GB. At 4 MP: ~22 GB. | Check actual latent dimensions in btrm_dataset. |
| 2 | **CUDA allocator fragmentation** | 2-5 GB gap between allocated and reserved. Checkpointing creates many alloc/free cycles per layer, which fragments the allocator. | Profile with `torch.cuda.memory_stats()`. Check `reserved_bytes.all.peak` vs `allocated_bytes.all.peak`. |
| 3 | **Latent + RoPE cache on GPU** | ~776 MB for 80 images, of which ~688 MB is redundant (same rope_cache duplicated per image). | `build_latent_loader` creates per-image rope_cache in `sweep_rtheta_lr.py:199`. |
| 4 | **Two concurrent computation graphs** | 2 * 0.93 GB instead of 1 * 0.93 GB. Excess: ~0.93 GB. | `btrm_training.py:373-380`: scores_a and scores_b both computed before backward. |
| 5 | **adaLN batched buffer** | 241 MB, completely unused during training. | `diffusion_model.py:811-812`: `_adaln_W` register_buffer created by `fuse_model()` which is called by `load_fp8_diffusion_model(fuse=True)`. |

---

## What to Fix

### Fix 1: Determine actual image resolution (DIAGNOSTIC)

Before fixing anything, check the actual latent dimensions:
```python
import torch
lat = torch.load("btrm_dataset/latents/traj_000000/step_00.pt")
print(f"Latent shape: {lat.shape}")  # (1, 16, H, W)
# H=128, W=128 -> 1 MP (1024x1024)
# H=192, W=192 -> 2.25 MP (1536x1536) -> THIS WOULD EXPLAIN OOM
```

If latents are 192x192 or larger, the per-token costs scale with (H/2 * W/2) and the two-forward budget balloons.

### Fix 2: Delete adaLN cache before training

The training forward path bypasses `_compute_adaln_params()` entirely. The buffer is dead weight.

```python
# After load_fp8_diffusion_model() returns, before creating BTRMCompoundModel:
if hasattr(diff_model, '_adaln_W'):
    del diff_model._adaln_W
    del diff_model._adaln_B
    torch.cuda.empty_cache()
```

Saves 241 MB.

### Fix 3: Share RoPE cache across same-resolution images

Instead of creating a new `rope_cache` per image index in `build_latent_loader()`, create one per unique `(H, W, num_tokens)` tuple:

```python
rope_cache_pool = {}

def load_latent(image_idx):
    ...
    key = (H, W, num_tokens)
    if key not in rope_cache_pool:
        rope_cache_pool[key] = make_rope_cache(diff_model, H, W, num_tokens, device)
    rope_cache = rope_cache_pool[key]
    ...
```

Saves ~688 MB if all images are the same resolution (80 copies -> 1 copy).

### Fix 4: Keep latents on CPU, transfer per-step

Latents are ~0.5 MB each. CPU->GPU transfer takes ~0.1 ms. The per-step cost is negligible compared to the ~10s forward pass:

```python
def load_latent(image_idx):
    ...
    latent = torch.load(pt_path, weights_only=True)  # stays on CPU
    conditioning = unique_prompts[prompt]  # stays on CPU
    # Only move to GPU when returned
    return (latent.to(device), timestep.to(device), conditioning.to(device), ...)
```

This eliminates the entire latent cache cost but means loading from disk each time. Better: cache on CPU, transfer per-step.

### Fix 5: Batched B=2 forward (reduces fragmentation, not total memory)

Replace the two sequential B=1 forwards with one B=2 forward. This does not reduce total activation memory (2 * N_tokens * dim is the same either way) but consolidates it into fewer, larger allocations that fragment less:

```python
# Instead of two score_differentiable() calls:
lat_both = torch.cat([lat_a, lat_b], dim=0)       # (2, C, H, W)
ts_both = torch.cat([ts_a, ts_b], dim=0)           # (2,)
cond_both = torch.cat([cond_a, cond_b], dim=0)     # (2, seq, dim)
scores_both = model.score_differentiable(lat_both, ts_both, cond_both, ...)
scores_a, scores_b = scores_both.split(1, dim=0)
```

Requires that both images have the same resolution and conditioning length (or padding).

### Fix 6: Reduce CUDA fragmentation

```python
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

Or for more aggressive defragmentation:
```python
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,expandable_segments:True'
```

---

## Detailed Answers to Specific Questions

### 1. Is the text encoder still in VRAM during training?

**NO.** `encode_all_prompts()` (`sweep_rtheta_lr.py:137-155`) loads the 7.5 GB Qwen3-4B encoder, encodes all unique prompts, stores results on CPU, then explicitly frees the encoder:
```python
del te_model, tokenizer
torch.cuda.empty_cache()
gc.collect()
```
This happens before `load_fp8_diffusion_model()` is called.

### 2. Are latents cached on GPU or CPU?

**GPU.** `build_latent_loader()` (`sweep_rtheta_lr.py:174-205`) loads each latent with `.to(device=device, dtype=dtype)` and stores the result in a closure-scoped `cache` dict. The pre-warming loop (`sweep_rtheta_lr.py:376-381`) eagerly populates the entire cache. For 80 unique images at ~9.7 MB each (latent + conditioning + rope_cache), total GPU cache is **~776 MB**, of which ~688 MB is redundant per-image RoPE duplication.

### 3. Is the model compiled with torch.compile?

**NO.** `load_fp8_diffusion_model()` is called with `compile_model=False` (`sweep_rtheta_lr.py:362`). The custom FP8 kernels operate through the `custom_op` mechanism and work without compilation. SageAttention is not explicitly configured; the default SDPA backend is used.

### 4. Are there duplicate weight tensors?

**One duplication exists.** `load_state_dict(assign=True)` avoids copies. `model.to(device)` is a no-op for tensors already on the device. The sole duplication is `_adaln_W` (241 MB BF16 buffer) created by `prepare_adaln_cache()`, which dequantizes FP8 adaLN weights. This buffer is never used during the training forward path.

The `fuse_w1w3()` function does `del self.w1; del self.w3` after creating the fused weight, so no FFN duplication exists.

### 5. Is gradient checkpointing actually applied?

**YES.** `extract_hidden_differentiable()` (`btrm_model.py:304-309`) uses `torch.utils.checkpoint.checkpoint(layer, ..., use_reentrant=False)` on each of the 30 main layers. The `gradient_checkpointing` parameter defaults to `True` and is passed through from `train_btrm_differentiable()`.

Note: only the 30 main layers are checkpointed. The 2 context_refiner + 2 noise_refiner layers run under `torch.no_grad()` (Phase 1 of `extract_hidden_differentiable()`), which is correct since they contain no trainable parameters.

### 6. Are there any BF16 copies of FP8 weights?

**Only `_adaln_W` (241 MB, unused in training).** The FP8 backward path creates **temporary** BF16 dequantizations via `dequantize_fp8_blockwise()` -- one layer at a time during backward recomputation, ~362 MB peak, freed after each layer's backward completes. No persistent BF16 master copies exist.

### 7. Is the VAE loaded during training?

**NO.** No VAE imports or model loading in `sweep_rtheta_lr.py`, `btrm_model.py`, or `btrm_training.py`.

### 8. Prompt embeddings: are all 8 unique prompt encodings resident on GPU simultaneously?

**YES, but multiplied by 10x due to per-image caching.** `encode_all_prompts()` stores ~8 unique embeddings on CPU. `build_latent_loader()` copies conditioning to GPU per-image-index, creating up to 80 GPU copies (one per cache entry, ~0.6 MB each = ~48 MB total). If deduplicated to 8 copies: ~5 MB. The 43 MB excess is negligible.

### 9. AdamW state: confirm it's only over LoRA params (10M) not full model (5.8B)?

**CONFIRMED.** `BTRMCompoundModel.optimizer()` (`btrm_model.py:161-177`) creates `torch.optim.AdamW(self.all_trainable_params(), ...)`. `all_trainable_params()` returns `self.adapter_params() + self.head_params()` -- only LoRA A/B matrices and the ScoreUnembedder. All backbone parameters have `requires_grad=False` (set at `btrm_model.py:94-95`). Total optimizer state: **~80 MB**.

### 10. Hidden state capture: does HiddenCapture hold a reference to the full hidden state tensor between forward and scoring?

**YES, but no extra memory cost.** The forward hook (`btrm_model.py` -> `training_utils.py:83`) stores a reference to the last layer's output in `self.captured`. During `extract_hidden_differentiable()`, the hook fires on the last `grad_ckpt()` call and stores a reference to the same `embed` tensor that the function returns. This is the same object already retained by the computation graph, so the hook reference adds zero bytes. Between the two `score_differentiable()` calls, image A's captured tensor is overwritten by image B's. The autograd graph retains image A's tensor independently of the hook.

---

## Files Examined

| File | Role |
|---|---|
| `/mnt/f/dox/repos/ai/futudiffu/scripts_ii/sweep_rtheta_lr.py` | Sweep script: orchestrates TE encode, model load, cache warm, training probes |
| `/mnt/f/dox/repos/ai/futudiffu/src_ii/btrm_model.py` | BTRMCompoundModel: coordinates backbone, adapter, head |
| `/mnt/f/dox/repos/ai/futudiffu/src_ii/btrm_training.py` | `train_btrm_differentiable()`: the training loop |
| `/mnt/f/dox/repos/ai/futudiffu/src/futudiffu/training_utils.py` | `HiddenCapture`, `forward_checkpointed()` |
| `/mnt/f/dox/repos/ai/futudiffu/src_ii/model_loading.py` | `load_fp8_diffusion_model()`: FP8 loading + fusion |
| `/mnt/f/dox/repos/ai/futudiffu/src/futudiffu/diffusion_model.py` | NextDiT model, `fuse_model()`, `prepare_adaln_cache()` |
| `/mnt/f/dox/repos/ai/futudiffu/src/futudiffu/text_encoder.py` | Qwen3-4B text encoder loading and encoding |
| `/mnt/f/dox/repos/ai/futudiffu/src/futudiffu/lora.py` | LoRA allocation, weight management, parameter counting |
| `/mnt/f/dox/repos/ai/futudiffu/src/futudiffu/fp8.py` | FP8Linear, custom_op autograd (backward dequantization) |
| `/mnt/f/dox/repos/ai/futudiffu/src/futudiffu/btrm.py` | ScoreUnembedder, Bradley-Terry loss |

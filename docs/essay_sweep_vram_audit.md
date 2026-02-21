# VRAM Audit: `scripts_ii/sweep_rtheta_lr.py`

## The Budget

The RTX 4090 has 24 GB of VRAM. Here is the static breakdown of what the
sweep script loads before any training begins:

| Component | Estimate | Notes |
|---|---|---|
| Diffusion model (FP8 weights + blockwise scales) | ~6.0 GB | 5.8B params stored as float8_e4m3fn with FP32 scales |
| LoRA adapter (rank 8, all 30 layers) | ~0.02 GB | ~600K params in bf16, negligible |
| BTRM head (ScoreUnembedder) | <0.01 GB | ~12K params in FP32, negligible |
| Latent cache (all unique images, on GPU) | ~0.06 GB | See calculation below |
| Conditioning cache (on GPU) | ~0.3 GB | See calculation below |
| RoPE cache (on GPU, per-image) | ~0.1 GB | See calculation below |
| **Subtotal: static residents** | **~6.5 GB** | |

This leaves ~17.5 GB for activations, autograd graph, and optimizer state
during training. Whether that is enough depends entirely on whether gradient
checkpointing is used.

### Latent cache calculation

Each latent is `(1, 16, H, W)` in bf16. The `.pt` files are ~534 KB, meaning
each latent is roughly `(1, 16, 128, 128)` -- a megapixel image at 1/8
resolution. That is `1 * 16 * 128 * 128 * 2 = 524,288 bytes` per latent.

The script uses `N_TRAJECTORIES = 10` trajectories, each with 8 sparse steps
(step_00, 04, 09, 14, 19, 24, 29, final). The training pairs index into
`per_image_scores`, which contains these 80 images. On lines 362--368, the
script pre-warms the cache by iterating over all unique indices referenced by
training pairs:

```python
for idx in sorted(unique_indices):
    load_latent_fn(idx)
```

If all 80 images are referenced, total latent VRAM is `80 * 0.5 MB = 40 MB`.
This is small. The latent cache itself is not the problem.

### Conditioning and RoPE cache calculation

Conditioning is more significant. Each unique prompt produces a
`(1, seq_len, 3584)` bf16 tensor (Qwen3-4B hidden dim = 3584). A typical
tokenized prompt might be 50-200 tokens. At 200 tokens:
`1 * 200 * 3584 * 2 = 1.4 MB` per prompt. With ~10 unique prompts that is
~14 MB. Still small.

However, the `build_latent_loader` cache (line 202) stores not just latents but
also conditioning, timesteps, and RoPE caches -- one tuple per image index.
The RoPE cache contains `freqs_cis` tensors of shape
`(1, n_axes, seq_len, 1, rope_dim, 2)` in float32. For a megapixel image with
~1000 image tokens and ~200 text tokens, this is roughly
`1 * 3 * 1200 * 1 * 64 * 2 * 4 = 1.8 MB` per cache entry. At 80 entries,
that is ~144 MB.

Total static cache VRAM: ~200 MB. This is not the oversubscription risk.

## The Actual Problem: Two Full Differentiable Forwards Per Training Step

The training loop (`train_btrm_differentiable`, called from `run_probe`) runs
one training step per iteration. Each step:

1. Loads a pair of images from the cache (already on GPU).
2. Calls `btrm_model.score_differentiable()` **twice** -- once for image A,
   once for image B.
3. Each `score_differentiable()` call runs `extract_hidden_differentiable()`,
   which is a full forward through all 30 transformer layers.

The critical question is whether gradient checkpointing is used. The sweep
script passes `gradient_checkpointing=True` on line 271, and this propagates
through to `extract_hidden_differentiable()` in `btrm_model.py` (line 304),
which does use `torch.utils.checkpoint.checkpoint` per-layer when the flag is
True.

**So gradient checkpointing IS used.** This is correct. But there is a subtlety
that still causes VRAM pressure.

## Issue 1: Two Live Computation Graphs Simultaneously

In `train_btrm_differentiable` (btrm_training.py, lines 345--353):

```python
scores_a = btrm_model.score_differentiable(
    lat_a, ts_a, cond_a, nt_a, rc_a,
    gradient_checkpointing=gradient_checkpointing,
)
scores_b = btrm_model.score_differentiable(
    lat_b, ts_b, cond_b, nt_b, rc_b,
    gradient_checkpointing=gradient_checkpointing,
)
```

Both `scores_a` and `scores_b` are live tensors with `grad_fn` when
`loss.backward()` is called on line 390. PyTorch must retain the checkpoint
metadata for both forward passes until backward completes.

With gradient checkpointing, each of the 30 layers saves only its inputs (not
intermediate activations). The per-layer input tensor is
`(1, ~1200, 3840)` in bf16 = ~9.2 MB. For 30 layers, that is ~276 MB per
forward pass. Two simultaneous forward passes means ~552 MB of checkpoint
storage.

During backward, each layer is re-executed to reconstruct intermediates. The
peak recomputation activations for one layer of a NextDiT block (QKV
projection, attention, FFN with SiLU+gate) at megapixel resolution are roughly
200--400 MB. This is manageable.

But the key question is: does backward recompute both graphs simultaneously?
No -- PyTorch processes the backward graph sequentially. It walks backward
through `scores_b`'s graph first (since it was computed last), then through
`scores_a`'s graph. The checkpoint recomputation for `scores_a` does not
overlap with `scores_b`'s recomputation.

**Verdict**: Two simultaneous computation graphs add ~550 MB to VRAM vs a
single-graph approach, but this is within the 17.5 GB headroom. This is a
moderate concern, not a fatal one.

## Issue 2: The Latent Cache Holds GPU Tensors Across the Entire Sweep

The `build_latent_loader` function (lines 158--205) creates a closure with a
`cache` dict. Every call to `load_latent_fn(idx)` populates this cache, and
nothing ever evicts from it. Lines 362--368 explicitly pre-warm the entire
cache:

```python
for idx in sorted(unique_indices):
    load_latent_fn(idx)
```

The cached tuples contain:
- `latent`: bf16 GPU tensor
- `timestep`: bf16 GPU tensor (tiny)
- `conditioning`: bf16 GPU tensor
- `rope_cache`: dict of float32 GPU tensors

These remain resident for the entire script lifetime -- across all probes in
Phase 1, Phase 2, and Phase 3. They are never freed between probes.

The total is ~200 MB as estimated above. This is not the catastrophic issue,
but it is architecturally wrong: these tensors should live on CPU and be moved
to GPU per-step. The `train_btrm_differentiable` reference function's
docstring says the `load_latent_fn` returns tensors "already on CUDA," but the
`encode_all_prompts` function (line 148) correctly stores encoded prompts on
CPU (`unique_prompts[prompt] = cond.cpu()`). The latent loader then undoes this
by moving them back to GPU and caching them there permanently (line 186--189).

**The fix**: `load_latent_fn` should store its cache on CPU and return
`.to(device)` copies each call, or (better) not cache at all and reload from
disk per-step. The latents are 0.5 MB each; disk I/O is negligible compared
to a 10-second forward pass.

## Issue 3: Conditioning Is Loaded to GPU Twice

In `encode_all_prompts` (line 148), prompts are encoded and stored on CPU:
```python
unique_prompts[prompt] = cond.cpu()
```

In `build_latent_loader` (line 189), they are moved back to GPU:
```python
conditioning = unique_prompts[prompt].to(device=device, dtype=dtype)
```

And then cached in the closure's `cache` dict (line 202). So each unique
conditioning tensor exists on GPU once (in the closure cache). The CPU copies
in `unique_prompts` also persist. This is a minor inefficiency (14 MB of
unnecessary CPU memory) but is not a VRAM issue per se.

## Issue 4: Phase 4 Attention Diff Loads Additional Latents Without Clearing the Cache

The `run_attention_diff` function (lines 527--644) loads 4 additional latents
to GPU (line 577--578), creates a new `BTRMCompoundModel` from disk (line 560),
and runs inference-mode forwards. These forwards are under `no_grad` (attention
capture), so they do not create autograd graphs. However:

1. The attention capture `stats_a` and `stats_b` dicts (lines 592--599) store
   per-layer attention statistics (`attn_received` tensors) on GPU. With 30+
   layers and per-head statistics, this could be significant.
2. The original latent cache from the training phases is still resident in the
   closure -- it was never freed.
3. A fresh `BTRMCompoundModel` is created (line 560), which allocates a new
   LoRA adapter on the backbone. The old adapter from the last probe was
   removed (line 317, `remove_all_adapters`), but the new one adds VRAM.

This phase is inference-only, so the absence of gradient checkpointing is
fine. But the cumulative VRAM from the still-resident latent cache plus
attention statistics could cause pressure if the attention capture stores
large per-layer tensors.

## Issue 5: AdamW Optimizer State

Each probe creates a fresh `BTRMCompoundModel` and calls `btrm_model.optimizer(lr=lr)`.
AdamW stores two momentum buffers (m and v) per parameter, each the same size
as the parameter. For rank-8 adapters across 30 layers:

- Adapter params: ~600K params in bf16 = ~1.2 MB for weights, but AdamW
  stores m and v in FP32 = `600K * 4 * 2 = 4.8 MB`
- Head params: ~12K params in FP32 = `12K * 4 * 2 = 96 KB`

Total optimizer state: ~5 MB. Negligible.

For rank-64 adapters (Phase 3 rank sweep), adapter params grow to ~4.8M,
and optimizer state to ~38 MB. Still negligible.

## Issue 6: No `torch.cuda.empty_cache()` Between the Two `score_differentiable` Calls

During training, the two sequential `score_differentiable` calls within a
single step do not release CUDA memory between them. The first call's
checkpoint metadata is held while the second call allocates its own. PyTorch's
caching allocator will reuse freed blocks where possible, but the peak
concurrent allocation includes both graphs' checkpoint storage.

The fix would be to compute scores sequentially with explicit cache clearing
between them, or (better) to restructure so backward runs after each forward
individually. However, the BT loss requires both scores simultaneously, so
this would require accumulating gradients manually.

## Summary: Is This Script Going to OOM on a 4090?

Probably not, but it is closer to the edge than it needs to be.

| Budget line | VRAM |
|---|---|
| FP8 model weights + scales | 6.0 GB |
| Cached latents + conditioning + RoPE (all on GPU) | 0.2 GB |
| LoRA adapter + head | 0.02 GB |
| Optimizer state (AdamW, rank 8) | 0.005 GB |
| **Checkpoint storage (2 graphs x 30 layers)** | **0.55 GB** |
| **Peak recomputation activations (1 layer)** | **0.2--0.4 GB** |
| PyTorch allocator overhead + fragmentation | 1--3 GB |
| **Estimated peak** | **~8--10 GB** |
| **Available** | **24 GB** |

The script should fit. The gradient checkpointing is correctly applied, the
batch size is 1 (one pair per step), and the latent cache is small. The main
risk factors are:

1. **Fragmentation**: The PyTorch caching allocator can fragment VRAM if
   allocations and deallocations happen in the wrong order. The per-probe
   create/destroy cycle (allocate adapter, train 100 steps, remove adapter,
   repeat 12 times) could fragment the address space.

2. **Phase 3 rank-64 sweep**: Rank 64 increases adapter parameter count by 8x.
   This does not change forward activation size (the LoRA pathway is
   additive to the base GEMM), but it increases checkpoint storage slightly
   because the LoRA intermediates become part of the saved inputs.

3. **Accumulation across probes**: If `remove_all_adapters` + `btrm.cleanup()`
   + `torch.cuda.empty_cache()` does not fully release all adapter-related
   VRAM, residual allocations accumulate across the 12+ probes.

4. **Phase 4 attention capture**: Storing per-layer attention statistics for
   30+ layers concurrently with the latent cache is the highest-risk moment
   for VRAM pressure, even though it is inference-only.

## What the Fix Should Be

1. **Move the latent cache to CPU.** `build_latent_loader` should store cached
   tuples on CPU and produce GPU copies on demand. The 0.5 MB per latent
   transfer is invisible next to a 10-second forward pass. This reclaims
   ~200 MB and, more importantly, eliminates a class of fragmentation risk
   from long-lived small GPU allocations.

2. **Clear the latent cache between phases.** After Phase 1 completes and a
   winner is selected, there is no reason to keep all 80 cached latents
   resident. At minimum, the cache should be cleared between Phases 1/2/3/4.

3. **Consider sequential score + backward.** Instead of computing both
   `scores_a` and `scores_b` before calling `loss.backward()`, compute
   `scores_a`, detach its value for the loss computation, backward through
   the first graph, then compute `scores_b` and backward through the second.
   This halves peak checkpoint storage from ~550 MB to ~275 MB. The gradient
   accumulation is trivial since both contribute additively to the BT loss.
   (This is how `compute_reinforce_step` in `training_utils.py` already works
   for the policy path -- one no_grad reference pass, one checkpointed policy
   pass.)

4. **Free attention capture tensors eagerly in Phase 4.** The `stats_a` and
   `stats_b` dicts should be moved to CPU or deleted after computing the
   per-layer diff, not held until the end of the loop.

5. **Add a VRAM watchdog.** The script already prints allocated/reserved VRAM
   after cache warmup (line 372--374). It should also print VRAM after each
   probe completes and at the start of Phase 4, to detect creeping leaks.

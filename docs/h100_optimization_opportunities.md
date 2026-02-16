# H100 (SM90) Optimization Opportunities

Analysis of four optimization opportunities available on H100 Hopper (SM90) that are
not available or not beneficial on our current RTX 4090 (SM89) target. This document
is a reference for implementation planning.

Cross-references:
- `remote_training_plan.md` -- GPU performance analysis, timeline estimates
- `multi_gpu_scaling.md` -- multi-GPU data parallelism design
- `fp8_kernels.py` -- current Triton FP8 GEMM kernels (SM89-tuned)
- `fp8.py` -- FP8Linear, custom_ops, blockwise quantization
- `server.py` -- model lifecycle management and offloading
- `training_utils.py` -- gradient checkpointing (forward_checkpointed)

---

## 1. torch._scaled_mm replacing custom Triton FP8 GEMM kernels

### Background

Our custom Triton FP8 GEMM kernels (`fp8_kernels.py`) exist because
`torch._scaled_mm` historically supported only scalar (tensorwise) scales. The
comment on line 17 of `fp8_kernels.py` says this explicitly:

> "torch._scaled_mm only supports scalar (tensorwise) scales, hence these custom kernels."

Our model uses **blockwise FP8 quantization** with `block_size=128`, producing 2D
scale tensors of shape `(ceil(N/128), ceil(K/128))` for weight matrices. This format
is identical to the DeepSeek-V3 quantization scheme: 128x128 blocks for weights,
1x128 blocks for activations (rowwise).

### Current state of torch._scaled_mm (PyTorch 2.10+, CUDA 12.8+)

**Measured fact** (from PyTorch source and issue tracker):

`torch._scaled_mm` supports the following scaling modes as of PyTorch 2.10:

| Mode | scale_a shape | scale_b shape | Backend | SM requirement |
|------|--------------|--------------|---------|----------------|
| Tensorwise | (1,) | (1,) | cuBLAS | SM89+ |
| Rowwise | (M, 1) | (1, N) | CUTLASS | SM90 only |
| BlockWise 1x128 | (M, K/128) | (N/128, K/128) | cuBLAS | SM90 only |
| BlockWise 128x128 | (M/128, K/128) | (N/128, K/128) | cuBLAS | SM90 only |

**Key finding**: PyTorch PR #158037, merged July 24, 2025, added `BlockWise128x128`
support to `torch._scaled_mm` on SM90+ via new cuBLAS bindings introduced in CUDA
12.9. This is exactly our quantization format.

The blockwise path uses cuBLAS (not CUTLASS), which means it benefits from NVIDIA's
hand-tuned Hopper PTX. Scale format is auto-detected from scale tensor dimensions --
no explicit mode flag needed. Passing `scale_a` with shape `(M/128, K/128)` and
`scale_b` with shape `(N/128, K/128)` automatically triggers the blockwise kernel.

**Important caveat**: The blockwise path requires CUDA 12.9+. Our current environment
has CUDA 12.8. The remote H100 node would need CUDA 12.9 or the PyTorch build must
bundle a sufficiently recent cuBLAS. PyTorch 2.10+cu128 may or may not include the
12.9 cuBLAS bindings -- this needs verification on the actual H100 node.

**SM89 limitation**: BlockWise128x128 is SM90-only. It will not work on our RTX 4090.
The tensorwise mode works on SM89 but is useless for our blockwise-quantized weights
(it would require dequantizing to a single scale, destroying precision).

### Migration path

If `BlockWise128x128` is available on the target H100:

1. **Replace the GEMM call in `FP8Linear.forward()`**: Instead of calling our custom
   `fp8_gemm_blockwise` Triton kernel, call `torch._scaled_mm(x_fp8, weight.T,
   scale_a=x_scale, scale_b=weight_scale)`. The activation quantization
   (`fp8_act_quant`) still runs our Triton kernel -- `_scaled_mm` only replaces the
   matmul itself.

2. **Simplify custom_ops infrastructure**: The four FP8 custom ops in `fp8.py`
   (`futudiffu::fp8_linear`, `fp8_linear_bias`, `fp8_linear_v1t`,
   `fp8_linear_bias_v1t`) exist to make our Triton kernels compatible with
   `torch.compile`. If `_scaled_mm` replaces the kernels, these custom ops become
   unnecessary -- `_scaled_mm` is already a first-class PyTorch op that torch.compile
   understands natively. The `register_fake` and `register_autograd` definitions can
   be removed.

3. **The fused FP8 FFN chain has no _scaled_mm equivalent**: Our
   `fp8_silu_gate_quant -> fp8_gemm_blockwise` fusion skips materializing the BF16
   intermediate between the SiLU-gate activation and the w2 GEMM. `_scaled_mm`
   cannot fuse with an upstream activation function -- it is a standalone matmul op.
   Options:
   - Accept the regression: materialize the BF16 intermediate, quantize it, then
     call `_scaled_mm`. On H100, the bandwidth cost of this intermediate
     (8576 x 10240 x 2 bytes = ~168MB round-trip at 3,350 GB/s = ~0.05ms per layer)
     is much smaller than on SM89 (same at 1,008 GB/s = ~0.17ms). Over 30 layers
     this is 1.5ms vs 5.1ms. Acceptable.
   - Keep the fused Triton kernel for the FFN chain on both architectures. This means
     maintaining two code paths but preserves the fusion benefit.
   - **Recommendation**: Accept the regression on H100. The cuBLAS blockwise GEMM
     will be faster than our Triton GEMM by enough to more than offset the lost
     fusion. Keep the fused path for SM89 only.

4. **Scale tensor format**: Our `weight_scale` tensors have shape
   `(ceil(N/128), ceil(K/128))` in float32. `_scaled_mm` expects the same format for
   BlockWise128x128. Our `x_scale` from `fp8_act_quant` has shape
   `(M, K/128)` which maps to BlockWise 1x128 for the activation side. This is a
   supported combination (mixed blockwise modes for A and B operands). No scale
   tensor reformatting needed.

### Runtime architecture dispatch

**Key design question**: Can the same codebase run optimally on both SM89 and SM90?

Yes. The dispatch is straightforward:

```python
_SM90_PLUS = torch.cuda.get_device_capability()[0] >= 9

class FP8Linear(nn.Module):
    def forward(self, x):
        if _SM90_PLUS and _SCALED_MM_BLOCKWISE_AVAILABLE:
            # cuBLAS blockwise path -- no custom ops needed
            x_fp8, x_scale = fp8_act_quant(x, block_size=self.block_size)
            return torch._scaled_mm(
                x_fp8, self.weight.T,
                scale_a=x_scale, scale_b=self.weight_scale,
                out_dtype=self.output_dtype,
            )
        else:
            # Triton custom kernel path (SM89)
            return fp8_linear_op(x, self.weight, self.weight_scale, ...)
```

The `_SCALED_MM_BLOCKWISE_AVAILABLE` flag should be set at import time by attempting
a small test matmul with blockwise scales and catching any `RuntimeError`. This
handles the case where the PyTorch build lacks the cuBLAS 12.9 bindings.

### Performance expectations

**Educated estimate**: cuBLAS FP8 on H100 should achieve 60-70% of peak 1,979
TFLOPS for our GEMM shapes, vs our Triton kernels achieving ~50-60% on H100 (or
~40% on SM89). This is because:

- cuBLAS uses hand-tuned Hopper PTX with wgmma + TMA, which Triton generates
  automatically but not optimally (Triton's SM90 codegen is improving but not yet
  at cuBLAS quality for all shapes)
- The blockwise scaling is handled in the cuBLAS epilogue without separate passes

Estimated effective throughput: 1,979 * 0.65 = **~1,286 TFLOPS** via `_scaled_mm`
vs ~1,088 TFLOPS from our Triton kernels on H100 (from `remote_training_plan.md`
estimate). This is a **~18% speedup on the GEMM portion**, which at 70% of forward
time translates to **~13% overall forward pass speedup** (from ~235ms to ~205ms per
step).

### Summary

| Aspect | Status |
|--------|--------|
| BlockWise128x128 in _scaled_mm | Merged (PR #158037, July 2025) |
| SM90 requirement | Yes, SM90 only |
| CUDA 12.9 requirement | Yes, needs verification on target node |
| Migration effort | Small (replace GEMM call, simplify custom_ops) |
| Fused FFN chain | No equivalent; accept regression or keep dual path |
| SM89 compatibility | Keep Triton kernels for SM89, dispatch at runtime |
| Estimated speedup on H100 | ~13% overall forward pass |

---

## 2. Triton kernel tuning configs for SM90

### Current SM89 configs

Our Triton GEMM kernels use the following autotune config space:

```python
fp8_gemm_configs = [
    Config({"BLOCK_SIZE_M": block_m}, num_stages=num_stages, num_warps=num_warps)
    for block_m in [64, 128, 256]
    for num_stages in [3, 4, 5]
    for num_warps in [4, 8]
]
```

`BLOCK_SIZE_N` and `BLOCK_SIZE_K` are pinned to `input_block_size` (128) so that
scale indexing is 1:1 with quantization blocks. This means our tiles are always
`M x 128 x 128` where M varies from 64 to 256.

SM89 (Ada Lovelace) has:
- 128 KB shared memory per SM (configurable up to ~164 KB with opt-in via
  `cudaFuncSetAttribute`)
- `mma.sync` instruction (warp-level, 1 warp per MMA)
- No TMA, no wgmma

### SM90 (Hopper) hardware advantages

SM90 has three architectural features that change the optimal tile configs:

**1. 228 KB shared memory per SM (227 KB usable per thread block)**

This is a 39% increase over SM89's maximum 164 KB. Larger shared memory enables:
- More pipeline stages (deeper software pipelining of global -> shared -> register
  transfers)
- Larger tile footprints (wider M or N tiles)
- Both simultaneously

**2. wgmma instruction (warp-group MMA)**

Instead of 1 warp (32 threads) computing a small MMA fragment, wgmma uses a
warp group (4 warps, 128 threads) to compute a larger fragment asynchronously. This
reduces instruction issue overhead per FLOP and enables larger output tiles per
scheduling unit.

Triton automatically uses wgmma when targeting SM90 (via the `sm_90a` target).
No source-level changes needed -- the Triton compiler emits wgmma instructions
for `tl.dot` on SM90.

**3. TMA (Tensor Memory Accelerator)**

Hardware unit for async bulk memory transfers between global and shared memory.
Eliminates address generation overhead and shared memory bank conflicts via automatic
swizzling. Triton is adding TMA support (partially available in Triton 3.x), but
the degree to which Triton's SM90 codegen uses TMA depends on the Triton version.

### What configs become legal/optimal on SM90

With 228 KB shared memory and N=K=128 (pinned to block_size), the shared memory per
tile is approximately:

```
Per-stage shared memory = BLOCK_M * 128 * 1 (FP8 A) + 128 * 128 * 1 (FP8 B)
                        = 128 * BLOCK_M + 16384  bytes per stage
```

For BLOCK_M=128: 128*128 + 16384 = 32768 bytes/stage. At 6 stages: 196 KB. Fits.
For BLOCK_M=256: 256*128 + 16384 = 49152 bytes/stage. At 4 stages: 196 KB. Fits.
For BLOCK_M=256: At 5 stages: 245 KB. Does NOT fit (exceeds 228 KB).
For BLOCK_M=512: 512*128 + 16384 = 81920 bytes/stage. At 2 stages: 164 KB. Fits.

Proposed expanded config space for SM90:

```python
fp8_gemm_configs_sm90 = [
    Config({"BLOCK_SIZE_M": block_m}, num_stages=num_stages, num_warps=num_warps)
    for block_m in [64, 128, 256, 512]
    for num_stages in [3, 4, 5, 6]
    for num_warps in [4, 8, 16]
] + [
    # Extremely deep pipelines for smaller tiles
    Config({"BLOCK_SIZE_M": 128}, num_stages=7, num_warps=8),
    Config({"BLOCK_SIZE_M": 128}, num_stages=8, num_warps=8),
]
```

Key additions over SM89:
- `num_warps=16`: wgmma uses 4 warps per group, so 16 warps = 4 warp groups.
  On SM89 with mma.sync, 16 warps would hurt occupancy. On SM90 with wgmma, 16
  warps is the natural scheduling unit for large tiles.
- `num_stages=6,7,8`: The deeper pipeline hides memory latency better, especially
  with TMA's async transfers. Legal on SM90 for BLOCK_M <= 256.
- `BLOCK_SIZE_M=512`: Very large M tile. Only 2-3 stages fit, but on shapes where
  M is large (8576 for our seq_len), this reduces the number of CTAs and improves
  L2 cache reuse.

### Optimal configs for our specific GEMM shapes

| Operation | Shape (M, K, N) | Likely optimal SM90 config |
|-----------|-----------------|---------------------------|
| QKV projection | (8576, 3840, 11520) | BLOCK_M=256, stages=4, warps=16 |
| FFN w1w3 (fused) | (8576, 3840, 10240) | BLOCK_M=256, stages=4, warps=16 |
| FFN w2 | (8576, 10240, 3840) | BLOCK_M=128, stages=6, warps=8 |
| adaLN (small) | (8576, 3840, 32) | BLOCK_M=128, stages=3, warps=4 |

For the large GEMMs (QKV, FFN), M=8576 is large enough that BLOCK_M=256 gives good
CTA parallelism (8576/256 = ~34 CTAs along M). The large N dimensions (11520, 10240)
with our fixed BLOCK_N=128 give 90 and 80 CTAs along N, respectively. Total grid:
34*90 = 3,060 CTAs for QKV -- enough to saturate H100's 132 SMs easily.

For the FFN w2 (K=10240), the large K dimension means many iterations of the inner
loop (10240/128 = 80 iterations). Deeper pipelining (stages=6) helps overlap the
last iterations' compute with the next iterations' loads.

### Will Triton autotune find these?

**Partially.** Triton autotune will test all configs in the list and pick the fastest
one for each shape. But it can only find configs that are in the list. Our current
list caps at `num_stages=5` and `num_warps=8`, which excludes the best SM90 configs.

**Recommendation**: Add the SM90 configs conditionally:

```python
if torch.cuda.get_device_capability()[0] >= 9:
    fp8_gemm_configs.extend(sm90_extra_configs)
```

This lets autotune explore the SM90-specific space without adding illegal configs
on SM89 (e.g., num_stages=6 with BLOCK_M=256 would exceed SM89's shared memory).

### Dependency on topic 1

If `torch._scaled_mm` replaces our Triton GEMM kernels on SM90 (topic 1), then SM90
tuning configs become moot -- we would not be running our own kernels on SM90 at all.

The SM90 configs are only relevant if:
- `_scaled_mm` blockwise is not available (CUDA version too old)
- We keep the Triton path for the fused FFN chain
- We want a fallback if `_scaled_mm` regresses on specific shapes

**Recommendation**: Add the configs anyway (low effort, ~10 lines of code). They
serve as a fallback and help the fused FFN chain if we keep it.

---

## 3. Batch size scaling and gradient checkpointing tradeoffs

### Current state on RTX 4090 (24 GB)

From `training_utils.py` and memory measurements:

| Component | VRAM |
|-----------|------|
| FP8 diffusion model | ~5.8 GB |
| LoRA adapters (rtheta + ptheta) | ~7 MB |
| BTRM head + optimizer | ~0.5 GB |
| Baseline total | ~6.3 GB |
| Remaining for activations | ~17.7 GB |
| Backward overhead (checkpointed) | +0.20 GB |

Gradient checkpointing (`forward_checkpointed`) is **necessary** on SM89:
- Without checkpointing: all 30 blocks' activations stored simultaneously
- With checkpointing: only 1 block's activations at a time, each block recomputed
  during backward

### Per-block activation memory estimate

For a single NextDiT block at B=2, seq_len=8576, dim=3840:

**Attention activations** (kept for backward if not checkpointed):
- QKV projections: 3 * B * seq * dim = 3 * 2 * 8576 * 3840 * 2 bytes = ~377 MB
- Attention scores: B * n_heads * seq * seq = 2 * 30 * 8576 * 8576 * 2 bytes
  = ~8.4 GB (this is the killer -- quadratic in seq_len)

Wait -- SDPA with `torch.nn.functional.scaled_dot_product_attention` uses FlashAttention
under the hood, which does NOT materialize the full attention matrix. FlashAttention
stores only the LSE (log-sum-exp) values: B * n_heads * seq = 2 * 30 * 8576 * 4 bytes
= ~2 MB. The backward recomputes attention blocks on the fly.

**Corrected per-block activation estimate** (with FlashAttention / SDPA):
- Input tensor (saved for residual): B * seq * dim * 2 = 2 * 8576 * 3840 * 2 = ~126 MB
- QKV projections output: ~377 MB (need Q, K, V for backward of attention)
- Attention output: B * seq * dim * 2 = ~126 MB
- FFN intermediates (w1, w3, gate activation): ~3 * 2 * 8576 * 10240 * 2 = ~1.0 GB
- adaLN modulation cache: small (~1 MB)
- LSE from SDPA: ~2 MB
- **Total per block: ~1.6 GB** (educated estimate)

For 30 blocks without checkpointing: **~48 GB**

This explains why checkpointing is mandatory on the 4090 (17.7 GB available, 48 GB
needed). And it informs the H100 analysis.

### H100 (80 GB) activation budget

| Component | VRAM |
|-----------|------|
| FP8 diffusion model | ~5.8 GB |
| LoRA + BTRM + optimizer | ~0.5 GB |
| TE (if co-resident, see topic 4) | ~7.5 GB |
| VAE (co-resident) | ~0.16 GB |
| Baseline total | ~14.0 GB |
| Remaining for activations | ~66 GB |

### Can we skip gradient checkpointing on H100?

At B=2 (CFG batch): 30 blocks * ~1.6 GB/block = **~48 GB**. This fits in 66 GB
with 18 GB to spare.

**Yes, gradient checkpointing can be skipped on H100 at B=2.**

The savings from skipping checkpointing:

- **FLOPs saved**: Each of the 30 main blocks is forward-computed twice during
  training with checkpointing (once in the forward pass, once in the backward pass
  during recomputation). Without checkpointing, each block runs forward only once.
  This saves 30 block forwards per training step.

- **Time saved per training step** (educated estimate): Each block's forward is
  dominated by 3 FP8 GEMMs (QKV, w1w3, w2) totaling ~15ms per block on H100
  (from the ~235ms/30 blocks in `remote_training_plan.md`, accounting for compute
  overlap with backward). Saving 30 forwards = **~450ms per backward pass**
  (but not all of this is on the critical path due to backward/forward overlap in
  checkpointing). Realistic wall-clock savings: **~200-350ms per training step**.
  Over 50 policy iterations with 5 sparse gradient steps each: ~250 steps *
  ~275ms = **~69 seconds** saved per training run.

- **Implementation**: Add a flag to `forward_checkpointed` (or a separate
  `forward_training` function) that stores all activations instead of using
  `torch.utils.checkpoint.checkpoint`. The logic is simple -- replace the
  `grad_ckpt(layer, ...)` call with a plain `layer(...)` call.

### Batch size scaling on H100

Without checkpointing, activation memory scales linearly with batch size:

| Batch size | Activations (30 blocks) | Total VRAM | Fits in 80 GB? |
|-----------|------------------------|-----------|----------------|
| B=2 (current, pos+neg CFG) | ~48 GB | ~62 GB | Yes |
| B=4 (2 rollouts batched) | ~96 GB | ~110 GB | No |
| B=4 with checkpointing | ~3.2 GB | ~17.2 GB | Yes |
| B=6 with checkpointing | ~4.8 GB | ~18.8 GB | Yes |
| B=8 with checkpointing | ~6.4 GB | ~20.4 GB | Yes |

**Key insight**: B=2 without checkpointing is the sweet spot on H100 for training.
Higher batch sizes require re-enabling checkpointing, which negates part of the
FLOPS savings.

However, the batch dimension here has a specific meaning:
- B=2 is pos+neg CFG (not two separate images)
- For policy training, each "rollout" requires B=2 (CFG). Multiple rollouts are
  sequential, not batched, because each rollout needs its own gradient accumulation
  with its own advantage weight.
- For trajectory GENERATION (inference only, no backward), there is no activation
  storage constraint. Inference at B=4, B=8, or even B=16 is feasible:

| Inference batch | Peak VRAM (forward only) | Fits in 80 GB? |
|----------------|-------------------------|----------------|
| B=2 (1 image, CFG) | ~3.2 GB activations + 14 GB model = ~17 GB | Yes |
| B=4 (2 images, CFG) | ~6.4 GB + 14 GB = ~20 GB | Yes |
| B=8 (4 images, CFG) | ~12.8 GB + 14 GB = ~27 GB | Yes |
| B=16 (8 images, CFG) | ~25.6 GB + 14 GB = ~40 GB | Yes |

With FlexAttention batch packing (see `flexattention_batch_packing.md`), B=8 or
B=16 CFG batches would be the practical inference target on H100. This is **4-8x**
more images per forward pass than the current B=2.

### Does H100 saturate at hardware batch size 1?

On SM89, we measured that our GEMMs hit ~40% utilization at B=2, and cuBLAS hit the
same ceiling. The utilization was not batch-dependent -- SM89 saturates its tensor
cores at B=1 for our GEMM shapes because M=8576 already provides enough work to
fill the SMs.

**Empirical question**: Does SM90 also saturate at B=1?

Educated estimate: **Probably yes for our shapes, but with a nuance.** H100 has 132
SMs (vs 128 on 4090). Our QKV GEMM has M=8576, N=11520, BLOCK_M=256, BLOCK_N=128,
giving 34 * 90 = 3,060 CTAs. At 132 SMs, that is ~23 waves -- more than enough to
saturate. Even at B=1 (M=4288), it would be ~17 * 90 = 1,530 CTAs, ~12 waves.
Still saturated.

Where higher batch size helps on H100:
- **Kernel launch amortization**: More work per launch = less launch overhead as a
  fraction. With torch.compile CUDA graph replay, this is already minimal, but
  for inference without compile warmup, it matters.
- **Memory access coalescing**: Larger batches improve global memory access patterns
  for the attention kernel (more heads to schedule concurrently).
- **FlexAttention packing efficiency**: Packing 4+ images into one sequence reduces
  padding waste and gives FlexAttention a longer packed sequence to tile over.

### Summary of tradeoffs

| Scenario | Checkpointing? | Batch size | VRAM on H100 | FLOPS saved vs 4090 config |
|----------|---------------|-----------|-------------|---------------------------|
| Training (policy backward) | No | B=2 | ~62 GB | ~30 block recomputes/step |
| Training (higher batch) | Yes | B=4-8 | ~17-20 GB | None (same as 4090 pattern) |
| Inference (traj generation) | N/A | B=8-16 | ~27-40 GB | N/A (inference only) |

**Recommendation**: Default to no-checkpointing at B=2 for training on H100. Add a
`--gradient-checkpointing` flag to `forward_checkpointed` that can be forced on for
larger batches or smaller GPUs. Auto-detect based on `torch.cuda.mem_get_info()` at
startup.

**Empirical questions** (need H100 benchmarking):
- Exact per-block activation memory (the 1.6 GB estimate needs measurement)
- Whether B=2 no-checkpointing actually fits with all three models co-resident
- Throughput scaling from B=2 to B=8 for inference (is it linear?)
- Whether wgmma achieves better utilization at B=1 vs B=2

---

## 4. Model offloading strategy on H100

### Current lifecycle on RTX 4090 (24 GB)

From `server.py`, the server has a mutual-exclusion model lifecycle:

```
Phase "te":        TE loaded (~7.5 GB),  diffusion NOT loaded
Phase "diffusion": Diffusion loaded (~5.8 GB + compile overhead), TE NOT loaded
Phase "vae":       VAE loaded (~0.16 GB), can coexist with diffusion
Phase None:        Nothing loaded
```

The `_ensure_te()` method frees the diffusion model before loading TE. The
`_ensure_diffusion()` method frees TE before loading diffusion. Each transition
involves:

1. `_snapshot_lora_weights()` -- copy LoRA state to CPU (~7 MB)
2. `del self._diff_model` -- free GPU memory
3. `torch.cuda.empty_cache()` -- return memory to CUDA allocator
4. Load new model from safetensors
5. `torch.compile()` -- recompile for the new model
6. Replay LoRA injections from saved configs
7. Restore LoRA weights from CPU snapshots

This lifecycle is **necessary on 24 GB** because TE (7.5 GB) + diffusion (5.8 GB)
= 13.3 GB, leaving only 10.7 GB for activations and compile overhead. With compile
overhead (~2 GB), actual remaining is ~8.7 GB -- tight even for inference at B=2.

### H100: all models fit simultaneously

| Component | VRAM |
|-----------|------|
| Text encoder (BF16) | 7.5 GB |
| Diffusion model (FP8) | 5.8 GB |
| VAE | 0.16 GB |
| LoRA adapters | ~7 MB |
| BTRM head + optimizer | ~0.5 GB |
| torch.compile overhead | ~2-3 GB |
| **Total model state** | **~16 GB** |
| Remaining (80 GB) | **~64 GB** |

All three models fit with 64 GB to spare. No offloading needed.

### What changes in server.py

**New "all_loaded" mode**: When VRAM is sufficient, skip all lifecycle transitions.

Concrete changes to `server.py`:

1. **Add `--no-offload` flag** (or auto-detect via `torch.cuda.mem_get_info()`):

   ```python
   def __init__(self, ..., no_offload: bool = False):
       self._no_offload = no_offload
   ```

2. **Modify `_ensure_te()`**: When `_no_offload` is True, skip freeing diffusion:

   ```python
   def _ensure_te(self):
       if self._te_model is not None:
           return
       # In no-offload mode, do NOT free diffusion -- both fit
       if not self._no_offload and self._diff_model is not None:
           self._snapshot_lora_weights()
           del self._diff_model, ...
           torch.cuda.empty_cache()
       # Load TE (no-offload: alongside diffusion; normal: alone)
       self._te_model = load_text_encoder(...)
       self._te_model = torch.compile(self._te_model, mode="default")
   ```

3. **Modify `_ensure_diffusion()`**: When `_no_offload` is True, skip freeing TE:

   ```python
   def _ensure_diffusion(self):
       if self._diff_model is not None:
           return
       if not self._no_offload:
           self._free_te()
       # Load diffusion (TE stays resident in no-offload mode)
       ...
   ```

4. **Eliminate LoRA weight snapshot/replay**: In no-offload mode, the diffusion
   model is never destroyed, so LoRA weights never need to be saved to CPU and
   replayed. The entire `_snapshot_lora_weights()` / replay mechanism in
   `_ensure_diffusion()` is dead code. This also eliminates the risk of weight
   precision loss during CPU round-trips (the CPU snapshot stores in model dtype,
   but the round-trip through `detach().cpu()` and `v.to(dtype=self.dtype)` is
   a potential source of subtle bugs).

5. **Startup becomes "load everything once"**:

   ```python
   def startup(self):
       if self._no_offload:
           self._ensure_te()
           self._ensure_diffusion()
           self._ensure_vae()
           # All models now resident. Phase tracking is vestigial.
           self._phase = "all_loaded"
       else:
           # Existing lazy-load behavior
           pass
   ```

6. **Text encoding at any time**: With TE always resident, `handle_encode_prompt`
   no longer triggers a lifecycle transition. No diffusion model unload/reload.
   No torch.compile re-warmup. The RPC is just tokenize -> forward -> return.

7. **VAE decoding at any time**: Similarly, `handle_vae_decode` always has the VAE
   available. No lifecycle delay.

### Time saved by not offloading

**Measured** (on RTX 4090): Each diffusion model load + compile + LoRA replay takes
~8-12 seconds. TE load + compile takes ~3-5 seconds. VAE load is ~1 second.

In a typical training session:
- Initial TE load for prompt encoding: 1 transition
- Switch to diffusion for trajectory generation: 1 transition
- If periodic eval renders need VAE decode + TE re-encode: 2-4 transitions
- Each policy iteration that needs fresh prompt encoding: N transitions

For a 50-iteration training run with eval renders every 10 iterations:
- ~5 eval render cycles * 2 transitions each = 10 transitions
- ~10 transitions * ~10 seconds each = **~100 seconds of lifecycle overhead**

On H100 with no-offload: **0 seconds**. All RPCs dispatch immediately to
always-resident models.

Additionally, torch.compile warmup happens only once per model (at startup) rather
than being re-triggered after each lifecycle reload. On H100, compile warmup may
take ~45-60 seconds per model on the first call. In no-offload mode, this is a
fixed one-time cost. In offload mode, it recurs every time the diffusion model is
reloaded.

### Auto-detection

Rather than requiring a manual `--no-offload` flag, the server can auto-detect:

```python
def _should_offload(self) -> bool:
    """Determine if model offloading is needed based on available VRAM."""
    total_model_memory = 16_000_000_000  # ~16 GB for all models + compile
    min_activation_headroom = 20_000_000_000  # ~20 GB for training activations
    total_mem = torch.cuda.get_device_properties(0).total_mem
    return total_mem < (total_model_memory + min_activation_headroom)
```

On a 24 GB GPU: 24 < 36 -> offload. On an 80 GB GPU: 80 > 36 -> no offload.

### Impact on multi-GPU scaling

From `multi_gpu_scaling.md`: "On an H100, both fit simultaneously (7.5GB TE + 8GB
diffusion + activations = ~20GB total out of 80GB available)." This was already
noted as an advantage. The no-offload mode formalizes it.

For the `MultiGPUClient` pattern where server 0 encodes all prompts: in no-offload
mode, server 0 can encode prompts AND generate trajectories without any lifecycle
transition. Other servers only need the diffusion model, but in no-offload mode they
too load all three models at startup (the TE and VAE are small enough that keeping
them resident has no meaningful cost).

### Summary

| Aspect | RTX 4090 (offload) | H100 (no-offload) |
|--------|-------------------|-------------------|
| Models in VRAM | 1 at a time | All 3 simultaneously |
| Lifecycle transitions | ~10-15 per training run | 0 |
| Time in transitions | ~100-150 seconds | 0 seconds |
| torch.compile warmups | Per-reload (multiple) | Once at startup |
| LoRA weight snapshots | Required (CPU round-trip) | Not needed |
| Text encoding | Requires lifecycle swap | Immediate |
| VAE decoding | Requires lifecycle swap | Immediate |
| Implementation effort | Existing code | ~50 lines of server.py changes |

---

## Cross-cutting considerations

### Interaction between the four optimizations

The four opportunities are largely independent but have some interactions:

1. If `_scaled_mm` replaces Triton GEMMs (topic 1), then Triton SM90 tuning
   (topic 2) is only relevant for the fused FFN chain and as a fallback.

2. No-offload mode (topic 4) increases baseline VRAM by ~7.5 GB (TE stays resident),
   which reduces the headroom for no-checkpointing (topic 3) from 66 GB to 58.5 GB.
   At ~48 GB for B=2 activations, this still fits with 10 GB margin.

3. Higher inference batch sizes (topic 3) benefit more from no-offload (topic 4)
   because they can encode prompts and generate trajectories without any lifecycle
   delay, enabling tighter pipelining.

### What can be estimated vs. what needs benchmarking

| Question | Estimable? | Method |
|----------|-----------|--------|
| Does _scaled_mm blockwise work on the target node? | No | Must test on actual H100 |
| _scaled_mm vs Triton GEMM speedup | Partially | Need H100 benchmark |
| Per-block activation memory | Partially | Need H100 measurement (torch.cuda.memory_allocated) |
| B=2 no-checkpointing fits in 66 GB | Probably yes | Estimate says 48 GB; need measurement |
| Offloading time savings | Yes | Already measured lifecycle overhead on 4090 |
| Optimal Triton SM90 tile configs | No | Triton autotune on actual H100 |
| Inference throughput at B=8-16 | No | Need H100 benchmark |

### Priority order for implementation

1. **No-offload mode** (topic 4): Highest ROI, lowest risk, ~50 lines. Saves ~100s
   per training run immediately, simplifies the codebase, eliminates the LoRA
   snapshot/replay complexity for H100 deployments.

2. **No-checkpointing mode** (topic 3): Medium ROI, low risk, ~20 lines. Saves ~69s
   per training run. Requires verifying activation memory fits; easy to fall back to
   checkpointing if it does not.

3. **torch._scaled_mm dispatch** (topic 1): High ROI if available, medium effort
   (~100 lines). The ~13% forward speedup compounds across all 30 steps and all
   iterations. But requires CUDA 12.9 verification and has the fused FFN chain
   question.

4. **Triton SM90 configs** (topic 2): Low effort (~10 lines), but lowest ROI if
   topic 1 succeeds. Add the configs speculatively; they cost nothing and help the
   fallback path.

---

## Sources

- [PyTorch issue #153555: Updated Scaled_mm to support more scaling formats via CuBlas](https://github.com/pytorch/pytorch/issues/153555)
- [PyTorch PR #158037: DeepSeek-style blockwise scaling for _scaled_mm](https://github.com/pytorch/pytorch/pull/158037)
- [PyTorch issue #147971: FP8 scaled-mm row-wise is substantially slower than tensor-wise](https://github.com/pytorch/pytorch/issues/147971)
- [PyTorch PR #145728: Limit f8f8bf16 rowwise scaled matmul to SM90](https://github.com/pytorch/pytorch/pull/145728)
- [Scaled MM API design notes (drisspg gist)](https://gist.github.com/drisspg/783616821043ab4594b9784f556c6714)
- [PyTorch blog: Accelerating Llama3 FP8 Inference with Triton Kernels (TK-GEMM)](https://pytorch.org/blog/accelerating-llama3/)
- [PyTorch blog: Deep Dive on the Hopper TMA Unit for FP8 GEMMs](https://pytorch.org/blog/hopper-tma-unit/)
- [NVIDIA Hopper Tuning Guide (shared memory: 228 KB per SM)](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [SM90 Hopper Architecture Features (CUTLASS DeepWiki)](https://deepwiki.com/NVIDIA/cutlass/7.1-sm90-hopper-architecture)
- [Colfax: CUTLASS Tutorial on WGMMA for Hopper GPUs](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)
- [Triton issue #2339: Matmul performance on H100](https://github.com/openai/triton/issues/2339)
- [DeepGEMM: clean and efficient FP8 GEMM kernels with fine-grained scaling](https://github.com/deepseek-ai/DeepGEMM)
- [Jianyu Huang: CUDA H100 GEMM Optimization](https://jianyuh.github.io/gemm/optimization/hopper/2024/12/29/h100_gemm.html)
- [Hamza's Blog: Optimising GEMM on H100 for cuBLAS-like Performance](https://hamzaelshafie.bearblog.dev/worklog-optimising-gemm-on-nvidia-h100-for-cublas-like-performance-wip/)

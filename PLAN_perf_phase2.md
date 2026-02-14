# Performance Phase 2: Deep Kernel & Pipeline Optimizations

## Context

Phase 1 fusions (w1w3, elementwise, FP8 chain) took 832ms → 768ms (-7.7%).
Utilization is 12.4% (95ms theoretical / 768ms actual). The remaining 673ms
gap breaks down as:

| Source | Estimated ms | Root cause |
|--------|-------------|------------|
| FP8 GEMM at 38% peak | ~360 | Uncoalesced B loads, rigid tiles, autotuner gaps |
| Python dispatch overhead | ~100 | @torch.compiler.disable blocks CUDA graphs |
| SDPA attention | ~193 | Already addressed by SageAttention (separate) |
| Bandwidth-bound serialization | ~60 | 13 b/w-bound ops per block, tensor cores idle |
| Misc (embeddings, final layer) | ~20 | Small, not worth targeting |

This plan targets the first four rows.

---

## Task A: FP8 GEMM Kernel Overhaul

**Files:** `fp8_kernels.py` (new kernel functions alongside existing ones)
**Impact:** ~150ms (20% of 768ms)
**Risk:** Medium

### A1: Weight pre-transpose [N,K] → [K,N]

The GEMM computes C = A @ B^T where B is stored [N, K]. The kernel loads
(BLOCK_K, BLOCK_N) tiles of B via:

```python
b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
```

Consecutive warp lanes access addresses with stride K (3840+ bytes apart),
causing ~3% cache line utilization on B loads. Pre-transposing B to [K, N]
at model load time makes consecutive N-addresses consecutive in memory →
fully coalesced loads.

**Changes:**
- New kernel `fp8_gemm_blockwise_v2_kernel` that accepts B in [K, N] layout
- B tile loading: `b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]`
- Scale layout: `b_s` stored as [K//bs, N//bs] (transposed from [N//bs, K//bs])
- Python wrapper `fp8_gemm_blockwise_v2()` with `weight_transposed=True` flag
- `FP8Linear.__init__` transposes weight and scale at construction time

### A2: Fused activation quantization in GEMM prologue

Currently: `fp8_act_quant(x)` → write FP8+scales to GMEM → GEMM reads FP8.
This is a full round-trip for quantized activations (132MB per QKV GEMM).

Fuse: GEMM kernel loads BF16 A tiles directly, quantizes in-register:
```python
a_bf16 = tl.load(a_ptr + ...).to(tl.float32)
amax = tl.max(tl.abs(a_bf16), axis=1)
scale = tl.maximum(amax / FP8_MAX, 1e-12)
a_fp8 = (a_bf16 * (FP8_MAX / tl.maximum(amax[:, None], 1e-12))).clamp(...)
```

The new kernel signature accepts BF16 input directly (no pre-quantized FP8).
This eliminates 170 act_quant kernel launches per forward (5 per block × 34).

### A3: Enlarged tile autotune configs

Currently pinned: BLOCK_N=128, BLOCK_K=128 (= input_block_size).
Decouple: allow BLOCK_N ∈ {128, 256} and BLOCK_K ∈ {128, 256} with
multi-block scale indexing:

```python
for k_sub in range(BLOCK_SIZE_K // input_block_size):
    a_s_sub = tl.load(a_s_ptr + k_scale_base + k_sub)
    b_s_sub = tl.load(b_s_ptr + ...)
    sub_accumulator += partial * a_s_sub[:, None] * b_s_sub[None, :]
```

Add configs: BLOCK_M ∈ {64, 128, 256}, BLOCK_N ∈ {128, 256},
BLOCK_K ∈ {128, 256}, num_stages ∈ {2, 3, 4}, num_warps ∈ {4, 8, 16}.

### Deliverable

New kernel functions in fp8_kernels.py (v2 variants) + standalone test
comparing output against existing kernels. Old kernels remain for
backwards compatibility until v2 is validated.

---

## Task B: Fused QKV Post-Process Kernel

**Files:** `fused_kernels.py` (add kernel), `diffusion_model.py` (wire in)
**Impact:** ~11ms (1.4%)
**Risk:** Medium

After QKV GEMM produces (B*seq, 3*dim) in BF16:
1. split into Q, K, V (view, free)
2. reshape to (B, seq, heads, head_dim) (view, free)
3. q_norm: RMSNorm per head_dim=128 (bandwidth-bound, full read+write)
4. k_norm: RMSNorm per head_dim=128 (bandwidth-bound, full read+write)
5. apply_rope_flux: 2x2 rotation (bandwidth-bound, full read+write)
6. 3× movedim(1,2) → implicit copy to (B, heads, seq, head_dim)

These 6 operations do ~6 full passes over ~66MB each = ~400MB GMEM traffic.

**Fused kernel:** One Triton kernel that:
1. Reads one row of QKV output (11520 elements = 3×3840 = 3×30×128)
2. For each head h in 0..29:
   - Extract Q[h], K[h] (128 elements each) from the correct offsets
   - Compute RMSNorm in-register (128-element reduction, trivially fits)
   - Apply Flux RoPE 2x2 rotation in-register (load freqs_cis for this head)
   - Extract V[h] (128 elements, no norm needed)
3. Writes Q, K, V directly in (B, heads, seq, head_dim) layout

Grid: (B * seq,) — one program per token position.
BLOCK_DIM: 128 (head_dim, fits in one tile).

This kernel needs to know n_heads=30 and head_dim=128 as constexprs, plus
the split offsets for Q/K/V within the 11520-dim output.

### Deliverable

Kernel + Python wrapper in fused_kernels.py. Standalone test comparing
output against the unfused JointAttention path. Does NOT modify
JointAttention — integration wired separately.

---

## Task C: custom_op for FP8Linear

**Files:** `fp8.py`
**Impact:** ~100ms (13%)
**Risk:** Low (proven pattern from sage_attention.py and lora_kernels.py)

Register FP8Linear's forward as `torch.library.custom_op("futudiffu::fp8_linear")`.
The compiler treats it as an opaque node — no graph break, CUDA graph
capture succeeds, Python dispatch overhead eliminated for the 30-step loop.

**Pattern (from sage_attention.py:386-412):**
```python
@torch.library.custom_op("futudiffu::fp8_linear", mutates_args=())
def fp8_linear_op(x, weight, weight_scale, bias_or_dummy, block_size, has_bias):
    # ... calls fp8_act_quant + fp8_gemm_blockwise ...
    return out

@fp8_linear_op.register_fake
def _fp8_linear_fake(x, weight, weight_scale, bias_or_dummy, block_size, has_bias):
    N = weight.shape[0]
    return x.new_empty(*x.shape[:-1], N, dtype=torch.bfloat16)
```

Then `FP8Linear.forward` drops `@torch.compiler.disable` and calls
`fp8_linear_op(...)`.

Note: custom_op args must be Tensor or primitive types (no torch.dtype).
Encode output_dtype as int and decode inside the op.

### Deliverable

Modified fp8.py with custom_op registration. Test: wrap model in
torch.compile(mode="reduce-overhead"), verify no graph breaks, verify
output correctness.

---

## Task D: Pre-batched adaLN Computation

**Files:** `diffusion_model.py`
**Impact:** ~3ms (0.4%) — small but eliminates 31 kernel launches
**Risk:** Low (pure Python refactor)

All 32 modulated blocks (2 noise_refiner + 30 main) compute
`self.adaLN_modulation(adaln_input)` independently using the SAME
`adaln_input` (timestep embedding). Each is a (2, 256) @ (256, 15360) GEMM
— tiny, bandwidth-bound, and serialized.

**Pre-batch:** In NextDiT.forward, before the layer loop:
1. Collect all 32 adaLN weight matrices into one (32*15360, 256) tensor
2. One GEMM: F.linear(adaln_input, W_cat, B_cat) → (batch, 32*15360)
3. Split into 32 × (batch, 15360)
4. Chunk each into (scale_msa, gate_msa, scale_mlp, gate_mlp)
5. Pass pre-computed params to each block's forward()

**JointTransformerBlock.forward signature change:**
```python
def forward(self, x, x_mask, freqs_cis,
            adaln_input=None,
            precomputed_adaln=None):  # NEW: (scale_msa, gate_msa, scale_mlp, gate_mlp)
    if precomputed_adaln is not None:
        scale_msa, gate_msa, scale_mlp, gate_mlp = precomputed_adaln
    elif self.modulation:
        scale_msa, gate_msa, scale_mlp, gate_mlp = (
            self.adaLN_modulation(adaln_input).chunk(4, dim=1)
        )
```

The weight matrix stacking should be done ONCE in a `prepare_adaln_cache()`
method, not per-forward.

### Deliverable

Modified NextDiT with `prepare_adaln_cache()` + pre-batched forward path.
Test: compare per-layer params against sequential computation (must be
bitwise identical — it's the same linear algebra, just batched).

---

## Task E: Persistent FP8 GEMM (Stretch Goal)

**Files:** `fp8_kernels.py`
**Impact:** ~30ms (3.9%)
**Risk:** High

Launch exactly 128 programs (one per SM). Each program loops over its
assigned tiles via `for tile_idx in range(pid, total_tiles, NUM_SMS)`.
Benefits: zero scheduling overhead, explicit L2 tile ordering, better
occupancy across waves.

**Deferred to Phase 3** — depends on Task A completing first.

---

## Dependency Graph

```
Task A (GEMM kernel)  ───────────────────────────┐
Task B (QKV postprocess)  ──────────────────────┐ │
Task C (custom_op)  ────────────────────────────┐│ │
Task D (batched adaLN)  ───────────────────────┐││ │
                                                ││││
                                         Integration + Benchmark
                                                │
                                         Task E (Persistent GEMM)
```

Tasks A, B, C, D are fully independent — different files, no shared state.
Integration reconciles them into the codebase. Task E depends on A.

## Results

### Benchmark (bench_fusion.py, RTX 4090, BF16, batch=2 CFG)

| Config | Time (ms) | vs Baseline | Delta |
|--------|-----------|-------------|-------|
| Pre-fusion (historical) | ~832 | — | — |
| w1w3 only (baseline) | 851.4 | — | — |
| Phase 1 (elem+chain) | 792.8 | -6.9% | -58.6ms |
| Phase 2 (+ adaLN + QKV) | 705.8 | -17.1% | -145.6ms |
| + torch.compile | 720.4 | -15.4% | -131.0ms |

### Findings

1. **Task A (GEMM overhaul): NO BENEFIT.** FP8 GEMMs are compute-bound at 250-295
   TFLOPS (38-45% peak). Weight pre-transpose [K,N] for coalesced access was 0.73-0.78x
   SLOWER — pipeline latency hiding already masks the B load inefficiency, and the
   simpler v1 kernel body wins on register pressure.

2. **Task B (fused QKV postprocess): ~45ms savings.** Single Triton kernel replaces
   split + 2x RMSNorm + RoPE + 3x movedim. 6.5x speedup on the QKV post-process
   portion alone.

3. **Task C (custom_op for FP8Linear): Enables torch.compile but net negative.**
   The `@torch.compiler.disable` on fused elementwise/QKV kernels causes graph breaks
   that hurt more than CUDA graph capture helps. Net: +14.7ms SLOWER with compile.
   The custom_ops work correctly and will become valuable once fused kernels are also
   wrapped as custom_ops.

4. **Task D (pre-batched adaLN): ~42ms savings.** 32 tiny GEMMs → 1 batched GEMM.
   Required FP8 weight dequantization since adaLN weights are FP8-quantized in the
   model file.

5. **Task E (persistent GEMM): Deferred.** GEMM is compute-bound, not scheduling-bound.

### Additional fixes during integration

- `prepare_adaln_cache()` now dequantizes FP8Linear weights via `dequantize_fp8_blockwise()`
- Fused elementwise wrappers need `@torch.compiler.disable` to prevent Inductor from
  trying to compile `tl_libdevice.tanh` references
- `_forward_fused_chain()` dispatches to `fp8_gemm_v1t` when w2 is transposed

## SM89 FP8 GEMM Utilization Analysis

Benchmarked our Triton kernel vs cuBLAS (torch._scaled_mm) on the 4 actual GEMM shapes
(bench_fp8_gemm.py, RTX 4090, FP8 E4M3, median of 20, CUDA events):

| Shape | Triton (TFLOPS) | %peak | cuBLAS (TFLOPS) | %peak | cuBLAS-fast | %peak | T/cB |
|-------|----------------|-------|-----------------|-------|-------------|-------|------|
| QKV (8576, 3840, 11520) | 265.7 | 40.3% | 291.9 | 44.2% | 265.3 | 40.2% | 1.10 |
| Out (8576, 3840, 3840)  | 288.5 | 43.7% | 299.7 | 45.4% | 283.9 | 43.0% | 1.04 |
| w1w3 (8576, 3840, 20480)| 249.8 | 37.8% | 260.2 | 39.4% | 268.5 | 40.7% | 1.08 |
| w2 (8576, 10240, 3840)  | 238.5 | 36.1% | 248.2 | 37.6% | 250.8 | 38.0% | 1.05 |

**Conclusion**: ~40% utilization is inherent to SM89 for FP8 GEMMs at these sizes.
cuBLAS with hand-tuned PTX hits the same ceiling (37-45%). Our Triton kernel is within
4-10% of cuBLAS. No meaningful kernel-level improvement possible.

## SageAttention Variant Analysis

Benchmarked all sage variants on actual diffusion model shapes (bench_sage_pv.py,
B=2, H=30, N=4288, D=128, CUDA events):

| Variant | ms/call | x34 total | vs SDPA | cos_sim |
|---------|---------|-----------|---------|---------|
| SDPA (BF16 baseline) | 6.543 | 222.5ms | 1.00x | 1.000000 |
| FP8 QK + BF16 PV | 5.072 | 172.4ms | 1.29x | 0.999289 |
| INT8 QK + BF16 PV | 4.844 | 164.7ms | 1.35x | 0.999746 |
| FP8 QK + FP8 PV | 7.050 | 239.7ms | 0.93x | 0.998759 |

**FP8 PV is DEAD**: 8% slower than SDPA. Per-column V quant + per-row P quant overhead
exceeds FP8 tensor core throughput gain at BLOCK_N=64. INT8 QK + BF16 PV is the
best variant (fastest AND most accurate).

## Overall Performance Summary

| Config | ms/NFE | vs pre-fusion |
|--------|--------|---------------|
| Pre-fusion (historical) | ~832 | — |
| Phase 2 compiled (SDPA) | 707 | -15.0% |
| Phase 2 compiled + INT8 sage | ~649 | -22.0% |

## Verification

After integration, run bench_fusion.py (extended) to compare:
- Pre-phase-2 baseline: 768ms
- Post-phase-2 actual: 706ms (-8.1% from phase 1, -17.1% from baseline)
- Correctness: not yet re-validated (should re-run generate.py to verify output)

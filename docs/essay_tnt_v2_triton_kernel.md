# TNT v2: Triton Kernel and Compilation for Grid Laplacian Matvec

## What Changed

The graph-based structural similarity pipeline in `src_ii/itten_cuter_grops.py`
was optimized via three complementary strategies:

1. **Triton kernel for grid Laplacian matvec** -- a fused stencil kernel that
   computes `L @ v` in a single launch, replacing 5 separate PyTorch kernel
   launches (degree multiply + 4 roll-multiply-subtract operations).

2. **torch.compile on Lanczos and connected components loops** -- the compiled
   graphs fuse all per-iteration kernel launches, eliminating Python loop
   overhead and CUDA launch latency across 30 Lanczos iterations and 15 CC
   iterations.

3. **Removal of CUDA stream parallelism** -- the old implementation ran image
   and reference pipelines on separate CUDA streams. With compiled graphs that
   saturate the GPU, stream management overhead exceeded any overlap benefit.
   Sequential execution on the default stream is 30% faster.

## Why Sparse Matvec Was Wrong for Grid Graphs

A sparse COO Laplacian on a 512x512 image has H*W = 262,144 rows and ~4*H*W =
1,048,576 nonzeros. This is a 262K x 262K matrix with 99.998% sparsity. The
`torch.sparse.mm` path constructs this matrix (COO indices + values), then
dispatches to cuSPARSE for each matvec.

But a regular image grid has uniform degree 4 (2-3 at boundaries), implicit
topology (neighbor at position (y,x) is always at (y+-1, x) or (y, x+-1)), and
dense contiguous edge weights. The sparse format adds:

- 2 * nnz int64 indices = 16 MB just for the COO coordinates
- cuSPARSE overhead for CSR conversion, load balancing, etc.
- Random access patterns from indirect addressing through index arrays

The dense stencil approach stores only 4 * H * W float32 edge weights (4 MB)
and uses regular strided access. The topology is implicit in the flat index
arithmetic: neighbor up = `idx - W`, neighbor down = `idx + W`, etc.

## The Triton Kernel

```python
@triton.jit
def _grid_laplacian_matvec_kernel(
    V_ptr, Out_ptr,
    W_up_ptr, W_down_ptr, W_left_ptr, W_right_ptr, Deg_ptr,
    H: tl.constexpr, W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < H * W

    v = tl.load(V_ptr + offs, mask=mask, other=0.0)
    deg = tl.load(Deg_ptr + offs, mask=mask, other=0.0)
    # ... load 4 edge weights ...

    # Neighbor indices: clamp to valid range, zero weights kill boundary values
    idx_up = tl.maximum(offs - W, zero)
    idx_down = tl.minimum(offs + W, max_idx)
    # ... left, right ...

    result = deg * v - w_up * v_up - w_down * v_down - ...
    tl.store(Out_ptr + offs, result, mask=mask)
```

Each program handles 1024 contiguous pixels. 7 loads (v, degree, 4 weights) +
4 neighbor loads + 1 store = 12 memory transactions. The arithmetic is 4 FMAs.
This is purely memory-bandwidth-bound.

**Correctness**: max absolute difference from the torch.roll reference
implementation is 2.38e-07 at 512x512 (float32 accumulation rounding).

**Performance**: 0.018ms per matvec (warmed, cached flat tensors) vs 0.122ms
for torch.roll -- 6.9x speedup.

## Why torch.compile Beats the Triton Kernel

The Triton kernel optimizes individual matvec cost: 0.018ms vs 0.122ms. But the
Lanczos iteration has 30 steps, and each step also does reorthogonalization
(matrix-vector products with the growing basis), dot products, norms, and scalar
arithmetic. With 30 Triton kernel launches, the total is:

    30 * 0.018ms (Triton matvec) + 30 * ~0.3ms (reorth + other) = ~10ms

With torch.compile on the entire Lanczos loop (using torch.roll for the matvec
so compile can trace it), the compiled graph fuses ALL 30 iterations' worth of
operations:

    ~4.5ms total (compiled, fused graph)

The compiled path uses torch.roll (which compile fuses into its elementwise
graph) rather than the Triton kernel (which compile can't fuse across). The
result: the Triton kernel is available via `laplacian_matvec()` for non-compiled
callers, but the compiled Lanczos uses pure-torch ops internally.

The same pattern holds for connected_components: compiled loop = 0.5-0.9ms vs
eager loop = 5-7ms (12x speedup from eliminating per-iteration sync barriers).

## Numerical Differences

The compiled Lanczos produces scores that differ from the eager version in the
6th decimal place (e.g., 0.359902 vs 0.359904). This is expected: torch.compile
may reorder floating-point operations within the fused graph. The relative
ordering of all images is preserved, and all 5/5 validation constraints pass
with identical outcomes.

## Performance Results

### Per-image pipeline breakdown (512x512, RTX 4090, post-warmup)

| Stage              | Before (torch.roll, eager) | After (compiled) |
|--------------------|---------------------------|-------------------|
| build_edge_weights | 0.8 ms                    | 0.25 ms           |
| lanczos (30 iter)  | 10.0 ms                   | 4.8 ms            |
| detect_contours    | 0.3 ms                    | 0.3 ms            |
| connected_components| 5.1 ms                   | 0.9 ms            |
| segment_signatures | 1.0 ms                    | 1.0 ms            |
| **single image**   | **~17 ms**                | **~7.3 ms**       |

### Full `structural_similarity_score` (2 images)

| Configuration           | Time (ms)  |
|--------------------------|-----------|
| Streams, eager           | 38 ms     |
| Sequential, eager        | 28 ms     |
| Streams, compiled        | 20 ms     |
| Sequential, compiled     | **14 ms** |

### Triton kernel microbenchmark

| Path                 | ms/matvec | Speedup |
|----------------------|-----------|---------|
| torch.roll (eager)   | 0.122     | 1.0x    |
| Triton (no cache)    | 0.027     | 4.5x    |
| Triton (cached flat) | 0.018     | 6.9x    |

### End-to-end `thisnotthat_score_v2` (steady-state, post-warmup)

| Before | After  |
|--------|--------|
| ~75 ms | ~30 ms |

Note: `thisnotthat_score_v2` calls `structural_similarity_score` twice (once
vs THIS reference, once vs THAT reference), so 2 * 14ms = 28ms plus
interpolation overhead.

### Compilation overhead

First call per unique resolution triggers torch.compile, which takes 60-80s on
Windows. Subsequent calls at the same resolution reuse the cached compiled graph.
For the validation suite with 3 distinct resolutions, the first 3 images incur
compilation overhead; the remaining 14 images run at steady-state speed.

## Constraint Validation

All 5/5 TNT validation constraints pass:

```
[PASS] 1_THIS_GT_THAT:     THIS_REF=3.346067 vs THAT_REF=-3.320323
[PASS] 2_THIS_GT_ALL:      THIS_REF=3.346067 vs max(others)=1.277342
[PASS] 3_THAT_LT_ALL:      THAT_REF=-3.320323 vs min(others)=0.359904
[PASS] 4_SKETCH_GT_COLOR:  min(SKETCH)=0.386091 vs max(COLOR)=0.359904
[PASS] 5_THAT_LT_NIGHTMODE: THAT_REF=-3.320323 vs NIGHTMODE=0.359904
```

## Architecture: Three Execution Paths

The code now has three paths for the Laplacian matvec:

1. **`_laplacian_matvec_triton()`**: Single Triton kernel launch. Used by
   `laplacian_matvec()` when Triton is available and input is on CUDA. Best
   for standalone matvec calls outside of loops.

2. **`_laplacian_matvec_torch()`**: Pure torch.roll stencil. CPU fallback and
   building block for the compiled Lanczos loop.

3. **`_lanczos_loop` (compiled)**: Inlines the torch.roll matvec and all other
   Lanczos operations into a single compiled graph. Used by `lanczos_fiedler()`
   on CUDA. Best overall because compile fuses across loop iterations.

The same pattern applies to connected_components: `_cc_propagation_loop`
(compiled) vs eager fallback.

## Files Modified

- `src_ii/itten_cuter_grops.py` -- all changes in this single file
- `docs/essay_tnt_v2_triton_kernel.md` -- this essay

## What Was NOT Changed

- `structural_similarity_score` signature -- identical
- `thisnotthat_score_v2` signature -- identical
- `_PipelineTimer` -- preserved
- `_process_single_image` -- preserved (internal helper)
- `compute_segment_signatures` -- no changes
- `match_segments` -- no changes
- `detect_contours` -- no changes

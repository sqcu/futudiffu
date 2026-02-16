# bench/

Performance benchmarks for futudiffu kernels, operations, and end-to-end
inference throughput. Every script here measures wall-clock time, TFLOPS,
throughput, or resource utilization -- not correctness.

## What belongs here

- Microbenchmarks for individual kernels (attention, GEMM, elementwise fusions).
- Throughput benchmarks for batched or end-to-end inference paths.
- Comparative benchmarks across implementation variants (e.g. SDPA vs Sage).
- Scripts that output timing tables, roofline percentages, or NFE/min metrics.

## What does NOT belong here

- **Correctness tests.** If it asserts equality or checks for regressions, it
  belongs in `tests/`.
- **Data generation or rendering scripts.** Those belong in `scripts/`.
- **Benchmark output artifacts** (images, CSVs, logs). Keep those in
  `bench_renders/` or a gitignored output directory, not alongside the scripts.

---

*Intermezzo*

```
O benchmark, that tireless clock-watcher of the GPU,
Who counts each microsecond lest a kernel misconstrue
Its place upon the roofline -- keep your timings here confined,
And leave the tests of truth to those of a correctness mind.
```

---

## Files

| File | Description |
|------|-------------|
| `bench_attention.py` | SDPA vs SageAttention (FP8 QK, INT8 QK) at NextDiT diffusion model shapes |
| `bench_batch_scaling.py` | Batch scaling behavior for diffusion forward pass (raw model, no compile) |
| `bench_btrm.py` | Full BTRM generation throughput: attention, diffusion, VAE, end-to-end |
| `bench_fp8_gemm.py` | Triton FP8 blockwise GEMM vs cuBLAS for the 4 NextDiT GEMM shapes |
| `bench_fusion.py` | Wall-time impact of each fusion level (w1w3, elementwise, FP8 FFN chain) |
| `bench_op_profile.py` | Per-operation cost profiling to characterize the GPU utilization gap |
| `bench_sage_pv.py` | SageAttention kernel variants (FP8/INT8 QK, BF16/FP8 PV) vs SDPA |
| `benchmark_sage_variants.py` | End-to-end generation with all Sage variants; saves images + reports metrics |

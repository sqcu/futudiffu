"""
Benchmark: Triton FP8 blockwise GEMM vs cuBLAS (torch._scaled_mm)
for the 4 GEMM shapes in NextDiT Z-Image model.

M = 2 * 4288 = 8576 (batch*seq with CFG batching)
Peak FP8 TFLOPS on RTX 4090: 660 TFLOPS
"""

import sys
sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

import torch
import math

assert torch.cuda.is_available(), "CUDA not available"
device = torch.device("cuda:0")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print()

from futudiffu.fp8_kernels import fp8_gemm_blockwise

M = 8576
SHAPES = [
    ("QKV proj",       M, 3840,  11520),
    ("Out proj",       M, 3840,  3840),
    ("w1w3 (FFN up)",  M, 3840,  20480),
    ("w2 (FFN down)",  M, 10240, 3840),
]

BLOCK_SIZE = 128
PEAK_TFLOPS = 660.0
WARMUP = 5
ITERS = 20

def total_flops(m, k, n):
    return 2 * m * k * n

def bench_triton(A_fp8, A_s, B_fp8, B_s, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        _ = fp8_gemm_blockwise(A_fp8, A_s, B_fp8, B_s, input_block_size=BLOCK_SIZE)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        _ = fp8_gemm_blockwise(A_fp8, A_s, B_fp8, B_s, input_block_size=BLOCK_SIZE)
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]

def bench_cublas(A_fp8, B_col_major, scale_a, scale_b, warmup=WARMUP, iters=ITERS):
    """
    torch._scaled_mm(A, B, scale_a, scale_b, out_dtype=...)
    A: (M, K) row-major (contiguous)
    B: (K, N) col-major (non-contiguous view from [N,K].t())
    """
    for _ in range(warmup):
        _ = torch._scaled_mm(A_fp8, B_col_major, scale_a, scale_b, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        _ = torch._scaled_mm(A_fp8, B_col_major, scale_a, scale_b, out_dtype=torch.bfloat16)
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]

# Also benchmark cuBLAS with use_fast_accum=True
def bench_cublas_fast(A_fp8, B_col_major, scale_a, scale_b, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        _ = torch._scaled_mm(A_fp8, B_col_major, scale_a, scale_b, out_dtype=torch.bfloat16, use_fast_accum=True)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        _ = torch._scaled_mm(A_fp8, B_col_major, scale_a, scale_b, out_dtype=torch.bfloat16, use_fast_accum=True)
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]

print(f"{'Shape':<16} {'M':>5} {'K':>5} {'N':>5} | {'Triton ms':>10} {'TF':>6} {'%pk':>5} | {'cuBLAS ms':>10} {'TF':>6} {'%pk':>5} | {'cuB-fast ms':>11} {'TF':>6} {'%pk':>5} | {'T/cB':>5}")
print("-" * 130)

for name, m, k, n in SHAPES:
    fl = total_flops(m, k, n)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    # Create random FP8 data
    # A: (M, K)
    A_bf16 = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    B_bf16 = torch.randn(n, k, device=device, dtype=torch.bfloat16)  # [N, K]

    # Blockwise quantize A -> A_fp8, A_s
    A_blocks = A_bf16.float().reshape(m, k // BLOCK_SIZE, BLOCK_SIZE)
    A_amax = A_blocks.abs().amax(dim=-1)  # (M, K//128)
    A_s = (A_amax / fp8_max).clamp(min=1e-12)
    A_quant = (A_blocks / A_s.unsqueeze(-1)).clamp(-fp8_max, fp8_max)
    A_fp8 = A_quant.reshape(m, k).to(torch.float8_e4m3fn)

    # Blockwise quantize B -> B_fp8, B_s
    # B_s shape: (N//128, K//128)
    n_blocks = n // BLOCK_SIZE
    k_blocks = k // BLOCK_SIZE
    B_blocks = B_bf16.float().reshape(n_blocks, BLOCK_SIZE, k_blocks, BLOCK_SIZE)
    B_amax = B_blocks.abs().amax(dim=(1, 3))  # (N//128, K//128)
    B_s = (B_amax / fp8_max).clamp(min=1e-12)
    # Quantize each block
    B_quant = B_blocks / B_s[:, None, :, None]
    B_quant = B_quant.clamp(-fp8_max, fp8_max).reshape(n, k)
    B_fp8 = B_quant.to(torch.float8_e4m3fn)

    # For cuBLAS: B needs to be (K, N) col-major = [N, K].t() (non-contiguous view)
    B_col_major = B_fp8.t()  # (K, N), non-contiguous, col-major
    assert not B_col_major.is_contiguous(), "B must be col-major (non-contiguous) for _scaled_mm"
    scale_a = torch.tensor(1.0, device=device, dtype=torch.float32)
    scale_b = torch.tensor(1.0, device=device, dtype=torch.float32)

    # Run benchmarks
    triton_times = bench_triton(A_fp8, A_s, B_fp8, B_s)
    triton_med = sorted(triton_times)[len(triton_times) // 2]
    triton_tf = (fl / (triton_med / 1000)) / 1e12

    try:
        cublas_times = bench_cublas(A_fp8, B_col_major, scale_a, scale_b)
        cublas_med = sorted(cublas_times)[len(cublas_times) // 2]
        cublas_tf = (fl / (cublas_med / 1000)) / 1e12
        cublas_str = f"{cublas_med:>10.3f} {cublas_tf:>6.1f} {100*cublas_tf/PEAK_TFLOPS:>4.1f}%"
    except Exception as e:
        cublas_med = float('inf')
        cublas_tf = 0
        cublas_str = f"  ERR: {str(e)[:35]}"

    try:
        cublas_fast_times = bench_cublas_fast(A_fp8, B_col_major, scale_a, scale_b)
        cublas_fast_med = sorted(cublas_fast_times)[len(cublas_fast_times) // 2]
        cublas_fast_tf = (fl / (cublas_fast_med / 1000)) / 1e12
        cublas_fast_str = f"{cublas_fast_med:>11.3f} {cublas_fast_tf:>6.1f} {100*cublas_fast_tf/PEAK_TFLOPS:>4.1f}%"
    except Exception as e:
        cublas_fast_med = float('inf')
        cublas_fast_tf = 0
        cublas_fast_str = f"  ERR: {str(e)[:35]}"

    ratio = triton_med / min(cublas_med, cublas_fast_med) if min(cublas_med, cublas_fast_med) < float('inf') else float('nan')
    triton_str = f"{triton_med:>10.3f} {triton_tf:>6.1f} {100*triton_tf/PEAK_TFLOPS:>4.1f}%"

    print(f"{name:<16} {m:>5} {k:>5} {n:>5} | {triton_str} | {cublas_str} | {cublas_fast_str} | {ratio:>5.2f}")

print()
print("Legend:")
print(f"  TF = TFLOPS, %pk = % of {PEAK_TFLOPS} TFLOPS peak, T/cB = Triton_time / best_cuBLAS_time")
print(f"  cuBLAS = torch._scaled_mm (tensorwise scales)")
print(f"  cuB-fast = torch._scaled_mm with use_fast_accum=True")
print(f"  Triton = fp8_gemm_blockwise (blockwise scales, block_size={BLOCK_SIZE})")
print(f"  Timing: median of {ITERS} iters, {WARMUP} warmup, CUDA events")

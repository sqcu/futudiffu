"""Benchmark: PyTorch SDPA vs SageAttention (FP8 QK, INT8 QK) on diffusion model shapes.

Shapes: B=2 (CFG), H=30 heads, N=4288 seq (128 caption + 4160 image), D=128 head_dim
All BF16 on CUDA, mask=None (non-causal).
"""
import sys
sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

import math
import torch
import torch.nn.functional as F

def main():
    # ---- Configuration ----
    B, H, N, D = 2, 30, 4288, 128
    WARMUP = 5
    ITERS = 20
    sm_scale = 1.0 / math.sqrt(D)

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Shape:  B={B}, H={H}, N={N}, D={D}")
    print(f"Warmup: {WARMUP}, Iterations: {ITERS}")
    print()

    # ---- Create inputs ----
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device=device)

    # ---- Benchmark helper ----
    def bench(fn, warmup=WARMUP, iters=ITERS):
        """Time fn() using CUDA events. Returns (result, median_ms, mean_ms, min_ms, max_ms)."""
        # Warmup
        for _ in range(warmup):
            result = fn()
        torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        times.sort()
        median = times[len(times)//2]
        mean = sum(times) / len(times)
        return result, median, mean, min(times), max(times)

    # ---- 1. PyTorch SDPA ----
    print("Benchmarking PyTorch SDPA (cutlassF / flash)...")
    def run_sdpa():
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)

    out_sdpa, sdpa_median, sdpa_mean, sdpa_min, sdpa_max = bench(run_sdpa)
    print(f"  Done. Median: {sdpa_median:.3f} ms")

    # ---- 2. SageAttention FP8 QK ----
    from futudiffu.sage_attention import sage_attn_forward, configure_sage

    print("Benchmarking SageAttention FP8 QK + BF16 PV...")
    configure_sage(smooth_k=True, qk_quant="fp8", pv_quant="bf16")

    def run_sage_fp8():
        return sage_attn_forward(q, k, v, sm_scale)

    out_fp8, fp8_median, fp8_mean, fp8_min, fp8_max = bench(run_sage_fp8)
    print(f"  Done. Median: {fp8_median:.3f} ms")

    # ---- 3. SageAttention INT8 QK ----
    print("Benchmarking SageAttention INT8 QK + BF16 PV...")
    configure_sage(smooth_k=True, qk_quant="int8", pv_quant="bf16")

    def run_sage_int8():
        return sage_attn_forward(q, k, v, sm_scale)

    out_int8, int8_median, int8_mean, int8_min, int8_max = bench(run_sage_int8)
    print(f"  Done. Median: {int8_median:.3f} ms")

    # ---- Accuracy comparison ----
    print()
    print("Computing accuracy metrics vs SDPA reference...")

    out_sdpa_flat = out_sdpa.float().flatten()
    out_fp8_flat = out_fp8.float().flatten()
    out_int8_flat = out_int8.float().flatten()

    cos_fp8 = F.cosine_similarity(out_fp8_flat.unsqueeze(0), out_sdpa_flat.unsqueeze(0)).item()
    cos_int8 = F.cosine_similarity(out_int8_flat.unsqueeze(0), out_sdpa_flat.unsqueeze(0)).item()

    mse_fp8 = ((out_fp8.float() - out_sdpa.float()) ** 2).mean().item()
    mse_int8 = ((out_int8.float() - out_sdpa.float()) ** 2).mean().item()

    max_err_fp8 = (out_fp8.float() - out_sdpa.float()).abs().max().item()
    max_err_int8 = (out_int8.float() - out_sdpa.float()).abs().max().item()

    # ---- Results table ----
    print()
    print("=" * 80)
    print(f"  ATTENTION BENCHMARK  |  B={B}, H={H}, N={N}, D={D}, BF16")
    print("=" * 80)
    print()
    print(f"{'Method':<28} {'Median':>8} {'Mean':>8} {'Min':>8} {'Max':>8} {'Speedup':>8}")
    print(f"{'':.<28} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8} {'vs SDPA':>8}")
    print("-" * 80)
    print(f"{'PyTorch SDPA (BF16)':<28} {sdpa_median:8.3f} {sdpa_mean:8.3f} {sdpa_min:8.3f} {sdpa_max:8.3f} {'1.00x':>8}")
    print(f"{'Sage FP8 QK + BF16 PV':<28} {fp8_median:8.3f} {fp8_mean:8.3f} {fp8_min:8.3f} {fp8_max:8.3f} {sdpa_median/fp8_median:7.2f}x")
    print(f"{'Sage INT8 QK + BF16 PV':<28} {int8_median:8.3f} {int8_mean:8.3f} {int8_min:8.3f} {int8_max:8.3f} {sdpa_median/int8_median:7.2f}x")
    print()
    print(f"{'Accuracy vs SDPA':<28} {'Cosine':>10} {'MSE':>14} {'Max Err':>12}")
    print("-" * 80)
    print(f"{'Sage FP8 QK + BF16 PV':<28} {cos_fp8:10.6f} {mse_fp8:14.2e} {max_err_fp8:12.4f}")
    print(f"{'Sage INT8 QK + BF16 PV':<28} {cos_int8:10.6f} {mse_int8:14.2e} {max_err_int8:12.4f}")
    print("=" * 80)

    # ---- Memory estimate ----
    elem_bytes = 2  # BF16
    total_bytes = 3 * B * H * N * D * elem_bytes  # Q + K + V
    print(f"\nInput memory (Q+K+V): {total_bytes / 1024**2:.1f} MB")
    print(f"Output memory:        {B * H * N * D * elem_bytes / 1024**2:.1f} MB")
    flops_attn = 2 * B * H * N * N * D  # QK^T
    flops_pv = 2 * B * H * N * N * D    # PV (same shape since non-causal full)
    # Actually PV is N x N times D, same as QK^T
    total_flops = flops_attn + flops_pv
    print(f"Attention FLOPs:      {total_flops / 1e12:.2f} TFLOP")
    for label, med in [("SDPA", sdpa_median), ("Sage FP8", fp8_median), ("Sage INT8", int8_median)]:
        tflops = total_flops / (med / 1000) / 1e12
        print(f"  {label:>12} throughput: {tflops:.1f} TFLOP/s")

if __name__ == "__main__":
    main()

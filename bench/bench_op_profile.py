"""Profile individual operation costs to characterize the 8x utilization gap.

Measures wall time for each operation type at actual model shapes to understand
where the overhead comes from: kernel launches, memory traffic, Triton dispatch,
or actual compute.

Usage:
    .venv/Scripts/python.exe bench_op_profile.py
"""

import os
import sys
import time

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")


def cuda_timer(fn, warmup=5, repeat=20):
    """Time a callable using CUDA events. Returns (mean_ms, std_ms)."""
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    return mean, std


def main():
    import torch
    import torch.nn.functional as F

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"Device: {torch.cuda.get_device_name()}", flush=True)
    print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB",
          flush=True)

    # Actual model shapes
    B = 2       # CFG batching
    seq = 4288  # 128 caption + 4160 image patches
    dim = 3840
    hidden = 10240  # FFN hidden dim
    n_heads = 30
    head_dim = 128

    M = B * seq  # token-rows for GEMM: 8576

    print(f"\nShapes: B={B}, seq={seq}, dim={dim}, hidden={hidden}, M={M}", flush=True)
    print(f"Heads: {n_heads}, head_dim: {head_dim}", flush=True)

    # =========================================================================
    # Section 1: Raw kernel launch overhead
    # =========================================================================
    print("\n" + "=" * 70, flush=True)
    print("  KERNEL LAUNCH OVERHEAD", flush=True)
    print("=" * 70, flush=True)

    # Empty CUDA kernel (just launch overhead)
    x_tiny = torch.zeros(1, device=device, dtype=dtype)
    mean, std = cuda_timer(lambda: torch.cuda.synchronize())
    print(f"  sync only:             {mean:.3f} +/- {std:.3f} ms", flush=True)

    # Minimal elementwise (identity)
    x = torch.randn(M, dim, device=device, dtype=dtype)
    mean, std = cuda_timer(lambda: x + 0)
    print(f"  x + 0 ({M}x{dim}):    {mean:.3f} +/- {std:.3f} ms", flush=True)

    # =========================================================================
    # Section 2: Elementwise operations at model scale
    # =========================================================================
    print("\n" + "=" * 70, flush=True)
    print("  ELEMENTWISE OPS (M={}, dim={})".format(M, dim), flush=True)
    print("=" * 70, flush=True)

    x = torch.randn(B, seq, dim, device=device, dtype=dtype)
    w = torch.randn(dim, device=device, dtype=dtype)
    scale = torch.randn(B, dim, device=device, dtype=dtype)
    gate = torch.randn(B, dim, device=device, dtype=dtype)

    # RMSNorm
    mean, std = cuda_timer(lambda: F.rms_norm(x, w.shape, weight=w, eps=1e-5))
    print(f"  RMSNorm:               {mean:.3f} +/- {std:.3f} ms", flush=True)

    # Modulate: x * (1 + scale)
    mean, std = cuda_timer(lambda: x * (1 + scale.unsqueeze(1)))
    print(f"  modulate:              {mean:.3f} +/- {std:.3f} ms", flush=True)

    # RMSNorm + modulate (unfused)
    def norm_mod():
        return F.rms_norm(x, w.shape, weight=w, eps=1e-5) * (1 + scale.unsqueeze(1))
    mean, std = cuda_timer(norm_mod)
    print(f"  RMSNorm+modulate:      {mean:.3f} +/- {std:.3f} ms", flush=True)

    # tanh
    mean, std = cuda_timer(lambda: gate.unsqueeze(1).tanh())
    print(f"  tanh(gate):            {mean:.3f} +/- {std:.3f} ms", flush=True)

    # gate * x (with broadcast)
    gt = gate.unsqueeze(1).tanh()
    mean, std = cuda_timer(lambda: gt * x)
    print(f"  gate * x:              {mean:.3f} +/- {std:.3f} ms", flush=True)

    # residual add
    y = torch.randn_like(x)
    mean, std = cuda_timer(lambda: x + y)
    print(f"  residual add:          {mean:.3f} +/- {std:.3f} ms", flush=True)

    # RMSNorm + gate + residual (unfused)
    def norm_gate_res():
        normed = F.rms_norm(x, w.shape, weight=w, eps=1e-5)
        return y + gate.unsqueeze(1).tanh() * normed
    mean, std = cuda_timer(norm_gate_res)
    print(f"  RMSNorm+gate+residual: {mean:.3f} +/- {std:.3f} ms", flush=True)

    # SiLU
    x_hidden = torch.randn(M, hidden, device=device, dtype=dtype)
    mean, std = cuda_timer(lambda: F.silu(x_hidden))
    print(f"  SiLU ({M}x{hidden}):   {mean:.3f} +/- {std:.3f} ms", flush=True)

    # SiLU + gate multiply
    x_hidden2 = torch.randn(M, hidden, device=device, dtype=dtype)
    mean, std = cuda_timer(lambda: F.silu(x_hidden) * x_hidden2)
    print(f"  SiLU * gate:           {mean:.3f} +/- {std:.3f} ms", flush=True)

    del x, w, scale, gate, gt, y, x_hidden, x_hidden2

    # =========================================================================
    # Section 3: FP8 operations
    # =========================================================================
    print("\n" + "=" * 70, flush=True)
    print("  FP8 OPS", flush=True)
    print("=" * 70, flush=True)

    from futudiffu.fp8_kernels import fp8_act_quant, fp8_gemm_blockwise

    # act_quant at different shapes
    for label, shape in [
        ("QKV input", (M, dim)),
        ("FFN input", (M, dim)),
        ("FFN mid", (M, hidden)),
    ]:
        x = torch.randn(*shape, device=device, dtype=dtype)
        mean, std = cuda_timer(lambda: fp8_act_quant(x, block_size=128))
        bytes_rw = x.numel() * 2 + x.numel()  # read bf16 + write fp8 + scales
        bw_gbps = bytes_rw / (mean / 1000) / 1e9
        print(f"  act_quant {label:12s} {shape}: {mean:.3f} +/- {std:.3f} ms  ({bw_gbps:.0f} GB/s)", flush=True)
    del x

    # FP8 GEMM at actual model shapes
    # Need FP8 weights. Create dummy ones.
    for label, M_dim, K_dim, N_dim in [
        ("QKV",     M, dim, 3 * dim),      # 8576 x 3840 x 11520
        ("out",     M, dim, dim),           # 8576 x 3840 x 3840
        ("w1",      M, dim, hidden),        # 8576 x 3840 x 10240
        ("w2",      M, hidden, dim),        # 8576 x 10240 x 3840
        ("w1w3",    M, dim, 2 * hidden),    # 8576 x 3840 x 20480 (fused)
        ("adaLN",   B, 256, 4 * dim),       # 2 x 256 x 15360
    ]:
        # Create FP8 inputs
        a_bf16 = torch.randn(M_dim, K_dim, device=device, dtype=dtype)
        a_fp8, a_s = fp8_act_quant(a_bf16, block_size=128)
        del a_bf16

        # Create FP8 weights (dummy)
        w_bf16 = torch.randn(N_dim, K_dim, device=device, dtype=dtype)
        w_fp8, w_s = fp8_act_quant(w_bf16.contiguous(), block_size=128)
        del w_bf16

        try:
            mean, std = cuda_timer(
                lambda: fp8_gemm_blockwise(a_fp8, a_s, w_fp8, w_s,
                                           input_block_size=128, output_dtype=dtype),
                warmup=3, repeat=10,
            )
            flops = 2 * M_dim * K_dim * N_dim
            tflops = flops / (mean / 1000) / 1e12
            print(f"  GEMM {label:8s} ({M_dim}x{K_dim}x{N_dim}): "
                  f"{mean:.3f} +/- {std:.3f} ms  ({tflops:.1f} TFLOPS)", flush=True)
        except Exception as e:
            print(f"  GEMM {label:8s} ({M_dim}x{K_dim}x{N_dim}): FAILED: {e}", flush=True)

        del a_fp8, a_s, w_fp8, w_s
        torch.cuda.empty_cache()

    # =========================================================================
    # Section 4: SDPA at model scale
    # =========================================================================
    print("\n" + "=" * 70, flush=True)
    print("  ATTENTION (SDPA)", flush=True)
    print("=" * 70, flush=True)

    q = torch.randn(B, n_heads, seq, head_dim, device=device, dtype=dtype)
    k = torch.randn(B, n_heads, seq, head_dim, device=device, dtype=dtype)
    v = torch.randn(B, n_heads, seq, head_dim, device=device, dtype=dtype)

    mean, std = cuda_timer(
        lambda: F.scaled_dot_product_attention(q, k, v),
        warmup=3, repeat=10,
    )
    flops_attn = 2 * 2 * B * n_heads * seq * seq * head_dim  # QK^T + PV
    tflops_attn = flops_attn / (mean / 1000) / 1e12
    print(f"  SDPA ({B}x{n_heads}x{seq}x{head_dim}): {mean:.3f} +/- {std:.3f} ms  ({tflops_attn:.1f} TFLOPS)", flush=True)
    del q, k, v

    # =========================================================================
    # Section 5: Estimated breakdown vs actual
    # =========================================================================
    print("\n" + "=" * 70, flush=True)
    print("  ESTIMATED BREAKDOWN (30 main layers, B=2)", flush=True)
    print("=" * 70, flush=True)

    # Re-run key timings for the estimate
    # We'll use the results from above. For now just print the accounting.
    print("  (Compare sum of estimated component costs vs actual 832ms forward)", flush=True)
    print("  See individual timings above to compute:", flush=True)
    print("    30 layers x (1 QKV + 1 out + 1 w1 + 1 w3 + 1 w2) GEMM", flush=True)
    print("    30 layers x (5 act_quant + ~12 elementwise)", flush=True)
    print("    30 layers x 1 SDPA", flush=True)
    print("    + context_refiner (2 layers, 128 tokens)", flush=True)
    print("    + noise_refiner (2 layers, 4160 tokens)", flush=True)
    print("    + embeddings + final_layer", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

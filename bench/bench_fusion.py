"""Benchmark fused vs unfused diffusion model forward pass.

Tests each fusion level's wall-time impact:
  1. Baseline (no fusion)
  2. w1w3 horizontal GEMM fusion only
  3. w1w3 + fused elementwise kernels
  4. w1w3 + FP8 FFN chain
  5. All three fusions

Uses CUDA events for timing. Loads the FP8 diffusion model once,
then toggles fusion flags between measurements.

Usage:
    .venv/Scripts/python.exe bench_fusion.py
"""

import os
import sys
import time
import gc

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")


def cuda_timer(fn, warmup=5, repeat=20):
    """Time a callable using CUDA events. Returns (mean_ms, std_ms, times)."""
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
    return mean, std, times


def load_fp8_model(device, dtype):
    """Load FP8 diffusion model (same as generate.py fast path)."""
    import torch
    from futudiffu.diffusion_model import (
        create_diffusion_model,
        _detect_cap_feat_dim,
        _detect_n_layers,
        _detect_qk_norm,
        _strip_diffusion_prefix,
    )
    from futudiffu.fp8 import replace_linear_with_fp8
    from safetensors.torch import load_file

    model_path = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
    print(f"Loading FP8 model from {model_path}...")
    diff_sd = load_file(model_path, device=str(device))
    remapped = _strip_diffusion_prefix(diff_sd)
    del diff_sd

    n_layers = _detect_n_layers(remapped.keys())
    cap_feat_dim = _detect_cap_feat_dim(remapped)
    qk_norm = _detect_qk_norm(remapped.keys())
    model = create_diffusion_model(dtype=dtype, n_layers=n_layers,
                                   cap_feat_dim=cap_feat_dim, qk_norm=qk_norm)

    replace_linear_with_fp8(model, remapped, block_size=128, output_dtype=dtype)

    remaining = {}
    for k, v in remapped.items():
        if k.endswith(".weight_scale") or k.endswith(".comfy_quant"):
            continue
        remaining[k] = v
    model.load_state_dict(remaining, strict=False, assign=True)
    del remapped, remaining

    model = model.to(device)
    model.eval()
    return model


def make_inputs(model, device, dtype, batch_size=2):
    """Create model inputs at actual generation shapes."""
    import torch

    # Actual shapes from generation: 1280x832 -> 160x104 latent -> 80x52 patches
    h, w = 104, 160  # latent dims (height//8, width//8)
    x = torch.randn(batch_size, 16, h, w, device=device, dtype=dtype)
    timesteps = torch.tensor([0.967] * batch_size, device=device, dtype=dtype)

    # Caption: 128 tokens with cap_feat_dim=2560
    cap_len = 128
    context = torch.randn(batch_size, cap_len, 2560, device=device, dtype=dtype)

    # Precompute RoPE cache
    padded_h = h + ((-h) % model.patch_size)
    padded_w = w + ((-w) % model.patch_size)
    rope_cache = model.prepare_rope_cache(padded_h, padded_w, cap_len, device)

    return x, timesteps, context, cap_len, rope_cache


def reset_fusion(model):
    """Disable all fusion flags without changing weights.

    Note: w1w3 weight fusion is structural (weights are physically concatenated),
    so we can't un-fuse after fuse_w1w3(). Instead we track the fusion state
    and only toggle the chain and elementwise flags.
    """
    from futudiffu.diffusion_model import FeedForward, JointTransformerBlock

    for module in model.modules():
        if isinstance(module, FeedForward):
            module._fused_chain = False
        if isinstance(module, JointTransformerBlock):
            module._use_fused_elementwise = False


def enable_fused_elementwise(model):
    """Enable fused elementwise kernels on modulated blocks."""
    from futudiffu.diffusion_model import JointTransformerBlock
    from futudiffu.fused_kernels import _HAS_TRITON
    if not _HAS_TRITON:
        print("  WARNING: Triton not available, skipping fused elementwise")
        return 0
    n = 0
    for module in model.modules():
        if isinstance(module, JointTransformerBlock) and module.modulation:
            module._use_fused_elementwise = True
            n += 1
    return n


def enable_fused_chain(model):
    """Enable FP8 FFN chain on fused FeedForward blocks."""
    from futudiffu.diffusion_model import FeedForward
    from futudiffu.fp8 import FP8Linear
    n = 0
    for module in model.modules():
        if isinstance(module, FeedForward) and module._fused:
            if isinstance(module.w2, FP8Linear):
                module._fused_chain = True
                n += 1
    return n


def main():
    import torch

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Dtype:  {dtype}")
    print()

    # Load model
    model = load_fp8_model(device, dtype)

    # Apply w1w3 fusion first (structural, can't undo)
    from futudiffu.diffusion_model import FeedForward
    n_fused = 0
    for module in model.modules():
        if isinstance(module, FeedForward):
            module.fuse_w1w3()
            if module._fused:
                n_fused += 1
    print(f"w1w3 fused: {n_fused} FeedForward layers")

    # Create inputs
    x, timesteps, context, cap_len, rope_cache = make_inputs(model, device, dtype)
    print(f"Input shapes: x={tuple(x.shape)}, context={tuple(context.shape)}")
    print()

    def run_forward():
        with torch.inference_mode():
            return model(x, timesteps, context, num_tokens=cap_len, rope_cache=rope_cache)

    def run_forward_fn(m, x_, t_, c_, cap_, rc_):
        with torch.inference_mode():
            return m(x_, t_, c_, num_tokens=cap_, rope_cache=rc_)

    # --- Config 1: w1w3 only (baseline with fusion) ---
    reset_fusion(model)
    print("=" * 70)
    print("  CONFIG 1: w1w3 fusion only (baseline)")
    print("=" * 70)
    mean, std, times = cuda_timer(run_forward, warmup=3, repeat=10)
    print(f"  Forward: {mean:.1f} +/- {std:.1f} ms")
    print(f"  Times:   {[f'{t:.1f}' for t in times]}")
    baseline_ms = mean
    print()

    # --- Config 2: w1w3 + fused elementwise ---
    reset_fusion(model)
    n = enable_fused_elementwise(model)
    print("=" * 70)
    print(f"  CONFIG 2: w1w3 + fused elementwise ({n} blocks)")
    print("=" * 70)
    mean, std, times = cuda_timer(run_forward, warmup=3, repeat=10)
    speedup = (baseline_ms - mean) / baseline_ms * 100
    print(f"  Forward: {mean:.1f} +/- {std:.1f} ms  ({speedup:+.1f}%)")
    print(f"  Times:   {[f'{t:.1f}' for t in times]}")
    print()

    # --- Config 3: w1w3 + FP8 chain ---
    reset_fusion(model)
    n = enable_fused_chain(model)
    print("=" * 70)
    print(f"  CONFIG 3: w1w3 + FP8 chain ({n} blocks)")
    print("=" * 70)
    mean, std, times = cuda_timer(run_forward, warmup=3, repeat=10)
    speedup = (baseline_ms - mean) / baseline_ms * 100
    print(f"  Forward: {mean:.1f} +/- {std:.1f} ms  ({speedup:+.1f}%)")
    print(f"  Times:   {[f'{t:.1f}' for t in times]}")
    print()

    # --- Config 4: All Phase 1 fusions ---
    reset_fusion(model)
    n_elem = enable_fused_elementwise(model)
    n_chain = enable_fused_chain(model)
    print("=" * 70)
    print(f"  CONFIG 4: All Phase 1 (w1w3 + elem({n_elem}) + chain({n_chain}))")
    print("=" * 70)
    mean, std, times = cuda_timer(run_forward, warmup=3, repeat=10)
    speedup = (baseline_ms - mean) / baseline_ms * 100
    print(f"  Forward: {mean:.1f} +/- {std:.1f} ms  ({speedup:+.1f}%)")
    print(f"  Times:   {[f'{t:.1f}' for t in times]}")
    phase1_ms = mean
    print()

    # --- Config 5: All Phase 1 + fuse_model (batched adaLN + fused QKV) ---
    from futudiffu.diffusion_model import fuse_model
    fuse_model(model)
    print("=" * 70)
    print("  CONFIG 5: Phase 1 + fuse_model (adaLN + QKV postprocess)")
    print("=" * 70)
    mean, std, times = cuda_timer(run_forward, warmup=3, repeat=10)
    speedup = (baseline_ms - mean) / baseline_ms * 100
    delta = phase1_ms - mean
    print(f"  Forward: {mean:.1f} +/- {std:.1f} ms  ({speedup:+.1f}% vs baseline, {delta:+.1f}ms vs Phase 1)")
    print(f"  Times:   {[f'{t:.1f}' for t in times]}")
    phase2_ms = mean
    print()

    # --- Config 6: Phase 2 + torch.compile ---
    print("=" * 70)
    print("  CONFIG 6: Phase 2 + torch.compile(mode='reduce-overhead')")
    print("=" * 70)
    print("  Compiling...")
    model_compiled = torch.compile(model, mode="reduce-overhead")
    # Extended warmup for compilation + CUDA graph capture
    for i in range(5):
        _ = run_forward_fn(model_compiled, x, timesteps, context, cap_len, rope_cache)
        torch.cuda.synchronize()
        print(f"    Warmup {i+1}/5")

    def run_compiled():
        with torch.inference_mode():
            return model_compiled(x, timesteps, context, num_tokens=cap_len, rope_cache=rope_cache)

    mean, std, times = cuda_timer(run_compiled, warmup=2, repeat=10)
    speedup = (baseline_ms - mean) / baseline_ms * 100
    delta = phase2_ms - mean
    print(f"  Forward: {mean:.1f} +/- {std:.1f} ms  ({speedup:+.1f}% vs baseline, {delta:+.1f}ms vs Phase 2)")
    print(f"  Times:   {[f'{t:.1f}' for t in times]}")
    print()

    # --- Summary ---
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Baseline (w1w3 only):    {baseline_ms:.1f} ms")
    print(f"  Phase 1 (all fusions):   {phase1_ms:.1f} ms  ({(baseline_ms-phase1_ms)/baseline_ms*100:+.1f}%)")
    print(f"  Phase 2 (+ adaLN + QKV): {phase2_ms:.1f} ms  ({(baseline_ms-phase2_ms)/baseline_ms*100:+.1f}%)")
    print(f"  Compiled (+ CUDA graph): {mean:.1f} ms  ({(baseline_ms-mean)/baseline_ms*100:+.1f}%)")
    print(f"  Pre-fusion baseline was ~832ms (from prior measurements).")
    print()
    print("Done.")


if __name__ == "__main__":
    main()

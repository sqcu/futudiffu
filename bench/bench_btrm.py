"""Benchmark inference kernels for BTRM dataset generation throughput.

Measures:
  1. Attention: SageAttention (FP8 QK, INT8 QK) vs PyTorch SDPA
  2. Diffusion forward: single vs CFG-batched, with/without torch.compile
  3. VAE decode: single vs batched latent decoding
  4. End-to-end: full t2i trajectory at 30 steps (the bottleneck)

Run from WSL:
    .venv/Scripts/python.exe bench_btrm.py [--skip-attention] [--skip-vae] [--skip-e2e]

Results are printed as a table and appended to bench_results.txt.
"""

import argparse
import sys
import time

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

import torch
import torch.nn.functional as F


def cuda_timer(fn, warmup=3, repeat=10, sync=True):
    """Time a callable using CUDA events. Returns (mean_ms, std_ms)."""
    # Warmup
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    return mean, std


def bench_attention(device, dtype):
    """Benchmark attention backends at diffusion model shapes."""
    print("\n" + "=" * 70)
    print("  ATTENTION BENCHMARKS")
    print("  Shape: (B, 30 heads, 4288 seq, 128 dim)")
    print("=" * 70)

    results = []

    for batch in [1, 2]:
        q = torch.randn(batch, 30, 4288, 128, device=device, dtype=dtype)
        k = torch.randn(batch, 30, 4288, 128, device=device, dtype=dtype)
        v = torch.randn(batch, 30, 4288, 128, device=device, dtype=dtype)
        sm_scale = 1.0 / (128 ** 0.5)

        # PyTorch SDPA
        def sdpa_fn():
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

        mean, std = cuda_timer(sdpa_fn)
        results.append(("SDPA", batch, mean, std))
        print(f"  SDPA          B={batch}: {mean:7.2f} +/- {std:.2f} ms")

        # SageAttention FP8 QK
        try:
            from futudiffu.sage_attention import sage_attn_op, configure_sage
            configure_sage(smooth_k=True, qk_quant="fp8", pv_quant="bf16")

            def sage_fp8_fn():
                return sage_attn_op(q, k, v, sm_scale)

            mean, std = cuda_timer(sage_fp8_fn)
            results.append(("Sage FP8 QK", batch, mean, std))
            print(f"  Sage FP8 QK   B={batch}: {mean:7.2f} +/- {std:.2f} ms")
        except Exception as e:
            print(f"  Sage FP8 QK   B={batch}: FAILED ({e})")

        # SageAttention INT8 QK
        try:
            from futudiffu.sage_attention import sage_attn_op, configure_sage
            configure_sage(smooth_k=True, qk_quant="int8", pv_quant="bf16")

            def sage_int8_fn():
                return sage_attn_op(q, k, v, sm_scale)

            mean, std = cuda_timer(sage_int8_fn)
            results.append(("Sage INT8 QK", batch, mean, std))
            print(f"  Sage INT8 QK  B={batch}: {mean:7.2f} +/- {std:.2f} ms")
        except Exception as e:
            print(f"  Sage INT8 QK  B={batch}: FAILED ({e})")

        del q, k, v
        torch.cuda.empty_cache()

    return results


def bench_vae_decode(device, dtype, vae_path):
    """Benchmark VAE decode at various batch sizes."""
    print("\n" + "=" * 70)
    print("  VAE DECODE BENCHMARKS")
    print("  Latent shape: (B, 16, 104, 160) -> Image (B, 3, 832, 1280)")
    print("=" * 70)

    from futudiffu.vae import load_vae, vae_decode

    model = load_vae(vae_path, device=device, dtype=dtype)
    model_compiled = torch.compile(model, mode="default")

    results = []

    for batch in [1, 2, 4]:
        latent = torch.randn(batch, 16, 104, 160, device=device, dtype=dtype)

        # Check if batch fits in VRAM (VAE decode is memory-hungry)
        try:
            # Uncompiled
            def vae_fn():
                from futudiffu.sampling import flux_process_out
                z = flux_process_out(latent)
                with torch.inference_mode():
                    decoded = model.decode(z)
                return decoded

            mean, std = cuda_timer(vae_fn, warmup=2, repeat=5)
            results.append(("VAE decode", batch, mean, std))
            print(f"  VAE decode (no compile) B={batch}: {mean:7.2f} +/- {std:.2f} ms")

            # Compiled
            def vae_compiled_fn():
                from futudiffu.sampling import flux_process_out
                z = flux_process_out(latent)
                with torch.inference_mode():
                    decoded = model_compiled.decode(z)
                return decoded

            mean, std = cuda_timer(vae_compiled_fn, warmup=2, repeat=5)
            results.append(("VAE decode compiled", batch, mean, std))
            print(f"  VAE decode (compiled)   B={batch}: {mean:7.2f} +/- {std:.2f} ms")

        except torch.cuda.OutOfMemoryError:
            print(f"  VAE decode B={batch}: OOM")
            torch.cuda.empty_cache()
            break

        del latent
        torch.cuda.empty_cache()

    del model, model_compiled
    torch.cuda.empty_cache()
    return results


def bench_vae_encode(device, dtype, vae_path):
    """Benchmark VAE encode (needed for i2i trajectories)."""
    print("\n" + "=" * 70)
    print("  VAE ENCODE BENCHMARKS")
    print("  Image shape: (B, 3, 832, 1280) -> Latent (B, 16, 104, 160)")
    print("=" * 70)

    from futudiffu.vae import load_vae

    model = load_vae(vae_path, device=device, dtype=dtype)

    results = []

    for batch in [1, 2]:
        image = torch.randn(batch, 3, 832, 1280, device=device, dtype=dtype)

        try:
            def vae_enc_fn():
                x = image * 2.0 - 1.0
                with torch.inference_mode():
                    return model.encode(x)

            mean, std = cuda_timer(vae_enc_fn, warmup=2, repeat=5)
            results.append(("VAE encode", batch, mean, std))
            print(f"  VAE encode  B={batch}: {mean:7.2f} +/- {std:.2f} ms")

        except torch.cuda.OutOfMemoryError:
            print(f"  VAE encode  B={batch}: OOM")
            torch.cuda.empty_cache()
            break

        del image
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    return results


def bench_diffusion_forward(device, dtype, diff_path, fp8=False):
    """Benchmark a single diffusion model forward pass."""
    label = "FP8" if fp8 else "BF16"
    print(f"\n{'=' * 70}")
    print(f"  DIFFUSION FORWARD BENCHMARKS ({label})")
    print(f"  Latent: (B, 16, 104, 160), Cond: (B, 128, 2560)")
    print(f"{'=' * 70}")

    if fp8:
        from futudiffu.diffusion_model import (
            create_diffusion_model,
            _detect_cap_feat_dim,
            _detect_n_layers,
            _detect_qk_norm,
            _strip_diffusion_prefix,
        )
        from futudiffu.fp8 import replace_linear_with_fp8
        from safetensors.torch import load_file

        diff_sd = load_file(diff_path, device=str(device))
        remapped = _strip_diffusion_prefix(diff_sd)
        del diff_sd

        n_layers = _detect_n_layers(remapped.keys())
        cap_feat_dim = _detect_cap_feat_dim(remapped)
        qk_norm = _detect_qk_norm(remapped.keys())
        model = create_diffusion_model(
            dtype=dtype, n_layers=n_layers,
            cap_feat_dim=cap_feat_dim, qk_norm=qk_norm)
        replace_linear_with_fp8(model, remapped, block_size=128, output_dtype=dtype)

        remaining = {k: v for k, v in remapped.items()
                     if not k.endswith((".weight_scale", ".comfy_quant"))}
        model.load_state_dict(remaining, strict=False, assign=True)
        del remapped, remaining
        model = model.to(device).eval()
    else:
        from futudiffu.diffusion_model import load_diffusion_model
        model = load_diffusion_model(diff_path, device=device, dtype=dtype)

    # Prepare inputs
    num_tokens = 128
    cond = torch.randn(1, num_tokens, 2560, device=device, dtype=dtype)

    latent_h, latent_w = 104, 160
    padded_h = latent_h + ((-latent_h) % model.patch_size)
    padded_w = latent_w + ((-latent_w) % model.patch_size)
    rope_cache = model.prepare_rope_cache(padded_h, padded_w, num_tokens, device)

    results = []

    # Single forward (B=1)
    x1 = torch.randn(1, 16, latent_h, latent_w, device=device, dtype=dtype)
    t1 = torch.tensor([0.5], device=device, dtype=dtype)

    def fwd_b1():
        with torch.inference_mode():
            return model(x1, t1, cond, num_tokens=num_tokens, rope_cache=rope_cache)

    mean, std = cuda_timer(fwd_b1, warmup=2, repeat=5)
    results.append((f"Diffusion {label} B=1", 1, mean, std))
    print(f"  Forward B=1 (no compile):  {mean:7.2f} +/- {std:.2f} ms")

    # CFG batched (B=2)
    x2 = torch.randn(1, 16, latent_h, latent_w, device=device, dtype=dtype).expand(2, -1, -1, -1)
    t2 = torch.tensor([0.5, 0.5], device=device, dtype=dtype)
    cond2 = cond.expand(2, -1, -1)

    def fwd_b2():
        with torch.inference_mode():
            return model(x2, t2, cond2, num_tokens=num_tokens, rope_cache=rope_cache)

    mean, std = cuda_timer(fwd_b2, warmup=2, repeat=5)
    results.append((f"Diffusion {label} B=2 CFG", 2, mean, std))
    print(f"  Forward B=2 CFG (no compile): {mean:7.2f} +/- {std:.2f} ms")

    # torch.compile reduce-overhead
    model_compiled = torch.compile(model, mode="reduce-overhead")

    def fwd_compiled_b2():
        with torch.inference_mode():
            return model_compiled(x2, t2, cond2, num_tokens=num_tokens, rope_cache=rope_cache)

    mean, std = cuda_timer(fwd_compiled_b2, warmup=3, repeat=5)
    results.append((f"Diffusion {label} B=2 compiled", 2, mean, std))
    print(f"  Forward B=2 CFG (compiled):   {mean:7.2f} +/- {std:.2f} ms")

    del model, model_compiled, x1, x2, t1, t2, cond, cond2
    torch.cuda.empty_cache()
    return results


def bench_e2e_trajectory(device, dtype, fp8_diff_path, te_path, vae_path):
    """End-to-end single trajectory benchmark: the real bottleneck."""
    print(f"\n{'=' * 70}")
    print(f"  END-TO-END TRAJECTORY BENCHMARK")
    print(f"  30-step t2i trajectory, FP8 diffusion, BF16 TE")
    print(f"{'=' * 70}")

    from generate_btrm_dataset import ModelCache, run_trajectory
    from pathlib import Path
    import tempfile

    models = ModelCache(device, dtype)

    # Load TE, encode one prompt, free TE
    print("  Loading TE...")
    models.load_text_encoder(te_path)
    prompt = 'qwen-3-4b, draw me a "laser shark swimming through a neon cityscape".'
    pos, neg = models.encode_prompt(prompt)
    te_cache = {prompt: (pos.detach().clone(), neg.detach().clone())}
    models.free_text_encoder()
    models.te_cache = te_cache
    models.encode_prompt = lambda p: models.te_cache[p]

    # Load FP8 diffusion
    print("  Loading FP8 diffusion model...")
    models.load_diffusion_fp8(fp8_diff_path)

    # Warmup torch.compile
    print("  Warming up torch.compile...")
    with tempfile.TemporaryDirectory() as tmpdir:
        run_trajectory(
            models, seed=0, prompt=prompt, n_steps=4, cfg=4.0,
            width=1280, height=832, sampling_shift=1.0, multiplier=1.0,
            save_steps=[], output_dir=Path(tmpdir) / "warmup",
        )

    # Benchmark: 30-step trajectory with SDPA (default)
    from futudiffu.attention import set_attention_backend
    results = []

    for backend_name, backend in [("SDPA", "sdpa"), ("Sage", "sage")]:
        set_attention_backend(backend)
        print(f"\n  --- {backend_name} backend ---")

        times = []
        for i in range(5):
            with tempfile.TemporaryDirectory() as tmpdir:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                run_trajectory(
                    models, seed=42 + i, prompt=prompt, n_steps=30, cfg=4.0,
                    width=1280, height=832, sampling_shift=1.0, multiplier=1.0,
                    save_steps=[0, 14, 29], output_dir=Path(tmpdir) / f"traj_{i}",
                )
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

        mean = sum(times) / len(times)
        std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
        results.append((f"E2E 30-step {backend_name}", 1, mean, std))
        print(f"  30-step trajectory: {mean:7.0f} +/- {std:.0f} ms")
        print(f"  Per-step:           {mean / 30:7.1f} ms")
        print(f"  Throughput:         {1000 / mean:.3f} traj/s, {30000 / mean:.1f} step/s")

    # Reset to SDPA
    set_attention_backend("sdpa")

    # Estimate full dataset timing (use best result)
    best_mean = min(r[2] for r in results)
    n_total = 2304
    est_hours = (best_mean / 1000) * n_total / 3600
    print(f"\n  Estimated full dataset ({n_total} trajectories):")
    print(f"    {est_hours:.1f} hours at best throughput")

    models.free_diffusion()
    torch.cuda.empty_cache()
    return results


def print_summary(all_results):
    """Print consolidated results table."""
    print(f"\n{'=' * 70}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Test':<35s} {'B':>2s} {'Mean (ms)':>10s} {'Std (ms)':>10s}")
    print(f"  {'-' * 35} {'--':>2s} {'-' * 10:>10s} {'-' * 10:>10s}")
    for name, batch, mean, std in all_results:
        print(f"  {name:<35s} {batch:>2d} {mean:>10.2f} {std:>10.2f}")


def main():
    parser = argparse.ArgumentParser(description="BTRM inference kernel benchmarks")
    parser.add_argument("--skip-attention", action="store_true")
    parser.add_argument("--skip-vae", action="store_true")
    parser.add_argument("--skip-diffusion", action="store_true")
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument("--fp8-diff",
                        default=r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors")
    parser.add_argument("--bf16-diff",
                        default=r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_bf16.safetensors")
    parser.add_argument("--te",
                        default=r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors")
    parser.add_argument("--vae",
                        default=r"F:\dox\ai\comfyui\ComfyUI\models\vae\zimage.safetensors")
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    all_results = []

    if not args.skip_attention:
        all_results.extend(bench_attention(device, dtype))

    if not args.skip_diffusion:
        all_results.extend(bench_diffusion_forward(device, dtype, args.fp8_diff, fp8=True))

    if not args.skip_vae:
        all_results.extend(bench_vae_decode(device, dtype, args.vae))
        all_results.extend(bench_vae_encode(device, dtype, args.vae))

    if not args.skip_e2e:
        all_results.extend(bench_e2e_trajectory(
            device, dtype, args.fp8_diff, args.te, args.vae))

    if all_results:
        print_summary(all_results)

    # Append to results file
    with open(r"F:\dox\repos\ai\futudiffu\bench_results.txt", "a") as f:
        import datetime
        f.write(f"\n--- {datetime.datetime.now().isoformat()} ---\n")
        f.write(f"Device: {torch.cuda.get_device_name()}\n")
        for name, batch, mean, std in all_results:
            f.write(f"  {name:<35s} B={batch:>2d}  {mean:>10.2f} +/- {std:.2f} ms\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Benchmark batch scaling for diffusion forward pass.

Tests raw model (no torch.compile) to measure true batch scaling behavior
without CUDA graph re-recording overhead masking results.

In-process watchdog thread hard-exits via os._exit(1) if any iteration
stalls for >10x the slowest completed iteration (floor 120s).

Each trajectory needs B=2 for CFG (pos + neg). So:
  k=1 trajectory  -> B=2
  k=2 trajectories -> B=4
  k=4 trajectories -> B=8
  ...

Usage:
    .venv/Scripts/python.exe bench_batch_scaling.py
"""

import os
import sys
import threading
import time

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")


class Watchdog:
    """In-process heartbeat watchdog. Daemon thread hard-exits on stall."""

    def __init__(self, stall_multiplier=10.0, floor_timeout=120.0,
                 initial_timeout=600.0):
        self._lock = threading.Lock()
        self._last_beat = time.monotonic()
        self._max_interval = 0.0
        self._stall_multiplier = stall_multiplier
        self._floor_timeout = floor_timeout
        self._initial_timeout = initial_timeout
        self._armed = False
        self._label = ""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def beat(self, label: str = ""):
        now = time.monotonic()
        with self._lock:
            interval = now - self._last_beat
            self._last_beat = now
            if self._armed and interval > self._max_interval:
                self._max_interval = interval
            self._label = label

    def arm(self):
        with self._lock:
            self._armed = True
            self._last_beat = time.monotonic()

    def _run(self):
        while True:
            time.sleep(1.0)
            with self._lock:
                elapsed = time.monotonic() - self._last_beat
                if self._max_interval > 0:
                    threshold = max(self._max_interval * self._stall_multiplier,
                                    self._floor_timeout)
                else:
                    threshold = self._initial_timeout

            if elapsed > threshold:
                print(f"\nWATCHDOG: stall! {elapsed:.0f}s since last beat "
                      f"(threshold={threshold:.0f}s, slowest={self._max_interval:.0f}s, "
                      f"last={self._label})", flush=True)
                os._exit(1)


def cuda_timer(fn, warmup=3, repeat=5):
    """Time a callable using CUDA events."""
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

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"Device: {torch.cuda.get_device_name()}", flush=True)
    print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB",
          flush=True)

    wd = Watchdog()

    # Load FP8 diffusion model
    print("Loading FP8 diffusion model...", flush=True)
    from futudiffu.diffusion_model import (
        create_diffusion_model, _detect_cap_feat_dim,
        _detect_n_layers, _detect_qk_norm, _strip_diffusion_prefix,
    )
    from futudiffu.fp8 import replace_linear_with_fp8
    from safetensors.torch import load_file

    diff_path = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
    sd = load_file(diff_path, device=str(device))
    remapped = _strip_diffusion_prefix(sd)
    del sd

    model = create_diffusion_model(
        dtype=dtype,
        n_layers=_detect_n_layers(remapped.keys()),
        cap_feat_dim=_detect_cap_feat_dim(remapped),
        qk_norm=_detect_qk_norm(remapped.keys()),
    )
    replace_linear_with_fp8(model, remapped, block_size=128, output_dtype=dtype)
    remaining = {k: v for k, v in remapped.items()
                 if not k.endswith((".weight_scale", ".comfy_quant"))}
    model.load_state_dict(remaining, strict=False, assign=True)
    del remapped, remaining
    model = model.to(device).eval()
    wd.beat("model_loaded")

    # Shared constants
    latent_h, latent_w = 104, 160
    num_tokens = 128
    padded_h = latent_h + ((-latent_h) % model.patch_size)
    padded_w = latent_w + ((-latent_w) % model.patch_size)
    rope_cache = model.prepare_rope_cache(padded_h, padded_w, num_tokens, device)

    # Header
    print(flush=True)
    print("=" * 80, flush=True)
    print("  BATCH SCALING: DIFFUSION FORWARD (FP8, raw model, no torch.compile)",
          flush=True)
    print("  k trajectories = B=2k (CFG). Latent (B,16,104,160) Cond (B,128,2560)",
          flush=True)
    print("=" * 80, flush=True)
    hdr = (f"  {'k':>3s}  {'B':>3s}  {'Total (ms)':>11s}  {'Per-traj (ms)':>13s}  "
           f"{'traj/s':>8s}  {'step/s':>8s}  {'Speedup':>8s}  {'VRAM (GB)':>9s}")
    sep = (f"  {'---':>3s}  {'---':>3s}  {'-'*11:>11s}  {'-'*13:>13s}  "
           f"{'-'*8:>8s}  {'-'*8:>8s}  {'-'*8:>8s}  {'-'*9:>9s}")
    print(hdr, flush=True)
    print(sep, flush=True)

    baseline = None
    results = []

    for k in [1, 2, 3, 4, 6, 8]:
        B = 2 * k
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            x = torch.randn(B, 16, latent_h, latent_w, device=device, dtype=dtype)
            t = torch.full((B,), 0.5, device=device, dtype=dtype)
            c = torch.randn(B, num_tokens, 2560, device=device, dtype=dtype)

            def fwd(x=x, t=t, c=c):
                with torch.inference_mode():
                    return model(x, t, c, num_tokens=num_tokens,
                                 rope_cache=rope_cache)

            mean, std = cuda_timer(fwd)
            vram = torch.cuda.max_memory_allocated() / 1024**3

            per = mean / k
            tps = k * 1000.0 / mean
            sps = tps * 30
            if baseline is None:
                baseline = per
            spd = baseline / per

            results.append((k, B, mean, per, tps, sps, spd, vram))
            print(f"  {k:>3d}  {B:>3d}  {mean:>10.1f}  {per:>12.1f}  "
                  f"{tps:>7.3f}  {sps:>7.1f}  {spd:>7.2f}x  {vram:>8.1f}",
                  flush=True)

            if not wd._armed:
                wd.arm()
            wd.beat(f"k={k} B={B}")

            del x, t, c

        except torch.cuda.OutOfMemoryError:
            print(f"  {k:>3d}  {B:>3d}  --- OOM ---", flush=True)
            wd.beat(f"k={k} OOM")
            torch.cuda.empty_cache()
            break

    # Summary
    if results:
        best = max(results, key=lambda r: r[4])
        print(f"\n  Best: k={best[0]} (B={best[1]}), "
              f"{best[4]:.3f} traj/s, {best[5]:.1f} step/s, "
              f"{best[6]:.2f}x, {best[7]:.1f} GB", flush=True)

        est_h = 2304 / best[4] / 3600
        print(f"  Full dataset (2304 traj): {est_h:.1f} hours", flush=True)

    with open(r"F:\dox\repos\ai\futudiffu\bench_results.txt", "a") as f:
        import datetime
        f.write(f"\n--- {datetime.datetime.now().isoformat()} BATCH SCALING (raw) ---\n")
        f.write(f"Device: {torch.cuda.get_device_name()}\n")
        for k, B, mean, per, tps, sps, spd, vram in results:
            f.write(f"  k={k:>2d} B={B:>2d}  total={mean:>8.1f}ms  "
                    f"per_traj={per:>7.1f}ms  {tps:.3f} traj/s  "
                    f"{spd:.2f}x  {vram:.1f}GB\n")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

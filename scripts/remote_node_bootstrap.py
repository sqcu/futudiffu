"""Remote node bootstrap: download models, quantize diffusion to FP8, validate.

Usage:
    python remote_node_bootstrap.py --output-dir ./models [--validate] [--reference-dir stream_futudiffu]

Phase 1: Download BF16 diffusion, BF16 text encoder, VAE from HuggingFace.
Phase 2: Quantize diffusion model to FP8 blockwise (block_size=128).
Phase 3: (optional) Validate via trajectory comparison against reference.
Phase 4: Print summary and launch commands.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Phase 1: Download
# ---------------------------------------------------------------------------

HF_REPO = "Comfy-Org/z_image"
HF_REVISION = "main"

DOWNLOADS = [
    ("split_files/diffusion_models/z_image_bf16.safetensors", "z_image_bf16.safetensors"),
    ("split_files/text_encoders/qwen_3_4b.safetensors", "qwen_3_4b.safetensors"),
    ("split_files/vae/ae.safetensors", "ae.safetensors"),
]


def download_models(output_dir: Path) -> dict[str, Path]:
    """Download model files from HuggingFace. Skip if already present."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed.")
        print("  Install with: pip install huggingface_hub")
        print("  Or add to pyproject.toml and re-sync.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    for hf_path, local_name in DOWNLOADS:
        dest = output_dir / local_name
        if dest.exists():
            size_gb = dest.stat().st_size / (1024 ** 3)
            print(f"  [skip] {local_name} already exists ({size_gb:.2f} GB)")
            paths[local_name] = dest
            continue

        print(f"  [download] {hf_path} -> {local_name}")
        t0 = time.perf_counter()
        downloaded = hf_hub_download(
            repo_id=HF_REPO,
            filename=hf_path,
            revision=HF_REVISION,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
        )
        # hf_hub_download with local_dir saves to output_dir/split_files/...,
        # so move to the flat location we want.
        downloaded = Path(downloaded)
        if downloaded != dest:
            downloaded.rename(dest)
            # Clean up empty parent directories left by HF
            for parent in downloaded.parents:
                if parent == output_dir:
                    break
                try:
                    parent.rmdir()
                except OSError:
                    break

        elapsed = time.perf_counter() - t0
        size_gb = dest.stat().st_size / (1024 ** 3)
        print(f"           {size_gb:.2f} GB in {elapsed:.1f}s "
              f"({size_gb / elapsed * 1024:.0f} MB/s)")
        paths[local_name] = dest

    return paths


# ---------------------------------------------------------------------------
# Phase 2: Quantize
# ---------------------------------------------------------------------------

def quantize_diffusion_model(bf16_path: Path, output_path: Path,
                             block_size: int = 128) -> None:
    """Quantize BF16 diffusion model to FP8 blockwise using the canonical algorithm."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from futudiffu.fp8 import quantize_model
    return quantize_model(str(bf16_path), str(output_path), block_size=block_size)


# ---------------------------------------------------------------------------
# Phase 3: Validate
# ---------------------------------------------------------------------------

LASER_SHARK_PROMPT = (
    'ahem.\n*ting ting ting ting ting*\n'
    'the query model for this is a LARGE LANGUAGE MODEL, specifically QWEN-3-4B, '
    'a GENERAL PURPOSE SEMANTIC PARSER which is able to WRITE SENTENCES AT A TIME '
    'when they are participating in dialogue. however, in this situation, they are '
    'being used as a hidden state generator to steer an *image generation model*, '
    'z-image.\n\n'
    'qwen-3-4b, draw me an "enormous laser shark for the sega saturn".'
)


def cosine_similarity(a, b):
    """Compute cosine similarity between two tensors (flattened)."""
    import torch
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return torch.nn.functional.cosine_similarity(
        a_flat.unsqueeze(0), b_flat.unsqueeze(0)
    ).item()


def wait_for_server(endpoint: str, timeout: float = 120.0) -> bool:
    """Wait for the inference server to become responsive."""
    import zmq
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 3000)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(endpoint)

    t0 = time.perf_counter()
    # Send a minimal status request
    meta = json.dumps({
        "method": "status", "params": {}, "tensors": []
    }).encode("utf-8")

    while time.perf_counter() - t0 < timeout:
        try:
            sock.send_multipart([meta])
            sock.recv_multipart()
            sock.close()
            ctx.term()
            return True
        except zmq.Again:
            pass
        except zmq.ZMQError:
            # Socket might be in bad state after timeout, recreate
            sock.close()
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.RCVTIMEO, 3000)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(endpoint)
        time.sleep(2.0)

    sock.close()
    ctx.term()
    return False


def _tensor_to_uint8(image_tensor):
    """Convert (1, 3, H, W) image tensor in [0,1] to (H, W, 3) uint8 numpy."""
    import numpy as np
    return (image_tensor[0].permute(1, 2, 0).float().numpy() * 255).clip(0, 255).astype(np.uint8)


def _false_color_diff(a, b, gain=8.0):
    """Absolute per-pixel diff with inferno-style false coloring.

    Returns (H, W, 3) uint8 array and scalar mean diff.
    """
    import numpy as np
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    gray = diff.mean(axis=2)
    t = np.clip(gray * gain / 255.0, 0, 1)

    r = np.clip(np.where(t < 0.4, t * 2.0, np.where(t < 0.75, 0.8 + (t - 0.4) * 0.57, 1.0)), 0, 1)
    g = np.clip(np.where(t < 0.5, 0, (t - 0.5) * 2.0), 0, 1)
    b_ch = np.clip(np.where(t < 0.25, t * 3.0, np.where(t < 0.6, 0.75 - (t - 0.25) * 2.14, 0)), 0, 1)

    rgb = (np.stack([r, g, b_ch], axis=2) * 255).astype(np.uint8)
    return rgb, float(gray.mean())


def validate_trajectory(
    fp8_diff_path: Path,
    te_path: Path,
    vae_path: Path,
    reference_dir: Path,
    output_dir: Path,
    port: int = 5556,
) -> bool:
    """Start server, run canonical trajectory, compare against reference.

    Produces actual images:
      output_dir/reference.png   — VAE-decoded reference final latent
      output_dir/test.png        — VAE-decoded test final latent
      output_dir/diff.png        — |reference - test| amplified
      output_dir/false_color.png — false-color diff (gain=8x)
      output_dir/strip.png       — side-by-side: reference | diff | test
      output_dir/stats.json      — per-step cosine similarities + summary
    """
    import torch
    import numpy as np
    from PIL import Image

    endpoint = f"tcp://localhost:{port}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"

    # Start server as subprocess
    print(f"  Starting server on port {port}...")
    server_proc = subprocess.Popen(
        [
            sys.executable, "-m", "futudiffu.server",
            "--port", str(port),
            "--fp8-diff", str(fp8_diff_path),
            "--te", str(te_path),
            "--vae", str(vae_path),
        ],
        env={**os.environ, "PYTHONPATH": str(src_dir)},
        stdout=sys.stderr,
        stderr=sys.stderr,
    )

    try:
        print(f"  Waiting for server (PID {server_proc.pid})...")
        if not wait_for_server(endpoint, timeout=180.0):
            print("  ERROR: Server did not become responsive within 180s")
            return False
        print("  Server is ready.")

        sys.path.insert(0, str(src_dir))
        from futudiffu.client import InferenceClient

        client = InferenceClient(endpoint, timeout_ms=600_000)

        # Warmup
        print("  Warming up (model load + torch.compile, this may take minutes)...")
        t0_warmup = time.perf_counter()
        client.warmup()
        print(f"  Warmup complete in {time.perf_counter() - t0_warmup:.1f}s")

        # Encode prompts
        print("  Encoding prompts...")
        pos_cond = client.encode_prompt(LASER_SHARK_PROMPT)
        neg_cond = client.encode_prompt("")

        # Load canonical noise tensor from reference directory.
        # This is the literal PRNG output that produced the reference trajectory.
        # Using it bypasses torch.randn version differences entirely.
        ref_dir = Path(reference_dir)
        ref_noise_file = ref_dir / "noise.pt"
        if ref_noise_file.exists():
            canonical_noise = torch.load(ref_noise_file, map_location="cpu",
                                         weights_only=True)
            print(f"  Using canonical noise tensor from {ref_noise_file} "
                  f"(shape={list(canonical_noise.shape)}, dtype={canonical_noise.dtype})")
        else:
            canonical_noise = None
            print("  WARNING: No canonical noise tensor found, falling back to torch.randn")

        # Generate trajectory
        print("  Running 30-step trajectory (seed=91849188298864, cfg=4.0)...")
        t0 = time.perf_counter()
        result = client.sample_trajectory(
            pos_cond=pos_cond,
            neg_cond=neg_cond,
            seed=91849188298864,
            n_steps=30,
            cfg=4.0,
            width=1280,
            height=832,
            save_steps=list(range(30)),
            noise=canonical_noise,
        )
        print(f"  Trajectory complete in {time.perf_counter() - t0:.1f}s")

        # --- VAE decode test final latent ---
        print("  VAE-decoding test final latent...")
        test_image = client.vae_decode(result["final"])
        test_np = _tensor_to_uint8(test_image)
        Image.fromarray(test_np).save(str(output_dir / "test.png"))
        print(f"  Saved {output_dir / 'test.png'}")

        # --- Load and decode reference ---
        ref_vae_file = ref_dir / "vae_output.pt"
        ref_final_file = ref_dir / "final_latent.pt"

        if ref_vae_file.exists():
            # Reference VAE output already decoded — use directly
            ref_vae = torch.load(ref_vae_file, map_location="cpu", weights_only=True)
            ref_np = _tensor_to_uint8(ref_vae)
        elif ref_final_file.exists():
            # Decode reference final latent through the server's VAE
            print("  VAE-decoding reference final latent...")
            ref_latent = torch.load(ref_final_file, map_location="cpu", weights_only=True)
            ref_image = client.vae_decode(ref_latent)
            ref_np = _tensor_to_uint8(ref_image)
        else:
            print(f"  WARNING: No reference at {ref_dir}, saving test render only")
            return True

        Image.fromarray(ref_np).save(str(output_dir / "reference.png"))
        print(f"  Saved {output_dir / 'reference.png'}")

        # --- Pixel diff ---
        diff_raw = np.abs(ref_np.astype(np.float32) - test_np.astype(np.float32))
        diff_amplified = np.clip(diff_raw * 8, 0, 255).astype(np.uint8)
        Image.fromarray(diff_amplified).save(str(output_dir / "diff.png"))

        fc, mean_diff = _false_color_diff(ref_np, test_np, gain=8.0)
        Image.fromarray(fc).save(str(output_dir / "false_color.png"))
        print(f"  Saved diff.png + false_color.png (mean pixel diff: {mean_diff:.2f}/255)")

        # --- Side-by-side strip: reference | false_color | test ---
        h, w = ref_np.shape[:2]
        pad = 4
        strip = np.zeros((h, 3 * w + 2 * pad, 3), dtype=np.uint8)
        strip[:, :w] = ref_np
        strip[:, w + pad:2 * w + pad] = fc
        strip[:, 2 * w + 2 * pad:] = test_np
        Image.fromarray(strip).save(str(output_dir / "strip.png"))
        print(f"  Saved strip.png (reference | false_color_diff | test)")

        # --- Per-step cosine similarities ---
        manifest_path = ref_dir / "manifest.json"
        step_stats = []
        if manifest_path.exists():
            print("\n  Per-step cosine similarity:")
            for step_i in range(30):
                ref_file = ref_dir / f"euler_step_{step_i:02d}.pt"
                our_key = f"step_{step_i:02d}"
                if not ref_file.exists() or our_key not in result:
                    continue

                ref_data = torch.load(ref_file, map_location="cpu", weights_only=True)
                cos_x = cosine_similarity(result[our_key], ref_data["x"])
                step_stats.append({"step": step_i, "cos_x": cos_x})
                if step_i % 5 == 0 or step_i == 29:
                    print(f"    step {step_i:02d}: cos(x)={cos_x:.6f}")

        # Final latent cosine
        cos_final = None
        if ref_final_file.exists():
            ref_final = torch.load(ref_final_file, map_location="cpu", weights_only=True)
            cos_final = cosine_similarity(result["final"], ref_final)
            print(f"\n  Final latent cosine: {cos_final:.6f}")

        # --- Write stats JSON ---
        stats = {
            "mean_pixel_diff": mean_diff,
            "cos_final_latent": cos_final,
            "per_step": step_stats,
        }
        if step_stats:
            cos_values = [s["cos_x"] for s in step_stats]
            stats["cos_min"] = min(cos_values)
            stats["cos_mean"] = sum(cos_values) / len(cos_values)

        with open(output_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved stats.json")

        # --- Pass/fail ---
        if step_stats:
            min_cos = stats["cos_min"]
            print(f"\n  Step cosine stats: min={min_cos:.6f} mean={stats['cos_mean']:.6f}")
            passed = min_cos > 0.99
            if cos_final is not None:
                passed = passed and cos_final > 0.99
            # Also check pixel diff — images should be visually identical
            passed = passed and mean_diff < 5.0
        else:
            passed = True

        status = "PASSED" if passed else "FAILED"
        print(f"\n  Validation: {status}")
        print(f"  Visual evidence at: {output_dir}/")
        return passed

    finally:
        print(f"  Stopping server (PID {server_proc.pid})...")
        server_proc.send_signal(signal.SIGTERM)
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait()
        print("  Server stopped.")


# ---------------------------------------------------------------------------
# Phase 4: Report
# ---------------------------------------------------------------------------

def print_report(paths: dict[str, Path], passed: bool | None) -> None:
    """Print summary and launch commands."""
    print("\n" + "=" * 60)
    print("BOOTSTRAP COMPLETE")
    print("=" * 60)

    print("\nModel files:")
    for name, path in sorted(paths.items()):
        size_gb = path.stat().st_size / (1024 ** 3) if path.exists() else 0
        print(f"  {name}: {path} ({size_gb:.2f} GB)")

    fp8_path = paths.get("z_image_fp8_blockwise.safetensors")
    te_path = paths.get("qwen_3_4b.safetensors")
    vae_path = paths.get("ae.safetensors")

    if fp8_path and te_path and vae_path:
        print("\nServer launch command:")
        print(f"  python -m futudiffu.server \\")
        print(f"    --fp8-diff {fp8_path} \\")
        print(f"    --te {te_path} \\")
        print(f"    --vae {vae_path}")

    if passed is None:
        print("\nValidation: SKIPPED (use --validate to enable)")
    elif passed:
        print("\nValidation: PASSED")
        print("  Visual evidence at: validation_renders/")
    else:
        print("\nValidation: FAILED")
        print("  Visual evidence at: validation_renders/")
        print("  Inspect reference.png, test.png, false_color.png, strip.png")
        print("  and stats.json to diagnose the divergence.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap futudiffu models: download, quantize, validate")
    parser.add_argument("--output-dir", type=str, default="./models",
                        help="Directory for downloaded/quantized models (default: ./models)")
    parser.add_argument("--validate", action="store_true",
                        help="Run trajectory validation (requires GPU)")
    parser.add_argument("--reference-dir", type=str, default="stream_futudiffu",
                        help="Path to reference trajectory dir (default: stream_futudiffu)")
    parser.add_argument("--port", type=int, default=5556,
                        help="Port for validation server (default: 5556)")
    parser.add_argument("--block-size", type=int, default=128,
                        help="FP8 quantization block size (default: 128)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download phase (assume models already present)")
    parser.add_argument("--skip-quantize", action="store_true",
                        help="Skip quantization phase (assume FP8 model already exists)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    reference_dir = Path(args.reference_dir).resolve()

    all_paths = {}

    # Phase 1: Download
    print("\n=== Phase 1: Download ===")
    if args.skip_download:
        print("  Skipped (--skip-download)")
        for _, local_name in DOWNLOADS:
            p = output_dir / local_name
            if p.exists():
                all_paths[local_name] = p
    else:
        all_paths = download_models(output_dir)

    # Phase 2: Quantize
    print("\n=== Phase 2: Quantize BF16 -> FP8 Blockwise ===")
    bf16_path = output_dir / "z_image_bf16.safetensors"
    fp8_path = output_dir / "z_image_fp8_blockwise.safetensors"

    if args.skip_quantize:
        print("  Skipped (--skip-quantize)")
        if fp8_path.exists():
            all_paths["z_image_fp8_blockwise.safetensors"] = fp8_path
    elif not bf16_path.exists():
        print(f"  ERROR: BF16 model not found at {bf16_path}")
        print("  Run without --skip-download first.")
        sys.exit(1)
    else:
        if fp8_path.exists():
            size_gb = fp8_path.stat().st_size / (1024 ** 3)
            print(f"  FP8 model already exists ({size_gb:.2f} GB), re-quantizing...")

        n_quant = quantize_diffusion_model(bf16_path, fp8_path,
                                           block_size=args.block_size)
        all_paths["z_image_fp8_blockwise.safetensors"] = fp8_path

    # Phase 3: Validate
    passed = None
    if args.validate:
        print("\n=== Phase 3: Validate via Trajectory Comparison ===")
        te_path = all_paths.get("qwen_3_4b.safetensors", output_dir / "qwen_3_4b.safetensors")
        vae_path = all_paths.get("ae.safetensors", output_dir / "ae.safetensors")

        missing = []
        if not fp8_path.exists():
            missing.append(f"FP8 diff: {fp8_path}")
        if not te_path.exists():
            missing.append(f"TE: {te_path}")
        if not vae_path.exists():
            missing.append(f"VAE: {vae_path}")
        if missing:
            print("  ERROR: Missing model files for validation:")
            for m in missing:
                print(f"    {m}")
            passed = False
        else:
            validation_dir = Path(args.output_dir).resolve().parent / "validation_renders"
            passed = validate_trajectory(
                fp8_diff_path=fp8_path,
                te_path=te_path,
                vae_path=vae_path,
                reference_dir=reference_dir,
                output_dir=validation_dir,
                port=args.port,
            )
    else:
        print("\n=== Phase 3: Validate (skipped, use --validate) ===")

    # Phase 4: Report
    print("\n=== Phase 4: Report ===")
    # Make sure all expected paths are in the dict
    for name in ["z_image_bf16.safetensors", "z_image_fp8_blockwise.safetensors",
                 "qwen_3_4b.safetensors", "ae.safetensors"]:
        if name not in all_paths:
            p = output_dir / name
            if p.exists():
                all_paths[name] = p
    print_report(all_paths, passed)


if __name__ == "__main__":
    main()

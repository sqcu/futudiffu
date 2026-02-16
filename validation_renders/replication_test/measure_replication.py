"""Measure run-rerun reproducibility of the futudiffu inference server.

Runs the EXACT same generation config TWICE (same seed, same prompt, same
everything), saves all intermediates from both runs, runs analyze_residuals.py
to compare, VAE-decodes both final latents, and creates a difference image.

All artifacts are written to disk under validation_renders/replication_test/.

Usage:
    python measure_replication.py
"""

import json
import os
import subprocess
import sys
import time
import traceback

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

import numpy as np
import torch

from futudiffu.client import InferenceClient

# ---------------------------------------------------------------------------
# Config — identical for both runs
# ---------------------------------------------------------------------------

PROMPT = (
    'ahem.\n*ting ting ting ting ting*\n'
    'the query model for this is a LARGE LANGUAGE MODEL, specifically QWEN-3-4B, '
    'a GENERAL PURPOSE SEMANTIC PARSER which is able to WRITE SENTENCES AT A TIME '
    'when they are participating in dialogue. however, in this situation, they are '
    'being used as a hidden state generator to steer an *image generation model*, '
    'z-image.\n\n'
    'qwen-3-4b, draw me an "enormous laser shark for the sega saturn".'
)

SEED = 91849188298864
N_STEPS = 30
CFG = 4.0
WIDTH = 1280
HEIGHT = 832
PORT = 5560

BASE_DIR = r"F:\dox\repos\ai\futudiffu\validation_renders\replication_test"
RUN_A_DIR = os.path.join(BASE_DIR, "run_a")
RUN_B_DIR = os.path.join(BASE_DIR, "run_b")
ANALYSIS_DIR = os.path.join(BASE_DIR, "residual_analysis")
ANALYZE_SCRIPT = r"F:\dox\repos\ai\futudiffu\analyze_residuals.py"


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def tensor_to_image(t):
    """Convert (1, 3, H, W) tensor in [0,1] to (H, W, 3) uint8 numpy."""
    img = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).float().numpy()
    return (img * 255).clip(0, 255).astype(np.uint8)


def save_ppm(path, img_uint8):
    """Save a uint8 numpy HWC array as PPM (no dependencies needed)."""
    h, w, c = img_uint8.shape
    with open(path, 'wb') as f:
        f.write(f'P6\n{w} {h}\n255\n'.encode())
        f.write(img_uint8.tobytes())


def save_image(path, img_uint8):
    """Save image as PNG (with PIL) or PPM (fallback)."""
    try:
        from PIL import Image
        Image.fromarray(img_uint8).save(path)
        return path
    except ImportError:
        ppm_path = path.replace(".png", ".ppm")
        save_ppm(ppm_path, img_uint8)
        return ppm_path


# ---------------------------------------------------------------------------
# Run one trajectory, save everything
# ---------------------------------------------------------------------------

def run_trajectory(client, run_dir, label):
    """Run a full trajectory and save all intermediates to run_dir.

    Returns: (result_dict, pos_cond, neg_cond, vae_image_tensor)
    """
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  {label}: encoding prompts")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    pos_cond = client.encode_prompt(PROMPT)
    elapsed_pos = time.perf_counter() - t0
    print(f"  pos_cond shape={list(pos_cond.shape)} dtype={pos_cond.dtype} "
          f"({elapsed_pos:.1f}s)")

    t0 = time.perf_counter()
    neg_cond = client.encode_prompt("")
    elapsed_neg = time.perf_counter() - t0
    print(f"  neg_cond shape={list(neg_cond.shape)} dtype={neg_cond.dtype} "
          f"({elapsed_neg:.1f}s)")

    # Save text encoder outputs
    torch.save(pos_cond, os.path.join(run_dir, "te_pos.pt"))
    torch.save(neg_cond, os.path.join(run_dir, "te_neg.pt"))
    print(f"  Saved te_pos.pt, te_neg.pt")

    # Run trajectory
    print(f"\n  {label}: running {N_STEPS}-step trajectory (seed={SEED})...")
    t0 = time.perf_counter()
    result = client.sample_trajectory(
        pos_cond=pos_cond,
        neg_cond=neg_cond,
        seed=SEED,
        n_steps=N_STEPS,
        cfg=CFG,
        width=WIDTH,
        height=HEIGHT,
        attention_backend="sdpa",
        sampling_shift=1.0,
        multiplier=1.0,
        save_steps=list(range(N_STEPS)),
    )
    elapsed_sample = time.perf_counter() - t0
    print(f"  Trajectory done in {elapsed_sample:.1f}s")
    print(f"  Result keys: {sorted(result.keys())[:5]}... ({len(result.keys())} total)")

    # Save all step latents
    for i in range(N_STEPS):
        step_key = f"step_{i:02d}"
        if step_key in result:
            torch.save(result[step_key], os.path.join(run_dir, f"step_{i:02d}.pt"))
    print(f"  Saved step_00..step_{N_STEPS-1:02d}.pt")

    # Save final latent
    final = result["final"]
    torch.save(final, os.path.join(run_dir, "final.pt"))
    print(f"  Saved final.pt shape={list(final.shape)} dtype={final.dtype}")

    # VAE decode
    print(f"  {label}: VAE decoding...")
    t0 = time.perf_counter()
    vae_image = client.vae_decode(final)
    elapsed_vae = time.perf_counter() - t0
    print(f"  VAE decode done in {elapsed_vae:.1f}s, "
          f"shape={list(vae_image.shape)} dtype={vae_image.dtype}")

    return result, pos_cond, neg_cond, vae_image, elapsed_sample


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0_total = time.perf_counter()
    print("=" * 60)
    print("REPLICATION TEST: Run-Rerun Reproducibility Measurement")
    print("=" * 60)
    print(f"Port: {PORT}")
    print(f"Seed: {SEED}")
    print(f"Steps: {N_STEPS}")
    print(f"CFG: {CFG}")
    print(f"Resolution: {WIDTH}x{HEIGHT}")
    print(f"Output: {BASE_DIR}")

    # Create output dirs
    for d in [RUN_A_DIR, RUN_B_DIR, ANALYSIS_DIR]:
        os.makedirs(d, exist_ok=True)

    # Connect to server
    endpoint = f"tcp://localhost:{PORT}"
    print(f"\nConnecting to server at {endpoint}...")
    client = InferenceClient(endpoint, timeout_ms=600_000)

    # Status check
    try:
        status = client.status()
        print(f"  Server phase: {status['phase']}")
        print(f"  VRAM: {status['vram_allocated_gb']:.2f} / {status['vram_total_gb']:.2f} GB")
    except Exception as e:
        msg = f"Cannot connect to server at {endpoint}: {e}"
        print(f"FATAL: {msg}")
        with open(os.path.join(BASE_DIR, "INCOMPLETE.txt"), "w") as f:
            f.write(f"INCOMPLETE: {msg}\n")
            f.write(f"The server must be started first:\n")
            f.write(f"  .venv/Scripts/python.exe -m futudiffu.server --port {PORT} \\\n")
            f.write(f"    --fp8-diff '...' --te '...' --vae '...'\n")
        sys.exit(1)

    # Warmup (needed for compile)
    print("\nWarming up server (SDPA)...")
    t0 = time.perf_counter()
    client.warmup(attention_backend="sdpa")
    print(f"  Warmup done in {time.perf_counter() - t0:.1f}s")

    # -----------------------------------------------------------------------
    # Run A
    # -----------------------------------------------------------------------
    result_a, pos_a, neg_a, vae_a, time_a = run_trajectory(client, RUN_A_DIR, "RUN A")

    # -----------------------------------------------------------------------
    # Run B
    # -----------------------------------------------------------------------
    result_b, pos_b, neg_b, vae_b, time_b = run_trajectory(client, RUN_B_DIR, "RUN B")

    # -----------------------------------------------------------------------
    # Save rendered images
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Saving rendered images")
    print("=" * 60)

    img_a = tensor_to_image(vae_a)
    img_b = tensor_to_image(vae_b)

    path_a = save_image(os.path.join(BASE_DIR, "run_a_render.png"), img_a)
    print(f"  Saved {path_a}")

    path_b = save_image(os.path.join(BASE_DIR, "run_b_render.png"), img_b)
    print(f"  Saved {path_b}")

    # Difference image: |A - B| * 10, clamped to [0, 255]
    diff = np.abs(img_a.astype(np.float32) - img_b.astype(np.float32))
    diff_10x = np.clip(diff * 10, 0, 255).astype(np.uint8)
    path_diff = save_image(os.path.join(BASE_DIR, "diff_10x.png"), diff_10x)
    print(f"  Saved {path_diff}")

    mean_pixel_diff = diff.mean()
    max_pixel_diff = diff.max()
    print(f"  Mean pixel diff: {mean_pixel_diff:.4f}")
    print(f"  Max pixel diff:  {max_pixel_diff:.4f}")

    # -----------------------------------------------------------------------
    # Quick inline comparison (before analyze_residuals)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Quick inline comparison")
    print("=" * 60)

    def cosine_sim(a, b):
        af = a.float().flatten()
        bf = b.float().flatten()
        return (torch.dot(af, bf) / (af.norm() * bf.norm())).item()

    def mse_val(a, b):
        return ((a.float() - b.float()) ** 2).mean().item()

    # TE comparison
    te_pos_cos = cosine_sim(pos_a, pos_b)
    te_neg_cos = cosine_sim(neg_a, neg_b)
    te_pos_bitwise = torch.equal(pos_a, pos_b)
    te_neg_bitwise = torch.equal(neg_a, neg_b)
    print(f"  te_pos: cos={te_pos_cos:.10f}  bitwise={'EXACT' if te_pos_bitwise else 'DIFFERS'}")
    print(f"  te_neg: cos={te_neg_cos:.10f}  bitwise={'EXACT' if te_neg_bitwise else 'DIFFERS'}")

    # Per-step comparison
    print(f"\n  {'Step':>6}  {'Cos Sim':>14}  {'MSE':>14}  {'Bitwise':>10}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*10}")
    for i in range(N_STEPS):
        key = f"step_{i:02d}"
        if key in result_a and key in result_b:
            cos = cosine_sim(result_a[key], result_b[key])
            m = mse_val(result_a[key], result_b[key])
            bw = torch.equal(result_a[key], result_b[key])
            print(f"  {i:>6}  {cos:>14.10f}  {m:>14.4e}  {'EXACT' if bw else 'DIFFERS':>10}")

    # Final latent
    final_cos = cosine_sim(result_a["final"], result_b["final"])
    final_mse = mse_val(result_a["final"], result_b["final"])
    final_bw = torch.equal(result_a["final"], result_b["final"])
    print(f"  {'final':>6}  {final_cos:>14.10f}  {final_mse:>14.4e}  {'EXACT' if final_bw else 'DIFFERS':>10}")

    # VAE output
    vae_cos = cosine_sim(vae_a, vae_b)
    vae_mse = mse_val(vae_a, vae_b)
    vae_bw = torch.equal(vae_a, vae_b)
    print(f"  {'vae':>6}  {vae_cos:>14.10f}  {vae_mse:>14.4e}  {'EXACT' if vae_bw else 'DIFFERS':>10}")

    # -----------------------------------------------------------------------
    # Run analyze_residuals.py
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Running analyze_residuals.py")
    print("=" * 60)

    try:
        cmd = [
            sys.executable,
            ANALYZE_SCRIPT,
            RUN_A_DIR,
            RUN_B_DIR,
            "--out", ANALYSIS_DIR,
        ]
        print(f"  Command: {' '.join(cmd)}")
        result_proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        print(result_proc.stdout)
        if result_proc.stderr:
            print("  STDERR:", result_proc.stderr[:2000])
        if result_proc.returncode != 0:
            print(f"  WARNING: analyze_residuals.py exited with code {result_proc.returncode}")
    except subprocess.TimeoutExpired:
        print("  WARNING: analyze_residuals.py timed out after 120s")
    except Exception as e:
        print(f"  WARNING: analyze_residuals.py failed: {e}")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    total_elapsed = time.perf_counter() - t0_total
    print("\n" + "=" * 60)
    print("  REPLICATION TEST SUMMARY")
    print("=" * 60)
    print(f"  Run A time:       {time_a:.1f}s")
    print(f"  Run B time:       {time_b:.1f}s")
    print(f"  Total elapsed:    {total_elapsed:.1f}s")
    print()
    print(f"  TE pos bitwise:   {'EXACT' if te_pos_bitwise else 'DIFFERS'}")
    print(f"  TE neg bitwise:   {'EXACT' if te_neg_bitwise else 'DIFFERS'}")
    print(f"  Final latent:")
    print(f"    cosine sim:     {final_cos:.10f}")
    print(f"    MSE:            {final_mse:.4e}")
    print(f"    bitwise:        {'EXACT' if final_bw else 'DIFFERS'}")
    print(f"  VAE output:")
    print(f"    cosine sim:     {vae_cos:.10f}")
    print(f"    MSE:            {vae_mse:.4e}")
    print(f"    bitwise:        {'EXACT' if vae_bw else 'DIFFERS'}")
    print(f"  Pixel diff:")
    print(f"    mean:           {mean_pixel_diff:.4f}")
    print(f"    max:            {max_pixel_diff:.4f}")
    print()

    # Verdict
    if final_bw and vae_bw and te_pos_bitwise and te_neg_bitwise:
        verdict = "BITWISE REPRODUCIBLE: All outputs identical between runs."
    elif final_cos > 0.999999 and vae_cos > 0.999999:
        verdict = "NEAR-REPRODUCIBLE: Not bitwise identical but extremely close (cos > 0.999999)."
    elif final_cos > 0.999:
        verdict = "WEAKLY REPRODUCIBLE: Measurable drift but same image (cos > 0.999)."
    elif final_cos > 0.99:
        verdict = "UNRELIABLE: Significant drift between runs (cos in [0.99, 0.999])."
    else:
        verdict = f"NOT REPRODUCIBLE: Large divergence (cos = {final_cos:.6f})."

    print(f"  VERDICT: {verdict}")
    print()

    # Write artifacts list
    artifacts = {
        "run_a_dir": RUN_A_DIR,
        "run_b_dir": RUN_B_DIR,
        "run_a_render": path_a,
        "run_b_render": path_b,
        "diff_10x": path_diff,
        "residual_analysis": ANALYSIS_DIR,
        "config": {
            "prompt": PROMPT,
            "seed": SEED,
            "n_steps": N_STEPS,
            "cfg": CFG,
            "width": WIDTH,
            "height": HEIGHT,
            "port": PORT,
        },
        "results": {
            "te_pos_bitwise": te_pos_bitwise,
            "te_neg_bitwise": te_neg_bitwise,
            "te_pos_cos": te_pos_cos,
            "te_neg_cos": te_neg_cos,
            "final_cos": final_cos,
            "final_mse": final_mse,
            "final_bitwise": final_bw,
            "vae_cos": vae_cos,
            "vae_mse": vae_mse,
            "vae_bitwise": vae_bw,
            "mean_pixel_diff": float(mean_pixel_diff),
            "max_pixel_diff": float(max_pixel_diff),
        },
        "timing": {
            "run_a_seconds": round(time_a, 2),
            "run_b_seconds": round(time_b, 2),
            "total_seconds": round(total_elapsed, 2),
        },
        "verdict": verdict,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "torch_version": torch.__version__,
    }

    manifest_path = os.path.join(BASE_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(artifacts, f, indent=2)
    print(f"  Manifest: {manifest_path}")

    client.close()
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        incomplete_path = os.path.join(BASE_DIR, "INCOMPLETE.txt")
        with open(incomplete_path, "w") as f:
            f.write(f"INCOMPLETE: Script crashed.\n\n")
            f.write(f"Exception: {e}\n\n")
            f.write(traceback.format_exc())
        print(f"\nFATAL: {e}")
        traceback.print_exc()
        print(f"\nWrote {incomplete_path}")
        sys.exit(1)

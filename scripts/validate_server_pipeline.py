"""Validate the FP8 inference server pipeline against the reference trajectory.

Connects to a running inference server, runs the canonical laser shark trajectory,
and compares against the reference in stream_futudiffu/.

Can be used standalone (python validate_server_pipeline.py) or imported:
    from scripts.validate_server_pipeline import validate_pipeline
    passed, stats = validate_pipeline(port=5555, ref_dir="stream_futudiffu", output_dir="out")
"""
import os
import sys
import time

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

import numpy as np
import torch

from futudiffu.client import InferenceClient
from futudiffu.rendering import save_image, tensor_to_uint8

# ---------------------------------------------------------------------------
# Config defaults (used by standalone mode)
# ---------------------------------------------------------------------------

DEFAULT_PORT = 5557
DEFAULT_REF_DIR = r"F:\dox\repos\ai\futudiffu\stream_futudiffu"
DEFAULT_OUT_DIR = r"F:\dox\repos\ai\futudiffu\validation_renders"

PROMPT = (
    'ahem.\n*ting ting ting ting ting*\nthe query model for this is a LARGE '
    'LANGUAGE MODEL, specifically QWEN-3-4B, a GENERAL PURPOSE SEMANTIC PARSER '
    'which is able to WRITE SENTENCES AT A TIME when they are participating in '
    'dialogue. however, in this situation, they are being used as a hidden state '
    'generator to steer an *image generation model*, z-image.\n\nqwen-3-4b, draw '
    'me an "enormous laser shark for the sega saturn".'
)

SEED = 91849188298864
N_STEPS = 30
CFG = 4.0
WIDTH = 1280
HEIGHT = 832


def cosine_sim(a, b):
    """Compute cosine similarity between two tensors (flattened)."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return (torch.dot(a_flat, b_flat) / (a_flat.norm() * b_flat.norm())).item()


def mse(a, b):
    """Compute MSE between two tensors."""
    return ((a.float() - b.float()) ** 2).mean().item()


def validate_pipeline(
    port: int = DEFAULT_PORT,
    ref_dir: str = DEFAULT_REF_DIR,
    output_dir: str = DEFAULT_OUT_DIR,
) -> tuple[bool, dict]:
    """Run pipeline validation against reference trajectory.

    Connects to a running server on `port`, runs the canonical laser shark
    trajectory, compares against reference data in `ref_dir`, and saves
    visual evidence to `output_dir`.

    Args:
        port: Server ZMQ port.
        ref_dir: Path to reference trajectory directory (stream_futudiffu/).
        output_dir: Path to save validation renders.

    Returns:
        (passed, stats) where passed is True if all checks pass, and stats
        is a dict with all measured cosine similarities, MSE values, timing.
    """
    os.makedirs(output_dir, exist_ok=True)
    t0_total = time.perf_counter()

    print("=" * 70)
    print("FP8 Inference Server Pipeline Validation")
    print("=" * 70)

    client = InferenceClient(f"tcp://localhost:{port}")

    # 1. Status check
    print("\n[1] Server status...")
    status = client.status()
    print(f"    Phase: {status['phase']}")
    print(f"    VRAM: {status['vram_allocated_gb']:.2f} / {status['vram_total_gb']:.2f} GB")

    # 2. Encode prompts
    print("\n[2] Encoding prompts...")
    t0 = time.perf_counter()
    pos_cond = client.encode_prompt(PROMPT)
    elapsed_pos = time.perf_counter() - t0
    print(f"    pos_cond shape: {pos_cond.shape}, dtype: {pos_cond.dtype}, "
          f"elapsed: {elapsed_pos:.1f}s")

    t0 = time.perf_counter()
    neg_cond = client.encode_prompt("")
    elapsed_neg = time.perf_counter() - t0
    print(f"    neg_cond shape: {neg_cond.shape}, dtype: {neg_cond.dtype}, "
          f"elapsed: {elapsed_neg:.1f}s")

    # Compare conditioning against reference
    ref_pos = torch.load(os.path.join(ref_dir, "text_encoder_pos.pt"),
                         map_location="cpu", weights_only=True)
    ref_neg = torch.load(os.path.join(ref_dir, "text_encoder_neg.pt"),
                         map_location="cpu", weights_only=True)

    pos_cos = cosine_sim(pos_cond, ref_pos)
    neg_cos = cosine_sim(neg_cond, ref_neg)
    pos_mse = mse(pos_cond, ref_pos)
    neg_mse = mse(neg_cond, ref_neg)
    print(f"    pos_cond vs ref: cos={pos_cos:.6f}, mse={pos_mse:.2e}")
    print(f"    neg_cond vs ref: cos={neg_cos:.6f}, mse={neg_mse:.2e}")

    # 3. Warmup diffusion model
    print("\n[3] Warming up diffusion model (compile + 4-step trajectory)...")
    t0 = time.perf_counter()
    client.warmup(attention_backend="sdpa")
    elapsed_warmup = time.perf_counter() - t0
    print(f"    Warmup elapsed: {elapsed_warmup:.1f}s")

    # Check VRAM after warmup
    status = client.status()
    print(f"    VRAM after warmup: {status['vram_allocated_gb']:.2f} / "
          f"{status['vram_total_gb']:.2f} GB")

    # 4. Run the canonical laser shark trajectory
    # Load canonical noise tensor from reference directory. Seed-based noise
    # is not portable across torch versions, so the reference ships the literal
    # PRNG output that produced the reference trajectory.
    noise_path = os.path.join(ref_dir, "noise.pt")
    canonical_noise = None
    if os.path.exists(noise_path):
        canonical_noise = torch.load(noise_path, map_location="cpu", weights_only=True)
        print(f"\n[4] Running laser shark trajectory (canonical noise from {noise_path})...")
    else:
        print(f"\n[4] Running laser shark trajectory (seed={SEED}, NO canonical noise)...")
        print(f"    WARNING: {noise_path} not found, seed-based noise may diverge")

    t0 = time.perf_counter()
    result = client.sample_trajectory(
        pos_cond, neg_cond,
        seed=SEED,
        n_steps=N_STEPS,
        cfg=CFG,
        width=WIDTH,
        height=HEIGHT,
        attention_backend="sdpa",
        save_steps=list(range(N_STEPS)),
        noise=canonical_noise,
    )
    elapsed_sample = time.perf_counter() - t0
    print(f"    Sampling elapsed: {elapsed_sample:.1f}s")
    print(f"    Result keys: {sorted(result.keys())[:5]}... + final")
    final_latent = result["final"]
    print(f"    final_latent shape: {final_latent.shape}, dtype: {final_latent.dtype}")

    # 5. Compare final latent against reference
    print("\n[5] Comparing final latent against reference...")
    ref_final = torch.load(os.path.join(ref_dir, "final_latent.pt"),
                           map_location="cpu", weights_only=True)
    final_cos = cosine_sim(final_latent, ref_final)
    final_mse_val = mse(final_latent, ref_final)
    print(f"    final_latent vs ref: cos={final_cos:.6f}, mse={final_mse_val:.2e}")

    # 6. Compare per-step latents
    print("\n[6] Per-step latent comparison...")
    step_cosines = []
    for step_i in range(N_STEPS):
        step_key = f"step_{step_i:02d}"
        if step_key not in result:
            print(f"    WARNING: {step_key} not in result")
            continue

        ref_step_path = os.path.join(ref_dir, f"euler_step_{step_i:02d}.pt")
        if not os.path.exists(ref_step_path):
            continue

        ref_step = torch.load(ref_step_path, map_location="cpu", weights_only=True)

        # Reference step files have {"x": tensor, "denoised": tensor, "sigma": float}
        if isinstance(ref_step, dict):
            ref_x = ref_step["x"]
        else:
            ref_x = ref_step

        test_x = result[step_key]
        step_cos = cosine_sim(test_x, ref_x)
        step_cosines.append(step_cos)

    if step_cosines:
        print(f"    Step 0  cos: {step_cosines[0]:.6f}")
        print(f"    Step 14 cos: {step_cosines[min(14, len(step_cosines)-1)]:.6f}")
        print(f"    Step 29 cos: {step_cosines[min(29, len(step_cosines)-1)]:.6f}")
        print(f"    Min cos: {min(step_cosines):.6f} (step {step_cosines.index(min(step_cosines))})")
        print(f"    Mean cos: {sum(step_cosines)/len(step_cosines):.6f}")

    # 7. VAE decode
    print("\n[7] VAE decoding...")
    t0 = time.perf_counter()
    test_image = client.vae_decode(final_latent)
    elapsed_vae = time.perf_counter() - t0
    print(f"    test_image shape: {test_image.shape}, dtype: {test_image.dtype}, "
          f"elapsed: {elapsed_vae:.1f}s")

    # Compare VAE output
    ref_vae = torch.load(os.path.join(ref_dir, "vae_output.pt"),
                         map_location="cpu", weights_only=True)
    vae_cos = cosine_sim(test_image, ref_vae)
    vae_mse_val = mse(test_image, ref_vae)
    print(f"    vae_output vs ref: cos={vae_cos:.6f}, mse={vae_mse_val:.2e}")

    # 8. Save images
    print("\n[8] Saving images...")
    test_arr = tensor_to_uint8(test_image)
    ref_arr = tensor_to_uint8(ref_vae)

    save_image(test_arr, os.path.join(output_dir, "test.png"))
    save_image(ref_arr, os.path.join(output_dir, "reference.png"))

    # Difference image (amplified for visibility)
    diff_arr = np.abs(test_arr.astype(float) - ref_arr.astype(float))
    mean_pixel_diff = diff_arr.mean()
    max_pixel_diff = diff_arr.max()
    diff_amplified = np.clip(diff_arr * 10, 0, 255).astype(np.uint8)
    save_image(diff_amplified, os.path.join(output_dir, "diff_10x.png"))

    print(f"    Mean pixel diff: {mean_pixel_diff:.2f}")
    print(f"    Max pixel diff: {max_pixel_diff:.2f}")
    print(f"    Saved: test.png, reference.png, diff_10x.png")

    # 9. Summary
    total_elapsed = time.perf_counter() - t0_total

    stats = {
        "pos_cos": pos_cos,
        "neg_cos": neg_cos,
        "pos_mse": pos_mse,
        "neg_mse": neg_mse,
        "final_cos": final_cos,
        "final_mse": final_mse_val,
        "vae_cos": vae_cos,
        "vae_mse": vae_mse_val,
        "mean_pixel_diff": float(mean_pixel_diff),
        "max_pixel_diff": float(max_pixel_diff),
        "step_cos_min": min(step_cosines) if step_cosines else None,
        "step_cos_mean": (sum(step_cosines) / len(step_cosines)) if step_cosines else None,
        "total_elapsed_s": round(total_elapsed, 1),
    }

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Text encoder pos cos: {pos_cos:.6f}")
    print(f"  Text encoder neg cos: {neg_cos:.6f}")
    print(f"  Final latent cos:     {final_cos:.6f}")
    print(f"  Final latent MSE:     {final_mse_val:.2e}")
    print(f"  VAE output cos:       {vae_cos:.6f}")
    print(f"  VAE output MSE:       {vae_mse_val:.2e}")
    print(f"  Mean pixel diff:      {mean_pixel_diff:.2f}")
    print(f"  Max pixel diff:       {max_pixel_diff:.2f}")
    if step_cosines:
        print(f"  Per-step cos min:     {min(step_cosines):.6f}")
        print(f"  Per-step cos mean:    {sum(step_cosines)/len(step_cosines):.6f}")
    print(f"  Total elapsed:        {total_elapsed:.1f}s")
    print()

    # Pass/fail criteria
    passed = True
    if final_cos < 0.90:
        print("  FAIL: final_latent cos < 0.90")
        passed = False
    else:
        print(f"  PASS: final_latent cos >= 0.90 ({final_cos:.6f})")

    if vae_cos < 0.90:
        print("  FAIL: vae_output cos < 0.90")
        passed = False
    else:
        print(f"  PASS: vae_output cos >= 0.90 ({vae_cos:.6f})")

    if pos_cos < 0.99:
        print(f"  FAIL: text encoder pos cos < 0.99 ({pos_cos:.6f})")
        passed = False
    else:
        print(f"  PASS: text encoder pos cos >= 0.99 ({pos_cos:.6f})")

    print()
    if passed:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED")

    client.close()

    return passed, stats


def main():
    passed, stats = validate_pipeline(
        port=DEFAULT_PORT,
        ref_dir=DEFAULT_REF_DIR,
        output_dir=DEFAULT_OUT_DIR,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

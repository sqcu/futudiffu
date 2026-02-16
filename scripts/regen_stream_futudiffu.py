"""Regenerate canonical reference trajectory in stream_futudiffu/.

Uses the current bf16 server (bf16 throughout, no float32 sampling loop).
Connects to server on localhost:5559, runs the full Config B pipeline with
canonical noise from stream_futudiffu/noise.pt, saves all intermediates.
"""

import json
import os
import sys
import time

import torch

# Ensure src is on path
sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

from futudiffu.client import InferenceClient
from futudiffu.sampling import build_sigmas, simple_scheduler

STREAM_DIR = r"F:\dox\repos\ai\futudiffu\stream_futudiffu"

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


def main():
    endpoint = "tcp://localhost:5559"
    print(f"Connecting to server at {endpoint}...")
    client = InferenceClient(endpoint, timeout_ms=600_000)  # 10min timeout

    # --- Step 1: Warmup ---
    print("Warming up server (SDPA)...")
    t0 = time.perf_counter()
    client.warmup(attention_backend="sdpa")
    print(f"  Warmup done in {time.perf_counter() - t0:.1f}s")

    # --- Step 2: Encode prompts ---
    print("Encoding positive prompt...")
    t0 = time.perf_counter()
    pos_cond = client.encode_prompt(PROMPT)
    print(f"  pos_cond shape={list(pos_cond.shape)} dtype={pos_cond.dtype} "
          f"in {time.perf_counter() - t0:.1f}s")

    print("Encoding negative prompt (empty)...")
    t0 = time.perf_counter()
    neg_cond = client.encode_prompt("")
    print(f"  neg_cond shape={list(neg_cond.shape)} dtype={neg_cond.dtype} "
          f"in {time.perf_counter() - t0:.1f}s")

    # --- Step 3: Load canonical noise ---
    noise_path = os.path.join(STREAM_DIR, "noise.pt")
    print(f"Loading canonical noise from {noise_path}...")
    noise = torch.load(noise_path, map_location="cpu", weights_only=True)
    print(f"  noise shape={list(noise.shape)} dtype={noise.dtype}")

    # --- Step 4: Generate sigmas ---
    print("Generating sigma schedule...")
    sigma_table = build_sigmas(shift=1.0, multiplier=1000.0)
    sigmas = simple_scheduler(sigma_table, N_STEPS)
    print(f"  sigmas shape={list(sigmas.shape)} dtype={sigmas.dtype}")
    print(f"  sigmas[0]={sigmas[0].item():.6f}, sigmas[-1]={sigmas[-1].item():.6f}")

    # --- Step 5: Sample trajectory ---
    print(f"Running {N_STEPS}-step trajectory with canonical noise...")
    t0 = time.perf_counter()
    result = client.sample_trajectory(
        pos_cond=pos_cond,
        neg_cond=neg_cond,
        seed=SEED,  # ignored since we supply noise
        n_steps=N_STEPS,
        cfg=CFG,
        width=WIDTH,
        height=HEIGHT,
        attention_backend="sdpa",
        sampling_shift=1.0,
        multiplier=1.0,
        save_steps=list(range(N_STEPS)),
        noise=noise,
    )
    elapsed_sample = time.perf_counter() - t0
    print(f"  Trajectory done in {elapsed_sample:.1f}s")
    print(f"  Result keys: {sorted(result.keys())}")

    final_latent = result["final"]
    print(f"  final_latent shape={list(final_latent.shape)} dtype={final_latent.dtype}")

    # --- Step 6: VAE decode ---
    print("VAE decoding final latent...")
    t0 = time.perf_counter()
    vae_output = client.vae_decode(final_latent)
    elapsed_vae = time.perf_counter() - t0
    print(f"  vae_output shape={list(vae_output.shape)} dtype={vae_output.dtype} "
          f"in {elapsed_vae:.1f}s")

    # --- Step 7: Save everything ---
    print(f"\nSaving to {STREAM_DIR}...")
    t_save_start = time.perf_counter()

    stages = []

    # Text encoder outputs
    te_pos_path = os.path.join(STREAM_DIR, "text_encoder_pos.pt")
    torch.save(pos_cond, te_pos_path)
    stages.append({
        "name": "text_encoder_pos",
        "filename": "text_encoder_pos.pt",
        "shape": list(pos_cond.shape),
        "dtype": str(pos_cond.dtype),
    })
    print(f"  Saved text_encoder_pos.pt ({os.path.getsize(te_pos_path):,} bytes)")

    te_neg_path = os.path.join(STREAM_DIR, "text_encoder_neg.pt")
    torch.save(neg_cond, te_neg_path)
    stages.append({
        "name": "text_encoder_neg",
        "filename": "text_encoder_neg.pt",
        "shape": list(neg_cond.shape),
        "dtype": str(neg_cond.dtype),
    })
    print(f"  Saved text_encoder_neg.pt ({os.path.getsize(te_neg_path):,} bytes)")

    # Noise (verify, do not overwrite)
    assert os.path.exists(noise_path), f"Canonical noise missing: {noise_path}"
    stages.append({
        "name": "noise",
        "filename": "noise.pt",
        "shape": list(noise.shape),
        "dtype": str(noise.dtype),
        "note": "canonical noise preserved (not overwritten)",
    })
    print(f"  Verified noise.pt exists ({os.path.getsize(noise_path):,} bytes)")

    # Sigmas
    sigmas_path = os.path.join(STREAM_DIR, "sigmas.pt")
    torch.save(sigmas, sigmas_path)
    stages.append({
        "name": "sigmas",
        "filename": "sigmas.pt",
        "shape": list(sigmas.shape),
        "dtype": str(sigmas.dtype),
    })
    print(f"  Saved sigmas.pt ({os.path.getsize(sigmas_path):,} bytes)")

    # Euler steps
    for i in range(N_STEPS):
        step_key = f"step_{i:02d}"
        step_tensor = result[step_key]
        step_dict = {
            "x": step_tensor,
            "denoised": step_tensor,  # server only saves x; duplicate for compat
        }
        step_path = os.path.join(STREAM_DIR, f"euler_step_{i:02d}.pt")
        torch.save(step_dict, step_path)
        stages.append({
            "name": f"euler_step_{i:02d}",
            "filename": f"euler_step_{i:02d}.pt",
            "shape": {
                "x": list(step_tensor.shape),
                "denoised": list(step_tensor.shape),
            },
            "dtype": {
                "x": str(step_tensor.dtype),
                "denoised": str(step_tensor.dtype),
            },
        })
        if i == 0 or i == N_STEPS - 1:
            print(f"  Saved euler_step_{i:02d}.pt "
                  f"({os.path.getsize(step_path):,} bytes) "
                  f"dtype={step_tensor.dtype}")

    print(f"  Saved euler_step_00..{N_STEPS-1:02d}.pt (30 files)")

    # Final latent
    final_path = os.path.join(STREAM_DIR, "final_latent.pt")
    torch.save(final_latent, final_path)
    stages.append({
        "name": "final_latent",
        "filename": "final_latent.pt",
        "shape": list(final_latent.shape),
        "dtype": str(final_latent.dtype),
    })
    print(f"  Saved final_latent.pt ({os.path.getsize(final_path):,} bytes)")

    # VAE output
    vae_path = os.path.join(STREAM_DIR, "vae_output.pt")
    torch.save(vae_output, vae_path)
    stages.append({
        "name": "vae_output",
        "filename": "vae_output.pt",
        "shape": list(vae_output.shape),
        "dtype": str(vae_output.dtype),
    })
    print(f"  Saved vae_output.pt ({os.path.getsize(vae_path):,} bytes)")

    # Manifest
    manifest = {
        "source": "futudiffu-bf16",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "torch_version": torch.__version__,
        "dtype_info": {
            "sampling_loop": "bfloat16",
            "text_encoder": "bfloat16",
            "diffusion_model": "FP8-blockwise + bfloat16",
            "vae": "bfloat16",
            "noise": str(noise.dtype),
            "sigmas": str(sigmas.dtype),
        },
        "config": {
            "prompt": PROMPT,
            "seed": SEED,
            "steps": N_STEPS,
            "cfg": CFG,
            "width": WIDTH,
            "height": HEIGHT,
            "sampling_shift": 1.0,
            "multiplier": 1.0,
            "denoise": 1.0,
            "attention_backend": "sdpa",
        },
        "timing": {
            "trajectory_seconds": round(elapsed_sample, 2),
            "vae_decode_seconds": round(elapsed_vae, 2),
        },
        "stages": stages,
    }

    manifest_path = os.path.join(STREAM_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved manifest.json ({os.path.getsize(manifest_path):,} bytes)")
    print(f"  Save total: {time.perf_counter() - t_save_start:.1f}s")

    # --- Step 8: Self-check round-trip ---
    print("\nSelf-check: reloading saved files and comparing...")
    errors = []

    # Check TE
    pos_reload = torch.load(te_pos_path, map_location="cpu", weights_only=True)
    if not torch.equal(pos_reload, pos_cond):
        errors.append("text_encoder_pos round-trip MISMATCH")
    else:
        print("  text_encoder_pos: bitwise match")

    neg_reload = torch.load(te_neg_path, map_location="cpu", weights_only=True)
    if not torch.equal(neg_reload, neg_cond):
        errors.append("text_encoder_neg round-trip MISMATCH")
    else:
        print("  text_encoder_neg: bitwise match")

    # Check sigmas
    sigmas_reload = torch.load(sigmas_path, map_location="cpu", weights_only=True)
    if not torch.equal(sigmas_reload, sigmas):
        errors.append("sigmas round-trip MISMATCH")
    else:
        print("  sigmas: bitwise match")

    # Check a few euler steps
    for i in [0, 14, 29]:
        step_path = os.path.join(STREAM_DIR, f"euler_step_{i:02d}.pt")
        step_reload = torch.load(step_path, map_location="cpu", weights_only=False)
        original = result[f"step_{i:02d}"]
        if not torch.equal(step_reload["x"], original):
            errors.append(f"euler_step_{i:02d} round-trip MISMATCH")
        else:
            print(f"  euler_step_{i:02d}: bitwise match")

    # Check final
    final_reload = torch.load(final_path, map_location="cpu", weights_only=True)
    if not torch.equal(final_reload, final_latent):
        errors.append("final_latent round-trip MISMATCH")
    else:
        print("  final_latent: bitwise match")

    # Check VAE
    vae_reload = torch.load(vae_path, map_location="cpu", weights_only=True)
    if not torch.equal(vae_reload, vae_output):
        errors.append("vae_output round-trip MISMATCH")
    else:
        print("  vae_output: bitwise match")

    if errors:
        print(f"\nSELF-CHECK FAILED: {len(errors)} errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\nSelf-check PASSED: all files round-trip correctly.")

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"  Stream dir: {STREAM_DIR}")
    total_bytes = 0
    for fname in os.listdir(STREAM_DIR):
        fpath = os.path.join(STREAM_DIR, fname)
        if os.path.isfile(fpath):
            sz = os.path.getsize(fpath)
            total_bytes += sz
    print(f"  Total size: {total_bytes:,} bytes ({total_bytes / 1024 / 1024:.1f} MB)")
    print(f"  Files: {len(os.listdir(STREAM_DIR))}")
    print(f"  Source: futudiffu-bf16")
    print(f"  Torch: {torch.__version__}")
    print(f"  Dtype: bf16 throughout (noise={noise.dtype}, sigmas={sigmas.dtype})")

    client.close()
    print("\nDone.")


if __name__ == "__main__":
    main()

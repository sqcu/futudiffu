"""Reproducibility test: re-run traj_000004 through inference server and compare bitwise.

Three-way comparison:
1. Run A: fresh trajectory from server
2. Run B: second trajectory from same server session (same compiled graphs)
3. Reference: saved trajectory from btrm_dataset

If A==B bitwise but A!=Reference, the divergence is cross-session torch.compile
nondeterminism (expected, not a bug).
"""

import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

import torch
from futudiffu.client import InferenceClient

# ---- Configuration ----
ENDPOINT = "tcp://localhost:5560"
SEED = 402418010
PROMPT = (
    'ahem.\n'
    '*ting ting ting ting ting*\n'
    'the query model for this is a LARGE LANGUAGE MODEL, specifically QWEN-3-4B, '
    'a GENERAL PURPOSE SEMANTIC PARSER which is able to WRITE SENTENCES AT A TIME '
    'when they are participating in dialogue. however, in this situation, they are '
    'being used as a hidden state generator to steer an *image generation model*, '
    'z-image.\n\n'
    'qwen-3-4b, draw me an "enormous laser shark for the sega saturn".'
)
N_STEPS = 30
CFG = 4.0
WIDTH = 1280
HEIGHT = 832
SAVE_STEPS = [0, 4, 9, 14, 19, 24, 29]

TRAJ_DIR = r"F:\dox\repos\ai\futudiffu\btrm_dataset\latents\traj_000004"
RENDER_DIR = r"F:\dox\repos\ai\futudiffu\validation_renders"
ORIG_RENDER = r"F:\dox\repos\ai\futudiffu\btrm_dataset\renders\traj_000004\final.png"

os.makedirs(RENDER_DIR, exist_ok=True)


def cosine_sim(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def compare_tensors(name, reproduced, reference):
    """Compare two tensors and report results."""
    bitwise = torch.equal(reproduced, reference)
    cos = cosine_sim(reproduced, reference)
    max_diff = (reproduced.float() - reference.float()).abs().max().item()
    mean_diff = (reproduced.float() - reference.float()).abs().mean().item()
    return {
        "name": name,
        "bitwise": bitwise,
        "cosine": cos,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "shape": list(reproduced.shape),
    }


def run_trajectory(client, pos_cond, neg_cond, label=""):
    """Run the trajectory and return results dict."""
    print(f"\n--- Running trajectory {label} ---")
    print(f"  seed={SEED}, n_steps={N_STEPS}, cfg={CFG}, {WIDTH}x{HEIGHT}")
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
        save_steps=SAVE_STEPS,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s, keys: {sorted(result.keys())}")
    return result


def print_comparison_table(label, all_results):
    """Print a formatted comparison table."""
    print(f"\n  {label}")
    print(f"  {'Name':<12} {'Bitwise':<10} {'Cosine':<18} {'Max Abs Diff':<16} {'Mean Abs Diff':<16}")
    print("  " + "-" * 72)
    for comp in all_results:
        bw_str = "YES" if comp["bitwise"] else "NO"
        print(f"  {comp['name']:<12} {bw_str:<10} {comp['cosine']:<18.12f} "
              f"{comp['max_abs_diff']:<16.8e} {comp['mean_abs_diff']:<16.8e}")


def main():
    print("=" * 78)
    print("REPRODUCIBILITY TEST: traj_000004 (three-way comparison)")
    print("=" * 78)

    client = InferenceClient(ENDPOINT, timeout_ms=0)

    # 1. Check server status
    status = client.status()
    print(f"\nServer status: {status}")

    # 2. Warmup
    print("\n--- Warmup ---")
    t0 = time.perf_counter()
    client.warmup(attention_backend="sdpa")
    print(f"  Warmup done in {time.perf_counter() - t0:.1f}s")

    # 3. Encode prompts
    print("\n--- Encoding prompts ---")
    t0 = time.perf_counter()
    pos_cond = client.encode_prompt(PROMPT)
    neg_cond = client.encode_prompt("")
    print(f"  pos_cond: {pos_cond.shape}, neg_cond: {neg_cond.shape}")
    print(f"  Encoding done in {time.perf_counter() - t0:.1f}s")

    # 4. Run trajectory TWICE (same compiled graphs, same session)
    result_a = run_trajectory(client, pos_cond, neg_cond, label="Run A")
    result_b = run_trajectory(client, pos_cond, neg_cond, label="Run B")

    # 5. Load reference
    step_names = [f"step_{s:02d}" for s in SAVE_STEPS] + ["final"]
    reference = {}
    for name in step_names:
        ref_path = os.path.join(TRAJ_DIR, f"{name}.pt")
        if os.path.exists(ref_path):
            reference[name] = torch.load(ref_path, map_location="cpu", weights_only=True)

    # 6. Three-way comparison
    print("\n" + "=" * 78)
    print("COMPARISONS")
    print("=" * 78)

    # A vs B (same session, same compiled graphs)
    ab_results = []
    for name in step_names:
        if name in result_a and name in result_b:
            ab_results.append(compare_tensors(name, result_a[name], result_b[name]))
    print_comparison_table("Run A vs Run B (same session reproducibility)", ab_results)

    # A vs Reference (cross-session)
    ar_results = []
    for name in step_names:
        if name in result_a and name in reference:
            ar_results.append(compare_tensors(name, result_a[name], reference[name]))
    print_comparison_table("Run A vs Reference (cross-session)", ar_results)

    # B vs Reference (cross-session)
    br_results = []
    for name in step_names:
        if name in result_b and name in reference:
            br_results.append(compare_tensors(name, result_b[name], reference[name]))
    print_comparison_table("Run B vs Reference (cross-session)", br_results)

    # 7. VAE decode Run A's final latent
    print("\n--- VAE decode ---")
    t0 = time.perf_counter()
    image = client.vae_decode(result_a["final"])
    print(f"  VAE decode done in {time.perf_counter() - t0:.1f}s")
    print(f"  Image shape: {image.shape}")

    # Save reproduced render
    img_np = (image.squeeze(0).permute(1, 2, 0).float().clamp(0, 1).numpy() * 255).astype(np.uint8)
    from PIL import Image as PILImage
    out_path = os.path.join(RENDER_DIR, "reproduced_traj_000004.png")
    PILImage.fromarray(img_np).save(out_path)
    print(f"  Saved to: {out_path}")

    # 8. Compare rendered images if original exists
    if os.path.exists(ORIG_RENDER):
        print("\n--- Pixel comparison with original render ---")
        orig_img = np.array(PILImage.open(ORIG_RENDER))
        if orig_img.shape == img_np.shape:
            pixel_match = np.array_equal(orig_img, img_np)
            pixel_diff = np.abs(orig_img.astype(np.int16) - img_np.astype(np.int16))
            max_pixel_diff = pixel_diff.max()
            mean_pixel_diff = pixel_diff.mean()
            psnr = 10 * np.log10(255**2 / (pixel_diff.astype(np.float64)**2).mean()) if pixel_diff.any() else float('inf')
            print(f"  Pixel-exact match: {pixel_match}")
            print(f"  Max pixel diff: {max_pixel_diff}")
            print(f"  Mean pixel diff: {mean_pixel_diff:.4f}")
            print(f"  PSNR: {psnr:.1f} dB")
        else:
            print(f"  Shape mismatch: orig={orig_img.shape} vs repro={img_np.shape}")
    else:
        print(f"\n  No original render found at {ORIG_RENDER}")

    # 9. Summary
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    ab_all_bitwise = all(r["bitwise"] for r in ab_results)
    ar_all_bitwise = all(r["bitwise"] for r in ar_results)

    print(f"  Run A vs Run B (intra-session): {'ALL BITWISE' if ab_all_bitwise else 'DIVERGENT'}")
    if not ab_all_bitwise:
        failed = [r["name"] for r in ab_results if not r["bitwise"]]
        print(f"    Failed steps: {failed}")
        min_cos = min(r["cosine"] for r in ab_results)
        print(f"    Min cosine: {min_cos:.12f}")

    print(f"  Run A vs Reference (cross-session): {'ALL BITWISE' if ar_all_bitwise else 'DIVERGENT'}")
    if not ar_all_bitwise:
        failed = [r["name"] for r in ar_results if not r["bitwise"]]
        print(f"    Failed steps: {failed}")
        min_cos = min(r["cosine"] for r in ar_results)
        max_diff = max(r["max_abs_diff"] for r in ar_results)
        print(f"    Min cosine: {min_cos:.12f}")
        print(f"    Max abs diff: {max_diff:.8e}")

    if ab_all_bitwise and not ar_all_bitwise:
        print("\n  DIAGNOSIS: Intra-session reproducibility is PERFECT.")
        print("  Cross-session divergence is from torch.compile recompilation")
        print("  (different operator fusion / register allocation / reduction order).")
        print("  This is EXPECTED behavior with torch.compile on CUDA.")

    client.close()
    return 0 if ab_all_bitwise else 1


if __name__ == "__main__":
    sys.exit(main())

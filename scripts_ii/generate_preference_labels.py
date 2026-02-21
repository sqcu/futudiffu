r"""Generate pairwise preference labels for PINKIFY and THISNOTTHAT.

Loads trajectory latents from btrm_dataset/, VAE-decodes each to pixel space,
applies both scoring functions, generates pairwise preferences for all pairs
within each trajectory, and writes results to disk.

Uses src_ii.vae_utils for VAE loading and decoding instead of inlining.

Execution:
  PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe \
      F:\dox\repos\ai\futudiffu\scripts_ii\generate_preference_labels.py

Output:
  pinkify_thisnotthat_output/preference_labels.json
  pinkify_thisnotthat_output/per_image_scores.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from PIL import Image

from src_ii.reward_functions import pinkify_score, thisnotthat_score_gpu, _pil_to_tensor, pairwise_preference
from src_ii.vae_utils import load_vae, decode_latent_to_pil

# --- Configuration ---
N_TRAJECTORIES = 10
VAE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\vae\zimage.safetensors"
THIS_PATH = REPO_ROOT / "i2i_off_policies" / "pizza-ratto.png"
THAT_PATH = REPO_ROOT / "i2i_off_policies" / "offhand_pleometric.png"
OUTPUT_DIR = REPO_ROOT / "pinkify_thisnotthat_output"


def main():
    t0_total = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load reference images for THISNOTTHAT
    print("Loading reference images...")
    this_ref_pil = Image.open(str(THIS_PATH))
    that_ref_pil = Image.open(str(THAT_PATH))
    print(f"  THIS: {THIS_PATH.name} ({this_ref_pil.size})")
    print(f"  THAT: {THAT_PATH.name} ({that_ref_pil.size})")

    # Load manifest
    manifest_path = REPO_ROOT / "btrm_dataset" / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    records = manifest["records"]
    print(f"  Manifest has {len(records)} trajectories, using first {N_TRAJECTORIES}")

    # --- Phase 1: Load VAE and decode all latents ---
    print("\n=== Phase 1: VAE Decode ===")
    device = torch.device("cuda")
    vae = load_vae(VAE_PATH, device=device, dtype=torch.bfloat16)
    print(f"  VAE loaded on CUDA")

    # Convert references to GPU tensors for thisnotthat_score_gpu
    this_ref_t = _pil_to_tensor(this_ref_pil, device)  # (1, 3, H, W)
    that_ref_t = _pil_to_tensor(that_ref_pil, device)

    all_image_scores = []

    for traj_idx in range(N_TRAJECTORIES):
        traj_dir = REPO_ROOT / "btrm_dataset" / "latents" / f"traj_{traj_idx:06d}"
        record = records[traj_idx]

        print(f"\n  Trajectory {traj_idx}: {record['prompt'][:60]}...")

        step_files = sorted(traj_dir.glob("step_*.pt"))
        final_file = traj_dir / "final.pt"

        all_files = [(sf.stem, sf) for sf in step_files]
        if final_file.exists():
            all_files.append(("final", final_file))

        for step_key, pt_path in all_files:
            latent = torch.load(str(pt_path), weights_only=True)
            pil_img = decode_latent_to_pil(vae, latent, device=device, dtype=torch.bfloat16)

            pink_s = pinkify_score(pil_img)
            img_t = _pil_to_tensor(pil_img, device).squeeze(0)  # (3, H, W)
            with torch.no_grad():
                tnt_s = float(thisnotthat_score_gpu(img_t, this_ref_t, that_ref_t).item())

            all_image_scores.append({
                "traj_idx": traj_idx,
                "step_key": step_key,
                "pinkify_score": pink_s,
                "thisnotthat_score": tnt_s,
                "prompt": record["prompt"],
            })

            print(f"    {step_key}: pinkify={pink_s:.6f}, thisnotthat={tnt_s:.6f}")

    del vae
    torch.cuda.empty_cache()
    print(f"\n  VAE freed. Total images scored: {len(all_image_scores)}")

    # --- Phase 2: Generate pairwise preferences ---
    print("\n=== Phase 2: Pairwise Preferences ===")

    traj_scores: dict[int, list[dict]] = {}
    for entry in all_image_scores:
        tidx = entry["traj_idx"]
        if tidx not in traj_scores:
            traj_scores[tidx] = []
        traj_scores[tidx].append(entry)

    preference_labels = []
    stats = {
        "n_trajectories": len(traj_scores),
        "n_images": len(all_image_scores),
        "n_pairs": 0,
        "pinkify_wins_a": 0, "pinkify_wins_b": 0, "pinkify_ties": 0,
        "thisnotthat_wins_a": 0, "thisnotthat_wins_b": 0, "thisnotthat_ties": 0,
    }

    for traj_idx, scores in sorted(traj_scores.items()):
        n = len(scores)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = scores[i], scores[j]
                pink_pref = pairwise_preference(a["pinkify_score"], b["pinkify_score"])
                tnt_pref = pairwise_preference(a["thisnotthat_score"], b["thisnotthat_score"])

                preference_labels.append({
                    "traj_idx": traj_idx,
                    "step_a": a["step_key"], "step_b": b["step_key"],
                    "pinkify_score_a": a["pinkify_score"],
                    "pinkify_score_b": b["pinkify_score"],
                    "pinkify_preference": pink_pref,
                    "thisnotthat_score_a": a["thisnotthat_score"],
                    "thisnotthat_score_b": b["thisnotthat_score"],
                    "thisnotthat_preference": tnt_pref,
                })
                stats["n_pairs"] += 1

                for name, pref in [("pinkify", pink_pref), ("thisnotthat", tnt_pref)]:
                    if pref == 1:
                        stats[f"{name}_wins_a"] += 1
                    elif pref == -1:
                        stats[f"{name}_wins_b"] += 1
                    else:
                        stats[f"{name}_ties"] += 1

    print(f"\n  Generated {stats['n_pairs']} pairwise comparisons")
    print(f"  Pinkify: A={stats['pinkify_wins_a']}, B={stats['pinkify_wins_b']}, ties={stats['pinkify_ties']}")
    print(f"  TNT: A={stats['thisnotthat_wins_a']}, B={stats['thisnotthat_wins_b']}, ties={stats['thisnotthat_ties']}")

    # --- Phase 3: Write outputs ---
    print("\n=== Phase 3: Writing outputs ===")

    with open(OUTPUT_DIR / "per_image_scores.json", "w") as f:
        json.dump(all_image_scores, f, indent=2)

    with open(OUTPUT_DIR / "preference_labels.json", "w") as f:
        json.dump({"stats": stats, "labels": preference_labels}, f, indent=2)

    elapsed = time.perf_counter() - t0_total
    print(f"\n=== Complete in {elapsed:.1f}s ===")


if __name__ == "__main__":
    main()

"""Score ~20 fully denoised V2 BTRM dataset images with the fixed pinkify_score().

Rubric:
  1. Uses ONLY "final" step latents (fully denoised images).
  2. At least 20 images scored.
  3. Scores sorted and printed in a table with trajectory ID and prompt snippet.
  4. Decoded images saved to disk for visual inspection.
  5. Score distribution statistics reported.
  6. Run via .venv/Scripts/python.exe (Windows Python from WSL).

Usage:
    /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe \
        F:\\dox\\repos\\ai\\futudiffu\\scripts\\score_pinkify_v2.py
"""

from __future__ import annotations

import math
import os
import sys
import time

# Windows-native Python -- add src and src_ii to path using Windows paths
sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")
sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src_ii")
sys.path.insert(0, r"F:\dox\repos\ai\futudiffu")

import torch

from futudiffu.dataset_v2 import DatasetReader
from reward_functions import pinkify_score
from vae_utils import decode_latent_to_pil, load_vae

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_DIR = r"F:\dox\repos\ai\futudiffu\btrm_dataset_v2"
VAE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\vae\zimage.safetensors"
OUTPUT_DIR = r"F:\dox\repos\ai\futudiffu\pinkify_test_output\v2_samples"
N_SAMPLES = 25  # >= 20 required by rubric; 25 gives some margin

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAE_DTYPE = torch.bfloat16

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")
print(f"Device: {DEVICE}")
print()

# ---------------------------------------------------------------------------
# Load dataset index
# ---------------------------------------------------------------------------

print("Loading V2 dataset index...")
reader = DatasetReader(DATASET_DIR)
print(f"  {len(reader)} trajectories in dataset")

# Sample trajectories that have a "final" latent.
# All 259 have has_final=True, but let's filter explicitly.
traj_ids_with_final = [
    tid for tid, meta in reader.iter_metadata()
    if meta["has_final"]
]
print(f"  {len(traj_ids_with_final)} trajectories with has_final=True")

import random
rng = random.Random(42)
sampled_ids = rng.sample(traj_ids_with_final, min(N_SAMPLES, len(traj_ids_with_final)))
sampled_ids.sort()
print(f"  Sampling {len(sampled_ids)} trajectories: {sampled_ids[:5]}...")
print()

# ---------------------------------------------------------------------------
# Load VAE
# ---------------------------------------------------------------------------

print(f"Loading VAE from {VAE_PATH} ...")
t0 = time.time()
vae = load_vae(VAE_PATH, device=DEVICE, dtype=VAE_DTYPE)
print(f"  VAE loaded in {time.time() - t0:.1f}s")
print()

# ---------------------------------------------------------------------------
# Score each image
# ---------------------------------------------------------------------------

results = []  # list of (traj_id, prompt_snippet, score)

print(f"{'traj_id':>8}  {'score':>8}  prompt (first 60 chars)")
print("-" * 85)

for traj_id in sampled_ids:
    meta, accessor = reader[traj_id]

    # Load ONLY the "final" latent (fully denoised)
    latent = accessor["final"]  # returns (1, C, H, W) bfloat16

    # VAE decode
    with torch.no_grad():
        pil_image = decode_latent_to_pil(vae, latent, device=DEVICE, dtype=VAE_DTYPE)

    # Save decoded image
    out_path = os.path.join(OUTPUT_DIR, f"traj_{traj_id:06d}_final.png")
    pil_image.save(out_path)

    # Score with fixed pinkify_score
    score = pinkify_score(pil_image)

    prompt = meta.get("prompt", "") or ""
    snippet = prompt[:60].replace("\n", " ")

    results.append((traj_id, snippet, score))
    print(f"{traj_id:>8}  {score:>8.5f}  {snippet}")

print()

# ---------------------------------------------------------------------------
# Sort by score (descending) and print table
# ---------------------------------------------------------------------------

results_sorted = sorted(results, key=lambda r: r[2], reverse=True)

print("=" * 85)
print("SORTED BY SCORE (descending)")
print("=" * 85)
print(f"{'rank':>4}  {'traj_id':>8}  {'score':>8}  prompt")
print("-" * 85)
for rank, (traj_id, snippet, score) in enumerate(results_sorted, 1):
    print(f"{rank:>4}  {traj_id:>8}  {score:>8.5f}  {snippet}")

# ---------------------------------------------------------------------------
# Distribution statistics
# ---------------------------------------------------------------------------

scores = [r[2] for r in results]
n = len(scores)
mean_score = sum(scores) / n
variance = sum((s - mean_score) ** 2 for s in scores) / n
stdev = math.sqrt(variance)
min_score = min(scores)
max_score = max(scores)
scores_sorted_vals = sorted(scores)
median_score = scores_sorted_vals[n // 2] if n % 2 == 1 else (
    (scores_sorted_vals[n // 2 - 1] + scores_sorted_vals[n // 2]) / 2
)

ratio_max_median = max_score / median_score if median_score > 1e-10 else float("inf")
ratio_max_min = max_score / min_score if min_score > 1e-10 else float("inf")

print()
print("=" * 85)
print("SCORE DISTRIBUTION STATISTICS")
print("=" * 85)
print(f"  N images scored  : {n}")
print(f"  Min              : {min_score:.5f}  (traj {results_sorted[-1][0]})")
print(f"  Max              : {max_score:.5f}  (traj {results_sorted[0][0]})")
print(f"  Median           : {median_score:.5f}")
print(f"  Mean             : {mean_score:.5f}")
print(f"  Stdev            : {stdev:.5f}")
print(f"  Max / Min ratio  : {ratio_max_min:.2f}x")
print(f"  Max / Median     : {ratio_max_median:.2f}x")
print()

# Old scoring function had mean=0.017, max=0.084 (range ~0.084)
# New function should produce wider separation
old_range = 0.084
new_range = max_score - min_score
print(f"  Score range      : {new_range:.5f}")
print(f"  Old range (ref)  : {old_range:.5f}")
if new_range > old_range:
    print(f"  -> Range is {new_range / old_range:.1f}x WIDER than old function (separation improved)")
else:
    print(f"  -> Range is {new_range / old_range:.2f}x of old range (separation narrower or comparable)")

print()
print(f"Images saved to: {OUTPUT_DIR}")
reader.close()

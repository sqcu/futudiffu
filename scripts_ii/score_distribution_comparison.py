r"""Compare BTRM head scores against literal rule function scores.

Uses src_ii.stats.spearman_rank_correlation instead of inlining.

Execution:
  PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe \
      F:\dox\repos\ai\futudiffu\scripts_ii\score_distribution_comparison.py

Output:
  pinkify_thisnotthat_output/score_distribution_comparison.json
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

from src_ii.stats import spearman_rank_correlation

OUTPUT_DIR = REPO_ROOT / "pinkify_thisnotthat_output"


def main():
    t0 = time.perf_counter()

    print("Loading literal rule scores...")
    with open(OUTPUT_DIR / "per_image_scores.json") as f:
        per_image_scores = json.load(f)
    n_images = len(per_image_scores)
    print(f"  {n_images} images")

    print("Loading BTRM head scores...")
    btrm_scores = torch.load(str(OUTPUT_DIR / "pre_persist_scores.pt"), weights_only=True)
    print(f"  Shape: {btrm_scores.shape}")

    print("\nBuilding comparison...")
    per_image_comparison = []
    for i in range(n_images):
        entry = per_image_scores[i]
        per_image_comparison.append({
            "traj_idx": entry["traj_idx"],
            "step_key": entry["step_key"],
            "literal_pinkify": entry["pinkify_score"],
            "literal_thisnotthat": entry["thisnotthat_score"],
            "btrm_pinkify": btrm_scores[i, 0].item(),
            "btrm_thisnotthat": btrm_scores[i, 1].item(),
        })

    print("Computing pairwise agreement...")
    traj_images: dict[int, list[int]] = {}
    for i, entry in enumerate(per_image_scores):
        tidx = entry["traj_idx"]
        if tidx not in traj_images:
            traj_images[tidx] = []
        traj_images[tidx].append(i)

    n_pairs = 0
    agree_pinkify = 0
    agree_thisnotthat = 0
    agree_both = 0
    disagree_pinkify_details = []
    disagree_thisnotthat_details = []

    for traj_idx, indices in sorted(traj_images.items()):
        for a_pos, a_idx in enumerate(indices):
            for b_pos in range(a_pos + 1, len(indices)):
                b_idx = indices[b_pos]

                lit_pink_a = per_image_scores[a_idx]["pinkify_score"]
                lit_pink_b = per_image_scores[b_idx]["pinkify_score"]
                lit_tnt_a = per_image_scores[a_idx]["thisnotthat_score"]
                lit_tnt_b = per_image_scores[b_idx]["thisnotthat_score"]

                lit_pink_pref = 1 if lit_pink_a > lit_pink_b else (-1 if lit_pink_a < lit_pink_b else 0)
                lit_tnt_pref = 1 if lit_tnt_a > lit_tnt_b else (-1 if lit_tnt_a < lit_tnt_b else 0)

                btrm_pink_a = btrm_scores[a_idx, 0].item()
                btrm_pink_b = btrm_scores[b_idx, 0].item()
                btrm_tnt_a = btrm_scores[a_idx, 1].item()
                btrm_tnt_b = btrm_scores[b_idx, 1].item()

                btrm_pink_pref = 1 if btrm_pink_a > btrm_pink_b else (-1 if btrm_pink_a < btrm_pink_b else 0)
                btrm_tnt_pref = 1 if btrm_tnt_a > btrm_tnt_b else (-1 if btrm_tnt_a < btrm_tnt_b else 0)

                n_pairs += 1
                pink_agree = (lit_pink_pref == btrm_pink_pref)
                tnt_agree = (lit_tnt_pref == btrm_tnt_pref)

                if pink_agree:
                    agree_pinkify += 1
                else:
                    disagree_pinkify_details.append({
                        "traj_idx": traj_idx,
                        "step_a": per_image_scores[a_idx]["step_key"],
                        "step_b": per_image_scores[b_idx]["step_key"],
                        "literal_pref": lit_pink_pref, "btrm_pref": btrm_pink_pref,
                    })

                if tnt_agree:
                    agree_thisnotthat += 1
                else:
                    disagree_thisnotthat_details.append({
                        "traj_idx": traj_idx,
                        "step_a": per_image_scores[a_idx]["step_key"],
                        "step_b": per_image_scores[b_idx]["step_key"],
                        "literal_pref": lit_tnt_pref, "btrm_pref": btrm_tnt_pref,
                    })

                if pink_agree and tnt_agree:
                    agree_both += 1

    pinkify_agreement = agree_pinkify / max(n_pairs, 1)
    thisnotthat_agreement = agree_thisnotthat / max(n_pairs, 1)
    both_agreement = agree_both / max(n_pairs, 1)

    print(f"\n  Total pairs: {n_pairs}")
    print(f"  Pinkify agreement:     {agree_pinkify}/{n_pairs} = {pinkify_agreement:.1%}")
    print(f"  ThisNotThat agreement: {agree_thisnotthat}/{n_pairs} = {thisnotthat_agreement:.1%}")
    print(f"  Both agree:            {agree_both}/{n_pairs} = {both_agreement:.1%}")

    print("\nComputing rank correlations...")
    literal_pink = [e["pinkify_score"] for e in per_image_scores]
    literal_tnt = [e["thisnotthat_score"] for e in per_image_scores]
    btrm_pink = btrm_scores[:, 0].tolist()
    btrm_tnt = btrm_scores[:, 1].tolist()

    rho_pinkify = spearman_rank_correlation(literal_pink, btrm_pink)
    rho_thisnotthat = spearman_rank_correlation(literal_tnt, btrm_tnt)

    print(f"  Spearman rho (pinkify):     {rho_pinkify:.4f}")
    print(f"  Spearman rho (thisnotthat): {rho_thisnotthat:.4f}")

    comparison = {
        "summary": {
            "n_images": n_images, "n_pairs": n_pairs,
            "pinkify_agreement": pinkify_agreement,
            "thisnotthat_agreement": thisnotthat_agreement,
            "both_agreement": both_agreement,
            "spearman_pinkify": rho_pinkify,
            "spearman_thisnotthat": rho_thisnotthat,
        },
        "per_image": per_image_comparison,
        "disagreements": {
            "pinkify": disagree_pinkify_details[:20],
            "thisnotthat": disagree_thisnotthat_details[:20],
        },
    }

    out_path = OUTPUT_DIR / "score_distribution_comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)

    elapsed = time.perf_counter() - t0
    print(f"\n=== Comparison complete in {elapsed:.1f}s ===")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()

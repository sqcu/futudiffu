r"""Regenerate funfetti diagnostic charts (06-10) from existing training data.

The 100-step funfetti training run completed successfully, but charts 06-10
were not generated because of a bug: the per-step funfetti metadata was
computed in the training loop but not passed through to TrainingArtifacts
via extra_metrics. The bug has since been fixed in btrm_training.py.

This script reconstructs the funfetti metadata by replaying the pair sampler
with the same RNG seed (42) and bin packer configuration used during training.
The sampler is deterministic: same seed + same dataset + same calling pattern
= same pairs. We then inject the reconstructed metadata into the JSONL data
and call generate_analysis() to produce all 10 charts.

No GPU needed -- this is pure data processing.

Execution:
  /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe ^
      F:\dox\repos\ai\futudiffu\scripts_ii\regenerate_funfetti_charts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

OUTPUT_DIR = REPO_ROOT / "funfetti_100step_output"
DATASET_DIR = REPO_ROOT / "multi_res_trajectories"
JSONL_PATH = OUTPUT_DIR / "training_metrics.jsonl"

N_STEPS = 100
PAIRS_PER_PACK = 2
GRAD_ACCUM = 2
HEAD_NAMES = ("pinkify", "thisnotthat")
RUN_NAME = "funfetti_100step"
RNG_SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file."""
    rows = []
    with open(str(path)) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    print("=" * 70)
    print("  REGENERATE FUNFETTI CHARTS (06-10)")
    print("=" * 70)

    print(f"\n  Loading JSONL from {JSONL_PATH}")
    data = load_jsonl(JSONL_PATH)
    print(f"  Loaded {len(data)} step entries")

    if len(data) != N_STEPS:
        print(f"  WARNING: Expected {N_STEPS} entries, got {len(data)}")

    has_funfetti = any("funfetti" in d for d in data)
    if has_funfetti:
        print("  Funfetti data already present in JSONL -- will regenerate charts directly")
    else:
        print("  No funfetti data in JSONL -- reconstructing from pair sampler replay")

    if not has_funfetti:
        print(f"\n  Loading dataset from {DATASET_DIR}")

        from futudiffu.dataset_v2 import DatasetReader
        from src_ii.pair_sampler import BTRMPairSampler, build_positions_from_v2
        from src_ii.flops_sampling import compute_flops_sampling_weights_from_positions
        from src_ii.bin_packer import BinPackScheduler, compute_effective_seq_len

        reader = DatasetReader(str(DATASET_DIR))
        n_available = len(reader)
        print(f"  Dataset: {n_available} trajectories")

        traj_ids = list(range(n_available))
        positions = build_positions_from_v2(reader, traj_ids=traj_ids)
        print(f"  Positions: {len(positions)} across {len(traj_ids)} trajectories")

        pos_lookup = {}
        for pos in positions:
            pos_lookup[(pos.traj_id, pos.step_key)] = (pos.width, pos.height)

        flops_weights = compute_flops_sampling_weights_from_positions(positions)

        sampler = BTRMPairSampler(
            positions=positions,
            allow_inter_trajectory=True,
            allow_intra_trajectory=True,
            rng_seed=RNG_SEED,
            flops_weights=flops_weights,
        )
        print(f"  Pair space: {sampler.pair_space_size:,} possible pairs")

        print(f"\n  Replaying {N_STEPS} steps x {GRAD_ACCUM} microbatches "
              f"x {PAIRS_PER_PACK} pairs = {N_STEPS * GRAD_ACCUM * PAIRS_PER_PACK} total pairs")

        for step_idx in range(N_STEPS):
            step_microbatch_meta = []

            for micro in range(GRAD_ACCUM):
                image_resolutions = []

                for k in range(PAIRS_PER_PACK):
                    pair = sampler.sample_pair()

                    key_a = (pair["traj_a"], pair["step_a"])
                    key_b = (pair["traj_b"], pair["step_b"])

                    w_a, h_a = pos_lookup[key_a]
                    w_b, h_b = pos_lookup[key_b]

                    image_resolutions.append((w_a, h_a))
                    image_resolutions.append((w_b, h_b))

                packer = BinPackScheduler()
                pack_items = []
                for img_idx, (w, h) in enumerate(image_resolutions):
                    seq_len = compute_effective_seq_len(w, h, packer.default_cap_tokens)
                    pack_items.append({
                        "img_idx": img_idx,
                        "seq_len": seq_len,
                        "width": w,
                        "height": h,
                    })

                bins = packer.pack(pack_items)

                micro_meta = {
                    "n_pairs": PAIRS_PER_PACK,
                    "n_images": len(image_resolutions),
                    "n_bins": len(bins),
                    "per_bin_item_count": [len(b) for b in bins],
                    "per_bin_context_len": [
                        sum(item["seq_len"] for item in b) for b in bins
                    ],
                    "image_resolutions": [
                        {"width": w, "height": h, "pixels": w * h}
                        for w, h in image_resolutions
                    ],
                    "total_context_len": sum(
                        sum(item["seq_len"] for item in b) for b in bins
                    ),
                }
                step_microbatch_meta.append(micro_meta)

            total_pairs = sum(m["n_pairs"] for m in step_microbatch_meta)
            total_ctx = sum(m["total_context_len"] for m in step_microbatch_meta)
            total_nfes = sum(m["n_images"] for m in step_microbatch_meta)
            all_resolutions = []
            for m in step_microbatch_meta:
                all_resolutions.extend(m["image_resolutions"])

            funfetti_meta = {
                "n_microbatches": len(step_microbatch_meta),
                "total_pairs": total_pairs,
                "total_context_len": total_ctx,
                "total_nfes": total_nfes,
                "resolutions": all_resolutions,
                "microbatches": step_microbatch_meta,
            }

            if step_idx < len(data):
                data[step_idx]["funfetti"] = funfetti_meta

        reader.close()

        print(f"  Sampler stats after replay: {sampler.stats()}")
        print(f"  Injected funfetti metadata into {min(N_STEPS, len(data))} entries")

    n_with_funfetti = sum(1 for d in data if "funfetti" in d)
    print(f"\n  Entries with funfetti data: {n_with_funfetti}/{len(data)}")

    if n_with_funfetti == 0:
        print("  ERROR: No funfetti data available. Cannot generate charts.")
        return 1

    from collections import Counter
    pixel_counts = Counter()
    for d in data:
        fm = d.get("funfetti")
        if fm:
            for res in fm.get("resolutions", []):
                pixel_counts[res["pixels"]] += 1
    total_images = sum(pixel_counts.values())
    print(f"\n  Resolution distribution across all steps:")
    for px in sorted(pixel_counts.keys()):
        ct = pixel_counts[px]
        print(f"    {px:>10,} px: {ct:>4} images ({ct/total_images:.1%})")

    print(f"\n  Generating analysis (all 10 charts)...")

    from src_ii.training_artifacts import TrainingArtifacts

    artifacts = TrainingArtifacts(
        output_dir=str(OUTPUT_DIR),
        run_name=RUN_NAME,
        head_names=HEAD_NAMES,
    )

    artifacts.close()

    artifacts._steps = data

    summary_path = OUTPUT_DIR / "run_summary.json"
    run_config = None
    if summary_path.exists():
        with open(str(summary_path)) as f:
            summary = json.load(f)
        run_config = {
            "mode": "funfetti_packed",
            "n_steps": summary.get("n_steps", N_STEPS),
            "pairs_per_pack": summary.get("pairs_per_pack", PAIRS_PER_PACK),
            "grad_accum_steps": summary.get("grad_accum_steps", GRAD_ACCUM),
            "lr": summary.get("lr"),
            "lr_schedule": summary.get("lr_schedule"),
            "grad_clip": summary.get("grad_clip"),
            "n_trajectories": summary.get("n_trajectories"),
            "resolution_dist": summary.get("resolution_dist"),
        }

    report_path = artifacts.generate_analysis(run_config=run_config)
    print(f"  Analysis generated: {report_path}")

    charts_dir = OUTPUT_DIR / "charts"
    expected_charts = [
        "01_loss_curve.png",
        "02_per_head_accuracy.png",
        "03_gradient_norms.png",
        "04_learning_rate.png",
        "05_step_timing.png",
        "06_resolution_pdf.png",
        "07_aspect_ratio_pdf.png",
        "08_metrics_by_resolution.png",
        "09_microbatch_pairs.png",
        "10_context_length.png",
    ]

    print(f"\n  Verifying charts in {charts_dir}:")
    all_present = True
    for chart_name in expected_charts:
        chart_path = charts_dir / chart_name
        if chart_path.exists():
            size_kb = chart_path.stat().st_size / 1024
            print(f"    OK  {chart_name} ({size_kb:.1f} KB)")
        else:
            print(f"    MISSING  {chart_name}")
            all_present = False

    if all_present:
        print(f"\n  All 10 charts generated successfully.")
    else:
        print(f"\n  WARNING: Some charts are missing!")

    print(f"\n{'=' * 70}")
    print(f"  REGENERATION COMPLETE")
    print(f"{'=' * 70}")

    return 0 if all_present else 1


if __name__ == "__main__":
    sys.exit(main())

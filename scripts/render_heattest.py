"""Render heattest dataset: VAE-decode all step latents + finals to PNG.

GPU phase -- reads v2 dataset, decodes via inference server, saves PNGs
and a manifest.json that the CPU-only analyze_heattest.py consumes.

Requires a running inference server:
    python -m futudiffu.server --port 5555 --fp8-diff ... --te ... --vae ...

Usage:
    .venv/Scripts/python.exe 'F:\\dox\\repos\\ai\\futudiffu\\scripts\\render_heattest.py'
    .venv/Scripts/python.exe 'F:\\dox\\repos\\ai\\futudiffu\\scripts\\render_heattest.py' \\
        --dataset-dir PATH --output-dir PATH
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

import torch

from futudiffu.client import InferenceClient
from futudiffu.dataset_v2 import DatasetReader
from futudiffu.rendering import decode_and_save
from futudiffu.sampling import build_sigma_schedule


def build_manifest_entry(meta: dict, traj_id: int, rendered_steps: list[str],
                         output_subdir: str) -> dict:
    """Build a manifest entry for one trajectory (no torch dependency)."""
    # Pre-compute sigma schedule so analyze_heattest never needs torch
    n_steps = meta["n_steps"]
    denoise = meta.get("denoise") or 1.0
    sigmas = build_sigma_schedule(
        n_steps=n_steps,
        denoise=denoise,
        dtype=torch.float32,
    )
    sigma_list = sigmas.tolist()

    return {
        "traj_id": traj_id,
        "output_dir": output_subdir,
        "rendered_steps": rendered_steps,
        "prompt": meta.get("prompt", ""),
        "seed": meta.get("seed"),
        "n_steps": n_steps,
        "denoise": denoise,
        "attention_backend": meta.get("attention_backend"),
        "batch_type": meta.get("batch_type"),
        "width": meta.get("width"),
        "height": meta.get("height"),
        "image_file": meta.get("image_file"),
        "parent_traj_id": meta.get("parent_traj_id"),
        "parent_step": meta.get("parent_step"),
        "parent_denoise": meta.get("parent_denoise"),
        "sigmas": sigma_list,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Render heattest v2 dataset to PNGs via inference server")
    parser.add_argument("--dataset-dir", type=str,
                        default=r"F:\dox\repos\ai\futudiffu\btrm_dataset_v2_heattest")
    parser.add_argument("--output-dir", type=str,
                        default=r"F:\dox\repos\ai\futudiffu\heattest_renders")
    parser.add_argument("--server", type=str, default="tcp://localhost:5555")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open dataset
    reader = DatasetReader(str(dataset_dir))
    n_traj = len(reader)
    print(f"Dataset: {dataset_dir}")
    print(f"Trajectories: {n_traj}")
    print(f"Output: {output_dir}")

    if n_traj == 0:
        print("No trajectories to render.")
        return 0

    # Connect to server
    client = InferenceClient(args.server)
    try:
        status = client.status()
        print(f"Server: {status.get('loaded_models', [])} loaded, "
              f"VRAM {status.get('vram_allocated_gb', '?')}GB")
    except Exception as e:
        print(f"Cannot connect to server at {args.server}: {e}")
        return 1

    # Iterate all trajectories
    manifest_entries = []
    total_pngs = 0
    wall_start = time.perf_counter()

    # Get all traj_ids from the dataset
    all_traj_ids = sorted(
        tid for tid, _ in reader.iter_metadata()
    )

    for i, traj_id in enumerate(all_traj_ids):
        meta, accessor = reader[traj_id]
        subdir = f"traj_{traj_id:06d}"
        traj_out = output_dir / subdir
        available = accessor.available_steps

        t0 = time.perf_counter()
        rendered_steps = []

        for step_label in available:
            latent = accessor[step_label]
            png_path = traj_out / f"{step_label}.png"
            decode_and_save(client, latent, png_path)
            rendered_steps.append(step_label)
            total_pngs += 1

        elapsed = time.perf_counter() - t0
        batch_type = meta.get("batch_type", "?")
        parent = meta.get("parent_traj_id")
        parent_info = f" parent={parent}" if parent is not None else ""
        print(f"  [{i+1}/{n_traj}] traj_{traj_id:06d} ({batch_type}{parent_info}): "
              f"{len(rendered_steps)} PNGs ({elapsed:.1f}s)")

        entry = build_manifest_entry(meta, traj_id, rendered_steps, subdir)
        manifest_entries.append(entry)

    reader.close()
    client.close()

    # Write manifest
    manifest = {
        "dataset_dir": str(dataset_dir),
        "n_trajectories": len(manifest_entries),
        "total_pngs": total_pngs,
        "trajectories": manifest_entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    wall_total = time.perf_counter() - wall_start
    print(f"\nDone: {total_pngs} PNGs from {n_traj} trajectories in {wall_total:.1f}s")
    print(f"Manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

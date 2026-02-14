"""Render existing trajectories to PNG via the inference server.

Walks a BTRM dataset output dir, finds trajectories matching filters,
and VAE-decodes their latents to images. Connects to the same inference
server used for generation.

Usage:
    # Render all i2i trajectories (final only):
    python render_trajectories.py --dataset-dir btrm_dataset --type i2i

    # Render all trajectories, including checkpoint steps:
    python render_trajectories.py --dataset-dir btrm_dataset --steps all

    # Render specific trajectories by index:
    python render_trajectories.py --dataset-dir btrm_dataset --traj 4 5 12 13

    # Render trajectories that don't already have renders:
    python render_trajectories.py --dataset-dir btrm_dataset --missing-only

    # Render only step checkpoints (not final):
    python render_trajectories.py --dataset-dir btrm_dataset --steps 0 14 29
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

import numpy as np
import torch
from PIL import Image

from futudiffu.client import InferenceClient


def find_trajectories(dataset_dir: Path, type_filter: str | None,
                      traj_indices: list[int] | None,
                      missing_only: bool) -> list[Path]:
    """Find trajectory dirs matching filters."""
    latents_dir = dataset_dir / "latents"
    if not latents_dir.exists():
        print(f"No latents directory at {latents_dir}")
        return []

    traj_dirs = sorted(
        d for d in latents_dir.iterdir()
        if d.is_dir() and d.name.startswith("traj_")
    )

    result = []
    for d in traj_dirs:
        meta_file = d / "meta.json"
        if not meta_file.exists():
            continue

        # Filter by index
        if traj_indices is not None:
            idx = int(d.name.split("_")[1])
            if idx not in traj_indices:
                continue

        # Filter by type
        if type_filter is not None:
            meta = json.loads(meta_file.read_text())
            if meta.get("type") != type_filter:
                continue

        # Filter by missing renders
        if missing_only:
            render_dir = dataset_dir / "renders" / d.name
            if (render_dir / "final.png").exists():
                continue

        result.append(d)

    return result


def render_latent(client: InferenceClient, latent_path: Path,
                  output_path: Path) -> None:
    """VAE-decode a .pt latent file and save as PNG."""
    latent = torch.load(str(latent_path), map_location="cpu", weights_only=True)
    image = client.vae_decode(latent)
    image_np = (image[0].permute(1, 2, 0).float().numpy() * 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_np).save(str(output_path))


def render_trajectory(client: InferenceClient, traj_dir: Path,
                      render_dir: Path, steps: list[str]) -> int:
    """Render selected latents from a trajectory. Returns number of PNGs written."""
    count = 0

    if "all" in steps or "final" in steps:
        final_pt = traj_dir / "final.pt"
        if final_pt.exists():
            render_latent(client, final_pt, render_dir / "final.png")
            count += 1

    # Step checkpoints
    if "all" in steps:
        # Render every step_*.pt file found
        for pt_file in sorted(traj_dir.glob("step_*.pt")):
            png_name = pt_file.stem + ".png"
            render_latent(client, pt_file, render_dir / png_name)
            count += 1
    else:
        # Render specific numbered steps
        for s in steps:
            if s == "final":
                continue
            try:
                step_idx = int(s)
            except ValueError:
                continue
            pt_file = traj_dir / f"step_{step_idx:02d}.pt"
            if pt_file.exists():
                render_latent(client, pt_file, render_dir / f"step_{step_idx:02d}.png")
                count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Render existing BTRM trajectories to PNG via inference server")
    parser.add_argument("--dataset-dir", type=str, required=True,
                        help="Path to BTRM dataset output directory")
    parser.add_argument("--server", type=str, default="tcp://localhost:5555",
                        help="Inference server endpoint")
    parser.add_argument("--type", type=str, default=None, choices=["t2i", "i2i"],
                        help="Only render trajectories of this type")
    parser.add_argument("--traj", type=int, nargs="+", default=None,
                        help="Only render these trajectory indices")
    parser.add_argument("--missing-only", action="store_true",
                        help="Only render trajectories that don't have renders yet")
    parser.add_argument("--steps", nargs="+", default=["final"],
                        help="Which latents to render: 'final', 'all', or step numbers (e.g. 0 14 29)")

    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)

    traj_dirs = find_trajectories(
        dataset_dir,
        type_filter=args.type,
        traj_indices=set(args.traj) if args.traj else None,
        missing_only=args.missing_only,
    )

    if not traj_dirs:
        print("No trajectories match the given filters.")
        return 0

    print(f"Found {len(traj_dirs)} trajectories to render")
    print(f"Steps: {args.steps}")

    with InferenceClient(args.server) as client:
        status = client.status()
        print(f"Server: {status.get('loaded_models', [])} loaded, "
              f"VRAM {status.get('vram_allocated_gb', '?')}GB")

        total_pngs = 0
        t_start = time.perf_counter()

        for i, traj_dir in enumerate(traj_dirs):
            render_dir = dataset_dir / "renders" / traj_dir.name
            meta = json.loads((traj_dir / "meta.json").read_text())
            traj_type = meta.get("type", "?")
            label = f"{traj_dir.name} ({traj_type})"

            t0 = time.perf_counter()
            n = render_trajectory(client, traj_dir, render_dir, args.steps)
            elapsed = time.perf_counter() - t0
            total_pngs += n
            print(f"  [{i+1}/{len(traj_dirs)}] {label}: {n} PNGs ({elapsed:.1f}s)")

        wall = time.perf_counter() - t_start
        print(f"\nDone: {total_pngs} PNGs from {len(traj_dirs)} trajectories in {wall:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())

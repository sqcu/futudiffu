"""Pack trajectory directories into safetensors archives + JSONL manifest.

Converts the per-trajectory directory format:
    btrm_dataset/latents/traj_NNNNNN/{step_00.pt, ..., final.pt, meta.json}
into HuggingFace-friendly archives:
    output_dir/{traj_000000.safetensors, ..., manifest.jsonl}
"""
import sys
sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

import argparse
import json
import os
import re
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file


def discover_trajectories(input_dir: Path) -> list[Path]:
    """Find all traj_NNNNNN directories sorted by index."""
    latents_dir = input_dir / "latents"
    if not latents_dir.exists():
        print(f"ERROR: {latents_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    dirs = sorted(
        [d for d in latents_dir.iterdir() if d.is_dir() and re.match(r"traj_\d{6}", d.name)],
        key=lambda d: d.name,
    )
    return dirs


def load_meta(traj_dir: Path) -> dict:
    """Load meta.json from a trajectory directory."""
    meta_path = traj_dir / "meta.json"
    with open(meta_path) as f:
        return json.load(f)


def discover_checkpoints(traj_dir: Path) -> list[str]:
    """Find checkpoint names (step_XX and final) in sorted order."""
    names = []
    for f in sorted(traj_dir.iterdir()):
        if f.suffix == ".pt":
            names.append(f.stem)  # "step_00", "step_04", ..., "final"
    # Sort: step_XX by number, final last
    steps = sorted([n for n in names if n.startswith("step_")], key=lambda s: int(s.split("_")[1]))
    finals = [n for n in names if n == "final"]
    return steps + finals


def pack_one(traj_dir: Path, output_path: Path, meta: dict) -> list[str]:
    """Pack one trajectory into a safetensors file. Returns checkpoint names."""
    checkpoints = discover_checkpoints(traj_dir)
    tensors = {}
    for ckpt_name in checkpoints:
        t = torch.load(traj_dir / f"{ckpt_name}.pt", map_location="cpu", weights_only=True)
        # Squeeze batch dimension: (1, 16, H, W) -> (16, H, W)
        if t.dim() == 4 and t.shape[0] == 1:
            t = t.squeeze(0)
        tensors[ckpt_name] = t

    # Safetensors metadata is string->string; stringify all meta fields
    sf_meta = {k: str(v) for k, v in meta.items()}
    save_file(tensors, str(output_path), metadata=sf_meta)
    return checkpoints


def verify_one(traj_dir: Path, output_path: Path, checkpoints: list[str]):
    """Round-trip verify: load safetensors back and compare against original .pt files."""
    loaded = load_file(str(output_path))
    for ckpt_name in checkpoints:
        original = torch.load(traj_dir / f"{ckpt_name}.pt", map_location="cpu", weights_only=True)
        if original.dim() == 4 and original.shape[0] == 1:
            original = original.squeeze(0)
        packed = loaded[ckpt_name]
        if not torch.equal(original, packed):
            raise AssertionError(
                f"Round-trip mismatch for {traj_dir.name}/{ckpt_name}: "
                f"original {original.shape} {original.dtype} vs packed {packed.shape} {packed.dtype}"
            )


def build_jsonl_record(traj_id: int, filename: str, meta: dict, checkpoints: list[str]) -> dict:
    """Build a typed JSONL record from meta.json fields + checkpoint list."""
    record = {
        "traj_id": traj_id,
        "file": filename,
        "type": meta.get("type", "t2i"),
        "seed": meta.get("seed"),
        "prompt": meta.get("prompt", ""),
        "n_steps": meta.get("n_steps"),
        "precision": meta.get("precision", "sdpa"),
        "checkpoints": checkpoints,
    }
    # Optional fields present in some trajectory types
    if "prompt_idx" in meta:
        record["prompt_idx"] = meta["prompt_idx"]
    if "denoise" in meta:
        record["denoise"] = meta["denoise"]
    if "image_file" in meta:
        record["image_file"] = meta["image_file"]
    if "output_width" in meta:
        record["output_width"] = meta["output_width"]
    if "output_height" in meta:
        record["output_height"] = meta["output_height"]
    return record


def main():
    parser = argparse.ArgumentParser(description="Pack trajectory directories into safetensors + JSONL")
    parser.add_argument("--input", required=True, help="Input dataset directory (contains latents/)")
    parser.add_argument("--output", required=True, help="Output directory for packed files")
    parser.add_argument("--verify-all", action="store_true", help="Round-trip verify every trajectory (slow)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    traj_dirs = discover_trajectories(input_dir)
    n_total = len(traj_dirs)
    if n_total == 0:
        print("No trajectories found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {n_total} trajectories in {input_dir / 'latents'}", file=sys.stderr)

    manifest_path = output_dir / "manifest.jsonl"
    # Load existing manifest to allow appending on re-runs
    existing_files = set()
    existing_records = []
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    existing_files.add(rec["file"])
                    existing_records.append(rec)

    n_skipped = 0
    n_packed = 0
    n_verified = 0
    records = list(existing_records)
    first_verify_done = False

    for i, traj_dir in enumerate(traj_dirs):
        # Extract trajectory index from directory name
        traj_id = int(traj_dir.name.split("_")[1])
        filename = f"traj_{traj_id:06d}.safetensors"
        output_path = output_dir / filename

        # Skip if already packed
        if filename in existing_files and output_path.exists():
            n_skipped += 1
            continue

        meta = load_meta(traj_dir)
        checkpoints = pack_one(traj_dir, output_path, meta)
        n_packed += 1

        # Verify: always verify the first one, plus all if --verify-all
        if args.verify_all or not first_verify_done:
            verify_one(traj_dir, output_path, checkpoints)
            n_verified += 1
            first_verify_done = True

        record = build_jsonl_record(traj_id, filename, meta, checkpoints)
        records.append(record)

        print(f"\r  packed {n_packed}/{n_total - n_skipped} (skipped {n_skipped})", end="", file=sys.stderr)

    print(file=sys.stderr)  # newline after progress

    # Write manifest (sorted by traj_id for determinism)
    records.sort(key=lambda r: r["traj_id"])
    # Deduplicate by traj_id (in case of partial re-runs)
    seen_ids = set()
    deduped = []
    for r in records:
        if r["traj_id"] not in seen_ids:
            seen_ids.add(r["traj_id"])
            deduped.append(r)
    records = deduped

    with open(manifest_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Done. {n_packed} packed, {n_skipped} skipped, {n_verified} verified.", file=sys.stderr)
    print(f"Manifest: {manifest_path} ({len(records)} entries)", file=sys.stderr)


if __name__ == "__main__":
    main()

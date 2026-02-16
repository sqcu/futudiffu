"""Migrate BTRM dataset from v1 (per-directory .pt) to v2 (parquet + safetensors blobs).

Reads the v1 format (manifest.json + latents/traj_NNNNNN/) and writes v2
format using DatasetWriter. Optionally verifies bitwise round-trip fidelity.

Usage (from WSL):
    .venv/Scripts/python.exe scripts/migrate_v1_to_v2.py \
        --v1-dir F:\dox\repos\ai\futudiffu\btrm_dataset \
        --v2-dir F:\dox\repos\ai\futudiffu\btrm_dataset_v2 \
        --verify
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

import torch

from futudiffu.dataset_v2 import DatasetWriter, DatasetReader


# ---------------------------------------------------------------------------
# V1 field mapping -> V2 metadata
# ---------------------------------------------------------------------------

def _v1_meta_to_v2(meta: dict, traj_dir: Path) -> dict:
    """Convert a v1 meta.json dict to v2 metadata dict.

    Follows the field mapping table from the spec (Section 10).
    """
    v2 = {}
    v2["prompt"] = meta["prompt"]
    v2["prompt_idx"] = meta.get("prompt_idx", -1)
    v2["seed"] = meta["seed"]
    v2["cfg"] = meta.get("cfg", 4.0)                  # Not in v1; use GENERATION_DEFAULTS
    v2["width"] = meta.get("output_width", 1280)       # i2i has output_width
    v2["height"] = meta.get("output_height", 832)      # i2i has output_height
    v2["n_steps"] = meta["n_steps"]
    v2["attention_backend"] = meta["precision"]         # Rename: precision -> attention_backend
    v2["batch_type"] = meta["type"]                     # Rename: type -> batch_type
    v2["denoise"] = meta.get("denoise")                 # None for t2i
    v2["image_file"] = meta.get("image_file")           # None for t2i
    v2["is_gold"] = (meta["precision"] == "sdpa" and meta["n_steps"] == 30)
    v2["batch_idx"] = meta.get("batch_idx", 0)
    v2["packed"] = meta.get("packed", False)
    v2["timing_seconds"] = None                         # v1 does not record timing

    return v2


def _load_v1_tensors(traj_dir: Path) -> dict[str, torch.Tensor]:
    """Load all .pt tensor files from a v1 trajectory directory.

    Returns a dict mapping step labels to tensors.
    """
    tensors = {}
    for fname in sorted(os.listdir(traj_dir)):
        if not fname.endswith(".pt"):
            continue
        if fname == "meta.json":
            continue

        fpath = traj_dir / fname
        tensor = torch.load(str(fpath), map_location="cpu", weights_only=True)

        if fname == "final.pt":
            label = "final"
        elif fname.startswith("step_"):
            label = fname[:-3]  # "step_04.pt" -> "step_04"
        else:
            continue

        tensors[label] = tensor
    return tensors


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def migrate_v1_to_v2(
    v1_dir: str | Path,
    v2_dir: str | Path,
    max_blob_bytes: int = 1_000_000_000,
    verify: bool = True,
) -> None:
    """Migrate a v1 dataset to v2 format.

    Args:
        v1_dir: Path to the v1 dataset root (contains manifest.json, latents/).
        v2_dir: Path for the new v2 dataset (will be created).
        max_blob_bytes: Blob size limit.
        verify: If True, round-trip verify every tensor after migration.
    """
    v1_dir = Path(v1_dir)
    v2_dir = Path(v2_dir)

    # Read v1 manifest
    manifest_path = v1_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: No manifest.json found at {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    records = manifest.get("records", [])
    print(f"V1 dataset: {len(records)} trajectories in manifest")

    # Enumerate actual trajectory directories (in case manifest is stale)
    latents_dir = v1_dir / "latents"
    traj_dirs = sorted(
        d for d in latents_dir.iterdir()
        if d.is_dir() and d.name.startswith("traj_")
    ) if latents_dir.exists() else []

    print(f"Trajectory directories on disk: {len(traj_dirs)}")

    # Build a lookup from traj_dir name to manifest record
    # Manifest records have traj_dir as a full Windows path; we match by
    # the traj_NNNNNN suffix.
    record_by_name: dict[str, dict] = {}
    for rec in records:
        traj_path = rec.get("traj_dir", "")
        # Extract the traj_NNNNNN part from the Windows path
        name = Path(traj_path).name if traj_path else ""
        if name:
            record_by_name[name] = rec

    total_bytes = 0
    t_start = time.perf_counter()

    with DatasetWriter(v2_dir, max_blob_bytes=max_blob_bytes) as writer:
        for traj_dir in traj_dirs:
            name = traj_dir.name  # "traj_000042"

            # Load meta.json from disk (authoritative source)
            meta_path = traj_dir / "meta.json"
            if not meta_path.exists():
                print(f"  SKIP {name}: no meta.json")
                continue

            with open(meta_path) as f:
                meta = json.load(f)

            # Merge with manifest record if available (manifest may have
            # additional fields like traj_dir, but meta.json is authoritative)
            if name in record_by_name:
                rec = record_by_name[name]
                # Use manifest record for any fields missing from meta.json
                for k, v in rec.items():
                    if k not in meta and k != "traj_dir":
                        meta[k] = v

            # Load tensors
            tensors = _load_v1_tensors(traj_dir)
            if not tensors:
                print(f"  SKIP {name}: no .pt files")
                continue

            # Convert metadata
            v2_meta = _v1_meta_to_v2(meta, traj_dir)

            # Write to v2
            traj_id = writer.add_trajectory(tensors=tensors, metadata=v2_meta)
            traj_bytes = sum(
                t.nelement() * t.element_size()
                for t in tensors.values()
            )
            total_bytes += traj_bytes

            print(f"  {name} -> traj_id={traj_id:06d}  "
                  f"({len(tensors)} tensors, {traj_bytes / 1e6:.1f} MB)")

    elapsed = time.perf_counter() - t_start
    n_written = len(traj_dirs)
    print(f"\nMigration complete: {n_written} trajectories, "
          f"{total_bytes / 1e9:.2f} GB, {elapsed:.1f}s")

    # Count blobs
    blobs = list((v2_dir / "blobs").glob("blob_*.safetensors"))
    print(f"Blobs created: {len(blobs)}")
    for b in blobs:
        print(f"  {b.name} ({b.stat().st_size / 1e6:.1f} MB)")

    # ---------------------------------------------------------------------------
    # Verification
    # ---------------------------------------------------------------------------
    if verify:
        print(f"\nVerifying round-trip fidelity...")
        reader = DatasetReader(v2_dir)
        n_checked = 0
        n_tensors_checked = 0
        mismatches = []

        for traj_dir in traj_dirs:
            name = traj_dir.name
            # v1 traj index -> v2 traj_id (they correspond 1:1 in order)
            traj_idx = int(name.split("_")[1])

            if traj_idx not in reader:
                print(f"  WARN: {name} (id={traj_idx}) not in v2 dataset")
                continue

            meta_v2, accessor = reader[traj_idx]

            # Load v1 tensors for comparison
            v1_tensors = _load_v1_tensors(traj_dir)

            for label, v1_t in v1_tensors.items():
                # v2 accessor returns (1, C, H, W); v1 tensors may be (1, C, H, W) or (C, H, W)
                v2_t = accessor[label]

                # Normalize shapes for comparison
                if v1_t.dim() == 4:
                    v1_cmp = v1_t
                else:
                    v1_cmp = v1_t.unsqueeze(0)

                if not torch.equal(v1_cmp, v2_t):
                    mismatches.append((name, label))
                    print(f"  MISMATCH: {name}/{label}")
                else:
                    n_tensors_checked += 1

            n_checked += 1

        reader.close()

        if mismatches:
            print(f"\nVERIFICATION FAILED: {len(mismatches)} tensor mismatches")
            for name, label in mismatches[:10]:
                print(f"  {name}/{label}")
            sys.exit(1)
        else:
            print(f"Verification PASSED: {n_checked} trajectories, "
                  f"{n_tensors_checked} tensors checked (bitwise identical)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Migrate BTRM dataset from v1 to v2 format")
    parser.add_argument("--v1-dir", type=str, required=True,
                        help="Path to v1 dataset root (manifest.json + latents/)")
    parser.add_argument("--v2-dir", type=str, required=True,
                        help="Path for new v2 dataset (will be created)")
    parser.add_argument("--max-blob-bytes", type=int, default=1_000_000_000,
                        help="Max blob size in bytes (default: 1 GB)")
    parser.add_argument("--verify", action="store_true", default=False,
                        help="Bitwise round-trip verification after migration")
    parser.add_argument("--no-verify", action="store_false", dest="verify")

    args = parser.parse_args()

    migrate_v1_to_v2(
        v1_dir=args.v1_dir,
        v2_dir=args.v2_dir,
        max_blob_bytes=args.max_blob_bytes,
        verify=args.verify,
    )


if __name__ == "__main__":
    main()

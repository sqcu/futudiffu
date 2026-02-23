#!/usr/bin/env python3
"""Migrate ALL V1 trajectory data from btrm_dataset/ into the existing V2 dataset.

This script appends 98 V1 trajectories (traj_000000 through traj_000097) to
the existing btrm_dataset_v2/ which already contains 161 trajectories from
the multi-GPU merge. It:

1. Reads the existing V2 index to determine next_traj_id and next_blob_number.
2. Scans all 98 V1 trajectory directories.
3. For each trajectory:
   - Loads meta.json for metadata.
   - Checks if it's in the V1 manifest (50 trajectories) or an extra
     policy rollout (48 additional).
   - Loads all .pt latent files via torch.load(weights_only=True).
   - Converts to safetensors blob format (squeeze batch dim, contiguous).
4. Adds provenance: source_dir="btrm_dataset", run_name="original_v1" or
   "policy_rollout_v1", source_device="local".
5. Writes new blobs starting at the next available blob number.
6. Appends rows to the existing parquet index.
7. Prints a summary.

Does NOT delete source directories.
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file


REPO_ROOT = Path(r"F:\dox\repos\ai\futudiffu")
V1_DIR = REPO_ROOT / "btrm_dataset"
V2_DIR = REPO_ROOT / "btrm_dataset_v2"
V1_MANIFEST = V1_DIR / "manifest.json"
V1_LATENTS = V1_DIR / "latents"
V2_INDEX = V2_DIR / "index.parquet"
V2_BLOBS = V2_DIR / "blobs"

MAX_BLOB_BYTES = 200_000_000  # 200 MB per blob (V1 data is small, keep blobs reasonable)


def load_v1_manifest() -> dict[str, dict]:
    """Load V1 manifest and build a lookup from traj dir name to record."""
    with open(str(V1_MANIFEST), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    lookup = {}
    for record in manifest["records"]:
        traj_dir = record["traj_dir"]
        basename = traj_dir.rstrip("\\").split("\\")[-1]
        lookup[basename] = record

    return lookup


def discover_v1_trajectories() -> list[str]:
    """Discover all trajectory directory names in V1 latents/, sorted."""
    dirs = []
    for entry in os.listdir(str(V1_LATENTS)):
        full = V1_LATENTS / entry
        if full.is_dir() and entry.startswith("traj_"):
            dirs.append(entry)
    dirs.sort()
    return dirs


def load_meta_json(traj_dir: Path) -> dict:
    """Load meta.json from a V1 trajectory directory."""
    meta_path = traj_dir / "meta.json"
    with open(str(meta_path), "r", encoding="utf-8") as f:
        return json.load(f)


def discover_step_files(traj_dir: Path) -> tuple[list[str], bool]:
    """Discover step_XX.pt and final.pt files in a V1 trajectory directory.

    Returns:
        (step_labels, has_final) where step_labels are like ["step_00", "step_04", ...]
    """
    step_labels = []
    has_final = False

    for fname in os.listdir(str(traj_dir)):
        if fname == "final.pt":
            has_final = True
        elif fname.startswith("step_") and fname.endswith(".pt"):
            label = fname[:-3]  # Remove .pt
            step_labels.append(label)

    step_labels.sort(key=lambda s: int(s.split("_")[1]))
    return step_labels, has_final


def load_v1_tensors(traj_dir: Path, step_labels: list[str], has_final: bool) -> dict[str, torch.Tensor]:
    """Load all .pt files from a V1 trajectory directory.

    Returns tensors squeezed to (C, H, W) and contiguous.
    """
    tensors = {}
    for label in step_labels:
        pt_path = traj_dir / f"{label}.pt"
        t = torch.load(str(pt_path), weights_only=True, map_location="cpu")
        if t.dim() == 4 and t.shape[0] == 1:
            t = t.squeeze(0)
        tensors[label] = t.contiguous()

    if has_final:
        pt_path = traj_dir / "final.pt"
        t = torch.load(str(pt_path), weights_only=True, map_location="cpu")
        if t.dim() == 4 and t.shape[0] == 1:
            t = t.squeeze(0)
        tensors["final"] = t.contiguous()

    return tensors


def build_v2_metadata(
    meta: dict,
    manifest_record: dict | None,
    is_manifest: bool,
) -> dict:
    """Build V2 metadata dict from V1 meta.json and optional manifest record.

    Field mapping follows docs/dataset_v2_spec.md section 10.
    """
    attention_backend = meta.get("precision", "sdpa")

    batch_type = meta.get("type", "t2i")

    n_steps = meta.get("n_steps", 30)

    if manifest_record and "output_width" in manifest_record:
        width = manifest_record["output_width"]
        height = manifest_record["output_height"]
    elif batch_type == "t2i":
        width = 1280
        height = 832
    else:
        width = 1280
        height = 832

    cfg = meta.get("cfg", 4.0)

    prompt = meta.get("prompt", "")

    seed = meta.get("seed", 0)

    prompt_idx = meta.get("prompt_idx", -1)

    batch_idx = meta.get("batch_idx", 0)

    denoise = meta.get("denoise") or (manifest_record.get("denoise") if manifest_record else None)

    image_file = meta.get("image_file") or (manifest_record.get("image_file") if manifest_record else None)

    is_gold = (attention_backend == "sdpa" and n_steps == 30)

    packed = meta.get("packed", False)

    if is_manifest:
        run_name = "original_v1"
    else:
        run_name = "policy_rollout_v1"

    return {
        "prompt": prompt,
        "prompt_idx": prompt_idx,
        "seed": seed,
        "cfg": cfg,
        "width": width,
        "height": height,
        "n_steps": n_steps,
        "attention_backend": attention_backend,
        "batch_type": batch_type,
        "denoise": denoise,
        "image_file": image_file,
        "is_gold": is_gold,
        "batch_idx": batch_idx,
        "packed": packed,
        "timing_seconds": None,  # V1 doesn't record per-trajectory timing
        "parent_traj_id": None,
        "parent_step": None,
        "parent_denoise": None,
        "source_dir": "btrm_dataset",
        "run_name": run_name,
        "source_device": "local",
    }


def main():
    print("=" * 70)
    print("V1 -> V2 Migration: btrm_dataset/ into btrm_dataset_v2/")
    print("=" * 70)

    print("\n[1/6] Reading existing V2 index...")
    existing_table = pq.read_table(str(V2_INDEX))
    existing_rows = existing_table.to_pylist()
    n_existing = len(existing_rows)
    existing_traj_ids = [r["traj_id"] for r in existing_rows]
    next_traj_id = max(existing_traj_ids) + 1 if existing_traj_ids else 0

    existing_blobs = sorted([
        f for f in os.listdir(str(V2_BLOBS))
        if f.startswith("blob_") and f.endswith(".safetensors")
    ])
    next_blob_num = len(existing_blobs)

    print(f"  Existing trajectories: {n_existing}")
    print(f"  Next traj_id: {next_traj_id}")
    print(f"  Existing blobs: {len(existing_blobs)} (next: blob_{next_blob_num:03d})")

    print("\n[2/6] Loading V1 manifest and discovering trajectories...")
    manifest_lookup = load_v1_manifest()
    traj_dirs = discover_v1_trajectories()

    n_manifest = 0
    n_extra = 0
    for d in traj_dirs:
        if d in manifest_lookup:
            n_manifest += 1
        else:
            n_extra += 1

    print(f"  V1 manifest records: {len(manifest_lookup)}")
    print(f"  Total trajectory dirs: {len(traj_dirs)}")
    print(f"    In manifest: {n_manifest}")
    print(f"    Extra (policy rollouts): {n_extra}")

    print("\n[3/6] Processing V1 trajectories...")

    new_rows = []
    current_blob_tensors: dict[str, torch.Tensor] = {}
    current_blob_bytes = 0
    blob_num = next_blob_num
    sealed_blobs = []

    def seal_blob():
        nonlocal current_blob_tensors, current_blob_bytes, blob_num
        if not current_blob_tensors:
            return
        blob_name = f"blob_{blob_num:03d}.safetensors"
        blob_path = V2_BLOBS / blob_name
        blob_meta = {
            "dataset_version": "2",
            "n_trajectories": str(sum(1 for k in current_blob_tensors if k.endswith("/final") or k.endswith("/step_00"))),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": "v1_migration",
        }
        save_file(current_blob_tensors, str(blob_path), metadata=blob_meta)
        n_tensors = len(current_blob_tensors)
        mb = current_blob_bytes / 1e6
        print(f"    Sealed {blob_name}: {n_tensors} tensors, {mb:.1f} MB")
        sealed_blobs.append(blob_name)

        for row in new_rows:
            if row["blob_file"] == "_wip_":
                row["blob_file"] = blob_name

        blob_num += 1
        current_blob_tensors = {}
        current_blob_bytes = 0

    total_bytes = 0
    total_tensors = 0

    for i, traj_dir_name in enumerate(traj_dirs):
        traj_path = V1_LATENTS / traj_dir_name
        meta = load_meta_json(traj_path)

        is_manifest = traj_dir_name in manifest_lookup
        manifest_record = manifest_lookup.get(traj_dir_name)

        step_labels, has_final = discover_step_files(traj_path)
        step_indices = [int(s.split("_")[1]) for s in step_labels]

        tensors = load_v1_tensors(traj_path, step_labels, has_final)

        traj_bytes = sum(t.nelement() * t.element_size() for t in tensors.values())
        n_tensors_traj = len(tensors)

        ref_tensor = next(iter(tensors.values()))
        c, h, w = ref_tensor.shape
        dtype_str = str(ref_tensor.dtype).replace("torch.", "")

        v2_meta = build_v2_metadata(meta, manifest_record, is_manifest)

        traj_id = next_traj_id + i
        key_prefix = f"{traj_id:06d}"

        if current_blob_tensors and (current_blob_bytes + traj_bytes > MAX_BLOB_BYTES):
            seal_blob()

        for label, t in tensors.items():
            blob_key = f"{key_prefix}/{label}"
            current_blob_tensors[blob_key] = t

        current_blob_bytes += traj_bytes
        total_bytes += traj_bytes
        total_tensors += n_tensors_traj

        now = datetime.now(timezone.utc)
        row = {
            "traj_id": traj_id,
            "prompt": v2_meta["prompt"],
            "prompt_idx": v2_meta["prompt_idx"],
            "seed": int(v2_meta["seed"]),
            "cfg": float(v2_meta["cfg"]),
            "width": int(v2_meta["width"]),
            "height": int(v2_meta["height"]),
            "n_steps": int(v2_meta["n_steps"]),
            "attention_backend": v2_meta["attention_backend"],
            "batch_type": v2_meta["batch_type"],
            "denoise": float(v2_meta["denoise"]) if v2_meta["denoise"] is not None else None,
            "image_file": v2_meta["image_file"],
            "is_gold": bool(v2_meta["is_gold"]),
            "batch_idx": int(v2_meta["batch_idx"]),
            "packed": bool(v2_meta["packed"]),
            "step_indices": step_indices,
            "has_final": has_final,
            "latent_channels": int(c),
            "latent_height": int(h),
            "latent_width": int(w),
            "latent_dtype": dtype_str,
            "blob_file": "_wip_",  # Will be updated when blob is sealed
            "key_prefix": key_prefix,
            "n_tensors": n_tensors_traj,
            "bytes_total": traj_bytes,
            "timing_seconds": None,
            "created_at": now,
            "parent_traj_id": None,
            "parent_step": None,
            "parent_denoise": None,
            "source_dir": v2_meta["source_dir"],
            "run_name": v2_meta["run_name"],
            "source_device": v2_meta["source_device"],
        }
        new_rows.append(row)

        tag = "manifest" if is_manifest else "extra"
        if (i + 1) % 10 == 0 or i == 0:
            print(f"    [{i+1:3d}/{len(traj_dirs)}] {traj_dir_name} -> traj_id {traj_id} "
                  f"({tag}, {n_tensors_traj} tensors, {traj_bytes/1e6:.2f} MB, "
                  f"{c}x{h}x{w} {dtype_str})")

    seal_blob()

    print(f"\n  Processed {len(traj_dirs)} trajectories:")
    print(f"    Total tensors: {total_tensors}")
    print(f"    Total bytes: {total_bytes / 1e6:.1f} MB")
    print(f"    New blobs created: {len(sealed_blobs)}")

    print("\n[4/6] Validating blob assignments...")
    bad_rows = [r for r in new_rows if r["blob_file"] == "_wip_"]
    if bad_rows:
        raise RuntimeError(f"{len(bad_rows)} rows still reference _wip_ blob!")
    print("  All rows have valid blob_file references.")

    print("\n[5/6] Merging with existing V2 index...")

    all_rows = existing_rows + new_rows

    merged_table = pa.Table.from_pylist(all_rows, schema=existing_table.schema)

    temp_path = V2_INDEX.with_suffix(".parquet.tmp")
    pq.write_table(
        merged_table,
        str(temp_path),
        compression="zstd",
        compression_level=3,
        write_statistics=True,
        use_dictionary=[
            "attention_backend",
            "batch_type",
            "latent_dtype",
            "blob_file",
            "parent_step",
            "source_dir",
            "run_name",
            "source_device",
        ],
        row_group_size=10_000,
    )
    os.replace(str(temp_path), str(V2_INDEX))
    print(f"  Wrote merged index: {len(all_rows)} total rows")

    print("\n[6/6] Verification and summary...")

    verify_table = pq.read_table(str(V2_INDEX))
    verify_n = len(verify_table)

    source_dirs = verify_table.column("source_dir").to_pylist()
    from collections import Counter
    source_dist = Counter(source_dirs)

    run_names = verify_table.column("run_name").to_pylist()
    run_dist = Counter(run_names)

    all_blobs = sorted([
        f for f in os.listdir(str(V2_BLOBS))
        if f.startswith("blob_") and f.endswith(".safetensors")
    ])

    total_blob_size = sum(
        os.path.getsize(str(V2_BLOBS / b)) for b in all_blobs
    )

    print(f"\n{'=' * 70}")
    print(f"MIGRATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total trajectories in V2: {verify_n}")
    print(f"    Previously existing:    {n_existing}")
    print(f"    Newly migrated:         {len(new_rows)}")
    print(f"    Expected:               {n_existing + len(traj_dirs)}")
    print(f"    Match: {'YES' if verify_n == n_existing + len(traj_dirs) else 'NO'}")
    print()
    print(f"  Total blobs: {len(all_blobs)}")
    for b in all_blobs:
        size = os.path.getsize(str(V2_BLOBS / b))
        print(f"    {b}: {size / 1e6:.1f} MB")
    print(f"  Total blob size: {total_blob_size / 1e6:.1f} MB")
    print()
    print(f"  Per-source_dir distribution:")
    for src, count in sorted(source_dist.items()):
        print(f"    {src}: {count}")
    print()
    print(f"  Per-run_name distribution:")
    for rn, count in sorted(run_dist.items()):
        print(f"    {rn}: {count}")
    print()

    traj_ids = verify_table.column("traj_id").to_pylist()
    print(f"  traj_id range: {min(traj_ids)} - {max(traj_ids)}")
    print(f"  traj_id count unique: {len(set(traj_ids))}")
    if len(set(traj_ids)) != verify_n:
        print("  WARNING: duplicate traj_ids detected!")
    else:
        print("  No duplicate traj_ids.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge per-GPU dataset_v2 staging directories into a unified dataset.

Each GPU writes to its own staging directory (e.g., dataset_v2_gpu0,
dataset_v2_gpu1, ...) to avoid write lock contention. This script reads
each staging directory's parquet index and safetensors blobs, re-numbers
trajectory IDs to be globally unique, rewrites blob keys, and produces
a single merged dataset_v2 directory.

Usage:
    python merge_staged_datasets.py \
        --staging-dirs dataset_v2_gpu0 dataset_v2_gpu1 ... \
        --output dataset_v2_merged

Layout of each staging dir (dataset_v2 format):
    staging_dir/
        index.parquet
        blobs/
            blob_000.safetensors
            blob_001.safetensors
            ...

Merged output has the same layout with globally unique traj_ids and
sequentially numbered blobs.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

import pyarrow as pa
import pyarrow.parquet as pq
from safetensors import safe_open
from safetensors.torch import save_file

from futudiffu.dataset_v2 import INDEX_SCHEMA, _PARQUET_WRITE_KWARGS


def _validate_staging_dir(staging_dir: Path) -> pa.Table:
    """Validate a staging directory and return its parquet index table."""
    if not staging_dir.exists():
        raise FileNotFoundError(f"Staging directory does not exist: {staging_dir}")

    index_path = staging_dir / "index.parquet"
    if not index_path.exists():
        # Empty staging dir (writer with 0 trajectories never writes index).
        # Return an empty table with the correct schema.
        print(f"  WARNING: {staging_dir} has no index.parquet (empty staging dir)")
        return INDEX_SCHEMA.empty_table()

    blobs_dir = staging_dir / "blobs"
    if not blobs_dir.exists():
        raise FileNotFoundError(f"No blobs/ directory in staging directory: {staging_dir}")

    table = pq.read_table(str(index_path))
    if len(table) == 0:
        print(f"  WARNING: {staging_dir} has 0 trajectories (empty index)")

    return table


def _remap_blob(
    src_blob_path: Path,
    id_remap: dict[int, int],
) -> dict[str, "torch.Tensor"]:
    """Read a safetensors blob and remap its keys to new traj_ids.

    Keys are formatted as '{traj_id:06d}/{step_label}'. This function
    parses the traj_id from each key, looks up the new global ID in
    id_remap, and returns a dict with remapped keys.

    Returns:
        Dict mapping new keys to tensors.
    """
    remapped = {}
    with safe_open(str(src_blob_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            # Key format: '000042/step_14' or '000042/final'
            parts = key.split("/", 1)
            if len(parts) != 2:
                raise ValueError(
                    f"Unexpected blob key format '{key}' in {src_blob_path}. "
                    f"Expected '{{traj_id:06d}}/{{step_label}}'."
                )
            old_id = int(parts[0])
            step_label = parts[1]

            if old_id not in id_remap:
                raise KeyError(
                    f"Blob key references traj_id {old_id} which is not in the "
                    f"parquet index of its staging directory. Blob: {src_blob_path}"
                )

            new_id = id_remap[old_id]
            new_key = f"{new_id:06d}/{step_label}"
            remapped[new_key] = f.get_tensor(key)

    return remapped


def merge_staged_datasets(
    staging_dirs: list[Path],
    output_dir: Path,
    max_blob_bytes: int = 1_000_000_000,
) -> int:
    """Merge multiple staging directories into one unified dataset_v2.

    Args:
        staging_dirs: List of staging directory paths to merge.
        output_dir: Output directory for the merged dataset.
        max_blob_bytes: Maximum bytes per output blob (default 1 GB).

    Returns:
        Total number of trajectories in the merged dataset.
    """
    # ------------------------------------------------------------------
    # Step 1: Validate all staging dirs and load their indices
    # ------------------------------------------------------------------
    print(f"Validating {len(staging_dirs)} staging directories...")
    tables: list[tuple[Path, pa.Table]] = []
    for sd in staging_dirs:
        table = _validate_staging_dir(sd)
        tables.append((sd, table))
        print(f"  {sd.name}: {len(table)} trajectories")

    total_traj = sum(len(t) for _, t in tables)
    print(f"Total trajectories to merge: {total_traj}")

    # ------------------------------------------------------------------
    # Step 2: Check idempotency -- skip if output already correct
    # ------------------------------------------------------------------
    output_index_path = output_dir / "index.parquet"
    if output_index_path.exists():
        existing = pq.read_table(str(output_index_path))
        if len(existing) == total_traj:
            print(f"Output already has {total_traj} trajectories. Skipping merge (idempotent).")
            return total_traj

    # ------------------------------------------------------------------
    # Step 3: Build global ID remapping
    # ------------------------------------------------------------------
    # GPU 0 gets IDs 0..N0-1, GPU 1 gets N0..N0+N1-1, etc.
    next_global_id = 0
    all_remaps: list[dict[int, int]] = []
    all_rows: list[list[dict]] = []

    for sd, table in tables:
        rows = table.to_pylist()
        id_remap = {}
        remapped_rows = []

        for row in rows:
            old_id = row["traj_id"]
            new_id = next_global_id
            id_remap[old_id] = new_id

            # Update row with new IDs
            row["traj_id"] = new_id
            row["key_prefix"] = f"{new_id:06d}"

            # Remap parent_traj_id if it references a trajectory in the
            # same staging dir. Cross-staging-dir references are not
            # expected in normal multi-GPU generation (each GPU generates
            # independently), but we handle the same-dir case for i2i2i.
            if row.get("parent_traj_id") is not None:
                parent_old = row["parent_traj_id"]
                if parent_old in id_remap:
                    row["parent_traj_id"] = id_remap[parent_old]
                # If parent is not in this staging dir, leave it as-is.
                # The caller is responsible for ensuring cross-dir
                # references are valid (unusual case).

            remapped_rows.append(row)
            next_global_id += 1

        all_remaps.append(id_remap)
        all_rows.append(remapped_rows)

    # ------------------------------------------------------------------
    # Step 4: Create output directory structure
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    output_blobs_dir = output_dir / "blobs"
    output_blobs_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Step 5: Copy and re-key blobs
    # ------------------------------------------------------------------
    print("Re-keying and writing blobs...")
    output_blob_idx = 0
    current_blob_tensors: dict[str, "torch.Tensor"] = {}
    current_blob_bytes = 0

    # Track which output blob each trajectory ends up in.
    # Key: new global traj_id -> output blob filename
    traj_to_blob: dict[int, str] = {}

    def _seal_blob():
        nonlocal output_blob_idx, current_blob_tensors, current_blob_bytes
        if not current_blob_tensors:
            return
        blob_name = f"blob_{output_blob_idx:03d}.safetensors"
        blob_path = output_blobs_dir / blob_name
        save_file(current_blob_tensors, str(blob_path))
        print(f"  Wrote {blob_name} ({len(current_blob_tensors)} tensors, "
              f"{current_blob_bytes / 1e6:.1f} MB)")
        output_blob_idx += 1
        current_blob_tensors = {}
        current_blob_bytes = 0

    for (sd, table), id_remap, rows in zip(tables, all_remaps, all_rows):
        blobs_dir = sd / "blobs"

        # Collect all unique blob files referenced by this staging dir
        blob_files = sorted(set(r["blob_file"] for r in rows))

        for blob_file in blob_files:
            src_path = blobs_dir / blob_file
            if not src_path.exists():
                raise FileNotFoundError(
                    f"Referenced blob file not found: {src_path}"
                )

            remapped_tensors = _remap_blob(src_path, id_remap)

            for key, tensor in remapped_tensors.items():
                tensor_bytes = tensor.nelement() * tensor.element_size()

                # Check if we need to seal the current blob
                if current_blob_tensors and (current_blob_bytes + tensor_bytes > max_blob_bytes):
                    _seal_blob()

                current_blob_tensors[key] = tensor
                current_blob_bytes += tensor_bytes

                # Track blob assignment for this trajectory
                new_traj_id = int(key.split("/")[0])
                pending_blob_name = f"blob_{output_blob_idx:03d}.safetensors"
                traj_to_blob[new_traj_id] = pending_blob_name

    # Seal the final blob
    _seal_blob()

    # ------------------------------------------------------------------
    # Step 6: Update blob_file references in rows and write merged index
    # ------------------------------------------------------------------
    print("Writing merged index...")
    merged_rows = []
    for rows in all_rows:
        for row in rows:
            tid = row["traj_id"]
            if tid in traj_to_blob:
                row["blob_file"] = traj_to_blob[tid]
            else:
                raise RuntimeError(
                    f"Trajectory {tid} has no blob assignment. "
                    f"This should not happen."
                )
            merged_rows.append(row)

    merged_table = pa.Table.from_pylist(merged_rows, schema=INDEX_SCHEMA)
    pq.write_table(merged_table, str(output_index_path), **_PARQUET_WRITE_KWARGS)

    print(f"Merged {total_traj} trajectories from {len(staging_dirs)} staging dirs "
          f"into {output_dir}")
    print(f"  Index: {output_index_path}")
    print(f"  Blobs: {output_blob_idx} files in {output_blobs_dir}")

    return total_traj


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-GPU dataset_v2 staging directories into a unified dataset."
    )
    parser.add_argument(
        "--staging-dirs", nargs="+", required=True, type=str,
        help="Paths to per-GPU staging directories to merge."
    )
    parser.add_argument(
        "--output", required=True, type=str,
        help="Output directory for the merged dataset."
    )
    parser.add_argument(
        "--max-blob-bytes", type=int, default=1_000_000_000,
        help="Maximum bytes per output blob file (default: 1 GB)."
    )

    args = parser.parse_args()

    staging_dirs = [Path(p) for p in args.staging_dirs]
    output_dir = Path(args.output)

    merge_staged_datasets(
        staging_dirs=staging_dirs,
        output_dir=output_dir,
        max_blob_bytes=args.max_blob_bytes,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

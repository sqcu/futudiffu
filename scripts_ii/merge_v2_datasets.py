#!/usr/bin/env python3
"""Merge fragmented V2 dataset splits from a 2xH100 training run into a
single unified btrm_dataset_v2/.

Reads parquet indices and safetensors blobs from all GPU split directories,
adds provenance metadata (source_dir, run_name, source_device), renumbers
traj_ids and blob files to avoid collisions, and writes a merged dataset.

Source directories are NOT deleted (copy, not move).

Usage:
    PYTHONUNBUFFERED=1 .venv/Scripts/python.exe \
        F:\\dox\\repos\\ai\\futudiffu\\scripts_ii\\merge_v2_datasets.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

import pyarrow as pa
import pyarrow.parquet as pq
from safetensors import safe_open
from safetensors.torch import save_file

from futudiffu.dataset_v2 import INDEX_SCHEMA, _PARQUET_WRITE_KWARGS


REPO_ROOT = Path(r"F:\dox\repos\ai\futudiffu")

SOURCE_DIRS = [
    REPO_ROOT / "btrm_dataset_v2_gpu0",
    REPO_ROOT / "btrm_dataset_v2_gpu0_gpu0",
    REPO_ROOT / "btrm_dataset_v2_gpu1",
    REPO_ROOT / "btrm_dataset_v2_gpu1_gpu1",
]

OUTPUT_DIR = REPO_ROOT / "btrm_dataset_v2"
REPORT_DIR = REPO_ROOT / "dataset_audit_output"
RUN_NAME = "2xh100_20260216"


def _parse_device(dir_name: str) -> str:
    """Extract the GPU device identifier from a directory name.

    Examples:
        btrm_dataset_v2_gpu0       -> 'gpu0'
        btrm_dataset_v2_gpu0_gpu0  -> 'gpu0'  (naming collision artifact)
        btrm_dataset_v2_gpu1       -> 'gpu1'
        btrm_dataset_v2_gpu1_gpu1  -> 'gpu1'  (naming collision artifact)
    """
    m = re.search(r"gpu(\d+)", dir_name)
    if m:
        return f"gpu{m.group(1)}"
    return "unknown"


def _read_source(src_dir: Path) -> tuple[pa.Table, list[Path]]:
    """Read a source directory's parquet index and list its blob files.

    Returns:
        (table, blob_paths) where table may be empty and blob_paths is
        a sorted list of existing blob files.
    """
    index_path = src_dir / "index.parquet"
    blobs_dir = src_dir / "blobs"

    if not index_path.exists():
        print(f"  WARNING: {src_dir.name} has no index.parquet -- skipping")
        return INDEX_SCHEMA.empty_table(), []

    table = pq.read_table(str(index_path))

    blob_paths = []
    if blobs_dir.exists():
        blob_paths = sorted(
            p for p in blobs_dir.iterdir()
            if p.name.startswith("blob_") and p.suffix == ".safetensors"
        )

    return table, blob_paths


def _remap_blob_tensors(
    src_blob_path: Path,
    id_remap: dict[int, int],
) -> dict[str, "torch.Tensor"]:
    """Read a safetensors blob and remap keys to new traj_ids.

    Keys are '{traj_id:06d}/{step_label}'. Returns dict with remapped keys.
    """
    remapped = {}
    with safe_open(str(src_blob_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            parts = key.split("/", 1)
            if len(parts) != 2:
                raise ValueError(
                    f"Unexpected blob key format '{key}' in {src_blob_path}"
                )
            old_id = int(parts[0])
            step_label = parts[1]

            if old_id not in id_remap:
                print(f"  WARNING: blob key references traj_id {old_id} not in "
                      f"parquet index of {src_blob_path.parent.parent.name}. Skipping.")
                continue

            new_id = id_remap[old_id]
            new_key = f"{new_id:06d}/{step_label}"
            remapped[new_key] = f.get_tensor(key)

    return remapped


def merge_v2_datasets() -> dict:
    """Main merge logic. Returns a summary dict for the report."""

    start_time = time.time()

    print(f"Merge V2 Datasets")
    print(f"  Run name:   {RUN_NAME}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Sources:    {len(SOURCE_DIRS)}")
    print()

    print("Reading source indices...")
    sources: list[tuple[Path, pa.Table, list[Path]]] = []
    for src_dir in SOURCE_DIRS:
        if not src_dir.exists():
            print(f"  SKIP: {src_dir.name} does not exist")
            continue
        table, blob_paths = _read_source(src_dir)
        n_traj = len(table)
        blob_sizes = [p.stat().st_size for p in blob_paths]
        total_blob_mb = sum(blob_sizes) / 1e6
        print(f"  {src_dir.name}: {n_traj} trajectories, "
              f"{len(blob_paths)} blob(s), {total_blob_mb:.1f} MB")
        if n_traj > 0:
            sources.append((src_dir, table, blob_paths))

    if not sources:
        print("ERROR: No non-empty source directories found. Nothing to merge.")
        return {"error": "no sources"}

    total_traj = sum(len(t) for _, t, _ in sources)
    print(f"\nTotal trajectories to merge: {total_traj}")

    provenance_fields = [
        pa.field("source_dir", pa.utf8()),
        pa.field("run_name", pa.utf8()),
        pa.field("source_device", pa.utf8()),
    ]
    extended_schema = pa.schema(list(INDEX_SCHEMA) + provenance_fields)

    print("\nBuilding global ID remapping...")
    next_global_id = 0
    per_source_data: list[tuple[Path, dict[int, int], list[dict]]] = []
    per_source_counts: dict[str, int] = {}

    for src_dir, table, blob_paths in sources:
        rows = table.to_pylist()
        id_remap: dict[int, int] = {}
        remapped_rows: list[dict] = []
        device = _parse_device(src_dir.name)

        for row in rows:
            old_id = row["traj_id"]
            new_id = next_global_id
            id_remap[old_id] = new_id

            row["traj_id"] = new_id
            row["key_prefix"] = f"{new_id:06d}"

            if row.get("parent_traj_id") is not None:
                parent_old = row["parent_traj_id"]
                if parent_old in id_remap:
                    row["parent_traj_id"] = id_remap[parent_old]

            row["source_dir"] = src_dir.name
            row["run_name"] = RUN_NAME
            row["source_device"] = device

            remapped_rows.append(row)
            next_global_id += 1

        per_source_data.append((src_dir, id_remap, remapped_rows))
        per_source_counts[src_dir.name] = len(remapped_rows)
        print(f"  {src_dir.name}: {len(remapped_rows)} trajs -> "
              f"ids [{min(id_remap.values())}..{max(id_remap.values())}]")

    print(f"\nPreparing output directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_blobs_dir = OUTPUT_DIR / "blobs"
    output_blobs_dir.mkdir(exist_ok=True)

    existing_blobs = list(output_blobs_dir.glob("blob_*.safetensors"))
    if existing_blobs:
        print(f"  Cleaning {len(existing_blobs)} stale blob(s) from previous run")
        for p in existing_blobs:
            p.unlink()

    print("\nRe-keying and writing blobs...")
    output_blob_idx = 0
    traj_to_blob: dict[int, str] = {}
    total_blob_bytes_written = 0

    for src_dir, id_remap, remapped_rows in per_source_data:
        blob_files_referenced = sorted(set(r["blob_file"] for r in remapped_rows
                                           if r.get("blob_file")))
        src_blobs_dir = src_dir / "blobs"

        for blob_file in blob_files_referenced:
            src_blob_path = src_blobs_dir / blob_file
            if not src_blob_path.exists():
                raise FileNotFoundError(
                    f"Referenced blob not found: {src_blob_path}"
                )

            remapped_tensors = _remap_blob_tensors(src_blob_path, id_remap)

            if not remapped_tensors:
                print(f"  WARNING: {src_dir.name}/{blob_file} produced 0 remapped tensors")
                continue

            out_blob_name = f"blob_{output_blob_idx:03d}.safetensors"
            out_blob_path = output_blobs_dir / out_blob_name

            blob_meta = {
                "dataset_version": "2",
                "n_trajectories": str(len(set(
                    int(k.split("/")[0]) for k in remapped_tensors
                ))),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_dir": src_dir.name,
                "source_blob": blob_file,
            }

            temp_path = out_blob_path.with_suffix(".tmp")
            save_file(remapped_tensors, str(temp_path), metadata=blob_meta)
            os.replace(str(temp_path), str(out_blob_path))

            blob_size = out_blob_path.stat().st_size
            total_blob_bytes_written += blob_size

            traj_ids_in_blob = set(int(k.split("/")[0]) for k in remapped_tensors)
            for tid in traj_ids_in_blob:
                traj_to_blob[tid] = out_blob_name

            n_tensors = len(remapped_tensors)
            print(f"  {src_dir.name}/{blob_file} -> {out_blob_name} "
                  f"({n_tensors} tensors, {blob_size / 1e6:.1f} MB)")

            output_blob_idx += 1

    print("\nWriting merged index.parquet...")
    merged_rows: list[dict] = []
    missing_blob_refs = 0

    for src_dir, id_remap, remapped_rows in per_source_data:
        for row in remapped_rows:
            tid = row["traj_id"]
            if tid in traj_to_blob:
                row["blob_file"] = traj_to_blob[tid]
            else:
                print(f"  WARNING: traj_id {tid} has no blob assignment "
                      f"(from {row.get('source_dir', '?')})")
                missing_blob_refs += 1
                continue
            merged_rows.append(row)

    merged_table = pa.Table.from_pylist(merged_rows, schema=extended_schema)
    output_index_path = OUTPUT_DIR / "index.parquet"
    temp_index = output_index_path.with_suffix(".parquet.tmp")

    write_kwargs = dict(_PARQUET_WRITE_KWARGS)
    write_kwargs["use_dictionary"] = list(_PARQUET_WRITE_KWARGS["use_dictionary"]) + [
        "source_dir", "run_name", "source_device",
    ]

    pq.write_table(merged_table, str(temp_index), **write_kwargs)
    os.replace(str(temp_index), str(output_index_path))

    index_size = output_index_path.stat().st_size
    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"  Trajectories merged: {len(merged_rows)}")
    print(f"  Output blobs:        {output_blob_idx}")
    print(f"  Total blob data:     {total_blob_bytes_written / 1e6:.1f} MB")
    print(f"  Index size:          {index_size / 1e3:.1f} KB")
    print(f"  Missing blob refs:   {missing_blob_refs}")
    print(f"  Elapsed:             {elapsed:.1f}s")
    print()
    print("  Per-source counts:")
    for name, count in per_source_counts.items():
        print(f"    {name}: {count}")
    print()
    print(f"  Output: {OUTPUT_DIR}")
    print(f"    index.parquet: {output_index_path}")
    print(f"    blobs/:        {output_blobs_dir}")

    summary = {
        "merge_timestamp": datetime.now(timezone.utc).isoformat(),
        "run_name": RUN_NAME,
        "total_trajectories": len(merged_rows),
        "total_blob_files": output_blob_idx,
        "total_blob_bytes": total_blob_bytes_written,
        "index_bytes": index_size,
        "missing_blob_refs": missing_blob_refs,
        "elapsed_seconds": round(elapsed, 2),
        "per_source": per_source_counts,
        "output_dir": str(OUTPUT_DIR),
    }

    return summary


def verify_merged_dataset(summary: dict) -> bool:
    """Verify the merged dataset is readable and consistent."""
    print()
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    index_path = OUTPUT_DIR / "index.parquet"
    if not index_path.exists():
        print("  FAIL: index.parquet does not exist")
        return False

    table = pq.read_table(str(index_path))
    n_rows = len(table)
    print(f"  index.parquet: {n_rows} rows")
    print(f"  Schema columns: {table.column_names}")

    for col in ["source_dir", "run_name", "source_device"]:
        if col not in table.column_names:
            print(f"  FAIL: missing provenance column '{col}'")
            return False
        vals = set(table.column(col).to_pylist())
        print(f"    {col}: {vals}")

    traj_ids = table.column("traj_id").to_pylist()
    if len(set(traj_ids)) != len(traj_ids):
        print(f"  FAIL: duplicate traj_ids detected")
        return False
    expected = list(range(n_rows))
    if traj_ids != expected:
        print(f"  WARN: traj_ids are not sequential 0..{n_rows-1}")
        print(f"    first 5: {traj_ids[:5]}, last 5: {traj_ids[-5:]}")
    else:
        print(f"  traj_ids: 0..{n_rows-1} (sequential, unique)")

    blob_files = set(table.column("blob_file").to_pylist())
    blobs_dir = OUTPUT_DIR / "blobs"
    missing_blobs = []
    for bf in blob_files:
        if not (blobs_dir / bf).exists():
            missing_blobs.append(bf)
    if missing_blobs:
        print(f"  FAIL: {len(missing_blobs)} referenced blob(s) missing: {missing_blobs}")
        return False
    print(f"  All {len(blob_files)} referenced blobs exist")

    total_blob_size = sum(
        (blobs_dir / bf).stat().st_size for bf in blob_files
    )
    print(f"  Total blob size on disk: {total_blob_size / 1e6:.1f} MB")

    print()
    print("  Sample rows (first 3):")
    for i in range(min(3, n_rows)):
        row = {col: table.column(col)[i].as_py() for col in table.column_names}
        print(f"    [{i}] traj_id={row['traj_id']}, prompt_idx={row['prompt_idx']}, "
              f"seed={row['seed']}, n_steps={row['n_steps']}, "
              f"backend={row['attention_backend']}, "
              f"blob={row['blob_file']}, prefix={row['key_prefix']}, "
              f"source={row['source_dir']}, device={row['source_device']}")

    print()
    print("  Sample rows (last 3):")
    for i in range(max(0, n_rows - 3), n_rows):
        row = {col: table.column(col)[i].as_py() for col in table.column_names}
        print(f"    [{i}] traj_id={row['traj_id']}, prompt_idx={row['prompt_idx']}, "
              f"seed={row['seed']}, n_steps={row['n_steps']}, "
              f"backend={row['attention_backend']}, "
              f"blob={row['blob_file']}, prefix={row['key_prefix']}, "
              f"source={row['source_dir']}, device={row['source_device']}")

    print()
    print("  Spot-checking tensor reads from each blob...")
    for bf in sorted(blob_files):
        blob_path = blobs_dir / bf
        try:
            with safe_open(str(blob_path), framework="pt", device="cpu") as f:
                keys = list(f.keys())
                if keys:
                    t = f.get_tensor(keys[0])
                    print(f"    {bf}: {len(keys)} tensors, "
                          f"sample key='{keys[0]}', shape={tuple(t.shape)}, "
                          f"dtype={t.dtype}")
                else:
                    print(f"    {bf}: 0 tensors (empty blob)")
        except Exception as e:
            print(f"    {bf}: FAIL: {e}")
            return False

    print()
    print("  VERIFICATION PASSED")
    return True


def save_report(summary: dict, verified: bool) -> Path:
    """Save the merge report to dataset_audit_output/merge_report.txt."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "merge_report.txt"

    lines = [
        "=" * 60,
        "V2 DATASET MERGE REPORT",
        "=" * 60,
        "",
        f"Timestamp:           {summary.get('merge_timestamp', 'N/A')}",
        f"Run name:            {summary.get('run_name', 'N/A')}",
        f"Verification:        {'PASSED' if verified else 'FAILED'}",
        "",
        "--- Merged Dataset ---",
        f"Total trajectories:  {summary.get('total_trajectories', 0)}",
        f"Total blob files:    {summary.get('total_blob_files', 0)}",
        f"Total blob bytes:    {summary.get('total_blob_bytes', 0):,}",
        f"Index bytes:         {summary.get('index_bytes', 0):,}",
        f"Missing blob refs:   {summary.get('missing_blob_refs', 0)}",
        f"Elapsed seconds:     {summary.get('elapsed_seconds', 0)}",
        f"Output directory:    {summary.get('output_dir', 'N/A')}",
        "",
        "--- Per-Source Counts ---",
    ]

    for name, count in summary.get("per_source", {}).items():
        lines.append(f"  {name}: {count}")

    lines.extend(["", "--- Raw Summary JSON ---", json.dumps(summary, indent=2)])

    report_text = "\n".join(lines) + "\n"

    with open(str(report_path), "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nReport saved to: {report_path}")
    return report_path


def main() -> int:
    summary = merge_v2_datasets()

    if "error" in summary:
        return 1

    verified = verify_merged_dataset(summary)
    save_report(summary, verified)

    return 0 if verified else 1


if __name__ == "__main__":
    sys.exit(main())

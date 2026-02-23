#!/usr/bin/env python3
"""Backfill sampling state identity hashes into the V2 dataset index.

Reads the existing btrm_dataset_v2/index.parquet, fills in hash columns
for trajectories where the model state is known, and writes the updated
index back.

Backfill rules:
  - run_name="original_v1" (50 trajectories):
      base_model_hash = "z_image_v1" (placeholder -- not a real content hash)
      adapter_set_hash = "" (no adapters active during generation)
      active_adapters = "[]"
      model_state_hash = computed from (base_model_hash, adapter_set_hash)
      trajectory_hash = computed from (model_state_hash, prompt, seed, cfg, ...)

  - run_name="policy_rollout_v1" (48 trajectories):
      base_model_hash = "z_image_v1" (same base model)
      adapter_set_hash = null (unknown -- adapter state at generation time not recorded)
      active_adapters = null (unknown)
      model_state_hash = null (cannot compute without adapter_set_hash)
      trajectory_hash = null (cannot compute without model_state_hash)

  - run_name="2xh100_20260216" (161 trajectories):
      base_model_hash = "z_image_v1" (same base model)
      adapter_set_hash = null (unknown -- adapter state at generation time not recorded)
      active_adapters = null (unknown)
      model_state_hash = null (cannot compute without adapter_set_hash)
      trajectory_hash = null (cannot compute without model_state_hash)

The base_model_hash "z_image_v1" is a placeholder. When the actual model
weights are hashed (e.g., SHA-256 of z_image_fp8_blockwise.safetensors),
this placeholder should be replaced. The important thing is that the schema
exists and the original_v1 trajectories are fully identified.

Usage:
    PYTHONUNBUFFERED=1 .venv/Scripts/python.exe ^
        F:\\dox\\repos\\ai\\futudiffu\\scripts_ii\\backfill_v2_hashes.py

    Optional flags:
        --dry-run    Print what would change without writing
        --v2-dir     Override V2 dataset directory
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import pyarrow as pa
import pyarrow.parquet as pq

from src_ii.sampling_identity import (
    compute_adapter_set_hash,
    compute_model_state_hash,
    compute_trajectory_hash,
    serialize_active_adapters,
)


BASE_MODEL_HASH_PLACEHOLDER = "z_image_v1"

KNOWN_BASE_RUN_NAMES = {"original_v1"}

UNKNOWN_ADAPTER_RUN_NAMES = {"policy_rollout_v1", "2xh100_20260216"}


def backfill_hashes(
    v2_dir: Path,
    dry_run: bool = False,
) -> dict:
    """Backfill hash columns in the V2 parquet index.

    Args:
        v2_dir: Path to the V2 dataset root directory.
        dry_run: If True, print what would change without writing.

    Returns:
        Summary dict with counts of rows updated by category.
    """
    index_path = v2_dir / "index.parquet"
    if not index_path.exists():
        raise FileNotFoundError(f"V2 index not found: {index_path}")

    table = pq.read_table(str(index_path))
    rows = table.to_pylist()

    n_known_filled = 0
    n_unknown_partial = 0
    n_already_filled = 0
    n_total = len(rows)

    for row in rows:
        if row.get("model_state_hash") is not None:
            n_already_filled += 1
            continue

        run_name = row.get("run_name", "")

        if run_name in KNOWN_BASE_RUN_NAMES:
            base_hash = BASE_MODEL_HASH_PLACEHOLDER
            adapter_list = []
            adapter_set_h = compute_adapter_set_hash(adapter_list)  # ""
            model_state_h = compute_model_state_hash(base_hash, adapter_set_h)
            traj_h = compute_trajectory_hash(
                model_state_hash=model_state_h,
                prompt=row["prompt"],
                seed=int(row["seed"]),
                cfg=float(row["cfg"]),
                n_steps=int(row["n_steps"]),
                width=int(row["width"]),
                height=int(row["height"]),
            )

            row["base_model_hash"] = base_hash
            row["adapter_set_hash"] = adapter_set_h  # ""
            row["model_state_hash"] = model_state_h
            row["trajectory_hash"] = traj_h
            row["active_adapters"] = serialize_active_adapters(adapter_list)  # "[]"
            n_known_filled += 1

        elif run_name in UNKNOWN_ADAPTER_RUN_NAMES:
            row["base_model_hash"] = BASE_MODEL_HASH_PLACEHOLDER
            row["adapter_set_hash"] = None  # unknown
            row["model_state_hash"] = None  # cannot compute
            row["trajectory_hash"] = None   # cannot compute
            row["active_adapters"] = None   # unknown
            n_unknown_partial += 1

        else:
            row["base_model_hash"] = None
            row["adapter_set_hash"] = None
            row["model_state_hash"] = None
            row["trajectory_hash"] = None
            row["active_adapters"] = None
            n_unknown_partial += 1

    summary = {
        "total_rows": n_total,
        "known_filled": n_known_filled,
        "unknown_partial": n_unknown_partial,
        "already_filled": n_already_filled,
        "dry_run": dry_run,
    }

    print(f"Backfill summary:")
    print(f"  Total rows:        {n_total}")
    print(f"  Fully filled:      {n_known_filled} (original_v1, all hashes computed)")
    print(f"  Partially filled:  {n_unknown_partial} (base_model_hash only, adapters unknown)")
    print(f"  Already filled:    {n_already_filled} (skipped)")

    if dry_run:
        print(f"\n  DRY RUN -- no changes written.")
        return summary

    hash_columns = {
        "model_state_hash": pa.utf8(),
        "base_model_hash": pa.utf8(),
        "adapter_set_hash": pa.utf8(),
        "trajectory_hash": pa.utf8(),
        "active_adapters": pa.utf8(),
    }

    existing_col_names = set(table.column_names)
    output_fields = list(table.schema)
    for col_name, col_type in hash_columns.items():
        if col_name not in existing_col_names:
            output_fields.append(pa.field(col_name, col_type))
    output_schema = pa.schema(output_fields)

    updated_table = pa.Table.from_pylist(rows, schema=output_schema)

    temp_path = index_path.with_suffix(".parquet.tmp")
    pq.write_table(
        updated_table,
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
            "base_model_hash",
        ],
        row_group_size=10_000,
    )
    os.replace(str(temp_path), str(index_path))

    print(f"\n  Written updated index: {index_path}")
    print(f"  Schema columns: {updated_table.column_names}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Backfill sampling state identity hashes into V2 dataset index."
    )
    parser.add_argument("--v2-dir", type=str,
                        default=str(REPO_ROOT / "btrm_dataset_v2"),
                        help="Path to V2 dataset directory.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change without writing.")
    args = parser.parse_args()

    v2_dir = Path(args.v2_dir)
    print(f"Backfilling hashes in: {v2_dir}")
    print()

    summary = backfill_hashes(v2_dir, dry_run=args.dry_run)

    if not args.dry_run:
        print("\nVerification sample:")
        table = pq.read_table(str(v2_dir / "index.parquet"))
        import pyarrow.compute as pc

        if "run_name" in table.column_names:
            v1_mask = pc.equal(table.column("run_name"), "original_v1")
            v1_rows = table.filter(v1_mask)
            if len(v1_rows) > 0:
                row = {col: v1_rows.column(col)[0].as_py() for col in v1_rows.column_names}
                print(f"\n  original_v1 sample (traj_id={row['traj_id']}):")
                print(f"    base_model_hash:  {row.get('base_model_hash')}")
                print(f"    adapter_set_hash: {row.get('adapter_set_hash')!r}")
                print(f"    model_state_hash: {row.get('model_state_hash')}")
                print(f"    trajectory_hash:  {row.get('trajectory_hash')}")
                print(f"    active_adapters:  {row.get('active_adapters')}")

            pr_mask = pc.equal(table.column("run_name"), "policy_rollout_v1")
            pr_rows = table.filter(pr_mask)
            if len(pr_rows) > 0:
                row = {col: pr_rows.column(col)[0].as_py() for col in pr_rows.column_names}
                print(f"\n  policy_rollout_v1 sample (traj_id={row['traj_id']}):")
                print(f"    base_model_hash:  {row.get('base_model_hash')}")
                print(f"    adapter_set_hash: {row.get('adapter_set_hash')}")
                print(f"    model_state_hash: {row.get('model_state_hash')}")
                print(f"    trajectory_hash:  {row.get('trajectory_hash')}")
                print(f"    active_adapters:  {row.get('active_adapters')}")


if __name__ == "__main__":
    main()

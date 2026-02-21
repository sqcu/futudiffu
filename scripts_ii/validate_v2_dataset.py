r"""Validate V2 dataset integrity: index, blobs, cross-reference with V1, VAE decode.

Five-phase validation:
  1. Index validation (no GPU): parquet schema, duplicate IDs, blob references,
     provenance metadata, per-source distributions.
  2. Blob integrity (no GPU): safetensors header parse, tensor key existence,
     shape verification against index metadata.
  3. Random sample VAE decode (GPU): decode sampled latents via inference server,
     check pixel validity, save decoded images, compare against V1 if available.
  4. Cross-reference with V1 (no GPU): bitwise tensor comparison for trajectories
     present in both V1 and V2.
  5. Report generation: PASS/FAIL per check, detailed failures, summary stats.

Execution:
  # Non-GPU validation:
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\validate_v2_dataset.py --skip-vae

  # Full validation with VAE decode (requires running inference server):
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\validate_v2_dataset.py

  # Custom dataset path and sample percentage:
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\validate_v2_dataset.py \
      --dataset-dir F:\dox\repos\ai\futudiffu\btrm_dataset_v2_gpu0 \
      --v1-dir F:\dox\repos\ai\futudiffu\btrm_dataset \
      --sample-pct 5
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyarrow as pa
import pyarrow.parquet as pq
from safetensors import safe_open

from futudiffu.dataset_v2 import INDEX_SCHEMA, DatasetReader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = REPO_ROOT / "dataset_audit_output" / "v2_validation"
EXPECTED_LATENT_CHANNELS = 16
EXPECTED_DTYPE = "bfloat16"

# Required metadata columns that must be non-null for every trajectory.
_REQUIRED_NON_NULL_COLS = [
    "traj_id", "prompt", "seed", "cfg", "width", "height",
    "n_steps", "attention_backend", "batch_type", "blob_file",
    "key_prefix", "n_tensors", "bytes_total", "created_at",
    "latent_channels", "latent_height", "latent_width", "latent_dtype",
]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def format_size(n_bytes: int) -> str:
    """Human-readable size string."""
    if n_bytes < 1024:
        return f"{n_bytes} B"
    elif n_bytes < 1024 ** 2:
        return f"{n_bytes / 1024:.1f} KB"
    elif n_bytes < 1024 ** 3:
        return f"{n_bytes / 1024 ** 2:.1f} MB"
    else:
        return f"{n_bytes / 1024 ** 3:.2f} GB"


class ValidationResult:
    """Accumulates PASS/FAIL results for a named check category."""

    def __init__(self, name: str):
        self.name = name
        self.passed: list[str] = []
        self.failed: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    def ok(self, msg: str) -> None:
        self.passed.append(msg)

    def fail(self, msg: str) -> None:
        self.failed.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def note(self, msg: str) -> None:
        self.info.append(msg)

    @property
    def status(self) -> str:
        if self.failed:
            return "FAIL"
        elif self.warnings:
            return "WARN"
        return "PASS"

    def render(self) -> list[str]:
        lines = []
        lines.append(f"  [{self.status}] {self.name}")
        for msg in self.passed:
            lines.append(f"    [PASS] {msg}")
        for msg in self.failed:
            lines.append(f"    [FAIL] {msg}")
        for msg in self.warnings:
            lines.append(f"    [WARN] {msg}")
        for msg in self.info:
            lines.append(f"    [INFO] {msg}")
        return lines


# ---------------------------------------------------------------------------
# Phase 1: Index validation
# ---------------------------------------------------------------------------

def validate_index(dataset_dir: Path) -> tuple[ValidationResult, pa.Table | None]:
    """Validate the parquet index file.

    Returns (result, table) where table is the loaded pyarrow Table or None
    if the index could not be loaded.
    """
    result = ValidationResult("Phase 1: Index Validation")

    index_path = dataset_dir / "index.parquet"
    if not index_path.exists():
        result.fail(f"index.parquet does not exist at {index_path}")
        return result, None

    result.ok(f"index.parquet exists ({format_size(index_path.stat().st_size)})")

    # Load the table
    try:
        table = pq.read_table(str(index_path))
    except Exception as e:
        result.fail(f"Failed to read index.parquet: {e}")
        return result, None

    result.ok(f"Parquet readable: {len(table)} rows, {len(table.column_names)} columns")

    # --- Schema check ---
    expected_cols = set(INDEX_SCHEMA.names)
    actual_cols = set(table.column_names)
    missing_cols = expected_cols - actual_cols
    extra_cols = actual_cols - expected_cols
    if missing_cols:
        result.fail(f"Missing columns vs INDEX_SCHEMA: {sorted(missing_cols)}")
    else:
        result.ok(f"All {len(expected_cols)} expected columns present")
    if extra_cols:
        result.warn(f"Extra columns not in INDEX_SCHEMA: {sorted(extra_cols)}")

    if len(table) == 0:
        result.warn("Index has 0 rows (empty dataset)")
        return result, table

    # --- Trajectory ID uniqueness ---
    traj_ids = table.column("traj_id").to_pylist()
    unique_ids = set(traj_ids)
    if len(unique_ids) < len(traj_ids):
        n_dups = len(traj_ids) - len(unique_ids)
        # Find which IDs are duplicated
        seen = set()
        dup_ids = set()
        for tid in traj_ids:
            if tid in seen:
                dup_ids.add(tid)
            seen.add(tid)
        result.fail(f"{n_dups} duplicate traj_id(s): {sorted(dup_ids)[:20]}")
    else:
        result.ok(f"All {len(traj_ids)} traj_ids are unique")

    # --- traj_id monotonicity ---
    is_monotonic = all(traj_ids[i] < traj_ids[i + 1] for i in range(len(traj_ids) - 1))
    if is_monotonic:
        result.ok(f"traj_ids are strictly monotonically increasing ({traj_ids[0]}..{traj_ids[-1]})")
    else:
        result.warn("traj_ids are NOT strictly monotonically increasing")

    # --- Required non-null columns ---
    for col_name in _REQUIRED_NON_NULL_COLS:
        if col_name not in table.column_names:
            continue  # Already flagged as missing above
        col = table.column(col_name)
        n_nulls = col.null_count
        if n_nulls > 0:
            result.fail(f"Column '{col_name}' has {n_nulls} null value(s) (expected 0)")
        else:
            result.ok(f"Column '{col_name}': 0 nulls")

    # --- Blob file references point to existing files ---
    blobs_dir = dataset_dir / "blobs"
    blob_files_referenced = set(table.column("blob_file").to_pylist())
    missing_blobs = []
    for bf in blob_files_referenced:
        blob_path = blobs_dir / bf
        if not blob_path.exists():
            missing_blobs.append(bf)
    if missing_blobs:
        result.fail(f"{len(missing_blobs)} referenced blob file(s) not found: {missing_blobs[:10]}")
    else:
        result.ok(f"All {len(blob_files_referenced)} referenced blob files exist on disk")

    # Check for WIP blob references (should not exist in a sealed dataset)
    if "blob_wip.safetensors" in blob_files_referenced:
        result.warn("Index references 'blob_wip.safetensors' -- dataset may not be fully sealed")

    # --- step_indices + has_final consistency with n_tensors ---
    rows = table.to_pylist()
    n_tensor_mismatches = []
    for row in rows:
        expected_n = len(row["step_indices"]) + (1 if row["has_final"] else 0)
        if row["n_tensors"] != expected_n:
            n_tensor_mismatches.append(
                f"traj_id={row['traj_id']}: n_tensors={row['n_tensors']} "
                f"but step_indices has {len(row['step_indices'])} entries + "
                f"has_final={row['has_final']} = expected {expected_n}"
            )
    if n_tensor_mismatches:
        result.fail(f"{len(n_tensor_mismatches)} n_tensors mismatches")
        for msg in n_tensor_mismatches[:5]:
            result.note(msg)
    else:
        result.ok("n_tensors consistent with step_indices + has_final for all rows")

    # --- Distribution reports ---
    # Per-source-device (attention_backend)
    backend_counts = Counter(table.column("attention_backend").to_pylist())
    result.note(f"Attention backend distribution: {dict(backend_counts.most_common())}")

    # Per-resolution
    widths = table.column("width").to_pylist()
    heights = table.column("height").to_pylist()
    res_counts = Counter(f"{w}x{h}" for w, h in zip(widths, heights))
    result.note(f"Resolution distribution: {dict(res_counts.most_common())}")

    # Per-step-count
    step_counts = Counter(table.column("n_steps").to_pylist())
    result.note(f"Step count distribution: {dict(step_counts.most_common())}")

    # Per batch_type
    type_counts = Counter(table.column("batch_type").to_pylist())
    result.note(f"Batch type distribution: {dict(type_counts.most_common())}")

    # Per blob_file (trajectory density)
    blob_counts = Counter(table.column("blob_file").to_pylist())
    result.note(f"Trajectories per blob: {dict(blob_counts.most_common())}")

    # Latent dimension distribution
    lc = Counter(table.column("latent_channels").to_pylist())
    result.note(f"Latent channels distribution: {dict(lc.most_common())}")

    # Latent dtype distribution
    ld = Counter(table.column("latent_dtype").to_pylist())
    result.note(f"Latent dtype distribution: {dict(ld.most_common())}")

    return result, table


# ---------------------------------------------------------------------------
# Phase 2: Blob integrity
# ---------------------------------------------------------------------------

def validate_blobs(dataset_dir: Path, table: pa.Table) -> ValidationResult:
    """Validate all safetensors blob files referenced by the index."""
    result = ValidationResult("Phase 2: Blob Integrity")

    blobs_dir = dataset_dir / "blobs"
    if not blobs_dir.exists():
        result.fail(f"blobs/ directory does not exist at {blobs_dir}")
        return result

    rows = table.to_pylist()

    # Group rows by blob_file for efficient per-blob validation
    blob_to_rows: dict[str, list[dict]] = {}
    for row in rows:
        bf = row["blob_file"]
        blob_to_rows.setdefault(bf, []).append(row)

    total_tensor_count = 0
    total_tensor_bytes = 0
    total_blob_bytes_on_disk = 0

    for blob_file, blob_rows in sorted(blob_to_rows.items()):
        blob_path = blobs_dir / blob_file
        if not blob_path.exists():
            # Already flagged in Phase 1; skip here.
            result.fail(f"Blob {blob_file} not found on disk")
            continue

        blob_disk_size = blob_path.stat().st_size
        total_blob_bytes_on_disk += blob_disk_size

        # Try to open the safetensors file
        try:
            handle = safe_open(str(blob_path), framework="pt", device="cpu")
        except Exception as e:
            result.fail(f"Blob {blob_file} failed to open as safetensors: {e}")
            continue

        # Get all keys in this blob
        blob_keys = set(handle.keys())

        # Validate each trajectory's tensor references
        for row in blob_rows:
            traj_id = row["traj_id"]
            key_prefix = row["key_prefix"]
            step_indices = row["step_indices"]
            has_final = row["has_final"]
            expected_c = row["latent_channels"]
            expected_h = row["latent_height"]
            expected_w = row["latent_width"]
            expected_dtype = row["latent_dtype"]

            # Build expected keys
            expected_keys = []
            for step_idx in step_indices:
                expected_keys.append(f"{key_prefix}/step_{step_idx:02d}")
            if has_final:
                expected_keys.append(f"{key_prefix}/final")

            # Check all expected keys exist in blob
            for key in expected_keys:
                if key not in blob_keys:
                    result.fail(
                        f"traj_id={traj_id}: key '{key}' missing from blob {blob_file}"
                    )
                    continue

                # Verify tensor shape
                try:
                    tensor = handle.get_tensor(key)
                except Exception as e:
                    result.fail(
                        f"traj_id={traj_id}: failed to load tensor '{key}' "
                        f"from {blob_file}: {e}"
                    )
                    continue

                total_tensor_count += 1
                total_tensor_bytes += tensor.nelement() * tensor.element_size()

                # Shape check: stored as (C, H, W) with batch dim squeezed
                if tensor.dim() != 3:
                    result.fail(
                        f"traj_id={traj_id}, key='{key}': expected 3D tensor (C,H,W), "
                        f"got {tensor.dim()}D shape {tuple(tensor.shape)}"
                    )
                    continue

                c, h, w = tensor.shape
                if c != expected_c:
                    result.fail(
                        f"traj_id={traj_id}, key='{key}': latent_channels mismatch: "
                        f"tensor has C={c}, index says {expected_c}"
                    )
                if h != expected_h:
                    result.fail(
                        f"traj_id={traj_id}, key='{key}': latent_height mismatch: "
                        f"tensor has H={h}, index says {expected_h}"
                    )
                if w != expected_w:
                    result.fail(
                        f"traj_id={traj_id}, key='{key}': latent_width mismatch: "
                        f"tensor has W={w}, index says {expected_w}"
                    )

                # Dtype check
                tensor_dtype_str = str(tensor.dtype).replace("torch.", "")
                if tensor_dtype_str != expected_dtype:
                    result.fail(
                        f"traj_id={traj_id}, key='{key}': dtype mismatch: "
                        f"tensor is {tensor_dtype_str}, index says {expected_dtype}"
                    )

        # Check for orphan keys in the blob (keys not referenced by any index row)
        referenced_keys = set()
        for row in blob_rows:
            kp = row["key_prefix"]
            for step_idx in row["step_indices"]:
                referenced_keys.add(f"{kp}/step_{step_idx:02d}")
            if row["has_final"]:
                referenced_keys.add(f"{kp}/final")

        orphan_keys = blob_keys - referenced_keys
        if orphan_keys:
            result.warn(
                f"Blob {blob_file} has {len(orphan_keys)} key(s) not referenced "
                f"by index: {sorted(orphan_keys)[:10]}"
            )

        del handle  # Release file handle

    # Also check for orphan blob files on disk not referenced by the index
    all_blobs_on_disk = set()
    for f in blobs_dir.iterdir():
        if f.name.startswith("blob_") and f.suffix == ".safetensors":
            all_blobs_on_disk.add(f.name)

    referenced_blob_files = set(blob_to_rows.keys())
    orphan_blobs = all_blobs_on_disk - referenced_blob_files
    if orphan_blobs:
        result.warn(
            f"{len(orphan_blobs)} blob file(s) on disk not referenced by index: "
            f"{sorted(orphan_blobs)[:10]}"
        )

    if not result.failed:
        result.ok(
            f"All tensor references valid across {len(blob_to_rows)} blob(s)"
        )

    result.note(f"Total tensors validated: {total_tensor_count}")
    result.note(f"Total tensor data: {format_size(total_tensor_bytes)}")
    result.note(f"Total blob data on disk: {format_size(total_blob_bytes_on_disk)}")

    return result


# ---------------------------------------------------------------------------
# Phase 3: VAE decode validation (GPU required)
# ---------------------------------------------------------------------------

def validate_vae_decode(
    dataset_dir: Path,
    table: pa.Table,
    sample_pct: float,
    server_endpoint: str,
    output_dir: Path,
    v1_dir: Path | None,
) -> ValidationResult:
    """Sample trajectories, VAE-decode latents, check pixel validity.

    Optionally compares decoded images against V1 source (pixel-space MSE).
    """
    import torch
    from futudiffu.client import InferenceClient

    result = ValidationResult("Phase 3: VAE Decode Validation")

    n_total = len(table)
    n_sample = max(5, min(20, int(n_total * sample_pct / 100.0)))
    n_sample = min(n_sample, n_total)

    result.note(f"Sampling {n_sample} of {n_total} trajectories ({sample_pct}% requested)")

    # Sample trajectory IDs
    import random
    rng = random.Random(42)  # deterministic for reproducibility
    traj_ids = table.column("traj_id").to_pylist()
    sampled_ids = rng.sample(traj_ids, n_sample)
    sampled_ids.sort()

    result.note(f"Sampled traj_ids: {sampled_ids}")

    # Open V2 reader
    reader = DatasetReader(str(dataset_dir))

    # Connect to inference server
    try:
        client = InferenceClient(server_endpoint, timeout_ms=120_000)
    except Exception as e:
        result.fail(f"Failed to connect to inference server at {server_endpoint}: {e}")
        return result

    # Optionally prepare V1 reader
    v1_available = v1_dir is not None and v1_dir.exists()
    v1_manifest = None
    if v1_available:
        manifest_path = v1_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                v1_manifest = json.load(f)

    samples_dir = output_dir / "v2_validation_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    for traj_id in sampled_ids:
        traj_tag = f"traj_{traj_id:06d}"
        print(f"  Validating {traj_tag} ...")

        try:
            meta, accessor = reader[traj_id]
        except Exception as e:
            result.fail(f"{traj_tag}: failed to load from V2: {e}")
            continue

        # Load all step latents
        try:
            all_latents = accessor.load_all()
        except Exception as e:
            result.fail(f"{traj_tag}: failed to load tensors: {e}")
            continue

        # Pick one latent to decode (prefer "final", fall back to last step)
        if "final" in all_latents:
            decode_label = "final"
        else:
            # Pick the highest numbered step
            step_labels = [k for k in all_latents if k.startswith("step_")]
            if not step_labels:
                result.fail(f"{traj_tag}: no step or final latent found")
                continue
            decode_label = sorted(step_labels)[-1]

        latent = all_latents[decode_label]

        # Ensure latent is on CPU and has batch dim (1, C, H, W)
        if latent.dim() == 3:
            latent = latent.unsqueeze(0)

        # VAE decode via inference server
        try:
            image = client.vae_decode(latent)
        except Exception as e:
            result.fail(f"{traj_tag}: VAE decode failed: {e}")
            continue

        # Check pixel validity
        if torch.isnan(image).any():
            result.fail(f"{traj_tag}: decoded image contains NaN values")
        elif torch.isinf(image).any():
            result.fail(f"{traj_tag}: decoded image contains Inf values")
        elif image.abs().max().item() == 0.0:
            result.fail(f"{traj_tag}: decoded image is all zeros")
        elif image.min().item() < -0.5 or image.max().item() > 1.5:
            result.warn(
                f"{traj_tag}: decoded image pixel range suspicious: "
                f"[{image.min().item():.3f}, {image.max().item():.3f}] "
                f"(expected roughly [0, 1])"
            )
        else:
            result.ok(
                f"{traj_tag}: decoded OK, pixel range "
                f"[{image.min().item():.3f}, {image.max().item():.3f}], "
                f"shape {tuple(image.shape)}"
            )

        # Save decoded image as PNG
        try:
            _save_image_tensor(
                image,
                samples_dir / f"{traj_tag}_{decode_label}.png",
            )
        except Exception as e:
            result.warn(f"{traj_tag}: failed to save PNG: {e}")

        # --- V1 cross-comparison (pixel-space MSE) ---
        if v1_available and v1_manifest is not None:
            # Try to find matching V1 trajectory. The V2 traj_id may not
            # correspond directly to V1 traj index. Match by prompt + seed.
            v1_match = _find_v1_match(
                v2_meta=meta,
                v1_manifest=v1_manifest,
                v1_dir=v1_dir,
            )
            if v1_match is not None:
                v1_traj_idx, v1_traj_dir = v1_match
                v1_pt_path = v1_traj_dir / f"{decode_label}.pt"
                if v1_pt_path.exists():
                    try:
                        v1_latent = torch.load(str(v1_pt_path), weights_only=True)
                        if v1_latent.dim() == 3:
                            v1_latent = v1_latent.unsqueeze(0)
                        # Decode V1 latent
                        v1_image = client.vae_decode(v1_latent)
                        # Compute pixel-space MSE
                        mse = (image.float() - v1_image.float()).pow(2).mean().item()
                        if mse == 0.0:
                            result.ok(
                                f"{traj_tag}: V1 cross-ref pixel MSE = 0.0 "
                                f"(exact match, v1_traj_idx={v1_traj_idx})"
                            )
                        elif mse < 1e-6:
                            result.ok(
                                f"{traj_tag}: V1 cross-ref pixel MSE = {mse:.2e} "
                                f"(near-zero, v1_traj_idx={v1_traj_idx})"
                            )
                        else:
                            result.fail(
                                f"{traj_tag}: V1 cross-ref pixel MSE = {mse:.6f} "
                                f"(non-zero! v1_traj_idx={v1_traj_idx})"
                            )
                    except Exception as e:
                        result.warn(
                            f"{traj_tag}: V1 cross-ref decode failed: {e}"
                        )

    reader.close()
    return result


from src_ii.rendering import save_tensor_as_png as _save_image_tensor  # noqa: E402


def _find_v1_match(
    v2_meta: dict,
    v1_manifest: dict,
    v1_dir: Path,
) -> tuple[int, Path] | None:
    """Find a V1 trajectory matching the V2 metadata by (prompt, seed).

    Returns (v1_traj_index, v1_traj_dir) or None if no match found.
    """
    v2_prompt = v2_meta.get("prompt", "")
    v2_seed = v2_meta.get("seed")
    records = v1_manifest.get("records", [])

    for i, rec in enumerate(records):
        if rec.get("prompt") == v2_prompt and rec.get("seed") == v2_seed:
            traj_dir = v1_dir / "latents" / f"traj_{i:06d}"
            if traj_dir.exists():
                return i, traj_dir

    return None


# ---------------------------------------------------------------------------
# Phase 4: Cross-reference with V1 (tensor-space, no GPU)
# ---------------------------------------------------------------------------

def validate_v1_crossref(
    dataset_dir: Path,
    table: pa.Table,
    v1_dir: Path,
) -> ValidationResult:
    """For trajectories with both V1 and V2 representations, compare tensors
    bitwise in tensor space (no VAE decode needed).
    """
    import torch

    result = ValidationResult("Phase 4: V1 Cross-Reference (Tensor Space)")

    if not v1_dir.exists():
        result.warn(f"V1 directory does not exist: {v1_dir}")
        return result

    manifest_path = v1_dir / "manifest.json"
    if not manifest_path.exists():
        result.warn(f"V1 manifest.json not found at {manifest_path}")
        return result

    with open(manifest_path) as f:
        v1_manifest = json.load(f)

    v1_records = v1_manifest.get("records", [])
    if not v1_records:
        result.warn("V1 manifest has no records")
        return result

    # Build V1 lookup: (prompt, seed) -> (v1_traj_index, record)
    v1_lookup: dict[tuple[str, int], tuple[int, dict]] = {}
    for i, rec in enumerate(v1_records):
        key = (rec.get("prompt", ""), rec.get("seed", -1))
        v1_lookup[key] = (i, rec)

    # Open V2 reader
    reader = DatasetReader(str(dataset_dir))
    rows = table.to_pylist()

    n_matched = 0
    n_compared = 0
    n_exact_match = 0
    n_mismatch = 0
    mismatch_details: list[str] = []

    for row in rows:
        v2_key = (row["prompt"], row["seed"])
        if v2_key not in v1_lookup:
            continue

        v1_idx, v1_rec = v1_lookup[v2_key]
        n_matched += 1

        traj_id = row["traj_id"]
        v1_traj_dir = v1_dir / "latents" / f"traj_{v1_idx:06d}"
        if not v1_traj_dir.exists():
            result.warn(
                f"V1 traj dir missing for matched trajectory: "
                f"v2_traj_id={traj_id}, v1_idx={v1_idx}"
            )
            continue

        # Load V2 tensors
        try:
            _, accessor = reader[traj_id]
            v2_tensors = accessor.load_all()
        except Exception as e:
            result.warn(f"traj_id={traj_id}: failed to load V2 tensors: {e}")
            continue

        # Compare each step that exists in both
        for step_label, v2_tensor in v2_tensors.items():
            v1_pt_path = v1_traj_dir / f"{step_label}.pt"
            if not v1_pt_path.exists():
                continue

            try:
                v1_tensor = torch.load(str(v1_pt_path), weights_only=True)
            except Exception as e:
                result.warn(
                    f"traj_id={traj_id}, {step_label}: failed to load V1 tensor: {e}"
                )
                continue

            # Ensure both have the same shape for comparison
            # V2 returns (1, C, H, W), V1 may be (1, C, H, W) or (C, H, W)
            if v1_tensor.dim() == 3:
                v1_tensor = v1_tensor.unsqueeze(0)
            if v2_tensor.dim() == 3:
                v2_tensor = v2_tensor.unsqueeze(0)

            n_compared += 1

            if v1_tensor.shape != v2_tensor.shape:
                msg = (
                    f"traj_id={traj_id} ({step_label}): shape mismatch: "
                    f"V1={tuple(v1_tensor.shape)}, V2={tuple(v2_tensor.shape)}"
                )
                mismatch_details.append(msg)
                n_mismatch += 1
                continue

            # Compute max absolute difference (tensor space, no dtype conversion)
            # Use float32 for stable comparison
            diff = (v2_tensor.float() - v1_tensor.float()).abs()
            max_abs_diff = diff.max().item()

            if max_abs_diff == 0.0:
                n_exact_match += 1
            else:
                n_mismatch += 1
                msg = (
                    f"traj_id={traj_id} ({step_label}): max_abs_diff={max_abs_diff:.2e} "
                    f"(v1_idx={v1_idx})"
                )
                mismatch_details.append(msg)

    reader.close()

    result.note(
        f"Matched {n_matched} V2 trajectories to V1 by (prompt, seed)"
    )
    result.note(
        f"Compared {n_compared} tensor pairs: "
        f"{n_exact_match} exact matches, {n_mismatch} mismatches"
    )

    if n_compared == 0:
        result.warn("No tensor pairs could be compared (no V1/V2 overlap)")
    elif n_mismatch == 0:
        result.ok(
            f"All {n_compared} tensor comparisons are bitwise exact"
        )
    else:
        result.fail(
            f"{n_mismatch} of {n_compared} tensor comparisons have non-zero differences"
        )
        for detail in mismatch_details[:20]:
            result.note(detail)

    return result


# ---------------------------------------------------------------------------
# Phase 5: Report generation
# ---------------------------------------------------------------------------

def generate_report(
    results: list[ValidationResult],
    dataset_dir: Path,
    elapsed_seconds: float,
) -> str:
    """Generate the full validation report as a string."""
    lines: list[str] = []
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines.append("=" * 78)
    lines.append(f"  V2 DATASET VALIDATION REPORT -- {now_str}")
    lines.append(f"  Dataset: {dataset_dir}")
    lines.append(f"  Elapsed: {elapsed_seconds:.1f}s")
    lines.append("=" * 78)

    # Summary
    lines.append("")
    lines.append("SUMMARY")
    lines.append("-" * 78)
    overall_pass = True
    for r in results:
        status = r.status
        if status == "FAIL":
            overall_pass = False
        lines.append(f"  {status:4s}  {r.name}")
    lines.append("")
    lines.append(f"  OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    lines.append("")

    # Detailed results
    lines.append("DETAILED RESULTS")
    lines.append("-" * 78)
    for r in results:
        lines.append("")
        lines.extend(r.render())

    lines.append("")
    lines.append("=" * 78)
    lines.append("  END OF VALIDATION REPORT")
    lines.append("=" * 78)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate V2 dataset integrity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(REPO_ROOT / "btrm_dataset_v2"),
        help="Path to the V2 dataset directory (default: btrm_dataset_v2/)",
    )
    parser.add_argument(
        "--v1-dir",
        type=str,
        default=str(REPO_ROOT / "btrm_dataset"),
        help="Path to V1 dataset directory for cross-reference (default: btrm_dataset/)",
    )
    parser.add_argument(
        "--skip-vae",
        action="store_true",
        help="Skip GPU-dependent VAE decode checks (run only index + blob + cross-ref)",
    )
    parser.add_argument(
        "--sample-pct",
        type=float,
        default=2.0,
        help="Percentage of trajectories to VAE-decode validate (default: 2%%)",
    )
    parser.add_argument(
        "--server-endpoint",
        type=str,
        default="tcp://localhost:5555",
        help="ZeroMQ endpoint for the inference server (default: tcp://localhost:5555)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for report and samples",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    v1_dir = Path(args.v1_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    results: list[ValidationResult] = []

    print(f"\n{'=' * 60}")
    print(f"  V2 Dataset Validation")
    print(f"  Dataset: {dataset_dir}")
    print(f"  V1 ref:  {v1_dir}")
    print(f"  Skip VAE: {args.skip_vae}")
    print(f"{'=' * 60}\n")

    # --- Phase 1: Index validation ---
    print("Phase 1: Index validation ...")
    idx_result, table = validate_index(dataset_dir)
    results.append(idx_result)
    print(f"  -> {idx_result.status} ({len(idx_result.passed)} passed, "
          f"{len(idx_result.failed)} failed, {len(idx_result.warnings)} warnings)")

    if table is None or len(table) == 0:
        print("\n  Cannot proceed without a valid, non-empty index.")
        if table is not None and len(table) == 0:
            # Still generate report for empty dataset
            elapsed = time.perf_counter() - t0
            report = generate_report(results, dataset_dir, elapsed)
            _write_report(report, output_dir)
            return 1 if any(r.status == "FAIL" for r in results) else 0
        elapsed = time.perf_counter() - t0
        report = generate_report(results, dataset_dir, elapsed)
        _write_report(report, output_dir)
        return 1

    # --- Phase 2: Blob integrity ---
    print("\nPhase 2: Blob integrity ...")
    blob_result = validate_blobs(dataset_dir, table)
    results.append(blob_result)
    print(f"  -> {blob_result.status} ({len(blob_result.passed)} passed, "
          f"{len(blob_result.failed)} failed, {len(blob_result.warnings)} warnings)")

    # --- Phase 3: VAE decode (if not skipped) ---
    if not args.skip_vae:
        print(f"\nPhase 3: VAE decode validation ({args.sample_pct}% sample) ...")
        vae_result = validate_vae_decode(
            dataset_dir=dataset_dir,
            table=table,
            sample_pct=args.sample_pct,
            server_endpoint=args.server_endpoint,
            output_dir=output_dir,
            v1_dir=v1_dir if v1_dir.exists() else None,
        )
        results.append(vae_result)
        print(f"  -> {vae_result.status} ({len(vae_result.passed)} passed, "
              f"{len(vae_result.failed)} failed, {len(vae_result.warnings)} warnings)")
    else:
        skipped = ValidationResult("Phase 3: VAE Decode Validation")
        skipped.note("Skipped (--skip-vae)")
        results.append(skipped)
        print("\nPhase 3: Skipped (--skip-vae)")

    # --- Phase 4: V1 cross-reference ---
    print("\nPhase 4: V1 cross-reference ...")
    crossref_result = validate_v1_crossref(dataset_dir, table, v1_dir)
    results.append(crossref_result)
    print(f"  -> {crossref_result.status} ({len(crossref_result.passed)} passed, "
          f"{len(crossref_result.failed)} failed, {len(crossref_result.warnings)} warnings)")

    # --- Phase 5: Report ---
    elapsed = time.perf_counter() - t0
    report = generate_report(results, dataset_dir, elapsed)
    _write_report(report, output_dir)

    print(f"\nTotal elapsed: {elapsed:.1f}s")

    # Return exit code: 0 if all pass, 1 if any fail
    any_fail = any(r.status == "FAIL" for r in results)
    return 1 if any_fail else 0


def _write_report(report: str, output_dir: Path) -> None:
    """Write the report to stdout and to disk."""
    print("\n" + report)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Timestamped copy
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_path = output_dir / f"v2_validation_report_{timestamp}.txt"
    with open(timestamped_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Latest copy
    latest_path = output_dir / "v2_validation_report.txt"
    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nReport saved to: {timestamped_path}")
    print(f"Latest copy at:  {latest_path}")


if __name__ == "__main__":
    sys.exit(main())

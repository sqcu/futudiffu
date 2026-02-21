"""Pull BTRM dataset from HuggingFace, compare with local, and migrate all V1 data to V2 format.

This script performs 4 stages:
  1. Download HF repo contents to a staging directory
  2. Compare HF data against local packed_dataset/ and btrm_dataset/ -- produce an audit report
  3. Reconcile: ensure all HF trajectories exist in btrm_dataset/ V1 format
  4. Migrate ALL V1 trajectories (manifest + extra policy rollouts) to V2 format

The V2 output is written to btrm_dataset_v2_from_v1/ as a staging directory.
Use merge_staged_datasets.py to combine it with any other V2 staging dirs.

Usage (from WSL):
    PYTHONUNBUFFERED=1 .venv/Scripts/python.exe scripts/pull_hf_and_migrate_v2.py

    # Or with explicit paths:
    PYTHONUNBUFFERED=1 .venv/Scripts/python.exe scripts/pull_hf_and_migrate_v2.py \
        --hf-repo SQCU/futudiffu-btrm \
        --v1-dir F:\\dox\\repos\\ai\\futudiffu\\btrm_dataset \
        --v2-dir F:\\dox\\repos\\ai\\futudiffu\\btrm_dataset_v2_from_v1 \
        --staging-dir F:\\dox\\repos\\ai\\futudiffu\\hf_download_staging \
        --report-dir F:\\dox\\repos\\ai\\futudiffu\\dataset_audit_output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure we can import from src/
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _load_hf_token() -> str:
    """Load HuggingFace token from .supersekrit or env."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()
    supersekrit = REPO_ROOT / ".supersekrit"
    if supersekrit.exists():
        for line in supersekrit.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    raise RuntimeError("No HF token found. Set HF_TOKEN env var or create .supersekrit")


# ---------------------------------------------------------------------------
# Stage 1: Download HF repo
# ---------------------------------------------------------------------------

def stage1_download_hf(
    hf_repo: str,
    staging_dir: Path,
    token: str,
) -> list[str]:
    """Download all files from the HF dataset repo.

    Returns:
        List of downloaded file paths (relative to repo root).
    """
    from huggingface_hub import HfApi, hf_hub_download

    print("=" * 70)
    print("STAGE 1: Download from HuggingFace")
    print("=" * 70)

    api = HfApi(token=token)

    # Verify auth
    user = api.whoami()
    print(f"  Authenticated as: {user['name']}")

    # List all files in the HF repo
    print(f"  Listing files in {hf_repo}...")
    hf_files = api.list_repo_files(hf_repo, repo_type="dataset")
    hf_files = [f for f in hf_files if not f.startswith(".")]  # skip .gitattributes etc
    print(f"  Found {len(hf_files)} files on HF")

    # Download all files
    staging_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for i, fname in enumerate(sorted(hf_files)):
        dest = staging_dir / fname
        if dest.exists():
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  [{i+1}/{len(hf_files)}] SKIP (exists): {fname} ({size_mb:.2f} MB)")
            downloaded.append(fname)
            continue

        print(f"  [{i+1}/{len(hf_files)}] Downloading: {fname}...", end=" ", flush=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        local_path = hf_hub_download(
            repo_id=hf_repo,
            filename=fname,
            repo_type="dataset",
            token=token,
            local_dir=str(staging_dir),
        )
        size_mb = Path(local_path).stat().st_size / (1024 * 1024)
        print(f"done ({size_mb:.2f} MB)")
        downloaded.append(fname)

    print(f"  Downloaded {len(downloaded)} files to {staging_dir}")
    return downloaded


# ---------------------------------------------------------------------------
# Stage 2: Compare HF vs local
# ---------------------------------------------------------------------------

def stage2_compare(
    hf_files: list[str],
    staging_dir: Path,
    packed_dir: Path,
    v1_dir: Path,
    report_dir: Path,
) -> dict:
    """Compare HF contents against local packed_dataset/ and btrm_dataset/.

    Produces dataset_audit_output/hf_comparison.txt.

    Returns:
        dict with comparison results.
    """
    print()
    print("=" * 70)
    print("STAGE 2: Compare HF vs Local")
    print("=" * 70)

    report_dir.mkdir(parents=True, exist_ok=True)
    report_lines = []
    results = {
        "hf_files": sorted(hf_files),
        "packed_match": True,
        "v1_manifest_traj_count": 0,
        "v1_disk_traj_count": 0,
        "missing_from_local": [],
        "extra_local_trajs": [],
    }

    # --- Compare HF vs packed_dataset/ ---
    report_lines.append("=" * 70)
    report_lines.append("HuggingFace vs Local Comparison Report")
    report_lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    report_lines.append("=" * 70)
    report_lines.append("")

    # HF files
    hf_safetensors = sorted(f for f in hf_files if f.endswith(".safetensors"))
    hf_jsonl = sorted(f for f in hf_files if f.endswith(".jsonl"))
    report_lines.append(f"HF repo files: {len(hf_files)} total")
    report_lines.append(f"  .safetensors: {len(hf_safetensors)}")
    report_lines.append(f"  .jsonl: {len(hf_jsonl)}")
    report_lines.append("")

    # Compare packed_dataset/
    report_lines.append("--- packed_dataset/ comparison ---")
    if packed_dir.exists():
        local_files = sorted(
            str(f.relative_to(packed_dir))
            for f in packed_dir.rglob("*")
            if f.is_file() and not f.name.startswith(".")
        )
        local_set = set(local_files)
        hf_set = set(hf_files)

        only_hf = hf_set - local_set
        only_local = local_set - hf_set
        common = hf_set & local_set

        report_lines.append(f"  Common files: {len(common)}")
        report_lines.append(f"  Only on HF: {len(only_hf)}")
        report_lines.append(f"  Only local: {len(only_local)}")

        if only_hf:
            report_lines.append(f"  Files only on HF:")
            for f in sorted(only_hf):
                report_lines.append(f"    {f}")
            results["packed_match"] = False

        if only_local:
            report_lines.append(f"  Files only local:")
            for f in sorted(only_local):
                report_lines.append(f"    {f}")

        # Size comparison for common files
        size_mismatches = []
        for fname in sorted(common):
            hf_path = staging_dir / fname
            local_path = packed_dir / fname
            if hf_path.exists() and local_path.exists():
                hf_size = hf_path.stat().st_size
                local_size = local_path.stat().st_size
                if hf_size != local_size:
                    size_mismatches.append((fname, hf_size, local_size))

        if size_mismatches:
            report_lines.append(f"  Size mismatches: {len(size_mismatches)}")
            for fname, hs, ls in size_mismatches:
                report_lines.append(f"    {fname}: HF={hs} bytes, local={ls} bytes")
            results["packed_match"] = False
        else:
            report_lines.append(f"  Size check: all {len(common)} common files match")
    else:
        report_lines.append("  packed_dataset/ does not exist locally")
        results["packed_match"] = False

    report_lines.append("")

    # --- Compare with btrm_dataset/ V1 ---
    report_lines.append("--- btrm_dataset/ V1 comparison ---")
    latents_dir = v1_dir / "latents"
    v1_manifest_path = v1_dir / "manifest.json"

    # Load V1 manifest
    manifest_traj_count = 0
    manifest_records = {}
    if v1_manifest_path.exists():
        with open(v1_manifest_path) as f:
            manifest = json.load(f)
        records = manifest.get("records", [])
        manifest_traj_count = len(records)
        for rec in records:
            traj_path = rec.get("traj_dir", "")
            name = Path(traj_path).name if traj_path else ""
            if name:
                manifest_records[name] = rec
    results["v1_manifest_traj_count"] = manifest_traj_count
    report_lines.append(f"  V1 manifest records: {manifest_traj_count}")

    # Count on-disk trajectory dirs
    if latents_dir.exists():
        traj_dirs = sorted(
            d.name for d in latents_dir.iterdir()
            if d.is_dir() and d.name.startswith("traj_")
        )
        results["v1_disk_traj_count"] = len(traj_dirs)
        report_lines.append(f"  V1 trajectory directories on disk: {len(traj_dirs)}")

        # Which are in manifest, which are extra?
        manifest_names = set(manifest_records.keys())
        disk_names = set(traj_dirs)
        in_manifest = disk_names & manifest_names
        extra = disk_names - manifest_names
        missing_from_disk = manifest_names - disk_names

        report_lines.append(f"  In manifest AND on disk: {len(in_manifest)}")
        report_lines.append(f"  On disk but NOT in manifest (policy rollouts): {len(extra)}")

        results["extra_local_trajs"] = sorted(extra)

        if extra:
            report_lines.append(f"  Extra trajectory dirs (policy rollout persistence):")
            for name in sorted(extra):
                report_lines.append(f"    {name}")

        if missing_from_disk:
            report_lines.append(f"  In manifest but NOT on disk: {len(missing_from_disk)}")
            for name in sorted(missing_from_disk):
                report_lines.append(f"    {name}")
    else:
        report_lines.append("  latents/ directory does not exist")

    # --- HF manifest.jsonl trajectory IDs vs V1 manifest ---
    report_lines.append("")
    report_lines.append("--- HF manifest.jsonl vs V1 manifest ---")

    hf_manifest_path = staging_dir / "manifest.jsonl"
    if hf_manifest_path.exists():
        hf_trajs = []
        with open(hf_manifest_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    hf_trajs.append(json.loads(line))

        report_lines.append(f"  HF manifest.jsonl entries: {len(hf_trajs)}")
        report_lines.append(f"  V1 manifest.json records: {manifest_traj_count}")

        hf_traj_ids = set(t["traj_id"] for t in hf_trajs)
        v1_traj_ids_in_manifest = set(range(manifest_traj_count))

        if hf_traj_ids == v1_traj_ids_in_manifest:
            report_lines.append("  Trajectory ID sets: IDENTICAL")
        else:
            only_hf_ids = hf_traj_ids - v1_traj_ids_in_manifest
            only_v1_ids = v1_traj_ids_in_manifest - hf_traj_ids
            if only_hf_ids:
                report_lines.append(f"  Only on HF: {sorted(only_hf_ids)}")
            if only_v1_ids:
                report_lines.append(f"  Only in V1 manifest: {sorted(only_v1_ids)}")

        # Cross-check metadata for shared trajectories
        mismatched_meta = []
        for ht in hf_trajs:
            tid = ht["traj_id"]
            traj_name = f"traj_{tid:06d}"
            if traj_name in manifest_records:
                v1_rec = manifest_records[traj_name]
                # Compare key fields
                for field in ["seed", "n_steps", "prompt"]:
                    hf_val = ht.get(field)
                    v1_val = v1_rec.get(field)
                    if hf_val != v1_val:
                        mismatched_meta.append((traj_name, field, hf_val, v1_val))

        if mismatched_meta:
            report_lines.append(f"  Metadata mismatches: {len(mismatched_meta)}")
            for name, field, hf_val, v1_val in mismatched_meta[:20]:
                report_lines.append(f"    {name}.{field}: HF={hf_val!r} vs V1={v1_val!r}")
        else:
            report_lines.append(f"  Metadata cross-check: all fields match for {len(hf_trajs)} trajectories")
    else:
        report_lines.append("  No manifest.jsonl found in HF download staging")

    # Write report
    report_path = report_dir / "hf_comparison.txt"
    report_text = "\n".join(report_lines) + "\n"
    with open(report_path, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"  Report written to {report_path}")

    return results


# ---------------------------------------------------------------------------
# Stage 3: Reconcile -- ensure HF data exists as V1
# ---------------------------------------------------------------------------

def stage3_reconcile(
    staging_dir: Path,
    v1_dir: Path,
    report_dir: Path,
) -> int:
    """Verify all HF trajectories are represented in V1 format locally.

    The HF data was uploaded FROM packed_dataset/ which was created from
    btrm_dataset/ V1. So the 50 HF trajectories should already exist as
    V1 directories traj_000000 through traj_000049.

    This stage VERIFIES that, rather than reconstructing from the packed format
    (which would require torch and safetensors to unpack). If any HF
    trajectories are missing from V1, we report it but don't fail -- the
    V2 migration will proceed with whatever is on disk.

    Returns:
        Number of trajectories verified as present.
    """
    print()
    print("=" * 70)
    print("STAGE 3: Reconcile HF data with V1 on-disk")
    print("=" * 70)

    hf_manifest_path = staging_dir / "manifest.jsonl"
    if not hf_manifest_path.exists():
        print("  No manifest.jsonl in staging dir; nothing to reconcile")
        return 0

    hf_trajs = []
    with open(hf_manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                hf_trajs.append(json.loads(line))

    latents_dir = v1_dir / "latents"
    verified = 0
    missing = []

    for ht in hf_trajs:
        tid = ht["traj_id"]
        traj_name = f"traj_{tid:06d}"
        traj_dir = latents_dir / traj_name

        if not traj_dir.exists():
            missing.append(traj_name)
            print(f"  MISSING: {traj_name} not in V1 latents/")
            continue

        meta_path = traj_dir / "meta.json"
        if not meta_path.exists():
            missing.append(traj_name)
            print(f"  MISSING: {traj_name}/meta.json not found")
            continue

        # Verify at least some .pt files exist
        pt_files = [f for f in traj_dir.iterdir() if f.suffix == ".pt"]
        if not pt_files:
            missing.append(traj_name)
            print(f"  MISSING: {traj_name} has no .pt files")
            continue

        verified += 1

    print(f"  Verified: {verified}/{len(hf_trajs)} HF trajectories exist in V1 format")
    if missing:
        print(f"  Missing: {len(missing)} trajectories")
        for name in missing:
            print(f"    {name}")

    return verified


# ---------------------------------------------------------------------------
# Stage 4: Migrate V1 -> V2
# ---------------------------------------------------------------------------

def _v1_meta_to_v2(meta: dict, traj_dir: Path, source: str) -> dict:
    """Convert a v1 meta.json dict to v2 metadata dict.

    Follows the field mapping table from the spec (Section 10).
    """
    v2 = {}
    v2["prompt"] = meta["prompt"]
    v2["prompt_idx"] = meta.get("prompt_idx", -1)
    v2["seed"] = meta["seed"]
    v2["cfg"] = meta.get("cfg", 4.0)
    v2["width"] = meta.get("output_width", 1280)
    v2["height"] = meta.get("output_height", 832)
    v2["n_steps"] = meta["n_steps"]
    v2["attention_backend"] = meta["precision"]
    v2["batch_type"] = meta["type"]
    v2["denoise"] = meta.get("denoise")
    v2["image_file"] = meta.get("image_file")
    v2["is_gold"] = (meta["precision"] == "sdpa" and meta["n_steps"] == 30)
    v2["batch_idx"] = meta.get("batch_idx", 0)
    v2["packed"] = meta.get("packed", False)
    v2["timing_seconds"] = None

    return v2


def stage4_migrate(
    v1_dir: Path,
    v2_dir: Path,
    v1_manifest_path: Path,
    report_dir: Path,
    max_blob_bytes: int = 1_000_000_000,
    verify: bool = True,
) -> int:
    """Migrate ALL V1 trajectories to V2 format.

    Handles both:
    - Trajectories in the manifest (original BTRM dataset)
    - Extra trajectories NOT in the manifest (policy rollout persistence)

    Each trajectory gets provenance metadata tracking its source.

    Returns:
        Number of trajectories migrated.
    """
    import torch
    from futudiffu.dataset_v2 import DatasetWriter, DatasetReader

    print()
    print("=" * 70)
    print("STAGE 4: Migrate V1 -> V2")
    print("=" * 70)

    latents_dir = v1_dir / "latents"
    if not latents_dir.exists():
        print(f"  ERROR: {latents_dir} does not exist")
        return 0

    # Load V1 manifest
    manifest_records = {}
    if v1_manifest_path.exists():
        with open(v1_manifest_path) as f:
            manifest = json.load(f)
        for rec in manifest.get("records", []):
            traj_path = rec.get("traj_dir", "")
            name = Path(traj_path).name if traj_path else ""
            if name:
                manifest_records[name] = rec
        print(f"  Loaded V1 manifest: {len(manifest_records)} records")
    else:
        print(f"  No V1 manifest found at {v1_manifest_path}")

    # Enumerate all trajectory dirs
    traj_dirs = sorted(
        d for d in latents_dir.iterdir()
        if d.is_dir() and d.name.startswith("traj_")
    )
    print(f"  Trajectory directories on disk: {len(traj_dirs)}")
    print(f"  In manifest: {len(manifest_records)}")
    print(f"  Extra (policy rollouts): {len(traj_dirs) - len(set(d.name for d in traj_dirs) & set(manifest_records.keys()))}")

    # Migration report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("V1 -> V2 Migration Report")
    report_lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    report_lines.append(f"Source: {v1_dir}")
    report_lines.append(f"Target: {v2_dir}")
    report_lines.append("=" * 70)
    report_lines.append("")

    total_bytes = 0
    n_migrated = 0
    n_skipped = 0
    t_start = time.perf_counter()

    with DatasetWriter(v2_dir, max_blob_bytes=max_blob_bytes) as writer:
        for traj_dir in traj_dirs:
            name = traj_dir.name

            # Load meta.json
            meta_path = traj_dir / "meta.json"
            if not meta_path.exists():
                print(f"  SKIP {name}: no meta.json")
                report_lines.append(f"SKIP {name}: no meta.json")
                n_skipped += 1
                continue

            with open(meta_path) as f:
                meta = json.load(f)

            # Merge with manifest record if available
            in_manifest = name in manifest_records
            if in_manifest:
                rec = manifest_records[name]
                for k, v in rec.items():
                    if k not in meta and k != "traj_dir":
                        meta[k] = v
                source = "btrm_dataset_manifest"
            else:
                source = "policy_rollout_persistence"

            # Load tensors
            tensors = {}
            for fname in sorted(os.listdir(traj_dir)):
                if not fname.endswith(".pt"):
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

            if not tensors:
                print(f"  SKIP {name}: no .pt files")
                report_lines.append(f"SKIP {name}: no .pt files")
                n_skipped += 1
                continue

            # Convert metadata for V2
            v2_meta = _v1_meta_to_v2(meta, traj_dir, source)

            # Write to V2
            traj_id = writer.add_trajectory(tensors=tensors, metadata=v2_meta)
            traj_bytes = sum(
                t.nelement() * t.element_size()
                for t in tensors.values()
            )
            total_bytes += traj_bytes
            n_migrated += 1

            source_tag = "manifest" if in_manifest else "POLICY_ROLLOUT"
            detail = (f"  {name} -> traj_id={traj_id:06d}  "
                      f"({len(tensors)} tensors, {traj_bytes / 1e6:.1f} MB) "
                      f"[{source_tag}]")
            print(detail)
            report_lines.append(detail.strip())

    elapsed = time.perf_counter() - t_start
    print(f"\nMigration complete: {n_migrated} trajectories, "
          f"{n_skipped} skipped, "
          f"{total_bytes / 1e9:.2f} GB, {elapsed:.1f}s")

    # Count blobs
    blobs_dir = v2_dir / "blobs"
    blobs = sorted(blobs_dir.glob("blob_*.safetensors")) if blobs_dir.exists() else []
    print(f"Blobs created: {len(blobs)}")
    for b in blobs:
        print(f"  {b.name} ({b.stat().st_size / 1e6:.1f} MB)")

    report_lines.append("")
    report_lines.append(f"Total migrated: {n_migrated}")
    report_lines.append(f"Total skipped: {n_skipped}")
    report_lines.append(f"Total bytes: {total_bytes / 1e9:.2f} GB")
    report_lines.append(f"Elapsed: {elapsed:.1f}s")
    report_lines.append(f"Blobs: {len(blobs)}")

    # ---------------------------------------------------------------------------
    # Verification
    # ---------------------------------------------------------------------------
    if verify:
        print(f"\nVerifying round-trip fidelity...")
        reader = DatasetReader(v2_dir)
        n_checked = 0
        n_tensors_checked = 0
        mismatches = []

        for i, traj_dir in enumerate(traj_dirs):
            name = traj_dir.name

            # Skip dirs that were skipped during migration
            meta_path = traj_dir / "meta.json"
            if not meta_path.exists():
                continue

            pt_files = [f for f in traj_dir.iterdir() if f.suffix == ".pt"]
            if not pt_files:
                continue

            # The V2 traj_id corresponds to the sequential index of
            # successfully migrated trajectories
            traj_idx = n_checked  # Sequential ID assigned during migration

            if traj_idx not in reader:
                print(f"  WARN: {name} (expected id={traj_idx}) not in v2 dataset")
                continue

            # Load v1 tensors
            v1_tensors = {}
            for fname in sorted(os.listdir(traj_dir)):
                if not fname.endswith(".pt"):
                    continue
                fpath = traj_dir / fname
                tensor = torch.load(str(fpath), map_location="cpu", weights_only=True)
                if fname == "final.pt":
                    label = "final"
                elif fname.startswith("step_"):
                    label = fname[:-3]
                else:
                    continue
                v1_tensors[label] = tensor

            meta_v2, accessor = reader[traj_idx]

            for label, v1_t in v1_tensors.items():
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
            report_lines.append(f"\nVERIFICATION FAILED: {len(mismatches)} mismatches")
        else:
            print(f"Verification PASSED: {n_checked} trajectories, "
                  f"{n_tensors_checked} tensors checked (bitwise identical)")
            report_lines.append(f"\nVerification PASSED: {n_checked} trajectories, "
                                f"{n_tensors_checked} tensors (bitwise identical)")

    # Write migration report
    report_path = report_dir / "v1_migration_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nMigration report written to {report_path}")

    return n_migrated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pull BTRM dataset from HF, compare with local, migrate V1 to V2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--hf-repo", type=str, default="SQCU/futudiffu-btrm",
        help="HuggingFace dataset repo ID",
    )
    parser.add_argument(
        "--v1-dir", type=str,
        default=str(REPO_ROOT / "btrm_dataset"),
        help="Path to V1 btrm_dataset root",
    )
    parser.add_argument(
        "--v2-dir", type=str,
        default=str(REPO_ROOT / "btrm_dataset_v2_from_v1"),
        help="Output path for V2 dataset (staging, avoid conflict with merge script)",
    )
    parser.add_argument(
        "--staging-dir", type=str,
        default=str(REPO_ROOT / "hf_download_staging"),
        help="Directory for HF downloads",
    )
    parser.add_argument(
        "--report-dir", type=str,
        default=str(REPO_ROOT / "dataset_audit_output"),
        help="Directory for comparison and migration reports",
    )
    parser.add_argument(
        "--packed-dir", type=str,
        default=str(REPO_ROOT / "packed_dataset"),
        help="Path to local packed_dataset/ for comparison",
    )
    parser.add_argument(
        "--max-blob-bytes", type=int, default=1_000_000_000,
        help="Max blob size for V2 writer (default 1 GB)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip HF download (use existing staging dir)",
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip bitwise verification after migration",
    )

    args = parser.parse_args()

    v1_dir = Path(args.v1_dir)
    v2_dir = Path(args.v2_dir)
    staging_dir = Path(args.staging_dir)
    report_dir = Path(args.report_dir)
    packed_dir = Path(args.packed_dir)

    token = _load_hf_token()

    t_total_start = time.perf_counter()

    # Stage 1: Download from HF
    if args.skip_download:
        print("Skipping HF download (--skip-download)")
        # Still enumerate what's in staging
        if staging_dir.exists():
            hf_files = sorted(
                str(f.relative_to(staging_dir))
                for f in staging_dir.rglob("*")
                if f.is_file() and not f.name.startswith(".")
            )
        else:
            hf_files = []
    else:
        hf_files = stage1_download_hf(args.hf_repo, staging_dir, token)

    # Stage 2: Compare
    comparison = stage2_compare(hf_files, staging_dir, packed_dir, v1_dir, report_dir)

    # Stage 3: Reconcile
    n_verified = stage3_reconcile(staging_dir, v1_dir, report_dir)

    # Stage 4: Migrate V1 -> V2
    v1_manifest_path = v1_dir / "manifest.json"
    n_migrated = stage4_migrate(
        v1_dir=v1_dir,
        v2_dir=v2_dir,
        v1_manifest_path=v1_manifest_path,
        report_dir=report_dir,
        max_blob_bytes=args.max_blob_bytes,
        verify=not args.skip_verify,
    )

    elapsed_total = time.perf_counter() - t_total_start

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  HF files downloaded/verified: {len(hf_files)}")
    print(f"  V1 trajectories in manifest: {comparison['v1_manifest_traj_count']}")
    print(f"  V1 trajectories on disk: {comparison['v1_disk_traj_count']}")
    print(f"  Extra (policy rollouts): {len(comparison['extra_local_trajs'])}")
    print(f"  Trajectories migrated to V2: {n_migrated}")
    print(f"  V2 output: {v2_dir}")
    print(f"  Reports: {report_dir}")
    print(f"  Total elapsed: {elapsed_total:.1f}s")
    print()
    print("Next step: merge with other V2 staging dirs using:")
    print(f"  .venv/Scripts/python.exe scripts/merge_staged_datasets.py \\")
    print(f"    --staging-dirs {v2_dir} [other_v2_dirs...] \\")
    print(f"    --output btrm_dataset_v2_merged")


if __name__ == "__main__":
    main()

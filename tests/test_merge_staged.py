#!/usr/bin/env python3
"""Test merge_staged_datasets: per-GPU staging dirs -> unified dataset_v2.

Creates 3 temporary staging dirs, each with 5 small trajectories (random
tensors), merges them, and verifies correctness.

Run with:
    .venv/Scripts/python.exe tests/test_merge_staged.py
"""

import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

import pyarrow.parquet as pq
import torch
from safetensors import safe_open

from futudiffu.dataset_v2 import DatasetReader, DatasetWriter

# Import the merge function
sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\scripts")
from merge_staged_datasets import merge_staged_datasets


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_staging_dir(
    root: Path,
    gpu_id: int,
    n_traj: int = 5,
    n_steps: int = 3,
    latent_shape: tuple = (16, 8, 8),
) -> tuple[Path, dict[int, dict[str, torch.Tensor]]]:
    """Create a staging directory with n_traj small trajectories.

    Returns (staging_dir, ground_truth) where ground_truth maps
    local traj_id -> {step_label: tensor}.
    """
    staging_dir = root / f"staging_gpu{gpu_id}"
    ground_truth: dict[int, dict[str, torch.Tensor]] = {}

    with DatasetWriter(staging_dir) as writer:
        for i in range(n_traj):
            # Deterministic random tensors keyed by (gpu_id, i)
            rng = torch.Generator().manual_seed(gpu_id * 1000 + i)
            tensors = {}
            for s in range(n_steps):
                label = f"step_{s:02d}"
                t = torch.randn(*latent_shape, generator=rng, dtype=torch.bfloat16)
                tensors[label] = t
            # Also add a "final" tensor
            tensors["final"] = torch.randn(*latent_shape, generator=rng, dtype=torch.bfloat16)

            metadata = {
                "prompt": f"test prompt gpu{gpu_id} traj{i}",
                "prompt_idx": i,
                "seed": gpu_id * 1000 + i,
                "cfg": 4.0,
                "width": 128,
                "height": 128,
                "n_steps": 30,
                "attention_backend": "sdpa",
                "batch_type": "t2i",
                "is_gold": True,
                "batch_idx": 0,
            }

            traj_id = writer.add_trajectory(tensors=tensors, metadata=metadata)
            ground_truth[traj_id] = {k: v.clone() for k, v in tensors.items()}

    return staging_dir, ground_truth


def _verify_tensor_bitwise(a: torch.Tensor, b: torch.Tensor, label: str):
    """Assert two tensors are bitwise identical."""
    # Squeeze batch dim if present (reader returns (1, C, H, W))
    if a.dim() == 4 and a.shape[0] == 1:
        a = a.squeeze(0)
    if b.dim() == 4 and b.shape[0] == 1:
        b = b.squeeze(0)
    if not torch.equal(a, b):
        raise AssertionError(
            f"Tensor mismatch for {label}. "
            f"Shapes: {a.shape} vs {b.shape}, "
            f"Max diff: {(a.float() - b.float()).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_merge():
    """Test 1: Create 3 staging dirs x 5 trajectories, merge, verify."""
    print("=" * 60)
    print("TEST: basic_merge (3 staging dirs x 5 trajectories)")
    print("=" * 60)

    tmpdir = Path(tempfile.mkdtemp(prefix="test_merge_staged_"))
    try:
        n_gpus = 3
        n_traj_per_gpu = 5
        n_steps = 3
        latent_shape = (16, 8, 8)

        # Create staging dirs
        staging_dirs = []
        all_ground_truth: list[dict[int, dict[str, torch.Tensor]]] = []
        for gpu_id in range(n_gpus):
            sd, gt = _make_staging_dir(
                tmpdir, gpu_id, n_traj=n_traj_per_gpu,
                n_steps=n_steps, latent_shape=latent_shape,
            )
            staging_dirs.append(sd)
            all_ground_truth.append(gt)
            print(f"  Created staging_gpu{gpu_id}: {len(gt)} trajectories")

        # Merge
        output_dir = tmpdir / "merged"
        t0 = time.perf_counter()
        total = merge_staged_datasets(staging_dirs, output_dir)
        dt = time.perf_counter() - t0
        print(f"  Merge completed in {dt:.2f}s")

        # Verify total count
        expected_total = n_gpus * n_traj_per_gpu
        assert total == expected_total, f"Expected {expected_total}, got {total}"
        print(f"  PASS: total count = {total}")

        # Verify via DatasetReader
        reader = DatasetReader(output_dir)
        assert len(reader) == expected_total, \
            f"Reader sees {len(reader)}, expected {expected_total}"

        # Verify globally unique traj_ids
        all_traj_ids = set()
        for traj_id, _ in reader.iter_metadata():
            assert traj_id not in all_traj_ids, f"Duplicate traj_id: {traj_id}"
            all_traj_ids.add(traj_id)
        assert len(all_traj_ids) == expected_total
        assert all_traj_ids == set(range(expected_total)), \
            f"Expected contiguous IDs 0..{expected_total-1}, got {sorted(all_traj_ids)}"
        print(f"  PASS: globally unique traj_ids = {{0..{expected_total-1}}}")

        # Verify tensor data preserved bitwise
        global_id = 0
        for gpu_id in range(n_gpus):
            gt = all_ground_truth[gpu_id]
            for local_id in sorted(gt.keys()):
                meta, accessor = reader[global_id]
                for step_label, expected_tensor in gt[local_id].items():
                    actual_tensor = accessor[step_label]
                    _verify_tensor_bitwise(actual_tensor, expected_tensor,
                                           f"gpu{gpu_id}/local{local_id}/{step_label}")
                # Verify metadata propagated
                assert meta["seed"] == gpu_id * 1000 + local_id, \
                    f"Seed mismatch for global_id={global_id}"
                assert meta["prompt"] == f"test prompt gpu{gpu_id} traj{local_id}", \
                    f"Prompt mismatch for global_id={global_id}"
                global_id += 1

        print(f"  PASS: all tensors bitwise identical")
        print(f"  PASS: all metadata preserved")

        reader.close()
        return output_dir, tmpdir, staging_dirs

    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise


def test_idempotency(output_dir: Path, staging_dirs: list[Path]):
    """Test 2: Run merge twice, verify same result."""
    print()
    print("=" * 60)
    print("TEST: idempotency (merge twice, same result)")
    print("=" * 60)

    # Read the first merge result
    table_before = pq.read_table(str(output_dir / "index.parquet"))
    n_before = len(table_before)

    # Run merge again on the same output
    total = merge_staged_datasets(staging_dirs, output_dir)

    assert total == n_before, f"Idempotent merge returned {total}, expected {n_before}"

    # Verify index is unchanged (the merge should have been skipped)
    table_after = pq.read_table(str(output_dir / "index.parquet"))
    assert table_before.equals(table_after), "Index changed after idempotent merge"

    print(f"  PASS: merge skipped (idempotent), {total} trajectories unchanged")


def test_empty_staging_dir():
    """Test 3: Merge with one empty staging dir."""
    print()
    print("=" * 60)
    print("TEST: empty staging dir (2 normal + 1 empty)")
    print("=" * 60)

    tmpdir = Path(tempfile.mkdtemp(prefix="test_merge_empty_"))
    try:
        # Create 2 normal staging dirs + 1 empty
        staging_dirs = []
        total_expected = 0

        for gpu_id in range(2):
            sd, gt = _make_staging_dir(tmpdir, gpu_id, n_traj=3)
            staging_dirs.append(sd)
            total_expected += len(gt)

        # Empty staging dir
        empty_dir = tmpdir / "staging_gpu2"
        with DatasetWriter(empty_dir) as writer:
            pass  # No trajectories
        staging_dirs.append(empty_dir)

        # Merge
        output_dir = tmpdir / "merged_with_empty"
        total = merge_staged_datasets(staging_dirs, output_dir)

        assert total == total_expected, f"Expected {total_expected}, got {total}"

        reader = DatasetReader(output_dir)
        assert len(reader) == total_expected
        reader.close()

        print(f"  PASS: merged {total_expected} trajectories (empty dir contributed 0)")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_single_staging_dir():
    """Test 4: Merge with a single staging dir (identity merge)."""
    print()
    print("=" * 60)
    print("TEST: single staging dir (identity merge)")
    print("=" * 60)

    tmpdir = Path(tempfile.mkdtemp(prefix="test_merge_single_"))
    try:
        sd, gt = _make_staging_dir(tmpdir, gpu_id=0, n_traj=4)

        output_dir = tmpdir / "merged_single"
        total = merge_staged_datasets([sd], output_dir)

        assert total == 4, f"Expected 4, got {total}"

        reader = DatasetReader(output_dir)
        assert len(reader) == 4

        # Verify data
        for local_id in sorted(gt.keys()):
            meta, accessor = reader[local_id]
            for step_label, expected_tensor in gt[local_id].items():
                actual_tensor = accessor[step_label]
                _verify_tensor_bitwise(actual_tensor, expected_tensor,
                                       f"single/local{local_id}/{step_label}")

        reader.close()
        print(f"  PASS: identity merge of 4 trajectories, data preserved")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.perf_counter()
    print(f"test_merge_staged.py -- {datetime.now(timezone.utc).isoformat()}")
    print()

    # Test 1 + 2 share state (basic merge result used for idempotency test)
    output_dir, tmpdir, staging_dirs = test_basic_merge()
    try:
        test_idempotency(output_dir, staging_dirs)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Test 3: empty staging dir
    test_empty_staging_dir()

    # Test 4: single staging dir
    test_single_staging_dir()

    dt = time.perf_counter() - t_start
    print()
    print("=" * 60)
    print(f"ALL TESTS PASSED ({dt:.2f}s)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

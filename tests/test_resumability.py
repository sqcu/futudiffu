"""Resumability test for dataset_v2: proves DatasetWriter can survive interruptions.

Verifies:
  1. Basic resume: close writer, reopen on same dir, continue adding
  2. Resume after partial seal: frequent blob rotation + multi-session append
  3. Simulated crash: unsealed WIP data is lost, sealed data survives
  4. Trajectory ID continuity: monotonically increasing across sessions

All tests use small random CPU tensors in temp directories. No GPU required.
"""

from __future__ import annotations

import gc
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

import torch

from futudiffu.dataset_v2 import DatasetReader, DatasetWriter, _release_lock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LATENT_SHAPE = (4, 8, 8)  # Small for speed: (C, H, W)


def _make_tensors(n_steps: int = 3) -> dict[str, torch.Tensor]:
    """Create small random bf16 tensors for a fake trajectory."""
    tensors = {}
    for i in range(n_steps):
        tensors[f"step_{i:02d}"] = torch.randn(1, *LATENT_SHAPE, dtype=torch.bfloat16)
    tensors["final"] = torch.randn(1, *LATENT_SHAPE, dtype=torch.bfloat16)
    return tensors


def _make_metadata(seed: int = 42) -> dict:
    """Create a minimal valid metadata dict."""
    return {
        "prompt": f"test prompt seed={seed}",
        "prompt_idx": 0,
        "seed": seed,
        "cfg": 4.0,
        "width": 256,
        "height": 256,
        "n_steps": 30,
        "attention_backend": "sdpa",
        "batch_type": "t2i",
    }


def _add_n_trajectories(
    writer: DatasetWriter,
    n: int,
    seed_offset: int = 0,
) -> list[tuple[int, dict[str, torch.Tensor]]]:
    """Add n trajectories to writer. Returns list of (traj_id, tensors)."""
    results = []
    for i in range(n):
        tensors = _make_tensors()
        meta = _make_metadata(seed=seed_offset + i)
        traj_id = writer.add_trajectory(tensors=tensors, metadata=meta)
        results.append((traj_id, tensors))
    return results


# ---------------------------------------------------------------------------
# Test 1: Basic resume after seal
# ---------------------------------------------------------------------------

def test_basic_resume():
    """Open writer, add 10, close. Reopen, verify next_traj_id==10, add 5 more.
    Open reader, verify 15 total, all tensors bitwise correct."""
    tmpdir = tempfile.mkdtemp(prefix="resumability_test1_")
    try:
        ds_path = Path(tmpdir) / "dataset_v2"
        all_written: list[tuple[int, dict[str, torch.Tensor]]] = []

        # Session 1: add 10 trajectories
        with DatasetWriter(ds_path) as writer:
            batch1 = _add_n_trajectories(writer, 10, seed_offset=0)
            all_written.extend(batch1)
            assert writer.n_trajectories == 10, (
                f"Session 1: expected 10 trajectories, got {writer.n_trajectories}"
            )

        # Session 2: reopen, verify state, add 5 more
        with DatasetWriter(ds_path) as writer:
            assert writer.next_traj_id == 10, (
                f"Resume: expected next_traj_id=10, got {writer.next_traj_id}"
            )
            assert writer.n_trajectories == 10, (
                f"Resume: expected 10 existing rows, got {writer.n_trajectories}"
            )
            batch2 = _add_n_trajectories(writer, 5, seed_offset=100)
            all_written.extend(batch2)

        # Verify with reader
        reader = DatasetReader(ds_path)
        assert len(reader) == 15, f"Expected 15 trajectories, got {len(reader)}"

        for traj_id, orig_tensors in all_written:
            assert traj_id in reader, f"traj_id {traj_id} not in reader"
            _, accessor = reader[traj_id]
            loaded = accessor.load_all()
            assert set(loaded.keys()) == set(orig_tensors.keys()), (
                f"Key mismatch for traj {traj_id}: "
                f"{set(loaded.keys())} vs {set(orig_tensors.keys())}"
            )
            for label, orig_t in orig_tensors.items():
                assert torch.equal(orig_t, loaded[label]), (
                    f"Tensor mismatch for traj {traj_id} / {label}"
                )

        reader.close()
        print("  test_basic_resume PASSED")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 2: Resume after partial seal (frequent blob rotation)
# ---------------------------------------------------------------------------

def test_resume_partial_seal():
    """Small max_blob_bytes forces frequent seals. Multi-session append."""
    tmpdir = tempfile.mkdtemp(prefix="resumability_test2_")
    try:
        ds_path = Path(tmpdir) / "dataset_v2"
        all_written: list[tuple[int, dict[str, torch.Tensor]]] = []

        # Each trajectory: 4 tensors * (4*8*8) * 2 bytes = 4 * 512 = 2048 bytes
        # With max_blob_bytes=1024, each trajectory should trigger a seal
        # (since even one traj's tensors exceed 1024 bytes).

        # Session 1: add 20 trajectories with tiny blob limit
        with DatasetWriter(ds_path, max_blob_bytes=1024) as writer:
            batch1 = _add_n_trajectories(writer, 20, seed_offset=0)
            all_written.extend(batch1)

        # Count sealed blobs
        blobs_dir = ds_path / "blobs"
        blob_files = sorted(blobs_dir.glob("blob_*.safetensors"))
        n_blobs_session1 = len(blob_files)
        assert n_blobs_session1 > 1, (
            f"Expected >1 blobs with 1024-byte limit, got {n_blobs_session1}"
        )
        print(f"    Session 1: {n_blobs_session1} blobs for 20 trajectories")

        # Session 2: reopen with same blob limit, add 10 more
        with DatasetWriter(ds_path, max_blob_bytes=1024) as writer:
            assert writer.next_traj_id == 20, (
                f"Resume: expected next_traj_id=20, got {writer.next_traj_id}"
            )
            batch2 = _add_n_trajectories(writer, 10, seed_offset=200)
            all_written.extend(batch2)

        # Count total blobs now
        blob_files = sorted(blobs_dir.glob("blob_*.safetensors"))
        n_blobs_total = len(blob_files)
        assert n_blobs_total > n_blobs_session1, (
            f"Expected more blobs after session 2: "
            f"was {n_blobs_session1}, now {n_blobs_total}"
        )
        print(f"    Session 2: {n_blobs_total} total blobs for 30 trajectories")

        # Verify sequential naming
        for i, bf in enumerate(blob_files):
            expected = f"blob_{i:03d}.safetensors"
            assert bf.name == expected, (
                f"Blob {i} named '{bf.name}', expected '{expected}'"
            )

        # Verify all 30 trajectories readable
        reader = DatasetReader(ds_path)
        assert len(reader) == 30, f"Expected 30 trajectories, got {len(reader)}"

        for traj_id, orig_tensors in all_written:
            _, accessor = reader[traj_id]
            loaded = accessor.load_all()
            for label, orig_t in orig_tensors.items():
                assert torch.equal(orig_t, loaded[label]), (
                    f"Tensor mismatch for traj {traj_id} / {label}"
                )

        reader.close()
        print("  test_resume_partial_seal PASSED")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 3: Simulated crash (context manager __exit__ not called)
# ---------------------------------------------------------------------------

def test_simulated_crash():
    """Sealed data survives crash; unsealed WIP data is lost."""
    tmpdir = tempfile.mkdtemp(prefix="resumability_test3_")
    try:
        ds_path = Path(tmpdir) / "dataset_v2"

        # --- Phase 1: write 5, seal manually, write 3 more (unsealed), crash ---
        writer = DatasetWriter(ds_path)
        writer.__enter__()

        sealed_written = _add_n_trajectories(writer, 5, seed_offset=0)
        blob_name = writer.seal_current_blob()
        assert blob_name is not None, "seal_current_blob should return a filename"
        print(f"    Sealed blob: {blob_name}")

        # These 3 are in WIP memory only -- will be lost on crash
        wip_written = _add_n_trajectories(writer, 3, seed_offset=500)

        # Simulate crash: release the lock but do NOT call __exit__
        # This means WIP tensors are never sealed, and the index is not updated
        # for the WIP rows.
        if writer._lock_fd is not None:
            _release_lock(writer._lock_fd)
            writer._lock_fd = None  # Prevent double-release

        # Force garbage collection to release any file handles
        del writer
        gc.collect()

        # --- Phase 2: recover ---
        # seal_current_blob() writes the parquet index after sealing.
        # The 3 WIP trajectories were added to _rows in memory AFTER that
        # seal, but since __exit__ was never called, _write_index() was
        # never invoked again. The on-disk index only has the 5 sealed
        # trajectories. The 3 WIP ones are cleanly lost -- this is the
        # intended crash-safe behavior (no WIP blob on disk = no corruption).

        # Check what's on disk
        blobs_dir = ds_path / "blobs"
        blob_files = sorted(blobs_dir.glob("blob_*.safetensors"))
        print(f"    Blobs on disk after crash: {[f.name for f in blob_files]}")

        # Read the index to see what rows exist
        import pyarrow.parquet as pq
        raw_table = pq.read_table(str(ds_path / "index.parquet"))
        raw_rows = raw_table.to_pylist()
        sealed_rows = [r for r in raw_rows if r["blob_file"] != "blob_wip.safetensors"]
        wip_rows = [r for r in raw_rows if r["blob_file"] == "blob_wip.safetensors"]
        print(f"    Index rows: {len(raw_rows)} total, "
              f"{len(sealed_rows)} sealed, {len(wip_rows)} WIP (orphaned)")

        # Only the sealed data should be in the index
        assert len(sealed_rows) == 5, (
            f"Expected 5 sealed rows, got {len(sealed_rows)}"
        )
        assert len(wip_rows) == 0, (
            f"Expected 0 WIP rows on disk (never flushed), got {len(wip_rows)}"
        )

        # Reopen writer -- it loads 5 rows, next_traj_id = 5
        with DatasetWriter(ds_path) as writer:
            loaded_count = writer.n_trajectories
            loaded_next_id = writer.next_traj_id
            print(f"    After recovery: n_trajectories={loaded_count}, "
                  f"next_traj_id={loaded_next_id}")

            # Only the 5 sealed trajectories survived
            assert loaded_count == 5, (
                f"Expected 5 surviving trajectories, got {loaded_count}"
            )
            assert loaded_next_id == 5, (
                f"Expected next_traj_id=5, got {loaded_next_id}"
            )

            # Add 5 more trajectories (IDs 5..9)
            recovery_written = _add_n_trajectories(writer, 5, seed_offset=1000)

        # --- Phase 3: verify final state ---
        reader = DatasetReader(ds_path)
        total = len(reader)
        print(f"    Final dataset: {total} trajectories")

        # 5 sealed + 5 new = 10 total (3 WIP lost as expected)
        assert total == 10, (
            f"Expected 10 total rows (5 sealed + 5 recovery), got {total}"
        )

        # Verify the originally sealed 5 are bitwise correct
        for traj_id, orig_tensors in sealed_written:
            assert traj_id in reader, f"Sealed traj {traj_id} missing"
            _, accessor = reader[traj_id]
            loaded = accessor.load_all()
            for label, orig_t in orig_tensors.items():
                assert torch.equal(orig_t, loaded[label]), (
                    f"Tensor mismatch for sealed traj {traj_id} / {label}"
                )

        # Verify the 5 recovery trajectories are correct
        for traj_id, orig_tensors in recovery_written:
            assert traj_id in reader, f"Recovery traj {traj_id} missing"
            _, accessor = reader[traj_id]
            loaded = accessor.load_all()
            for label, orig_t in orig_tensors.items():
                assert torch.equal(orig_t, loaded[label]), (
                    f"Tensor mismatch for recovery traj {traj_id} / {label}"
                )

        # The 3 WIP trajectory IDs (5, 6, 7 from original) should NOT be in
        # the dataset -- they were lost in the crash. The recovery trajectories
        # reused IDs starting from 5.
        wip_orig_ids = [tid for tid, _ in wip_written]
        recovery_ids = [tid for tid, _ in recovery_written]
        print(f"    Lost WIP traj_ids: {wip_orig_ids}")
        print(f"    Recovery traj_ids: {recovery_ids}")

        reader.close()
        print("  test_simulated_crash PASSED")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 4: Trajectory ID continuity across sessions
# ---------------------------------------------------------------------------

def test_traj_id_continuity():
    """IDs are monotonically increasing across 3 separate writer sessions."""
    tmpdir = tempfile.mkdtemp(prefix="resumability_test4_")
    try:
        ds_path = Path(tmpdir) / "dataset_v2"
        all_ids: list[int] = []

        # Session 1: traj_ids 0..4
        with DatasetWriter(ds_path) as writer:
            batch1 = _add_n_trajectories(writer, 5, seed_offset=0)
            ids1 = [tid for tid, _ in batch1]
            all_ids.extend(ids1)
            assert ids1 == list(range(0, 5)), (
                f"Session 1 ids: expected [0..4], got {ids1}"
            )

        # Session 2: traj_ids 5..9
        with DatasetWriter(ds_path) as writer:
            assert writer.next_traj_id == 5, (
                f"Session 2: expected next_traj_id=5, got {writer.next_traj_id}"
            )
            batch2 = _add_n_trajectories(writer, 5, seed_offset=100)
            ids2 = [tid for tid, _ in batch2]
            all_ids.extend(ids2)
            assert ids2 == list(range(5, 10)), (
                f"Session 2 ids: expected [5..9], got {ids2}"
            )

        # Session 3: traj_ids 10..14
        with DatasetWriter(ds_path) as writer:
            assert writer.next_traj_id == 10, (
                f"Session 3: expected next_traj_id=10, got {writer.next_traj_id}"
            )
            batch3 = _add_n_trajectories(writer, 5, seed_offset=200)
            ids3 = [tid for tid, _ in batch3]
            all_ids.extend(ids3)
            assert ids3 == list(range(10, 15)), (
                f"Session 3 ids: expected [10..14], got {ids3}"
            )

        # Verify full sequence
        assert all_ids == list(range(15)), (
            f"Full ID sequence should be [0..14], got {all_ids}"
        )

        # Verify with reader
        reader = DatasetReader(ds_path)
        assert len(reader) == 15, f"Expected 15 trajectories, got {len(reader)}"

        # Check all IDs are present and monotonic
        for i in range(15):
            assert i in reader, f"traj_id {i} missing from reader"

        reader.close()
        print("  test_traj_id_continuity PASSED")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Dataset V2 Resumability Tests")
    print("=" * 60)
    print()

    tests = [
        ("1. Basic resume after seal", test_basic_resume),
        ("2. Resume after partial seal", test_resume_partial_seal),
        ("3. Simulated crash", test_simulated_crash),
        ("4. Trajectory ID continuity", test_traj_id_continuity),
    ]

    n_passed = 0
    n_failed = 0
    failures = []

    for name, test_fn in tests:
        print(f"--- {name} ---")
        t0 = time.perf_counter()
        try:
            test_fn()
            elapsed = time.perf_counter() - t0
            print(f"    ({elapsed:.3f}s)")
            n_passed += 1
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            n_failed += 1
            failures.append((name, exc))
            import traceback
            print(f"  FAILED ({elapsed:.3f}s): {exc}")
            traceback.print_exc()
        print()

    print("=" * 60)
    print(f"  Results: {n_passed} passed, {n_failed} failed")
    if failures:
        for name, exc in failures:
            print(f"    FAIL: {name}: {exc}")
    print("=" * 60)

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

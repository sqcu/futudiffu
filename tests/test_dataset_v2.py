"""Integration test for dataset_v2: DatasetWriter + DatasetReader round-trip.

Verifies the v2 dataset format (parquet index + safetensors blob storage)
without requiring GPU or the inference server. All tensors are random bf16
on CPU in temp directories.

Tests:
  1. Write + read round-trip (5 trajectories, bitwise tensor fidelity)
  2. Blob rotation under small max_blob_bytes
  3. Filter and sample operations
  4. Flush mid-generation (seal + continue)
  5. Empty dataset handling
"""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from futudiffu.dataset_v2 import DatasetReader, DatasetWriter, INDEX_SCHEMA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LATENT_SHAPE = (16, 32, 48)  # (C, H, W) -- small for speed


def _make_tensors(step_indices: list[int], include_final: bool = True) -> dict[str, torch.Tensor]:
    """Create random bf16 tensors keyed by step label."""
    tensors = {}
    for idx in step_indices:
        tensors[f"step_{idx:02d}"] = torch.randn(1, *LATENT_SHAPE, dtype=torch.bfloat16)
    if include_final:
        tensors["final"] = torch.randn(1, *LATENT_SHAPE, dtype=torch.bfloat16)
    return tensors


def _make_metadata(
    prompt: str = "a laser shark",
    seed: int = 42,
    batch_type: str = "t2i",
    n_steps: int = 30,
    attention_backend: str = "sdpa",
    denoise: float | None = None,
    image_file: str | None = None,
) -> dict:
    """Create a valid metadata dict for add_trajectory."""
    meta = {
        "prompt": prompt,
        "prompt_idx": 0,
        "seed": seed,
        "cfg": 4.0,
        "width": 1280,
        "height": 832,
        "n_steps": n_steps,
        "attention_backend": attention_backend,
        "batch_type": batch_type,
    }
    if denoise is not None:
        meta["denoise"] = denoise
    if image_file is not None:
        meta["image_file"] = image_file
    return meta


# ---------------------------------------------------------------------------
# Test definitions for each scenario. Each is a standalone function that
# receives a fresh temp directory via _run_in_tempdir.
# ---------------------------------------------------------------------------

# Reference data: 5 trajectories with varied metadata, stored so we can
# compare after round-trip.

_TRAJ_SPECS = [
    dict(
        prompt="a laser shark swimming through space",
        seed=1000,
        batch_type="t2i",
        n_steps=30,
        attention_backend="sdpa",
        step_indices=[0, 4, 9, 14, 19, 24, 29],
        include_final=True,
    ),
    dict(
        prompt="a cat wearing a top hat",
        seed=2000,
        batch_type="t2i",
        n_steps=30,
        attention_backend="sage",
        step_indices=[0, 9, 19, 29],
        include_final=True,
    ),
    dict(
        prompt="mountain landscape at sunset",
        seed=3000,
        batch_type="i2i",
        n_steps=20,
        attention_backend="sdpa",
        step_indices=[0, 4, 9, 14, 19],
        include_final=True,
        denoise=0.85,
        image_file="input_landscape.png",
    ),
    dict(
        prompt="cyberpunk cityscape neon rain",
        seed=4000,
        batch_type="t2i",
        n_steps=30,
        attention_backend="sage",
        step_indices=[0, 14, 29],
        include_final=False,
    ),
    dict(
        prompt="portrait of a robot philosopher",
        seed=5000,
        batch_type="i2i",
        n_steps=15,
        attention_backend="sdpa",
        step_indices=[0, 7, 14],
        include_final=True,
        denoise=0.90,
        image_file="input_portrait.png",
    ),
]


def _write_trajectories(writer: DatasetWriter, specs: list[dict]) -> list[tuple[int, dict[str, torch.Tensor], dict]]:
    """Write trajectories from specs, return (traj_id, tensors, metadata) triples."""
    written = []
    for spec in specs:
        tensors = _make_tensors(spec["step_indices"], spec.get("include_final", True))
        metadata = _make_metadata(
            prompt=spec["prompt"],
            seed=spec["seed"],
            batch_type=spec["batch_type"],
            n_steps=spec["n_steps"],
            attention_backend=spec["attention_backend"],
            denoise=spec.get("denoise"),
            image_file=spec.get("image_file"),
        )
        traj_id = writer.add_trajectory(tensors=tensors, metadata=metadata)
        written.append((traj_id, tensors, metadata))
    return written


# ---------------------------------------------------------------------------
# Test 1: Write + read round-trip
# ---------------------------------------------------------------------------

def test_write_read_roundtrip():
    """Write 5 trajectories, close writer, open reader, verify everything."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ds_path = Path(tmpdir) / "dataset_v2"

        # Write
        with DatasetWriter(ds_path) as writer:
            written = _write_trajectories(writer, _TRAJ_SPECS)

        # Read
        reader = DatasetReader(ds_path)

        # -- Length check
        assert len(reader) == 5, f"Expected 5 trajectories, got {len(reader)}"

        # -- Per-trajectory verification
        for traj_id, orig_tensors, orig_meta in written:
            assert traj_id in reader, f"traj_id {traj_id} not in reader"
            row, accessor = reader[traj_id]

            # Metadata checks
            assert row["prompt"] == orig_meta["prompt"], \
                f"Prompt mismatch for traj {traj_id}"
            assert row["seed"] == orig_meta["seed"], \
                f"Seed mismatch for traj {traj_id}"
            assert row["batch_type"] == orig_meta["batch_type"], \
                f"batch_type mismatch for traj {traj_id}"
            assert row["n_steps"] == orig_meta["n_steps"], \
                f"n_steps mismatch for traj {traj_id}"
            assert row["attention_backend"] == orig_meta["attention_backend"], \
                f"attention_backend mismatch for traj {traj_id}"
            assert row["latent_channels"] == LATENT_SHAPE[0]
            assert row["latent_height"] == LATENT_SHAPE[1]
            assert row["latent_width"] == LATENT_SHAPE[2]
            assert row["latent_dtype"] == "bfloat16"

            if orig_meta.get("denoise") is not None:
                assert abs(row["denoise"] - orig_meta["denoise"]) < 1e-5, \
                    f"denoise mismatch for traj {traj_id}"
            if orig_meta.get("image_file") is not None:
                assert row["image_file"] == orig_meta["image_file"], \
                    f"image_file mismatch for traj {traj_id}"

            # Tensor bitwise checks
            loaded = accessor.load_all()
            assert set(loaded.keys()) == set(orig_tensors.keys()), \
                f"Tensor key mismatch for traj {traj_id}: {set(loaded.keys())} vs {set(orig_tensors.keys())}"

            for label, orig_t in orig_tensors.items():
                read_t = loaded[label]
                # Accessor returns (1, C, H, W); orig_t is already (1, C, H, W)
                assert torch.equal(orig_t, read_t), \
                    f"Tensor mismatch for traj {traj_id} / {label}"

        # -- Parquet schema check
        table = pq.read_table(str(ds_path / "index.parquet"))
        for field in INDEX_SCHEMA:
            assert field.name in table.column_names, \
                f"Missing column '{field.name}' in parquet index"

        reader.close()
        print("  test_write_read_roundtrip PASSED")


# ---------------------------------------------------------------------------
# Test 2: Blob rotation
# ---------------------------------------------------------------------------

def test_blob_rotation():
    """Small max_blob_bytes forces multiple blob files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ds_path = Path(tmpdir) / "dataset_v2"

        # Each trajectory's tensors: ~8 step tensors * 16*32*48 * 2 bytes
        # = 8 * 49152 = ~393 KB per trajectory. With max_blob_bytes=100KB,
        # each trajectory should go into its own blob (or at most a couple
        # share one).
        n_trajs = 8
        specs = []
        for i in range(n_trajs):
            specs.append(dict(
                prompt=f"rotation test {i}",
                seed=9000 + i,
                batch_type="t2i",
                n_steps=30,
                attention_backend="sdpa",
                step_indices=[0, 4, 9, 14, 19, 24, 29],
                include_final=True,
            ))

        with DatasetWriter(ds_path, max_blob_bytes=100_000) as writer:
            written = _write_trajectories(writer, specs)

        # Check blob files
        blobs_dir = ds_path / "blobs"
        blob_files = sorted(blobs_dir.glob("blob_*.safetensors"))
        n_blobs = len(blob_files)

        assert n_blobs >= 2, \
            f"Expected at least 2 blobs with 100KB limit, got {n_blobs}"

        # Verify sequential naming
        for i, bf in enumerate(blob_files):
            expected_name = f"blob_{i:03d}.safetensors"
            assert bf.name == expected_name, \
                f"Blob {i} named '{bf.name}', expected '{expected_name}'"

        # Verify all trajectories still readable
        reader = DatasetReader(ds_path)
        assert len(reader) == n_trajs, \
            f"Expected {n_trajs} trajectories, got {len(reader)}"

        for traj_id, orig_tensors, _ in written:
            _, accessor = reader[traj_id]
            loaded = accessor.load_all()
            for label, orig_t in orig_tensors.items():
                assert torch.equal(orig_t, loaded[label]), \
                    f"Tensor mismatch after rotation: traj {traj_id} / {label}"

        reader.close()
        print(f"  test_blob_rotation PASSED ({n_blobs} blobs created)")


# ---------------------------------------------------------------------------
# Test 3: Filter and sample
# ---------------------------------------------------------------------------

def test_filter_and_sample():
    """Verify filter by batch_type and sample operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ds_path = Path(tmpdir) / "dataset_v2"

        with DatasetWriter(ds_path) as writer:
            _write_trajectories(writer, _TRAJ_SPECS)

        reader = DatasetReader(ds_path)

        # Count expected t2i and i2i from specs
        expected_t2i = sum(1 for s in _TRAJ_SPECS if s["batch_type"] == "t2i")
        expected_i2i = sum(1 for s in _TRAJ_SPECS if s["batch_type"] == "i2i")

        # Filter by batch_type
        t2i_view = reader.filter(batch_type="t2i")
        assert len(t2i_view) == expected_t2i, \
            f"t2i filter: expected {expected_t2i}, got {len(t2i_view)}"

        i2i_view = reader.filter(batch_type="i2i")
        assert len(i2i_view) == expected_i2i, \
            f"i2i filter: expected {expected_i2i}, got {len(i2i_view)}"

        # Verify disjointness
        t2i_ids = set(t2i_view.traj_ids)
        i2i_ids = set(i2i_view.traj_ids)
        assert t2i_ids.isdisjoint(i2i_ids), "t2i and i2i sets overlap"
        assert len(t2i_ids | i2i_ids) == 5, "t2i + i2i should cover all 5 trajectories"

        # Sample
        rng = random.Random(42)
        sampled = reader.sample(n=2, rng=rng)
        assert len(sampled) == 2, f"sample(2) returned {len(sampled)} items"
        assert all(isinstance(tid, int) for tid in sampled)

        # Sample from filtered view
        t2i_sampled = t2i_view.sample(n=2, rng=random.Random(42))
        assert len(t2i_sampled) == min(2, expected_t2i)
        assert all(tid in t2i_ids for tid in t2i_sampled)

        # scrimble_split and scrongle_split should not crash
        sdpa_ids, sage_ids = reader.scrimble_split()
        assert isinstance(sdpa_ids, list)
        assert isinstance(sage_ids, list)

        full_ids, reduced_ids = reader.scrongle_split()
        assert isinstance(full_ids, list)
        assert isinstance(reduced_ids, list)

        reader.close()
        print(f"  test_filter_and_sample PASSED "
              f"(t2i={expected_t2i}, i2i={expected_i2i})")


# ---------------------------------------------------------------------------
# Test 4: Flush mid-generation
# ---------------------------------------------------------------------------

def test_flush_mid_generation():
    """flush() seals first batch; remaining trajectories go to a second blob."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ds_path = Path(tmpdir) / "dataset_v2"

        with DatasetWriter(ds_path) as writer:
            # Add first 3
            first_batch = _write_trajectories(writer, _TRAJ_SPECS[:3])

            # Flush -- this should seal a blob and write the parquet index
            writer.flush()

            # Verify the first blob exists on disk
            blobs_dir = ds_path / "blobs"
            blob_files_after_flush = sorted(blobs_dir.glob("blob_*.safetensors"))
            assert len(blob_files_after_flush) == 1, \
                f"Expected 1 sealed blob after flush, got {len(blob_files_after_flush)}"
            assert blob_files_after_flush[0].name == "blob_000.safetensors"

            # Verify the parquet index exists and has 3 rows
            table_mid = pq.read_table(str(ds_path / "index.parquet"))
            assert len(table_mid) == 3, \
                f"Expected 3 rows in index after flush, got {len(table_mid)}"

            # The first 3 trajectories should reference blob_000
            blob_col = table_mid.column("blob_file").to_pylist()
            assert all(b == "blob_000.safetensors" for b in blob_col), \
                f"First 3 rows should reference blob_000, got {blob_col}"

            # Add 2 more
            second_batch = _write_trajectories(writer, _TRAJ_SPECS[3:])

        # After close, the second batch should be in blob_001
        reader = DatasetReader(ds_path)
        assert len(reader) == 5, f"Expected 5 total, got {len(reader)}"

        blob_files_final = sorted(blobs_dir.glob("blob_*.safetensors"))
        assert len(blob_files_final) == 2, \
            f"Expected 2 blobs (flush + close), got {len(blob_files_final)}"
        assert blob_files_final[0].name == "blob_000.safetensors"
        assert blob_files_final[1].name == "blob_001.safetensors"

        # Verify all tensors from both batches
        for traj_id, orig_tensors, _ in first_batch + second_batch:
            _, accessor = reader[traj_id]
            loaded = accessor.load_all()
            for label, orig_t in orig_tensors.items():
                assert torch.equal(orig_t, loaded[label]), \
                    f"Tensor mismatch: traj {traj_id} / {label}"

        # Verify blob_file column is correct for each trajectory
        table_final = pq.read_table(str(ds_path / "index.parquet"))
        rows = table_final.to_pylist()
        first_ids = {tid for tid, _, _ in first_batch}
        second_ids = {tid for tid, _, _ in second_batch}
        for row in rows:
            if row["traj_id"] in first_ids:
                assert row["blob_file"] == "blob_000.safetensors", \
                    f"traj {row['traj_id']} should be in blob_000"
            elif row["traj_id"] in second_ids:
                assert row["blob_file"] == "blob_001.safetensors", \
                    f"traj {row['traj_id']} should be in blob_001"

        reader.close()
        print("  test_flush_mid_generation PASSED")


# ---------------------------------------------------------------------------
# Test 5: Empty dataset
# ---------------------------------------------------------------------------

def test_empty_dataset():
    """Create and immediately close a writer; reader handles it gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ds_path = Path(tmpdir) / "dataset_v2"

        # Write nothing
        with DatasetWriter(ds_path) as writer:
            assert writer.n_trajectories == 0

        # Reader should handle missing or empty index
        reader = DatasetReader(ds_path)
        assert len(reader) == 0, f"Expected 0 trajectories, got {len(reader)}"

        # sample on empty dataset should return empty list
        sampled = reader.sample(n=5)
        assert sampled == [], f"Expected empty sample, got {sampled}"

        # filter should return empty view
        view = reader.filter(batch_type="t2i")
        assert len(view) == 0

        # scrimble_split / scrongle_split should not crash
        sdpa, sage = reader.scrimble_split()
        assert sdpa == [] and sage == []
        full, reduced = reader.scrongle_split()
        assert full == [] and reduced == []

        reader.close()
        print("  test_empty_dataset PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Dataset V2 Integration Tests")
    print("=" * 60)
    print()

    tests = [
        ("1. Write + read round-trip", test_write_read_roundtrip),
        ("2. Blob rotation", test_blob_rotation),
        ("3. Filter and sample", test_filter_and_sample),
        ("4. Flush mid-generation", test_flush_mid_generation),
        ("5. Empty dataset", test_empty_dataset),
    ]

    n_passed = 0
    n_failed = 0
    failures = []

    for name, test_fn in tests:
        print(f"--- {name} ---")
        try:
            test_fn()
            n_passed += 1
        except Exception as exc:
            n_failed += 1
            failures.append((name, exc))
            print(f"  FAILED: {exc}")
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

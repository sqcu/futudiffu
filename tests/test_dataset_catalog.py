"""Tests for the dataset catalog, loader, and inspector.

Tests cover:
  - Registration of V2 datasets (synthetic and real)
  - Integrity verification (mutation detection)
  - Split creation and loading
  - Index summary computation (unique counts, ranges, distributions)
  - CatalogedDataset loading with integrity checks
  - Error handling (not found, mutated, already registered)
  - Inspector output structure (list and inspect commands)
  - Idempotent re-registration

These tests create temporary V2 datasets using synthetic parquet data.
No GPU, no torch tensors, no model weights required. Pure metadata tests.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# Ensure repo root is importable
import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src_ii.dataset_catalog import (
    CatalogedDataset,
    DatasetAlreadyRegisteredError,
    DatasetCatalog,
    DatasetIntegrityError,
    DatasetNotFoundError,
    _compute_index_hash,
    _compute_index_summary,
    register_dataset,
    load_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    """Create a temp directory, clean up after test."""
    d = tempfile.mkdtemp(prefix="test_catalog_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def _make_synthetic_dataset(
    base_dir: Path,
    name: str = "test_dataset",
    n_traj: int = 10,
    n_widths: int = 3,
    n_backends: int = 2,
) -> Path:
    """Create a minimal synthetic V2 dataset with a parquet index and blob dirs.

    No actual safetensors blobs -- just the index.parquet and directory
    structure needed for catalog operations.
    """
    dataset_dir = base_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    blobs_dir = dataset_dir / "blobs"
    blobs_dir.mkdir(exist_ok=True)

    # Create some fake blob files (empty, just for counting)
    for i in range(2):
        (blobs_dir / f"blob_{i:03d}.safetensors").write_bytes(b"\x00" * 1024)

    # Build synthetic parquet data
    widths = [256, 512, 1024][:n_widths]
    backends = ["sdpa", "sage"][:n_backends]

    rows = []
    for i in range(n_traj):
        w = widths[i % len(widths)]
        h = w  # Square for simplicity
        backend = backends[i % len(backends)]
        rows.append({
            "traj_id": i,
            "prompt": f"test prompt {i % 5}",
            "prompt_idx": i % 5,
            "seed": 10000 + i,
            "cfg": 4.0,
            "width": w,
            "height": h,
            "n_steps": 30,
            "attention_backend": backend,
            "batch_type": "t2i",
            "denoise": None,
            "image_file": None,
            "is_gold": backend == "sdpa",
            "batch_idx": 0,
            "packed": False,
            "step_indices": [0, 4, 9, 14, 19, 24, 29],
            "has_final": True,
            "latent_channels": 16,
            "latent_height": h // 8,
            "latent_width": w // 8,
            "latent_dtype": "bfloat16",
            "blob_file": f"blob_{i // 5:03d}.safetensors",
            "key_prefix": f"{i:06d}",
            "n_tensors": 8,
            "bytes_total": 1024 * (h // 8) * (w // 8) * 16 * 2,
            "sampling_shift": 1.0 + (w * h / 1048576.0),
            "timing_seconds": 2.5 + i * 0.1,
            "created_at": datetime(2026, 2, 19, 14, 0, 0, tzinfo=timezone.utc),
            "parent_traj_id": None,
            "parent_step": None,
            "parent_denoise": None,
            "model_state_hash": None,
            "base_model_hash": None,
            "adapter_set_hash": None,
            "trajectory_hash": None,
            "active_adapters": None,
        })

    # Use the same schema as dataset_v2.py
    from futudiffu.dataset_v2 import INDEX_SCHEMA
    table = pa.Table.from_pylist(rows, schema=INDEX_SCHEMA)
    index_path = dataset_dir / "index.parquet"
    pq.write_table(table, str(index_path), compression="zstd")

    return dataset_dir


# ---------------------------------------------------------------------------
# Test: Index hashing
# ---------------------------------------------------------------------------

class TestIndexHashing:
    def test_hash_deterministic(self, tmp_dir):
        """Same file produces same hash."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=5)
        h1 = _compute_index_hash(ds / "index.parquet")
        h2 = _compute_index_hash(ds / "index.parquet")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256

    def test_hash_changes_on_mutation(self, tmp_dir):
        """Modifying the parquet file changes the hash."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=5)
        h1 = _compute_index_hash(ds / "index.parquet")

        # Mutate: rewrite with extra rows
        ds2 = _make_synthetic_dataset(tmp_dir, name="mutated", n_traj=10)
        shutil.copy(str(ds2 / "index.parquet"), str(ds / "index.parquet"))

        h2 = _compute_index_hash(ds / "index.parquet")
        assert h1 != h2


# ---------------------------------------------------------------------------
# Test: Index summary
# ---------------------------------------------------------------------------

class TestIndexSummary:
    def test_basic_summary(self, tmp_dir):
        """Summary has expected structure and counts."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=10, n_widths=3, n_backends=2)
        summary = _compute_index_summary(ds / "index.parquet")

        assert "columns" in summary
        assert "n_rows" in summary
        assert "unique_counts" in summary
        assert "value_ranges" in summary
        assert "value_distributions" in summary

        assert summary["n_rows"] == 10

    def test_unique_counts(self, tmp_dir):
        """Unique counts are correct for known columns."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=10, n_widths=3, n_backends=2)
        summary = _compute_index_summary(ds / "index.parquet")
        uc = summary["unique_counts"]

        assert uc["traj_id"] == 10
        assert uc["attention_backend"] == 2
        assert uc["width"] == 3
        assert uc["n_steps"] == 1  # all 30

    def test_value_ranges(self, tmp_dir):
        """Value ranges cover the expected min/max."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=10, n_widths=3)
        summary = _compute_index_summary(ds / "index.parquet")
        vr = summary["value_ranges"]

        assert "width" in vr
        assert vr["width"][0] == 256
        assert vr["width"][1] == 1024

        assert "seed" in vr
        assert vr["seed"][0] == 10000
        assert vr["seed"][1] == 10009

    def test_value_distributions(self, tmp_dir):
        """Low-cardinality columns get full histograms."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=10, n_backends=2)
        summary = _compute_index_summary(ds / "index.parquet")
        vd = summary["value_distributions"]

        assert "attention_backend" in vd
        assert "sage" in vd["attention_backend"]
        assert "sdpa" in vd["attention_backend"]

        # n_steps should be in distributions (1 unique value)
        assert "n_steps" in vd
        assert "30" in vd["n_steps"]

    def test_constant_column_detected(self, tmp_dir):
        """Columns with no variation (n_steps=30) have unique_count=1."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=10)
        summary = _compute_index_summary(ds / "index.parquet")

        assert summary["unique_counts"]["n_steps"] == 1


# ---------------------------------------------------------------------------
# Test: Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_basic(self, tmp_dir):
        """Register a synthetic dataset and verify the catalog entry."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=10)
        catalog_path = tmp_dir / "catalog.json"
        catalog = DatasetCatalog(catalog_path)

        did = catalog.register(ds, name="test_v2", tags=["synthetic", "test"])
        assert "test_v2" in did
        assert "10traj" in did

        entry = catalog.get(did)
        assert entry["n_trajectories"] == 10
        assert entry["format_version"] == "v2"
        assert "synthetic" in entry["tags"]
        assert entry["blob_count"] == 2
        assert entry["total_size_mb"] >= 0  # synthetic blobs are tiny

    def test_register_idempotent(self, tmp_dir):
        """Re-registering the same dataset returns the same ID."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=5)
        catalog_path = tmp_dir / "catalog.json"
        catalog = DatasetCatalog(catalog_path)

        did1 = catalog.register(ds, name="idem")
        did2 = catalog.register(ds, name="idem")
        assert did1 == did2

    def test_register_mutated_raises(self, tmp_dir):
        """Re-registering a mutated dataset raises an error."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=5)
        catalog_path = tmp_dir / "catalog.json"
        catalog = DatasetCatalog(catalog_path)

        did = catalog.register(ds, name="will_mutate")

        # Mutate the parquet file
        ds2 = _make_synthetic_dataset(tmp_dir, name="source", n_traj=15)
        shutil.copy(str(ds2 / "index.parquet"), str(ds / "index.parquet"))

        with pytest.raises(DatasetAlreadyRegisteredError, match="mutated"):
            catalog.register(ds, name="will_mutate")

    def test_register_no_index(self, tmp_dir):
        """Registering a directory without index.parquet fails."""
        empty_dir = tmp_dir / "empty"
        empty_dir.mkdir()
        catalog = DatasetCatalog(tmp_dir / "catalog.json")

        with pytest.raises(FileNotFoundError, match="index.parquet"):
            catalog.register(empty_dir)

    def test_register_persists(self, tmp_dir):
        """Catalog survives re-instantiation from disk."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=5)
        catalog_path = tmp_dir / "catalog.json"

        did = DatasetCatalog(catalog_path).register(ds, name="persist")

        # New catalog instance should see it
        catalog2 = DatasetCatalog(catalog_path)
        assert did in catalog2.list_datasets()
        entry = catalog2.get(did)
        assert entry["n_trajectories"] == 5

    def test_module_level_register(self, tmp_dir):
        """Test the module-level register_dataset() convenience."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=5)
        catalog_path = tmp_dir / "catalog.json"

        did = register_dataset(ds, name="conv", catalog_path=catalog_path)
        assert "conv" in did


# ---------------------------------------------------------------------------
# Test: Integrity verification
# ---------------------------------------------------------------------------

class TestVerification:
    def test_verify_intact(self, tmp_dir):
        """Freshly registered dataset passes verification."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=5)
        catalog = DatasetCatalog(tmp_dir / "catalog.json")
        did = catalog.register(ds)

        ok, msg = catalog.verify(did)
        assert ok is True
        assert "INTACT" in msg

    def test_verify_mutated(self, tmp_dir):
        """Mutated dataset fails verification."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=5)
        catalog = DatasetCatalog(tmp_dir / "catalog.json")
        did = catalog.register(ds)

        # Mutate
        ds2 = _make_synthetic_dataset(tmp_dir, name="source2", n_traj=20)
        shutil.copy(str(ds2 / "index.parquet"), str(ds / "index.parquet"))

        ok, msg = catalog.verify(did)
        assert ok is False
        assert "MISMATCH" in msg

    def test_verify_missing_file(self, tmp_dir):
        """Deleted index.parquet fails verification."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=5)
        catalog = DatasetCatalog(tmp_dir / "catalog.json")
        did = catalog.register(ds)

        # Delete the index
        os.remove(str(ds / "index.parquet"))

        ok, msg = catalog.verify(did)
        assert ok is False
        assert "not found" in msg

    def test_verify_all(self, tmp_dir):
        """verify_all checks all datasets."""
        ds1 = _make_synthetic_dataset(tmp_dir, name="ds1", n_traj=5)
        ds2 = _make_synthetic_dataset(tmp_dir, name="ds2", n_traj=10)
        catalog = DatasetCatalog(tmp_dir / "catalog.json")
        did1 = catalog.register(ds1, name="ds1")
        did2 = catalog.register(ds2, name="ds2")

        results = catalog.verify_all()
        assert len(results) == 2
        assert all(ok for ok, _ in results.values())


# ---------------------------------------------------------------------------
# Test: Splits
# ---------------------------------------------------------------------------

class TestSplits:
    def test_define_split_by_ids(self, tmp_dir):
        """Define a split with explicit traj_ids."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=10)
        catalog = DatasetCatalog(tmp_dir / "catalog.json")
        did = catalog.register(ds)

        result = catalog.define_split(did, "train", traj_ids=[0, 1, 2, 3, 4, 5, 6, 7])
        assert result == [0, 1, 2, 3, 4, 5, 6, 7]

        result = catalog.define_split(did, "val", traj_ids=[8, 9])
        assert result == [8, 9]

        # Verify retrieval
        assert catalog.get_split(did, "train") == [0, 1, 2, 3, 4, 5, 6, 7]
        assert catalog.get_split(did, "val") == [8, 9]

    def test_define_split_by_predicate(self, tmp_dir):
        """Define a split with a predicate function."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=10)
        catalog = DatasetCatalog(tmp_dir / "catalog.json")
        did = catalog.register(ds)

        result = catalog.define_split(
            did, "train",
            predicate=lambda row: row["traj_id"] < 8,
        )
        assert result == list(range(8))

    def test_split_persists(self, tmp_dir):
        """Splits survive catalog re-instantiation."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=10)
        catalog_path = tmp_dir / "catalog.json"
        catalog = DatasetCatalog(catalog_path)
        did = catalog.register(ds)
        catalog.define_split(did, "train", traj_ids=[0, 1, 2])

        # Re-load catalog
        catalog2 = DatasetCatalog(catalog_path)
        assert catalog2.get_split(did, "train") == [0, 1, 2]

    def test_split_not_found(self, tmp_dir):
        """Accessing a nonexistent split raises KeyError."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=5)
        catalog = DatasetCatalog(tmp_dir / "catalog.json")
        did = catalog.register(ds)

        with pytest.raises(KeyError, match="nonexistent"):
            catalog.get_split(did, "nonexistent")

    def test_remove_split(self, tmp_dir):
        """Removing a split removes it from the catalog."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=5)
        catalog = DatasetCatalog(tmp_dir / "catalog.json")
        did = catalog.register(ds)
        catalog.define_split(did, "test_split", traj_ids=[0, 1])

        catalog.remove_split(did, "test_split")

        with pytest.raises(KeyError):
            catalog.get_split(did, "test_split")


# ---------------------------------------------------------------------------
# Test: CatalogedDataset loading
# ---------------------------------------------------------------------------

class TestCatalogedDataset:
    def test_load_basic(self, tmp_dir):
        """Load a dataset through the catalog."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=10)
        catalog_path = tmp_dir / "catalog.json"
        catalog = DatasetCatalog(catalog_path)
        did = catalog.register(ds)

        loaded = CatalogedDataset.load(did, catalog_path=catalog_path)
        assert loaded.n_trajectories == 10
        assert loaded.dataset_id == did
        assert len(loaded.traj_ids) == 10

    def test_load_with_split(self, tmp_dir):
        """Load a specific split."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=10)
        catalog_path = tmp_dir / "catalog.json"
        catalog = DatasetCatalog(catalog_path)
        did = catalog.register(ds)
        catalog.define_split(did, "val", traj_ids=[8, 9])

        loaded = CatalogedDataset.load(did, split="val", catalog_path=catalog_path)
        assert loaded.n_trajectories == 2
        assert set(loaded.traj_ids) == {8, 9}

    def test_load_not_found(self, tmp_dir):
        """Loading a nonexistent dataset raises DatasetNotFoundError."""
        catalog_path = tmp_dir / "catalog.json"

        with pytest.raises(DatasetNotFoundError):
            CatalogedDataset.load("nonexistent_id", catalog_path=catalog_path)

    def test_load_mutated_raises(self, tmp_dir):
        """Loading a mutated dataset raises DatasetIntegrityError."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=5)
        catalog_path = tmp_dir / "catalog.json"
        catalog = DatasetCatalog(catalog_path)
        did = catalog.register(ds)

        # Mutate the index
        ds2 = _make_synthetic_dataset(tmp_dir, name="source3", n_traj=20)
        shutil.copy(str(ds2 / "index.parquet"), str(ds / "index.parquet"))

        with pytest.raises(DatasetIntegrityError, match="modified"):
            CatalogedDataset.load(did, catalog_path=catalog_path)

    def test_load_module_level(self, tmp_dir):
        """Test the module-level load_dataset() convenience."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=5)
        catalog_path = tmp_dir / "catalog.json"
        did = register_dataset(ds, name="conv_load", catalog_path=catalog_path)

        loaded = load_dataset(did, catalog_path=catalog_path)
        assert loaded.n_trajectories == 5

    def test_iter_metadata(self, tmp_dir):
        """iter_metadata returns dicts with expected keys."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=5)
        catalog_path = tmp_dir / "catalog.json"
        catalog = DatasetCatalog(catalog_path)
        did = catalog.register(ds)

        loaded = CatalogedDataset.load(did, catalog_path=catalog_path)
        rows = loaded.iter_metadata()
        assert len(rows) == 5
        assert "traj_id" in rows[0]
        assert "width" in rows[0]
        assert "attention_backend" in rows[0]


# ---------------------------------------------------------------------------
# Test: Catalog operations
# ---------------------------------------------------------------------------

class TestCatalogOperations:
    def test_list_empty(self, tmp_dir):
        """Empty catalog returns empty list."""
        catalog = DatasetCatalog(tmp_dir / "catalog.json")
        assert catalog.list_datasets() == []

    def test_unregister(self, tmp_dir):
        """Unregister removes the entry."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=5)
        catalog = DatasetCatalog(tmp_dir / "catalog.json")
        did = catalog.register(ds)

        catalog.unregister(did)
        assert did not in catalog.list_datasets()

    def test_unregister_not_found(self, tmp_dir):
        """Unregistering a nonexistent dataset raises."""
        catalog = DatasetCatalog(tmp_dir / "catalog.json")
        with pytest.raises(DatasetNotFoundError):
            catalog.unregister("ghost")

    def test_find_by_tag(self, tmp_dir):
        """find_by_tag returns matching datasets."""
        ds1 = _make_synthetic_dataset(tmp_dir, name="ds1", n_traj=5)
        ds2 = _make_synthetic_dataset(tmp_dir, name="ds2", n_traj=5)
        catalog = DatasetCatalog(tmp_dir / "catalog.json")
        did1 = catalog.register(ds1, tags=["multi_res"])
        did2 = catalog.register(ds2, tags=["production"])

        assert did1 in catalog.find_by_tag("multi_res")
        assert did2 not in catalog.find_by_tag("multi_res")
        assert did2 in catalog.find_by_tag("production")

    def test_find_by_path(self, tmp_dir):
        """find_by_path returns the dataset ID for a path."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=5)
        catalog = DatasetCatalog(tmp_dir / "catalog.json")
        did = catalog.register(ds)

        found = catalog.find_by_path(ds)
        assert found == did

    def test_catalog_json_is_readable(self, tmp_dir):
        """The catalog.json is valid, human-readable JSON."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=5)
        catalog_path = tmp_dir / "catalog.json"
        catalog = DatasetCatalog(catalog_path)
        catalog.register(ds, name="readable", tags=["test"])

        text = catalog_path.read_text()
        data = json.loads(text)
        assert "datasets" in data


# ---------------------------------------------------------------------------
# Test: Inspector output (integration)
# ---------------------------------------------------------------------------

class TestInspectorOutput:
    """Test the inspector commands produce expected output."""

    def test_list_output(self, tmp_dir, capsys):
        """List command produces structured output."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=10)
        catalog_path = tmp_dir / "catalog.json"
        catalog = DatasetCatalog(catalog_path)
        did = catalog.register(ds, name="inspect_test", tags=["test"])

        from scripts_ii.inspect_datasets import cmd_list
        cmd_list(catalog)

        captured = capsys.readouterr()
        assert "DATASET CATALOG" in captured.out
        assert "1 datasets registered" in captured.out
        assert did in captured.out
        assert "Trajectories: 10" in captured.out

    def test_inspect_output(self, tmp_dir, capsys):
        """Inspect command produces variation analysis."""
        ds = _make_synthetic_dataset(tmp_dir, n_traj=10, n_widths=3, n_backends=2)
        catalog_path = tmp_dir / "catalog.json"
        catalog = DatasetCatalog(catalog_path)
        did = catalog.register(ds, name="inspect_deep")

        from scripts_ii.inspect_datasets import cmd_inspect
        cmd_inspect(catalog, did)

        captured = capsys.readouterr()
        assert "DATASET:" in captured.out
        assert "variation analysis" in captured.out
        assert "NO VARIATION" in captured.out or "NO variation" in captured.out.lower()
        # n_steps=30 should be flagged as constant
        assert "n_steps" in captured.out

    def test_list_empty(self, tmp_dir, capsys):
        """List command on empty catalog produces helpful message."""
        catalog = DatasetCatalog(tmp_dir / "catalog.json")

        from scripts_ii.inspect_datasets import cmd_list
        cmd_list(catalog)

        captured = capsys.readouterr()
        assert "empty" in captured.out.lower() or "No datasets" in captured.out


# ---------------------------------------------------------------------------
# Test: Real dataset (if available)
# ---------------------------------------------------------------------------

class TestRealDataset:
    """Tests against the actual multi_res_trajectories dataset on disk.

    These tests are skipped if the dataset directory does not exist.
    """

    REAL_DATASET_PATH = REPO_ROOT / "multi_res_trajectories"

    @pytest.fixture(autouse=True)
    def skip_if_no_real_data(self):
        if not (self.REAL_DATASET_PATH / "index.parquet").exists():
            pytest.skip("multi_res_trajectories not available on disk")

    def test_register_real(self, tmp_dir):
        """Register the real multi_res dataset."""
        catalog = DatasetCatalog(tmp_dir / "catalog.json")
        did = catalog.register(
            self.REAL_DATASET_PATH,
            name="multi_res_v2",
            tags=["multi_res", "logsnr_uniform"],
        )

        entry = catalog.get(did)
        # The real dataset has either 60 or 120 trajectories
        assert entry["n_trajectories"] >= 60
        assert entry["blob_count"] >= 6
        assert entry["format_version"] == "v2"

    def test_summary_real(self, tmp_dir):
        """Index summary of real dataset has expected variation."""
        summary = _compute_index_summary(
            self.REAL_DATASET_PATH / "index.parquet"
        )

        uc = summary["unique_counts"]
        # Should have multiple resolutions
        assert uc.get("width", 0) >= 10
        # Should have 2 backends
        assert uc.get("attention_backend", 0) == 2
        # Should have multiple prompts
        assert uc.get("prompt_idx", 0) >= 5

    def test_verify_real(self, tmp_dir):
        """Real dataset passes integrity verification."""
        catalog = DatasetCatalog(tmp_dir / "catalog.json")
        did = catalog.register(self.REAL_DATASET_PATH, name="real")

        ok, msg = catalog.verify(did)
        assert ok is True

    def test_inspect_real(self, tmp_dir, capsys):
        """Inspector works on real dataset."""
        catalog = DatasetCatalog(tmp_dir / "catalog.json")
        did = catalog.register(self.REAL_DATASET_PATH, name="real_inspect")

        from scripts_ii.inspect_datasets import cmd_inspect
        cmd_inspect(catalog, did)

        captured = capsys.readouterr()
        assert "DATASET:" in captured.out
        assert "attention_backend" in captured.out

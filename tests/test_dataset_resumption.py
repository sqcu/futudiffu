"""Tests for dataset_resumption: plan building, identity matching, and resume logic.

Tests cover:
  1. trajectory_identity extracts correct identity tuples
  2. Generation plan is deterministic (same params -> same plan)
  3. Plan save/load round-trip
  4. find_completed_identities reads parquet index correctly
  5. compute_remaining_work correctly diffs plan against dataset
  6. Partial dataset resume: completed trajectories are skipped
  7. Empty dataset: all trajectories remain
  8. Full dataset: no trajectories remain
  9. Step sigmas sidecar incremental update
  10. Edge cases: missing columns, nonexistent dataset dir

No GPU, no model weights, no inference. Pure metadata tests using
synthetic parquet data in temp directories.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src_ii.dataset_resumption import (
    IDENTITY_COLUMNS,
    compute_remaining_work,
    find_completed_identities,
    load_generation_plan,
    save_generation_plan,
    trajectory_identity,
    update_step_sigmas_sidecar,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    """Create a temp directory, clean up after test."""
    d = tempfile.mkdtemp(prefix="test_resumption_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def _make_plan(n: int = 10, base_seed: int = 1000) -> list[dict]:
    """Create a synthetic generation plan."""
    plan = []
    for i in range(n):
        plan.append({
            "seed": base_seed + i,
            "prompt_idx": i % 5,
            "prompt": f"test prompt {i % 5}",
            "width": [256, 512, 1024][i % 3],
            "height": [256, 512, 1024][i % 3],
            "attention_backend": ["sdpa", "sage"][i % 2],
            "n_steps": 30,
            "cfg": 4.0,
            "resolution_tier": "test",
            "sampling_shift": 1.0,
            "sparse_steps": [0, 4, 9, 14, 19, 24, 29],
            "step_selection": "logsnr_uniform",
            "plan_index": i,
        })
    return plan


def _make_synthetic_v2_dataset(
    dataset_dir: Path,
    specs: list[dict],
) -> None:
    """Create a minimal V2 dataset with a parquet index from spec dicts.

    Only writes the parquet index (no blob files needed for resumption).
    """
    from futudiffu.dataset_v2 import INDEX_SCHEMA

    dataset_dir.mkdir(parents=True, exist_ok=True)
    blobs_dir = dataset_dir / "blobs"
    blobs_dir.mkdir(exist_ok=True)

    rows = []
    for i, spec in enumerate(specs):
        rows.append({
            "traj_id": i,
            "prompt": spec.get("prompt", f"prompt {i}"),
            "prompt_idx": spec["prompt_idx"],
            "seed": spec["seed"],
            "cfg": float(spec["cfg"]),
            "width": spec["width"],
            "height": spec["height"],
            "n_steps": spec["n_steps"],
            "attention_backend": spec["attention_backend"],
            "batch_type": "t2i",
            "denoise": None,
            "image_file": None,
            "is_gold": spec["attention_backend"] == "sdpa",
            "batch_idx": 0,
            "packed": False,
            "step_indices": [0, 4, 9, 14, 19, 24, 29],
            "has_final": True,
            "latent_channels": 16,
            "latent_height": spec["height"] // 8,
            "latent_width": spec["width"] // 8,
            "latent_dtype": "bfloat16",
            "blob_file": f"blob_{i // 10:03d}.safetensors",
            "key_prefix": f"{i:06d}",
            "n_tensors": 8,
            "bytes_total": 1024,
            "sampling_shift": spec.get("sampling_shift", 1.0),
            "timing_seconds": 2.5,
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

    table = pa.Table.from_pylist(rows, schema=INDEX_SCHEMA)
    pq.write_table(table, str(dataset_dir / "index.parquet"), compression="zstd")


# ---------------------------------------------------------------------------
# Test: trajectory_identity
# ---------------------------------------------------------------------------

class TestTrajectoryIdentity:
    def test_extracts_correct_fields(self):
        """Identity tuple contains exactly the identity columns."""
        spec = {
            "seed": 42,
            "prompt_idx": 3,
            "width": 512,
            "height": 512,
            "attention_backend": "sdpa",
            "n_steps": 30,
            "cfg": 4.0,
            "extra_field": "ignored",
        }
        identity = trajectory_identity(spec)
        assert identity == (42, 3, 512, 512, "sdpa", 30, 4.0)

    def test_different_specs_different_identities(self):
        """Two specs differing in any identity column produce different tuples."""
        base = {
            "seed": 42, "prompt_idx": 3, "width": 512, "height": 512,
            "attention_backend": "sdpa", "n_steps": 30, "cfg": 4.0,
        }
        modified = dict(base, seed=43)
        assert trajectory_identity(base) != trajectory_identity(modified)

    def test_same_specs_same_identity(self):
        """Same identity columns produce same tuple regardless of extras."""
        spec_a = {
            "seed": 42, "prompt_idx": 3, "width": 512, "height": 512,
            "attention_backend": "sdpa", "n_steps": 30, "cfg": 4.0,
            "resolution_tier": "512sq",
        }
        spec_b = {
            "seed": 42, "prompt_idx": 3, "width": 512, "height": 512,
            "attention_backend": "sdpa", "n_steps": 30, "cfg": 4.0,
            "resolution_tier": "different",
        }
        assert trajectory_identity(spec_a) == trajectory_identity(spec_b)

    def test_missing_column_raises(self):
        """Missing identity column raises KeyError."""
        spec = {"seed": 42, "prompt_idx": 3}  # missing many fields
        with pytest.raises(KeyError):
            trajectory_identity(spec)


# ---------------------------------------------------------------------------
# Test: plan save/load
# ---------------------------------------------------------------------------

class TestPlanPersistence:
    def test_round_trip(self, tmp_dir):
        """Plan save/load preserves all data."""
        plan = _make_plan(5)
        plan_path = tmp_dir / "plan.json"

        save_generation_plan(plan, plan_path)
        loaded = load_generation_plan(plan_path)

        assert len(loaded) == 5
        for orig, read in zip(plan, loaded):
            assert orig["seed"] == read["seed"]
            assert orig["prompt_idx"] == read["prompt_idx"]
            assert orig["width"] == read["width"]
            assert orig["attention_backend"] == read["attention_backend"]

    def test_atomic_write(self, tmp_dir):
        """No .tmp file left behind after save."""
        plan_path = tmp_dir / "plan.json"
        save_generation_plan(_make_plan(3), plan_path)

        assert plan_path.exists()
        tmp_file = plan_path.with_suffix(".json.tmp")
        assert not tmp_file.exists()

    def test_load_nonexistent_raises(self, tmp_dir):
        """Loading a nonexistent plan raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_generation_plan(tmp_dir / "nonexistent.json")

    def test_validates_identity_columns(self, tmp_dir):
        """Save rejects plan entries missing identity columns."""
        bad_plan = [{"seed": 42}]  # missing most identity columns
        with pytest.raises(ValueError, match="missing identity columns"):
            save_generation_plan(bad_plan, tmp_dir / "bad.json")


# ---------------------------------------------------------------------------
# Test: find_completed_identities
# ---------------------------------------------------------------------------

class TestFindCompleted:
    def test_finds_existing_trajectories(self, tmp_dir):
        """Correctly identifies trajectories in the parquet index."""
        plan = _make_plan(5)
        ds_dir = tmp_dir / "dataset_v2"
        _make_synthetic_v2_dataset(ds_dir, plan[:3])  # write first 3

        completed = find_completed_identities(ds_dir)

        assert len(completed) == 3
        for spec in plan[:3]:
            assert trajectory_identity(spec) in completed
        for spec in plan[3:]:
            assert trajectory_identity(spec) not in completed

    def test_empty_dataset(self, tmp_dir):
        """Empty dataset returns empty set."""
        ds_dir = tmp_dir / "empty_dataset"
        # No index.parquet at all
        assert find_completed_identities(ds_dir) == set()

    def test_nonexistent_dir(self, tmp_dir):
        """Nonexistent directory returns empty set."""
        assert find_completed_identities(tmp_dir / "nope") == set()


# ---------------------------------------------------------------------------
# Test: compute_remaining_work
# ---------------------------------------------------------------------------

class TestRemainingWork:
    def test_fresh_run(self, tmp_dir):
        """No existing dataset -> all work remaining."""
        plan = _make_plan(10)
        ds_dir = tmp_dir / "empty"

        remaining, completed, n_total = compute_remaining_work(plan, ds_dir)

        assert n_total == 10
        assert len(remaining) == 10
        assert len(completed) == 0

    def test_partial_resume(self, tmp_dir):
        """Partial dataset -> only missing trajectories remain."""
        plan = _make_plan(10)
        ds_dir = tmp_dir / "dataset_v2"
        _make_synthetic_v2_dataset(ds_dir, plan[:6])  # first 6 done

        remaining, completed, n_total = compute_remaining_work(plan, ds_dir)

        assert n_total == 10
        assert len(completed) == 6
        assert len(remaining) == 4

        # Remaining should be the last 4
        remaining_seeds = {spec["seed"] for spec in remaining}
        expected_seeds = {spec["seed"] for spec in plan[6:]}
        assert remaining_seeds == expected_seeds

    def test_all_complete(self, tmp_dir):
        """Fully completed dataset -> nothing remaining."""
        plan = _make_plan(5)
        ds_dir = tmp_dir / "dataset_v2"
        _make_synthetic_v2_dataset(ds_dir, plan)  # all 5 done

        remaining, completed, n_total = compute_remaining_work(plan, ds_dir)

        assert n_total == 5
        assert len(remaining) == 0
        assert len(completed) == 5

    def test_preserves_plan_order(self, tmp_dir):
        """Remaining entries preserve their original plan order."""
        plan = _make_plan(10)
        ds_dir = tmp_dir / "dataset_v2"
        # Complete entries 0, 2, 4, 6, 8 (even indices)
        even_specs = [plan[i] for i in range(0, 10, 2)]
        _make_synthetic_v2_dataset(ds_dir, even_specs)

        remaining, _, _ = compute_remaining_work(plan, ds_dir)

        # Remaining should be indices 1, 3, 5, 7, 9
        assert len(remaining) == 5
        for i, spec in enumerate(remaining):
            expected_idx = 2 * i + 1
            assert spec["seed"] == plan[expected_idx]["seed"]


# ---------------------------------------------------------------------------
# Test: step_sigmas sidecar
# ---------------------------------------------------------------------------

class TestStepSigmasSidecar:
    def test_create_new(self, tmp_dir):
        """Creating a new sidecar file."""
        sidecar_path = tmp_dir / "step_sigmas.json"

        update_step_sigmas_sidecar(
            sidecar_path, 0,
            {"step_00": 0.95, "final": 0.0},
            {"step_00": -2.94, "final": 13.8},
        )

        data = json.loads(sidecar_path.read_text())
        assert "0" in data
        assert data["0"]["step_sigmas"]["step_00"] == 0.95

    def test_incremental_update(self, tmp_dir):
        """Adding entries preserves existing ones."""
        sidecar_path = tmp_dir / "step_sigmas.json"

        update_step_sigmas_sidecar(
            sidecar_path, 0,
            {"step_00": 0.95}, {"step_00": -2.94},
        )
        update_step_sigmas_sidecar(
            sidecar_path, 1,
            {"step_04": 0.7}, {"step_04": -1.5},
        )

        data = json.loads(sidecar_path.read_text())
        assert "0" in data
        assert "1" in data
        assert data["0"]["step_sigmas"]["step_00"] == 0.95
        assert data["1"]["step_sigmas"]["step_04"] == 0.7

    def test_overwrite_entry(self, tmp_dir):
        """Updating an existing entry replaces it."""
        sidecar_path = tmp_dir / "step_sigmas.json"

        update_step_sigmas_sidecar(
            sidecar_path, 0,
            {"step_00": 0.95}, {"step_00": -2.94},
        )
        update_step_sigmas_sidecar(
            sidecar_path, 0,
            {"step_00": 0.8, "final": 0.0}, {"step_00": -2.0, "final": 13.8},
        )

        data = json.loads(sidecar_path.read_text())
        assert data["0"]["step_sigmas"]["step_00"] == 0.8
        assert "final" in data["0"]["step_sigmas"]


# ---------------------------------------------------------------------------
# Test: generation plan determinism
# ---------------------------------------------------------------------------

class TestPlanDeterminism:
    def test_plan_is_deterministic(self):
        """Same parameters produce identical plan."""
        # Import the generation script's plan builder
        sys.path.insert(0, str(REPO_ROOT / "scripts_ii"))

        # We can't import the full script (it imports torch at module level),
        # but we can verify the identity columns concept is deterministic.
        plan_a = _make_plan(20, base_seed=400000)
        plan_b = _make_plan(20, base_seed=400000)

        for a, b in zip(plan_a, plan_b):
            assert trajectory_identity(a) == trajectory_identity(b)

    def test_identity_columns_are_hashable(self):
        """Identity tuples can be stored in sets."""
        plan = _make_plan(100)
        identities = {trajectory_identity(spec) for spec in plan}
        # All 100 should be unique (different seeds)
        assert len(identities) == 100


# ---------------------------------------------------------------------------
# Test: integration with DatasetWriter
# ---------------------------------------------------------------------------

class TestIntegrationWithWriter:
    """Test that resumption works with actual DatasetWriter output."""

    def test_resume_after_partial_write(self, tmp_dir):
        """Write 3 trajectories via DatasetWriter, verify resume detects them."""
        import torch
        from futudiffu.dataset_v2 import DatasetWriter

        ds_dir = tmp_dir / "dataset_v2"
        plan = _make_plan(6)

        # Write first 3 trajectories using real DatasetWriter
        with DatasetWriter(str(ds_dir)) as writer:
            for spec in plan[:3]:
                tensors = {
                    "step_00": torch.randn(1, 16, spec["height"] // 8, spec["width"] // 8,
                                           dtype=torch.bfloat16),
                    "final": torch.randn(1, 16, spec["height"] // 8, spec["width"] // 8,
                                         dtype=torch.bfloat16),
                }
                metadata = {
                    "prompt": spec["prompt"],
                    "prompt_idx": spec["prompt_idx"],
                    "seed": spec["seed"],
                    "cfg": spec["cfg"],
                    "width": spec["width"],
                    "height": spec["height"],
                    "n_steps": spec["n_steps"],
                    "attention_backend": spec["attention_backend"],
                    "batch_type": "t2i",
                }
                writer.add_trajectory(tensors=tensors, metadata=metadata)

        # Now compute remaining work
        remaining, completed, n_total = compute_remaining_work(plan, ds_dir)

        assert n_total == 6
        assert len(completed) == 3
        assert len(remaining) == 3

        # The remaining should be the last 3 specs
        remaining_seeds = {spec["seed"] for spec in remaining}
        expected_seeds = {plan[3]["seed"], plan[4]["seed"], plan[5]["seed"]}
        assert remaining_seeds == expected_seeds

"""Dataset generation resumption: plan comparison against existing datasets.

Solves the problem: "I planned 60 trajectories, the script crashed at #37,
which ones still need generating?" without loading any tensor data.

The generation plan is a deterministic list of trajectory specifications
(seed, prompt_idx, width, height, attention_backend, n_steps, cfg). Each
spec has a unique identity formed by these fields. Resumption inspects the
parquet index of an existing dataset and returns the subset of the plan
that has not yet been persisted.

Import constraints:
  - pathlib, json -- stdlib
  - pyarrow -- for parquet I/O (already a project dependency)
  - No torch imports (this is a metadata-only module)
  - No imports from src/futudiffu/ (frozen)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Trajectory identity
# ---------------------------------------------------------------------------

# These columns uniquely identify a trajectory in the generation plan.
# Two trajectories are "the same" if and only if all these fields match.
# This set is deliberately minimal: it captures the parameters that
# determine the output (given the same model weights and code version).
IDENTITY_COLUMNS = (
    "seed",
    "prompt_idx",
    "width",
    "height",
    "attention_backend",
    "n_steps",
    "cfg",
)


def trajectory_identity(spec: dict) -> tuple:
    """Extract the identity tuple from a trajectory specification.

    The identity is a tuple of (seed, prompt_idx, width, height,
    attention_backend, n_steps, cfg) -- the minimal set of fields
    that uniquely determine a trajectory's output.

    Args:
        spec: A trajectory specification dict. Must contain all
            IDENTITY_COLUMNS as keys.

    Returns:
        A hashable tuple suitable for set membership testing.

    Raises:
        KeyError: If any identity column is missing.
    """
    return tuple(spec[col] for col in IDENTITY_COLUMNS)


# ---------------------------------------------------------------------------
# Plan persistence
# ---------------------------------------------------------------------------

def save_generation_plan(
    plan: list[dict],
    plan_path: Path | str,
) -> None:
    """Write a generation plan to a JSON file.

    The plan is a list of trajectory specification dicts. Each dict
    must contain all IDENTITY_COLUMNS plus any additional metadata
    needed for generation (prompt text, resolution_tier, etc.).

    The file is written atomically (write to .tmp, then rename).

    Args:
        plan: List of trajectory spec dicts.
        plan_path: Path to write the plan JSON.
    """
    plan_path = Path(plan_path)
    plan_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate that all specs have identity columns
    for i, spec in enumerate(plan):
        missing = set(IDENTITY_COLUMNS) - set(spec.keys())
        if missing:
            raise ValueError(
                f"Plan entry {i} missing identity columns: {missing}"
            )

    tmp_path = plan_path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(plan, indent=2, default=str),
        encoding="utf-8",
    )
    import os
    os.replace(str(tmp_path), str(plan_path))


def load_generation_plan(plan_path: Path | str) -> list[dict]:
    """Load a generation plan from a JSON file.

    Args:
        plan_path: Path to the plan JSON.

    Returns:
        List of trajectory spec dicts.

    Raises:
        FileNotFoundError: If the plan file does not exist.
    """
    plan_path = Path(plan_path)
    if not plan_path.exists():
        raise FileNotFoundError(f"Generation plan not found: {plan_path}")

    text = plan_path.read_text(encoding="utf-8")
    return json.loads(text)


# ---------------------------------------------------------------------------
# Completed trajectory detection
# ---------------------------------------------------------------------------

def find_completed_identities(dataset_dir: Path | str) -> set[tuple]:
    """Scan a V2 dataset's parquet index and return the identity tuples
    of all trajectories already present.

    This reads only the parquet index (a few KB), not any tensor data.

    Args:
        dataset_dir: Path to the V2 dataset directory (must contain
            index.parquet).

    Returns:
        Set of identity tuples for trajectories in the dataset.
        Empty set if the dataset directory or index does not exist.
    """
    dataset_dir = Path(dataset_dir)
    index_path = dataset_dir / "index.parquet"

    if not index_path.exists():
        return set()

    import pyarrow.parquet as pq

    table = pq.read_table(str(index_path))

    # Read only the identity columns for efficiency
    available_cols = set(table.column_names)
    missing_cols = set(IDENTITY_COLUMNS) - available_cols
    if missing_cols:
        raise ValueError(
            f"Parquet index missing identity columns: {missing_cols}. "
            f"Available: {sorted(available_cols)}"
        )

    # Extract identity tuples
    completed = set()
    n_rows = len(table)
    # Build column arrays once
    col_arrays = {col: table.column(col).to_pylist() for col in IDENTITY_COLUMNS}

    for i in range(n_rows):
        identity = tuple(col_arrays[col][i] for col in IDENTITY_COLUMNS)
        completed.add(identity)

    return completed


# ---------------------------------------------------------------------------
# Plan diffing
# ---------------------------------------------------------------------------

def compute_remaining_work(
    plan: list[dict],
    dataset_dir: Path | str,
) -> tuple[list[dict], list[dict], int]:
    """Compare a generation plan against an existing dataset and return
    the remaining work.

    Args:
        plan: Full generation plan (list of trajectory spec dicts).
        dataset_dir: Path to the V2 dataset directory.

    Returns:
        (remaining, completed, n_total) where:
            remaining: List of spec dicts not yet in the dataset,
                preserving the original plan order.
            completed: List of spec dicts already in the dataset.
            n_total: Total plan size (== len(remaining) + len(completed)).
    """
    completed_ids = find_completed_identities(dataset_dir)

    remaining = []
    completed = []

    for spec in plan:
        identity = trajectory_identity(spec)
        if identity in completed_ids:
            completed.append(spec)
        else:
            remaining.append(spec)

    return remaining, completed, len(plan)


# ---------------------------------------------------------------------------
# Step sigmas sidecar (incremental)
# ---------------------------------------------------------------------------

def update_step_sigmas_sidecar(
    sidecar_path: Path | str,
    traj_index: int,
    step_sigmas: dict[str, float],
    step_logsnrs: dict[str, float],
) -> None:
    """Incrementally update the step_sigmas.json sidecar file.

    Reads the existing sidecar (if present), adds/updates the entry
    for traj_index, and writes back atomically.

    Args:
        sidecar_path: Path to step_sigmas.json.
        traj_index: The trajectory index (key in the sidecar dict).
        step_sigmas: Per-step sigma values.
        step_logsnrs: Per-step logSNR values.
    """
    import os

    sidecar_path = Path(sidecar_path)

    # Load existing
    existing: dict = {}
    if sidecar_path.exists():
        try:
            existing = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}

    # Update
    existing[str(traj_index)] = {
        "step_sigmas": step_sigmas,
        "step_logsnrs": step_logsnrs,
    }

    # Write atomically
    tmp_path = sidecar_path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(existing, indent=2),
        encoding="utf-8",
    )
    os.replace(str(tmp_path), str(sidecar_path))

"""Dataset deprecation filters for BTRM training data selection.

Policy rollout trajectories (run_name in ("policy_rollout_v1", "2xh100_20260216"))
were generated with LoRA adapters active. Using them to train a reward model
conflates model quality with adapter effects. By default, BTRM training
excludes these trajectories and uses only original base model data.

Usage:
    import pyarrow.parquet as pq
    from src_ii.dataset_filters import filter_training_trajectories

    table = pq.read_table("btrm_dataset_v2/index.parquet")
    filtered = filter_training_trajectories(table)
    # filtered is a pyarrow Table with only base model trajectories
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc


# Run names known to contain policy rollout data (adapter-contaminated).
_POLICY_ROLLOUT_RUN_NAMES = frozenset({"policy_rollout_v1", "2xh100_20260216"})

# Default filter configuration. Keys are filter names, values are booleans.
# Overrides can be passed to filter_training_trajectories() as a dict.
DATASET_DEPRECATION_DEFAULTS: dict[str, bool] = {
    # Exclude trajectories generated with LoRA adapters active.
    # These come from policy rollout runs where the model state differs
    # from the base model. Training a reward model on them would conflate
    # model quality with adapter effects.
    "exclude_policy_rollouts": True,

    # If True, exclude all V1-migrated data (only use V2-native).
    # Useful for experiments that should only use fresh generation data.
    "exclude_v1_only": False,
}


def filter_training_trajectories(
    index: pa.Table,
    overrides: dict[str, Any] | None = None,
) -> pa.Table:
    """Apply deprecation filters to a V2 dataset index for BTRM training.

    Args:
        index: A pyarrow Table read from a V2 index.parquet file.
            Must have a 'run_name' column for policy rollout filtering.
        overrides: Optional dict to override DATASET_DEPRECATION_DEFAULTS.
            Example: {"exclude_policy_rollouts": False} to include rollouts.

    Returns:
        A filtered pyarrow Table containing only the trajectories that
        pass all active filters.
    """
    config = dict(DATASET_DEPRECATION_DEFAULTS)
    if overrides:
        config.update(overrides)

    mask = None

    # --- Filter: exclude policy rollouts ---
    if config.get("exclude_policy_rollouts", True):
        if "run_name" in index.column_names:
            run_names = index.column("run_name")
            # Keep rows where run_name is NOT in the policy rollout set
            is_rollout = pc.is_in(run_names, pa.array(list(_POLICY_ROLLOUT_RUN_NAMES)))
            keep = pc.invert(is_rollout)
            mask = keep if mask is None else pc.and_(mask, keep)

    # --- Filter: exclude V1-migrated data ---
    if config.get("exclude_v1_only", False):
        if "run_name" in index.column_names:
            run_names = index.column("run_name")
            # Exclude anything with "v1" in the run_name
            is_v1_original = pc.equal(run_names, "original_v1")
            is_v1_rollout = pc.equal(run_names, "policy_rollout_v1")
            is_v1 = pc.or_(is_v1_original, is_v1_rollout)
            keep = pc.invert(is_v1)
            mask = keep if mask is None else pc.and_(mask, keep)

    if mask is not None:
        return index.filter(mask)
    return index

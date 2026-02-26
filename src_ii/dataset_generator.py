"""BTRM trajectory dataset generation: schedule-driven, server-backed.

Generates diffusion trajectories via an inference server (InferenceClient)
and writes them to V2 format (DatasetWriter). Supports mixed-resolution
generation, attention backend variation, provenance tracking, and bin-packed
FlexAttention batches.

Replaces the monolithic scripts/generate_btrm_dataset.py with a composable
library module. The CLI wrapper lives in scripts_ii/generate_btrm_dataset.py.

Outer specification axes affected (from user_dataflow_and_lifecycle_rollup.md):
  2. Trajectory Persistence: V2 parquet + safetensors blobs (sealed, no WIP on disk)
  3. Side-Channel Observability: sampling state hashes (model, adapter, trajectory)
  10. Sequence Packing: bin-packed FlexAttention batches for mixed-resolution

Import constraints:
  - IMPORTS from src_ii: bin_packer, sampling_identity
  - IMPORTS from futudiffu: client (InferenceClient), dataset_v2 (DatasetWriter)
  - DOES NOT import: server, model_manager, diffusion_model, training code
"""

from __future__ import annotations

import copy
import itertools
import json
import random
import signal
import sys
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import torch

from .bin_packer import (
    DEFAULT_CAP_TOKENS,
    REFERENCE_SEQ_LEN,
    REFERENCE_TOTAL_LEN,
    RESOLUTION_TIERS,
    BinPackScheduler,
    build_generation_plan,
    compute_effective_seq_len,
    compute_seq_len,
)
from .sigma_schedule import resolution_shift
from .sampling_identity import (
    compute_adapter_set_hash,
    compute_model_state_hash,
    compute_trajectory_hash,
    serialize_active_adapters,
)



# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DatasetGenerationConfig:
    """Configuration for a generation run.

    Attributes:
        prompts: List of prompt strings to generate from.
        seeds: List of PRNG seeds. None = generate random seeds.
        resolution_tiers: Which resolution tiers to generate ("full", "medium", "small").
        attention_backends: Which attention backends to use ("sdpa", "sage").
        n_steps: Number of Euler diffusion steps.
        cfg: Classifier-free guidance scale.
        output_dir: Path to write V2 dataset.
        run_name: Human-readable name for this generation run (stored in provenance).
        base_model_hash: Identifier for the base model weights. Placeholder until
            we can hash the actual weights at generation time.
        sparse_steps: Which step indices to save latents for.
        sampling_shift: Sigma schedule shift parameter. None = auto-compute
            per-item from resolution_shift(width, height). Explicit float
            overrides the automatic computation for all items.
        multiplier: Timestep multiplier.
        server_endpoint: ZeroMQ endpoint for the inference server.
        flush_interval: Seal the current blob every N trajectories (limits
            data loss on crash).
        render_count: How many trajectories to VAE-decode and save as PNGs.
        active_adapters: List of adapter descriptors currently active on the
            model. Each dict has "name", "strength", "param_hash". Empty list
            means base model only.
        source_device: Human-readable device identifier (e.g. "rtx4090_0").
    """
    prompts: list[str]
    seeds: list[int] | None = None
    resolution_tiers: list[str] = field(default_factory=lambda: ["full"])
    attention_backends: list[str] = field(default_factory=lambda: ["sdpa", "sage"])
    n_steps: int = 30
    cfg: float = 4.0
    output_dir: str = "btrm_dataset_v2"
    run_name: str = "generation"
    base_model_hash: str = "z_image_v1"
    sparse_steps: list[int] = field(default_factory=lambda: [0, 4, 9, 14, 19, 24, 29])
    sampling_shift: float | None = None
    multiplier: float = 1.0
    server_endpoint: str = "tcp://localhost:5555"
    flush_interval: int = 50
    render_count: int = 6
    active_adapters: list[dict[str, Any]] = field(default_factory=list)
    source_device: str = "unknown"
    plan_spec: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        Drops non-serializable fields (callables, objects). Every field
        that is a plain Python type (str, int, float, bool, list, dict, None)
        is included as-is.
        """
        d: dict[str, Any] = {}
        for f in fields(self):
            # plan_spec is transient (used for lazy tile iteration), skip it
            if f.name == "plan_spec":
                continue
            val = getattr(self, f.name)
            # Skip anything that isn't JSON-native
            if isinstance(val, (str, int, float, bool, type(None))):
                d[f.name] = val
            elif isinstance(val, (list, dict)):
                d[f.name] = copy.deepcopy(val)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DatasetGenerationConfig":
        """Construct from a dict, filling defaults for missing keys.

        Unknown keys in *d* are silently ignored so that plan-level
        metadata (``phases``, ``node_partitioning``, ``prompts_file``,
        ``sparse_steps_mode``, ``n_save``, ``seeds_per_combo``,
        ``base_seed``) can be present without raising.
        """
        valid_names = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in valid_names}
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Plan loading / saving
# ---------------------------------------------------------------------------

def _resolve_prompts(plan: dict[str, Any], plan_dir: Path | None = None) -> list[str]:
    """Resolve prompts from either ``prompts`` or ``prompts_file`` key.

    Args:
        plan: Plan dict.
        plan_dir: Directory containing the plan file. If provided,
            relative ``prompts_file`` paths are resolved relative to this
            directory (not cwd). This makes plans work on remote nodes
            regardless of working directory.
    """
    if "prompts" in plan and plan["prompts"]:
        return list(plan["prompts"])
    if "prompts_file" in plan and plan["prompts_file"]:
        p = Path(plan["prompts_file"])
        if not p.is_absolute() and not p.exists() and plan_dir is not None:
            # Relative path doesn't exist from cwd — try relative to plan dir
            p_from_plan = plan_dir / p
            if p_from_plan.exists():
                p = p_from_plan
            else:
                # Also try relative to plan dir's parent (repo root convention:
                # plans/ is one level below repo root, prompts_file paths are
                # relative to repo root)
                p_from_repo = plan_dir.parent / p
                if p_from_repo.exists():
                    p = p_from_repo
        if not p.exists():
            raise FileNotFoundError(
                f"prompts_file not found: {plan['prompts_file']!r} "
                f"(tried cwd, plan dir {plan_dir}, plan dir parent)"
            )
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        return [line.strip() for line in lines if line.strip()]
    raise ValueError("Plan must contain either 'prompts' or 'prompts_file'")


def _resolve_sparse_steps(
    plan: dict[str, Any],
    n_steps: int,
) -> list[int] | None:
    """Resolve sparse_steps from explicit list or mode + n_save.

    Returns None when the mode is ``logsnr_uniform`` — the generator
    must compute per-trajectory steps using resolution info at runtime.
    Returns a concrete list for ``step_uniform`` or explicit indices.
    """
    if "sparse_steps" in plan and plan["sparse_steps"] is not None:
        return list(plan["sparse_steps"])

    mode = plan.get("sparse_steps_mode", "logsnr_uniform")
    n_save = plan.get("n_save", 7)

    if mode == "step_uniform":
        # Evenly spaced step indices [0, ..., n_steps-1]
        if n_save >= n_steps:
            return list(range(n_steps))
        return [round(i * (n_steps - 1) / (n_save - 1)) for i in range(n_save)]

    if mode == "logsnr_uniform":
        # Cannot resolve without per-trajectory resolution.  Return None;
        # the DatasetGenerator will use compute_logsnr_uniform_steps()
        # per-item at generation time.
        return None

    raise ValueError(f"Unknown sparse_steps_mode: {mode!r}")


def _resolve_seeds(
    plan: dict[str, Any],
    n_prompts: int,
    resolution_tiers: list[str],
    attention_backends: list[str],
    node_rank: int = 0,
) -> list[int] | None:
    """Build a deterministic seed list from base_seed + seeds_per_combo.

    When ``node_partitioning`` is present, offsets the base seed by
    ``node_rank * seeds_per_node``.

    Returns None when neither base_seed nor seeds are specified (the
    generator will auto-generate random seeds).
    """
    if "seeds" in plan and plan["seeds"] is not None:
        return list(plan["seeds"])

    seeds_per_combo = plan.get("seeds_per_combo")
    if seeds_per_combo is None:
        return None

    # base_seed can come from the plan directly or from node_partitioning
    base_seed = plan.get("base_seed")
    partitioning = plan.get("node_partitioning")
    if base_seed is None and partitioning:
        base_seed = partitioning.get("base_seed")

    if base_seed is None:
        return None

    # Apply node partitioning offset
    if partitioning and node_rank > 0:
        seeds_per_node = partitioning.get("seeds_per_node", seeds_per_combo)
        base_seed = base_seed + node_rank * seeds_per_node

    # Generate sequential seeds
    return list(range(base_seed, base_seed + seeds_per_combo))


def _plan_to_config(
    plan: dict[str, Any],
    prompts: list[str],
    seeds: list[int] | None,
    sparse_steps: list[int] | None,
) -> DatasetGenerationConfig:
    """Convert resolved plan fields to a DatasetGenerationConfig."""
    cfg_dict: dict[str, Any] = {
        "prompts": prompts,
        "seeds": seeds,
    }
    if sparse_steps is not None:
        cfg_dict["sparse_steps"] = sparse_steps

    # Copy over fields that map directly
    direct_fields = [
        "resolution_tiers", "attention_backends", "n_steps", "cfg",
        "output_dir", "run_name", "base_model_hash", "sampling_shift",
        "multiplier", "server_endpoint", "flush_interval", "render_count",
        "source_device",
    ]
    for key in direct_fields:
        if key in plan and plan[key] is not None:
            cfg_dict[key] = plan[key]

    return DatasetGenerationConfig.from_dict(cfg_dict)


def _has_distribution_fields(plan: dict[str, Any]) -> bool:
    """Detect whether a plan uses distribution-valued fields.

    Returns True if any of:
      - ``n_steps`` is a list or dict (not a scalar)
      - ``resolution`` key exists (compound distribution)
      - ``cfg`` is a dict (distribution, not scalar)
    """
    from .config_distributions import is_distribution, is_enumeration

    n_steps = plan.get("n_steps")
    if n_steps is not None and (is_enumeration(n_steps) or is_distribution(n_steps)):
        return True
    if "resolution" in plan:
        return True
    cfg_val = plan.get("cfg")
    if cfg_val is not None and (is_enumeration(cfg_val) or is_distribution(cfg_val)):
        return True
    return False


def _parse_distribution_axes(
    plan_dict: dict[str, Any],
) -> dict[str, Any]:
    """Parse a distribution-valued plan dict into its axis structure.

    Returns a dict with all the parsed configuration needed by both
    ``_iter_distribution_tiles`` and ``distribution_plan_summary``.
    Pure computation, no side effects.
    """
    from .config_distributions import is_distribution, is_enumeration

    enum_axes: dict[str, list] = {}

    backends = plan_dict.get("attention_backends", ["sdpa", "sage"])
    if isinstance(backends, list):
        enum_axes["attention_backend"] = backends
    else:
        enum_axes["attention_backend"] = [backends]

    n_steps_spec = plan_dict.get("n_steps", 30)
    n_steps_is_enum = is_enumeration(n_steps_spec)
    n_steps_is_dist = is_distribution(n_steps_spec)
    if n_steps_is_enum:
        enum_axes["n_steps"] = list(n_steps_spec)
    elif not n_steps_is_dist:
        enum_axes["n_steps"] = [n_steps_spec]

    cfg_spec = plan_dict.get("cfg", 4.0)
    cfg_is_enum = is_enumeration(cfg_spec)
    cfg_is_dist = is_distribution(cfg_spec)
    if cfg_is_enum:
        enum_axes["cfg"] = list(cfg_spec)
    elif not cfg_is_dist:
        enum_axes["cfg"] = [cfg_spec]

    enum_keys = sorted(enum_axes.keys())
    enum_values = [enum_axes[k] for k in enum_keys]
    enum_combos = list(itertools.product(*enum_values))

    return {
        "enum_keys": enum_keys,
        "enum_combos": enum_combos,
        "n_steps_spec": n_steps_spec,
        "n_steps_is_enum": n_steps_is_enum,
        "n_steps_is_dist": n_steps_is_dist,
        "cfg_spec": cfg_spec,
        "cfg_is_dist": cfg_is_dist,
        "resolution_spec": plan_dict.get("resolution"),
        "sparse_steps_mode": plan_dict.get("sparse_steps_mode", "logsnr_uniform"),
        "n_save": plan_dict.get("n_save", 7),
    }


def _iter_distribution_tiles(
    plan_dict: dict[str, Any],
    prompts: list[str],
    base_seed: int,
    seeds_per_combo: int,
    node_rank: int = 0,
):
    """Yield comparison tiles from a distribution-valued plan.

    Each tile is a list of items sharing a single (seed, prompt, W, H).
    Items within a tile enumerate all combinations of the enumeration axes
    (n_steps, backend, etc.), ordered by (n_steps, backend) for bin packing.

    Tiles are yielded in (seed, prompt_idx) order. Every completed tile
    is a self-contained comparison set — spot preemption at any tile
    boundary yields a dataset where every persisted seed has full
    scrimblo + scrongle data.

    Memory: O(tile_size) per yield, not O(total_items).

    Yields:
        list[dict]: One tile's worth of plan items (all same seq_len).
    """
    from .config_distributions import resolve_scalar, resolve_resolution

    axes = _parse_distribution_axes(plan_dict)
    enum_keys = axes["enum_keys"]
    enum_combos = axes["enum_combos"]
    n_steps_spec = axes["n_steps_spec"]
    n_steps_is_dist = axes["n_steps_is_dist"]
    cfg_spec = axes["cfg_spec"]
    cfg_is_dist = axes["cfg_is_dist"]
    resolution_spec = axes["resolution_spec"]
    sparse_steps_mode = axes["sparse_steps_mode"]
    n_save = axes["n_save"]

    # Node partitioning
    partitioning = plan_dict.get("node_partitioning")
    effective_base_seed = base_seed
    if partitioning and node_rank > 0:
        seeds_per_node = partitioning.get("seeds_per_node", seeds_per_combo)
        effective_base_seed = base_seed + node_rank * seeds_per_node

    # Pre-import for logsnr_uniform mode (avoid import per-item in inner loop)
    _compute_logsnr = None
    if sparse_steps_mode == "logsnr_uniform":
        from .sigma_schedule import compute_logsnr_uniform_steps
        _compute_logsnr = compute_logsnr_uniform_steps

    for seed_offset in range(seeds_per_combo):
        seed = effective_base_seed + seed_offset
        for prompt_idx, prompt in enumerate(prompts):
            rng = random.Random(hash((seed, prompt_idx)))

            if resolution_spec is not None:
                w, h = resolve_resolution(resolution_spec, rng)
            else:
                w, h = 1024, 1024

            sampled_n_steps = None
            if n_steps_is_dist:
                sampled_n_steps = resolve_scalar(n_steps_spec, rng)

            sampled_cfg = None
            if cfg_is_dist:
                sampled_cfg = resolve_scalar(cfg_spec, rng)

            tile: list[dict[str, Any]] = []
            seq_len = compute_seq_len(w, h)

            for combo in enum_combos:
                combo_dict = dict(zip(enum_keys, combo))

                item_n_steps = sampled_n_steps if sampled_n_steps is not None else combo_dict.get("n_steps", 30)
                item_cfg = sampled_cfg if sampled_cfg is not None else combo_dict.get("cfg", 4.0)
                item_backend = combo_dict.get("attention_backend", "sdpa")

                if sparse_steps_mode == "logsnr_uniform":
                    sparse_steps = _compute_logsnr(w, h, item_n_steps, n_save)
                elif sparse_steps_mode == "step_uniform":
                    if n_save >= item_n_steps:
                        sparse_steps = list(range(item_n_steps))
                    else:
                        sparse_steps = [round(i * (item_n_steps - 1) / (n_save - 1)) for i in range(n_save)]
                else:
                    sparse_steps = plan_dict.get("sparse_steps", [0, 4, 9, 14, 19, 24, 29])

                tile.append({
                    "prompt": prompt,
                    "prompt_idx": prompt_idx,
                    "seed": seed,
                    "width": w,
                    "height": h,
                    "n_steps": item_n_steps,
                    "cfg": item_cfg,
                    "attention_backend": item_backend,
                    "sparse_steps": sparse_steps,
                    "batch_type": "t2i",
                    "seq_len": seq_len,
                })

            # Sort within tile: group by n_steps for bin packing
            tile.sort(key=lambda x: (x["n_steps"], x["attention_backend"]))
            yield tile


def _pack_tile(
    tile: list[dict[str, Any]],
    max_total_len: int = REFERENCE_TOTAL_LEN,
    cap_tokens: int = DEFAULT_CAP_TOKENS,
) -> list[list[dict[str, Any]]]:
    """Pack a single tile into bins (no cross-tile buffering).

    All items in a tile share the same (W, H), so the same effective seq_len.
    Groups by n_steps, chunks by capacity. For cross-tile packing that fills
    bins across multiple tiles, use ``TilePacker`` instead.

    Args:
        tile: List of items from one (seed, prompt) group, all same resolution.
        max_total_len: Bin capacity in tokens (image + text).
        cap_tokens: Caption token count for effective seq_len computation.

    Returns:
        List of bins (each bin = list of items).
    """
    if not tile:
        return []

    w, h = tile[0]["width"], tile[0]["height"]
    effective = compute_effective_seq_len(w, h, cap_tokens)
    items_per_bin = max(1, max_total_len // effective)

    bins: list[list[dict[str, Any]]] = []
    by_steps: dict[int, list[dict[str, Any]]] = {}
    for item in tile:
        by_steps.setdefault(item["n_steps"], []).append(item)

    for n_steps in sorted(by_steps.keys()):
        group = by_steps[n_steps]
        for i in range(0, len(group), items_per_bin):
            bins.append(group[i:i + items_per_bin])

    return bins


class TilePacker:
    """Cross-tile bin packer that buffers items from consecutive tiles.

    Items from different (seed, prompt) tiles that share the same
    (n_steps, effective_seq_len) can share a FlexAttention bin. This class
    accumulates items keyed by (n_steps, effective_seq_len) and emits full
    bins as soon as the buffer reaches capacity.

    Tiles are still processed in (seed, prompt) order — the packer just
    allows small-resolution items from consecutive tiles to fill bins that
    would otherwise run at 2/3 or 2/8 capacity.

    Usage::

        packer = TilePacker()
        for tile in tile_iterator:
            yield from packer.add_tile(tile)
        yield from packer.flush()

    Memory: O(items_per_bin * n_active_groups). A group is (n_steps, eff_len).
    With 9 step counts and ~6 resolutions, that's ~54 groups, each holding
    at most items_per_bin items. Worst case ~500 items buffered.
    """

    def __init__(
        self,
        max_total_len: int = REFERENCE_TOTAL_LEN,
        cap_tokens: int = DEFAULT_CAP_TOKENS,
    ):
        self.max_total_len = max_total_len
        self.cap_tokens = cap_tokens
        # Buffers keyed by (n_steps, effective_seq_len)
        self._buffers: dict[tuple[int, int], list[dict[str, Any]]] = {}
        # Cache: effective_seq_len -> items_per_bin
        self._capacity_cache: dict[int, int] = {}

    def _items_per_bin(self, effective: int) -> int:
        if effective not in self._capacity_cache:
            self._capacity_cache[effective] = max(1, self.max_total_len // effective)
        return self._capacity_cache[effective]

    def add_tile(self, tile: list[dict[str, Any]]):
        """Add a tile's items to the buffer. Yields full bins as they fill.

        Args:
            tile: All items from one (seed, prompt) group, same (W, H).

        Yields:
            list[dict]: Complete bins ready for generation.
        """
        if not tile:
            return

        w, h = tile[0]["width"], tile[0]["height"]
        effective = compute_effective_seq_len(w, h, self.cap_tokens)
        capacity = self._items_per_bin(effective)

        # Group tile items by n_steps
        by_steps: dict[int, list[dict[str, Any]]] = {}
        for item in tile:
            by_steps.setdefault(item["n_steps"], []).append(item)

        for n_steps, group in by_steps.items():
            key = (n_steps, effective)
            buf = self._buffers.setdefault(key, [])
            for item in group:
                buf.append(item)
                if len(buf) >= capacity:
                    yield buf[:capacity]
                    self._buffers[key] = buf[capacity:]
                    buf = self._buffers[key]

    def flush(self):
        """Yield all remaining buffered items as partial bins.

        Call this after all tiles have been added.

        Yields:
            list[dict]: Remaining bins (may be smaller than capacity).
        """
        for key in sorted(self._buffers.keys()):
            buf = self._buffers[key]
            if buf:
                yield buf
        self._buffers.clear()


def distribution_plan_summary(
    plan_dict: dict[str, Any],
    prompts: list[str],
    base_seed: int,
    seeds_per_combo: int,
    node_rank: int = 0,
) -> dict[str, Any]:
    """Compute summary statistics for a distribution-valued plan without
    materializing the full item list.

    Iterates tiles and accumulates counts. Memory: O(unique_resolutions +
    unique_steps + unique_backends), not O(total_items).

    Returns:
        Dict with total_items, tile_size, unique counts, anchor distribution, etc.
    """
    from collections import Counter
    from .resolution_sampling import assign_budget_tier, ANCHOR_LABELS

    axes = _parse_distribution_axes(plan_dict)
    tile_size = len(axes["enum_combos"])
    n_tiles = seeds_per_combo * len(prompts)

    total_items = 0
    step_counts: Counter = Counter()
    backend_counts: Counter = Counter()
    cfg_counts: Counter = Counter()
    res_counts: Counter = Counter()
    anchor_counts: Counter = Counter()
    seeds: set[int] = set()

    for tile in _iter_distribution_tiles(plan_dict, prompts, base_seed,
                                          seeds_per_combo, node_rank):
        total_items += len(tile)
        for item in tile:
            step_counts[item["n_steps"]] += 1
            backend_counts[item["attention_backend"]] += 1
            cfg_counts[item["cfg"]] += 1
            res_key = f"{item['width']}x{item['height']}"
            res_counts[res_key] += 1
            anchor_counts[assign_budget_tier(item["width"], item["height"])] += 1
            seeds.add(item["seed"])

    return {
        "total_items": total_items,
        "tile_size": tile_size,
        "n_tiles": n_tiles,
        "n_unique_seeds": len(seeds),
        "seed_range": (min(seeds), max(seeds)) if seeds else (0, 0),
        "step_counts": dict(step_counts),
        "backend_counts": dict(backend_counts),
        "cfg_counts": dict(cfg_counts),
        "n_unique_resolutions": len(res_counts),
        "anchor_counts": {
            ANCHOR_LABELS.get(a, f"{a}px"): c
            for a, c in sorted(anchor_counts.items())
        },
    }


def load_plan(
    path: str | Path,
    node_rank: int = 0,
) -> list[DatasetGenerationConfig]:
    """Load a generation plan from a JSON file.

    Returns a list of DatasetGenerationConfig objects — one per phase if
    the plan has a ``phases`` array, otherwise a single-element list.

    Args:
        path: Path to the JSON plan file.
        node_rank: Node rank for seed partitioning (default 0).

    Returns:
        List of DatasetGenerationConfig, one per phase.
    """
    path = Path(path)
    plan_dir = path.resolve().parent
    with open(path, encoding="utf-8") as f:
        plan = json.load(f)

    prompts = _resolve_prompts(plan, plan_dir=plan_dir)

    # --- Distribution-valued plan: store spec for lazy tile iteration ---
    if _has_distribution_fields(plan):
        # Resolve node partitioning into the spec for the generator
        plan_spec = dict(plan)
        plan_spec["_prompts"] = prompts
        plan_spec["_node_rank"] = node_rank

        base_seed = plan.get("base_seed")
        partitioning = plan.get("node_partitioning")
        if base_seed is None and partitioning:
            base_seed = partitioning.get("base_seed")
        if base_seed is None:
            base_seed = 42
        plan_spec["_base_seed"] = base_seed
        plan_spec["_seeds_per_combo"] = plan.get("seeds_per_combo", 1)

        cfg_dict: dict[str, Any] = {"prompts": prompts, "plan_spec": plan_spec}
        for key in ["output_dir", "run_name", "base_model_hash", "sampling_shift",
                     "multiplier", "server_endpoint", "flush_interval", "render_count",
                     "source_device"]:
            if key in plan and plan[key] is not None:
                cfg_dict[key] = plan[key]

        # Partition output directory by node rank to avoid DatasetWriter
        # lock contention. Each node writes to its own subdirectory;
        # datasets are merged after all nodes complete.
        if partitioning:
            base_dir = cfg_dict.get("output_dir", "btrm_dataset_v2")
            cfg_dict["output_dir"] = f"{base_dir}/node_{node_rank}"

        return [DatasetGenerationConfig.from_dict(cfg_dict)]

    # --- Legacy plan format: phases or single-phase with resolution_tiers ---
    phases = plan.get("phases")

    if phases:
        # Multi-phase plan: each phase inherits top-level fields,
        # then overrides with phase-specific fields.
        configs = []
        for phase in phases:
            merged = {k: v for k, v in plan.items() if k != "phases"}
            merged.update(phase)

            n_steps = merged.get("n_steps", 30)
            sparse_steps = _resolve_sparse_steps(merged, n_steps)

            resolution_tiers = merged.get("resolution_tiers", ["full"])
            attention_backends = merged.get("attention_backends", ["sdpa", "sage"])
            seeds = _resolve_seeds(
                merged, len(prompts), resolution_tiers, attention_backends,
                node_rank=node_rank,
            )

            cfg = _plan_to_config(merged, prompts, seeds, sparse_steps)
            configs.append(cfg)
        return configs
    else:
        # Single-phase plan
        n_steps = plan.get("n_steps", 30)
        sparse_steps = _resolve_sparse_steps(plan, n_steps)

        resolution_tiers = plan.get("resolution_tiers", ["full"])
        attention_backends = plan.get("attention_backends", ["sdpa", "sage"])
        seeds = _resolve_seeds(
            plan, len(prompts), resolution_tiers, attention_backends,
            node_rank=node_rank,
        )

        cfg = _plan_to_config(plan, prompts, seeds, sparse_steps)
        return [cfg]


def save_plan(config: DatasetGenerationConfig, path: str | Path) -> None:
    """Save a DatasetGenerationConfig as a JSON plan file.

    Args:
        config: Configuration to serialize.
        path: Output path for the JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
        f.write("\n")


# ---------------------------------------------------------------------------
# DatasetGenerator
# ---------------------------------------------------------------------------

class DatasetGenerator:
    """Generates BTRM training trajectories and writes to V2 format.

    Lifecycle:
      1. __init__: accept config and client
      2. generate_all(): run the full generation plan
         - Phase 1: Build generation plan (Cartesian product of params)
         - Phase 2: Bin-pack plan items into FlexAttention batches
         - Phase 3: Encode all unique prompts via server
         - Phase 4: Warmup diffusion model for each (backend, resolution) combo
         - Phase 5: Execute bins sequentially, writing V2 after each trajectory
         - Phase 6: Render a sample of trajectories
         - Phase 7: Print throughput profile

    The generator does NOT import torch at class level or call any model code.
    All model interaction happens through InferenceClient RPCs.
    """

    def __init__(self, config: DatasetGenerationConfig, client):
        """Initialize the generator.

        Args:
            config: Generation configuration.
            client: An InferenceClient instance (or compatible interface).
                Must support: encode_prompt, sample_trajectory, free, warmup,
                warmup_packed, status, vae_decode.
        """
        self.config = config
        self.client = client
        self._interrupted = False

        # Compute identity hashes once (they don't change during a run)
        adapter_set_hash = compute_adapter_set_hash(config.active_adapters)
        self._model_state_hash = compute_model_state_hash(
            config.base_model_hash, adapter_set_hash,
        )
        self._adapter_set_hash = adapter_set_hash
        self._active_adapters_json = serialize_active_adapters(config.active_adapters)

    def _install_signal_handler(self):
        """Install a SIGINT handler that sets the interrupted flag on first press."""
        def _handler(signum, frame):
            if self._interrupted:
                print("\nForce quit.")
                sys.exit(1)
            self._interrupted = True
            print("\nInterrupt received. Finishing current trajectory then saving...")

        signal.signal(signal.SIGINT, _handler)

    def generate_all(self) -> dict[str, Any]:
        """Run the full generation pipeline.

        Returns:
            Summary dict with timing, counts, and efficiency stats.
        """
        self._install_signal_handler()

        cfg = self.config
        output_dir = Path(cfg.output_dir)

        # ---------------------------------------------------------------
        # Phase 1+2: Build generation plan / tile iterator
        # ---------------------------------------------------------------
        use_tile_iterator = cfg.plan_spec is not None

        if use_tile_iterator:
            spec = cfg.plan_spec
            summary = distribution_plan_summary(
                spec, spec["_prompts"], spec["_base_seed"],
                spec["_seeds_per_combo"], spec["_node_rank"],
            )
            print(f"Generation plan: {summary['total_items']} trajectories "
                  f"({summary['n_tiles']} tiles x {summary['tile_size']}/tile)")
            print(f"  Prompts: {len(cfg.prompts)}")
            print(f"  Steps: {sorted(summary['step_counts'].keys())}")
            print(f"  Backends: {sorted(summary['backend_counts'].keys())}")
            print(f"  Resolutions: {summary['n_unique_resolutions']} unique (W,H)")
            print(f"  Output: {output_dir}")
        else:
            seeds = cfg.seeds
            if seeds is None:
                rng = random.Random(42)
                max_rollouts = sum(
                    RESOLUTION_TIERS[t]["rollouts_per_prompt"]
                    for t in cfg.resolution_tiers
                )
                n_seeds_needed = len(cfg.prompts) * max_rollouts * len(cfg.attention_backends)
                seeds = [rng.randint(0, 2**32 - 1) for _ in range(n_seeds_needed)]

            plan = build_generation_plan(
                prompts=cfg.prompts,
                seeds=seeds,
                resolution_tiers=cfg.resolution_tiers,
                attention_backends=cfg.attention_backends,
                n_steps=cfg.n_steps,
                cfg=cfg.cfg,
                sparse_steps=cfg.sparse_steps,
            )

            print(f"Generation plan: {len(plan)} trajectories")
            print(f"  Prompts: {len(cfg.prompts)}")
            print(f"  Resolution tiers: {cfg.resolution_tiers}")
            print(f"  Attention backends: {cfg.attention_backends}")
            print(f"  Steps: {cfg.n_steps}, CFG: {cfg.cfg}")
            print(f"  Output: {output_dir}")

            # Legacy path: global bin packing
            scheduler = BinPackScheduler(max_seq_len=REFERENCE_SEQ_LEN)
            bins = scheduler.pack_generation_plan(plan)
            efficiency = scheduler.estimate_efficiency(bins)
            print(f"\nBin packing: {efficiency['n_items']} items -> {efficiency['n_bins']} bins")
            print(f"  Utilization: {efficiency['utilization']:.1%}")
            print(f"  Wasted seq_len: {efficiency['total_wasted']}")

        # ---------------------------------------------------------------
        # Phase 3: Encode prompts
        # ---------------------------------------------------------------
        print(f"\nEncoding {len(cfg.prompts)} unique prompts + 1 negative...")

        timing = {
            "te_encode": 0.0,
            "warmup": 0.0,
            "diffusion": 0.0,
            "vae_decode": 0.0,
            "overhead": 0.0,
        }
        wall_start = time.perf_counter()

        t0 = time.perf_counter()
        neg_cond = self.client.encode_prompt("")
        te_cache: dict[str, torch.Tensor] = {}
        for prompt in cfg.prompts:
            if prompt not in te_cache:
                te_cache[prompt] = self.client.encode_prompt(prompt)
        timing["te_encode"] = time.perf_counter() - t0
        print(f"  Done ({timing['te_encode']:.1f}s)")

        # Free TE on server (diffusion model will be loaded on next call)
        self.client.free("te")

        # ---------------------------------------------------------------
        # Phase 4: Warmup (lazy for tile iterator, eager for legacy)
        # ---------------------------------------------------------------
        warmed_up: set[tuple[str, int, int]] = set()
        warmed_packed: set[int] = set()

        def _ensure_warmup(items: list[dict]) -> None:
            """Warmup any new (backend, W, H) or packed sizes on first encounter."""
            for item in items:
                key = (item["attention_backend"], item["width"], item["height"])
                if key not in warmed_up:
                    warmed_up.add(key)
                    print(f"  Warming up {key[0]} @ {key[1]}x{key[2]}...")
                    t0 = time.perf_counter()
                    self.client.warmup(attention_backend=key[0], width=key[1], height=key[2])
                    timing["warmup"] += time.perf_counter() - t0
            if len(items) > 1 and len(items) not in warmed_packed:
                warmed_packed.add(len(items))
                print(f"  Warming up packed forward (n_images={len(items)})...")
                t0 = time.perf_counter()
                self.client.warmup_packed(n_images=len(items))
                timing["warmup"] += time.perf_counter() - t0

        if not use_tile_iterator:
            # Eager warmup for legacy path
            warmup_configs: set[tuple[str, int, int]] = set()
            for item in plan:
                warmup_configs.add((item["attention_backend"], item["width"], item["height"]))
            for backend, w, h in sorted(warmup_configs):
                _ensure_warmup([{"attention_backend": backend, "width": w, "height": h}])
            bin_sizes = sorted(set(len(b) for b in bins))
            for ps in [s for s in bin_sizes if s > 1]:
                if ps not in warmed_packed:
                    warmed_packed.add(ps)
                    print(f"  Warming up packed forward (n_images={ps})...")
                    t0 = time.perf_counter()
                    self.client.warmup_packed(n_images=ps)
                    timing["warmup"] += time.perf_counter() - t0

        # ---------------------------------------------------------------
        # Phase 5: Generate trajectories
        # ---------------------------------------------------------------
        from futudiffu.dataset_v2 import DatasetWriter

        render_queue: list[tuple[torch.Tensor, Path]] = []
        generated_count = 0
        renders_done = 0

        def _generate_bins(bin_iter, writer):
            """Generate from an iterable of bins. Returns count generated."""
            nonlocal generated_count
            for bin_items in bin_iter:
                if self._interrupted:
                    break

                _ensure_warmup(bin_items)

                if len(bin_items) == 1:
                    generated_count += self._generate_single(
                        bin_items[0], te_cache, neg_cond, writer, timing,
                    )
                else:
                    # Group by n_steps (FlexAttention lockstep constraint).
                    # _pack_tile already ensures same n_steps per bin, but
                    # legacy bins may mix.
                    by_steps: dict[int, list[dict]] = {}
                    for item in bin_items:
                        by_steps.setdefault(item["n_steps"], []).append(item)
                    for _, group_items in sorted(by_steps.items()):
                        if self._interrupted:
                            break
                        generated_count += self._generate_packed(
                            group_items, te_cache, neg_cond, writer, timing,
                        )

                if generated_count > 0 and generated_count % cfg.flush_interval == 0:
                    writer.flush()
                    print(f"  [flush] {generated_count} trajectories sealed")

        with DatasetWriter(output_dir) as writer:
            if use_tile_iterator:
                # Streaming: iterate tiles, pack per-tile, generate per-bin
                tile_iter = _iter_distribution_tiles(
                    spec, spec["_prompts"], spec["_base_seed"],
                    spec["_seeds_per_combo"], spec["_node_rank"],
                )
                packer = TilePacker()
                def _bins_from_tiles():
                    for tile in tile_iter:
                        yield from packer.add_tile(tile)
                    yield from packer.flush()
                _generate_bins(_bins_from_tiles(), writer)
                # On interrupt, the generator is abandoned before packer.flush().
                # Partial bins hold plan items that were never generated — no
                # data loss. On resume, the tile iterator replays from the start;
                # already-written trajectories are harmless duplicates
                # (deterministic generation, same identity hash).
            else:
                # Legacy: pre-packed bins
                _generate_bins(bins, writer)

            # Final summary
            print(f"\nGenerated {generated_count} trajectories")
            print(f"Total in dataset: {writer.n_trajectories}")

        # ---------------------------------------------------------------
        # Phase 6: Render sample (skipped if no renders queued)
        # ---------------------------------------------------------------
        if render_queue and not self._interrupted:
            render_dir = output_dir / "renders"
            render_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nRendering {len(render_queue)} trajectories...")
            for final_tensor, render_path in render_queue:
                t0 = time.perf_counter()
                self._render_latent(final_tensor, render_path)
                timing["vae_decode"] += time.perf_counter() - t0

        # ---------------------------------------------------------------
        # Phase 7: Throughput profile
        # ---------------------------------------------------------------
        wall_total = time.perf_counter() - wall_start
        timing["overhead"] = wall_total - sum(timing.values())

        result = self._print_throughput_profile(timing, wall_total, generated_count)
        if not use_tile_iterator:
            result["n_bins"] = len(bins)
            result["packing_efficiency"] = efficiency
        return result

    # -------------------------------------------------------------------
    # Internal generation methods
    # -------------------------------------------------------------------

    def _generate_single(
        self,
        item: dict,
        te_cache: dict[str, torch.Tensor],
        neg_cond: torch.Tensor,
        writer,
        timing: dict,
    ) -> int:
        """Generate a single trajectory (no packing).

        Returns:
            1 if generated, 0 if interrupted.
        """
        if self._interrupted:
            return 0

        cfg = self.config
        prompt = item["prompt"]
        w, h = item["width"], item["height"]

        # Resolve per-item sampling_shift: explicit override or auto from resolution
        if cfg.sampling_shift is not None:
            item_shift = cfg.sampling_shift
        else:
            item_shift = resolution_shift(w, h)

        t0 = time.perf_counter()
        result = self.client.sample_trajectory(
            pos_cond=te_cache[prompt],
            neg_cond=neg_cond,
            seed=item["seed"],
            n_steps=item["n_steps"],
            cfg=item["cfg"],
            width=w,
            height=h,
            attention_backend=item["attention_backend"],
            sampling_shift=item_shift,
            multiplier=cfg.multiplier,
            save_steps=item["sparse_steps"],
        )
        dt = time.perf_counter() - t0
        timing["diffusion"] += dt

        # Compute trajectory hash
        traj_hash = compute_trajectory_hash(
            self._model_state_hash,
            prompt=prompt,
            seed=item["seed"],
            cfg=item["cfg"],
            n_steps=item["n_steps"],
            width=w,
            height=h,
        )

        metadata = self._build_metadata(item, packed=False, timing_seconds=dt,
                                         trajectory_hash=traj_hash,
                                         sampling_shift=item_shift)
        writer.add_trajectory(tensors=result, metadata=metadata)

        label = f"{w}x{h} {item['attention_backend']}"
        print(f"  traj {writer.n_trajectories - 1:06d}: {label} ({dt:.1f}s)")

        return 1

    def _generate_packed(
        self,
        items: list[dict],
        te_cache: dict[str, torch.Tensor],
        neg_cond: torch.Tensor,
        writer,
        timing: dict,
    ) -> int:
        """Generate a packed batch of trajectories via FlexAttention.

        All items must share the same n_steps. They may differ in resolution,
        prompt, seed, and attention_backend. Mixed-resolution bins use
        per-image widths/heights and per-image sigma schedule shifts.

        Returns:
            Number of trajectories generated.
        """
        if self._interrupted or not items:
            return 0

        n_steps = items[0]["n_steps"]

        # Compute per-item shifts
        per_item_shifts = []
        for item in items:
            if self.config.sampling_shift is not None:
                per_item_shifts.append(self.config.sampling_shift)
            else:
                per_item_shifts.append(resolution_shift(item["width"], item["height"]))

        pos_conds = [te_cache[item["prompt"]] for item in items]
        seeds = [item["seed"] for item in items]
        widths = [item["width"] for item in items]
        heights = [item["height"] for item in items]

        # Use the first item's attention_backend for the batch
        # (packed forward currently uses a single backend for all items)
        backend = items[0]["attention_backend"]

        t0 = time.perf_counter()
        results = self.client.sample_trajectory_packed(
            pos_conds=pos_conds,
            neg_cond=neg_cond,
            seeds=seeds,
            n_steps=n_steps,
            cfg=items[0]["cfg"],
            widths=widths,
            heights=heights,
            attention_backend=backend,
            sampling_shifts=per_item_shifts,
            multiplier=self.config.multiplier,
            save_steps=items[0]["sparse_steps"],
        )
        dt = time.perf_counter() - t0
        timing["diffusion"] += dt

        dt_per_image = dt / len(items)

        for i, (item, result) in enumerate(zip(items, results)):
            traj_hash = compute_trajectory_hash(
                self._model_state_hash,
                prompt=item["prompt"],
                seed=item["seed"],
                cfg=item["cfg"],
                n_steps=item["n_steps"],
                width=item["width"],
                height=item["height"],
            )

            metadata = self._build_metadata(
                item, packed=True, timing_seconds=dt_per_image,
                trajectory_hash=traj_hash,
                sampling_shift=per_item_shifts[i],
            )
            writer.add_trajectory(tensors=result, metadata=metadata)

        idxs_start = writer.n_trajectories - len(items)
        unique_res = set(zip(widths, heights))
        if len(unique_res) == 1:
            w0, h0 = next(iter(unique_res))
            res_label = f"{w0}x{h0}"
        else:
            res_label = "mixed-res"
        label = f"{len(items)}x {res_label} {backend}"
        print(f"  traj {idxs_start:06d}..{writer.n_trajectories - 1:06d}: "
              f"packed {label} ({dt:.1f}s, {dt_per_image:.1f}s/img)")

        return len(items)

    def _build_metadata(
        self,
        item: dict,
        packed: bool,
        timing_seconds: float,
        trajectory_hash: str,
        sampling_shift: float = 1.0,
    ) -> dict[str, Any]:
        """Build V2 metadata dict for one trajectory.

        Args:
            item: Generation plan item dict.
            packed: Whether this was part of a packed batch.
            timing_seconds: Wall time for this trajectory.
            trajectory_hash: Full trajectory identity hash.
            sampling_shift: The sigma schedule shift used for this trajectory.

        Returns:
            Metadata dict compatible with DatasetWriter.add_trajectory().
        """
        cfg = self.config
        return {
            "prompt": item["prompt"],
            "prompt_idx": item.get("prompt_idx", -1),
            "seed": item["seed"],
            "cfg": item["cfg"],
            "width": item["width"],
            "height": item["height"],
            "n_steps": item["n_steps"],
            "attention_backend": item["attention_backend"],
            "batch_type": item.get("batch_type", "t2i"),
            "denoise": item.get("denoise"),
            "image_file": item.get("image_file"),
            "is_gold": (
                item["attention_backend"] == "sdpa"
                and item["n_steps"] == 30
            ),
            "batch_idx": 0,
            "packed": packed,
            "timing_seconds": timing_seconds,
            "sampling_shift": sampling_shift,
            # Provenance / identity hashes
            "model_state_hash": self._model_state_hash,
            "base_model_hash": cfg.base_model_hash,
            "adapter_set_hash": self._adapter_set_hash,
            "trajectory_hash": trajectory_hash,
            "active_adapters": self._active_adapters_json,
        }

    def _render_latent(self, latent: torch.Tensor, output_path: Path) -> None:
        """VAE-decode a latent and save as PNG.

        Args:
            latent: (1, 16, H, W) latent tensor on CPU.
            output_path: Path for the output PNG file.
        """
        from src_ii.rendering import save_tensor_as_png

        output_path.parent.mkdir(parents=True, exist_ok=True)
        # client.vae_decode returns (1, 3, H*8, W*8) in [0, 1]
        image = self.client.vae_decode(latent)
        save_tensor_as_png(image, output_path)

    def _print_throughput_profile(
        self,
        timing: dict[str, float],
        wall_total: float,
        generated_count: int,
    ) -> dict[str, Any]:
        """Print throughput summary and return summary dict."""
        print(f"\n{'=' * 60}")
        print(f"  THROUGHPUT PROFILE ({generated_count} trajectories)")
        print(f"{'=' * 60}")
        print(f"  {'Phase':<20} {'Time (s)':>10} {'%':>6}")
        print(f"  {'-' * 38}")
        for phase, t in sorted(timing.items(), key=lambda x: -x[1]):
            if t > 0.01:
                print(f"  {phase:<20} {t:>10.1f} {100 * t / wall_total:>5.1f}%")
        print(f"  {'-' * 38}")
        print(f"  {'TOTAL':<20} {wall_total:>10.1f}")

        summary: dict[str, Any] = {
            "generated_count": generated_count,
            "wall_total": wall_total,
            "timing": dict(timing),
        }

        if generated_count > 0:
            avg = timing["diffusion"] / generated_count
            imgs_per_min = 60.0 / (wall_total / generated_count)
            gpu_active = timing["diffusion"] + timing["vae_decode"] + timing["te_encode"]
            print(f"\n  Avg diffusion: {avg:.1f}s/trajectory")
            print(f"  Throughput:    {imgs_per_min:.2f} images/min")
            print(f"  GPU util:      {100 * gpu_active / wall_total:.1f}%")
            summary["avg_diffusion_s"] = avg
            summary["throughput_imgs_per_min"] = imgs_per_min
            summary["gpu_utilization"] = gpu_active / wall_total

        print(f"{'=' * 60}")
        return summary

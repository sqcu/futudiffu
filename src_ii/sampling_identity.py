"""Sampling state hash computation for trajectory identity.

Provides deterministic hash functions that uniquely identify:
  - A model state (base model + active adapter set)
  - A specific sampling trajectory (model state + sampling parameters)

Hash design:
  - All hashes use SHA-256
  - Short form: first 16 hex chars (for display, filenames)
  - Full form: all 64 hex chars (for storage in parquet, collision resistance)
  - Adapter set hash is ORDER-INDEPENDENT (frozenset semantics)
  - Adapters with strength=0 are excluded (they don't affect output)

Why this matters:
  When a useful policy adapter is distilled into the base model
  (materialized: base_new = base_old + adapter_A @ adapter_B), the
  base_model_hash changes. Multiple "versions" of the base model will
  exist, each with a different hash. The trajectory_hash captures the
  full identity: which exact model state produced this specific trajectory.

Usage:
    from src_ii.sampling_identity import (
        compute_adapter_param_hash,
        compute_adapter_set_hash,
        compute_model_state_hash,
        compute_trajectory_hash,
    )

    # Hash a single adapter's parameters
    param_hash = compute_adapter_param_hash(adapter.state_dict())

    # Hash the full adapter set (order-independent, excludes strength=0)
    adapters = [
        {"name": "rtheta", "strength": 1.0, "param_hash": param_hash},
    ]
    set_hash = compute_adapter_set_hash(adapters)

    # Combine with base model identity
    model_hash = compute_model_state_hash("z_image_v1", set_hash)

    # Full trajectory identity
    traj_hash = compute_trajectory_hash(
        model_hash, "a photo of a cat", seed=42,
        cfg=4.0, n_steps=30, width=1280, height=832,
    )
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
from torch import Tensor


def compute_adapter_param_hash(adapter_state_dict: dict[str, Tensor]) -> str:
    """Hash adapter parameters using SHA-256 of sorted key-value pairs.

    The hash is computed over the concatenated bytes of all parameter tensors,
    sorted by key name for determinism. Each tensor is converted to contiguous
    bytes on CPU.

    Args:
        adapter_state_dict: Mapping of parameter names to tensors.
            Typically from adapter.state_dict() or a filtered subset.

    Returns:
        Full 64-char hex digest string.
    """
    h = hashlib.sha256()
    for key in sorted(adapter_state_dict.keys()):
        t = adapter_state_dict[key]
        # Ensure CPU, contiguous for stable byte representation
        t_cpu = t.detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(t_cpu.numpy().tobytes())
    return h.hexdigest()


def compute_adapter_set_hash(active_adapters: list[dict[str, Any]]) -> str:
    """Hash an unordered set of (strength, param_hash) pairs.

    Adapters with strength=0 are excluded -- they don't affect model output.
    The hash is order-independent: the adapter list is converted to a
    frozenset of (strength, param_hash) tuples before hashing.

    An empty adapter set (or all adapters at strength=0) returns the
    empty string "", which is the canonical "no adapters" value.

    Args:
        active_adapters: List of dicts, each with keys:
            "name" (str): Adapter name (not included in hash, only for display).
            "strength" (float): Adapter strength/scale. 0 means disabled.
            "param_hash" (str): Hash of the adapter's current parameters.

    Returns:
        Full 64-char hex digest, or "" if no active adapters.
    """
    # Filter out disabled adapters (strength == 0)
    active = [
        (float(a["strength"]), a["param_hash"])
        for a in active_adapters
        if float(a.get("strength", 0)) != 0.0
    ]

    if not active:
        return ""

    # Order-independent: sort the tuples for deterministic hashing
    # (frozenset doesn't have a canonical serialization, but sorted tuples do)
    active_sorted = sorted(active)

    h = hashlib.sha256()
    for strength, param_hash in active_sorted:
        # Use repr for float to ensure consistent string representation
        h.update(f"{strength!r}:{param_hash}".encode("utf-8"))
    return h.hexdigest()


def compute_model_state_hash(base_model_hash: str, adapter_set_hash: str) -> str:
    """Combine base model and adapter set hashes into a model state hash.

    Args:
        base_model_hash: Hash (or placeholder identifier) of the base model.
        adapter_set_hash: Hash of the active adapter set ("" for no adapters).

    Returns:
        Full 64-char hex digest.
    """
    h = hashlib.sha256()
    h.update(f"base:{base_model_hash}".encode("utf-8"))
    h.update(f"adapters:{adapter_set_hash}".encode("utf-8"))
    return h.hexdigest()


def compute_trajectory_hash(
    model_state_hash: str,
    prompt: str,
    seed: int,
    cfg: float,
    n_steps: int,
    width: int,
    height: int,
) -> str:
    """Compute the full trajectory identity hash.

    This uniquely identifies a specific diffusion sampling run: the exact
    model state combined with all sampling parameters that affect the output.

    Args:
        model_state_hash: From compute_model_state_hash().
        prompt: Full prompt text.
        seed: PRNG seed for noise generation.
        cfg: Classifier-free guidance scale.
        n_steps: Total diffusion steps.
        width: Output image width in pixels.
        height: Output image height in pixels.

    Returns:
        Full 64-char hex digest.
    """
    h = hashlib.sha256()
    h.update(f"model:{model_state_hash}".encode("utf-8"))
    h.update(f"prompt:{prompt}".encode("utf-8"))
    h.update(f"seed:{seed}".encode("utf-8"))
    h.update(f"cfg:{cfg!r}".encode("utf-8"))
    h.update(f"n_steps:{n_steps}".encode("utf-8"))
    h.update(f"width:{width}".encode("utf-8"))
    h.update(f"height:{height}".encode("utf-8"))
    return h.hexdigest()


def short_hash(full_hex: str) -> str:
    """Return the first 16 characters of a hex digest for display purposes.

    Args:
        full_hex: A 64-char hex digest string.

    Returns:
        First 16 characters of the hex digest.
    """
    return full_hex[:16]


def serialize_active_adapters(active_adapters: list[dict[str, Any]]) -> str:
    """Serialize active adapter list to JSON string for parquet storage.

    Filters out adapters with strength=0 before serialization.

    Args:
        active_adapters: List of adapter dicts with "name", "strength",
            "param_hash" keys.

    Returns:
        JSON string representation. "[]" for no active adapters.
    """
    filtered = [
        {"name": a["name"], "strength": float(a["strength"]), "param_hash": a["param_hash"]}
        for a in active_adapters
        if float(a.get("strength", 0)) != 0.0
    ]
    return json.dumps(filtered, sort_keys=True)

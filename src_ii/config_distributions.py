"""Distribution-valued configuration fields for generation plans.

A scalar is a point mass. An array is an enumeration axis (Cartesian product).
A dict with values/weights/min/max is a sampling axis (one draw per group).

Distribution spec syntax:
  30                                          -> point mass (fixed)
  [8, 10, 12, 14, 16, 18, 20, 22, 30]        -> set (enumerate, Cartesian)
  {"values": [8, 30], "weights": [0.7, 0.3]}  -> weighted categorical (sample)
  {"min": 8, "max": 22}                       -> uniform range (sample; int if both bounds int)
  {"min": 0.5, "max": 2.0, "distribution": "log_uniform"} -> log-uniform (sample)

Resolution is a compound distribution:
  {"megapixels": <spec>, "aspect_ratio": <spec>, "quantize": 32}
  Sub-fields are always sampled (one anchor, one aspect per draw).

Import constraints:
  - Pure Python + math. No torch, no numpy.
  - Reuses resolution_sampling.sample_resolution() for compound resolution.
"""

from __future__ import annotations

import math
import random


def is_distribution(value) -> bool:
    """True if value is a distribution object (dict with distribution keys)."""
    if not isinstance(value, dict):
        return False
    return bool({"values", "min", "max", "weights", "megapixels"} & value.keys())


def is_enumeration(value) -> bool:
    """True if value is a plain array (enumeration axis, Cartesian product)."""
    return isinstance(value, list)


def resolve_scalar(spec, rng: random.Random) -> int | float | str:
    """Sample one value from a distribution spec. Scalars pass through.

    Args:
        spec: A scalar, or a distribution dict.
        rng: Random instance for reproducibility.

    Returns:
        A single sampled value.
    """
    if not isinstance(spec, dict):
        return spec

    # Weighted categorical: {"values": [...], "weights": [...]}
    if "values" in spec and "weights" in spec:
        return rng.choices(spec["values"], weights=spec["weights"], k=1)[0]

    # Unweighted categorical (sample, not enumerate): {"values": [...]}
    if "values" in spec and "weights" not in spec:
        return rng.choice(spec["values"])

    # Range: {"min": ..., "max": ..., "distribution": "uniform"|"log_uniform"}
    if "min" in spec and "max" in spec:
        lo, hi = spec["min"], spec["max"]
        dist = spec.get("distribution", "uniform")

        if dist == "log_uniform":
            log_val = rng.uniform(math.log(lo), math.log(hi))
            val = math.exp(log_val)
            # Return int if both bounds are int
            if isinstance(lo, int) and isinstance(hi, int):
                return round(val)
            return val

        # uniform
        val = rng.uniform(lo, hi)
        if isinstance(lo, int) and isinstance(hi, int):
            return rng.randint(lo, hi)
        return val

    raise ValueError(f"Unrecognized distribution spec: {spec}")


def resolve_resolution(
    spec: dict,
    rng: random.Random,
) -> tuple[int, int]:
    """Sample (width, height) from a compound resolution distribution.

    Spec format:
      {"megapixels": <spec>, "aspect_ratio": <spec>, "quantize": 32}

    1. Resolve megapixels sub-field -> one anchor pixel count
    2. Resolve aspect_ratio sub-field -> one float
    3. Compute aligned (W, H) from budget and aspect

    Args:
        spec: Compound resolution distribution dict.
        rng: Random instance for reproducibility.

    Returns:
        (width, height) tuple, both multiples of quantize and >= 64.
    """
    from .resolution_sampling import sample_resolution

    mega_spec = spec.get("megapixels", 262144)
    aspect_spec = spec.get("aspect_ratio", 1.0)
    quantize = spec.get("quantize", 32)

    budget = resolve_scalar(mega_spec, rng)
    if isinstance(budget, float):
        budget = round(budget)

    aspect = resolve_scalar(aspect_spec, rng)
    if isinstance(aspect, str):
        raise ValueError(f"aspect_ratio resolved to string: {aspect!r}")

    return sample_resolution(budget, float(aspect), step=quantize)

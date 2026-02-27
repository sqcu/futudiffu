"""Distribution-valued configuration fields for generation plans.

A scalar is a point mass. An array is an enumeration axis (Cartesian product).
A dict with values/weights/min/max is a sampling axis (one draw per group).

Distribution spec syntax:
  30                                          -> point mass (fixed)
  [8, 10, 12, 14, 16, 18, 20, 22, 30]        -> set (enumerate, Cartesian)
  {"values": [8, 30], "weights": [0.7, 0.3]}  -> weighted categorical (sample)
  {"min": 8, "max": 22}                       -> uniform range (sample; int if both bounds int)
  {"min": 1.0, "max": 8.0, "step": 0.1}      -> stepped range (arange; finite cardinality)
  {"min": 0.5, "max": 2.0, "distribution": "log_uniform"} -> log-uniform (sample)
  {"min": 0.5, "max": 2.0, "step": 0.01, "distribution": "log_uniform"} -> log-uniform stepped

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

    # Range: {"min": ..., "max": ..., "step": ..., "distribution": "uniform"|"log_uniform"}
    if "min" in spec and "max" in spec:
        lo, hi = spec["min"], spec["max"]
        dist = spec.get("distribution", "uniform")
        step = spec.get("step")

        if dist == "log_uniform":
            log_val = rng.uniform(math.log(lo), math.log(hi))
            val = math.exp(log_val)
            if step is not None and step > 0:
                val = lo + round((val - lo) / step) * step
                return max(lo, min(hi, val))
            if isinstance(lo, int) and isinstance(hi, int):
                return round(val)
            return val

        # uniform
        if step is not None and step > 0:
            n_grid = max(0, int((hi - lo) / step))
            idx = rng.randint(0, n_grid)
            val = lo + idx * step
            if isinstance(lo, int) and isinstance(step, int):
                return int(val)
            return round(val, 10)

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


def resolve_generation_config(config: dict, rng: random.Random) -> dict:
    """Resolve all distribution-valued fields in a generation config to scalars.

    Every field is distributional: prompt arrays are enum-sampled like any other.
    Only 'k' is meta (draw count) and passes through unchanged.
    """
    resolved = {}
    for key, value in config.items():
        if key == "k":
            resolved[key] = value
        elif key == "resolution" and isinstance(value, dict) and "megapixels" in value:
            # Resolve sub-fields to scalars while preserving the input shape.
            # This makes resolved configs round-trippable: pasting a resolved
            # config back as input produces the same image.
            mega = resolve_scalar(value.get("megapixels", 262144), rng)
            if isinstance(mega, float):
                mega = round(mega)
            aspect = float(resolve_scalar(value.get("aspect_ratio", 1.0), rng))
            resolved["resolution"] = {
                "megapixels": mega,
                "aspect_ratio": aspect,
                "quantize": value.get("quantize", 32),
            }
        elif is_enumeration(value):
            resolved[key] = rng.choice(value)
        elif is_distribution(value):
            resolved[key] = resolve_scalar(value, rng)
        else:
            resolved[key] = value
    return resolved


def describe_config_space(config: dict) -> dict[str, str]:
    """Return human-readable descriptions for each distributional field.

    Scalar fields are omitted. Only fields that are distributions or
    enumerations get descriptions.
    """
    desc = {}
    for key, value in config.items():
        if key == "k":
            continue
        if key == "resolution" and isinstance(value, dict) and "megapixels" in value:
            parts = []
            for sub_key in ("megapixels", "aspect_ratio"):
                sv = value.get(sub_key)
                if sv is not None and (is_distribution(sv) or is_enumeration(sv)):
                    parts.append(f"{sub_key}: {_describe_one(sv)}")
            if parts:
                desc[key] = ", ".join(parts)
        elif is_enumeration(value):
            desc[key] = f"one of {value}"
        elif is_distribution(value):
            desc[key] = _describe_one(value)
    return desc


HUMAN_LABELS = {
    "prompt": "Prompt",
    "negative_prompt": "NegPrompt",
    "seed": "Seed",
    "n_steps": "Steps",
    "cfg": "CFG",
    "sampling_shift": "Shift",
    "multiplier": "Mult",
    "denoise": "Denoise",
    "attention_backend": "Attn",
    "resolution.megapixels": "px\u00b2",
    "resolution.aspect_ratio": "Aspect",
}


def _field_volume(spec) -> dict | None:
    """Compute volume info for a single distribution spec. Returns None if scalar."""
    if isinstance(spec, list):
        n = len(spec)
        if n <= 1:
            return None
        return {
            "kind": "enum",
            "cardinality": n,
            "log_volume": math.log2(n),
            "values": spec,
            "bounds": None,
        }
    if isinstance(spec, dict):
        if "values" in spec and "weights" in spec:
            # Weighted categorical: effective cardinality = 2^H (Shannon entropy)
            weights = spec["weights"]
            total = sum(weights)
            probs = [w / total for w in weights]
            h = -sum(p * math.log2(p) for p in probs if p > 0)
            eff = 2 ** h
            return {
                "kind": "weighted_cat",
                "cardinality": eff,
                "log_volume": h,
                "values": spec["values"],
                "bounds": None,
            }
        if "values" in spec:
            n = len(spec["values"])
            if n <= 1:
                return None
            return {
                "kind": "cat",
                "cardinality": n,
                "log_volume": math.log2(n),
                "values": spec["values"],
                "bounds": None,
            }
        if "min" in spec and "max" in spec:
            lo, hi = spec["min"], spec["max"]
            dist = spec.get("distribution", "uniform")
            step = spec.get("step")

            # Stepped range (arange semantics): finite cardinality
            if step is not None and step > 0:
                n = int((hi - lo) / step) + 1
                if n <= 1:
                    return None
                kind = "range_stepped"
                if dist == "log_uniform":
                    kind = "log_uniform_stepped"
                return {
                    "kind": kind,
                    "cardinality": n,
                    "log_volume": math.log2(n),
                    "values": None,
                    "bounds": [lo, hi],
                    "step": step,
                }

            if isinstance(lo, int) and isinstance(hi, int) and dist != "log_uniform":
                n = hi - lo + 1
                if n <= 1:
                    return None
                return {
                    "kind": "range_int",
                    "cardinality": n,
                    "log_volume": math.log2(n),
                    "values": None,
                    "bounds": [lo, hi],
                }
            # Continuous range (no step — heuristic cardinality)
            eff = 100
            kind = "log_uniform" if dist == "log_uniform" else "range_float"
            return {
                "kind": kind,
                "cardinality": eff,
                "log_volume": math.log2(eff),
                "values": None,
                "bounds": [lo, hi],
            }
    return None


def compute_config_volumes(config: dict, k: int = 1) -> list[dict]:
    """Compute volume info for each distributional field in a generation config.

    Returns a list of dicts, one per distributional field (cardinality > 1).
    Scalar fields are excluded. Resolution is decomposed into sub-fields.
    """
    volumes = []
    for key, value in config.items():
        if key == "k":
            continue
        if key == "resolution" and isinstance(value, dict) and "megapixels" in value:
            for sub_key in ("megapixels", "aspect_ratio"):
                sv = value.get(sub_key)
                if sv is None:
                    continue
                if is_enumeration(sv) or is_distribution(sv):
                    vol = _field_volume(sv)
                    if vol is not None:
                        field_key = f"resolution.{sub_key}"
                        vol["key"] = field_key
                        vol["label"] = HUMAN_LABELS.get(field_key, sub_key)
                        vol["k"] = k
                        vol["exploration"] = min(1.0, k / vol["cardinality"])
                        volumes.append(vol)
        elif is_enumeration(value) or is_distribution(value):
            vol = _field_volume(value)
            if vol is not None:
                vol["key"] = key
                vol["label"] = HUMAN_LABELS.get(key, key)
                vol["k"] = k
                vol["exploration"] = min(1.0, k / vol["cardinality"])
                volumes.append(vol)
    return volumes


def _describe_one(spec) -> str:
    """Describe a single distribution spec."""
    if isinstance(spec, list):
        return f"one of {spec}"
    if isinstance(spec, dict):
        if "values" in spec and "weights" in spec:
            pairs = [f"{v}({w})" for v, w in zip(spec["values"], spec["weights"])]
            return f"weighted {{{', '.join(pairs)}}}"
        if "values" in spec:
            return f"uniform {{{', '.join(str(v) for v in spec['values'])}}}"
        if "min" in spec and "max" in spec:
            dist = spec.get("distribution", "uniform")
            step = spec.get("step")
            step_str = f" step {step}" if step else ""
            return f"{dist}[{spec['min']}, {spec['max']}{step_str}]"
    return str(spec)

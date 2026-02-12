"""Tensor recording, loading, and comparison infrastructure for pipeline validation.

Records named tensors at each pipeline stage to disk as .pt files with a manifest.
Provides comparison functions matching the validation criteria in CLAUDE.md.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch


class TensorRecorder:
    """Saves pipeline tensors to a directory as .pt files with a manifest.

    Recording format:
        stream_dir/
            manifest.json   # source, config, stage listing, timestamps
            sigmas.pt
            noise.pt
            text_encoder_pos.pt
            text_encoder_neg.pt
            euler_step_00.pt  # dict: {"x": tensor, "denoised": tensor, "sigma": float}
            ...
            final_latent.pt
            vae_output.pt
    """

    def __init__(self, output_dir: str | Path, source: str, config_metadata: dict[str, Any] | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.source = source
        self.config_metadata = config_metadata or {}
        self.stages: list[dict[str, Any]] = []
        self._start_time = time.monotonic()

    def emit(self, name: str, data: torch.Tensor | dict[str, Any]) -> None:
        """Save a named tensor or dict of tensors/scalars to disk.

        Args:
            name: Stage name (e.g. "sigmas", "euler_step_00").
            data: A tensor, or a dict mapping names to tensors/floats.
        """
        filename = f"{name}.pt"
        filepath = self.output_dir / filename

        if isinstance(data, torch.Tensor):
            save_data = data.detach().cpu()
            shape_info = list(data.shape)
            dtype_info = str(data.dtype)
        elif isinstance(data, dict):
            save_data = {}
            shape_info = {}
            dtype_info = {}
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    save_data[k] = v.detach().cpu()
                    shape_info[k] = list(v.shape)
                    dtype_info[k] = str(v.dtype)
                else:
                    save_data[k] = v
                    shape_info[k] = "scalar"
                    dtype_info[k] = type(v).__name__
        else:
            raise TypeError(f"emit() expects Tensor or dict, got {type(data)}")

        torch.save(save_data, filepath)

        self.stages.append({
            "name": name,
            "filename": filename,
            "shape": shape_info,
            "dtype": dtype_info,
            "elapsed": time.monotonic() - self._start_time,
        })

    def close(self) -> None:
        """Write the manifest file."""
        manifest = {
            "source": self.source,
            "config": self.config_metadata,
            "stages": self.stages,
            "total_elapsed": time.monotonic() - self._start_time,
        }
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)


class TensorEmitter:
    """Facade dispatching emit() calls to multiple backends (recorder, etc.)."""

    def __init__(self, backends: list[TensorRecorder]):
        self.backends = backends

    def emit(self, name: str, data: torch.Tensor | dict[str, Any]) -> None:
        for backend in self.backends:
            backend.emit(name, data)

    def close(self) -> None:
        for backend in self.backends:
            backend.close()


def load_stream(stream_dir: str | Path) -> dict[str, Any]:
    """Load a recorded tensor stream from a directory.

    Returns:
        Dict mapping stage names to their loaded data (Tensor or dict).
    """
    stream_dir = Path(stream_dir)
    manifest_path = stream_dir / "manifest.json"

    with open(manifest_path) as f:
        manifest = json.load(f)

    result: dict[str, Any] = {}
    for stage in manifest["stages"]:
        filepath = stream_dir / stage["filename"]
        result[stage["name"]] = torch.load(filepath, weights_only=False)

    return result


def compare_bitwise(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    """Exact bitwise comparison of two tensors.

    Returns dict with 'match' bool and diagnostic info.
    """
    if a.shape != b.shape:
        return {"match": False, "reason": f"shape mismatch: {a.shape} vs {b.shape}"}
    if a.dtype != b.dtype:
        # Compare in float32 for mixed dtype
        a_f = a.float()
        b_f = b.float()
    else:
        a_f = a
        b_f = b
    match = torch.equal(a_f, b_f)
    result: dict[str, Any] = {"match": match}
    if not match:
        diff = (a_f - b_f).abs()
        result["max_abs_diff"] = float(diff.max())
        result["num_mismatched"] = int((diff > 0).sum())
        result["total_elements"] = int(a.numel())
    return result


def compare_mse(a: torch.Tensor, b: torch.Tensor, threshold: float = 1e-6) -> dict[str, Any]:
    """MSE comparison with threshold.

    Returns dict with 'pass' bool, 'mse' float, 'max_abs_diff' float.
    """
    if a.shape != b.shape:
        return {"pass": False, "reason": f"shape mismatch: {a.shape} vs {b.shape}"}
    a_f = a.float()
    b_f = b.float()
    diff = a_f - b_f
    mse = float((diff ** 2).mean())
    max_abs = float(diff.abs().max())
    return {
        "pass": mse < threshold,
        "mse": mse,
        "max_abs_diff": max_abs,
        "threshold": threshold,
    }


def compare_cosine(a: torch.Tensor, b: torch.Tensor, threshold: float = 0.99) -> dict[str, Any]:
    """Cosine similarity comparison with threshold.

    Returns dict with 'pass' bool, 'cosine_sim' float.
    """
    if a.shape != b.shape:
        return {"pass": False, "reason": f"shape mismatch: {a.shape} vs {b.shape}"}
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    cos_sim = float(torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)))
    return {
        "pass": cos_sim >= threshold,
        "cosine_sim": cos_sim,
        "threshold": threshold,
    }


# Default validation config per stage, matching CLAUDE.md criteria
VALIDATION_CONFIG_B: dict[str, dict[str, Any]] = {
    "sigmas":            {"method": "bitwise"},
    "noise":             {"method": "bitwise"},
    "text_encoder_pos":  {"method": "mse", "threshold": 1e-6},
    "text_encoder_neg":  {"method": "mse", "threshold": 1e-6},
    "final_latent":      {"method": "mse", "threshold": 1e-6},
    "vae_output":        {"method": "mse", "threshold": 1e-4},
}

VALIDATION_CONFIG_A: dict[str, dict[str, Any]] = {
    "sigmas":            {"method": "bitwise"},
    "noise":             {"method": "bitwise"},
    "text_encoder_pos":  {"method": "cosine", "threshold": 0.99},
    "text_encoder_neg":  {"method": "cosine", "threshold": 0.99},
    "final_latent":      {"method": "cosine", "threshold": 0.99},
    "vae_output":        {"method": "cosine", "threshold": 0.99},
}


def _compare_one(a_data: Any, b_data: Any, method: str, threshold: float | None = None) -> dict[str, Any]:
    """Compare a single stage's data using the specified method."""
    # Handle dict data (euler steps have {x, denoised, sigma})
    if isinstance(a_data, dict) and isinstance(b_data, dict):
        results = {}
        all_pass = True
        for key in a_data:
            if key not in b_data:
                results[key] = {"pass": False, "reason": f"key '{key}' missing from b"}
                all_pass = False
                continue
            av, bv = a_data[key], b_data[key]
            if isinstance(av, torch.Tensor) and isinstance(bv, torch.Tensor):
                r = _compare_one(av, bv, method, threshold)
                results[key] = r
                if not r.get("pass", r.get("match", False)):
                    all_pass = False
            else:
                # Scalar comparison
                match = av == bv
                results[key] = {"match": match, "a": av, "b": bv}
                if not match:
                    all_pass = False
        results["pass"] = all_pass
        return results

    if not isinstance(a_data, torch.Tensor) or not isinstance(b_data, torch.Tensor):
        return {"pass": a_data == b_data, "a": str(a_data), "b": str(b_data)}

    if method == "bitwise":
        r = compare_bitwise(a_data, b_data)
        r["pass"] = r["match"]
        return r
    elif method == "mse":
        return compare_mse(a_data, b_data, threshold=threshold or 1e-6)
    elif method == "cosine":
        return compare_cosine(a_data, b_data, threshold=threshold or 0.99)
    else:
        raise ValueError(f"Unknown comparison method: {method}")


def compare_streams(
    dir_a: str | Path,
    dir_b: str | Path,
    config: str = "B",
) -> dict[str, Any]:
    """Full comparison of two recorded streams.

    Args:
        dir_a: Path to first stream directory.
        dir_b: Path to second stream directory.
        config: "A" for defective/cosine or "B" for golden/MSE validation.

    Returns:
        Dict with per-stage results and overall pass/fail.
    """
    stream_a = load_stream(dir_a)
    stream_b = load_stream(dir_b)

    validation = VALIDATION_CONFIG_B if config == "B" else VALIDATION_CONFIG_A

    # Euler step default: MSE for config B, cosine for config A
    euler_method = "mse" if config == "B" else "cosine"
    euler_threshold = 1e-6 if config == "B" else 0.99

    results: dict[str, Any] = {}
    all_pass = True

    # Compare all stages present in both streams
    all_stages = sorted(set(stream_a.keys()) | set(stream_b.keys()))
    for stage in all_stages:
        if stage not in stream_a:
            results[stage] = {"pass": False, "reason": "missing from stream A"}
            all_pass = False
            continue
        if stage not in stream_b:
            results[stage] = {"pass": False, "reason": "missing from stream B"}
            all_pass = False
            continue

        if stage in validation:
            cfg = validation[stage]
            method = cfg["method"]
            threshold = cfg.get("threshold")
        elif stage.startswith("euler_step_"):
            method = euler_method
            threshold = euler_threshold
        else:
            # Unknown stage: default to MSE
            method = "mse"
            threshold = 1e-6

        r = _compare_one(stream_a[stage], stream_b[stage], method, threshold)
        results[stage] = r
        if not r.get("pass", False):
            all_pass = False

    return {"stages": results, "all_pass": all_pass, "config": config}


def make_euler_callback(emitter: TensorEmitter, existing_callback=None):
    """Wrap a sample_euler callback to emit per-step tensors.

    Args:
        emitter: TensorEmitter to record to.
        existing_callback: Optional existing callback to also invoke.

    Returns:
        A callback function compatible with sample_euler's callback interface.
    """
    def callback(info: dict[str, Any]) -> None:
        i = info["i"]
        emitter.emit(f"euler_step_{i:02d}", {
            "x": info["x"],
            "denoised": info["denoised"],
            "sigma": float(info["sigma"]),
        })
        if existing_callback is not None:
            existing_callback(info)
    return callback

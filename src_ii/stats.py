"""Reusable statistical functions.

Extracted from scripts_ii/score_distribution_comparison.py.
Extended with finite_differences, running_average, and sliding_std from
scripts_ii/plot_sweep_curves.py and scripts_ii/analyze_sweep_curves.py.

Import constraints:
  - numpy for array operations
  - math for sqrt in sliding_std
  - No torch, no futudiffu imports
"""

from __future__ import annotations

import math

import numpy as np


def spearman_rank_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation coefficient.

    Args:
        x: First sequence of values.
        y: Second sequence of values (same length as x).

    Returns:
        Spearman's rho in [-1, 1].
    """
    n = len(x)
    if n < 2:
        return 0.0

    def _rank(arr: list[float]) -> np.ndarray:
        sorted_indices = sorted(range(n), key=lambda i: arr[i])
        ranks = np.zeros(n)
        for r, idx in enumerate(sorted_indices):
            ranks[idx] = r + 1
        return ranks

    rx = _rank(x)
    ry = _rank(y)

    d = rx - ry
    d2_sum = (d * d).sum()
    rho = 1 - (6 * d2_sum) / (n * (n * n - 1))
    return float(rho)


def sigma_for_step(
    step_key: str,
    n_steps: int,
    device=None,
    dtype=None,
) -> "torch.Tensor":
    """Look up the sigma value for a given step key.

    Handles the "step_NN" -> sigma[NN] and "final" -> sigma[-2] mapping
    that was duplicated 2x across train_pinkify_btrm.py and
    attention_interpretability.py.

    Args:
        step_key: "step_00", "step_04", ..., "step_29", or "final".
        n_steps: Number of Euler steps in the trajectory.
        device: Target device for sigma tensor.
        dtype: Target dtype for sigma tensor.

    Returns:
        Scalar sigma tensor.
    """
    import torch
    from src_ii.sigma_schedule import build_sigma_schedule

    sigmas = build_sigma_schedule(n_steps, device=device, dtype=dtype)

    if step_key == "final":
        # "final" is after the last euler step; sigma effectively 0
        # Use the sigma from the last actual step to avoid division issues
        return sigmas[-2] if len(sigmas) > 1 else torch.tensor(
            0.01, device=device, dtype=dtype
        )
    else:
        step_idx = int(step_key.split("_")[1])
        return sigmas[step_idx]


# ---------------------------------------------------------------------------
# Time-series utilities
# ---------------------------------------------------------------------------

def finite_differences(values: list[float]) -> list[float]:
    """Forward finite difference: d[i] = values[i+1] - values[i].

    Returns a list of length len(values) - 1.

    Canonical implementation replacing:
      - scripts_ii/plot_sweep_curves.py::finite_diff()
      - scripts_ii/analyze_sweep_curves.py::compute_finite_differences()

    Args:
        values: Sequence of numeric values.

    Returns:
        Forward differences of length len(values) - 1.
    """
    return [values[i + 1] - values[i] for i in range(len(values) - 1)]


def running_average(values: list[float]) -> list[float]:
    """Cumulative running (expanding) mean.

    running_average(values)[i] = mean(values[0..i]).

    Canonical implementation replacing:
      - scripts_ii/analyze_sweep_curves.py::compute_running_mean()

    Args:
        values: Sequence of numeric values.

    Returns:
        Running mean of the same length as values.
    """
    if not values:
        return []
    result = []
    total = 0.0
    for i, v in enumerate(values):
        total += v
        result.append(total / (i + 1))
    return result


def sliding_std(values: list[float], window: int = 20) -> list[float]:
    """Sliding window population standard deviation.

    For indices 0 .. window-2 (insufficient history), returns None.
    For indices window-1 .. len(values)-1, returns the population std of the
    last `window` elements.

    Canonical implementation replacing:
      - scripts_ii/analyze_sweep_curves.py::compute_sliding_std()

    Args:
        values: Sequence of numeric values.
        window: Window size. Default 20.

    Returns:
        List of length len(values); entries are None for the first
        (window - 1) positions, then float population std thereafter.
    """
    result: list[float | None] = []
    for i in range(len(values)):
        if i < window - 1:
            result.append(None)
            continue
        w = values[i - window + 1 : i + 1]
        mean = sum(w) / len(w)
        var = sum((x - mean) ** 2 for x in w) / len(w)
        result.append(math.sqrt(var))
    return result

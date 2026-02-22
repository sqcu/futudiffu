"""Cross-head decorrelation measurement for multi-head BTRM models.

Measures Spearman rank correlation between head scores on a shared
set of images. Low correlation = the heads are learning different
features from the residual stream. High correlation = both heads
converge to the same discriminator (failure mode).

The key diagnostic for the manifold hypothesis: if two decorrelated
heads each track their ground truth function, the residual stream
supports multiple independent linear readouts.

Import constraints:
  - torch for GPU scoring
  - scipy.stats for Spearman correlation (optional, pure-Python fallback)
  - DOES NOT import: model_manager, server, client
"""

from __future__ import annotations

from typing import Sequence

import torch


def _spearman_rho(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation between two score vectors.

    Uses scipy if available, otherwise a pure-Python fallback.

    Args:
        x, y: Score vectors of equal length.

    Returns:
        Spearman rho in [-1, 1]. Returns 0.0 if either vector is constant.
    """
    n = len(x)
    if n < 3:
        return 0.0

    # Check for constant vectors (would produce nan)
    if all(v == x[0] for v in x) or all(v == y[0] for v in y):
        return 0.0

    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(x, y)
        if rho != rho:  # nan check
            return 0.0
        return float(rho)
    except ImportError:
        pass

    # Pure-Python fallback: rank values, then Pearson on ranks
    def _rank(vals):
        indexed = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(indexed):
            j = i
            while j < len(indexed) - 1 and vals[indexed[j + 1]] == vals[indexed[j]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0  # 1-based
            for k in range(i, j + 1):
                ranks[indexed[k]] = avg_rank
            i = j + 1
        return ranks

    rx = _rank(x)
    ry = _rank(y)

    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    std_rx = (sum((rx[i] - mean_rx) ** 2 for i in range(n))) ** 0.5
    std_ry = (sum((ry[i] - mean_ry) ** 2 for i in range(n))) ** 0.5

    if std_rx < 1e-10 or std_ry < 1e-10:
        return 0.0

    return cov / (std_rx * std_ry)


def measure_cross_head_decorrelation(
    model,
    latent_cache: dict[str, dict],
    head_names: Sequence[str] = ("pinkify", "thisnotthat"),
    device: torch.device | None = None,
) -> dict:
    """Measure Spearman rank correlation between head pairs on cached latents.

    Scores all images in latent_cache with each head, then computes
    pairwise Spearman rho between head score vectors.

    Args:
        model: ZImageRLAIF model instance.
        latent_cache: Dict mapping image label to cached latent data
            (same format as score_pinkify_cached uses: each value is a
            dict with "latent", "timestep", "conditioning", "num_tokens").
        head_names: Names of heads to measure. Length must match model.n_score_heads.
        device: CUDA device.

    Returns:
        dict with:
            "head_scores": {head_name: {label: score, ...}, ...}
            "cross_rho": {(head_a, head_b): rho, ...} as serializable dict
            "summary": human-readable summary string
    """
    if device is None:
        device = torch.device("cuda")

    from src_ii.btrm_lifecycle import score_serial

    # Resolve head indices (positional — head_names[i] = score column i)
    head_indices = {name: i for i, name in enumerate(head_names)}

    model.eval()

    # Score all images with each head
    head_scores = {name: {} for name in head_names}
    labels = sorted(latent_cache.keys())

    with torch.no_grad():
        for label in labels:
            cached = latent_cache[label]
            latent = cached["latent"]
            timestep = cached["timestep"]
            conditioning = cached["conditioning"]
            num_tokens = cached["num_tokens"]

            score_tensor = score_serial(
                model, latent, timestep, conditioning, num_tokens,
                gradient_checkpointing=False,
            )  # (1, N_heads)

            for name in head_names:
                idx = head_indices[name]
                head_scores[name][label] = float(score_tensor[0, idx].item())

    # Compute pairwise Spearman rho
    cross_rho = {}
    for i, name_a in enumerate(head_names):
        for j, name_b in enumerate(head_names):
            if j <= i:
                continue
            scores_a = [head_scores[name_a][label] for label in labels]
            scores_b = [head_scores[name_b][label] for label in labels]
            rho = _spearman_rho(scores_a, scores_b)
            cross_rho[f"{name_a}_vs_{name_b}"] = rho

    # Summary
    summary_parts = []
    for key, rho in cross_rho.items():
        interpretation = "DECORRELATED" if abs(rho) < 0.5 else "CORRELATED (WARNING)"
        summary_parts.append(f"  {key}: rho={rho:.4f} [{interpretation}]")
    summary = "\n".join(summary_parts) if summary_parts else "No cross-head pairs to measure"

    return {
        "head_scores": head_scores,
        "cross_rho": cross_rho,
        "summary": summary,
        "n_images": len(labels),
        "labels": labels,
    }


def measure_cross_head_from_pixel_scores(
    head_scores: dict[str, dict[str, float]],
    head_names: Sequence[str] = ("pinkify", "thisnotthat"),
) -> dict:
    """Compute cross-head Spearman rho from pre-computed pixel-space scores.

    This variant operates on scores that have already been computed
    (e.g., from ground truth reward functions or from a separate scoring
    pass). It does not need the model or latent cache.

    Args:
        head_scores: {head_name: {label: score, ...}, ...}
        head_names: Names of heads to measure.

    Returns:
        dict with: "cross_rho", "summary"
    """
    labels = sorted(head_scores[head_names[0]].keys())

    cross_rho = {}
    for i, name_a in enumerate(head_names):
        for j, name_b in enumerate(head_names):
            if j <= i:
                continue
            scores_a = [head_scores[name_a][label] for label in labels]
            scores_b = [head_scores[name_b][label] for label in labels]
            rho = _spearman_rho(scores_a, scores_b)
            cross_rho[f"{name_a}_vs_{name_b}"] = rho

    summary_parts = []
    for key, rho in cross_rho.items():
        interpretation = "DECORRELATED" if abs(rho) < 0.5 else "CORRELATED (WARNING)"
        summary_parts.append(f"  {key}: rho={rho:.4f} [{interpretation}]")
    summary = "\n".join(summary_parts) if summary_parts else "No cross-head pairs"

    return {
        "cross_rho": cross_rho,
        "summary": summary,
        "n_images": len(labels),
    }

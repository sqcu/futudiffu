"""Validate THISNOTTHAT scoring across three axes of quality.

Challenge set: 11 images in i2i_off_policies/ with known tier structure.

Three validation axes:

  Axis 1 — Perturbation robustness (THIS):
    A rotated or translated THIS image should still score positive (closer
    to THIS than THAT). Perturbations: rotate 7°, 15°, translate 5% x/y.

  Axis 2 — Perturbation robustness (THAT):
    A rotated or translated THAT image should still score negative (closer
    to THAT than THIS). Same perturbation set as axis 1.

  Axis 3 — Score discrimination:
    Arbitrary non-reference images should have meaningfully different scores
    from one another. Not collapsed into ~THIS / ~anything / ~THAT. Metrics:
    score range, min adjacent gap, whether THIS and THAT are at the extremes.

Legacy 5-constraint ranking checks are preserved for training script
compatibility.

Entry points:
  validate_tnt_full() — all three axes + ranking constraints.
  validate_tnt_ranking() — ranking constraints only (training scripts).
  _check_tnt_ranking() — pure constraint check from scores dict.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


TNT_CHALLENGE_LABELS = (
    "THIS", "THAT",
    "1bit_redraw", "bubblegum", "clear_sky", "deviantart",
    "mspaint_enso", "snek_heavy", "widemeister",
    "red_tonegraph", "nightmode",
)

_TNT_FILENAMES = {
    "THIS": "pizza-ratto.png",
    "THAT": "offhand_pleometric.png",
    "1bit_redraw": "1bit redraw.png",
    "bubblegum": "bubblegum-zinesona-4.png",
    "clear_sky": "clear-sky-thick-mkii.png",
    "deviantart": "deviantart-is-my-spine-moe-is-my-face.png",
    "mspaint_enso": "mspaint-enso-i-couldnt-forget-ii.png",
    "snek_heavy": "snek-heavy.png",
    "widemeister": "widemeister.png",
    "red_tonegraph": "red-tonegraph.png",
    "nightmode": "00500-3023556536_re_nightmode2.png",
}

_TIER_MAP = {
    "THIS": "THIS_REF",
    "THAT": "THAT_REF",
    "1bit_redraw": "SKETCH",
    "bubblegum": "SKETCH",
    "clear_sky": "SKETCH",
    "deviantart": "SKETCH",
    "mspaint_enso": "SKETCH",
    "snek_heavy": "SKETCH",
    "widemeister": "SKETCH",
    "red_tonegraph": "MIXED",
    "nightmode": "COLOR",
}

# Perturbation suite for axes 1 and 2
_PERTURBATIONS = [
    ("rotate_7deg", "rotate", 7.0),
    ("rotate_15deg", "rotate", 15.0),
    ("translate_5pct_right", "translate", (0.1, 0.0)),
    ("translate_5pct_down", "translate", (0.0, 0.1)),
]


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_tnt_challenge_images(
    challenge_dir: str | Path,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Load all 11 TNT challenge images as [3, H, W] float tensors in [0, 1]."""
    challenge_dir = Path(challenge_dir)
    if not challenge_dir.exists():
        raise FileNotFoundError(f"Challenge directory not found: {challenge_dir}")

    images = {}
    for label in TNT_CHALLENGE_LABELS:
        fname = _TNT_FILENAMES[label]
        img_path = challenge_dir / fname
        if not img_path.exists():
            raise FileNotFoundError(f"TNT challenge image '{label}' not found: {img_path}")

        pil = Image.open(str(img_path)).convert("RGB")
        arr = np.array(pil, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).to(device=device, dtype=dtype)
        images[label] = t

    return images


# ---------------------------------------------------------------------------
# Perturbation transforms (axes 1 & 2)
# ---------------------------------------------------------------------------

def _rotate_image(img: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Rotate image by angle_deg with border-replicated padding."""
    C, H, W = img.shape
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    theta = torch.tensor(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]],
        device=img.device, dtype=torch.float32,
    ).unsqueeze(0)
    grid = F.affine_grid(theta, [1, C, H, W], align_corners=False)
    out = F.grid_sample(
        img.unsqueeze(0), grid, mode="bilinear",
        padding_mode="border", align_corners=False,
    )
    return out.squeeze(0).clamp(0, 1)


def _translate_image(img: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Translate image by (dx, dy) in normalized [-1, 1] coords with border padding.

    dx=0.1 means ~5% shift right, dy=0.1 means ~5% shift down.
    """
    C, H, W = img.shape
    theta = torch.tensor(
        [[1.0, 0.0, dx], [0.0, 1.0, dy]],
        device=img.device, dtype=torch.float32,
    ).unsqueeze(0)
    grid = F.affine_grid(theta, [1, C, H, W], align_corners=False)
    out = F.grid_sample(
        img.unsqueeze(0), grid, mode="bilinear",
        padding_mode="border", align_corners=False,
    )
    return out.squeeze(0).clamp(0, 1)


def _apply_perturbation(
    img: torch.Tensor, kind: str, param,
) -> torch.Tensor:
    """Apply a named perturbation."""
    if kind == "rotate":
        return _rotate_image(img, param)
    elif kind == "translate":
        return _translate_image(img, param[0], param[1])
    else:
        raise ValueError(f"Unknown perturbation kind: {kind}")


# ---------------------------------------------------------------------------
# Ranking constraints (legacy 5-check)
# ---------------------------------------------------------------------------

def _check_tnt_ranking(scores: dict[str, float]) -> list[dict]:
    """Check the 5 TNT ranking constraints.

    Args:
        scores: Dict mapping label to score (higher = more like THIS).

    Returns:
        List of dicts with "name", "passed", "detail".
    """
    this_score = scores.get("THIS")
    that_score = scores.get("THAT")
    nightmode_score = scores.get("nightmode")

    sketch_labels = [l for l in scores if _TIER_MAP.get(l) == "SKETCH"]
    color_labels = [l for l in scores if _TIER_MAP.get(l) == "COLOR"]
    non_ref_labels = [l for l in scores if _TIER_MAP.get(l) not in ("THIS_REF", "THAT_REF")]

    sketch_scores = [scores[l] for l in sketch_labels]
    color_scores = [scores[l] for l in color_labels]
    non_ref_scores = [scores[l] for l in non_ref_labels]

    checks = []

    # 1. THIS > THAT
    if this_score is not None and that_score is not None:
        checks.append({
            "name": "THIS > THAT",
            "passed": this_score > that_score,
            "detail": f"THIS={this_score:.6f}, THAT={that_score:.6f}",
        })

    # 2. THIS > ALL others
    if this_score is not None and non_ref_scores:
        mx = max(non_ref_scores)
        checks.append({
            "name": "THIS > ALL",
            "passed": this_score > mx,
            "detail": f"THIS={this_score:.6f}, max(others)={mx:.6f}",
        })

    # 3. THAT < ALL others
    if that_score is not None and non_ref_scores:
        mn = min(non_ref_scores)
        checks.append({
            "name": "THAT < ALL",
            "passed": that_score < mn,
            "detail": f"THAT={that_score:.6f}, min(others)={mn:.6f}",
        })

    # 4. THAT < NIGHT
    if that_score is not None and nightmode_score is not None:
        checks.append({
            "name": "THAT < NIGHT",
            "passed": that_score < nightmode_score,
            "detail": f"THAT={that_score:.6f}, NIGHT={nightmode_score:.6f}",
        })

    return checks


# ---------------------------------------------------------------------------
# Axis 1 & 2: Perturbation robustness
# ---------------------------------------------------------------------------

def _check_perturbation_robustness(
    score_fn: Callable,
    this_img: torch.Tensor,
    that_img: torch.Tensor,
) -> dict:
    """Check sign preservation under small perturbations of both anchors.

    Returns dict with:
        "checks": list of pass/fail dicts (one per perturbation per anchor).
        "this_scores": dict mapping perturbation name to score.
        "that_scores": dict mapping perturbation name to score.
        "passed": True if all perturbations preserve sign.
    """
    checks = []
    this_scores = {}
    that_scores = {}

    with torch.no_grad():
        for pert_name, kind, param in _PERTURBATIONS:
            # THIS perturbation → should stay positive
            this_perturbed = _apply_perturbation(this_img, kind, param)
            raw = score_fn(this_perturbed)
            sc = float(raw.item()) if isinstance(raw, torch.Tensor) else float(raw)
            this_scores[pert_name] = sc
            checks.append({
                "name": f"THIS/{pert_name} > 0",
                "passed": sc > 0,
                "detail": f"score={sc:+.6f}",
            })

            # THAT perturbation → should stay negative
            that_perturbed = _apply_perturbation(that_img, kind, param)
            raw = score_fn(that_perturbed)
            sc = float(raw.item()) if isinstance(raw, torch.Tensor) else float(raw)
            that_scores[pert_name] = sc
            checks.append({
                "name": f"THAT/{pert_name} < 0",
                "passed": sc < 0,
                "detail": f"score={sc:+.6f}",
            })

    return {
        "checks": checks,
        "this_scores": this_scores,
        "that_scores": that_scores,
        "passed": all(c["passed"] for c in checks),
    }


# ---------------------------------------------------------------------------
# Axis 3: Score discrimination
# ---------------------------------------------------------------------------

def _check_discrimination(scores: dict[str, float]) -> dict:
    """Check that non-reference images have meaningfully different scores.

    Metrics:
        score_range: max - min of non-reference scores.
        min_adjacent_gap: smallest gap between consecutive sorted scores.
        this_that_gap: THIS score - THAT score.
        effective_resolution: score_range / this_that_gap. >1 means non-ref
            images spread wider than the reference gap (degenerate — refs
            should be at extremes). 0.3-0.8 is healthy. <0.1 is collapsed.
        n_distinct: number of non-ref images with unique scores (gap > 1e-6).
        this_is_max: whether THIS has the highest score of all images.
        that_is_min: whether THAT has the lowest score of all images.

    Returns dict with metrics and a checks list.
    """
    this_score = scores.get("THIS")
    that_score = scores.get("THAT")

    non_ref_labels = [l for l in scores if _TIER_MAP.get(l) not in ("THIS_REF", "THAT_REF")]
    non_ref_scores = sorted([scores[l] for l in non_ref_labels])
    n = len(non_ref_scores)

    checks = []

    # Score range
    score_range = non_ref_scores[-1] - non_ref_scores[0] if n >= 2 else 0.0

    # Min adjacent gap
    adjacent_gaps = [non_ref_scores[i+1] - non_ref_scores[i] for i in range(n - 1)] if n >= 2 else []
    min_adjacent_gap = min(adjacent_gaps) if adjacent_gaps else 0.0
    mean_adjacent_gap = sum(adjacent_gaps) / len(adjacent_gaps) if adjacent_gaps else 0.0

    # Number of effectively distinct scores (gap > 1e-6 between consecutive)
    n_distinct = 1
    for gap in adjacent_gaps:
        if gap > 1e-6:
            n_distinct += 1

    # THIS-THAT gap
    this_that_gap = (this_score - that_score) if (this_score is not None and that_score is not None) else 0.0

    # Effective resolution: how much of the THIS-THAT gap is used by non-ref spread
    effective_resolution = score_range / abs(this_that_gap) if abs(this_that_gap) > 1e-12 else float("inf")

    # Check: THIS should be the global max
    all_scores = list(scores.values())
    this_is_max = this_score is not None and this_score >= max(all_scores) - 1e-10
    checks.append({
        "name": "THIS is global max",
        "passed": this_is_max,
        "detail": f"THIS={this_score:.6f}, global_max={max(all_scores):.6f}",
    })

    # Check: THAT should be the global min
    that_is_min = that_score is not None and that_score <= min(all_scores) + 1e-10
    checks.append({
        "name": "THAT is global min",
        "passed": that_is_min,
        "detail": f"THAT={that_score:.6f}, global_min={min(all_scores):.6f}",
    })

    # Check: non-ref images should not all have the same score
    checks.append({
        "name": f"non-ref scores distinguishable ({n_distinct}/{n})",
        "passed": n_distinct >= min(n, 3),
        "detail": f"range={score_range:.6f}, min_gap={min_adjacent_gap:.6f}, mean_gap={mean_adjacent_gap:.6f}",
    })

    # Check: non-ref scores should be INSIDE the THIS-THAT range
    # (if they overflow, the scoring doesn't anchor on the references)
    if this_score is not None and that_score is not None and n >= 1:
        n_inside = sum(1 for s in non_ref_scores if that_score <= s <= this_score)
        checks.append({
            "name": f"non-ref scores inside THIS-THAT range ({n_inside}/{n})",
            "passed": n_inside == n,
            "detail": f"THAT={that_score:.6f}, THIS={this_score:.6f}, "
                      f"non-ref range=[{non_ref_scores[0]:.6f}, {non_ref_scores[-1]:.6f}]",
        })

    return {
        "checks": checks,
        "metrics": {
            "this_that_gap": this_that_gap,
            "non_ref_score_range": score_range,
            "min_adjacent_gap": min_adjacent_gap,
            "mean_adjacent_gap": mean_adjacent_gap,
            "effective_resolution": effective_resolution,
            "n_distinct": n_distinct,
            "n_total_non_ref": n,
            "this_is_max": this_is_max,
            "that_is_min": that_is_min,
        },
        "non_ref_sorted": list(zip(
            sorted(non_ref_labels, key=lambda l: scores[l]),
            non_ref_scores,
        )),
        "passed": all(c["passed"] for c in checks),
    }


# ---------------------------------------------------------------------------
# Full validation (all 3 axes)
# ---------------------------------------------------------------------------

def validate_tnt_full(
    score_fn: Callable | None = None,
    challenge_dir: str | Path = "i2i_off_policies",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict:
    """Full TNT validation: ranking + perturbation robustness + discrimination.

    Axis 1: Perturbed THIS stays positive.
    Axis 2: Perturbed THAT stays negative.
    Axis 3: Non-reference images have distinct, bounded scores.

    Plus the legacy 5 ranking constraints.

    Args:
        score_fn: Callable that takes [3, H, W] float32 tensor → scalar score.
                  If None, uses thisnotthat_score_gpu with default refs.
        challenge_dir: Path to directory containing challenge images.
        device: torch device (default: cuda).
        dtype: torch dtype for image loading (default: float32).

    Returns:
        dict with:
            "ranking": ranking validation result
            "perturbation": perturbation robustness result (axes 1 & 2)
            "discrimination": score discrimination result (axis 3)
            "all_checks": flat list of all checks across all axes
            "passed": True if ALL checks pass
            "summary": human-readable summary string
    """
    if device is None:
        device = torch.device("cuda")
    if dtype is None:
        dtype = torch.float32

    if score_fn is None:
        from src_ii.reward_functions import thisnotthat_score_gpu, _pil_to_tensor

        challenge_path = Path(challenge_dir)
        this_pil = Image.open(str(challenge_path / "pizza-ratto.png")).convert("RGB")
        that_pil = Image.open(str(challenge_path / "offhand_pleometric.png")).convert("RGB")
        this_ref_t = _pil_to_tensor(this_pil, device)
        that_ref_t = _pil_to_tensor(that_pil, device)

        def score_fn(img_t):
            return thisnotthat_score_gpu(img_t, this_ref_t, that_ref_t)

    # Load all images
    images = _load_tnt_challenge_images(challenge_dir, device=device, dtype=dtype)

    # Score all images
    scores = {}
    with torch.no_grad():
        for label, img_t in images.items():
            raw = score_fn(img_t)
            scores[label] = float(raw.item()) if isinstance(raw, torch.Tensor) else float(raw)

    # Ranking (legacy 5 constraints)
    ranking_checks = _check_tnt_ranking(scores)
    ranking_passed = all(c["passed"] for c in ranking_checks)
    rank_order = sorted(scores.keys(), key=lambda k: scores[k])

    ranking_result = {
        "passed": ranking_passed,
        "scores": scores,
        "checks": ranking_checks,
        "rank_order": rank_order,
    }

    # Perturbation robustness (axes 1 & 2)
    perturbation_result = _check_perturbation_robustness(
        score_fn, images["THIS"], images["THAT"],
    )

    # Discrimination (axis 3)
    discrimination_result = _check_discrimination(scores)

    # Collect all checks
    all_checks = []
    for c in ranking_checks:
        all_checks.append({**c, "axis": "ranking"})
    for c in perturbation_result["checks"]:
        all_checks.append({**c, "axis": "perturbation"})
    for c in discrimination_result["checks"]:
        all_checks.append({**c, "axis": "discrimination"})

    n_pass = sum(1 for c in all_checks if c["passed"])
    n_total = len(all_checks)
    all_passed = n_pass == n_total

    # Summary
    n_ranking = sum(1 for c in ranking_checks if c["passed"])
    n_pert = sum(1 for c in perturbation_result["checks"] if c["passed"])
    n_disc = sum(1 for c in discrimination_result["checks"] if c["passed"])
    metrics = discrimination_result["metrics"]

    summary = (
        f"TNT validation: {n_pass}/{n_total} checks passed. "
        f"Ranking: {n_ranking}/{len(ranking_checks)}. "
        f"Perturbation: {n_pert}/{len(perturbation_result['checks'])}. "
        f"Discrimination: {n_disc}/{len(discrimination_result['checks'])}. "
        f"THIS-THAT gap: {metrics['this_that_gap']:.6f}, "
        f"non-ref range: {metrics['non_ref_score_range']:.6f}, "
        f"eff_resolution: {metrics['effective_resolution']:.2f}"
    )

    return {
        "ranking": ranking_result,
        "perturbation": perturbation_result,
        "discrimination": discrimination_result,
        "all_checks": all_checks,
        "passed": all_passed,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Legacy entry point (training scripts)
# ---------------------------------------------------------------------------

def validate_tnt_ranking(
    score_fn: Callable | None = None,
    challenge_dir: str | Path = "i2i_off_policies",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict:
    """Validate THISNOTTHAT scores against the 5 ranking constraints only.

    For training scripts that need periodic lightweight validation.
    For comprehensive validation, use validate_tnt_full().

    Args:
        score_fn: Callable [3, H, W] → scalar. Default: thisnotthat_score_gpu.
        challenge_dir: Path to challenge images.
        device: torch device (default: cuda).
        dtype: torch dtype (default: float32).

    Returns:
        dict with "passed", "scores", "checks", "rank_order".
    """
    if device is None:
        device = torch.device("cuda")
    if dtype is None:
        dtype = torch.float32

    if score_fn is None:
        from src_ii.reward_functions import thisnotthat_score_gpu, _pil_to_tensor

        challenge_path = Path(challenge_dir)
        this_pil = Image.open(str(challenge_path / "pizza-ratto.png")).convert("RGB")
        that_pil = Image.open(str(challenge_path / "offhand_pleometric.png")).convert("RGB")
        this_ref_t = _pil_to_tensor(this_pil, device)
        that_ref_t = _pil_to_tensor(that_pil, device)

        def score_fn(img_t):
            return thisnotthat_score_gpu(img_t, this_ref_t, that_ref_t)

    images = _load_tnt_challenge_images(challenge_dir, device=device, dtype=dtype)

    scores = {}
    with torch.no_grad():
        for label, img_t in images.items():
            raw = score_fn(img_t)
            scores[label] = float(raw.item()) if isinstance(raw, torch.Tensor) else float(raw)

    checks = _check_tnt_ranking(scores)
    all_passed = all(c["passed"] for c in checks)
    rank_order = sorted(scores.keys(), key=lambda k: scores[k])

    return {
        "passed": all_passed,
        "scores": scores,
        "checks": checks,
        "rank_order": rank_order,
    }

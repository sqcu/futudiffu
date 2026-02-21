"""Score the PINKIFY challenge set with pinkify_score() and the reflect-mode variant.

Usage:
    python score_pinkify_cases.py
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, r'F:\dox\repos\ai\futudiffu\src_ii')
sys.path.insert(0, r'F:\dox\repos\ai\futudiffu\src')

import numpy as np
from PIL import Image
from reward_functions import (
    pinkify_score,
    _rgb_to_hsv_array,
    _is_pink_mask,
    _continuous_pinkness,
    _coverage_contrast_noscipy,
)

BASE = r'F:\dox\repos\ai\futudiffu\i2i_off_policies\PINKIFY_cases'
NAMES = ['A', 'B', 'C', 'D', 'E', 'F']


def pinkify_score_reflect(image: Image.Image) -> float:
    """Same as pinkify_score but uses mode='reflect' in the box filter."""
    img = image.convert("RGB")
    rgb = np.array(img)
    hsv = _rgb_to_hsv_array(rgb)
    pink_mask = _is_pink_mask(hsv)

    # Use numpy integral-image path with reflect padding
    pink_f = pink_mask.astype(np.float32)
    h, w = pink_f.shape
    kernel_size = 7
    pad = kernel_size // 2

    # Reflect padding (numpy reflect = mirror, no repeated edge pixel)
    padded = np.pad(pink_f, pad, mode='reflect')
    ph, pw = padded.shape

    integral = np.zeros((ph + 1, pw + 1), dtype=np.float64)
    integral[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)

    r1 = np.arange(h)
    r2 = r1 + kernel_size
    c1 = np.arange(w)
    c2 = c1 + kernel_size

    local_sum = (integral[np.ix_(r2, c2)]
                 - integral[np.ix_(r1, c2)]
                 - integral[np.ix_(r2, c1)]
                 + integral[np.ix_(r1, c1)])

    area_kernel = kernel_size * kernel_size
    local_pink_frac = local_sum / area_kernel
    contrast = (1.0 - local_pink_frac.astype(np.float32)) * pink_f

    total_area = h * w
    return float(contrast.sum() / total_area)


def load_and_diagnose(name: str) -> dict:
    path = os.path.join(BASE, f'PINKER_{name}.png')
    img = Image.open(path)
    rgb = np.array(img.convert("RGB"))
    hsv = _rgb_to_hsv_array(rgb)
    pink_mask = _is_pink_mask(hsv)
    h, w = pink_mask.shape
    area = h * w
    pink_px = int(pink_mask.sum())

    score_orig = pinkify_score(img)
    score_reflect = pinkify_score_reflect(img)

    return {
        'name': name,
        'score_orig': score_orig,
        'score_reflect': score_reflect,
        'pink_pixels': pink_px,
        'area': area,
        'pink_frac': pink_px / area,
    }


def check_ranking(scores: dict[str, float], label: str) -> None:
    """Check if scores satisfy the expected ranking constraints."""
    print(f"\n=== Ranking check: {label} ===")
    A, B, C = scores['A'], scores['B'], scores['C']
    D, E = scores['D'], scores['E']
    F = scores['F']

    tier_low = [A, B, C]
    tier_mid = [D, E]
    tier_high = [F]

    checks = []

    # A < B < C
    checks.append(("A < B", A < B, f"{A:.6f} < {B:.6f}"))
    checks.append(("B < C", B < C, f"{B:.6f} < {C:.6f}"))

    # D ~= E (within 20% relative)
    if D > 0 or E > 0:
        rel_diff = abs(D - E) / (max(D, E) + 1e-12)
        approx_eq = rel_diff < 0.20
    else:
        approx_eq = True
        rel_diff = 0.0
    checks.append(("D ~= E", approx_eq, f"|{D:.6f} - {E:.6f}| / max = {rel_diff:.3f}"))

    # max({A,B,C}) < min({D,E})
    checks.append(("max(A,B,C) < min(D,E)", max(tier_low) < min(tier_mid),
                   f"max={max(tier_low):.6f} < min={min(tier_mid):.6f}"))

    # max({D,E}) < F
    checks.append(("max(D,E) < F", max(tier_mid) < F,
                   f"max={max(tier_mid):.6f} < F={F:.6f}"))

    all_pass = True
    for desc, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {desc}: {detail}")
        if not ok:
            all_pass = False

    if all_pass:
        print("  => ALL CONSTRAINTS SATISFIED")
    else:
        print("  => RANKING MISMATCH")


def main():
    print("=" * 60)
    print("PINKIFY Challenge Set Scoring")
    print("=" * 60)

    results = [load_and_diagnose(n) for n in NAMES]

    print("\n=== RAW SCORES ===")
    print(f"{'Image':<10} {'score_orig':>12} {'score_reflect':>14} {'pink_frac':>10} {'pink_px':>8}")
    print("-" * 60)
    for r in results:
        print(f"PINKER_{r['name']:<3}  {r['score_orig']:>12.6f}  {r['score_reflect']:>14.6f}  "
              f"{r['pink_frac']:>10.4f}  {r['pink_pixels']:>8d}")

    scores_orig = {r['name']: r['score_orig'] for r in results}
    scores_reflect = {r['name']: r['score_reflect'] for r in results}

    print("\n=== SORTED BY score_orig ===")
    for name, s in sorted(scores_orig.items(), key=lambda x: x[1]):
        print(f"  PINKER_{name}: {s:.6f}")

    print("\n=== SORTED BY score_reflect ===")
    for name, s in sorted(scores_reflect.items(), key=lambda x: x[1]):
        print(f"  PINKER_{name}: {s:.6f}")

    check_ranking(scores_orig, "original (mode=constant)")
    check_ranking(scores_reflect, "reflect (mode=reflect)")

    # Pixel-level diff between tied pairs
    print("\n=== PIXEL DIFF BETWEEN TIED IMAGES ===")
    imgs = {}
    for name in NAMES:
        path = os.path.join(BASE, f'PINKER_{name}.png')
        imgs[name] = np.array(Image.open(path).convert('RGB'))

    for a, b in [('B', 'C'), ('D', 'E'), ('D', 'F'), ('E', 'F')]:
        diff = np.abs(imgs[a].astype(np.int32) - imgs[b].astype(np.int32))
        max_diff = diff.max()
        mean_diff = diff.mean()
        nonzero = int(np.sum(diff.max(axis=-1) > 0))
        print(f"  PINKER_{a} vs PINKER_{b}: max_diff={max_diff}, mean_diff={mean_diff:.4f}, "
              f"nonzero_pixels={nonzero} / {imgs[a].shape[0]*imgs[a].shape[1]}")


if __name__ == "__main__":
    main()

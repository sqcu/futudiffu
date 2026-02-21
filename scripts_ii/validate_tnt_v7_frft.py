"""TNT validation: Fractional Fourier Transform high-D descriptor.

Validates thisnotthat_score_gpu (FrFT-based, 2D isotropic FrFT at multiple
angles, compared via cosine similarity in 24K-dimensional space).

Part A: Score all 11 images in i2i_off_policies/.
         Check 5 TNT constraints. Report per-image scores and timing.

Part B: Full invariance suite on both anchors:
         10 basic transforms + 6 fractional rotations with BORDER padding.
         Sign preservation check.

Part C: Discrimination diagnostic:
         Cross-image vs self-rotation descriptor distances in the FrFT space.

Key properties:
- Resolution invariant: normalized coordinates, same evaluation grid.
- Rotation soft: isotropic 2D FrFT commutes with spatial rotation.
  Rotated input → rotated output → smooth perturbation of descriptor.
- Shallow/wide: two matmuls per angle per channel. No iterative solvers.
- High-D comparison: 24,576 dimensions for rotation manifold room.

All outputs saved to tnt_v7_validation/.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src_ii.reward_functions import thisnotthat_score_gpu
from src_ii.frft import frft_descriptor, frft_similarity

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

I2I_DIR = PROJECT_ROOT / "i2i_off_policies"
THIS_PATH = I2I_DIR / "pizza-ratto.png"
THAT_PATH = I2I_DIR / "offhand_pleometric.png"
OUTPUT_DIR = PROJECT_ROOT / "tnt_v7_validation"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TIER_MAP = {
    "pizza-ratto.png": "THIS_REF",
    "offhand_pleometric.png": "THAT_REF",
    "1bit redraw.png": "SKETCH",
    "bubblegum-zinesona-4.png": "SKETCH",
    "clear-sky-thick-mkii.png": "SKETCH",
    "deviantart-is-my-spine-moe-is-my-face.png": "SKETCH",
    "mspaint-enso-i-couldnt-forget-ii.png": "SKETCH",
    "snek-heavy.png": "SKETCH",
    "widemeister.png": "SKETCH",
    "red-tonegraph.png": "MIXED",
    "00500-3023556536_re_nightmode2.png": "COLOR",
}

SCORING_FUNCTIONS = ["frft"]

ANCHOR_NAMES = ["pizza-ratto", "offhand_pleometric"]

BASIC_TRANSFORM_NAMES = [
    "identity",
    "horizontal_flip",
    "rotate_90",
    "shear_015",
    "scale_70pct_pad",
    "hue_shift_60deg",
    "invert_colors",
    "gaussian_blur_s3",
    "gaussian_noise_005",
    "center_crop_80pct",
]

FRACTIONAL_ROTATION_ANGLES = [
    ("rot_72_pentagonal", 72.0),
    ("rot_51.4_septagonal", 360.0 / 7.0),
    ("rot_40_nonagonal", 40.0),
    ("rot_15", 15.0),
    ("rot_7", 7.0),
    ("rot_137.5_golden", 137.5),
]


# ---------------------------------------------------------------------------
# Image loading + transforms (reused from v6 validation)
# ---------------------------------------------------------------------------

def load_image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).to(device)


def discover_images(root: Path) -> list[Path]:
    images = []
    for f in sorted(root.iterdir()):
        if f.is_file() and f.suffix.lower() == ".png":
            images.append(f)
    return images


def transform_identity(img: torch.Tensor) -> torch.Tensor:
    return img.clone()

def transform_horizontal_flip(img: torch.Tensor) -> torch.Tensor:
    return img.flip(dims=[2])

def transform_rotate_90(img: torch.Tensor) -> torch.Tensor:
    return img.transpose(1, 2).flip(dims=[2])

def transform_shear_015(img: torch.Tensor) -> torch.Tensor:
    C, H, W = img.shape
    theta = torch.tensor([[1.0, 0.15, 0.0], [0.0, 1.0, 0.0]],
                         device=img.device, dtype=torch.float32).unsqueeze(0)
    grid = F.affine_grid(theta, [1, C, H, W], align_corners=False)
    return F.grid_sample(img.unsqueeze(0), grid, mode="bilinear",
                         padding_mode="zeros", align_corners=False).squeeze(0).clamp(0, 1)

def transform_scale_70pct_pad(img: torch.Tensor) -> torch.Tensor:
    C, H, W = img.shape
    sH, sW = int(H * 0.7), int(W * 0.7)
    small = F.interpolate(img.unsqueeze(0), size=(sH, sW), mode="bilinear",
                          align_corners=False).squeeze(0)
    pad_top = (H - sH) // 2
    pad_bottom = H - sH - pad_top
    pad_left = (W - sW) // 2
    pad_right = W - sW - pad_left
    return F.pad(small, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")

def _rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb[0], rgb[1], rgb[2]
    cmax, cmax_idx = rgb.max(dim=0)
    cmin = rgb.min(dim=0).values
    delta = cmax - cmin
    h = torch.zeros_like(delta)
    mask = delta > 0
    m_r = mask & (cmax_idx == 0)
    h[m_r] = ((g[m_r] - b[m_r]) / delta[m_r]) % 6.0
    m_g = mask & (cmax_idx == 1)
    h[m_g] = ((b[m_g] - r[m_g]) / delta[m_g]) + 2.0
    m_b = mask & (cmax_idx == 2)
    h[m_b] = ((r[m_b] - g[m_b]) / delta[m_b]) + 4.0
    h = (h / 6.0) % 1.0
    s = torch.where(cmax > 0, delta / cmax, torch.zeros_like(delta))
    return torch.stack([h, s, cmax], dim=0)

def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    h, s, v = hsv[0], hsv[1], hsv[2]
    h6 = h * 6.0
    i = h6.long() % 6
    f = h6 - h6.floor()
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    r = torch.where(i == 0, v, torch.where(i == 1, q, torch.where(i == 2, p,
         torch.where(i == 3, p, torch.where(i == 4, t, v)))))
    g = torch.where(i == 0, t, torch.where(i == 1, v, torch.where(i == 2, v,
         torch.where(i == 3, q, torch.where(i == 4, p, p)))))
    b = torch.where(i == 0, p, torch.where(i == 1, p, torch.where(i == 2, t,
         torch.where(i == 3, v, torch.where(i == 4, v, q)))))
    return torch.stack([r, g, b], dim=0)

def transform_hue_shift_60deg(img: torch.Tensor) -> torch.Tensor:
    hsv = _rgb_to_hsv(img)
    hsv[0] = (hsv[0] + 60.0 / 360.0) % 1.0
    return _hsv_to_rgb(hsv).clamp(0, 1)

def transform_invert_colors(img: torch.Tensor) -> torch.Tensor:
    return 1.0 - img

def _gaussian_kernel_1d(sigma: float, kernel_size: int, device: torch.device) -> torch.Tensor:
    x = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()

def transform_gaussian_blur_s3(img: torch.Tensor) -> torch.Tensor:
    sigma, ks = 3.0, 19
    C = img.shape[0]
    k1d = _gaussian_kernel_1d(sigma, ks, img.device)
    p = ks // 2
    x = F.pad(img.unsqueeze(0), (p, p, 0, 0), mode="reflect")
    x = F.conv2d(x, k1d.view(1, 1, 1, ks).expand(C, 1, 1, ks), groups=C)
    x = F.pad(x, (0, 0, p, p), mode="reflect")
    x = F.conv2d(x, k1d.view(1, 1, ks, 1).expand(C, 1, ks, 1), groups=C)
    return x.squeeze(0)

def transform_gaussian_noise_005(img: torch.Tensor) -> torch.Tensor:
    gen = torch.Generator(device=img.device)
    gen.manual_seed(12345)
    return (img + torch.randn_like(img, generator=gen) * 0.05).clamp(0, 1)

def transform_center_crop_80pct(img: torch.Tensor) -> torch.Tensor:
    C, H, W = img.shape
    cH, cW = int(H * 0.8), int(W * 0.8)
    top, left = (H - cH) // 2, (W - cW) // 2
    cropped = img[:, top:top + cH, left:left + cW]
    return F.interpolate(cropped.unsqueeze(0), size=(H, W), mode="bilinear",
                         align_corners=False).squeeze(0)

BASIC_TRANSFORM_FNS = [
    transform_identity, transform_horizontal_flip, transform_rotate_90,
    transform_shear_015, transform_scale_70pct_pad, transform_hue_shift_60deg,
    transform_invert_colors, transform_gaussian_blur_s3,
    transform_gaussian_noise_005, transform_center_crop_80pct,
]

def rotate_image_border(img: torch.Tensor, angle_deg: float) -> torch.Tensor:
    C, H, W = img.shape
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    theta = torch.tensor([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]],
                         device=img.device, dtype=torch.float32).unsqueeze(0)
    grid = F.affine_grid(theta, [1, C, H, W], align_corners=False)
    return F.grid_sample(img.unsqueeze(0), grid, mode="bilinear",
                         padding_mode="border", align_corners=False).squeeze(0).clamp(0, 1)


# ---------------------------------------------------------------------------
# Scoring wrappers
# ---------------------------------------------------------------------------

def score_frft(image: torch.Tensor, this_ref: torch.Tensor, that_ref: torch.Tensor) -> float:
    score = thisnotthat_score_gpu(image, this_ref.unsqueeze(0), that_ref.unsqueeze(0))
    return float(score.item())

SCORE_FNS = {
    "frft": score_frft,
}


# ---------------------------------------------------------------------------
# Constraint checking
# ---------------------------------------------------------------------------

def check_constraints(scores: dict[str, float]) -> dict[str, dict]:
    this_ref_score = that_ref_score = nightmode_score = None
    sketch_scores, color_scores, all_non_ref = [], [], []

    for name, sc in scores.items():
        tier = TIER_MAP.get(name)
        if tier == "THIS_REF":
            this_ref_score = sc
        elif tier == "THAT_REF":
            that_ref_score = sc
        elif tier == "SKETCH":
            sketch_scores.append(sc)
            all_non_ref.append(sc)
        elif tier == "MIXED":
            all_non_ref.append(sc)
        elif tier == "COLOR":
            nightmode_score = sc
            color_scores.append(sc)
            all_non_ref.append(sc)
        else:
            all_non_ref.append(sc)

    results = {}
    if this_ref_score is not None and that_ref_score is not None:
        results["1_THIS_GT_THAT"] = {
            "passed": this_ref_score > that_ref_score,
            "detail": f"THIS={this_ref_score:.6f} THAT={that_ref_score:.6f}",
        }
    if this_ref_score is not None and all_non_ref:
        mx = max(all_non_ref)
        results["2_THIS_GT_ALL"] = {"passed": this_ref_score > mx,
                                     "detail": f"THIS={this_ref_score:.6f} max(others)={mx:.6f}"}
    if that_ref_score is not None and all_non_ref:
        mn = min(all_non_ref)
        results["3_THAT_LT_ALL"] = {"passed": that_ref_score < mn,
                                     "detail": f"THAT={that_ref_score:.6f} min(others)={mn:.6f}"}
    if sketch_scores and color_scores:
        results["4_SKETCH_GT_COLOR"] = {
            "passed": min(sketch_scores) > max(color_scores),
            "detail": f"min(sketch)={min(sketch_scores):.6f} max(color)={max(color_scores):.6f}",
        }
    if that_ref_score is not None and nightmode_score is not None:
        results["5_THAT_LT_NIGHT"] = {
            "passed": that_ref_score < nightmode_score,
            "detail": f"THAT={that_ref_score:.6f} NIGHT={nightmode_score:.6f}",
        }
    return results


# ---------------------------------------------------------------------------
# Part A: Constraint scoring
# ---------------------------------------------------------------------------

def run_part_a(this_ref: torch.Tensor, that_ref: torch.Tensor, images: list[Path]):
    print("\n" + "=" * 70)
    print("PART A: Constraint scoring on all images")
    print("=" * 70)

    all_scores = {}
    all_timings = {}

    for method_name in SCORING_FUNCTIONS:
        score_fn = SCORE_FNS[method_name]
        scores = {}
        t0 = time.perf_counter()

        for img_path in images:
            img = load_image_tensor(img_path, DEVICE)
            sc = score_fn(img, this_ref, that_ref)
            scores[img_path.name] = sc

        elapsed = (time.perf_counter() - t0) * 1000
        avg_ms = elapsed / len(images)
        all_scores[method_name] = scores
        all_timings[method_name] = avg_ms

        print(f"\n--- {method_name} ({avg_ms:.1f} ms/image) ---")
        for name in sorted(scores.keys()):
            tier = TIER_MAP.get(name, "???")
            print(f"  {name:50s} {tier:10s} {scores[name]:+.6f}")

        constraints = check_constraints(scores)
        n_pass = sum(1 for c in constraints.values() if c["passed"])
        print(f"  Constraints: {n_pass}/5")
        for cname, cval in sorted(constraints.items()):
            status = "PASS" if cval["passed"] else "FAIL"
            print(f"    {cname}: {status} ({cval['detail']})")

    return all_scores, all_timings


# ---------------------------------------------------------------------------
# Part B: Invariance suite
# ---------------------------------------------------------------------------

def run_part_b(this_ref: torch.Tensor, that_ref: torch.Tensor, anchors: dict[str, torch.Tensor]):
    print("\n" + "=" * 70)
    print("PART B: Invariance suite (sign preservation)")
    print("=" * 70)

    results = {}

    for method_name in SCORING_FUNCTIONS:
        score_fn = SCORE_FNS[method_name]
        method_results = {}

        for anchor_name, anchor_img in anchors.items():
            expected_sign = 1 if anchor_name == "pizza-ratto" else -1

            # Basic transforms
            for tx_name, tx_fn in zip(BASIC_TRANSFORM_NAMES, BASIC_TRANSFORM_FNS):
                transformed = tx_fn(anchor_img)
                sc = score_fn(transformed, this_ref, that_ref)
                passed = (sc > 0) == (expected_sign > 0)
                key = f"{anchor_name}/{tx_name}"
                method_results[key] = {"score": sc, "passed": passed}

            # Fractional rotations
            for rot_name, rot_angle in FRACTIONAL_ROTATION_ANGLES:
                rotated = rotate_image_border(anchor_img, rot_angle)
                sc = score_fn(rotated, this_ref, that_ref)
                passed = (sc > 0) == (expected_sign > 0)
                key = f"{anchor_name}/{rot_name}"
                method_results[key] = {"score": sc, "passed": passed}

        results[method_name] = method_results

        # Summary
        n_pass = sum(1 for v in method_results.values() if v["passed"])
        n_total = len(method_results)

        # Per-anchor counts
        pizza_pass = sum(1 for k, v in method_results.items()
                        if k.startswith("pizza-ratto") and v["passed"])
        pizza_total = sum(1 for k in method_results if k.startswith("pizza-ratto"))
        offhand_pass = sum(1 for k, v in method_results.items()
                          if k.startswith("offhand_pleometric") and v["passed"])
        offhand_total = sum(1 for k in method_results if k.startswith("offhand_pleometric"))

        # Fractional rotation counts
        frac_pass = sum(1 for k, v in method_results.items()
                       if "rot_" in k and v["passed"])
        frac_total = sum(1 for k in method_results if "rot_" in k)

        print(f"\n--- {method_name} ---")
        print(f"  Overall: {n_pass}/{n_total}")
        print(f"  pizza-ratto: {pizza_pass}/{pizza_total}  offhand_pleometric: {offhand_pass}/{offhand_total}")
        print(f"  Fractional rotations: {frac_pass}/{frac_total}")

        # Show failures
        failures = [(k, v) for k, v in method_results.items() if not v["passed"]]
        if failures:
            print(f"  Failures:")
            for k, v in failures:
                print(f"    {k}: {v['score']:+.6f}")
        else:
            print(f"  No failures!")

    return results


# ---------------------------------------------------------------------------
# Part C: Discrimination diagnostic (FrFT descriptor space)
# ---------------------------------------------------------------------------

def run_part_c(this_ref: torch.Tensor, that_ref: torch.Tensor,
               anchors: dict[str, torch.Tensor]):
    print("\n" + "=" * 70)
    print("PART C: Discrimination diagnostic in FrFT descriptor space")
    print("=" * 70)

    # Compute descriptors at native resolution
    desc_this = frft_descriptor(anchors["pizza-ratto"])
    desc_that = frft_descriptor(anchors["offhand_pleometric"])

    # Cross-image distance
    cross_sim = frft_similarity(desc_this, desc_that)
    cross_dist = 1.0 - float(cross_sim.item())

    # Self-rotation distances for each anchor at multiple angles
    results = {"cross_cosine_sim": float(cross_sim.item()), "cross_dist": cross_dist}

    for anchor_name, anchor_img in anchors.items():
        desc_orig = frft_descriptor(anchor_img)
        rot_results = []

        for rot_name, rot_angle in FRACTIONAL_ROTATION_ANGLES:
            rotated = rotate_image_border(anchor_img, rot_angle)
            desc_rot = frft_descriptor(rotated)

            self_sim = frft_similarity(desc_orig, desc_rot)
            self_dist = 1.0 - float(self_sim.item())

            # Also: cosine sim between rotated and the OTHER anchor
            desc_other = desc_that if anchor_name == "pizza-ratto" else desc_this
            cross_rot_sim = frft_similarity(desc_rot, desc_other)

            rot_results.append({
                "angle": rot_name,
                "self_cosine_sim": float(self_sim.item()),
                "self_dist": self_dist,
                "cross_rot_cosine_sim": float(cross_rot_sim.item()),
                "ratio": cross_dist / (self_dist + 1e-12),
            })

        results[anchor_name] = rot_results

        print(f"\n--- {anchor_name} ---")
        print(f"  Cross-image dist (1-cos): {cross_dist:.6f}")
        for r in rot_results:
            ratio_str = f"{r['ratio']:.2f}" if r['self_dist'] > 1e-8 else "inf"
            print(f"  {r['angle']:30s}  self_dist={r['self_dist']:.6f}  "
                  f"ratio(cross/self)={ratio_str}  "
                  f"rot→other_sim={r['cross_rot_cosine_sim']:.6f}")

    # Descriptor dimensionality info
    print(f"\n  Descriptor dim: {desc_this.shape[0]}")
    print(f"  THIS norm: {torch.linalg.norm(desc_this):.2f}")
    print(f"  THAT norm: {torch.linalg.norm(desc_that):.2f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"THIS: {THIS_PATH}")
    print(f"THAT: {THAT_PATH}")

    this_ref = load_image_tensor(THIS_PATH, DEVICE)
    that_ref = load_image_tensor(THAT_PATH, DEVICE)
    print(f"THIS shape: {this_ref.shape}")
    print(f"THAT shape: {that_ref.shape}")

    images = discover_images(I2I_DIR)
    print(f"Found {len(images)} images in {I2I_DIR}")

    anchors = {
        "pizza-ratto": this_ref,
        "offhand_pleometric": that_ref,
    }

    # Part A
    all_scores, all_timings = run_part_a(this_ref, that_ref, images)

    # Part B
    invariance_results = run_part_b(this_ref, that_ref, anchors)

    # Part C
    discrimination_results = run_part_c(this_ref, that_ref, anchors)

    # Save results
    summary = {
        "device": str(DEVICE),
        "scoring_functions": SCORING_FUNCTIONS,
        "part_a_timings": all_timings,
        "part_a_constraint_pass_counts": {},
        "part_b_pass_rates": {},
    }

    for method in SCORING_FUNCTIONS:
        constraints = check_constraints(all_scores[method])
        summary["part_a_constraint_pass_counts"][method] = sum(
            1 for c in constraints.values() if c["passed"]
        )
        inv = invariance_results[method]
        n_pass = sum(1 for v in inv.values() if v["passed"])
        frac_pass = sum(1 for k, v in inv.items() if "rot_" in k and v["passed"])
        frac_total = sum(1 for k in inv if "rot_" in k)
        summary["part_b_pass_rates"][method] = {
            "overall": f"{n_pass}/{len(inv)}",
            "fractional_rotations": f"{frac_pass}/{frac_total}",
        }

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(OUTPUT_DIR / "part_a_scores.json", "w") as f:
        json.dump(all_scores, f, indent=2)

    with open(OUTPUT_DIR / "part_b_invariance.json", "w") as f:
        json.dump(invariance_results, f, indent=2, default=str)

    with open(OUTPUT_DIR / "part_c_discrimination.json", "w") as f:
        json.dump(discrimination_results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")

    # Final summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<25s} {'Constraints':<15s} {'Invariance':<15s} {'Frac Rot':<15s} {'ms/img':<10s}")
    print("-" * 70)
    for method in SCORING_FUNCTIONS:
        c = summary["part_a_constraint_pass_counts"][method]
        inv = summary["part_b_pass_rates"][method]
        t = all_timings[method]
        print(f"{method:<25s} {c}/5{'':<11s} {inv['overall']:<15s} {inv['fractional_rotations']:<15s} {t:<10.1f}")


if __name__ == "__main__":
    main()

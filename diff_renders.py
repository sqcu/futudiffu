"""False-color diff visualization across all bench_renders variants.

Compares:
  1. Each attention variant against sdpa (baseline)
  2. Each attention variant against policy_on
  3. policy_on against sdpa

Outputs per-pair false-color diffs and a summary grid.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw


RENDER_DIR = os.path.join(os.path.dirname(__file__), "bench_renders")

# All single-image renders to compare (order: reference first)
VARIANTS = [
    ("sdpa", "sdpa.png"),
    ("fp8+bf16", "fp8_qk_p_bf16_pv.png"),
    ("int8+bf16", "int8_qk_p_bf16_pv.png"),
    ("fp8+fp8", "fp8_fp8_isolated.png"),
    ("policy_on", "policy_on.png"),
    ("policy_off", "policy_off.png"),
]


def load_img(path):
    return np.array(Image.open(path)).astype(np.float32)


def false_color_diff(a, b, gain=8.0):
    """Absolute per-pixel diff -> grayscale -> inferno-ish false color.

    gain: multiplier on raw [0,255] diff. gain=8 saturates at ~32/255 diff.
    """
    diff = np.abs(a - b)
    gray = diff.mean(axis=2)  # (H, W)
    t = np.clip(gray * gain / 255.0, 0, 1)

    # inferno-inspired: black -> indigo -> red -> yellow -> white
    r = np.clip(np.where(t < 0.4, t * 2.0, np.where(t < 0.75, 0.8 + (t - 0.4) * 0.57, 1.0)), 0, 1)
    g = np.clip(np.where(t < 0.5, 0, (t - 0.5) * 2.0), 0, 1)
    b_ch = np.clip(np.where(t < 0.25, t * 3.0, np.where(t < 0.6, 0.75 - (t - 0.25) * 2.14, 0)), 0, 1)

    rgb = np.stack([r, g, b_ch], axis=2)
    return (rgb * 255).astype(np.uint8), gray


def annotate(img_np, text, size_hint=16):
    """Add text label at top-left with shadow."""
    img = Image.fromarray(img_np.copy())
    draw = ImageDraw.Draw(img)
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]:
        draw.text((10 + dx, 10 + dy), text, fill=(0, 0, 0))
    draw.text((10, 10), text, fill=(255, 255, 255))
    return np.array(img)


def stats_line(a, b):
    diff = np.abs(a - b)
    gray = diff.mean(axis=2)
    mean = gray.mean()
    p95 = np.percentile(gray, 95)
    p99 = np.percentile(gray, 99)
    pct_nonzero = 100 * np.count_nonzero(diff) / diff.size
    return f"mean={mean:.1f} p95={p95:.0f} p99={p99:.0f} nz={pct_nonzero:.0f}%"


def main():
    # Load all images
    images = {}
    for name, fname in VARIANTS:
        path = os.path.join(RENDER_DIR, fname)
        if os.path.exists(path):
            images[name] = load_img(path)
            print(f"  Loaded {name}: {images[name].shape}")
        else:
            print(f"  MISSING: {fname}")

    sdpa = images.get("sdpa")
    policy = images.get("policy_on")

    if sdpa is None or policy is None:
        print("Need both sdpa.png and policy_on.png")
        return

    # ---- Numeric summary table ----
    print("\n=== Pairwise diff statistics ===")
    print(f"{'A':>12s} vs {'B':>12s} | {'mean':>6s} {'p95':>5s} {'p99':>5s} {'nz%':>5s}")
    print("-" * 60)

    pairs = []
    for name_a, img_a in images.items():
        for name_b, img_b in images.items():
            if name_a >= name_b:
                continue
            diff = np.abs(img_a - img_b)
            gray = diff.mean(axis=2)
            mean_d = gray.mean()
            p95 = np.percentile(gray, 95)
            p99 = np.percentile(gray, 99)
            nz = 100 * np.count_nonzero(diff) / diff.size
            print(f"{name_a:>12s} vs {name_b:>12s} | {mean_d:6.1f} {p95:5.0f} {p99:5.0f} {nz:5.1f}")
            pairs.append((name_a, name_b, mean_d))

    # ---- False-color diffs: each variant vs sdpa ----
    gain = 8.0
    print(f"\n=== False-color diffs (gain={gain}x) ===")

    # Row 1: variant vs sdpa
    # Row 2: variant vs policy_on
    variant_names = [n for n in images if n not in ("sdpa", "policy_off")]
    h, w = sdpa.shape[:2]
    n_variants = len(variant_names)

    # Build a grid: 2 rows x n_variants columns
    # Row 0: X vs sdpa
    # Row 1: X vs policy_on
    pad = 4
    grid_h = 2 * h + pad
    grid_w = n_variants * w + (n_variants - 1) * pad
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for col, name in enumerate(variant_names):
        img = images[name]
        x_off = col * (w + pad)

        # Row 0: vs sdpa
        fc, _ = false_color_diff(sdpa, img, gain=gain)
        fc = annotate(fc, f"{name} vs sdpa | {stats_line(sdpa, img)}")
        grid[:h, x_off:x_off + w] = fc

        # Row 1: vs policy_on
        fc2, _ = false_color_diff(policy, img, gain=gain)
        fc2 = annotate(fc2, f"{name} vs policy | {stats_line(policy, img)}")
        grid[h + pad:, x_off:x_off + w] = fc2

    out_path = os.path.join(RENDER_DIR, "diff_grid.png")
    Image.fromarray(grid).save(out_path)
    print(f"Saved: {out_path}")

    # ---- Individual high-res diffs for the key pairs ----
    key_pairs = [
        ("sdpa", "policy_on"),
        ("sdpa", "fp8+bf16"),
        ("sdpa", "int8+bf16"),
        ("sdpa", "fp8+fp8"),
        ("fp8+bf16", "policy_on"),
        ("int8+bf16", "policy_on"),
    ]
    for name_a, name_b in key_pairs:
        if name_a not in images or name_b not in images:
            continue
        a, b = images[name_a], images[name_b]
        fc, gray = false_color_diff(a, b, gain=gain)

        # Compose: original A | false color diff | original B
        strip = np.zeros((h, 3 * w + 2 * pad, 3), dtype=np.uint8)
        strip[:, :w] = annotate(a.astype(np.uint8), name_a)
        strip[:, w + pad:2 * w + pad] = annotate(fc, f"|{name_a} - {name_b}| x{int(gain)}")
        strip[:, 2 * w + 2 * pad:] = annotate(b.astype(np.uint8), name_b)

        safe_name = f"diff_{name_a}_vs_{name_b}".replace("+", "_")
        out_path = os.path.join(RENDER_DIR, f"{safe_name}.png")
        Image.fromarray(strip).save(out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

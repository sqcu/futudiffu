"""Diagnostic: measure per-step divergence of i2i trajectories from source images.

For each i2i trajectory, loads the source image (preprocessed the same way as
generation), loads each rendered intermediate step, and computes:
  - MSE(step_output, source) at each checkpoint
  - d_MSE / d_step (divergence rate)

This reveals whether the i2i pipeline is actually editing the source or just
reproducing it, and quantifies the effect of center-padding vs resize.

Usage:
    .venv/Scripts/python.exe diagnose_i2i.py [--dataset-dir btrm_dataset]
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

I2I_DIR = Path(r"F:\dox\repos\ai\futudiffu\i2i_off_policies")


def load_source_padded(image_path: Path, target_h: int, target_w: int) -> np.ndarray:
    """Replicate the exact preprocessing from generate_btrm_dataset.load_i2i_image.

    Returns (H, W, 3) float32 array in [0, 1].
    """
    img = Image.open(str(image_path)).convert("RGB")
    img_w, img_h = img.size
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    paste_x = max(0, (target_w - img_w) // 2)
    paste_y = max(0, (target_h - img_h) // 2)
    crop_x = max(0, (img_w - target_w) // 2)
    crop_y = max(0, (img_h - target_h) // 2)
    if img_w > target_w or img_h > target_h:
        img = img.crop((
            crop_x, crop_y,
            crop_x + min(img_w, target_w),
            crop_y + min(img_h, target_h),
        ))
        paste_x = max(0, (target_w - img.size[0]) // 2)
        paste_y = max(0, (target_h - img.size[1]) // 2)
    canvas.paste(img, (paste_x, paste_y))
    return np.array(canvas, dtype=np.float32) / 255.0


def load_render(render_path: Path) -> np.ndarray:
    """Load a rendered PNG as (H, W, 3) float32 in [0, 1]."""
    img = Image.open(str(render_path)).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error between two images."""
    return float(np.mean((a - b) ** 2))


def content_region_mse(
    render: np.ndarray,
    source_padded: np.ndarray,
    source_img: Image.Image,
    target_h: int,
    target_w: int,
) -> float:
    """MSE computed only over the content region (non-black area of source)."""
    img_w, img_h = source_img.size
    paste_x = max(0, (target_w - min(img_w, target_w)) // 2)
    paste_y = max(0, (target_h - min(img_h, target_h)) // 2)
    actual_w = min(img_w, target_w)
    actual_h = min(img_h, target_h)

    region_render = render[paste_y:paste_y + actual_h, paste_x:paste_x + actual_w]
    region_source = source_padded[paste_y:paste_y + actual_h, paste_x:paste_x + actual_w]
    return float(np.mean((region_render - region_source) ** 2))


def border_region_mse(
    render: np.ndarray,
    source_img: Image.Image,
    target_h: int,
    target_w: int,
) -> float:
    """MSE of the border region (black padding area) vs actual black."""
    img_w, img_h = source_img.size
    paste_x = max(0, (target_w - min(img_w, target_w)) // 2)
    paste_y = max(0, (target_h - min(img_h, target_h)) // 2)
    actual_w = min(img_w, target_w)
    actual_h = min(img_h, target_h)

    # Create mask of border pixels
    mask = np.ones((target_h, target_w), dtype=bool)
    mask[paste_y:paste_y + actual_h, paste_x:paste_x + actual_w] = False
    n_border = mask.sum()
    if n_border == 0:
        return 0.0

    border_pixels = render[mask]
    # Border should be black (0) if model preserved it
    return float(np.mean(border_pixels ** 2))


def analyze_trajectory(
    traj_dir: Path,
    render_dir: Path,
    target_h: int = 832,
    target_w: int = 1280,
) -> dict | None:
    """Analyze one i2i trajectory's divergence from source."""
    meta_path = traj_dir / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    if meta.get("type") != "i2i":
        return None
    if not render_dir.exists():
        return None

    image_file = meta["image_file"]
    image_path = I2I_DIR / image_file
    if not image_path.exists():
        print(f"  WARNING: source image not found: {image_path}")
        return None

    source_img = Image.open(str(image_path)).convert("RGB")
    source_padded = load_source_padded(image_path, target_h, target_w)

    # Collect rendered steps
    renders = {}
    for f in sorted(render_dir.iterdir()):
        if f.suffix == ".png":
            name = f.stem
            renders[name] = load_render(f)

    if not renders:
        return None

    # Parse step indices
    steps = {}
    for name, img_arr in renders.items():
        if name.startswith("step_"):
            step_idx = int(name.split("_")[1])
            steps[step_idx] = img_arr
        elif name == "final":
            steps["final"] = img_arr

    # Compute MSE at each step
    results = {
        "traj": traj_dir.name,
        "image_file": image_file,
        "source_size": f"{source_img.size[0]}x{source_img.size[1]}",
        "denoise": meta.get("denoise", 1.0),
        "n_steps": meta["n_steps"],
        "precision": meta["precision"],
        "steps": {},
    }

    # Content area fraction
    img_w, img_h = source_img.size
    content_area = min(img_w, target_w) * min(img_h, target_h)
    total_area = target_w * target_h
    results["content_fraction"] = round(content_area / total_area, 3)

    for step_key in sorted(steps.keys(), key=lambda x: -1 if x == "final" else x):
        img_arr = steps[step_key]
        full_mse = mse(img_arr, source_padded)
        c_mse = content_region_mse(img_arr, source_padded, source_img, target_h, target_w)
        b_mse = border_region_mse(img_arr, source_img, target_h, target_w)

        results["steps"][str(step_key)] = {
            "full_mse": round(full_mse, 6),
            "content_mse": round(c_mse, 6),
            "border_mse": round(b_mse, 6),
        }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose i2i trajectory divergence")
    parser.add_argument("--dataset-dir", type=str,
                        default=r"F:\dox\repos\ai\futudiffu\btrm_dataset")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    latents_dir = dataset_dir / "latents"
    renders_dir = dataset_dir / "renders"

    if not latents_dir.exists():
        print(f"No latents dir at {latents_dir}")
        return

    # Find all i2i trajectories
    results = []
    for traj_dir in sorted(latents_dir.iterdir()):
        if not traj_dir.is_dir():
            continue
        meta_path = traj_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("type") != "i2i":
            continue

        render_dir = renders_dir / traj_dir.name
        result = analyze_trajectory(traj_dir, render_dir)
        if result:
            results.append(result)

    if not results:
        print("No rendered i2i trajectories found.")
        return

    # Print report
    print(f"{'='*80}")
    print(f"  I2I DIVERGENCE DIAGNOSTIC — {len(results)} trajectories")
    print(f"{'='*80}")
    print()

    for r in results:
        print(f"--- {r['traj']} ---")
        print(f"  Source: {r['image_file']} ({r['source_size']})")
        print(f"  Content fraction: {r['content_fraction']:.1%} of 1280x832 canvas")
        print(f"  Denoise: {r['denoise']:.3f}, Steps: {r['n_steps']}, Precision: {r['precision']}")
        print()

        # Step table
        print(f"  {'Step':>8}  {'Full MSE':>10}  {'Content MSE':>12}  {'Border MSE':>12}  {'d_content':>10}")
        print(f"  {'-'*58}")

        prev_c_mse = None
        prev_step = None
        for step_key, data in sorted(r["steps"].items(),
                                      key=lambda x: (-1 if x[0] == "final" else int(x[0]))):
            step_label = f"step_{step_key:>02}" if step_key != "final" else "final"
            d_content = ""
            if prev_c_mse is not None and step_key != "final":
                step_num = int(step_key)
                delta = data["content_mse"] - prev_c_mse
                d_steps = step_num - prev_step if prev_step is not None else 1
                rate = delta / max(d_steps, 1)
                d_content = f"{rate:+.6f}"

            print(f"  {step_label:>8}  {data['full_mse']:>10.6f}  "
                  f"{data['content_mse']:>12.6f}  {data['border_mse']:>12.6f}  "
                  f"{d_content:>10}")

            if step_key != "final":
                prev_c_mse = data["content_mse"]
                prev_step = int(step_key)

        # Summary stats
        step_keys_numeric = [k for k in r["steps"] if k != "final"]
        if step_keys_numeric:
            first_step = min(step_keys_numeric, key=int)
            last_step = max(step_keys_numeric, key=int)
            first_content = r["steps"][first_step]["content_mse"]
            last_content = r["steps"][last_step]["content_mse"]
            final_content = r["steps"].get("final", {}).get("content_mse", last_content)

            print()
            print(f"  Content MSE: start={first_content:.6f} → end={final_content:.6f}")
            print(f"  Total content divergence: {final_content - first_content:+.6f}")
            if first_content > 0:
                print(f"  Relative change: {(final_content - first_content) / first_content:+.1%}")

            # Border analysis
            final_border = r["steps"].get("final", {}).get("border_mse", 0)
            print(f"  Final border MSE (vs black): {final_border:.6f}")
            border_fraction = 1.0 - r["content_fraction"]
            print(f"  Border pixel fraction: {border_fraction:.1%}")

        print()

    # Summary table
    print(f"\n{'='*80}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"  {'Traj':>14}  {'Source':>20}  {'Content%':>8}  {'Denoise':>7}  "
          f"{'Final Content MSE':>18}  {'Final Border MSE':>17}")
    print(f"  {'-'*92}")

    for r in results:
        final = r["steps"].get("final", {})
        print(f"  {r['traj']:>14}  {r['source_size']:>20}  "
              f"{r['content_fraction']:>7.1%}  {r['denoise']:>7.3f}  "
              f"{final.get('content_mse', 0):>18.6f}  {final.get('border_mse', 0):>17.6f}")


if __name__ == "__main__":
    main()

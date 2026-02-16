"""Analyze rendered heattest images: PSNR, naturalness, clustering, parent-child.

CPU only -- reads manifest.json + PNGs from render_heattest.py output.
No torch, no server, no dataset_v2.

Five analyses:
  1. PSNR monotonicity: psnr(step_i, final) must increase with step index
  2. Naturalness statistics: entropy, spectral slope, Laplacian on finals
  3. Color histogram clustering: same-prompt > same-seed similarity
  4. Parent-child correlation: divergence from parent step to child step_00
  5. Summary report: analysis_report.json + analysis_summary.txt

Usage:
    .venv/Scripts/python.exe 'F:\\dox\\repos\\ai\\futudiffu\\scripts\\analyze_heattest.py'
    .venv/Scripts/python.exe 'F:\\dox\\repos\\ai\\futudiffu\\scripts\\analyze_heattest.py' \\
        --render-dir PATH
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

import numpy as np

from futudiffu.image_stats import (
    color_histogram,
    histogram_cosine_sim,
    naturalness_report,
    psnr,
)
from futudiffu.rendering import load_image_array


def load_manifest(render_dir: Path) -> dict:
    """Load manifest.json from render_heattest output."""
    manifest_path = render_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"No manifest.json at {manifest_path}")
        print("Run render_heattest.py first.")
        sys.exit(1)
    return json.loads(manifest_path.read_text())


# -----------------------------------------------------------------------
# Analysis 1: PSNR monotonicity
# -----------------------------------------------------------------------

def analyze_psnr_monotonicity(render_dir: Path, trajectories: list[dict]) -> dict:
    """Check that psnr(step_i, final) increases as step index increases."""
    print("\n=== Analysis 1: PSNR Monotonicity ===")

    results = []
    all_monotonic = True

    for traj in trajectories:
        traj_dir = render_dir / traj["output_dir"]
        final_path = traj_dir / "final.png"
        if not final_path.exists():
            continue

        final_img = load_image_array(final_path)

        # Collect step PNGs (exclude "final")
        step_labels = sorted(
            s for s in traj["rendered_steps"]
            if s.startswith("step_")
        )
        if not step_labels:
            continue

        psnr_values = []
        for step_label in step_labels:
            step_path = traj_dir / f"{step_label}.png"
            if not step_path.exists():
                continue
            step_img = load_image_array(step_path)
            p = psnr(step_img, final_img)
            step_idx = int(step_label.split("_")[1])
            psnr_values.append((step_idx, p))

        psnr_values.sort(key=lambda x: x[0])

        # Check monotonicity
        is_monotonic = True
        for j in range(1, len(psnr_values)):
            if psnr_values[j][1] < psnr_values[j - 1][1]:
                is_monotonic = False
                break

        if not is_monotonic:
            all_monotonic = False

        traj_result = {
            "traj_id": traj["traj_id"],
            "psnr_by_step": {str(idx): round(p, 2) for idx, p in psnr_values},
            "is_monotonic": is_monotonic,
        }
        results.append(traj_result)

        status = "PASS" if is_monotonic else "FAIL"
        first_p = psnr_values[0][1] if psnr_values else 0
        last_p = psnr_values[-1][1] if psnr_values else 0
        print(f"  traj_{traj['traj_id']:06d}: {status}  "
              f"PSNR {first_p:.1f} -> {last_p:.1f} dB "
              f"({len(psnr_values)} steps)")

    status = "PASS" if all_monotonic else "FAIL"
    print(f"  Overall: {status} ({sum(r['is_monotonic'] for r in results)}"
          f"/{len(results)} monotonic)")

    return {
        "all_monotonic": all_monotonic,
        "trajectories": results,
    }


# -----------------------------------------------------------------------
# Analysis 2: Naturalness statistics
# -----------------------------------------------------------------------

def analyze_naturalness(render_dir: Path, trajectories: list[dict]) -> dict:
    """Run naturalness_report on all final images."""
    print("\n=== Analysis 2: Naturalness Statistics ===")

    results = []
    flags = []

    for traj in trajectories:
        traj_dir = render_dir / traj["output_dir"]
        final_path = traj_dir / "final.png"
        if not final_path.exists():
            continue

        img = load_image_array(final_path)
        report = naturalness_report(img)
        report["traj_id"] = traj["traj_id"]
        results.append(report)

        # Flag checks
        traj_flags = []
        if report["spectral_slope"] > -0.5:
            traj_flags.append(f"spectral_slope={report['spectral_slope']:.2f} (> -0.5)")
        if report["mean_entropy"] > 7.5:
            traj_flags.append(f"mean_entropy={report['mean_entropy']:.2f} (> 7.5)")

        status = "FLAG" if traj_flags else "OK"
        print(f"  traj_{traj['traj_id']:06d}: {status}  "
              f"slope={report['spectral_slope']:.2f}  "
              f"entropy={report['mean_entropy']:.2f}  "
              f"laplacian={report['laplacian_variance']:.0f}")

        if traj_flags:
            flags.append({"traj_id": traj["traj_id"], "flags": traj_flags})

    all_pass = len(flags) == 0
    status = "PASS" if all_pass else f"FLAG ({len(flags)} flagged)"
    print(f"  Overall: {status}")

    return {
        "all_pass": all_pass,
        "flagged": flags,
        "reports": results,
    }


# -----------------------------------------------------------------------
# Analysis 3: Color histogram clustering
# -----------------------------------------------------------------------

def analyze_histogram_clustering(render_dir: Path, trajectories: list[dict]) -> dict:
    """Compute pairwise histogram cosine similarity on finals.

    Group by prompt and seed. Test: same-prompt > same-seed similarity.
    """
    print("\n=== Analysis 3: Color Histogram Clustering ===")

    # Compute histograms for all finals
    hists = {}
    meta_lookup = {}
    for traj in trajectories:
        traj_dir = render_dir / traj["output_dir"]
        final_path = traj_dir / "final.png"
        if not final_path.exists():
            continue

        img = load_image_array(final_path)
        h = color_histogram(img)
        tid = traj["traj_id"]
        hists[tid] = h
        meta_lookup[tid] = traj

    tids = sorted(hists.keys())
    n = len(tids)

    if n < 2:
        print("  Not enough trajectories for clustering.")
        return {"n_trajectories": n, "skip": True}

    # Pairwise similarity matrix
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = histogram_cosine_sim(hists[tids[i]], hists[tids[j]])

    # Group pairs by relationship
    same_prompt_sims = []
    same_seed_sims = []
    other_sims = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = sim_matrix[i, j]
            ti = meta_lookup[tids[i]]
            tj = meta_lookup[tids[j]]

            same_prompt = (ti.get("prompt") == tj.get("prompt")
                           and ti.get("prompt") is not None)
            same_seed = (ti.get("seed") == tj.get("seed")
                         and ti.get("seed") is not None)

            if same_prompt:
                same_prompt_sims.append(sim)
            elif same_seed:
                same_seed_sims.append(sim)
            else:
                other_sims.append(sim)

    mean_prompt = float(np.mean(same_prompt_sims)) if same_prompt_sims else None
    mean_seed = float(np.mean(same_seed_sims)) if same_seed_sims else None
    mean_other = float(np.mean(other_sims)) if other_sims else None

    # Primary test: same-prompt > same-seed. Fallback when no same-seed
    # pairs exist (unique seeds per trajectory): same-prompt > other.
    if mean_prompt is not None and mean_seed is not None:
        prompt_gt_seed = mean_prompt > mean_seed
    elif mean_prompt is not None and mean_other is not None:
        prompt_gt_seed = mean_prompt > mean_other
    else:
        prompt_gt_seed = None

    if mean_prompt is not None:
        print(f"  Same-prompt pairs: {len(same_prompt_sims)}, "
              f"mean sim = {mean_prompt:.4f}")
    else:
        print(f"  Same-prompt pairs: 0")
    if mean_seed is not None:
        print(f"  Same-seed pairs: {len(same_seed_sims)}, "
              f"mean sim = {mean_seed:.4f}")
    else:
        print(f"  Same-seed pairs: 0")
    if mean_other is not None:
        print(f"  Other pairs: {len(other_sims)}, "
              f"mean sim = {mean_other:.4f}")
    else:
        print(f"  Other pairs: 0")

    if prompt_gt_seed is not None:
        baseline = "same-seed" if mean_seed is not None else "other"
        status = "PASS" if prompt_gt_seed else "FAIL"
        print(f"  same-prompt > {baseline}: {status}")
    else:
        print("  Insufficient data for clustering comparison")

    return {
        "n_trajectories": n,
        "same_prompt_pairs": len(same_prompt_sims),
        "same_seed_pairs": len(same_seed_sims),
        "other_pairs": len(other_sims),
        "mean_same_prompt_sim": mean_prompt,
        "mean_same_seed_sim": mean_seed,
        "mean_other_sim": mean_other,
        "prompt_gt_seed": prompt_gt_seed,
        "sim_matrix": sim_matrix.tolist(),
        "traj_ids": tids,
    }


# -----------------------------------------------------------------------
# Analysis 4: Parent-child correlation
# -----------------------------------------------------------------------

def analyze_parent_child(render_dir: Path, trajectories: list[dict]) -> dict:
    """Compare parent's sampled step vs child's step_00, parent vs child finals."""
    print("\n=== Analysis 4: Parent-Child Correlation ===")

    # Build lookup
    traj_by_id = {t["traj_id"]: t for t in trajectories}

    pairs = []
    for traj in trajectories:
        parent_id = traj.get("parent_traj_id")
        if parent_id is None or parent_id not in traj_by_id:
            continue
        parent = traj_by_id[parent_id]
        pairs.append((parent, traj))

    if not pairs:
        print("  No parent-child pairs found.")
        return {"n_pairs": 0, "skip": True}

    results = []
    for parent, child in pairs:
        parent_dir = render_dir / parent["output_dir"]
        child_dir = render_dir / child["output_dir"]

        # Parent's sampled step vs child's step_00
        parent_step = child.get("parent_step", "step_00")
        parent_step_path = parent_dir / f"{parent_step}.png"
        child_step0_path = child_dir / "step_00.png"

        step_sim = None
        if parent_step_path.exists() and child_step0_path.exists():
            p_img = load_image_array(parent_step_path)
            c_img = load_image_array(child_step0_path)
            h_p = color_histogram(p_img)
            h_c = color_histogram(c_img)
            step_sim = histogram_cosine_sim(h_p, h_c)

        # Parent final vs child final
        parent_final_path = parent_dir / "final.png"
        child_final_path = child_dir / "final.png"

        final_sim = None
        final_psnr_val = None
        if parent_final_path.exists() and child_final_path.exists():
            p_final = load_image_array(parent_final_path)
            c_final = load_image_array(child_final_path)
            h_pf = color_histogram(p_final)
            h_cf = color_histogram(c_final)
            final_sim = histogram_cosine_sim(h_pf, h_cf)
            final_psnr_val = psnr(p_final, c_final)

        pair_result = {
            "parent_traj_id": parent["traj_id"],
            "child_traj_id": child["traj_id"],
            "parent_step": parent_step,
            "step_histogram_sim": step_sim,
            "final_histogram_sim": final_sim,
            "final_psnr": final_psnr_val,
        }
        results.append(pair_result)

        step_str = f"{step_sim:.4f}" if step_sim is not None else "N/A"
        final_str = f"{final_sim:.4f}" if final_sim is not None else "N/A"
        psnr_str = f"{final_psnr_val:.1f}" if final_psnr_val is not None else "N/A"
        print(f"  parent {parent['traj_id']:06d} -> child {child['traj_id']:06d}: "
              f"step_sim={step_str}  final_sim={final_str}  psnr={psnr_str}")

    # Expect divergence: finals should be less similar than steps
    step_sims = [r["step_histogram_sim"] for r in results if r["step_histogram_sim"] is not None]
    final_sims = [r["final_histogram_sim"] for r in results if r["final_histogram_sim"] is not None]

    mean_step_sim = float(np.mean(step_sims)) if step_sims else None
    mean_final_sim = float(np.mean(final_sims)) if final_sims else None

    if mean_step_sim is not None and mean_final_sim is not None:
        print(f"  Mean step sim: {mean_step_sim:.4f}")
        print(f"  Mean final sim: {mean_final_sim:.4f}")
    print(f"  Pairs analyzed: {len(results)}")

    return {
        "n_pairs": len(results),
        "pairs": results,
        "mean_step_sim": mean_step_sim,
        "mean_final_sim": mean_final_sim,
    }


# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------

def write_summary(render_dir: Path, report: dict) -> None:
    """Write analysis_report.json and analysis_summary.txt."""
    # JSON report
    report_path = render_dir / "analysis_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    # Text summary
    lines = []
    w = lines.append

    w("=" * 60)
    w("HEATTEST ANALYSIS SUMMARY")
    w("=" * 60)
    w("")

    # PSNR monotonicity
    psnr_result = report.get("psnr_monotonicity", {})
    psnr_pass = psnr_result.get("all_monotonic", False)
    n_mono = sum(1 for t in psnr_result.get("trajectories", []) if t.get("is_monotonic"))
    n_total = len(psnr_result.get("trajectories", []))
    w(f"1. PSNR Monotonicity: {'PASS' if psnr_pass else 'FAIL'} "
      f"({n_mono}/{n_total} monotonic)")

    # Naturalness
    nat_result = report.get("naturalness", {})
    nat_pass = nat_result.get("all_pass", False)
    n_flagged = len(nat_result.get("flagged", []))
    w(f"2. Naturalness: {'PASS' if nat_pass else f'FLAG ({n_flagged} flagged)'}")
    for flag in nat_result.get("flagged", []):
        w(f"   traj_{flag['traj_id']:06d}: {', '.join(flag['flags'])}")

    # Histogram clustering
    hist_result = report.get("histogram_clustering", {})
    if not hist_result.get("skip"):
        prompt_gt_seed = hist_result.get("prompt_gt_seed")
        if prompt_gt_seed is None:
            w("3. Histogram Clustering: SKIP (insufficient data)")
        else:
            w(f"3. Histogram Clustering: {'PASS' if prompt_gt_seed else 'FAIL'}")
        ms_prompt = hist_result.get("mean_same_prompt_sim")
        ms_seed = hist_result.get("mean_same_seed_sim")
        ms_other = hist_result.get("mean_other_sim")
        parts = []
        if ms_prompt is not None:
            parts.append(f"same-prompt={ms_prompt:.4f}")
        if ms_seed is not None:
            parts.append(f"same-seed={ms_seed:.4f}")
        if ms_other is not None:
            parts.append(f"other={ms_other:.4f}")
        if parts:
            w(f"   {'  '.join(parts)}")
    else:
        w("3. Histogram Clustering: SKIP (insufficient data)")

    # Parent-child
    pc_result = report.get("parent_child", {})
    if not pc_result.get("skip"):
        w(f"4. Parent-Child: {pc_result['n_pairs']} pairs analyzed")
        if pc_result.get("mean_step_sim") is not None:
            w(f"   mean_step_sim={pc_result['mean_step_sim']:.4f}  "
              f"mean_final_sim={pc_result['mean_final_sim']:.4f}")
    else:
        w("4. Parent-Child: SKIP (no pairs found)")

    # Overall
    w("")
    w("=" * 60)
    checks_passed = psnr_pass and nat_pass
    if hist_result.get("prompt_gt_seed") is not None:
        checks_passed = checks_passed and hist_result["prompt_gt_seed"]
    w(f"OVERALL: {'ALL CHECKS PASSED' if checks_passed else 'SOME CHECKS FAILED'}")
    w("=" * 60)

    text = "\n".join(lines)

    summary_path = render_dir / "analysis_summary.txt"
    summary_path.write_text(text)

    print(f"\n{text}")
    print(f"\nReport: {report_path}")
    print(f"Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze rendered heattest images (CPU only, no torch)")
    parser.add_argument("--render-dir", type=str,
                        default=r"F:\dox\repos\ai\futudiffu\heattest_renders")
    args = parser.parse_args()

    render_dir = Path(args.render_dir)
    manifest = load_manifest(render_dir)
    trajectories = manifest["trajectories"]

    print(f"Render dir: {render_dir}")
    print(f"Trajectories: {len(trajectories)}")
    print(f"Total PNGs: {manifest.get('total_pngs', '?')}")

    report = {}

    # Analysis 1: PSNR monotonicity
    report["psnr_monotonicity"] = analyze_psnr_monotonicity(render_dir, trajectories)

    # Analysis 2: Naturalness
    report["naturalness"] = analyze_naturalness(render_dir, trajectories)

    # Analysis 3: Histogram clustering
    report["histogram_clustering"] = analyze_histogram_clustering(render_dir, trajectories)

    # Analysis 4: Parent-child correlation
    report["parent_child"] = analyze_parent_child(render_dir, trajectories)

    # Write summary
    write_summary(render_dir, report)

    return 0


if __name__ == "__main__":
    sys.exit(main())

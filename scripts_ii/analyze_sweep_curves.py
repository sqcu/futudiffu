r"""Post-facto analysis of sweep training curves.

Reads training_curve.json from each probe directory in a sweep output,
computes smoothed metrics (EMA loss, running mean accuracy, loss derivatives,
sliding window variance), and writes per-probe + summary analysis to JSON.

Works on both v1 and v2 sweep data (v1 lacks `pre_clip_grad_norm` and `lr`
fields -- these are simply omitted from analysis when absent).

Usage:
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\analyze_sweep_curves.py ^
      --sweep-dir F:\dox\repos\ai\futudiffu\rtheta_sweep_output_v2

  # Or analyze v1 data:
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\analyze_sweep_curves.py ^
      --sweep-dir F:\dox\repos\ai\futudiffu\rtheta_sweep_output
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src_ii.stats import finite_differences, running_average, sliding_std


def compute_ema(values: list[float], alpha: float = 0.1) -> list[float]:
    """Exponential moving average. First value seeds the EMA."""
    if not values:
        return []
    ema = [values[0]]
    for v in values[1:]:
        ema.append(alpha * v + (1.0 - alpha) * ema[-1])
    return ema


def analyze_probe(probe_dir: Path) -> dict | None:
    """Analyze a single probe's training curve. Returns None if no data."""
    curve_path = probe_dir / "training_curve.json"
    if not curve_path.exists():
        return None

    with open(curve_path) as f:
        curve = json.load(f)

    if not curve:
        return None

    n_steps = len(curve)
    probe_name = probe_dir.name

    # Extract raw series
    losses = [e["loss"] for e in curve]
    bt_losses = [e.get("loss", e.get("bt_loss", 0.0)) for e in curve]
    acc_pink = [e.get("accuracy_pinkify", 0.0) for e in curve]
    acc_tnt = [e.get("accuracy_thisnotthat", 0.0) for e in curve]
    grad_norms = [e["grad_norm"] for e in curve]
    times = [e["time_s"] for e in curve]

    # Optional v2 fields
    has_pre_clip = "pre_clip_grad_norm" in curve[0]
    has_lr = "lr" in curve[0]
    pre_clip_norms = [e["pre_clip_grad_norm"] for e in curve] if has_pre_clip else None
    lr_schedule = [e["lr"] for e in curve] if has_lr else None

    # EMA (alpha=0.1)
    ema_loss = compute_ema(losses, alpha=0.1)
    ema_norm_loss = compute_ema(bt_losses, alpha=0.1)

    # Running mean accuracy
    running_mean_pink = running_average(acc_pink)
    running_mean_tnt = running_average(acc_tnt)

    # Finite differences of EMA loss (d_loss/d_step)
    d_ema_loss = finite_differences(ema_loss)

    # Sliding window std (window=20)
    sliding_std_loss = sliding_std(losses, window=20)

    # Summary scalars
    mean_acc_pink = sum(acc_pink) / n_steps
    mean_acc_tnt = sum(acc_tnt) / n_steps

    # Tail statistics (last 20 steps)
    tail_n = min(20, n_steps)
    tail_losses = losses[-tail_n:]
    tail_mean = sum(tail_losses) / tail_n
    tail_std = math.sqrt(sum((x - tail_mean) ** 2 for x in tail_losses) / tail_n)

    tail_norm = bt_losses[-tail_n:]
    tail_norm_mean = sum(tail_norm) / tail_n

    # Mean d_ema_loss over last 20 steps (convergence rate)
    d_ema_tail = d_ema_loss[-tail_n:] if len(d_ema_loss) >= tail_n else d_ema_loss
    mean_d_ema_tail = sum(d_ema_tail) / len(d_ema_tail) if d_ema_tail else 0.0

    # Min loss and step at which it occurred
    min_loss = min(losses)
    min_loss_step = losses.index(min_loss)

    # Check for divergence: is the loss monotonically increasing in the last quarter?
    quarter_n = max(n_steps // 4, 2)
    last_quarter_ema = ema_loss[-quarter_n:]
    diverging = all(
        last_quarter_ema[i + 1] > last_quarter_ema[i]
        for i in range(len(last_quarter_ema) - 1)
    )

    result = {
        "probe_name": probe_name,
        "n_steps": n_steps,
        "format_version": "v2" if has_pre_clip else "v1",
        # Summary scalars
        "mean_acc_pinkify": round(mean_acc_pink, 4),
        "mean_acc_thisnotthat": round(mean_acc_tnt, 4),
        "final_running_mean_pink": round(running_mean_pink[-1], 4),
        "final_running_mean_tnt": round(running_mean_tnt[-1], 4),
        "ema_loss_final": round(ema_loss[-1], 6),
        "ema_norm_loss_final": round(ema_norm_loss[-1], 6),
        "final_loss": round(losses[-1], 6),
        "min_loss": round(min_loss, 6),
        "min_loss_step": min_loss_step,
        "loss_std_last_20": round(tail_std, 6),
        "loss_mean_last_20": round(tail_mean, 6),
        "norm_loss_mean_last_20": round(tail_norm_mean, 6),
        "mean_d_ema_loss_last_20": round(mean_d_ema_tail, 8),
        "diverging_last_quarter": diverging,
        "mean_grad_norm": round(sum(grad_norms) / n_steps, 6),
        "mean_step_time_s": round(sum(times) / n_steps, 3),
        "total_time_s": round(sum(times), 1),
        # Per-step time series (for detailed inspection)
        "series": {
            "ema_loss": [round(v, 6) for v in ema_loss],
            "ema_norm_loss": [round(v, 6) for v in ema_norm_loss],
            "running_mean_acc_pinkify": [round(v, 4) for v in running_mean_pink],
            "running_mean_acc_thisnotthat": [round(v, 4) for v in running_mean_tnt],
            "d_ema_loss_d_step": [round(v, 8) for v in d_ema_loss],
            "sliding_std_loss_w20": [
                round(v, 6) if v is not None else None for v in sliding_std_loss
            ],
        },
    }

    if has_pre_clip:
        result["mean_pre_clip_grad_norm"] = round(
            sum(pre_clip_norms) / n_steps, 4
        )
        clip_ratio = sum(
            1 for p, c in zip(pre_clip_norms, grad_norms)
            if p > c * 1.01  # clipped if pre_clip > post_clip by > 1%
        ) / n_steps
        result["grad_clip_ratio"] = round(clip_ratio, 4)

    if has_lr:
        result["lr_range"] = [lr_schedule[0], lr_schedule[-1]]

    return result


def analyze_sweep(sweep_dir: Path) -> dict:
    """Analyze all probes in a sweep directory."""
    probe_dirs = sorted(
        [d for d in sweep_dir.iterdir() if d.is_dir() and (d / "training_curve.json").exists()]
    )

    if not probe_dirs:
        print(f"  No probes with training_curve.json found in {sweep_dir}")
        return {"sweep_dir": str(sweep_dir), "probes": [], "n_probes": 0}

    print(f"  Found {len(probe_dirs)} probes in {sweep_dir}")

    probes = []
    for pd in probe_dirs:
        print(f"    Analyzing {pd.name}...", end=" ")
        result = analyze_probe(pd)
        if result is not None:
            probes.append(result)
            print(f"OK ({result['n_steps']} steps, "
                  f"mean_pink={result['mean_acc_pinkify']:.3f}, "
                  f"mean_tnt={result['mean_acc_thisnotthat']:.3f}, "
                  f"ema_loss={result['ema_loss_final']:.4f})")
        else:
            print("SKIP (empty)")

    # Cross-probe comparison
    comparison = {}
    if probes:
        best_ema = min(probes, key=lambda p: p["ema_loss_final"])
        best_min = min(probes, key=lambda p: p["min_loss"])
        best_pink = max(probes, key=lambda p: p["mean_acc_pinkify"])
        best_tnt = max(probes, key=lambda p: p["mean_acc_thisnotthat"])
        lowest_noise = min(probes, key=lambda p: p["loss_std_last_20"])

        comparison = {
            "best_ema_loss": {
                "probe": best_ema["probe_name"],
                "value": best_ema["ema_loss_final"],
            },
            "best_min_loss": {
                "probe": best_min["probe_name"],
                "value": best_min["min_loss"],
            },
            "best_mean_acc_pinkify": {
                "probe": best_pink["probe_name"],
                "value": best_pink["mean_acc_pinkify"],
            },
            "best_mean_acc_thisnotthat": {
                "probe": best_tnt["probe_name"],
                "value": best_tnt["mean_acc_thisnotthat"],
            },
            "lowest_loss_noise": {
                "probe": lowest_noise["probe_name"],
                "value": lowest_noise["loss_std_last_20"],
            },
            "any_diverging": any(p["diverging_last_quarter"] for p in probes),
            "diverging_probes": [
                p["probe_name"] for p in probes if p["diverging_last_quarter"]
            ],
        }

    summary = {
        "sweep_dir": str(sweep_dir),
        "n_probes": len(probes),
        "comparison": comparison,
        "probes": probes,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Post-facto analysis of sweep training curves"
    )
    parser.add_argument(
        "--sweep-dir",
        type=str,
        required=True,
        help="Path to sweep output directory (contains lr_* probe subdirs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: <sweep-dir>/post_facto_analysis.json)",
    )
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"ERROR: sweep directory does not exist: {sweep_dir}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else sweep_dir / "post_facto_analysis.json"

    print(f"Analyzing sweep: {sweep_dir}")
    summary = analyze_sweep(sweep_dir)

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAnalysis written to {output_path}")

    # Print comparison table
    if summary["comparison"]:
        c = summary["comparison"]
        print(f"\n{'=' * 70}")
        print(f"  CROSS-PROBE COMPARISON")
        print(f"{'=' * 70}")
        print(f"  Best EMA loss:       {c['best_ema_loss']['probe']} = {c['best_ema_loss']['value']:.6f}")
        print(f"  Best min loss:       {c['best_min_loss']['probe']} = {c['best_min_loss']['value']:.6f}")
        print(f"  Best mean acc pink:  {c['best_mean_acc_pinkify']['probe']} = {c['best_mean_acc_pinkify']['value']:.4f}")
        print(f"  Best mean acc tnt:   {c['best_mean_acc_thisnotthat']['probe']} = {c['best_mean_acc_thisnotthat']['value']:.4f}")
        print(f"  Lowest loss noise:   {c['lowest_loss_noise']['probe']} = {c['lowest_loss_noise']['value']:.6f}")
        if c["diverging_probes"]:
            print(f"  DIVERGING: {', '.join(c['diverging_probes'])}")
        else:
            print(f"  No diverging probes detected.")

    # Per-probe summary table
    if summary["probes"]:
        print(f"\n{'Probe':<20s} {'Steps':>6s} {'EMALoss':>9s} {'MinLoss':>9s} "
              f"{'MnPink':>7s} {'MnTNT':>7s} {'LStd20':>8s} {'dEMA/ds':>10s} {'Divg':>5s}")
        print("-" * 95)
        for p in sorted(summary["probes"], key=lambda x: x["ema_loss_final"]):
            print(f"{p['probe_name']:<20s} {p['n_steps']:6d} "
                  f"{p['ema_loss_final']:9.4f} {p['min_loss']:9.4f} "
                  f"{p['mean_acc_pinkify']:7.3f} {p['mean_acc_thisnotthat']:7.3f} "
                  f"{p['loss_std_last_20']:8.4f} "
                  f"{p['mean_d_ema_loss_last_20']:10.6f} "
                  f"{'YES' if p['diverging_last_quarter'] else 'no':>5s}")


if __name__ == "__main__":
    main()

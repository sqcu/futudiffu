"""CLI tool for comparing two tensor stream recordings.

Usage:
    python -m futudiffu.compare <stream_a> <stream_b> [--config B]

Loads two recordings, runs validation comparisons per stage, prints
a formatted report with per-stage PASS/FAIL, and exits 0 (all pass) or 1.
"""

from __future__ import annotations

import argparse
import sys

from .tensor_stream import compare_streams


def format_report(results: dict) -> str:
    """Format comparison results into a human-readable report."""
    lines = []
    config = results["config"]
    lines.append(f"Tensor Stream Comparison Report (Config {config})")
    lines.append("=" * 60)

    stages = results["stages"]
    for stage_name in sorted(stages.keys()):
        stage = stages[stage_name]
        passed = stage.get("pass", False)
        status = "PASS" if passed else "FAIL"
        line = f"  [{status}] {stage_name}"

        # Add diagnostic details
        details = []
        if "reason" in stage:
            details.append(stage["reason"])
        if "mse" in stage:
            details.append(f"MSE={stage['mse']:.2e}")
        if "max_abs_diff" in stage:
            details.append(f"max_abs={stage['max_abs_diff']:.2e}")
        if "cosine_sim" in stage:
            details.append(f"cos={stage['cosine_sim']:.6f}")
        if "match" in stage and not passed:
            if "num_mismatched" in stage:
                details.append(f"{stage['num_mismatched']}/{stage['total_elements']} mismatched")

        # For dict stages (euler steps), summarize sub-keys
        if isinstance(stage, dict) and any(isinstance(v, dict) for v in stage.values()):
            for k, v in stage.items():
                if isinstance(v, dict) and k != "pass":
                    sub_pass = v.get("pass", v.get("match", False))
                    sub_status = "ok" if sub_pass else "FAIL"
                    sub_detail = ""
                    if "mse" in v:
                        sub_detail = f" MSE={v['mse']:.2e}"
                    elif "max_abs_diff" in v:
                        sub_detail = f" max_abs={v['max_abs_diff']:.2e}"
                    details.append(f"{k}:{sub_status}{sub_detail}")

        if details:
            line += f"  ({', '.join(details)})"
        lines.append(line)

    lines.append("-" * 60)
    overall = "ALL PASS" if results["all_pass"] else "SOME FAILURES"
    lines.append(f"Overall: {overall}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two tensor stream recordings from futudiffu/ComfyUI pipeline runs."
    )
    parser.add_argument("stream_a", help="Path to first stream directory")
    parser.add_argument("stream_b", help="Path to second stream directory")
    parser.add_argument(
        "--config", default="B", choices=["A", "B"],
        help="Validation config: B=golden (MSE), A=defective (cosine). Default: B",
    )

    args = parser.parse_args()
    results = compare_streams(args.stream_a, args.stream_b, config=args.config)

    print(format_report(results))
    sys.exit(0 if results["all_pass"] else 1)


if __name__ == "__main__":
    main()

r"""Wallclock benchmark for config-driven training via GPU server HTTP API.

Submits compact BTRM and DDGRPO configs to POST /training/start,
streams SSE events until completion, records per-step and total
wallclock times, writes a run report to disk.

This is the canonical way to benchmark training after the script purge:
no local model loading, no variant forward paths. Everything flows
through the server's TrainingOrchestrator.

Usage:
    .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\bench_config_driven.py
    .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\bench_config_driven.py --server http://172.26.160.1:8787
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    import httpx
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx"])
    import httpx


BTRM_CONFIG = {
    "run_type": "btrm",
    "dataset_path": "multi_res_trajectories",
    "n_steps": 5,
    "lr": 3e-4,
    "head_names": ["pinkify", "thisnotthat"],
    "pref_keys": ["pinkify_pref", "thisnotthat_pref"],
    "macrobatch_budget": 3.0,
    "adapter_name": "rtheta",
    "adapter_rank": 8,
    "adapter_alpha": 16.0,
    "lr_schedule": "warmup_cosine",
    "warmup_steps": 1,
    "gradient_checkpointing": True,
    "max_grad_norm": 0.1,
    "clean_fraction": 0.8,
    "output_dir": "training_output/bench_btrm_5step",
}

DDGRPO_CONFIG = {
    "run_type": "ddgrpo",
    "btrm_checkpoint": "training_output/bench_btrm_5step",
    "n_iters": 3,
    "prompts": ["a beautiful mountain landscape at sunset"],
    "n_rollouts_per_prompt": 2,
    "rollout_steps": 20,
    "eta_scale": 0.1,
    "policy_lr": 2e-5,
    "max_grad_norm": 0.1,
    "btrm_adapter_name": "rtheta",
    "policy_adapter_name": "policy_bench",
    "adapter_rank": 8,
    "adapter_alpha": 16.0,
    "init_b_std": 0.01,
    "advantage_threshold": 0.01,
    "resolution_budgets": [65536],
    "output_dir": "training_output/bench_ddgrpo_3iter",
}

OUTPUT_DIR = REPO_ROOT / "training_output" / "bench_reports"


def stream_training_run(
    client: httpx.Client,
    config: dict,
    label: str,
) -> dict:
    """Submit config, stream SSE events, return timing report."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Config: {json.dumps({k: v for k, v in config.items() if k != 'prompts'}, indent=2)[:500]}")

    wall_start = time.perf_counter()

    # Submit
    resp = client.post("/training/start", json=config, timeout=30.0)
    if resp.status_code != 200:
        print(f"  ERROR: {resp.status_code} {resp.text[:500]}")
        return {"status": "FAIL", "error": resp.text[:500]}

    result = resp.json()
    run_id = result["run_id"]
    stream_url = f"/training/stream/{run_id}"
    print(f"  Run ID: {run_id}")
    print(f"  Stream: {stream_url}")

    submit_time = time.perf_counter() - wall_start

    # Stream SSE events
    step_times = []
    step_metrics = []
    last_step_time = time.perf_counter()
    final_data = None

    with client.stream("GET", stream_url, timeout=600.0) as sse:
        buffer = ""
        for chunk in sse.iter_text():
            buffer += chunk
            while "\n\n" in buffer:
                message, buffer = buffer.split("\n\n", 1)
                event_type = None
                event_data = None
                for line in message.strip().split("\n"):
                    if line.startswith("event: "):
                        event_type = line[7:]
                    elif line.startswith("data: "):
                        try:
                            event_data = json.loads(line[6:])
                        except json.JSONDecodeError:
                            event_data = line[6:]

                if event_type is None or event_data is None:
                    continue

                now = time.perf_counter()

                if event_type == "step":
                    step_wall = now - last_step_time
                    last_step_time = now
                    step_times.append(step_wall)
                    step_metrics.append(event_data)
                    step_num = event_data.get("step", len(step_times))
                    loss = event_data.get("loss", event_data.get("bt_loss", "?"))
                    print(f"    Step {step_num}: {step_wall:.1f}s, loss={loss}")

                elif event_type == "complete":
                    final_data = event_data
                    print(f"  Complete: {event_data}")

                elif event_type == "error":
                    print(f"  ERROR: {event_data}")
                    return {
                        "status": "FAIL",
                        "error": str(event_data),
                        "wall_time_s": now - wall_start,
                    }

                elif event_type in ("checkpoint", "artifact_ready"):
                    print(f"    [{event_type}] {event_data.get('path', event_data.get('step', ''))}")

    wall_total = time.perf_counter() - wall_start

    report = {
        "label": label,
        "status": "PASS",
        "config": config,
        "wall_time_s": wall_total,
        "submit_time_s": submit_time,
        "n_steps": len(step_times),
        "step_times_s": step_times,
        "mean_step_time_s": sum(step_times) / len(step_times) if step_times else 0,
        "step_metrics": step_metrics,
        "final_data": final_data,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    print(f"\n  Wall time: {wall_total:.1f}s")
    print(f"  Steps: {len(step_times)}")
    if step_times:
        print(f"  Mean step: {report['mean_step_time_s']:.1f}s")
        print(f"  First step: {step_times[0]:.1f}s (includes compile/warmup)")
        if len(step_times) > 1:
            print(f"  Steady-state: {sum(step_times[1:])/(len(step_times)-1):.1f}s/step")

    return report


def main():
    parser = argparse.ArgumentParser(description="Config-driven training benchmark")
    parser.add_argument("--server", default="http://172.26.160.1:8787",
                        help="GPU server URL")
    parser.add_argument("--btrm-steps", type=int, default=5,
                        help="BTRM training steps (default 5)")
    parser.add_argument("--ddgrpo-iters", type=int, default=3,
                        help="DDGRPO iterations (default 3)")
    parser.add_argument("--skip-btrm", action="store_true",
                        help="Skip BTRM, only run DDGRPO")
    parser.add_argument("--skip-ddgrpo", action="store_true",
                        help="Skip DDGRPO, only run BTRM")
    args = parser.parse_args()

    BTRM_CONFIG["n_steps"] = args.btrm_steps
    DDGRPO_CONFIG["n_iters"] = args.ddgrpo_iters

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = httpx.Client(base_url=args.server, timeout=30.0)

    # Verify server health
    try:
        health = client.get("/health", timeout=5.0)
        print(f"Server: {args.server} — {health.json()}")
    except Exception as e:
        print(f"Cannot reach server at {args.server}: {e}")
        return 1

    reports = []
    bench_start = time.perf_counter()

    # Run 1: BTRM 5-step
    if not args.skip_btrm:
        btrm_report = stream_training_run(
            client, BTRM_CONFIG,
            f"BTRM {args.btrm_steps}-step training",
        )
        reports.append(btrm_report)

    # Run 2: DDGRPO 3-iter (uses BTRM checkpoint from run 1)
    if not args.skip_ddgrpo:
        ddgrpo_report = stream_training_run(
            client, DDGRPO_CONFIG,
            f"DDGRPO {args.ddgrpo_iters}-iter policy optimization",
        )
        reports.append(ddgrpo_report)

    bench_total = time.perf_counter() - bench_start

    # Summary
    summary = {
        "total_wall_time_s": bench_total,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "server": args.server,
        "runs": reports,
    }

    # Save report
    report_name = f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path = OUTPUT_DIR / report_name
    with open(str(report_path), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"  Total wall time: {bench_total:.1f}s")
    for r in reports:
        status = r.get("status", "?")
        label = r.get("label", "?")
        wall = r.get("wall_time_s", 0)
        steps = r.get("n_steps", 0)
        mean = r.get("mean_step_time_s", 0)
        print(f"  {label}: {status} — {wall:.1f}s ({steps} steps, {mean:.1f}s/step)")
    print(f"  Report: {report_path}")

    return 0 if all(r.get("status") == "PASS" for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())

r"""Generate BTRM training dataset via relay client API.

Thin CLI wrapper that submits a dataset generation plan to the server.
Supports JSON plan files and inline prompt specification.

Usage:
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\generate_btrm_dataset.py \
      --plan plans\scrimblo_only.json
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\generate_btrm_dataset.py \
      --plan plans\scrimblo_only.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src_ii.client_yeetums.bridge import InferenceBridge


def main():
    parser = argparse.ArgumentParser(description="Generate BTRM dataset via relay API")
    parser.add_argument("--plan", type=str, help="JSON generation plan file")
    parser.add_argument("--server", default="http://localhost:9090", help="BFF URL")
    parser.add_argument("--output-dir", type=str, default="btrm_dataset_v2")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    args = parser.parse_args()

    if args.plan:
        with open(args.plan) as f:
            plan = json.load(f)
    else:
        plan = {
            "prompts": ["a beautiful landscape"],
            "resolution_tiers": ["full"],
            "attention_backends": ["sdpa", "sage"],
            "n_steps": 30,
        }

    if args.dry_run:
        print("=== Dry Run ===")
        print(json.dumps(plan, indent=2))
        return

    config = {
        "run_type": "dataset_generation",
        "plan": plan,
        "output_dir": args.output_dir,
    }

    bridge = InferenceBridge(args.server)

    print(f"Starting dataset generation -> {args.output_dir}")
    result = bridge.start_training_run(config)
    run_id = result["run_id"]
    print(f"Run started: {run_id}")

    for event in bridge.stream_training_events(run_id):
        etype = event["type"]
        data = event["data"]

        if etype == "step":
            traj = data.get("trajectory", "?")
            total = data.get("total", "?")
            print(f"  trajectory {traj}/{total}")

        elif etype == "complete":
            print(f"Dataset generation complete: {data.get('elapsed_s', 0):.1f}s")
            break

        elif etype == "error":
            print(f"ERROR: {data.get('error', 'unknown')}", file=sys.stderr)
            sys.exit(1)

    bridge.close()


if __name__ == "__main__":
    main()

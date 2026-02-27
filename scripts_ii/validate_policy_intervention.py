r"""Policy intervention validation via relay client API.

Compares model outputs before and after policy adapter activation.
Submits config to the server for server-side comparison.

Usage:
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\validate_policy_intervention.py
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

DEFAULT_CONFIG = {
    "adapter_name": "policy_pinkify",
    "prompts": [
        "a beautiful mountain landscape at sunset",
        "a cat sitting on a windowsill",
    ],
    "seeds": [42, 137],
    "n_steps": 20,
    "width": 320,
    "height": 208,
    "output_dir": "validation_renders/policy_intervention",
}


def main():
    parser = argparse.ArgumentParser(description="Policy intervention validation via relay API")
    parser.add_argument("--config", type=str, help="JSON config file")
    parser.add_argument("--server", default="http://localhost:9090", help="BFF URL")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    bridge = InferenceBridge(args.server)
    config["run_type"] = "policy_intervention"

    print(f"Starting policy intervention validation")
    result = bridge.start_training_run(config)
    run_id = result["run_id"]
    print(f"Run started: {run_id}")

    for event in bridge.stream_training_events(run_id):
        etype = event["type"]
        data = event["data"]

        if etype == "step":
            print(f"  {data.get('phase', '?')}: {data.get('detail', '')}")

        elif etype == "complete":
            print(f"Validation complete: {data.get('elapsed_s', 0):.1f}s")
            if "diff_stats" in data:
                print(f"  diff stats: {json.dumps(data['diff_stats'], indent=2)}")
            break

        elif etype == "error":
            print(f"ERROR: {data.get('error', 'unknown')}", file=sys.stderr)
            sys.exit(1)

    bridge.close()


if __name__ == "__main__":
    main()

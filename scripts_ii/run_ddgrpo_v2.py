r"""DDGRPO v2: Policy optimization via relay client API.

Submits a DDGRPO config to the inference server and streams metrics.
All orchestration happens server-side via TrainingOrchestrator.

Usage:
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\run_ddgrpo_v2.py
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\run_ddgrpo_v2.py --config ddgrpo_config.json
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
    "btrm_checkpoint": "training_output/reward_function_run_tnt_v2",
    "n_iters": 100,
    "rollout_steps": 20,
    "eta_scale": 0.1,
    "policy_lr": 2e-5,
    "max_grad_norm": 0.1,
    "btrm_adapter_name": "rtheta",
    "policy_adapter_name": "policy_pinkify",
    "adapter_rank": 8,
    "adapter_alpha": 16.0,
    "init_b_std": 0.01,
    "prompts": [
        "a beautiful mountain landscape at sunset",
        "a cat sitting on a windowsill",
        "abstract geometric patterns",
        "a portrait of a person reading a book",
    ],
    "n_rollouts_per_prompt": 2,
    "advantage_threshold": 0.01,
    "resolution_budgets": [65536, 102400, 147456, 262144],
    "output_dir": "training_output/ddgrpo_v2",
}


def main():
    parser = argparse.ArgumentParser(description="DDGRPO policy optimization via relay API")
    parser.add_argument("--config", type=str, help="JSON config file (overrides defaults)")
    parser.add_argument("--server", default="http://localhost:9090", help="BFF URL")
    parser.add_argument("--n-iters", type=int, help="Override n_iters")
    parser.add_argument("--lr", type=float, help="Override policy_lr")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))
    if args.n_iters is not None:
        config["n_iters"] = args.n_iters
    if args.lr is not None:
        config["policy_lr"] = args.lr
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    bridge = InferenceBridge(args.server)
    config["run_type"] = "ddgrpo"

    print(f"Starting DDGRPO: {config['n_iters']} iters, lr={config['policy_lr']}")
    result = bridge.start_training_run(config)
    run_id = result["run_id"]
    print(f"Run started: {run_id} -> {result.get('stream_url', '')}")

    for event in bridge.stream_training_events(run_id):
        etype = event["type"]
        data = event["data"]

        if etype == "step":
            it = data.get("iteration", data.get("step", "?"))
            n = data.get("n_steps", "?")
            reward = data.get("mean_reward", "?")
            gnorm = data.get("grad_norm", "?")
            print(f"  iter {it}/{n}  reward={reward}  grad_norm={gnorm}")

        elif etype == "complete":
            print(f"DDGRPO complete: {data.get('elapsed_s', 0):.1f}s")
            print(f"Output: {data.get('output_dir', '')}")
            break

        elif etype == "error":
            print(f"ERROR: {data.get('error', 'unknown')}", file=sys.stderr)
            sys.exit(1)

    bridge.close()


if __name__ == "__main__":
    main()

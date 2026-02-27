r"""Run 03: BTRM training via relay client API.

Submits a BTRM training config to the inference server and streams metrics.
All orchestration happens server-side via TrainingOrchestrator.

Usage:
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\run03_btrm_training.py
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\run03_btrm_training.py --config config.json
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
    "dataset_path": "multi_res_trajectories",
    "n_steps": 100,
    "lr": 3e-4,
    "head_names": ["pinkify", "thisnotthat"],
    "pref_keys": ["pinkify_pref", "thisnotthat_pref"],
    "gradient_checkpointing": True,
    "max_grad_norm": 0.1,
    "warmup_steps": 5,
    "lr_schedule": "warmup_cosine",
    "macrobatch_budget": 3.0,
    "megapixel_flops_fraction": 0.33,
    "checkpoint_steps": [25, 50, 75, 100, 125],
    "adapter_name": "rtheta",
    "adapter_rank": 8,
    "adapter_alpha": 16.0,
    "clean_fraction": 0.8,
    "output_dir": "training_output/run03",
}


def main():
    parser = argparse.ArgumentParser(description="BTRM training via relay API")
    parser.add_argument("--config", type=str, help="JSON config file (overrides defaults)")
    parser.add_argument("--server", default="http://localhost:9090", help="BFF URL")
    parser.add_argument("--n-steps", type=int, help="Override n_steps")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))
    if args.n_steps is not None:
        config["n_steps"] = args.n_steps
    if args.lr is not None:
        config["lr"] = args.lr
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    bridge = InferenceBridge(args.server)
    config["run_type"] = "btrm"

    print(f"Starting BTRM training: {config['n_steps']} steps, lr={config['lr']}")
    result = bridge.start_training_run(config)
    run_id = result["run_id"]
    print(f"Run started: {run_id} -> {result.get('stream_url', '')}")

    for event in bridge.stream_training_events(run_id):
        etype = event["type"]
        data = event["data"]

        if etype == "step":
            step = data.get("step", "?")
            n = data.get("n_steps", "?")
            loss = data.get("loss", "?")
            acc = data.get("per_head_accuracy", {})
            print(f"  step {step}/{n}  loss={loss}  acc={acc}")

        elif etype == "checkpoint":
            print(f"  checkpoint saved at step {data.get('step')}")

        elif etype == "complete":
            print(f"Training complete: {data.get('elapsed_s', 0):.1f}s")
            print(f"Output: {data.get('output_dir', '')}")
            break

        elif etype == "error":
            print(f"ERROR: {data.get('error', 'unknown')}", file=sys.stderr)
            sys.exit(1)

    bridge.close()


if __name__ == "__main__":
    main()

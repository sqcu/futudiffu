"""Replicate LoRA + BTRM head state from one server to another.

Used mid-session when training was launched on one GPU but needs to expand
to multiple GPUs without losing trained adapter weights.

Usage:
    python replicate_server_state.py --source 5555 --target 5556
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from futudiffu.client import InferenceClient


def main():
    parser = argparse.ArgumentParser(description="Replicate server state")
    parser.add_argument("--source", type=int, required=True, help="Source server port")
    parser.add_argument("--target", type=int, required=True, help="Target server port")
    args = parser.parse_args()

    src = InferenceClient(f"tcp://localhost:{args.source}")
    tgt = InferenceClient(f"tcp://localhost:{args.target}")

    s0 = src.status()
    s1 = tgt.status()
    print(f"Source (:{args.source}): phase={s0.get('phase')}")
    print(f"Target (:{args.target}): phase={s1.get('phase')}")

    # Get adapter info from source
    adapters = s0.get("adapters", {})
    print(f"Source adapters: {list(adapters.keys())}")

    # Replicate each adapter
    for name, info in adapters.items():
        print(f"\n--- Replicating adapter '{name}' ---")
        rank = info.get("rank", 8)
        alpha = info.get("alpha", 16.0)
        scale = info.get("scale", 1.0)
        n_modules = info.get("n_modules", 0)

        # Determine layer_indices from module count
        # rtheta: 6 adapters = layers 28-29 (3 targets per layer x 2 layers)
        # ptheta: 102 adapters = all layers
        if n_modules <= 10:
            layer_indices = [28, 29]
        else:
            layer_indices = None  # all layers

        # Dump weights from source
        sd = src.get_lora_state_dict(name)
        print(f"  Got {len(sd)} tensors from source")

        # Inject on target
        n = tgt.inject_lora(name, rank=rank, alpha=alpha, layer_indices=layer_indices)
        print(f"  Injected on target: {n} adapters")

        # Load weights
        tgt.update_lora_weights(name, sd)
        print(f"  Loaded trained weights")

        # Set scale and freeze state
        frozen = scale == 0.0 or name == "rtheta"
        tgt.set_adapter_config(name, frozen=frozen, scale=scale)
        print(f"  Config: scale={scale}, frozen={frozen}")

    # Replicate BTRM head
    print("\n--- Replicating BTRM head ---")
    btrm_meta = tgt.inject_btrm_head(
        head_names=["scrimble", "scrongle"],
        logit_cap=10.0,
        lr=0.001,
    )
    print(f"  Injected BTRM head on target: {btrm_meta}")

    # Find and load trained BTRM weights from checkpoint
    repo_root = Path(__file__).resolve().parent.parent
    ckpt_dirs = sorted(repo_root.glob("training_output/btrm_step_*"))
    if ckpt_dirs:
        latest = ckpt_dirs[-1]
        manifests = sorted(latest.glob("dump_manifest_*.json"))
        if manifests:
            with open(manifests[-1]) as f:
                manifest = json.load(f)
            btrm_info = manifest.get("btrm_head")
            if btrm_info and btrm_info.get("path"):
                from safetensors.torch import load_file
                btrm_sd = load_file(btrm_info["path"])
                tgt.update_btrm_head(btrm_sd)
                print(f"  Loaded BTRM head weights from {btrm_info['path']}")
            else:
                print("  WARNING: No BTRM head in checkpoint manifest")
        else:
            print(f"  WARNING: No manifest in {latest}")
    else:
        print("  WARNING: No checkpoints found in training_output/")

    print("\n--- Warmup target ---")
    tgt.warmup(attention_backend="sdpa")
    print("  Warmup complete")

    src.close()
    tgt.close()
    print("\nDone. Both servers have matching state.")


if __name__ == "__main__":
    main()

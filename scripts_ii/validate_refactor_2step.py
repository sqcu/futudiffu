r"""2-step validation run: exercises all refactored shared modules.

Tests the full BTRM training pipeline with N_STEPS=2 to verify
that model_paths, dataset_io, training_setup, and the decomposed
btrm_training.py all wire together correctly.

Execution:
  PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe ^
      F:\dox\repos\ai\futudiffu\scripts_ii\validate_refactor_2step.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

# --- Shared modules under test ---
from src_ii.model_paths import FP8_PATH, TE_PATH, VAE_PATH, TOKENIZER_PATH
from src_ii.training_setup import (
    encode_training_prompts,
    build_dataset_positions,
    load_training_backbone,
)
from src_ii.dataset_io import make_load_latent_fn

DATASET_DIR = REPO_ROOT / "multi_res_trajectories"
OUTPUT_DIR = REPO_ROOT / "training_output" / "refactor_validation_2step"

N_STEPS = 2
MACROBATCH_BUDGET = 3.0
LR = 3e-4
HEAD_NAMES = ("pinkify", "thisnotthat")
PREF_KEYS = ("pinkify_pref", "thisnotthat_pref")


def main():
    wall_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("=" * 60)
    print("  REFACTOR VALIDATION: 2-step BTRM training")
    print("=" * 60)

    # Phase 1: Dataset loading via training_setup
    print("\n  Phase 1: build_dataset_positions()")
    reader, positions, sampler, traj_ids = build_dataset_positions(
        DATASET_DIR, clean_fraction=0.8,
    )
    print(f"  OK: {len(traj_ids)} trajectories, {len(positions)} positions")

    # Phase 2: TE encoding via training_setup
    print("\n  Phase 2: encode_training_prompts()")
    prompt_cache = encode_training_prompts(
        reader, traj_ids, TOKENIZER_PATH, TE_PATH, device=device, dtype=dtype,
    )
    print(f"  OK: {len(prompt_cache)} unique prompts cached")

    # Phase 3: load_latent_fn via dataset_io
    print("\n  Phase 3: make_load_latent_fn()")
    load_latent_fn = make_load_latent_fn(reader, prompt_cache, device=device, dtype=dtype)

    # Verify it works
    test_key = (traj_ids[0], "final")
    lat, ts, cond, nt, rope = load_latent_fn(test_key)
    assert ts.item() == 0.0, f"Final sigma must be 0.0, got {ts.item()}"
    assert rope is None, "5th return value must be None"
    print(f"  OK: latent={lat.shape}, sigma=0.0, cond={cond.shape}, rope=None")
    del lat, ts, cond

    # Phase 4: Backbone loading via training_setup
    print("\n  Phase 4: load_training_backbone()")
    raw_model, optimizer, head_names_loaded = load_training_backbone(
        FP8_PATH, device=device, dtype=dtype, lr=LR,
    )
    print(f"  OK: heads={head_names_loaded}")

    # Phase 5: Sigma-based preference function (simplest)
    def preference_fn(pair: dict) -> dict:
        prefs = {}
        for pref_key in PREF_KEYS:
            sigma_a = pair.get("sigma_a", 0.5)
            sigma_b = pair.get("sigma_b", 0.5)
            if sigma_a < sigma_b - 0.001:
                prefs[pref_key] = 1
            elif sigma_b < sigma_a - 0.001:
                prefs[pref_key] = -1
            else:
                prefs[pref_key] = 0
        return prefs

    # Phase 6: 2-step training via refactored btrm_training
    print("\n  Phase 6: train_btrm_differentiable() — 2 steps")
    from src_ii.btrm_training import train_btrm_differentiable

    t0 = time.perf_counter()
    training_curve = train_btrm_differentiable(
        model=raw_model,
        pair_sampler=sampler,
        load_latent_fn=load_latent_fn,
        preference_fn=preference_fn,
        n_steps=N_STEPS,
        lr=LR,
        head_names=HEAD_NAMES,
        pref_keys=PREF_KEYS,
        gradient_checkpointing=True,
        max_grad_norm=0.1,
        log_interval=1,
        warmup_steps=1,
        lr_schedule="warmup_cosine",
        macrobatch_budget=MACROBATCH_BUDGET,
        macrobatch_cross_resolution=True,
        output_dir=str(OUTPUT_DIR),
    )
    train_time = time.perf_counter() - t0

    assert len(training_curve) == N_STEPS, (
        f"Expected {N_STEPS} entries, got {len(training_curve)}"
    )

    # Phase 7: Verify outputs
    print("\n  Phase 7: Verify outputs")
    for i, entry in enumerate(training_curve):
        loss = entry.get("loss", entry.get("bt_loss", 0.0))
        grad = entry.get("grad_norm", 0.0)
        accs = {name: entry.get(f"accuracy_{name}", 0.0) for name in HEAD_NAMES}
        print(f"    Step {i}: loss={loss:.4f}, grad_norm={grad:.4f}, accs={accs}")

    # Save results
    results = {
        "status": "PASS",
        "n_steps": N_STEPS,
        "train_time_s": train_time,
        "wall_time_s": time.perf_counter() - wall_start,
        "training_curve": training_curve,
        "modules_tested": [
            "src_ii.model_paths",
            "src_ii.dataset_io.make_load_latent_fn",
            "src_ii.training_setup.encode_training_prompts",
            "src_ii.training_setup.build_dataset_positions",
            "src_ii.training_setup.load_training_backbone",
            "src_ii.btrm_training.train_btrm_differentiable",
            "src_ii.btrm_training._flops_budget_step",
            "src_ii.btrm_training._compute_pair_bt_loss",
            "src_ii.btrm_training._bin_pack_images",
            "src_ii.btrm_training._compute_sigma_weight_for_pairs",
        ],
    }

    results_path = OUTPUT_DIR / "validation_results.json"
    with open(str(results_path), "w") as f:
        json.dump(results, f, indent=2, default=str)

    reader.close()
    torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"  REFACTOR VALIDATION: PASS")
    print(f"  {N_STEPS} steps in {train_time:.1f}s ({train_time/N_STEPS:.1f}s/step)")
    print(f"  Wall time: {time.perf_counter() - wall_start:.1f}s")
    print(f"  Results: {results_path}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

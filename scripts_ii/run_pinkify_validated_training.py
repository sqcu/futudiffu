r"""BTRM training with periodic PINKIFY holdout validation.

Trains a BTRM model on the multi-res trajectory dataset (420+ trajectories,
multi-resolution, multi-prompt) with clean-biased sampling (80/20), and
evaluates the PINKIFY holdout challenge set every ~10 training steps.

The goal: identify whether the reward model learns a non-vacuous pinkify
signal -- one that preserves the ground-truth ranking:
    A < B < C, D ~ E, {A,B,C} < {D,E} < F
rather than just learning the trivial noise-vs-clean discriminator.

Critical optimization: VAE-encode the 6 PINKIFY challenge images to latents
ONCE before training starts. For subsequent evaluations, just re-score the
cached latents with the evolving BTRM model. No VAE reload every 10 steps.

Execution:
  PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe ^
      F:\dox\repos\ai\futudiffu\scripts_ii\run_pinkify_validated_training.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch


from src_ii.model_paths import FP8_PATH, TE_PATH, VAE_PATH, TOKENIZER_PATH

DATASET_DIR = REPO_ROOT / "multi_res_trajectories"
OUTPUT_DIR = REPO_ROOT / "training_output" / "pinkify_validation_run"
CHALLENGE_DIR = REPO_ROOT / "i2i_off_policies" / "PINKIFY_cases"

N_STEPS = 150
MACROBATCH_BUDGET = 3.0
MACROBATCH_CROSS_RES = True
LR = 3e-4
GRAD_CLIP = 0.1
WARMUP_STEPS = 5
LR_SCHEDULE = "warmup_cosine"
CHECKPOINT_STEPS = [25, 50, 75, 100, 125]
CLEAN_FRACTION = 0.8

HEAD_NAMES = ("pinkify", "thisnotthat")
PREF_KEYS = ("pinkify_pref", "thisnotthat_pref")

PINKIFY_EVAL_INTERVAL = 10  # Evaluate pinkify every N steps
PINKIFY_HEAD_NAME = "pinkify"

RUN_NAME = "pinkify_validated_training"



def encode_pinkify_challenge_latents(
    challenge_dir: Path,
    vae_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, dict]:
    """Load PINKIFY challenge images, VAE-encode to latents, return cached data.

    Returns dict mapping label ("A" through "F") to:
        {
            "latent": (1, 16, H//8, W//8) tensor on device,
            "timestep": (1,) zero tensor,
            "conditioning": (1, 1, 2560) zero tensor,
            "num_tokens": 1,
            "width": int,
            "height": int,
        }
    """
    from PIL import Image
    import numpy as np
    from src_ii.vae_utils import load_vae
    from futudiffu.vae import vae_encode

    print(f"  Loading VAE for PINKIFY challenge encoding...")
    vae = load_vae(vae_path, device=device, dtype=dtype)

    labels = ("A", "B", "C", "D", "E", "F")
    cache = {}

    for label in labels:
        candidates = [
            challenge_dir / f"PINKER_{label}.png",
            challenge_dir / f"{label}.png",
        ]
        img_path = None
        for c in candidates:
            if c.exists():
                img_path = c
                break
        if img_path is None:
            raise FileNotFoundError(f"Challenge image for '{label}' not found: {candidates}")

        pil = Image.open(str(img_path)).convert("RGB")
        arr = np.array(pil, dtype=np.float32) / 255.0  # (H, W, 3)
        img_t = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
        pixel_batch = img_t.unsqueeze(0).to(device=device, dtype=dtype)  # (1, 3, H, W)

        with torch.no_grad():
            latent = vae_encode(vae, pixel_batch)  # (1, 16, H//8, W//8)

        _, _, lat_h, lat_w = latent.shape
        w_pixels = lat_w * 8
        h_pixels = lat_h * 8

        cache[label] = {
            "latent": latent,
            "timestep": torch.zeros(1, device=device, dtype=dtype),
            "conditioning": torch.zeros(1, 1, 2560, device=device, dtype=dtype),
            "num_tokens": 1,
            "width": w_pixels,
            "height": h_pixels,
        }
        print(f"    {label}: {w_pixels}x{h_pixels}, latent shape {latent.shape}")

    del vae
    torch.cuda.empty_cache()
    print(f"  VAE freed after encoding {len(cache)} challenge images.")

    return cache


def score_pinkify_cached(
    model,
    latent_cache: dict[str, dict],
    head_name: str = "pinkify",
    head_names: tuple[str, ...] = ("pinkify", "thisnotthat"),
    device: torch.device = torch.device("cuda"),
) -> dict:
    """Score cached PINKIFY latents with the current BTRM model state.

    Returns the same dict format as validate_btrm_pinkify_ranking:
        "passed", "scores", "checks", "rank_order", "head_name", "head_index"
    """
    from src_ii.pinkify_validation import _check_ranking
    from src_ii.btrm_lifecycle import score_serial

    head_index = list(head_names).index(head_name)

    model.gradient_checkpointing = False
    model.eval()
    scores = {}

    with torch.no_grad():
        for label, cached in latent_cache.items():
            latent = cached["latent"]
            timestep = cached["timestep"]
            conditioning = cached["conditioning"]
            num_tokens = cached["num_tokens"]

            score_tensor = score_serial(
                model, latent, timestep, conditioning, num_tokens,
                gradient_checkpointing=False,
            )
            scores[label] = float(score_tensor[0, head_index].item())

    checks = _check_ranking(scores)
    all_passed = all(c["passed"] for c in checks)
    rank_order = sorted(scores.keys(), key=lambda k: scores[k])

    return {
        "passed": all_passed,
        "scores": scores,
        "checks": checks,
        "rank_order": rank_order,
        "head_name": head_name,
        "head_index": head_index,
    }


def append_pinkify_log(log_path: Path, entry: dict) -> None:
    """Append one evaluation entry to the pinkify validation log (JSONL)."""
    with open(str(log_path), "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def main():
    wall_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("=" * 70)
    print(f"  PINKIFY-VALIDATED BTRM TRAINING")
    print(f"  Steps: {N_STEPS}, macrobatch_budget: {MACROBATCH_BUDGET}")
    print(f"  Clean fraction: {CLEAN_FRACTION}")
    print(f"  PINKIFY eval interval: every {PINKIFY_EVAL_INTERVAL} steps")
    print(f"  LR: {LR}, schedule: {LR_SCHEDULE}, grad_clip: {GRAD_CLIP}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)

    if not CHALLENGE_DIR.exists():
        print(f"\n  FATAL: PINKIFY challenge directory not found: {CHALLENGE_DIR}")
        print(f"  Expected files: PINKER_A.png through PINKER_F.png")
        return 1

    if not DATASET_DIR.exists():
        print(f"\n  FATAL: Multi-res dataset not found: {DATASET_DIR}")
        return 1

    print("\n" + "=" * 60)
    print("  Phase 1: Loading multi-res V2 dataset")
    print("=" * 60)

    from src_ii.training_setup import build_dataset_positions

    reader, positions, sampler, traj_ids = build_dataset_positions(
        DATASET_DIR, clean_fraction=CLEAN_FRACTION,
    )

    if len(traj_ids) < 10:
        print(f"  ERROR: Need at least 10 trajectories, have {len(traj_ids)}")
        return 1

    res_dist = {}
    for pos in positions:
        key = f"{pos.width}x{pos.height}"
        res_dist[key] = res_dist.get(key, 0) + 1
    n_unique_res = len(res_dist)

    print("\n" + "=" * 60)
    print("  Phase 2: Encoding prompts")
    print("=" * 60)

    from src_ii.training_setup import encode_training_prompts

    prompt_cache = encode_training_prompts(
        reader, traj_ids, TOKENIZER_PATH, TE_PATH, device=device, dtype=dtype,
    )
    n_prompts = len(prompt_cache)

    print("\n" + "=" * 60)
    print("  Phase 3: Encoding PINKIFY challenge images to latents")
    print("=" * 60)

    pinkify_latent_cache = encode_pinkify_challenge_latents(
        CHALLENGE_DIR, VAE_PATH, device, dtype,
    )
    print(f"  Cached {len(pinkify_latent_cache)} challenge latents")

    print("\n" + "=" * 60)
    print("  Phase 3b: Ground truth PINKIFY scores (pixel-space)")
    print("=" * 60)

    from src_ii.pinkify_validation import validate_pinkify_ranking

    ground_truth = validate_pinkify_ranking(
        challenge_dir=str(CHALLENGE_DIR),
        device=device,
    )

    gt_path = OUTPUT_DIR / "pinkify_ground_truth.json"
    with open(str(gt_path), "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"  Ground truth scores:")
    for label in ("A", "B", "C", "D", "E", "F"):
        print(f"    {label}: {ground_truth['scores'][label]:.6f}")
    print(f"  Rank order: {' < '.join(ground_truth['rank_order'])}")
    print(f"  All constraints passed: {ground_truth['passed']}")
    for check in ground_truth["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"    [{status}] {check['name']}: {check['detail']}")
    print(f"  Saved to {gt_path}")

    print("\n" + "=" * 60)
    print("  Phase 4: Loading backbone + creating BTRM compound model")
    print("=" * 60)

    from src_ii.training_setup import load_training_backbone
    from src_ii.btrm_lifecycle import persist_btrm

    raw_model, optimizer, head_names_loaded = load_training_backbone(
        FP8_PATH, device=device, dtype=dtype, lr=LR,
    )

    from src_ii.multi_lora import get_adapter_params
    n_adapter = sum(p.numel() for p in get_adapter_params(raw_model, "rtheta").values())
    n_head = sum(p.numel() for p in raw_model.score_proj.parameters()) + \
             sum(p.numel() for p in raw_model.score_norm.parameters())

    print("\n" + "=" * 60)
    print("  Phase 4b: Initial PINKIFY evaluation (before training)")
    print("=" * 60)

    pinkify_log_path = OUTPUT_DIR / "pinkify_validation_log.jsonl"
    if pinkify_log_path.exists():
        pinkify_log_path.unlink()

    initial_eval = score_pinkify_cached(
        raw_model, pinkify_latent_cache,
        head_name=PINKIFY_HEAD_NAME, head_names=HEAD_NAMES, device=device,
    )
    initial_eval["step"] = -1  # before training
    initial_eval["timestamp"] = datetime.now(timezone.utc).isoformat()
    append_pinkify_log(pinkify_log_path, initial_eval)

    print(f"  Initial BTRM pinkify scores (untrained):")
    for label in ("A", "B", "C", "D", "E", "F"):
        print(f"    {label}: {initial_eval['scores'][label]:.6f}")
    print(f"  Rank order: {' < '.join(initial_eval['rank_order'])}")
    print(f"  Constraints passed: {sum(1 for c in initial_eval['checks'] if c['passed'])}/5")
    for check in initial_eval["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"    [{status}] {check['name']}")

    from src_ii.dataset_io import make_load_latent_fn
    load_latent_fn = make_load_latent_fn(reader, prompt_cache, device=device, dtype=dtype)

    def preference_fn(pair: dict) -> dict:
        """Deterministic preference: cleaner image (lower sigma) wins."""
        prefs = {}
        for pref_key in PREF_KEYS:
            sigma_a = pair.get("sigma_a", 0.5)
            sigma_b = pair.get("sigma_b", 0.5)
            if sigma_a < sigma_b - 0.001:
                prefs[pref_key] = 1   # A is cleaner, A wins
            elif sigma_b < sigma_a - 0.001:
                prefs[pref_key] = -1  # B is cleaner, B wins
            else:
                prefs[pref_key] = 0   # tie
        return prefs

    print("\n" + "=" * 60)
    print(f"  Phase 5: Training ({N_STEPS} steps) + PINKIFY validation")
    print(f"  macrobatch_budget={MACROBATCH_BUDGET}, clean_fraction={CLEAN_FRACTION}")
    print(f"  PINKIFY eval at steps: 0, {PINKIFY_EVAL_INTERVAL}, "
          f"{2*PINKIFY_EVAL_INTERVAL}, ... and final")
    print("=" * 60)

    from src_ii.btrm_training import train_btrm_differentiable
    from src_ii.training_artifacts import TrainingArtifacts
    from src_ii.incremental_save import TrainingCurveWriter

    artifacts = TrainingArtifacts(
        output_dir=str(OUTPUT_DIR),
        run_name=RUN_NAME,
        head_names=HEAD_NAMES,
    )

    curve_writer = TrainingCurveWriter(OUTPUT_DIR / "training_curve.jsonl")

    pinkify_eval_count = 0

    def pinkify_validation_callback(step: int, entry: dict) -> None:
        nonlocal pinkify_eval_count

        is_eval_step = (
            step == 0
            or step % PINKIFY_EVAL_INTERVAL == 0
            or step == N_STEPS - 1
        )
        if not is_eval_step:
            return

        t_eval_start = time.perf_counter()
        eval_result = score_pinkify_cached(
            raw_model, pinkify_latent_cache,
            head_name=PINKIFY_HEAD_NAME, head_names=HEAD_NAMES, device=device,
        )
        eval_time = time.perf_counter() - t_eval_start

        eval_result["step"] = step
        eval_result["timestamp"] = datetime.now(timezone.utc).isoformat()
        eval_result["eval_time_s"] = eval_time
        eval_result["training_loss"] = entry.get("loss", entry.get("bt_loss", 0.0))

        for name in HEAD_NAMES:
            eval_result[f"training_accuracy_{name}"] = entry.get(f"accuracy_{name}", 0.0)

        append_pinkify_log(pinkify_log_path, eval_result)
        pinkify_eval_count += 1

        n_passed = sum(1 for c in eval_result["checks"] if c["passed"])
        rank_str = " < ".join(eval_result["rank_order"])
        print(f"  [PINKIFY step {step}] {n_passed}/5 constraints | "
              f"rank: {rank_str} | "
              f"{'ALL PASS' if eval_result['passed'] else 'INCOMPLETE'} | "
              f"{eval_time:.1f}s")

        raw_model.gradient_checkpointing = True
        raw_model.train()

    t_train_start = time.perf_counter()

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
        max_grad_norm=GRAD_CLIP,
        log_interval=5,
        warmup_steps=WARMUP_STEPS,
        lr_schedule=LR_SCHEDULE,
        output_dir=str(OUTPUT_DIR),
        artifacts=artifacts,
        checkpoint_steps=CHECKPOINT_STEPS,
        macrobatch_budget=MACROBATCH_BUDGET,
        macrobatch_cross_resolution=MACROBATCH_CROSS_RES,
        curve_writer=curve_writer,
        val_metrics_save_interval=10,
        summary_path=str(OUTPUT_DIR / "run_summary.json"),
        callback=pinkify_validation_callback,
    )

    curve_writer.close()
    train_time = time.perf_counter() - t_train_start
    print(f"\n  Training complete: {train_time:.1f}s "
          f"({train_time / N_STEPS:.1f}s/step)")

    print("\n" + "=" * 60)
    print("  Phase 6: Final analysis")
    print("=" * 60)

    run_config = {
        "mode": "pinkify_validated_training",
        "n_steps": N_STEPS,
        "macrobatch_budget": MACROBATCH_BUDGET,
        "macrobatch_cross_resolution": MACROBATCH_CROSS_RES,
        "lr": LR,
        "lr_schedule": LR_SCHEDULE,
        "grad_clip": GRAD_CLIP,
        "warmup_steps": WARMUP_STEPS,
        "clean_fraction": CLEAN_FRACTION,
        "dataset": str(DATASET_DIR),
        "n_trajectories": len(traj_ids),
        "n_unique_resolutions": n_unique_res,
        "n_unique_prompts": n_prompts,
        "pinkify_eval_interval": PINKIFY_EVAL_INTERVAL,
        "pinkify_eval_count": pinkify_eval_count,
    }

    report_path = artifacts.generate_analysis(run_config=run_config)
    print(f"  Analysis: {report_path}")

    persist_info = persist_btrm(raw_model, "rtheta", str(OUTPUT_DIR))
    print(f"  Model persisted: {persist_info}")

    pinkify_entries = []
    with open(str(pinkify_log_path), "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    pinkify_entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    pinkify_summary = {
        "n_evaluations": len(pinkify_entries),
        "ground_truth": ground_truth,
        "evaluations": [],
    }

    for entry in pinkify_entries:
        summary_entry = {
            "step": entry["step"],
            "passed": entry["passed"],
            "n_constraints_passed": sum(1 for c in entry["checks"] if c["passed"]),
            "scores": entry["scores"],
            "rank_order": entry["rank_order"],
            "checks": entry["checks"],
        }
        if "training_loss" in entry:
            summary_entry["training_loss"] = entry["training_loss"]
        pinkify_summary["evaluations"].append(summary_entry)

    first_all_pass = None
    for e in pinkify_summary["evaluations"]:
        if e["passed"]:
            first_all_pass = e["step"]
            break

    pinkify_summary["first_all_pass_step"] = first_all_pass
    pinkify_summary["final_passed"] = pinkify_entries[-1]["passed"] if pinkify_entries else False
    pinkify_summary["final_n_constraints"] = (
        sum(1 for c in pinkify_entries[-1]["checks"] if c["passed"])
        if pinkify_entries else 0
    )

    pinkify_json_path = OUTPUT_DIR / "pinkify_validation_log.json"
    with open(str(pinkify_json_path), "w") as f:
        json.dump(pinkify_summary, f, indent=2, default=str)
    print(f"  PINKIFY summary: {pinkify_json_path}")

    print("\n" + "=" * 60)
    print("  Phase 7: Summary")
    print("=" * 60)

    wall_total = time.perf_counter() - wall_start

    losses = [e.get("loss", e.get("bt_loss", 0.0)) for e in training_curve]

    summary = {
        "run_name": RUN_NAME,
        "wall_time_s": wall_total,
        "train_time_s": train_time,
        "n_steps": N_STEPS,
        "macrobatch_budget": MACROBATCH_BUDGET,
        "clean_fraction": CLEAN_FRACTION,
        "lr": LR,
        "lr_schedule": LR_SCHEDULE,
        "grad_clip": GRAD_CLIP,
        "n_trajectories": len(traj_ids),
        "n_unique_resolutions": n_unique_res,
        "n_unique_prompts": n_prompts,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
        "n_adapter_params": n_adapter,
        "n_head_params": n_head,
        "pinkify_eval_count": pinkify_eval_count,
        "pinkify_first_all_pass": first_all_pass,
        "pinkify_final_passed": pinkify_summary["final_passed"],
        "pinkify_final_n_constraints": pinkify_summary["final_n_constraints"],
        "ground_truth_passed": ground_truth["passed"],
        "ground_truth_rank_order": ground_truth["rank_order"],
        "sampler_stats": sampler.stats(),
        "end_time": datetime.now(timezone.utc).isoformat(),
    }

    for name in HEAD_NAMES:
        accs = [e.get(f"accuracy_{name}", 0.0) for e in training_curve]
        if accs:
            summary[f"overall_accuracy_{name}"] = sum(accs) / len(accs)
            last_20 = accs[-20:]
            summary[f"last_20_accuracy_{name}"] = sum(last_20) / len(last_20)

    summary_path = OUTPUT_DIR / "run_summary.json"
    with open(str(summary_path), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    curve_path = OUTPUT_DIR / "training_curve.json"
    with open(str(curve_path), "w") as f:
        json.dump(training_curve, f, indent=2, default=str)

    print(f"\n  Wall time: {wall_total:.1f}s ({wall_total / 60:.1f} min)")
    print(f"  Train time: {train_time:.1f}s ({train_time / N_STEPS:.1f}s/step)")
    if losses:
        print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f} "
              f"(min={min(losses):.4f} at step {losses.index(min(losses))})")
    for name in HEAD_NAMES:
        accs = [e.get(f"accuracy_{name}", 0.0) for e in training_curve]
        if accs:
            print(f"  {name}: overall={sum(accs)/len(accs):.1%}, "
                  f"last-20={sum(accs[-20:])/len(accs[-20:]):.1%}")
    print(f"  PINKIFY evaluations: {pinkify_eval_count}")
    print(f"  PINKIFY first all-pass: step {first_all_pass}")
    print(f"  PINKIFY final: {pinkify_summary['final_n_constraints']}/5 constraints")
    print(f"  Clean fraction (measured): {sampler.get_clean_fraction():.1%}")
    print(f"  Output: {OUTPUT_DIR}")

    reader.close()
    torch.cuda.empty_cache()

    print(f"\n{'=' * 70}")
    print(f"  PINKIFY-VALIDATED TRAINING COMPLETE")
    print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

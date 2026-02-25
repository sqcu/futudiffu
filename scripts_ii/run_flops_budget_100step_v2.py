r"""100-step FLOPS-budget BTRM training v2: multi-res + corrected logSNR.

Changes from v1 (run_flops_budget_100step.py):
  - Dataset: multi_res_trajectories/ with 60 trajectories across 26 non-square
    resolutions in 6 megapixel tiers (was: 60 trajectories at 3 square resolutions)
  - LogSNR step weighting: threshold=10.0, decay_rate=0.5 (corrected defaults)
  - Final position sigma: sigma=0.0 for "final" step (was: sigmas[-2] bug)
  - New charts: logSNR distribution histogram, sampling weight by tier,
    FLOPS consumed per macrobatch, pairs per macrobatch, cross-res fraction,
    resolution tier distribution
  - Output: flops_budget_100step_v2_output/

The logSNR corrections are in the src_ii/ defaults. The sigma=0.0 fix is in
this script's load_latent_fn AND in pair_sampler.build_positions_from_v2().

Execution:
  PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe ^
      F:\dox\repos\ai\futudiffu\scripts_ii\run_flops_budget_100step_v2.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch


from src_ii.model_paths import FP8_PATH, TE_PATH, VAE_PATH, TOKENIZER_PATH

DATASET_DIR = REPO_ROOT / "multi_res_trajectories"
OUTPUT_DIR = REPO_ROOT / "flops_budget_100step_v2_output"

N_STEPS = 100
MACROBATCH_BUDGET = 3.0   # FLOPS budget in 1024^2-equivalent units
MACROBATCH_CROSS_RES = True  # enable cross-resolution pairs
LR = 3e-4
GRAD_CLIP = 0.1
WARMUP_STEPS = 5
LR_SCHEDULE = "warmup_cosine"
CHECKPOINT_STEPS = [25, 50, 75]

HEAD_NAMES = ("pinkify", "thisnotthat")
PREF_KEYS = ("pinkify_pref", "thisnotthat_pref")

RUN_NAME = "flops_budget_100step_v2"


def main():
    wall_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("=" * 70)
    print(f"  FLOPS-BUDGET 100-STEP BTRM TRAINING v2 (multi-res + logSNR fix)")
    print(f"  Steps: {N_STEPS}, macrobatch_budget: {MACROBATCH_BUDGET}")
    print(f"  Cross-resolution pairs: {MACROBATCH_CROSS_RES}")
    print(f"  LR: {LR}, schedule: {LR_SCHEDULE}, grad_clip: {GRAD_CLIP}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)

    from src_ii.pair_sampler import logsnr_sampling_weight, logsnr_sampling_logit
    import inspect

    sig_weight = inspect.signature(logsnr_sampling_weight)
    sig_logit = inspect.signature(logsnr_sampling_logit)

    threshold_default = sig_weight.parameters["threshold"].default
    decay_default = sig_weight.parameters["decay_rate"].default
    print(f"\n  [VERIFY] logsnr_sampling_weight defaults: "
          f"threshold={threshold_default}, decay_rate={decay_default}")
    assert threshold_default == 10.0, f"Expected threshold=10.0, got {threshold_default}"
    assert decay_default == 0.5, f"Expected decay_rate=0.5, got {decay_default}"

    threshold_logit = sig_logit.parameters["threshold"].default
    decay_logit = sig_logit.parameters["decay_rate"].default
    print(f"  [VERIFY] logsnr_sampling_logit defaults: "
          f"threshold={threshold_logit}, decay_rate={decay_logit}")
    assert threshold_logit == 10.0
    assert decay_logit == 0.5

    w_sigma0 = logsnr_sampling_weight(0.0)
    w_sigma_clean = logsnr_sampling_weight(0.034)  # step_29 at 1280x832
    w_sigma_noisy = logsnr_sampling_weight(0.867)  # step_04 at 1280x832
    print(f"  [VERIFY] Weight at sigma=0.0: {w_sigma0:.3f} (expected 1.000)")
    print(f"  [VERIFY] Weight at sigma=0.034: {w_sigma_clean:.3f} (expected ~0.632)")
    print(f"  [VERIFY] Weight at sigma=0.867: {w_sigma_noisy:.3f} (expected ~0.149)")
    print(f"  [VERIFY] Clean/noisy ratio: {w_sigma_clean/w_sigma_noisy:.2f}x (expected ~4.25x)")
    print(f"  LogSNR defaults VERIFIED.\n")

    print("=" * 60)
    print("  Phase 1: Loading multi-res V2 dataset")
    print("=" * 60)

    from futudiffu.dataset_v2 import DatasetReader
    from src_ii.pair_sampler import BTRMPairSampler, build_positions_from_v2
    from src_ii.flops_sampling import (
        compute_flops_sampling_weights_from_positions,
        summarize_flops_weights,
    )

    reader = DatasetReader(str(DATASET_DIR))
    n_available = len(reader)
    print(f"  Dataset: {n_available} trajectories")

    if n_available < 10:
        print(f"  ERROR: Need at least 10 trajectories, have {n_available}")
        return 1

    traj_ids = list(range(n_available))
    print(f"  Using all {len(traj_ids)} trajectories")

    positions = build_positions_from_v2(reader, traj_ids=traj_ids)
    print(f"  Positions: {len(positions)} across {len(traj_ids)} trajectories")

    res_dist = {}
    for pos in positions:
        key = f"{pos.width}x{pos.height}"
        res_dist[key] = res_dist.get(key, 0) + 1
    print(f"  Resolution distribution ({len(res_dist)} unique):")
    for rk in sorted(res_dist.keys()):
        print(f"    {rk}: {res_dist[rk]} positions")

    final_positions = [p for p in positions if p.step_key == "final"]
    n_final_correct = sum(1 for p in final_positions if p.sigma == 0.0)
    print(f"  Final positions: {len(final_positions)} total, "
          f"{n_final_correct} with sigma=0.0 "
          f"({'ALL CORRECT' if n_final_correct == len(final_positions) else 'BUG!'})")
    assert n_final_correct == len(final_positions), \
        f"Final positions must have sigma=0.0, but {len(final_positions) - n_final_correct} do not"

    flops_weights = compute_flops_sampling_weights_from_positions(positions)

    traj_resolutions = {}
    for pos in positions:
        if pos.traj_id not in traj_resolutions:
            traj_resolutions[pos.traj_id] = (pos.width, pos.height)
    flops_summary = summarize_flops_weights(flops_weights, traj_resolutions)
    print(f"  FLOPS weights summary:")
    print(f"    {json.dumps(flops_summary, indent=4)}")

    unique_weights = set(round(w, 8) for w in flops_weights.values())
    print(f"  Unique FLOPS weight values: {len(unique_weights)} "
          f"(degenerate={len(unique_weights) <= 1})")

    sampler = BTRMPairSampler(
        positions=positions,
        allow_inter_trajectory=True,
        allow_intra_trajectory=True,
        rng_seed=42,
        flops_weights=flops_weights,
    )
    print(f"  Pair space: {sampler.pair_space_size:,} possible pairs")
    print(f"  Populated tiers: {sampler.populated_tiers}")
    print(f"  Tier trajectory counts: {sampler.tier_trajectory_counts}")

    test_macro = sampler.sample_macrobatch(
        budget_units=MACROBATCH_BUDGET,
        tier_flops_targets={1048576: 0.33},
        allow_cross_resolution=MACROBATCH_CROSS_RES,
    )
    test_consumed = sum(ps.flops_cost for ps in test_macro)
    test_cross = sum(1 for ps in test_macro if ps.cross_resolution)
    print(f"  Test macrobatch: {len(test_macro)} pairs, "
          f"{test_consumed:.3f} FLOPS consumed, "
          f"{test_cross} cross-resolution")

    print("\n" + "=" * 60)
    print("  Phase 2: Encoding prompts")
    print("=" * 60)

    from src_ii.training_setup import encode_training_prompts

    prompt_cache = encode_training_prompts(
        reader, traj_ids, TOKENIZER_PATH, TE_PATH, device=device, dtype=dtype,
    )
    n_prompts = len(prompt_cache)

    print("\n" + "=" * 60)
    print("  Phase 3: Loading backbone + creating BTRM compound model")
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

    from src_ii.dataset_io import make_load_latent_fn
    load_latent_fn = make_load_latent_fn(reader, prompt_cache, device=device, dtype=dtype)

    # Verify final sigma = 0.0 (critical correctness check)
    test_key = (traj_ids[0], positions[0].step_key)
    lat, ts, cond, nt, _ = load_latent_fn(test_key)
    print(f"  Test load: latent={lat.shape}, timestep={ts.shape}, cond={cond.shape}")

    test_final_key = (traj_ids[0], "final")
    _, ts_final, _, _, _ = load_latent_fn(test_final_key)
    final_sigma = ts_final.item()
    print(f"  Final position sigma: {final_sigma:.6f} "
          f"({'CORRECT (0.0)' if final_sigma == 0.0 else f'BUG! Expected 0.0, got {final_sigma}'})")
    assert final_sigma == 0.0, f"Final sigma must be 0.0, got {final_sigma}"

    del lat, ts, cond, ts_final
    torch.cuda.empty_cache()

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

    sampled_logsnr_values = []  # populated during training via callback

    print("\n" + "=" * 60)
    print(f"  Phase 4: FLOPS-budget training ({N_STEPS} steps)")
    print(f"  macrobatch_budget={MACROBATCH_BUDGET}, "
          f"cross_resolution={MACROBATCH_CROSS_RES}")
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
        packed=True,
        output_dir=str(OUTPUT_DIR),
        artifacts=artifacts,
        checkpoint_steps=CHECKPOINT_STEPS,
        macrobatch_budget=MACROBATCH_BUDGET,
        macrobatch_cross_resolution=MACROBATCH_CROSS_RES,
        curve_writer=curve_writer,
        val_metrics_save_interval=10,
        summary_path=str(OUTPUT_DIR / "run_summary.json"),
    )

    curve_writer.close()
    print(f"  Training curve: {curve_writer.n_written} steps written to "
          f"{OUTPUT_DIR / 'training_curve.jsonl'}")

    train_time = time.perf_counter() - t_train_start
    print(f"\n  Training complete: {train_time:.1f}s "
          f"({train_time / N_STEPS:.1f}s/step)")

    print("\n" + "=" * 60)
    print("  Phase 5: Generating standard analysis + charts")
    print("=" * 60)

    fb_stats = _compute_flops_budget_stats(training_curve)

    run_config = {
        "mode": "flops_budget_packed_v2",
        "n_steps": N_STEPS,
        "macrobatch_budget": MACROBATCH_BUDGET,
        "macrobatch_cross_resolution": MACROBATCH_CROSS_RES,
        "lr": LR,
        "lr_schedule": LR_SCHEDULE,
        "grad_clip": GRAD_CLIP,
        "warmup_steps": WARMUP_STEPS,
        "dataset": str(DATASET_DIR),
        "n_trajectories": len(traj_ids),
        "resolution_dist": res_dist,
        "n_unique_prompts": n_prompts,
        "checkpoint_steps": str(CHECKPOINT_STEPS),
        "flops_budget_stats": fb_stats,
        "logsnr_threshold": 10.0,
        "logsnr_decay_rate": 0.5,
        "final_sigma_fix": True,
    }

    report_path = artifacts.generate_analysis(run_config=run_config)
    print(f"  Standard analysis generated: {report_path}")

    persist_info = persist_btrm(raw_model, "rtheta", str(OUTPUT_DIR))
    print(f"  Model persisted: {persist_info}")

    print("\n" + "=" * 60)
    print("  Phase 5b: Generating v2-specific charts")
    print("=" * 60)

    charts_dir = OUTPUT_DIR / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    _generate_v2_charts(training_curve, charts_dir, positions, sampler)
    print(f"  v2 charts saved to {charts_dir}")

    print("\n" + "=" * 60)
    print("  Phase 6: Rendering exemplar images")
    print("=" * 60)

    exemplars_rendered = False
    try:
        from src_ii.exemplar_renderer import render_exemplars_from_model

        sample_keys = []
        seen_keys = set()  # (traj_id, step_key) dedup

        for pos in positions:
            if pos.step_key == "final" and (pos.traj_id, pos.step_key) not in seen_keys:
                sample_keys.append((pos.traj_id, pos.step_key))
                seen_keys.add((pos.traj_id, pos.step_key))
            if len(sample_keys) >= 12:
                break

        for pos in positions:
            if pos.step_key == "step_29" and (pos.traj_id, pos.step_key) not in seen_keys:
                sample_keys.append((pos.traj_id, pos.step_key))
                seen_keys.add((pos.traj_id, pos.step_key))
            if len(sample_keys) >= 18:
                break

        for pos in positions:
            if pos.step_key == "step_14" and (pos.traj_id, pos.step_key) not in seen_keys:
                sample_keys.append((pos.traj_id, pos.step_key))
                seen_keys.add((pos.traj_id, pos.step_key))
            if len(sample_keys) >= 24:
                break

        for pos in positions:
            if pos.step_key == "step_04" and (pos.traj_id, pos.step_key) not in seen_keys:
                sample_keys.append((pos.traj_id, pos.step_key))
                seen_keys.add((pos.traj_id, pos.step_key))
            if len(sample_keys) >= 30:
                break

        print(f"  Scoring {len(sample_keys)} images for exemplars "
              f"(diverse sigma levels)")

        exemplar_manifest = render_exemplars_from_model(
            output_dir=str(OUTPUT_DIR / "exemplars"),
            btrm_model=raw_model,
            load_latent_fn=load_latent_fn,
            sample_keys=sample_keys,
            vae_path=VAE_PATH,
            top_k=3,
            head_names=list(HEAD_NAMES),
            device=device,
            dtype=dtype,
        )
        exemplars_rendered = True
        print(f"  Exemplars rendered: {exemplar_manifest}")
    except Exception as e:
        print(f"  Exemplar rendering failed (non-fatal): {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("  Phase 7: Summary")
    print("=" * 60)

    wall_total = time.perf_counter() - wall_start

    losses = [e.get("loss", e.get("bt_loss", 0.0)) for e in training_curve]
    grad_norms_pre = [e.get("pre_clip_grad_norm", 0.0) for e in training_curve]
    grad_norms_post = [e.get("grad_norm", 0.0) for e in training_curve]
    step_times = [e.get("time_s", 0.0) for e in training_curve]

    n_pairs_per_step = []
    n_bins_per_step = []
    consumed_per_step = []
    n_cross_res_per_step = []
    for e in training_curve:
        fm = e.get("funfetti", {})
        n_pairs_per_step.append(fm.get("total_pairs", 0))
        n_bins_per_step.append(fm.get("n_bins", 0))
        consumed_per_step.append(fm.get("macrobatch_consumed", 0.0))
        n_cross_res_per_step.append(fm.get("n_cross_resolution_pairs", 0))

    from src_ii.flops_sampling import _attention_flops_ratio, _MEGAPIXEL_THRESHOLD
    from src_ii.resolution_sampling import assign_budget_tier, ANCHOR_LABELS

    all_pixel_counts = Counter()
    pixel_to_wh = {}
    tier_pair_counts = Counter()
    for entry in training_curve:
        fm = entry.get("funfetti")
        if not fm:
            continue
        for res in fm.get("resolutions", []):
            px = res["pixels"]
            all_pixel_counts[px] += 1
            if px not in pixel_to_wh:
                pixel_to_wh[px] = (res["width"], res["height"])
        for ppr in fm.get("per_pair_resolutions", []):
            wa, ha = ppr.get("width_a", 0), ppr.get("height_a", 0)
            wb, hb = ppr.get("width_b", 0), ppr.get("height_b", 0)
            if wa > 0 and ha > 0:
                tier_a = assign_budget_tier(wa, ha)
                tier_pair_counts[tier_a] += 1
            if wb > 0 and hb > 0:
                tier_b = assign_budget_tier(wb, hb)
                tier_pair_counts[tier_b] += 1

    total_images = sum(all_pixel_counts.values())

    total_flops_weighted = 0.0
    mega_flops_weighted = 0.0
    for px, ct in all_pixel_counts.items():
        if px in pixel_to_wh:
            w, h = pixel_to_wh[px]
        else:
            w = h = int(math.sqrt(px))
        flops_r = _attention_flops_ratio(w, h)
        weighted = ct * flops_r
        total_flops_weighted += weighted
        if px >= _MEGAPIXEL_THRESHOLD:
            mega_flops_weighted += weighted
    small_flops_weighted = total_flops_weighted - mega_flops_weighted
    mega_flops_pct = (mega_flops_weighted / total_flops_weighted * 100) if total_flops_weighted > 0 else 0
    small_flops_pct = (small_flops_weighted / total_flops_weighted * 100) if total_flops_weighted > 0 else 0

    print("  Collecting logSNR distribution sample (1000 pairs)...")
    logsnr_sample_pairs = []
    for _ in range(1000):
        sp = sampler._sample_pair_spec(allow_cross_resolution=MACROBATCH_CROSS_RES)
        logsnr_sample_pairs.append(sp)

    logsnr_sample_sigmas = []
    logsnr_sample_values = []
    for sp in logsnr_sample_pairs:
        for pos in [sp.image_a, sp.image_b]:
            sigma = pos.sigma
            logsnr_sample_sigmas.append(sigma)
            if sigma <= 0:
                logsnr_sample_values.append(15.0)  # cap for visualization
            elif sigma >= 1.0:
                logsnr_sample_values.append(-15.0)
            else:
                logsnr_sample_values.append(2.0 * math.log((1.0 - sigma) / sigma))

    logsnr_dist_data = {
        "n_samples": len(logsnr_sample_values),
        "sigma_values": logsnr_sample_sigmas,
        "logsnr_values": logsnr_sample_values,
        "sigma_0_count": sum(1 for s in logsnr_sample_sigmas if s <= 0),
        "sigma_0_fraction": sum(1 for s in logsnr_sample_sigmas if s <= 0) / max(len(logsnr_sample_sigmas), 1),
    }
    with open(str(OUTPUT_DIR / "logsnr_distribution.json"), "w") as f:
        json.dump(logsnr_dist_data, f, indent=2)
    print(f"  logSNR distribution: {logsnr_dist_data['sigma_0_count']} sigma=0 "
          f"({logsnr_dist_data['sigma_0_fraction']:.1%}), "
          f"mean logSNR={sum(logsnr_sample_values)/len(logsnr_sample_values):.2f}")

    summary = {
        "run_name": RUN_NAME,
        "wall_time_s": wall_total,
        "train_time_s": train_time,
        "n_steps": N_STEPS,
        "macrobatch_budget": MACROBATCH_BUDGET,
        "macrobatch_cross_resolution": MACROBATCH_CROSS_RES,
        "lr": LR,
        "lr_schedule": LR_SCHEDULE,
        "grad_clip": GRAD_CLIP,
        "warmup_steps": WARMUP_STEPS,
        "n_trajectories": len(traj_ids),
        "n_unique_resolutions": len(res_dist),
        "resolution_dist": res_dist,
        "flops_weights_summary": flops_summary,
        "sampler_stats": sampler.stats(),
        "logsnr_defaults": {"threshold": 10.0, "decay_rate": 0.5},
        "final_sigma_fix": True,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
        "max_loss": max(losses) if losses else None,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "std_loss": _std(losses),
        "mean_grad_norm_pre": sum(grad_norms_pre) / max(len(grad_norms_pre), 1),
        "max_grad_norm_pre": max(grad_norms_pre) if grad_norms_pre else None,
        "mean_grad_norm_post": sum(grad_norms_post) / max(len(grad_norms_post), 1),
        "flops_budget_stats": fb_stats,
        "mean_pairs_per_step": sum(n_pairs_per_step) / max(len(n_pairs_per_step), 1),
        "min_pairs_per_step": min(n_pairs_per_step) if n_pairs_per_step else 0,
        "max_pairs_per_step": max(n_pairs_per_step) if n_pairs_per_step else 0,
        "mean_bins_per_step": sum(n_bins_per_step) / max(len(n_bins_per_step), 1),
        "min_bins_per_step": min(n_bins_per_step) if n_bins_per_step else 0,
        "max_bins_per_step": max(n_bins_per_step) if n_bins_per_step else 0,
        "mean_consumed_per_step": sum(consumed_per_step) / max(len(consumed_per_step), 1),
        "min_consumed_per_step": min(consumed_per_step) if consumed_per_step else 0,
        "max_consumed_per_step": max(consumed_per_step) if consumed_per_step else 0,
        "total_cross_res_pairs": sum(n_cross_res_per_step),
        "mean_cross_res_per_step": sum(n_cross_res_per_step) / max(len(n_cross_res_per_step), 1),
        "total_images_sampled": total_images,
        "mega_flops_pct": mega_flops_pct,
        "small_flops_pct": small_flops_pct,
        "tier_pair_counts": {str(k): v for k, v in sorted(tier_pair_counts.items())},
        "logsnr_sigma0_fraction": logsnr_dist_data["sigma_0_fraction"],
        "logsnr_mean": sum(logsnr_sample_values) / len(logsnr_sample_values) if logsnr_sample_values else 0,
        "step0_time_s": step_times[0] if step_times else None,
        "mean_steady_state_time_s": sum(step_times[1:]) / max(len(step_times) - 1, 1) if len(step_times) > 1 else None,
        "exemplars_rendered": exemplars_rendered,
        "n_adapter_params": n_adapter,
        "n_head_params": n_head,
        "end_time": datetime.now(timezone.utc).isoformat(),
    }

    for name in HEAD_NAMES:
        accs = [e.get(f"accuracy_{name}", 0.0) for e in training_curve]
        if accs:
            summary[f"overall_accuracy_{name}"] = sum(accs) / len(accs)
            last_20 = accs[-20:]
            summary[f"last_20_accuracy_{name}"] = sum(last_20) / len(last_20)

    summary_out_path = OUTPUT_DIR / "run_summary.json"
    with open(str(summary_out_path), "w") as f:
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
    print(f"  Sampler stats: {sampler.stats()}")
    print(f"  FLOPS budget: mean_consumed={summary['mean_consumed_per_step']:.3f}, "
          f"mean_pairs={summary['mean_pairs_per_step']:.1f}, "
          f"mean_bins={summary['mean_bins_per_step']:.1f}")
    print(f"  Cross-res pairs: {summary['total_cross_res_pairs']} total, "
          f"{summary['mean_cross_res_per_step']:.1f}/step")
    print(f"  FLOPS breakdown: mega={mega_flops_pct:.1f}%, small={small_flops_pct:.1f}%")
    print(f"  logSNR: sigma=0 fraction={logsnr_dist_data['sigma_0_fraction']:.1%}, "
          f"mean={logsnr_dist_data.get('logsnr_mean', 0):.2f}")
    print(f"  Exemplars: {'rendered' if exemplars_rendered else 'FAILED'}")
    print(f"  Output: {OUTPUT_DIR}")

    reader.close()
    torch.cuda.empty_cache()

    print(f"\n{'=' * 70}")
    print(f"  FLOPS-BUDGET 100-STEP v2 RUN COMPLETE")
    print(f"{'=' * 70}")

    return 0



def _generate_v2_charts(training_curve, charts_dir, positions, sampler):
    """Generate the 10 requested charts with proper axis scales.

    Charts 01-04 are generated by TrainingArtifacts.generate_analysis().
    This function generates ADDITIONAL charts or REPLACES the standard ones
    with corrected axis scales.
    """
    from src_ii.training_artifacts import PILChart, _ema, _running_average
    from src_ii.resolution_sampling import assign_budget_tier, ANCHOR_LABELS
    from src_ii.flops_sampling import _attention_flops_ratio, _MEGAPIXEL_THRESHOLD

    steps = [e["step"] for e in training_curve]
    losses = [e.get("loss", e.get("bt_loss", 0.0)) for e in training_curve]

    chart = PILChart()
    chart.set_title(f"flops_budget_100step_v2: Loss per-term (y: 0-1)")
    chart.set_labels("Step", "Loss (per-term avg)")
    ema_losses = _ema(losses, alpha=0.1)
    chart.add_scatter(steps, losses, color="#bbddff", label="raw loss", size=2)
    chart.add_line(steps, ema_losses, color="#1155cc", label="EMA(0.1)", line_width=2)
    chart.add_line([0, max(steps)], [0.693, 0.693], color="#cccccc", label="random (ln2)", style="dashed")
    chart.save(str(charts_dir / "01_loss_curve_bounded.png"))

    chart = PILChart()
    chart.set_title("flops_budget_100step_v2: Per-Head Accuracy (%)")
    chart.set_labels("Step", "Accuracy (%)")
    colors = ["#1155cc", "#cc7711"]
    scatter_colors = ["#aaddff", "#ffddaa"]
    for i, name in enumerate(HEAD_NAMES):
        accs = [e.get(f"accuracy_{name}", 0.0) * 100 for e in training_curve]
        ravg = _running_average(accs, window=5)
        chart.add_scatter(steps, accs, color=scatter_colors[i], size=2)
        chart.add_line(steps, ravg, color=colors[i], label=f"{name} ravg(5)", line_width=2)
    chart.add_line([0, max(steps)], [50.0, 50.0], color="#cccccc", label="random (50%)", style="dashed")
    chart.save(str(charts_dir / "02_per_head_accuracy_bounded.png"))

    grad_norms_pre = [e.get("pre_clip_grad_norm", 0.0) for e in training_curve]
    grad_norms_post = [e.get("grad_norm", 0.0) for e in training_curve]
    chart = PILChart()
    chart.set_title("flops_budget_100step_v2: Gradient Norms")
    chart.set_labels("Step", "Gradient Norm")
    chart.add_scatter(steps, grad_norms_pre, color="#bbffbb", label="pre-clip", size=2)
    chart.add_line(steps, _ema(grad_norms_pre, 0.15), color="#117711", label="pre-clip EMA", line_width=2)
    chart.add_line(steps, grad_norms_post, color="#cc1111", label="post-clip", line_width=2)
    chart.add_line([0, max(steps)], [GRAD_CLIP, GRAD_CLIP], color="#999999", label=f"clip={GRAD_CLIP}", style="dashed")
    chart.save(str(charts_dir / "03_gradient_norms_dual.png"))

    consumed_vals = []
    consumed_steps = []
    for e in training_curve:
        fm = e.get("funfetti", {})
        c = fm.get("macrobatch_consumed", 0.0)
        if c > 0:
            consumed_vals.append(c)
            consumed_steps.append(e["step"])
    if consumed_vals:
        chart = PILChart()
        chart.set_title("flops_budget_100step_v2: FLOPS Consumed per Macrobatch")
        chart.set_labels("Step", "FLOPS Units (1024^2-equiv)")
        chart.add_scatter(consumed_steps, consumed_vals, color="#1155cc", label="consumed", size=3)
        chart.add_line(consumed_steps, _ema(consumed_vals, 0.15), color="#1155cc", label="EMA(0.15)", line_width=2)
        chart.add_line([0, max(consumed_steps)], [MACROBATCH_BUDGET, MACROBATCH_BUDGET],
                       color="#cc1111", label=f"budget={MACROBATCH_BUDGET}", style="dashed")
        chart.save(str(charts_dir / "05_flops_consumed.png"))

    pairs_vals = []
    pairs_steps = []
    for e in training_curve:
        fm = e.get("funfetti", {})
        p = fm.get("total_pairs", 0)
        if p > 0:
            pairs_vals.append(p)
            pairs_steps.append(e["step"])
    if pairs_vals:
        chart = PILChart()
        chart.set_title("flops_budget_100step_v2: Pairs per Macrobatch")
        chart.set_labels("Step", "Pair Count")
        chart.add_scatter(pairs_steps, pairs_vals, color="#cc7711", label="pairs/step", size=3)
        chart.add_line(pairs_steps, _ema(pairs_vals, 0.15), color="#cc7711", label="EMA(0.15)", line_width=2)
        chart.save(str(charts_dir / "06_pairs_per_macrobatch.png"))

    cross_fracs = []
    cross_steps = []
    for e in training_curve:
        fm = e.get("funfetti", {})
        total = fm.get("total_pairs", 0)
        cross = fm.get("n_cross_resolution_pairs", 0)
        if total > 0:
            cross_fracs.append(cross / total * 100)
            cross_steps.append(e["step"])
    if cross_fracs:
        chart = PILChart()
        chart.set_title("flops_budget_100step_v2: Cross-Resolution Pair Fraction (%)")
        chart.set_labels("Step", "Cross-Res %")
        chart.add_scatter(cross_steps, cross_fracs, color="#11aa44", label="cross-res %", size=3)
        chart.add_line(cross_steps, _ema(cross_fracs, 0.15), color="#11aa44", label="EMA(0.15)", line_width=2)
        chart.add_line([0, max(cross_steps)], [30.0, 30.0], color="#999999", label="target 30%", style="dashed")
        chart.save(str(charts_dir / "07_cross_res_fraction.png"))

    tier_pair_counts = Counter()
    for entry in training_curve:
        fm = entry.get("funfetti")
        if not fm:
            continue
        for ppr in fm.get("per_pair_resolutions", []):
            wa, ha = ppr.get("width_a", 0), ppr.get("height_a", 0)
            wb, hb = ppr.get("width_b", 0), ppr.get("height_b", 0)
            if wa > 0 and ha > 0:
                tier_a = assign_budget_tier(wa, ha)
                label_a = ANCHOR_LABELS.get(tier_a, str(tier_a))
                tier_pair_counts[label_a] += 1
            if wb > 0 and hb > 0:
                tier_b = assign_budget_tier(wb, hb)
                label_b = ANCHOR_LABELS.get(tier_b, str(tier_b))
                tier_pair_counts[label_b] += 1

    if tier_pair_counts:
        sorted_tiers = sorted(tier_pair_counts.keys())
        total_tier = sum(tier_pair_counts.values())
        tier_fracs = [tier_pair_counts[t] / total_tier * 100 for t in sorted_tiers]
        xs = list(range(len(sorted_tiers)))

        chart = PILChart()
        chart.set_title("flops_budget_100step_v2: Resolution Tier Distribution (% of images)")
        chart.set_labels("Tier Index", "Fraction (%)")
        chart.add_scatter(xs, tier_fracs, color="#7722cc", label="tier %", size=6)
        if len(xs) > 1:
            chart.add_line(xs, tier_fracs, color="#7722cc", line_width=2)
        chart.save(str(charts_dir / "08_resolution_tier_distribution.png"))

        tier_map = {str(i): {"label": t, "count": tier_pair_counts[t], "pct": tier_fracs[i]}
                    for i, t in enumerate(sorted_tiers)}
        with open(str(charts_dir.parent / "tier_distribution.json"), "w") as f:
            json.dump(tier_map, f, indent=2)

    logsnr_values = []
    sigma_values = []
    for _ in range(2000):
        sp = sampler._sample_pair_spec(allow_cross_resolution=MACROBATCH_CROSS_RES)
        for pos in [sp.image_a, sp.image_b]:
            sigma = pos.sigma
            sigma_values.append(sigma)
            if sigma <= 0:
                logsnr_values.append(15.0)  # cap for visualization
            elif sigma >= 1.0:
                logsnr_values.append(-15.0)
            else:
                logsnr_values.append(2.0 * math.log((1.0 - sigma) / sigma))

    if logsnr_values:
        n_bins = 30
        min_val = min(logsnr_values)
        max_val = max(logsnr_values)
        bin_width = (max_val - min_val) / n_bins
        bin_edges = [min_val + i * bin_width for i in range(n_bins + 1)]
        bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(n_bins)]
        bin_counts = [0] * n_bins
        for v in logsnr_values:
            idx = min(int((v - min_val) / bin_width), n_bins - 1)
            bin_counts[idx] += 1

        total_count = len(logsnr_values)
        bin_density = [c / total_count for c in bin_counts]

        chart = PILChart()
        chart.set_title("flops_budget_100step_v2: logSNR Distribution of Sampled Positions")
        chart.set_labels("logSNR (nats)", "Density")
        chart.add_scatter(bin_centers, bin_density, color="#1155cc", label="logSNR density", size=4)
        chart.add_line(bin_centers, bin_density, color="#1155cc", line_width=2)
        sigma0_frac = sum(1 for s in sigma_values if s <= 0) / len(sigma_values)
        chart.add_vline(15.0, color="#cc1111", label=f"sigma=0 ({sigma0_frac:.0%})")
        chart.add_vline(10.0, color="#999999", label="threshold=10", style="dashed")
        chart.save(str(charts_dir / "09_logsnr_distribution.png"))

    from src_ii.pair_sampler import logsnr_sampling_weight

    tier_weights_data: dict[str, list[float]] = {}
    for pos in positions:
        tier = assign_budget_tier(pos.width, pos.height)
        label = ANCHOR_LABELS.get(tier, str(tier))
        w = logsnr_sampling_weight(pos.sigma)
        tier_weights_data.setdefault(label, []).append(w)

    if tier_weights_data:
        sorted_tier_labels = sorted(tier_weights_data.keys())
        mean_weights = [sum(tier_weights_data[t]) / len(tier_weights_data[t]) for t in sorted_tier_labels]
        xs = list(range(len(sorted_tier_labels)))

        chart = PILChart()
        chart.set_title("flops_budget_100step_v2: Mean Sampling Weight by Resolution Tier")
        chart.set_labels("Tier Index", "Mean logSNR Weight")
        chart.add_scatter(xs, mean_weights, color="#cc1111", label="mean weight", size=6)
        if len(xs) > 1:
            chart.add_line(xs, mean_weights, color="#cc1111", line_width=2)
        chart.save(str(charts_dir / "10_sampling_weight_by_tier.png"))

        tier_weight_map = {str(i): {"label": t, "mean_weight": mean_weights[i],
                                     "n_positions": len(tier_weights_data[t])}
                          for i, t in enumerate(sorted_tier_labels)}
        with open(str(charts_dir.parent / "tier_sampling_weights.json"), "w") as f:
            json.dump(tier_weight_map, f, indent=2)



def _std(xs):
    """Standard deviation."""
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def _compute_flops_budget_stats(training_curve):
    """Extract FLOPS-budget specific statistics from training curve."""
    stats = {
        "n_steps_with_flops_data": 0,
        "per_step_pairs": [],
        "per_step_bins": [],
        "per_step_consumed": [],
        "per_step_cross_res": [],
    }
    for e in training_curve:
        fm = e.get("funfetti", {})
        if fm.get("macrobatch_budget") is not None:
            stats["n_steps_with_flops_data"] += 1
            stats["per_step_pairs"].append(fm.get("total_pairs", 0))
            stats["per_step_bins"].append(fm.get("n_bins", 0))
            stats["per_step_consumed"].append(fm.get("macrobatch_consumed", 0.0))
            stats["per_step_cross_res"].append(fm.get("n_cross_resolution_pairs", 0))

    if stats["per_step_pairs"]:
        stats["mean_pairs"] = sum(stats["per_step_pairs"]) / len(stats["per_step_pairs"])
        stats["min_pairs"] = min(stats["per_step_pairs"])
        stats["max_pairs"] = max(stats["per_step_pairs"])
        stats["mean_bins"] = sum(stats["per_step_bins"]) / len(stats["per_step_bins"])
        stats["min_bins"] = min(stats["per_step_bins"])
        stats["max_bins"] = max(stats["per_step_bins"])
        stats["mean_consumed"] = sum(stats["per_step_consumed"]) / len(stats["per_step_consumed"])
        stats["min_consumed"] = min(stats["per_step_consumed"])
        stats["max_consumed"] = max(stats["per_step_consumed"])
        stats["total_cross_res"] = sum(stats["per_step_cross_res"])
        stats["mean_cross_res"] = sum(stats["per_step_cross_res"]) / len(stats["per_step_cross_res"])

    return stats


if __name__ == "__main__":
    sys.exit(main())

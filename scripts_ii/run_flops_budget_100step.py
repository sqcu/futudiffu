r"""100-step FLOPS-budget BTRM training with multi-resolution data.

Exercises the FLOPS-budget macrobatch architecture:
  - BTRMPairSampler.sample_macrobatch() with budget_units=3.0
  - Variable pair count per optimizer step (adapts to resolution mix)
  - Cross-resolution pairs (images from different resolution tiers)
  - Per-bin gradient accumulation (memory-efficient backward)
  - BinPackScheduler for mixed-resolution FlexAttention batches
  - ValidationMetrics multi-indexed covariance tracker
  - TrainingArtifacts with funfetti diagnostic charts (Plots A-F)
  - Exemplar image rendering (top/bottom per head)

Key difference from run_funfetti_100step.py and run_funfetti_stratified.py:
  - Uses macrobatch_budget=3.0 instead of pairs_per_pack + grad_accum_steps.
  - The pair count is VARIABLE per step (determined by sampler + resolution mix).
  - Cross-resolution pairs are enabled by default.
  - Everything else (backbone loading, TE encoding, artifacts, exemplars) is the same.

Dataset: multi_res_trajectories/ (60 trajectories: 20x256^2, 20x512^2, 20x1024^2)

Memory lifecycle (CRITICAL):
  Phase 1: Text encoder load -> encode all prompts -> free -> empty_cache (~8 GB peak)
  Phase 2: FP8 backbone load (~6 GB) + BTRMCompoundModel setup
  Phase 3: Training (packed, gradient checkpointing) (~16 GB peak)
  Phase 4: VAE load for exemplar rendering (~160 MB, backbone stays resident)
  Phase 5: Analysis + charts generation (no GPU needed)

Expected runtime: ~10-20 min on RTX 4090 (100 steps, variable time per step).

Execution:
  PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe ^
      F:\dox\repos\ai\futudiffu\scripts_ii\run_flops_budget_100step.py
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FP8_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
TE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"
VAE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\vae\zimage.safetensors"
TOKENIZER_PATH = str(REPO_ROOT / "src" / "futudiffu" / "tokenizer")

DATASET_DIR = REPO_ROOT / "multi_res_trajectories"
OUTPUT_DIR = REPO_ROOT / "flops_budget_100step_output"

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

RUN_NAME = "flops_budget_100step"


def main():
    wall_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("=" * 70)
    print(f"  FLOPS-BUDGET 100-STEP BTRM TRAINING")
    print(f"  Steps: {N_STEPS}, macrobatch_budget: {MACROBATCH_BUDGET}")
    print(f"  Cross-resolution pairs: {MACROBATCH_CROSS_RES}")
    print(f"  LR: {LR}, schedule: {LR_SCHEDULE}, grad_clip: {GRAD_CLIP}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)

    # ==================================================================
    # Phase 1: Load dataset + build pair sampler with FLOPS weights
    # ==================================================================
    print("\n" + "=" * 60)
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

    # Use all available trajectories
    traj_ids = list(range(n_available))
    print(f"  Using all {len(traj_ids)} trajectories")

    positions = build_positions_from_v2(reader, traj_ids=traj_ids)
    print(f"  Positions: {len(positions)} across {len(traj_ids)} trajectories")

    # Resolution distribution
    res_dist = {}
    for pos in positions:
        key = f"{pos.width}x{pos.height}"
        res_dist[key] = res_dist.get(key, 0) + 1
    print(f"  Resolution distribution: {res_dist}")

    # Compute FLOPS weights
    flops_weights = compute_flops_sampling_weights_from_positions(positions)

    traj_resolutions = {}
    for pos in positions:
        if pos.traj_id not in traj_resolutions:
            traj_resolutions[pos.traj_id] = (pos.width, pos.height)
    flops_summary = summarize_flops_weights(flops_weights, traj_resolutions)
    print(f"  FLOPS weights summary:")
    print(f"    {json.dumps(flops_summary, indent=4)}")

    # Check non-degeneracy
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

    # Quick validation of macrobatch sampling
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

    # ==================================================================
    # Phase 2: Encode prompts with text encoder (then free)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Phase 2: Encoding prompts")
    print("=" * 60)

    from futudiffu.text_encoder import create_tokenizer, load_text_encoder, encode_prompt

    tokenizer = create_tokenizer(TOKENIZER_PATH)
    te_model = load_text_encoder(TE_PATH, device=device, dtype=dtype)

    prompt_cache = {}
    for idx in traj_ids:
        meta, _ = reader[idx]
        prompt = meta.get("prompt", "")
        if prompt and prompt not in prompt_cache:
            cond = encode_prompt(te_model, tokenizer, prompt, device=device)
            prompt_cache[prompt] = cond.cpu()

    n_prompts = len(prompt_cache)
    print(f"  Encoded {n_prompts} unique prompts")

    del te_model, tokenizer
    torch.cuda.empty_cache()
    vram_after_te_free = torch.cuda.memory_allocated() / 1e9
    print(f"  TE freed. VRAM: {vram_after_te_free:.2f} GB")

    # ==================================================================
    # Phase 3: Load FP8 backbone + create BTRMCompoundModel
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Phase 3: Loading backbone + creating BTRM compound model")
    print("=" * 60)

    from src_ii.zimage_model import load_zimage_rlaif
    from src_ii.btrm_lifecycle import setup_btrm_training, persist_btrm
    from src_ii.multi_lora import get_adapter_params
    from src_ii.sigma_schedule import build_sigma_schedule, resolution_shift

    # No torch.compile for training -- per-block gradient checkpointing
    # is incompatible with whole-model compile.
    _, raw_model = load_zimage_rlaif(
        FP8_PATH, device=device, dtype=dtype,
        compile_model=False, fuse=True,
    )
    vram_after_backbone = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM after backbone: {vram_after_backbone:.2f} GB")

    optimizer = setup_btrm_training(
        raw_model,
        adapter_name="rtheta",
        adapter_rank=8,
        adapter_alpha=16.0,
        adapter_init_b_std=0.01,
    )

    n_adapter = sum(p.numel() for p in get_adapter_params(raw_model, "rtheta").values())
    n_head = sum(p.numel() for p in raw_model.score_proj.parameters()) + \
             sum(p.numel() for p in raw_model.score_norm.parameters())
    print(f"  Adapter params: {n_adapter:,}")
    print(f"  Head params: {n_head:,}")
    print(f"  Total trainable: {n_adapter + n_head:,}")
    vram_after_btrm = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM after BTRM: {vram_after_btrm:.2f} GB")

    # ==================================================================
    # Build load_latent_fn
    # ==================================================================
    _v2_meta_cache = {}

    def _get_v2_meta(traj_id):
        if traj_id not in _v2_meta_cache:
            meta, accessor = reader[traj_id]
            _v2_meta_cache[traj_id] = (meta, accessor)
        return _v2_meta_cache[traj_id]

    def load_latent_fn(key):
        traj_id, step_key = key
        meta, accessor = _get_v2_meta(traj_id)
        latent = accessor[step_key].to(device=device, dtype=dtype)
        if latent.dim() == 3:
            latent = latent.unsqueeze(0)

        n_steps_traj = meta.get("n_steps", 30)
        w = meta.get("width", 1280)
        h = meta.get("height", 832)
        recorded_shift = meta.get("sampling_shift")
        if recorded_shift is not None:
            shift = float(recorded_shift)
        else:
            shift = resolution_shift(w, h)

        sigmas = build_sigma_schedule(
            n_steps_traj, sampling_shift=shift, device="cpu", dtype=torch.float32,
        )

        if step_key == "final":
            sigma_val = float(sigmas[-2].item()) if len(sigmas) > 1 else 0.01
        else:
            step_idx = int(step_key.split("_")[1])
            sigma_val = float(sigmas[step_idx].item()) if step_idx < len(sigmas) else 0.01

        timestep = torch.tensor([sigma_val], device=device, dtype=dtype)

        prompt = meta.get("prompt", "")
        cond = prompt_cache.get(prompt)
        if cond is None:
            raise ValueError(f"No cached prompt for traj {traj_id}: '{prompt[:60]}...'")
        cond = cond.to(device=device, dtype=dtype)

        num_tokens = cond.shape[1]

        return latent, timestep, cond, num_tokens

    # Validate load_latent_fn
    test_key = (traj_ids[0], positions[0].step_key)
    lat, ts, cond, nt = load_latent_fn(test_key)
    print(f"  Test load: latent={lat.shape}, timestep={ts.shape}, cond={cond.shape}")
    del lat, ts, cond
    torch.cuda.empty_cache()

    # ==================================================================
    # Build preference function (deterministic sigma-based)
    # ==================================================================
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

    # ==================================================================
    # Phase 4: Run 100-step FLOPS-budget training
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"  Phase 4: FLOPS-budget training ({N_STEPS} steps)")
    print(f"  macrobatch_budget={MACROBATCH_BUDGET}, "
          f"cross_resolution={MACROBATCH_CROSS_RES}")
    print("=" * 60)

    from src_ii.btrm_training import train_btrm_differentiable
    from src_ii.training_artifacts import TrainingArtifacts

    artifacts = TrainingArtifacts(
        output_dir=str(OUTPUT_DIR),
        run_name=RUN_NAME,
        head_names=HEAD_NAMES,
    )

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
        # FLOPS-budget specific parameters:
        macrobatch_budget=MACROBATCH_BUDGET,
        macrobatch_cross_resolution=MACROBATCH_CROSS_RES,
    )

    train_time = time.perf_counter() - t_train_start
    print(f"\n  Training complete: {train_time:.1f}s "
          f"({train_time / N_STEPS:.1f}s/step)")

    # ==================================================================
    # Phase 5: Generate analysis (charts A-F + standard)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Phase 5: Generating analysis + charts")
    print("=" * 60)

    # Compute FLOPS-budget statistics from training curve
    fb_stats = _compute_flops_budget_stats(training_curve)

    run_config = {
        "mode": "flops_budget_packed",
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
    }

    report_path = artifacts.generate_analysis(run_config=run_config)
    print(f"  Analysis generated: {report_path}")

    # Persist the model
    persist_info = persist_btrm(raw_model, "rtheta", str(OUTPUT_DIR))
    print(f"  Model persisted: {persist_info}")

    # ==================================================================
    # Phase 6: Exemplar image rendering (VAE decode)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Phase 6: Rendering exemplar images")
    print("=" * 60)

    exemplars_rendered = False
    try:
        from src_ii.exemplar_renderer import render_exemplars_from_model

        # Score a sample of images across all resolutions
        sample_keys = []
        # Pick 1 image per trajectory (up to 18)
        for idx in traj_ids:
            for pos in positions:
                if pos.traj_id == idx:
                    sample_keys.append((pos.traj_id, pos.step_key))
                    break  # one per trajectory
            if len(sample_keys) >= 18:
                break

        print(f"  Scoring {len(sample_keys)} images for exemplars")

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

    # ==================================================================
    # Phase 7: Final summary
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Phase 7: Summary")
    print("=" * 60)

    wall_total = time.perf_counter() - wall_start

    # Compute training statistics
    losses = [e.get("loss", e.get("bt_loss", 0.0)) for e in training_curve]
    grad_norms = [e.get("pre_clip_grad_norm", 0.0) for e in training_curve]
    step_times = [e.get("time_s", 0.0) for e in training_curve]

    # FLOPS-budget specific: per-step pair counts, bin counts, consumed FLOPS
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

    # Resolution tier distribution from training curve
    from collections import Counter
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
        # Per-pair tier info
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

    # FLOPS-normalized resolution breakdown
    total_flops_weighted = 0.0
    mega_flops_weighted = 0.0
    for px, ct in all_pixel_counts.items():
        if px in pixel_to_wh:
            w, h = pixel_to_wh[px]
        else:
            import math
            w = h = int(math.sqrt(px))
        flops_r = _attention_flops_ratio(w, h)
        weighted = ct * flops_r
        total_flops_weighted += weighted
        if px >= _MEGAPIXEL_THRESHOLD:
            mega_flops_weighted += weighted
    small_flops_weighted = total_flops_weighted - mega_flops_weighted
    mega_flops_pct = (mega_flops_weighted / total_flops_weighted * 100) if total_flops_weighted > 0 else 0
    small_flops_pct = (small_flops_weighted / total_flops_weighted * 100) if total_flops_weighted > 0 else 0

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
        "resolution_dist": res_dist,
        "flops_weights_summary": flops_summary,
        "sampler_stats": sampler.stats(),
        # Loss statistics
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
        "max_loss": max(losses) if losses else None,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "std_loss": _std(losses),
        # Gradient norms
        "mean_grad_norm": sum(grad_norms) / max(len(grad_norms), 1),
        "max_grad_norm": max(grad_norms) if grad_norms else None,
        # FLOPS-budget statistics
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
        # Resolution statistics
        "total_images_sampled": total_images,
        "mega_flops_pct": mega_flops_pct,
        "small_flops_pct": small_flops_pct,
        "tier_pair_counts": {str(k): v for k, v in sorted(tier_pair_counts.items())},
        # Step timing
        "step0_time_s": step_times[0] if step_times else None,
        "mean_steady_state_time_s": sum(step_times[1:]) / max(len(step_times) - 1, 1) if len(step_times) > 1 else None,
        # Model info
        "exemplars_rendered": exemplars_rendered,
        "n_adapter_params": n_adapter,
        "n_head_params": n_head,
        "end_time": datetime.now(timezone.utc).isoformat(),
    }

    # Per-head accuracy summary
    for name in HEAD_NAMES:
        accs = [e.get(f"accuracy_{name}", 0.0) for e in training_curve]
        if accs:
            summary[f"overall_accuracy_{name}"] = sum(accs) / len(accs)
            last_20 = accs[-20:]
            summary[f"last_20_accuracy_{name}"] = sum(last_20) / len(last_20)

    summary_path = OUTPUT_DIR / "run_summary.json"
    with open(str(summary_path), "w") as f:
        json.dump(summary, f, indent=2, default=str)

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
    print(f"  Exemplars: {'rendered' if exemplars_rendered else 'FAILED'}")
    print(f"  Output: {OUTPUT_DIR}")

    # Cleanup
    reader.close()
    torch.cuda.empty_cache()

    print(f"\n{'=' * 70}")
    print(f"  FLOPS-BUDGET 100-STEP RUN COMPLETE")
    print(f"{'=' * 70}")

    return 0


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

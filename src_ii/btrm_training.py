"""BTRM training loop — FLOPS-budget macrobatch path.

Single training mode:
  train_btrm_differentiable(): Full forward through the 6B backbone per step.
  Gradients flow through the adapter's LoRA matrices. FLOPS-budget macrobatch
  sampling with per-bin immediate backward for O(1) graph memory.

Import constraints:
  - IMPORTS from src_ii.btrm_lifecycle: make_training_optimizer,
    get_all_trainable_params, score_packed
  - DOES NOT import: model_manager, server, client
"""

from __future__ import annotations

import math
import os
import time
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src_ii.btrm_lifecycle import (
    make_training_optimizer,
    get_all_trainable_params,
    score_packed,
)
from src_ii.validation_metrics import ValidationMetrics, PairResult
from src_ii.incremental_save import (
    TrainingCurveWriter,
    PeriodicSaver,
    atomic_json_save,
)


# -------------------------------------------------------------------------
# Log-SNR sampling weights (Section 4 of docs/essay_training_data_distributions.md)
# -------------------------------------------------------------------------

# Canonical implementation lives in pair_sampler.logsnr_sampling_weight.
# This module re-exports it as log_snr_weight for backward compatibility
# and provides pair_sigma_weight on top.
from src_ii.pair_sampler import logsnr_sampling_weight as _logsnr_sampling_weight


def log_snr_weight(
    sigma: float,
    threshold: float = 10.0,
    interval: float = 5.0,
    decay_rate: float = 0.5,
) -> float:
    """Geometric decay weight by log-SNR. Higher weight for cleaner images.

    Delegates to pair_sampler.logsnr_sampling_weight (single source of truth).

    Flat at 1.0 for logSNR >= threshold, then decay_rate^((threshold - logSNR) / interval)
    below. sigma=0 (fully denoised, logSNR=+inf) gets FULL weight.
    """
    return _logsnr_sampling_weight(
        sigma,
        threshold=threshold,
        interval=interval,
        decay_rate=decay_rate,
    )


def pair_sigma_weight(
    sigma_a: float,
    sigma_b: float,
    threshold: float = 10.0,
    interval: float = 5.0,
    decay_rate: float = 0.5,
) -> float:
    """Compute pair weight as geometric mean of individual log-SNR weights.

    pair_weight = sqrt(weight(sigma_a) * weight(sigma_b))

    Used as a SAMPLING PROBABILITY weight (not loss multiplier).
    """
    return math.sqrt(
        log_snr_weight(sigma_a, threshold=threshold, interval=interval, decay_rate=decay_rate)
        * log_snr_weight(sigma_b, threshold=threshold, interval=interval, decay_rate=decay_rate)
    )


# -------------------------------------------------------------------------
# Extracted helpers
# -------------------------------------------------------------------------

def _compute_pair_bt_loss(
    scores_a: Tensor,
    scores_b: Tensor,
    pair: dict,
    head_names: Sequence[str],
    pref_keys: Sequence[str],
    val_tracker: ValidationMetrics,
    resolution_a: tuple[int, int],
    resolution_b: tuple[int, int],
) -> tuple[Tensor, int, dict[str, float], dict[str, int]]:
    """Per-pair BT loss across all heads + validation tracking.

    Returns (bt_sum, active_heads, accuracy_accum, accuracy_counts).
    bt_sum is a scalar tensor with grad_fn. Caller normalizes.
    """
    device = scores_a.device
    bt_sum = torch.zeros((), device=device)
    n_active = 0
    accs: dict[str, float] = {}
    counts: dict[str, int] = {}

    for head_idx, (name, pref_key) in enumerate(zip(head_names, pref_keys)):
        pref = pair[pref_key]
        if pref == 0:
            continue

        if pref > 0:
            pos_s = scores_a[head_idx]
            neg_s = scores_b[head_idx]
        else:
            pos_s = scores_b[head_idx]
            neg_s = scores_a[head_idx]

        bt = -F.logsigmoid(pos_s - neg_s)
        bt_sum = bt_sum + bt
        n_active += 1

        with torch.no_grad():
            correct = 1.0 if (pos_s > neg_s).item() else 0.0
            accs[name] = accs.get(name, 0.0) + correct
            counts[name] = counts.get(name, 0) + 1

            w_a, h_a = resolution_a
            w_b, h_b = resolution_b
            val_tracker.update(PairResult(
                head_name=name,
                correct=correct,
                loss_contribution=bt.item(),
                score_preferred=pos_s.item(),
                score_rejected=neg_s.item(),
                width_a=w_a, height_a=h_a, sigma_a=pair.get("sigma_a", 0.5),
                width_b=w_b, height_b=h_b, sigma_b=pair.get("sigma_b", 0.5),
                source_a="flops_budget_training",
                source_b="flops_budget_training",
                traj_a=pair.get("traj_a", -1),
                step_a=pair.get("step_a", ""),
                traj_b=pair.get("traj_b", -1),
                step_b=pair.get("step_b", ""),
            ))

    return bt_sum, n_active, accs, counts


def _bin_pack_images(
    image_resolutions: list[tuple[int, int]],
) -> list[list[dict]]:
    """FFD bin-pack images by effective token length.

    Returns list of bins, each bin = list of {img_idx, seq_len, width, height}.
    """
    from src_ii.bin_packer import BinPackScheduler, compute_effective_seq_len

    packer = BinPackScheduler()
    pack_items = []
    for img_idx, (w, h) in enumerate(image_resolutions):
        seq_len = compute_effective_seq_len(w, h, packer.default_cap_tokens)
        pack_items.append({
            "img_idx": img_idx,
            "seq_len": seq_len,
            "width": w,
            "height": h,
        })
    return packer.pack(pack_items)


def _compute_sigma_weight_for_pairs(
    pairs: list[dict],
    logsnr_threshold: float = 10.0,
    logsnr_interval: float = 5.0,
    logsnr_decay_rate: float = 0.5,
) -> float:
    """Average pair_sigma_weight across all pairs in a macrobatch."""
    if not pairs:
        return 1.0
    total = 0.0
    for pd in pairs:
        sigma_a = pd.get("sigma_a", 0.5)
        sigma_b = pd.get("sigma_b", 0.5)
        total += pair_sigma_weight(
            sigma_a, sigma_b,
            threshold=logsnr_threshold,
            interval=logsnr_interval,
            decay_rate=logsnr_decay_rate,
        )
    return total / len(pairs)


def _persist_step(
    step: int,
    entry: dict,
    n_steps: int,
    training_curve: list[dict],
    curve_writer: TrainingCurveWriter | None,
    val_tracker: ValidationMetrics,
    val_metrics_saver: PeriodicSaver | None,
    callback: Callable | None,
    artifacts: object | None,
    checkpoint_fn: Callable | None,
    checkpoint_steps: list[int] | None,
    output_dir: str | None,
    summary_path: str | None,
    head_names: Sequence[str],
    t_total: float,
    log_interval: int,
    model: object | None = None,
) -> None:
    """All post-step persistence: JSONL write, val metrics save, callbacks,
    checkpointing, logging. Called once per optimizer step."""
    pre_clip_val = entry["pre_clip_grad_norm"]
    post_clip_val = entry["grad_norm"]
    current_lr = entry["lr"]
    step_time = entry["time_s"]

    # Include validation tracker summary at log intervals
    if step % log_interval == 0 or step == n_steps - 1:
        entry["validation"] = val_tracker.summary()

    training_curve.append(entry)

    if curve_writer is not None:
        curve_writer.write_step(entry)

    if val_metrics_saver is not None:
        val_metrics_saver.maybe_save(step)

    if callback is not None:
        callback(step, entry)

    # TrainingArtifacts integration
    if artifacts is not None:
        _acc_dict = {n: entry.get(f"accuracy_{n}", 0.0) for n in head_names}
        _extra = {
            "bt_loss": entry["bt_loss"],
            "grad_norm_post_clip": post_clip_val,
            "pair_weight": entry["pair_weight"],
        }
        if "funfetti" in entry:
            _extra["funfetti"] = entry["funfetti"]
        artifacts.log_step(
            step=step,
            loss=entry["loss"],
            accuracy_dict=_acc_dict,
            grad_norm=pre_clip_val,
            lr=current_lr,
            extra_metrics=_extra,
            step_time=step_time,
        )
        if checkpoint_steps is not None and step in checkpoint_steps:
            print(f"  [ARTIFACTS CHECKPOINT] Saving checkpoint at step {step}")
            artifacts.save_checkpoint(step, model)

    # Legacy checkpoint path
    if checkpoint_fn is not None and checkpoint_steps is not None:
        if step in checkpoint_steps:
            print(f"  [CHECKPOINT] Saving checkpoint at step {step}")
            checkpoint_fn(step, model)
            if output_dir is not None:
                val_tracker.save_json(os.path.join(output_dir, "validation_metrics.json"))

    # Force val_metrics save + summary at checkpoints
    _is_checkpoint = checkpoint_steps is not None and step in checkpoint_steps
    if _is_checkpoint:
        if val_metrics_saver is not None:
            val_metrics_saver.flush(step)
        if summary_path is not None:
            _incremental_summary = _build_incremental_summary(
                training_curve, head_names, n_steps, t_total, step,
            )
            atomic_json_save(_incremental_summary, summary_path)

    if step % log_interval == 0 or step == n_steps - 1:
        acc_str = ", ".join(
            f"acc_{n}={entry[f'accuracy_{n}']:.1f}"
            for n in head_names
        )
        elapsed = time.perf_counter() - t_total
        print(f"  Step {step:4d}/{n_steps}: loss={entry['loss']:.4f} "
              f"gnorm={pre_clip_val:.3e}->{post_clip_val:.3e} "
              f"lr={current_lr:.3e} {acc_str} "
              f"({step_time:.1f}s, elapsed={elapsed:.0f}s)")


# -------------------------------------------------------------------------
# Main training function
# -------------------------------------------------------------------------

def train_btrm_differentiable(
    model,
    pair_sampler,
    preference_fn: Callable[[dict], dict[str, int]],
    load_latent_fn: Callable,
    head_names: Sequence[str],
    pref_keys: Sequence[str],
    n_steps: int = 100,
    lr: float = 1e-3,
    gradient_checkpointing: bool = True,
    max_grad_norm: float = 0.1,
    log_interval: int = 10,
    callback: Callable[[int, dict], None] | None = None,
    warmup_steps: int = 40,
    optimizer_type: str = "adam",
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    logsnr_threshold: float = 10.0,
    logsnr_interval: float = 5.0,
    logsnr_decay_rate: float = 0.5,
    lr_schedule: str = "warmup_only",
    checkpoint_fn: Callable[[int, object], None] | None = None,
    checkpoint_steps: list[int] | None = None,
    output_dir: str | None = None,
    artifacts: object | None = None,
    megapixel_flops_fraction: float = 0.33,
    macrobatch_budget: float = 3.0,
    macrobatch_cross_resolution: bool = True,
    curve_writer: TrainingCurveWriter | None = None,
    val_metrics_save_interval: int = 10,
    summary_path: str | None = None,
    adapter_name: str = "rtheta",
    gradient_sync_fn: Callable[[], None] | None = None,
    start_step: int = 0,
    pool=None,
) -> list[dict]:
    """Train ZImageRLAIF model with FLOPS-budget macrobatch forward passes.

    Each optimizer step:
      1. Zero gradients
      2. Sample a macrobatch of pairs via pair_sampler.sample_macrobatch()
      3. Compute preferences for each pair via preference_fn
      4. Load all latents, bin-pack all images
      5. Per-bin: score_packed → BT loss for completed pairs → immediate backward
      6. Clip gradients, (optional gradient sync), optimizer.step(), scheduler.step()

    The adapter's LoRA parameters get real gradients because the computation
    graph is intact from backbone forward through to the loss.

    Args:
        model: ZImageRLAIF model with adapter + score head.
        pair_sampler: BTRMPairSampler instance with sample_macrobatch() method.
        preference_fn: Callable(pair_metadata_dict) -> dict[str, int].
            Returns per-head preferences as {pref_key: +1/-1/0}.
            Build via dataset_io.make_reward_manifest_preference_fn() for
            ground-truth reward function scoring.
        load_latent_fn: Callable((traj_id, step_key)) -> (latent, timestep,
            conditioning, num_tokens, rope_cache). All tensors on CUDA.
        n_steps: Number of optimizer steps.
        lr: Learning rate.
        head_names: Names of the scoring heads.
        pref_keys: Keys that preference_fn returns in its output dict.
        gradient_checkpointing: Per-block checkpointing for 24 GB VRAM.
        max_grad_norm: Gradient clipping norm.
        log_interval: Print every N steps.
        callback: Optional callback(step, entry) called every optimizer step.
        warmup_steps: Number of warmup steps for LR scheduler.
        optimizer_type: "adam" or "muon".
        muon_lr: Learning rate for Muon param group.
        muon_momentum: Momentum for Muon.
        logsnr_threshold: logSNR value where geometric decay begins.
        logsnr_interval: logSNR nats per decay step.
        logsnr_decay_rate: Multiplicative factor per interval.
        lr_schedule: "warmup_only" or "warmup_cosine".
        checkpoint_fn: Optional callable(step, model) at checkpoint steps.
        checkpoint_steps: Step numbers at which to checkpoint.
        output_dir: Output directory for validation metrics.
        artifacts: Optional TrainingArtifacts instance.
        megapixel_flops_fraction: Fraction of budget for megapixel pairs.
        macrobatch_budget: FLOPS budget per step in 1024^2-equivalent units.
        macrobatch_cross_resolution: Allow cross-resolution pairs.
        curve_writer: Optional TrainingCurveWriter for incremental JSONL.
        val_metrics_save_interval: Auto-save ValidationMetrics every N steps.
        summary_path: Path for incremental summary JSON.
        adapter_name: LoRA adapter name.
        gradient_sync_fn: Optional all-reduce for multi-GPU gradient sync.
        start_step: Resume training from this step number.
        pool: Optional AcceleratorPool for multi-device dispatch.
            If provided, used as executor for score_packed and
            gather_gradients/broadcast_params are called around optimizer.step().
            If None, existing single-device behavior is unchanged.

    Returns:
        List of dicts (one per optimizer step) with keys: step, loss, bt_loss,
        accuracy_<head_name>, time_s, grad_norm, pair_weight.
    """
    if load_latent_fn is None:
        raise ValueError("load_latent_fn is required.")
    if preference_fn is None:
        raise ValueError("preference_fn is required.")
    if not hasattr(pair_sampler, 'sample_macrobatch'):
        raise ValueError(
            "pair_sampler must have a sample_macrobatch() method. "
            "Use BTRMPairSampler with flops_weights for FLOPS-budget sampling."
        )

    # Default K=1 pool: same behavior as today, no branching downstream.
    if pool is None:
        from src_ii.accelerator_pool import AcceleratorPool
        _device = next(model.parameters()).device
        pool = AcceleratorPool(
            model_factory=lambda d, _m=model: _m,
            devices=[_device],
        )

    optimizer = make_training_optimizer(
        model, adapter_name,
        lr=lr,
        optimizer_type=optimizer_type,
        muon_lr=muon_lr,
        muon_momentum=muon_momentum,
    )

    # Build LR scheduler
    if lr_schedule == "warmup_cosine":
        warmup_sched = LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_steps = max(n_steps - warmup_steps, 1)
        cosine_sched = CosineAnnealingLR(
            optimizer, T_max=cosine_steps, eta_min=0.0,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_steps],
        )
    else:
        scheduler = LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0,
            total_iters=warmup_steps,
        )

    model.gradient_checkpointing = gradient_checkpointing
    model.train()

    val_tracker = ValidationMetrics()

    _val_metrics_saver = None
    if output_dir is not None and val_metrics_save_interval > 0:
        _val_metrics_path = os.path.join(output_dir, "validation_metrics.json")
        _val_metrics_saver = PeriodicSaver(
            save_fn=lambda step: val_tracker.save_json(_val_metrics_path),
            interval=val_metrics_save_interval,
        )

    training_curve: list[dict] = []
    t_total = time.perf_counter()

    for step in range(start_step, n_steps):
        t0 = time.perf_counter()
        optimizer.zero_grad()

        accum_accs: dict[str, float] = {name: 0.0 for name in head_names}
        accum_acc_counts: dict[str, int] = {name: 0 for name in head_names}

        # Phase 1: Sample macrobatch
        from src_ii.pair_sampler import PairSpec
        macro_pair_specs = pair_sampler.sample_macrobatch(
            budget_units=macrobatch_budget,
            tier_flops_targets={1048576: megapixel_flops_fraction},
            allow_cross_resolution=macrobatch_cross_resolution,
        )

        macro_pairs: list[dict] = []
        macro_flops_consumed = 0.0
        n_cross_res = 0
        for ps in macro_pair_specs:
            pd = ps.to_pair_dict()
            prefs = preference_fn(pd)
            pd.update(prefs)
            macro_pairs.append(pd)
            macro_flops_consumed += ps.flops_cost
            if ps.cross_resolution:
                n_cross_res += 1

        # Phase 2: Load all latents
        all_images: list[tuple] = []
        image_resolutions: list[tuple[int, int]] = []
        pair_image_indices: list[tuple[int, int]] = []

        for pd in macro_pairs:
            key_a = (pd["traj_a"], pd["step_a"])
            key_b = (pd["traj_b"], pd["step_b"])

            lat_a, ts_a, cond_a, nt_a, _rc_a = load_latent_fn(key_a)
            lat_b, ts_b, cond_b, nt_b, _rc_b = load_latent_fn(key_b)

            idx_a = len(all_images)
            all_images.append((lat_a, ts_a, cond_a, nt_a))
            _, _, lh_a, lw_a = lat_a.shape
            image_resolutions.append((lw_a * 8, lh_a * 8))

            idx_b = len(all_images)
            all_images.append((lat_b, ts_b, cond_b, nt_b))
            _, _, lh_b, lw_b = lat_b.shape
            image_resolutions.append((lw_b * 8, lh_b * 8))

            pair_image_indices.append((idx_a, idx_b))

        # Phase 3: Bin-pack all images
        bins = _bin_pack_images(image_resolutions)

        step_microbatch_meta = [{
            "n_pairs": len(macro_pairs),
            "n_images": len(all_images),
            "n_bins": len(bins),
            "per_bin_item_count": [len(b) for b in bins],
            "per_bin_context_len": [
                sum(item["seq_len"] for item in b) for b in bins
            ],
            "image_resolutions": [
                {"width": w, "height": h, "pixels": w * h}
                for w, h in image_resolutions
            ],
            "total_context_len": sum(
                sum(item["seq_len"] for item in b) for b in bins
            ),
            "macrobatch_budget": macrobatch_budget,
            "macrobatch_consumed": macro_flops_consumed,
            "n_cross_resolution_pairs": n_cross_res,
            "per_pair_resolutions": [
                {
                    "width_a": pd.get("width_a", 0),
                    "height_a": pd.get("height_a", 0),
                    "width_b": pd.get("width_b", 0),
                    "height_b": pd.get("height_b", 0),
                    "cross_resolution": pd.get("cross_resolution", False),
                    "flops_cost": pd.get("flops_cost", 0.0),
                }
                for pd in macro_pairs
            ],
        }]

        # Pre-count active heads (preferences known before scoring)
        _precount_active_heads = 0
        for pd in macro_pairs:
            for pref_key in pref_keys:
                if pd.get(pref_key, 0) != 0:
                    _precount_active_heads += 1

        _norm_denom = max(_precount_active_heads, 1)

        # Build image → pair membership map for smart detaching
        _img_pair_membership: dict[int, list[int]] = {}
        for k, (idx_a, idx_b) in enumerate(pair_image_indices):
            _img_pair_membership.setdefault(idx_a, []).append(k)
            _img_pair_membership.setdefault(idx_b, []).append(k)

        # Phase 4+5: Per-bin scoring + immediate backward
        img_idx_to_score: dict[int, Tensor] = {}
        pair_processed: list[bool] = [False] * len(macro_pairs)
        total_bt_val = 0.0
        device = next(model.parameters()).device

        for bin_items in bins:
            bin_images = [
                all_images[item["img_idx"]] for item in bin_items
            ]
            bin_scores = score_packed(
                model, bin_images,
                gradient_checkpointing=gradient_checkpointing,
                executor=pool,
            )

            for local_idx, item in enumerate(bin_items):
                img_idx_to_score[item["img_idx"]] = bin_scores[local_idx]

            # Compute loss for pairs with both images now scored
            bin_bt = torch.zeros((), device=device)
            bin_active = 0

            for k, pd in enumerate(macro_pairs):
                if pair_processed[k]:
                    continue
                idx_a, idx_b = pair_image_indices[k]
                if idx_a not in img_idx_to_score or idx_b not in img_idx_to_score:
                    continue

                pair_processed[k] = True
                bt_sum, n_active, accs_delta, counts_delta = _compute_pair_bt_loss(
                    img_idx_to_score[idx_a],
                    img_idx_to_score[idx_b],
                    pd, head_names, pref_keys,
                    val_tracker,
                    image_resolutions[idx_a],
                    image_resolutions[idx_b],
                )
                bin_bt = bin_bt + bt_sum
                bin_active += n_active
                for name in accs_delta:
                    accum_accs[name] += accs_delta[name]
                    accum_acc_counts[name] += counts_delta[name]

            # Immediate backward for this bin's contributions
            if bin_active > 0:
                partial_loss = bin_bt / _norm_denom
                if partial_loss.requires_grad or partial_loss.grad_fn is not None:
                    _all_pairs_done = all(pair_processed)
                    partial_loss.backward(retain_graph=not _all_pairs_done)
                total_bt_val += bin_bt.item()

            # Detach images whose ALL pairs have been processed
            for item in bin_items:
                img_idx = item["img_idx"]
                if img_idx not in img_idx_to_score:
                    continue
                pairs_for_img = _img_pair_membership.get(img_idx, [])
                if all(pair_processed[pk] for pk in pairs_for_img):
                    img_idx_to_score[img_idx] = \
                        img_idx_to_score[img_idx].detach()

        # Compute step metrics
        total_loss_val = total_bt_val / _norm_denom if _norm_denom > 0 else 0.0
        pw = _compute_sigma_weight_for_pairs(
            macro_pairs,
            logsnr_threshold=logsnr_threshold,
            logsnr_interval=logsnr_interval,
            logsnr_decay_rate=logsnr_decay_rate,
        )

        # Multi-device gradient gathering (within-node, no-op for K=1)
        pool.gather_gradients()

        # Gradient sync (cross-node)
        if gradient_sync_fn is not None:
            gradient_sync_fn()

        # Clip gradients and step optimizer
        all_params = get_all_trainable_params(model, adapter_name)
        pre_clip_norm = torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
        pre_clip_val = pre_clip_norm.item() if isinstance(pre_clip_norm, Tensor) else pre_clip_norm
        post_clip_norm = torch.nn.utils.clip_grad_norm_(all_params, float('inf'))
        post_clip_val = post_clip_norm.item() if isinstance(post_clip_norm, Tensor) else post_clip_norm

        optimizer.step()
        scheduler.step()

        # Broadcast updated params to all replicas (no-op for K=1)
        pool.broadcast_params()

        step_time = time.perf_counter() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        # Build entry dict
        accs = {}
        for name in head_names:
            if accum_acc_counts[name] > 0:
                accs[name] = accum_accs[name] / accum_acc_counts[name]
            else:
                accs[name] = 0.0

        entry: dict = {
            "step": step,
            "loss": total_loss_val,
            "bt_loss": total_bt_val,
            "pre_clip_grad_norm": pre_clip_val,
            "grad_norm": post_clip_val,
            "lr": current_lr,
            "time_s": step_time,
            "pair_weight": pw,
        }
        for name in head_names:
            entry[f"accuracy_{name}"] = accs.get(name, 0.0)

        # Funfetti metadata
        if step_microbatch_meta:
            total_pairs = sum(m["n_pairs"] for m in step_microbatch_meta)
            total_ctx = sum(m["total_context_len"] for m in step_microbatch_meta)
            total_nfes = sum(m["n_images"] for m in step_microbatch_meta)
            all_resolutions = []
            for m in step_microbatch_meta:
                all_resolutions.extend(m["image_resolutions"])
            funfetti_meta = {
                "n_microbatches": len(step_microbatch_meta),
                "total_pairs": total_pairs,
                "total_context_len": total_ctx,
                "total_nfes": total_nfes,
                "pre_clip_grad_norm": pre_clip_val,
                "microbatches": step_microbatch_meta,
                "resolutions": all_resolutions,
            }
            m0 = step_microbatch_meta[0]
            funfetti_meta["macrobatch_budget"] = m0.get("macrobatch_budget", macrobatch_budget)
            funfetti_meta["macrobatch_consumed"] = m0.get("macrobatch_consumed", 0.0)
            funfetti_meta["n_bins"] = m0.get("n_bins", 0)
            funfetti_meta["n_cross_resolution_pairs"] = m0.get("n_cross_resolution_pairs", 0)
            funfetti_meta["per_pair_resolutions"] = m0.get("per_pair_resolutions", [])
            entry["funfetti"] = funfetti_meta

        _persist_step(
            step=step,
            entry=entry,
            n_steps=n_steps,
            training_curve=training_curve,
            curve_writer=curve_writer,
            val_tracker=val_tracker,
            val_metrics_saver=_val_metrics_saver,
            callback=callback,
            artifacts=artifacts,
            checkpoint_fn=checkpoint_fn,
            checkpoint_steps=checkpoint_steps,
            output_dir=output_dir,
            summary_path=summary_path,
            head_names=head_names,
            t_total=t_total,
            log_interval=log_interval,
            model=model,
        )

    # Post-loop persistence
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        val_tracker.save_json(os.path.join(output_dir, "validation_metrics.json"))
        print(f"  ValidationMetrics saved to {output_dir}/validation_metrics.json "
              f"({val_tracker._n_updates} pair results tracked)")

    if _val_metrics_saver is not None and n_steps > 0:
        _val_metrics_saver.flush(n_steps - 1)

    if summary_path is not None and training_curve:
        _final_summary = _build_incremental_summary(
            training_curve, head_names, n_steps, t_total, n_steps - 1,
        )
        _final_summary["status"] = "completed"
        atomic_json_save(_final_summary, summary_path)

    model.gradient_checkpointing = False
    model.eval()
    return training_curve


# -------------------------------------------------------------------------
# Incremental summary builder
# -------------------------------------------------------------------------

def _build_incremental_summary(
    training_curve: list[dict],
    head_names: Sequence[str],
    n_steps_total: int,
    t_total_start: float,
    current_step: int,
) -> dict:
    """Build a summary dict from the training curve accumulated so far."""
    n = len(training_curve)
    if n == 0:
        return {"status": "in_progress", "steps_completed": 0}

    losses = [e.get("loss", e.get("bt_loss", 0.0)) for e in training_curve]
    grad_norms = [e.get("pre_clip_grad_norm", 0.0) for e in training_curve]
    step_times = [e.get("time_s", 0.0) for e in training_curve]

    elapsed = time.perf_counter() - t_total_start

    summary = {
        "status": "in_progress",
        "steps_completed": current_step + 1,
        "steps_total": n_steps_total,
        "progress_pct": (current_step + 1) / max(n_steps_total, 1) * 100,
        "elapsed_s": elapsed,
        "initial_loss": losses[0],
        "current_loss": losses[-1],
        "min_loss": min(losses),
        "min_loss_step": losses.index(min(losses)),
        "max_loss": max(losses),
        "mean_loss": sum(losses) / len(losses),
        "mean_grad_norm": sum(grad_norms) / max(len(grad_norms), 1),
        "max_grad_norm": max(grad_norms) if grad_norms else 0.0,
        "mean_step_time_s": sum(step_times) / max(len(step_times), 1),
    }

    for name in head_names:
        accs = [e.get(f"accuracy_{name}", 0.0) for e in training_curve]
        if accs:
            summary[f"overall_accuracy_{name}"] = sum(accs) / len(accs)
            last_20 = accs[-min(20, len(accs)):]
            summary[f"last_20_accuracy_{name}"] = sum(last_20) / len(last_20)

    if step_times and current_step < n_steps_total - 1:
        steady_times = step_times[1:] if len(step_times) > 1 else step_times
        avg_step = sum(steady_times) / len(steady_times)
        remaining = n_steps_total - current_step - 1
        summary["estimated_remaining_s"] = avg_step * remaining

    return summary

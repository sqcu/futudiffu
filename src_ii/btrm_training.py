"""BTRM training loop as a named, importable function.

Two training modes:
  1. train_btrm(): Original detached path. Pre-extracted hidden states on CPU.
     Fast (~0.1s/epoch) but adapter gets zero meaningful gradients.
  2. train_btrm_differentiable(): Full forward through the 6B backbone per step.
     Slow (~10s/step) but gradients flow through the adapter's LoRA matrices.

Import constraints:
  - IMPORTS from futudiffu.btrm: bradley_terry_loss
  - DOES NOT import: model_manager, server, client
"""

from __future__ import annotations

import math
import random
import time
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from futudiffu.btrm import bradley_terry_loss
from src_ii.validation_metrics import ValidationMetrics, PairResult
from src_ii.incremental_save import (
    TrainingCurveWriter,
    PeriodicSaver,
    atomic_json_save,
    load_training_curve_jsonl,
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

    User spec: "noisy latents with log(snr(t)) > 10 get sampled uniformly,
    every -5 step decrement below log(snr(t)) 10 gets p-% geometric decay."

    Flat at 1.0 for logSNR >= threshold, then decay_rate^((threshold - logSNR) / interval)
    below. This is a one-sided exponential ramp, NOT a sigmoid or binned function.

    sigma=0 (fully denoised, logSNR=+inf) gets FULL weight -- the most
    important training signal for the reward model.

    The CONST noise model uses: x_t = sigma * noise + (1 - sigma) * x_0
    so log-SNR = 2 * ln((1 - sigma) / sigma).

    The schedule is tunable:
        threshold: logSNR value where decay begins (default 10.0)
        interval: logSNR nats per decay step (default 5.0)
        decay_rate: multiplicative factor per interval (default 0.5)

    Verification table (default params: threshold=10.0, interval=5.0, decay_rate=0.5):
        sigma=0.000  logSNR=+inf   weight=1.000  (fully denoised)
        sigma=0.034  logSNR=+6.69  weight=0.632  (near-clean)
        sigma=0.200  logSNR=+2.77  weight=0.367  (low noise)
        sigma=0.367  logSNR=+1.09  weight=0.291  (signal-dominant)
        sigma=0.500  logSNR=0.00   weight=0.250  (equal noise/signal)
        sigma=0.700  logSNR=-1.69  weight=0.198  (noise-dominant)
        sigma=0.867  logSNR=-3.75  weight=0.149  (very noisy)
        sigma=1.000  logSNR=-inf   weight~0.000  (pure noise)

    Resolution awareness: The sigma at a given step index depends on
    the resolution's sigma schedule. For 256x256 with shift~4.0, step_29
    has sigma=0.124 (logSNR=+3.9, weight=0.430). For 1280x832 with
    shift=1.0, step_29 has sigma=0.034 (logSNR=+6.7, weight=0.632).
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

    Used as a SAMPLING PROBABILITY weight (not loss multiplier). Pairs with
    higher weight (cleaner images, higher log-SNR) are sampled more often.
    This changes the gradient noise structure and critical batch size --
    formally nonequivalent to loss weighting except at infinite data.
    """
    return math.sqrt(
        log_snr_weight(sigma_a, threshold=threshold, interval=interval, decay_rate=decay_rate)
        * log_snr_weight(sigma_b, threshold=threshold, interval=interval, decay_rate=decay_rate)
    )


def compute_pairwise_bt_loss(
    scores_a: Tensor,
    scores_b: Tensor,
    prefs: list[int],
    head_idx: int,
) -> tuple[Tensor, float]:
    """Compute Bradley-Terry pairwise loss for a single head.

    Routes winners and losers based on preference labels, computes BT loss
    and accuracy. The logsquare regularizer has been removed -- the
    ScoreUnembedder's soft_tanh_cap(10.0) already bounds score magnitudes
    without imposing a target magnitude. See
    docs/directive_remove_logsquare_regularizer.md for the rationale.

    Args:
        scores_a: (B, N_heads) scores for image A in each pair.
        scores_b: (B, N_heads) scores for image B in each pair.
        prefs: List of +1 (A wins), -1 (B wins), 0 (tie) for each pair.
        head_idx: Which head column to use from scores.

    Returns:
        (bt_loss, accuracy) where bt_loss is a scalar tensor and accuracy
        is a float.
    """
    device = scores_a.device
    prefs_t = torch.tensor(prefs, device=device)

    a_wins = prefs_t > 0
    b_wins = prefs_t < 0

    n_a_wins = a_wins.sum().item()
    n_b_wins = b_wins.sum().item()

    bt_loss = scores_a.new_zeros(())
    accuracy = 0.0

    if n_a_wins > 0 and n_b_wins > 0:
        # When A wins: score_a should be > score_b
        # When B wins: score_b should be > score_a
        pos_scores = torch.cat([
            scores_a[a_wins, head_idx],
            scores_b[b_wins, head_idx],
        ])
        neg_scores = torch.cat([
            scores_b[a_wins, head_idx],
            scores_a[b_wins, head_idx],
        ])

        bt_loss = bradley_terry_loss(pos_scores, neg_scores)

        # Accuracy
        with torch.no_grad():
            accuracy = (pos_scores > neg_scores).float().mean().item()

    return bt_loss, accuracy


def train_btrm(
    btrm_model,
    training_pairs: list[dict],
    hidden_states_cpu: list[Tensor],
    n_epochs: int = 40,
    lr: float = 1e-3,
    logsquare_weight: float = 0.0,
    batch_size: int = 16,
    head_names: Sequence[str] = ("pinkify", "thisnotthat"),
    pref_keys: Sequence[str] = ("pinkify_pref", "thisnotthat_pref"),
    device: torch.device | None = None,
    grad_clip: float = 0.01,
    warmup_steps: int = 40,
) -> list[dict]:
    """Train BTRM head on PRE-EXTRACTED hidden states (HEAD-ONLY, no adapter gradients).

    WARNING: This function trains on detached hidden states. The adapter's
    LoRA parameters are in the optimizer but receive ZERO meaningful gradients
    because the hidden states were extracted outside the training loop and have
    no grad_fn connecting them to the adapter. Only the ScoreUnembedder parameters
    learn anything.

    If you want the adapter to train, use train_btrm_differentiable() instead,
    which runs full forwards through the backbone during training.

    This function exists for:
      - Fast hyperparameter sweeps where you want to reuse hidden states
      - Probe-style training where you intentionally freeze the adapter
      - Backward compatibility with existing scripts

    Note: logsquare_weight is retained as a parameter for backward compatibility
    but is ignored. The logsquare regularizer has been removed. See
    docs/directive_remove_logsquare_regularizer.md for the rationale.

    Args:
        btrm_model: BTRMCompoundModel instance (or any object with
            .optimizer(), .head, .train_mode(), .eval_mode() methods).
        training_pairs: List of dicts with keys:
            "idx_a", "idx_b": indices into hidden_states_cpu
            + one key per head in pref_keys with values +1/-1/0
        hidden_states_cpu: List of (1, N_tokens, hidden_dim) CPU tensors.
            These are DETACHED -- no grad_fn, no autograd connection to the
            model that produced them.
        n_epochs: Number of training epochs.
        lr: Learning rate.
        logsquare_weight: Ignored. Kept for backward compatibility only.
        batch_size: Mini-batch size.
        head_names: Names of the heads (for logging).
        pref_keys: Keys in training_pairs dicts for each head's preference.
        device: CUDA device. Defaults to cuda.

    Returns:
        List of dicts (one per epoch) with keys: epoch, loss, bt_loss,
        accuracy_<head_name> for each head.
    """
    if device is None:
        device = torch.device("cuda")

    # --- Detached-head guard ---
    # Check if the model has adapter params. If so, warn that this function
    # will not train them meaningfully.
    if hasattr(btrm_model, 'adapter_params'):
        n_adapter = sum(p.numel() for p in btrm_model.adapter_params())
        if n_adapter > 0:
            import warnings
            warnings.warn(
                f"train_btrm() called with a model that has {n_adapter} adapter "
                f"parameters, but this function trains on pre-extracted hidden "
                f"states (detached). The adapter will receive ZERO meaningful "
                f"gradients. Use train_btrm_differentiable() for full-forward "
                f"training that flows gradients through the adapter.",
                UserWarning,
                stacklevel=2,
            )

    # Create optimizer via compound model (includes adapter + head params)
    optimizer = btrm_model.optimizer(lr=lr)
    scheduler = LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps,
    )
    btrm_model.train_mode()
    head = btrm_model.head

    training_curve = []
    global_step = 0

    for epoch in range(n_epochs):
        random.shuffle(training_pairs)
        epoch_loss = 0.0
        epoch_bt_loss = 0.0
        epoch_grad_norm = 0.0
        epoch_correct = {name: 0.0 for name in head_names}
        epoch_total = {name: 0 for name in head_names}
        n_batches = 0

        for batch_start in range(0, len(training_pairs), batch_size):
            batch = training_pairs[batch_start:batch_start + batch_size]

            # Collect hidden states for this batch
            all_hidden_a = []
            all_hidden_b = []
            per_head_prefs = {name: [] for name in head_names}

            for pair in batch:
                all_hidden_a.append(hidden_states_cpu[pair["idx_a"]])
                all_hidden_b.append(hidden_states_cpu[pair["idx_b"]])
                for name, key in zip(head_names, pref_keys):
                    per_head_prefs[name].append(pair[key])

            # Mean-pool each hidden state individually (variable seq lengths),
            # then stack into (B, hidden_dim)
            pooled_a = torch.stack(
                [h.mean(dim=1).squeeze(0) for h in all_hidden_a], dim=0
            ).to(device=device, dtype=torch.float32)
            pooled_b = torch.stack(
                [h.mean(dim=1).squeeze(0) for h in all_hidden_b], dim=0
            ).to(device=device, dtype=torch.float32)

            # Score using the head's forward (not manual norm+proj+cap)
            # We pass pre-pooled vectors as (B, 1, hidden_dim) so the head's
            # mean(dim=1) is a no-op
            scores_a = head(pooled_a.unsqueeze(1))
            scores_b = head(pooled_b.unsqueeze(1))

            # Compute per-head BT loss (loss = BT loss only, no regularizer)
            total_bt = scores_a.new_zeros(())
            active_heads = 0

            for head_idx, name in enumerate(head_names):
                bt, acc = compute_pairwise_bt_loss(
                    scores_a, scores_b,
                    per_head_prefs[name],
                    head_idx,
                )

                if bt.item() != 0.0:
                    total_bt = total_bt + bt
                    active_heads += 1

                    n_pairs_in_batch = sum(
                        1 for p in per_head_prefs[name] if p != 0
                    )
                    epoch_correct[name] += acc * n_pairs_in_batch
                    epoch_total[name] += n_pairs_in_batch

            if active_heads > 0:
                total_bt = total_bt / active_heads

            loss = total_bt

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping -- log pre-clip norm alongside post-clip
            all_params = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
            if all_params:
                pre_clip_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=grad_clip)
            else:
                pre_clip_norm = torch.tensor(0.0)

            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_loss += loss.item()
            epoch_bt_loss += total_bt.item()
            epoch_grad_norm += (pre_clip_norm.item() if isinstance(pre_clip_norm, Tensor) else pre_clip_norm)
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_bt = epoch_bt_loss / max(n_batches, 1)
        avg_grad_norm = epoch_grad_norm / max(n_batches, 1)

        current_lr = optimizer.param_groups[0]["lr"]
        entry = {
            "epoch": epoch,
            "loss": avg_loss,
            "bt_loss": avg_bt,
            "pre_clip_grad_norm": avg_grad_norm,
            "lr": current_lr,
        }
        for name in head_names:
            acc = epoch_correct[name] / max(epoch_total[name], 1)
            entry[f"accuracy_{name}"] = acc

        training_curve.append(entry)

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            accs = ", ".join(
                f"acc_{name}={entry[f'accuracy_{name}']:.3f}"
                for name in head_names
            )
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f} "
                  f"gnorm={avg_grad_norm:.3e} "
                  f"lr={current_lr:.3e} {accs}")

    btrm_model.eval_mode()
    return training_curve


# -------------------------------------------------------------------------
# Full-forward differentiable training (gradients flow through adapter)
# -------------------------------------------------------------------------

def train_btrm_differentiable(
    btrm_model,
    training_pairs: list[dict] | None = None,
    load_latent_fn: Callable[[int], tuple[Tensor, Tensor, Tensor, int, dict]] | None = None,
    n_steps: int = 100,
    lr: float = 1e-3,
    logsquare_weight: float = 0.0,
    head_names: Sequence[str] = ("pinkify", "thisnotthat"),
    pref_keys: Sequence[str] = ("pinkify_pref", "thisnotthat_pref"),
    gradient_checkpointing: bool = True,
    max_grad_norm: float = 0.1,
    log_interval: int = 10,
    callback: Callable[[int, dict], None] | None = None,
    warmup_steps: int = 40,
    use_sigma_weighting: bool = True,
    sigma_lookup_fn: Callable[[int], float] | None = None,
    optimizer_type: str = "adam",
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    grad_accum_steps: int = 1,
    pair_sampler: object | None = None,
    preference_fn: Callable[[dict], dict[str, int]] | None = None,
    logsnr_threshold: float = 10.0,
    logsnr_interval: float = 5.0,
    logsnr_decay_rate: float = 0.5,
    lr_schedule: str = "warmup_only",
    checkpoint_fn: Callable[[int, object], None] | None = None,
    checkpoint_steps: list[int] | None = None,
    packed: bool = False,
    pairs_per_pack: int = 2,
    force_sdpa: bool = False,
    output_dir: str | None = None,
    artifacts: object | None = None,
    target_flops_ratio: float | None = None,
    min_microbatches: int = 2,
    max_microbatches: int = 8,
    megapixel_flops_fraction: float = 0.33,
    macrobatch_budget: float | None = None,
    macrobatch_cross_resolution: bool = True,
    curve_writer: TrainingCurveWriter | None = None,
    val_metrics_save_interval: int = 10,
    summary_path: str | None = None,
    reward_manifest: dict[str, Callable] | None = None,
    vae: object | None = None,
) -> list[dict]:
    """Train BTRM compound model with full forward passes through the backbone.

    Each optimizer step:
      1. Zero gradients
      2. For each microbatch (grad_accum_steps times):
         a. Pick a random pair from training_pairs OR pair_sampler
         b. Load both latents, sigmas, conditioning via load_latent_fn
         c. Full differentiable forward through backbone -> hidden -> head -> score
         d. Bradley-Terry loss between winner/loser scores
         e. (loss / grad_accum_steps).backward()
      3. Clip gradients, optimizer.step(), scheduler.step()

    This is the correct training path: the adapter's LoRA parameters get
    real gradients because the computation graph is intact from backbone
    forward through to the loss.

    Pair sourcing (mutually exclusive):
      - training_pairs + load_latent_fn: Original materialized pair table.
        Pairs are dicts with "idx_a", "idx_b" (integer indices into
        load_latent_fn) and pref_keys. Preferences come from the pair dict.
      - pair_sampler + load_latent_fn + preference_fn: On-the-fly pair
        sampling from the full combinatorial space. The sampler returns
        dicts with "traj_a", "step_a", "traj_b", "step_b" keys (NO
        preferences). Preferences are computed by preference_fn, which
        takes the pair metadata dict and returns per-head preferences.

    The preference_fn abstraction allows swapping reward functions:
      - PINKIFY/TNT: VAE decode both images -> apply reward fn -> preference
      - Scrimblo/scrongle: Different scoring -> same interface
      - Human labels: Look up from a database -> same interface
      - Self-training: Use BTRM's own predictions -> same interface

    Args:
        btrm_model: BTRMCompoundModel with adapter + head.
        training_pairs: List of dicts with keys:
            "idx_a", "idx_b": integer indices for load_latent_fn
            + one key per head in pref_keys with values +1/-1/0
            Mutually exclusive with pair_sampler.
        load_latent_fn: Callable(image_idx_or_tuple) -> (latent, timestep,
            conditioning, num_tokens, rope_cache). All tensors already on CUDA.
            When used with training_pairs: accepts int (image index).
            When used with pair_sampler: accepts (traj_id, step_key) tuple.
        n_steps: Number of optimizer steps (each may involve multiple microbatches).
        lr: Learning rate for AdamW (all params when optimizer_type="adam",
            or ScoreUnembedder only when optimizer_type="muon").
        logsquare_weight: Ignored. Kept for backward compatibility only.
            The logsquare regularizer has been removed. See
            docs/directive_remove_logsquare_regularizer.md.
        head_names: Names of the heads (for logging).
        pref_keys: Keys in training_pairs dicts for each head's preference.
            When using pair_sampler, these are the keys that preference_fn
            must return in its output dict.
        gradient_checkpointing: Per-block checkpointing for 24 GB VRAM.
        max_grad_norm: Gradient clipping norm.
        log_interval: Print every N steps.
        callback: Optional callback(step, entry) called every optimizer step.
        warmup_steps: Number of warmup steps for LR scheduler.
        use_sigma_weighting: If True, sample training pairs with probability
            proportional to pair_sigma_weight(sigma_a, sigma_b) instead of
            uniformly. Cleaner pairs (higher log-SNR) are sampled more often.
            This is SAMPLING weighting, not loss weighting -- it changes the
            gradient noise structure. Requires sigma_lookup_fn. See
            docs/essay_training_data_distributions.md Section 4.
            Only applies when using training_pairs (not pair_sampler, which
            has its own logSNR weighting built in).
        sigma_lookup_fn: Callable(image_idx) -> sigma_float. Returns the
            sigma value for a given image index. Required when
            use_sigma_weighting=True. If use_sigma_weighting=True and this
            is None, sigma weighting is silently disabled.
            Only applies when using training_pairs (not pair_sampler).
        optimizer_type: "adam" or "muon". When "muon", uses Muon for LoRA
            adapter params and AdamW for ScoreUnembedder params.
        muon_lr: Learning rate for Muon param group (default 0.02).
        muon_momentum: Momentum for Muon (default 0.95).
        grad_accum_steps: Number of microbatches to accumulate before each
            optimizer step. Each microbatch is one random pair. Loss is
            divided by grad_accum_steps for correct scaling. Default 1
            preserves existing behavior.
        pair_sampler: On-the-fly pair sampler (BTRMPairSampler instance).
            Must have a .sample_pair() method returning dicts with keys:
            "traj_a", "step_a", "sigma_a", "traj_b", "step_b", "sigma_b".
            Does NOT include preferences (those come from preference_fn).
            Mutually exclusive with training_pairs.
        preference_fn: Callable(pair_metadata_dict) -> dict[str, int].
            Takes a pair metadata dict (as returned by pair_sampler) and
            returns a dict mapping pref_key -> preference (+1/-1/0).
            Required when using pair_sampler. Ignored when using
            training_pairs (preferences come from the pair dicts directly).
        logsnr_threshold: logSNR value where geometric decay begins (default 5.0).
            Passed to log_snr_weight / pair_sigma_weight for sigma sampling weights.
        logsnr_interval: logSNR nats per decay step (default 5.0).
        logsnr_decay_rate: Multiplicative factor per interval (default 0.75).
        lr_schedule: LR schedule type. Options:
            "warmup_only" (default): Linear warmup then constant LR.
            "warmup_cosine": Linear warmup then cosine decay to 0 at n_steps.
        checkpoint_fn: Optional callable(step, btrm_model) called at each
            checkpoint step. Use this to save intermediate adapter state.
        checkpoint_steps: List of step numbers at which to call checkpoint_fn.
            If None, no intermediate checkpoints are saved.
        packed: If True, use FlexAttention batch packing for multi-image
            forward passes. Samples pairs_per_pack pairs per microbatch,
            collects all 2*pairs_per_pack images, and scores them in a
            single packed forward via score_differentiable_packed().
            Default False preserves existing serial behavior.
        pairs_per_pack: Number of pairs to sample per packed microbatch.
            Only used when packed=True. Each pair contributes 2 images,
            so a packed forward processes 2*pairs_per_pack images.
            Default 2 (4 images per packed forward).
        force_sdpa: When packed=True, force SDPA attention instead of
            SageAttention. Useful for correctness validation.
        output_dir: Optional output directory path. When provided,
            the ValidationMetrics tracker is saved to
            ``{output_dir}/validation_metrics.json`` at each checkpoint
            step and at the end of training. The tracker summary is also
            included in each log-interval entry.
        artifacts: Optional TrainingArtifacts instance
            (from src_ii.training_artifacts). When provided, the training
            loop automatically logs each step via artifacts and saves
            checkpoints at checkpoint_steps. The artifacts callback is
            composed with the existing callback (both fire). After the
            training loop, callers should call artifacts.generate_analysis()
            to produce charts and markdown.
        target_flops_ratio: When set (float > 0) and packed=True, replaces
            the fixed grad_accum_steps loop with FLOPS-budget-based
            accumulation. Each macrobatch (optimizer step) accumulates
            microbatches until cumulative FLOPS ratio meets this target.
            The FLOPS ratio is relative to one 1280x832 reference forward.
            E.g., target_flops_ratio=4.0 means "accumulate enough
            microbatches to total 4x the FLOPS of one megapixel forward."
            When set, uses two-phase sampling per macrobatch:
              Phase 1: Megapixel pairs (33% of target by default)
              Phase 2: Small pairs (67% of target by default)
            This directly implements the funfetti spec's 33/67 FLOPS split.
            When None (default), uses fixed grad_accum_steps (legacy).
        min_microbatches: Minimum microbatches per macrobatch when using
            FLOPS-normalized accumulation. Prevents degenerate single-
            microbatch steps. Default 2.
        max_microbatches: Maximum microbatches per macrobatch when using
            FLOPS-normalized accumulation. Prevents runaway accumulation
            if only tiny images are sampled. Default 8.
        megapixel_flops_fraction: Fraction of target_flops_ratio to
            allocate to megapixel pairs in Phase 1. Default 0.33 (33%).
            The remainder (1 - megapixel_flops_fraction) goes to small
            pairs in Phase 2.
        macrobatch_budget: FLOPS budget per optimizer step in 1024^2-equivalent
            units. When set (float > 0) and packed=True, replaces the fixed
            ``pairs_per_pack * grad_accum_steps`` pair count with variable-
            length FLOPS-budget macrobatches. The pair_sampler must have a
            ``sample_macrobatch()`` method. The training loop samples ALL
            pairs for the macrobatch upfront, packs them into bins, and runs
            however many forward passes the packer produces.
            Default budget: 3.0 (~ one 1024^2 pair + many small pairs).
            When None (default), uses the old fixed pair-count mode.
        macrobatch_cross_resolution: When macrobatch_budget is set, allow
            cross-resolution pairs (images from different resolution tiers).
            Default True.
        curve_writer: Optional TrainingCurveWriter instance. When provided,
            each step's metrics dict is written as a JSONL line immediately
            after computation. This makes the training curve incrementally
            durable -- if the process crashes, all completed steps are on
            disk. The writer is NOT closed by this function (the caller
            owns its lifecycle). When None, no incremental writing occurs
            and the training curve is only available from the returned list.
        val_metrics_save_interval: How often (in steps) to auto-save the
            ValidationMetrics tracker to its JSON file. Default 10. Only
            effective when output_dir is set. The save is atomic (write to
            temp file, rename). Set to 0 to disable auto-saving (only
            save at checkpoints and end-of-run, as before).
        summary_path: Optional path for an incremental summary JSON file.
            When provided, the summary dict is written at each checkpoint
            step and at end-of-run. The summary includes loss statistics,
            per-head accuracy, and timing data computed from the training
            curve so far. The write is atomic.
        reward_manifest: Optional dict mapping head name strings to callable
            scoring functions. When provided (with vae), preference labels
            are derived from ground truth reward function evaluation:
              1. VAE-decode both latents to pixel tensors (torch.no_grad)
              2. Score each pixel tensor with manifest[head_name]()
              3. Preference = +1 if score_a > score_b, -1 if score_b > score_a
            Each head gets INDEPENDENT preferences. This replaces the
            external preference_fn for the reward-function-based path.
            When None (default), uses the external preference_fn or
            training_pairs preferences (backward compatible).
            Scoring function interface: (3, H, W) float32 tensor in [0,1]
            -> scalar tensor.
        vae: Optional loaded VAE model for decoding latents to pixel
            tensors. Required when reward_manifest is provided. The VAE
            is used only inside torch.no_grad() for preference label
            generation -- it is NOT part of the training graph.

    Returns:
        List of dicts (one per optimizer step) with keys: step, loss, bt_loss,
        accuracy_<head_name>, time_s, grad_norm, pair_weight.
        Note: logsq_loss is no longer present (regularizer removed).
    """
    # Validate pair sourcing: exactly one of training_pairs or pair_sampler
    _using_sampler = pair_sampler is not None
    if _using_sampler and training_pairs is not None:
        raise ValueError(
            "training_pairs and pair_sampler are mutually exclusive. "
            "Provide one or the other, not both."
        )
    if not _using_sampler and training_pairs is None:
        raise ValueError(
            "Either training_pairs or pair_sampler must be provided."
        )
    if load_latent_fn is None:
        raise ValueError("load_latent_fn is required.")

    # Reward manifest: when provided, build a preference_fn that VAE-decodes
    # both latents and scores them with each head's ground truth function.
    # This replaces any externally-provided preference_fn.
    _using_reward_manifest = reward_manifest is not None
    if _using_reward_manifest:
        if vae is None:
            raise ValueError(
                "vae is required when reward_manifest is provided. The VAE "
                "decodes latents to pixel tensors for reward function scoring."
            )
        # Validate that manifest keys align with head_names and pref_keys
        for head_name, pref_key in zip(head_names, pref_keys):
            if head_name not in reward_manifest:
                raise ValueError(
                    f"reward_manifest missing key '{head_name}'. "
                    f"Available: {list(reward_manifest.keys())}. "
                    f"Expected keys matching head_names: {list(head_names)}"
                )

        from futudiffu.vae import vae_decode as _vae_decode

        def _reward_manifest_preference_fn(pair: dict) -> dict[str, int]:
            """Compute per-head preferences by VAE-decoding and scoring.

            All operations inside torch.no_grad(). The VAE decode and
            reward function scores produce LABELS, not gradients.
            """
            with torch.no_grad():
                # Load latents for both images
                key_a = (pair["traj_a"], pair["step_a"])
                key_b = (pair["traj_b"], pair["step_b"])

                lat_a, _, _, _, _ = load_latent_fn(key_a)
                lat_b, _, _, _, _ = load_latent_fn(key_b)

                # VAE decode: latent -> pixel tensor [B, 3, H*8, W*8] in [0, 1]
                pixel_a = _vae_decode(vae, lat_a)  # (1, 3, H, W) float
                pixel_b = _vae_decode(vae, lat_b)  # (1, 3, H, W) float

                # Squeeze to (3, H, W) and ensure float32 for scoring
                pixel_a = pixel_a[0].float()
                pixel_b = pixel_b[0].float()

                prefs = {}
                for head_name, pref_key in zip(head_names, pref_keys):
                    score_fn = reward_manifest[head_name]
                    score_a = score_fn(pixel_a)
                    score_b = score_fn(pixel_b)

                    sa = float(score_a.item()) if isinstance(score_a, Tensor) else float(score_a)
                    sb = float(score_b.item()) if isinstance(score_b, Tensor) else float(score_b)

                    if sa > sb:
                        prefs[pref_key] = 1   # A wins for this head
                    elif sb > sa:
                        prefs[pref_key] = -1  # B wins for this head
                    else:
                        prefs[pref_key] = 0   # tie

                return prefs

        # Override any externally-provided preference_fn with the
        # reward-manifest-based one. The manifest is the sole source
        # of preference labels.
        preference_fn = _reward_manifest_preference_fn

    if _using_sampler and preference_fn is None:
        raise ValueError(
            "preference_fn is required when using pair_sampler (and no "
            "reward_manifest is provided). The sampler returns pair metadata "
            "without preferences; preference_fn computes them. Example: "
            "preference_fn=lambda pair: {'pinkify_pref': 1, 'thisnotthat_pref': -1}"
        )

    if _using_sampler:
        # When using pair_sampler, sigma weighting is built into the sampler.
        # Disable the materialized-pair sigma weighting path.
        use_sigma_weighting = False
        pair_weights = None
    else:
        # Resolve sigma weighting: need the lookup function
        if use_sigma_weighting and sigma_lookup_fn is None:
            import warnings
            warnings.warn(
                "use_sigma_weighting=True but sigma_lookup_fn is None. "
                "Sigma weighting disabled for this run.",
                UserWarning,
                stacklevel=2,
            )
            use_sigma_weighting = False

        # Pre-compute sampling weights for all training pairs.
        # When use_sigma_weighting is True, pairs involving cleaner images
        # (higher log-SNR) get sampled proportionally more often.
        # When False, all pairs have equal weight (uniform sampling).
        if use_sigma_weighting:
            pair_weights = [
                pair_sigma_weight(
                    sigma_lookup_fn(pair["idx_a"]),
                    sigma_lookup_fn(pair["idx_b"]),
                    threshold=logsnr_threshold,
                    interval=logsnr_interval,
                    decay_rate=logsnr_decay_rate,
                )
                for pair in training_pairs
            ]
        else:
            pair_weights = None  # uniform sampling

    optimizer = btrm_model.optimizer(
        lr=lr,
        optimizer_type=optimizer_type,
        muon_lr=muon_lr,
        muon_momentum=muon_momentum,
    )

    # Build LR scheduler based on lr_schedule parameter
    if lr_schedule == "warmup_cosine":
        # Phase 1: linear warmup from ~0 to peak LR
        warmup_sched = LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0,
            total_iters=warmup_steps,
        )
        # Phase 2: cosine decay from peak LR to 0
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
        # Default: warmup only, then constant LR
        scheduler = LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0,
            total_iters=warmup_steps,
        )
    btrm_model.train_mode()

    # Validation metrics tracker (Layer 4: multi-indexed covariance)
    val_tracker = ValidationMetrics()

    # Incremental persistence: auto-save ValidationMetrics every N steps.
    # The PeriodicSaver wraps the atomic save_json call so the training
    # loop does not need interval logic.
    _val_metrics_saver = None
    if output_dir is not None and val_metrics_save_interval > 0:
        import os as _os
        _val_metrics_path = _os.path.join(output_dir, "validation_metrics.json")
        _val_metrics_saver = PeriodicSaver(
            save_fn=lambda step: val_tracker.save_json(_val_metrics_path),
            interval=val_metrics_save_interval,
        )

    training_curve = []
    t_total = time.perf_counter()

    for step in range(n_steps):
        t0 = time.perf_counter()

        optimizer.zero_grad()

        accum_loss = 0.0
        accum_bt = 0.0
        accum_pw = 0.0
        accum_accs = {name: 0.0 for name in head_names}
        accum_acc_counts = {name: 0 for name in head_names}

        # Per-step metadata for funfetti diagnostics
        step_microbatch_meta = []  # one entry per microbatch

        # ---------------------------------------------------------------
        # Determine macrobatch mode:
        #   1. FLOPS-budget mode (macrobatch_budget is set, packed=True,
        #      pair_sampler has sample_macrobatch). Variable pair count
        #      per optimizer step, variable bin count.
        #   2. Stratified fixed-count mode (legacy, packed=True,
        #      pair_sampler has sample_stratified_batch). Fixed pair count
        #      = grad_accum_steps * pairs_per_pack.
        #   3. Unstratified fixed-count mode (packed or serial, default).
        # ---------------------------------------------------------------
        _use_flops_budget = (
            macrobatch_budget is not None
            and packed
            and _using_sampler
            and hasattr(pair_sampler, 'sample_macrobatch')
        )

        # Track actual forward pass count for FLOPS-budget mode
        _actual_n_bins = 0

        if _use_flops_budget:
            # ===========================================================
            # FLOPS-BUDGET MACROBATCH PATH
            #
            # The macrobatch is defined by a FLOPS budget, not pair count.
            # 1. Sample all pairs upfront via sample_macrobatch()
            # 2. Compute preferences for each pair
            # 3. Load all latents, collect all images
            # 4. Bin-pack all images into J bins
            # 5. Run J forward passes, accumulate gradients
            # 6. One optimizer step
            # ===========================================================
            from src_ii.bin_packer import BinPackScheduler, compute_effective_seq_len
            from src_ii.pair_sampler import PairSpec

            # Phase 1: Sample macrobatch
            macro_pair_specs = pair_sampler.sample_macrobatch(
                budget_units=macrobatch_budget,
                tier_flops_targets={1048576: megapixel_flops_fraction},
                allow_cross_resolution=macrobatch_cross_resolution,
            )

            # Convert PairSpec objects to pair dicts and compute preferences
            macro_pairs = []
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

            # Phase 2: Load all latents and collect images
            all_images = []       # (latent, timestep, conditioning, num_tokens)
            image_resolutions = []  # (width, height) for bin packing
            # Map: pair_index -> (img_idx_a, img_idx_b) in all_images
            pair_image_indices = []

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

            # Phase 3: Bin-pack ALL images into bins
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

            bins = packer.pack(pack_items)
            _actual_n_bins = len(bins)

            # Collect metadata
            step_microbatch_meta.append({
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
            })

            # Phase 4+5: Per-bin gradient accumulation.
            #
            # Score each bin via packed forward, then IMMEDIATELY compute
            # BT loss for any pairs that now have BOTH images scored, and
            # backward the partial loss. This frees the computation graph
            # for each bin after its gradients are accumulated, preventing
            # all J bins' graphs from being held in GPU memory simultaneously.
            #
            # The alternative (scoring all bins, then computing loss, then
            # one backward) requires O(J * graph_size) memory. Per-bin
            # accumulation requires O(max_active_graphs) memory, where
            # max_active_graphs is at most 2 (current bin + one prior bin
            # with an unmatched cross-bin pair partner).
            #
            # Loss normalization: we PRE-COUNT active_heads from the pair
            # metadata (preferences are known before scoring). This gives
            # a constant normalization denominator that does not change as
            # bins are processed, ensuring consistent gradient scale
            # regardless of bin processing order.

            # Pre-count active heads from metadata (no computation graph needed)
            _precount_active_heads = 0
            for pd in macro_pairs:
                for pref_key in pref_keys:
                    if pd.get(pref_key, 0) != 0:
                        _precount_active_heads += 1

            img_idx_to_score: dict[int, Tensor] = {}
            # Track which pairs have been processed (both images scored)
            pair_processed: list[bool] = [False] * len(macro_pairs)
            total_bt_val = 0.0
            active_heads = 0
            device = next(btrm_model.backbone.parameters()).device

            # Normalization denominator: use pre-counted heads if nonzero,
            # else fall back to 1 to avoid division by zero.
            _norm_denom = max(_precount_active_heads, 1)

            # Build image -> pair membership map for smart detaching.
            # An image's computation graph can only be freed once ALL pairs
            # it participates in have been processed.
            _img_pair_membership: dict[int, list[int]] = {}
            for k, (idx_a, idx_b) in enumerate(pair_image_indices):
                _img_pair_membership.setdefault(idx_a, []).append(k)
                _img_pair_membership.setdefault(idx_b, []).append(k)

            for bin_items in bins:
                bin_images = [
                    all_images[item["img_idx"]] for item in bin_items
                ]
                bin_scores = btrm_model.score_differentiable_packed(
                    bin_images,
                    gradient_checkpointing=gradient_checkpointing,
                    force_sdpa=force_sdpa,
                )  # (len(bin_items), N_heads) with grad_fn

                for local_idx, item in enumerate(bin_items):
                    img_idx_to_score[item["img_idx"]] = bin_scores[local_idx]

                # After scoring this bin, compute loss for any pairs that
                # now have BOTH images scored and haven't been processed yet.
                bin_bt = torch.zeros((), device=device)
                bin_active = 0

                for k, pd in enumerate(macro_pairs):
                    if pair_processed[k]:
                        continue
                    idx_a, idx_b = pair_image_indices[k]
                    if idx_a not in img_idx_to_score or idx_b not in img_idx_to_score:
                        continue

                    # Both images are now scored -- compute pairwise BT loss
                    pair_processed[k] = True
                    scores_a_k = img_idx_to_score[idx_a]
                    scores_b_k = img_idx_to_score[idx_b]

                    for head_idx, (name, pref_key) in enumerate(zip(head_names, pref_keys)):
                        pref = pd[pref_key]
                        if pref == 0:
                            continue

                        if pref > 0:
                            pos_s = scores_a_k[head_idx]
                            neg_s = scores_b_k[head_idx]
                        else:
                            pos_s = scores_b_k[head_idx]
                            neg_s = scores_a_k[head_idx]

                        bt = -F.logsigmoid(pos_s - neg_s)
                        bin_bt = bin_bt + bt
                        bin_active += 1
                        active_heads += 1

                        with torch.no_grad():
                            correct = 1.0 if (pos_s > neg_s).item() else 0.0
                            accum_accs[name] += correct
                            accum_acc_counts[name] += 1

                            w_a, h_a = image_resolutions[idx_a]
                            w_b, h_b = image_resolutions[idx_b]
                            sigma_a_val = pd.get("sigma_a", 0.5)
                            sigma_b_val = pd.get("sigma_b", 0.5)
                            val_tracker.update(PairResult(
                                head_name=name,
                                correct=correct,
                                loss_contribution=bt.item(),
                                score_preferred=pos_s.item(),
                                score_rejected=neg_s.item(),
                                width_a=w_a, height_a=h_a, sigma_a=sigma_a_val,
                                width_b=w_b, height_b=h_b, sigma_b=sigma_b_val,
                                source_a="flops_budget_training",
                                source_b="flops_budget_training",
                                traj_a=pd.get("traj_a", -1),
                                step_a=pd.get("step_a", ""),
                                traj_b=pd.get("traj_b", -1),
                                step_b=pd.get("step_b", ""),
                            ))

                # Backward the partial loss for this bin's pair contributions.
                # Gradients accumulate into the adapter/head parameters across
                # all bins. Normalization uses the pre-counted denominator for
                # consistent gradient scale across all bins.
                #
                # retain_graph: We must retain the graph when there are still
                # unprocessed pairs whose images span across bins. If ALL pairs
                # are now processed, this is the last backward and we can free
                # the graph. Otherwise, some image scores from prior bins may
                # be referenced by future cross-bin pairs, and their graph
                # segments must survive until those pairs are backward'd.
                if bin_active > 0:
                    partial_loss = bin_bt / _norm_denom
                    if partial_loss.requires_grad or partial_loss.grad_fn is not None:
                        _all_pairs_done = all(pair_processed)
                        partial_loss.backward(retain_graph=not _all_pairs_done)
                    total_bt_val += bin_bt.item()

                # Detach images whose ALL pairs have been processed. This frees
                # the computation graph for that bin. Images with unprocessed
                # cross-bin pairs must keep their graph alive until the partner
                # bin is scored and the pair loss is backward'd.
                for item in bin_items:
                    img_idx = item["img_idx"]
                    if img_idx not in img_idx_to_score:
                        continue
                    pairs_for_img = _img_pair_membership.get(img_idx, [])
                    if all(pair_processed[pk] for pk in pairs_for_img):
                        img_idx_to_score[img_idx] = \
                            img_idx_to_score[img_idx].detach()

            # Compute normalized loss value for reporting
            total_loss_val = total_bt_val / _norm_denom if _norm_denom > 0 else 0.0

            accum_loss += total_loss_val
            accum_bt += total_bt_val

            # Average sigma weight across all pairs
            pw = 0.0
            for pd in macro_pairs:
                sigma_a = pd.get("sigma_a", 0.5)
                sigma_b = pd.get("sigma_b", 0.5)
                pw += pair_sigma_weight(
                    sigma_a, sigma_b,
                    threshold=logsnr_threshold,
                    interval=logsnr_interval,
                    decay_rate=logsnr_decay_rate,
                )
            pw /= max(len(macro_pairs), 1)
            accum_pw += pw

        else:
            # ===========================================================
            # LEGACY FIXED-PAIR-COUNT PATH (stratified or unstratified)
            # ===========================================================

            # Stratified macrobatch sampling (packed path only).
            _stratified_pairs = None
            if packed and _using_sampler and hasattr(pair_sampler, 'sample_stratified_batch'):
                total_pairs_this_macro = grad_accum_steps * pairs_per_pack
                _stratified_pairs = pair_sampler.sample_stratified_batch(
                    total_pairs_this_macro,
                    mega_fraction=megapixel_flops_fraction,
                )
                # Compute preferences for all stratified pairs
                for sp in _stratified_pairs:
                    prefs = preference_fn(sp)
                    sp.update(prefs)

            for micro in range(grad_accum_steps):
                if packed:
                    # -------------------------------------------------------
                    # PACKED PATH: Sample K pairs, collect 2K images, bin-pack
                    # into FlexAttention batches, score via packed forward(s).
                    # -------------------------------------------------------
                    from src_ii.bin_packer import BinPackScheduler, compute_effective_seq_len

                    pairs_this_micro = []
                    all_images = []       # (latent, timestep, conditioning, num_tokens)
                    image_pair_map = []   # (pair_idx, "a" or "b") for each image
                    image_resolutions = []  # (width, height) for bin packing

                    for k in range(pairs_per_pack):
                        if _stratified_pairs is not None:
                            # Use pre-sampled stratified pair
                            pair_idx = micro * pairs_per_pack + k
                            pair = _stratified_pairs[pair_idx]
                            key_a = (pair["traj_a"], pair["step_a"])
                            key_b = (pair["traj_b"], pair["step_b"])
                        elif _using_sampler:
                            pair = pair_sampler.sample_pair()
                            prefs = preference_fn(pair)
                            pair.update(prefs)
                            key_a = (pair["traj_a"], pair["step_a"])
                            key_b = (pair["traj_b"], pair["step_b"])
                        else:
                            if pair_weights is not None:
                                pair = random.choices(training_pairs, weights=pair_weights, k=1)[0]
                            else:
                                pair = random.choice(training_pairs)
                            key_a = pair["idx_a"]
                            key_b = pair["idx_b"]

                        pairs_this_micro.append(pair)

                        # Load latents (returns latent, timestep, cond, num_tokens, rope_cache)
                        lat_a, ts_a, cond_a, nt_a, _rc_a = load_latent_fn(key_a)
                        lat_b, ts_b, cond_b, nt_b, _rc_b = load_latent_fn(key_b)

                        idx_a = len(all_images)
                        all_images.append((lat_a, ts_a, cond_a, nt_a))
                        image_pair_map.append((k, "a"))
                        # Infer resolution from latent shape: (1, C, H/8, W/8)
                        # Pixel resolution = latent_spatial * vae_scale (8)
                        _, _, lh_a, lw_a = lat_a.shape
                        image_resolutions.append((lw_a * 8, lh_a * 8))

                        idx_b = len(all_images)
                        all_images.append((lat_b, ts_b, cond_b, nt_b))
                        image_pair_map.append((k, "b"))
                        _, _, lh_b, lw_b = lat_b.shape
                        image_resolutions.append((lw_b * 8, lh_b * 8))

                    # --- Bin-pack images into FlexAttention batches ---
                    # Build bin packing items with seq_len computed from resolution
                    packer = BinPackScheduler()
                    pack_items = []
                    for img_idx, (img_tuple, (w, h)) in enumerate(
                        zip(all_images, image_resolutions)
                    ):
                        seq_len = compute_effective_seq_len(w, h, packer.default_cap_tokens)
                        pack_items.append({
                            "img_idx": img_idx,
                            "seq_len": seq_len,
                            "width": w,
                            "height": h,
                        })

                    bins = packer.pack(pack_items)

                    # Collect per-microbatch metadata for funfetti diagnostics
                    micro_meta = {
                        "n_pairs": len(pairs_this_micro),
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
                    }
                    step_microbatch_meta.append(micro_meta)

                    # Score each bin via packed forward, collecting scores
                    # keyed by original image index
                    img_idx_to_score: dict[int, Tensor] = {}

                    for bin_items in bins:
                        bin_images = [
                            all_images[item["img_idx"]] for item in bin_items
                        ]
                        bin_scores = btrm_model.score_differentiable_packed(
                            bin_images,
                            gradient_checkpointing=gradient_checkpointing,
                            force_sdpa=force_sdpa,
                        )  # (len(bin_items), N_heads) with grad_fn

                        for local_idx, item in enumerate(bin_items):
                            img_idx_to_score[item["img_idx"]] = bin_scores[local_idx]

                    # Reassemble scores in original image order -> (2K, N_heads)
                    all_scores = torch.stack(
                        [img_idx_to_score[i] for i in range(len(all_images))],
                        dim=0,
                    )

                    # Compute BT loss across all K pairs
                    # Loss is normalized by number of active head-pair contributions,
                    # NOT by number of images. This ensures the gradient magnitude
                    # scales with the number of informative pairs, not batch size.
                    total_bt = all_scores.new_zeros(())
                    active_heads = 0

                    for k, pair in enumerate(pairs_this_micro):
                        scores_a_k = all_scores[2 * k]      # (N_heads,)
                        scores_b_k = all_scores[2 * k + 1]  # (N_heads,)

                        for head_idx, (name, pref_key) in enumerate(zip(head_names, pref_keys)):
                            pref = pair[pref_key]
                            if pref == 0:
                                continue

                            if pref > 0:
                                pos_s = scores_a_k[head_idx]
                                neg_s = scores_b_k[head_idx]
                            else:
                                pos_s = scores_b_k[head_idx]
                                neg_s = scores_a_k[head_idx]

                            bt = -F.logsigmoid(pos_s - neg_s)
                            total_bt = total_bt + bt
                            active_heads += 1

                            with torch.no_grad():
                                correct = 1.0 if (pos_s > neg_s).item() else 0.0
                                accum_accs[name] += correct
                                accum_acc_counts[name] += 1

                                # Track in ValidationMetrics
                                w_a, h_a = image_resolutions[2 * k]
                                w_b, h_b = image_resolutions[2 * k + 1]
                                sigma_a_val = pair.get("sigma_a", 0.5) if _using_sampler else 0.5
                                sigma_b_val = pair.get("sigma_b", 0.5) if _using_sampler else 0.5
                                val_tracker.update(PairResult(
                                    head_name=name,
                                    correct=correct,
                                    loss_contribution=bt.item(),
                                    score_preferred=pos_s.item(),
                                    score_rejected=neg_s.item(),
                                    width_a=w_a, height_a=h_a, sigma_a=sigma_a_val,
                                    width_b=w_b, height_b=h_b, sigma_b=sigma_b_val,
                                    source_a="packed_training",
                                    source_b="packed_training",
                                    traj_a=pair.get("traj_a", -1) if _using_sampler else -1,
                                    step_a=pair.get("step_a", "") if _using_sampler else "",
                                    traj_b=pair.get("traj_b", -1) if _using_sampler else -1,
                                    step_b=pair.get("step_b", "") if _using_sampler else "",
                                ))

                    if active_heads > 0:
                        total_bt = total_bt / active_heads

                    loss = total_bt

                    # Sigma weight: average across all pairs in the pack
                    pw = 0.0
                    for pair in pairs_this_micro:
                        if _using_sampler:
                            sigma_a = pair.get("sigma_a", 0.5)
                            sigma_b = pair.get("sigma_b", 0.5)
                        elif use_sigma_weighting and sigma_lookup_fn is not None:
                            sigma_a = sigma_lookup_fn(pair.get("idx_a", key_a))
                            sigma_b = sigma_lookup_fn(pair.get("idx_b", key_b))
                        else:
                            sigma_a = sigma_b = 0.5
                        pw += pair_sigma_weight(
                            sigma_a, sigma_b,
                            threshold=logsnr_threshold,
                            interval=logsnr_interval,
                            decay_rate=logsnr_decay_rate,
                        )
                    pw /= max(len(pairs_this_micro), 1)

                else:
                    # -------------------------------------------------------
                    # SERIAL PATH: Original 1-pair-at-a-time scoring.
                    # -------------------------------------------------------
                    # Pick a pair: from sampler (on-the-fly) or materialized pair list
                    if _using_sampler:
                        pair = pair_sampler.sample_pair()
                        # Compute preferences via the caller-supplied preference_fn.
                        # The sampler returns only pair metadata (traj/step/sigma);
                        # preference_fn evaluates the reward function and returns
                        # per-head preferences as {pref_key: +1/-1/0}.
                        prefs = preference_fn(pair)
                        pair.update(prefs)
                        # Sampler returns (traj_id, step_key) pairs; load via tuple keys
                        key_a = (pair["traj_a"], pair["step_a"])
                        key_b = (pair["traj_b"], pair["step_b"])
                    else:
                        if pair_weights is not None:
                            pair = random.choices(training_pairs, weights=pair_weights, k=1)[0]
                        else:
                            pair = random.choice(training_pairs)
                        key_a = pair["idx_a"]
                        key_b = pair["idx_b"]

                    # Load latents for both images
                    lat_a, ts_a, cond_a, nt_a, rc_a = load_latent_fn(key_a)
                    lat_b, ts_b, cond_b, nt_b, rc_b = load_latent_fn(key_b)

                    # Full differentiable forward for both images
                    scores_a = btrm_model.score_differentiable(
                        lat_a, ts_a, cond_a, nt_a, rc_a,
                        gradient_checkpointing=gradient_checkpointing,
                    )  # (1, N_heads) with grad_fn
                    scores_b = btrm_model.score_differentiable(
                        lat_b, ts_b, cond_b, nt_b, rc_b,
                        gradient_checkpointing=gradient_checkpointing,
                    )  # (1, N_heads) with grad_fn

                    # Compute per-head BT loss (total loss = BT loss only, no regularizer)
                    # The ScoreUnembedder's soft_tanh_cap(10.0) already bounds scores.
                    # See docs/directive_remove_logsquare_regularizer.md.
                    total_bt = scores_a.new_zeros(())
                    active_heads = 0

                    for head_idx, (name, key) in enumerate(zip(head_names, pref_keys)):
                        pref = pair[key]
                        if pref == 0:
                            continue

                        if pref > 0:
                            pos_s = scores_a[0, head_idx]
                            neg_s = scores_b[0, head_idx]
                        else:
                            pos_s = scores_b[0, head_idx]
                            neg_s = scores_a[0, head_idx]

                        bt = -F.logsigmoid(pos_s - neg_s)
                        total_bt = total_bt + bt
                        active_heads += 1

                        with torch.no_grad():
                            correct = 1.0 if (pos_s > neg_s).item() else 0.0
                            accum_accs[name] += correct
                            accum_acc_counts[name] += 1

                            # Track in ValidationMetrics
                            sigma_a_val = pair.get("sigma_a", 0.5) if _using_sampler else 0.5
                            sigma_b_val = pair.get("sigma_b", 0.5) if _using_sampler else 0.5
                            val_tracker.update(PairResult(
                                head_name=name,
                                correct=correct,
                                loss_contribution=bt.item(),
                                score_preferred=pos_s.item(),
                                score_rejected=neg_s.item(),
                                sigma_a=sigma_a_val,
                                sigma_b=sigma_b_val,
                                source_a="serial_training",
                                source_b="serial_training",
                                traj_a=pair.get("traj_a", -1) if _using_sampler else -1,
                                step_a=pair.get("step_a", "") if _using_sampler else "",
                                traj_b=pair.get("traj_b", -1) if _using_sampler else -1,
                                step_b=pair.get("step_b", "") if _using_sampler else "",
                            ))

                    if active_heads > 0:
                        total_bt = total_bt / active_heads

                    loss = total_bt

                    # Log the pair's sigma weight for observability (which sigma regime
                    # was sampled). This is NOT applied to the loss -- the weighting
                    # happens via the sampling distribution, not loss scaling.
                    pw = 1.0
                    if _using_sampler:
                        # Sampler provides sigma directly in the pair dict
                        sigma_a = pair.get("sigma_a", 0.5)
                        sigma_b = pair.get("sigma_b", 0.5)
                        pw = pair_sigma_weight(
                            sigma_a, sigma_b,
                            threshold=logsnr_threshold,
                            interval=logsnr_interval,
                            decay_rate=logsnr_decay_rate,
                        )
                    elif use_sigma_weighting and sigma_lookup_fn is not None:
                        sigma_a = sigma_lookup_fn(key_a)
                        sigma_b = sigma_lookup_fn(key_b)
                        pw = pair_sigma_weight(
                            sigma_a, sigma_b,
                            threshold=logsnr_threshold,
                            interval=logsnr_interval,
                            decay_rate=logsnr_decay_rate,
                        )

                # Scale loss for gradient accumulation.
                # Guard: skip backward when loss has no grad_fn. Two causes:
                # 1. Both heads returned pref==0 (tied pair) -> loss is
                #    scores_a.new_zeros(()) which does NOT inherit requires_grad.
                # 2. F.logsigmoid(very_large_diff) rounds to exactly 0.0 in
                #    float32, losing the computation graph.
                # In either case, there's nothing to backpropagate.
                scaled_loss = loss / grad_accum_steps
                if scaled_loss.requires_grad or scaled_loss.grad_fn is not None:
                    scaled_loss.backward()

                accum_loss += loss.item()
                accum_bt += total_bt.item()
                accum_pw += pw

        # Clip gradients -- pre_clip_norm is the total norm BEFORE clipping
        all_params = btrm_model.all_trainable_params()
        pre_clip_norm = torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
        pre_clip_val = pre_clip_norm.item() if isinstance(pre_clip_norm, Tensor) else pre_clip_norm
        # Compute post-clip norm for logging
        post_clip_norm = torch.nn.utils.clip_grad_norm_(all_params, float('inf'))
        post_clip_val = post_clip_norm.item() if isinstance(post_clip_norm, Tensor) else post_clip_norm

        optimizer.step()
        scheduler.step()

        step_time = time.perf_counter() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        # Report accumulated (mean) metrics
        accs = {}
        for name in head_names:
            if accum_acc_counts[name] > 0:
                accs[name] = accum_accs[name] / accum_acc_counts[name]
            else:
                accs[name] = 0.0

        # For FLOPS-budget mode, there is 1 effective accumulation step
        # (all bins are computed in a single macrobatch). For legacy mode,
        # the divisor is grad_accum_steps.
        _n_accum = 1 if _use_flops_budget else grad_accum_steps

        entry = {
            "step": step,
            "loss": accum_loss / _n_accum,
            "bt_loss": accum_bt / _n_accum,
            "pre_clip_grad_norm": pre_clip_val,
            "grad_norm": post_clip_val,
            "lr": current_lr,
            "time_s": step_time,
            "pair_weight": accum_pw / _n_accum,
        }
        for name in head_names:
            entry[f"accuracy_{name}"] = accs.get(name, 0.0)

        # Attach per-step funfetti metadata if packed training was used
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
            # Add FLOPS-budget-specific metadata
            if _use_flops_budget and step_microbatch_meta:
                m0 = step_microbatch_meta[0]
                funfetti_meta["macrobatch_budget"] = m0.get("macrobatch_budget", macrobatch_budget)
                funfetti_meta["macrobatch_consumed"] = m0.get("macrobatch_consumed", 0.0)
                funfetti_meta["n_bins"] = m0.get("n_bins", 0)
                funfetti_meta["n_cross_resolution_pairs"] = m0.get("n_cross_resolution_pairs", 0)
                funfetti_meta["per_pair_resolutions"] = m0.get("per_pair_resolutions", [])
            entry["funfetti"] = funfetti_meta

        # Include validation tracker summary at log intervals
        if step % log_interval == 0 or step == n_steps - 1:
            entry["validation"] = val_tracker.summary()

        training_curve.append(entry)

        # Incremental persistence: write step to JSONL immediately
        if curve_writer is not None:
            curve_writer.write_step(entry)

        # Incremental persistence: auto-save ValidationMetrics periodically
        if _val_metrics_saver is not None:
            _val_metrics_saver.maybe_save(step)

        if callback is not None:
            callback(step, entry)

        # TrainingArtifacts integration: log step + auto-checkpoint
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
                artifacts.save_checkpoint(step, btrm_model)

        # Intermediate checkpoints (legacy path, still honored when artifacts is None)
        if checkpoint_fn is not None and checkpoint_steps is not None:
            if step in checkpoint_steps:
                print(f"  [CHECKPOINT] Saving checkpoint at step {step}")
                checkpoint_fn(step, btrm_model)
                # Persist validation metrics alongside checkpoint
                if output_dir is not None:
                    import os
                    val_tracker.save_json(os.path.join(output_dir, "validation_metrics.json"))

        # Incremental persistence: force val_metrics save + summary at checkpoints
        _is_checkpoint = (checkpoint_steps is not None and step in checkpoint_steps)
        if _is_checkpoint:
            if _val_metrics_saver is not None:
                _val_metrics_saver.flush(step)
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

    # Persist validation metrics at end of training
    if output_dir is not None:
        import os
        os.makedirs(output_dir, exist_ok=True)
        val_tracker.save_json(os.path.join(output_dir, "validation_metrics.json"))
        print(f"  ValidationMetrics saved to {output_dir}/validation_metrics.json "
              f"({val_tracker._n_updates} pair results tracked)")

    # Final flush of incremental persistence
    if _val_metrics_saver is not None and n_steps > 0:
        _val_metrics_saver.flush(n_steps - 1)

    if summary_path is not None and training_curve:
        _final_summary = _build_incremental_summary(
            training_curve, head_names, n_steps, t_total, n_steps - 1,
        )
        _final_summary["status"] = "completed"
        atomic_json_save(_final_summary, summary_path)

    btrm_model.eval_mode()
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
    """Build a summary dict from the training curve accumulated so far.

    This is called at checkpoint steps and at end-of-run to produce a
    partial or complete summary. The summary is designed to be useful
    even when training is only partially complete.

    Args:
        training_curve: List of per-step entry dicts accumulated so far.
        head_names: Names of the scoring heads.
        n_steps_total: Total planned training steps.
        t_total_start: perf_counter timestamp at training start.
        current_step: The step number at which this summary is being built.

    Returns:
        Summary dict suitable for JSON serialization.
    """
    import time as _time

    n = len(training_curve)
    if n == 0:
        return {"status": "in_progress", "steps_completed": 0}

    losses = [e.get("loss", e.get("bt_loss", 0.0)) for e in training_curve]
    grad_norms = [e.get("pre_clip_grad_norm", 0.0) for e in training_curve]
    step_times = [e.get("time_s", 0.0) for e in training_curve]

    elapsed = _time.perf_counter() - t_total_start

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

    # Per-head accuracy
    for name in head_names:
        accs = [e.get(f"accuracy_{name}", 0.0) for e in training_curve]
        if accs:
            summary[f"overall_accuracy_{name}"] = sum(accs) / len(accs)
            last_20 = accs[-min(20, len(accs)):]
            summary[f"last_20_accuracy_{name}"] = sum(last_20) / len(last_20)

    # ETA estimate
    if step_times and current_step < n_steps_total - 1:
        # Use steady-state step time (exclude step 0 which includes compilation)
        steady_times = step_times[1:] if len(step_times) > 1 else step_times
        avg_step = sum(steady_times) / len(steady_times)
        remaining = n_steps_total - current_step - 1
        summary["estimated_remaining_s"] = avg_step * remaining

    return summary

"""Production training script: BTRM head training + policy optimization.

Phased execution:
  Phase 0: (stub) Trajectory generation — skipped, use generate_btrm_dataset.py
  Phase 1: BTRM training from stored trajectories via TrajectoryPool
  Phase 2: Policy optimization with live rollouts and diverse prompts
  Phase 3: Final adapter dump + optional eval renders

All GPU computation happens on the inference server via ZMQ RPCs.
This script is a pure scheduling client — no torch.cuda, no model loading.

Prerequisites:
  Running inference server:
    .venv/Scripts/python.exe -m futudiffu.server --port 5555 \
        --fp8-diff <path> --te <path> --vae <path>

Usage:
    .venv/Scripts/python.exe train.py --dataset-dir F:\\path\\to\\btrm_dataset \
        --output-dir F:\\path\\to\\run_output
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn.functional as F

from futudiffu.btrm_dataset import PROMPT_TEMPLATES
from futudiffu.client import InferenceClient
from futudiffu.multi_gpu_client import MultiGPUClient
from futudiffu.image_stats import naturalness_report
from futudiffu.policy_loss import compute_group_advantages
from futudiffu.rendering import decode_and_save
from futudiffu.sampling import build_sigmas, simple_scheduler
from futudiffu.trajectory_loader import TrajectoryPool

# Type alias: both client types share the same RPC interface
Client = InferenceClient | MultiGPUClient


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class MetricsLogger:
    """Structured JSONL logger + human-readable stderr summaries."""

    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self._logfile_path = os.path.join(output_dir, "metrics.jsonl")
        self._logfile = open(self._logfile_path, "a", buffering=1)
        self._t_start = time.monotonic()

    def log(self, record: dict) -> None:
        """Write one JSONL line with wall_clock_s injected."""
        record["wall_clock_s"] = round(time.monotonic() - self._t_start, 2)
        record["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._logfile.write(json.dumps(record, default=_json_default) + "\n")

    def close(self) -> None:
        self._logfile.close()

    @property
    def path(self) -> str:
        return self._logfile_path


def _json_default(obj):
    """JSON serializer for objects not serializable by default json code."""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return str(obj)
    return str(obj)


def _stderr(msg: str) -> None:
    """Print to stderr (human-readable output)."""
    print(msg, file=sys.stderr, flush=True)


def _section(title: str) -> None:
    _stderr(f"\n{'=' * 70}")
    _stderr(f"  {title}")
    _stderr(f"{'=' * 70}\n")


def _get_vram_gb() -> float:
    """Get current VRAM usage in GB, or 0 if not available."""
    try:
        return round(torch.cuda.memory_allocated() / (1024 ** 3), 2)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Prompt encoding + conditioning cache
# ---------------------------------------------------------------------------

def encode_all_prompts(
    client: Client,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Encode all 24 prompt templates + negative prompt.

    Returns:
        (pos_conds, neg_cond): list of 24 conditioning tensors, one neg_cond.
        All tensors are on CPU, shape (1, seq_len_i, 2560).
    """
    _stderr(f"  Encoding {len(PROMPT_TEMPLATES)} prompts + negative...")
    t0 = time.monotonic()

    neg_cond = client.encode_prompt("")
    pos_conds = []
    for i, prompt in enumerate(PROMPT_TEMPLATES):
        cond = client.encode_prompt(prompt)
        pos_conds.append(cond)
        if (i + 1) % 8 == 0:
            _stderr(f"    Encoded {i + 1}/{len(PROMPT_TEMPLATES)}")

    dt = time.monotonic() - t0
    _stderr(f"  All prompts encoded in {dt:.1f}s")
    _stderr(f"  neg_cond shape: {neg_cond.shape}")
    _stderr(f"  pos_cond shapes: {pos_conds[0].shape} .. {pos_conds[-1].shape}")

    return pos_conds, neg_cond


def pad_conds(
    pos_cond: torch.Tensor,
    neg_cond: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad pos/neg conditioning to same seq length for CFG batching."""
    pos_len = pos_cond.shape[1]
    neg_len = neg_cond.shape[1]
    max_len = max(pos_len, neg_len)
    if pos_len < max_len:
        pos_cond = F.pad(pos_cond, (0, 0, 0, max_len - pos_len))
    if neg_len < max_len:
        neg_cond = F.pad(neg_cond, (0, 0, 0, max_len - neg_len))
    return pos_cond, neg_cond


# ---------------------------------------------------------------------------
# Rollout render health check
# ---------------------------------------------------------------------------

def _rollout_health_check(
    client: Client,
    latent: torch.Tensor,
    label: str,
    output_dir: str,
    logger: MetricsLogger,
) -> dict:
    """VAE-decode a rollout latent, compute image quality stats, save render.

    Catches pathological outputs (NaN, all-black, all-white, flat) that could
    go unnoticed when only numerical BTRM scores are inspected.

    Capture rate guidance:
      - Phase 1 (BTRM): 1 r_theta trajectory per checkpoint_every macrobatches
      - Phase 2 (policy): 1 rollout final per render_every iterations
      - VAE decode is ~0.5s vs ~30s trajectory generation — negligible overhead

    Args:
        client: InferenceClient (for VAE decode).
        latent: (1, 16, H, W) final latent tensor.
        label: Descriptive label for this render (used in filename + logs).
        output_dir: Root output directory.
        logger: MetricsLogger for structured logging.

    Returns:
        Dict with latent stats, image stats, and pathology status.
    """
    render_dir = os.path.join(output_dir, "rollout_renders")
    os.makedirs(render_dir, exist_ok=True)

    # Latent-domain checks (cheap, catches NaN/Inf before VAE decode)
    latent_f = latent.float()
    latent_stats = {
        "latent_mean": round(float(latent_f.mean()), 4),
        "latent_std": round(float(latent_f.std()), 4),
        "latent_min": round(float(latent_f.min()), 4),
        "latent_max": round(float(latent_f.max()), 4),
        "latent_has_nan": bool(torch.isnan(latent_f).any()),
        "latent_has_inf": bool(torch.isinf(latent_f).any()),
    }

    if latent_stats["latent_has_nan"] or latent_stats["latent_has_inf"]:
        _stderr(f"  PATHOLOGICAL: {label} latent has NaN/Inf!")
        logger.log({"phase": "render_check", "label": label,
                     "status": "PATHOLOGICAL_NANF", **latent_stats})
        return {**latent_stats, "status": "PATHOLOGICAL_NANF"}

    # VAE decode -> render -> image quality stats
    render_path = os.path.join(render_dir, f"{label}.png")
    try:
        arr = decode_and_save(client, latent, render_path)
        img_stats = naturalness_report(arr)

        # Pathology thresholds
        is_black = all(m < 5.0 for m in img_stats["channel_means"])
        is_white = all(m > 250.0 for m in img_stats["channel_means"])
        is_flat = all(s < 1.0 for s in img_stats["channel_stds"])

        status = "OK"
        if is_black:
            status = "PATHOLOGICAL_BLACK"
        elif is_white:
            status = "PATHOLOGICAL_WHITE"
        elif is_flat:
            status = "PATHOLOGICAL_FLAT"

        all_stats = {**latent_stats, **img_stats, "status": status}
        logger.log({"phase": "render_check", "label": label, **all_stats})

        if status != "OK":
            _stderr(f"  PATHOLOGICAL: {label} render is {status}")
        else:
            _stderr(f"    render {label}: OK "
                    f"(entropy={img_stats['mean_entropy']:.2f}, "
                    f"slope={img_stats['spectral_slope']:.2f})")

        return all_stats

    except Exception as e:
        _stderr(f"  WARNING: render check failed for {label}: {e}")
        logger.log({"phase": "render_check", "label": label,
                     "status": "RENDER_FAILED", "error": str(e),
                     **latent_stats})
        return {**latent_stats, "status": "RENDER_FAILED"}


# ---------------------------------------------------------------------------
# Phase 1: BTRM Training
# ---------------------------------------------------------------------------

def phase_btrm(
    client: Client,
    pool: TrajectoryPool,
    logger: MetricsLogger,
    n_macrobatches: int,
    macrobatch_size: int,
    lr: float,
    logsq_weight: float,
    checkpoint_every: int,
    output_dir: str,
    lora_rank: int,
    lora_alpha: float,
    prompt_cache: dict[str, torch.Tensor],
    neg_cond: torch.Tensor | None = None,
    render_every: int = 0,
) -> None:
    """Phase 1: Train BTRM head on stored trajectory checkpoints.

    Rollout capture rate (when render_every > 0):
      - 1 r_theta trajectory generated per render_every macrobatches
      - Final latent VAE-decoded and saved to rollout_renders/
      - Image stats logged (entropy, spectral slope, pathology flags)
      - Uses fixed seed=42 + first PROMPT_TEMPLATE for visual consistency
    """
    _section("Phase 1: BTRM Training")

    # -- Inject rtheta LoRA on last 2 layers --
    n_rtheta = client.inject_lora(
        "rtheta", rank=lora_rank, alpha=lora_alpha,
        layer_indices=[28, 29],
    )
    _stderr(f"  Injected rtheta: {n_rtheta} adapters on layers 28-29")

    # Warmup compiled model with new LoRA
    _stderr("  Warming up compiled model...")
    t0 = time.monotonic()
    client.warmup(attention_backend="sdpa")
    _stderr(f"  Warmup done in {time.monotonic() - t0:.1f}s")

    # -- Inject BTRM head --
    btrm_meta = client.inject_btrm_head(
        head_names=["scrimble", "scrongle"],
        logit_cap=10.0,
        lr=lr,
    )
    _stderr(f"  BTRM head injected: {btrm_meta.get('n_params', '?')} params, lr={lr}")

    # -- Get data splits --
    sdpa_idx, sage_idx = pool.scrimble_split()
    full_idx, reduced_idx = pool.scrongle_split()
    _stderr(f"  Scrimble split: {len(sdpa_idx)} SDPA, {len(sage_idx)} Sage")
    _stderr(f"  Scrongle split: {len(full_idx)} full-step, {len(reduced_idx)} reduced-step")

    # Build deduped index sets
    all_indices = sorted(set(sdpa_idx + sage_idx + full_idx + reduced_idx))
    sdpa_set = set(sdpa_idx)
    sage_set = set(sage_idx)
    full_set = set(full_idx)
    reduced_set = set(reduced_idx)
    _stderr(f"  Total unique examples: {len(all_indices)}")

    # Prompt cache provided by caller (pre-encoded before any model swaps)
    unique_prompts = {pool.examples[i].prompt for i in all_indices}
    _stderr(f"  Using pre-encoded cache: {len(prompt_cache)} prompts "
            f"({len(unique_prompts)} needed)")

    # -- Training loop --
    rng = random.Random(42)
    macrobatch_count = 0

    for mb in range(n_macrobatches):
        t0 = time.monotonic()

        # Shuffle and sample a macrobatch worth of examples
        rng.shuffle(all_indices)
        batch_examples: list[dict] = []

        for ex_idx in all_indices:
            if len(batch_examples) >= macrobatch_size:
                break

            ex = pool.examples[ex_idx]
            if ex.step_idx == -1:  # skip final checkpoints (sigma=0)
                continue

            latent = pool.load_checkpoint(ex)
            cond = prompt_cache[ex.prompt]

            # Head 0 (scrimble): SDPA=positive, Sage=negative
            if ex_idx in sdpa_set:
                batch_examples.append({
                    "latent": latent, "sigma": ex.sigma,
                    "conditioning": cond, "head_idx": 0, "is_positive": True,
                })
            elif ex_idx in sage_set:
                batch_examples.append({
                    "latent": latent, "sigma": ex.sigma,
                    "conditioning": cond, "head_idx": 0, "is_positive": False,
                })

            # Head 1 (scrongle): full-step=positive, reduced=negative
            if ex_idx in full_set:
                batch_examples.append({
                    "latent": latent, "sigma": ex.sigma,
                    "conditioning": cond, "head_idx": 1, "is_positive": True,
                })
            elif ex_idx in reduced_set:
                batch_examples.append({
                    "latent": latent, "sigma": ex.sigma,
                    "conditioning": cond, "head_idx": 1, "is_positive": False,
                })

        if len(batch_examples) < 4:
            _stderr(f"  WARNING: macrobatch {mb} has only {len(batch_examples)} examples, skipping")
            continue

        # Truncate to macrobatch_size
        batch_examples = batch_examples[:macrobatch_size]

        # Train on server
        metrics = client.train_btrm_step(
            batch_examples,
            logsquare_weight=logsq_weight,
            attention_backend="sdpa",
        )
        dt = time.monotonic() - t0

        macrobatch_count += 1

        # Log
        acc = metrics.get("per_head_accuracy", {})
        acc_str = ", ".join(f"{k}={v:.2%}" for k, v in acc.items())

        record = {
            "phase": "btrm",
            "step": macrobatch_count,
            "n_examples": len(batch_examples),
            "loss": metrics["loss"],
            "bt_loss": metrics.get("bt_loss", 0.0),
            "logsq_loss": metrics.get("logsq_loss", 0.0),
            "per_head_accuracy": acc,
            "dt_s": round(dt, 2),
            "vram_gb": _get_vram_gb(),
        }
        logger.log(record)

        _stderr(
            f"  btrm {macrobatch_count:3d}/{n_macrobatches} | "
            f"n={len(batch_examples)} | loss={metrics['loss']:.4f} | "
            f"bt={metrics.get('bt_loss', 0):.4f} | "
            f"acc=[{acc_str}] | {dt:.1f}s"
        )

        # Checkpoint
        if checkpoint_every > 0 and macrobatch_count % checkpoint_every == 0:
            _checkpoint(client, output_dir, f"btrm_step_{macrobatch_count:04d}")

        # R_theta sanity render: generate one trajectory with r_theta active,
        # VAE-decode the final, check for pathological outputs.
        if (render_every > 0 and macrobatch_count % render_every == 0
                and neg_cond is not None):
            render_cond = prompt_cache.get(PROMPT_TEMPLATES[0])
            if render_cond is not None:
                _stderr(f"  R_theta render check (mb {macrobatch_count})...")
                rc_pos, rc_neg = pad_conds(render_cond, neg_cond)
                rc_traj = client.sample_trajectory(
                    rc_pos, rc_neg, seed=42, n_steps=30, cfg=4.0,
                )
                _rollout_health_check(
                    client, rc_traj["final"],
                    f"rtheta_mb{macrobatch_count:04d}",
                    output_dir, logger,
                )
                del rc_traj

    _stderr(f"\n  Phase 1 complete: {macrobatch_count} macrobatches trained.")


# ---------------------------------------------------------------------------
# Phase 2: Policy Optimization
# ---------------------------------------------------------------------------

def phase_policy(
    client: Client,
    pos_conds: list[torch.Tensor],
    neg_cond: torch.Tensor,
    logger: MetricsLogger,
    n_iterations: int,
    group_size: int,
    rollout_steps: int,
    n_sparse: int,
    lr: float,
    cfg: float,
    checkpoint_every: int,
    output_dir: str,
    lora_rank: int,
    lora_alpha: float,
    render_every: int = 0,
) -> None:
    """Phase 2: Policy optimization with live rollouts and diverse prompts.

    Rollout capture rate (when render_every > 0):
      - 1/K rollout finals VAE-decoded per render_every iterations
      - group_size rollouts generated per iteration (default K=4)
      - sparse_steps checkpoints kept transiently for gradient computation
      - All rollout checkpoint latents are consumed then discarded (not persisted)
      - VAE decode of 1 final is ~0.5s vs ~30s per trajectory — negligible
    """
    _section("Phase 2: Policy Optimization")

    # -- Freeze rtheta and silence it --
    client.set_adapter_config("rtheta", frozen=True, scale=0.0)
    _stderr("  rtheta frozen, scale=0")

    # -- Inject ptheta on all layers --
    n_ptheta = client.inject_lora(
        "ptheta", rank=lora_rank, alpha=lora_alpha,
        init_b_std=0.01,
    )
    _stderr(f"  Injected ptheta: {n_ptheta} adapters (all layers)")

    # Warmup after injection
    _stderr("  Warming up compiled model...")
    t0 = time.monotonic()
    client.warmup(attention_backend="sdpa")
    _stderr(f"  Warmup done in {time.monotonic() - t0:.1f}s")

    # -- Build sigma schedule --
    sigma_table = build_sigmas(shift=1.0, multiplier=1000.0)
    sigmas = simple_scheduler(sigma_table, rollout_steps)
    _stderr(f"  Sigmas ({len(sigmas)}): [{sigmas[0]:.4f} .. {sigmas[-1]:.4f}]")

    # -- Sparse step indices (evenly spaced) --
    n_sparse_actual = min(n_sparse, rollout_steps)
    sparse_indices = [
        int(i * (rollout_steps - 1) / max(1, n_sparse_actual - 1))
        for i in range(n_sparse_actual)
    ]
    _stderr(f"  Sparse steps: {sparse_indices}")
    _stderr(f"  Rollout steps: {rollout_steps}, group size: {group_size}")
    _stderr(f"  Prompts available: {len(pos_conds)}")

    # -- Prompt RNG (deterministic for reproducibility) --
    prompt_rng = random.Random(12345)

    # -- Policy training loop --
    base_seed = 70000

    for iteration in range(n_iterations):
        t0 = time.monotonic()

        # Sample a prompt for this iteration's group
        prompt_idx = prompt_rng.randint(0, len(pos_conds) - 1)
        pos_cond_raw = pos_conds[prompt_idx]
        pos_cond, neg_cond_padded = pad_conds(pos_cond_raw, neg_cond)

        # -- Generate K rollouts --
        rollout_rewards = []
        rollout_checkpoints: list[dict[int, torch.Tensor]] = []
        render_latent = None  # Keep first rollout final for render check

        mid_idx = sparse_indices[len(sparse_indices) // 2]

        if hasattr(client, 'generate_trajectories'):
            # Multi-GPU: parallel rollout generation across servers
            jobs = []
            for k in range(group_size):
                seed = base_seed + iteration * 1000 + k
                jobs.append({
                    "pos_cond": pos_cond, "neg_cond": neg_cond_padded,
                    "seed": seed, "n_steps": rollout_steps,
                    "cfg": cfg, "save_steps": sparse_indices,
                    "score_at_step": mid_idx,
                    "attention_backend": "sage",
                })
            trajectories = client.generate_trajectories(jobs)

            for k, trajectory in enumerate(trajectories):
                # Extract reward (inline or fallback)
                btrm_scores = trajectory.pop("_btrm_scores", None)
                if btrm_scores is not None:
                    reward = btrm_scores[0][0]
                else:
                    step_key = f"step_{mid_idx:02d}"
                    if step_key in trajectory:
                        x_for_score = trajectory[step_key]
                        sigma_for_score = sigmas[mid_idx]
                    else:
                        x_for_score = trajectory["final"]
                        sigma_for_score = sigmas[-2]
                    sigma_tensor = torch.tensor([float(sigma_for_score)])
                    scores = client.score_btrm(
                        x_for_score, sigma_tensor, pos_cond[:1],
                        attention_backend="sdpa",
                    )
                    reward = scores[0][0]

                rollout_rewards.append(reward)

                # Collect checkpoints for gradient computation
                ckpts: dict[int, torch.Tensor] = {}
                for si in sparse_indices:
                    sk = f"step_{si:02d}"
                    if sk in trajectory:
                        ckpts[si] = trajectory[sk]
                rollout_checkpoints.append(ckpts)

                if k == 0 and render_every > 0:
                    render_latent = trajectory.get("final")

                del trajectory
        else:
            # Single-GPU: sequential rollout generation
            for k in range(group_size):
                seed = base_seed + iteration * 1000 + k

                trajectory = client.sample_trajectory(
                    pos_cond, neg_cond_padded, seed=seed,
                    n_steps=rollout_steps,
                    cfg=cfg,
                    save_steps=sparse_indices,
                    score_at_step=mid_idx,
                )

                # Read BTRM scores from inline scoring (avoids separate RPC)
                btrm_scores = trajectory.pop("_btrm_scores", None)
                if btrm_scores is not None:
                    reward = btrm_scores[0][0]
                else:
                    # Fallback: standalone BTRM scoring RPC
                    step_key = f"step_{mid_idx:02d}"
                    if step_key in trajectory:
                        x_for_score = trajectory[step_key]
                        sigma_for_score = sigmas[mid_idx]
                    else:
                        x_for_score = trajectory["final"]
                        sigma_for_score = sigmas[-2]
                    sigma_tensor = torch.tensor([float(sigma_for_score)])
                    scores = client.score_btrm(
                        x_for_score, sigma_tensor, pos_cond[:1],
                        attention_backend="sdpa",
                    )
                    reward = scores[0][0]

                # Use head 0 (scrimble / bit_quality) as reward signal
                rollout_rewards.append(reward)

                # Collect checkpoints for gradient computation
                ckpts: dict[int, torch.Tensor] = {}
                for si in sparse_indices:
                    sk = f"step_{si:02d}"
                    if sk in trajectory:
                        ckpts[si] = trajectory[sk]
                rollout_checkpoints.append(ckpts)

                # Keep first rollout final for periodic render health check
                if k == 0 and render_every > 0:
                    render_latent = trajectory.get("final")

                del trajectory

        t_rollout = time.monotonic() - t0

        # -- Compute advantages --
        rewards = torch.tensor(rollout_rewards)
        advantages = compute_group_advantages(rewards)

        # -- Accumulate gradients on server --
        for k in range(group_size):
            if not rollout_checkpoints[k]:
                continue
            client.accumulate_policy_gradients(
                checkpoints=rollout_checkpoints[k],
                sigmas=sigmas,
                conditioning=pos_cond[:1],
                adapter_name="ptheta",
                advantage=advantages[k].item(),
            )

        # -- Optimizer step on server --
        step_meta = client.policy_optimizer_step(
            adapter_name="ptheta",
            max_grad_norm=1.0,
            lr=lr,
        )

        dt = time.monotonic() - t0
        mean_reward = rewards.mean().item()
        grad_norm = step_meta["grad_norm"]

        # Log
        record = {
            "phase": "policy",
            "iter": iteration,
            "prompt_idx": prompt_idx,
            "rewards": rollout_rewards,
            "mean_reward": mean_reward,
            "advantages": advantages.tolist(),
            "grad_norm": grad_norm,
            "n_params": step_meta["n_params"],
            "rollout_time_s": round(t_rollout, 2),
            "dt_s": round(dt, 2),
            "vram_gb": _get_vram_gb(),
        }
        logger.log(record)

        prompt_snippet = PROMPT_TEMPLATES[prompt_idx][:50].replace("\n", " ")
        _stderr(
            f"  iter {iteration:3d}/{n_iterations} | "
            f"p={prompt_idx:2d} \"{prompt_snippet}...\" | "
            f"rewards=[{', '.join(f'{r:+.4f}' for r in rollout_rewards)}] | "
            f"mean={mean_reward:+.4f} | "
            f"grad_norm={grad_norm:.3e} | "
            f"rollout={t_rollout:.1f}s | total={dt:.1f}s"
        )

        # Render health check on first rollout's final latent
        if (render_every > 0 and render_latent is not None
                and (iteration + 1) % render_every == 0):
            _stderr(f"  Policy render check (iter {iteration + 1})...")
            _rollout_health_check(
                client, render_latent,
                f"ptheta_iter{iteration + 1:04d}",
                output_dir, logger,
            )

        # Checkpoint
        if checkpoint_every > 0 and (iteration + 1) % checkpoint_every == 0:
            _checkpoint(client, output_dir, f"policy_iter_{iteration + 1:04d}")

    _stderr(f"\n  Phase 2 complete: {n_iterations} policy iterations.")


# ---------------------------------------------------------------------------
# Phase 3: Final dump + eval
# ---------------------------------------------------------------------------

def phase_final(
    client: Client,
    output_dir: str,
    logger: MetricsLogger,
) -> None:
    """Phase 3: Final adapter dump."""
    _section("Phase 3: Final Adapter Dump")

    _checkpoint(client, output_dir, "final")

    logger.log({"phase": "final", "event": "adapters_dumped"})
    _stderr("  Final adapters saved.")


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------

def _checkpoint(client: Client, output_dir: str, label: str) -> None:
    """Dump all LoRA adapters + BTRM head to output_dir/label/."""
    ckpt_dir = os.path.join(output_dir, label)
    _stderr(f"  Saving checkpoint: {ckpt_dir}")
    try:
        result = client.dump_all_loras(output_dir=ckpt_dir)
        files = result.get("files", [])
        _stderr(f"    Saved {len(files)} files")
    except Exception as e:
        _stderr(f"    WARNING: checkpoint failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Production training: BTRM + policy optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Connection
    parser.add_argument("--port", type=int, default=5555,
                        help="Server ZMQ port")
    parser.add_argument("--ports", type=int, nargs="+", default=None,
                        help="Multiple server ports for multi-GPU. Overrides --port. "
                             "First port is primary (handles training RPCs).")

    # Dataset
    parser.add_argument("--dataset-dir", type=str, required=True,
                        help="Path to btrm_dataset/ directory")

    # Output
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path for checkpoints + metrics JSONL")

    # BTRM config
    parser.add_argument("--btrm-macrobatches", type=int, default=30,
                        help="Number of BTRM training macrobatches")
    parser.add_argument("--btrm-batch-size", type=int, default=32,
                        help="Examples per BTRM macrobatch")
    parser.add_argument("--btrm-lr", type=float, default=1e-3,
                        help="BTRM head + rtheta learning rate")
    parser.add_argument("--btrm-logsq-weight", type=float, default=0.1,
                        help="Logsquare regularization weight for BTRM")

    # Policy config
    parser.add_argument("--policy-iterations", type=int, default=50,
                        help="Number of policy optimization iterations")
    parser.add_argument("--policy-group-size", type=int, default=4,
                        help="Rollouts per policy iteration (K for advantages)")
    parser.add_argument("--policy-rollout-steps", type=int, default=10,
                        help="Euler steps per rollout")
    parser.add_argument("--policy-sparse-steps", type=int, default=5,
                        help="Sparse gradient steps per rollout")
    parser.add_argument("--policy-lr", type=float, default=1e-4,
                        help="Policy optimizer learning rate")

    # LoRA config
    parser.add_argument("--lora-rank", type=int, default=8,
                        help="LoRA rank for both rtheta and ptheta")
    parser.add_argument("--lora-alpha", type=float, default=16.0,
                        help="LoRA alpha for both rtheta and ptheta")

    # Generation config
    parser.add_argument("--cfg", type=float, default=4.0,
                        help="CFG scale for policy rollouts")

    # Checkpoint cadence
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="Save adapters every N steps/iterations (0 to disable)")

    # Render monitoring
    parser.add_argument("--render-every", type=int, default=8,
                        help="VAE-decode a rollout final every N steps/iters for "
                             "pathology checks (0 to disable)")

    # Phase control
    parser.add_argument("--skip-btrm", action="store_true",
                        help="Skip phase 1 (BTRM training)")
    parser.add_argument("--skip-policy", action="store_true",
                        help="Skip phase 2 (policy optimization)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume mode: skip BTRM, assume adapters exist on server")

    args = parser.parse_args()

    # -- Setup --
    os.makedirs(args.output_dir, exist_ok=True)
    logger = MetricsLogger(args.output_dir)

    _section("futudiffu production training")
    _stderr(f"  Output dir: {args.output_dir}")
    _stderr(f"  Metrics log: {logger.path}")
    _stderr(f"  Config: {json.dumps(vars(args), indent=2)}")

    # Log full config
    logger.log({"phase": "config", "config": vars(args)})

    # -- Connect to server --
    if args.ports and len(args.ports) > 1:
        endpoints = [f"tcp://localhost:{p}" for p in args.ports]
        _stderr(f"\n  Connecting to {len(args.ports)} servers (multi-GPU)...")
        client = MultiGPUClient(endpoints)
        _stderr(f"  Multi-GPU mode: {len(args.ports)} servers, "
                f"primary=:{args.ports[0]}")
    else:
        port = args.ports[0] if args.ports else args.port
        endpoint = f"tcp://localhost:{port}"
        _stderr(f"\n  Connecting to server at {endpoint}...")
        client = InferenceClient(endpoint)

    status = client.status()
    _stderr(f"  Server phase: {status.get('phase', '?')}")
    _stderr(f"  VRAM: {status.get('vram_allocated_gb', '?')} / "
            f"{status.get('vram_total_gb', '?')} GB")
    logger.log({"phase": "startup", "server_status": status})

    if isinstance(client, MultiGPUClient):
        all_status = client.status_all()
        for i, s in enumerate(all_status):
            tag = "primary" if i == 0 else f"worker-{i}"
            _stderr(f"    [{tag}] phase={s.get('phase', '?')}, "
                    f"VRAM={s.get('vram_allocated_gb', '?')}/"
                    f"{s.get('vram_total_gb', '?')} GB")
        logger.log({"phase": "startup", "all_server_status": all_status})

    # ===================================================================
    # Pre-encode all prompts (TE loads once, then freed permanently)
    # ===================================================================

    run_btrm = not (args.resume or args.skip_btrm)
    run_policy = not args.skip_policy

    if run_btrm or run_policy:
        _stderr("")
        pos_conds, neg_cond = encode_all_prompts(client)
        prompt_cache = {
            prompt: cond
            for prompt, cond in zip(PROMPT_TEMPLATES, pos_conds)
        }
        client.free("te")
        _stderr("  TE freed after encoding all prompts.")
        logger.log({"phase": "encoding", "event": "prompts_encoded",
                     "n_prompts": len(pos_conds)})
    else:
        pos_conds, neg_cond, prompt_cache = None, None, None

    # ===================================================================
    # Phase 1: BTRM Training
    # ===================================================================

    if not run_btrm:
        _stderr("\n  Skipping Phase 1 (BTRM training)"
                + (" [--resume]" if args.resume else " [--skip-btrm]"))
        logger.log({"phase": "btrm", "event": "skipped",
                     "reason": "resume" if args.resume else "skip-btrm"})
    else:
        # Load trajectory pool
        _stderr(f"\n  Loading trajectories from {args.dataset_dir}...")
        pool = TrajectoryPool(args.dataset_dir, include_i2i=False)
        _stderr(f"  {len(pool.examples)} examples loaded")

        phase_btrm(
            client=client,
            pool=pool,
            logger=logger,
            n_macrobatches=args.btrm_macrobatches,
            macrobatch_size=args.btrm_batch_size,
            lr=args.btrm_lr,
            logsq_weight=args.btrm_logsq_weight,
            checkpoint_every=args.checkpoint_every,
            output_dir=args.output_dir,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            prompt_cache=prompt_cache,
            neg_cond=neg_cond,
            render_every=args.render_every,
        )

        del pool  # free memory

    # ===================================================================
    # Phase 2: Policy Optimization
    # ===================================================================

    if not run_policy:
        _stderr("\n  Skipping Phase 2 (policy optimization) [--skip-policy]")
        logger.log({"phase": "policy", "event": "skipped", "reason": "skip-policy"})
    else:
        phase_policy(
            client=client,
            pos_conds=pos_conds,
            neg_cond=neg_cond,
            logger=logger,
            n_iterations=args.policy_iterations,
            group_size=args.policy_group_size,
            rollout_steps=args.policy_rollout_steps,
            n_sparse=args.policy_sparse_steps,
            lr=args.policy_lr,
            cfg=args.cfg,
            checkpoint_every=args.checkpoint_every,
            output_dir=args.output_dir,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            render_every=args.render_every,
        )

    # ===================================================================
    # Phase 3: Final Dump
    # ===================================================================

    # Dump whenever we actually ran training (BTRM or policy or both)
    ran_btrm = not (args.resume or args.skip_btrm)
    ran_policy = not args.skip_policy
    if ran_btrm or ran_policy:
        phase_final(client, args.output_dir, logger)

    # ===================================================================
    # Summary
    # ===================================================================
    _section("Training Complete")
    _stderr(f"  Output: {args.output_dir}")
    _stderr(f"  Metrics: {logger.path}")

    logger.log({"phase": "done", "event": "training_complete"})
    logger.close()
    client.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

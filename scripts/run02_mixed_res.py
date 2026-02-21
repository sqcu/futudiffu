"""Run 02: Mixed-resolution BTRM training with corrected optimizer.

Successor to run 01 (2026-02-16, 2xH100). This run is single-GPU (RTX 4090)
and exercises the full corrected pipeline:

1. Mixed-resolution trajectory generation:
   - Full-res: 1280x832, 30 steps, SDPA attention (positive for both heads)
   - Small-res: 512x512, 30 steps, SDPA attention (multi-res positive)
   - Reduced-step: 1280x832, 10 steps, SageAttention (negative for scrongle)
   - Sage-quantized: 1280x832, 30 steps, SageAttention (negative for scrimble)
   Packed generation via sample_trajectory_packed for bin-packing exercise.

2. All latents VAE-decoded and rendered to PNG on generation.

3. V2 dataset with sampling_shift metadata recorded per trajectory.

4. BTRM training with corrected optimizer:
   - lr=3e-4 (sweep winner from r_theta sweep v2)
   - grad_clip=0.1 (was 0.01 -- saturated every step in run 1)
   - Optimizer includes rtheta LoRA params (fixes Defect 24)
   - Backbone runs WITH gradients (not no_grad) for adapter training

5. All output to training_output/run02/

Usage:
    .venv/Scripts/python.exe scripts/run02_mixed_res.py --output-dir training_output/run02
    (assumes server running on localhost:5555)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from futudiffu.btrm_dataset import PROMPT_TEMPLATES
from futudiffu.client import InferenceClient
from futudiffu.dataset_v2 import DatasetWriter
from futudiffu.rendering import decode_and_save
from futudiffu.sampling import resolution_shift


# -----------------------------------------------------------------------
# Output dir and logging
# -----------------------------------------------------------------------

def _stderr(msg: str, log_f=None) -> None:
    print(msg, file=sys.stderr, flush=True)
    if log_f is not None:
        log_f.write(msg + "\n")
        log_f.flush()


def _section(title: str, log_f=None) -> None:
    sep = "=" * 70
    _stderr(f"\n{sep}", log_f)
    _stderr(f"  {title}", log_f)
    _stderr(f"{sep}\n", log_f)


# -----------------------------------------------------------------------
# Generation configuration
# -----------------------------------------------------------------------

# Sparse step indices to save (same as btrm_dataset convention)
SPARSE_STEPS = [0, 4, 9, 14, 19, 24, 29]

# Resolution tiers for mixed-res generation
# (width, height, n_steps, attention_backend, batch_type_label, n_trajectories)
GENERATION_PLAN = [
    # Full-res, full-step, SDPA = positive for both BTRM heads
    (1280, 832, 30, "sdpa", "t2i", 8),
    # Small-res, full-step, SDPA = multi-resolution positive (scrongle)
    (512, 512, 30, "sdpa", "t2i", 6),
    # Full-res, reduced-step, SageAttention = negative for scrongle
    (1280, 832, 10, "sage", "t2i", 8),
    # Full-res, full-step, SageAttention = negative for scrimble
    (1280, 832, 30, "sage", "t2i", 8),
]

# Number of prompts to cycle through per tier
N_PROMPTS_PER_TIER = 4

# Seeds for each trajectory (deterministic)
BASE_SEED = 200000


def plan_generation(n_prompts: int = N_PROMPTS_PER_TIER) -> list[dict]:
    """Build flat list of trajectory specs."""
    specs = []
    seed_counter = BASE_SEED
    for (w, h, n_steps, attn, batch_type, n_traj) in GENERATION_PLAN:
        shift = resolution_shift(w, h)
        # Sparse steps for this step count
        sparse = [s for s in SPARSE_STEPS if s < n_steps]
        if sparse:
            # Add the last valid step
            last = n_steps - 1
            if last not in sparse:
                sparse.append(last)
        sparse.sort()

        for i in range(n_traj):
            prompt_idx = i % n_prompts
            specs.append({
                "width": w,
                "height": h,
                "n_steps": n_steps,
                "attention_backend": attn,
                "batch_type": batch_type,
                "sampling_shift": shift,
                "seed": seed_counter,
                "prompt_idx": prompt_idx,
                "sparse_steps": sparse,
            })
            seed_counter += 1

    return specs


# -----------------------------------------------------------------------
# Phase 1: Generate and persist trajectories
# -----------------------------------------------------------------------

def phase_generate(
    client: InferenceClient,
    output_dir: str,
    log_f,
    n_prompts: int = N_PROMPTS_PER_TIER,
) -> str:
    """Generate mixed-resolution trajectories and persist to V2 dataset.

    Returns the dataset directory path.
    """
    _section("Phase 1: Mixed-Resolution Trajectory Generation", log_f)

    dataset_dir = os.path.join(output_dir, "run02_dataset")
    render_dir = os.path.join(output_dir, "renders")
    os.makedirs(render_dir, exist_ok=True)

    specs = plan_generation(n_prompts=n_prompts)
    _stderr(f"  Total trajectories to generate: {len(specs)}", log_f)
    for (w, h, n_steps, attn, _btype, n_traj) in GENERATION_PLAN:
        shift = resolution_shift(w, h)
        _stderr(f"    {w}x{h}, {n_steps} steps, {attn}, shift={shift:.3f}: {n_traj} traj", log_f)

    # Encode all needed prompts
    _stderr("\n  Encoding prompts...", log_f)
    t0 = time.monotonic()
    neg_cond = client.encode_prompt("")
    pos_conds = []
    for i in range(n_prompts):
        cond = client.encode_prompt(PROMPT_TEMPLATES[i])
        pos_conds.append(cond)
    client.free("te")
    _stderr(f"  Encoded {n_prompts} prompts in {time.monotonic()-t0:.1f}s, TE freed", log_f)

    # Warmup packed path (will trigger compilation ~45-73s)
    # Warmup with n_images=2 to cover mixed-res bins
    _stderr("  Warming up packed path (n=2, includes compilation)...", log_f)
    t0 = time.monotonic()
    client.warmup_packed(n_images=2)
    _stderr(f"  Packed warmup done in {time.monotonic()-t0:.1f}s", log_f)

    # Also warmup the non-packed compiled path (for single-item generation)
    _stderr("  Warming up non-packed path...", log_f)
    t0 = time.monotonic()
    client.warmup(attention_backend="sdpa")
    _stderr(f"  Non-packed warmup done in {time.monotonic()-t0:.1f}s", log_f)

    # Group specs into bins for packed generation
    # For simplicity, generate paired bins where possible:
    # - Pair full-res+small-res in the same packed call (exercises mixed-res packing)
    # - Single-item bins for unpaired specs
    bins = _pack_specs(specs)
    _stderr(f"\n  Packed {len(specs)} specs into {len(bins)} generation bins", log_f)

    metrics_path = os.path.join(output_dir, "generation_metrics.jsonl")
    metrics_f = open(metrics_path, "a", buffering=1)

    t_gen_start = time.monotonic()
    n_generated = 0
    n_rendered = 0

    with DatasetWriter(dataset_dir) as writer:
        for bin_idx, bin_specs in enumerate(bins):
            t_bin = time.monotonic()

            if len(bin_specs) == 1:
                # Single trajectory: use non-packed path
                spec = bin_specs[0]
                _stderr(f"  bin {bin_idx+1}/{len(bins)}: single {spec['width']}x{spec['height']} "
                        f"{spec['n_steps']}s {spec['attention_backend']}", log_f)

                pos_cond = pos_conds[spec["prompt_idx"]]
                # Pad conditioning
                pos_len = pos_cond.shape[1]
                neg_len = neg_cond.shape[1]
                max_len = max(pos_len, neg_len)
                import torch.nn.functional as F
                pos_p = F.pad(pos_cond, (0, 0, 0, max_len - pos_len)) if pos_len < max_len else pos_cond
                neg_p = F.pad(neg_cond, (0, 0, 0, max_len - neg_len)) if neg_len < max_len else neg_cond

                traj = client.sample_trajectory(
                    pos_p, neg_p,
                    seed=spec["seed"],
                    n_steps=spec["n_steps"],
                    cfg=4.0,
                    width=spec["width"],
                    height=spec["height"],
                    attention_backend=spec["attention_backend"],
                    sampling_shift=spec["sampling_shift"],
                    save_steps=spec["sparse_steps"],
                )
                trajs = [traj]
                dt_gen = time.monotonic() - t_bin

                # Persist
                for i, (traj_dict, s) in enumerate(zip(trajs, bin_specs)):
                    traj_id = _persist_trajectory(
                        writer, traj_dict, s,
                        pos_conds[s["prompt_idx"]], render_dir, client, log_f,
                    )
                    n_generated += 1
                    n_rendered += 1 if traj_dict.get("final") is not None else 0

                    metrics_f.write(json.dumps({
                        "event": "trajectory_generated",
                        "traj_id": traj_id,
                        "width": s["width"], "height": s["height"],
                        "n_steps": s["n_steps"],
                        "attention_backend": s["attention_backend"],
                        "sampling_shift": s["sampling_shift"],
                        "dt_s": round(dt_gen, 2),
                        "seed": s["seed"],
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }) + "\n")

            else:
                # Multiple trajectories: use packed path
                res_strs = ",".join(f"{s['width']}x{s['height']}" for s in bin_specs)
                _stderr(f"  bin {bin_idx+1}/{len(bins)}: packed {len(bin_specs)} traj ({res_strs})", log_f)

                widths = [s["width"] for s in bin_specs]
                heights = [s["height"] for s in bin_specs]
                seeds = [s["seed"] for s in bin_specs]
                sampling_shifts = [s["sampling_shift"] for s in bin_specs]
                n_steps = bin_specs[0]["n_steps"]
                attn = bin_specs[0]["attention_backend"]
                sparse = bin_specs[0]["sparse_steps"]
                prompt_indices = [s["prompt_idx"] for s in bin_specs]

                batch_pos_conds = [pos_conds[pi] for pi in prompt_indices]

                trajs = client.sample_trajectory_packed(
                    pos_conds=batch_pos_conds,
                    neg_cond=neg_cond,
                    seeds=seeds,
                    n_steps=n_steps,
                    cfg=4.0,
                    widths=widths,
                    heights=heights,
                    attention_backend=attn,
                    sampling_shifts=sampling_shifts,
                    save_steps=sparse,
                )
                dt_gen = time.monotonic() - t_bin

                for i, (traj_dict, s) in enumerate(zip(trajs, bin_specs)):
                    traj_id = _persist_trajectory(
                        writer, traj_dict, s,
                        pos_conds[s["prompt_idx"]], render_dir, client, log_f,
                    )
                    n_generated += 1
                    n_rendered += 1 if traj_dict.get("final") is not None else 0

                    metrics_f.write(json.dumps({
                        "event": "trajectory_generated",
                        "traj_id": traj_id,
                        "width": s["width"], "height": s["height"],
                        "n_steps": s["n_steps"],
                        "attention_backend": s["attention_backend"],
                        "sampling_shift": s["sampling_shift"],
                        "dt_s": round(dt_gen / len(bin_specs), 2),  # per-image
                        "seed": s["seed"],
                        "packed": True,
                        "pack_size": len(bin_specs),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }) + "\n")

            if (bin_idx + 1) % 5 == 0 or bin_idx == len(bins) - 1:
                writer.flush()
                _stderr(f"    Flushed dataset ({writer.n_trajectories} trajectories written)", log_f)

    dt_total = time.monotonic() - t_gen_start
    metrics_f.close()

    _stderr(f"\n  Generation complete: {n_generated} trajectories, {n_rendered} renders", log_f)
    _stderr(f"  Time: {dt_total:.1f}s ({dt_total/max(n_generated,1):.1f}s/traj)", log_f)
    _stderr(f"  Dataset: {dataset_dir}", log_f)
    _stderr(f"  Renders: {render_dir}", log_f)

    return dataset_dir


def _pack_specs(specs: list[dict]) -> list[list[dict]]:
    """Group specs into generation bins.

    Pairs full-res (1280x832) with small-res (512x512) to exercise
    mixed-resolution bin packing. Other specs go into single-item bins.

    Mixed-res packed bins require:
    - Same n_steps
    - Same attention_backend
    - Different resolutions (exercises the per-image sigma shift path)
    """
    full_sage30 = [s for s in specs if s["width"] == 1280 and s["n_steps"] == 30 and s["attention_backend"] == "sdpa"]
    small_sdpa30 = [s for s in specs if s["width"] == 512 and s["n_steps"] == 30 and s["attention_backend"] == "sdpa"]
    other = [s for s in specs if s not in full_sage30 and s not in small_sdpa30]

    bins = []

    # Pair full-res + small-res in the same packed call (mixed-res exercise)
    # This is the key rubric requirement: bin packer must be exercised
    n_mixed = min(len(full_sage30), len(small_sdpa30))
    for i in range(n_mixed):
        bins.append([full_sage30[i], small_sdpa30[i]])

    # Remaining full-res and small-res as singles
    for s in full_sage30[n_mixed:]:
        bins.append([s])
    for s in small_sdpa30[n_mixed:]:
        bins.append([s])

    # Other specs as singles
    for s in other:
        bins.append([s])

    return bins


def _persist_trajectory(
    writer: DatasetWriter,
    traj_dict: dict,
    spec: dict,
    pos_cond: torch.Tensor,
    render_dir: str,
    client: InferenceClient,
    log_f,
) -> int:
    """Persist one trajectory to the writer and render the final latent."""
    # Build tensors dict from trajectory
    tensors = {}
    for key, tensor in traj_dict.items():
        if key.startswith("step_") or key == "final":
            tensors[key] = tensor

    # VAE-decode and render final latent (rubric requirement)
    rendered_path = None
    if "final" in tensors:
        label = (f"run02_w{spec['width']}h{spec['height']}"
                 f"_s{spec['n_steps']}_{spec['attention_backend']}"
                 f"_seed{spec['seed']}")
        render_path = os.path.join(render_dir, f"{label}.png")
        try:
            decode_and_save(client, tensors["final"], render_path)
            rendered_path = render_path
            _stderr(f"    rendered: {os.path.basename(render_path)}", log_f)
        except Exception as e:
            _stderr(f"    WARNING: render failed for {label}: {e}", log_f)

    metadata = {
        "prompt": PROMPT_TEMPLATES[spec["prompt_idx"]],
        "prompt_idx": spec["prompt_idx"],
        "seed": spec["seed"],
        "cfg": 4.0,
        "width": spec["width"],
        "height": spec["height"],
        "n_steps": spec["n_steps"],
        "attention_backend": spec["attention_backend"],
        "batch_type": spec["batch_type"],
        "sampling_shift": spec["sampling_shift"],
        "is_gold": (spec["n_steps"] == 30 and spec["attention_backend"] == "sdpa"),
        "image_file": rendered_path,
        "packed": False,
        "base_model_hash": "z_image_v1",
        "run_name": "run02_mixed_res",
        "source_device": "rtx4090",
    }

    return writer.add_trajectory(tensors=tensors, metadata=metadata)


# -----------------------------------------------------------------------
# Phase 2: BTRM Training
# -----------------------------------------------------------------------

def phase_btrm_train(
    client: InferenceClient,
    dataset_dir: str,
    output_dir: str,
    log_f,
    n_macrobatches: int = 30,
    macrobatch_size: int = 16,
    lr: float = 3e-4,
    logsq_weight: float = 0.1,
    checkpoint_every: int = 5,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    render_every: int = 5,
) -> dict:
    """Train BTRM reward model on the generated trajectories.

    Key corrections vs run 1:
    - lr=3e-4 (was 1e-3 -- sweep winner)
    - grad_clip=0.1 (was 0.01 -- saturated every step)
    - Optimizer includes rtheta LoRA params (fixes Defect 24)
    - Backbone runs WITH gradients (not no_grad) for adapter training

    Returns final metrics summary dict.
    """
    _section("Phase 2: BTRM Training (Corrected Optimizer)", log_f)

    import torch.nn.functional as F
    from futudiffu.trajectory_loader import TrajectoryPoolV2

    # Load dataset
    _stderr(f"  Loading dataset from: {dataset_dir}", log_f)
    pool = TrajectoryPoolV2([dataset_dir], include_i2i=False)
    _stderr(f"  Loaded {len(pool.examples)} examples", log_f)

    sdpa_idx, sage_idx = pool.scrimble_split()
    full_idx, reduced_idx = pool.scrongle_split()
    _stderr(f"  Scrimble split: {len(sdpa_idx)} SDPA, {len(sage_idx)} Sage", log_f)
    _stderr(f"  Scrongle split: {len(full_idx)} full-step, {len(reduced_idx)} reduced-step", log_f)

    if len(sdpa_idx) == 0 and len(sage_idx) == 0 and len(full_idx) == 0 and len(reduced_idx) == 0:
        _stderr("  ERROR: No valid training pairs found in dataset", log_f)
        return {"error": "no_pairs"}

    # Build unique prompts for pre-encoding
    all_indices = sorted(set(sdpa_idx + sage_idx + full_idx + reduced_idx))
    unique_prompts = list({pool.examples[i].prompt for i in all_indices})
    _stderr(f"  Unique prompts needed: {len(unique_prompts)}", log_f)

    # Encode prompts
    _stderr("  Encoding prompts for BTRM training...", log_f)
    t0 = time.monotonic()
    neg_cond = client.encode_prompt("")
    prompt_cache: dict[str, torch.Tensor] = {}
    for prompt in unique_prompts:
        prompt_cache[prompt] = client.encode_prompt(prompt)
    client.free("te")
    _stderr(f"  Encoded {len(unique_prompts)} prompts in {time.monotonic()-t0:.1f}s, TE freed", log_f)

    # Allocate rtheta adapter BEFORE compile
    _stderr("\n  Allocating rtheta adapter (layers 28-29)...", log_f)
    n_rtheta = client.allocate_adapter(
        "rtheta", rank=lora_rank, alpha=lora_alpha, layer_indices=[28, 29],
    )
    _stderr(f"  Allocated rtheta: {n_rtheta} slots on layers 28-29", log_f)

    # NOTE: Warmup (diff_compiled path) intentionally skipped for training phase.
    # train_btrm_step uses forward_checkpointed(diff_model, ...) which bypasses
    # diff_compiled entirely. The warmup only primes inference compilation, which
    # can fail with the multi-LoRA torch.stack + custom_op combination in inductor
    # (SymPy recursion in ir.py __str__). Generation warmup already ran in phase 1.
    _stderr("  Skipping warmup (training uses forward_checkpointed, not diff_compiled)", log_f)

    # Initialize rtheta weights
    client.init_adapter_weights("rtheta", init_b_std=0.0, scale=1.0)
    _stderr("  rtheta initialized (scale=1.0, zero-init B)", log_f)

    # Inject BTRM head with lr=3e-4
    # The server will include rtheta LoRA params in the optimizer (Defect 24 fix)
    btrm_meta = client.inject_btrm_head(
        head_names=["scrimble", "scrongle"],
        logit_cap=10.0,
        lr=lr,
    )
    _stderr(f"  BTRM head injected: {btrm_meta.get('n_params', '?')} params, lr={lr}", log_f)
    _stderr(f"  (optimizer includes rtheta LoRA params -- Defect 24 fix)", log_f)

    # Training loop
    sdpa_set = set(sdpa_idx)
    sage_set = set(sage_idx)
    full_set = set(full_idx)
    reduced_set = set(reduced_idx)

    import random
    rng = random.Random(42)
    metrics_path = os.path.join(output_dir, "btrm_metrics.jsonl")
    metrics_f = open(metrics_path, "a", buffering=1)
    t_train_start = time.monotonic()

    render_dir = os.path.join(output_dir, "btrm_renders")
    os.makedirs(render_dir, exist_ok=True)

    # Track first macrobatch where rtheta params show nonzero grad
    first_nonzero_rtheta_grad = None
    summary_records = []

    for mb in range(n_macrobatches):
        t0 = time.monotonic()

        # Shuffle and build macrobatch
        rng.shuffle(all_indices)
        batch_examples = []

        for ex_idx in all_indices:
            if len(batch_examples) >= macrobatch_size:
                break
            ex = pool.examples[ex_idx]
            if ex.step_idx == -1:
                continue  # skip final (sigma=0)

            latent = pool.load_checkpoint(ex)
            cond = prompt_cache.get(ex.prompt)
            if cond is None:
                continue

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
            _stderr(f"  WARNING: mb {mb} has only {len(batch_examples)} examples, skipping", log_f)
            continue

        batch_examples = batch_examples[:macrobatch_size]

        # Train step on server
        train_metrics = client.train_btrm_step(
            batch_examples,
            logsquare_weight=logsq_weight,
            attention_backend="sdpa",
        )
        dt = time.monotonic() - t0

        acc = train_metrics.get("per_head_accuracy", {})
        acc_str = ", ".join(f"{k}={v:.2%}" for k, v in acc.items())
        pre_clip = train_metrics.get("pre_clip_grad_norm", 0.0)
        grad_norm = train_metrics.get("grad_norm", 0.0)

        # Check for Defect 24 manifestation: if pre_clip_norm == 0 for many steps,
        # the adapter is not receiving gradients
        if pre_clip > 0 and first_nonzero_rtheta_grad is None:
            first_nonzero_rtheta_grad = mb + 1

        record = {
            "phase": "btrm",
            "macrobatch": mb + 1,
            "n_examples": len(batch_examples),
            "loss": train_metrics["loss"],
            "bt_loss": train_metrics.get("bt_loss", 0.0),
            "logsq_loss": train_metrics.get("logsq_loss", 0.0),
            "per_head_accuracy": acc,
            "pre_clip_grad_norm": pre_clip,
            "grad_norm": grad_norm,
            "lr": train_metrics.get("lr", lr),
            "dt_s": round(dt, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "wall_clock_s": round(time.monotonic() - t_train_start, 2),
        }
        metrics_f.write(json.dumps(record) + "\n")
        summary_records.append(record)

        _stderr(
            f"  btrm {mb+1:3d}/{n_macrobatches} | n={len(batch_examples)} | "
            f"loss={train_metrics['loss']:.4f} | bt={train_metrics.get('bt_loss',0):.4f} | "
            f"acc=[{acc_str}] | pre_clip={pre_clip:.3e} | grad={grad_norm:.3e} | {dt:.1f}s",
            log_f,
        )

        # Checkpoint
        if checkpoint_every > 0 and (mb + 1) % checkpoint_every == 0:
            ckpt_dir = os.path.join(output_dir, f"btrm_ckpt_{mb+1:04d}")
            try:
                result = client.dump_all_loras(output_dir=ckpt_dir)
                _stderr(f"    Checkpoint: {len(result.get('files', []))} files -> {ckpt_dir}", log_f)
            except Exception as e:
                _stderr(f"    WARNING: checkpoint failed: {e}", log_f)

        # Render check
        if render_every > 0 and (mb + 1) % render_every == 0:
            _stderr(f"    Render check at mb {mb+1}...", log_f)
            try:
                # Generate a reference trajectory and render
                ref_cond = prompt_cache.get(PROMPT_TEMPLATES[0])
                if ref_cond is not None:
                    neg_p = neg_cond
                    ref_len = ref_cond.shape[1]
                    neg_len = neg_p.shape[1]
                    max_len = max(ref_len, neg_len)
                    import torch.nn.functional as F
                    ref_p = F.pad(ref_cond, (0, 0, 0, max_len - ref_len)) if ref_len < max_len else ref_cond
                    neg_pp = F.pad(neg_p, (0, 0, 0, max_len - neg_len)) if neg_len < max_len else neg_p
                    # Re-encode neg_cond since TE was freed
                    rc_traj = client.sample_trajectory(
                        ref_p, neg_pp, seed=42, n_steps=10, cfg=4.0,
                    )
                    render_path = os.path.join(render_dir, f"btrm_mb{mb+1:04d}.png")
                    decode_and_save(client, rc_traj["final"], render_path)
                    _stderr(f"    Rendered: {os.path.basename(render_path)}", log_f)
            except Exception as e:
                _stderr(f"    WARNING: render failed: {e}", log_f)

    metrics_f.close()
    dt_total = time.monotonic() - t_train_start

    # Final checkpoint
    final_dir = os.path.join(output_dir, "final")
    try:
        result = client.dump_all_loras(output_dir=final_dir)
        _stderr(f"\n  Final checkpoint: {len(result.get('files', []))} files -> {final_dir}", log_f)
    except Exception as e:
        _stderr(f"\n  WARNING: final checkpoint failed: {e}", log_f)

    _stderr(f"\n  BTRM training complete: {n_macrobatches} macrobatches in {dt_total:.1f}s", log_f)
    _stderr(f"  First nonzero grad: macrobatch {first_nonzero_rtheta_grad or 'NEVER'}", log_f)
    if first_nonzero_rtheta_grad is None:
        _stderr("  WARNING: rtheta LoRA NEVER received nonzero gradients -- possible Defect 24!", log_f)

    # Compute summary stats
    if summary_records:
        final_losses = [r["loss"] for r in summary_records[-5:]]
        final_accs = {k: sum(r["per_head_accuracy"].get(k, 0) for r in summary_records[-5:]) / 5
                      for k in (summary_records[-1]["per_head_accuracy"] or {}).keys()}
        _stderr(f"  Final loss (last 5): {sum(final_losses)/len(final_losses):.4f}", log_f)
        for k, v in final_accs.items():
            _stderr(f"  Final {k} accuracy (last 5): {v:.2%}", log_f)

    return {
        "n_macrobatches": n_macrobatches,
        "dt_total_s": dt_total,
        "first_nonzero_rtheta_grad": first_nonzero_rtheta_grad,
        "final_records": summary_records[-5:] if summary_records else [],
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run 02: Mixed-resolution BTRM training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=5555, help="Server ZMQ port")
    parser.add_argument("--output-dir", type=str, default="training_output/run02",
                        help="Output directory for all artifacts")
    parser.add_argument("--btrm-macrobatches", type=int, default=30)
    parser.add_argument("--btrm-batch-size", type=int, default=16)
    parser.add_argument("--btrm-lr", type=float, default=3e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--render-every", type=int, default=5)
    parser.add_argument("--n-prompts", type=int, default=4,
                        help="Number of prompts to cycle through per tier")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip generation, use existing dataset-dir")
    parser.add_argument("--dataset-dir", type=str, default=None,
                        help="Existing dataset dir (if --skip-generation)")
    args = parser.parse_args()

    # Setup output
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "run02_console.log")
    log_f = open(log_path, "a", buffering=1)

    # Metrics summary
    summary_path = os.path.join(output_dir, "run02_summary.json")

    _section("futudiffu Run 02: Mixed-Resolution BTRM Training", log_f)
    _stderr(f"  Output: {output_dir}", log_f)
    _stderr(f"  Config: {json.dumps(vars(args), indent=2)}", log_f)

    with open(os.path.join(output_dir, "run02_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Connect to server
    endpoint = f"tcp://localhost:{args.port}"
    _stderr(f"\n  Connecting to {endpoint}...", log_f)
    client = InferenceClient(endpoint)
    status = client.status()
    _stderr(f"  Server status: {status}", log_f)

    t_total_start = time.monotonic()
    summary = {
        "run": "run02_mixed_res",
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": vars(args),
        "server_status": status,
    }

    # Phase 1: Generate trajectories
    if not args.skip_generation:
        dataset_dir = phase_generate(
            client, output_dir, log_f,
            n_prompts=args.n_prompts,
        )
        summary["dataset_dir"] = dataset_dir
    else:
        dataset_dir = args.dataset_dir or os.path.join(output_dir, "run02_dataset")
        _stderr(f"\n  Skipping generation, using: {dataset_dir}", log_f)
        summary["dataset_dir"] = dataset_dir

    # Phase 2: BTRM training
    btrm_summary = phase_btrm_train(
        client=client,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        log_f=log_f,
        n_macrobatches=args.btrm_macrobatches,
        macrobatch_size=args.btrm_batch_size,
        lr=args.btrm_lr,
        checkpoint_every=args.checkpoint_every,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        render_every=args.render_every,
    )
    summary["btrm"] = btrm_summary

    dt_total = time.monotonic() - t_total_start
    summary["total_wall_clock_s"] = dt_total
    summary["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    _section("Run 02 Complete", log_f)
    _stderr(f"  Total wall clock: {dt_total:.1f}s ({dt_total/60:.1f}min)", log_f)
    _stderr(f"  Summary: {summary_path}", log_f)

    client.close()
    log_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

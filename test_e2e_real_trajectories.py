"""E2E training test: BTRM + Policy on real trajectories via server RPCs.

Two-phase test that exercises the full server-based training pipeline
against stored trajectory data from btrm_dataset/.

Phase 1 (BTRM): Score stored checkpoints via server's train_btrm_step
(backbone + BTRM head + loss + backward + step all server-side).

Phase 2 (Policy): Live rollouts via sample_trajectory, BTRM scoring
via score_btrm, policy gradients via accumulate_policy_gradients,
optimizer step via policy_optimizer_step -- all on server.

Requires a running server (see invocation at bottom of file).
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time

import torch
import torch.nn.functional as F

from futudiffu.client import InferenceClient
from futudiffu.policy_loss import compute_group_advantages
from futudiffu.sampling import build_sigmas, simple_scheduler
from futudiffu.trajectory_loader import TrajectoryPool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log_section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Phase 1: BTRM Training
# ---------------------------------------------------------------------------

def phase_btrm(
    client: InferenceClient,
    pool: TrajectoryPool,
    n_epochs: int = 3,
    macrobatch_size: int = 32,
    lr: float = 1e-3,
    logsq_weight: float = 0.1,
) -> list[dict]:
    """Train BTRM head on stored trajectory checkpoints.

    Accumulates examples into macrobatches, then sends each macrobatch
    to the server via train_btrm_step (all computation server-side).

    Returns list of per-macrobatch metric dicts.
    """
    log_section("Phase 1: BTRM Training on Stored Trajectories")

    # Get splits
    sdpa_idx, sage_idx = pool.scrimble_split()
    full_idx, reduced_idx = pool.scrongle_split()
    print(f"  Scrimble split: {len(sdpa_idx)} SDPA, {len(sage_idx)} Sage")
    print(f"  Scrongle split: {len(full_idx)} full-step, {len(reduced_idx)} reduced-step")

    # Build unified index set (deduplicated)
    all_indices = sorted(set(sdpa_idx + sage_idx + full_idx + reduced_idx))
    print(f"  Total unique examples: {len(all_indices)}")

    # Precompute which indices belong to which splits (as sets for O(1) lookup)
    sdpa_set = set(sdpa_idx)
    sage_set = set(sage_idx)
    full_set = set(full_idx)
    reduced_set = set(reduced_idx)

    # Cache encoded prompts (text -> conditioning tensor)
    prompt_cache: dict[str, torch.Tensor] = {}
    unique_prompts = {pool.examples[i].prompt for i in all_indices}
    print(f"  Encoding {len(unique_prompts)} unique prompts...")
    for prompt_text in unique_prompts:
        prompt_cache[prompt_text] = client.encode_prompt(prompt_text)
    print(f"  Prompt encoding done.")

    # Inject BTRM head on server with optimizer
    client.inject_btrm_head(
        head_names=["scrimble", "scrongle"],
        logit_cap=10.0,
        lr=lr,
    )
    print(f"  BTRM head injected on server (lr={lr})")

    log_entries = []

    for epoch in range(n_epochs):
        log_section(f"BTRM Epoch {epoch + 1}/{n_epochs}")

        # Shuffle example order
        epoch_indices = all_indices.copy()
        random.shuffle(epoch_indices)

        # Accumulators for macrobatch
        batch_examples: list[dict] = []
        batch_meta: list[dict] = []  # track provenance for accuracy reporting
        n_scored = 0
        epoch_t0 = time.time()

        for ex_idx in epoch_indices:
            ex = pool.examples[ex_idx]

            # Skip examples with sigma=0 (final checkpoints)
            if ex.step_idx == -1:
                continue

            # Load checkpoint and get conditioning
            latent = pool.load_checkpoint(ex)  # (1, 16, 104, 160)
            cond = prompt_cache[ex.prompt]

            # Determine head_idx and is_positive from provenance
            # Head 0 (scrimble): SDPA=positive, Sage=negative
            if ex_idx in sdpa_set:
                batch_examples.append({
                    "latent": latent, "sigma": ex.sigma,
                    "conditioning": cond, "head_idx": 0, "is_positive": True,
                })
                batch_meta.append({"head": "scrimble", "pos": True})
            elif ex_idx in sage_set:
                batch_examples.append({
                    "latent": latent, "sigma": ex.sigma,
                    "conditioning": cond, "head_idx": 0, "is_positive": False,
                })
                batch_meta.append({"head": "scrimble", "pos": False})

            # Head 1 (scrongle): full-step=positive, reduced=negative
            # An example can belong to both heads (e.g. SDPA + full-step)
            if ex_idx in full_set:
                batch_examples.append({
                    "latent": latent, "sigma": ex.sigma,
                    "conditioning": cond, "head_idx": 1, "is_positive": True,
                })
                batch_meta.append({"head": "scrongle", "pos": True})
            elif ex_idx in reduced_set:
                batch_examples.append({
                    "latent": latent, "sigma": ex.sigma,
                    "conditioning": cond, "head_idx": 1, "is_positive": False,
                })
                batch_meta.append({"head": "scrongle", "pos": False})

            n_scored += 1
            if n_scored % 20 == 0:
                print(f"    Accumulated {n_scored}/{len(epoch_indices)} examples")

            # Flush macrobatch
            if len(batch_examples) >= macrobatch_size:
                t0 = time.time()
                metrics = client.train_btrm_step(
                    batch_examples,
                    logsquare_weight=logsq_weight,
                    attention_backend="sdpa",
                )
                dt = time.time() - t0

                entry = dict(
                    epoch=epoch,
                    batch=len(log_entries),
                    n_examples=len(batch_examples),
                    loss=metrics["loss"],
                    bt_loss=metrics["bt_loss"],
                    logsq_loss=metrics["logsq_loss"],
                    per_head_accuracy=metrics.get("per_head_accuracy", {}),
                    time=dt,
                )
                log_entries.append(entry)

                acc_str = ", ".join(
                    f"{k}={v:.2%}" for k, v in entry["per_head_accuracy"].items()
                )
                print(
                    f"    batch {entry['batch']:3d} | n={len(batch_examples)} | "
                    f"loss={metrics['loss']:.4f} | bt={metrics['bt_loss']:.4f} | "
                    f"acc=[{acc_str}] | {dt:.1f}s"
                )

                batch_examples = []
                batch_meta = []

        # Flush remaining
        if len(batch_examples) >= 4:
            t0 = time.time()
            metrics = client.train_btrm_step(
                batch_examples,
                logsquare_weight=logsq_weight,
                attention_backend="sdpa",
            )
            dt = time.time() - t0

            entry = dict(
                epoch=epoch,
                batch=len(log_entries),
                n_examples=len(batch_examples),
                loss=metrics["loss"],
                bt_loss=metrics["bt_loss"],
                logsq_loss=metrics["logsq_loss"],
                per_head_accuracy=metrics.get("per_head_accuracy", {}),
                time=dt,
            )
            log_entries.append(entry)

        epoch_dt = time.time() - epoch_t0
        print(f"\n  Epoch {epoch + 1} done in {epoch_dt:.1f}s, "
              f"scored {n_scored} examples")

    return log_entries


# ---------------------------------------------------------------------------
# Phase 2: Policy Optimization
# ---------------------------------------------------------------------------

def phase_policy(
    client: InferenceClient,
    pos_cond: torch.Tensor,
    neg_cond: torch.Tensor,
    n_iterations: int = 10,
    group_size: int = 2,
    rollout_steps: int = 10,
    lr: float = 1e-4,
    grad_clip: float = 1.0,
) -> list[dict]:
    """Policy optimization via live rollouts + server-side BTRM scoring + server-side gradients.

    1. Freeze rtheta (scale=0).
    2. Inject ptheta LoRA on all layers.
    3. For each iteration: rollout -> score_btrm -> accumulate_policy_gradients -> policy_optimizer_step.
    """
    log_section("Phase 2: Policy Optimization via Live Rollouts")

    # Freeze rtheta
    print("  Freezing rtheta (scale=0, frozen=True)...")
    client.set_adapter_config("rtheta", scale=0.0, frozen=True)

    # Inject ptheta on all layers
    print("  Injecting ptheta LoRA on all layers...")
    n_adapters = client.inject_lora("ptheta", rank=8, alpha=16.0, init_b_std=0.01)
    print(f"    {n_adapters} adapters injected")

    # Warmup after LoRA injection (recompile)
    print("  Warming up after ptheta injection...")
    client.warmup(attention_backend="sdpa")

    # Snapshot initial weights for before/after comparison
    initial_sd = client.get_lora_state_dict("ptheta")
    initial_snapshot = {k: v.clone() for k, v in initial_sd.items()}
    print(f"    {len(initial_sd)} weight tensors")

    # Build sigma schedule for rollouts
    sigma_table = build_sigmas(shift=1.0, multiplier=1000.0)
    sigmas = simple_scheduler(sigma_table, rollout_steps)

    # Pick sparse steps for gradient computation (evenly spaced)
    n_sparse = min(5, rollout_steps)
    sparse_steps = [int(i * (rollout_steps - 1) / (n_sparse - 1))
                    for i in range(n_sparse)]
    print(f"  Rollout steps: {rollout_steps}, sparse steps: {sparse_steps}")

    log_entries = []
    base_seed = 42000

    for iteration in range(n_iterations):
        t0 = time.time()

        # -- Generate K rollouts --
        rewards = []
        rollout_data = []  # (checkpoints_dict, seed) per rollout

        for k in range(group_size):
            seed = base_seed + iteration * group_size + k
            result = client.sample_trajectory(
                pos_cond=pos_cond,
                neg_cond=neg_cond,
                seed=seed,
                n_steps=rollout_steps,
                cfg=4.0,
                attention_backend="sdpa",
                save_steps=sparse_steps,
            )

            # Score via server-side BTRM
            final_latent = result["final"]
            mid_idx = rollout_steps // 2
            sigma_mid = torch.tensor([float(sigmas[mid_idx])])

            scores = client.score_btrm(
                latent=final_latent,
                sigma=sigma_mid,
                conditioning=pos_cond,
            )
            reward = scores[0][0]  # first example, head 0 (scrimble / bit quality)
            rewards.append(reward)

            # Collect checkpoints for gradient computation
            checkpoints = {}
            for step in sparse_steps:
                key = f"step_{step:02d}"
                if key in result:
                    checkpoints[step] = result[key]
            rollout_data.append((checkpoints, seed))

        # -- Compute advantages --
        rewards_tensor = torch.tensor(rewards)
        advantages = compute_group_advantages(rewards_tensor)

        # -- Accumulate gradients on server for each rollout --
        for k in range(group_size):
            checkpoints, seed = rollout_data[k]
            adv = advantages[k].item()

            client.accumulate_policy_gradients(
                checkpoints=checkpoints,
                sigmas=sigmas,
                conditioning=pos_cond,
                adapter_name="ptheta",
                advantage=adv,
            )

        # -- Optimizer step on server --
        step_meta = client.policy_optimizer_step(
            adapter_name="ptheta",
            max_grad_norm=grad_clip,
            lr=lr,
        )

        dt = time.time() - t0

        grad_norm_val = step_meta["grad_norm"]

        entry = dict(
            iteration=iteration,
            rewards=rewards,
            mean_reward=rewards_tensor.mean().item(),
            advantages=advantages.tolist(),
            grad_norm=grad_norm_val,
            n_params=step_meta["n_params"],
            time=dt,
        )
        log_entries.append(entry)

        print(
            f"  iter {iteration:3d} | "
            f"rewards={[f'{r:+.4f}' for r in rewards]} | "
            f"mean={rewards_tensor.mean().item():+.4f} | "
            f"grad_norm={grad_norm_val:.3e} | "
            f"n_params={step_meta['n_params']} | {dt:.1f}s"
        )

    # -- Verification: weights changed --
    final_sd = client.get_lora_state_dict("ptheta")
    weight_changed = False
    for key in initial_snapshot:
        if key in final_sd:
            cos = F.cosine_similarity(
                initial_snapshot[key].flatten().float(),
                final_sd[key].flatten().float(),
                dim=0,
            ).item()
            if cos < 1.0 - 1e-6:
                weight_changed = True
                print(f"  Weight changed: {key} (cos={cos:.6f})")
                break

    if weight_changed:
        print("  Verification: ptheta weights CHANGED (training had effect)")
    else:
        print("  Verification: ptheta weights UNCHANGED (WARNING)")

    return log_entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="E2E training test on real trajectories"
    )
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--dataset", type=str,
                        default=r"F:\dox\repos\ai\futudiffu\btrm_dataset")
    parser.add_argument("--btrm-epochs", type=int, default=3)
    parser.add_argument("--macrobatch", type=int, default=32)
    parser.add_argument("--policy-steps", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=10)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--btrm-lr", type=float, default=1e-3)
    parser.add_argument("--policy-lr", type=float, default=1e-4)
    parser.add_argument("--skip-policy", action="store_true",
                        help="Run only BTRM phase")
    args = parser.parse_args()

    log_section("E2E Training Test: BTRM + Policy on Real Trajectories")

    # -- Connect --
    endpoint = f"tcp://localhost:{args.port}"
    print(f"Connecting to server at {endpoint}...")
    client = InferenceClient(endpoint)

    status = client.status()
    print(f"  Server status: {status}")

    # -- Load trajectory pool --
    print(f"\nLoading trajectories from {args.dataset}...")
    pool = TrajectoryPool(args.dataset, include_i2i=False)
    print(f"  {len(pool.examples)} examples loaded")

    sdpa_idx, sage_idx = pool.scrimble_split()
    full_idx, reduced_idx = pool.scrongle_split()
    print(f"  Scrimble: {len(sdpa_idx)} SDPA / {len(sage_idx)} Sage")
    print(f"  Scrongle: {len(full_idx)} full / {len(reduced_idx)} reduced")

    # -- Inject rtheta LoRA (for BTRM training backbone adaptation) --
    print("\nInjecting rtheta LoRA on layers 28-29...")
    n_rtheta = client.inject_lora("rtheta", rank=8, alpha=16.0,
                                  layer_indices=[28, 29])
    print(f"  {n_rtheta} adapters injected")

    # Warmup after injection
    print("Warming up...")
    client.warmup(attention_backend="sdpa")

    # ===================================================================
    # Phase 1: BTRM
    # ===================================================================
    btrm_log = phase_btrm(
        client=client,
        pool=pool,
        n_epochs=args.btrm_epochs,
        macrobatch_size=args.macrobatch,
        lr=args.btrm_lr,
    )

    # ===================================================================
    # Phase 2: Policy
    # ===================================================================
    policy_log = []
    if not args.skip_policy:
        # Encode prompts for rollouts (use laser shark as canonical)
        laser_shark_prompt = (
            'ahem.\n*ting ting ting ting ting*\n'
            'the query model for this is a LARGE LANGUAGE MODEL, specifically '
            'QWEN-3-4B, a GENERAL PURPOSE SEMANTIC PARSER which is able to '
            'WRITE SENTENCES AT A TIME when they are participating in dialogue. '
            'however, in this situation, they are being used as a hidden state '
            'generator to steer an *image generation model*, z-image.\n\n'
            'qwen-3-4b, draw me an "enormous laser shark for the sega saturn".'
        )
        print("\nEncoding prompts for policy rollouts...")
        pos_cond = client.encode_prompt(laser_shark_prompt)
        neg_cond = client.encode_prompt("")

        policy_log = phase_policy(
            client=client,
            pos_cond=pos_cond,
            neg_cond=neg_cond,
            n_iterations=args.policy_steps,
            group_size=args.group_size,
            rollout_steps=args.rollout_steps,
            lr=args.policy_lr,
        )

    # ===================================================================
    # Summary
    # ===================================================================
    log_section("Results Summary")

    # BTRM checks
    btrm_ok = True
    if btrm_log:
        any_nan = any(math.isnan(e["loss"]) for e in btrm_log)
        if any_nan:
            print("  BTRM: FAIL - NaN in losses")
            btrm_ok = False
        else:
            print("  BTRM: No NaN in losses")

        # Check accuracy trend
        per_head_accs = [e.get("per_head_accuracy", {}) for e in btrm_log]
        first_with_scrimble = next((a for a in per_head_accs if "scrimble" in a), None)
        last_with_scrimble = next((a for a in reversed(per_head_accs) if "scrimble" in a), None)
        if first_with_scrimble and last_with_scrimble:
            print(f"  Scrimble acc: first={first_with_scrimble['scrimble']:.2%} "
                  f"-> last={last_with_scrimble['scrimble']:.2%}")
            if last_with_scrimble["scrimble"] > 0.6:
                print("  Scrimble: > 60% accuracy (PASS)")

        first_with_scrongle = next((a for a in per_head_accs if "scrongle" in a), None)
        last_with_scrongle = next((a for a in reversed(per_head_accs) if "scrongle" in a), None)
        if first_with_scrongle and last_with_scrongle:
            print(f"  Scrongle acc: first={first_with_scrongle['scrongle']:.2%} "
                  f"-> last={last_with_scrongle['scrongle']:.2%}")
            if last_with_scrongle["scrongle"] > 0.6:
                print("  Scrongle: > 60% accuracy (PASS)")

        # Loss trend
        losses = [e["loss"] for e in btrm_log]
        if len(losses) >= 2:
            first_half = sum(losses[:len(losses)//2]) / max(len(losses)//2, 1)
            second_half = sum(losses[len(losses)//2:]) / max(len(losses) - len(losses)//2, 1)
            trend = "decreasing" if second_half < first_half else "increasing"
            print(f"  Loss trend: {trend} "
                  f"(first half avg={first_half:.4f}, second half avg={second_half:.4f})")

    # Policy checks
    policy_ok = True
    if policy_log:
        any_nan_rewards = any(
            any(math.isnan(r) for r in e["rewards"]) for e in policy_log
        )
        if any_nan_rewards:
            print("  Policy: FAIL - NaN in rewards")
            policy_ok = False
        else:
            print("  Policy: No NaN in rewards")

        grad_norms = [e["grad_norm"] for e in policy_log]
        all_finite = all(math.isfinite(g) for g in grad_norms)
        all_nonzero = all(g > 0 for g in grad_norms)
        print(f"  Grad norms: finite={all_finite}, nonzero={all_nonzero}")
        if not (all_finite and all_nonzero):
            policy_ok = False

    # Overall
    overall = btrm_ok and (policy_ok or not policy_log)
    print(f"\n  Overall: {'PASS' if overall else 'FAIL'}")

    client.close()
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())

"""End-to-end training smoke test: BTRM + policy optimization via inference server.

Verifies the full server-based training pipeline:
  1. Server manages FP8 diffusion model + LoRA adapters + BTRM head
  2. BTRM head training happens server-side (train_btrm_step)
  3. Policy gradients accumulate on server (accumulate_policy_gradients)
  4. Optimizer step happens on server (policy_optimizer_step)

No model or optimizer state is held locally -- all computation on the server.
The client is a pure scheduling process.

Prerequisites:
  Running inference server:
    .venv/Scripts/python.exe -m futudiffu.server --port 5555 \
        --fp8-diff <path> --te <path> --vae <path>

Usage:
    .venv/Scripts/python.exe smoke_test_e2e_training.py [--port 5555]
"""

import argparse
import math
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

from futudiffu.client import InferenceClient
from futudiffu.policy_loss import compute_group_advantages
from futudiffu.sampling import build_sigmas, simple_scheduler


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LASER_SHARK_PROMPT = (
    'ahem.\n*ting ting ting ting ting*\n'
    'the query model for this is a LARGE LANGUAGE MODEL, specifically QWEN-3-4B, '
    'a GENERAL PURPOSE SEMANTIC PARSER which is able to WRITE SENTENCES AT A TIME '
    'when they are participating in dialogue. however, in this situation, they are '
    'being used as a hidden state generator to steer an *image generation model*, '
    'z-image.\n\nqwen-3-4b, draw me an "enormous laser shark for the sega saturn".'
)


def main():
    parser = argparse.ArgumentParser(
        description="E2E training smoke test (server-based)")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--btrm-steps", type=int, default=10)
    parser.add_argument("--policy-steps", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=10,
                        help="Euler steps per rollout (fewer = faster)")
    parser.add_argument("--group-size", type=int, default=2,
                        help="Rollouts per policy step (K for advantages)")
    parser.add_argument("--n-sparse", type=int, default=3,
                        help="Sparse steps for gradient computation")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--btrm-lr", type=float, default=1e-3)
    parser.add_argument("--policy-lr", type=float, default=1e-4)
    parser.add_argument("--cfg", type=float, default=4.0)
    args = parser.parse_args()

    endpoint = f"tcp://localhost:{args.port}"
    client = InferenceClient(endpoint)

    print(f"Connected to server at {endpoint}")
    status = client.status()
    print(f"  Phase: {status.get('phase')}")
    print(f"  VRAM: {status.get('vram_allocated_gb', '?')} / "
          f"{status.get('vram_total_gb', '?')} GB")

    # ------------------------------------------------------------------
    # Encode prompts (TE loaded on server, then freed when diff loads)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Encoding prompts")
    print("=" * 60)

    t0 = time.perf_counter()
    pos_cond = client.encode_prompt(LASER_SHARK_PROMPT)
    neg_cond = client.encode_prompt("")
    te_time = time.perf_counter() - t0
    print(f"  pos_cond: {pos_cond.shape}  neg_cond: {neg_cond.shape}")
    print(f"  Encode time: {te_time:.1f}s")

    # Pad to same length for CFG batching
    pos_len = pos_cond.shape[1]
    neg_len = neg_cond.shape[1]
    max_len = max(pos_len, neg_len)
    if pos_len < max_len:
        pos_cond = F.pad(pos_cond, (0, 0, 0, max_len - pos_len))
    if neg_len < max_len:
        neg_cond = F.pad(neg_cond, (0, 0, 0, max_len - neg_len))

    # Build sigma schedule locally (pure math, no GPU)
    sigma_table = build_sigmas(shift=1.0, multiplier=1000.0)
    sigmas = simple_scheduler(sigma_table, args.rollout_steps)
    print(f"  Sigmas ({len(sigmas)}): [{sigmas[0]:.4f} .. {sigmas[-1]:.4f}]")

    # Sparse step indices (evenly spaced)
    sparse_indices = [
        int(i * (args.rollout_steps - 1) / max(1, args.n_sparse - 1))
        for i in range(args.n_sparse)
    ]
    print(f"  Sparse steps: {sparse_indices}")

    # ==================================================================
    # Phase 1: BTRM Training (10 steps)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Phase 1: BTRM Head Training")
    print("=" * 60)

    # Inject rtheta LoRA on last 2 layers for BTRM backbone
    n_rtheta = client.inject_lora(
        "rtheta", rank=args.lora_rank, alpha=args.lora_alpha,
        layer_indices=[28, 29],
    )
    print(f"  Injected rtheta: {n_rtheta} adapters on layers 28-29")

    # Warmup compiled model with new LoRA structure
    print("  Warming up compiled model...", end="", flush=True)
    t0 = time.perf_counter()
    client.warmup(attention_backend="sdpa")
    print(f" {time.perf_counter() - t0:.1f}s")

    # Create BTRM head on server with optimizer
    btrm_meta = client.inject_btrm_head(
        head_names=["bit_quality", "step_quality"],
        logit_cap=10.0,
        lr=args.btrm_lr,
    )
    print(f"\n  BTRM head on server: {btrm_meta['n_params']:,} params")
    print(f"  Training {args.btrm_steps} steps...\n")

    btrm_losses = []
    for step in range(args.btrm_steps):
        t0 = time.perf_counter()

        # Vary sigma across steps for diverse training signal
        sigma_val = 0.3 + 0.5 * (step / max(1, args.btrm_steps - 1))

        # Generate two different noisy latents at the same sigma.
        # "Positive" = first noise realization, "negative" = second.
        noise_pos = torch.randn(1, 16, 104, 160)
        noise_neg = torch.randn(1, 16, 104, 160)
        x_pos = sigma_val * noise_pos
        x_neg = sigma_val * noise_neg

        # Train BTRM with one step on server (backbone + head + loss + backward + step)
        examples = [
            {"latent": x_pos, "sigma": sigma_val, "conditioning": pos_cond[:1],
             "head_idx": 0, "is_positive": True},
            {"latent": x_neg, "sigma": sigma_val, "conditioning": pos_cond[:1],
             "head_idx": 0, "is_positive": False},
        ]
        metrics = client.train_btrm_step(examples, attention_backend="sdpa")

        btrm_losses.append(metrics["loss"])
        dt = time.perf_counter() - t0

        acc_str = ", ".join(f"{k}={v:.2%}" for k, v in metrics.get("per_head_accuracy", {}).items())
        print(f"  btrm {step:2d} | sigma={sigma_val:.3f} | "
              f"loss={metrics['loss']:.4f} | acc=[{acc_str}] | {dt:.1f}s")

    print(f"\n  BTRM losses: first={btrm_losses[0]:.4f}, last={btrm_losses[-1]:.4f}")

    # ==================================================================
    # Phase 2: Policy Optimization (10 steps)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Phase 2: Policy Optimization")
    print("=" * 60)

    # Freeze rtheta and silence it
    client.set_adapter_config("rtheta", frozen=True, scale=0.0)
    print("  rtheta frozen, scale=0")

    # Inject ptheta on all layers with non-zero init_b_std
    n_ptheta = client.inject_lora(
        "ptheta", rank=args.lora_rank, alpha=args.lora_alpha,
        init_b_std=0.01,
    )
    print(f"  Injected ptheta: {n_ptheta} adapters (all layers)")

    # Warmup compiled model with new LoRA structure
    print("  Warming up compiled model...", end="", flush=True)
    t0 = time.perf_counter()
    client.warmup(attention_backend="sdpa")
    print(f" {time.perf_counter() - t0:.1f}s")

    print(f"\n  Running {args.policy_steps} policy iterations "
          f"(K={args.group_size}, {args.rollout_steps} steps, "
          f"{args.n_sparse} sparse)...\n")

    rewards_over_time = []
    grad_norms = []

    for iteration in range(args.policy_steps):
        t0 = time.perf_counter()

        # --- Generate K rollouts and score them ---
        rollout_rewards = []
        rollout_checkpoints: list[dict[int, torch.Tensor]] = []

        for k in range(args.group_size):
            seed = 42000 + iteration * 100 + k

            # Rollout via server (compiled model, inference_mode)
            trajectory = client.sample_trajectory(
                pos_cond, neg_cond, seed=seed,
                n_steps=args.rollout_steps,
                cfg=args.cfg,
                save_steps=sparse_indices,
            )

            # Score with BTRM on server: use a mid-schedule checkpoint
            mid_idx = sparse_indices[len(sparse_indices) // 2]
            step_key = f"step_{mid_idx:02d}"
            if step_key in trajectory:
                x_for_score = trajectory[step_key]
            else:
                x_for_score = trajectory["final"]
            sigma_mid = sigmas[mid_idx] if step_key in trajectory else sigmas[-2]
            sigma_tensor = torch.tensor([float(sigma_mid)])

            scores = client.score_btrm(
                x_for_score, sigma_tensor, pos_cond[:1],
                attention_backend="sdpa",
            )
            reward = scores[0][0]  # first example, head 0
            rollout_rewards.append(reward)

            # Collect checkpoints for gradient computation
            ckpts: dict[int, torch.Tensor] = {}
            for si in sparse_indices:
                sk = f"step_{si:02d}"
                if sk in trajectory:
                    ckpts[si] = trajectory[sk]
            rollout_checkpoints.append(ckpts)

            del trajectory

        # --- Compute advantages ---
        rewards = torch.tensor(rollout_rewards)
        advantages = compute_group_advantages(rewards)

        t_rollout = time.perf_counter() - t0

        # --- Accumulate gradients on server ---
        for k in range(args.group_size):
            if not rollout_checkpoints[k]:
                continue

            client.accumulate_policy_gradients(
                checkpoints=rollout_checkpoints[k],
                sigmas=sigmas,
                conditioning=pos_cond[:1],
                adapter_name="ptheta",
                advantage=advantages[k].item(),
            )

        # --- Optimizer step on server ---
        step_meta = client.policy_optimizer_step(
            adapter_name="ptheta",
            max_grad_norm=1.0,
            lr=args.policy_lr,
        )

        dt = time.perf_counter() - t0
        mean_reward = rewards.mean().item()
        rewards_over_time.append(mean_reward)
        grad_norms.append(step_meta["grad_norm"])

        print(f"  iter {iteration:2d} | "
              f"rewards=[{', '.join(f'{r:+.4f}' for r in rollout_rewards)}] | "
              f"mean={mean_reward:+.4f} | "
              f"grad_norm={step_meta['grad_norm']:.3e} | "
              f"n_params={step_meta['n_params']} | "
              f"rollout={t_rollout:.1f}s | total={dt:.1f}s")

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)

    print(f"\n  BTRM losses ({args.btrm_steps} steps):")
    print(f"    first: {btrm_losses[0]:.4f}")
    print(f"    last:  {btrm_losses[-1]:.4f}")

    print(f"\n  Policy rewards ({args.policy_steps} iterations):")
    print(f"    first: {rewards_over_time[0]:+.4f}")
    print(f"    last:  {rewards_over_time[-1]:+.4f}")

    no_nan = not any(math.isnan(r) for r in rewards_over_time + btrm_losses)
    has_grad_norms = all(math.isfinite(g) and g > 0 for g in grad_norms)

    checks = {
        "no NaN in rewards/losses": no_nan,
        "BTRM loss finite": all(math.isfinite(l) for l in btrm_losses),
        "policy grad norms finite and nonzero": has_grad_norms,
        "all server RPCs succeeded": True,  # would crash if not
    }

    print("\n  Checks:")
    all_pass = True
    for name, ok in checks.items():
        status_str = "PASS" if ok else "FAIL"
        print(f"    [{status_str}] {name}")
        if not ok:
            all_pass = False

    print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}")

    client.close()
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

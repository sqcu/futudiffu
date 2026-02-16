"""Test: Split pi/ref forward in REINFORCE (Refactor 3 validation).

Exercises the refactored compute_reinforce_step which uses two B=1 passes
(ref under no_grad, pi with checkpointing) instead of one concurrent B=2.

Validation criteria:
  1. forward_no_grad matches forward_checkpointed output (detached)
  2. compute_reinforce_step produces nonzero LoRA gradients
  3. Gradients and weights remain finite over multiple iterations
  4. Rollout latents captured at each iteration are numerically sane
  5. (Optional, server) r_theta-active trajectories still converge visually

Uses the S-S-S (Stubbed-Skinny-Shared) model for fast GPU testing (~200MB).

Usage:
    .venv/Scripts/python.exe tests/test_split_piref.py [--iterations 20]
    .venv/Scripts/python.exe tests/test_split_piref.py --server-test --port 5555
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from stubbed_skinny_shared import (
    load_sss_model,
    make_random_conditioning,
    SSS_CAP_FEAT_DIM,
    SSS_DIM,
)

from futudiffu.lora import (
    inject_lora,
    set_lora_scale,
    clear_lora_scale,
    get_lora_params,
    freeze_adapter,
)
from futudiffu.training_utils import (
    forward_checkpointed,
    forward_no_grad,
    compute_reinforce_step,
    prepare_latent_state,
)
from futudiffu.sampling import const_calculate_denoised


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def _check(name: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    suffix = f" ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    return passed


def _latent_stats(x: torch.Tensor) -> dict:
    """Quick latent health stats."""
    xf = x.float()
    return {
        "mean": float(xf.mean()),
        "std": float(xf.std()),
        "min": float(xf.min()),
        "max": float(xf.max()),
        "has_nan": bool(torch.isnan(xf).any()),
        "has_inf": bool(torch.isinf(xf).any()),
    }


# ---------------------------------------------------------------------------
# Test 1: forward_no_grad vs forward_checkpointed agreement
# ---------------------------------------------------------------------------

def test_forward_agreement(model, device, dtype):
    """Verify forward_no_grad output matches forward_checkpointed (detached)."""
    _section("Test 1: forward_no_grad vs forward_checkpointed agreement")

    B, C, H, W = 1, 16, 64, 64
    x = torch.randn(B, C, H, W, device=device, dtype=dtype)
    t = torch.tensor([0.5], device=device, dtype=dtype)
    ctx = make_random_conditioning(B, 20, device=device, dtype=dtype)
    num_tokens = 20

    rope_cache, sigmas, _, _ = prepare_latent_state(
        model, 512, 512, num_tokens, device, dtype,
    )

    # forward_no_grad
    out_ng = forward_no_grad(model, x, t, ctx, num_tokens, rope_cache)

    # forward_checkpointed
    out_ck, last_h = forward_checkpointed(model, x, t, ctx, num_tokens, rope_cache)
    out_ck_det = out_ck.detach()

    # Compare
    diff = (out_ng - out_ck_det).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos_sim = nn.functional.cosine_similarity(
        out_ng.flatten().unsqueeze(0),
        out_ck_det.flatten().unsqueeze(0),
    ).item()

    print(f"  max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, cos_sim={cos_sim:.6f}")

    ok = True
    ok &= _check("outputs finite", torch.isfinite(out_ng).all().item()
                  and torch.isfinite(out_ck_det).all().item())
    ok &= _check("max_diff < 1e-3", max_diff < 1e-3,
                  f"max_diff={max_diff:.2e}")
    ok &= _check("cosine_similarity > 0.999", cos_sim > 0.999,
                  f"cos_sim={cos_sim:.6f}")
    ok &= _check("last_hidden has gradient connection",
                  last_h.requires_grad)

    return ok


# ---------------------------------------------------------------------------
# Test 2: compute_reinforce_step produces nonzero gradients
# ---------------------------------------------------------------------------

def test_reinforce_gradients(model, device, dtype):
    """Verify split pi/ref produces nonzero, finite gradients on LoRA B."""
    _section("Test 2: compute_reinforce_step gradient validity")

    B, C, H, W = 1, 16, 64, 64
    x = torch.randn(B, C, H, W, device=device, dtype=dtype)
    sigma = torch.tensor(0.5, device=device, dtype=dtype)
    ctx = make_random_conditioning(B, 20, device=device, dtype=dtype)
    num_tokens = 20

    rope_cache, _, _, _ = prepare_latent_state(
        model, 512, 512, num_tokens, device, dtype,
    )

    # Zero any existing gradients
    for p in get_lora_params(model, adapter_name="ptheta"):
        if p.grad is not None:
            p.grad = None
        p.requires_grad_(True)

    log_ratio = compute_reinforce_step(
        model, x, sigma, ctx, num_tokens, rope_cache,
        multiplier=1.0, advantage=1.0, adapter_name="ptheta",
    )

    # Check gradients
    lora_params = list(get_lora_params(model, adapter_name="ptheta"))
    n_with_grad = sum(1 for p in lora_params if p.grad is not None)
    n_nonzero_grad = sum(
        1 for p in lora_params
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    n_nan_grad = sum(
        1 for p in lora_params
        if p.grad is not None and torch.isnan(p.grad).any()
    )
    total_grad_norm = torch.sqrt(sum(
        p.grad.norm() ** 2 for p in lora_params
        if p.grad is not None
    )).item()

    print(f"  log_ratio={log_ratio:.6f}")
    print(f"  {n_with_grad}/{len(lora_params)} params have gradients")
    print(f"  {n_nonzero_grad} nonzero, {n_nan_grad} NaN")
    print(f"  total_grad_norm={total_grad_norm:.3e}")

    ok = True
    ok &= _check("log_ratio finite", math.isfinite(log_ratio))
    ok &= _check("some params have grads", n_with_grad > 0,
                  f"{n_with_grad}/{len(lora_params)}")
    ok &= _check("nonzero grads exist", n_nonzero_grad > 0,
                  f"{n_nonzero_grad}")
    ok &= _check("no NaN grads", n_nan_grad == 0)
    ok &= _check("grad_norm finite and > 0",
                  math.isfinite(total_grad_norm) and total_grad_norm > 0,
                  f"grad_norm={total_grad_norm:.3e}")

    # Clean up
    for p in lora_params:
        if p.grad is not None:
            p.grad = None
    clear_lora_scale(model, adapter_name="ptheta")

    return ok


# ---------------------------------------------------------------------------
# Test 3: Multi-iteration stability + rollout capture
# ---------------------------------------------------------------------------

def test_multi_iteration(model, device, dtype, n_iters: int, output_dir: str):
    """Run N REINFORCE iterations, verify numerical stability throughout.

    Captures rollout latents (denoised outputs) at each iteration and
    verifies they remain sane (no NaN, Inf, degenerate statistics).
    """
    _section(f"Test 3: {n_iters}-iteration stability + rollout capture")

    B, C, H, W = 1, 16, 64, 64
    num_tokens = 20

    rope_cache, _, _, _ = prepare_latent_state(
        model, 512, 512, num_tokens, device, dtype,
    )

    # Simple AdamW optimizer on ptheta params
    lora_params = list(get_lora_params(model, adapter_name="ptheta"))
    optimizer = torch.optim.AdamW(lora_params, lr=1e-4)

    # Track metrics
    log_ratios = []
    grad_norms = []
    weight_norms = []
    rollout_stats = []

    os.makedirs(output_dir, exist_ok=True)

    for it in range(n_iters):
        # Fresh random latent and noise each iteration
        x = torch.randn(B, C, H, W, device=device, dtype=dtype)
        sigma = torch.tensor(0.3 + 0.4 * (it / max(1, n_iters - 1)),
                             device=device, dtype=dtype)
        ctx = make_random_conditioning(B, num_tokens, device=device, dtype=dtype)

        # Zero grads
        optimizer.zero_grad()

        # REINFORCE step
        lr = compute_reinforce_step(
            model, x, sigma, ctx, num_tokens, rope_cache,
            multiplier=1.0, advantage=1.0, adapter_name="ptheta",
        )
        log_ratios.append(lr)

        # Grad norm
        gn = torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        grad_norms.append(gn.item() if isinstance(gn, torch.Tensor) else gn)

        # Step
        optimizer.step()

        # Weight norm after step
        wn = torch.sqrt(sum(p.data.norm() ** 2 for p in lora_params)).item()
        weight_norms.append(wn)

        # Capture rollout: run a forward pass with current weights and save
        # the denoised output as a latent sanity check
        with torch.no_grad():
            set_lora_scale(
                model, torch.tensor([1.0], device=device, dtype=dtype),
                adapter_name="ptheta",
            )
            out = forward_no_grad(
                model, x, (sigma * 1.0).unsqueeze(0), ctx, num_tokens, rope_cache,
            )
            denoised = const_calculate_denoised(sigma, out, x)
            stats = _latent_stats(denoised)
            rollout_stats.append(stats)

            # Save latent to disk every 10 iterations
            if (it + 1) % max(1, n_iters // 10) == 0 or it == 0:
                path = os.path.join(output_dir, f"rollout_iter{it:04d}.pt")
                torch.save(denoised.cpu(), path)

        clear_lora_scale(model, adapter_name="ptheta")

        if (it + 1) % max(1, n_iters // 10) == 0 or it == 0:
            print(f"  iter {it:3d} | log_ratio={lr:+.4f} | "
                  f"grad_norm={grad_norms[-1]:.3e} | "
                  f"weight_norm={wn:.3e} | "
                  f"latent_mean={stats['mean']:.3f} std={stats['std']:.3f}")

    # Checks
    ok = True

    # No NaN in log ratios
    ok &= _check("no NaN log_ratios",
                  not any(math.isnan(r) for r in log_ratios))

    # All grad norms finite
    ok &= _check("all grad_norms finite",
                  all(math.isfinite(g) for g in grad_norms))

    # All weight norms finite and nonzero
    ok &= _check("all weight_norms finite and > 0",
                  all(math.isfinite(w) and w > 0 for w in weight_norms),
                  f"range=[{min(weight_norms):.3e}, {max(weight_norms):.3e}]")

    # No NaN/Inf in any rollout latent
    n_nan = sum(1 for s in rollout_stats if s["has_nan"])
    n_inf = sum(1 for s in rollout_stats if s["has_inf"])
    ok &= _check("no NaN in rollout latents", n_nan == 0, f"{n_nan}/{n_iters}")
    ok &= _check("no Inf in rollout latents", n_inf == 0, f"{n_inf}/{n_iters}")

    # Rollout latent stds should be reasonable (not collapsed to zero)
    stds = [s["std"] for s in rollout_stats]
    ok &= _check("rollout latent stds > 0.01",
                  all(s > 0.01 for s in stds),
                  f"min_std={min(stds):.4f}")

    # Weight norms shouldn't explode
    wn_ratio = weight_norms[-1] / max(weight_norms[0], 1e-10)
    ok &= _check("weight norm ratio < 100x",
                  wn_ratio < 100,
                  f"ratio={wn_ratio:.2f}")

    print(f"\n  Saved {len([f for f in os.listdir(output_dir) if f.endswith('.pt')])} "
          f"rollout latents to {output_dir}")

    return ok


# ---------------------------------------------------------------------------
# Test 4 (optional): r_theta-active trajectory convergence via server
# ---------------------------------------------------------------------------

def test_rtheta_trajectories(port: int, n_iters: int, output_dir: str):
    """Generate trajectories with r_theta active after training for N iterations.

    Validates that the diffusion model still produces convergent trajectories
    even with the reward head LoRA distorting its outputs. This is a
    methodological safety guardrail -- r_theta isn't *trained* to produce
    good images, but by virtue of being a low-rank adapter it should not
    destroy convergence for moderate iteration counts.

    Requires a running inference server.
    """
    _section(f"Test 4: r_theta-active trajectory convergence ({n_iters} iters)")

    from futudiffu.client import InferenceClient
    from futudiffu.image_stats import naturalness_report
    from futudiffu.rendering import decode_and_save

    render_dir = os.path.join(output_dir, "rtheta_renders")
    os.makedirs(render_dir, exist_ok=True)

    client = InferenceClient(f"tcp://localhost:{port}")
    status = client.status()
    print(f"  Server status: {status.get('phase', '?')}")

    # Encode prompts
    prompt = ("a highly detailed watercolor painting of a mountain landscape "
              "with a crystal clear lake reflecting the sunset sky")
    pos_cond = client.encode_prompt(prompt)
    neg_cond = client.encode_prompt("")

    # Pad conditioning
    import torch.nn.functional as F
    pl, nl = pos_cond.shape[1], neg_cond.shape[1]
    ml = max(pl, nl)
    if pl < ml:
        pos_cond = F.pad(pos_cond, (0, 0, 0, ml - pl))
    if nl < ml:
        neg_cond = F.pad(neg_cond, (0, 0, 0, ml - nl))

    client.free("te")

    # Inject rtheta LoRA
    n_adapters = client.inject_lora("rtheta", rank=8, alpha=16.0,
                                     layer_indices=[28, 29])
    print(f"  Injected rtheta: {n_adapters} adapters")

    # Inject BTRM head with optimizer
    btrm_meta = client.inject_btrm_head(
        head_names=["scrimble", "scrongle"],
        logit_cap=10.0,
        lr=1e-3,
    )
    print(f"  BTRM head: {btrm_meta['n_params']:,} params")

    # Warmup
    client.warmup(attention_backend="sdpa")

    # Train BTRM for N iterations with random noise pairs
    print(f"\n  Training BTRM for {n_iters} iterations...")
    for it in range(n_iters):
        sigma_val = 0.3 + 0.5 * (it / max(1, n_iters - 1))
        noise_pos = torch.randn(1, 16, 104, 160)
        noise_neg = torch.randn(1, 16, 104, 160)
        examples = [
            {"latent": sigma_val * noise_pos, "sigma": sigma_val,
             "conditioning": pos_cond[:1], "head_idx": 0, "is_positive": True},
            {"latent": sigma_val * noise_neg, "sigma": sigma_val,
             "conditioning": pos_cond[:1], "head_idx": 0, "is_positive": False},
        ]
        metrics = client.train_btrm_step(examples, attention_backend="sdpa")

        if (it + 1) % max(1, n_iters // 5) == 0:
            print(f"    btrm {it+1:3d}/{n_iters} | loss={metrics['loss']:.4f}")

    # Generate trajectories with r_theta active (scale=1.0 is default)
    render_checkpoints = [0, n_iters // 4, n_iters // 2, 3 * n_iters // 4, n_iters]
    render_checkpoints = sorted(set(c for c in render_checkpoints if c <= n_iters))

    print(f"\n  Generating validation trajectories...")

    all_stats = []
    for ci, ckpt_iter in enumerate(render_checkpoints):
        label = f"rtheta_{ckpt_iter:04d}iters"
        seed = 42 + ci

        traj = client.sample_trajectory(
            pos_cond, neg_cond, seed=seed,
            n_steps=30, cfg=4.0,
        )

        final = traj["final"]
        latent_s = _latent_stats(final)

        # VAE decode and save
        render_path = os.path.join(render_dir, f"{label}.png")
        arr = decode_and_save(client, final, render_path)
        img_stats = naturalness_report(arr)

        # Pathology detection
        is_black = all(m < 5.0 for m in img_stats["channel_means"])
        is_white = all(m > 250.0 for m in img_stats["channel_means"])
        is_flat = all(s < 1.0 for s in img_stats["channel_stds"])

        status_str = "OK"
        if latent_s["has_nan"]:
            status_str = "NaN"
        elif latent_s["has_inf"]:
            status_str = "Inf"
        elif is_black:
            status_str = "BLACK"
        elif is_white:
            status_str = "WHITE"
        elif is_flat:
            status_str = "FLAT"

        all_stats.append({
            "label": label,
            "status": status_str,
            "latent": latent_s,
            "img": img_stats,
        })

        print(f"    {label}: {status_str} | "
              f"entropy={img_stats['mean_entropy']:.2f} | "
              f"slope={img_stats['spectral_slope']:.2f} | "
              f"lat_std={latent_s['std']:.3f}")

        # Save latent too
        torch.save(final.cpu(),
                    os.path.join(render_dir, f"{label}_latent.pt"))

        del traj

    # Checks
    ok = True
    n_pathological = sum(1 for s in all_stats if s["status"] != "OK")
    ok &= _check("no pathological renders",
                  n_pathological == 0,
                  f"{n_pathological}/{len(all_stats)} pathological")

    # All entropies should be > 3.0 (not collapsed to flat color)
    entropies = [s["img"]["mean_entropy"] for s in all_stats]
    ok &= _check("all mean_entropy > 3.0",
                  all(e > 3.0 for e in entropies),
                  f"min={min(entropies):.2f}")

    # Spectral slopes should be < -0.5 (natural image structure)
    slopes = [s["img"]["spectral_slope"] for s in all_stats]
    ok &= _check("all spectral_slope < -0.5",
                  all(s < -0.5 for s in slopes),
                  f"max={max(slopes):.2f}")

    print(f"\n  Renders saved to {render_dir}")

    client.close()
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test split pi/ref forward + rollout capture")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Multi-iteration test count (Test 3)")
    parser.add_argument("--output-dir", type=str, default="test_split_piref_output",
                        help="Directory for rollout latent captures")
    parser.add_argument("--server-test", action="store_true",
                        help="Run Test 4 (r_theta convergence via server)")
    parser.add_argument("--port", type=int, default=5555,
                        help="Server port for Test 4")
    parser.add_argument("--rtheta-iters", type=int, default=100,
                        help="BTRM training iterations for Test 4")
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    all_pass = True
    t_start = time.perf_counter()

    # Load S-S-S model (FP8, production path)
    _section("Loading S-S-S model")
    model = load_sss_model(device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} params, dim={model.dim}, layers={len(model.layers)}")

    # Inject ptheta LoRA on all layers (with init_b_std for gradients)
    adapters = inject_lora(model, "ptheta", rank=8, alpha=16.0, init_b_std=0.01)
    n_lora_params = sum(p.numel() for p in get_lora_params(model, "ptheta"))
    print(f"  Injected ptheta: {len(adapters)} adapters, {n_lora_params:,} params")

    # Test 1
    all_pass &= test_forward_agreement(model, device, dtype)

    # Test 2
    all_pass &= test_reinforce_gradients(model, device, dtype)

    # Test 3
    all_pass &= test_multi_iteration(
        model, device, dtype, args.iterations,
        os.path.join(args.output_dir, "rollouts"),
    )

    # Test 4 (optional, server-dependent)
    if args.server_test:
        all_pass &= test_rtheta_trajectories(
            args.port, args.rtheta_iters,
            args.output_dir,
        )

    # Summary
    elapsed = time.perf_counter() - t_start
    _section("Summary")
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

r"""Validation test: packed vs serial differentiable BTRM scoring.

Packs 2-4 images of mixed resolutions (1280x832 + 512x512), scores them
via score_differentiable_packed(), scores them serially via
score_differentiable(), and confirms scores match within tolerance.

Also verifies gradient connectivity: loss.backward() must produce nonzero
gradients on adapter parameters via the packed path.

Uses the Stubbed-Skinny-Shared (S-S-S) model for fast GPU testing (~200 MB
instead of 5.8 GB).

Output: test_packed_scoring_output/ with JSON results and timing data.

Usage:
    PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe \
        F:\dox\repos\ai\futudiffu\tests\test_packed_differentiable_scoring.py
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
sys.path.insert(0, str(REPO_ROOT / "tests"))

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Mixed resolutions to test (heterogeneous packing)
IMAGE_SPECS = [
    {"width": 1280, "height": 832, "label": "full_landscape"},
    {"width": 512, "height": 512, "label": "medium_square"},
    {"width": 256, "height": 256, "label": "small_square"},
    {"width": 640, "height": 384, "label": "medium_landscape"},
]

OUTPUT_DIR = REPO_ROOT / "test_packed_scoring_output"

# Tolerance for packed vs serial divergence.
# FlexAttention vs SDPA produces ~0.0625 max_abs per step.
# After 4 layers (SSS model), this can accumulate.
# For scores (which are scalar projections of hidden states), we allow 0.1.
SCORE_TOLERANCE = 0.1

HEAD_NAMES = ("pinkify", "thisnotthat")
HIDDEN_DIM_SSS = 1024  # S-S-S model hidden dim


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tests": [],
    }

    # ===================================================================
    # Phase 1: Load S-S-S model
    # ===================================================================
    print("=" * 60)
    print("  Phase 1: Loading S-S-S model")
    print("=" * 60)

    from stubbed_skinny_shared import load_sss_model, make_random_conditioning, SSS_DIM
    from src_ii.btrm_model import BTRMCompoundModel

    t0 = time.perf_counter()
    backbone = load_sss_model(device=device)
    load_time = time.perf_counter() - t0
    print(f"  Model loaded in {load_time:.2f}s")
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Create BTRMCompoundModel with S-S-S hidden dim
    btrm = BTRMCompoundModel(
        backbone,
        adapter_name="rtheta",
        adapter_rank=8,
        adapter_alpha=16.0,
        adapter_init_b_std=0.01,
        head_names=HEAD_NAMES,
        hidden_dim=SSS_DIM,
        logit_cap=10.0,
        device=device,
    )
    btrm.train_mode()

    adapter_params = btrm.adapter_params()
    n_adapter = sum(p.numel() for p in adapter_params)
    print(f"  Adapter params: {n_adapter:,}")
    print(f"  Head params: {sum(p.numel() for p in btrm.head_params()):,}")

    # ===================================================================
    # Phase 2: Prepare test images
    # ===================================================================
    print("\n" + "=" * 60)
    print("  Phase 2: Preparing test images")
    print("=" * 60)

    from src_ii.rollout import make_rope_cache

    # Create synthetic latents and conditioning for each image
    test_images = []  # (latent, timestep, conditioning, num_tokens, rope_cache)
    for spec in IMAGE_SPECS:
        w, h = spec["width"], spec["height"]
        latent_h, latent_w = h // 8, w // 8

        # Random latent
        lat = torch.randn(1, 16, latent_h, latent_w, device=device, dtype=dtype)
        # Random timestep
        ts = torch.tensor([0.5], device=device, dtype=dtype)
        # Random conditioning (variable length to test packing)
        num_tokens = 20 + hash(spec["label"]) % 15  # 20-34 tokens
        cond = make_random_conditioning(1, num_tokens, device=device, dtype=dtype)

        # RoPE cache
        rc = make_rope_cache(backbone, latent_h, latent_w, num_tokens, device)

        test_images.append((lat, ts, cond, num_tokens, rc))
        print(f"  {spec['label']}: latent={lat.shape}, tokens={num_tokens}")

    # ===================================================================
    # Phase 3: Serial scoring (reference)
    # ===================================================================
    print("\n" + "=" * 60)
    print("  Phase 3: Serial differentiable scoring")
    print("=" * 60)

    serial_scores = []
    serial_times = []
    for i, (lat, ts, cond, nt, rc) in enumerate(test_images):
        t0 = time.perf_counter()
        score = btrm.score_differentiable(
            lat, ts, cond, nt, rc,
            gradient_checkpointing=True,
        )
        elapsed = time.perf_counter() - t0
        serial_scores.append(score.detach().clone())
        serial_times.append(elapsed)
        print(f"  Image {i} ({IMAGE_SPECS[i]['label']}): "
              f"scores={score.detach().cpu().tolist()} ({elapsed:.3f}s)")

    # Stack serial scores for comparison: (N, n_heads)
    serial_all = torch.cat(serial_scores, dim=0)  # (N, n_heads)
    print(f"  Serial scores shape: {serial_all.shape}")
    print(f"  Serial total time: {sum(serial_times):.3f}s")

    # ===================================================================
    # Phase 4: Packed scoring
    # ===================================================================
    print("\n" + "=" * 60)
    print("  Phase 4: Packed differentiable scoring")
    print("=" * 60)

    # Build images list for packed scoring (no rope_cache needed)
    packed_images = [
        (lat, ts, cond, nt)
        for lat, ts, cond, nt, _rc in test_images
    ]

    t0 = time.perf_counter()
    packed_all = btrm.score_differentiable_packed(
        packed_images,
        gradient_checkpointing=True,
        force_sdpa=False,  # Use SageAttention masked
    )
    packed_time = time.perf_counter() - t0
    print(f"  Packed scores shape: {packed_all.shape}")
    print(f"  Packed total time: {packed_time:.3f}s")
    for i in range(len(IMAGE_SPECS)):
        print(f"  Image {i} ({IMAGE_SPECS[i]['label']}): "
              f"scores={packed_all[i].detach().cpu().tolist()}")

    # ===================================================================
    # Phase 5: Compare serial vs packed
    # ===================================================================
    print("\n" + "=" * 60)
    print("  Phase 5: Serial vs Packed comparison")
    print("=" * 60)

    diff = (packed_all.detach() - serial_all).float()
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()

    per_image_diffs = []
    for i in range(len(IMAGE_SPECS)):
        d = diff[i]
        per_img = {
            "label": IMAGE_SPECS[i]["label"],
            "resolution": f"{IMAGE_SPECS[i]['width']}x{IMAGE_SPECS[i]['height']}",
            "max_abs": d.abs().max().item(),
            "mean_abs": d.abs().mean().item(),
            "serial_scores": serial_all[i].cpu().tolist(),
            "packed_scores": packed_all[i].detach().cpu().tolist(),
        }
        per_image_diffs.append(per_img)
        verdict = "PASS" if per_img["max_abs"] <= SCORE_TOLERANCE else "FAIL"
        print(f"  Image {i} ({per_img['label']}): max_abs={per_img['max_abs']:.6f} "
              f"mean_abs={per_img['mean_abs']:.6f} [{verdict}]")

    overall_verdict = "PASS" if max_abs <= SCORE_TOLERANCE else "FAIL"
    print(f"\n  Overall: max_abs={max_abs:.6f}, mean_abs={mean_abs:.6f} [{overall_verdict}]")

    test_comparison = {
        "name": "serial_vs_packed_scores",
        "overall_max_abs": max_abs,
        "overall_mean_abs": mean_abs,
        "tolerance": SCORE_TOLERANCE,
        "verdict": overall_verdict,
        "serial_time_s": sum(serial_times),
        "packed_time_s": packed_time,
        "speedup": sum(serial_times) / max(packed_time, 1e-6),
        "per_image": per_image_diffs,
    }
    results["tests"].append(test_comparison)

    # ===================================================================
    # Phase 6: Gradient connectivity verification
    # ===================================================================
    print("\n" + "=" * 60)
    print("  Phase 6: Gradient connectivity verification (packed path)")
    print("=" * 60)

    # Zero all gradients
    for p in btrm.all_trainable_params():
        if p.grad is not None:
            p.grad.zero_()

    # Score via packed path
    packed_scores = btrm.score_differentiable_packed(
        packed_images,
        gradient_checkpointing=True,
        force_sdpa=False,
    )

    # Construct a dummy BT loss from two images
    pos_s = packed_scores[0, 0]  # First image, first head
    neg_s = packed_scores[1, 0]  # Second image, first head
    dummy_loss = -F.logsigmoid(pos_s - neg_s)

    print(f"  Dummy BT loss: {dummy_loss.item():.6f}")
    print(f"  Loss has grad_fn: {dummy_loss.grad_fn is not None}")

    dummy_loss.backward()

    # Check adapter gradients
    n_with_grad = 0
    n_nonzero_grad = 0
    max_grad = 0.0
    for p in adapter_params:
        if p.grad is not None:
            n_with_grad += 1
            g_max = p.grad.abs().max().item()
            if g_max > 0:
                n_nonzero_grad += 1
            max_grad = max(max_grad, g_max)

    # Check head gradients
    head_n_with_grad = 0
    head_n_nonzero = 0
    head_max_grad = 0.0
    for p in btrm.head_params():
        if p.grad is not None:
            head_n_with_grad += 1
            g_max = p.grad.abs().max().item()
            if g_max > 0:
                head_n_nonzero += 1
            head_max_grad = max(head_max_grad, g_max)

    adapter_grad_ok = n_nonzero_grad > 0
    head_grad_ok = head_n_nonzero > 0
    grad_verdict = "PASS" if adapter_grad_ok and head_grad_ok else "FAIL"

    print(f"  Adapter: {n_nonzero_grad}/{n_with_grad} params have nonzero grad, "
          f"max={max_grad:.6e} [{'PASS' if adapter_grad_ok else 'FAIL'}]")
    print(f"  Head: {head_n_nonzero}/{head_n_with_grad} params have nonzero grad, "
          f"max={head_max_grad:.6e} [{'PASS' if head_grad_ok else 'FAIL'}]")
    print(f"  Overall gradient connectivity: [{grad_verdict}]")

    test_gradient = {
        "name": "packed_gradient_connectivity",
        "adapter_params_with_grad": n_with_grad,
        "adapter_params_nonzero_grad": n_nonzero_grad,
        "adapter_max_grad": max_grad,
        "head_params_with_grad": head_n_with_grad,
        "head_params_nonzero_grad": head_n_nonzero,
        "head_max_grad": head_max_grad,
        "dummy_loss": dummy_loss.item(),
        "loss_has_grad_fn": dummy_loss.grad_fn is not None,
        "verdict": grad_verdict,
    }
    results["tests"].append(test_gradient)

    # ===================================================================
    # Phase 7: Two-image pack test (minimal packing)
    # ===================================================================
    print("\n" + "=" * 60)
    print("  Phase 7: Two-image pack (1280x832 + 512x512)")
    print("=" * 60)

    two_images = packed_images[:2]
    t0 = time.perf_counter()
    two_scores = btrm.score_differentiable_packed(
        two_images,
        gradient_checkpointing=True,
        force_sdpa=False,
    )
    two_time = time.perf_counter() - t0

    # Compare against serial scores for the same two images
    two_serial = serial_all[:2]
    two_diff = (two_scores.detach() - two_serial).float()
    two_max_abs = two_diff.abs().max().item()
    two_verdict = "PASS" if two_max_abs <= SCORE_TOLERANCE else "FAIL"

    print(f"  Two-image packed scores: {two_scores.detach().cpu().tolist()}")
    print(f"  Two-image serial scores: {two_serial.cpu().tolist()}")
    print(f"  max_abs diff: {two_max_abs:.6f} [{two_verdict}]")
    print(f"  Time: {two_time:.3f}s")

    test_two = {
        "name": "two_image_pack",
        "max_abs": two_max_abs,
        "tolerance": SCORE_TOLERANCE,
        "verdict": two_verdict,
        "time_s": two_time,
    }
    results["tests"].append(test_two)

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    all_verdicts = [t["verdict"] for t in results["tests"]]
    overall = "PASS" if all(v == "PASS" for v in all_verdicts) else "FAIL"
    results["overall_verdict"] = overall

    for t in results["tests"]:
        print(f"  [{t['verdict']}] {t['name']}")
    print(f"\n  Overall: [{overall}]")

    # Save results
    report_path = OUTPUT_DIR / "report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Report saved to: {report_path}")

    # Cleanup
    btrm.cleanup()

    if overall == "FAIL":
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()

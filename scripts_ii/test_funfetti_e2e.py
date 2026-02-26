r"""End-to-end functional test for funfetti batching integration.

This is NOT a unit test. It loads real FP8 weights, runs real GPU forwards
through the 6B backbone, produces real persisted artifacts, and verifies
the FLOPS-budget training path.

What it exercises:
  1. ZImageRLAIF with real FP8 backbone
  2. BTRMPairSampler with FLOPS weights (non-degenerate for multi-res)
  3. train_btrm_differentiable() with FLOPS-budget macrobatch
  4. ValidationMetrics tracker wired into the training loop
  5. All outputs persisted to disk in funfetti_e2e_output/

Supports two dataset modes:
  USE_MULTI_RES=False: V1 btrm_dataset/ (monoresolution 1280x832, FLOPS weights degenerate)
  USE_MULTI_RES=True:  V2 multi_res_trajectories/ (256/512/1024, FLOPS weights non-degenerate)

Execution:
  PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe ^
      F:\dox\repos\ai\futudiffu\scripts_ii\test_funfetti_e2e.py

Expected runtime: ~3-5 minutes on RTX 4090 (3 steps FLOPS-budget).
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch


FP8_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
TE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"
TOKENIZER_PATH = str(REPO_ROOT / "src" / "futudiffu" / "tokenizer")

USE_MULTI_RES = True

V1_DATASET_DIR = REPO_ROOT / "btrm_dataset"
V2_DATASET_DIR = REPO_ROOT / "multi_res_trajectories"
DATASET_DIR = V2_DATASET_DIR if USE_MULTI_RES else V1_DATASET_DIR

OUTPUT_DIR = REPO_ROOT / "funfetti_e2e_output"
PACKED_DIR = OUTPUT_DIR / "packed"

N_STEPS = 3
N_TRAJECTORIES = 8 if USE_MULTI_RES else 4  # more trajectories for multi-res diversity
LR = 3e-4
GRAD_CLIP = 0.1
WARMUP_STEPS = 1
HEAD_NAMES = ("pinkify", "thisnotthat")
PREF_KEYS = ("pinkify_pref", "thisnotthat_pref")


def main():
    wall_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PACKED_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    verdict = "FAIL"  # default to fail; set to PASS only if everything succeeds
    results = {
        "test_name": "funfetti_e2e",
        "start_time": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_steps": N_STEPS,
            "n_trajectories": N_TRAJECTORIES,
            "macrobatch_budget": 3.0,
            "lr": LR,
            "grad_clip": GRAD_CLIP,
        },
    }

    print("=" * 60)
    print(f"  Phase 1: Loading dataset ({'V2 multi-res' if USE_MULTI_RES else 'V1 monores'})")
    print("=" * 60)

    from src_ii.pair_sampler import BTRMPairSampler, build_positions_from_manifest, build_positions_from_v2
    from src_ii.flops_sampling import (
        compute_flops_sampling_weights_from_positions,
        summarize_flops_weights,
    )

    v2_reader = None
    traj_records = None  # V1 records (only set for V1 mode)
    traj_indices = None

    if USE_MULTI_RES:
        from futudiffu.dataset_v2 import DatasetReader
        v2_reader = DatasetReader(str(V2_DATASET_DIR))
        n_available = len(v2_reader)
        print(f"  V2 dataset: {n_available} trajectories")

        if n_available < N_TRAJECTORIES:
            print(f"  ERROR: Need at least {N_TRAJECTORIES} trajectories, have {n_available}")
            results["verdict"] = "FAIL"
            results["error"] = f"insufficient trajectories: {n_available} < {N_TRAJECTORIES}"
            _save_results(results)
            return 1

        traj_indices = [0, 5, 10, 20, 25, 30, 40, 45][:N_TRAJECTORIES]
        print(f"  Selected traj_indices: {traj_indices}")

        positions = build_positions_from_v2(v2_reader, traj_ids=traj_indices)
        print(f"  Positions: {len(positions)} across {len(traj_indices)} trajectories")

        res_dist = {}
        for pos in positions:
            key = f"{pos.width}x{pos.height}"
            res_dist[key] = res_dist.get(key, 0) + 1
        print(f"  Resolution distribution: {res_dist}")

    else:
        manifest_path = DATASET_DIR / "manifest.json"
        if not manifest_path.exists():
            print(f"  ERROR: manifest not found at {manifest_path}")
            results["verdict"] = "FAIL"
            results["error"] = f"manifest not found: {manifest_path}"
            _save_results(results)
            return 1

        with open(str(manifest_path)) as f:
            manifest = json.load(f)

        if isinstance(manifest, list):
            traj_records = manifest
        elif "records" in manifest:
            traj_records = manifest["records"]
        else:
            traj_records = manifest.get("trajectories", [])
        n_available = len(traj_records)
        print(f"  Manifest: {n_available} trajectories")

        if n_available < N_TRAJECTORIES:
            print(f"  ERROR: Need at least {N_TRAJECTORIES} trajectories, have {n_available}")
            results["verdict"] = "FAIL"
            results["error"] = f"insufficient trajectories: {n_available} < {N_TRAJECTORIES}"
            _save_results(results)
            return 1

        traj_indices = list(range(N_TRAJECTORIES))

        positions = build_positions_from_manifest(
            traj_records, traj_indices, DATASET_DIR,
        )
        print(f"  Positions: {len(positions)} across {len(traj_indices)} trajectories")

    if len(positions) < 4:
        print(f"  ERROR: Need at least 4 positions, have {len(positions)}")
        results["verdict"] = "FAIL"
        results["error"] = f"insufficient positions: {len(positions)}"
        _save_results(results)
        return 1

    flops_weights = compute_flops_sampling_weights_from_positions(positions)

    traj_resolutions = {}
    for pos in positions:
        if pos.traj_id not in traj_resolutions:
            traj_resolutions[pos.traj_id] = (pos.width, pos.height)
    flops_summary = summarize_flops_weights(flops_weights, traj_resolutions)
    print(f"  FLOPS weights summary: {json.dumps(flops_summary, indent=2)}")
    results["flops_weights_summary"] = flops_summary
    results["use_multi_res"] = USE_MULTI_RES

    if USE_MULTI_RES:
        unique_weights = set(flops_weights.values())
        n_unique = len(unique_weights)
        print(f"  FLOPS weights non-degeneracy: {n_unique} unique weights "
              f"(degenerate={n_unique <= 1})")
        results["flops_weights_degenerate"] = (n_unique <= 1)

    sampler = BTRMPairSampler(
        positions=positions,
        allow_inter_trajectory=True,
        allow_intra_trajectory=True,
        rng_seed=42,
        flops_weights=flops_weights,
    )
    print(f"  Pair space: {sampler.pair_space_size:,} possible pairs")

    # Sampler will be passed directly to train_btrm_differentiable()
    print(f"  Sampler ready for FLOPS-budget macrobatch sampling")

    print("\n" + "=" * 60)
    print("  Phase 2: Encoding prompts")
    print("=" * 60)

    from futudiffu.text_encoder import create_tokenizer, load_text_encoder, encode_prompt

    tokenizer = create_tokenizer(TOKENIZER_PATH)
    te_model = load_text_encoder(TE_PATH, device=device, dtype=dtype)

    prompt_cache = {}
    for idx in traj_indices:
        if USE_MULTI_RES:
            meta, _ = v2_reader[idx]
            prompt = meta.get("prompt", "")
        else:
            rec = traj_records[idx]
            prompt = rec.get("prompt", "")
        if prompt and prompt not in prompt_cache:
            cond = encode_prompt(te_model, tokenizer, prompt, device=device)
            prompt_cache[prompt] = cond.cpu()
            print(f"    Encoded: '{prompt[:50]}...' -> {cond.shape}")

    del te_model, tokenizer
    torch.cuda.empty_cache()
    print(f"  TE freed. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\n" + "=" * 60)
    print("  Phase 3: Loading backbone, creating compound model")
    print("=" * 60)

    from src_ii.zimage_model import load_zimage_rlaif
    from src_ii.btrm_lifecycle import setup_btrm_training, persist_btrm, load_btrm, get_all_trainable_params
    from src_ii.multi_lora import install_multi_lora, get_adapter_params
    from src_ii.sigma_schedule import build_sigma_schedule, resolution_shift

    raw_model = load_zimage_rlaif(
        FP8_PATH, device=device, dtype=dtype,
        compile_model=False, fuse=True,
    )
    print(f"  VRAM after backbone: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    _v2_meta_cache = {}

    def _get_v2_meta(traj_id):
        if traj_id not in _v2_meta_cache:
            meta, accessor = v2_reader[traj_id]
            _v2_meta_cache[traj_id] = (meta, accessor)
        return _v2_meta_cache[traj_id]

    def load_latent_fn(key):
        """Load latent from dataset.
        key: (traj_id, step_key) tuple.
        """
        traj_id, step_key = key

        if USE_MULTI_RES:
            meta, accessor = _get_v2_meta(traj_id)
            latent = accessor[step_key].to(device=device, dtype=dtype)
            if latent.dim() == 3:
                latent = latent.unsqueeze(0)

            n_steps = meta.get("n_steps", 30)
            w = meta.get("width", 1280)
            h = meta.get("height", 832)
            recorded_shift = meta.get("sampling_shift")
            if recorded_shift is not None:
                shift = float(recorded_shift)
            else:
                shift = resolution_shift(w, h)

            sigmas = build_sigma_schedule(
                n_steps, sampling_shift=shift, device="cpu", dtype=torch.float32,
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

        else:
            rec = traj_records[traj_id]

            traj_dir = V1_DATASET_DIR / "latents" / f"traj_{traj_id:06d}"
            if step_key == "final":
                lat_path = traj_dir / "final.pt"
            else:
                lat_path = traj_dir / f"{step_key}.pt"

            latent = torch.load(str(lat_path), map_location=device, weights_only=True)
            if latent.dtype != dtype:
                latent = latent.to(dtype=dtype)
            if latent.device != device:
                latent = latent.to(device=device)

            n_steps = rec.get("n_steps", 30)
            sigmas = build_sigma_schedule(n_steps, device="cpu", dtype=torch.float32)

            if step_key == "final":
                sigma_val = float(sigmas[-2].item()) if len(sigmas) > 1 else 0.01
            else:
                step_idx = int(step_key.split("_")[1])
                sigma_val = float(sigmas[step_idx].item()) if step_idx < len(sigmas) else 0.01

            timestep = torch.tensor([sigma_val], device=device, dtype=dtype)

            prompt = rec.get("prompt", "")
            cond = prompt_cache.get(prompt)
            if cond is None:
                raise ValueError(f"No cached prompt for traj {traj_id}: '{prompt[:60]}...'")
            cond = cond.to(device=device, dtype=dtype)

            num_tokens = cond.shape[1]

            return latent, timestep, cond, num_tokens

    test_key = (traj_indices[0], positions[0].step_key)
    lat, ts, cond, nt = load_latent_fn(test_key)
    print(f"  Test load: latent={lat.shape}, timestep={ts.shape}, cond={cond.shape}, nt={nt}")
    del lat, ts, cond
    torch.cuda.empty_cache()

    def preference_fn(pair: dict) -> dict:
        """Deterministic preference based on sigma: cleaner image wins."""
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

    print("\n" + "=" * 60)
    print(f"  Phase 4: FLOPS-budget training ({N_STEPS} steps)")
    print("=" * 60)

    optimizer = setup_btrm_training(raw_model)

    from src_ii.btrm_training import train_btrm_differentiable
    from src_ii.training_artifacts import TrainingArtifacts

    t_packed_start = time.perf_counter()

    packed_artifacts = TrainingArtifacts(
        output_dir=str(PACKED_DIR),
        run_name="funfetti_packed",
        head_names=HEAD_NAMES,
    )

    packed_curve = train_btrm_differentiable(
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
        log_interval=1,
        warmup_steps=WARMUP_STEPS,
        lr_schedule="warmup_only",
        macrobatch_budget=3.0,
        output_dir=str(PACKED_DIR),
        artifacts=packed_artifacts,
    )

    packed_time = time.perf_counter() - t_packed_start
    print(f"  Packed training: {packed_time:.1f}s ({packed_time / N_STEPS:.1f}s/step)")

    packed_artifacts.generate_analysis(run_config={
        "mode": "flops_budget", "n_steps": N_STEPS, "macrobatch_budget": 3.0,
        "lr": LR, "grad_clip": GRAD_CLIP,
    })
    print(f"  Packed analysis generated: {PACKED_DIR / 'charts'}")

    packed_grad_stats = _collect_grad_stats(raw_model)
    results["packed"] = {
        "time_s": packed_time,
        "time_per_step_s": packed_time / N_STEPS,
        "curve": packed_curve,
        "grad_stats": packed_grad_stats,
    }

    packed_persist = persist_btrm(raw_model, "rtheta", str(PACKED_DIR))
    results["packed"]["persist"] = packed_persist

    torch.cuda.empty_cache()

    # Serial path removed — FLOPS-budget is the only valid training path.

    print("\n" + "=" * 60)
    print("  Phase 6: Verification")
    print("=" * 60)

    checks = []

    packed_losses = [e["loss"] for e in packed_curve]
    packed_finite = all(
        not (l != l) and l < float('inf') for l in packed_losses  # not NaN, not inf
    )
    checks.append(("packed_losses_finite", packed_finite, packed_losses))
    print(f"  [{'PASS' if packed_finite else 'FAIL'}] Packed losses finite: {packed_losses}")

    packed_nonzero = packed_grad_stats["n_nonzero_grad"] > 0
    checks.append(("packed_adapter_gradients", packed_nonzero, packed_grad_stats))
    print(f"  [{'PASS' if packed_nonzero else 'FAIL'}] Packed adapter grads nonzero: "
          f"{packed_grad_stats['n_nonzero_grad']}/{packed_grad_stats['n_with_grad']}")

    packed_vm_path = PACKED_DIR / "validation_metrics.json"
    packed_vm_exists = packed_vm_path.exists()
    checks.append(("packed_validation_metrics_saved", packed_vm_exists, str(packed_vm_path)))
    print(f"  [{'PASS' if packed_vm_exists else 'FAIL'}] Packed validation_metrics.json exists")

    packed_vm_entries = 0
    packed_vm_data = {}
    if packed_vm_exists:
        with open(str(packed_vm_path)) as f:
            packed_vm_data = json.load(f)
        packed_vm_entries = packed_vm_data.get("n_updates", 0)

    packed_vm_nonempty = packed_vm_entries > 0
    checks.append(("packed_validation_has_entries", packed_vm_nonempty, packed_vm_entries))
    print(f"  [{'PASS' if packed_vm_nonempty else 'FAIL'}] Packed VM entries: {packed_vm_entries}")

    if packed_losses:
        packed_mean = sum(packed_losses) / len(packed_losses)
        comparable = 0.0 < packed_mean < 5.0
        checks.append(("loss_range_reasonable", comparable, {"packed_mean": packed_mean}))
        print(f"  [{'PASS' if comparable else 'FAIL'}] Loss range: "
              f"packed_mean={packed_mean:.4f}")

    packed_charts_exist = (PACKED_DIR / "charts" / "01_loss_curve.png").exists()
    checks.append(("packed_charts_generated", packed_charts_exist, str(PACKED_DIR / "charts")))
    print(f"  [{'PASS' if packed_charts_exist else 'FAIL'}] Packed charts generated")

    packed_analysis_exists = (PACKED_DIR / "training_analysis.md").exists()
    checks.append(("packed_analysis_generated", packed_analysis_exists, str(PACKED_DIR / "training_analysis.md")))
    print(f"  [{'PASS' if packed_analysis_exists else 'FAIL'}] Packed training_analysis.md generated")

    if USE_MULTI_RES:
        flops_nondegen = not results.get("flops_weights_degenerate", True)
        checks.append(("flops_weights_nondegenerate", flops_nondegen, flops_summary))
        print(f"  [{'PASS' if flops_nondegen else 'FAIL'}] FLOPS weights non-degenerate")

    if USE_MULTI_RES and packed_vm_exists:
        packed_vm_res_buckets = sorted(packed_vm_data.get("by_resolution", {}).keys())
        vm_has_res_data = len(packed_vm_res_buckets) > 0
        checks.append(("vm_resolution_buckets_present", vm_has_res_data,
                       {"buckets": packed_vm_res_buckets}))
        print(f"  [{'PASS' if vm_has_res_data else 'FAIL'}] VM resolution buckets: "
              f"{packed_vm_res_buckets}")

        small_bucket_count = 0
        for bk, bv in packed_vm_data.get("by_resolution", {}).items():
            if "0.1" in bk or "< 0.1" in bk:
                small_bucket_count += bv.get("accuracy", {}).get("count", 0)
        total_count = packed_vm_data.get("n_updates", 0)
        if total_count > 0:
            small_frac = small_bucket_count / total_count
            print(f"  Small-image fraction in VM: {small_frac:.1%} "
                  f"({small_bucket_count}/{total_count} -- "
                  f"FLOPS oversampling {'working' if small_frac > 0.5 else 'NOT working'})")

    print("\n" + "=" * 60)
    print("  Phase 7: Rendering exemplar images")
    print("=" * 60)

    VAE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\vae\zimage.safetensors"
    exemplars_rendered = False

    try:
        from src_ii.exemplar_renderer import render_exemplars_from_model

        # Use the packed-trained model (already in memory) for exemplar scoring

        sample_keys = []
        for idx in traj_indices:
            for pos in positions:
                if pos.traj_id == idx:
                    sample_keys.append((pos.traj_id, pos.step_key))
        sample_keys = sample_keys[:min(12, len(sample_keys))]
        print(f"  Scoring {len(sample_keys)} images for exemplar rendering")

        exemplar_manifest = render_exemplars_from_model(
            output_dir=str(OUTPUT_DIR / "exemplars"),
            btrm_model=raw_model,
            load_latent_fn=load_latent_fn,
            sample_keys=sample_keys,
            vae_path=VAE_PATH,
            top_k=2,
            head_names=list(HEAD_NAMES),
            device=device,
            dtype=dtype,
        )
        exemplars_rendered = True
        print(f"  Exemplars rendered: {exemplar_manifest}")

        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  Exemplar rendering failed (non-fatal): {e}")
        import traceback
        traceback.print_exc()

    checks.append(("exemplars_rendered", exemplars_rendered, str(OUTPUT_DIR / "exemplars")))
    print(f"  [{'PASS' if exemplars_rendered else 'FAIL'}] Exemplar images rendered")

    all_pass = all(ok for _, ok, _ in checks)
    verdict = "PASS" if all_pass else "FAIL"

    results["checks"] = [
        {"name": name, "passed": ok, "detail": str(detail)}
        for name, ok, detail in checks
    ]
    results["verdict"] = verdict
    results["wall_time_s"] = time.perf_counter() - wall_start
    results["packed_time_s"] = packed_time
    results["end_time"] = datetime.now(timezone.utc).isoformat()

    _save_results(results)

    print(f"\n{'=' * 60}")
    print(f"  FUNFETTI E2E TEST: {verdict}")
    print(f"  Dataset: {'V2 multi-res' if USE_MULTI_RES else 'V1 monores'}")
    print(f"{'=' * 60}")
    print(f"  Wall time: {results['wall_time_s']:.1f}s")
    print(f"  Packed: {packed_time:.1f}s ({packed_time/N_STEPS:.1f}s/step)")
    print(f"  Packed losses: {packed_losses}")
    print(f"  Packed VM entries: {packed_vm_entries}")
    print(f"  Packed charts: {packed_charts_exist}")
    print(f"  Exemplars rendered: {exemplars_rendered}")
    print(f"  Checks: {sum(1 for _, ok, _ in checks if ok)}/{len(checks)} passed")
    print(f"  Output: {OUTPUT_DIR}")

    if v2_reader is not None:
        v2_reader.close()

    torch.cuda.empty_cache()
    return 0 if verdict == "PASS" else 1



def _collect_grad_stats(model) -> dict:
    """Collect gradient statistics from the model's trainable parameters."""
    n_total = 0
    n_with_grad = 0
    n_nonzero_grad = 0
    max_grad = 0.0
    mean_grad = 0.0

    for p in get_all_trainable_params(model, "rtheta"):
        n_total += 1
        if p.grad is not None:
            n_with_grad += 1
            g_max = p.grad.abs().max().item()
            g_mean = p.grad.abs().mean().item()
            if g_max > 0:
                n_nonzero_grad += 1
            max_grad = max(max_grad, g_max)
            mean_grad += g_mean

    return {
        "n_total": n_total,
        "n_with_grad": n_with_grad,
        "n_nonzero_grad": n_nonzero_grad,
        "max_grad": max_grad,
        "mean_grad": mean_grad / max(n_with_grad, 1),
    }


def _save_results(results: dict):
    """Persist the summary results to disk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "summary.json"
    with open(str(summary_path), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {summary_path}")


if __name__ == "__main__":
    sys.exit(main())

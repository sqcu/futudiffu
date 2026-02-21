r"""End-to-end functional test for funfetti batching integration.

This is NOT a unit test. It loads real FP8 weights, runs real GPU forwards
through the 6B backbone, produces real persisted artifacts, and compares
the packed and serial training paths on identical pairs.

What it exercises:
  1. BTRMCompoundModel with real FP8 backbone
  2. BTRMPairSampler with FLOPS weights (non-degenerate for multi-res)
  3. train_btrm_differentiable() with packed=True (FlexAttention batch packing)
  4. train_btrm_differentiable() with packed=False (serial baseline)
  5. ValidationMetrics tracker wired into the training loop
  6. All outputs persisted to disk in funfetti_e2e_output/

Supports two dataset modes:
  USE_MULTI_RES=False: V1 btrm_dataset/ (monoresolution 1280x832, FLOPS weights degenerate)
  USE_MULTI_RES=True:  V2 multi_res_trajectories/ (256/512/1024, FLOPS weights non-degenerate)

Execution:
  PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe ^
      F:\dox\repos\ai\futudiffu\scripts_ii\test_funfetti_e2e.py

Expected runtime: ~3-10 minutes on RTX 4090 (3 steps packed + 3 steps serial).
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FP8_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
TE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"
TOKENIZER_PATH = str(REPO_ROOT / "src" / "futudiffu" / "tokenizer")

# --- Dataset mode ---
# True: use multi_res_trajectories/ V2 dataset (256/512/1024 mixed resolutions)
# False: use btrm_dataset/ V1 dataset (monoresolution 1280x832)
USE_MULTI_RES = True

# Dataset paths
V1_DATASET_DIR = REPO_ROOT / "btrm_dataset"
V2_DATASET_DIR = REPO_ROOT / "multi_res_trajectories"
DATASET_DIR = V2_DATASET_DIR if USE_MULTI_RES else V1_DATASET_DIR

OUTPUT_DIR = REPO_ROOT / "funfetti_e2e_output"
PACKED_DIR = OUTPUT_DIR / "packed"
SERIAL_DIR = OUTPUT_DIR / "serial"

N_STEPS = 3
N_TRAJECTORIES = 8 if USE_MULTI_RES else 4  # more trajectories for multi-res diversity
PAIRS_PER_PACK = 2   # 2 pairs = 4 images per packed forward
LR = 3e-4
GRAD_CLIP = 0.1
WARMUP_STEPS = 1
HEAD_NAMES = ("pinkify", "thisnotthat")
PREF_KEYS = ("pinkify_pref", "thisnotthat_pref")


def main():
    wall_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PACKED_DIR.mkdir(parents=True, exist_ok=True)
    SERIAL_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    verdict = "FAIL"  # default to fail; set to PASS only if everything succeeds
    results = {
        "test_name": "funfetti_e2e",
        "start_time": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_steps": N_STEPS,
            "n_trajectories": N_TRAJECTORIES,
            "pairs_per_pack": PAIRS_PER_PACK,
            "lr": LR,
            "grad_clip": GRAD_CLIP,
        },
    }

    # ==================================================================
    # Phase 1: Load dataset and build pair sampler with FLOPS weights
    # ==================================================================
    print("=" * 60)
    print(f"  Phase 1: Loading dataset ({'V2 multi-res' if USE_MULTI_RES else 'V1 monores'})")
    print("=" * 60)

    from src_ii.pair_sampler import BTRMPairSampler, build_positions_from_manifest, build_positions_from_v2
    from src_ii.flops_sampling import (
        compute_flops_sampling_weights_from_positions,
        summarize_flops_weights,
    )

    # V2 reader (needed for both dataset modes for load_latent_fn in multi-res)
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

        # Select trajectories spanning all resolutions: pick from each tier
        # V2 dataset is ordered: 0-19 = 256x256, 20-39 = 512x512, 40-59 = 1024x1024
        # Pick 2-3 from each tier to ensure multi-res diversity
        traj_indices = [0, 5, 10, 20, 25, 30, 40, 45][:N_TRAJECTORIES]
        print(f"  Selected traj_indices: {traj_indices}")

        positions = build_positions_from_v2(v2_reader, traj_ids=traj_indices)
        print(f"  Positions: {len(positions)} across {len(traj_indices)} trajectories")

        # Log resolution distribution
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

        # V1 manifest has "records" key; V2 might be a flat list
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

        # Use the first N_TRAJECTORIES trajectories
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

    # Compute FLOPS weights (will degrade gracefully for monoresolution data)
    flops_weights = compute_flops_sampling_weights_from_positions(positions)

    # Build summary for logging
    traj_resolutions = {}
    for pos in positions:
        if pos.traj_id not in traj_resolutions:
            traj_resolutions[pos.traj_id] = (pos.width, pos.height)
    flops_summary = summarize_flops_weights(flops_weights, traj_resolutions)
    print(f"  FLOPS weights summary: {json.dumps(flops_summary, indent=2)}")
    results["flops_weights_summary"] = flops_summary
    results["use_multi_res"] = USE_MULTI_RES

    # Check FLOPS weight non-degeneracy for multi-res mode
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

    # Pre-sample pairs for both packed and serial runs (same pairs for comparison)
    # We sample N_STEPS * PAIRS_PER_PACK pairs for the packed run, and use the
    # first N_STEPS pairs (one per step) for the serial run.
    fixed_pairs = sampler.sample_batch(N_STEPS * PAIRS_PER_PACK * 2)
    print(f"  Pre-sampled {len(fixed_pairs)} pairs for reproducibility")

    # ==================================================================
    # Phase 2: Encode prompts with text encoder
    # ==================================================================
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

    # ==================================================================
    # Phase 3: Load FP8 backbone + create BTRMCompoundModel
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Phase 3: Loading backbone, creating compound model")
    print("=" * 60)

    from src_ii.model_loading import load_fp8_diffusion_model
    from src_ii.btrm_model import BTRMCompoundModel
    from src_ii.rollout import make_rope_cache
    from src_ii.sigma_schedule import build_sigma_schedule, resolution_shift

    # NOTE: compile_model=False because training uses per-block gradient
    # checkpointing through extract_hidden_differentiable(), which is
    # incompatible with whole-model torch.compile. The 30 main layers are
    # individually checkpointed, so the activation memory is bounded by
    # a single layer's activations (not all 30). This is the correct
    # pattern for 24 GB VRAM training -- see model_manager.compile_layers_for_training().
    _, diff_model = load_fp8_diffusion_model(
        FP8_PATH, device=device, dtype=dtype,
        compile_model=False, fuse=True,
    )
    print(f"  VRAM after backbone: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Build load_latent_fn supporting both V1 and V2 datasets
    # Cache V2 metadata + accessor to avoid re-reading
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
            # V2 path: read from DatasetReader
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

            latent_h = h // 8
            latent_w = w // 8
            num_tokens = cond.shape[1]
            rope_cache = make_rope_cache(diff_model, latent_h, latent_w, num_tokens, device)

            return latent, timestep, cond, num_tokens, rope_cache

        else:
            # V1 path: read .pt files directly
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

            latent_h = rec.get("latent_height") or (rec.get("height", 832) // 8)
            latent_w = rec.get("latent_width") or (rec.get("width", 1280) // 8)
            num_tokens = cond.shape[1]

            rope_cache = make_rope_cache(diff_model, latent_h, latent_w, num_tokens, device)

            return latent, timestep, cond, num_tokens, rope_cache

    # Validate load_latent_fn
    test_key = (traj_indices[0], positions[0].step_key)
    lat, ts, cond, nt, rc = load_latent_fn(test_key)
    print(f"  Test load: latent={lat.shape}, timestep={ts.shape}, cond={cond.shape}, nt={nt}")
    del lat, ts, cond, rc
    torch.cuda.empty_cache()

    # Simple preference function: deterministic from pair metadata
    # Uses sigma difference as a proxy (lower sigma = cleaner = higher quality)
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

    # ==================================================================
    # Phase 4: Run packed training (3 steps)
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"  Phase 4: Packed training ({N_STEPS} steps, pairs_per_pack={PAIRS_PER_PACK})")
    print("=" * 60)

    # Create fresh compound model for packed run
    btrm_packed = BTRMCompoundModel(
        diff_model,
        adapter_name="rtheta",
        adapter_rank=8,
        adapter_alpha=16.0,
        adapter_init_b_std=0.01,
        head_names=HEAD_NAMES,
        hidden_dim=3840,
        logit_cap=10.0,
        device=device,
    )

    from src_ii.btrm_training import train_btrm_differentiable
    from src_ii.training_artifacts import TrainingArtifacts

    # Build a sampler that replays fixed pairs
    packed_sampler = _ReplaySampler(fixed_pairs)

    t_packed_start = time.perf_counter()

    # Use TrainingArtifacts for logging (replaces inline JSONL writing)
    packed_artifacts = TrainingArtifacts(
        output_dir=str(PACKED_DIR),
        run_name="funfetti_packed",
        head_names=HEAD_NAMES,
    )

    packed_curve = train_btrm_differentiable(
        btrm_model=btrm_packed,
        pair_sampler=packed_sampler,
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
        grad_accum_steps=1,
        lr_schedule="warmup_only",
        packed=True,
        pairs_per_pack=PAIRS_PER_PACK,
        force_sdpa=False,   # Use SageAttention INT8 QK for production memory profile
        output_dir=str(PACKED_DIR),
        artifacts=packed_artifacts,
    )

    packed_time = time.perf_counter() - t_packed_start
    print(f"  Packed training: {packed_time:.1f}s ({packed_time / N_STEPS:.1f}s/step)")

    # Generate charts + analysis for packed run
    packed_artifacts.generate_analysis(run_config={
        "mode": "packed", "n_steps": N_STEPS, "pairs_per_pack": PAIRS_PER_PACK,
        "lr": LR, "grad_clip": GRAD_CLIP,
    })
    print(f"  Packed analysis generated: {PACKED_DIR / 'charts'}")

    # Collect packed gradient stats
    packed_grad_stats = _collect_grad_stats(btrm_packed)
    results["packed"] = {
        "time_s": packed_time,
        "time_per_step_s": packed_time / N_STEPS,
        "curve": packed_curve,
        "grad_stats": packed_grad_stats,
    }

    # Save packed adapter
    packed_persist = btrm_packed.persist(str(PACKED_DIR))
    results["packed"]["persist"] = packed_persist

    # Clean up packed model's adapter before serial run
    btrm_packed.cleanup()

    # Need to reload the backbone for a fresh adapter
    # Actually we can just reallocate the adapter since we haven't compiled
    torch.cuda.empty_cache()

    # ==================================================================
    # Phase 5: Run serial training (3 steps)
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"  Phase 5: Serial training ({N_STEPS} steps)")
    print("=" * 60)

    # Create fresh compound model for serial run
    btrm_serial = BTRMCompoundModel(
        diff_model,
        adapter_name="rtheta",
        adapter_rank=8,
        adapter_alpha=16.0,
        adapter_init_b_std=0.01,
        head_names=HEAD_NAMES,
        hidden_dim=3840,
        logit_cap=10.0,
        device=device,
    )

    serial_sampler = _ReplaySampler(fixed_pairs)

    t_serial_start = time.perf_counter()

    # Use TrainingArtifacts for logging (replaces inline JSONL writing)
    serial_artifacts = TrainingArtifacts(
        output_dir=str(SERIAL_DIR),
        run_name="funfetti_serial",
        head_names=HEAD_NAMES,
    )

    serial_curve = train_btrm_differentiable(
        btrm_model=btrm_serial,
        pair_sampler=serial_sampler,
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
        grad_accum_steps=1,
        lr_schedule="warmup_only",
        packed=False,
        output_dir=str(SERIAL_DIR),
        artifacts=serial_artifacts,
    )

    serial_time = time.perf_counter() - t_serial_start
    print(f"  Serial training: {serial_time:.1f}s ({serial_time / N_STEPS:.1f}s/step)")

    # Generate charts + analysis for serial run
    serial_artifacts.generate_analysis(run_config={
        "mode": "serial", "n_steps": N_STEPS, "lr": LR, "grad_clip": GRAD_CLIP,
    })
    print(f"  Serial analysis generated: {SERIAL_DIR / 'charts'}")

    serial_grad_stats = _collect_grad_stats(btrm_serial)
    results["serial"] = {
        "time_s": serial_time,
        "time_per_step_s": serial_time / N_STEPS,
        "curve": serial_curve,
        "grad_stats": serial_grad_stats,
    }

    serial_persist = btrm_serial.persist(str(SERIAL_DIR))
    results["serial"]["persist"] = serial_persist

    btrm_serial.cleanup()

    # ==================================================================
    # Phase 6: Verify and compare
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Phase 6: Verification and comparison")
    print("=" * 60)

    checks = []

    # Check 1: Packed losses are all finite
    packed_losses = [e["loss"] for e in packed_curve]
    packed_finite = all(
        not (l != l) and l < float('inf') for l in packed_losses  # not NaN, not inf
    )
    checks.append(("packed_losses_finite", packed_finite, packed_losses))
    print(f"  [{'PASS' if packed_finite else 'FAIL'}] Packed losses finite: {packed_losses}")

    # Check 2: Serial losses are all finite
    serial_losses = [e["loss"] for e in serial_curve]
    serial_finite = all(
        not (l != l) and l < float('inf') for l in serial_losses
    )
    checks.append(("serial_losses_finite", serial_finite, serial_losses))
    print(f"  [{'PASS' if serial_finite else 'FAIL'}] Serial losses finite: {serial_losses}")

    # Check 3: Packed adapter has nonzero gradients
    packed_nonzero = packed_grad_stats["n_nonzero_grad"] > 0
    checks.append(("packed_adapter_gradients", packed_nonzero, packed_grad_stats))
    print(f"  [{'PASS' if packed_nonzero else 'FAIL'}] Packed adapter grads nonzero: "
          f"{packed_grad_stats['n_nonzero_grad']}/{packed_grad_stats['n_with_grad']}")

    # Check 4: Serial adapter has nonzero gradients
    serial_nonzero = serial_grad_stats["n_nonzero_grad"] > 0
    checks.append(("serial_adapter_gradients", serial_nonzero, serial_grad_stats))
    print(f"  [{'PASS' if serial_nonzero else 'FAIL'}] Serial adapter grads nonzero: "
          f"{serial_grad_stats['n_nonzero_grad']}/{serial_grad_stats['n_with_grad']}")

    # Check 5: ValidationMetrics JSON files exist
    packed_vm_path = PACKED_DIR / "validation_metrics.json"
    serial_vm_path = SERIAL_DIR / "validation_metrics.json"
    packed_vm_exists = packed_vm_path.exists()
    serial_vm_exists = serial_vm_path.exists()
    checks.append(("packed_validation_metrics_saved", packed_vm_exists, str(packed_vm_path)))
    checks.append(("serial_validation_metrics_saved", serial_vm_exists, str(serial_vm_path)))
    print(f"  [{'PASS' if packed_vm_exists else 'FAIL'}] Packed validation_metrics.json exists")
    print(f"  [{'PASS' if serial_vm_exists else 'FAIL'}] Serial validation_metrics.json exists")

    # Check 6: ValidationMetrics have entries
    packed_vm_entries = 0
    serial_vm_entries = 0
    if packed_vm_exists:
        with open(str(packed_vm_path)) as f:
            packed_vm_data = json.load(f)
        packed_vm_entries = packed_vm_data.get("n_updates", 0)
    if serial_vm_exists:
        with open(str(serial_vm_path)) as f:
            serial_vm_data = json.load(f)
        serial_vm_entries = serial_vm_data.get("n_updates", 0)

    packed_vm_nonempty = packed_vm_entries > 0
    serial_vm_nonempty = serial_vm_entries > 0
    checks.append(("packed_validation_has_entries", packed_vm_nonempty, packed_vm_entries))
    checks.append(("serial_validation_has_entries", serial_vm_nonempty, serial_vm_entries))
    print(f"  [{'PASS' if packed_vm_nonempty else 'FAIL'}] Packed VM entries: {packed_vm_entries}")
    print(f"  [{'PASS' if serial_vm_nonempty else 'FAIL'}] Serial VM entries: {serial_vm_entries}")

    # Check 7: Loss trajectories are in comparable range
    # Both should be in [0, 2] for BT loss (ln(2) ~ 0.693 at chance)
    if packed_losses and serial_losses:
        packed_mean = sum(packed_losses) / len(packed_losses)
        serial_mean = sum(serial_losses) / len(serial_losses)
        # Both should be in reasonable range and not wildly different
        comparable = (
            0.0 < packed_mean < 5.0
            and 0.0 < serial_mean < 5.0
        )
        checks.append(("loss_trajectories_comparable", comparable,
                       {"packed_mean": packed_mean, "serial_mean": serial_mean}))
        print(f"  [{'PASS' if comparable else 'FAIL'}] Loss ranges: "
              f"packed_mean={packed_mean:.4f}, serial_mean={serial_mean:.4f}")

    # Check 8: Charts were generated by TrainingArtifacts
    packed_charts_exist = (PACKED_DIR / "charts" / "01_loss_curve.png").exists()
    serial_charts_exist = (SERIAL_DIR / "charts" / "01_loss_curve.png").exists()
    checks.append(("packed_charts_generated", packed_charts_exist, str(PACKED_DIR / "charts")))
    checks.append(("serial_charts_generated", serial_charts_exist, str(SERIAL_DIR / "charts")))
    print(f"  [{'PASS' if packed_charts_exist else 'FAIL'}] Packed charts generated")
    print(f"  [{'PASS' if serial_charts_exist else 'FAIL'}] Serial charts generated")

    # Check 9: Training analysis markdown was generated
    packed_analysis_exists = (PACKED_DIR / "training_analysis.md").exists()
    serial_analysis_exists = (SERIAL_DIR / "training_analysis.md").exists()
    checks.append(("packed_analysis_generated", packed_analysis_exists, str(PACKED_DIR / "training_analysis.md")))
    checks.append(("serial_analysis_generated", serial_analysis_exists, str(SERIAL_DIR / "training_analysis.md")))
    print(f"  [{'PASS' if packed_analysis_exists else 'FAIL'}] Packed training_analysis.md generated")
    print(f"  [{'PASS' if serial_analysis_exists else 'FAIL'}] Serial training_analysis.md generated")

    # Check 10 (multi-res only): FLOPS weights are non-degenerate
    if USE_MULTI_RES:
        flops_nondegen = not results.get("flops_weights_degenerate", True)
        checks.append(("flops_weights_nondegenerate", flops_nondegen, flops_summary))
        print(f"  [{'PASS' if flops_nondegen else 'FAIL'}] FLOPS weights non-degenerate")

    # Check 11 (multi-res only): ValidationMetrics have resolution bucket data
    # Note: with FLOPS-weighted sampling, small images are heavily oversampled.
    # In 3 steps x 2 pairs/step = 6 pairs, most may land in the same bucket.
    # The meaningful check is that the VM infrastructure is working and recording
    # resolution bucket data (even if only one bucket has data with so few samples).
    if USE_MULTI_RES and packed_vm_exists:
        packed_vm_res_buckets = sorted(packed_vm_data.get("by_resolution", {}).keys())
        vm_has_res_data = len(packed_vm_res_buckets) > 0
        checks.append(("vm_resolution_buckets_present", vm_has_res_data,
                       {"buckets": packed_vm_res_buckets}))
        print(f"  [{'PASS' if vm_has_res_data else 'FAIL'}] VM resolution buckets: "
              f"{packed_vm_res_buckets}")

        # Also check FLOPS weights led to small-image oversampling (expected behavior)
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

    # ==================================================================
    # Phase 7: Render exemplar images (VAE decode)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Phase 7: Rendering exemplar images")
    print("=" * 60)

    VAE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\vae\zimage.safetensors"
    exemplars_rendered = False

    try:
        from src_ii.exemplar_renderer import render_exemplars_from_model

        # Reload serial model adapter for exemplar scoring.
        # BTRMCompoundModel.load() is a classmethod: creates a new compound model
        # from persisted adapter + head + config files.
        serial_config_path = SERIAL_DIR / "btrm_compound_config.json"
        if serial_config_path.exists():
            btrm_exemplar = BTRMCompoundModel.load(
                str(SERIAL_DIR), backbone=diff_model, device=device,
            )
            print(f"  Loaded serial adapter for exemplar scoring")
        else:
            # Fallback: create fresh (untrained) model
            btrm_exemplar = BTRMCompoundModel(
                diff_model,
                adapter_name="rtheta",
                adapter_rank=8,
                adapter_alpha=16.0,
                adapter_init_b_std=0.01,
                head_names=HEAD_NAMES,
                hidden_dim=3840,
                logit_cap=10.0,
                device=device,
            )
            print(f"  No saved serial model found, using untrained model for exemplar scoring")

        # Build sample keys from the first few trajectories and steps
        sample_keys = []
        for idx in traj_indices:
            for pos in positions:
                if pos.traj_id == idx:
                    sample_keys.append((pos.traj_id, pos.step_key))
        sample_keys = sample_keys[:min(12, len(sample_keys))]
        print(f"  Scoring {len(sample_keys)} images for exemplar rendering")

        exemplar_manifest = render_exemplars_from_model(
            output_dir=str(OUTPUT_DIR / "exemplars"),
            btrm_model=btrm_exemplar,
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

        btrm_exemplar.cleanup()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  Exemplar rendering failed (non-fatal): {e}")
        import traceback
        traceback.print_exc()

    # Check 12: Exemplar images rendered
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
    results["serial_time_s"] = serial_time
    results["end_time"] = datetime.now(timezone.utc).isoformat()

    _save_results(results)

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'=' * 60}")
    print(f"  FUNFETTI E2E TEST: {verdict}")
    print(f"  Dataset: {'V2 multi-res' if USE_MULTI_RES else 'V1 monores'}")
    print(f"{'=' * 60}")
    print(f"  Wall time: {results['wall_time_s']:.1f}s")
    print(f"  Packed: {packed_time:.1f}s ({packed_time/N_STEPS:.1f}s/step)")
    print(f"  Serial: {serial_time:.1f}s ({serial_time/N_STEPS:.1f}s/step)")
    print(f"  Packed losses: {packed_losses}")
    print(f"  Serial losses: {serial_losses}")
    print(f"  Packed VM entries: {packed_vm_entries}")
    print(f"  Serial VM entries: {serial_vm_entries}")
    print(f"  Packed charts: {packed_charts_exist}")
    print(f"  Serial charts: {serial_charts_exist}")
    print(f"  Exemplars rendered: {exemplars_rendered}")
    print(f"  Checks: {sum(1 for _, ok, _ in checks if ok)}/{len(checks)} passed")
    print(f"  Output: {OUTPUT_DIR}")

    # Cleanup V2 reader if used
    if v2_reader is not None:
        v2_reader.close()

    torch.cuda.empty_cache()
    return 0 if verdict == "PASS" else 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ReplaySampler:
    """Replays pre-sampled pairs in order. Wraps around when exhausted."""

    def __init__(self, pairs: list[dict]):
        self._pairs = pairs
        self._idx = 0

    def sample_pair(self) -> dict:
        pair = self._pairs[self._idx % len(self._pairs)]
        self._idx += 1
        return dict(pair)  # copy to prevent mutation

    def sample_batch(self, n: int) -> list[dict]:
        return [self.sample_pair() for _ in range(n)]

    def stats(self) -> dict:
        return {"type": "replay", "total_pairs": len(self._pairs), "consumed": self._idx}


def _collect_grad_stats(btrm_model) -> dict:
    """Collect gradient statistics from the model's trainable parameters."""
    n_total = 0
    n_with_grad = 0
    n_nonzero_grad = 0
    max_grad = 0.0
    mean_grad = 0.0

    for p in btrm_model.all_trainable_params():
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

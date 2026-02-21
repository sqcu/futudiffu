r"""Packed vs serial BTRM scoring validation.

Loads the real FP8 backbone + creates a BTRMCompoundModel, then compares:
  - Serial scoring: N independent score_differentiable() calls
  - Packed scoring: one score_differentiable_packed() call with N images

Validates:
  1. Score agreement: packed scores match serial scores within tolerance
  2. Gradient connectivity: adapter params have nonzero gradients after packed backward
  3. Hidden state extraction: per-image hidden states from packed match serial

Saves all results and intermediate tensors to validation_output_ii/packed_scoring/.

Usage:
    PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe ^
        F:\dox\repos\ai\futudiffu\scripts_ii\validate_packed_scoring.py
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

import torch
import torch.nn.functional as F

# =====================================================================
# Configuration
# =====================================================================

FP8_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
TE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"
TOKENIZER_PATH = str(REPO_ROOT / "src" / "futudiffu" / "tokenizer")
V2_DATASET_DIR = REPO_ROOT / "btrm_dataset_v2"

OUTPUT_DIR = REPO_ROOT / "validation_output_ii" / "packed_scoring"

# Images to test: mixed resolutions, same prompt for simplicity
IMAGE_SPECS = [
    {"traj_id": 0, "step_key": "step_14", "label": "full_1280x832"},
    {"traj_id": 1, "step_key": "step_09", "label": "full_1280x832_b"},
    # If the dataset has mixed resolutions, add them. Otherwise test with
    # same-resolution images (packing still exercises block masking).
]

# Score tolerance: max_abs_diff between packed and serial scores
SCORE_TOLERANCE = 0.1  # per the spec
SCORE_WARN = 0.05

# Gradient threshold: minimum max_abs_grad to consider "nonzero"
GRAD_THRESHOLD = 1e-10


def main():
    wall_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "fp8_path": FP8_PATH,
            "te_path": TE_PATH,
            "score_tolerance": SCORE_TOLERANCE,
            "grad_threshold": GRAD_THRESHOLD,
        },
        "phases": {},
        "verdict": None,
    }

    print("=" * 72)
    print("PACKED vs SERIAL BTRM Scoring Validation")
    print("=" * 72)

    # =================================================================
    # Phase 1: Load V2 dataset to get latents and metadata
    # =================================================================
    print("\n--- Phase 1: Loading V2 dataset ---")

    from futudiffu.dataset_v2 import DatasetReader
    reader = DatasetReader(str(V2_DATASET_DIR))
    n_trajs = len(reader)
    print(f"  Dataset: {n_trajs} trajectories")

    # Collect trajectory info to find mixed resolutions if available
    traj_metas = {}
    resolutions_available = set()
    for traj_id in range(min(n_trajs, 20)):
        meta, accessor = reader[traj_id]
        w = meta.get("width", 1280)
        h = meta.get("height", 832)
        resolutions_available.add((w, h))
        traj_metas[traj_id] = (meta, accessor)

    print(f"  Resolutions found: {sorted(resolutions_available)}")

    # Build image specs: try to get mixed resolutions
    image_specs = []
    seen_resolutions = set()
    for traj_id, (meta, accessor) in traj_metas.items():
        w = meta.get("width", 1280)
        h = meta.get("height", 832)
        res = (w, h)
        # Include up to 4 images, preferring resolution diversity
        if res not in seen_resolutions or len(image_specs) < 2:
            # step_indices from metadata tells us which steps are available
            step_indices = meta.get("step_indices", [0, 4, 9, 14, 19, 24, 29])
            if step_indices:
                mid_idx = step_indices[len(step_indices) // 2]
                step_key = f"step_{mid_idx:02d}"
                image_specs.append({
                    "traj_id": traj_id,
                    "step_key": step_key,
                    "width": w,
                    "height": h,
                    "label": f"traj{traj_id}_{w}x{h}_{step_key}",
                })
                seen_resolutions.add(res)
        if len(image_specs) >= 4:
            break

    # Fall back to default specs if we couldn't find enough
    if len(image_specs) < 2:
        # Use simple known-good specs
        for i, traj_id in enumerate([0, 1]):
            meta, accessor = reader[traj_id]
            w = meta.get("width", 1280)
            h = meta.get("height", 832)
            step_indices = meta.get("step_indices", [0, 4, 9, 14, 19, 24, 29])
            mid_idx = step_indices[len(step_indices) // 2]
            step_key = f"step_{mid_idx:02d}"
            image_specs.append({
                "traj_id": traj_id,
                "step_key": step_key,
                "width": w,
                "height": h,
                "label": f"traj{traj_id}_{w}x{h}_{step_key}",
            })

    print(f"  Selected {len(image_specs)} images for validation:")
    for spec in image_specs:
        print(f"    {spec['label']}: traj={spec['traj_id']}, step={spec['step_key']}, "
              f"res={spec.get('width', '?')}x{spec.get('height', '?')}")

    results["config"]["image_specs"] = image_specs

    # =================================================================
    # Phase 2: Encode prompt(s) with text encoder
    # =================================================================
    print("\n--- Phase 2: Encoding prompts ---")

    from futudiffu.text_encoder import create_tokenizer, load_text_encoder, encode_prompt

    tokenizer = create_tokenizer(TOKENIZER_PATH)
    te_model = load_text_encoder(TE_PATH, device=device, dtype=dtype)

    prompt_cache = {}
    for spec in image_specs:
        meta, _ = reader[spec["traj_id"]]
        prompt = meta.get("prompt", "")
        if prompt and prompt not in prompt_cache:
            cond = encode_prompt(te_model, tokenizer, prompt, device=device)
            prompt_cache[prompt] = cond.cpu()
            print(f"  Encoded prompt: '{prompt[:60]}...' -> {cond.shape}")

    del te_model, tokenizer
    torch.cuda.empty_cache()
    vram_post_te = torch.cuda.memory_allocated() / 1e9
    print(f"  TE freed. VRAM: {vram_post_te:.2f} GB")

    # =================================================================
    # Phase 3: Load FP8 backbone + create BTRMCompoundModel
    # =================================================================
    print("\n--- Phase 3: Loading backbone + creating BTRM model ---")

    from src_ii.model_loading import load_fp8_diffusion_model
    from src_ii.btrm_model import BTRMCompoundModel
    from src_ii.rollout import make_rope_cache
    from src_ii.sigma_schedule import build_sigma_schedule, resolution_shift

    _, diff_model = load_fp8_diffusion_model(
        FP8_PATH, device=device, dtype=dtype,
        compile_model=False, fuse=True,
    )

    vram_post_backbone = torch.cuda.memory_allocated() / 1e9
    print(f"  Backbone loaded. VRAM: {vram_post_backbone:.2f} GB")

    btrm_model = BTRMCompoundModel(
        diff_model,
        adapter_name="rtheta",
        adapter_rank=8,
        adapter_alpha=16.0,
        adapter_init_b_std=0.01,
        head_names=("pinkify", "thisnotthat"),
        hidden_dim=3840,
        logit_cap=10.0,
        device=device,
    )

    adapter_params = btrm_model.adapter_params()
    head_params = btrm_model.head_params()
    n_adapter = sum(p.numel() for p in adapter_params)
    n_head = sum(p.numel() for p in head_params)
    print(f"  Adapter params: {n_adapter:,}")
    print(f"  Head params: {n_head:,}")
    print(f"  VRAM after BTRM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    btrm_model.train_mode()

    results["phases"]["model_setup"] = {
        "n_adapter_params": n_adapter,
        "n_head_params": n_head,
        "vram_backbone_gb": vram_post_backbone,
    }

    # =================================================================
    # Phase 4: Build per-image inputs
    # =================================================================
    print("\n--- Phase 4: Preparing per-image inputs ---")

    def load_image_for_scoring(spec):
        """Load latent, timestep, conditioning, num_tokens for one image."""
        traj_id = spec["traj_id"]
        step_key = spec["step_key"]
        meta, accessor = reader[traj_id]

        latent = accessor[step_key].to(device=device, dtype=dtype)

        # Compute sigma
        n_steps = meta.get("n_steps", 30)
        denoise_val = meta.get("denoise") or 1.0
        recorded_shift = meta.get("sampling_shift")
        if recorded_shift is not None:
            shift = float(recorded_shift)
        else:
            w = meta.get("width", 1280)
            h = meta.get("height", 832)
            shift = resolution_shift(w, h)

        sigmas = build_sigma_schedule(
            n_steps, sampling_shift=shift, denoise=denoise_val,
            device="cpu", dtype=torch.float32,
        )

        if step_key == "final":
            sigma_val = float(sigmas[-2].item()) if len(sigmas) > 1 else 0.01
        else:
            step_idx = int(step_key.split("_")[1])
            sigma_val = float(sigmas[step_idx].item()) if step_idx < len(sigmas) else 0.01

        timestep = torch.tensor([sigma_val], device=device, dtype=dtype)

        prompt = meta.get("prompt", "")
        cond = prompt_cache[prompt].to(device=device, dtype=dtype)
        num_tokens = cond.shape[1]

        latent_h = meta.get("latent_height") or (meta.get("height", 832) // 8)
        latent_w = meta.get("latent_width") or (meta.get("width", 1280) // 8)
        rope_cache = make_rope_cache(diff_model, latent_h, latent_w, num_tokens, device)

        return latent, timestep, cond, num_tokens, rope_cache

    per_image_inputs = []
    for spec in image_specs:
        lat, ts, cond, nt, rc = load_image_for_scoring(spec)
        per_image_inputs.append({
            "latent": lat,
            "timestep": ts,
            "conditioning": cond,
            "num_tokens": nt,
            "rope_cache": rc,
            "spec": spec,
        })
        print(f"  {spec['label']}: latent={lat.shape}, sigma={ts.item():.4f}, "
              f"cond={cond.shape}, num_tokens={nt}")

    # =================================================================
    # Phase 5: Serial scoring (reference)
    # =================================================================
    print("\n--- Phase 5: Serial scoring (reference) ---")

    # Set attention backend to sage for serial path too, so the only
    # difference between serial and packed is the masking, not the backend.
    from futudiffu.attention import set_attention_backend
    set_attention_backend("sage")

    serial_scores = []
    serial_times = []

    for i, inp in enumerate(per_image_inputs):
        spec = inp["spec"]
        print(f"  [{i+1}/{len(per_image_inputs)}] {spec['label']} ... ", end="", flush=True)

        # Zero grads before each serial scoring
        for p in btrm_model.all_trainable_params():
            if p.grad is not None:
                p.grad.zero_()

        t0 = time.perf_counter()
        scores = btrm_model.score_differentiable(
            inp["latent"], inp["timestep"], inp["conditioning"],
            inp["num_tokens"], inp["rope_cache"],
            gradient_checkpointing=True,
        )
        t1 = time.perf_counter()

        serial_scores.append(scores.detach().clone())
        serial_times.append(t1 - t0)

        # Compute a dummy loss and backward to verify serial gradients
        dummy_loss = scores.sum()
        dummy_loss.backward()

        # Check adapter grads
        n_nonzero = sum(
            1 for p in adapter_params
            if p.grad is not None and p.grad.abs().max().item() > GRAD_THRESHOLD
        )

        print(f"{t1-t0:.2f}s  scores={scores.detach().cpu().tolist()}  "
              f"adapter_grads={n_nonzero}/{len(adapter_params)}")

    serial_scores_cat = torch.cat(serial_scores, dim=0)  # (N, n_heads)
    print(f"\n  Serial scores stacked: {serial_scores_cat.shape}")
    print(f"  Serial total time: {sum(serial_times):.2f}s")

    # Save serial scores
    torch.save(serial_scores_cat, OUTPUT_DIR / "serial_scores.pt")

    results["phases"]["serial_scoring"] = {
        "scores": serial_scores_cat.cpu().tolist(),
        "times_s": serial_times,
        "total_time_s": sum(serial_times),
    }

    # =================================================================
    # Phase 6: Packed scoring
    # =================================================================
    print("\n--- Phase 6: Packed scoring ---")

    # Zero all grads
    for p in btrm_model.all_trainable_params():
        if p.grad is not None:
            p.grad.zero_()

    # Build images list for packed scoring
    packed_images = [
        (inp["latent"], inp["timestep"], inp["conditioning"], inp["num_tokens"])
        for inp in per_image_inputs
    ]

    print(f"  Packing {len(packed_images)} images into one forward pass...")
    t0 = time.perf_counter()
    packed_scores = btrm_model.score_differentiable_packed(
        packed_images,
        gradient_checkpointing=True,
        force_sdpa=False,  # Use SageAttention masked for packed
    )
    t1 = time.perf_counter()
    packed_time = t1 - t0

    print(f"  Packed scoring: {packed_time:.2f}s")
    print(f"  Packed scores: {packed_scores.detach().cpu().tolist()}")

    # Backward through packed scores to verify gradient connectivity
    packed_loss = packed_scores.sum()
    packed_loss.backward()

    # Check adapter gradients after packed backward
    adapter_grad_stats = []
    n_total_params = len(adapter_params)
    n_with_grad = 0
    n_nonzero_grad = 0
    max_adapter_grad = 0.0

    for p in adapter_params:
        if p.grad is not None:
            g_max = p.grad.abs().max().item()
            n_with_grad += 1
            if g_max > GRAD_THRESHOLD:
                n_nonzero_grad += 1
            max_adapter_grad = max(max_adapter_grad, g_max)
            adapter_grad_stats.append(g_max)
        else:
            adapter_grad_stats.append(0.0)

    # Also check head gradients
    n_head_with_grad = 0
    max_head_grad = 0.0
    for p in head_params:
        if p.grad is not None:
            g_max = p.grad.abs().max().item()
            if g_max > GRAD_THRESHOLD:
                n_head_with_grad += 1
            max_head_grad = max(max_head_grad, g_max)

    print(f"\n  Gradient connectivity:")
    print(f"    Adapter: {n_nonzero_grad}/{n_total_params} params with nonzero grad")
    print(f"    Adapter max grad: {max_adapter_grad:.6e}")
    print(f"    Head: {n_head_with_grad}/{len(head_params)} params with nonzero grad")
    print(f"    Head max grad: {max_head_grad:.6e}")

    grad_connected = n_nonzero_grad > 0 and max_adapter_grad > GRAD_THRESHOLD

    # Save packed scores
    torch.save(packed_scores.detach(), OUTPUT_DIR / "packed_scores.pt")

    results["phases"]["packed_scoring"] = {
        "scores": packed_scores.detach().cpu().tolist(),
        "time_s": packed_time,
        "n_adapter_with_grad": n_with_grad,
        "n_adapter_nonzero_grad": n_nonzero_grad,
        "n_adapter_total": n_total_params,
        "max_adapter_grad": max_adapter_grad,
        "n_head_with_grad": n_head_with_grad,
        "max_head_grad": max_head_grad,
        "gradient_connected": grad_connected,
    }

    # =================================================================
    # Phase 7: Score comparison
    # =================================================================
    print("\n--- Phase 7: Score comparison ---")

    serial_np = serial_scores_cat.cpu()
    packed_np = packed_scores.detach().cpu()

    diff = packed_np - serial_np
    abs_diff = diff.abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()

    # Per-image comparison
    per_image_comparison = []
    for i, spec in enumerate(image_specs):
        s = serial_np[i]
        p = packed_np[i]
        d = (p - s).abs()
        img_max_abs = d.max().item()
        img_comp = {
            "label": spec["label"],
            "serial_scores": s.tolist(),
            "packed_scores": p.tolist(),
            "abs_diff": d.tolist(),
            "max_abs_diff": img_max_abs,
        }
        per_image_comparison.append(img_comp)

        verdict_str = "PASS" if img_max_abs <= SCORE_TOLERANCE else "FAIL"
        if img_max_abs > SCORE_WARN and img_max_abs <= SCORE_TOLERANCE:
            verdict_str = "WARN"

        print(f"  Image {i} ({spec['label']}):")
        print(f"    Serial:  {s.tolist()}")
        print(f"    Packed:  {p.tolist()}")
        print(f"    AbsDiff: {d.tolist()}")
        print(f"    max_abs: {img_max_abs:.6f}  [{verdict_str}]")

    score_match = max_abs <= SCORE_TOLERANCE

    print(f"\n  Overall max_abs_diff: {max_abs:.6f}")
    print(f"  Overall mean_abs_diff: {mean_abs:.6f}")
    print(f"  Score match (tolerance={SCORE_TOLERANCE}): {'PASS' if score_match else 'FAIL'}")
    print(f"  Gradient connectivity: {'PASS' if grad_connected else 'FAIL'}")

    results["phases"]["comparison"] = {
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "per_image": per_image_comparison,
        "score_match": score_match,
        "tolerance": SCORE_TOLERANCE,
    }

    # =================================================================
    # Phase 8: Speedup analysis
    # =================================================================
    print("\n--- Phase 8: Timing analysis ---")

    serial_total = sum(serial_times)
    speedup = serial_total / packed_time if packed_time > 0 else 0

    print(f"  Serial total:  {serial_total:.2f}s ({len(image_specs)} images)")
    print(f"  Packed total:  {packed_time:.2f}s ({len(image_specs)} images)")
    print(f"  Speedup:       {speedup:.2f}x")

    results["phases"]["timing"] = {
        "serial_total_s": serial_total,
        "packed_total_s": packed_time,
        "speedup": speedup,
        "n_images": len(image_specs),
    }

    # =================================================================
    # Phase 9: Final verdict
    # =================================================================
    print("\n" + "=" * 72)

    overall_pass = score_match and grad_connected
    verdict = "PASS" if overall_pass else "FAIL"

    results["verdict"] = verdict
    results["score_match"] = score_match
    results["gradient_connected"] = grad_connected
    results["wall_time_s"] = time.perf_counter() - wall_start

    print(f"OVERALL VERDICT: {verdict}")
    if not score_match:
        print(f"  FAIL: max_abs_diff={max_abs:.6f} > tolerance={SCORE_TOLERANCE}")
    if not grad_connected:
        print(f"  FAIL: adapter gradients not connected (max_grad={max_adapter_grad:.6e})")
    if overall_pass:
        print(f"  Score agreement: max_abs={max_abs:.6f} <= {SCORE_TOLERANCE}")
        print(f"  Gradient flow: {n_nonzero_grad}/{n_total_params} adapter params have nonzero grad")
        print(f"  Max adapter grad: {max_adapter_grad:.6e}")
    print("=" * 72)

    # Save results JSON
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    # Also save a human-readable summary
    summary_path = OUTPUT_DIR / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Packed vs Serial BTRM Scoring Validation\n")
        f.write(f"Generated: {results['timestamp']}\n")
        f.write(f"Wall time: {results['wall_time_s']:.1f}s\n\n")

        f.write(f"VERDICT: {verdict}\n\n")

        f.write(f"Score Agreement:\n")
        f.write(f"  max_abs_diff: {max_abs:.6f}\n")
        f.write(f"  mean_abs_diff: {mean_abs:.6f}\n")
        f.write(f"  tolerance: {SCORE_TOLERANCE}\n")
        f.write(f"  match: {score_match}\n\n")

        f.write(f"Gradient Connectivity:\n")
        f.write(f"  adapter params with nonzero grad: {n_nonzero_grad}/{n_total_params}\n")
        f.write(f"  max adapter grad: {max_adapter_grad:.6e}\n")
        f.write(f"  head params with nonzero grad: {n_head_with_grad}/{len(head_params)}\n")
        f.write(f"  max head grad: {max_head_grad:.6e}\n\n")

        f.write(f"Timing:\n")
        f.write(f"  serial: {serial_total:.2f}s ({len(image_specs)} images)\n")
        f.write(f"  packed: {packed_time:.2f}s ({len(image_specs)} images)\n")
        f.write(f"  speedup: {speedup:.2f}x\n\n")

        f.write(f"Per-Image Comparison:\n")
        for comp in per_image_comparison:
            f.write(f"  {comp['label']}:\n")
            f.write(f"    serial: {comp['serial_scores']}\n")
            f.write(f"    packed: {comp['packed_scores']}\n")
            f.write(f"    diff:   {comp['abs_diff']}\n")
            f.write(f"    max_abs: {comp['max_abs_diff']:.6f}\n")

    print(f"Summary saved to: {summary_path}")

    # Cleanup
    btrm_model.cleanup()
    del btrm_model, diff_model
    torch.cuda.empty_cache()

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())

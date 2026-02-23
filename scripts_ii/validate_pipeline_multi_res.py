r"""Pipeline validation: mixed-resolution diffusion end-to-end sanity check.

Generates 6+ denoising trajectories at different resolutions and aspect ratios,
VAE-decodes every saved latent to PNG, computes image statistics (PSNR, mean
RGB, variance RGB), and produces diagnostic plots.

This is a sanity check that the whole inference pipeline works end-to-end
across diverse resolutions.

VRAM lifecycle (sequential, never co-resident):
  Phase 1: Load TE (~7.5GB), encode prompts, free TE
  Phase 2: Load FP8 diffusion model (~5.8GB + compile), generate trajectories
  Phase 3: Load VAE (~160MB), decode all saved latents to PNG, free VAE
  Phase 4: Compute statistics + generate plots (CPU only)

Usage:
  set PYTHONUNBUFFERED=1
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\validate_pipeline_multi_res.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch


FP8_WEIGHTS = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
TE_WEIGHTS = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"
VAE_WEIGHTS = r"F:\dox\ai\comfyui\ComfyUI\models\vae\zimage.safetensors"
TOKENIZER_PATH = str(REPO_ROOT / "src" / "futudiffu" / "tokenizer")

OUTPUT_DIR = REPO_ROOT / "pipeline_validation"
RENDER_DIR = OUTPUT_DIR / "renders"
STATS_DIR = OUTPUT_DIR / "stats"

from src_ii.resolution_sampling import (
    MEGAPIXEL_ANCHORS,
    ANCHOR_LABELS,
    sample_random_resolution,
)
import random as _random

RESOLUTION_PLAN: list[tuple[int, int, str]] = []  # (W, H, label) -- built below


def _build_resolution_plan() -> list[tuple[int, int, str]]:
    """Build 8 diverse resolutions across 4+ megapixel tiers, all non-square."""
    rng = _random.Random(42)
    plan = []

    plan.append((96, 64, "256sq_landscape"))    # ~0.006 MP, 3:2 landscape
    plan.append((64, 96, "256sq_portrait"))     # ~0.006 MP, 2:3 portrait

    w, h = sample_random_resolution(147456, rng, aspect_min=0.6, aspect_max=0.8)
    plan.append((w, h, "384sq_portrait"))       # ~0.15 MP portrait

    w, h = sample_random_resolution(262144, rng, aspect_min=1.3, aspect_max=1.8)
    plan.append((w, h, "512sq_landscape"))      # ~0.26 MP landscape

    w, h = sample_random_resolution(495616, rng, aspect_min=0.55, aspect_max=0.75)
    plan.append((w, h, "704sq_portrait"))       # ~0.50 MP portrait

    w, h = sample_random_resolution(1048576, rng, aspect_min=1.4, aspect_max=1.9)
    plan.append((w, h, "1024sq_landscape"))     # ~1.0 MP landscape
    w, h = sample_random_resolution(1048576, rng, aspect_min=0.55, aspect_max=0.75)
    plan.append((w, h, "1024sq_portrait"))      # ~1.0 MP portrait

    plan.append((1280, 832, "reference_1280x832"))

    return plan


RESOLUTION_PLAN = _build_resolution_plan()

N_STEPS = 30
CFG = 4.0
SAVE_STEPS = {0, 5, 10, 15, 20, 25, 29}
BASE_SEED = 500000

from futudiffu.btrm_dataset import PROMPT_TEMPLATES

PROMPTS = PROMPT_TEMPLATES[:8]


def _log(msg: str) -> None:
    print(msg, flush=True)



def phase1_encode_prompts(device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Load TE, encode prompts, free TE."""
    _log("\n" + "=" * 60)
    _log("  PHASE 1: TEXT ENCODER")
    _log("=" * 60)

    from futudiffu.text_encoder import create_tokenizer, load_text_encoder, encode_prompt

    t0 = time.perf_counter()

    tokenizer = create_tokenizer(TOKENIZER_PATH)
    te_model = load_text_encoder(TE_WEIGHTS, device=device, dtype=dtype)
    _log(f"  VRAM after TE load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    prompt_cache: dict[str, torch.Tensor] = {}

    neg_cond = encode_prompt(te_model, tokenizer, "", device=device)
    prompt_cache[""] = neg_cond.cpu()
    _log(f"  neg_cond shape: {neg_cond.shape}")

    for i, prompt in enumerate(PROMPTS):
        cond = encode_prompt(te_model, tokenizer, prompt, device=device)
        prompt_cache[prompt] = cond.cpu()
        _log(f"  prompt {i}: shape={cond.shape}, '{prompt[:60]}...'")

    del te_model, tokenizer
    torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    _log(f"  Phase 1 done: {elapsed:.1f}s, {len(prompt_cache) - 1} prompts encoded")

    return prompt_cache



def phase2_generate(
    prompt_cache: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict]:
    """Generate mixed-resolution trajectories with dense step saving."""
    _log("\n" + "=" * 60)
    _log("  PHASE 2: DIFFUSION MODEL")
    _log("=" * 60)

    from src_ii.zimage_model import load_zimage_rlaif
    from src_ii.rollout import rollout
    from src_ii.sigma_schedule import resolution_shift, build_sigma_schedule_py

    t0 = time.perf_counter()

    diff_model = load_zimage_rlaif(
        FP8_WEIGHTS, device=device, dtype=dtype,
        compile_model=True, fuse=True, use_sage=True,
    )
    _log(f"  VRAM after model load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    trajectories = []
    neg_cond = prompt_cache[""]

    n_traj = len(RESOLUTION_PLAN)
    _log(f"  Generating {n_traj} trajectories")
    _log(f"  Save steps: {sorted(SAVE_STEPS)}")

    timing_data = {}

    for idx, (width, height, label) in enumerate(RESOLUTION_PLAN):
        prompt_idx = idx % len(PROMPTS)
        prompt = PROMPTS[prompt_idx]
        seed = BASE_SEED + idx

        shift = resolution_shift(width, height)
        pixels = width * height

        pos_cond = prompt_cache[prompt].to(device=device, dtype=dtype)
        neg_c = neg_cond.to(device=device, dtype=dtype)

        _log(f"\n  [{idx + 1}/{n_traj}] {width}x{height} ({label})")
        _log(f"    pixels={pixels:,}, shift={shift:.4f}, seed={seed}")

        t_traj = time.perf_counter()

        result_tensors, meta = rollout(
            model=diff_model,
            pos_cond=pos_cond,
            neg_cond=neg_c,
            seed=seed,
            n_steps=N_STEPS,
            cfg=CFG,
            width=width,
            height=height,
            device=device,
            dtype=dtype,
            sampling_shift=shift,
            save_steps=SAVE_STEPS,
        )

        dt = time.perf_counter() - t_traj
        _log(f"    Done in {dt:.1f}s, saved {len(result_tensors)} tensors")

        sigma_schedule = build_sigma_schedule_py(N_STEPS, sampling_shift=shift)

        traj_metadata = {
            "traj_idx": idx,
            "prompt": prompt,
            "prompt_idx": prompt_idx,
            "seed": seed,
            "cfg": CFG,
            "width": width,
            "height": height,
            "n_steps": N_STEPS,
            "sampling_shift": shift,
            "label": label,
            "pixels": pixels,
            "generation_time_s": dt,
            "sigma_schedule": sigma_schedule,
            "saved_steps": sorted(SAVE_STEPS),
        }

        cpu_tensors = {k: v.cpu() for k, v in result_tensors.items()}
        del result_tensors
        torch.cuda.empty_cache()

        trajectories.append({
            "tensors": cpu_tensors,
            "metadata": traj_metadata,
        })

        timing_data[f"{width}x{height}"] = dt

    del diff_model, diff_model
    torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    _log(f"\n  Phase 2 done: {len(trajectories)} trajectories in {elapsed:.1f}s")

    return trajectories



def phase3_vae_decode(
    trajectories: list[dict],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, dict[str, str]]:
    """VAE decode every saved latent, save as PNG. Returns mapping traj_idx -> step_key -> path."""
    _log("\n" + "=" * 60)
    _log("  PHASE 3: VAE DECODE")
    _log("=" * 60)

    from src_ii.vae_utils import load_vae, decode_latent_to_pil

    RENDER_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    vae = load_vae(VAE_WEIGHTS, device=device, dtype=dtype)
    _log(f"  VRAM after VAE load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    render_paths: dict[int, dict[str, str]] = {}
    total_renders = 0

    for traj in trajectories:
        meta = traj["metadata"]
        idx = meta["traj_idx"]
        w = meta["width"]
        h = meta["height"]
        render_paths[idx] = {}

        for step_key in sorted(traj["tensors"].keys()):
            latent = traj["tensors"][step_key]
            if latent.dim() == 3:
                latent = latent.unsqueeze(0)

            if step_key == "final":
                step_num = "final"
            else:
                step_num = step_key  # e.g. "step_00"

            fname = f"traj_{idx:02d}_{step_num}_{w}x{h}.png"
            fpath = RENDER_DIR / fname

            try:
                pil_img = decode_latent_to_pil(vae, latent, device=device, dtype=dtype)
                pil_img.save(str(fpath))
                render_paths[idx][step_key] = str(fpath)
                total_renders += 1

                if total_renders % 10 == 0 or total_renders == 1:
                    _log(f"    [{total_renders}] {fname} ({pil_img.size[0]}x{pil_img.size[1]})")
            except Exception as e:
                _log(f"    RENDER FAILED: {fname}: {e}")

    del vae
    torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    _log(f"  Phase 3 done: {total_renders} renders in {elapsed:.1f}s")

    return render_paths



def _compute_rgb_from_latent_pil(pil_img) -> tuple:
    """Compute mean RGB, variance RGB from PIL image. Returns (mean_r,g,b), (var_r,g,b)."""
    import numpy as np
    arr = np.array(pil_img).astype(float) / 255.0  # (H, W, 3) in [0, 1]
    mean_rgb = arr.mean(axis=(0, 1))  # (3,)
    var_rgb = arr.var(axis=(0, 1))    # (3,)
    return tuple(mean_rgb), tuple(var_rgb)


def _psnr_from_pils(img_a, img_ref) -> float:
    """Compute PSNR between two PIL images (both RGB uint8)."""
    import numpy as np
    a = np.array(img_a).astype(float)
    ref = np.array(img_ref).astype(float)
    if a.shape != ref.shape:
        return 0.0
    mse = ((a - ref) ** 2).mean()
    if mse < 1e-10:
        return 100.0  # Cap at 100 dB for identical images
    return 10.0 * math.log10(255.0 ** 2 / mse)


def phase4_statistics(
    trajectories: list[dict],
    render_paths: dict[int, dict[str, str]],
) -> list[dict]:
    """Compute per-trajectory statistics and generate plots."""
    _log("\n" + "=" * 60)
    _log("  PHASE 4: STATISTICS + PLOTS")
    _log("=" * 60)

    from PIL import Image
    from src_ii.training_artifacts import PILChart

    STATS_DIR.mkdir(parents=True, exist_ok=True)

    all_stats = []

    for traj in trajectories:
        meta = traj["metadata"]
        idx = meta["traj_idx"]
        w = meta["width"]
        h = meta["height"]
        label = meta["label"]

        traj_renders = render_paths.get(idx, {})
        if not traj_renders:
            _log(f"  Skipping traj {idx}: no renders")
            continue

        if "final" not in traj_renders:
            _log(f"  Skipping traj {idx}: no final render")
            continue

        ref_img = Image.open(traj_renders["final"])

        step_keys = sorted(
            [k for k in traj_renders.keys() if k.startswith("step_")],
            key=lambda k: int(k.split("_")[1]),
        )
        if "final" not in step_keys:
            step_keys.append("final")

        step_indices = []
        psnr_values = []
        mean_r, mean_g, mean_b = [], [], []
        var_r, var_g, var_b = [], [], []

        for step_key in step_keys:
            fpath = traj_renders.get(step_key)
            if not fpath or not os.path.exists(fpath):
                continue

            img = Image.open(fpath)

            if step_key == "final":
                step_num = N_STEPS  # After step 29, this is the final clean image
            else:
                step_num = int(step_key.split("_")[1])

            step_indices.append(step_num)

            psnr = _psnr_from_pils(img, ref_img)
            psnr_values.append(psnr)

            mean_rgb, var_rgb = _compute_rgb_from_latent_pil(img)
            mean_r.append(mean_rgb[0])
            mean_g.append(mean_rgb[1])
            mean_b.append(mean_rgb[2])
            var_r.append(var_rgb[0])
            var_g.append(var_rgb[1])
            var_b.append(var_rgb[2])

        chart = PILChart(width=900, height=500)
        chart.set_title(f"Traj {idx}: PSNR vs Step ({w}x{h}, {label})")
        chart.set_labels("Step", "PSNR (dB)")
        chart.add_line(
            [float(s) for s in step_indices],
            psnr_values,
            color="#1155cc", label="PSNR vs final", line_width=2,
        )
        chart.add_scatter(
            [float(s) for s in step_indices],
            psnr_values,
            color="#1155cc", size=4,
        )
        psnr_path = STATS_DIR / f"traj_{idx:02d}_psnr_{w}x{h}.png"
        chart.save(str(psnr_path))

        chart = PILChart(width=900, height=500)
        chart.set_title(f"Traj {idx}: Mean RGB vs Step ({w}x{h}, {label})")
        chart.set_labels("Step", "Mean Channel Value")
        xs_f = [float(s) for s in step_indices]
        chart.add_line(xs_f, mean_r, color="#cc1111", label="Red", line_width=2)
        chart.add_line(xs_f, mean_g, color="#11aa44", label="Green", line_width=2)
        chart.add_line(xs_f, mean_b, color="#1155cc", label="Blue", line_width=2)
        chart.add_scatter(xs_f, mean_r, color="#cc1111", size=3)
        chart.add_scatter(xs_f, mean_g, color="#11aa44", size=3)
        chart.add_scatter(xs_f, mean_b, color="#1155cc", size=3)
        mean_path = STATS_DIR / f"traj_{idx:02d}_mean_rgb_{w}x{h}.png"
        chart.save(str(mean_path))

        chart = PILChart(width=900, height=500)
        chart.set_title(f"Traj {idx}: RGB Variance vs Step ({w}x{h}, {label})")
        chart.set_labels("Step", "Channel Variance")
        chart.add_line(xs_f, var_r, color="#cc1111", label="Red", line_width=2)
        chart.add_line(xs_f, var_g, color="#11aa44", label="Green", line_width=2)
        chart.add_line(xs_f, var_b, color="#1155cc", label="Blue", line_width=2)
        chart.add_scatter(xs_f, var_r, color="#cc1111", size=3)
        chart.add_scatter(xs_f, var_g, color="#11aa44", size=3)
        chart.add_scatter(xs_f, var_b, color="#1155cc", size=3)
        var_path = STATS_DIR / f"traj_{idx:02d}_var_rgb_{w}x{h}.png"
        chart.save(str(var_path))

        traj_stats = {
            "traj_idx": idx,
            "width": w,
            "height": h,
            "label": label,
            "step_indices": step_indices,
            "psnr_values": psnr_values,
            "mean_rgb": {
                "r": mean_r, "g": mean_g, "b": mean_b,
            },
            "var_rgb": {
                "r": var_r, "g": var_g, "b": var_b,
            },
            "psnr_monotonic": all(
                psnr_values[i] <= psnr_values[i + 1]
                for i in range(len(psnr_values) - 1)
            ),
        }
        all_stats.append(traj_stats)

        _log(f"  Traj {idx} ({w}x{h}): PSNR range [{min(psnr_values):.1f}, {max(psnr_values):.1f}] dB, "
             f"monotonic={traj_stats['psnr_monotonic']}")

    stats_path = STATS_DIR / "all_stats.json"
    with open(str(stats_path), "w") as f:
        json.dump(all_stats, f, indent=2)

    _log(f"  Phase 4 done: {len(all_stats)} trajectory stats computed")

    return all_stats



def main() -> int:
    wall_start = time.perf_counter()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    _log("=" * 60)
    _log("  PIPELINE VALIDATION: MIXED-RESOLUTION DIFFUSION")
    _log("=" * 60)
    _log(f"  Output: {OUTPUT_DIR}")
    _log(f"  Device: {device}")
    _log(f"  VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    _log(f"\n  Resolution plan ({len(RESOLUTION_PLAN)} trajectories):")
    for w, h, label in RESOLUTION_PLAN:
        _log(f"    {w}x{h} ({label}, {w * h:,} pixels)")

    prompt_cache = phase1_encode_prompts(device, dtype)

    trajectories = phase2_generate(prompt_cache, device, dtype)

    render_paths = phase3_vae_decode(trajectories, device, dtype)

    all_stats = phase4_statistics(trajectories, render_paths)

    wall_total = time.perf_counter() - wall_start

    report = {
        "script": "validate_pipeline_multi_res.py",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wall_time_s": wall_total,
        "n_trajectories": len(trajectories),
        "n_steps": N_STEPS,
        "cfg": CFG,
        "save_steps": sorted(SAVE_STEPS),
        "resolutions": [],
        "total_renders": sum(len(v) for v in render_paths.values()),
    }

    for traj in trajectories:
        meta = traj["metadata"]
        report["resolutions"].append({
            "traj_idx": meta["traj_idx"],
            "width": meta["width"],
            "height": meta["height"],
            "pixels": meta["pixels"],
            "label": meta["label"],
            "sampling_shift": meta["sampling_shift"],
            "sigma_schedule_first5": meta["sigma_schedule"][:5],
            "sigma_schedule_last5": meta["sigma_schedule"][-5:],
            "generation_time_s": meta["generation_time_s"],
        })

    if all_stats:
        psnr_monotonic_count = sum(1 for s in all_stats if s["psnr_monotonic"])
        report["stats_summary"] = {
            "trajectories_analyzed": len(all_stats),
            "psnr_monotonic": psnr_monotonic_count,
            "psnr_monotonic_pct": psnr_monotonic_count / len(all_stats) * 100,
        }

    report_path = OUTPUT_DIR / "generation_report.json"
    with open(str(report_path), "w") as f:
        json.dump(report, f, indent=2)

    _log(f"\n{'=' * 60}")
    _log(f"  VALIDATION COMPLETE")
    _log(f"{'=' * 60}")
    _log(f"  Wall time: {wall_total:.1f}s ({wall_total / 60:.1f} min)")
    _log(f"  Trajectories: {len(trajectories)}")
    _log(f"  Total renders: {report['total_renders']}")
    _log(f"  Report: {report_path}")

    if all_stats:
        mono = sum(1 for s in all_stats if s["psnr_monotonic"])
        _log(f"  PSNR monotonic: {mono}/{len(all_stats)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

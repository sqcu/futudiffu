r"""3-way comparison of DDGRPO policy interventions across resolutions and seeds.

Resumable 3-pass architecture with incremental persistence:

  Pass 1 (backbone loaded): For each (slug, seed, resolution) triple, sample
    3 configs (ref/v2/v2b) -> write 3 latent .safetensors + 3 step record .json
    sidecars -> free tensors. Resume: skip triples whose 3 latent files exist.

  Pass 2 (VAE loaded): Free backbone, load VAE. For each triple, load 3
    latents from disk -> decode -> write PNGs + false-color diffs -> free.
    After all seeds for a (slug, res): compute mean_diff -> write .npy + .png.
    Resume: skip triples whose 3 PNG files exist.

  Pass 3 (CPU only): Free VAE. Build composite panels, charts, score JSONL,
    diff stats JSON, resolution generalization JSON, manifest JSON from
    on-disk artifacts. Resume: skip composites that already exist.

Adapters coexist on the model:
  0: "rtheta"    -- BTRM reward model (always active)
  1: "policy_v2" -- trained policy from ddgrpo_v2
  2: "policy_v2b"-- trained policy from ddgrpo_v2b

Configs select which policy is active via adapter_scales (dim=3):
  "ref":  [1, 0, 0]  -- BTRM only
  "v2":   [1, 1, 0]  -- BTRM + v2 policy
  "v2b":  [1, 0, 1]  -- BTRM + v2b policy

Output: validation_renders/policy_intervention_v2_v2b/{run_name}/
  {slug}/{WxH}/ nesting per resolution.

Usage:
    .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\validate_policy_intervention.py
    .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\validate_policy_intervention.py --run-name my_run
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from src_ii.infer.text_encoding import encode_prompts
from src_ii.infer.model_setup import load_and_prepare_model
from src_ii.infer.trajectory import sample_trajectory
from src_ii.infer.charts import draw_score_chart
from src_ii.infer.composites import build_comparison_composite, build_grid_composite
from src_ii.infer.diff_analysis import (
    compute_latent_covariance, compute_mean_diff, compute_pixel_diff_stats,
    make_false_color_diff, compute_spatial_autocorrelation,
)
from src_ii.forward_packed import prepare_packed_forward
from src_ii.sigma_schedule import build_sigma_schedule, resolution_shift

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FP8_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
TE_PATH  = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"
VAE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\vae\zimage.safetensors"
BTRM_DIR = REPO_ROOT / "training_output" / "reward_function_run_tnt_v2"
V2_CKPT  = REPO_ROOT / "training_output" / "ddgrpo_v2" / "policy_adapter_final.safetensors"
V2B_CKPT = REPO_ROOT / "training_output" / "ddgrpo_v2b" / "policy_adapter_final.safetensors"

DEVICE = torch.device("cuda")
DTYPE  = torch.bfloat16

PROMPTS = [
    ("portrait",  "a portrait of a woman with flowers in her hair"),
    ("cityscape", "a futuristic city skyline at night"),
    ("garden",    "a japanese garden with cherry blossoms"),
    ("cabin",     "a cozy cabin in the woods during winter"),
]
SEEDS = [42, 137, 256, 999]
RESOLUTIONS = [(320, 320), (512, 512), (832, 640), (1280, 832)]
N_STEPS = 20

ADAPTER_RANK  = 8
ADAPTER_ALPHA = 16.0

CONFIGS = {
    "ref": [1, 0, 0],
    "v2":  [1, 1, 0],
    "v2b": [1, 0, 1],
}
CONFIG_NAMES = list(CONFIGS.keys())  # ["ref", "v2", "v2b"]
CONFIG_COLORS = {
    "ref": (50, 50, 200),
    "v2":  (200, 50, 50),
    "v2b": (50, 180, 50),
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _default_run_name() -> str:
    """First 8 hex chars of sha256(timestamp_ns + PID)."""
    payload = f"{time.time_ns()}-{os.getpid()}"
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# File naming helpers — single source of truth for all 3 passes
# ---------------------------------------------------------------------------

def _latent_path(out_dir: Path, slug: str, res_key: str, seed: int, cfg: str) -> Path:
    return out_dir / slug / res_key / f"seed{seed}_{cfg}.safetensors"


def _steps_path(out_dir: Path, slug: str, res_key: str, seed: int, cfg: str) -> Path:
    return out_dir / slug / res_key / f"seed{seed}_{cfg}_steps.json"


def _png_path(out_dir: Path, slug: str, res_key: str, seed: int, cfg: str) -> Path:
    return out_dir / slug / res_key / f"seed{seed}_{cfg}.png"


def _diff_png_path(out_dir: Path, slug: str, res_key: str, seed: int, cfg: str) -> Path:
    return out_dir / slug / res_key / f"seed{seed}_diff_{cfg}_ref.png"


def _mean_diff_npy_path(out_dir: Path, slug: str, res_key: str, cfg: str) -> Path:
    return out_dir / slug / res_key / f"mean_diff_{cfg}.npy"


def _mean_diff_png_path(out_dir: Path, slug: str, res_key: str, cfg: str) -> Path:
    return out_dir / slug / res_key / f"mean_diff_{cfg}.png"


def _composite_path(out_dir: Path, slug: str, res_key: str) -> Path:
    return out_dir / slug / res_key / "composite.png"


def _cross_res_path(out_dir: Path, slug: str) -> Path:
    return out_dir / slug / "cross_resolution.png"


# ---------------------------------------------------------------------------
# Resume detection helpers
# ---------------------------------------------------------------------------

def _pass1_done(out_dir: Path, slug: str, res_key: str, seed: int) -> bool:
    """True if all 3 config latents + step records exist for this triple."""
    for cfg in CONFIG_NAMES:
        if not _latent_path(out_dir, slug, res_key, seed, cfg).exists():
            return False
        if not _steps_path(out_dir, slug, res_key, seed, cfg).exists():
            return False
    return True


def _pass2_triple_done(out_dir: Path, slug: str, res_key: str, seed: int) -> bool:
    """True if all 3 config PNGs + diff PNGs exist for this triple."""
    for cfg in CONFIG_NAMES:
        if not _png_path(out_dir, slug, res_key, seed, cfg).exists():
            return False
    for cfg in ["v2", "v2b"]:
        if not _diff_png_path(out_dir, slug, res_key, seed, cfg).exists():
            return False
    return True


def _pass2_meandiff_done(out_dir: Path, slug: str, res_key: str) -> bool:
    """True if mean diff npy + png exist for both policy configs."""
    for cfg in ["v2", "v2b"]:
        if not _mean_diff_npy_path(out_dir, slug, res_key, cfg).exists():
            return False
        if not _mean_diff_png_path(out_dir, slug, res_key, cfg).exists():
            return False
    return True


# ---------------------------------------------------------------------------
# Pass 1: backbone sampling with incremental latent persistence
# ---------------------------------------------------------------------------

def pass1_sample(out_dir: Path, conds: dict[str, torch.Tensor]) -> None:
    """Load backbone, sample all trajectories, persist latents + step records."""
    _log("\n" + "=" * 60)
    _log("  PASS 1: BACKBONE SAMPLING")
    _log("=" * 60)

    # Check how many triples need work
    total_triples = len(PROMPTS) * len(SEEDS) * len(RESOLUTIONS)
    skip_count = sum(
        1 for W, H in RESOLUTIONS
        for slug, _ in PROMPTS
        for seed in SEEDS
        if _pass1_done(out_dir, slug, f"{W}x{H}", seed)
    )
    if skip_count == total_triples:
        _log("  All pass 1 triples already complete -- skipping backbone load entirely.")
        return
    _log(f"  {skip_count}/{total_triples} triples cached, {total_triples - skip_count} to compute.")

    # Load model
    adapter_configs = [
        {"name": "rtheta",    "rank": ADAPTER_RANK, "alpha": ADAPTER_ALPHA},
        {"name": "policy_v2", "rank": ADAPTER_RANK, "alpha": ADAPTER_ALPHA},
        {"name": "policy_v2b","rank": ADAPTER_RANK, "alpha": ADAPTER_ALPHA},
    ]
    extra_adapter_loads = [
        {"target_name": "policy_v2",  "path": str(V2_CKPT)},
        {"target_name": "policy_v2b", "path": str(V2B_CKPT)},
    ]
    model, raw_model, head_names = load_and_prepare_model(
        FP8_PATH, adapter_configs, BTRM_DIR, "rtheta",
        extra_adapter_loads=extra_adapter_loads, device=DEVICE, dtype=DTYPE,
    )

    # Persist head_names for pass 3 (CPU-only pass needs this)
    head_names_path = out_dir / "_head_names.json"
    with open(head_names_path, "w") as f:
        json.dump(head_names, f)

    for W, H in RESOLUTIONS:
        res_key = f"{W}x{H}"
        lh, lw = H // 8, W // 8
        alpha = resolution_shift(W, H)
        sigmas = build_sigma_schedule(N_STEPS, sampling_shift=alpha, device=DEVICE, dtype=DTYPE)
        _log(f"\n{'='*60}\n  Resolution {res_key}  alpha={alpha:.4f}\n{'='*60}")

        for slug, _ in PROMPTS:
            cond = conds[slug].to(DEVICE)
            plan = prepare_packed_forward(model, [cond], [(lh, lw)], [cond.shape[1]], DEVICE)

            for seed in SEEDS:
                if _pass1_done(out_dir, slug, res_key, seed):
                    _log(f"  [skip] {slug} seed={seed} {res_key} -- latents exist")
                    continue

                gen = torch.Generator(device=DEVICE).manual_seed(seed)
                x_init = sigmas[0] * torch.randn(1, 16, lh, lw, dtype=DTYPE, device=DEVICE, generator=gen)

                triple_dir = out_dir / slug / res_key
                triple_dir.mkdir(parents=True, exist_ok=True)

                for cfg_name, scales_list in CONFIGS.items():
                    scales = torch.tensor([scales_list], device=DEVICE, dtype=torch.float32)
                    final_lat, step_records = sample_trajectory(model, plan, sigmas, x_init.clone(), scales)

                    # Persist latent as safetensors
                    lat_cpu = final_lat.cpu()
                    save_file({"latent": lat_cpu}, str(_latent_path(out_dir, slug, res_key, seed, cfg_name)))

                    # Persist step records as JSON sidecar
                    enriched_records = []
                    for rec in step_records:
                        enriched_records.append({
                            "slug": slug, "seed": seed, "resolution": res_key,
                            "config": cfg_name, **rec,
                        })
                    with open(_steps_path(out_dir, slug, res_key, seed, cfg_name), "w") as f:
                        json.dump(enriched_records, f)

                    del final_lat, lat_cpu
                    torch.cuda.empty_cache()

                _log(f"  {slug} seed={seed} {res_key} done -- latents + step records persisted")

    # Free backbone
    del model, raw_model
    gc.collect()
    torch.cuda.empty_cache()
    _log(f"\nBackbone freed. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


# ---------------------------------------------------------------------------
# Pass 2: VAE decoding with incremental PNG persistence
# ---------------------------------------------------------------------------

def pass2_decode(out_dir: Path) -> None:
    """Load VAE, decode persisted latents to PNGs + false-color diffs + mean diffs."""
    from src_ii.vae_utils import load_vae, decode_latent_to_pil

    _log("\n" + "=" * 60)
    _log("  PASS 2: VAE DECODING")
    _log("=" * 60)

    # Check how many triples need decoding
    total_triples = len(PROMPTS) * len(SEEDS) * len(RESOLUTIONS)
    skip_decode = sum(
        1 for W, H in RESOLUTIONS
        for slug, _ in PROMPTS
        for seed in SEEDS
        if _pass2_triple_done(out_dir, slug, f"{W}x{H}", seed)
    )
    total_meandiffs = len(PROMPTS) * len(RESOLUTIONS)
    skip_meandiffs = sum(
        1 for W, H in RESOLUTIONS
        for slug, _ in PROMPTS
        if _pass2_meandiff_done(out_dir, slug, f"{W}x{H}")
    )
    if skip_decode == total_triples and skip_meandiffs == total_meandiffs:
        _log("  All pass 2 outputs already exist -- skipping VAE load entirely.")
        return
    _log(f"  Decode: {skip_decode}/{total_triples} triples cached, {total_triples - skip_decode} to compute.")
    _log(f"  Mean diffs: {skip_meandiffs}/{total_meandiffs} cached, {total_meandiffs - skip_meandiffs} to compute.")

    vae = load_vae(VAE_PATH, device=DEVICE, dtype=DTYPE)
    _log(f"VAE loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    for W, H in RESOLUTIONS:
        res_key = f"{W}x{H}"

        for slug, _ in PROMPTS:
            # --- Per-seed decode + false-color diffs ---
            for seed in SEEDS:
                if _pass2_triple_done(out_dir, slug, res_key, seed):
                    _log(f"  [skip] {slug} seed={seed} {res_key} -- PNGs exist")
                    continue

                # Load 3 latents, decode, write PNGs
                images: dict[str, Image.Image] = {}
                for cfg in CONFIG_NAMES:
                    lat_data = load_file(str(_latent_path(out_dir, slug, res_key, seed, cfg)))
                    lat = lat_data["latent"]  # (1, 16, lh, lw) bfloat16
                    img = decode_latent_to_pil(vae, lat, device=DEVICE, dtype=DTYPE)
                    img.save(_png_path(out_dir, slug, res_key, seed, cfg))
                    images[cfg] = img
                    del lat, lat_data
                    torch.cuda.empty_cache()

                # False-color diffs (v2 vs ref, v2b vs ref)
                for cfg in ["v2", "v2b"]:
                    fc = make_false_color_diff(images["ref"], images[cfg])
                    fc.save(_diff_png_path(out_dir, slug, res_key, seed, cfg))

                del images
                _log(f"  {slug} seed={seed} {res_key} decoded + diffs written")

            # --- Mean diffs across seeds (per slug, res) ---
            if _pass2_meandiff_done(out_dir, slug, res_key):
                _log(f"  [skip] {slug} {res_key} mean diffs -- already exist")
                continue

            # Load ref latents for all seeds
            refs = []
            for seed in SEEDS:
                lat_data = load_file(str(_latent_path(out_dir, slug, res_key, seed, "ref")))
                refs.append(lat_data["latent"].squeeze(0))  # (16, lh, lw)
                del lat_data

            for cfg in ["v2", "v2b"]:
                pols = []
                for seed in SEEDS:
                    lat_data = load_file(str(_latent_path(out_dir, slug, res_key, seed, cfg)))
                    pols.append(lat_data["latent"].squeeze(0))  # (16, lh, lw)
                    del lat_data

                md = compute_mean_diff(pols, refs)  # (16, lh, lw)
                np.save(str(_mean_diff_npy_path(out_dir, slug, res_key, cfg)), md.numpy())

                # Decode mean diff to image
                md_img = decode_latent_to_pil(vae, md.unsqueeze(0), device=DEVICE, dtype=DTYPE)
                md_img.save(_mean_diff_png_path(out_dir, slug, res_key, cfg))

                del pols, md, md_img
                torch.cuda.empty_cache()

            del refs
            _log(f"  {slug} {res_key} mean diffs written")

    del vae
    gc.collect()
    torch.cuda.empty_cache()
    _log(f"VAE freed. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


# ---------------------------------------------------------------------------
# Pass 3: CPU-only composite rendering and aggregation
# ---------------------------------------------------------------------------

def pass3_composites(out_dir: Path) -> None:
    """Build composites, charts, and aggregate statistics from on-disk artifacts."""
    _log("\n" + "=" * 60)
    _log("  PASS 3: COMPOSITE RENDERING + AGGREGATION")
    _log("=" * 60)

    # Load head_names persisted by pass 1
    head_names_path = out_dir / "_head_names.json"
    with open(head_names_path) as f:
        head_names: list[str] = json.load(f)

    # Collect all step records from disk
    all_step_records: list[dict] = []
    for W, H in RESOLUTIONS:
        res_key = f"{W}x{H}"
        for slug, _ in PROMPTS:
            for seed in SEEDS:
                for cfg in CONFIG_NAMES:
                    steps_file = _steps_path(out_dir, slug, res_key, seed, cfg)
                    with open(steps_file) as f:
                        records = json.load(f)
                    all_step_records.extend(records)

    diff_stats_all: dict[str, dict] = {}
    res_gen_all: dict[str, dict] = {}

    for slug, _ in PROMPTS:
        diff_stats_all[slug] = {}
        res_gen_all[slug] = {}

        for W, H in RESOLUTIONS:
            res_key = f"{W}x{H}"

            # --- Build composite panel from seed-0 ---
            composite_file = _composite_path(out_dir, slug, res_key)
            seed0 = SEEDS[0]

            if composite_file.exists():
                _log(f"  [skip] {slug} {res_key} composite -- exists")
                # Still need diff_stats for JSON aggregation, so compute those
                ref_img = Image.open(_png_path(out_dir, slug, res_key, seed0, "ref"))
                slug_diff_stats: dict[str, dict] = {}
                for cfg in ["v2", "v2b"]:
                    pol_img = Image.open(_png_path(out_dir, slug, res_key, seed0, cfg))
                    pds = compute_pixel_diff_stats(ref_img, pol_img)
                    diff_arr = np.abs(
                        np.array(ref_img, dtype="float32") / 255.0
                        - np.array(pol_img, dtype="float32") / 255.0
                    )
                    ac = compute_spatial_autocorrelation(diff_arr)
                    md_lat = torch.from_numpy(np.load(str(_mean_diff_npy_path(out_dir, slug, res_key, cfg))))
                    cov = compute_latent_covariance(md_lat)
                    slug_diff_stats[cfg] = {
                        "pixel_diff": pds, "autocorrelation": ac, "covariance": cov,
                    }
                diff_stats_all[slug][res_key] = slug_diff_stats
                res_gen_all[slug][res_key] = {
                    cfg: slug_diff_stats[cfg]["pixel_diff"]["mean"] for cfg in ["v2", "v2b"]
                }
                continue

            # Load seed-0 images from disk for panels
            panels: list[Image.Image] = []
            panel_labels: list[str] = []
            seed0_images: dict[str, Image.Image] = {}
            for cfg in CONFIG_NAMES:
                img = Image.open(_png_path(out_dir, slug, res_key, seed0, cfg))
                seed0_images[cfg] = img
                panels.append(img)
                panel_labels.append(cfg)

            ref_img = seed0_images["ref"]

            # Add false-color diff panels
            for cfg in ["v2", "v2b"]:
                fc = make_false_color_diff(ref_img, seed0_images[cfg])
                panels.append(fc)
                panel_labels.append(f"diff_{cfg}")

            # Score charts per head
            step_recs_here = [r for r in all_step_records
                              if r["slug"] == slug and r["resolution"] == res_key and r["seed"] == seed0]
            charts = []
            for head_idx, head_name in enumerate(head_names):
                named_series = {}
                for cfg_name, color in CONFIG_COLORS.items():
                    recs = [r for r in step_recs_here if r["config"] == cfg_name]
                    recs.sort(key=lambda r: r["step"])
                    named_series[cfg_name] = {
                        "values": [r["scores"][head_idx] for r in recs],
                        "color": color,
                    }
                logsnrs = [r["logsnr"] for r in recs]
                charts.append(draw_score_chart(logsnrs, named_series, head_name))

            # Stats lines + diff stats
            stats_lines = []
            slug_diff_stats = {}
            for cfg in ["v2", "v2b"]:
                pol_img = seed0_images[cfg]
                pds = compute_pixel_diff_stats(ref_img, pol_img)
                diff_arr = np.abs(
                    np.array(ref_img, dtype="float32") / 255.0
                    - np.array(pol_img, dtype="float32") / 255.0
                )
                ac = compute_spatial_autocorrelation(diff_arr)
                md_lat = torch.from_numpy(np.load(str(_mean_diff_npy_path(out_dir, slug, res_key, cfg))))
                cov = compute_latent_covariance(md_lat)
                stats_lines.append(
                    f"{cfg}: pixel_diff_mean={pds['mean']:.4f}  "
                    f"autocorr={ac['max_autocorrelation']:.4f} ({ac['verdict']})  "
                    f"eff_rank={cov['effective_rank']:.2f}"
                )
                slug_diff_stats[cfg] = {
                    "pixel_diff": pds, "autocorrelation": ac, "covariance": cov,
                }
            diff_stats_all[slug][res_key] = slug_diff_stats

            # Build composite
            composite = build_comparison_composite(
                panels, panel_labels, charts,
                title=f"{slug} {res_key}",
                stats_lines=stats_lines,
                target_row_height=256,
            )
            composite.save(composite_file)
            _log(f"  {slug} {res_key} composite saved")

            # Resolution generalization entry
            res_gen_all[slug][res_key] = {
                cfg: slug_diff_stats[cfg]["pixel_diff"]["mean"] for cfg in ["v2", "v2b"]
            }

        # --- Cross-resolution grid composite (seed-0 images) ---
        cross_res_file = _cross_res_path(out_dir, slug)
        if cross_res_file.exists():
            _log(f"  [skip] {slug} cross-resolution composite -- exists")
        else:
            col_labels = [f"{W}x{H}" for W, H in RESOLUTIONS]
            row_labels = list(CONFIGS.keys())
            grid: list[list[Image.Image]] = []  # grid[col][row]
            for W, H in RESOLUTIONS:
                res_key = f"{W}x{H}"
                col = [Image.open(_png_path(out_dir, slug, res_key, SEEDS[0], cfg))
                       for cfg in CONFIG_NAMES]
                grid.append(col)
            cross_res = build_grid_composite(
                grid, col_labels, row_labels,
                title=f"{slug}: cross-resolution policy comparison",
                target_row_height=256,
            )
            cross_res.save(cross_res_file)
            _log(f"  {slug}: cross-resolution composite saved")

    # ---- Write JSONL + JSON outputs ----
    scores_path = out_dir / "scores_per_step.jsonl"
    with open(scores_path, "w") as f:
        for rec in all_step_records:
            f.write(json.dumps(rec) + "\n")
    _log(f"scores_per_step.jsonl: {len(all_step_records)} entries")

    with open(out_dir / "diff_stats.json", "w") as f:
        json.dump(diff_stats_all, f, indent=2)

    with open(out_dir / "resolution_generalization.json", "w") as f:
        json.dump(res_gen_all, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Policy intervention validation (resumable)")
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Run name for output directory. Default: 8-char hash from timestamp+PID.",
    )
    args = parser.parse_args()

    run_name = args.run_name or _default_run_name()
    out_dir = REPO_ROOT / "validation_renders" / "policy_intervention_v2_v2b" / run_name

    _log(f"Policy intervention comparison -- {datetime.now(timezone.utc).isoformat()}")
    _log(f"Run name: {run_name}")
    _log(f"Output dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    t_total = time.perf_counter()

    # ---- Text encoding (stateless, always re-run -- fast and no persistence needed) ----
    conds = encode_prompts(PROMPTS, TE_PATH, DEVICE, DTYPE)

    # ---- Pass 1: backbone sampling ----
    pass1_sample(out_dir, conds)

    # ---- Pass 2: VAE decoding ----
    pass2_decode(out_dir)

    # ---- Pass 3: composites + aggregation ----
    pass3_composites(out_dir)

    # ---- Write manifest (always overwritten to capture final timing) ----
    # Load head_names for manifest
    with open(out_dir / "_head_names.json") as f:
        head_names = json.load(f)

    total_elapsed = time.perf_counter() - t_total
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "prompts": {s: p for s, p in PROMPTS},
        "seeds": SEEDS,
        "resolutions": [f"{W}x{H}" for W, H in RESOLUTIONS],
        "configs": {k: v for k, v in CONFIGS.items()},
        "n_steps": N_STEPS,
        "head_names": head_names,
        "btrm_dir": str(BTRM_DIR),
        "v2_ckpt": str(V2_CKPT),
        "v2b_ckpt": str(V2B_CKPT),
        "total_trajectories": len(PROMPTS) * len(SEEDS) * len(RESOLUTIONS) * len(CONFIGS),
        "total_elapsed_s": total_elapsed,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    _log(f"\n{'='*60}")
    _log(f"  DONE -- {total_elapsed:.1f}s total")
    _log(f"  {len(PROMPTS)*len(SEEDS)*len(RESOLUTIONS)*len(CONFIGS)} trajectories")
    _log(f"  Output: {out_dir}")
    _log(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

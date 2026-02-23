r"""Generate multi-resolution trajectory data for funfetti batching exercise.

Produces trajectories across all 6 megapixel anchor tiers with randomly
sampled aspect ratios, complementing the existing 96% 1280x832 V2 dataset.
This enables the funfetti training path to exercise non-degenerate
FLOPS-weighted sampling, bin packing, and continuous resolution coverage.

Resolution sampling:
  - 6 megapixel anchors: 256^2, 320^2, 384^2, 512^2, 704^2, 1024^2
  - Aspect ratios sampled log-uniformly from [0.5, 2.0]
  - W, H quantized to 32px alignment (>= 64px minimum)
  - Each trajectory gets a unique (W, H) from sample_random_resolution()

Direct model loading: no ZMQ server dependency. Loads TE, diffusion model,
and VAE directly on the RTX 4090.

VRAM lifecycle:
  Phase 1: Load TE (~7.5GB), encode prompts, free TE
  Phase 2: Load FP8 diffusion model (~5.8GB), generate + persist each
           trajectory immediately (streaming write-through)
  Phase 3: Load VAE (~160MB), decode final latents to PNG, free VAE

Crash-resumability:
  Re-running with the same parameters skips already-generated trajectories.
  The parquet index is inspected for matching (seed, prompt_idx, width,
  height, attention_backend, n_steps, cfg) tuples. Only missing entries
  are generated. DatasetWriter.flush() is called every 10 trajectories
  and after the last trajectory, ensuring at most ~60s of data loss on
  crash (one trajectory generation cycle).

Usage:
  set PYTHONUNBUFFERED=1
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\generate_multi_res_trajectories.py

Output:
  multi_res_trajectories/   -- V2 dataset with multi-resolution trajectories
  multi_res_trajectories/renders/  -- VAE-decoded PNGs of final latents
  multi_res_trajectories/generation_report.json  -- timing, stats, verification
"""

from __future__ import annotations

import argparse
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

OUTPUT_DIR = REPO_ROOT / "multi_res_trajectories"
RENDER_DIR = OUTPUT_DIR / "renders"

from src_ii.resolution_sampling import (
    MEGAPIXEL_ANCHORS,
    ANCHOR_LABELS,
    sample_random_resolution,
)
from src_ii.sigma_schedule import compute_logsnr_uniform_steps
import random as _random

TRAJECTORIES_PER_TIER = 5  # per anchor, per backend (5 x 6 tiers x 2 backends = 60 total)
ASPECT_MIN = 0.5
ASPECT_MAX = 2.0
RESOLUTION_RNG_SEED = 777  # deterministic resolution sampling

def _build_resolution_plan() -> list[tuple[int, int, int, str]]:
    """Build (width, height, n_per_backend, label) list from anchors.

    Each anchor gets TRAJECTORIES_PER_TIER entries, each with a
    unique randomly sampled aspect ratio.

    Returns:
        List of (width, height, 1, label) tuples. Each entry is one
        trajectory slot (n_per_backend=1 because each has a unique
        resolution; the outer loop handles backend replication).
    """
    rng = _random.Random(RESOLUTION_RNG_SEED)
    plan = []
    for anchor in MEGAPIXEL_ANCHORS:
        label = ANCHOR_LABELS.get(anchor, f"{anchor}px")
        for _ in range(TRAJECTORIES_PER_TIER):
            w, h = sample_random_resolution(
                anchor, rng, aspect_min=ASPECT_MIN, aspect_max=ASPECT_MAX,
            )
            plan.append((w, h, 1, label))
    return plan

RESOLUTION_TIERS = _build_resolution_plan()

N_STEPS = 30
CFG = 4.0
N_SAVE = 7  # number of sparse steps to save per trajectory (excluding "final")
LEGACY_SPARSE_STEPS = {0, 4, 9, 14, 19, 24, 29}  # old step-uniform baseline

MULTI_PROMPTS: list[str] = [
    'ahem.\n*ting ting ting ting ting*\nthe query model for this is a LARGE LANGUAGE MODEL, specifically QWEN-3-4B, a GENERAL PURPOSE SEMANTIC PARSER which is able to WRITE SENTENCES AT A TIME when they are participating in dialogue. however, in this situation, they are being used as a hidden state generator to steer an *image generation model*, z-image.\n\nqwen-3-4b, draw me an "enormous laser shark for the sega saturn".',
    'qwen-3-4b, draw me a "gigantic laser shark breaching out of the ocean at sunset".',
    'A neon sign reading "OPEN 24 HOURS" above a rain-soaked Tokyo alleyway at night.',
    'A cat sitting on top of a stack of books next to a window with rain outside, warm interior lighting.',
    'An astronaut riding a horse across a desert under a starfield, photorealistic.',
    'Extreme macro photograph of a butterfly wing showing individual scales, iridescent blue and green.',
    'A cozy medieval tavern interior, firelight, wooden beams, illustrated in the style of a fantasy RPG sourcebook.',
    'Aerial view of a brutalist concrete building surrounded by cherry blossom trees in full bloom.',
    'qwen-3-4b, draw me a "laser shark swimming through a neon cyberpunk cityscape at night".',
    'qwen-3-4b, draw me a "laser shark made of chrome and glass, studio lighting, product photography".',
    'A storefront window with painted gold lettering reading "ANTIQUES & CURIOSITIES" in a foggy English village.',
    'A robot watering potted plants on a balcony overlooking a futuristic city skyline at dawn.',
]

N_PROMPTS = len(MULTI_PROMPTS)
BASE_SEED = 400000  # deterministic seeds, offset from existing dataset

ATTENTION_BACKENDS = ["sdpa", "sage"]


def _log(msg: str) -> None:
    """Print with flush for real-time output."""
    print(msg, flush=True)


def _sigma_to_logsnr(sigma: float) -> float:
    """Compute logSNR = 2 * ln((1-sigma)/sigma), clamped to avoid infinities."""
    s = max(0.001, min(0.999, sigma))
    return 2.0 * math.log((1.0 - s) / s)


def _get_sparse_steps(
    width: int,
    height: int,
    override: set[int] | None = None,
) -> list[int]:
    """Return sparse step indices for a trajectory.

    If *override* is not None, returns it as a sorted list (legacy mode).
    Otherwise, computes logSNR-uniform step indices via
    compute_logsnr_uniform_steps() for the given (width, height).
    """
    if override is not None:
        return sorted(override)
    return compute_logsnr_uniform_steps(width, height, n_steps=N_STEPS, n_save=N_SAVE)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for optional backward-compatible overrides."""
    parser = argparse.ArgumentParser(
        description="Generate multi-resolution trajectories for BTRM training.",
    )
    parser.add_argument(
        "--sparse-steps",
        type=str,
        default=None,
        help=(
            "Override logSNR-uniform step selection with a fixed comma-separated "
            "list of step indices (e.g. '0,4,9,14,19,24,29'). "
            "When set, all trajectories use these exact step indices regardless "
            "of resolution."
        ),
    )
    return parser.parse_args()



def phase1_encode_prompts(device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Load TE, encode all needed prompts, free TE.

    Returns:
        Dict mapping prompt string -> (1, seq_len, 2560) conditioning tensor on CPU.
    """
    _log("\n" + "=" * 60)
    _log("  PHASE 1: TEXT ENCODER -- ENCODE PROMPTS")
    _log("=" * 60)

    from futudiffu.text_encoder import create_tokenizer, load_text_encoder, encode_prompt

    t0 = time.perf_counter()

    _log(f"  Loading tokenizer from {TOKENIZER_PATH}")
    tokenizer = create_tokenizer(TOKENIZER_PATH)

    _log(f"  Loading TE from {TE_WEIGHTS}")
    te_model = load_text_encoder(TE_WEIGHTS, device=device, dtype=dtype)
    _log(f"  VRAM after TE load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    prompt_cache: dict[str, torch.Tensor] = {}

    _log(f"  Encoding negative prompt...")
    neg_cond = encode_prompt(te_model, tokenizer, "", device=device)
    prompt_cache[""] = neg_cond.cpu()
    _log(f"    neg_cond shape: {neg_cond.shape}")

    prompts_to_encode = MULTI_PROMPTS
    for i, prompt in enumerate(prompts_to_encode):
        cond = encode_prompt(te_model, tokenizer, prompt, device=device)
        prompt_cache[prompt] = cond.cpu()
        _log(f"    prompt {i}: shape={cond.shape}, '{prompt[:60]}...'")

    del te_model, tokenizer
    torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    _log(f"  TE freed. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    _log(f"  Phase 1 complete: {elapsed:.1f}s, {len(prompt_cache) - 1} prompts encoded")

    return prompt_cache



def _build_generation_plan(
    sparse_steps_override: set[int] | None = None,
) -> list[dict]:
    """Build a deterministic, ordered generation plan.

    Each entry is a trajectory specification dict containing all fields
    needed for generation and for identity matching during resumption.
    The plan order is deterministic: resolution tiers (outer) x backends
    (inner), with seeds assigned sequentially from BASE_SEED.

    Args:
        sparse_steps_override: If set, use these fixed step indices
            instead of logSNR-uniform computation.

    Returns:
        List of trajectory specification dicts. Each dict contains:
            seed, prompt_idx, prompt, width, height, attention_backend,
            n_steps, cfg, resolution_tier, sampling_shift, sparse_steps,
            step_selection, plan_index.
    """
    from src_ii.sigma_schedule import resolution_shift

    plan = []
    seed_counter = BASE_SEED
    plan_index = 0

    for width, height, n_traj, tier_label in RESOLUTION_TIERS:
        shift = resolution_shift(width, height)
        sparse_steps_list = _get_sparse_steps(width, height, override=sparse_steps_override)

        for backend in ATTENTION_BACKENDS:
            for i in range(n_traj):
                prompt_idx = plan_index % len(MULTI_PROMPTS)
                prompt = MULTI_PROMPTS[prompt_idx]
                seed = seed_counter
                seed_counter += 1

                plan.append({
                    "seed": seed,
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "attention_backend": backend,
                    "n_steps": N_STEPS,
                    "cfg": CFG,
                    "resolution_tier": tier_label,
                    "sampling_shift": shift,
                    "sparse_steps": sparse_steps_list,
                    "step_selection": "logsnr_uniform" if sparse_steps_override is None else "step_uniform",
                    "plan_index": plan_index,
                })
                plan_index += 1

    return plan


def phase2_generate_and_persist(
    prompt_cache: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    sparse_steps_override: set[int] | None = None,
) -> tuple[Path, list[dict], int]:
    """Load FP8 diffusion model, generate trajectories with streaming writes.

    Each trajectory is written to the V2 dataset immediately after generation.
    DatasetWriter.flush() is called every 10 newly-written trajectories and
    after the final trajectory. The step_sigmas sidecar is updated
    incrementally after each trajectory.

    On resume: inspects the existing dataset for trajectories matching the
    generation plan. Already-completed trajectories are skipped.

    Args:
        prompt_cache: Pre-encoded prompt tensors.
        device: CUDA device.
        dtype: Working dtype.
        sparse_steps_override: If set, use these fixed step indices instead of
            logSNR-uniform computation. Enables backward-compatible --sparse-steps.

    Returns:
        (dataset_dir, trajectory_metadata, n_generated) where:
            dataset_dir: Path to the V2 dataset on disk.
            trajectory_metadata: List of metadata dicts for ALL trajectories
                (both previously completed and newly generated).
            n_generated: Number of trajectories generated this run (excludes
                previously completed ones).
    """
    _log("\n" + "=" * 60)
    _log("  PHASE 2: GENERATE + PERSIST (streaming write-through)")
    _log("=" * 60)

    from src_ii.zimage_model import load_zimage_rlaif
    from src_ii.rollout import rollout
    from src_ii.sigma_schedule import resolution_shift
    from futudiffu.dataset_v2 import DatasetWriter
    from src_ii.dataset_resumption import (
        compute_remaining_work,
        trajectory_identity,
        save_generation_plan,
        update_step_sigmas_sidecar,
    )

    t0 = time.perf_counter()

    full_plan = _build_generation_plan(sparse_steps_override)
    total_planned = len(full_plan)

    plan_path = OUTPUT_DIR / "generation_plan.json"
    save_generation_plan(full_plan, plan_path)
    _log(f"  Generation plan saved: {plan_path} ({total_planned} trajectories)")

    dataset_dir = OUTPUT_DIR
    remaining, completed, n_total = compute_remaining_work(full_plan, dataset_dir)
    n_skipped = len(completed)

    if n_skipped > 0:
        _log(f"  RESUMING: {n_skipped}/{total_planned} trajectories already exist, "
             f"{len(remaining)} remaining")
    else:
        _log(f"  Fresh run: {total_planned} trajectories to generate")

    if not remaining:
        _log(f"  All {total_planned} trajectories already exist. Nothing to generate.")
        trajectory_metadata = _build_metadata_from_plan(full_plan)
        return dataset_dir, trajectory_metadata, 0

    _log(f"  Resolutions: {[(w, h, n) for w, h, n, _ in RESOLUTION_TIERS]}")
    _log(f"  Backends: {ATTENTION_BACKENDS}")
    _log(f"  Steps: {N_STEPS}, CFG: {CFG}")
    if sparse_steps_override is not None:
        _log(f"  Sparse steps: OVERRIDE {sorted(sparse_steps_override)} (step-uniform)")
    else:
        _log(f"  Sparse steps: logSNR-uniform, {N_SAVE} per trajectory (resolution-aware)")

    if sparse_steps_override is None:
        _log(f"\n  --- LogSNR-uniform step indices per resolution ---")
        seen_resolutions: set[tuple[int, int]] = set()
        for width, height, _, tier_label in RESOLUTION_TIERS:
            if (width, height) not in seen_resolutions:
                seen_resolutions.add((width, height))
                steps = _get_sparse_steps(width, height, override=None)
                shift = resolution_shift(width, height)
                _log(f"    {tier_label:>8s} {width:>5d}x{height:<5d} shift={shift:.3f}  steps={steps}")

    _log(f"  Loading FP8 diffusion model (with compile for reduced activation memory)...")
    diff_model = load_zimage_rlaif(
        FP8_WEIGHTS,
        device=device,
        dtype=dtype,
        compile_model=True,
        fuse=True,
        use_sage=True,
    )
    _log(f"  VRAM after model load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    remaining_identities = {trajectory_identity(spec) for spec in remaining}

    prompts = MULTI_PROMPTS
    neg_cond = prompt_cache[""]
    sidecar_path = OUTPUT_DIR / "step_sigmas.json"

    gen_count = 0          # newly generated this run
    total_count = 0        # total processed (including skipped)
    new_since_flush = 0    # trajectories written since last flush
    timing_per_res: dict[str, dict] = {}
    trajectory_metadata: list[dict] = []  # metadata-only (no tensors)

    dataset_dir.mkdir(parents=True, exist_ok=True)

    with DatasetWriter(str(dataset_dir)) as writer:
        current_backend = None

        for spec in full_plan:
            total_count += 1
            identity = trajectory_identity(spec)

            width = spec["width"]
            height = spec["height"]
            backend = spec["attention_backend"]
            seed = spec["seed"]
            prompt_idx = spec["prompt_idx"]
            prompt = spec["prompt"]
            shift = spec["sampling_shift"]
            sparse_steps_list = spec["sparse_steps"]
            tier_label = spec["resolution_tier"]

            if identity not in remaining_identities:
                trajectory_metadata.append({
                    "metadata": {
                        "prompt": prompt,
                        "prompt_idx": prompt_idx,
                        "seed": seed,
                        "cfg": CFG,
                        "width": width,
                        "height": height,
                        "n_steps": N_STEPS,
                        "attention_backend": backend,
                        "resolution_tier": tier_label,
                    },
                    "width": width,
                    "height": height,
                })
                continue

            if backend != current_backend:
                current_backend = backend
                _log(f"      Attention backend: {backend}")

            save_steps = set(sparse_steps_list)

            pos_cond = prompt_cache[prompt].to(device=device, dtype=dtype)
            neg_c = neg_cond.to(device=device, dtype=dtype)

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
                save_steps=save_steps,
            )

            dt = time.perf_counter() - t_traj
            gen_count += 1

            step_sigmas = meta.get("step_sigmas", {})

            step_logsnrs: dict[str, float] = {}
            for step_key, sigma_val in step_sigmas.items():
                step_logsnrs[step_key] = _sigma_to_logsnr(sigma_val)

            traj_metadata = {
                "prompt": prompt,
                "prompt_idx": prompt_idx,
                "seed": seed,
                "cfg": CFG,
                "width": width,
                "height": height,
                "n_steps": N_STEPS,
                "attention_backend": backend,
                "batch_type": "t2i",
                "sampling_shift": shift,
                "is_gold": (backend == "sdpa" and N_STEPS == 30),
                "packed": False,
                "base_model_hash": "z_image_v1",
                "run_name": "multi_res_gen",
                "source_device": "rtx4090_0",
                "resolution_tier": tier_label,
                "step_sigmas": step_sigmas,
                "step_logsnrs": step_logsnrs,
                "sparse_steps": sparse_steps_list,
                "step_selection": spec["step_selection"],
            }

            cpu_tensors = {k: v.cpu() for k, v in result_tensors.items()}
            del result_tensors
            torch.cuda.empty_cache()

            traj_id = writer.add_trajectory(
                tensors=cpu_tensors,
                metadata=traj_metadata,
            )
            del cpu_tensors  # free CPU RAM immediately
            new_since_flush += 1

            if new_since_flush >= 10:
                writer.flush()
                _log(f"    Flushed after {gen_count} new trajectories (traj_id up to {traj_id})")
                new_since_flush = 0

            if step_sigmas:
                update_step_sigmas_sidecar(
                    sidecar_path, traj_id, step_sigmas, step_logsnrs,
                )

            trajectory_metadata.append({
                "metadata": traj_metadata,
                "width": width,
                "height": height,
            })

            res_key = f"{width}x{height}"
            if res_key not in timing_per_res:
                timing_per_res[res_key] = {"count": 0, "total_s": 0.0}
            timing_per_res[res_key]["count"] += 1
            timing_per_res[res_key]["total_s"] += dt

            _log(f"    [{gen_count}/{len(remaining)}] {width}x{height} {backend} "
                 f"seed={seed} prompt={prompt_idx} ({dt:.1f}s) -> traj_id={traj_id}")

        if new_since_flush > 0:
            writer.flush()
            _log(f"    Final flush: {new_since_flush} trajectories")


    del diff_model, diff_model
    torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    _log(f"\n  Phase 2 complete: {gen_count} new trajectories in {elapsed:.1f}s "
         f"({n_skipped} skipped from previous run)")
    _log(f"  VRAM after model free: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    if timing_per_res:
        _log(f"  Timing per resolution:")
        for res, stats in timing_per_res.items():
            avg = stats["total_s"] / stats["count"] if stats["count"] else 0
            _log(f"    {res}: {stats['count']} traj, {stats['total_s']:.1f}s total, {avg:.1f}s avg")

    return dataset_dir, trajectory_metadata, gen_count


def _build_metadata_from_plan(plan: list[dict]) -> list[dict]:
    """Build trajectory_metadata list from a generation plan (for the case
    where all trajectories were already completed and no model was loaded).

    Returns the same structure as phase2_generate_and_persist's
    trajectory_metadata output.
    """
    metadata = []
    for spec in plan:
        metadata.append({
            "metadata": {
                "prompt": spec["prompt"],
                "prompt_idx": spec["prompt_idx"],
                "seed": spec["seed"],
                "cfg": spec["cfg"],
                "width": spec["width"],
                "height": spec["height"],
                "n_steps": spec["n_steps"],
                "attention_backend": spec["attention_backend"],
                "resolution_tier": spec["resolution_tier"],
            },
            "width": spec["width"],
            "height": spec["height"],
        })
    return metadata





def phase4_render(
    trajectories: list[dict],
    device: torch.device,
    dtype: torch.dtype,
) -> list[str]:
    """Load VAE, decode final latents, save PNGs.

    Returns:
        List of rendered PNG paths.
    """
    _log("\n" + "=" * 60)
    _log("  PHASE 4: VAE DECODE + RENDER")
    _log("=" * 60)

    from src_ii.vae_utils import load_vae, decode_latent_to_pil

    RENDER_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    _log(f"  Loading VAE from {VAE_WEIGHTS}")
    vae = load_vae(VAE_WEIGHTS, device=device, dtype=dtype)
    _log(f"  VRAM after VAE load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    rendered = []
    for i, traj in enumerate(trajectories):
        meta = traj["metadata"]
        final_latent = traj["tensors"].get("final")
        if final_latent is None:
            _log(f"    [{i}] No final latent, skipping render")
            continue

        if final_latent.dim() == 3:
            final_latent = final_latent.unsqueeze(0)

        w = meta["width"]
        h = meta["height"]
        backend = meta["attention_backend"]
        seed = meta["seed"]
        prompt_idx = meta["prompt_idx"]

        fname = f"mr_{w}x{h}_{backend}_p{prompt_idx}_s{seed}.png"
        fpath = RENDER_DIR / fname

        try:
            pil_img = decode_latent_to_pil(vae, final_latent, device=device, dtype=dtype)
            pil_img.save(str(fpath))
            rendered.append(str(fpath))

            if (i + 1) % 10 == 0 or i == 0:
                _log(f"    [{i + 1}/{len(trajectories)}] Rendered {fname} ({pil_img.size[0]}x{pil_img.size[1]})")
        except Exception as e:
            _log(f"    [{i + 1}/{len(trajectories)}] RENDER FAILED for {fname}: {e}")

    del vae
    torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    _log(f"  Phase 4 complete: {len(rendered)} images rendered in {elapsed:.1f}s")
    _log(f"  Renders: {RENDER_DIR}")

    return rendered



def phase4_render_from_dataset(
    dataset_dir: Path,
    trajectory_metadata: list[dict],
    device: torch.device,
    dtype: torch.dtype,
    n_intermediates: int = 2,
) -> list[str]:
    """Load VAE, decode final + near-clean intermediate latents, save PNGs.

    Memory-efficient: loads one latent at a time from disk instead of keeping
    all trajectory tensors in CPU RAM.

    For each trajectory, also renders the *n_intermediates* non-final steps
    with the lowest sigma (highest logSNR, i.e. cleanest intermediates).
    These go into renders/intermediates/ for visual comparison against the
    clean final.

    Returns:
        List of rendered PNG paths (finals + intermediates).
    """
    _log("\n" + "=" * 60)
    _log("  PHASE 4: VAE DECODE + RENDER (from dataset)")
    _log("=" * 60)

    from src_ii.vae_utils import load_vae, decode_latent_to_pil
    from futudiffu.dataset_v2 import DatasetReader

    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    intermediates_dir = RENDER_DIR / "intermediates"
    intermediates_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    _log(f"  Loading VAE from {VAE_WEIGHTS}")
    vae = load_vae(VAE_WEIGHTS, device=device, dtype=dtype)
    _log(f"  VRAM after VAE load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    _log(f"  Intermediate renders: {n_intermediates} cleanest non-final steps per trajectory")

    reader = DatasetReader(str(dataset_dir))
    n_traj = len(reader)
    _log(f"  Dataset has {n_traj} trajectories")

    rendered = []
    n_intermediates_rendered = 0
    for i in range(n_traj):
        meta_db, accessor = reader[i]
        meta = trajectory_metadata[i]["metadata"] if i < len(trajectory_metadata) else meta_db

        avail = accessor.available_steps
        if "final" not in avail:
            _log(f"    [{i}] No final step, skipping render")
            continue

        w = meta["width"]
        h = meta["height"]
        backend = meta["attention_backend"]
        seed = meta["seed"]
        prompt_idx = meta["prompt_idx"]
        step_sigmas = meta.get("step_sigmas", {})

        final_latent = accessor["final"]
        if final_latent.dim() == 3:
            final_latent = final_latent.unsqueeze(0)

        fname = f"mr_{w}x{h}_{backend}_p{prompt_idx}_s{seed}.png"
        fpath = RENDER_DIR / fname

        try:
            pil_img = decode_latent_to_pil(vae, final_latent, device=device, dtype=dtype)
            pil_img.save(str(fpath))
            rendered.append(str(fpath))

            if (i + 1) % 10 == 0 or i == 0:
                _log(f"    [{i + 1}/{n_traj}] Rendered {fname} ({pil_img.size[0]}x{pil_img.size[1]})")
        except Exception as e:
            _log(f"    [{i + 1}/{n_traj}] RENDER FAILED for {fname}: {e}")

        del final_latent

        nonfinal_steps = [k for k in avail if k != "final" and k in step_sigmas]
        if nonfinal_steps:
            nonfinal_steps.sort(key=lambda k: step_sigmas[k])
            to_render = nonfinal_steps[:n_intermediates]

            for step_key in to_render:
                sigma_val = step_sigmas[step_key]
                step_num = step_key.replace("step_", "")
                int_fname = (
                    f"mr_{w}x{h}_{backend}_p{prompt_idx}_s{seed}"
                    f"_step{step_num}_sigma{sigma_val:.3f}.png"
                )
                int_fpath = intermediates_dir / int_fname

                try:
                    lat = accessor[step_key]
                    if lat.dim() == 3:
                        lat = lat.unsqueeze(0)
                    pil_int = decode_latent_to_pil(vae, lat, device=device, dtype=dtype)
                    pil_int.save(str(int_fpath))
                    rendered.append(str(int_fpath))
                    n_intermediates_rendered += 1
                    del lat
                except Exception as e:
                    _log(f"    [{i + 1}/{n_traj}] INTERMEDIATE RENDER FAILED "
                         f"for {int_fname}: {e}")

            if (i + 1) % 10 == 0 or i == 0:
                rendered_steps = ", ".join(
                    f"{k}(sigma={step_sigmas[k]:.3f})" for k in to_render
                )
                _log(f"      intermediates: {rendered_steps}")

    del vae
    torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    n_finals = len(rendered) - n_intermediates_rendered
    _log(f"  Phase 4 complete: {n_finals} finals + {n_intermediates_rendered} intermediates "
         f"= {len(rendered)} images rendered in {elapsed:.1f}s")
    _log(f"  Renders: {RENDER_DIR}")
    _log(f"  Intermediates: {intermediates_dir}")

    return rendered



def phase5_verify_from_dataset(
    trajectory_metadata: list[dict],
    dataset_dir: Path,
    rendered: list[str],
) -> dict:
    """Verify generated data integrity using persisted dataset on disk.

    Memory-efficient: reads latents from disk one at a time instead of
    keeping all tensors in CPU RAM.

    Returns:
        Verification report dict.
    """
    _log("\n" + "=" * 60)
    _log("  PHASE 5: VERIFICATION (from dataset)")
    _log("=" * 60)

    report = {
        "total_trajectories": len(trajectory_metadata),
        "total_rendered": len(rendered),
        "resolution_distribution": {},
        "backend_distribution": {},
        "latent_shape_checks": [],
        "all_passed": True,
    }

    res_counts = Counter()
    backend_counts = Counter()
    for traj in trajectory_metadata:
        w, h = traj["width"], traj["height"]
        res_counts[f"{w}x{h}"] += 1
        backend_counts[traj["metadata"]["attention_backend"]] += 1

    report["resolution_distribution"] = dict(res_counts)
    report["backend_distribution"] = dict(backend_counts)

    _log(f"  Resolution distribution: {dict(res_counts)}")
    _log(f"  Backend distribution: {dict(backend_counts)}")

    n_shape_ok = 0
    n_shape_fail = 0
    n_valid_finals = 0
    n_invalid_finals = 0

    try:
        from futudiffu.dataset_v2 import DatasetReader
        reader = DatasetReader(str(dataset_dir))
        n_read = len(reader)
        report["n_readable_trajectories"] = n_read
        _log(f"  V2 dataset readable: {n_read} trajectories")

        if n_read != len(trajectory_metadata):
            _log(f"    WARNING: expected {len(trajectory_metadata)}, got {n_read}")
            report["all_passed"] = False

        checked_res = set()
        for traj_id in range(n_read):
            meta_db, accessor = reader[traj_id]
            traj_meta = trajectory_metadata[traj_id] if traj_id < len(trajectory_metadata) else {"width": meta_db.get("width", 0), "height": meta_db.get("height", 0)}
            w = traj_meta["width"]
            h = traj_meta["height"]
            expected = (16, h // 8, w // 8)
            res_key = f"{w}x{h}"

            avail = accessor.available_steps
            if res_key not in checked_res:
                checked_res.add(res_key)
                _log(f"    traj {traj_id}: {res_key}, steps={avail}")

            if "final" in avail:
                final = accessor["final"]
                t = final.squeeze(0) if final.dim() == 4 else final
                actual = tuple(t.shape)

                if actual == expected:
                    n_shape_ok += 1
                else:
                    n_shape_fail += 1
                    report["all_passed"] = False

                if final.abs().max().item() < 1e-10:
                    n_invalid_finals += 1
                    report["all_passed"] = False
                elif torch.isnan(final).any():
                    n_invalid_finals += 1
                    report["all_passed"] = False
                else:
                    n_valid_finals += 1

                del final
            else:
                _log(f"      WARNING: no final step")
                report["all_passed"] = False

    except Exception as e:
        _log(f"  WARNING: V2 dataset read failed: {e}")
        report["all_passed"] = False
        report["read_error"] = str(e)

    report["n_shape_ok"] = n_shape_ok
    report["n_shape_fail"] = n_shape_fail
    report["n_valid_finals"] = n_valid_finals
    report["n_invalid_finals"] = n_invalid_finals

    _log(f"  Latent shape checks: {n_shape_ok} OK, {n_shape_fail} FAIL")
    _log(f"  Final latent validity: {n_valid_finals} valid, {n_invalid_finals} invalid")

    n_renders_exist = sum(1 for p in rendered if os.path.exists(p))
    report["n_renders_on_disk"] = n_renders_exist
    _log(f"  Renders on disk: {n_renders_exist}/{len(rendered)}")

    verdict = "PASS" if report["all_passed"] else "FAIL"
    _log(f"\n  Verification verdict: {verdict}")

    return report



def phase6_packing_analysis_from_metadata(trajectory_metadata: list[dict]) -> dict:
    """Run bin packing analysis on metadata only (no tensors needed).

    Returns:
        Packing analysis dict.
    """
    _log("\n" + "=" * 60)
    _log("  PHASE 6: BIN PACKING ANALYSIS")
    _log("=" * 60)

    from src_ii.bin_packer import (
        BinPackScheduler,
        compute_effective_seq_len,
        compute_seq_len,
        DEFAULT_CAP_TOKENS,
        REFERENCE_TOTAL_LEN,
    )

    scheduler = BinPackScheduler()

    items = []
    for traj in trajectory_metadata:
        w = traj["width"]
        h = traj["height"]
        items.append({
            "width": w,
            "height": h,
            "seq_len": compute_effective_seq_len(w, h, DEFAULT_CAP_TOKENS),
            "img_seq_len": compute_seq_len(w, h),
            "resolution": f"{w}x{h}",
        })

    bins = scheduler.pack(items)
    efficiency = scheduler.estimate_efficiency(bins)

    _log(f"  Items: {len(items)}")
    _log(f"  Bins: {efficiency['n_bins']}")
    _log(f"  Utilization: {efficiency['utilization']:.1%}")
    _log(f"  Sparse compute ratio: {efficiency['sparse_compute_ratio']:.4f}")

    bin_sizes = Counter(len(b) for b in bins)
    _log(f"  Bin size distribution:")
    for size in sorted(bin_sizes):
        _log(f"    {size} items/bin: {bin_sizes[size]} bins")

    for i, b in enumerate(bins[:min(5, len(bins))]):
        desc = ", ".join(f"{item['resolution']}({item['seq_len']})" for item in b)
        used = sum(item["seq_len"] for item in b)
        _log(f"    bin {i}: [{desc}] = {used}/{REFERENCE_TOTAL_LEN} ({used/REFERENCE_TOTAL_LEN:.0%})")

    multi_item_bins = sum(1 for b in bins if len(b) > 1)
    _log(f"\n  Multi-item bins (packing exercised): {multi_item_bins}/{len(bins)}")

    analysis = {
        "n_items": len(items),
        "n_bins": efficiency["n_bins"],
        "utilization": efficiency["utilization"],
        "sparse_compute_ratio": efficiency["sparse_compute_ratio"],
        "bin_size_distribution": dict(bin_sizes),
        "multi_item_bins": multi_item_bins,
        "reference_total_len": REFERENCE_TOTAL_LEN,
        "per_bin": efficiency["per_bin"],
    }

    return analysis



def phase5_verify(trajectories: list[dict], dataset_dir: Path, rendered: list[str]) -> dict:
    """Verify generated data integrity.

    Returns:
        Verification report dict.
    """
    _log("\n" + "=" * 60)
    _log("  PHASE 5: VERIFICATION")
    _log("=" * 60)

    report = {
        "total_trajectories": len(trajectories),
        "total_rendered": len(rendered),
        "resolution_distribution": {},
        "backend_distribution": {},
        "latent_shape_checks": [],
        "all_passed": True,
    }

    res_counts = Counter()
    backend_counts = Counter()
    for traj in trajectories:
        w, h = traj["width"], traj["height"]
        res_counts[f"{w}x{h}"] += 1
        backend_counts[traj["metadata"]["attention_backend"]] += 1

    report["resolution_distribution"] = dict(res_counts)
    report["backend_distribution"] = dict(backend_counts)

    _log(f"  Resolution distribution: {dict(res_counts)}")
    _log(f"  Backend distribution: {dict(backend_counts)}")

    expected_shapes = {
        256: (16, 256 // 8, 256 // 8),   # (16, 32, 32)
        512: (16, 512 // 8, 512 // 8),   # (16, 64, 64)
        1024: (16, 1024 // 8, 1024 // 8), # (16, 128, 128)
    }

    n_shape_ok = 0
    n_shape_fail = 0

    for i, traj in enumerate(trajectories):
        w = traj["width"]
        h = traj["height"]
        expected = (16, h // 8, w // 8)

        for step_key, tensor in traj["tensors"].items():
            t = tensor.squeeze(0) if tensor.dim() == 4 else tensor
            actual = tuple(t.shape)

            if actual == expected:
                n_shape_ok += 1
            else:
                n_shape_fail += 1
                report["all_passed"] = False
                report["latent_shape_checks"].append({
                    "traj_idx": i,
                    "step": step_key,
                    "expected": expected,
                    "actual": actual,
                    "passed": False,
                })

    report["n_shape_ok"] = n_shape_ok
    report["n_shape_fail"] = n_shape_fail

    _log(f"  Latent shape checks: {n_shape_ok} OK, {n_shape_fail} FAIL")

    n_valid_finals = 0
    n_invalid_finals = 0

    for traj in trajectories:
        final = traj["tensors"].get("final")
        if final is None:
            n_invalid_finals += 1
            continue

        if final.abs().max().item() < 1e-10:
            n_invalid_finals += 1
            report["all_passed"] = False
            _log(f"    WARNING: traj has all-zero final latent")
        elif torch.isnan(final).any():
            n_invalid_finals += 1
            report["all_passed"] = False
            _log(f"    WARNING: traj has NaN final latent")
        else:
            n_valid_finals += 1

    report["n_valid_finals"] = n_valid_finals
    report["n_invalid_finals"] = n_invalid_finals

    _log(f"  Final latent validity: {n_valid_finals} valid, {n_invalid_finals} invalid")

    n_renders_exist = sum(1 for p in rendered if os.path.exists(p))
    report["n_renders_on_disk"] = n_renders_exist
    _log(f"  Renders on disk: {n_renders_exist}/{len(rendered)}")

    try:
        from futudiffu.dataset_v2 import DatasetReader
        reader = DatasetReader(str(dataset_dir))
        n_read = len(reader)
        report["n_readable_trajectories"] = n_read
        _log(f"  V2 dataset readable: {n_read} trajectories")

        if n_read != len(trajectories):
            _log(f"    WARNING: expected {len(trajectories)}, got {n_read}")
            report["all_passed"] = False

        checked_res = set()
        for traj_id in range(n_read):
            meta, accessor = reader[traj_id]
            res_key = f"{meta['width']}x{meta['height']}"
            if res_key not in checked_res:
                checked_res.add(res_key)
                avail = accessor.available_steps
                _log(f"    traj {traj_id}: {res_key}, steps={avail}")
                if "final" not in avail:
                    _log(f"      WARNING: no final step")
                    report["all_passed"] = False
    except Exception as e:
        _log(f"  WARNING: V2 dataset read failed: {e}")
        report["all_passed"] = False
        report["read_error"] = str(e)

    verdict = "PASS" if report["all_passed"] else "FAIL"
    _log(f"\n  Verification verdict: {verdict}")

    return report



def phase6_packing_analysis(trajectories: list[dict]) -> dict:
    """Run bin packing analysis on the generated data to verify non-degeneracy.

    Returns:
        Packing analysis dict.
    """
    _log("\n" + "=" * 60)
    _log("  PHASE 6: BIN PACKING ANALYSIS")
    _log("=" * 60)

    from src_ii.bin_packer import (
        BinPackScheduler,
        compute_effective_seq_len,
        compute_seq_len,
        DEFAULT_CAP_TOKENS,
        REFERENCE_TOTAL_LEN,
    )

    scheduler = BinPackScheduler()

    items = []
    for traj in trajectories:
        w = traj["width"]
        h = traj["height"]
        items.append({
            "width": w,
            "height": h,
            "seq_len": compute_effective_seq_len(w, h, DEFAULT_CAP_TOKENS),
            "img_seq_len": compute_seq_len(w, h),
            "resolution": f"{w}x{h}",
        })

    bins = scheduler.pack(items)
    efficiency = scheduler.estimate_efficiency(bins)

    _log(f"  Items: {len(items)}")
    _log(f"  Bins: {efficiency['n_bins']}")
    _log(f"  Utilization: {efficiency['utilization']:.1%}")
    _log(f"  Sparse compute ratio: {efficiency['sparse_compute_ratio']:.4f}")

    bin_sizes = Counter(len(b) for b in bins)
    _log(f"  Bin size distribution:")
    for size in sorted(bin_sizes):
        _log(f"    {size} items/bin: {bin_sizes[size]} bins")

    for i, b in enumerate(bins[:min(5, len(bins))]):
        desc = ", ".join(f"{item['resolution']}({item['seq_len']})" for item in b)
        used = sum(item["seq_len"] for item in b)
        _log(f"    bin {i}: [{desc}] = {used}/{REFERENCE_TOTAL_LEN} ({used/REFERENCE_TOTAL_LEN:.0%})")

    multi_item_bins = sum(1 for b in bins if len(b) > 1)
    _log(f"\n  Multi-item bins (packing exercised): {multi_item_bins}/{len(bins)}")

    analysis = {
        "n_items": len(items),
        "n_bins": efficiency["n_bins"],
        "utilization": efficiency["utilization"],
        "sparse_compute_ratio": efficiency["sparse_compute_ratio"],
        "bin_size_distribution": dict(bin_sizes),
        "multi_item_bins": multi_item_bins,
        "reference_total_len": REFERENCE_TOTAL_LEN,
        "per_bin": efficiency["per_bin"],
    }

    return analysis



def main() -> int:
    args = _parse_args()

    sparse_steps_override: set[int] | None = None
    if args.sparse_steps is not None:
        sparse_steps_override = set(int(x.strip()) for x in args.sparse_steps.split(","))
        assert all(0 <= s < N_STEPS for s in sparse_steps_override), (
            f"All sparse step indices must be in [0, {N_STEPS}), got {sparse_steps_override}"
        )

    wall_start = time.perf_counter()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RENDER_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    _log("=" * 60)
    _log("  MULTI-RESOLUTION TRAJECTORY GENERATION")
    _log("  Continuous resolution sampling from 6 megapixel anchors")
    _log("=" * 60)
    _log(f"  Output: {OUTPUT_DIR}")
    _log(f"  Device: {device}")
    _log(f"  VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    _log(f"  Megapixel anchors: {MEGAPIXEL_ANCHORS}")
    _log(f"  Trajectories per tier: {TRAJECTORIES_PER_TIER}")
    _log(f"  Aspect ratio range: [{ASPECT_MIN}, {ASPECT_MAX}]")
    _log(f"  Unique resolutions: {len(set((w, h) for w, h, _, _ in RESOLUTION_TIERS))}")
    _log(f"  Backends: {ATTENTION_BACKENDS}")
    _log(f"  Prompts: {N_PROMPTS} (cycling across trajectories)")
    _log(f"  Total planned: {sum(n for _, _, n, _ in RESOLUTION_TIERS) * len(ATTENTION_BACKENDS)}")
    if sparse_steps_override is not None:
        _log(f"  Step selection: OVERRIDE (step-uniform): {sorted(sparse_steps_override)}")
    else:
        _log(f"  Step selection: logSNR-uniform, {N_SAVE} steps per trajectory")

    prompt_cache = phase1_encode_prompts(device, dtype)

    dataset_dir, trajectory_metadata, n_generated = phase2_generate_and_persist(
        prompt_cache, device, dtype, sparse_steps_override=sparse_steps_override,
    )

    n_trajectories_total = len(trajectory_metadata)

    rendered = phase4_render_from_dataset(dataset_dir, trajectory_metadata, device, dtype)

    verification = phase5_verify_from_dataset(trajectory_metadata, dataset_dir, rendered)

    packing = phase6_packing_analysis_from_metadata(trajectory_metadata)

    wall_total = time.perf_counter() - wall_start
    res_tier_summary = Counter()
    for tm in trajectory_metadata:
        w, h = tm["width"], tm["height"]
        tier = tm["metadata"].get("resolution_tier", "unknown")
        res_tier_summary[f"{w}x{h} ({tier})"] += 1

    intermediates_dir = RENDER_DIR / "intermediates"
    n_intermediate_renders = len(list(intermediates_dir.glob("*.png"))) if intermediates_dir.exists() else 0
    n_final_renders = len(rendered) - n_intermediate_renders

    report = {
        "script": "generate_multi_res_trajectories.py",
        "start_time": datetime.now(timezone.utc).isoformat(),
        "wall_time_s": wall_total,
        "n_trajectories": n_trajectories_total,
        "n_generated_this_run": n_generated,
        "n_resumed_from_previous": n_trajectories_total - n_generated,
        "n_rendered": len(rendered),
        "n_final_renders": n_final_renders,
        "n_intermediate_renders": n_intermediate_renders,
        "resolution_distribution": dict(res_tier_summary),
        "n_unique_resolutions": len(set(
            (tm["width"], tm["height"]) for tm in trajectory_metadata
        )),
        "megapixel_anchors": [str(a) for a in MEGAPIXEL_ANCHORS],
        "trajectories_per_tier": TRAJECTORIES_PER_TIER,
        "aspect_range": [ASPECT_MIN, ASPECT_MAX],
        "attention_backends": ATTENTION_BACKENDS,
        "n_steps": N_STEPS,
        "cfg": CFG,
        "n_save": N_SAVE,
        "step_selection": (
            f"override:{sorted(sparse_steps_override)}"
            if sparse_steps_override is not None
            else "logsnr_uniform"
        ),
        "n_prompts": N_PROMPTS,
        "verification": verification,
        "packing_analysis": packing,
    }

    report_path = OUTPUT_DIR / "generation_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))

    n_skipped = n_trajectories_total - n_generated
    _log(f"\n{'=' * 60}")
    _log(f"  GENERATION COMPLETE")
    _log(f"{'=' * 60}")
    _log(f"  Wall time: {wall_total:.1f}s ({wall_total/60:.1f} min)")
    _log(f"  Trajectories: {n_trajectories_total} total "
         f"({n_generated} new, {n_skipped} resumed)")
    _log(f"  Rendered: {len(rendered)} ({n_final_renders} finals + {n_intermediate_renders} intermediates)")
    _log(f"  Verification: {'PASS' if verification['all_passed'] else 'FAIL'}")
    _log(f"  Packing bins: {packing['n_bins']} (multi-item: {packing['multi_item_bins']})")
    _log(f"  Report: {report_path}")
    _log(f"  Dataset: {dataset_dir}")
    _log(f"  Renders: {RENDER_DIR}")
    _log(f"  Intermediates: {RENDER_DIR / 'intermediates'}")

    return 0 if verification["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

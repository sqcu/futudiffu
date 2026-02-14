"""Generate BTRM training dataset: schedule-driven diffusion trajectory generation.

Pure scheduling client -- all model loading and inference happens on the
inference server (futudiffu.server). This script owns: schedule config, RNG,
prompt selection, metadata, disk I/O, resumability.

Requires a running inference server:
    python -m futudiffu.server --port 5555 --fp8-diff ... --te ... --vae ...

Usage:
    python generate_btrm_dataset.py --schedule schedule.json [--output-dir PATH]
    python generate_btrm_dataset.py --t2i 20 --i2i 10 [--output-dir PATH]

Schedule JSON format:
    [
        {"type": "t2i", "count": 10, "precision": "sdpa", "steps": 30},
        {"type": "t2i", "count": 10, "precision": "sage", "steps": 30},
        {"type": "t2i", "count": 10, "precision": "sdpa", "steps": [8, 22]},
        {"type": "i2i", "count": 5, "precision": "sdpa"},
        {"type": "i2i", "count": 5, "precision": "sage"}
    ]

Output structure:
    output_dir/
        manifest.json           # Schedule + all trajectory metadata
        latents/
            traj_000000/        # One dir per trajectory
                step_00.pt      # Latent at checkpoint step
                final.pt        # Final latent (post inverse_noise_scaling)
                meta.json       # {seed, prompt, n_steps, precision, type, ...}
        renders/
            traj_000000/        # VAE-decoded images for visual QA
                final.png
"""

import argparse
import json
import random
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

import numpy as np
import torch
from PIL import Image

from futudiffu.btrm_dataset import (
    I2I_IMAGES,
    PROMPT_TEMPLATES,
    TRANSFORMATIVE_LABELS,
)
from futudiffu.client import InferenceClient


# ---------------------------------------------------------------------------
# Graceful interruption
# ---------------------------------------------------------------------------

_interrupted = False


def _signal_handler(signum, frame):
    global _interrupted
    if _interrupted:
        print("\nForce quit.")
        sys.exit(1)
    _interrupted = True
    print("\nInterrupt received. Finishing current trajectory then saving...")


signal.signal(signal.SIGINT, _signal_handler)


# ---------------------------------------------------------------------------
# Image loading for i2i
# ---------------------------------------------------------------------------

def load_i2i_image(image_path: Path, multiple: int = 16) -> tuple[torch.Tensor, int, int]:
    """Load image for i2i at native resolution.  No resampling.

    The image is center-cropped to the nearest multiple of `multiple`
    (16 = VAE 8x downscale * patch_size 2).  At most 15 pixels are trimmed
    from each edge — no interpolation, no spectral distortion.

    Returns (tensor, width, height) where tensor is (1, 3, H, W) in [0, 1]
    bf16, and width/height are the cropped pixel dimensions.
    """
    img = Image.open(str(image_path)).convert("RGB")
    src_w, src_h = img.size

    crop_w = (src_w // multiple) * multiple
    crop_h = (src_h // multiple) * multiple

    if crop_w != src_w or crop_h != src_h:
        left = (src_w - crop_w) // 2
        top = (src_h - crop_h) // 2
        img = img.crop((left, top, left + crop_w, top + crop_h))

    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(dtype=torch.bfloat16), crop_w, crop_h


# ---------------------------------------------------------------------------
# Schedule parameter sampling
# ---------------------------------------------------------------------------

GENERATION_DEFAULTS = {
    "cfg": 4.0,
    "width": 1280,
    "height": 832,
    "sampling_shift": 1.0,
    "multiplier": 1.0,
    "save_steps": [0, 4, 9, 14, 19, 24, 29],
}

# Maximum images to pack into a single FlexAttention forward pass.
# 4 x 1280x832 ≈ 16768 tokens packed, well within 4090 VRAM budget.
PACK_SIZE = 4


def _sample_t2i_params(rng: random.Random, batch: dict) -> dict:
    """Sample random parameters for one t2i trajectory from a batch spec."""
    steps_spec = batch.get("steps", 30)
    if isinstance(steps_spec, list):
        n_steps = rng.randint(steps_spec[0], steps_spec[1])
    else:
        n_steps = steps_spec

    prompt_idx = rng.randrange(len(PROMPT_TEMPLATES))
    return {
        "type": "t2i",
        "seed": rng.randint(0, 2**32 - 1),
        "prompt_idx": prompt_idx,
        "prompt": PROMPT_TEMPLATES[prompt_idx],
        "n_steps": n_steps,
        "precision": batch["precision"],
    }


def _sample_i2i_params(rng: random.Random, batch: dict) -> dict:
    """Sample random parameters for one i2i trajectory from a batch spec."""
    # Z-Image flow-matching (CONST model) has a very strong structural prior.
    # denoise < 0.75 produces negligible transformation regardless of CFG.
    # Effective range is [0.75, 0.95] for this model.
    denoise_spec = batch.get("denoise", [0.75, 0.95])
    if isinstance(denoise_spec, list):
        denoise = rng.uniform(denoise_spec[0], denoise_spec[1])
    else:
        denoise = denoise_spec

    # Step count is independent of denoise — use batch config (default 30).
    # Denoise controls only the starting sigma (how much structure to preserve).
    # Scrongle variants get fewer steps from the batch's "steps" field, not from denoise.
    steps_spec = batch.get("steps", 30)
    if isinstance(steps_spec, list):
        n_steps = rng.randint(steps_spec[0], steps_spec[1])
    else:
        n_steps = steps_spec

    img_info = rng.choice(I2I_IMAGES)
    transform = rng.choice(TRANSFORMATIVE_LABELS)
    prompt = f"{img_info['object_label']}, {transform}"

    return {
        "type": "i2i",
        "seed": rng.randint(0, 2**32 - 1),
        "prompt": prompt,
        "n_steps": n_steps,
        "denoise": round(denoise, 3),
        "image_file": img_info["file"],
        "precision": batch["precision"],
    }


def _next_traj_idx(latents_dir: Path) -> int:
    """Find the next available trajectory index by scanning existing dirs."""
    if not latents_dir.exists():
        return 0
    existing = [d.name for d in latents_dir.iterdir() if d.is_dir()
                and d.name.startswith("traj_")]
    if not existing:
        return 0
    return max(int(name.split("_")[1]) for name in existing) + 1


def _load_existing_records(latents_dir: Path) -> list[dict]:
    """Load meta.json from all existing trajectory dirs."""
    records = []
    if not latents_dir.exists():
        return records
    for d in sorted(latents_dir.iterdir()):
        meta_file = d / "meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            meta["traj_dir"] = str(d)
            records.append(meta)
    return records


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------

def save_trajectory(traj_dir: Path, result: dict[str, torch.Tensor], params: dict):
    """Save trajectory latents and metadata to disk."""
    traj_dir.mkdir(parents=True, exist_ok=True)

    # Save each tensor
    for name, tensor in result.items():
        torch.save(tensor, traj_dir / f"{name}.pt")

    # Save metadata
    meta = {k: v for k, v in params.items()}
    (traj_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def render_latent_to_png(
    client: InferenceClient,
    latent_path: Path,
    output_path: Path,
):
    """VAE-decode a latent file and save as PNG."""
    latent = torch.load(str(latent_path), map_location="cpu", weights_only=True)
    image = client.vae_decode(latent)
    image_np = (image[0].permute(1, 2, 0).float().numpy() * 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_np).save(str(output_path))


# ---------------------------------------------------------------------------
# Packed batch generation
# ---------------------------------------------------------------------------

def _flush_packed_batch(
    client: InferenceClient,
    pending: list[tuple[int, dict, Path]],
    te_cache: dict[str, torch.Tensor],
    neg_cond: torch.Tensor,
    defs: dict,
    timing: dict,
    batch_idx: int,
) -> int:
    """Generate a packed batch of t2i trajectories via FlexAttention.

    Args:
        pending: List of (traj_idx, params_dict, traj_dir) to generate together.
        te_cache: Prompt -> conditioning tensor cache.
        neg_cond: Shared negative conditioning.
        defs: GENERATION_DEFAULTS.
        timing: Timing accumulator dict.
        batch_idx: Current schedule batch index.

    Returns:
        Number of trajectories generated.
    """
    if not pending:
        return 0

    pos_conds = [te_cache[p["prompt"]] for _, p, _ in pending]
    seeds = [p["seed"] for _, p, _ in pending]
    n_steps = pending[0][1]["n_steps"]

    t0 = time.perf_counter()
    results = client.sample_trajectory_packed(
        pos_conds=pos_conds,
        neg_cond=neg_cond,
        seeds=seeds,
        n_steps=n_steps,
        cfg=defs["cfg"],
        width=defs["width"],
        height=defs["height"],
        sampling_shift=defs["sampling_shift"],
        multiplier=defs["multiplier"],
        save_steps=defs["save_steps"],
    )
    timing["diffusion"] += time.perf_counter() - t0

    for (idx, params, traj_dir), result in zip(pending, results):
        meta = {**params, "batch_idx": batch_idx, "packed": True,
                "pack_size": len(pending)}
        save_trajectory(traj_dir, result, meta)

    return len(pending)


# ---------------------------------------------------------------------------
# Schedule execution
# ---------------------------------------------------------------------------

def run_schedule(
    schedule: list[dict],
    client: InferenceClient,
    output_dir: Path,
    rng_seed: int = 42,
    i2i_dir: str = "",
):
    """Execute a generation schedule via the inference server.

    Each entry in schedule is a batch: {type, count, precision, steps, render, ...}.
    Batches execute in order. Resume is automatic.
    """
    global _interrupted

    defs = GENERATION_DEFAULTS
    output_dir.mkdir(parents=True, exist_ok=True)
    latents_dir = output_dir / "latents"

    # Print schedule summary
    total_requested = sum(b["count"] for b in schedule)
    print(f"Schedule: {len(schedule)} batches, {total_requested} trajectories total")
    for i, batch in enumerate(schedule):
        steps_desc = batch.get("steps", 30)
        if isinstance(steps_desc, list):
            steps_desc = f"{steps_desc[0]}-{steps_desc[1]}"
        render_n = batch.get("render", 0)
        print(f"  [{i}] {batch['type']:3s} x{batch['count']:4d}  "
              f"prec={batch['precision']:4s}  steps={steps_desc}"
              f"{'  render=' + str(render_n) if render_n else ''}")

    # Timing
    timing = {
        "te_encode": 0.0, "warmup": 0.0,
        "diffusion": 0.0, "vae_decode": 0.0, "overhead": 0.0,
    }
    wall_start = time.perf_counter()

    # ---------------------------------------------------------------
    # Phase 1: Collect all prompts and encode them via server
    # ---------------------------------------------------------------
    rng_preview = random.Random(rng_seed)
    all_prompts: set[str] = set()
    for batch in schedule:
        for _ in range(batch["count"]):
            if batch["type"] == "i2i":
                params = _sample_i2i_params(rng_preview, batch)
            else:
                params = _sample_t2i_params(rng_preview, batch)
            all_prompts.add(params["prompt"])

    print(f"\nEncoding {len(all_prompts)} unique prompts + 1 negative...")
    t0 = time.perf_counter()
    neg_cond = client.encode_prompt("")
    te_cache: dict[str, torch.Tensor] = {}
    for prompt in sorted(all_prompts):
        te_cache[prompt] = client.encode_prompt(prompt)
    timing["te_encode"] += time.perf_counter() - t0
    print(f"  Done ({time.perf_counter() - t0:.1f}s)")

    # Free TE on server (diffusion model will be loaded on next call)
    client.free("te")

    # ---------------------------------------------------------------
    # Phase 2: Warmup diffusion model
    # ---------------------------------------------------------------
    precisions_needed = {b["precision"] for b in schedule}
    for variant in sorted(precisions_needed):
        print(f"  Warming up {variant}...")
        t0 = time.perf_counter()
        client.warmup(variant)
        timing["warmup"] += time.perf_counter() - t0

    # Warmup packed path (FlexAttention + torch.compile for forward_packed)
    has_t2i = any(b["type"] == "t2i" for b in schedule)
    if has_t2i:
        print(f"  Warming up packed forward (FlexAttention)...")
        t0 = time.perf_counter()
        client.warmup_packed(n_images=min(PACK_SIZE, 2))
        timing["warmup"] += time.perf_counter() - t0

    # ---------------------------------------------------------------
    # Phase 3: Generate trajectories
    # ---------------------------------------------------------------
    traj_idx = _next_traj_idx(latents_dir)  # Append after any existing trajectories.
    start_idx = traj_idx
    # RNG is deterministic within this schedule: every trajectory in the schedule
    # advances the RNG whether it's generated or skipped, so resume after interrupt
    # produces the same params for remaining trajectories.
    generated_count = 0
    skipped_count = 0
    rng = random.Random(rng_seed)
    all_records = _load_existing_records(latents_dir)

    print(f"\nExisting trajectories: {len(all_records)}")
    print()

    # Collect trajectories that need rendering
    render_queue: list[tuple[Path, Path]] = []

    for batch_i, batch in enumerate(schedule):
        if _interrupted:
            break

        batch_type = batch["type"]
        batch_count = batch["count"]
        batch_precision = batch["precision"]
        batch_render = batch.get("render", 0)
        render_done = 0
        batch_generated = 0

        # Packing accumulator: adjacent t2i trajectories with the same
        # n_steps are batched into a single FlexAttention forward pass.
        pending_pack: list[tuple[int, dict, Path]] = []

        def _do_flush():
            """Flush the pending pack and update counters."""
            nonlocal batch_generated, generated_count, render_done
            n = _flush_packed_batch(
                client, pending_pack, te_cache, neg_cond,
                defs, timing, batch_i,
            )
            # Queue renders for the flushed trajectories
            for idx, params_p, tdir in pending_pack:
                if render_done < batch_render:
                    rdir = output_dir / "renders" / f"traj_{idx:06d}"
                    render_queue.append((tdir, rdir))
                    render_done += 1
            batch_generated += n
            generated_count += n

        for j in range(batch_count):
            if _interrupted:
                break

            # Sample params (always, to advance RNG deterministically)
            if batch_type == "i2i":
                params = _sample_i2i_params(rng, batch)
            else:
                params = _sample_t2i_params(rng, batch)

            # Skip if this trajectory was already generated (resume)
            traj_dir = latents_dir / f"traj_{traj_idx:06d}"
            if (traj_dir / "meta.json").exists():
                traj_idx += 1
                skipped_count += 1
                continue

            if batch_type == "t2i":
                # Check compatibility with pending pack (must share n_steps)
                if pending_pack:
                    pack_n_steps = pending_pack[0][1]["n_steps"]
                    if params["n_steps"] != pack_n_steps:
                        _do_flush()
                        pending_pack = []

                pending_pack.append((traj_idx, params, traj_dir))

                if len(pending_pack) >= PACK_SIZE:
                    _do_flush()
                    pending_pack = []
            else:
                # i2i: flush any pending t2i pack, then run individually
                if pending_pack:
                    _do_flush()
                    pending_pack = []

                if not i2i_dir:
                    print(f"  SKIP i2i traj_{traj_idx:06d}: no --i2i-dir")
                    traj_idx += 1
                    continue
                image_path = Path(i2i_dir) / params["image_file"]
                image_tensor, i2i_w, i2i_h = load_i2i_image(image_path)
                clean_latent = client.vae_encode(image_tensor)

                t_traj = time.perf_counter()
                result = client.sample_trajectory(
                    pos_cond=te_cache[params["prompt"]],
                    neg_cond=neg_cond,
                    seed=params["seed"],
                    n_steps=params["n_steps"],
                    cfg=defs["cfg"],
                    width=i2i_w,
                    height=i2i_h,
                    attention_backend=batch_precision,
                    sampling_shift=defs["sampling_shift"],
                    multiplier=defs["multiplier"],
                    save_steps=defs["save_steps"],
                    denoise=params["denoise"],
                    clean_latent=clean_latent,
                )
                timing["diffusion"] += time.perf_counter() - t_traj

                meta = {**params, "batch_idx": batch_i,
                        "output_width": i2i_w, "output_height": i2i_h}
                save_trajectory(traj_dir, result, meta)

                if render_done < batch_render:
                    render_dir = output_dir / "renders" / f"traj_{traj_idx:06d}"
                    render_queue.append((traj_dir, render_dir))
                    render_done += 1

                batch_generated += 1
                generated_count += 1

            traj_idx += 1

        # Flush remaining pack at end of batch
        if pending_pack:
            _do_flush()
            pending_pack = []

        if batch_generated > 0:
            print(f"  Batch [{batch_i}] {batch_type}/{batch_precision}: "
                  f"generated {batch_generated}/{batch_count}"
                  f"{f', rendered {render_done}' if render_done else ''}")

    # ---------------------------------------------------------------
    # Phase 4: Render selected trajectories
    # ---------------------------------------------------------------
    if render_queue and not _interrupted:
        print(f"\nRendering {len(render_queue)} trajectories...")
        for traj_dir, render_dir in render_queue:
            t0 = time.perf_counter()
            final_file = traj_dir / "final.pt"
            if final_file.exists():
                render_latent_to_png(client, final_file, render_dir / "final.png")
            timing["vae_decode"] += time.perf_counter() - t0

    # ---------------------------------------------------------------
    # Save manifest
    # ---------------------------------------------------------------
    all_records = _load_existing_records(latents_dir)
    manifest = {
        "schedule": schedule,
        "rng_seed": rng_seed,
        "records": all_records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Throughput summary
    wall_total = time.perf_counter() - wall_start
    timing["overhead"] = wall_total - sum(timing.values())
    print(f"\n{'='*60}")
    print(f"  THROUGHPUT PROFILE ({generated_count} new, {skipped_count} skipped)")
    print(f"{'='*60}")
    print(f"  {'Phase':<20} {'Time (s)':>10} {'%':>6}")
    print(f"  {'-'*38}")
    for phase, t in sorted(timing.items(), key=lambda x: -x[1]):
        if t > 0.01:
            print(f"  {phase:<20} {t:>10.1f} {100*t/wall_total:>5.1f}%")
    print(f"  {'-'*38}")
    print(f"  {'TOTAL':<20} {wall_total:>10.1f}")
    if generated_count > 0:
        avg = timing["diffusion"] / generated_count
        imgs_per_min = 60.0 / (wall_total / generated_count)
        gpu_active = timing["diffusion"] + timing["vae_decode"] + timing["te_encode"]
        print(f"\n  Avg diffusion: {avg:.1f}s/trajectory")
        print(f"  Throughput:    {imgs_per_min:.2f} images/min")
        print(f"  GPU util:      {100*gpu_active/wall_total:.1f}%")
    print(f"  Total records: {len(all_records)}")
    print(f"  Manifest:      {manifest_path}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_SCHEDULE = [
    {"type": "t2i", "count": 10, "precision": "sdpa", "steps": 30, "render": 3},
    {"type": "t2i", "count": 10, "precision": "sage", "steps": 30, "render": 3},
    {"type": "t2i", "count": 10, "precision": "sdpa", "steps": [8, 22]},
    {"type": "t2i", "count": 10, "precision": "sage", "steps": [8, 22]},
    {"type": "i2i", "count": 5,  "precision": "sdpa", "render": 3},
    {"type": "i2i", "count": 5,  "precision": "sage", "render": 3},
]


def main():
    parser = argparse.ArgumentParser(
        description="Generate BTRM training dataset (schedule-driven, server-backed)")
    parser.add_argument("--output-dir", type=str,
                        default=r"F:\dox\repos\ai\futudiffu\btrm_dataset")
    parser.add_argument("--schedule", type=str, default=None,
                        help="Path to schedule JSON file")
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument("--server", type=str, default="tcp://localhost:5555",
                        help="Inference server endpoint")

    # Shorthand: if no --schedule, build from counts
    parser.add_argument("--t2i", type=int, default=None,
                        help="Total t2i trajectories (split across gold/sage/scrongle)")
    parser.add_argument("--i2i", type=int, default=None,
                        help="Total i2i trajectories (split sdpa/sage)")
    parser.add_argument("--render", type=int, default=6,
                        help="Total trajectories to render across all batches")

    parser.add_argument("--i2i-dir", type=str,
                        default=r"F:\dox\repos\ai\futudiffu\i2i_off_policies")

    parser.add_argument("--dry-run", action="store_true",
                        help="Print schedule and exit")

    args = parser.parse_args()

    # Build schedule
    if args.schedule:
        schedule = json.loads(Path(args.schedule).read_text())
    elif args.t2i is not None or args.i2i is not None:
        schedule = []
        t2i_total = args.t2i or 0
        i2i_total = args.i2i or 0
        renders_left = args.render

        if t2i_total > 0:
            quarter = max(1, t2i_total // 4)
            remainder = t2i_total - quarter * 4

            r = min(renders_left, max(1, quarter // 3))
            schedule.append({"type": "t2i", "count": quarter, "precision": "sdpa",
                             "steps": 30, "render": r})
            renders_left -= r

            r = min(renders_left, max(1, quarter // 3))
            schedule.append({"type": "t2i", "count": quarter, "precision": "sage",
                             "steps": 30, "render": r})
            renders_left -= r

            schedule.append({"type": "t2i", "count": quarter, "precision": "sdpa",
                             "steps": [8, 22]})
            schedule.append({"type": "t2i", "count": quarter + remainder,
                             "precision": "sage", "steps": [8, 22]})

        if i2i_total > 0:
            half = max(1, i2i_total // 2)
            remainder = i2i_total - half * 2

            r = min(renders_left, max(1, half // 2))
            schedule.append({"type": "i2i", "count": half, "precision": "sdpa",
                             "render": r})
            renders_left -= r

            r = min(renders_left, max(1, (half + remainder) // 2))
            schedule.append({"type": "i2i", "count": half + remainder,
                             "precision": "sage", "render": r})
    else:
        schedule = DEFAULT_SCHEDULE

    if args.dry_run:
        total = sum(b["count"] for b in schedule)
        print(f"Schedule: {len(schedule)} batches, {total} trajectories")
        for i, batch in enumerate(schedule):
            steps = batch.get("steps", 30)
            if isinstance(steps, list):
                steps = f"{steps[0]}-{steps[1]}"
            r = batch.get("render", 0)
            print(f"  [{i}] {batch['type']:3s} x{batch['count']:4d}  "
                  f"prec={batch['precision']:4s}  steps={steps}"
                  f"{'  render=' + str(r) if r else ''}")
        return 0

    with InferenceClient(args.server) as client:
        # Verify server is reachable
        try:
            status = client.status()
            print(f"Connected to server: {status.get('loaded_models', [])} loaded, "
                  f"VRAM {status.get('vram_allocated_gb', '?')}GB allocated")
        except Exception as e:
            print(f"Cannot connect to inference server at {args.server}: {e}")
            print("Start the server first: python -m futudiffu.server ...")
            return 1

        run_schedule(
            schedule=schedule,
            client=client,
            output_dir=Path(args.output_dir),
            rng_seed=args.rng_seed,
            i2i_dir=args.i2i_dir,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

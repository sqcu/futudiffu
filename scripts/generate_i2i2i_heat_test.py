"""i2i2i Heat Test: nested image-to-image trajectory generation.

Generates 10 parent i2i trajectories (5 per source image), then picks
intermediate steps from those, forward-noises them, and runs i2i denoise
again under different configs. Writes everything to dataset v2 with parent
trajectory linkage.

Requires a running inference server:
    python -m futudiffu.server --port 5555 --fp8-diff ... --te ... --vae ...

Usage:
    .venv/Scripts/python.exe 'F:\\dox\\repos\\ai\\futudiffu\\scripts\\generate_i2i2i_heat_test.py'
    .venv/Scripts/python.exe 'F:\\dox\\repos\\ai\\futudiffu\\scripts\\generate_i2i2i_heat_test.py' --output-dir PATH
"""

import argparse
import random
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

import numpy as np
import torch
from PIL import Image

from futudiffu.btrm_dataset import I2I_IMAGES, TRANSFORMATIVE_LABELS
from futudiffu.client import InferenceClient
from futudiffu.dataset_v2 import DatasetWriter, DatasetReader
from futudiffu.sampling import build_sigmas, simple_scheduler


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
# Image loading (same as generate_btrm_dataset.py)
# ---------------------------------------------------------------------------

def load_i2i_image(image_path: Path, multiple: int = 16) -> tuple[torch.Tensor, int, int]:
    """Load image for i2i at native resolution, center-crop to nearest multiple.

    Returns (tensor, width, height) where tensor is (1, 3, H, W) in [0, 1] bf16.
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
# Sigma computation
# ---------------------------------------------------------------------------

def compute_sigma_for_step(n_steps: int, denoise: float, step_idx: int) -> float:
    """Compute the sigma at a given step index for a given schedule.

    Replicates the server's schedule construction: if denoise < 1.0, we
    expand the step count and slice from the end.
    """
    sigma_table = build_sigmas(shift=1.0, multiplier=1000.0)
    if denoise < 1.0:
        expanded = int(n_steps / denoise)
        full = simple_scheduler(sigma_table, expanded)
        sigmas = full[-(n_steps + 1):]
    else:
        sigmas = simple_scheduler(sigma_table, n_steps)
    return float(sigmas[step_idx])


# ---------------------------------------------------------------------------
# Source images for heat test
# ---------------------------------------------------------------------------

# 2 source images, 5 transforms each = 10 parent trajectories
SOURCE_IMAGES = [
    "00500-3023556536_re_nightmode2.png",
    "deviantart-is-my-spine-moe-is-my-face.png",
]

# New transforms are indices 16-25 in TRANSFORMATIVE_LABELS
NEW_TRANSFORM_INDICES = list(range(16, 26))

# 10 unique parent configs: (image_idx, transform_idx, seed, denoise, n_steps, attn_backend)
def build_parent_configs(rng: random.Random) -> list[dict]:
    """Build 10 parent i2i configs: 5 per source image."""
    configs = []

    denoise_pool = [0.75, 0.80, 0.85, 0.90, 0.95]
    steps_pool = [18, 20, 22, 25, 30]
    backend_pool = ["sdpa", "sage"]

    # Shuffle transform indices for variety
    transforms = list(NEW_TRANSFORM_INDICES)
    rng.shuffle(transforms)

    for img_i, img_file in enumerate(SOURCE_IMAGES):
        # Find the I2I_IMAGES entry for this file
        img_info = next(info for info in I2I_IMAGES if info["file"] == img_file)

        for j in range(5):
            idx = img_i * 5 + j
            transform_idx = transforms[idx]
            transform_label = TRANSFORMATIVE_LABELS[transform_idx]
            prompt = f"{img_info['object_label']}, {transform_label}"

            configs.append({
                "image_file": img_file,
                "object_label": img_info["object_label"],
                "prompt": prompt,
                "transform_idx": transform_idx,
                "seed": 70000 + idx * 111,
                "denoise": denoise_pool[j],
                "n_steps": steps_pool[j],
                "attention_backend": backend_pool[idx % 2],
            })

    return configs


def build_child_configs(
    parent_configs: list[dict],
    parent_traj_ids: list[int],
    reader: DatasetReader,
    rng: random.Random,
) -> list[dict]:
    """Build 10 i2i2i child configs, one per parent trajectory."""
    configs = []

    for parent_cfg, parent_tid in zip(parent_configs, parent_traj_ids):
        meta, accessor = reader[parent_tid]

        # Pick a random intermediate step (exclude "final")
        available = [s for s in accessor.available_steps if s != "final"]
        step_label = rng.choice(available)
        step_idx = int(step_label.split("_")[1])

        # Compute sigma at that step
        sigma_step = compute_sigma_for_step(
            parent_cfg["n_steps"], parent_cfg["denoise"], step_idx
        )

        # i2i2i denoise: max(sigma_step, 0.75), clamped to [0.75, 0.95]
        i2i2i_denoise = max(sigma_step, 0.75)
        i2i2i_denoise = min(i2i2i_denoise, 0.95)

        # Vary config: new seed, different transform, flip attn backend, vary n_steps
        new_seed = parent_cfg["seed"] + 50000

        # Pick a different transform from the new set
        other_transforms = [t for t in NEW_TRANSFORM_INDICES
                           if t != parent_cfg["transform_idx"]]
        new_transform_idx = rng.choice(other_transforms)
        new_transform_label = TRANSFORMATIVE_LABELS[new_transform_idx]

        # Use same object label, new transform
        new_prompt = f"{parent_cfg['object_label']}, {new_transform_label}"

        # Flip attention backend
        new_backend = "sage" if parent_cfg["attention_backend"] == "sdpa" else "sdpa"

        # Vary n_steps: pick from a different set than parent
        child_steps_pool = [18, 20, 22, 25, 30]
        child_steps_pool = [s for s in child_steps_pool if s != parent_cfg["n_steps"]]
        new_n_steps = rng.choice(child_steps_pool)

        configs.append({
            "image_file": parent_cfg["image_file"],
            "object_label": parent_cfg["object_label"],
            "prompt": new_prompt,
            "transform_idx": new_transform_idx,
            "seed": new_seed,
            "denoise": round(i2i2i_denoise, 4),
            "n_steps": new_n_steps,
            "attention_backend": new_backend,
            # Parent linkage
            "parent_traj_id": parent_tid,
            "parent_step": step_label,
            "parent_denoise": round(i2i2i_denoise, 4),
            "sigma_step": round(sigma_step, 6),
            # For loading the step latent
            "step_label": step_label,
        })

    return configs


# ---------------------------------------------------------------------------
# Generation defaults
# ---------------------------------------------------------------------------

GENERATION_DEFAULTS = {
    "cfg": 4.0,
    "sampling_shift": 1.0,
    "multiplier": 1.0,
    "save_steps": [0, 4, 9, 14, 19, 24, 29],
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="i2i2i Heat Test: nested image-to-image trajectory generation")
    parser.add_argument("--output-dir", type=str,
                        default=r"F:\dox\repos\ai\futudiffu\btrm_dataset_v2_heattest")
    parser.add_argument("--i2i-dir", type=str,
                        default=r"F:\dox\repos\ai\futudiffu\i2i_off_policies")
    parser.add_argument("--server", type=str, default="tcp://localhost:5555")
    parser.add_argument("--rng-seed", type=int, default=7777)
    args = parser.parse_args()

    defs = GENERATION_DEFAULTS
    rng = random.Random(args.rng_seed)
    output_dir = Path(args.output_dir)
    i2i_dir = Path(args.i2i_dir)

    # Build parent configs
    parent_configs = build_parent_configs(rng)

    print("i2i2i Heat Test")
    print(f"  Output: {output_dir}")
    print(f"  Source images: {SOURCE_IMAGES}")
    print(f"  Parent configs: {len(parent_configs)}")
    print()

    with InferenceClient(args.server) as client:
        # Verify server
        try:
            status = client.status()
            print(f"Connected to server: {status.get('loaded_models', [])} loaded, "
                  f"VRAM {status.get('vram_allocated_gb', '?')}GB allocated")
        except Exception as e:
            print(f"Cannot connect to inference server at {args.server}: {e}")
            print("Start the server first: python -m futudiffu.server ...")
            return 1

        timing = {
            "te_encode": 0.0, "warmup": 0.0,
            "diffusion": 0.0, "vae_encode": 0.0, "overhead": 0.0,
        }
        wall_start = time.perf_counter()

        # ---------------------------------------------------------------
        # Phase 1: Encode prompts
        # ---------------------------------------------------------------
        print("\n=== Phase 1: Encode prompts ===")
        all_prompts = set(cfg["prompt"] for cfg in parent_configs)
        print(f"  Encoding {len(all_prompts)} unique prompts + 1 negative...")

        t0 = time.perf_counter()
        neg_cond = client.encode_prompt("")
        te_cache: dict[str, torch.Tensor] = {}
        for prompt in sorted(all_prompts):
            te_cache[prompt] = client.encode_prompt(prompt)
        timing["te_encode"] += time.perf_counter() - t0
        print(f"  Done ({time.perf_counter() - t0:.1f}s)")

        # Free TE, warmup diffusion
        client.free("te")

        precisions_needed = set(cfg["attention_backend"] for cfg in parent_configs)
        for variant in sorted(precisions_needed):
            print(f"  Warming up {variant}...")
            t0 = time.perf_counter()
            client.warmup(variant)
            timing["warmup"] += time.perf_counter() - t0

        # ---------------------------------------------------------------
        # Phase 2: Generate 10 parent i2i trajectories
        # ---------------------------------------------------------------
        print("\n=== Phase 2: Generate 10 parent i2i trajectories ===")

        # Pre-encode source images
        image_cache: dict[str, tuple[torch.Tensor, int, int]] = {}
        latent_cache: dict[str, torch.Tensor] = {}
        for img_file in SOURCE_IMAGES:
            image_path = i2i_dir / img_file
            image_tensor, w, h = load_i2i_image(image_path)
            image_cache[img_file] = (image_tensor, w, h)
            print(f"  VAE-encoding {img_file} ({w}x{h})...")
            t0 = time.perf_counter()
            latent_cache[img_file] = client.vae_encode(image_tensor)
            timing["vae_encode"] += time.perf_counter() - t0

        parent_traj_ids: list[int] = []

        with DatasetWriter(str(output_dir)) as writer:
            for i, cfg in enumerate(parent_configs):
                if _interrupted:
                    break

                img_file = cfg["image_file"]
                clean_latent = latent_cache[img_file]
                _, i2i_w, i2i_h = image_cache[img_file]

                print(f"  [{i+1}/10] seed={cfg['seed']} denoise={cfg['denoise']:.2f} "
                      f"steps={cfg['n_steps']} attn={cfg['attention_backend']} "
                      f"img={img_file[:20]}...", end="", flush=True)

                t0 = time.perf_counter()
                result = client.sample_trajectory(
                    pos_cond=te_cache[cfg["prompt"]],
                    neg_cond=neg_cond,
                    seed=cfg["seed"],
                    n_steps=cfg["n_steps"],
                    cfg=defs["cfg"],
                    width=i2i_w,
                    height=i2i_h,
                    attention_backend=cfg["attention_backend"],
                    sampling_shift=defs["sampling_shift"],
                    multiplier=defs["multiplier"],
                    save_steps=defs["save_steps"],
                    denoise=cfg["denoise"],
                    clean_latent=clean_latent,
                )
                dt = time.perf_counter() - t0
                timing["diffusion"] += dt
                print(f" {dt:.1f}s")

                v2_meta = {
                    "prompt": cfg["prompt"],
                    "prompt_idx": -1,
                    "seed": cfg["seed"],
                    "cfg": defs["cfg"],
                    "width": i2i_w,
                    "height": i2i_h,
                    "n_steps": cfg["n_steps"],
                    "attention_backend": cfg["attention_backend"],
                    "batch_type": "i2i",
                    "denoise": cfg["denoise"],
                    "image_file": img_file,
                    "is_gold": False,
                    "batch_idx": 0,
                    "packed": False,
                    "timing_seconds": dt,
                }
                traj_id = writer.add_trajectory(tensors=result, metadata=v2_meta)
                parent_traj_ids.append(traj_id)

            # Seal parent blob
            print("  Sealing parent blob...")
            writer.flush()

        if _interrupted:
            print("\nInterrupted during parent generation.")
            return 1

        print(f"  Parent traj_ids: {parent_traj_ids}")

        # ---------------------------------------------------------------
        # Phase 3: Sample steps and compute i2i2i params
        # ---------------------------------------------------------------
        print("\n=== Phase 3: Sample steps and compute i2i2i params ===")

        reader = DatasetReader(str(output_dir))
        child_configs = build_child_configs(parent_configs, parent_traj_ids, reader, rng)

        for i, ccfg in enumerate(child_configs):
            pcfg = parent_configs[i]
            print(f"  Parent {parent_traj_ids[i]:06d} -> child: "
                  f"step={ccfg['step_label']} sigma={ccfg['sigma_step']:.4f} "
                  f"denoise={ccfg['denoise']:.4f} "
                  f"steps={ccfg['n_steps']} attn={ccfg['attention_backend']}")

        # ---------------------------------------------------------------
        # Phase 4: Generate 10 i2i2i trajectories
        # ---------------------------------------------------------------
        print("\n=== Phase 4: Generate 10 i2i2i trajectories ===")

        # Re-encode any new prompts not in te_cache
        new_prompts = set(ccfg["prompt"] for ccfg in child_configs) - set(te_cache.keys())
        if new_prompts:
            print(f"  Re-encoding {len(new_prompts)} new prompts...")
            # Need TE back -- warmup will handle model swaps on server
            t0 = time.perf_counter()
            for prompt in sorted(new_prompts):
                te_cache[prompt] = client.encode_prompt(prompt)
            timing["te_encode"] += time.perf_counter() - t0

            # Free TE, re-warmup diffusion
            client.free("te")
            child_precisions = set(ccfg["attention_backend"] for ccfg in child_configs)
            for variant in sorted(child_precisions):
                print(f"  Re-warming up {variant}...")
                t0 = time.perf_counter()
                client.warmup(variant)
                timing["warmup"] += time.perf_counter() - t0

        child_traj_ids: list[int] = []

        with DatasetWriter(str(output_dir)) as writer:
            for i, ccfg in enumerate(child_configs):
                if _interrupted:
                    break

                # Load the intermediate step latent from parent
                parent_tid = ccfg["parent_traj_id"]
                step_label = ccfg["step_label"]
                _, accessor = reader[parent_tid]
                step_latent = accessor[step_label]  # (1, C, H, W)

                img_file = ccfg["image_file"]
                _, i2i_w, i2i_h = image_cache[img_file]

                print(f"  [{i+1}/10] parent={parent_tid:06d}:{step_label} "
                      f"seed={ccfg['seed']} denoise={ccfg['denoise']:.4f} "
                      f"steps={ccfg['n_steps']} attn={ccfg['attention_backend']}",
                      end="", flush=True)

                t0 = time.perf_counter()
                result = client.sample_trajectory(
                    pos_cond=te_cache[ccfg["prompt"]],
                    neg_cond=neg_cond,
                    seed=ccfg["seed"],
                    n_steps=ccfg["n_steps"],
                    cfg=defs["cfg"],
                    width=i2i_w,
                    height=i2i_h,
                    attention_backend=ccfg["attention_backend"],
                    sampling_shift=defs["sampling_shift"],
                    multiplier=defs["multiplier"],
                    save_steps=defs["save_steps"],
                    denoise=ccfg["denoise"],
                    clean_latent=step_latent,
                )
                dt = time.perf_counter() - t0
                timing["diffusion"] += dt
                print(f" {dt:.1f}s")

                v2_meta = {
                    "prompt": ccfg["prompt"],
                    "prompt_idx": -1,
                    "seed": ccfg["seed"],
                    "cfg": defs["cfg"],
                    "width": i2i_w,
                    "height": i2i_h,
                    "n_steps": ccfg["n_steps"],
                    "attention_backend": ccfg["attention_backend"],
                    "batch_type": "i2i2i",
                    "denoise": ccfg["denoise"],
                    "image_file": img_file,
                    "is_gold": False,
                    "batch_idx": 1,
                    "packed": False,
                    "timing_seconds": dt,
                    "parent_traj_id": ccfg["parent_traj_id"],
                    "parent_step": ccfg["parent_step"],
                    "parent_denoise": ccfg["parent_denoise"],
                }
                traj_id = writer.add_trajectory(tensors=result, metadata=v2_meta)
                child_traj_ids.append(traj_id)

            # Seal child blob
            print("  Sealing child blob...")
            writer.flush()

        reader.close()

        # ---------------------------------------------------------------
        # Phase 5: Verification summary
        # ---------------------------------------------------------------
        print("\n=== Phase 5: Verification summary ===")

        reader = DatasetReader(str(output_dir))
        print(f"\nTotal trajectories: {len(reader)}")
        print()

        # Print parent trajectories
        print("PARENT TRAJECTORIES (i2i):")
        print(f"  {'TID':>6}  {'Seed':>8}  {'Denoise':>7}  {'Steps':>5}  "
              f"{'Attn':>5}  {'Image':<30}")
        print(f"  {'-'*6}  {'-'*8}  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*30}")
        for tid in parent_traj_ids:
            meta, _ = reader[tid]
            print(f"  {tid:6d}  {meta['seed']:8d}  {meta['denoise']:7.3f}  "
                  f"{meta['n_steps']:5d}  {meta['attention_backend']:>5}  "
                  f"{(meta.get('image_file') or '')[:30]}")

        print()
        print("CHILD TRAJECTORIES (i2i2i):")
        print(f"  {'TID':>6}  {'Parent':>6}  {'PStep':>8}  {'Sigma':>8}  "
              f"{'Denoise':>7}  {'Steps':>5}  {'Attn':>5}")
        print(f"  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*5}  {'-'*5}")
        for i, tid in enumerate(child_traj_ids):
            meta, _ = reader[tid]
            ccfg = child_configs[i]
            print(f"  {tid:6d}  {meta.get('parent_traj_id', '?'):6}  "
                  f"{meta.get('parent_step', '?'):>8}  "
                  f"{ccfg['sigma_step']:8.4f}  "
                  f"{meta['denoise']:7.4f}  "
                  f"{meta['n_steps']:5d}  {meta['attention_backend']:>5}")

        reader.close()

        # Timing summary
        wall_total = time.perf_counter() - wall_start
        timing["overhead"] = wall_total - sum(timing.values())
        print(f"\n{'='*60}")
        print(f"  THROUGHPUT PROFILE (20 trajectories)")
        print(f"{'='*60}")
        print(f"  {'Phase':<20} {'Time (s)':>10} {'%':>6}")
        print(f"  {'-'*38}")
        for phase, t in sorted(timing.items(), key=lambda x: -x[1]):
            if t > 0.01:
                print(f"  {phase:<20} {t:>10.1f} {100*t/wall_total:>5.1f}%")
        print(f"  {'-'*38}")
        print(f"  {'TOTAL':<20} {wall_total:>10.1f}")
        print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

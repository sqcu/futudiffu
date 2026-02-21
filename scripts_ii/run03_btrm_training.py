r"""Run 03: BTRM training using src_ii library exclusively.

Two-phase training run:
  Phase 1: Generate mixed-resolution trajectories via inference server
  Phase 2: Train BTRM compound model (rtheta LoRA + ScoreUnembedder) using
           full differentiable forward with gradient checkpointing

All imports from src_ii/ or futudiffu/ (for InferenceClient, DatasetReader/Writer).
Zero imports from scripts/.

Usage:
  set PYTHONUNBUFFERED=1
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\run03_btrm_training.py 2>&1 | tee training_output/run03/run03_tee.log
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# src_ii imports ONLY (R1: no imports from scripts/)
# ---------------------------------------------------------------------------
from src_ii.bin_packer import (
    REFERENCE_SEQ_LEN,
    REFERENCE_TOTAL_LEN,
    BinPackScheduler,
    build_generation_plan,
    compute_seq_len,
)
from src_ii.btrm_model import BTRMCompoundModel
from src_ii.btrm_training import train_btrm_differentiable
from src_ii.dataset_generator import DatasetGenerationConfig, DatasetGenerator
from src_ii.pair_sampler import BTRMPairSampler, build_positions_from_v2
from src_ii.rendering import save_tensor_as_png
from src_ii.sigma_schedule import build_sigma_schedule, resolution_shift

# futudiffu imports (allowed: these are the core library, not scripts/)
from futudiffu.client import InferenceClient
from futudiffu.dataset_v2 import DatasetReader, DatasetWriter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = REPO_ROOT / "training_output" / "run03"
RENDER_DIR = OUTPUT_DIR / "renders"
DATASET_DIR = OUTPUT_DIR / "run03_dataset"
METRICS_PATH = OUTPUT_DIR / "btrm_metrics.jsonl"
CONFIG_PATH = OUTPUT_DIR / "run03_config.json"
SUMMARY_PATH = OUTPUT_DIR / "run03_summary.json"

# Existing V2 datasets to merge for training
EXISTING_V2_DATASETS = [
    REPO_ROOT / "btrm_dataset_v2",
    REPO_ROOT / "training_output" / "run02" / "run02_dataset",
]

# Model weights
FP8_WEIGHTS = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
VAE_WEIGHTS = r"F:\dox\ai\comfyui\ComfyUI\models\vae\zimage.safetensors"

SERVER_ENDPOINT = "tcp://localhost:5555"

# Prompts for new trajectory generation (subset of 24 built-in)
GENERATION_PROMPTS = [
    'ahem.\n*ting ting ting ting ting*\nthe query model for this is a LARGE LANGUAGE MODEL, specifically QWEN-3-4B, a GENERAL PURPOSE SEMANTIC PARSER which is able to WRITE SENTENCES AT A TIME when they are participating in dialogue. however, in this situation, they are being used as a hidden state generator to steer an *image generation model*, z-image.\n\nqwen-3-4b, draw me an "enormous laser shark for the sega saturn".',
    'qwen-3-4b, draw me a "gigantic laser shark breaching out of the ocean at sunset".',
    'A neon sign reading "OPEN 24 HOURS" above a rain-soaked Tokyo alleyway at night.',
    'A cat sitting on top of a stack of books next to a window with rain outside, warm interior lighting.',
    'Macro photograph of a dewdrop on a spider web, reflecting a tiny garden scene, shallow depth of field.',
    'An oil painting of a harbor at twilight, fishing boats with warm lanterns, impressionist style, visible brushstrokes.',
]

# Generation plan: 6 prompts x 2 tiers x 2 backends = 24 base trajectories
# Plus reduced-step variants for scrongle signal
RESOLUTION_TIERS = ["full", "medium"]
ATTENTION_BACKENDS = ["sdpa", "sage"]
N_STEPS_FULL = 30
N_STEPS_REDUCED = 10

# BTRM training hyperparameters
BTRM_LR = 3e-4
BTRM_GRAD_CLIP = 0.1
BTRM_N_STEPS = 100  # optimizer steps (doubled from run02's 30 macrobatches)
BTRM_WARMUP = 40
BTRM_GRAD_ACCUM = 2  # 2 microbatches per optimizer step
BTRM_CHECKPOINT_INTERVAL = 10

# Head names: scrimble (quantization) and scrongle (step count)
HEAD_NAMES = ("scrimble", "scrongle")
PREF_KEYS = ("scrimble_pref", "scrongle_pref")


def _make_preference_fn():
    """Build a preference function for the pair sampler.

    Scrimble: SDPA wins over SageAttention (quantization quality)
    Scrongle: 30-step wins over reduced-step (step count quality)

    Returns a function that takes a pair dict and returns preferences.
    """
    # We need access to the dataset metadata to determine preferences.
    # The pair sampler gives us (traj_a, step_a, traj_b, step_b).
    # We need to look up each trajectory's attention_backend and n_steps.
    _meta_cache = {}

    def _load_meta(reader, traj_id):
        if traj_id not in _meta_cache:
            meta, _ = reader[traj_id]
            _meta_cache[traj_id] = meta
        return _meta_cache[traj_id]

    def preference_fn(pair, reader):
        meta_a = _load_meta(reader, pair["traj_a"])
        meta_b = _load_meta(reader, pair["traj_b"])

        prefs = {}

        # Scrimble: SDPA > SageAttention (at same step count)
        backend_a = meta_a.get("attention_backend", "sdpa")
        backend_b = meta_b.get("attention_backend", "sdpa")
        if backend_a != backend_b:
            if backend_a == "sdpa" and backend_b == "sage":
                prefs["scrimble_pref"] = 1   # A wins (higher quality)
            elif backend_a == "sage" and backend_b == "sdpa":
                prefs["scrimble_pref"] = -1  # B wins
            else:
                prefs["scrimble_pref"] = 0
        else:
            prefs["scrimble_pref"] = 0  # same backend, no scrimble signal

        # Scrongle: more steps > fewer steps (at same backend)
        steps_a = meta_a.get("n_steps", 30)
        steps_b = meta_b.get("n_steps", 30)
        if steps_a != steps_b:
            if steps_a > steps_b:
                prefs["scrongle_pref"] = 1   # A has more steps, wins
            else:
                prefs["scrongle_pref"] = -1  # B has more steps, wins
        else:
            prefs["scrongle_pref"] = 0  # same steps, no scrongle signal

        return prefs

    return preference_fn, _meta_cache


def phase1_generate(client: InferenceClient) -> Path:
    """Phase 1: Generate mixed-resolution trajectories.

    Returns the path to the generated V2 dataset.
    """
    print("=" * 60)
    print("  PHASE 1: TRAJECTORY GENERATION")
    print("=" * 60)

    t0 = time.perf_counter()

    # Generate full-step trajectories (SDPA + Sage, full + medium res)
    config_full = DatasetGenerationConfig(
        prompts=GENERATION_PROMPTS,
        resolution_tiers=RESOLUTION_TIERS,
        attention_backends=ATTENTION_BACKENDS,
        n_steps=N_STEPS_FULL,
        cfg=4.0,
        output_dir=str(DATASET_DIR),
        run_name="run03_full",
        base_model_hash="z_image_v1",
        sparse_steps=[0, 4, 9, 14, 19, 24, 29],
        server_endpoint=SERVER_ENDPOINT,
        flush_interval=50,
        render_count=0,  # We'll render separately after generation
        source_device="rtx4090_0",
    )

    generator_full = DatasetGenerator(config_full, client)
    summary_full = generator_full.generate_all()

    # Now generate reduced-step trajectories (for scrongle signal)
    # Only SDPA at full resolution, 10 steps
    config_reduced = DatasetGenerationConfig(
        prompts=GENERATION_PROMPTS[:4],  # Use first 4 prompts for reduced
        resolution_tiers=["full"],
        attention_backends=["sdpa"],
        n_steps=N_STEPS_REDUCED,
        cfg=4.0,
        output_dir=str(DATASET_DIR),  # Same dataset dir (append)
        run_name="run03_reduced",
        base_model_hash="z_image_v1",
        sparse_steps=[0, 4, 9],  # Fewer sparse steps for 10-step traj
        server_endpoint=SERVER_ENDPOINT,
        flush_interval=50,
        render_count=0,
        source_device="rtx4090_0",
    )

    generator_reduced = DatasetGenerator(config_reduced, client)
    summary_reduced = generator_reduced.generate_all()

    gen_time = time.perf_counter() - t0

    # Save generation metrics
    gen_metrics = {
        "full_step": summary_full,
        "reduced_step": summary_reduced,
        "total_time_s": gen_time,
    }
    gen_metrics_path = OUTPUT_DIR / "generation_metrics.json"
    gen_metrics_path.write_text(json.dumps(gen_metrics, indent=2, default=str))

    print(f"\nPhase 1 complete: {gen_time:.1f}s total")
    return DATASET_DIR


def phase1_render(client: InferenceClient, dataset_dir: Path):
    """Render final latents from generated trajectories as PNGs."""
    print("\n" + "=" * 60)
    print("  PHASE 1b: RENDERING")
    print("=" * 60)

    reader = DatasetReader(str(dataset_dir))
    n_traj = len(reader)
    print(f"Rendering {n_traj} trajectories from {dataset_dir}")

    RENDER_DIR.mkdir(parents=True, exist_ok=True)

    rendered = 0
    t0 = time.perf_counter()

    for traj_id in range(n_traj):
        if traj_id not in reader:
            continue
        meta, accessor = reader[traj_id]

        # Get the final latent
        if "final" not in accessor.available_steps:
            print(f"  traj {traj_id:06d}: no final latent, skipping")
            continue

        final_latent = accessor["final"]

        # VAE decode via server
        decoded = client.vae_decode(final_latent)

        # Build filename from metadata
        w = meta.get("width", 0)
        h = meta.get("height", 0)
        backend = meta.get("attention_backend", "unknown")
        steps = meta.get("n_steps", 0)
        seed = meta.get("seed", 0)
        fname = f"run03_w{w}h{h}_s{steps}_{backend}_seed{seed}.png"

        save_tensor_as_png(decoded, RENDER_DIR / fname)
        rendered += 1

    render_time = time.perf_counter() - t0
    print(f"  Rendered {rendered} images in {render_time:.1f}s")


def phase2_train(client: InferenceClient, gen_dataset_dir: Path):
    """Phase 2: BTRM training with full differentiable forward.

    Uses BTRMCompoundModel (prevents Defect 24) and
    train_btrm_differentiable (gradient-checkpointed forward through LoRA).
    """
    import torch

    print("\n" + "=" * 60)
    print("  PHASE 2: BTRM TRAINING")
    print("=" * 60)

    t0_phase = time.perf_counter()

    # ---------------------------------------------------------------
    # 2a: Merge datasets for training
    # ---------------------------------------------------------------
    print("\n--- Merging datasets ---")

    # Collect all available V2 datasets
    all_readers = []
    total_trajs = 0

    for ds_path in EXISTING_V2_DATASETS:
        if ds_path.exists() and (ds_path / "index.parquet").exists():
            r = DatasetReader(str(ds_path))
            n = len(r)
            print(f"  {ds_path.name}: {n} trajectories")
            all_readers.append((r, ds_path))
            total_trajs += n

    # Add run03's own generated data
    if gen_dataset_dir.exists() and (gen_dataset_dir / "index.parquet").exists():
        r = DatasetReader(str(gen_dataset_dir))
        n = len(r)
        print(f"  run03_dataset: {n} trajectories")
        all_readers.append((r, gen_dataset_dir))
        total_trajs += n

    print(f"  Total available: {total_trajs} trajectories")

    # Use the largest dataset as primary reader (for pair sampling)
    # We'll build positions from all datasets
    all_positions = []
    all_traj_meta = {}  # global_id -> (reader, local_traj_id, meta)
    global_id_offset = 0

    for reader, ds_path in all_readers:
        import pyarrow.parquet as pq

        # Apply dataset filters
        from src_ii.dataset_filters import filter_training_trajectories

        raw_table = pq.read_table(str(ds_path / "index.parquet"))

        # Only apply filter if run_name column exists
        if "run_name" in raw_table.column_names:
            filtered = filter_training_trajectories(raw_table)
        else:
            filtered = raw_table

        traj_ids = filtered.column("traj_id").to_pylist()

        # Build positions from this reader
        positions = build_positions_from_v2(
            reader, traj_ids=traj_ids,
        )

        # Remap positions to global IDs
        local_to_global = {}
        for pos in positions:
            if pos.traj_id not in local_to_global:
                gid = global_id_offset + pos.traj_id
                local_to_global[pos.traj_id] = gid
                # Cache metadata
                meta, accessor = reader[pos.traj_id]
                all_traj_meta[gid] = {
                    "reader": reader,
                    "local_traj_id": pos.traj_id,
                    "meta": meta,
                    "accessor": accessor,
                }
            pos.traj_id = local_to_global[pos.traj_id]

        all_positions.extend(positions)
        global_id_offset += 10000  # Large offset to avoid collisions

    print(f"  Positions for sampling: {len(all_positions)} across {len(all_traj_meta)} trajectories")

    # ---------------------------------------------------------------
    # 2b: Build pair sampler
    # ---------------------------------------------------------------
    print("\n--- Building pair sampler ---")

    sampler = BTRMPairSampler(
        positions=all_positions,
        allow_inter_trajectory=True,
        allow_intra_trajectory=False,
        rng_seed=42,
    )
    print(f"  Pair space: {sampler.pair_space_size:,} possible pairs")
    print(f"  Trajectories: {sampler.n_trajectories}")
    print(f"  Positions: {sampler.n_positions}")

    # ---------------------------------------------------------------
    # 2c: Build preference function
    # ---------------------------------------------------------------
    print("\n--- Building preference function ---")

    # The preference function uses trajectory metadata to determine
    # which image is "better" for each head.
    def preference_fn(pair):
        """Compute pairwise preferences from trajectory metadata.

        Scrimble: SDPA > SageAttention (quantization quality)
        Scrongle: more steps > fewer steps (step count quality)
        """
        gid_a = pair["traj_a"]
        gid_b = pair["traj_b"]

        meta_a = all_traj_meta[gid_a]["meta"]
        meta_b = all_traj_meta[gid_b]["meta"]

        prefs = {}

        # Scrimble: SDPA wins over SageAttention
        backend_a = meta_a.get("attention_backend", "sdpa")
        backend_b = meta_b.get("attention_backend", "sdpa")
        if backend_a == "sdpa" and backend_b == "sage":
            prefs["scrimble_pref"] = 1
        elif backend_a == "sage" and backend_b == "sdpa":
            prefs["scrimble_pref"] = -1
        else:
            prefs["scrimble_pref"] = 0

        # Scrongle: more steps wins
        steps_a = meta_a.get("n_steps", 30)
        steps_b = meta_b.get("n_steps", 30)
        if steps_a > steps_b:
            prefs["scrongle_pref"] = 1
        elif steps_a < steps_b:
            prefs["scrongle_pref"] = -1
        else:
            prefs["scrongle_pref"] = 0

        return prefs

    # ---------------------------------------------------------------
    # 2d: Load backbone model directly (not via server)
    # ---------------------------------------------------------------
    print("\n--- Loading backbone model ---")

    from src_ii.model_loading import load_fp8_diffusion_model

    # Load WITHOUT compilation (training uses forward_checkpointed, not diff_compiled)
    # This avoids Defect R2-03 (inductor SymPy recursion with LoRA)
    _, diff_model = load_fp8_diffusion_model(
        FP8_WEIGHTS,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        compile_model=False,
        fuse=True,
    )

    # ---------------------------------------------------------------
    # 2e: Create BTRMCompoundModel (R2: prevents Defect 24)
    # ---------------------------------------------------------------
    print("\n--- Creating BTRM compound model ---")

    btrm_model = BTRMCompoundModel(
        diff_model,
        adapter_name="rtheta",
        adapter_rank=8,
        adapter_alpha=16.0,
        adapter_init_b_std=0.01,
        head_names=HEAD_NAMES,
        hidden_dim=3840,
        logit_cap=10.0,
    )

    # Verify adapter params are in the model
    adapter_params = btrm_model.adapter_params()
    head_params = btrm_model.head_params()
    all_params = btrm_model.all_trainable_params()

    n_adapter = sum(p.numel() for p in adapter_params)
    n_head = sum(p.numel() for p in head_params)
    print(f"  Adapter params: {n_adapter:,}")
    print(f"  Head params: {n_head:,}")
    print(f"  Total trainable: {n_adapter + n_head:,}")

    # ---------------------------------------------------------------
    # 2f: Build load_latent_fn
    # ---------------------------------------------------------------
    print("\n--- Preparing latent loader ---")

    # Pre-encode prompts for conditioning
    prompt_cache = {}
    neg_cond = None

    # Collect unique prompts from all trajectories
    unique_prompts = set()
    for gid, info in all_traj_meta.items():
        prompt = info["meta"].get("prompt", "")
        if prompt:
            unique_prompts.add(prompt)

    print(f"  {len(unique_prompts)} unique prompts to encode")

    # Encode prompts via server
    neg_cond = client.encode_prompt("")
    for prompt in unique_prompts:
        prompt_cache[prompt] = client.encode_prompt(prompt)

    # Free text encoder on server
    client.free("te")
    print(f"  Prompts encoded, TE freed")

    def load_latent_fn(key):
        """Load a latent + conditioning for BTRM training.

        key is a (traj_id, step_key) tuple (from pair sampler).
        Returns: (latent, timestep, conditioning, num_tokens, rope_cache)
        """
        traj_id, step_key = key
        info = all_traj_meta[traj_id]
        meta = info["meta"]
        accessor = info["accessor"]

        # Load latent
        latent = accessor[step_key].to(device="cuda", dtype=torch.bfloat16)

        # Get sigma for this step
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

        # Determine sigma from step_key
        if step_key == "final":
            sigma_val = float(sigmas[-2].item()) if len(sigmas) > 1 else 0.01
        else:
            step_idx = int(step_key.split("_")[1])
            if step_idx < len(sigmas):
                sigma_val = float(sigmas[step_idx].item())
            else:
                sigma_val = 0.01

        timestep = torch.tensor([sigma_val], device="cuda", dtype=torch.bfloat16)

        # Get conditioning
        prompt = meta.get("prompt", "")
        if prompt in prompt_cache:
            cond = prompt_cache[prompt].to(device="cuda")
        else:
            # Fallback: use negative conditioning
            cond = neg_cond.to(device="cuda")

        # Build RoPE cache
        from src_ii.rollout import make_rope_cache

        latent_h = meta.get("latent_height") or (meta.get("height", 832) // 8)
        latent_w = meta.get("latent_width") or (meta.get("width", 1280) // 8)
        num_tokens = cond.shape[1]

        rope_cache = make_rope_cache(
            diff_model, latent_h, latent_w, num_tokens,
            device=torch.device("cuda"),
        )

        return latent, timestep, cond, num_tokens, rope_cache

    # ---------------------------------------------------------------
    # 2g: Run training
    # ---------------------------------------------------------------
    print("\n--- Starting BTRM training ---")
    print(f"  Steps: {BTRM_N_STEPS}")
    print(f"  LR: {BTRM_LR}")
    print(f"  Grad clip: {BTRM_GRAD_CLIP}")
    print(f"  Grad accum: {BTRM_GRAD_ACCUM}")
    print(f"  Warmup: {BTRM_WARMUP}")
    print(f"  Heads: {HEAD_NAMES}")

    # Open metrics file for streaming writes
    metrics_file = open(METRICS_PATH, "w")

    def training_callback(step, entry):
        """Write each training step's metrics to JSONL."""
        record = {
            "phase": "btrm",
            "step": step,
            **entry,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        metrics_file.write(json.dumps(record, default=str) + "\n")
        metrics_file.flush()

        # Checkpoint at intervals
        if (step + 1) % BTRM_CHECKPOINT_INTERVAL == 0:
            ckpt_dir = OUTPUT_DIR / f"btrm_ckpt_{step + 1:04d}"
            btrm_model.persist(str(ckpt_dir))
            print(f"    [checkpoint] saved to {ckpt_dir}")

    training_curve = train_btrm_differentiable(
        btrm_model=btrm_model,
        pair_sampler=sampler,
        load_latent_fn=load_latent_fn,
        preference_fn=preference_fn,
        n_steps=BTRM_N_STEPS,
        lr=BTRM_LR,
        logsquare_weight=0.05,
        head_names=HEAD_NAMES,
        pref_keys=PREF_KEYS,
        gradient_checkpointing=True,
        max_grad_norm=BTRM_GRAD_CLIP,
        log_interval=5,
        callback=training_callback,
        warmup_steps=BTRM_WARMUP,
        grad_accum_steps=BTRM_GRAD_ACCUM,
    )

    metrics_file.close()

    train_time = time.perf_counter() - t0_phase
    print(f"\nPhase 2 complete: {train_time:.1f}s")

    # ---------------------------------------------------------------
    # 2h: Save final weights
    # ---------------------------------------------------------------
    print("\n--- Saving final weights ---")
    final_dir = OUTPUT_DIR / "final"
    manifest = btrm_model.persist(str(final_dir))

    # Print sampler stats
    stats = sampler.stats()
    print(f"\nSampler stats: {json.dumps(stats, indent=2)}")

    # ---------------------------------------------------------------
    # 2i: Training summary
    # ---------------------------------------------------------------
    if training_curve:
        first = training_curve[0]
        last = training_curve[-1]
        print(f"\n--- Training Summary ---")
        print(f"  Loss: {first['loss']:.4f} -> {last['loss']:.4f}")
        for name in HEAD_NAMES:
            k = f"accuracy_{name}"
            print(f"  {name}: {first.get(k, 0):.3f} -> {last.get(k, 0):.3f}")
        print(f"  Grad norm: {first.get('pre_clip_grad_norm', 0):.2f} -> {last.get('pre_clip_grad_norm', 0):.2f}")
        print(f"  Total time: {train_time:.1f}s")

    return training_curve, train_time


def main():
    wall_start = time.perf_counter()

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RENDER_DIR.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        "run_name": "run03",
        "generation_prompts": len(GENERATION_PROMPTS),
        "resolution_tiers": RESOLUTION_TIERS,
        "attention_backends": ATTENTION_BACKENDS,
        "n_steps_full": N_STEPS_FULL,
        "n_steps_reduced": N_STEPS_REDUCED,
        "btrm_lr": BTRM_LR,
        "btrm_grad_clip": BTRM_GRAD_CLIP,
        "btrm_n_steps": BTRM_N_STEPS,
        "btrm_warmup": BTRM_WARMUP,
        "btrm_grad_accum": BTRM_GRAD_ACCUM,
        "btrm_checkpoint_interval": BTRM_CHECKPOINT_INTERVAL,
        "head_names": list(HEAD_NAMES),
        "existing_datasets": [str(p) for p in EXISTING_V2_DATASETS],
        "start_time": datetime.now(timezone.utc).isoformat(),
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2))

    # Connect to server
    print("Connecting to inference server...")
    client = InferenceClient(SERVER_ENDPOINT, timeout_ms=0)

    try:
        status = client.status()
        print(f"Server status: {status.get('loaded_models', [])}, "
              f"VRAM: {status.get('vram_allocated_gb', '?')}GB")
    except Exception as e:
        print(f"Cannot connect to server: {e}")
        print("Start the server first.")
        return 1

    # ---------------------------------------------------------------
    # Phase 1: Generate trajectories
    # ---------------------------------------------------------------
    gen_dataset_dir = phase1_generate(client)

    # Render all generated trajectories
    phase1_render(client, gen_dataset_dir)

    # ---------------------------------------------------------------
    # Phase 2: BTRM Training
    # ---------------------------------------------------------------
    # Free diffusion model on server before loading locally
    client.free("diffusion")

    training_curve, train_time = phase2_train(client, gen_dataset_dir)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    wall_total = time.perf_counter() - wall_start

    summary = {
        "run_name": "run03",
        "wall_total_s": wall_total,
        "train_time_s": train_time,
        "n_training_steps": BTRM_N_STEPS,
        "end_time": datetime.now(timezone.utc).isoformat(),
    }
    if training_curve:
        summary["final_loss"] = training_curve[-1]["loss"]
        for name in HEAD_NAMES:
            k = f"accuracy_{name}"
            summary[f"final_{k}"] = training_curve[-1].get(k, 0)

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n{'=' * 60}")
    print(f"  RUN 03 COMPLETE")
    print(f"  Wall time: {wall_total:.1f}s ({wall_total/60:.1f} min)")
    print(f"  Artifacts: {OUTPUT_DIR}")
    print(f"{'=' * 60}")

    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

r"""Differentiable BTRM training for PINKIFY and THISNOTTHAT reward heads (v5).

v5 changes from v4:
  - N_STEPS = 170 (stop before Phase 3 instability onset at ~131)
  - Cosine LR decay after warmup (peak LR 3e-4 -> 0 at step 170)
  - Checkpoints every 50 steps (steps 50, 100, 150, and final)
  - Output to differentiable_run_v5/

v4 showed three-phase dynamics:
  Phase 1 (0-36): warmup + early descent, grad norms ~1.6
  Phase 2 (37-130): aggressive learning, loss sub-0.10
  Phase 3 (131+): instability, grad norms up to 83.1, crashed at step 197
The cosine schedule reduces effective LR as the model enters the
overfit-prone region. With LR decaying toward 0 by step 170, the gradient
explosions from Phase 3 should be attenuated.

Uses the full differentiable forward path (train_btrm_differentiable), which
flows gradients through the backbone via the r_theta LoRA adapter. This is
the CORRECT path for adapter training -- NOT the detached path (train_btrm)
which gives the adapter zero meaningful gradients.

On-the-fly GPU scoring: the preference function VAE-decodes each latent and
scores with pinkify_score_gpu / thisnotthat_score_gpu (<2ms each). The VAE
stays resident alongside the backbone (~160 MB vs ~6 GB backbone). No
pre-computed score cache needed.

Phases:
  1. Load V2 dataset + VAE + reference images
  2. Build pair sampler + on-the-fly preference function
  3. Validate challenge set ranking (A < B < C, D ~ E, {A,B,C} < {D,E} < F)
  4. Encode prompts with text encoder (then free TE)
  5. Load FP8 backbone + create BTRMCompoundModel
  6. Run train_btrm_differentiable() with gradient checkpointing + cosine LR
  7. Verify adapter gradients after macrobatch 1
  8. Pre-persist test scores
  9. Persist trained compound model + metrics

Execution:
  PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe \
      F:\dox\repos\ai\futudiffu\scripts_ii\train_pinkify_differentiable.py

Output:
  pinkify_thisnotthat_output/differentiable_run_v5/rtheta_adapter.safetensors
  pinkify_thisnotthat_output/differentiable_run_v5/btrm_head.safetensors
  pinkify_thisnotthat_output/differentiable_run_v5/btrm_compound_config.json
  pinkify_thisnotthat_output/differentiable_run_v5/training_metrics.jsonl
  pinkify_thisnotthat_output/differentiable_run_v5/pre_persist_scores.json
  pinkify_thisnotthat_output/differentiable_run_v5/run_summary.json
  pinkify_thisnotthat_output/differentiable_run_v5/checkpoint_step050/
  pinkify_thisnotthat_output/differentiable_run_v5/checkpoint_step100/
  pinkify_thisnotthat_output/differentiable_run_v5/checkpoint_step150/
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Model weights
FP8_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
TE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"
VAE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\vae\zimage.safetensors"
TOKENIZER_PATH = str(REPO_ROOT / "src" / "futudiffu" / "tokenizer")

# Reference images for THISNOTTHAT
THIS_PATH = REPO_ROOT / "i2i_off_policies" / "pizza-ratto.png"
THAT_PATH = REPO_ROOT / "i2i_off_policies" / "offhand_pleometric.png"

# Dataset
V2_DATASET_DIR = REPO_ROOT / "btrm_dataset_v2"

# Output (v5 -- 170 steps with cosine LR decay + checkpoints)
OUTPUT_DIR = REPO_ROOT / "pinkify_thisnotthat_output" / "differentiable_run_v5"
METRICS_PATH = OUTPUT_DIR / "training_metrics.jsonl"
PRE_PERSIST_SCORES_PATH = OUTPUT_DIR / "pre_persist_scores.json"
SUMMARY_PATH = OUTPUT_DIR / "run_summary.json"

# Training hyperparameters
# v5: 170 steps (stop before Phase 3 instability at ~131, with margin to
# EMA loss minimum at step 166 from v4 analysis). Cosine LR decay from
# peak after warmup to 0 at step 170.
N_STEPS = 170          # optimizer steps (macrobatches) -- reduced from 200
LR = 3e-4
GRAD_CLIP = 0.1
# LOGSQUARE_WEIGHT removed -- logsquare regularizer has been removed from
# train_btrm_differentiable. Total loss = BT loss only. The ScoreUnembedder's
# soft_tanh_cap(10.0) already bounds score magnitudes without imposing a
# target magnitude. See docs/directive_remove_logsquare_regularizer.md.
WARMUP_STEPS = 10      # same warmup as v3/v4
GRAD_ACCUM = 1         # 1 microbatch per optimizer step (tight VRAM)
LR_SCHEDULE = "warmup_cosine"  # cosine decay from peak LR to 0 at N_STEPS
CHECKPOINT_STEPS = [50, 100, 150]  # save adapter state at these steps

HEAD_NAMES = ("pinkify", "thisnotthat")
PREF_KEYS = ("pinkify_pref", "thisnotthat_pref")


def main():
    wall_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # ===================================================================
    # Phase 1: Load V2 dataset + VAE + reference images
    # ===================================================================
    print("=" * 60)
    print("  Phase 1: Loading V2 dataset, VAE, and reference images")
    print("=" * 60)

    from futudiffu.dataset_v2 import DatasetReader
    from src_ii.pair_sampler import BTRMPairSampler, build_positions_from_v2
    from src_ii.flops_sampling import compute_flops_sampling_weights_from_positions
    from src_ii.vae_utils import load_vae, decode_latent_to_pil
    from src_ii.reward_functions import (
        pinkify_score_gpu,
        thisnotthat_score_gpu,
        pairwise_preference,
        _pil_to_tensor,
    )
    from PIL import Image
    import torch.nn.functional as F

    reader = DatasetReader(str(V2_DATASET_DIR))
    traj_ids = reader._table.column("traj_id").to_pylist()
    print(f"  V2 dataset: {len(reader)} trajectories")

    # Load VAE -- stays resident throughout training (~160 MB)
    vae = load_vae(VAE_PATH, device=device, dtype=dtype)
    vram_after_vae = torch.cuda.memory_allocated() / 1e9
    print(f"  VAE loaded and resident. VRAM: {vram_after_vae:.2f} GB")

    # Load reference images for THISNOTTHAT -- kept on GPU
    this_ref_pil = Image.open(str(THIS_PATH)).convert("RGB")
    that_ref_pil = Image.open(str(THAT_PATH)).convert("RGB")
    this_ref_t = _pil_to_tensor(this_ref_pil, device)  # (1, 3, H, W) float32 on GPU
    that_ref_t = _pil_to_tensor(that_ref_pil, device)
    print(f"  THIS ref: {THIS_PATH.name} ({this_ref_pil.size})")
    print(f"  THAT ref: {THAT_PATH.name} ({that_ref_pil.size})")

    # ===================================================================
    # Phase 2: Build pair sampler + on-the-fly preference function
    # ===================================================================
    print("\n" + "=" * 60)
    print("  Phase 2: Building pair sampler and on-the-fly preference function")
    print("=" * 60)

    # Build positions from V2 dataset (all trajectories)
    positions = build_positions_from_v2(reader, traj_ids=traj_ids)
    print(f"  Positions: {len(positions)} across {len(traj_ids)} trajectories")

    # Compute FLOPS sampling weights for resolution-aware training
    # (Layer 3: funfetti batching PDF). Even with 96% monoresolution data,
    # the code path is exercised and degrades gracefully to uniform.
    flops_weights = compute_flops_sampling_weights_from_positions(positions)
    print(f"  FLOPS weights: {len(flops_weights)} trajectories weighted")

    sampler = BTRMPairSampler(
        positions=positions,
        allow_inter_trajectory=True,
        allow_intra_trajectory=False,
        rng_seed=42,
        flops_weights=flops_weights,
    )
    print(f"  Pair space: {sampler.pair_space_size:,} possible pairs")

    # Cache metadata + accessor per trajectory for fast access
    _meta_cache = {}

    def _get_meta(traj_id):
        if traj_id not in _meta_cache:
            meta, accessor = reader[traj_id]
            _meta_cache[traj_id] = (meta, accessor)
        return _meta_cache[traj_id]

    # Score counters for observability
    _score_calls = [0]
    _score_time = [0.0]

    def _score_latent(traj_id: int, step_key: str) -> dict[str, float]:
        """VAE-decode a latent and score it with both reward functions.

        Returns dict with 'pinkify' and 'thisnotthat' scores.
        All computation happens on GPU under torch.no_grad().
        """
        t0 = time.perf_counter()

        meta, accessor = _get_meta(traj_id)
        latent = accessor[step_key].to(device=device, dtype=dtype)

        # VAE decode to pixel space (returns PIL Image)
        pil_img = decode_latent_to_pil(vae, latent, device=device, dtype=dtype)

        # Convert PIL to GPU tensor for scoring
        import numpy as np
        rgb = pil_img.convert("RGB")
        arr = np.array(rgb, dtype=np.float32) / 255.0  # (H, W, 3)
        img_t = torch.from_numpy(arr).permute(2, 0, 1).to(device)  # (3, H, W)

        with torch.no_grad():
            pink_score = pinkify_score_gpu(img_t).item()

            # Resize refs to match image resolution for thisnotthat
            _, H, W = img_t.shape
            this_resized = F.interpolate(
                this_ref_t.float(), size=(H, W), mode='bilinear', align_corners=False,
            )
            that_resized = F.interpolate(
                that_ref_t.float(), size=(H, W), mode='bilinear', align_corners=False,
            )
            tnt_score = thisnotthat_score_gpu(img_t, this_resized, that_resized).item()

        _score_calls[0] += 1
        _score_time[0] += time.perf_counter() - t0

        return {"pinkify": pink_score, "thisnotthat": tnt_score}

    # On-the-fly preference function: no pre-computed cache
    def preference_fn(pair: dict) -> dict:
        """Compute pairwise preferences via on-the-fly GPU scoring.

        For each head (pinkify, thisnotthat): higher score wins.
        Returns {pref_key: +1/-1/0} for each head.
        """
        scores_a = _score_latent(pair["traj_a"], pair["step_a"])
        scores_b = _score_latent(pair["traj_b"], pair["step_b"])

        prefs = {}
        for head_name, pref_key in zip(HEAD_NAMES, PREF_KEYS):
            prefs[pref_key] = pairwise_preference(
                scores_a[head_name], scores_b[head_name], margin=0.0,
            )
        return prefs

    # Validate preference function with a test sample
    test_pair = sampler.sample_pair()
    test_prefs = preference_fn(test_pair)
    print(f"  Test pair: traj_a={test_pair['traj_a']}, step_a={test_pair['step_a']}, "
          f"traj_b={test_pair['traj_b']}, step_b={test_pair['step_b']}")
    print(f"  Test prefs: {test_prefs}")
    print(f"  Scoring time: {_score_time[0] * 1000:.1f}ms for {_score_calls[0]} calls "
          f"({_score_time[0] / max(_score_calls[0], 1) * 1000:.1f}ms/call)")

    # ===================================================================
    # Phase 3: Validate challenge set ranking
    # ===================================================================
    print("\n" + "=" * 60)
    print("  Phase 3: Validating challenge set ranking")
    print("=" * 60)

    # Challenge set: reference images from i2i_off_policies/PINKIFY_cases/
    # Expected ranking: A < B < C, D ~ E, {A,B,C} < {D,E} < F
    challenge_dir = REPO_ROOT / "i2i_off_policies" / "PINKIFY_cases"
    # Try both naming conventions: "A.png" and "PINKER_A.png"
    challenge_labels = ["A", "B", "C", "D", "E", "F"]
    challenge_scores = {}

    if challenge_dir.exists():
        import numpy as np
        for label in challenge_labels:
            # Try both naming conventions
            candidates = [
                challenge_dir / f"{label}.png",
                challenge_dir / f"PINKER_{label}.png",
            ]
            img_path = None
            for c in candidates:
                if c.exists():
                    img_path = c
                    break

            if img_path is not None:
                pil = Image.open(str(img_path)).convert("RGB")
                arr = np.array(pil, dtype=np.float32) / 255.0
                img_t = torch.from_numpy(arr).permute(2, 0, 1).to(device)
                with torch.no_grad():
                    ps = pinkify_score_gpu(img_t).item()
                challenge_scores[label] = ps
                print(f"    {img_path.name}: pinkify={ps:.6f}")

        if len(challenge_scores) == 6:
            # Verify ranking: A < B < C, D ~ E, {A,B,C} < {D,E} < F
            A, B, C = challenge_scores["A"], challenge_scores["B"], challenge_scores["C"]
            D, E, F_ = challenge_scores["D"], challenge_scores["E"], challenge_scores["F"]

            checks = [
                ("A < B", A < B),
                ("B < C", B < C),
                ("C < D (or C < E)", C < D or C < E),
                ("D ~ E (|D-E|/max < 0.5)", abs(D - E) / max(abs(D), abs(E), 1e-10) < 0.5),
                ("max(D,E) < F", max(D, E) < F_),
            ]
            all_pass = True
            for label, ok in checks:
                status = "PASS" if ok else "FAIL"
                if not ok:
                    all_pass = False
                print(f"    [{status}] {label}")

            if all_pass:
                print("  Challenge set ranking PRESERVED.")
            else:
                print("  WARNING: Challenge set ranking has deviations.")
        else:
            print(f"  Only found {len(challenge_scores)}/6 challenge images -- skipping ranking check")
    else:
        print(f"  Challenge dir not found: {challenge_dir}")
        print("  Skipping challenge set validation")

    # ===================================================================
    # Phase 4: Encode prompts with text encoder
    # ===================================================================
    print("\n" + "=" * 60)
    print("  Phase 4: Encoding prompts")
    print("=" * 60)

    from futudiffu.text_encoder import create_tokenizer, load_text_encoder, encode_prompt

    tokenizer = create_tokenizer(TOKENIZER_PATH)
    te_model = load_text_encoder(TE_PATH, device=device, dtype=dtype)

    # Collect unique prompts from all trajectories
    prompt_cache = {}
    for traj_id in traj_ids:
        meta, _ = reader[traj_id]
        prompt = meta.get("prompt", "")
        if prompt and prompt not in prompt_cache:
            prompt_cache[prompt] = None

    print(f"  {len(prompt_cache)} unique prompts to encode")

    for prompt in prompt_cache:
        cond = encode_prompt(te_model, tokenizer, prompt, device=device)
        prompt_cache[prompt] = cond.cpu()

    del te_model, tokenizer
    torch.cuda.empty_cache()
    print(f"  TE freed. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ===================================================================
    # Phase 5: Load FP8 backbone + create BTRMCompoundModel
    # ===================================================================
    print("\n" + "=" * 60)
    print("  Phase 5: Loading backbone and creating compound model")
    print("=" * 60)

    from src_ii.zimage_model import load_zimage_rlaif
    from src_ii.btrm_lifecycle import setup_btrm_training, persist_btrm, score_serial
    from src_ii.multi_lora import get_adapter_params
    from src_ii.sigma_schedule import build_sigma_schedule, resolution_shift

    # Load WITHOUT compilation -- training uses gradient-checkpointed forward
    _, raw_model = load_zimage_rlaif(
        FP8_PATH, device=device, dtype=dtype,
        compile_model=False, fuse=True,
    )

    print(f"  VRAM after backbone load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Set up BTRM training (installs adapter + score head + optimizer)
    optimizer = setup_btrm_training(
        raw_model,
        adapter_name="rtheta",
        adapter_rank=8,
        adapter_alpha=16.0,
        adapter_init_b_std=0.01,
    )

    # Report parameter counts
    adapter_params_dict = get_adapter_params(raw_model, "rtheta")
    n_adapter = sum(p.numel() for p in adapter_params_dict.values())
    n_head = sum(p.numel() for p in raw_model.score_proj.parameters()) + \
             sum(p.numel() for p in raw_model.score_norm.parameters())
    print(f"  Adapter params: {n_adapter:,}")
    print(f"  Head params: {n_head:,}")
    print(f"  Total trainable: {n_adapter + n_head:,}")
    print(f"  VRAM after BTRM setup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Confirm VAE is still resident
    print(f"  VAE still resident: {next(vae.parameters()).device}")

    # ===================================================================
    # Phase 6: Build load_latent_fn
    # ===================================================================
    print("\n" + "=" * 60)
    print("  Phase 6: Preparing latent loader")
    print("=" * 60)

    def load_latent_fn(key):
        """Load a latent + conditioning for BTRM training.

        key is a (traj_id, step_key) tuple (from pair sampler).
        Returns: (latent, timestep, conditioning, num_tokens)
        """
        traj_id, step_key = key
        meta, accessor = _get_meta(traj_id)

        # Load latent
        latent = accessor[step_key].to(device=device, dtype=dtype)

        # Compute sigma for this step
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
            if step_idx < len(sigmas):
                sigma_val = float(sigmas[step_idx].item())
            else:
                sigma_val = 0.01

        timestep = torch.tensor([sigma_val], device=device, dtype=dtype)

        # Get conditioning from prompt cache
        prompt = meta.get("prompt", "")
        cond = prompt_cache.get(prompt)
        if cond is None:
            raise ValueError(f"No cached prompt encoding for traj {traj_id}: '{prompt[:60]}...'")
        cond = cond.to(device=device, dtype=dtype)

        num_tokens = cond.shape[1]

        return latent, timestep, cond, num_tokens

    # Validate load_latent_fn with a test load
    test_key = (traj_ids[0], "step_00")
    lat, ts, cond, nt = load_latent_fn(test_key)
    print(f"  Test load: latent={lat.shape}, timestep={ts.shape}, cond={cond.shape}, "
          f"num_tokens={nt}")
    del lat, ts, cond
    torch.cuda.empty_cache()

    # ===================================================================
    # Phase 7: Run differentiable training
    # ===================================================================
    print("\n" + "=" * 60)
    print("  Phase 7: Differentiable BTRM training")
    print("=" * 60)
    print(f"  Steps: {N_STEPS}")
    print(f"  LR: {LR}")
    print(f"  LR schedule: {LR_SCHEDULE}")
    print(f"  Grad clip: {GRAD_CLIP}")
    print(f"  Grad accum: {GRAD_ACCUM}")
    print(f"  Warmup: {WARMUP_STEPS}")
    print(f"  Checkpoint steps: {CHECKPOINT_STEPS}")
    print(f"  Heads: {HEAD_NAMES}")
    print(f"  Gradient checkpointing: True")
    print(f"  Scoring: on-the-fly GPU (VAE resident)")

    from src_ii.btrm_training import train_btrm_differentiable

    # Open metrics file for streaming writes
    metrics_file = open(str(METRICS_PATH), "w")

    adapter_grad_verified = False

    def training_callback(step, entry):
        """Write each training step's metrics to JSONL + verify adapter grads."""
        nonlocal adapter_grad_verified

        record = {
            "phase": "btrm_differentiable_v5",
            "step": step,
            **entry,
            "scoring_method": "on_the_fly_gpu",
            "score_calls_cumulative": _score_calls[0],
            "score_time_cumulative_s": _score_time[0],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # After macrobatch 0 (first step), verify adapter receives gradients
        if step == 0 and not adapter_grad_verified:
            pre_clip = entry.get("pre_clip_grad_norm", 0.0)
            record["adapter_grad_verified"] = pre_clip > 0.0

            if pre_clip > 0.0:
                print(f"  [VERIFIED] Adapter receives gradients: "
                      f"pre_clip_grad_norm={pre_clip:.6f}")
            else:
                print(f"  [WARNING] pre_clip_grad_norm={pre_clip:.6f} -- "
                      f"adapter may not be receiving gradients!")

            # Additional check: inspect actual LoRA param gradients
            n_grads = 0
            n_nonzero = 0
            max_grad = 0.0
            for p in adapter_params:
                if p.grad is not None:
                    n_grads += 1
                    g_max = p.grad.abs().max().item()
                    if g_max > 0:
                        n_nonzero += 1
                    max_grad = max(max_grad, g_max)

            record["adapter_params_with_grad"] = n_grads
            record["adapter_params_nonzero_grad"] = n_nonzero
            record["adapter_max_grad"] = max_grad

            print(f"  [ADAPTER GRAD DETAIL] {n_nonzero}/{n_grads} params have "
                  f"nonzero grad, max_grad={max_grad:.6e}")

            adapter_grad_verified = True

        metrics_file.write(json.dumps(record, default=str) + "\n")
        metrics_file.flush()

    # Checkpoint function: save adapter + head state to a subdirectory
    def save_checkpoint(step, model):
        ckpt_dir = OUTPUT_DIR / f"checkpoint_step{step:03d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        persist_btrm(model, "rtheta", str(ckpt_dir))
        print(f"  [CHECKPOINT] Saved to {ckpt_dir}")

    t_train_start = time.perf_counter()

    training_curve = train_btrm_differentiable(
        model=raw_model,
        pair_sampler=sampler,
        load_latent_fn=load_latent_fn,
        preference_fn=preference_fn,
        n_steps=N_STEPS,
        lr=LR,
        # logsquare_weight not passed -- regularizer removed, BT loss only
        head_names=HEAD_NAMES,
        pref_keys=PREF_KEYS,
        gradient_checkpointing=True,
        max_grad_norm=GRAD_CLIP,
        log_interval=1,  # log every step
        callback=training_callback,
        warmup_steps=WARMUP_STEPS,
        grad_accum_steps=GRAD_ACCUM,
        lr_schedule=LR_SCHEDULE,
        checkpoint_fn=save_checkpoint,
        checkpoint_steps=CHECKPOINT_STEPS,
    )

    metrics_file.close()
    t_train_end = time.perf_counter()
    train_time = t_train_end - t_train_start

    print(f"\n  Training complete: {train_time:.1f}s ({train_time / N_STEPS:.1f}s/step)")
    print(f"  Total scoring: {_score_calls[0]} calls in {_score_time[0]:.1f}s "
          f"({_score_time[0] / max(_score_calls[0], 1) * 1000:.1f}ms/call)")

    # ===================================================================
    # Phase 8: Pre-persist test scores
    # ===================================================================
    print("\n" + "=" * 60)
    print("  Phase 8: Scoring test inputs before persist")
    print("=" * 60)

    raw_model.gradient_checkpointing = False
    raw_model.eval()

    # Score a sample of images to verify the trained model produces
    # different outputs than an untrained model
    import random
    rng = random.Random(123)
    test_traj_ids = rng.sample(traj_ids, min(10, len(traj_ids)))
    test_scores = []

    for traj_id in test_traj_ids:
        meta, accessor = reader[traj_id]
        # Use "final" step if available, else first available step
        steps = accessor.available_steps
        step_key = "final" if "final" in steps else steps[0]

        lat, ts, cond, nt = load_latent_fn((traj_id, step_key))

        with torch.no_grad():
            scores = score_serial(raw_model, lat, ts, cond, nt, gradient_checkpointing=False)

        score_dict = {
            "traj_id": traj_id,
            "step_key": step_key,
        }
        for head_idx, name in enumerate(HEAD_NAMES):
            score_dict[f"score_{name}"] = float(scores[0, head_idx].item())
        test_scores.append(score_dict)

        del lat, ts, cond
        torch.cuda.empty_cache()

    with open(str(PRE_PERSIST_SCORES_PATH), "w") as f:
        json.dump(test_scores, f, indent=2)
    print(f"  Pre-persist scores saved: {len(test_scores)} entries")
    for s in test_scores[:3]:
        print(f"    traj {s['traj_id']:3d}/{s['step_key']}: "
              f"pinkify={s['score_pinkify']:.4f}, thisnotthat={s['score_thisnotthat']:.4f}")

    # ===================================================================
    # Phase 9: Persist compound model
    # ===================================================================
    print("\n" + "=" * 60)
    print("  Phase 9: Persisting trained compound model")
    print("=" * 60)

    persist_info = persist_btrm(raw_model, "rtheta", str(OUTPUT_DIR))
    print(f"  Persisted: {persist_info}")

    # ===================================================================
    # Summary
    # ===================================================================
    wall_total = time.perf_counter() - wall_start

    # Report adapter weight statistics
    adapter_params_list = list(get_adapter_params(raw_model, "rtheta").values())
    if adapter_params_list:
        lora_b_stats = []
        for p in adapter_params_list:
            lora_b_stats.append({
                "shape": list(p.shape),
                "abs_max": float(p.abs().max().item()),
                "abs_mean": float(p.abs().mean().item()),
            })
        any_nonzero = any(s["abs_max"] > 0 for s in lora_b_stats)
    else:
        any_nonzero = False
        lora_b_stats = []

    summary = {
        "run_name": "pinkify_thisnotthat_differentiable_v5",
        "wall_total_s": wall_total,
        "train_time_s": train_time,
        "n_training_steps": N_STEPS,
        "lr": LR,
        "grad_clip": GRAD_CLIP,
        "grad_accum": GRAD_ACCUM,
        "warmup_steps": WARMUP_STEPS,
        "logsquare_weight": 0.0,  # removed -- logsquare regularizer removed
        "lr_schedule": LR_SCHEDULE,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "head_names": list(HEAD_NAMES),
        "n_trajectories": len(traj_ids),
        "scoring_method": "on_the_fly_gpu",
        "total_score_calls": _score_calls[0],
        "total_score_time_s": _score_time[0],
        "avg_score_time_ms": _score_time[0] / max(_score_calls[0], 1) * 1000,
        "adapter_trained": any_nonzero,
        "adapter_grad_verified_step0": adapter_grad_verified,
        "n_adapter_params": n_adapter,
        "n_head_params": n_head,
        "sampler_stats": sampler.stats(),
        "challenge_set_scores": challenge_scores if challenge_scores else None,
        "end_time": datetime.now(timezone.utc).isoformat(),
    }

    if training_curve:
        first = training_curve[0]
        last = training_curve[-1]
        summary["initial_loss"] = first["loss"]
        summary["final_loss"] = last["loss"]
        for name in HEAD_NAMES:
            k = f"accuracy_{name}"
            summary[f"initial_{k}"] = first.get(k, 0)
            summary[f"final_{k}"] = last.get(k, 0)
        summary["final_grad_norm"] = last.get("pre_clip_grad_norm", 0)

    with open(str(SUMMARY_PATH), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"  DIFFERENTIABLE BTRM TRAINING COMPLETE (v5: cosine LR decay, checkpoints)")
    print(f"{'=' * 60}")
    print(f"  Wall time: {wall_total:.1f}s ({wall_total / 60:.1f} min)")
    print(f"  Train time: {train_time:.1f}s ({train_time / N_STEPS:.1f}s/step)")
    if training_curve:
        print(f"  Loss: {first['loss']:.4f} -> {last['loss']:.4f}")
        for name in HEAD_NAMES:
            k = f"accuracy_{name}"
            print(f"  {name}: {first.get(k, 0):.3f} -> {last.get(k, 0):.3f}")
        print(f"  Grad norm: {first.get('pre_clip_grad_norm', 0):.4f} -> "
              f"{last.get('pre_clip_grad_norm', 0):.4f}")
    print(f"  Scoring: {_score_calls[0]} on-the-fly calls, "
          f"{_score_time[0] / max(_score_calls[0], 1) * 1000:.1f}ms/call avg")
    print(f"  Adapter trained (nonzero weights): {any_nonzero}")
    print(f"  Adapter grad verified step 0: {adapter_grad_verified}")
    print(f"  Output: {OUTPUT_DIR}")

    # Cleanup
    del vae
    reader.close()
    torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    sys.exit(main())

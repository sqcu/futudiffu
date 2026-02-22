r"""BTRM training with reward-function-derived preference labels.

Replaces sigma-based preference labels with ground truth reward function
evaluation. For each training pair:
  1. VAE-decode both latents to pixel tensors (torch.no_grad)
  2. Score each with pinkify_score_gpu and thisnotthat_score_gpu
  3. Per-head preference: +1 if score_a > score_b, -1 if B wins, 0 tie

This is the strong form of the manifold hypothesis test:
  - Two decorrelated reward heads
  - Each trained on its own ground truth scoring function
  - Cross-head Spearman rho should stay low while per-head correlation
    with ground truth increases

Validation at each eval checkpoint:
  1. PINKIFY validation: A < B < C, D ~ E, {A,B,C} < {D,E} < F
  2. TNT validation: THIS_REF > sketches > color > THAT_REF
  3. Cross-head decorrelation: Spearman rho between head scores

Execution:
  PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe ^
      F:\dox\repos\ai\futudiffu\scripts_ii\run_reward_validated_training.py
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

FP8_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
TE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"
VAE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\vae\zimage.safetensors"
TOKENIZER_PATH = str(REPO_ROOT / "src" / "futudiffu" / "tokenizer")

DATASET_DIR = REPO_ROOT / "multi_res_trajectories"
OUTPUT_DIR = REPO_ROOT / "training_output" / "reward_function_run_tnt_v2"
PINKIFY_CHALLENGE_DIR = REPO_ROOT / "i2i_off_policies" / "PINKIFY_cases"
TNT_CHALLENGE_DIR = REPO_ROOT / "i2i_off_policies"

N_STEPS = 150
MACROBATCH_BUDGET = 3.0
MACROBATCH_CROSS_RES = True
LR = 3e-4
GRAD_CLIP = 0.1
WARMUP_STEPS = 5
LR_SCHEDULE = "warmup_cosine"
CHECKPOINT_STEPS = [25, 50, 75, 100, 125]
CLEAN_FRACTION = 0.8

HEAD_NAMES = ("pinkify", "thisnotthat")
PREF_KEYS = ("pinkify_pref", "thisnotthat_pref")

EVAL_INTERVAL = 10  # Evaluate all heads every N steps

RUN_NAME = "reward_function_validated_training_tnt_v2"


# ---------------------------------------------------------------------------
# Encode challenge images to latents (PINKIFY and TNT combined)
# ---------------------------------------------------------------------------

def encode_pinkify_challenge_latents(
    challenge_dir: Path,
    vae,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, dict]:
    """Load PINKIFY challenge images, VAE-encode to latents."""
    from PIL import Image
    import numpy as np
    from futudiffu.vae import vae_encode

    labels = ("A", "B", "C", "D", "E", "F")
    cache = {}

    for label in labels:
        candidates = [
            challenge_dir / f"PINKER_{label}.png",
            challenge_dir / f"{label}.png",
        ]
        img_path = None
        for c in candidates:
            if c.exists():
                img_path = c
                break
        if img_path is None:
            raise FileNotFoundError(f"Challenge image for '{label}' not found: {candidates}")

        pil = Image.open(str(img_path)).convert("RGB")
        arr = np.array(pil, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(arr).permute(2, 0, 1)
        pixel_batch = img_t.unsqueeze(0).to(device=device, dtype=dtype)

        with torch.no_grad():
            latent = vae_encode(vae, pixel_batch)

        _, _, lat_h, lat_w = latent.shape
        cache[label] = {
            "latent": latent,
            "timestep": torch.zeros(1, device=device, dtype=dtype),
            "conditioning": torch.zeros(1, 1, 2560, device=device, dtype=dtype),
            "num_tokens": 1,
            "width": lat_w * 8,
            "height": lat_h * 8,
        }

    return cache


def encode_tnt_challenge_latents(
    challenge_dir: Path,
    vae,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, dict]:
    """Load TNT challenge images, VAE-encode to latents."""
    from PIL import Image
    import numpy as np
    from futudiffu.vae import vae_encode
    from src_ii.tnt_validation import TNT_CHALLENGE_LABELS, _TNT_FILENAMES

    cache = {}

    for label in TNT_CHALLENGE_LABELS:
        fname = _TNT_FILENAMES[label]
        img_path = challenge_dir / fname
        if not img_path.exists():
            raise FileNotFoundError(f"TNT challenge image '{label}' not found: {img_path}")

        pil = Image.open(str(img_path)).convert("RGB")
        arr = np.array(pil, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(arr).permute(2, 0, 1)
        pixel_batch = img_t.unsqueeze(0).to(device=device, dtype=dtype)

        with torch.no_grad():
            latent = vae_encode(vae, pixel_batch)

        _, _, lat_h, lat_w = latent.shape
        cache[label] = {
            "latent": latent,
            "timestep": torch.zeros(1, device=device, dtype=dtype),
            "conditioning": torch.zeros(1, 1, 2560, device=device, dtype=dtype),
            "num_tokens": 1,
            "width": lat_w * 8,
            "height": lat_h * 8,
        }

    return cache


# ---------------------------------------------------------------------------
# BTRM scoring on cached latents (per-head)
# ---------------------------------------------------------------------------

def score_cached_latents(
    model,
    latent_cache: dict[str, dict],
    head_name: str,
    head_names_all: tuple,
    device: torch.device,
) -> dict[str, float]:
    """Score cached latents with a specific BTRM head.

    Returns dict mapping label to score.
    """
    from src_ii.btrm_lifecycle import score_serial

    head_index = list(head_names_all).index(head_name)
    model.gradient_checkpointing = False
    model.eval()
    scores = {}

    with torch.no_grad():
        for label, cached in latent_cache.items():
            latent = cached["latent"]
            timestep = cached["timestep"]
            conditioning = cached["conditioning"]
            num_tokens = cached["num_tokens"]

            score_tensor = score_serial(
                model, latent, timestep, conditioning, num_tokens,
                gradient_checkpointing=False,
            )
            scores[label] = float(score_tensor[0, head_index].item())

    return scores


def score_all_heads_cached(
    model,
    latent_cache: dict[str, dict],
    head_names: tuple,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Score cached latents with ALL heads. Returns {head_name: {label: score}}."""
    from src_ii.btrm_lifecycle import score_serial

    head_indices = {name: i for i, name in enumerate(head_names)}
    model.gradient_checkpointing = False
    model.eval()
    results = {name: {} for name in head_names}

    with torch.no_grad():
        for label, cached in latent_cache.items():
            latent = cached["latent"]
            timestep = cached["timestep"]
            conditioning = cached["conditioning"]
            num_tokens = cached["num_tokens"]

            score_tensor = score_serial(
                model, latent, timestep, conditioning, num_tokens,
                gradient_checkpointing=False,
            )

            for name in head_names:
                idx = head_indices[name]
                results[name][label] = float(score_tensor[0, idx].item())

    return results


# ---------------------------------------------------------------------------
# Validation log helpers
# ---------------------------------------------------------------------------

def append_validation_log(log_path: Path, entry: dict) -> None:
    """Append one evaluation entry to a validation log (JSONL)."""
    with open(str(log_path), "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    wall_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("=" * 70)
    print("  REWARD-FUNCTION-VALIDATED BTRM TRAINING")
    print(f"  Steps: {N_STEPS}, macrobatch_budget: {MACROBATCH_BUDGET}")
    print(f"  Clean fraction: {CLEAN_FRACTION}")
    print(f"  Eval interval: every {EVAL_INTERVAL} steps")
    print(f"  LR: {LR}, schedule: {LR_SCHEDULE}, grad_clip: {GRAD_CLIP}")
    print(f"  Preference labels: REWARD FUNCTION (not sigma)")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)

    # ==================================================================
    # Phase 0: Pre-flight checks
    # ==================================================================
    if not PINKIFY_CHALLENGE_DIR.exists():
        print(f"\n  FATAL: PINKIFY challenge directory not found: {PINKIFY_CHALLENGE_DIR}")
        return 1

    if not TNT_CHALLENGE_DIR.exists():
        print(f"\n  FATAL: TNT challenge directory not found: {TNT_CHALLENGE_DIR}")
        return 1

    if not DATASET_DIR.exists():
        print(f"\n  FATAL: Multi-res dataset not found: {DATASET_DIR}")
        return 1

    # ==================================================================
    # Phase 1: Load dataset + build pair sampler
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Phase 1: Loading multi-res V2 dataset")
    print("=" * 60)

    from futudiffu.dataset_v2 import DatasetReader
    from src_ii.pair_sampler import BTRMPairSampler, build_positions_from_v2
    from src_ii.flops_sampling import compute_flops_sampling_weights_from_positions

    reader = DatasetReader(str(DATASET_DIR))
    n_available = len(reader)
    print(f"  Dataset: {n_available} trajectories")

    if n_available < 10:
        print(f"  ERROR: Need at least 10 trajectories, have {n_available}")
        return 1

    traj_ids = list(range(n_available))
    positions = build_positions_from_v2(reader, traj_ids=traj_ids)
    print(f"  Positions: {len(positions)} across {len(traj_ids)} trajectories")

    res_dist = {}
    for pos in positions:
        key = f"{pos.width}x{pos.height}"
        res_dist[key] = res_dist.get(key, 0) + 1
    n_unique_res = len(res_dist)
    print(f"  Unique resolutions: {n_unique_res}")

    flops_weights = compute_flops_sampling_weights_from_positions(positions)

    sampler = BTRMPairSampler(
        positions=positions,
        allow_inter_trajectory=True,
        allow_intra_trajectory=True,
        rng_seed=42,
        flops_weights=flops_weights,
        clean_fraction=CLEAN_FRACTION,
    )
    print(f"  Pair space: {sampler.pair_space_size:,} possible pairs")
    print(f"  Clean fraction: {CLEAN_FRACTION}")

    # ==================================================================
    # Phase 2: Encode prompts with text encoder (then free)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Phase 2: Encoding prompts")
    print("=" * 60)

    from futudiffu.text_encoder import create_tokenizer, load_text_encoder, encode_prompt

    tokenizer = create_tokenizer(TOKENIZER_PATH)
    te_model = load_text_encoder(TE_PATH, device=device, dtype=dtype)

    prompt_cache = {}
    for idx in traj_ids:
        meta, _ = reader[idx]
        prompt = meta.get("prompt", "")
        if prompt and prompt not in prompt_cache:
            cond = encode_prompt(te_model, tokenizer, prompt, device=device)
            prompt_cache[prompt] = cond.cpu()

    n_prompts = len(prompt_cache)
    print(f"  Encoded {n_prompts} unique prompts")

    del te_model, tokenizer
    torch.cuda.empty_cache()
    print(f"  TE freed. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ==================================================================
    # Phase 3: Load VAE (kept for entire training -- used for preference
    #          labels via reward manifest AND for challenge encoding)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Phase 3: Loading VAE (persistent for reward function scoring)")
    print("=" * 60)

    from src_ii.vae_utils import load_vae

    vae = load_vae(VAE_PATH, device=device, dtype=dtype)
    print(f"  VAE loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ==================================================================
    # Phase 3b: Encode challenge images to latents for eval (with VAE)
    # ==================================================================
    print("\n  Encoding PINKIFY challenge images to latents...")
    pinkify_latent_cache = encode_pinkify_challenge_latents(
        PINKIFY_CHALLENGE_DIR, vae, device, dtype,
    )
    print(f"  Cached {len(pinkify_latent_cache)} PINKIFY challenge latents")

    print("  Encoding TNT challenge images to latents...")
    tnt_latent_cache = encode_tnt_challenge_latents(
        TNT_CHALLENGE_DIR, vae, device, dtype,
    )
    print(f"  Cached {len(tnt_latent_cache)} TNT challenge latents")

    # Build a COMBINED latent cache for cross-head decorrelation
    # (union of both validation sets, with prefixed labels to avoid collisions)
    combined_latent_cache = {}
    for label, data in pinkify_latent_cache.items():
        combined_latent_cache[f"PINK_{label}"] = data
    for label, data in tnt_latent_cache.items():
        combined_latent_cache[f"TNT_{label}"] = data
    print(f"  Combined cache for cross-head: {len(combined_latent_cache)} images")

    # ==================================================================
    # Phase 3c: Ground truth validation (pixel-space)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Phase 3c: Ground truth scores (pixel-space)")
    print("=" * 60)

    from src_ii.pinkify_validation import validate_pinkify_ranking
    from src_ii.tnt_validation import validate_tnt_ranking

    # PINKIFY ground truth
    pinkify_gt = validate_pinkify_ranking(
        challenge_dir=str(PINKIFY_CHALLENGE_DIR), device=device,
    )
    print(f"  PINKIFY ground truth ranking: {' < '.join(pinkify_gt['rank_order'])}")
    print(f"  PINKIFY all constraints passed: {pinkify_gt['passed']}")
    for check in pinkify_gt["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"    [{status}] {check['name']}: {check['detail']}")

    # TNT ground truth
    tnt_gt = validate_tnt_ranking(
        challenge_dir=str(TNT_CHALLENGE_DIR), device=device,
    )
    print(f"\n  TNT ground truth ranking: {' < '.join(tnt_gt['rank_order'])}")
    print(f"  TNT all constraints passed: {tnt_gt['passed']}")
    for check in tnt_gt["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"    [{status}] {check['name']}: {check['detail']}")

    # Save ground truth
    gt_path = OUTPUT_DIR / "ground_truth.json"
    with open(str(gt_path), "w") as f:
        json.dump({"pinkify": pinkify_gt, "tnt": tnt_gt}, f, indent=2)
    print(f"\n  Ground truth saved to {gt_path}")

    # Ground truth cross-head decorrelation (pixel-space scores)
    from src_ii.cross_head_decorrelation import measure_cross_head_from_pixel_scores

    # Compute pinkify and TNT scores for all images in the combined set
    gt_head_scores = {"pinkify": {}, "thisnotthat": {}}
    for label in pinkify_gt["scores"]:
        gt_head_scores["pinkify"][f"PINK_{label}"] = pinkify_gt["scores"][label]
    for label in tnt_gt["scores"]:
        gt_head_scores["thisnotthat"][f"TNT_{label}"] = tnt_gt["scores"][label]

    # For cross-head measurement, we need scores from BOTH heads on the SAME images.
    # Score PINKIFY images with TNT ground truth, and TNT images with PINKIFY ground truth.
    from src_ii.reward_functions import pinkify_score_gpu, thisnotthat_score_gpu, _pil_to_tensor
    from PIL import Image
    import numpy as np
    import torch.nn.functional as F

    # Load TNT references for scoring
    this_pil = Image.open(str(TNT_CHALLENGE_DIR / "pizza-ratto.png")).convert("RGB")
    that_pil = Image.open(str(TNT_CHALLENGE_DIR / "offhand_pleometric.png")).convert("RGB")
    this_ref_tensor = _pil_to_tensor(this_pil, device)
    that_ref_tensor = _pil_to_tensor(that_pil, device)

    # Build a bound TNT score function (captures references)
    def tnt_score_bound(img_t: torch.Tensor) -> torch.Tensor:
        """TNT score with THIS/THAT references pre-bound."""
        return thisnotthat_score_gpu(img_t, this_ref_tensor, that_ref_tensor)

    # Score PINKIFY images with TNT and vice versa, building a shared score table
    gt_cross_scores = {"pinkify": {}, "thisnotthat": {}}

    # Score all PINKIFY challenge images
    from src_ii.pinkify_validation import _load_challenge_images as _load_pinkify_images
    pinkify_pixel_images = _load_pinkify_images(PINKIFY_CHALLENGE_DIR, device=device)
    with torch.no_grad():
        for label, img_t in pinkify_pixel_images.items():
            key = f"PINK_{label}"
            gt_cross_scores["pinkify"][key] = float(pinkify_score_gpu(img_t).item())
            gt_cross_scores["thisnotthat"][key] = float(tnt_score_bound(img_t).item())

    # Score all TNT challenge images
    from src_ii.tnt_validation import _load_tnt_challenge_images
    tnt_pixel_images = _load_tnt_challenge_images(TNT_CHALLENGE_DIR, device=device)
    with torch.no_grad():
        for label, img_t in tnt_pixel_images.items():
            key = f"TNT_{label}"
            gt_cross_scores["pinkify"][key] = float(pinkify_score_gpu(img_t).item())
            gt_cross_scores["thisnotthat"][key] = float(tnt_score_bound(img_t).item())

    gt_cross = measure_cross_head_from_pixel_scores(
        gt_cross_scores, head_names=("pinkify", "thisnotthat"),
    )
    print(f"\n  Ground truth cross-head decorrelation:")
    print(f"  {gt_cross['summary']}")
    print(f"  (n_images={gt_cross['n_images']})")

    gt_cross_path = OUTPUT_DIR / "ground_truth_cross_head.json"
    with open(str(gt_cross_path), "w") as f:
        json.dump(gt_cross, f, indent=2, default=str)

    # ==================================================================
    # Phase 4: Load FP8 backbone + create BTRMCompoundModel
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Phase 4: Loading backbone + creating BTRM compound model")
    print("=" * 60)

    from src_ii.zimage_model import load_zimage_rlaif
    from src_ii.btrm_lifecycle import setup_btrm_training, persist_btrm
    from src_ii.multi_lora import get_adapter_params
    from src_ii.sigma_schedule import build_sigma_schedule, resolution_shift

    # compile_model=False is CORRECT here: whole-model torch.compile is
    # incompatible with per-block gradient checkpointing.
    _, raw_model = load_zimage_rlaif(
        FP8_PATH, device=device, dtype=dtype,
        compile_model=False, fuse=True,
    )
    print(f"  VRAM after backbone: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    optimizer = setup_btrm_training(
        raw_model,
        adapter_name="rtheta",
        adapter_rank=8,
        adapter_alpha=16.0,
        adapter_init_b_std=0.01,
    )

    n_adapter = sum(p.numel() for p in get_adapter_params(raw_model, "rtheta").values())
    n_head = sum(p.numel() for p in raw_model.score_proj.parameters()) + \
             sum(p.numel() for p in raw_model.score_norm.parameters())
    print(f"  Adapter params: {n_adapter:,}")
    print(f"  Head params: {n_head:,}")
    print(f"  Total trainable: {n_adapter + n_head:,}")

    # ==================================================================
    # Phase 4b: Initial evaluation (before training)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Phase 4b: Initial evaluation (before training)")
    print("=" * 60)

    from src_ii.pinkify_validation import _check_ranking as check_pinkify_ranking
    from src_ii.tnt_validation import _check_tnt_ranking as check_tnt_ranking
    from src_ii.cross_head_decorrelation import measure_cross_head_decorrelation

    # Init validation log files
    pinkify_log_path = OUTPUT_DIR / "pinkify_validation_log.jsonl"
    tnt_log_path = OUTPUT_DIR / "tnt_validation_log.jsonl"
    cross_head_log_path = OUTPUT_DIR / "cross_head_decorrelation_log.jsonl"

    for p in [pinkify_log_path, tnt_log_path, cross_head_log_path]:
        if p.exists():
            p.unlink()

    # Score with initial (untrained) model
    pinkify_scores = score_cached_latents(
        raw_model, pinkify_latent_cache, "pinkify", HEAD_NAMES, device,
    )
    pinkify_checks = check_pinkify_ranking(pinkify_scores)

    tnt_scores = score_cached_latents(
        raw_model, tnt_latent_cache, "thisnotthat", HEAD_NAMES, device,
    )
    tnt_checks = check_tnt_ranking(tnt_scores)

    cross_head_result = measure_cross_head_decorrelation(
        raw_model, combined_latent_cache, head_names=HEAD_NAMES, device=device,
    )

    initial_eval = {
        "step": -1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pinkify": {
            "scores": pinkify_scores,
            "passed": all(c["passed"] for c in pinkify_checks),
            "n_passed": sum(1 for c in pinkify_checks if c["passed"]),
            "rank_order": sorted(pinkify_scores.keys(), key=lambda k: pinkify_scores[k]),
        },
        "tnt": {
            "scores": tnt_scores,
            "passed": all(c["passed"] for c in tnt_checks),
            "n_passed": sum(1 for c in tnt_checks if c["passed"]),
            "rank_order": sorted(tnt_scores.keys(), key=lambda k: tnt_scores[k]),
        },
        "cross_head": cross_head_result["cross_rho"],
    }

    append_validation_log(pinkify_log_path, {
        "step": -1, "scores": pinkify_scores, "checks": pinkify_checks,
        "passed": initial_eval["pinkify"]["passed"],
        "rank_order": initial_eval["pinkify"]["rank_order"],
    })
    append_validation_log(tnt_log_path, {
        "step": -1, "scores": tnt_scores, "checks": tnt_checks,
        "passed": initial_eval["tnt"]["passed"],
        "rank_order": initial_eval["tnt"]["rank_order"],
    })
    append_validation_log(cross_head_log_path, {
        "step": -1, "cross_rho": cross_head_result["cross_rho"],
    })

    print(f"  Initial PINKIFY: {initial_eval['pinkify']['n_passed']}/{len(pinkify_checks)} constraints")
    print(f"  Initial TNT: {initial_eval['tnt']['n_passed']}/{len(tnt_checks)} constraints")
    print(f"  Initial cross-head: {cross_head_result['summary']}")

    # ==================================================================
    # Build load_latent_fn
    # ==================================================================
    _v2_meta_cache = {}

    def _get_v2_meta(traj_id):
        if traj_id not in _v2_meta_cache:
            meta, accessor = reader[traj_id]
            _v2_meta_cache[traj_id] = (meta, accessor)
        return _v2_meta_cache[traj_id]

    def load_latent_fn(key):
        traj_id, step_key = key
        meta, accessor = _get_v2_meta(traj_id)
        latent = accessor[step_key].to(device=device, dtype=dtype)
        if latent.dim() == 3:
            latent = latent.unsqueeze(0)

        n_steps_traj = meta.get("n_steps", 30)
        w = meta.get("width", 1280)
        h = meta.get("height", 832)
        recorded_shift = meta.get("sampling_shift")
        if recorded_shift is not None:
            shift = float(recorded_shift)
        else:
            shift = resolution_shift(w, h)

        sigmas = build_sigma_schedule(
            n_steps_traj, sampling_shift=shift, device="cpu", dtype=torch.float32,
        )

        if step_key == "final":
            sigma_val = 0.0
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

    # ==================================================================
    # Build reward manifest (THE KEY CHANGE)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Building reward manifest")
    print("=" * 60)

    reward_manifest = {
        "pinkify": pinkify_score_gpu,
        "thisnotthat": tnt_score_bound,
    }

    print(f"  Manifest heads: {list(reward_manifest.keys())}")
    print(f"  pinkify: pinkify_score_gpu (continuous pinkness with local contrast)")
    print(f"  thisnotthat: thisnotthat_score_gpu (pixel similarity to THIS vs THAT)")
    print(f"  Preference labels: derived from reward function scores")
    print(f"  NOT sigma-based. Each head gets INDEPENDENT preferences.")

    # ==================================================================
    # Phase 5: Training loop with validation callback
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"  Phase 5: Training ({N_STEPS} steps) + reward-function validation")
    print(f"  macrobatch_budget={MACROBATCH_BUDGET}, clean_fraction={CLEAN_FRACTION}")
    print(f"  Eval at steps: 0, {EVAL_INTERVAL}, {2*EVAL_INTERVAL}, ... and final")
    print("=" * 60)

    from src_ii.btrm_training import train_btrm_differentiable
    from src_ii.training_artifacts import TrainingArtifacts
    from src_ii.incremental_save import TrainingCurveWriter

    # Disable donated buffer optimization: compiled backward with
    # retain_graph=True (needed for cross-bin pair processing) is
    # incompatible with donated buffers in torch 2.10+.
    torch._functorch.config.donated_buffer = False

    # Disable CUDA graph trees: gradient checkpointing replays the
    # compiled forward, which overwrites CUDA graph output tensors
    # that are still referenced by the first forward's computation graph.
    torch._inductor.config.triton.cudagraph_trees = False

    artifacts = TrainingArtifacts(
        output_dir=str(OUTPUT_DIR),
        run_name=RUN_NAME,
        head_names=HEAD_NAMES,
    )

    curve_writer = TrainingCurveWriter(OUTPUT_DIR / "training_curve.jsonl")

    # Validation callback: runs PINKIFY, TNT, and cross-head at each eval step
    eval_count = 0

    def validation_callback(step: int, entry: dict) -> None:
        nonlocal eval_count

        is_eval_step = (
            step == 0
            or step % EVAL_INTERVAL == 0
            or step == N_STEPS - 1
        )
        if not is_eval_step:
            return

        t_eval_start = time.perf_counter()

        # PINKIFY validation
        pinkify_scores_val = score_cached_latents(
            raw_model, pinkify_latent_cache, "pinkify", HEAD_NAMES, device,
        )
        pinkify_checks_val = check_pinkify_ranking(pinkify_scores_val)
        pinkify_passed = all(c["passed"] for c in pinkify_checks_val)
        n_pinkify_passed = sum(1 for c in pinkify_checks_val if c["passed"])

        append_validation_log(pinkify_log_path, {
            "step": step,
            "scores": pinkify_scores_val,
            "checks": pinkify_checks_val,
            "passed": pinkify_passed,
            "rank_order": sorted(pinkify_scores_val.keys(), key=lambda k: pinkify_scores_val[k]),
            "training_loss": entry.get("loss", entry.get("bt_loss", 0.0)),
            "training_accuracy_pinkify": entry.get("accuracy_pinkify", 0.0),
        })

        # TNT validation
        tnt_scores_val = score_cached_latents(
            raw_model, tnt_latent_cache, "thisnotthat", HEAD_NAMES, device,
        )
        tnt_checks_val = check_tnt_ranking(tnt_scores_val)
        tnt_passed = all(c["passed"] for c in tnt_checks_val)
        n_tnt_passed = sum(1 for c in tnt_checks_val if c["passed"])

        append_validation_log(tnt_log_path, {
            "step": step,
            "scores": tnt_scores_val,
            "checks": tnt_checks_val,
            "passed": tnt_passed,
            "rank_order": sorted(tnt_scores_val.keys(), key=lambda k: tnt_scores_val[k]),
            "training_loss": entry.get("loss", entry.get("bt_loss", 0.0)),
            "training_accuracy_thisnotthat": entry.get("accuracy_thisnotthat", 0.0),
        })

        # Cross-head decorrelation
        cross_result = measure_cross_head_decorrelation(
            raw_model, combined_latent_cache,
            head_names=HEAD_NAMES, device=device,
        )
        append_validation_log(cross_head_log_path, {
            "step": step,
            "cross_rho": cross_result["cross_rho"],
            "head_scores": cross_result["head_scores"],
        })

        eval_time = time.perf_counter() - t_eval_start
        eval_count += 1

        rho_str = ", ".join(
            f"{k}={v:.4f}" for k, v in cross_result["cross_rho"].items()
        )
        print(f"  [EVAL step {step}] "
              f"PINK {n_pinkify_passed}/{len(pinkify_checks_val)} | "
              f"TNT {n_tnt_passed}/{len(tnt_checks_val)} | "
              f"cross-rho: {rho_str} | "
              f"{eval_time:.1f}s")

        # Re-enable training mode
        raw_model.gradient_checkpointing = True
        raw_model.train()

    t_train_start = time.perf_counter()

    training_curve = train_btrm_differentiable(
        model=raw_model,
        pair_sampler=sampler,
        load_latent_fn=load_latent_fn,
        # preference_fn is NOT needed -- reward_manifest replaces it
        n_steps=N_STEPS,
        lr=LR,
        head_names=HEAD_NAMES,
        pref_keys=PREF_KEYS,
        gradient_checkpointing=True,
        max_grad_norm=GRAD_CLIP,
        log_interval=5,
        warmup_steps=WARMUP_STEPS,
        lr_schedule=LR_SCHEDULE,
        packed=True,
        output_dir=str(OUTPUT_DIR),
        artifacts=artifacts,
        checkpoint_steps=CHECKPOINT_STEPS,
        macrobatch_budget=MACROBATCH_BUDGET,
        macrobatch_cross_resolution=MACROBATCH_CROSS_RES,
        curve_writer=curve_writer,
        val_metrics_save_interval=10,
        summary_path=str(OUTPUT_DIR / "run_summary.json"),
        callback=validation_callback,
        reward_manifest=reward_manifest,
        vae=vae,
    )

    curve_writer.close()
    train_time = time.perf_counter() - t_train_start
    print(f"\n  Training complete: {train_time:.1f}s "
          f"({train_time / N_STEPS:.1f}s/step)")

    # ==================================================================
    # Phase 6: Final analysis
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Phase 6: Final analysis")
    print("=" * 60)

    run_config = {
        "mode": "reward_function_validated_training",
        "preference_source": "reward_manifest (pinkify_score_gpu + thisnotthat_score_gpu)",
        "n_steps": N_STEPS,
        "macrobatch_budget": MACROBATCH_BUDGET,
        "macrobatch_cross_resolution": MACROBATCH_CROSS_RES,
        "lr": LR,
        "lr_schedule": LR_SCHEDULE,
        "grad_clip": GRAD_CLIP,
        "warmup_steps": WARMUP_STEPS,
        "clean_fraction": CLEAN_FRACTION,
        "dataset": str(DATASET_DIR),
        "n_trajectories": len(traj_ids),
        "n_unique_resolutions": n_unique_res,
        "n_unique_prompts": n_prompts,
        "eval_interval": EVAL_INTERVAL,
        "eval_count": eval_count,
    }

    report_path = artifacts.generate_analysis(run_config=run_config)
    print(f"  Analysis: {report_path}")

    persist_info = persist_btrm(raw_model, "rtheta", str(OUTPUT_DIR))
    print(f"  Model persisted: {persist_info}")

    # ==================================================================
    # Phase 6b: Load and summarize all validation logs
    # ==================================================================

    def load_jsonl(path):
        entries = []
        with open(str(path), "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries

    pinkify_entries = load_jsonl(pinkify_log_path)
    tnt_entries = load_jsonl(tnt_log_path)
    cross_entries = load_jsonl(cross_head_log_path)

    # Build comprehensive validation summary
    validation_summary = {
        "ground_truth": {
            "pinkify": pinkify_gt,
            "tnt": tnt_gt,
            "cross_head": gt_cross,
        },
        "pinkify_trajectory": [
            {
                "step": e.get("step", -1),
                "n_passed": sum(1 for c in e.get("checks", []) if c.get("passed")),
                "passed": e.get("passed", False),
                "rank_order": e.get("rank_order", []),
            }
            for e in pinkify_entries
        ],
        "tnt_trajectory": [
            {
                "step": e.get("step", -1),
                "n_passed": sum(1 for c in e.get("checks", []) if c.get("passed")),
                "passed": e.get("passed", False),
                "rank_order": e.get("rank_order", []),
            }
            for e in tnt_entries
        ],
        "cross_head_trajectory": [
            {
                "step": e.get("step", -1),
                "cross_rho": e.get("cross_rho", {}),
            }
            for e in cross_entries
        ],
    }

    # Find first step where all constraints pass (per head)
    first_pinkify_all_pass = None
    for e in pinkify_entries:
        if e.get("passed"):
            first_pinkify_all_pass = e.get("step")
            break

    first_tnt_all_pass = None
    for e in tnt_entries:
        if e.get("passed"):
            first_tnt_all_pass = e.get("step")
            break

    validation_summary["first_pinkify_all_pass"] = first_pinkify_all_pass
    validation_summary["first_tnt_all_pass"] = first_tnt_all_pass

    # Final cross-head rho
    final_cross_rho = cross_entries[-1]["cross_rho"] if cross_entries else {}
    validation_summary["final_cross_rho"] = final_cross_rho

    val_summary_path = OUTPUT_DIR / "validation_summary.json"
    with open(str(val_summary_path), "w") as f:
        json.dump(validation_summary, f, indent=2, default=str)

    # ==================================================================
    # Phase 7: Run summary
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Phase 7: Summary")
    print("=" * 60)

    wall_total = time.perf_counter() - wall_start

    losses = [e.get("loss", e.get("bt_loss", 0.0)) for e in training_curve]

    summary = {
        "run_name": RUN_NAME,
        "preference_source": "reward_manifest",
        "wall_time_s": wall_total,
        "train_time_s": train_time,
        "n_steps": N_STEPS,
        "macrobatch_budget": MACROBATCH_BUDGET,
        "clean_fraction": CLEAN_FRACTION,
        "lr": LR,
        "lr_schedule": LR_SCHEDULE,
        "grad_clip": GRAD_CLIP,
        "n_trajectories": len(traj_ids),
        "n_unique_resolutions": n_unique_res,
        "n_unique_prompts": n_prompts,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
        "n_adapter_params": n_adapter,
        "n_head_params": n_head,
        "eval_count": eval_count,
        "pinkify_first_all_pass": first_pinkify_all_pass,
        "tnt_first_all_pass": first_tnt_all_pass,
        "final_cross_rho": final_cross_rho,
        "pinkify_ground_truth_passed": pinkify_gt["passed"],
        "tnt_ground_truth_passed": tnt_gt["passed"],
        "sampler_stats": sampler.stats(),
        "end_time": datetime.now(timezone.utc).isoformat(),
    }

    for name in HEAD_NAMES:
        accs = [e.get(f"accuracy_{name}", 0.0) for e in training_curve]
        if accs:
            summary[f"overall_accuracy_{name}"] = sum(accs) / len(accs)
            last_20 = accs[-20:]
            summary[f"last_20_accuracy_{name}"] = sum(last_20) / len(last_20)

    summary_path_final = OUTPUT_DIR / "run_summary_final.json"
    with open(str(summary_path_final), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    curve_path = OUTPUT_DIR / "training_curve.json"
    with open(str(curve_path), "w") as f:
        json.dump(training_curve, f, indent=2, default=str)

    print(f"\n  Wall time: {wall_total:.1f}s ({wall_total / 60:.1f} min)")
    print(f"  Train time: {train_time:.1f}s ({train_time / N_STEPS:.1f}s/step)")
    if losses:
        print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f} "
              f"(min={min(losses):.4f} at step {losses.index(min(losses))})")
    for name in HEAD_NAMES:
        accs = [e.get(f"accuracy_{name}", 0.0) for e in training_curve]
        if accs:
            print(f"  {name}: overall={sum(accs)/len(accs):.1%}, "
                  f"last-20={sum(accs[-20:])/len(accs[-20:]):.1%}")
    print(f"  Evaluations: {eval_count}")
    print(f"  PINKIFY first all-pass: step {first_pinkify_all_pass}")
    print(f"  TNT first all-pass: step {first_tnt_all_pass}")
    print(f"  Final cross-head rho: {final_cross_rho}")
    print(f"  Clean fraction (measured): {sampler.get_clean_fraction():.1%}")
    print(f"  Output: {OUTPUT_DIR}")

    # Cleanup: free VAE last
    del vae
    reader.close()
    torch.cuda.empty_cache()

    print(f"\n{'=' * 70}")
    print(f"  REWARD-FUNCTION-VALIDATED TRAINING COMPLETE")
    print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

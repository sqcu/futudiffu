"""Validate PINKIFY scores against a challenge set with known ranking.

Challenge set: 6 artisanally constructed images PINKER_A through PINKER_F
with known ranking:
    A < B < C, D ~ E, {A,B,C} < {D,E} < F

Two entry points:
  1. validate_pinkify_ranking() -- scores raw pixel images with pinkify_score_gpu
     (or a user-supplied score_fn). No model, no VAE. Pure pixel-space validation.
  2. validate_btrm_pinkify_ranking() -- VAE-encodes images to latent space, scores
     with a trained BTRM model's pinkify head, checks same ranking constraints.

Import constraints:
  - PIL for image loading
  - torch for GPU scoring
  - validate_pinkify_ranking: imports from src_ii/reward_functions only
  - validate_btrm_pinkify_ranking: imports from src_ii/vae_utils, src_ii/rollout,
    and futudiffu.vae (vae_encode) following the same pattern as src_ii/server.py
  - DOES NOT import: model_manager, server, client
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image

CHALLENGE_LABELS = ("A", "B", "C", "D", "E", "F")


def _load_challenge_images(
    challenge_dir: str | Path,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Load PINKER_A through PINKER_F as [3, H, W] float32 tensors in [0, 1].

    Tries both naming conventions: "PINKER_A.png" and "A.png".

    Returns:
        Dict mapping label ("A" through "F") to [3, H, W] tensor on device.

    Raises:
        FileNotFoundError: If challenge_dir does not exist or any image is missing.
    """
    import numpy as np

    challenge_dir = Path(challenge_dir)
    if not challenge_dir.exists():
        raise FileNotFoundError(f"Challenge directory not found: {challenge_dir}")

    images = {}
    for label in CHALLENGE_LABELS:
        # Try both naming conventions
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
            raise FileNotFoundError(
                f"Challenge image for label '{label}' not found. "
                f"Tried: {[str(c) for c in candidates]}"
            )

        pil = Image.open(str(img_path)).convert("RGB")
        arr = np.array(pil, dtype=np.float32) / 255.0  # (H, W, 3)
        t = torch.from_numpy(arr).permute(2, 0, 1).to(device=device, dtype=dtype)
        images[label] = t

    return images


def _check_ranking(scores: dict[str, float]) -> list[dict]:
    """Run the ranking constraint checks.

    Constraints (from spec: A < B < C, D ~ E, {A,B,C} < {D,E} < F):
        A < B
        B < C
        max(A,B,C) < min(D,E)  -- {A,B,C} < {D,E} as set comparison
        D ~ E  (|D - E| / max(|D|, |E|) < 0.5)
        max(D, E) < F

    Returns:
        List of dicts, each with "name", "passed", "detail".
    """
    A = scores["A"]
    B = scores["B"]
    C = scores["C"]
    D = scores["D"]
    E = scores["E"]
    F_ = scores["F"]

    checks = []

    # A < B
    checks.append({
        "name": "A < B",
        "passed": A < B,
        "detail": f"A={A:.6f}, B={B:.6f}, diff={B - A:.6f}",
    })

    # B < C
    checks.append({
        "name": "B < C",
        "passed": B < C,
        "detail": f"B={B:.6f}, C={C:.6f}, diff={C - B:.6f}",
    })

    # {A,B,C} < {D,E}: every element of the first set < every element of the second
    # Equivalent to max(A,B,C) < min(D,E) since A<B<C means C=max(A,B,C)
    abc_max = max(A, B, C)
    de_min = min(D, E)
    checks.append({
        "name": "max(A,B,C) < min(D,E)",
        "passed": abc_max < de_min,
        "detail": f"max(A,B,C)={abc_max:.6f}, min(D,E)={de_min:.6f}, gap={de_min - abc_max:.6f}",
    })

    # D ~ E: relative difference < 50%
    denom = max(abs(D), abs(E), 1e-10)
    rel_diff = abs(D - E) / denom
    checks.append({
        "name": "D ~ E (relative diff < 50%)",
        "passed": rel_diff < 0.5,
        "detail": f"D={D:.6f}, E={E:.6f}, |D-E|/max={rel_diff:.4f}",
    })

    # max(D,E) < F
    de_max = max(D, E)
    checks.append({
        "name": "max(D,E) < F",
        "passed": de_max < F_,
        "detail": f"max(D,E)={de_max:.6f}, F={F_:.6f}, gap={F_ - de_max:.6f}",
    })

    return checks


def validate_pinkify_ranking(
    score_fn: Callable[[torch.Tensor], torch.Tensor | float] | None = None,
    challenge_dir: str | Path = "i2i_off_policies/PINKIFY_cases",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict:
    """Validate PINKIFY scores against the challenge set ranking.

    Loads PINKER_A-F, scores each with score_fn (or pinkify_score_gpu by default),
    and checks the ranking constraints:
        A < B < C
        D ~ E (relative difference < 50%)
        max(A,B,C) < min(D,E)
        max(D,E) < F

    Args:
        score_fn: Callable that takes a [3, H, W] float32 tensor in [0, 1] and
                  returns a scalar score (tensor or float). If None, uses
                  pinkify_score_gpu from src_ii.reward_functions.
        challenge_dir: Path to directory containing PINKER_A-F.png.
        device: torch device (default: cuda).
        dtype: torch dtype for image loading (default: float32, since pinkify
               needs float32 precision).

    Returns:
        dict with keys:
            "passed": bool -- True if all ranking constraints hold.
            "scores": dict[str, float] -- {"A": score_A, "B": score_B, ...}.
            "checks": list[dict] -- each with "name", "passed", "detail".
            "rank_order": list[str] -- actual sorted order (lowest to highest).
    """
    if device is None:
        device = torch.device("cuda")
    if dtype is None:
        dtype = torch.float32

    if score_fn is None:
        from src_ii.reward_functions import pinkify_score_gpu
        score_fn = pinkify_score_gpu

    images = _load_challenge_images(challenge_dir, device=device, dtype=dtype)

    scores = {}
    with torch.no_grad():
        for label, img_t in images.items():
            raw = score_fn(img_t)
            scores[label] = float(raw.item()) if isinstance(raw, torch.Tensor) else float(raw)

    checks = _check_ranking(scores)
    all_passed = all(c["passed"] for c in checks)

    # Sort labels by score (ascending)
    rank_order = sorted(scores.keys(), key=lambda k: scores[k])

    return {
        "passed": all_passed,
        "scores": scores,
        "checks": checks,
        "rank_order": rank_order,
    }


def validate_btrm_pinkify_ranking(
    btrm_model,
    challenge_dir: str | Path = "i2i_off_policies/PINKIFY_cases",
    vae=None,
    vae_path: str | None = None,
    head_name: str = "pinkify",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict:
    """Validate a trained BTRM model's pinkify head against the challenge set.

    This is the end-of-training validation: does the learned reward model
    preserve the ground-truth PINKIFY ranking?

    Loads PINKER_A-F, VAE-encodes them to latent space, scores with the BTRM
    model's pinkify head at sigma=0 (fully denoised), and checks the same
    ranking constraints.

    Note: The BTRM scores latents, not pixels. So this function:
    1. Loads PNG -> pixel tensor [3, H, W] in [0, 1]
    2. VAE-encodes pixel -> latent (SD process_input + VAE encode + Flux process_in)
    3. Passes latent through BTRM at sigma=0 (fully denoised)
    4. Extracts the pinkify head's score
    5. Checks ranking

    Args:
        btrm_model: Trained BTRMCompoundModel instance.
        challenge_dir: Path to directory containing PINKER_A-F.png.
        vae: Loaded VAE model. If None, loaded from vae_path.
        vae_path: Path to VAE safetensors (used only if vae is None).
        head_name: Name of the pinkify head in the BTRM model.
        device: torch device (default: cuda).
        dtype: torch dtype for VAE and model (default: bfloat16 for VAE/model,
               images are loaded as float32 then cast).

    Returns:
        dict with keys:
            "passed": bool -- True if all ranking constraints hold.
            "scores": dict[str, float] -- {"A": score_A, "B": score_B, ...}.
            "checks": list[dict] -- each with "name", "passed", "detail".
            "rank_order": list[str] -- actual sorted order (lowest to highest).
            "head_name": str -- which head was evaluated.
            "head_index": int -- index of the head in the model.
    """
    if device is None:
        device = torch.device("cuda")
    if dtype is None:
        dtype = torch.bfloat16

    # Load VAE if not provided
    _vae_loaded_here = False
    if vae is None:
        if vae_path is None:
            raise ValueError("Either vae or vae_path must be provided for BTRM validation.")
        from src_ii.vae_utils import load_vae
        vae = load_vae(vae_path, device=device, dtype=dtype)
        _vae_loaded_here = True

    # Resolve head index
    head_names = btrm_model.head_names if hasattr(btrm_model, "head_names") else []
    if head_name not in head_names:
        raise ValueError(
            f"Head '{head_name}' not found in model. Available: {list(head_names)}"
        )
    head_index = list(head_names).index(head_name)

    # Load challenge images as pixel tensors (float32 for precision)
    images = _load_challenge_images(challenge_dir, device=device, dtype=torch.float32)

    # VAE encode: pixel -> latent
    # The VAE encode pipeline is: (pixel * 2 - 1) -> model.encode() -> flux_process_in
    # flux_process_in: (latent - 0.1159) * 0.3611
    from futudiffu.vae import vae_encode

    scores = {}
    btrm_model.eval_mode()

    with torch.no_grad():
        for label, img_t in images.items():
            # img_t is [3, H, W] float32. VAE expects [B, 3, H, W] bfloat16.
            pixel_batch = img_t.unsqueeze(0).to(dtype=dtype)  # (1, 3, H, W)

            # VAE encode -> latent
            latent = vae_encode(vae, pixel_batch)  # (1, 16, H//8, W//8)

            # Score at sigma=0 (fully denoised)
            timestep = torch.zeros(1, device=device, dtype=dtype)

            # Build a dummy conditioning. The BTRM scores latents with
            # conditioning context; for the challenge set we use an empty
            # prompt equivalent. We need to know the expected dimensions.
            # BTRMCompoundModel.score() needs: latent, timestep, conditioning,
            # num_tokens, rope_cache.
            #
            # For validation, we use a single zero-padded conditioning token.
            # The scoring head aggregates via mean-pool over image tokens from
            # the hidden capture hook, so the text conditioning content matters
            # less than the latent content.
            cap_feat_dim = 2560  # from architecture: NextDiT cap_feat_dim
            conditioning = torch.zeros(1, 1, cap_feat_dim, device=device, dtype=dtype)
            num_tokens = 1

            # Build RoPE cache
            _, _, lat_h, lat_w = latent.shape
            from src_ii.rollout import make_rope_cache
            rope_cache = make_rope_cache(
                btrm_model.backbone, lat_h, lat_w, num_tokens, device,
            )

            # Score
            score_tensor = btrm_model.score(
                latent, timestep, conditioning, num_tokens, rope_cache,
            )
            # score_tensor: (1, N_heads) -> extract the pinkify head
            scores[label] = float(score_tensor[0, head_index].item())

    # Free VAE if we loaded it
    if _vae_loaded_here:
        del vae
        torch.cuda.empty_cache()

    checks = _check_ranking(scores)
    all_passed = all(c["passed"] for c in checks)
    rank_order = sorted(scores.keys(), key=lambda k: scores[k])

    return {
        "passed": all_passed,
        "scores": scores,
        "checks": checks,
        "rank_order": rank_order,
        "head_name": head_name,
        "head_index": head_index,
    }

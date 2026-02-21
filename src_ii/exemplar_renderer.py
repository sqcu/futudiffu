"""Exemplar image renderer: VAE-decode and save high/low scoring images.

Canonical module for rendering training exemplars. For each scoring head,
finds the top-K and bottom-K images by score, VAE-decodes them, and saves
as PNG with metadata-rich filenames.

Each head selects its exemplars independently from its own score ranking.
When deduplicate_across_heads=True (the default), images already selected
by a previous head are excluded from subsequent heads' candidate pools,
ensuring distinct exemplars per head. This prevents the degenerate case
where correlated heads (e.g., trained on the same preference signal)
produce identical exemplar sets.

Usage:
    from src_ii.exemplar_renderer import render_exemplars

    render_exemplars(
        output_dir="training_output/run03/exemplars/",
        trajectories=trajectory_data,
        scores=score_dict,
        vae=vae_model,
        top_k=3,
    )

Import constraints:
    - PIL for image saving
    - torch for VAE decode (lazy import)
    - DOES NOT import from src/futudiffu/ (uses src_ii/vae_utils re-export)
    - DOES NOT import: model_manager, server, client
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _compute_rank_correlation(
    scores: dict[str, dict[str, float]],
    head_names: list[str],
) -> dict[str, float]:
    """Compute Spearman rank correlation between each pair of heads.

    Returns dict like {"pinkify_vs_thisnotthat": 0.95}.
    Used for manifest diagnostics to flag correlated heads.
    """
    if len(head_names) < 2:
        return {}

    # Build per-head rankings
    head_rankings: dict[str, list[tuple[str, float]]] = {}
    for head in head_names:
        ranked = []
        for img_key, sd in scores.items():
            if head in sd:
                ranked.append((img_key, sd[head]))
        ranked.sort(key=lambda x: x[1])
        head_rankings[head] = ranked

    correlations: dict[str, float] = {}
    for i, h1 in enumerate(head_names):
        for h2 in head_names[i + 1:]:
            r1 = head_rankings[h1]
            r2 = head_rankings[h2]

            # Build rank maps (0-indexed)
            rank_map_1 = {k: idx for idx, (k, _) in enumerate(r1)}
            rank_map_2 = {k: idx for idx, (k, _) in enumerate(r2)}

            # Only compare images present in both rankings
            common_keys = set(rank_map_1.keys()) & set(rank_map_2.keys())
            if len(common_keys) < 3:
                continue

            n = len(common_keys)
            sum_d2 = sum(
                (rank_map_1[k] - rank_map_2[k]) ** 2
                for k in common_keys
            )
            # Spearman rho = 1 - 6*sum(d^2) / (n*(n^2-1))
            rho = 1.0 - (6.0 * sum_d2) / (n * (n * n - 1))
            correlations[f"{h1}_vs_{h2}"] = round(rho, 4)

    return correlations


def render_exemplars(
    output_dir: str | Path,
    trajectories: list[dict],
    scores: dict[str, dict[str, float]],
    vae=None,
    vae_path: str | None = None,
    top_k: int = 3,
    head_names: list[str] | None = None,
    device=None,
    dtype=None,
    deduplicate_across_heads: bool = True,
) -> Path:
    """Render top-K and bottom-K exemplar images per scoring head.

    For each head, finds the top_k highest and top_k lowest scoring images,
    VAE-decodes them to pixel space, and saves as PNG.

    When deduplicate_across_heads=True (default), images selected for an
    earlier head's exemplars are excluded from later heads' candidate pools.
    This ensures each head shows DIFFERENT images, which is critical when
    heads have correlated rankings (e.g., trained on the same preference
    signal). Without deduplication, correlated heads produce identical
    exemplar sets, making the per-head visualization useless.

    Args:
        output_dir: Directory to write exemplar images and manifest.
        trajectories: List of dicts, each with at minimum:
            "traj_id": int, "step_key": str, "latent": Tensor (1,16,H,W)
            Additional fields are passed through to the manifest.
        scores: Nested dict: {image_key: {head_name: score_value}}.
            image_key is "{traj_id}_{step_key}" (matching trajectory entries).
        vae: Loaded VAE model. If None, loaded from vae_path.
        vae_path: Path to VAE safetensors (used only if vae is None).
        top_k: Number of top and bottom images to render per head.
        head_names: List of scoring head names. If None, inferred from scores.
        device: CUDA device. Defaults to cuda.
        dtype: Working dtype. Defaults to bfloat16.
        deduplicate_across_heads: If True, exclude images already selected
            by a previous head from subsequent heads' candidate pools.
            Default True.

    Returns:
        Path to the exemplars_manifest.json.
    """
    import torch
    from PIL import Image

    if device is None:
        device = torch.device("cuda")
    if dtype is None:
        dtype = torch.bfloat16

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load VAE if not provided
    _vae_loaded_here = False
    if vae is None:
        if vae_path is None:
            raise ValueError("Either vae or vae_path must be provided.")
        from src_ii.vae_utils import load_vae
        vae = load_vae(vae_path, device=device, dtype=dtype)
        _vae_loaded_here = True

    # Build lookup from image_key to trajectory entry
    traj_lookup: dict[str, dict] = {}
    for t in trajectories:
        key = f"{t['traj_id']}_{t['step_key']}"
        traj_lookup[key] = t

    # Infer head names if not provided
    if head_names is None:
        head_names_set: set[str] = set()
        for score_dict in scores.values():
            head_names_set.update(score_dict.keys())
        head_names = sorted(head_names_set)

    # Compute rank correlation diagnostics
    rank_correlations = _compute_rank_correlation(scores, head_names)
    if rank_correlations:
        for pair_name, rho in rank_correlations.items():
            if rho > 0.9:
                print(f"  [exemplar_renderer] WARNING: heads {pair_name} have "
                      f"rank correlation rho={rho:.3f} (>0.9). "
                      f"Deduplication {'ON' if deduplicate_across_heads else 'OFF'}.")

    # Track which images have been selected across heads (for deduplication)
    used_keys_top: set[str] = set()
    used_keys_bottom: set[str] = set()

    manifest_entries: list[dict] = []

    for head in head_names:
        # Collect (image_key, score) pairs for this head
        head_scores: list[tuple[str, float]] = []
        for img_key, score_dict in scores.items():
            if head in score_dict:
                head_scores.append((img_key, score_dict[head]))

        if not head_scores:
            continue

        # Sort by score (ascending: low to high)
        head_scores.sort(key=lambda x: x[1])

        # Select top-K (highest scores), excluding already-used images
        top_k_items: list[tuple[str, float]] = []
        for img_key, score_val in reversed(head_scores):
            if deduplicate_across_heads and img_key in used_keys_top:
                continue
            top_k_items.append((img_key, score_val))
            if len(top_k_items) >= top_k:
                break

        # Select bottom-K (lowest scores), excluding already-used images
        bottom_k_items: list[tuple[str, float]] = []
        for img_key, score_val in head_scores:
            if deduplicate_across_heads and img_key in used_keys_bottom:
                continue
            bottom_k_items.append((img_key, score_val))
            if len(bottom_k_items) >= top_k:
                break

        # Mark selected images as used
        for img_key, _ in top_k_items:
            used_keys_top.add(img_key)
        for img_key, _ in bottom_k_items:
            used_keys_bottom.add(img_key)

        for rank, (img_key, score_val) in enumerate(top_k_items):
            entry = _render_one(
                out_dir, traj_lookup, img_key, vae, device, dtype,
                head=head, rank=rank, score=score_val, category="top",
            )
            if entry:
                manifest_entries.append(entry)

        for rank, (img_key, score_val) in enumerate(bottom_k_items):
            entry = _render_one(
                out_dir, traj_lookup, img_key, vae, device, dtype,
                head=head, rank=rank, score=score_val, category="bottom",
            )
            if entry:
                manifest_entries.append(entry)

    # Save all scores to disk for diagnostics (allows post-hoc analysis)
    all_scores_path = out_dir / "all_scores.json"
    with open(str(all_scores_path), "w") as f:
        json.dump(scores, f, indent=2, default=str)

    # Write manifest
    manifest_path = out_dir / "exemplars_manifest.json"
    manifest = {
        "n_exemplars": len(manifest_entries),
        "top_k": top_k,
        "heads": head_names,
        "deduplicated": deduplicate_across_heads,
        "rank_correlations": rank_correlations,
        "n_images_scored": len(scores),
        "entries": manifest_entries,
    }
    with open(str(manifest_path), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # Free VAE if we loaded it
    if _vae_loaded_here:
        del vae
        torch.cuda.empty_cache()

    return manifest_path


def render_exemplars_from_model(
    output_dir: str | Path,
    btrm_model,
    load_latent_fn,
    sample_keys: list[tuple[int, str]],
    vae=None,
    vae_path: str | None = None,
    top_k: int = 3,
    head_names: list[str] | None = None,
    device=None,
    dtype=None,
    deduplicate_across_heads: bool = True,
) -> Path:
    """Score images with the BTRM model and render top/bottom exemplars.

    Convenience function that scores a set of images, then renders exemplars.
    Useful when you have a trained BTRM model and want to see what it considers
    high vs low quality.

    Args:
        output_dir: Directory for output.
        btrm_model: Trained BTRMCompoundModel (in eval mode).
        load_latent_fn: Callable((traj_id, step_key)) -> (latent, ts, cond, nt, rc).
        sample_keys: List of (traj_id, step_key) tuples to score and render.
        vae: Loaded VAE model (or None to load from vae_path).
        vae_path: Path to VAE safetensors.
        top_k: Number of top/bottom to render per head.
        head_names: Scoring head names.
        device: CUDA device.
        dtype: Working dtype.
        deduplicate_across_heads: If True, exclude images already selected
            by a previous head from subsequent heads' candidate pools.
            Default True.

    Returns:
        Path to exemplars_manifest.json.
    """
    import torch

    if device is None:
        device = torch.device("cuda")
    if dtype is None:
        dtype = torch.bfloat16

    if head_names is None:
        head_names = list(btrm_model.head_names) if hasattr(btrm_model, "head_names") else ["head_0"]

    trajectories: list[dict] = []
    scores: dict[str, dict[str, float]] = {}

    btrm_model.eval_mode()

    for traj_id, step_key in sample_keys:
        lat, ts, cond, nt, rc = load_latent_fn((traj_id, step_key))

        with torch.no_grad():
            score_tensor = btrm_model.score(lat, ts, cond, nt, rc)
            # score_tensor: (1, N_heads)

        img_key = f"{traj_id}_{step_key}"
        score_dict = {}
        for head_idx, name in enumerate(head_names):
            score_dict[name] = float(score_tensor[0, head_idx].item())

        scores[img_key] = score_dict
        # Store latent on CPU to free GPU VRAM for scoring the next image.
        # These are only needed later for VAE decode in render_exemplars(),
        # where they'll be moved back to GPU one at a time.
        trajectories.append({
            "traj_id": traj_id,
            "step_key": step_key,
            "latent": lat.detach().cpu(),
        })

        del lat, ts, cond, rc

    torch.cuda.empty_cache()

    return render_exemplars(
        output_dir=output_dir,
        trajectories=trajectories,
        scores=scores,
        vae=vae,
        vae_path=vae_path,
        top_k=top_k,
        head_names=head_names,
        device=device,
        dtype=dtype,
        deduplicate_across_heads=deduplicate_across_heads,
    )


def _render_one(
    out_dir: Path,
    traj_lookup: dict[str, dict],
    img_key: str,
    vae,
    device,
    dtype,
    head: str,
    rank: int,
    score: float,
    category: str,
) -> dict | None:
    """Decode and save a single exemplar image.

    Returns manifest entry dict, or None if the image could not be rendered.
    """
    import torch

    traj_entry = traj_lookup.get(img_key)
    if traj_entry is None:
        return None

    latent = traj_entry.get("latent")
    if latent is None:
        return None

    traj_id = traj_entry.get("traj_id", "?")
    step_key = traj_entry.get("step_key", "?")

    # VAE decode: latent (1, 16, H, W) -> pixel (1, 3, H*8, W*8) in [0, 1]
    pil_img = _decode_latent(vae, latent, device, dtype)

    # Filename encodes metadata
    fname = f"{head}_{category}{rank:02d}_{traj_id}_{step_key}_{score:.3f}.png"
    out_path = out_dir / fname
    pil_img.save(str(out_path))

    return {
        "filename": fname,
        "head": head,
        "category": category,
        "rank": rank,
        "traj_id": traj_id,
        "step_key": step_key,
        "score": score,
    }


def _decode_latent(vae, latent, device, dtype):
    """Decode a latent tensor to a PIL Image.

    Implements the standard AutoencoderKL decode pipeline:
    1. Process out: (latent / 0.3611) + 0.1159
    2. VAE decode
    3. Clamp [0, 1] -> [0, 255] uint8 -> PIL Image

    Uses src_ii.vae_utils.decode_latent_to_pil which encapsulates this.
    """
    from src_ii.vae_utils import decode_latent_to_pil
    return decode_latent_to_pil(vae, latent, device=device, dtype=dtype)

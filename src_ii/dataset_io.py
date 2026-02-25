"""Canonical dataset loading and preference computation.

Replaces 11 copies of load_latent_fn closures scattered across scripts_ii/.
Also provides make_reward_manifest_preference_fn() extracted from
btrm_training.py's inline closure builder.

Import constraints:
  - torch for tensor ops
  - src_ii.sigma_schedule for schedule construction
  - No server/client imports
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import Tensor


def make_load_latent_fn(
    reader,
    prompt_cache: dict[str, Tensor],
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Callable:
    """Build a canonical load_latent_fn closure for V2 datasets.

    The returned callable accepts (traj_id, step_key) tuples and returns
    (latent, timestep, conditioning, num_tokens, None).

    The 5th element (rope_cache) is None — it's included for compatibility
    with callers that unpack 5 values. Rope caches are constructed by the
    model's forward pass, not by data loading.

    Args:
        reader: V2DatasetReader instance.
        prompt_cache: dict mapping prompt strings to encoded conditioning
            tensors of shape (1, num_tokens, hidden_dim).
        device: Target device for tensors.
        dtype: Target dtype for tensors.
    """
    from src_ii.sigma_schedule import build_sigma_schedule, resolution_shift

    # Per-trajectory metadata cache (avoid re-reading parquet per call)
    _meta_cache: dict[int, tuple[dict, object]] = {}

    def _get_meta(traj_id: int):
        if traj_id not in _meta_cache:
            _meta_cache[traj_id] = reader[traj_id]
        return _meta_cache[traj_id]

    def load_latent_fn(key):
        traj_id, step_key = key
        meta, accessor = _get_meta(traj_id)
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

        return latent, timestep, cond, num_tokens, None

    return load_latent_fn


def make_reward_manifest_preference_fn(
    reward_manifest: dict[str, Callable],
    vae,
    load_latent_fn: Callable,
    head_names: Sequence[str],
    pref_keys: Sequence[str],
) -> Callable[[dict], dict[str, int]]:
    """Build a preference_fn from ground truth reward functions.

    The returned callable takes a pair metadata dict (from pair_sampler)
    and returns per-head preferences computed by VAE-decoding both images
    and scoring them with each head's reward function.

    All operations inside torch.no_grad(). The VAE decode and reward
    function scores produce LABELS, not gradients.

    Args:
        reward_manifest: Dict mapping head_name -> scoring function.
            Each scoring function takes a (3, H, W) float32 tensor in [0,1]
            and returns a scalar tensor.
        vae: Loaded VAE model for decoding latents to pixel tensors.
        load_latent_fn: Callable accepting (traj_id, step_key) tuples.
        head_names: Names of the scoring heads.
        pref_keys: Corresponding preference keys in output dict.
    """
    from futudiffu.vae import vae_decode as _vae_decode

    def preference_fn(pair: dict) -> dict[str, int]:
        with torch.no_grad():
            key_a = (pair["traj_a"], pair["step_a"])
            key_b = (pair["traj_b"], pair["step_b"])
            lat_a, _, _, _, _ = load_latent_fn(key_a)
            lat_b, _, _, _, _ = load_latent_fn(key_b)
            pixel_a = _vae_decode(vae, lat_a)[0].float()
            pixel_b = _vae_decode(vae, lat_b)[0].float()

            prefs = {}
            for head_name, pref_key in zip(head_names, pref_keys):
                score_fn = reward_manifest[head_name]
                sa = score_fn(pixel_a)
                sb = score_fn(pixel_b)
                sa = float(sa.item()) if isinstance(sa, Tensor) else float(sa)
                sb = float(sb.item()) if isinstance(sb, Tensor) else float(sb)
                if sa > sb:
                    prefs[pref_key] = 1
                elif sb > sa:
                    prefs[pref_key] = -1
                else:
                    prefs[pref_key] = 0
            return prefs

    return preference_fn

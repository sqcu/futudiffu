r"""Minimal NaN diagnostic for refactored BTRM training.

Exercises the exact same code path as run_reward_validated_training.py
but with explicit NaN checks after each stage.

Execution:
  PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe ^
      F:\dox\repos\ai\futudiffu\scripts_ii\diagnose_nan_training.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.nn.functional as F

from src_ii.model_paths import FP8_PATH, TE_PATH, VAE_PATH, TOKENIZER_PATH
from src_ii.training_setup import (
    encode_training_prompts,
    build_dataset_positions,
    load_training_backbone,
)
from src_ii.dataset_io import make_load_latent_fn

DATASET_DIR = REPO_ROOT / "multi_res_trajectories"

device = torch.device("cuda")
dtype = torch.bfloat16


def check_nan(name: str, t: torch.Tensor):
    """Check if tensor has NaN/Inf and print status."""
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    status = "OK" if not (has_nan or has_inf) else "FAIL"
    extras = []
    if has_nan:
        extras.append("HAS_NAN")
    if has_inf:
        extras.append("HAS_INF")
    extra_str = f" [{', '.join(extras)}]" if extras else ""
    print(f"  [{status}] {name}: shape={t.shape}, dtype={t.dtype}, "
          f"min={t.min().item():.6g}, max={t.max().item():.6g}{extra_str}")
    return not (has_nan or has_inf)


def main():
    print("=" * 60)
    print("  NaN Diagnostic for Refactored BTRM Training")
    print("=" * 60)

    # Phase 1: Load dataset
    print("\n--- Phase 1: Dataset ---")
    reader, positions, sampler, traj_ids = build_dataset_positions(
        DATASET_DIR, clean_fraction=0.8,
    )

    # Phase 2: Encode prompts
    print("\n--- Phase 2: TE ---")
    prompt_cache = encode_training_prompts(
        reader, traj_ids, TOKENIZER_PATH, TE_PATH, device=device, dtype=dtype,
    )

    # Phase 3: Load backbone
    print("\n--- Phase 3: Model ---")
    raw_model, optimizer_from_setup, head_names = load_training_backbone(
        FP8_PATH, device=device, dtype=dtype, lr=3e-4,
    )
    print(f"  head_names from setup: {head_names}")

    # Phase 4: Build load_latent_fn
    print("\n--- Phase 4: load_latent_fn ---")
    load_latent_fn = make_load_latent_fn(reader, prompt_cache, device=device, dtype=dtype)

    # Phase 5: Load a pair of latents
    print("\n--- Phase 5: Load one pair ---")
    pair_spec = sampler.sample_pair()
    key_a = (pair_spec["traj_a"], pair_spec["step_a"])
    key_b = (pair_spec["traj_b"], pair_spec["step_b"])
    print(f"  Pair: {key_a} vs {key_b}")

    lat_a, ts_a, cond_a, nt_a, _ = load_latent_fn(key_a)
    lat_b, ts_b, cond_b, nt_b, _ = load_latent_fn(key_b)

    check_nan("lat_a", lat_a)
    check_nan("lat_b", lat_b)
    check_nan("ts_a", ts_a)
    check_nan("ts_b", ts_b)
    check_nan("cond_a", cond_a)
    check_nan("cond_b", cond_b)

    # Phase 6: Run score_packed (eval mode first)
    print("\n--- Phase 6a: score_packed (eval mode, no grad) ---")
    raw_model.eval()
    from src_ii.btrm_lifecycle import score_packed
    with torch.no_grad():
        images = [(lat_a, ts_a, cond_a, nt_a), (lat_b, ts_b, cond_b, nt_b)]
        scores_eval = score_packed(raw_model, images, gradient_checkpointing=False)
        check_nan("scores_eval", scores_eval)
        print(f"  scores_eval:\n    img_a = {scores_eval[0].tolist()}\n    img_b = {scores_eval[1].tolist()}")

    # Phase 6b: score_packed (train mode, WITH grad)
    print("\n--- Phase 6b: score_packed (train mode, with grad) ---")
    raw_model.gradient_checkpointing = True
    raw_model.train()
    images = [(lat_a, ts_a, cond_a, nt_a), (lat_b, ts_b, cond_b, nt_b)]
    scores_train = score_packed(raw_model, images, gradient_checkpointing=True)
    ok = check_nan("scores_train", scores_train)
    print(f"  scores_train:\n    img_a = {scores_train[0].tolist()}\n    img_b = {scores_train[1].tolist()}")
    print(f"  scores_train.requires_grad: {scores_train.requires_grad}")
    print(f"  scores_train.grad_fn: {scores_train.grad_fn}")

    if not ok:
        print("\n  DIAGNOSIS: Model produces NaN scores in training mode.")
        print("  This is upstream of the BT loss computation.")
        reader.close()
        return 1

    # Phase 7: Compute BT loss
    print("\n--- Phase 7: BT loss ---")
    scores_a = scores_train[0]
    scores_b = scores_train[1]
    for head_idx in range(scores_a.shape[0]):
        diff = scores_a[head_idx] - scores_b[head_idx]
        bt = -F.logsigmoid(diff)
        ok_bt = check_nan(f"bt_head_{head_idx}", bt.unsqueeze(0))
        print(f"    head {head_idx}: score_a={scores_a[head_idx].item():.6f}, "
              f"score_b={scores_b[head_idx].item():.6f}, diff={diff.item():.6f}, "
              f"bt={bt.item():.6f}")

    # Compute total loss
    total_bt = sum(-F.logsigmoid(scores_a[i] - scores_b[i]) for i in range(scores_a.shape[0]))
    loss = total_bt / scores_a.shape[0]
    check_nan("loss", loss.unsqueeze(0))
    print(f"  loss = {loss.item():.6f}")

    # Phase 8: Backward
    print("\n--- Phase 8: Backward ---")
    loss.backward()

    from src_ii.btrm_lifecycle import get_all_trainable_params
    all_params = get_all_trainable_params(raw_model, "rtheta")

    n_none = 0
    n_nan = 0
    n_ok = 0
    for p in all_params:
        if p.grad is None:
            n_none += 1
        elif torch.isnan(p.grad).any().item():
            n_nan += 1
        else:
            n_ok += 1

    print(f"  Gradient stats: {n_ok} OK, {n_nan} NaN, {n_none} None")

    grad_norm = torch.nn.utils.clip_grad_norm_(all_params, float('inf'))
    print(f"  Total grad norm: {grad_norm.item():.6g}")

    # Phase 9: Also test make_training_optimizer from train_btrm_differentiable
    print("\n--- Phase 9: Optimizer from training loop ---")
    from src_ii.btrm_lifecycle import make_training_optimizer
    opt2 = make_training_optimizer(raw_model, "rtheta", lr=3e-4)
    print(f"  optimizer2 has {len(opt2.param_groups)} param groups")
    for i, pg in enumerate(opt2.param_groups):
        print(f"    group {i}: {len(pg['params'])} params, lr={pg['lr']}")

    # Phase 10: Test the full _flops_budget_step with diagnostic prints
    print("\n--- Phase 10: Full FLOPS-budget step ---")
    raw_model.zero_grad()

    # Sample a macrobatch
    macro_pair_specs = sampler.sample_macrobatch(
        budget_units=3.0,
        tier_flops_targets={1048576: 0.33},
        allow_cross_resolution=True,
    )
    print(f"  Macrobatch: {len(macro_pair_specs)} pairs")

    # Load VAE for reward_manifest
    from futudiffu.vae import load_vae, vae_decode

    vae = load_vae(VAE_PATH, device=device, dtype=dtype)
    print(f"  VAE loaded")

    from src_ii.reward_functions import pinkify_score_gpu
    from src_ii.tnt_validation import thisnotthat_score_gpu_cached

    THIS_PATH = str(REPO_ROOT / "i2i_off_policies" / "PINKIFY_cases" / "reference_THIS.png")
    THAT_PATH = str(REPO_ROOT / "i2i_off_policies" / "PINKIFY_cases" / "reference_THAT.png")
    tnt_score_bound = thisnotthat_score_gpu_cached(THIS_PATH, THAT_PATH, device=device)

    reward_manifest = {"pinkify": pinkify_score_gpu, "thisnotthat": tnt_score_bound}
    head_names_train = ("pinkify", "thisnotthat")
    pref_keys = ("pinkify_pref", "thisnotthat_pref")

    # Compute preferences for first 3 pairs
    n_prefs_nonzero = 0
    n_prefs_total = 0
    for i, ps in enumerate(macro_pair_specs[:min(3, len(macro_pair_specs))]):
        pd = ps.to_pair_dict()
        with torch.no_grad():
            key_a = (pd["traj_a"], pd["step_a"])
            key_b = (pd["traj_b"], pd["step_b"])
            lat_a_p, _, _, _, _ = load_latent_fn(key_a)
            lat_b_p, _, _, _, _ = load_latent_fn(key_b)
            pixel_a = vae_decode(vae, lat_a_p)[0].float()
            pixel_b = vae_decode(vae, lat_b_p)[0].float()

            print(f"\n  Pair {i}: {key_a} vs {key_b}")
            check_nan(f"  pixel_a", pixel_a)
            check_nan(f"  pixel_b", pixel_b)

            for head_name, pref_key in zip(head_names_train, pref_keys):
                score_fn = reward_manifest[head_name]
                sa = score_fn(pixel_a)
                sb = score_fn(pixel_b)
                sa_val = float(sa.item()) if isinstance(sa, torch.Tensor) else float(sa)
                sb_val = float(sb.item()) if isinstance(sb, torch.Tensor) else float(sb)
                pref = 1 if sa_val > sb_val else (-1 if sb_val > sa_val else 0)
                print(f"    {head_name}: score_a={sa_val:.6f}, score_b={sb_val:.6f}, pref={pref}")
                n_prefs_total += 1
                if pref != 0:
                    n_prefs_nonzero += 1

    print(f"\n  Preference stats (first 3 pairs): {n_prefs_nonzero}/{n_prefs_total} non-zero")

    reader.close()
    del vae
    torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"  NaN Diagnostic Complete")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

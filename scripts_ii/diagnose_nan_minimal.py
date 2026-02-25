r"""Minimal NaN diagnostic: bypass BatchExecutor, test backward directly.

Tests whether the FP8 backward through packed_forward produces NaN
gradients for LoRA adapter params.

Execution:
  PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe ^
      F:\dox\repos\ai\futudiffu\scripts_ii\diagnose_nan_minimal.py
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

from src_ii.model_paths import FP8_PATH, TE_PATH, TOKENIZER_PATH
from src_ii.training_setup import (
    encode_training_prompts,
    build_dataset_positions,
    load_training_backbone,
)
from src_ii.dataset_io import make_load_latent_fn
from src_ii.forward_packed import prepare_packed_forward, packed_forward
from src_ii.multi_lora import MultiLoRALinear, get_adapter_params

DATASET_DIR = REPO_ROOT / "multi_res_trajectories"

device = torch.device("cuda")
dtype = torch.bfloat16


def check_nan(name: str, t: torch.Tensor):
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
    print("  Minimal NaN Diagnostic: Direct packed_forward")
    print("=" * 60)

    # Load dataset + TE
    reader, positions, sampler, traj_ids = build_dataset_positions(
        DATASET_DIR, clean_fraction=0.8,
    )
    prompt_cache = encode_training_prompts(
        reader, traj_ids, TOKENIZER_PATH, TE_PATH, device=device, dtype=dtype,
    )
    load_latent_fn = make_load_latent_fn(reader, prompt_cache, device=device, dtype=dtype)

    # Load model
    raw_model, optimizer, head_names = load_training_backbone(
        FP8_PATH, device=device, dtype=dtype, lr=3e-4,
    )

    # Load one image
    pair_spec = sampler.sample_pair()
    key_a = (pair_spec["traj_a"], pair_spec["step_a"])
    lat_a, ts_a, cond_a, nt_a, _ = load_latent_fn(key_a)
    print(f"\n  Image: {key_a}, lat={lat_a.shape}, sigma={ts_a.item()}")

    # Build adapter_scales
    n_adapters = 0
    for m in raw_model.modules():
        if isinstance(m, MultiLoRALinear) and m.n_adapters > 0:
            n_adapters = m.n_adapters
            break
    adapter_scales = torch.ones(1, n_adapters, device=device)
    print(f"  adapter_scales: {adapter_scales.shape} (n_adapters={n_adapters})")

    # --- Test 1: Direct packed_forward (NO gradient checkpointing) ---
    print("\n--- Test 1: packed_forward WITHOUT gradient checkpointing ---")
    raw_model.gradient_checkpointing = False
    raw_model.train()

    x_list = [lat_a]
    timesteps = [ts_a]
    cap_list = [cond_a]
    cap_lens_list = [nt_a]
    img_sizes = [(lat_a.shape[2], lat_a.shape[3])]  # (H, W) in latent space

    state = prepare_packed_forward(raw_model, cap_list, img_sizes, cap_lens_list, device)
    fields, scores = packed_forward(
        raw_model, x_list, timesteps,
        state["refined_caps"], state["packing_info"],
        state["block_mask"], state["packed_rope"],
        adapter_scales=adapter_scales,
    )

    check_nan("scores", scores)
    print(f"  scores = {scores.tolist()}")
    print(f"  scores.requires_grad = {scores.requires_grad}")
    print(f"  scores.grad_fn = {scores.grad_fn}")

    # Compute simple loss and backward
    loss = scores.sum()
    print(f"  loss = {loss.item()}")
    loss.backward()

    adapter_ps = list(get_adapter_params(raw_model, "rtheta").values())
    head_ps = list(raw_model.score_proj.parameters()) + list(raw_model.score_norm.parameters())

    n_ok_adapter, n_nan_adapter, n_none_adapter = 0, 0, 0
    for p in adapter_ps:
        if p.grad is None:
            n_none_adapter += 1
        elif torch.isnan(p.grad).any().item():
            n_nan_adapter += 1
        else:
            n_ok_adapter += 1

    n_ok_head, n_nan_head, n_none_head = 0, 0, 0
    for p in head_ps:
        if p.grad is None:
            n_none_head += 1
        elif torch.isnan(p.grad).any().item():
            n_nan_head += 1
        else:
            n_ok_head += 1

    print(f"  Adapter grads: {n_ok_adapter} OK, {n_nan_adapter} NaN, {n_none_adapter} None")
    print(f"  Head grads: {n_ok_head} OK, {n_nan_head} NaN, {n_none_head} None")

    if n_nan_adapter > 0 or n_none_adapter > 0:
        # Dig deeper: check each layer
        print("\n  --- Layer-by-layer gradient check ---")
        for name, p in raw_model.named_parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                frac = torch.isnan(p.grad).float().mean().item()
                print(f"    NaN: {name} ({p.shape}) — {frac*100:.1f}% NaN")

    raw_model.zero_grad()
    torch.cuda.empty_cache()

    # --- Test 2: Direct packed_forward WITH gradient checkpointing ---
    print("\n--- Test 2: packed_forward WITH gradient checkpointing ---")
    raw_model.gradient_checkpointing = True

    state2 = prepare_packed_forward(raw_model, cap_list, img_sizes, cap_lens_list, device)
    fields2, scores2 = packed_forward(
        raw_model, x_list, timesteps,
        state2["refined_caps"], state2["packing_info"],
        state2["block_mask"], state2["packed_rope"],
        adapter_scales=adapter_scales,
    )

    check_nan("scores2", scores2)
    print(f"  scores2 = {scores2.tolist()}")
    loss2 = scores2.sum()
    loss2.backward()

    n_ok_adapter2, n_nan_adapter2, n_none_adapter2 = 0, 0, 0
    for p in adapter_ps:
        if p.grad is None:
            n_none_adapter2 += 1
        elif torch.isnan(p.grad).any().item():
            n_nan_adapter2 += 1
        else:
            n_ok_adapter2 += 1

    n_ok_head2, n_nan_head2, n_none_head2 = 0, 0, 0
    for p in head_ps:
        if p.grad is None:
            n_none_head2 += 1
        elif torch.isnan(p.grad).any().item():
            n_nan_head2 += 1
        else:
            n_ok_head2 += 1

    print(f"  Adapter grads: {n_ok_adapter2} OK, {n_nan_adapter2} NaN, {n_none_adapter2} None")
    print(f"  Head grads: {n_ok_head2} OK, {n_nan_head2} NaN, {n_none_head2} None")

    if n_ok_adapter > 0 and n_nan_adapter == 0:
        print("\n  >> Test 1 PASSED: No grad checkpointing works!")
    if n_ok_adapter2 > 0 and n_nan_adapter2 == 0:
        print("\n  >> Test 2 PASSED: Grad checkpointing works!")
    if n_nan_adapter > 0 and n_nan_adapter2 == 0:
        print("\n  >> DIAGNOSIS: Gradient checkpointing causes NaN")
    if n_nan_adapter == 0 and n_nan_adapter2 > 0:
        print("\n  >> DIAGNOSIS: Gradient checkpointing causes NaN (but non-checkpointed is fine)")
    if n_nan_adapter > 0 and n_nan_adapter2 > 0:
        print("\n  >> DIAGNOSIS: NaN present in both modes — problem is in FP8 backward itself")

    reader.close()
    torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"  Minimal NaN Diagnostic Complete")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

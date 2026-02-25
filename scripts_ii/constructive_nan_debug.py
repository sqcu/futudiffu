r"""Constructive NaN debugging: start from pure BT loss, add layers toward model.

Level 0: Pure BT loss on random scores from a linear layer. Must not NaN.
Level 1: Replace linear with score_packed on random latents.
Level 2: Add bin-packing orchestration.
Level 3: Add real latents from the dataset.
Level 4: Full training step.

Each level prints PASS/FAIL. First FAIL is the bug location.

Execution:
  PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe ^
      F:\dox\repos\ai\futudiffu\scripts_ii\constructive_nan_debug.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")
dtype = torch.bfloat16


def bt_loss(scores_a, scores_b, pref):
    """Exact BT loss from btrm_training.py L1025-1037."""
    total = torch.zeros((), device=scores_a.device)
    active = 0
    for head_idx in range(scores_a.shape[0]):
        if pref[head_idx] == 0:
            continue
        if pref[head_idx] > 0:
            pos_s = scores_a[head_idx]
            neg_s = scores_b[head_idx]
        else:
            pos_s = scores_b[head_idx]
            neg_s = scores_a[head_idx]
        bt = -F.logsigmoid(pos_s - neg_s)
        total = total + bt
        active += 1
    return total, active


def check(name, t):
    ok = not (torch.isnan(t).any().item() or torch.isinf(t).any().item())
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}: {t.item():.6f}" if t.numel() == 1
          else f"  [{tag}] {name}: shape={t.shape} min={t.min().item():.4g} max={t.max().item():.4g}")
    return ok


# =====================================================================
# LEVEL 0: Pure BT loss, standalone linear, random input
# =====================================================================
def level_0():
    print("\n" + "=" * 60)
    print("LEVEL 0: Pure BT loss + standalone linear + random input")
    print("=" * 60)

    # Mimic score head: linear over pooled hidden states
    proj = nn.Linear(3840, 2, bias=False).to(device=device, dtype=dtype)
    nn.init.zeros_(proj.weight)  # zero-init like ZImageRLAIF
    opt = torch.optim.AdamW(proj.parameters(), lr=3e-4)
    opt.zero_grad()

    # Two random "hidden state" vectors (as if mean-pooled from transformer)
    h_a = torch.randn(1, 3840, device=device, dtype=dtype)
    h_b = torch.randn(1, 3840, device=device, dtype=dtype)

    scores_a = proj(h_a).squeeze(0)  # (2,)
    scores_b = proj(h_b).squeeze(0)  # (2,)

    check("scores_a", scores_a)
    check("scores_b", scores_b)

    loss, active = bt_loss(scores_a, scores_b, pref=[1, -1])
    loss = loss / max(active, 1)

    ok = check("loss", loss)
    if not ok:
        return False

    loss.backward()
    grad_ok = check("proj.weight.grad", proj.weight.grad)
    norm = torch.nn.utils.clip_grad_norm_(proj.parameters(), 0.1)
    check("grad_norm", norm)

    opt.step()
    print("  LEVEL 0: PASS")
    return True


# =====================================================================
# LEVEL 1: score_packed with ONE synthetic image (random latent)
# =====================================================================
def level_1():
    print("\n" + "=" * 60)
    print("LEVEL 1: score_packed with synthetic random latent")
    print("=" * 60)

    from src_ii.zimage_model import load_zimage_rlaif
    from src_ii.btrm_lifecycle import setup_btrm_training, score_packed, get_all_trainable_params

    FP8_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"

    model = load_zimage_rlaif(FP8_PATH, device=device, dtype=dtype, compile_model=False, fuse=True)
    setup_btrm_training(model, adapter_name="rtheta", adapter_rank=8, adapter_alpha=16.0, adapter_init_b_std=0.01)

    model.gradient_checkpointing = True
    model.train()

    # Synthetic latent: (1, 16, 16, 16) = 128x128 pixel image
    lat = torch.randn(1, 16, 16, 16, device=device, dtype=dtype) * 0.1
    ts = torch.tensor([0.5], device=device, dtype=dtype)
    # Synthetic conditioning: (1, 128, 2560)
    cond = torch.randn(1, 128, 2560, device=device, dtype=dtype) * 0.01
    nt = 20

    images = [(lat, ts, cond, nt)]
    scores = score_packed(model, images, gradient_checkpointing=True)
    ok = check("scores", scores)
    print(f"  scores = {scores.tolist()}")
    print(f"  requires_grad = {scores.requires_grad}, grad_fn = {scores.grad_fn}")

    if not ok:
        print("  LEVEL 1: FAIL — score_packed produces NaN on synthetic input")
        return False, model

    # BT loss: score this image against itself (dummy, just testing backward)
    images2 = [(lat, ts, cond, nt), (lat.clone(), ts.clone(), cond.clone(), nt)]
    scores2 = score_packed(model, images2, gradient_checkpointing=True)
    check("scores2", scores2)

    loss, active = bt_loss(scores2[0], scores2[1], pref=[1, -1])
    loss = loss / max(active, 1)
    ok_loss = check("loss", loss)

    if not ok_loss:
        print("  LEVEL 1: FAIL — BT loss is NaN despite valid scores")
        return False, model

    loss.backward()
    all_params = get_all_trainable_params(model, "rtheta")
    n_none = sum(1 for p in all_params if p.grad is None)
    n_nan = sum(1 for p in all_params if p.grad is not None and torch.isnan(p.grad).any())
    n_ok = sum(1 for p in all_params if p.grad is not None and not torch.isnan(p.grad).any())
    print(f"  Grads: {n_ok} OK, {n_nan} NaN, {n_none} None")

    head_ps = list(model.score_proj.parameters()) + list(model.score_norm.parameters())
    for i, p in enumerate(head_ps):
        if p.grad is not None:
            check(f"head_param_{i}.grad", p.grad)
        else:
            print(f"  head_param_{i}: grad=None")

    grad_norm = torch.nn.utils.clip_grad_norm_(all_params, float('inf'))
    check("grad_norm", grad_norm)

    model.zero_grad()
    torch.cuda.empty_cache()

    if n_nan > 0:
        print("  LEVEL 1: FAIL — NaN gradients")
        return False, model

    print("  LEVEL 1: PASS")
    return True, model


# =====================================================================
# LEVEL 2: score_packed with REAL latent from dataset
# =====================================================================
def level_2(model):
    print("\n" + "=" * 60)
    print("LEVEL 2: score_packed with real dataset latent")
    print("=" * 60)

    from futudiffu.dataset_v2 import DatasetReader
    from src_ii.sigma_schedule import build_sigma_schedule, resolution_shift
    from src_ii.btrm_lifecycle import score_packed, get_all_trainable_params, make_training_optimizer
    from futudiffu.text_encoder import create_tokenizer, load_text_encoder, encode_prompt

    DATASET_DIR = REPO_ROOT / "multi_res_trajectories"
    TE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"
    TOKENIZER_PATH = str(REPO_ROOT / "src" / "futudiffu" / "tokenizer")

    reader = DatasetReader(str(DATASET_DIR))
    traj_ids = sorted(reader._row_lookup.keys())
    tid = traj_ids[0]
    row, accessor = reader[tid]

    # Load first available step latent
    step_key = accessor.available_steps[0]  # e.g. "step_00"
    lat = accessor[step_key].to(device=device, dtype=dtype)
    if lat.dim() == 3:
        lat = lat.unsqueeze(0)

    w = row.get("width", 1280)
    h = row.get("height", 832)
    shift = resolution_shift(w, h)
    sigmas = build_sigma_schedule(row.get("n_steps", 30), sampling_shift=shift, device="cpu", dtype=torch.float32)
    ts = torch.tensor([float(sigmas[0].item())], device=device, dtype=dtype)

    # Encode prompt
    tokenizer = create_tokenizer(TOKENIZER_PATH)
    te = load_text_encoder(TE_PATH, device=device, dtype=dtype)
    prompt = row.get("prompt", "")
    cond = encode_prompt(te, tokenizer, prompt, device=device)
    del te
    torch.cuda.empty_cache()

    nt = cond.shape[1]

    check("real_lat", lat)
    check("real_ts", ts)
    check("real_cond", cond)
    print(f"  Image: traj={tid}, {w}x{h}, sigma={ts.item():.4f}, tokens={nt}")

    # Score single image
    model.gradient_checkpointing = True
    model.train()
    optimizer = make_training_optimizer(model, "rtheta", lr=3e-4)
    optimizer.zero_grad()

    images = [(lat, ts, cond, nt)]
    scores = score_packed(model, images, gradient_checkpointing=True)
    ok = check("scores", scores)
    print(f"  scores = {scores.tolist()}, grad_fn = {scores.grad_fn}")

    if not ok:
        print("  LEVEL 2: FAIL — NaN scores on real latent")
        reader.close()
        return False

    # Now score TWO real images (pair)
    tid2 = traj_ids[1]
    row2, accessor2 = reader[tid2]
    lat2 = accessor2[accessor2.available_steps[0]].to(device=device, dtype=dtype)
    if lat2.dim() == 3:
        lat2 = lat2.unsqueeze(0)
    w2, h2 = row2.get("width", 1280), row2.get("height", 832)
    shift2 = resolution_shift(w2, h2)
    sigmas2 = build_sigma_schedule(row2.get("n_steps", 30), sampling_shift=shift2, device="cpu", dtype=torch.float32)
    ts2 = torch.tensor([float(sigmas2[0].item())], device=device, dtype=dtype)
    prompt2 = row2.get("prompt", "")
    cond2 = cond if prompt2 == prompt else cond  # reuse if same, fine for diag

    images_pair = [
        (lat, ts, cond, nt),
        (lat2, ts2, cond2, cond2.shape[1]),
    ]
    scores_pair = score_packed(model, images_pair, gradient_checkpointing=True)
    check("scores_pair", scores_pair)
    print(f"  pair scores = {scores_pair.tolist()}")

    # BT loss + backward
    loss, active = bt_loss(scores_pair[0], scores_pair[1], pref=[1, -1])
    loss = loss / max(active, 1)
    ok_loss = check("loss", loss)

    if not ok_loss:
        print("  LEVEL 2: FAIL — NaN BT loss on real pair")
        reader.close()
        return False

    loss.backward()
    all_params = get_all_trainable_params(model, "rtheta")
    n_nan = sum(1 for p in all_params if p.grad is not None and torch.isnan(p.grad).any())
    n_ok = sum(1 for p in all_params if p.grad is not None and not torch.isnan(p.grad).any())
    n_none = sum(1 for p in all_params if p.grad is None)
    print(f"  Grads: {n_ok} OK, {n_nan} NaN, {n_none} None")

    grad_norm = torch.nn.utils.clip_grad_norm_(all_params, 0.1)
    check("grad_norm", grad_norm)

    optimizer.step()
    print(f"  Optimizer stepped.")

    model.zero_grad()
    torch.cuda.empty_cache()
    reader.close()

    if n_nan > 0:
        print("  LEVEL 2: FAIL — NaN gradients on real pair")
        return False

    print("  LEVEL 2: PASS")
    return True


# =====================================================================
# LEVEL 3: Multi-image bin-packed scoring (the macrobatch path)
# =====================================================================
def level_3(model):
    print("\n" + "=" * 60)
    print("LEVEL 3: Bin-packed multi-image scoring")
    print("=" * 60)

    from futudiffu.dataset_v2 import DatasetReader
    from src_ii.sigma_schedule import build_sigma_schedule, resolution_shift
    from src_ii.btrm_lifecycle import score_packed, get_all_trainable_params, make_training_optimizer
    from src_ii.bin_packer import BinPackScheduler, compute_effective_seq_len
    from futudiffu.text_encoder import create_tokenizer, load_text_encoder, encode_prompt

    DATASET_DIR = REPO_ROOT / "multi_res_trajectories"
    TE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"
    TOKENIZER_PATH = str(REPO_ROOT / "src" / "futudiffu" / "tokenizer")

    reader = DatasetReader(str(DATASET_DIR))
    traj_ids = sorted(reader._row_lookup.keys())

    # Encode prompts
    tokenizer = create_tokenizer(TOKENIZER_PATH)
    te = load_text_encoder(TE_PATH, device=device, dtype=dtype)
    prompt_cache = {}
    for tid in traj_ids[:20]:
        row, _ = reader[tid]
        p = row.get("prompt", "")
        if p not in prompt_cache:
            prompt_cache[p] = encode_prompt(te, tokenizer, p, device=device).cpu()
    del te
    torch.cuda.empty_cache()

    # Load 8 images (4 pairs)
    all_images = []
    image_resolutions = []
    for tid in traj_ids[:8]:
        row, accessor = reader[tid]
        lat = accessor["final"].to(device=device, dtype=dtype)
        if lat.dim() == 3:
            lat = lat.unsqueeze(0)
        w, h = row.get("width", 1280), row.get("height", 832)
        ts = torch.tensor([0.0], device=device, dtype=dtype)  # final = sigma 0
        prompt = row.get("prompt", "")
        cond = prompt_cache[prompt].to(device=device, dtype=dtype)
        all_images.append((lat, ts, cond, cond.shape[1]))
        image_resolutions.append((w, h))

    print(f"  Loaded {len(all_images)} images")

    # Bin-pack
    packer = BinPackScheduler()
    pack_items = []
    for img_idx, (w, h) in enumerate(image_resolutions):
        seq_len = compute_effective_seq_len(w, h, packer.default_cap_tokens)
        pack_items.append({"img_idx": img_idx, "seq_len": seq_len, "width": w, "height": h})
    bins = packer.pack(pack_items)
    print(f"  {len(bins)} bins: {[len(b) for b in bins]}")

    # Score per-bin, exactly as btrm_training.py does
    model.gradient_checkpointing = True
    model.train()
    optimizer = make_training_optimizer(model, "rtheta", lr=3e-4)
    optimizer.zero_grad()

    img_idx_to_score = {}
    pair_indices = [(0, 1), (2, 3), (4, 5), (6, 7)]
    prefs = [[1, -1], [-1, 1], [1, 1], [-1, -1]]  # per-pair per-head
    pair_processed = [False] * 4
    _norm_denom = 8  # 4 pairs * 2 heads
    total_bt_val = 0.0

    for bi, bin_items in enumerate(bins):
        bin_images = [all_images[item["img_idx"]] for item in bin_items]
        scores = score_packed(model, bin_images, gradient_checkpointing=True)
        ok = check(f"bin_{bi}_scores", scores)

        if not ok:
            print(f"  LEVEL 3: FAIL — NaN in bin {bi}")
            reader.close()
            return False

        for local_idx, item in enumerate(bin_items):
            img_idx_to_score[item["img_idx"]] = scores[local_idx]

        # BT loss for completed pairs (exactly as btrm_training.py L1013-1082)
        bin_bt = torch.zeros((), device=device)
        bin_active = 0
        for k, (idx_a, idx_b) in enumerate(pair_indices):
            if pair_processed[k]:
                continue
            if idx_a not in img_idx_to_score or idx_b not in img_idx_to_score:
                continue
            pair_processed[k] = True
            loss_k, active_k = bt_loss(
                img_idx_to_score[idx_a], img_idx_to_score[idx_b], prefs[k]
            )
            bin_bt = bin_bt + loss_k
            bin_active += active_k

        if bin_active > 0:
            partial_loss = bin_bt / _norm_denom
            has_grad = partial_loss.requires_grad or partial_loss.grad_fn is not None
            print(f"    bin {bi}: bt={bin_bt.item():.6f}, active={bin_active}, "
                  f"partial={partial_loss.item():.6f}, has_grad={has_grad}")
            if has_grad:
                _all_done = all(pair_processed)
                partial_loss.backward(retain_graph=not _all_done)
            total_bt_val += bin_bt.item()

        # Detach completed images
        for item in bin_items:
            idx = item["img_idx"]
            pairs_for_img = [k for k, (a, b) in enumerate(pair_indices) if a == idx or b == idx]
            if all(pair_processed[pk] for pk in pairs_for_img):
                img_idx_to_score[idx] = img_idx_to_score[idx].detach()

    print(f"  total_bt_val = {total_bt_val:.6f}")
    total_loss = total_bt_val / _norm_denom
    print(f"  normalized loss = {total_loss:.6f}")

    all_params = get_all_trainable_params(model, "rtheta")
    n_nan = sum(1 for p in all_params if p.grad is not None and torch.isnan(p.grad).any())
    n_ok = sum(1 for p in all_params if p.grad is not None and not torch.isnan(p.grad).any())
    n_none = sum(1 for p in all_params if p.grad is None)
    print(f"  Grads: {n_ok} OK, {n_nan} NaN, {n_none} None")

    grad_norm = torch.nn.utils.clip_grad_norm_(all_params, 0.1)
    ok_norm = check("grad_norm", grad_norm)

    optimizer.step()

    model.zero_grad()
    torch.cuda.empty_cache()
    reader.close()

    if n_nan > 0 or not ok_norm:
        print("  LEVEL 3: FAIL")
        return False

    print("  LEVEL 3: PASS")
    return True


def main():
    print("=" * 60)
    print("  Constructive NaN Debug")
    print("=" * 60)

    if not level_0():
        return 1

    ok1, model = level_1()
    if not ok1:
        return 1

    if not level_2(model):
        return 1

    if not level_3(model):
        return 1

    print("\n" + "=" * 60)
    print("  ALL LEVELS PASSED — bug is in training loop orchestration")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

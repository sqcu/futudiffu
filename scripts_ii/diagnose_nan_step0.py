r"""Reproduce EXACT step 0 of run_reward_validated_training.py with NaN checks.

Follows the identical code path: load model, setup BTRM, build sampler,
sample macrobatch, compute preferences, load latents, bin-pack, score per bin,
compute BT loss. NaN check at each stage.

Execution:
  PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe ^
      F:\dox\repos\ai\futudiffu\scripts_ii\diagnose_nan_step0.py
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

device = torch.device("cuda")
dtype = torch.bfloat16

FP8_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
TE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"
TOKENIZER_PATH = str(REPO_ROOT / "src" / "futudiffu" / "tokenizer")
DATASET_DIR = REPO_ROOT / "multi_res_trajectories"


def check_nan(name, t):
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    ok = not (has_nan or has_inf)
    flag = " [NaN!]" if has_nan else (" [Inf!]" if has_inf else " [OK]")
    print(f"  {name}: shape={t.shape} dtype={t.dtype} min={t.min().item():.4g} max={t.max().item():.4g}{flag}")
    return ok


def main():
    print("=" * 60)
    print("  NaN Step-0 Reproduction")
    print("=" * 60)

    # ---- Phase 1: Dataset ----
    from futudiffu.dataset_v2 import DatasetReader
    from src_ii.pair_sampler import BTRMPairSampler, build_positions_from_v2
    from src_ii.flops_sampling import compute_flops_sampling_weights_from_positions
    from src_ii.sigma_schedule import build_sigma_schedule, resolution_shift

    reader = DatasetReader(str(DATASET_DIR))
    traj_ids = sorted(reader._row_lookup.keys())
    print(f"  {len(traj_ids)} trajectories")

    positions = build_positions_from_v2(reader)
    flops_w = compute_flops_sampling_weights_from_positions(positions)
    sampler = BTRMPairSampler(positions, flops_weights=flops_w, seed=42)
    print(f"  {len(positions)} positions")

    # ---- Phase 2: Text encoder ----
    from futudiffu.text_encoder import create_tokenizer, load_text_encoder, encode_prompt

    unique_prompts = set()
    for tid in traj_ids:
        row, _ = reader[tid]
        unique_prompts.add(row.get("prompt", ""))

    tokenizer = create_tokenizer(TOKENIZER_PATH)
    te = load_text_encoder(TE_PATH, device=device, dtype=dtype)

    prompt_cache = {}
    for prompt in sorted(unique_prompts):
        cond = encode_prompt(prompt, tokenizer, te, device=device, max_length=128)
        prompt_cache[prompt] = cond.cpu()
    print(f"  {len(prompt_cache)} prompts cached")

    del te
    torch.cuda.empty_cache()

    # ---- Phase 3: Model ----
    from src_ii.zimage_model import load_zimage_rlaif
    from src_ii.btrm_lifecycle import setup_btrm_training, score_packed, get_all_trainable_params, make_training_optimizer
    from src_ii.multi_lora import get_adapter_params

    raw_model = load_zimage_rlaif(FP8_PATH, device=device, dtype=dtype, compile_model=False, fuse=True)
    optimizer_script = setup_btrm_training(raw_model, adapter_name="rtheta", adapter_rank=8, adapter_alpha=16.0, adapter_init_b_std=0.01)
    print(f"  Model loaded, VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Check model weights for NaN
    n_nan_params = 0
    for name, p in raw_model.named_parameters():
        if torch.isnan(p.data).any():
            print(f"  NaN in param: {name}")
            n_nan_params += 1
    print(f"  Model param NaN check: {n_nan_params} NaN params")

    # ---- Phase 4: build load_latent_fn ----
    _v2_meta_cache = {}

    def _get_v2_meta(traj_id):
        if traj_id not in _v2_meta_cache:
            row, accessor = reader[traj_id]
            _v2_meta_cache[traj_id] = (row, accessor)
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
        shift = float(recorded_shift) if recorded_shift is not None else resolution_shift(w, h)

        sigmas = build_sigma_schedule(n_steps_traj, sampling_shift=shift, device="cpu", dtype=torch.float32)

        if step_key == "final":
            sigma_val = 0.0
        else:
            step_idx = int(step_key.split("_")[1])
            sigma_val = float(sigmas[step_idx].item()) if step_idx < len(sigmas) else 0.01

        timestep = torch.tensor([sigma_val], device=device, dtype=dtype)
        prompt = meta.get("prompt", "")
        cond = prompt_cache.get(prompt)
        if cond is None:
            raise ValueError(f"Missing prompt for traj {traj_id}")
        cond = cond.to(device=device, dtype=dtype)
        num_tokens = cond.shape[1]
        return latent, timestep, cond, num_tokens, None

    # ---- Phase 5: Sample macrobatch (seed=42, first draw) ----
    macro_pair_specs = sampler.sample_macrobatch(
        budget_units=3.0,
        tier_flops_targets={1048576: 0.33},
        allow_cross_resolution=True,
    )
    print(f"\n  Macrobatch: {len(macro_pair_specs)} pairs")

    # ---- Phase 6: Compute preferences (skip VAE, use dummy +1/-1) ----
    HEAD_NAMES = ("pinkify", "thisnotthat")
    PREF_KEYS = ("pinkify_pref", "thisnotthat_pref")

    macro_pairs = []
    for i, ps in enumerate(macro_pair_specs):
        pd = ps.to_pair_dict()
        # Use alternating preferences: this guarantees non-zero BT terms
        pd["pinkify_pref"] = 1 if i % 2 == 0 else -1
        pd["thisnotthat_pref"] = -1 if i % 3 == 0 else 1
        macro_pairs.append(pd)

    # ---- Phase 7: Load all latents ----
    all_images = []
    image_resolutions = []
    pair_image_indices = []

    for pd in macro_pairs:
        key_a = (pd["traj_a"], pd["step_a"])
        key_b = (pd["traj_b"], pd["step_b"])

        lat_a, ts_a, cond_a, nt_a, _ = load_latent_fn(key_a)
        lat_b, ts_b, cond_b, nt_b, _ = load_latent_fn(key_b)

        idx_a = len(all_images)
        all_images.append((lat_a, ts_a, cond_a, nt_a))
        _, _, lh_a, lw_a = lat_a.shape
        image_resolutions.append((lw_a * 8, lh_a * 8))

        idx_b = len(all_images)
        all_images.append((lat_b, ts_b, cond_b, nt_b))
        _, _, lh_b, lw_b = lat_b.shape
        image_resolutions.append((lw_b * 8, lh_b * 8))

        pair_image_indices.append((idx_a, idx_b))

    print(f"  Loaded {len(all_images)} images")

    # Check inputs for NaN
    for i, (lat, ts, cond, nt) in enumerate(all_images[:4]):
        check_nan(f"lat[{i}]", lat)
        check_nan(f"ts[{i}]", ts)
        check_nan(f"cond[{i}]", cond)

    # ---- Phase 8: Bin-pack ----
    from src_ii.bin_packer import BinPackScheduler, compute_effective_seq_len

    packer = BinPackScheduler()
    pack_items = []
    for img_idx, (w, h) in enumerate(image_resolutions):
        seq_len = compute_effective_seq_len(w, h, packer.default_cap_tokens)
        pack_items.append({"img_idx": img_idx, "seq_len": seq_len, "width": w, "height": h})

    bins = packer.pack(pack_items)
    print(f"  Packed into {len(bins)} bins")
    for bi, b in enumerate(bins):
        total = sum(item["seq_len"] for item in b)
        print(f"    bin {bi}: {len(b)} items, context_len={total}")

    # ---- Phase 9: Score each bin, check for NaN ----
    raw_model.gradient_checkpointing = True
    raw_model.train()

    # Replicate the EXACT training optimizer from train_btrm_differentiable
    optimizer = make_training_optimizer(raw_model, "rtheta", lr=3e-4)
    optimizer.zero_grad()

    img_idx_to_score = {}
    all_ok = True

    for bi, bin_items in enumerate(bins):
        bin_images = []
        for item in bin_items:
            idx = item["img_idx"]
            bin_images.append(all_images[idx])

        scores = score_packed(raw_model, bin_images, gradient_checkpointing=True)

        ok = check_nan(f"bin_{bi}_scores", scores)
        if not ok:
            all_ok = False
            print(f"    >> BIN {bi} PRODUCED NaN SCORES!")
        else:
            print(f"    bin {bi}: scores min={scores.min().item():.6f} max={scores.max().item():.6f} "
                  f"grad_fn={scores.grad_fn}")

        for local_idx, item in enumerate(bin_items):
            img_idx_to_score[item["img_idx"]] = scores[local_idx]

    if not all_ok:
        print("\n  >> DIAGNOSIS: score_packed produces NaN. Bug is in forward pass.")
        reader.close()
        return 1

    # ---- Phase 10: BT loss ----
    print("\n--- Phase 10: BT loss ---")
    total_bt = torch.zeros((), device=device)
    active_heads = 0
    pair_processed = [False] * len(macro_pairs)

    # Pre-count active heads
    precount = 0
    for pd in macro_pairs:
        for pref_key in PREF_KEYS:
            if pd.get(pref_key, 0) != 0:
                precount += 1
    norm_denom = max(precount, 1)
    print(f"  precount_active_heads={precount}, norm_denom={norm_denom}")

    for k, pd in enumerate(macro_pairs):
        idx_a, idx_b = pair_image_indices[k]
        scores_a = img_idx_to_score[idx_a]
        scores_b = img_idx_to_score[idx_b]

        for head_idx, (name, pref_key) in enumerate(zip(HEAD_NAMES, PREF_KEYS)):
            pref = pd[pref_key]
            if pref == 0:
                continue
            if pref > 0:
                pos_s = scores_a[head_idx]
                neg_s = scores_b[head_idx]
            else:
                pos_s = scores_b[head_idx]
                neg_s = scores_a[head_idx]

            bt = -F.logsigmoid(pos_s - neg_s)
            total_bt = total_bt + bt
            active_heads += 1

            if active_heads <= 5:
                print(f"  pair {k} head {name}: pos={pos_s.item():.6f} neg={neg_s.item():.6f} "
                      f"bt={bt.item():.6f} bt_nan={torch.isnan(bt).item()}")

    print(f"  active_heads={active_heads}")
    loss = total_bt / norm_denom
    ok_loss = check_nan("loss", loss.unsqueeze(0))

    if not ok_loss:
        print("\n  >> DIAGNOSIS: BT loss is NaN despite valid scores")
        reader.close()
        return 1

    # ---- Phase 11: Backward ----
    print("\n--- Phase 11: Backward ---")
    loss.backward()

    all_params = get_all_trainable_params(raw_model, "rtheta")
    n_none, n_nan, n_ok = 0, 0, 0
    for p in all_params:
        if p.grad is None:
            n_none += 1
        elif torch.isnan(p.grad).any():
            n_nan += 1
        else:
            n_ok += 1

    print(f"  Grads: {n_ok} OK, {n_nan} NaN, {n_none} None (total={len(all_params)})")

    grad_norm = torch.nn.utils.clip_grad_norm_(all_params, float('inf'))
    print(f"  Total grad norm: {grad_norm.item():.6g}")

    # Check score head specifically
    head_ps = list(raw_model.score_proj.parameters()) + list(raw_model.score_norm.parameters())
    for i, p in enumerate(head_ps):
        if p.grad is not None:
            check_nan(f"head_param_{i}_grad", p.grad)
        else:
            print(f"  head_param_{i}: grad=None")

    reader.close()
    torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"  Diagnosis Complete")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

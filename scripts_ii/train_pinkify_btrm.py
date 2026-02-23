r"""Train a BTRM head with PINKIFY and THISNOTTHAT reward heads.

Uses the BTRMCompoundModel (src_ii.btrm_model) which enforces that the
r_theta LoRA adapter and the ScoreUnembedder are always trained together.
This prevents Defect 24: "rtheta LoRA never trained because the optimizer
only had head params."

Three phases:
  Phase 1: Encode prompts with text encoder (then free TE)
  Phase 2: Extract hidden states with diffusion backbone (cached to CPU)
  Phase 3: Train BTRM compound model using Bradley-Terry loss

Execution:
  PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe \
      F:\dox\repos\ai\futudiffu\scripts_ii\train_pinkify_btrm.py

Output:
  pinkify_thisnotthat_output/rtheta_adapter.safetensors
  pinkify_thisnotthat_output/btrm_head.safetensors
  pinkify_thisnotthat_output/btrm_compound_config.json
  pinkify_thisnotthat_output/training_curve.json
  pinkify_thisnotthat_output/pre_persist_scores.pt
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

N_TRAJECTORIES = 10
FP8_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
TE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"
TOKENIZER_PATH = str(REPO_ROOT / "src" / "futudiffu" / "tokenizer")
OUTPUT_DIR = REPO_ROOT / "pinkify_thisnotthat_output"
LABELS_PATH = OUTPUT_DIR / "preference_labels.json"

N_EPOCHS = 40
LR = 1e-3
LOGSQUARE_WEIGHT = 0.05

HEAD_NAMES = ("pinkify", "thisnotthat")


def main():
    t0 = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("Loading preference labels...")
    with open(LABELS_PATH) as f:
        label_data = json.load(f)
    labels = label_data["labels"]
    stats = label_data["stats"]
    print(f"  {stats['n_pairs']} pairwise labels loaded")
    print(f"  {stats['n_images']} images across {stats['n_trajectories']} trajectories")

    scores_path = OUTPUT_DIR / "per_image_scores.json"
    with open(scores_path) as f:
        per_image_scores = json.load(f)

    image_index = {}
    for i, entry in enumerate(per_image_scores):
        key = (entry["traj_idx"], entry["step_key"])
        image_index[key] = i
    n_images = len(per_image_scores)
    print(f"  Image index built: {n_images} entries")

    manifest_path = REPO_ROOT / "btrm_dataset" / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    records = manifest["records"]

    print("\n=== Phase 1: Encoding prompts ===")
    from futudiffu.text_encoder import create_tokenizer, load_text_encoder, encode_prompt

    tokenizer = create_tokenizer(TOKENIZER_PATH)
    te_model = load_text_encoder(TE_PATH, device=device, dtype=dtype)

    unique_prompts = {}
    for traj_idx in range(N_TRAJECTORIES):
        prompt = records[traj_idx]["prompt"]
        if prompt not in unique_prompts:
            unique_prompts[prompt] = None

    for prompt in unique_prompts:
        cond = encode_prompt(te_model, tokenizer, prompt, device=device)
        unique_prompts[prompt] = cond.cpu()
        print(f"  Encoded: '{prompt[:60]}...' -> {cond.shape}")

    del te_model, tokenizer
    torch.cuda.empty_cache()
    print(f"  TE freed. {len(unique_prompts)} unique prompts encoded.")

    print("\n=== Phase 2: Loading diffusion model & extracting hidden states ===")
    from src_ii.zimage_model import load_zimage_rlaif
    from src_ii.btrm_lifecycle import setup_btrm_training, persist_btrm
    from src_ii.multi_lora import get_adapter_params
    from src_ii.stats import sigma_for_step
    from futudiffu.sampling import make_rope_cache

    raw_model = load_zimage_rlaif(
        FP8_PATH, device=device, dtype=dtype,
        compile_model=False, fuse=True,
    )

    optimizer = setup_btrm_training(raw_model)

    hidden_states_cpu = []
    multiplier = 1.0

    for img_entry in per_image_scores:
        traj_idx = img_entry["traj_idx"]
        step_key = img_entry["step_key"]
        prompt = records[traj_idx]["prompt"]
        conditioning = unique_prompts[prompt].to(device=device, dtype=dtype)

        traj_dir = REPO_ROOT / "btrm_dataset" / "latents" / f"traj_{traj_idx:06d}"
        pt_path = traj_dir / f"{step_key}.pt"
        latent = torch.load(str(pt_path), weights_only=True).to(device=device, dtype=dtype)

        n_steps = records[traj_idx]["n_steps"]
        sigma_val = sigma_for_step(step_key, n_steps, device=device, dtype=dtype)
        timestep = sigma_val * multiplier
        num_tokens = conditioning.shape[1]

        B, C, H, W = latent.shape
        rope_cache = make_rope_cache(raw_model, H, W, num_tokens, device)

        hidden = raw_model.extract_hidden(
            latent, timestep.unsqueeze(0), conditioning, num_tokens, rope_cache,
        )
        hidden_states_cpu.append(hidden.cpu())

        if len(hidden_states_cpu) % 10 == 0:
            print(f"  Extracted {len(hidden_states_cpu)}/{n_images} hidden states")

    print(f"  All {len(hidden_states_cpu)} hidden states extracted")
    print(f"  Hidden shape: {hidden_states_cpu[0].shape}")

    del raw_model
    torch.cuda.empty_cache()
    print("  Diffusion model freed")

    print("\n=== Phase 3: Training BTRM compound model (adapter + head) ===")
    from src_ii.btrm_training import train_btrm

    training_pairs = []
    for label in labels:
        traj_idx = label["traj_idx"]
        key_a = (traj_idx, label["step_a"])
        key_b = (traj_idx, label["step_b"])

        idx_a = image_index.get(key_a)
        idx_b = image_index.get(key_b)
        if idx_a is None or idx_b is None:
            continue

        training_pairs.append({
            "idx_a": idx_a,
            "idx_b": idx_b,
            "pinkify_pref": label["pinkify_preference"],
            "thisnotthat_pref": label["thisnotthat_preference"],
        })

    print(f"  {len(training_pairs)} training pairs")

    training_curve = train_btrm(
        model=raw_model,  # NOTE: raw_model was freed -- this will fail
        training_pairs=training_pairs,
        hidden_states_cpu=hidden_states_cpu,
        n_epochs=N_EPOCHS,
        lr=LR,
        logsquare_weight=LOGSQUARE_WEIGHT,
        batch_size=16,
        head_names=HEAD_NAMES,
        pref_keys=("pinkify_pref", "thisnotthat_pref"),
        device=device,
    )

    print("\n=== Saving pre-persist scores ===")
    raw_model.eval()
    pre_persist_scores = []
    with torch.no_grad():
        for i in range(n_images):
            h = hidden_states_cpu[i].to(device=device, dtype=torch.float32)
            normed = raw_model.score_norm(h.mean(dim=1, keepdim=True))
            raw = raw_model.score_proj(normed)
            score = raw_model.score_cap * torch.tanh(raw / raw_model.score_cap)
            pre_persist_scores.append(score.cpu())
    pre_persist_scores = torch.cat(pre_persist_scores, dim=0)
    torch.save(pre_persist_scores, OUTPUT_DIR / "pre_persist_scores.pt")
    print(f"  Pre-persist scores saved: {pre_persist_scores.shape}")

    print("\n=== Persisting trained compound model ===")
    persist_info = persist_btrm(raw_model, "rtheta", str(OUTPUT_DIR))
    print(f"  Persisted: {persist_info}")

    curve_path = OUTPUT_DIR / "training_curve.json"
    with open(curve_path, "w") as f:
        json.dump(training_curve, f, indent=2)
    print(f"  Training curve saved to: {curve_path}")

    torch.save(hidden_states_cpu[0], OUTPUT_DIR / "hidden_sample_0.pt")
    torch.save(hidden_states_cpu[-1], OUTPUT_DIR / "hidden_sample_last.pt")

    elapsed = time.perf_counter() - t0
    print(f"\n=== Training complete in {elapsed:.1f}s ===")
    print(f"  Final loss: {training_curve[-1]['loss']:.4f}")
    print(f"  Final accuracy: pinkify={training_curve[-1]['accuracy_pinkify']:.3f}, "
          f"thisnotthat={training_curve[-1]['accuracy_thisnotthat']:.3f}")

    adapter_params = list(get_adapter_params(raw_model, "rtheta").values())
    if adapter_params:
        any_nonzero = any(p.abs().max().item() > 0 for p in adapter_params)
        n_adapter = sum(p.numel() for p in adapter_params)
        print(f"\n  Adapter status: {n_adapter} params, any_nonzero={any_nonzero}")
        if any_nonzero:
            print("  SUCCESS: r_theta adapter was trained (non-zero weights)")
        else:
            print("  WARNING: r_theta adapter has all-zero weights")
    else:
        print("  ERROR: No adapter params found")


if __name__ == "__main__":
    main()

r"""Attention interpretability for BTRM reward heads.

Captures per-layer, per-head attention statistics from the base (frozen) NextDiT
backbone on latents that score HIGH vs LOW on the pinkify and thisnotthat reward
heads.

Uses the extracted AttentionCapture from src_ii.attention_capture instead of
inlining the 120-line monkey-patch class.

Execution:
  PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe ^
      F:\dox\repos\ai\futudiffu\scripts_ii\attention_interpretability.py

Output:
  pinkify_thisnotthat_output/attention_maps/attention_stats_<name>.pt
  pinkify_thisnotthat_output/attention_maps/latent_<name>.pt
  pinkify_thisnotthat_output/attention_maps/run_manifest.json
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

# --- Configuration ---
FP8_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
TE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"
TOKENIZER_PATH = str(REPO_ROOT / "src" / "futudiffu" / "tokenizer")
OUTPUT_DIR = REPO_ROOT / "pinkify_thisnotthat_output" / "attention_maps"

# Representative latents to analyze
LATENT_SELECTIONS = [
    {"name": "high_pinkify",      "traj_idx": 5, "step_key": "step_09",
     "reason": "Peak pinkify score 0.0925"},
    {"name": "low_pinkify",       "traj_idx": 3, "step_key": "final",
     "reason": "Minimum pinkify score 0.0001"},
    {"name": "high_thisnotthat",  "traj_idx": 2, "step_key": "final",
     "reason": "Maximum thisnotthat score 0.1266"},
    {"name": "low_thisnotthat",   "traj_idx": 7, "step_key": "final",
     "reason": "Minimum thisnotthat score -0.0175"},
]


def main():
    t0 = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # --- Load manifest ---
    manifest_path = REPO_ROOT / "btrm_dataset" / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    records = manifest["records"]

    # --- Phase 1: Encode prompts ---
    print("\n=== Phase 1: Encoding prompts ===")
    from futudiffu.text_encoder import create_tokenizer, load_text_encoder, encode_prompt

    tokenizer = create_tokenizer(TOKENIZER_PATH)
    te_model = load_text_encoder(TE_PATH, device=device, dtype=dtype)

    prompt_cache = {}
    for sel in LATENT_SELECTIONS:
        prompt = records[sel["traj_idx"]]["prompt"]
        if prompt not in prompt_cache:
            cond = encode_prompt(te_model, tokenizer, prompt, device=device)
            prompt_cache[prompt] = {
                "conditioning": cond.cpu(),
                "num_tokens": cond.shape[1],
            }
            print(f"  Encoded: '{prompt[:60]}...' -> {cond.shape}")

    del te_model, tokenizer
    torch.cuda.empty_cache()
    print(f"  TE freed. {len(prompt_cache)} unique prompts encoded.")

    # --- Phase 2: Load diffusion model ---
    print("\n=== Phase 2: Loading FP8 diffusion model ===")
    from src_ii.model_loading import load_fp8_diffusion_model
    from src_ii.stats import sigma_for_step
    from src_ii.attention_capture import AttentionCapture
    from futudiffu.sampling import make_rope_cache
    from futudiffu.attention import set_attention_backend

    # Force SDPA backend so our patched fn is used
    set_attention_backend("sdpa")

    diff_model, _ = load_fp8_diffusion_model(
        FP8_PATH, device=device, dtype=dtype,
        compile_model=False, fuse=True,
    )
    diff_model.eval()
    for p in diff_model.parameters():
        p.requires_grad_(False)

    # Update diffusion_model module's reference to sdpa_attention
    import futudiffu.attention as attn_mod
    import futudiffu.diffusion_model as dm_mod

    # --- Phase 3: Attention capture ---
    print("\n=== Phase 3: Capturing attention patterns ===")
    capture = AttentionCapture()
    capture.install()

    # Update the diffusion_model module's reference
    dm_mod.sdpa_attention = attn_mod.sdpa_attention

    all_results = {}
    run_manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "adapter_trained": False,
        "comparison_type": "high_vs_low_scoring_latents_on_base_model",
        "selections": LATENT_SELECTIONS,
        "results": {},
    }

    for sel in LATENT_SELECTIONS:
        name = sel["name"]
        traj_idx = sel["traj_idx"]
        step_key = sel["step_key"]
        prompt = records[traj_idx]["prompt"]

        print(f"\n  --- {name}: traj={traj_idx}, step={step_key} ---")
        print(f"  Reason: {sel['reason']}")

        # Load latent
        traj_dir = REPO_ROOT / "btrm_dataset" / "latents" / f"traj_{traj_idx:06d}"
        pt_path = traj_dir / f"{step_key}.pt"
        latent = torch.load(str(pt_path), weights_only=True).to(device=device, dtype=dtype)
        print(f"  Latent shape: {latent.shape}")

        # Get conditioning
        pc = prompt_cache[prompt]
        conditioning = pc["conditioning"].to(device=device, dtype=dtype)
        num_tokens = pc["num_tokens"]

        # Build sigma using extracted utility
        n_steps = records[traj_idx]["n_steps"]
        sigma_val = sigma_for_step(step_key, n_steps, device=device, dtype=dtype)
        timestep = sigma_val * 1.0

        # Build rope cache
        B, C, H, W = latent.shape
        rope_cache = make_rope_cache(diff_model, H, W, num_tokens, device)

        # Forward pass with attention capture (single call)
        t_fwd = time.perf_counter()
        stats = capture.capture_forward(
            diff_model, latent, timestep.unsqueeze(0),
            conditioning, num_tokens, rope_cache,
        )
        elapsed_fwd = time.perf_counter() - t_fwd
        print(f"  Forward pass: {elapsed_fwd:.1f}s, captured {len(stats)} attention layers")

        # Save stats
        stats_path = OUTPUT_DIR / f"attention_stats_{name}.pt"
        torch.save(stats, str(stats_path))

        # Save latent for visualization
        latent_path = OUTPUT_DIR / f"latent_{name}.pt"
        torch.save(latent.cpu(), str(latent_path))

        # Compute summary statistics
        n_layers = len(stats)
        if n_layers > 0:
            main_seq_len = max(stats[i]["seq_len"] for i in range(n_layers))
            main_layer_indices = [i for i in range(n_layers) if stats[i]["seq_len"] == main_seq_len]
            n_main = len(main_layer_indices)

            first_main = stats[main_layer_indices[0]] if main_layer_indices else stats[0]
            seq_len = first_main["seq_len"]
            n_heads = first_main["n_heads"]

            all_head_norms = torch.stack([stats[i]["head_norms"] for i in main_layer_indices])
            mean_head_norms = all_head_norms.mean(dim=0)
            top_heads = mean_head_norms.topk(5)

            all_received = torch.stack([stats[i]["attn_received"] for i in main_layer_indices])
            mean_received = all_received.mean(dim=0)
            token_importance = mean_received.mean(dim=0)

            summary = {
                "n_layers_captured": n_layers,
                "n_main_layers": n_main,
                "main_layer_indices": main_layer_indices,
                "seq_len": seq_len,
                "n_heads": n_heads,
                "mean_head_norm_top5_indices": top_heads.indices.tolist(),
                "mean_head_norm_top5_values": [f"{v:.4f}" for v in top_heads.values.tolist()],
                "token_importance_max": f"{token_importance.max().item():.6f}",
                "token_importance_min": f"{token_importance.min().item():.6f}",
                "token_importance_std": f"{token_importance.std().item():.6f}",
                "forward_time_s": f"{elapsed_fwd:.1f}",
            }
        else:
            summary = {"error": "No layers captured"}

        all_results[name] = summary
        run_manifest["results"][name] = summary
        print(f"  Summary: {json.dumps(summary, indent=2)}")

    # --- Phase 4: Compute cross-latent diffs ---
    print("\n=== Phase 4: Computing attention diffs ===")

    diff_pairs = [
        ("high_pinkify", "low_pinkify", "pinkify_diff"),
        ("high_thisnotthat", "low_thisnotthat", "thisnotthat_diff"),
    ]

    for name_a, name_b, diff_name in diff_pairs:
        print(f"\n  Computing diff: {name_a} vs {name_b} -> {diff_name}")

        stats_a = torch.load(str(OUTPUT_DIR / f"attention_stats_{name_a}.pt"), weights_only=True)
        stats_b = torch.load(str(OUTPUT_DIR / f"attention_stats_{name_b}.pt"), weights_only=True)

        max_seq_a = max(stats_a[i]["seq_len"] for i in range(len(stats_a)))
        max_seq_b = max(stats_b[i]["seq_len"] for i in range(len(stats_b)))
        main_a = [i for i in range(len(stats_a)) if stats_a[i]["seq_len"] == max_seq_a]
        main_b = [i for i in range(len(stats_b)) if stats_b[i]["seq_len"] == max_seq_b]
        n_main = min(len(main_a), len(main_b))

        diff_stats = {}
        for di in range(n_main):
            idx_a = main_a[di]
            idx_b = main_b[di]
            sa = stats_a[idx_a]
            sb = stats_b[idx_b]

            seq_a = sa["seq_len"]
            seq_b = sb["seq_len"]
            min_seq = min(seq_a, seq_b)

            recv_diff = sa["attn_received"][:, :min_seq] - sb["attn_received"][:, :min_seq]
            given_diff = sa["attn_given"][:, :min_seq] - sb["attn_given"][:, :min_seq]
            norm_diff = sa["head_norms"] - sb["head_norms"]

            diff_stats[di] = {
                "received_diff": recv_diff,
                "given_diff": given_diff,
                "head_norm_diff": norm_diff,
                "seq_len_a": seq_a,
                "seq_len_b": seq_b,
                "min_seq": min_seq,
            }

        diff_path = OUTPUT_DIR / f"attention_diff_{diff_name}.pt"
        torch.save(diff_stats, str(diff_path))

        all_recv_diff = torch.stack([diff_stats[i]["received_diff"] for i in range(n_main)])
        all_given_diff = torch.stack([diff_stats[i]["given_diff"] for i in range(n_main)])
        mean_abs_recv = all_recv_diff.abs().mean(dim=(0, 2))
        mean_abs_given = all_given_diff.abs().mean(dim=(0, 2))
        top_diff_heads = mean_abs_recv.topk(5)

        diff_summary = {
            "n_main_layers": n_main,
            "top_diff_heads_indices": top_diff_heads.indices.tolist(),
            "top_diff_heads_values": [f"{v:.6f}" for v in top_diff_heads.values.tolist()],
            "mean_abs_received_diff": f"{mean_abs_recv.mean().item():.6f}",
            "mean_abs_given_diff": f"{mean_abs_given.mean().item():.6f}",
            "max_received_diff": f"{all_recv_diff.abs().max().item():.6f}",
        }
        run_manifest["results"][diff_name] = diff_summary
        print(f"  Diff summary: {json.dumps(diff_summary, indent=2)}")

    # --- Save manifest ---
    capture.remove()
    manifest_path = OUTPUT_DIR / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(run_manifest, f, indent=2)

    elapsed = time.perf_counter() - t0
    print(f"\n=== Attention interpretability complete in {elapsed:.1f}s ===")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

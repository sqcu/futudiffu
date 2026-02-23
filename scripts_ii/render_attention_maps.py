r"""Render attention heatmaps from captured attention statistics.

Loads reduced attention stats from attention_interpretability.py, VAE-decodes
the latents, and renders spatial heatmaps overlaid on decoded images.

Uses extracted modules:
  - src_ii.vae_utils for VAE loading and decoding
  - src_ii.visualization for heatmap rendering, strips, and bar charts

Execution:
  PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe ^
      F:\dox\repos\ai\futudiffu\scripts_ii\render_attention_maps.py

Output:
  pinkify_thisnotthat_output/attention_maps/*.png
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

from src_ii.vae_utils import load_vae, decode_latent_to_pil
from src_ii.visualization import (
    render_attention_map,
    render_heatmap_overlay,
    render_layer_head_heatmap,
    render_strip,
    render_text_token_bar_chart,
)

VAE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\vae\zimage.safetensors"
OUTPUT_DIR = REPO_ROOT / "pinkify_thisnotthat_output" / "attention_maps"

LATENT_NAMES = ["high_pinkify", "low_pinkify", "high_thisnotthat", "low_thisnotthat"]
DIFF_NAMES = ["pinkify_diff", "thisnotthat_diff"]

PATCH_SIZE = 2
VAE_SCALE = 8
PIXELS_PER_TOKEN = PATCH_SIZE * VAE_SCALE


def main():
    t0 = time.perf_counter()
    device = torch.device("cuda")
    dtype = torch.bfloat16

    for name in LATENT_NAMES:
        stats_path = OUTPUT_DIR / f"attention_stats_{name}.pt"
        if not stats_path.exists():
            print(f"ERROR: Missing {stats_path}. Run attention_interpretability.py first.")
            sys.exit(1)

    print("=== Loading VAE ===")
    vae = load_vae(VAE_PATH, device=device, dtype=dtype)

    print("\n=== Rendering attention maps ===")

    decoded_images = {}
    summary_images = []

    for name in LATENT_NAMES:
        print(f"\n  --- {name} ---")

        latent = torch.load(str(OUTPUT_DIR / f"latent_{name}.pt"), weights_only=True)
        stats = torch.load(str(OUTPUT_DIR / f"attention_stats_{name}.pt"), weights_only=True)

        decoded = decode_latent_to_pil(vae, latent, device=device, dtype=dtype)
        decoded.save(str(OUTPUT_DIR / f"decoded_{name}.png"))
        decoded_images[name] = decoded
        print(f"  Decoded: {decoded.size}")

        B, C, H_lat, W_lat = latent.shape
        H_tokens = H_lat // PATCH_SIZE
        W_tokens = W_lat // PATCH_SIZE
        n_img_tokens = H_tokens * W_tokens
        n_img_padded = n_img_tokens + ((-n_img_tokens) % 32)

        first_stats = stats[0]
        seq_len = first_stats["seq_len"]
        cap_len = seq_len - n_img_padded

        print(f"  Spatial: {H_tokens}x{W_tokens} tokens = {n_img_tokens} img tokens")
        print(f"  Caption: {cap_len} tokens")
        print(f"  Layers captured: {len(stats)}")

        overlay = render_attention_map(
            decoded, stats, n_img_tokens, H_tokens, W_tokens, cap_len,
            alpha=0.4, colormap="hot",
        )
        overlay.save(str(OUTPUT_DIR / f"overlay_{name}.png"))
        print(f"  Overlay saved: overlay_{name}.png")

        n_layers = len(stats)
        agg_received = torch.zeros(first_stats["n_heads"], seq_len)
        for layer_idx in range(n_layers):
            agg_received += stats[layer_idx]["attn_received"]
        agg_received /= n_layers
        text_attention = agg_received[:, :cap_len].mean(dim=0)

        text_chart = render_text_token_bar_chart(
            text_attention, width=600, height=200,
            title=f"Text Token Attention: {name}",
        )
        text_chart.save(str(OUTPUT_DIR / f"text_attn_{name}.png"))

        summary_images.append((name, overlay))

    print("\n=== Rendering attention diff maps ===")

    diff_configs = [
        ("pinkify_diff", "high_pinkify", "low_pinkify", "Pinkify: HIGH - LOW"),
        ("thisnotthat_diff", "high_thisnotthat", "low_thisnotthat", "TNT: HIGH - LOW"),
    ]

    diff_overlays = []

    for diff_name, name_a, name_b, title in diff_configs:
        diff_path = OUTPUT_DIR / f"attention_diff_{diff_name}.pt"
        if not diff_path.exists():
            print(f"  Skipping {diff_name} -- file not found")
            continue

        print(f"\n  --- {diff_name} ---")
        diff_stats = torch.load(str(diff_path), weights_only=True)
        n_layers = len(diff_stats)

        base_image = decoded_images[name_a]
        latent_a = torch.load(str(OUTPUT_DIR / f"latent_{name_a}.pt"), weights_only=True)
        B, C, H_lat, W_lat = latent_a.shape
        H_tokens = H_lat // PATCH_SIZE
        W_tokens = W_lat // PATCH_SIZE
        n_img_tokens = H_tokens * W_tokens
        n_img_padded = n_img_tokens + ((-n_img_tokens) % 32)

        first_diff = diff_stats[0]
        min_seq = first_diff["min_seq"]
        n_heads_diff = first_diff["received_diff"].shape[0]
        cap_len_a = first_diff["seq_len_a"] - n_img_padded
        img_start_a = cap_len_a

        agg_diff = torch.zeros(n_heads_diff, min_seq)
        for layer_idx in range(n_layers):
            agg_diff += diff_stats[layer_idx]["received_diff"]
        agg_diff /= n_layers

        if img_start_a + n_img_tokens <= min_seq:
            img_diff = agg_diff[:, img_start_a:img_start_a + n_img_tokens].mean(dim=0)
        else:
            available = min_seq - img_start_a
            img_diff = agg_diff[:, img_start_a:img_start_a + available].mean(dim=0)
            if available < n_img_tokens:
                img_diff = torch.cat([img_diff, torch.zeros(n_img_tokens - available)])

        overlay_diff = render_heatmap_overlay(
            base_image, img_diff, H_tokens, W_tokens,
            alpha=0.5, colormap="diverging",
        )
        overlay_diff.save(str(OUTPUT_DIR / f"overlay_{diff_name}.png"))
        print(f"  Diff overlay saved: overlay_{diff_name}.png")
        diff_overlays.append((title, overlay_diff))

        text_diff = agg_diff[:, :min(cap_len_a, min_seq)].mean(dim=0)
        text_chart = render_text_token_bar_chart(
            text_diff, width=600, height=200,
            title=f"Text Token Diff: {title}",
        )
        text_chart.save(str(OUTPUT_DIR / f"text_diff_{diff_name}.png"))

        layer_head_img = render_layer_head_heatmap(
            diff_stats, n_layers, n_heads_diff,
            width=800, height=400,
            title=f"Layer x Head |Diff|: {title}",
        )
        layer_head_img.save(str(OUTPUT_DIR / f"layer_head_{diff_name}.png"))
        print(f"  Layer-head heatmap saved: layer_head_{diff_name}.png")

    print("\n=== Creating summary images ===")
    if summary_images:
        strip = render_strip(summary_images, max_width=1600)
        strip.save(str(OUTPUT_DIR / "summary_all_overlays.png"))
        print(f"  Summary strip saved: summary_all_overlays.png")

    if diff_overlays:
        diff_strip = render_strip(diff_overlays, max_width=1200)
        diff_strip.save(str(OUTPUT_DIR / "summary_diff_overlays.png"))
        print(f"  Diff summary saved: summary_diff_overlays.png")

    del vae
    torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    print(f"\n=== Rendering complete in {elapsed:.1f}s ===")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

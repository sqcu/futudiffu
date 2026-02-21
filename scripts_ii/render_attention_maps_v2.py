r"""Render attention heatmaps from captured attention statistics.

Fixed version of render_attention_maps.py. Key differences:
- Handles mixed seq_len across layers (context/noise refiners have shorter
  seq_len than main DiT blocks). Filters to dominant seq_len.
- Saves to renders/ subdirectory inside the attention_maps dir.
- Uses VAE decode of latents to produce decoded base images.

Execution:
  PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe \
      F:\dox\repos\ai\futudiffu\scripts_ii\render_attention_maps_v2.py

Output:
  pinkify_thisnotthat_output/attention_maps/renders/*.png
"""

from __future__ import annotations

import json
import os
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

# --- Configuration ---
VAE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\vae\zimage.safetensors"
ATTN_DIR = REPO_ROOT / "pinkify_thisnotthat_output" / "attention_maps"
OUTPUT_DIR = ATTN_DIR / "renders"

LATENT_NAMES = ["high_pinkify", "low_pinkify", "high_thisnotthat", "low_thisnotthat"]
DIFF_NAMES = ["pinkify_diff", "thisnotthat_diff"]

PATCH_SIZE = 2
VAE_SCALE = 8
PIXELS_PER_TOKEN = PATCH_SIZE * VAE_SCALE  # 16


def get_dominant_seq_len(stats: dict) -> int:
    """Return the most common seq_len in the stats dict (excludes refiner blocks)."""
    from collections import Counter
    counts = Counter(v["seq_len"] for v in stats.values())
    return counts.most_common(1)[0][0]


def filter_stats_by_seq_len(stats: dict, seq_len: int) -> dict:
    """Return only layers whose seq_len matches the target."""
    return {k: v for k, v in stats.items() if v["seq_len"] == seq_len}


def main():
    t0 = time.perf_counter()
    device = torch.device("cuda")
    dtype = torch.bfloat16

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Output dir:", OUTPUT_DIR)

    # Check that attention stats exist
    for name in LATENT_NAMES:
        stats_path = ATTN_DIR / ("attention_stats_" + name + ".pt")
        if not stats_path.exists():
            print("ERROR: Missing " + str(stats_path))
            sys.exit(1)

    # --- Load VAE ---
    print("=== Loading VAE ===")
    vae = load_vae(VAE_PATH, device=device, dtype=dtype)

    # --- Decode latents and render attention overlays ---
    print("\n=== Rendering attention maps ===")

    decoded_images = {}
    summary_images = []

    for name in LATENT_NAMES:
        print("\n  --- " + name + " ---")

        latent_path = str(ATTN_DIR / ("latent_" + name + ".pt"))
        stats_path = str(ATTN_DIR / ("attention_stats_" + name + ".pt"))

        latent = torch.load(latent_path, weights_only=True)
        all_stats = torch.load(stats_path, weights_only=True)

        # The attention stats dict has mixed seq_lens: context/noise refiners
        # (layers 0-3) have seq_len 32-4160, main DiT blocks have the full seq_len.
        # Use only layers with the dominant seq_len.
        dom_seq_len = get_dominant_seq_len(all_stats)
        stats = filter_stats_by_seq_len(all_stats, dom_seq_len)
        n_layers_used = len(stats)
        n_layers_total = len(all_stats)
        print("  Layers total: " + str(n_layers_total) + " | used (seq_len=" + str(dom_seq_len) + "): " + str(n_layers_used))

        # Renumber to contiguous 0..N-1 for render_attention_map compatibility
        stats_reindexed = {new_i: v for new_i, (_, v) in enumerate(sorted(stats.items()))}

        decoded = decode_latent_to_pil(vae, latent, device=device, dtype=dtype)
        decoded.save(str(OUTPUT_DIR / ("decoded_" + name + ".png")))
        decoded_images[name] = decoded
        print("  Decoded: " + str(decoded.size))

        B, C, H_lat, W_lat = latent.shape
        H_tokens = H_lat // PATCH_SIZE
        W_tokens = W_lat // PATCH_SIZE
        n_img_tokens = H_tokens * W_tokens
        n_img_padded = n_img_tokens + ((-n_img_tokens) % 32)
        cap_len = dom_seq_len - n_img_padded

        print("  Spatial: " + str(H_tokens) + "x" + str(W_tokens) + " tokens = " + str(n_img_tokens) + " img tokens")
        print("  img_padded=" + str(n_img_padded) + ", caption tokens=" + str(cap_len))

        # Render attention overlay
        overlay = render_attention_map(
            decoded, stats_reindexed, n_img_tokens, H_tokens, W_tokens, cap_len,
            alpha=0.4, colormap="hot",
        )
        overlay.save(str(OUTPUT_DIR / ("overlay_" + name + ".png")))
        print("  Overlay saved: overlay_" + name + ".png")

        # Text token attention bar chart (aggregate over layers, mean over heads)
        first_stats = stats_reindexed[0]
        n_heads = first_stats["n_heads"]
        agg_received = torch.zeros(n_heads, dom_seq_len)
        for layer_idx in range(n_layers_used):
            agg_received += stats_reindexed[layer_idx]["attn_received"]
        agg_received /= n_layers_used

        effective_cap = max(0, cap_len)
        if effective_cap > 0:
            text_attention = agg_received[:, :effective_cap].mean(dim=0)
            text_chart = render_text_token_bar_chart(
                text_attention, width=600, height=200,
                title="Text Token Attention: " + name,
            )
            text_chart.save(str(OUTPUT_DIR / ("text_attn_" + name + ".png")))
            print("  Text attention chart saved")
        else:
            print("  No text tokens found (cap_len=" + str(cap_len) + "), skipping text chart")

        summary_images.append((name, overlay))

    # --- Render diff overlays ---
    print("\n=== Rendering attention diff maps ===")

    diff_configs = [
        ("pinkify_diff", "high_pinkify", "low_pinkify", "Pinkify: HIGH - LOW"),
        ("thisnotthat_diff", "high_thisnotthat", "low_thisnotthat", "TNT: HIGH - LOW"),
    ]

    diff_overlays = []

    for diff_name, name_a, name_b, title in diff_configs:
        diff_path = ATTN_DIR / ("attention_diff_" + diff_name + ".pt")
        if not diff_path.exists():
            print("  Skipping " + diff_name + " -- file not found")
            continue

        print("\n  --- " + diff_name + " ---")
        diff_stats = torch.load(str(diff_path), weights_only=True)
        n_layers = len(diff_stats)

        base_image = decoded_images[name_a]
        latent_a = torch.load(str(ATTN_DIR / ("latent_" + name_a + ".pt")), weights_only=True)
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

        print("  n_layers=" + str(n_layers) + ", min_seq=" + str(min_seq) + ", cap_len_a=" + str(cap_len_a))

        # Aggregate diff across layers
        agg_diff = torch.zeros(n_heads_diff, min_seq)
        for layer_idx in range(n_layers):
            agg_diff += diff_stats[layer_idx]["received_diff"]
        agg_diff /= n_layers

        if img_start_a + n_img_tokens <= min_seq:
            img_diff = agg_diff[:, img_start_a:img_start_a + n_img_tokens].mean(dim=0)
        else:
            available = min_seq - img_start_a
            if available > 0:
                img_diff_partial = agg_diff[:, img_start_a:img_start_a + available].mean(dim=0)
                img_diff = torch.cat([img_diff_partial, torch.zeros(n_img_tokens - available)])
            else:
                img_diff = torch.zeros(n_img_tokens)

        overlay_diff = render_heatmap_overlay(
            base_image, img_diff, H_tokens, W_tokens,
            alpha=0.5, colormap="diverging",
        )
        overlay_diff.save(str(OUTPUT_DIR / ("overlay_" + diff_name + ".png")))
        print("  Diff overlay saved: overlay_" + diff_name + ".png")
        diff_overlays.append((title, overlay_diff))

        # Text token diff
        if cap_len_a > 0 and cap_len_a <= min_seq:
            text_diff = agg_diff[:, :cap_len_a].mean(dim=0)
            text_chart = render_text_token_bar_chart(
                text_diff, width=600, height=200,
                title="Text Token Diff: " + title,
            )
            text_chart.save(str(OUTPUT_DIR / ("text_diff_" + diff_name + ".png")))
            print("  Text diff chart saved")

        # Layer x Head heatmap
        layer_head_img = render_layer_head_heatmap(
            diff_stats, n_layers, n_heads_diff,
            width=800, height=400,
            title="Layer x Head |Diff|: " + title,
        )
        layer_head_img.save(str(OUTPUT_DIR / ("layer_head_" + diff_name + ".png")))
        print("  Layer-head heatmap saved: layer_head_" + diff_name + ".png")

    # --- Summary strips ---
    print("\n=== Creating summary images ===")
    if summary_images:
        strip = render_strip(summary_images, max_width=1600)
        strip.save(str(OUTPUT_DIR / "summary_all_overlays.png"))
        print("  Summary strip saved: summary_all_overlays.png")

    if diff_overlays:
        diff_strip = render_strip(diff_overlays, max_width=1200)
        diff_strip.save(str(OUTPUT_DIR / "summary_diff_overlays.png"))
        print("  Diff summary saved: summary_diff_overlays.png")

    # --- Composite: decoded side-by-side with overlay ---
    print("\n=== Creating decoded + overlay composites ===")
    for name in LATENT_NAMES:
        base = decoded_images[name]
        overlay_path = OUTPUT_DIR / ("overlay_" + name + ".png")
        if overlay_path.exists():
            from PIL import Image
            overlay = Image.open(str(overlay_path))
            # Resize to same height
            h = base.height
            if overlay.height != h:
                ratio = h / overlay.height
                overlay = overlay.resize((int(overlay.width * ratio), h))
            composite = Image.new("RGB", (base.width + overlay.width, h))
            composite.paste(base, (0, 0))
            composite.paste(overlay, (base.width, 0))
            composite.save(str(OUTPUT_DIR / ("composite_" + name + ".png")))
            print("  Composite saved: composite_" + name + ".png")

    del vae
    torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    print("\n=== Rendering complete in " + str(round(elapsed, 1)) + "s ===")
    print("Output: " + str(OUTPUT_DIR))

    # List produced files
    produced = sorted(OUTPUT_DIR.glob("*.png"))
    print("\nProduced " + str(len(produced)) + " PNG files:")
    for p in produced:
        print("  " + p.name)


if __name__ == "__main__":
    main()

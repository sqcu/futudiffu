"""Visualization utilities: heatmaps, strips, bar charts.

Extracted from scripts_ii/render_attention_maps.py. Pure PIL rendering
functions with no model dependencies and no matplotlib.

Import constraints:
  - PIL for image rendering
  - torch for tensor operations
  - DOES NOT import: futudiffu, model_manager, server, client
"""

from __future__ import annotations

import torch
from PIL import Image, ImageDraw


# Z-Image defaults
PATCH_SIZE = 2
VAE_SCALE = 8
PIXELS_PER_TOKEN = PATCH_SIZE * VAE_SCALE  # 16


def render_heatmap_overlay(
    base_image: Image.Image,
    token_values: torch.Tensor,
    H_tokens: int,
    W_tokens: int,
    alpha: float = 0.5,
    colormap: str = "diverging",
    pixels_per_token: int = PIXELS_PER_TOKEN,
) -> Image.Image:
    """Overlay a spatial heatmap on a decoded image.

    Each token maps to a pixels_per_token x pixels_per_token pixel region.

    Args:
        base_image: PIL Image (decoded latent).
        token_values: (n_tokens,) -- one value per image token.
        H_tokens: Token grid height.
        W_tokens: Token grid width.
        alpha: Blend factor (0 = base only, 1 = heatmap only).
        colormap: "diverging" (blue-white-red) or "hot" (black-red-yellow-white).
        pixels_per_token: Pixel size of each token cell.

    Returns:
        Blended PIL Image.
    """
    W_img, H_img = base_image.size

    # Create heatmap image at token resolution
    heatmap = Image.new("RGB", (W_tokens, H_tokens))

    vals = token_values[:H_tokens * W_tokens].float()
    if colormap == "diverging":
        vmax = max(vals.abs().max().item(), 1e-8)
        normalized = vals / vmax  # [-1, 1]
        for idx in range(H_tokens * W_tokens):
            row = idx // W_tokens
            col = idx % W_tokens
            v = normalized[idx].item()
            if v > 0:
                r = 255
                g = int(255 * (1 - v))
                b = int(255 * (1 - v))
            else:
                r = int(255 * (1 + v))
                g = int(255 * (1 + v))
                b = 255
            heatmap.putpixel((col, row), (r, g, b))
    else:
        # Hot colormap (black -> red -> yellow -> white)
        vmin = vals.min().item()
        vmax_val = vals.max().item()
        vrange = max(vmax_val - vmin, 1e-8)
        for idx in range(H_tokens * W_tokens):
            row = idx // W_tokens
            col = idx % W_tokens
            v = (vals[idx].item() - vmin) / vrange  # [0, 1]
            if v < 0.33:
                r = int(255 * v / 0.33)
                g, b = 0, 0
            elif v < 0.66:
                r = 255
                g = int(255 * (v - 0.33) / 0.33)
                b = 0
            else:
                r, g = 255, 255
                b = int(255 * (v - 0.66) / 0.34)
            heatmap.putpixel((col, row), (r, g, b))

    # Upscale heatmap to image resolution
    heatmap_upscaled = heatmap.resize(
        (W_tokens * pixels_per_token, H_tokens * pixels_per_token),
        Image.NEAREST,
    )

    # Crop/pad to match base image size
    heatmap_final = Image.new("RGB", (W_img, H_img), (128, 128, 128))
    heatmap_final.paste(heatmap_upscaled, (0, 0))

    return Image.blend(base_image, heatmap_final, alpha)


def render_strip(
    images: list[tuple[str, Image.Image]],
    max_width: int = 1600,
) -> Image.Image:
    """Create a labeled horizontal strip of images.

    Args:
        images: List of (label, image) tuples.
        max_width: Maximum total width.

    Returns:
        PIL Image with all images side by side, labeled.
    """
    n = len(images)
    if n == 0:
        return Image.new("RGB", (100, 100), (255, 255, 255))

    target_w = max_width // n
    scaled = []
    for label, img in images:
        ratio = target_w / img.width
        new_h = int(img.height * ratio)
        scaled.append((label, img.resize((target_w, new_h), Image.LANCZOS)))

    max_h = max(s[1].height for s in scaled) + 25  # room for label
    strip = Image.new("RGB", (max_width, max_h), (255, 255, 255))
    draw = ImageDraw.Draw(strip)

    x = 0
    for label, img in scaled:
        strip.paste(img, (x, 25))
        draw.text((x + 5, 5), label, fill=(0, 0, 0))
        x += target_w

    return strip


def render_text_token_bar_chart(
    token_values: torch.Tensor,
    width: int = 600,
    height: int = 200,
    title: str = "Text Token Attention",
) -> Image.Image:
    """Render a bar chart of per-text-token attention values.

    Args:
        token_values: (n_text_tokens,) tensor of values.
        width: Image width.
        height: Image height.
        title: Chart title.

    Returns:
        PIL Image with bar chart.
    """
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    n_tokens = len(token_values)
    if n_tokens == 0:
        draw.text((10, 10), "No text tokens", fill=(0, 0, 0))
        return img

    vals = token_values.float()
    vmax = max(vals.abs().max().item(), 1e-8)

    draw.text((10, 5), title, fill=(0, 0, 0))

    bar_top = 25
    bar_bottom = height - 10
    bar_height = bar_bottom - bar_top
    mid_y = bar_top + bar_height // 2

    draw.line([(0, mid_y), (width, mid_y)], fill=(200, 200, 200), width=1)

    bar_width = max(1, (width - 20) // n_tokens)
    x_start = 10

    for i in range(min(n_tokens, (width - 20) // max(bar_width, 1))):
        v = vals[i].item() / vmax
        x = x_start + i * bar_width

        if v > 0:
            y_top = int(mid_y - v * (bar_height // 2))
            draw.rectangle([x, y_top, x + bar_width - 1, mid_y], fill=(220, 50, 50))
        else:
            y_bot = int(mid_y - v * (bar_height // 2))
            draw.rectangle([x, mid_y, x + bar_width - 1, y_bot], fill=(50, 50, 220))

    return img


def render_layer_head_heatmap(
    diff_stats: dict,
    n_layers: int,
    n_heads: int,
    width: int = 800,
    height: int = 400,
    title: str = "Layer x Head Attention Diff",
) -> Image.Image:
    """Render a heatmap of mean absolute attention diff per layer per head.

    Args:
        diff_stats: Dict mapping layer_idx -> {"received_diff": (n_heads, seq), ...}.
        n_layers: Number of layers.
        n_heads: Number of attention heads.
        width: Image width.
        height: Image height.
        title: Chart title.

    Returns:
        PIL Image with layer x head heatmap.
    """
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 5), title, fill=(0, 0, 0))

    matrix = torch.zeros(n_layers, n_heads)
    for layer_idx in range(n_layers):
        if layer_idx in diff_stats:
            recv_diff = diff_stats[layer_idx]["received_diff"]
            matrix[layer_idx] = recv_diff.abs().mean(dim=1)

    vmax = max(matrix.max().item(), 1e-8)

    cell_w = max(1, (width - 60) // n_heads)
    cell_h = max(1, (height - 40) // n_layers)
    x_off = 40
    y_off = 25

    for li in range(n_layers):
        for hi in range(n_heads):
            v = matrix[li, hi].item() / vmax
            intensity = int(255 * (1 - v))
            color = (255, intensity, intensity)
            x = x_off + hi * cell_w
            y = y_off + li * cell_h
            draw.rectangle(
                [x, y, x + cell_w - 1, y + cell_h - 1], fill=color
            )

    draw.text((5, y_off), "L0", fill=(0, 0, 0))
    draw.text((5, y_off + (n_layers - 1) * cell_h), f"L{n_layers-1}", fill=(0, 0, 0))
    draw.text((x_off, height - 15), f"Heads 0..{n_heads-1}", fill=(0, 0, 0))

    return img


def render_attention_map(
    decoded_image: Image.Image,
    attention_stats: dict[int, dict],
    n_img_tokens: int,
    H_tokens: int,
    W_tokens: int,
    cap_len: int,
    alpha: float = 0.4,
    colormap: str = "hot",
) -> Image.Image:
    """Render an attention heatmap overlay from captured attention stats.

    Aggregates attention received across all layers for image tokens,
    averages across heads, and overlays on the decoded image.

    Args:
        decoded_image: PIL Image (decoded latent).
        attention_stats: Dict from AttentionCapture.get_stats().
        n_img_tokens: Number of image tokens.
        H_tokens: Token grid height.
        W_tokens: Token grid width.
        cap_len: Caption token length (image tokens start at this index).
        alpha: Blend factor.
        colormap: Color scheme.

    Returns:
        PIL Image with attention overlay.
    """
    n_layers = len(attention_stats)
    first_stats = attention_stats[0]
    seq_len = first_stats["seq_len"]
    n_heads = first_stats["n_heads"]

    # Aggregate attention received across all layers
    agg_received = torch.zeros(n_heads, seq_len)
    for layer_idx in range(n_layers):
        agg_received += attention_stats[layer_idx]["attn_received"]
    agg_received /= n_layers

    # Extract image token attention (mean across heads)
    img_start = cap_len
    img_attention = agg_received[:, img_start:img_start + n_img_tokens].mean(dim=0)

    return render_heatmap_overlay(
        decoded_image, img_attention, H_tokens, W_tokens,
        alpha=alpha, colormap=colormap,
    )

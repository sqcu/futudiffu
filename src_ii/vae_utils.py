"""VAE utility functions: load + decode to PIL.

Provides a clean narrow interface over futudiffu.vae for the common
"load VAE, decode latent to PIL Image" pattern that was duplicated 3x
across generate_preference_labels.py, render_attention_maps.py, and
render_comparison.py.

Import constraints:
  - IMPORTS from futudiffu.vae: load_vae, vae_decode
  - PIL for image output
  - DOES NOT import: model_manager, server, client
"""

from __future__ import annotations

import torch
from PIL import Image


def load_vae(
    path: str,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
):
    """Load the AutoencoderKL VAE from safetensors.

    Args:
        path: Path to the VAE safetensors file.
        device: Target device.
        dtype: Working dtype.

    Returns:
        Loaded VAE model in eval mode.
    """
    from futudiffu.vae import load_vae as _load_vae
    vae = _load_vae(path, device=device, dtype=dtype)
    vae.eval()
    return vae


def decode_latent_to_pil(
    vae,
    latent: torch.Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Image.Image:
    """Decode a single latent to a PIL Image.

    Args:
        vae: Loaded AutoencoderKL (from load_vae()).
        latent: (1, 16, H, W) latent tensor.
        device: Device to move latent to (if None, uses latent's device).
        dtype: Dtype to cast latent to (if None, uses latent's dtype).

    Returns:
        PIL RGB Image of size (W*8, H*8).
    """
    from futudiffu.vae import vae_decode as _vae_decode

    if device is not None:
        latent = latent.to(device=device)
    if dtype is not None:
        latent = latent.to(dtype=dtype)

    # vae_decode returns (B, 3, H*8, W*8) in [0, 1] range
    pixels = _vae_decode(vae, latent)
    pixels = (pixels[0] * 255).byte()
    pixels = pixels.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    return Image.fromarray(pixels, "RGB")

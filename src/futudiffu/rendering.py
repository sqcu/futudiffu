"""Image rendering and I/O utilities for diffusion pipeline outputs.

Consolidated from inline implementations across 5+ scripts into
canonical functions. All image conversion, saving, and loading goes
through this module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def tensor_to_uint8(t) -> np.ndarray:
    """Convert a (1, 3, H, W) float tensor in [0,1] to (H, W, 3) uint8 numpy.

    Handles both torch.Tensor and numpy arrays. Clamps to [0, 1] before
    scaling to avoid overflow artifacts from VAE decode.
    """
    # Support both torch tensors and numpy
    if hasattr(t, "squeeze"):
        img = t.squeeze(0)
        if hasattr(img, "permute"):
            # torch tensor
            img = img.permute(1, 2, 0).clamp(0, 1).float().cpu().numpy()
        else:
            # numpy with batch dim already squeezed
            img = np.transpose(img, (1, 2, 0))
            img = np.clip(img, 0, 1).astype(np.float32)
    else:
        img = np.clip(t, 0, 1).astype(np.float32)

    return (img * 255).clip(0, 255).astype(np.uint8)


def save_image(arr: np.ndarray, path: Path | str, mkdir: bool = True) -> None:
    """Save (H, W, 3) uint8 array as PNG via PIL."""
    path = Path(path)
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(str(path))


def decode_and_save(client, latent, path: Path | str) -> np.ndarray:
    """VAE-decode latent -> save PNG -> return uint8 array.

    Args:
        client: InferenceClient with vae_decode() method.
        latent: (1, C, H, W) latent tensor (bfloat16).
        path: Output PNG path.

    Returns:
        (H, W, 3) uint8 numpy array of the decoded image.
    """
    pixels = client.vae_decode(latent)
    arr = tensor_to_uint8(pixels)
    save_image(arr, path)
    return arr


def load_image_array(path: Path | str, normalize: bool = False) -> np.ndarray:
    """Load PNG as (H, W, 3) array.

    Args:
        path: Path to image file.
        normalize: If True, returns [0,1] float32. Otherwise uint8.

    Returns:
        (H, W, 3) numpy array.
    """
    img = Image.open(str(path)).convert("RGB")
    arr = np.array(img)
    if normalize:
        return arr.astype(np.float32) / 255.0
    return arr

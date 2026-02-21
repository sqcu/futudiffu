"""Pixel-space reward functions for PINKIFY and THISNOTTHAT.

PINKIFY: local pinkness with contrast scoring.
  - CPU path: PIL/numpy, ~150ms/image.
  - GPU path: Pure torch tensor ops, <1ms/image.
  - Auto-dispatch via `pinkify_score()`.

THISNOTTHAT: FrFT high-D descriptor comparison.
  - GPU only: `thisnotthat_score_gpu()`.
  - 0.7ms/image eager, 0.3ms compiled (any resolution up to 2MP).
  - Resolution/aspect-ratio quasi-invariant, rotation-soft, color-sensitive.

Import constraints:
  - PIL (Pillow) for image manipulation
  - numpy for array operations
  - torch is optional (GPU path only, graceful fallback)
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Optional torch import -- GPU path disabled if unavailable
try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_TORCH = False
    _HAS_CUDA = False


# ---------------------------------------------------------------------------
# PINKIFY: local pinkness with contrast scoring
# ---------------------------------------------------------------------------

def _rgb_to_hsv_array(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 array to HSV float array.

    Args:
        rgb: (H, W, 3) uint8 array.

    Returns:
        (H, W, 3) float array with H in [0, 360), S in [0, 1], V in [0, 1].
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Hue
    h = np.zeros_like(delta)
    mask_r = (cmax == r) & (delta > 0)
    mask_g = (cmax == g) & (delta > 0)
    mask_b = (cmax == b) & (delta > 0)

    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

    # Saturation (avoid divide-by-zero where cmax is 0)
    s = np.where(cmax > 0, delta / np.maximum(cmax, 1e-10), 0.0)

    # Value
    v = cmax

    return np.stack([h, s, v], axis=-1)


def _is_pink_mask(hsv: np.ndarray, sat_thresh: float = 0.15, val_thresh: float = 0.3) -> np.ndarray:
    """Create a binary mask of pink pixels (DEPRECATED -- use _continuous_pinkness).

    Retained for backward compatibility with analysis code that imports it.

    Pink is defined as:
      - Hue in the pink/magenta range: [280, 360) or [0, 30) (wraps around red)
      - Saturation above threshold (not gray/white)
      - Value above threshold (not too dark)

    Args:
        hsv: (H, W, 3) HSV array.
        sat_thresh: Minimum saturation for pink.
        val_thresh: Minimum value for pink.

    Returns:
        (H, W) boolean mask.
    """
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Pink/magenta hue range: wraps around 0/360
    # 300-360 (magenta/pink) and 0-30 (reddish-pink)
    hue_pink = ((h >= 300) & (h <= 360)) | ((h >= 0) & (h < 30))

    # Also include rose/hot pink in 330-360 range with lower saturation requirement
    # and fuchsia in 280-320 range
    hue_extended = (h >= 280) & (h < 340)

    hue_match = hue_pink | hue_extended
    sat_match = s >= sat_thresh
    val_match = v >= val_thresh

    return hue_match & sat_match & val_match


def _hue_pink_weight(h: np.ndarray) -> np.ndarray:
    """Compute continuous hue membership in the pink range.

    Core pink hue range: [300, 360) or equivalently wrapping to [0, 30).
    Smooth falloff extends +-20 degrees beyond boundaries:
      - Below 280: weight 0 (not pink at all)
      - 280-300: linear ramp from 0 to 1 (fuchsia/violet transition)
      - 300-360 or 0-30: weight 1 (core pink/magenta/rose)
      - 30-50: linear ramp from 1 to 0 (orange-red transition)
      - Above 50: weight 0 (not pink)

    Args:
        h: (H, W) hue array in [0, 360).

    Returns:
        (H, W) float array in [0, 1].
    """
    falloff = 20.0  # degrees of smooth transition outside core range

    # Distance to core range
    inside_core = (h >= 300.0) | (h <= 30.0)

    # Distance for hues above 30 (going away toward warm colors)
    dist_above = h - 30.0   # positive when h > 30
    # Distance for hues below 300 (going away toward cool colors)
    dist_below = 300.0 - h  # positive when h < 300

    # Pick the correct distance based on which side of the range we're on
    # For h in (30, 180]: dist_above is correct
    # For h in (180, 300): dist_below is correct
    distance = np.where(inside_core, 0.0,
                        np.where(h <= 180.0, dist_above, dist_below))

    weight = np.clip(1.0 - distance / falloff, 0.0, 1.0)
    return weight.astype(np.float32)


def _continuous_pinkness(hsv: np.ndarray, val_floor: float = 0.1) -> np.ndarray:
    """Compute continuous pinkness score per pixel.

    Replaces the binary _is_pink_mask with a smooth [0, 1] score that captures:
      - Hue proximity to the pink range (smooth falloff at boundaries)
      - Saturation intensity (linear contribution, no hard cutoff)
      - Value floor (very dark pixels get zero regardless of hue/saturation)

    The pinkness of a pixel is:
        pinkness = hue_weight(h) * saturation * value_gate(v)

    where:
      - hue_weight is 1.0 inside [300,360]+[0,30], smooth falloff 20 deg outside
      - saturation contributes linearly (S=0.40 is 8x pinker than S=0.05)
      - value_gate is 0 for V < val_floor, linear ramp from val_floor to 2*val_floor, 1 above

    Args:
        hsv: (H, W, 3) HSV array. H in [0,360), S in [0,1], V in [0,1].
        val_floor: Value below which pixels are not pink at all.

    Returns:
        (H, W) float32 array in [0, 1].
    """
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Hue membership: smooth weight based on proximity to pink range
    hue_w = _hue_pink_weight(h)

    # Value gate: hard floor at val_floor, linear ramp to 2*val_floor
    val_ramp_top = val_floor * 2.0
    val_gate = np.clip((v - val_floor) / (val_ramp_top - val_floor + 1e-10), 0.0, 1.0)

    # Pinkness = hue_weight * saturation * value_gate
    # Saturation contributes linearly: S=0.5 is 10x S=0.05
    pinkness = hue_w * s * val_gate

    return pinkness.astype(np.float32)


def _coverage_contrast(presence: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """Compute local contrast for a binary pink-presence mask.

    For each present pixel, the contrast score is the fraction of non-present
    pixels in its local neighborhood. This measures how much a pink pixel
    stands out against its local context, regardless of pinkness intensity.

    Uses reflect-padding at image boundaries to eliminate the edge artifact
    that inflated scores for uniformly-colored images with mode='constant'.

    Args:
        presence: (H, W) float32 binary mask (1.0 where pink, 0.0 elsewhere).
        kernel_size: Size of the local neighborhood.

    Returns:
        (H, W) float array of coverage contrast scores.
    """
    from scipy.ndimage import uniform_filter

    # Local fraction of pink-present pixels (reflect at edges)
    local_fraction = uniform_filter(presence, size=kernel_size, mode='reflect')

    # Contrast = presence * fraction of NON-pink in neighborhood
    contrast = presence * (1.0 - local_fraction)

    return contrast


def _coverage_contrast_noscipy(presence: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """Compute coverage contrast without scipy (pure numpy fallback).

    Uses integral image (summed-area table) for O(1) per-pixel box filter.
    Reflect-pads at boundaries to avoid edge artifacts.
    """
    h, w = presence.shape
    pad = kernel_size // 2

    # Reflect-pad to match scipy mode='reflect'
    padded = np.pad(presence, pad, mode='reflect')

    # Build integral image with a leading zero row and column
    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.float64)
    integral[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)

    r1 = np.arange(h)
    r2 = r1 + kernel_size
    c1 = np.arange(w)
    c2 = c1 + kernel_size

    local_sum = (integral[np.ix_(r2, c2)]
                 - integral[np.ix_(r1, c2)]
                 - integral[np.ix_(r2, c1)]
                 + integral[np.ix_(r1, c1)])

    area = kernel_size * kernel_size
    local_fraction = (local_sum / area).astype(np.float32)

    # Contrast: presence * fraction of non-pink in neighborhood
    contrast = presence * (1.0 - local_fraction)

    return contrast


def pinkify_score_cpu(image: Image.Image) -> float:
    """Score an image's pinkness using continuous pinkness with local contrast (CPU path).

    Uses a continuous pinkness measure per pixel (not a binary mask) so that:
      - More saturated pink scores higher than faintly saturated pink
      - Low-saturation pink washes contribute proportionally (no hard cutoff)
      - Local contrast counts pink pixels that stand out against surroundings
      - Uniform pink contributes via a base term (not zero)

    The scoring pipeline:

      1. Compute continuous pinkness per pixel (hue_weight * saturation * value_gate).
      2. Apply sqrt compression (power=0.5) to the pinkness values. This gives
         more weight to spatial coverage over per-pixel intensity: a pixel with
         pinkness 0.18 contributes 69% as much as one with pinkness 0.37 (instead
         of 47% linearly). This rewards images with more pink features.
      3. Compute **coverage contrast**: a binary presence mask (pinkness > threshold)
         is convolved with a box filter. Each present pixel's contribution is
         proportional to the fraction of non-pink pixels in its 7x7 neighborhood.
         This term rewards pink that stands out against non-pink surroundings.
      4. Compute **intensity term**: mean of the sqrt-compressed pinkness values,
         weighted at 0.2. This provides a floor so that uniformly pink images
         or pink-washed backgrounds score above zero.

    The final score is: coverage_contrast/area + mean_compressed_pinkness * 0.2

    Higher score = more/pinker pink with good contrast. A uniformly pink image
    scores lower than one with vivid pink against non-pink, but higher than
    an image with no pink at all.

    Args:
        image: PIL Image (any mode, will be converted to RGB).

    Returns:
        Scalar pinkness score (normalized by image area).
    """
    img = image.convert("RGB")
    rgb = np.array(img)
    hsv = _rgb_to_hsv_array(rgb)

    # Continuous pinkness per pixel (replaces binary mask)
    pinkness = _continuous_pinkness(hsv)

    # Sqrt compression: rewards coverage over per-pixel intensity
    compressed = np.sqrt(np.maximum(pinkness, 0.0))

    # Binary presence mask for coverage contrast
    presence_thresh = 0.01
    presence = (pinkness > presence_thresh).astype(np.float32)

    # Coverage contrast: counts pink pixels weighted by local non-pink fraction
    try:
        coverage = _coverage_contrast(presence)
    except ImportError:
        coverage = _coverage_contrast_noscipy(presence)

    h, w = pinkness.shape
    area = h * w

    # Coverage contrast term: rewards pink regions that stand out
    contrast_score = float(coverage.sum() / area)

    # Intensity term: mean compressed pinkness provides a floor for
    # uniform pink and pink washes. Weighted at 0.2 so contrast still
    # dominates for images with pink accents against non-pink backgrounds.
    intensity_weight = 0.2
    mean_compressed = float(compressed.sum() / area)
    intensity_score = mean_compressed * intensity_weight

    score = contrast_score + intensity_score
    return score


def pinkify_score(image: Image.Image) -> float:
    """Score an image's pinkness. Auto-dispatches to GPU when available.

    Transparent wrapper: uses the GPU path when torch+CUDA are available,
    otherwise falls back to the CPU path. The GPU path converts the PIL
    image to a CUDA tensor internally.

    Args:
        image: PIL Image (any mode, will be converted to RGB).

    Returns:
        Scalar pinkness score (normalized by image area).
    """
    if _can_use_gpu():
        t = _pil_to_gpu_tensor(image)  # (1, 3, H, W) on CUDA
        with torch.no_grad():
            score = pinkify_score_gpu(t.squeeze(0))  # pass (3, H, W)
        return float(score.item())
    return pinkify_score_cpu(image)




# ---------------------------------------------------------------------------
# GPU PINKIFY: pure torch tensor ops
# ---------------------------------------------------------------------------

def _rgb_to_hsv_torch(rgb: 'torch.Tensor') -> 'torch.Tensor':
    """Convert RGB float tensor to HSV float tensor.

    Args:
        rgb: (..., 3, H, W) float tensor in [0, 1].

    Returns:
        (..., 3, H, W) float tensor with H in [0, 360), S in [0, 1], V in [0, 1].
    """
    r, g, b = rgb[..., 0:1, :, :], rgb[..., 1:2, :, :], rgb[..., 2:3, :, :]

    cmax = torch.max(torch.max(r, g), b)  # (..., 1, H, W)
    cmin = torch.min(torch.min(r, g), b)
    delta = cmax - cmin

    # Hue computation: conditional on which channel is max
    # Default hue = 0 (when delta == 0)
    h = torch.zeros_like(delta)

    # Masks are NOT mutually exclusive -- matching numpy behavior where
    # later assignments overwrite earlier ones. Order: r, g, b (b wins ties).
    mask_r = (cmax == r) & (delta > 0)
    mask_g = (cmax == g) & (delta > 0)
    mask_b = (cmax == b) & (delta > 0)

    # Safe division (delta is > 0 where masks are True)
    safe_delta = torch.where(delta > 0, delta, torch.ones_like(delta))

    h_r = 60.0 * (((g - b) / safe_delta) % 6.0)
    h_g = 60.0 * (((b - r) / safe_delta) + 2.0)
    h_b = 60.0 * (((r - g) / safe_delta) + 4.0)

    # Apply in same order as numpy: r first, g second (overwrites r), b last (overwrites g)
    h = torch.where(mask_r, h_r, h)
    h = torch.where(mask_g, h_g, h)
    h = torch.where(mask_b, h_b, h)

    # Saturation: delta / cmax (0 where cmax == 0)
    s = torch.where(cmax > 0, delta / torch.clamp(cmax, min=1e-10), torch.zeros_like(delta))

    # Value
    v = cmax

    return torch.cat([h, s, v], dim=-3)  # (..., 3, H, W)


def _hue_pink_weight_torch(h: 'torch.Tensor') -> 'torch.Tensor':
    """Compute continuous hue membership in the pink range (torch version).

    Same semantics as _hue_pink_weight but operates on torch tensors.

    Args:
        h: (..., H, W) hue tensor in [0, 360).

    Returns:
        (..., H, W) float tensor in [0, 1].
    """
    falloff = 20.0

    inside_core = (h >= 300.0) | (h <= 30.0)

    dist_above = h - 30.0   # positive when h > 30
    dist_below = 300.0 - h  # positive when h < 300

    distance = torch.where(inside_core, torch.zeros_like(h),
                           torch.where(h <= 180.0, dist_above, dist_below))

    weight = torch.clamp(1.0 - distance / falloff, 0.0, 1.0)
    return weight


def _continuous_pinkness_torch(hsv: 'torch.Tensor', val_floor: float = 0.1) -> 'torch.Tensor':
    """Compute continuous pinkness score per pixel (torch version).

    Args:
        hsv: (..., 3, H, W) HSV tensor. H in [0,360), S in [0,1], V in [0,1].
        val_floor: Value below which pixels are not pink at all.

    Returns:
        (..., H, W) float32 tensor in [0, 1].
    """
    h = hsv[..., 0, :, :]  # (..., H, W)
    s = hsv[..., 1, :, :]
    v = hsv[..., 2, :, :]

    hue_w = _hue_pink_weight_torch(h)

    val_ramp_top = val_floor * 2.0
    val_gate = torch.clamp((v - val_floor) / (val_ramp_top - val_floor + 1e-10), 0.0, 1.0)

    pinkness = hue_w * s * val_gate
    return pinkness


def _coverage_contrast_torch(presence: 'torch.Tensor', kernel_size: int = 7) -> 'torch.Tensor':
    """Compute coverage contrast using avg_pool2d (torch version).

    Replaces scipy.ndimage.uniform_filter. Uses reflect padding + avg_pool2d
    to compute the local fraction of pink-present pixels.

    Args:
        presence: (B, 1, H, W) float tensor (binary: 1.0 or 0.0).
        kernel_size: Size of the local neighborhood.

    Returns:
        (B, 1, H, W) coverage contrast tensor.
    """
    pad = kernel_size // 2

    # Reflect-pad to match the CPU implementation's mode='reflect'
    padded = F.pad(presence, (pad, pad, pad, pad), mode='reflect')

    # avg_pool2d with kernel_size and stride=1 computes the local mean
    # which is equivalent to uniform_filter / box filter
    local_fraction = F.avg_pool2d(padded, kernel_size=kernel_size, stride=1, padding=0)

    # Contrast = presence * (1 - local_fraction)
    contrast = presence * (1.0 - local_fraction)
    return contrast


def pinkify_score_gpu(
    image_tensor: 'torch.Tensor',
    device: 'torch.device | None' = None,
) -> 'torch.Tensor':
    """Score image pinkness using pure torch GPU ops.

    Semantically identical to pinkify_score() but runs as tensor operations.

    Args:
        image_tensor: [3, H, W] or [B, 3, H, W] float tensor in [0, 1] range.
        device: Target device. If None, uses the tensor's device.

    Returns:
        Scalar tensor (if unbatched) or [B] tensor of pinkness scores.
    """
    unbatched = image_tensor.ndim == 3
    if unbatched:
        image_tensor = image_tensor.unsqueeze(0)  # (1, 3, H, W)

    if device is not None:
        image_tensor = image_tensor.to(device)

    # Ensure float32 for numerical consistency with CPU path
    image_tensor = image_tensor.float()

    B, C, H, W = image_tensor.shape
    area = H * W

    # Step 1: RGB -> HSV
    hsv = _rgb_to_hsv_torch(image_tensor)  # (B, 3, H, W)

    # Step 2: Continuous pinkness
    pinkness = _continuous_pinkness_torch(hsv)  # (B, H, W)

    # Step 3: Sqrt compression
    compressed = torch.sqrt(torch.clamp(pinkness, min=0.0))  # (B, H, W)

    # Step 4: Binary presence mask for coverage contrast
    presence_thresh = 0.01
    presence = (pinkness > presence_thresh).float().unsqueeze(1)  # (B, 1, H, W)

    # Step 5: Coverage contrast via avg_pool2d
    coverage = _coverage_contrast_torch(presence, kernel_size=7)  # (B, 1, H, W)

    # Step 6: Compute scores
    contrast_score = coverage.squeeze(1).sum(dim=(-2, -1)) / area  # (B,)
    mean_compressed = compressed.sum(dim=(-2, -1)) / area  # (B,)
    intensity_weight = 0.2
    intensity_score = mean_compressed * intensity_weight  # (B,)

    scores = contrast_score + intensity_score  # (B,)

    if unbatched:
        return scores.squeeze(0)  # scalar tensor
    return scores


# ---------------------------------------------------------------------------
# GPU THISNOTTHAT: FrFT high-D descriptor comparison
# ---------------------------------------------------------------------------

def thisnotthat_score_gpu(
    image_tensor: 'torch.Tensor',
    this_ref: 'torch.Tensor',
    that_ref: 'torch.Tensor',
    n_angles: int = 16,
    n_eval: int = 16,
    verbose: bool = False,
) -> 'torch.Tensor':
    """Score how much an image resembles THIS vs THAT using FrFT descriptors.

    Lifts each image into a high-dimensional space via the 2D isotropic
    fractional Fourier transform at multiple angles, then projects onto the
    Fisher discriminant axis between THIS and THAT references.

    Uses centered cosine scoring: midpoint = (desc_THIS + desc_THAT) / 2,
    discriminant = desc_THIS - desc_THAT. Score = cos_sim(desc_img - midpoint,
    discriminant). THIS -> +1.0, THAT -> -1.0 by construction.

    Resolution/aspect ratio invariant: coordinates are normalized to [-1, 1],
    evaluation points are fixed regardless of input resolution.

    Args:
        image_tensor: [3, H, W] or [B, 3, H, W] float tensor in [0, 1].
        this_ref: [1, 3, H_r, W_r] THIS reference tensor.
        that_ref: [1, 3, H_r, W_r] THAT reference tensor.
        n_angles: Number of FrFT angles (more = richer descriptor).
        n_eval: Evaluation grid size per dimension (n_eval^2 points).
        verbose: Print timing information.

    Returns:
        Scalar tensor (if unbatched) or [B] tensor of scores in [-1, +1].
    """
    from src_ii.frft import frft_descriptor, frft_discriminant_score
    import time

    unbatched = image_tensor.ndim == 3
    if unbatched:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.float()
    B = image_tensor.shape[0]

    t0 = time.perf_counter() if verbose else 0

    # Compute reference descriptors (at their native resolution — no resize!)
    # The FrFT descriptor is resolution-invariant by construction.
    desc_this = frft_descriptor(
        this_ref.squeeze(0).float().to(image_tensor.device),
        n_angles=n_angles, n_eval=n_eval,
    )
    desc_that = frft_descriptor(
        that_ref.squeeze(0).float().to(image_tensor.device),
        n_angles=n_angles, n_eval=n_eval,
    )

    if verbose:
        t1 = time.perf_counter()
        print(f"  ref descriptors: {(t1-t0)*1000:.1f}ms")

    scores = []
    for b in range(B):
        img = image_tensor[b]  # (3, H, W)
        desc_img = frft_descriptor(
            img, n_angles=n_angles, n_eval=n_eval,
        )

        scores.append(frft_discriminant_score(desc_img, desc_this, desc_that))

    result = torch.stack(scores)

    if verbose:
        t2 = time.perf_counter()
        print(f"  total: {(t2-t0)*1000:.1f}ms ({(t2-t0)*1000/B:.1f}ms/image)")

    if unbatched:
        return result.squeeze(0)
    return result


# ---------------------------------------------------------------------------
# PIL convenience wrappers with GPU auto-dispatch
# ---------------------------------------------------------------------------

def _pil_to_gpu_tensor(image: Image.Image) -> 'torch.Tensor':
    """Convert PIL Image to (1, 3, H, W) CUDA float32 tensor in [0, 1]."""
    rgb = image.convert("RGB")
    arr = np.array(rgb, dtype=np.float32) / 255.0  # (H, W, 3)
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return t.cuda()


def _pil_to_tensor(image: Image.Image, device: 'torch.device') -> 'torch.Tensor':
    """Convert PIL Image to (1, 3, H, W) float32 tensor in [0, 1]."""
    rgb = image.convert("RGB")
    arr = np.array(rgb, dtype=np.float32) / 255.0  # (H, W, 3)
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return t.to(device)


def _can_use_gpu() -> bool:
    """Check if GPU scoring path is available."""
    return _HAS_TORCH and _HAS_CUDA


# ---------------------------------------------------------------------------
# Pairwise preference from scores
# ---------------------------------------------------------------------------

def pairwise_preference(score_a: float, score_b: float, margin: float = 0.0) -> int:
    """Determine pairwise preference from scores.

    Args:
        score_a: Score for image A.
        score_b: Score for image B.
        margin: Minimum margin for a preference (0 = any difference counts).

    Returns:
        +1 if A wins, -1 if B wins, 0 if tie (within margin).
    """
    diff = score_a - score_b
    if diff > margin:
        return 1
    elif diff < -margin:
        return -1
    else:
        return 0

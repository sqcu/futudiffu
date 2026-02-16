"""Image-domain statistics for validating diffusion outputs.

Pure numpy + PIL, no torch dependency. Reusable for any image quality
assessment: PSNR, entropy, spectral slope, sharpness, histograms.
"""

from __future__ import annotations

import numpy as np


def psnr(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    """Peak signal-to-noise ratio between two images.

    Args:
        a, b: Images as numpy arrays (same shape). uint8 or float.
        max_val: Maximum pixel value (255 for uint8, 1.0 for normalized).

    Returns:
        PSNR in dB. Higher = more similar. Returns float('inf') if identical.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(max_val ** 2 / mse))


def shannon_entropy_per_channel(img: np.ndarray) -> list[float]:
    """Per-channel Shannon entropy (bits).

    Natural images: 5-7 bits. Pure noise: ~7.9 bits (for uint8).

    Args:
        img: (H, W, 3) uint8 array.

    Returns:
        List of 3 entropy values, one per channel.
    """
    entropies = []
    for c in range(img.shape[2]):
        channel = img[:, :, c].ravel()
        counts = np.bincount(channel, minlength=256)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropies.append(float(-np.sum(probs * np.log2(probs))))
    return entropies


def spectral_slope(img: np.ndarray) -> float:
    """1/f spectral slope via 2D FFT radial average + log-log linear fit.

    Natural images: ~-2. Pure noise: ~0. Flag slope > -0.5 as unnatural.

    Args:
        img: (H, W, 3) uint8 or float array.

    Returns:
        Slope of log(power) vs log(frequency). More negative = more natural.
    """
    # Convert to grayscale float
    if img.ndim == 3:
        gray = np.mean(img.astype(np.float64), axis=2)
    else:
        gray = img.astype(np.float64)

    H, W = gray.shape
    fft2 = np.fft.fft2(gray)
    power = np.abs(np.fft.fftshift(fft2)) ** 2

    cy, cx = H // 2, W // 2
    max_r = min(cy, cx)

    # Radial average
    y_grid, x_grid = np.ogrid[:H, :W]
    r_grid = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2).astype(int)

    radial = np.zeros(max_r)
    for r in range(max_r):
        mask = r_grid == r
        if mask.any():
            radial[r] = power[mask].mean()

    # Log-log fit, skip DC (r=0) and very low frequencies
    valid = radial[1:] > 0
    if valid.sum() < 2:
        return 0.0

    freqs = np.arange(1, max_r)
    log_f = np.log10(freqs[valid])
    log_p = np.log10(radial[1:][valid])

    # Linear fit: log_p = slope * log_f + intercept
    coeffs = np.polyfit(log_f, log_p, 1)
    return float(coeffs[0])


def laplacian_variance(img: np.ndarray) -> float:
    """Sharpness measure via Laplacian. Higher = sharper edges.

    Uses a simple 3x3 Laplacian kernel. No OpenCV dependency.

    Args:
        img: (H, W, 3) uint8 array.

    Returns:
        Variance of Laplacian response (grayscale).
    """
    if img.ndim == 3:
        gray = np.mean(img.astype(np.float64), axis=2)
    else:
        gray = img.astype(np.float64)

    # Laplacian via convolution: [[0,1,0],[1,-4,1],[0,1,0]]
    # Pad to handle edges
    padded = np.pad(gray, 1, mode="edge")
    lap = (
        padded[:-2, 1:-1] + padded[2:, 1:-1] +
        padded[1:-1, :-2] + padded[1:-1, 2:] -
        4 * padded[1:-1, 1:-1]
    )
    return float(np.var(lap))


def color_histogram(img: np.ndarray, bins: int = 32) -> np.ndarray:
    """96-dim (3*bins) normalized histogram feature vector.

    Args:
        img: (H, W, 3) uint8 array.
        bins: Number of bins per channel.

    Returns:
        Normalized histogram as (3*bins,) float64 array.
    """
    hists = []
    for c in range(3):
        h, _ = np.histogram(img[:, :, c], bins=bins, range=(0, 256))
        hists.append(h.astype(np.float64))

    feature = np.concatenate(hists)
    total = feature.sum()
    if total > 0:
        feature /= total
    return feature


def histogram_cosine_sim(h1: np.ndarray, h2: np.ndarray) -> float:
    """Cosine similarity between two histogram feature vectors.

    Returns:
        Cosine similarity in [-1, 1]. Higher = more similar.
    """
    dot = np.dot(h1, h2)
    norm1 = np.linalg.norm(h1)
    norm2 = np.linalg.norm(h2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def naturalness_report(img: np.ndarray) -> dict:
    """All-in-one image quality report.

    Combines entropy, spectral slope, Laplacian sharpness, and basic
    channel statistics into a single dict.

    Args:
        img: (H, W, 3) uint8 array.

    Returns:
        Dict with keys: entropy, spectral_slope, laplacian_variance,
        channel_means, channel_stds.
    """
    entropy = shannon_entropy_per_channel(img)
    slope = spectral_slope(img)
    lap_var = laplacian_variance(img)

    channel_means = [float(img[:, :, c].mean()) for c in range(3)]
    channel_stds = [float(img[:, :, c].astype(np.float64).std()) for c in range(3)]

    return {
        "entropy": entropy,
        "mean_entropy": float(np.mean(entropy)),
        "spectral_slope": slope,
        "laplacian_variance": lap_var,
        "channel_means": channel_means,
        "channel_stds": channel_stds,
    }

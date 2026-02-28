"""Sigma schedule construction.

Pure math -- ported from ComfyUI's model_sampling.py and samplers.py.
These are the CONST noise model sigma tables and scheduling functions.

Import constraints:
  - IMPORTS nothing from futudiffu
  - Pure torch math only
"""

import math

import torch


MAX_SHIFT: float = 8.0
"""Default upper bound for resolution_shift alpha.

With 30 Euler steps, alpha > 8 produces schedules where > 50% of steps
operate above sigma=0.9, wasting step budget in the high-noise regime where
the model's denoised predictions are near-random.  Alpha=13+ (e.g. 96x64 at
ref 1280x832) makes 18/30 steps > 0.9 and the last step jumps dt=-0.32,
producing featureless flat-color outputs.

Capped at 8.0: the worst-case schedule has ~47% of steps above sigma=0.9
and a max dt of ~0.22, keeping Euler integration well-conditioned.

SD3 used a fixed alpha=3.0 for 1024x1024 generation.  ComfyUI's
ModelSamplingAuraFlow accepts shift up to 100 but practical values are 1-6.
Community Z-Image experiments recommend shift=3-7 for non-native resolutions.
"""


def resolution_shift(
    width: int,
    height: int,
    ref_width: int = 1280,
    ref_height: int = 832,
    max_shift: float = MAX_SHIFT,
) -> float:
    """SD3 Eq.23: alpha = sqrt(ref_pixels / target_pixels), clamped.

    For the reference resolution (1280x832), returns 1.0 (identity/no shift).
    For smaller resolutions, returns alpha > 1.0 (shift toward higher noise),
    capped at *max_shift* to prevent degenerate Euler schedules.

    The cap prevents extreme alpha values (e.g. 13+ for tiny images) that
    compress the sigma schedule so severely that most Euler steps operate
    above sigma=0.9 where model predictions are near-random.

    Args:
        width: Target image width in pixels.
        height: Target image height in pixels.
        ref_width: Reference width (default 1280).
        ref_height: Reference height (default 832).
        max_shift: Maximum alpha value. Default MAX_SHIFT (8.0).

    Returns:
        Clamped alpha value in [1.0, max_shift].
    """
    ref_pixels = ref_width * ref_height
    target_pixels = width * height
    if target_pixels <= 0:
        raise ValueError(f"Invalid resolution: {width}x{height}")
    alpha = math.sqrt(ref_pixels / target_pixels)
    return min(alpha, max_shift)


def time_snr_shift(alpha: float, t: torch.Tensor) -> torch.Tensor:
    """SNR shift function from ComfyUI model_sampling."""
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


def build_sigmas(shift: float = 1.0, multiplier: float = 1000.0, timesteps: int = 1000) -> torch.Tensor:
    """Build the sigma table for ModelSamplingDiscreteFlow."""
    ts = torch.arange(1, timesteps + 1, 1, dtype=torch.float32) / timesteps
    ts = ts * multiplier
    sigmas = time_snr_shift(shift, ts / multiplier)
    return sigmas


def simple_scheduler(sigmas: torch.Tensor, steps: int) -> torch.Tensor:
    """Port of comfy/samplers.py:simple_scheduler.

    Args:
        sigmas: 1D tensor of 1000 sigma values from build_sigmas.
        steps: Number of sampling steps.

    Returns:
        Tensor of (steps+1,) sigma values, ending with 0.0.
    """
    sigs = []
    ss = len(sigmas) / steps
    for x in range(steps):
        sigs.append(float(sigmas[-(1 + int(x * ss))]))
    sigs.append(0.0)
    return torch.FloatTensor(sigs)


def build_sigma_schedule(
    n_steps: int,
    sampling_shift: float = 1.0,
    multiplier: float = 1.0,
    denoise: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Build the full sigma schedule for a trajectory.

    Handles both t2i (denoise=1.0) and i2i (denoise<1.0, expanded schedule).

    Args:
        n_steps: Number of euler steps.
        sampling_shift: Shift for build_sigmas.
        multiplier: Multiplier for build_sigmas (passed as multiplier * 1000).
        denoise: Denoise strength (0-1). 1.0 = full t2i.
        device: Target device.
        dtype: Target dtype.

    Returns:
        sigmas: (n_steps+1,) tensor on device/dtype.
    """
    sigma_table = build_sigmas(shift=sampling_shift, multiplier=multiplier * 1000)

    if denoise < 1.0:
        expanded_steps = int(n_steps / denoise)
        full_sigmas = simple_scheduler(sigma_table, expanded_steps)
        full_sigmas = full_sigmas.to(device=device, dtype=dtype)
        sigmas = full_sigmas[-(n_steps + 1):]
    else:
        sigmas = simple_scheduler(sigma_table, n_steps)
        sigmas = sigmas.to(device=device, dtype=dtype)

    return sigmas


# ---------------------------------------------------------------------------
# Pure-Python equivalents (no torch) for use in no-GPU audit/analysis scripts
# ---------------------------------------------------------------------------

def _time_snr_shift_py(alpha: float, t: float) -> float:
    """Pure-Python SNR shift for a single t value."""
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


def build_sigma_schedule_py(
    n_steps: int,
    sampling_shift: float = 1.0,
    multiplier: float = 1.0,
) -> list[float]:
    """Pure-Python ComfyUI simple_scheduler sigma schedule.

    Equivalent to build_sigma_schedule() but operates on plain Python floats,
    with no torch dependency. Use this in no-GPU scripts (e.g., audit_dataset.py).

    Matches the server schedule exactly: build_sigmas -> simple_scheduler.

    Args:
        n_steps: Number of euler steps.
        sampling_shift: SNR shift alpha. 1.0 = no shift.
        multiplier: Scale factor for build_sigmas (multiplier * 1000).

    Returns:
        List of (n_steps + 1,) sigma values as plain Python floats.
        The last value is always 0.0 (terminal sigma).
    """
    timesteps = 1000
    m = multiplier * 1000.0
    # Build the 1000-sigma table (equivalent to build_sigmas)
    sigma_table = []
    for i in range(1, timesteps + 1):
        t = (i / timesteps) * m
        sigma_table.append(_time_snr_shift_py(sampling_shift, t / m))

    # simple_scheduler over sigma_table
    sigs = []
    ss = len(sigma_table) / n_steps
    for x in range(n_steps):
        sigs.append(sigma_table[-(1 + int(x * ss))])
    sigs.append(0.0)
    return sigs


# --- CONST noise model functions ---

def sigma_to_logsnr(sigma: float) -> float:
    """Convert a noise level sigma to log signal-to-noise ratio.

    Clamps sigma to (0.001, 0.999) before computing to avoid log(0).
    Returns logSNR = 2 * log((1 - sigma) / sigma), which is negative
    in the high-noise regime and positive near sigma=0.
    """
    s = max(0.001, min(0.999, sigma))
    return 2.0 * math.log((1.0 - s) / s)


def const_noise_scaling(sigma: torch.Tensor, noise: torch.Tensor, latent_image: torch.Tensor) -> torch.Tensor:
    """CONST.noise_scaling: sigma * noise + (1 - sigma) * latent_image"""
    sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
    return sigma * noise + (1 - sigma) * latent_image


def const_inverse_noise_scaling(sigma: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    """CONST.inverse_noise_scaling: latent / (1 - sigma)"""
    sigma = sigma.view(sigma.shape[:1] + (1,) * (latent.ndim - 1))
    return latent / (1 - sigma)


# ---------------------------------------------------------------------------
# LogSNR-uniform sparse step selection
# ---------------------------------------------------------------------------

def compute_logsnr_uniform_steps(
    width: int,
    height: int,
    n_steps: int = 30,
    n_save: int = 7,
    ref_pixels: int = 1024 * 1024,
) -> list[int]:
    """Select step indices whose logSNR values are approximately uniformly spaced.

    Instead of saving steps at uniform step-index intervals (e.g. {0,4,9,14,19,24,29}),
    this selects steps such that their logSNR values (after resolution-dependent sigma
    shifting per SD3 Eq.23) are approximately equidistant. This concentrates saved steps
    in the clean regime (high logSNR, low sigma) where the BTRM reward model needs the
    most training data, rather than wasting resolution in the noisy regime where all
    images look alike.

    Algorithm:
      1. Build the sigma schedule for (width, height) with resolution shift.
      2. For each step i in 0..n_steps-1, compute logSNR[i] = 2*ln((1-sigma)/sigma),
         clamping sigma to [0.001, 0.999] to avoid infinities.
      3. Step 0 (pure noise anchor) and step n_steps-1 (cleanest non-final step) are
         always included.
      4. For the remaining n_save-2 slots, place target logSNR values uniformly between
         logSNR[0] and logSNR[n_steps-1], then for each target find the step index whose
         logSNR is closest.

    The "final" step (sigma=0, fully denoised) is NOT included -- the caller always
    adds it separately.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        n_steps: Total number of Euler steps in the trajectory (default 30).
        n_save: Number of step indices to return (default 7).
        ref_pixels: Reference pixel count for resolution shift (default 1024*1024).

    Returns:
        Sorted list of exactly n_save step indices from 0..n_steps-1.

    Raises:
        ValueError: If n_save < 2 or n_save > n_steps.
    """
    if n_save < 2:
        raise ValueError(f"n_save must be >= 2, got {n_save}")
    if n_save > n_steps:
        raise ValueError(f"n_save ({n_save}) > n_steps ({n_steps})")

    # Compute resolution-dependent sigma shift
    ref_w = int(math.sqrt(ref_pixels * 1280 / 832))  # ~1280 for 1024^2 ref
    ref_h = int(ref_pixels / ref_w)                    # ~832
    # Use the actual resolution_shift function for consistency
    shift = resolution_shift(width, height, ref_width=1280, ref_height=832)

    # Build sigma schedule: (n_steps+1,) tensor, last element is 0.0
    sigmas = build_sigma_schedule(
        n_steps, sampling_shift=shift, device=torch.device("cpu"), dtype=torch.float32,
    )

    # Compute logSNR for steps 0..n_steps-1 (skip the terminal sigma=0 at index n_steps)
    SIGMA_MIN = 0.001
    SIGMA_MAX = 0.999
    logsnrs = []
    for i in range(n_steps):
        s = float(sigmas[i].item())
        s = max(SIGMA_MIN, min(SIGMA_MAX, s))
        logsnr = 2.0 * math.log((1.0 - s) / s)
        logsnrs.append(logsnr)

    # Step 0 and step n_steps-1 are always included
    logsnr_lo = logsnrs[0]       # Most negative (noisiest)
    logsnr_hi = logsnrs[-1]      # Most positive (cleanest non-final)

    if n_save == 2:
        return [0, n_steps - 1]

    # n_save-2 interior targets, uniformly spaced between logsnr_lo and logsnr_hi
    # (endpoints are step 0 and step n_steps-1 which are already included)
    n_interior = n_save - 2
    targets = [
        logsnr_lo + (logsnr_hi - logsnr_lo) * (k + 1) / (n_interior + 1)
        for k in range(n_interior)
    ]

    # For each target, find the step index (excluding 0 and n_steps-1) with closest logSNR
    selected = {0, n_steps - 1}
    candidate_indices = list(range(1, n_steps - 1))

    for target in targets:
        best_idx = None
        best_dist = float("inf")
        for idx in candidate_indices:
            if idx in selected:
                continue
            dist = abs(logsnrs[idx] - target)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is not None:
            selected.add(best_idx)

    # If we somehow have fewer than n_save (shouldn't happen if n_steps >= n_save),
    # fill from remaining candidates by closest distance to any unfilled target
    remaining = [i for i in candidate_indices if i not in selected]
    while len(selected) < n_save and remaining:
        # Pick the remaining step that is most distant from any already-selected step's logSNR
        selected_logsnrs = sorted(logsnrs[i] for i in selected)
        best_idx = None
        best_gap = -1.0
        for idx in remaining:
            # Find the gap this step would fill
            lsnr = logsnrs[idx]
            # Find where this logSNR sits among selected
            min_dist = min(abs(lsnr - sl) for sl in selected_logsnrs)
            if min_dist > best_gap:
                best_gap = min_dist
                best_idx = idx
        if best_idx is not None:
            selected.add(best_idx)
            remaining.remove(best_idx)

    return sorted(selected)

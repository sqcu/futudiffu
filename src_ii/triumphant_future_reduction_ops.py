"""K-tuple scatter-gather reduction ops for guided diffusion sampling.

SCATTER maps a base latent into K guidance queries (copies, downsampled variants).
GATHER reduces K denoised estimates back to one guided trajectory via signed scales.
The branch count K is a parameter — these ops are correct for K=1, K=2, and K=11000000.

See docs/triumphant_future_reduction_ops_readme.md for full documentation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .sigma_schedule import resolution_shift, build_sigma_schedule


def noise_field(max_h, max_w, seed, device, dtype):
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(1, 16, max_h, max_w, dtype=dtype, device=device, generator=gen)


def aperture(master, h, w):
    _, _, mh, mw = master.shape
    y0 = (mh - h) // 2
    x0 = (mw - w) // 2
    return master[:, :, y0:y0 + h, x0:x0 + w]


def scatter(x_base, spec):
    _, _, base_h, base_w = x_base.shape
    out = []
    for cond, (res_w, res_h), scale in spec:
        lh, lw = res_h // 8, res_w // 8
        if lh == base_h and lw == base_w:
            out.append(x_base.clone())
        else:
            out.append(F.interpolate(x_base, size=(lh, lw), mode='bilinear', align_corners=False))
    return out


def gather(denoised_list, spec):
    base = denoised_list[0]
    _, _, base_h, base_w = base.shape
    result = base.clone()
    for i in range(1, len(spec)):
        _, (res_w, res_h), scale = spec[i]
        d_i = denoised_list[i]
        lh, lw = res_h // 8, res_w // 8
        if lh != base_h or lw != base_w:
            d_i = F.interpolate(d_i, size=(base_h, base_w), mode='bilinear', align_corners=False)
        result = result + scale * (d_i - base)
    return result


def denoise_all(x_list, fields, sigmas):
    return [x - field * sigma for x, field, sigma in zip(x_list, fields, sigmas)]


def euler_step(x_base, guided, sigma, sigma_next):
    d = (x_base - guided) / sigma
    return x_base + d * (sigma_next - sigma)


def euler_sde_step(x_base, guided, sigma, sigma_next, eta_t):
    """Euler step with noise injection. Returns (x_next, mu).

    mu is the deterministic Euler mean. x_next = mu + eta_t * z.
    If eta_t < 1e-12 or sigma_next == 0 (final step), x_next = mu (no noise).

    The noise converts the deterministic ODE step into a proper Gaussian
    transition π_θ(x_{t-1} | x_t) = N(μ_θ, η_t² I), required for
    REINFORCE score function estimation. See ddreinforce.py docstring.
    """
    d = (x_base - guided) / sigma
    mu = x_base + d * (sigma_next - sigma)
    if eta_t < 1e-12 or float(sigma_next) == 0.0:
        return mu, mu
    z = torch.randn_like(mu)
    return mu + eta_t * z, mu


def latent_padded(res_w, res_h, patch_size=2):
    lh, lw = res_h // 8, res_w // 8
    return (lh + ((-lh) % patch_size), lw + ((-lw) % patch_size))


def build_per_image_sigmas(spec, n_steps, device, dtype):
    schedules = []
    for cond, (res_w, res_h), scale in spec:
        alpha = resolution_shift(res_w, res_h)
        schedules.append(build_sigma_schedule(n_steps, sampling_shift=alpha, device=device, dtype=dtype))
    return schedules


# --- Factory helpers ---

def cfg1(cond, res):
    return [(cond, res, 1.0)]


def cfg2(pos, neg, res, scale):
    return [
        (pos, res, 1.0),
        (neg, res, -(scale - 1.0)),
    ]


def cfg6(base, shrimp, typo, banana, base_res, lr1, lr2, scales):
    s_base, s_shrimp, s_typo, s_lr1, s_lr2, s_banana = scales
    return [
        (base, base_res, s_base),
        (shrimp, base_res, s_shrimp),
        (typo, base_res, s_typo),
        (base, lr1, s_lr1),
        (base, lr2, s_lr2),
        (banana, base_res, s_banana),
    ]


# --- Spherical reduction with post-gain ---

def gather_residual_gain(denoised_list, spec, gain):
    """Polar decomposition of multi-direction guidance.

    Each guidance residual defines a direction on a hypersphere. Per_scales
    in the spec are angular weights: they control WHERE the combined guidance
    points (rotation between directions). Gain controls HOW FAR the combined
    push extends (radius multiplier). The radius is the |scale|-weighted mean
    of raw residual norms — preserving the "power level" of conditioning
    without double-counting when multiple directions are parallel.

    Permutation invariant. Smooth across K values. Collapses to standard CFG
    at K=2 with gain = cfg - 1.
    """
    base = denoised_list[0]
    _, _, base_h, base_w = base.shape
    shape = base.shape

    raw_residuals = []
    scales = []
    for i in range(1, len(spec)):
        _, (res_w, res_h), scale = spec[i]
        d_i = denoised_list[i]
        lh, lw = res_h // 8, res_w // 8
        if lh != base_h or lw != base_w:
            d_i = F.interpolate(d_i, size=(base_h, base_w), mode='bilinear', align_corners=False)
        raw_residuals.append((d_i - base).flatten())
        scales.append(scale)

    if not raw_residuals:
        return base.clone()

    norms = [r.norm().clamp(min=1e-8) for r in raw_residuals]
    unit_dirs = [r / n for r, n in zip(raw_residuals, norms)]

    weighted_dir = sum(s * u for s, u in zip(scales, unit_dirs))
    direction = weighted_dir / weighted_dir.norm().clamp(min=1e-8)

    abs_scales = [abs(s) for s in scales]
    total_abs = sum(abs_scales) + 1e-12
    radius = sum(a * n for a, n in zip(abs_scales, norms)) / total_abs

    return base + (gain * radius) * direction.reshape(shape)


def cfg6_residual_gain(base, shrimp, typo, banana, base_res, lr1, lr2, per_scales, gain):
    s_base, s_shrimp, s_typo, s_lr1, s_lr2, s_banana = per_scales
    spec = [
        (base, base_res, s_base),
        (shrimp, base_res, s_shrimp),
        (typo, base_res, s_typo),
        (base, lr1, s_lr1),
        (base, lr2, s_lr2),
        (banana, base_res, s_banana),
    ]
    return spec, gain

"""Fractional Fourier Transform (FrFT) for high-dimensional image descriptors.

The 2D isotropic FrFT (separable with equal angles) commutes with spatial
rotation: F_β[f∘R_θ] = F_β[f] ∘ R_{-θ}. This means rotating the input
rotates the output, preserving shape information through the transform.

By evaluating the FrFT at multiple angles β (from near-spatial to near-frequency)
and at sparse output points, we construct a high-dimensional descriptor where:
- Resolution invariance comes from evaluating the same continuous transform
  at the same normalized output coordinates regardless of input sampling
- Aspect ratio invariance comes from the descriptor being a function of the
  field's content, not its spatial extent
- Rotation softness comes from the high-D structure: a rotated image's
  descriptor is a smooth perturbation that stays closer to the original
  than to any genuinely different image

All operations are shallow (two batched matmuls across all angles) and wide
(embarrassingly parallel across pixels, output points, and angles).

Performance (RTX 4090, compiled, 16 angles, 16x16 eval grid):
  512x512:   0.23ms    1024x1024: 0.29ms
  2048x2048: 0.51ms    3072x3072: 1.20ms
"""

import torch


def _make_normalized_coords(n: int, device: torch.device) -> torch.Tensor:
    """Map pixel indices [0, n-1] to normalized coordinates [-1, 1]."""
    if n == 1:
        return torch.zeros(1, device=device)
    return torch.linspace(-1.0, 1.0, n, device=device)


def frft_descriptor(
    image: torch.Tensor,
    n_angles: int = 16,
    n_eval: int = 16,
    angle_min: float = 0.05,
    angle_max: float = 1.52,
) -> torch.Tensor:
    """Compute FrFT descriptor — all angles batched, real-arithmetic decomposition.

    Evaluates the 2D isotropic FrFT at n_angles angles simultaneously via
    batched real-valued einsum. No Python loops, no complex dtype (inductor
    can't codegen complex ops). Mathematically equivalent to the naive
    per-angle complex matmul but 35x faster under torch.compile.

    Each angle's contribution is normalized to unit norm before concatenation,
    ensuring every FrFT angle contributes equally regardless of energy scale.

    Descriptor dimension = 2 * C * n_angles * n_eval^2.
    With defaults (C=3, 16 angles, 16x16 grid): 24,576 dimensions.

    Args:
        image: (C, H, W) tensor in [0, 1]. Any resolution/aspect ratio.
        n_angles: Number of FrFT angles to evaluate.
        n_eval: Number of evaluation points per spatial dimension.
        angle_min: Minimum FrFT angle (>0, away from identity).
        angle_max: Maximum FrFT angle (<pi/2, approaching but not reaching FFT).

    Returns:
        (D,) float32 tensor, the flattened descriptor.
    """
    C, H, W = image.shape
    device = image.device

    coord_x = _make_normalized_coords(H, device)
    coord_y = _make_normalized_coords(W, device)
    eval_u = _make_normalized_coords(n_eval, device)
    eval_v = _make_normalized_coords(n_eval, device)

    angles = torch.linspace(angle_min, angle_max, n_angles, device=device)
    sin_a = torch.sin(angles)
    cos_a = torch.cos(angles)

    # --- Phase matrices for all angles at once ---
    # K_α(u, x) = exp(i * phase) = cos(phase) + i*sin(phase)
    # phase(u, x) = ((u² + x²)*cos(α) - 2*u*x) / (2*sin(α))

    u_sq_h = eval_u ** 2                                   # (M,)
    u_sq_w = eval_v ** 2                                   # (M,)
    x_sq_h = coord_x ** 2                                  # (H,)
    x_sq_w = coord_y ** 2                                  # (W,)
    ux_h = eval_u[:, None] * coord_x[None, :]              # (M, H)
    ux_w = eval_v[:, None] * coord_y[None, :]              # (M, W)
    u_plus_x_h = u_sq_h[:, None] + x_sq_h[None, :]        # (M, H)
    u_plus_x_w = u_sq_w[:, None] + x_sq_w[None, :]        # (M, W)

    cos_k = cos_a[:, None, None]                           # (K, 1, 1)
    sin_k = sin_a[:, None, None]                           # (K, 1, 1)

    phase_h = (u_plus_x_h.unsqueeze(0) * cos_k - 2.0 * ux_h.unsqueeze(0)) / (2.0 * sin_k)
    phase_w = (u_plus_x_w.unsqueeze(0) * cos_k - 2.0 * ux_w.unsqueeze(0)) / (2.0 * sin_k)

    # Real/imag kernel components — no complex dtype
    Kx_re = torch.cos(phase_h)   # (K, M, H)
    Kx_im = torch.sin(phase_h)
    Ky_re = torch.cos(phase_w)   # (K, M, W)
    Ky_im = torch.sin(phase_w)

    f = image.float()  # (C, H, W)

    # --- Matmul 1: temp = f @ K_y^T (plain transpose, image is real) ---
    # f @ (Ky_re + i*Ky_im)^T = f @ Ky_re^T + i * f @ Ky_im^T
    Ky_re_t = Ky_re.transpose(-2, -1)   # (K, W, M)
    Ky_im_t = Ky_im.transpose(-2, -1)   # (K, W, M)

    temp_re = torch.einsum('chw,kwm->kchm', f, Ky_re_t)   # (K, C, H, M)
    temp_im = torch.einsum('chw,kwm->kchm', f, Ky_im_t)   # (K, C, H, M)

    # --- Matmul 2: result = K_x @ temp (complex × complex) ---
    # (Kx_re + i*Kx_im) @ (temp_re + i*temp_im)
    # real part = Kx_re*temp_re - Kx_im*temp_im
    # imag part = Kx_re*temp_im + Kx_im*temp_re
    res_re = (torch.einsum('kuh,kchm->kcum', Kx_re, temp_re)
            - torch.einsum('kuh,kchm->kcum', Kx_im, temp_im))
    res_im = (torch.einsum('kuh,kchm->kcum', Kx_re, temp_im)
            + torch.einsum('kuh,kchm->kcum', Kx_im, temp_re))

    # --- Per-angle normalization ---
    flat_re = res_re.reshape(n_angles, -1)   # (K, C*M*M)
    flat_im = res_im.reshape(n_angles, -1)   # (K, C*M*M)
    flat = torch.cat([flat_re, flat_im], dim=1)   # (K, 2*C*M*M)

    norms = torch.linalg.norm(flat, dim=1, keepdim=True).clamp(min=1e-12)
    flat = flat / norms

    return flat.reshape(-1)


def frft_similarity(
    desc_a: torch.Tensor,
    desc_b: torch.Tensor,
) -> torch.Tensor:
    """Cosine similarity between two FrFT descriptors."""
    a = desc_a.float()
    b = desc_b.float()
    return torch.dot(a, b) / (torch.linalg.norm(a) * torch.linalg.norm(b) + 1e-12)


def frft_discriminant_score(
    desc_img: torch.Tensor,
    desc_this: torch.Tensor,
    desc_that: torch.Tensor,
) -> torch.Tensor:
    """Project image descriptor onto the THIS-THAT discriminant axis.

    Uses centered cosine scoring: the midpoint of THIS and THAT descriptors
    defines the origin, and the discriminant axis is THIS - THAT. The image
    descriptor is projected onto this axis via cosine similarity.

    This guarantees THIS -> +1.0, THAT -> -1.0 by construction, and all
    other images fall in between based on their alignment with the
    discriminant direction. Fixes the raw-cosine-difference bug where
    unrelated images could score higher than the identity.

    Returns:
        Scalar tensor in [-1, +1].
    """
    d = desc_img.float()
    t = desc_this.float()
    h = desc_that.float()

    midpoint = (t + h) * 0.5
    axis = t - h

    centered = d - midpoint
    axis_norm = torch.linalg.norm(axis).clamp(min=1e-12)
    centered_norm = torch.linalg.norm(centered).clamp(min=1e-12)

    return torch.dot(centered, axis) / (centered_norm * axis_norm)

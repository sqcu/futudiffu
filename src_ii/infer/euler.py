"""
Euler ODE stepping and sigma/logSNR conversion for probability flow sampling.

This module contains exactly two canonical algorithms extracted from
demonstrate_rtheta_policy.py. It has no src_ii or futudiffu imports:
only torch and math. Any module in src_ii/infer/ that needs these
functions imports from here; no inlining permitted.
"""

import math

import torch
from torch import Tensor


def euler_step(x: Tensor, field: Tensor, sigma_i: Tensor, sigma_next: Tensor) -> Tensor:
    """First-order Euler integration of the probability flow ODE.

    Converts the model velocity field to a denoised estimate, then steps
    toward sigma_next. When sigma_next == 0 the step collapses exactly
    to the denoised estimate with no approximation error.
    """
    denoised = x - field * sigma_i
    if sigma_next > 0:
        d = (x - denoised) / sigma_i
        x = x + d * (sigma_next - sigma_i)
    else:
        x = denoised
    return x


def sigma_to_logsnr(sigma: float) -> float:
    """Convert a noise level sigma to log signal-to-noise ratio.

    Clamps sigma to (0.001, 0.999) before computing to avoid log(0).
    Returns logSNR = 2 * log((1 - sigma) / sigma), which is negative
    in the high-noise regime and positive near sigma=0.
    """
    s = max(0.001, min(0.999, sigma))
    return 2.0 * math.log((1.0 - s) / s)

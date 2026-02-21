"""Euler ODE solver: euler_solve().

The fourth layer in the five-function decomposition.
Takes a denoiser_fn (output of make_guided_denoiser) and a sigma schedule,
produces a trajectory of latents.

Import constraints:
  - IMPORTS nothing from futudiffu (pure torch math)
  - DOES NOT import: model_manager, server, client, sampling, training_utils
"""

import torch


def to_d(x: torch.Tensor, sigma: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    """Convert a denoiser output to a Karras ODE derivative.

    d = (x - denoised) / sigma

    Args:
        x: (B, C, H, W) current noisy latent.
        sigma: Scalar or (B,) sigma value.
        denoised: (B, C, H, W) denoised estimate from the denoiser.

    Returns:
        (B, C, H, W) ODE derivative.
    """
    sigma = sigma.view(sigma.shape[:1] + (1,) * (x.ndim - 1))
    return (x - denoised) / sigma


@torch.inference_mode()
def euler_solve(
    denoiser_fn,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    callback=None,
) -> torch.Tensor:
    """Euler ODE solver (Algorithm 2, Karras et al. 2022).

    Takes a denoiser_fn whose type signature is (x, sigma) -> denoised.
    This is the output of make_guided_denoiser() -- the solver never sees
    raw model predictions, only denoised estimates.

    Args:
        denoiser_fn: Callable (x, sigma) -> denoised.
        x: (B, C, H, W) initial noisy latent.
        sigmas: 1D tensor of (steps+1,) sigma values, ending at 0 or near-0.
        callback: Optional callback(dict) called per step with:
            {'x': x_before_step, 'i': step_index, 'sigma': sigma_i,
             'sigma_hat': sigma_hat, 'denoised': denoised_estimate}

    Returns:
        (B, C, H, W) final latent after integration.
    """
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        sigma_hat = sigmas[i]
        denoised = denoiser_fn(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({
                'x': x,
                'i': i,
                'sigma': sigmas[i],
                'sigma_hat': sigma_hat,
                'denoised': denoised,
            })
        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt
    return x

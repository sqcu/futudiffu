"""Policy gradient accumulation for DDGRPO.

Opaque to the reduction function. The policy gradient path uses the SAME
executor, gather function, and Euler formula as the inference path. Whether
CFG uses 2-tuple, 6-tuple, or hypersphere reduction is invisible here --
the executor handles packing and adapter routing, the gather function
handles reduction, and this module only sees the guided output.

Per the DDPO paper (Black et al. 2023), the per-step loss is pure
REINFORCE with no KL regularization and uniform step weighting:

    loss_t = -(advantage * log_prob_mean)
    loss_t.backward()

Drift between policy and reference is tracked as MSE between guided
outputs (diagnostic only, not a loss term). LoRA rank provides an
implicit trust region (see rlaif_policy_gradients_readme.md Section 4).

Steps with eta_t < 1e-6 are skipped entirely: deterministic transitions
have no stochastic gradient signal (the Gaussian is degenerate and
log pi_theta -> -inf).

Import constraints:
  - torch
  - src_ii.ddreinforce (step_log_prob)
  - src_ii.triumphant_future_reduction_ops (gather -- default gather_fn)
  - src_ii.multi_lora (get_adapter_params -- for optimizer init only)
  - No futudiffu imports
"""

from __future__ import annotations

import logging

import torch

from .ddreinforce import step_log_prob
from .triumphant_future_reduction_ops import gather

logger = logging.getLogger("futudiffu.policy_step")


# ---------------------------------------------------------------------------
# Gradient accumulation
# ---------------------------------------------------------------------------

def accumulate_reinforce_gradients(
    executor,
    spec,
    query_sigmas: torch.Tensor,
    trajectory: dict,
    sparse_steps: list[int],
    advantage: float,
    adapter_scales=None,
    ref_adapter_scales=None,
    gather_fn=None,
) -> dict:
    """Accumulate REINFORCE gradients for one trajectory.

    For each sparse step:
      1. Call executor with policy adapter scales (with grad)
      2. Gather guided output via gather_fn (same as inference)
      3. Compute mu_theta from Euler formula (same as euler_step)
      4. Call executor with reference adapter scales (no grad)
      5. Gather guided_ref, compute mu_ref for drift tracking
      6. lp_theta = step_log_prob(x_next, mu_theta, eta_t)
      7. loss_t = -(advantage * lp_theta.mean())
      8. loss_t.backward() (accumulates into LoRA params)

    The executor is the SAME callable used for rollout generation. It
    handles all packing, CFG forking, sigma shifting, and adapter routing.
    The gather_fn is the SAME reduction function used during inference.
    This function does not know what reduction is used.

    All sparse steps receive equal weight (no logSNR reweighting).
    No KL regularization (per DDPO paper, Section 3 and Appendix C).

    Args:
        executor: Callable with .query_sigmas attribute. Called as
            executor([x_t], [spec], step_i, adapter_scales) ->
            (denoised_per_query, scores). Must preserve gradient graph
            (no .detach() on denoised outputs).
        spec: The k-tuple spec for this trajectory (e.g. from cfg2()).
            Same spec used during rollout generation.
        query_sigmas: (n_steps+1,) sigma schedule for this trajectory.
            Same schedule used during rollout generation.
        trajectory: Dict with checkpoint_N entries (1,16,H,W latents)
            and eta_used (list of per-step noise scales).
        sparse_steps: Step indices to compute gradients for.
        advantage: Scalar advantage for this trajectory.
        adapter_scales: Adapter scales for the policy forward (with
            the policy adapter active). Constructed by the caller.
        ref_adapter_scales: Adapter scales for the reference forward
            (policy adapter zeroed out). If None, skips reference
            forward and drift tracking.
        gather_fn: Reduction function. Defaults to gather() from
            triumphant_future_reduction_ops (standard linear CFG).

    Returns:
        Dict with total_log_prob, total_drift_mse, n_steps.
    """
    reduce = gather if gather_fn is None else gather_fn
    eta_used = trajectory.get("eta_used")
    device = query_sigmas.device

    total_log_prob = 0.0
    total_drift_mse = 0.0
    n_computed = 0

    for step_idx in sparse_steps:
        sigma_t = query_sigmas[step_idx]
        sigma_next = query_sigmas[step_idx + 1]
        dt = sigma_next - sigma_t

        if eta_used is not None and step_idx < len(eta_used):
            eta_t = eta_used[step_idx]
        else:
            eta_t = float(abs(sigma_next - sigma_t)) * 0.1

        if eta_t < 1e-6:
            continue

        x_t = trajectory[f"checkpoint_{step_idx}"].to(device)

        # Policy forward: adapter active, with gradients.
        # Set query_sigmas on executor so it reads the correct sigma.
        executor.query_sigmas = [query_sigmas]
        denoised_per_query, _scores = executor([x_t], [spec], step_idx, adapter_scales)
        guided_theta = reduce(denoised_per_query[0], spec)

        d_theta = (x_t - guided_theta) / sigma_t
        mu_theta = x_t + d_theta * dt

        # Reference forward: adapter zeroed, no gradients.
        if ref_adapter_scales is not None:
            with torch.no_grad():
                denoised_ref, _ = executor([x_t], [spec], step_idx, ref_adapter_scales)
                guided_ref = reduce(denoised_ref[0], spec)
                d_ref = (x_t - guided_ref) / sigma_t
                mu_ref = x_t + d_ref * dt

        # x_next from recorded trajectory
        next_key = f"checkpoint_{step_idx + 1}"
        if next_key in trajectory:
            x_next = trajectory[next_key].to(device)
        else:
            x_next = mu_theta.detach()

        lp_theta = step_log_prob(x_next.detach(), mu_theta, eta_t)

        step_loss = -advantage * lp_theta.mean()
        step_loss.backward()

        with torch.no_grad():
            if ref_adapter_scales is not None:
                drift_mse = (guided_theta.detach() - guided_ref).pow(2).mean()
                total_drift_mse += float(drift_mse)
        total_log_prob += float(lp_theta.mean().detach())
        n_computed += 1

    return {
        "total_log_prob": total_log_prob,
        "total_drift_mse": total_drift_mse,
        "n_steps": n_computed,
    }


# ---------------------------------------------------------------------------
# Optimizer step
# ---------------------------------------------------------------------------

def policy_optimizer_step(
    model,
    policy_optimizers: dict,
    device: torch.device,
    dtype: torch.dtype,
    params: dict,
    policy_schedulers: dict | None = None,
) -> dict:
    """Clip gradients, step optimizer, zero grads.

    Lazy-inits optimizer per adapter_name on first call.

    Args:
        model: The diffusion model with MultiLoRALinear modules.
        policy_optimizers: Mutable dict of {adapter_name: optimizer}.
        device: GPU device.
        dtype: Compute dtype.
        params: Dict with adapter_name, max_grad_norm, lr.
        policy_schedulers: Optional dict of {adapter_name: scheduler}.

    Returns:
        Dict with grad_norm, n_params, lr.
    """
    from src_ii.multi_lora import get_adapter_params

    adapter_name = params["adapter_name"]
    max_grad_norm = params.get("max_grad_norm", 1.0)
    lr = params.get("lr", 1e-4)

    # Lazy-init optimizer
    if adapter_name not in policy_optimizers:
        adapter_param_dict = get_adapter_params(model, adapter_name)
        adapter_params = list(adapter_param_dict.values())
        if not adapter_params:
            return {"grad_norm": 0.0, "n_params": 0, "lr": lr,
                    "error": f"no params found for adapter {adapter_name}"}
        policy_optimizers[adapter_name] = torch.optim.AdamW(
            adapter_params, lr=lr)

    optimizer = policy_optimizers[adapter_name]

    # Collect params with gradients for clipping
    param_list = [p for group in optimizer.param_groups for p in group["params"]
                  if p.grad is not None]

    if not param_list:
        return {"grad_norm": 0.0, "n_params": 0, "lr": lr}

    grad_norm = float(torch.nn.utils.clip_grad_norm_(param_list, max_grad_norm))
    optimizer.step()

    if policy_schedulers and adapter_name in policy_schedulers:
        policy_schedulers[adapter_name].step()

    optimizer.zero_grad()

    current_lr = optimizer.param_groups[0]["lr"]

    return {
        "grad_norm": grad_norm,
        "n_params": sum(p.numel() for p in param_list),
        "lr": current_lr,
    }

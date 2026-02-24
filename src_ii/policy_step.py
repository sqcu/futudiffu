"""Policy gradient accumulation for DDGRPO.

Opaque to the reduction function. The policy gradient path uses the SAME
executor, gather function, and Euler formula as the inference path. Whether
CFG uses 2-tuple, 6-tuple, or hypersphere reduction is invisible here --
the executor handles packing and adapter routing, the gather function
handles reduction, and this module only sees the guided output.

Per the DDPO paper (Black et al. 2023), the per-step loss is pure
REINFORCE with no KL regularization and uniform step weighting.

The outer loop is step-first. For each step, trajectories are processed
in micro-batches of `microbatch_size`, with a backward() after each
micro-batch. Gradients accumulate additively into LoRA params — this is
mathematically identical to processing all trajectories at once, but
bounds GPU memory to 1 bin's activations per backward call.

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
  - No futudiffu imports
"""

from __future__ import annotations

import logging

import torch

from .ddreinforce import step_log_prob
from .triumphant_future_reduction_ops import gather

logger = logging.getLogger("futudiffu.policy_step")


# ---------------------------------------------------------------------------
# Sign agreement tracking
# ---------------------------------------------------------------------------

class SignAgreementTracker:
    """Online measurement of gradient coherence across optimizer steps.

    Compares the sign of (post_step - pre_step) parameter diffs between
    successive iterations. High sign agreement (>90%) means the optimizer
    is consistently pushing parameters in the same direction despite
    per-iteration reward noise.

    Memory cost: one int8 tensor (~29 KB for rank-8 LoRA across 240 groups)
    plus a transient bf16 clone during pre_step (~58 KB). Negligible.
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self._prev_diff_signs: torch.Tensor | None = None
        self._prev_diff_norm: float | None = None
        self._pre_snapshot: list[torch.Tensor] | None = None
        self._iteration: int = 0

    def _all_params(self) -> list[torch.nn.Parameter]:
        return [p for g in self.optimizer.param_groups for p in g["params"]]

    def _snapshot(self) -> list[torch.Tensor]:
        return [p.data.clone() for p in self._all_params()]

    def pre_step(self):
        """Call BEFORE optimizer.step(). Captures current param values."""
        self._pre_snapshot = self._snapshot()

    def post_step(self) -> dict | None:
        """Call AFTER optimizer.step(). Returns sign agreement metrics or None
        if this is the first iteration (no previous diff to compare)."""
        params = self._all_params()
        diffs = [p.data - snap for p, snap in zip(params, self._pre_snapshot)]
        diff_flat = torch.cat([d.flatten() for d in diffs])
        diff_signs = diff_flat.sign().to(torch.int8)
        diff_norm = float(diff_flat.norm())
        zero_frac = float((diff_flat == 0).float().mean())

        result = None
        if self._prev_diff_signs is not None:
            agree = float((diff_signs == self._prev_diff_signs).float().mean())
            result = {
                "sign_agreement": agree,
                "zero_frac": zero_frac,
                "diff_norm": diff_norm,
                "prev_diff_norm": self._prev_diff_norm,
                "iteration": self._iteration,
            }

        self._prev_diff_signs = diff_signs
        self._prev_diff_norm = diff_norm
        self._pre_snapshot = None
        self._iteration += 1
        return result


# ---------------------------------------------------------------------------
# Gradient accumulation
# ---------------------------------------------------------------------------

def accumulate_reinforce_gradients(
    executor,
    specs: list,
    query_sigmas: list[torch.Tensor],
    trajectories: list[dict],
    gradient_steps: list[int],
    advantages: list[float],
    adapter_scales=None,
    ref_adapter_scales=None,
    gather_fn=None,
    microbatch_size: int = 1,
) -> dict:
    """Accumulate REINFORCE gradients for N trajectories, step-first.

    Outer loop = step indices. For each step, active trajectories are
    chunked into micro-batches of `microbatch_size`. Each micro-batch
    gets one executor call, one loss computation, one backward(). Gradients
    accumulate additively into LoRA params across micro-batches.

    microbatch_size=1: 1 trajectory (2 cfg2 entries) per backward. Lowest
    memory (~12 GB peak on 4090). One bin per call.

    microbatch_size=N: all trajectories at once. Fastest (fewest forward
    passes), but O(bins) activation memory held simultaneously. Only viable
    when total bins × ~6 GB/bin fits in VRAM.

    Args:
        executor: Callable with .query_sigmas attribute. Called as
            executor(x_bases, specs, step_i, adapter_scales) ->
            (denoised_per_query, scores). Must preserve gradient graph.
        specs: N k-tuple specs (one per trajectory, e.g. from cfg2()).
        query_sigmas: N (n_steps+1,) sigma schedules.
        trajectories: N dicts with checkpoint_i entries (1,16,H,W latents)
            and eta_used (list of per-step noise scales).
        gradient_steps: Step indices to differentiate (shared superset;
            per-step filtering skips trajectories missing checkpoints).
        advantages: N advantage scalars.
        adapter_scales: Adapter scales for the policy forward.
        ref_adapter_scales: Adapter scales for the reference forward
            (no_grad). If None, skips drift tracking.
        gather_fn: Reduction function. Defaults to gather().
        microbatch_size: Trajectories per executor call. Controls memory
            vs throughput tradeoff.

    Returns:
        Dict with total_log_prob, total_drift_mse, n_steps,
        per_step (list of per-step diagnostics).
    """
    reduce = gather if gather_fn is None else gather_fn
    device = query_sigmas[0].device
    N = len(trajectories)

    total_log_prob = 0.0
    total_drift_mse = 0.0
    n_computed = 0
    per_step_diag: list[dict] = []

    for step_idx in gradient_steps:
        # --- Filter active trajectories for this step ---
        active: list[tuple[int, float]] = []  # (traj_index, eta_t)
        for i in range(N):
            traj = trajectories[i]
            if f"checkpoint_{step_idx}" not in traj:
                continue
            if f"checkpoint_{step_idx + 1}" not in traj:
                continue

            eta_used = traj.get("eta_used")
            if eta_used is not None and step_idx < len(eta_used):
                eta_t = eta_used[step_idx]
            else:
                eta_t = float(abs(query_sigmas[i][step_idx + 1]
                                  - query_sigmas[i][step_idx])) * 0.1

            if eta_t < 1e-6:
                per_step_diag.append({
                    "step_idx": step_idx, "trajectory": i, "skipped": True,
                    "sigma_t": float(query_sigmas[i][step_idx]), "eta_t": eta_t,
                })
                continue

            active.append((i, eta_t))

        if not active:
            continue

        # --- Process in micro-batches ---
        for mb_start in range(0, len(active), microbatch_size):
            mb_active = active[mb_start:mb_start + microbatch_size]
            mb_indices = [a[0] for a in mb_active]

            x_ts = [trajectories[i][f"checkpoint_{step_idx}"].to(device)
                    for i in mb_indices]
            mb_specs = [specs[i] for i in mb_indices]
            mb_sigmas = [query_sigmas[i] for i in mb_indices]

            # --- ONE executor call per micro-batch ---
            executor.query_sigmas = mb_sigmas
            denoised_per_query, _scores = executor(
                x_ts, mb_specs, step_idx, adapter_scales)

            # --- Euler chain → micro-batch loss ---
            mb_loss = torch.tensor(0.0, device=device)
            mb_guided_detached: list[torch.Tensor] = []
            mb_lp_values: list[float] = []

            for q, (active_i, eta_t) in enumerate(mb_active):
                sigma_t = query_sigmas[active_i][step_idx]
                sigma_next = query_sigmas[active_i][step_idx + 1]
                dt = sigma_next - sigma_t

                guided_theta = reduce(denoised_per_query[q], mb_specs[q])

                if guided_theta.grad_fn is None:
                    logger.error(
                        "grad_fn is None on guided_theta at step %d traj %d",
                        step_idx, active_i,
                    )

                d_theta = (x_ts[q] - guided_theta) / sigma_t
                mu_theta = x_ts[q] + d_theta * dt

                x_next = trajectories[active_i][f"checkpoint_{step_idx + 1}"].to(device)
                lp = step_log_prob(x_next.detach(), mu_theta, eta_t)

                mb_loss = mb_loss - advantages[active_i] * lp.mean()
                mb_guided_detached.append(guided_theta.detach())
                mb_lp_values.append(float(lp.mean().detach()))

            # --- backward per micro-batch: graph freed after each ---
            mb_loss.backward()

            # --- Reference forward for drift (no_grad) ---
            if ref_adapter_scales is not None:
                with torch.no_grad():
                    executor.query_sigmas = mb_sigmas
                    denoised_ref, _ = executor(
                        x_ts, mb_specs, step_idx, ref_adapter_scales)
                    for q, active_i in enumerate(mb_indices):
                        guided_ref = reduce(denoised_ref[q], mb_specs[q])
                        drift_mse = (mb_guided_detached[q] - guided_ref).pow(2).mean()
                        total_drift_mse += float(drift_mse)

            # --- Per-query diagnostics ---
            for q, (active_i, eta_t) in enumerate(mb_active):
                total_log_prob += mb_lp_values[q]
                n_computed += 1
                per_step_diag.append({
                    "step_idx": step_idx,
                    "trajectory": active_i,
                    "skipped": False,
                    "sigma_t": float(query_sigmas[active_i][step_idx]),
                    "eta_t": eta_t,
                    "inv_eta_sq": 1.0 / (eta_t * eta_t),
                    "log_prob": mb_lp_values[q],
                    "n_active": len(active),
                })

    return {
        "total_log_prob": total_log_prob,
        "total_drift_mse": total_drift_mse,
        "n_steps": n_computed,
        "per_step": per_step_diag,
    }


# ---------------------------------------------------------------------------
# Optimizer step
# ---------------------------------------------------------------------------

def policy_optimizer_step(
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float = 1.0,
    scheduler=None,
    sign_tracker: SignAgreementTracker | None = None,
) -> dict:
    """Clip gradients and step optimizer.

    The caller owns zero_grad (before accumulation) and optimizer creation
    (at setup time). This function is clip + step only.

    Args:
        optimizer: The policy optimizer (created at setup, not here).
        max_grad_norm: Gradient clipping threshold.
        scheduler: Optional LR scheduler to step after optimizer.
        sign_tracker: Optional SignAgreementTracker. If provided, snapshots
            params before step and computes sign agreement after.

    Returns:
        Dict with grad_norm, n_params, n_params_with_grad, lr,
        and optionally sign_agreement (dict or None).
    """
    param_list = [p for group in optimizer.param_groups for p in group["params"]
                  if p.grad is not None]

    if not param_list:
        return {"grad_norm": 0.0, "n_params": 0,
                "n_params_with_grad": 0,
                "lr": optimizer.param_groups[0]["lr"]}

    grad_norm = float(torch.nn.utils.clip_grad_norm_(param_list, max_grad_norm))

    if sign_tracker is not None:
        sign_tracker.pre_step()

    optimizer.step()

    sign_result = None
    if sign_tracker is not None:
        sign_result = sign_tracker.post_step()

    if scheduler is not None:
        scheduler.step()

    result = {
        "grad_norm": grad_norm,
        "n_params": sum(p.numel() for p in param_list),
        "n_params_with_grad": len(param_list),
        "lr": optimizer.param_groups[0]["lr"],
    }
    if sign_result is not None:
        result["sign_agreement"] = sign_result
    return result

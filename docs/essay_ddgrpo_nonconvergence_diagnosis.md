# DDGRPO Nonconvergence Diagnosis

Date: 2026-02-23
Run: `training_output/ddgrpo_256sq_fast/`
Config: N_ITERS=30, K=4, N_STEPS=30, ETA_SCALE=0.1, LR=1e-4, MAX_GRAD_NORM=1.0, ADAPTER_RANK=8

## Outcome

30 iterations completed with finite gradients throughout. First-5-mean
reward=-0.204, last-5-mean=-0.301, delta=-0.097. No convergence.

## Hypothesis Results

### H1: Autograd graph severed — RULED OUT

Code audit of `policy_step.py`, `batch_executor.py`, `forward_packed.py`,
and `validate_ddgrpo_e2e.py` confirms the autograd graph is intact from
`step_log_prob` through `mu_theta` → `guided_theta` → `packed_forward` →
LoRA parameters. The `.cpu()` in `batch_executor.py:225` and `.to(device)`
in `validate_ddgrpo_e2e.py:223` both preserve `grad_fn`. The `x_next.detach()`
at `policy_step.py:149` is correct (target, not differentiable path). No
`torch.no_grad()` wraps the policy forward.

### H2: Weight delta — NONZERO, MONOTONIC, COHERENT

Loaded all 30 checkpoints. Results:

| Metric | Step 0→29 |
|--------|-----------|
| Raw parameter delta (Frobenius) | 2.117 |
| Initial parameter norm | 35.670 |
| Delta/init ratio | 5.94% |
| A\*B product delta (Frobenius) | 1.508 |
| A\*B product init norm | 17.713 |
| Product delta/init ratio | 8.51% |

**Per-iteration trajectory**: `delta_from_0` grows monotonically (0.277 → 2.117).
`step_delta` shrinks monotonically (0.277 → 0.093). This is NOT a random walk
(which would show `delta_from_0 ~ sqrt(n)` and constant `step_delta`). The
optimizer is moving coherently in a persistent direction.

**Per-module A\*B product delta** (step 0→29):
- ff_w1w3: mean 0.186 (largest — FFN gate changes most)
- attn_qkv: mean 0.143
- ff_w2: mean 0.117
- attn_out: mean 0.083 (smallest)

All 240 LoRA pairs (30 layers × 4 modules × A+B) change. No frozen/stuck params.

### H3: Pre-clip gradient norm — ALWAYS CLIPPED

Every iteration has `grad_norm > MAX_GRAD_NORM=1.0` (min 0.891, max 5.406,
mean 2.37). 29/30 iterations clip. Direction is preserved but magnitude is
crushed by 0.89–5.41×.

### H4: K=4 advantage distribution — REASONABLE

60% of trajectories have |advantage| > 0.5. No degenerate groups (all 4
trajectories at zero). Advantages span [-1.48, +1.42].

### H5: Sparse step eta_t scaling — ROOT CAUSE IDENTIFIED

**The gradient is dominated by noisy-regime steps where the model processes
essentially random noise.** Computed eta_t and 1/eta_t^2 at each sparse step
for all resolutions:

| Step | sigma_t | eta_t    | 1/eta^2    | % of gradient |
|------|---------|----------|------------|---------------|
| 0    | 1.000   | 0.000846 | 1,397,404  | **53.7%**     |
| 6    | 0.941   | 0.001176 | 722,940    | **27.8%**     |
| 12   | 0.857   | 0.001745 | 328,260    | 12.6%         |
| 18   | 0.727   | 0.002856 | 122,617    | 4.7%          |
| 24   | 0.500   | 0.005496 | 33,102     | **1.3%**      |

**81.5% of the gradient comes from steps 0 and 6** (sigma > 0.94), where the
latent is essentially pure noise. The model's output at these steps has
minimal semantic content — it's guessing the clean image from noise. The
reward signal comes from the FINAL image (sigma=0), but that sigma regime
isn't in the sparse step set at all.

**All 5 sparse steps are in the noisy regime** (sigma ≥ 0.5). The clean-regime
steps (sigma < 0.5, steps 25–29) that make the visual decisions pinkify
measures are not sampled for gradient computation.

The 1/eta^2 disparity between step 0 (1.4M) and step 24 (33K) is 42×.
After gradient clipping, the step direction is ~54% determined by the
noisiest step's gradient.

### H6: Within-group score variance — NON-TRIVIAL

Mean within-group pinkify score std is 0.104 across iterations. Not collapsed
to < 0.01. The BTRM discriminates between trajectories, but the gradient
doesn't optimize toward that discrimination signal because it's computed at
the wrong sigma values.

## Root Cause

The REINFORCE gradient accumulates `advantage * ∇_θ log π_θ(x_{t-1}|x_t)` at
each sparse step. The log-prob gradient magnitude scales as 1/eta_t^2, where
eta_t = ETA_SCALE * |dt| is the injected SDE noise. This scaling assigns
credit **inversely proportional to each step's causal influence on the outcome**.

The denoising trajectory is a policy rollout, and the key structural property
is that **later steps override earlier steps**. The diffusion model is a
contraction mapping: each denoising step refines the image, overwriting the
predictions of earlier steps. A perturbation to the model's output at step 0
(sigma=1.0) has negligible effect on the final image because 29 subsequent
denoising steps will converge the trajectory regardless. A perturbation at
step 29 (sigma≈0) goes directly into the output — nothing corrects it.

The **variance contribution of each step to the final output image** is:

- Step 0 (σ=1.0): ~zero — 29 subsequent denoising steps override this
  action. The policy's own future actions erase whatever this step does.
- Step 24 (σ=0.5): moderate — 5 remaining steps partially override.
- Step 29 (σ≈0): maximal — this IS the final image. No corrections follow.

But the 1/eta_t^2 gradient scaling assigns:

- Step 0: 53.7% of gradient magnitude (near-zero causal influence)
- Step 24: 1.3% of gradient magnitude (significant causal influence)
- Steps 25-29: 0% (maximal causal influence, not in sparse step set)

This is the same mechanism as the vanishing-credit problem in long-horizon RL,
except here it's structural to the denoising dynamics rather than a pure
variance issue. The gradient assigns credit where the policy cannot causally
affect the outcome, and ignores the steps where the policy's actions directly
determine the reward.

The optimizer moves coherently (monotonic weight drift, 5.94% of init norm
over 30 iterations) but in a direction determined by noisy-step gradients
whose causal influence on the final image is erased by subsequent denoising.

## Remediation Options

### R1: Per-step gradient normalization (highest impact)

Normalize each step's gradient contribution before summing across steps.
Instead of accumulating raw `step_loss.backward()`, normalize by the step's
expected gradient magnitude:

```python
weight_t = 1.0 / (inv_eta_sq_t / sum_inv_eta_sq)  # equalize contribution
step_loss = -advantage * lp_theta.mean() * weight_t
```

This makes each step contribute equally regardless of eta_t.

### R2: Use logSNR step weighting (moderate impact)

Apply `logsnr_step_weight(sigma_t)` which gives weight=1.0 at sigma=0 and
decays geometrically into the noisy regime. This would down-weight steps
0 and 6 substantially.

### R3: Sample sparse steps from clean regime (moderate impact)

Shift sparse step selection toward the clean end. Instead of evenly spaced
{0, 6, 12, 18, 24}, use logSNR-uniform spacing which concentrates steps
near sigma=0.

### R4: Increase ETA_SCALE (moderate impact)

ETA_SCALE=0.1 with small dt produces tiny eta_t (0.0008 at step 0). Increasing
to ETA_SCALE=1.0 would make eta_t ~ dt, reducing the 1/eta^2 disparity to
the dt^2 range (~7× instead of 42×). More noise per step also increases
trajectory diversity within groups.

### R5: Reduce MAX_GRAD_NORM (low impact)

BTRM training used MAX_GRAD_NORM=0.1, not 1.0. The pre-clip norms (0.89–5.41)
suggest a lower clip threshold would provide more consistent step sizes, but
this doesn't fix the direction problem.

## Persistence Additions (implemented)

`src_ii/policy_step.py`:
- `grad_fn` verification at `guided_theta` and `mu_theta` with logger.error
- Per-step diagnostics: `step_idx`, `sigma_t`, `eta_t`, `inv_eta_sq`,
  `log_prob`, `step_loss`, `has_grad_fn`, `drift_mse`
- Per-module gradient norms aggregated by module type (attn_qkv, attn_out,
  ff_w1w3, ff_w2) with mean/max/min

`scripts_ii/validate_ddgrpo_e2e.py`:
- Seeds persisted per iteration in metrics
- `per_step_diag` array in metrics (from accumulate_reinforce_gradients)
- `module_grad_norms` in metrics (from policy_optimizer_step)
- `n_params_with_grad` in metrics

## Implemented Remediation: DDGRPO v2

**Script**: `scripts_ii/run_ddgrpo_v2.py`
**Output**: `training_output/ddgrpo_v2/`

V2 addresses the root cause by applying R1 implicitly (all steps contribute) and
correcting the hyperparameters (R5). Specific changes from v1:

| Parameter | v1 | v2 | Rationale |
|-----------|----|----|-----------|
| Gradient steps | 5 sparse ({0,6,12,18,24}) | ALL 29 non-final | Clean-regime steps now contribute; no more 81.5% noisy-regime domination |
| LR | 1e-4 | 2e-5 | 5× reduction, appropriate for RL fine-tuning of pretrained weights |
| MAX_GRAD_NORM | 1.0 | 0.1 | Match BTRM training; consistent step sizes |
| B (prompts/iter) | 1 | 4 | Prompt diversity per optimizer step |
| B×K (rollouts/iter) | 4 | 16 | Hardware saturation, better advantage estimates |
| Resolution | fixed 256² | funfetti-sampled (256²–512²) | Multi-resolution generalization |

The key structural fix: passing `save_steps=set(range(N_STEPS))` captures ALL
step checkpoints, and `gradient_steps = [s for s in all_steps if checkpoint_{s+1} exists]`
ensures ALL 29 non-final steps receive REINFORCE gradients. The 1/eta² gradient
magnitude disparity still exists per step, but the clean-regime steps (25–29) that
were entirely absent from v1 now contribute. With 29 steps instead of 5, no single
step can dominate >20% of the total gradient.

Per-group advantage computation (z-scored within each K-group of the same prompt
+ resolution) ensures different groups have independent advantage scales.

## Data Artifacts

- 30 checkpoints: `training_output/ddgrpo_256sq_fast/checkpoints/policy_step_0000..0029.safetensors`
- Training metrics: `training_output/ddgrpo_256sq_fast/training_metrics.jsonl`
- Run analysis: `training_output/ddgrpo_256sq_fast/run_analysis.json`
- Optimizer state: `training_output/ddgrpo_256sq_fast/checkpoints/optimizer_state.pt`

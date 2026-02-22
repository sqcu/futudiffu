# DDGRPO Code Review: Gap Analysis Between Spec and Implementation

This document audits the end-to-end policy optimization code in `src_ii/` and
`scripts_ii/` for adherence to the DDGRPO algorithm specified in
`docs/essay_ddpo_to_ddgrpo.md`.

---

## 1. Executive Summary

The current codebase implements **BTRM (Bradley-Terry Reward Model) training**,
not DDGRPO policy optimization. These are fundamentally different algorithms that
serve different purposes in the policy optimization pipeline:

- **BTRM training** (what exists): Trains a reward model to score images by
  pairwise preference learning. The diffusion backbone is frozen; only a LoRA
  adapter and scoring head are trained. The loss is Bradley-Terry log-sigmoid on
  (preferred, rejected) pairs. No trajectories are generated during training.
  No log-probabilities are computed. No advantages are normalized.

- **DDGRPO** (what the spec describes): Optimizes the diffusion policy itself to
  maximize expected reward. Requires generating multiple trajectories per prompt,
  computing per-step log-probabilities, normalizing advantages within prompt groups,
  clipping importance ratios, and penalizing KL divergence from a reference policy.

The gap is categorical, not incremental. BTRM training is the **reward model
training phase** that would precede DDGRPO. DDGRPO is the **policy optimization
phase** that would consume BTRM scores as its reward signal. The spec document
(Section 5.4) explicitly identifies this relationship: the BTRM produces scalar
scores suitable as reward signals for DDGRPO.

There is, however, meaningful partial infrastructure:

1. A frozen-codebase (`src/futudiffu/policy_loss.py`) implementation of DRGRPO
   loss components: `compute_group_advantages`, `compute_step_log_ratios`,
   `clipped_policy_loss`, `latent_entropy_bonus`, `reference_anchor_loss`, and
   the combined `drgrpo_diffusion_loss`. This code is READ-ONLY under the src/
   freeze policy.

2. A frozen-codebase (`src/futudiffu/training_utils.py`) REINFORCE gradient
   accumulation function `compute_reinforce_step` that computes per-step
   MSE-based log-ratios between policy and reference model outputs.

3. A frozen-codebase orchestration script (`scripts/train.py`) that generates K
   rollouts per prompt, computes group advantages, and accumulates REINFORCE
   gradients via the server RPC.

4. `src_ii/` server endpoints (`/accumulate_policy_gradients`,
   `/policy_optimizer_step`) and client methods that delegate to the frozen
   `src/futudiffu/training_utils.py` implementation.

None of this infrastructure has been ported to `src_ii/` as independent,
test-ready modules. The `src_ii/` training code (`btrm_training.py`,
`btrm_model.py`) is exclusively BTRM reward model training.

---

## 2. Component-by-Component Audit

### 2a. Trajectory Generation with Stochastic Policy

**DDGRPO Spec (Section 4.2, 5.1):** Generate K trajectories per prompt from
the current policy pi_{theta_old}. The denoising process must be stochastic
(not deterministic ODE) to define a Gaussian policy with computable
log-probabilities. The spec recommends injecting noise at each Euler step:
`x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * v_theta(x_t, c, t) + eta_t * z`.

**Status: MISSING in src_ii/**

The `src_ii/solver.py` `euler_solve()` function (line 32-71) is a deterministic
Euler ODE solver decorated with `@torch.inference_mode()`. There is no noise
injection, no eta_t parameter, no stochastic variant. The solver computes
`x = x + d * dt` where `d = (x - denoised) / sigma` and `dt = sigmas[i+1] - sigmas[i]`.
This is a pure ODE integration with zero stochasticity.

The `src_ii/rollout.py` `rollout()` function (line 40-182) wraps `euler_solve`
and runs entirely under `torch.inference_mode()` (line 155). It generates
deterministic trajectories from a seed. There is no mechanism to inject per-step
noise.

The frozen `scripts/train.py` generates rollouts via `client.sample_trajectory()`
(line 576), which calls the server's rollout endpoint. The server's rollout also
uses the deterministic Euler solver.

**Gap:** A stochastic Euler solver (SDE variant) is needed. The spec recommends
`eta_t proportional to |sigma_{t-1} - sigma_t|` (Section 5.1). This is a new
function in `src_ii/solver.py`.

### 2b. Per-Step Log-Probability Tracking

**DDGRPO Spec (Section 4.2):** For each trajectory k and step t, compute
`log p_theta(x_{t-1}^k | x_t^k, c)`. For a Gaussian policy with injected noise
eta_t: `log p = -||x_{t-1} - mu_theta||^2 / (2 eta_t^2) + const`.

**Status: PARTIAL (frozen codebase only)**

The frozen `src/futudiffu/policy_loss.py` `compute_step_log_ratios()` (line 40-69)
computes per-step log-ratios using the MSE between policy and reference denoised
outputs:
```python
log_ratios[t] = -mse / (2.0 * sigma_t * sigma_t + 1e-10)
```

This uses sigma_t^2 as the variance denominator, which is the DDPM formulation.
For a flow-matching model with injected noise eta_t, the denominator should be
eta_t^2 instead. This is a subtle but important distinction identified in the
spec (Section 5.1).

The frozen `src/futudiffu/training_utils.py` `compute_reinforce_step()` (line
408-479) computes the same log-ratio for a single step and immediately backwards
into LoRA params, using `sigma * sigma` as the variance term.

**Neither function exists in `src_ii/`.**

**Gap:** The log-ratio computation needs to be reimplemented in `src_ii/` with
the correct variance term (eta_t^2 from the injected noise, not sigma_t^2 from
the noise schedule). The existing implementations in the frozen codebase are
reference-quality but use the wrong denominator for flow matching with injected
stochasticity.

### 2c. Group-Relative Advantage Normalization

**DDGRPO Spec (Section 4.2):** `A_hat_k = (r_k - mean(r_1...r_K)) / (std(r_1...r_K) + eps)`
computed within a group of K trajectories for the same prompt.

**Status: PARTIAL (frozen codebase only)**

The frozen `src/futudiffu/policy_loss.py` `compute_group_advantages()` (line
22-37) implements exactly this:
```python
def compute_group_advantages(rewards, eps=1e-8):
    mean = rewards.mean()
    std = rewards.std()
    return (rewards - mean) / (std + eps)
```

The frozen `scripts/train.py` calls this at line 593:
```python
advantages = compute_group_advantages(rewards)
```

**This function does NOT exist in `src_ii/`.** It is a 4-line function but it
must be reimplemented (not imported from `src/futudiffu/`) per the src/ freeze
policy.

**Gap:** Trivial to implement. ~5 lines of pure torch math.

### 2d. PPO-Style Importance Ratio Clipping

**DDGRPO Spec (Section 4.5, step 3d):**
`L[k,t] = -min(rho[k,t] * A_hat_k, clip(rho[k,t], 1-eps, 1+eps) * A_hat_k)`

**Status: PARTIAL (frozen codebase only)**

The frozen `src/futudiffu/policy_loss.py` `clipped_policy_loss()` (line 72-113)
implements asymmetric DAPO-style clipping:
```python
ratio = torch.exp(log_ratio_sum)
ratio_clipped = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
surr1 = ratio * adv
surr2 = ratio_clipped * adv
total = total + torch.min(surr1, surr2)
```

Two differences from the spec:
1. The frozen code sums per-step log-ratios into a trajectory-level ratio before
   clipping. The DDGRPO spec applies clipping per-step (`rho_{k,t}` not
   `prod_t rho_{k,t}`). Per-step clipping is more conservative and prevents
   the product of many small ratios from producing a very large trajectory ratio.
2. The frozen code uses asymmetric clips (DAPO: clip_low=0.2, clip_high=0.28).
   The spec uses symmetric epsilon clipping. Both are valid; asymmetric is
   arguably better.

**This function does NOT exist in `src_ii/`.**

**Gap:** Needs reimplementation in `src_ii/` with per-step (not trajectory-level)
clipping to match the DDGRPO spec.

### 2e. KL Penalty Against Reference Policy

**DDGRPO Spec (Section 4.2):** An explicit KL divergence term
`beta * D_KL^t(pi_theta || pi_ref)` using the unbiased estimator:
`D_KL = exp(log(pi_ref/pi_theta)) - log(pi_ref/pi_theta) - 1`.
pi_ref is a frozen reference policy (the initial model before any fine-tuning).

**Status: PARTIAL (frozen codebase, different formulation)**

The frozen `src/futudiffu/policy_loss.py` `reference_anchor_loss()` (line
139-155) implements MSE between policy and reference outputs:
```python
return torch.nn.functional.mse_loss(pi_output, ref_output.detach())
```

This is a KL-like regularizer but NOT the spec's unbiased KL estimator. MSE in
output space is related to KL divergence for Gaussian policies, but the
relationship depends on the variance (which the MSE anchor ignores). The spec's
unbiased estimator `exp(x) - x - 1` where `x = log(pi_ref/pi_theta)` is
guaranteed non-negative and is the formally correct KL penalty.

The frozen `compute_reinforce_step()` uses the reference model (LoRA scale=0)
as pi_ref, which matches the spec's intent: the reference is the base model
without the policy adapter.

**No KL penalty computation exists in `src_ii/`.**

**Gap:** The unbiased KL estimator needs to be implemented fresh in `src_ii/`.
The MSE anchor in the frozen code is a reasonable approximation but does not
match the spec. The reference policy mechanism (LoRA scale=0) is structurally
sound and transfers to `src_ii/` via `BTRMCompoundModel.set_adapter_scale(0.0)`.

### 2f. Per-Step Credit Assignment

**DDGRPO Spec (Section 4.2):** The objective sums over all denoising steps:
`(1/T) sum_{t=0}^{T} [clipped_loss_t - beta * kl_t]`. Each step gets the
same trajectory-level advantage A_hat_k, but has its own importance ratio
rho_{k,t} and KL penalty D_KL^t.

**Status: PARTIAL (frozen codebase)**

The frozen `compute_step_log_ratios()` computes per-step log-ratios. The frozen
`compute_reinforce_step()` backwards per-step into LoRA params. The frozen
`clipped_policy_loss()` sums step-level log-ratios into trajectory-level ratios
before clipping (which collapses per-step credit assignment into trajectory-level).

The existing BTRM training in `src_ii/` has no per-step credit assignment at all
because it does not do policy optimization. It scores individual images at
specific sigma values, not trajectories.

**Gap:** Per-step credit assignment needs full reimplementation in `src_ii/`.
The infrastructure for sparse step saving (7+1 steps per trajectory) exists in
the dataset pipeline, but DDGRPO requires full 30-step trajectories generated
on-the-fly, not loaded from disk.

### 2g. LogSNR-Weighted Step Contributions

**DDGRPO Spec (Section 5.2):** Weight each step's gradient contribution by its
logSNR sampling weight: `nabla J ~ sum_{t} w_logSNR(sigma_t) * nabla log p * A_hat`.
The spec notes that sigma=0 gets full weight=1.0 and 80% of training signal
comes from sigma=0 positions.

**Status: DIFFERENT (implemented for BTRM, not for policy gradients)**

The `src_ii/pair_sampler.py` `logsnr_sampling_weight()` (line 78-143)
implements the geometric decay weight function. This is used to weight pair
SAMPLING in BTRM training -- cleaner pairs are sampled more often.

The `src_ii/btrm_training.py` `pair_sigma_weight()` (line 96-115) computes
the geometric mean of two individual weights for pair sampling.

These are SAMPLING weights for BTRM pair selection, NOT loss/gradient weights
for policy optimization. They operate on pairs of pre-recorded images at fixed
sigma values, not on denoising steps within a trajectory.

For DDGRPO, the weight would be applied per denoising step within a trajectory,
multiplying the gradient contribution of each step. The weight function is the
same (`logsnr_sampling_weight`), but the application context is different.

**Gap:** The weight function exists and is correct. Applying it per denoising
step in a DDGRPO loss computation is straightforward but requires the DDGRPO
loss function to exist first.

### 2h. Multi-Head Reward Composition

**DDGRPO Spec (Section 5.4):**
`r_k = alpha_pinkify * score_pinkify(x_0^k) + alpha_tnt * score_tnt(x_0^k)`
or separate per-head optimization.

**Status: DIFFERENT (BTRM per-head loss, not combined reward)**

The `src_ii/btrm_training.py` computes INDEPENDENT per-head BT loss:
```python
for head_idx, (name, pref_key) in enumerate(zip(head_names, pref_keys)):
    bt = -F.logsigmoid(pos_s - neg_s)
    total_bt = total_bt + bt
    active_heads += 1
```
Loss is normalized by `active_heads` (line 1063, 1287, 1395).

For DDGRPO, the reward heads would need to be combined into a SCALAR reward
r_k for each trajectory. The combination could be a weighted sum (as the spec
suggests) or handled as separate optimization objectives.

The BTRM model (`src_ii/btrm_model.py`) already produces multi-headed scores
as `(B, N_heads)` tensors via `ScoreUnembedder`. These are directly usable as
reward function outputs for DDGRPO.

**Gap:** A reward composition function `r_k = f(scores_k)` needs to be defined.
This is a design decision, not an engineering task.

---

## 3. Current Training Paradigm Analysis

### What the Code Actually Does Today

The codebase implements a three-phase pipeline:

**Phase A: Trajectory Dataset Generation** (`src_ii/rollout.py`,
`scripts_ii/generate_multi_res_trajectories.py`)
- Deterministic Euler ODE rollouts from the frozen base model
- Saves 7 logSNR-uniform intermediate steps + final per trajectory
- Multi-resolution (6 tiers, 26 resolutions in the current dataset)
- Two attention backends (SDPA, SageAttention) for quantization artifact diversity
- Result: static dataset of latent trajectories on disk

**Phase B: BTRM Reward Model Training** (`src_ii/btrm_training.py`,
`src_ii/btrm_model.py`, `scripts_ii/run_reward_validated_training.py`)
- Loads pre-generated trajectory latents from disk
- Samples pairwise comparisons on-the-fly from ~1.6M pair space
- Computes preference labels via ground truth reward functions (pinkify_score_gpu,
  thisnotthat_score_gpu) by VAE-decoding latents to pixels, scoring, and
  comparing scores
- Trains a LoRA adapter + ScoreUnembedder on the frozen diffusion backbone
  using Bradley-Terry pairwise ranking loss
- FlexAttention batch packing for multi-image scoring
- FLOPS-budget macrobatch sampling across resolution tiers
- Validation: PINKIFY 6-image ranking, TNT 4-image ranking, cross-head
  decorrelation via Spearman rho

**Phase C: Policy Optimization (partially implemented in frozen code)**
- `scripts/train.py`: Generates K rollouts per prompt via server RPC
- Computes BTRM scores as rewards
- Group-relative advantage normalization
- REINFORCE gradient accumulation at sparse steps
- `src/futudiffu/policy_loss.py`: DRGRPO loss components (clipping, entropy,
  anchor)
- Server-side gradient accumulation and optimizer step

### Key Architectural Observations

1. **Phase B is mature and GPU-validated.** The BTRM training pipeline runs
   end-to-end with reward-function-derived labels, FlexAttention packing,
   multi-resolution support, and real-time validation. This is production-quality
   infrastructure.

2. **Phase C exists in frozen code only.** The `src/futudiffu/policy_loss.py`
   and `src/futudiffu/training_utils.py:compute_reinforce_step` implement the
   core DDGRPO math but are in the frozen `src/` directory. The `src_ii/server.py`
   routes to these frozen implementations via import delegation (line 1183:
   `from futudiffu.training_utils import accumulate_policy_gradients`). This
   violates the spirit of the src/ freeze: new training code should not route
   through frozen implementations.

3. **The rollout pipeline is deterministic.** Both `src_ii/solver.py` and the
   frozen `src/futudiffu/sampling.py` use deterministic Euler integration.
   DDGRPO requires stochastic trajectories to define a policy distribution.

4. **The reward model is differentiable.** `BTRMCompoundModel.score_differentiable()`
   and `score_differentiable_packed()` compute scores with full autograd graph
   intact. This enables the "reward gradient" approach (Section 5.4 of the spec):
   `nabla_theta r(x_0, c)` directly, sidestepping the policy gradient formulation.
   This is DRGPO, not DDGRPO, but the infrastructure supports both.

---

## 4. Implementation Roadmap

Dependency-ordered list of what needs to be built in `src_ii/` for DDGRPO:

### Tier 1: Core Math (no dependencies, can be built immediately)

**4.1 `src_ii/policy_math.py` — Pure math functions for policy optimization**
- `compute_group_advantages(rewards, eps=1e-8) -> Tensor`
  Z-score normalization. ~5 lines. Direct port of frozen `policy_loss.py:22-37`.
- `compute_step_log_ratio(mu_pi, mu_ref, eta_t) -> float`
  Per-step Gaussian log-ratio with eta_t^2 denominator (not sigma_t^2).
- `unbiased_kl_divergence(log_pi_ref, log_pi_theta) -> Tensor`
  `exp(x) - x - 1` where `x = log_pi_ref - log_pi_theta`. ~3 lines.
- `clipped_surrogate_loss(ratio, advantage, eps=0.2) -> Tensor`
  Per-step (not trajectory-level) PPO clip. ~5 lines.
- `ddgrpo_step_loss(ratio, advantage, kl, beta, eps) -> Tensor`
  Combined per-step loss: `-min(ratio*A, clip(ratio)*A) + beta*kl`.

Estimated effort: ~50 lines of pure torch math. No imports from `src/futudiffu/`.

### Tier 2: Stochastic Solver (depends on nothing)

**4.2 `src_ii/solver.py` — Add `euler_solve_sde()` alongside existing `euler_solve()`**
- Same interface as `euler_solve()` plus `eta_schedule: Tensor` parameter.
- At each step: `x = x + d * dt + eta_t * z` where `z ~ N(0, I)`.
- Returns trajectory AND per-step `(mu_theta, eta_t, z_t)` tuples for
  log-probability computation.
- `eta_t = eta_scale * |dt|` where dt = sigma_{t+1} - sigma_t.
- Must NOT be decorated with `@torch.inference_mode()` — gradients may need to
  flow for DDGRPO training.

Estimated effort: ~40 lines. Straightforward extension of existing `euler_solve()`.

### Tier 3: Stochastic Rollout (depends on Tier 2)

**4.3 `src_ii/rollout.py` — Add `rollout_stochastic()` alongside existing `rollout()`**
- Same interface as `rollout()` plus `eta_scale: float` parameter.
- Uses `euler_solve_sde()` instead of `euler_solve()`.
- Returns per-step log-probabilities alongside the trajectory.
- Stores `(mu_theta, eta_t, x_{t-1})` at each step for offline log-prob
  recomputation.

Estimated effort: ~30 lines. Thin wrapper around new solver.

### Tier 4: DDGRPO Loss Function (depends on Tier 1)

**4.4 `src_ii/ddgrpo_loss.py` — Complete DDGRPO loss computation**
- `ddgrpo_loss(log_probs_old, rewards, reference_log_probs, ...)`
- Computes group advantages, per-step importance ratios, per-step KL, and
  the clipped surrogate loss.
- Supports logSNR step weighting.
- Supports multi-head reward composition.

Estimated effort: ~80 lines. Composes Tier 1 functions.

### Tier 5: DDGRPO Training Loop (depends on Tiers 1-4 + existing BTRM infrastructure)

**4.5 `src_ii/ddgrpo_training.py` — The training orchestration**
- For each training iteration:
  1. Sample B prompts
  2. For each prompt, generate K stochastic trajectories (Tier 3)
  3. Score final images with the trained BTRM (existing infrastructure)
  4. Compute DDGRPO loss (Tier 4)
  5. Backward and optimizer step
- Uses existing: `BTRMCompoundModel.score()` for reward, prompt encoding,
  FlexAttention packing for batch scoring, gradient checkpointing.
- Sequential trajectory generation (one at a time in VRAM), parallel scoring.
- Must manage three model states: pi_theta (policy, trainable), pi_ref (frozen
  reference), and r_theta (BTRM reward model, frozen during DDGRPO).

Estimated effort: ~200 lines. Significant integration complexity.

### Tier 6: Orchestration Script (depends on Tier 5)

**4.6 `scripts_ii/run_ddgrpo_training.py` — The run script**
- Loads BTRM checkpoint (reward model)
- Initializes policy LoRA adapter (separate from BTRM adapter)
- Saves reference policy state (initial weights)
- Runs DDGRPO training loop
- Validation: render exemplars, track reward statistics

Estimated effort: ~300 lines (following the pattern of
`run_reward_validated_training.py`).

### Total Estimated Effort

| Tier | Component | Lines | Difficulty |
|------|-----------|-------|------------|
| 1 | policy_math.py | ~50 | Easy |
| 2 | solver.py (SDE) | ~40 | Easy |
| 3 | rollout.py (stochastic) | ~30 | Easy |
| 4 | ddgrpo_loss.py | ~80 | Medium |
| 5 | ddgrpo_training.py | ~200 | Hard |
| 6 | run script | ~300 | Medium |
| **Total** | | **~700** | |

---

## 5. Risk Assessment

### 5.1 What's Easy

- **Group advantage normalization**: 5 lines of torch math. Zero risk.
- **LogSNR step weighting**: The weight function already exists
  (`logsnr_sampling_weight`). Applying it per-step is trivial.
- **Multi-head reward composition**: A weighted sum of scalar scores. Design
  decision, not engineering challenge.
- **Stochastic solver**: Adding `+ eta_t * z` to each Euler step is
  mechanically simple. The eta_t schedule is a hyperparameter.

### 5.2 What's Medium Difficulty

- **Per-step log-probability computation**: The math is exact (Gaussian
  log-density), but numerical precision matters. `eta_t^2` can be very small
  near sigma=0 (clean images), causing `1 / (2 * eta_t^2)` to explode. The
  spec (Section 5.3) notes this must be computed in FP32. The existing FP8
  forward pass produces BF16 outputs, so `mu_theta` is BF16. The log-prob
  subtraction `||x_{t-1} - mu_theta||^2` can lose precision when both are
  BF16. Upcasting to FP32 for the log-prob math is mandatory.

- **Reference policy management**: The reference is the base model with
  adapter scale=0. `BTRMCompoundModel.set_adapter_scale(0.0)` achieves this.
  But during DDGRPO, we need BOTH the reference forward (scale=0, no_grad) and
  the policy forward (scale=1.0, with grad) at each step. The frozen
  `compute_reinforce_step()` solves this with two sequential B=1 passes. This
  doubles the forward passes per step (from T to 2T per trajectory).

- **KL estimator numerical stability**: `exp(x) - x - 1` is well-conditioned
  near x=0 but can overflow for large |x|. In practice, the importance ratios
  should be near 1.0 (the clipping bounds them), so x should be small. But
  gradient checkpointing can amplify numerical noise through re-computation.

### 5.3 What's Hard

- **VRAM budget**: Generating K trajectories per prompt with 30 steps each
  requires K*30 forward passes for generation, plus K*30 forward passes for
  log-prob recomputation during PPO epochs. With K=4 and T=30, that is 120+120
  = 240 backbone forwards per optimizer step. On an RTX 4090 with 24GB VRAM,
  each forward uses ~8GB compiled. Sequential generation (one trajectory at a
  time) is mandatory. The total wall time per DDGRPO step would be
  240 * 0.7s = 168s (vs ~10-20s for a BTRM step).

- **Gradient flow through stochastic rollout**: The SDE solver introduces
  sampled noise z_t at each step. For on-policy DDGRPO (fresh trajectories,
  no trajectory reuse), the importance ratios are identically 1 and there is
  no need to differentiate through the solver. But for multi-epoch DDGRPO
  (PPO-style, reusing trajectories), the log-probabilities must be
  recomputed under the CURRENT policy, requiring differentiable forward
  passes. This means gradient checkpointing across 30 layers * 30 steps =
  900 checkpointed layer calls per trajectory. Peak VRAM will be dominated
  by the checkpoint replay.

- **Interaction with FlexAttention packing**: DDGRPO trajectory generation
  is per-image (each trajectory produces one image). But reward scoring can
  use batch packing (score K final images in one packed forward). The packing
  infrastructure exists and is validated. The risk is in managing the
  lifecycle: trajectory generation is sequential with full autograd,
  reward scoring is batched with no_grad.

- **Interaction with per-layer compilation**: The `BTRMCompoundModel` compiles
  individual transformer layers for gradient-checkpointed training. This
  works for BTRM (one image per forward, or packed multi-image with block
  masks). For DDGRPO, the forward pass structure is different: two sequential
  B=1 passes (reference + policy) at each step, 30 steps per trajectory,
  K trajectories. The compiled graph may break on the first DDGRPO iteration
  due to different tensor shapes or control flow. The current run script
  (`run_reward_validated_training.py`) already disables per-layer compilation
  (line 526: `compile_layers=False`) due to torch 2.10 issues with
  `gradient_checkpointing + retain_graph`. DDGRPO will likely face the same
  issue.

- **Spot instance interruption**: A single DDGRPO step takes ~3 minutes.
  The streaming-write requirement (CLAUDE.md: "No phase may buffer >5 min of
  unwritten compute") is satisfied per step but K trajectories per step are
  generated sequentially. If the process is killed after generating 3 of 4
  trajectories, those 3 trajectories are lost (no partial-step persistence).
  Adding per-trajectory persistence within a DDGRPO step adds significant
  complexity.

### 5.4 What Might Break

- **Deterministic solver regression**: Adding a stochastic solver must not
  affect the existing deterministic `euler_solve()`. The stochastic variant
  should be a separate function, not a flag on the existing one. The
  `@torch.inference_mode()` decorator on `euler_solve()` is correct for
  inference but incompatible with training -- the SDE solver must NOT have
  this decorator.

- **LoRA adapter collision**: BTRM uses a "rtheta" adapter. DDGRPO would train
  a separate "ptheta" adapter (the policy). Both cannot be active simultaneously
  on the same backbone. The frozen `scripts/train.py` already handles this by
  toggling adapter scales. In `src_ii/`, the `BTRMCompoundModel` manages one
  adapter. A DDGRPO compound model would need to manage two (or use a separate
  backbone for the reward model, doubling VRAM).

- **Reward model stationarity**: The BTRM is trained on static trajectories.
  During DDGRPO, the policy changes, producing trajectories that are
  out-of-distribution for the BTRM. The reward model's accuracy on policy-
  generated images is unvalidated. This is the "reward hacking" risk that
  the KL penalty is designed to mitigate.

---

## Appendix: File Reference

| File | Role | DDGRPO Relevance |
|------|------|-----------------|
| `src_ii/btrm_training.py` | BTRM training loop | Reward model training (Phase B). Not DDGRPO. |
| `src_ii/btrm_model.py` | BTRMCompoundModel | Reward scoring infrastructure. Reusable for DDGRPO reward. |
| `src_ii/pair_sampler.py` | Pair sampling for BTRM | Not applicable to DDGRPO (no pairs in policy optimization). |
| `src_ii/flops_sampling.py` | Resolution-tier FLOPS | Reusable for DDGRPO trajectory generation budget. |
| `src_ii/sigma_schedule.py` | Sigma schedule construction | Directly reusable for DDGRPO trajectory generation. |
| `src_ii/forward_packed.py` | FlexAttention packing | Reusable for batch reward scoring of K final images. |
| `src_ii/solver.py` | Deterministic Euler ODE | Needs SDE variant for DDGRPO. |
| `src_ii/rollout.py` | Deterministic rollout | Needs stochastic variant for DDGRPO. |
| `src_ii/server.py` | FastAPI server | Has `/accumulate_policy_gradients` endpoint (delegates to frozen code). |
| `src_ii/http_client.py` | HTTP client | Has `accumulate_policy_gradients()` method. |
| `src_ii/server_models.py` | Pydantic models | Has `AccumulatePolicyGradientsRequest` with advantage field. |
| `src/futudiffu/policy_loss.py` | DRGRPO loss (FROZEN) | Reference implementation. Must be reimplemented in src_ii/. |
| `src/futudiffu/training_utils.py` | REINFORCE step (FROZEN) | Reference implementation. Must be reimplemented in src_ii/. |
| `scripts/train.py` | Policy optimization (FROZEN) | Reference orchestration. Must be reimplemented in scripts_ii/. |
| `scripts_ii/run_reward_validated_training.py` | BTRM training script | Template for DDGRPO script structure. |

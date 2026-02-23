# RLAIF Policy Gradients: DDPO Paper Review and Implementation Audit

Results of a code review of `src_ii/policy_step.py`, `src_ii/ddreinforce.py`,
`src_ii/reward_env.py`, and `scripts_ii/run_ddgrpo.py`, cross-referenced against
the DDPO paper (Black et al. 2023).

---

## 1. Reference Material

The DDPO paper source (Black et al. 2023, "Training Diffusion Models with
Reinforcement Learning", arXiv:2305.13301) is in `ref_papers/text/` in this
repository. Key files:

- `20-preliminaries.tex` -- MDP formalization, Eq. 2 (isotropic Gaussian sampler)
- `30-method.tex` -- DDPO_SF (REINFORCE) and DDPO_IS (PPO clipping), algorithm derivation
- `80-appendix.tex` -- hyperparameter table, clip range = 1e-4, overoptimization analysis

The paper source is gitignored. To re-download:

```bash
wget "https://arxiv.org/e-print/2305.13301" -O ref_papers/ddpo_source.tar.gz
tar xzf ref_papers/ddpo_source.tar.gz -C ref_papers/
```

---

## 2. What DDPO Does (from the paper)

The paper maps the denoising process to a multi-step MDP where each denoising
step is one action:

- State: s_t = (context c, timestep t, noisy image x_t)
- Action: a_t = x_{t-1} (the denoised output of one step)
- Policy: pi(a_t | s_t) = N(x_{t-1} ; mu_theta(x_t, c, t), sigma_t^2 I)
- Reward: r(x_0, c) at the terminal step only; 0 at all intermediate steps

The per-step log-probability is the raw Gaussian log-density:

    log p_theta(x_{t-1} | x_t, c) = -||x_{t-1} - mu_theta||^2 / (2 * sigma_t^2) + const

The constant -(d/2)log(2*pi*sigma_t^2) is independent of theta and drops from
all gradient expressions. The log-prob is NOT normalized by dimension d.
The denominator is sigma_t^2, the DDPM reverse process variance.

Two gradient estimators:

1. **DDPO_SF** (Eq. 5, REINFORCE / score function): On-policy only. One gradient
   update per round of data collection. Sum of grad_theta(log p_theta) * r(x_0, c)
   over all T steps. No importance weights.

2. **DDPO_IS** (Eq. 6, PPO-style): Off-policy via importance sampling. Per-step
   importance ratio rho_t = p_theta / p_{theta_old}. PPO clipping with clip range
   1e-4 (extremely tight -- see Appendix D hyperparameter table). Multiple gradient
   updates per data collection round (4 minibatches of 64 from 256 samples).

Key design choices in the paper:

- **No KL regularization.** The paper explicitly states this (Appendix C, comparison
  with DPOK): "Unlike DPOK, we do not employ KL regularization." Drift prevention
  relies entirely on PPO clipping (for DDPO_IS) or early stopping.
- **Equal timestep weighting.** The policy gradient sums over all T steps with
  uniform weight. No logSNR reweighting. No step importance sampling.
- **SDE rollout required.** A deterministic ODE step x_{t-1} = f_theta(x_t) is a
  Dirac delta with no density and no score function. REINFORCE requires a proper
  probability distribution at each step, which the Gaussian noise provides.
- **Reward normalization.** Per-prompt running mean/std tracked independently for
  each prompt. This is a variance-reduction baseline, not a trust region mechanism.
- **CFG training.** The guided epsilon-prediction is used during both training and
  sampling. Training on the conditional objective alone causes performance
  degradation after the first finetuning round.

---

## 3. What Our Implementation Should Do (design decisions)

### Denominator: eta_t^2, not sigma_t^2

The DDPO paper uses sigma_t^2 because DDPM reverse processes have intrinsic
stochasticity with variance sigma_t^2 = beta_t. Our model is flow-matching, not
DDPM. The deterministic Euler ODE has no intrinsic stochasticity. We inject
Gaussian noise with scale eta_t to create a proper density:

    pi_theta(x_{t-1} | x_t, c) = N(mu_theta, eta_t^2 I)
    eta_t = eta_scale * |sigma_{t-1} - sigma_t|

The log-prob denominator is eta_t^2 (the injected noise variance), NOT sigma_t^2
(the noise schedule level). This is the correct adaptation of DDPO Eq. 2 to
flow-matching. `ddreinforce.py` gets this right. `policy_step.py` also uses
eta_t^2 via the `step_log_prob` call.

### No KL regularization

The KL penalty term (beta * unbiased_kl) was imported from DeepSeek-V3/GRPO and
applied to Gaussian transition kernels. The KL between N(mu_theta, eta^2 I) and
N(mu_ref, eta^2 I) is ||mu_theta - mu_ref||^2 / (2 * eta_t^2). When eta_t is
small (near-clean steps), this divides by ~1e-8 and produces infinity. The KL
penalty was never part of the DDPO paper. It is not in our spec. Remove it
entirely.

The paper's overoptimization analysis (Appendix A) explicitly notes that KL
regularization "may be empirically equivalent to early stopping" and does not
use it.

### No logSNR step weighting in the policy gradient

logSNR weighting exists in the BTRM training pipeline, where it controls which
sigma regimes the reward model emphasizes during pairwise ranking loss computation.
The policy gradient should weight all sparse steps equally, per the paper (Eq. 5
sums uniformly over t=0..T). The `w_t = logsnr_step_weight(sigma_t)` multiplier
in `policy_step.py` line 180 is incorrect for the policy gradient loss.

### GRPO-style group advantages

Z-scored within a group of K trajectories for the same prompt. This IS in our
spec and correctly implemented in `ddreinforce.py`:

    group_advantages(rewards) = (rewards - mean(rewards)) / (std(rewards) + eps)

Returns zeros if K < 2.

### PPO clipping (DDPO_IS variant)

Use per-step clipping, not trajectory-level clipping. The paper uses clip range
1e-4. `ddreinforce.py` implements `clipped_surrogate_loss()` correctly as
per-step PPO with configurable clip_eps. The default clip_eps=0.2 in
`ddreinforce.py` is too loose for diffusion policy optimization -- the paper
found 1e-4 necessary.

### Sparse step sampling

We use 3-5 of 30 steps per rollout for gradient computation. This is a compute
optimization. The REINFORCE estimator is unbiased with any subset of steps
because rewards are terminal (only at t=0) and each step's gradient contribution
is independent. The `run_ddgrpo.py` script constructs the sparse step set
correctly (lines 301-308).

### eta_t floor

Skip steps where eta_t < 1e-6. Below this threshold the step is effectively
deterministic. The log-prob gradient contains 1/eta_t^2, which explodes when
eta_t approaches zero. The current floor in `policy_step.py` is 1e-12 (line
132), which is far too small -- at eta_t = 1e-7, the denominator is 1e-14
and gradients are numerically degenerate in float32.

---

## 4. Low-Rank Adapters as Implicit Trust Region

### Proof sketch

A LoRA adapter with rank r constrains the weight update to a rank-r subspace.
For a linear layer W in R^{m x n}:

- **Full-rank update**: delta_W can be any matrix in R^{m x n}. This has m*n
  degrees of freedom.
- **LoRA update**: delta_W = B @ A where B in R^{m x r}, A in R^{r x n}. This
  has r*(m+n) degrees of freedom, with r << min(m,n).

**Frobenius norm bound on the weight perturbation:**

    ||delta_W||_F = ||B @ A||_F <= ||B||_F * ||A||_F

With init_b_std = 0.01 and A initialized from Kaiming/He normal:

    ||B||_F ~ 0.01 * sqrt(m * r)
    ||A||_F ~ sqrt(n)
    ||delta_W||_F ~ 0.01 * sqrt(m * r * n)

Compare to the pretrained weight magnitude ||W||_F ~ sqrt(m * n). The ratio:

    ||delta_W||_F / ||W||_F ~ 0.01 * sqrt(r)

For rank r = 8: the LoRA update is bounded at ~2.8% of the pretrained weight
norm at initialization, and grows slowly under gradient descent because only
r*(m+n) parameters are being optimized.

**Functional change bound:**

For input x with ||x|| bounded:

    ||delta_W @ x|| <= ||delta_W||_op * ||x|| <= ||delta_W||_F * ||x||

The operator norm ||delta_W||_op is the largest singular value of B @ A. Since
rank(B @ A) <= r, there are at most r nonzero singular values. The operator norm
is bounded by:

    ||delta_W||_op <= min(||B||_op * ||A||_op, ||delta_W||_F)

**KL divergence bound:**

The KL divergence between old and new policy at step t depends on the change in
predicted mean mu_theta. For isotropic Gaussian policies with shared variance
eta_t^2:

    KL(pi_new || pi_old) = ||mu_new - mu_old||^2 / (2 * eta_t^2)

The change in mu is a function of delta_W applied through the network. For a
single linear layer, ||delta_mu|| <= C * ||delta_W||_F * ||x|| for some
architecture-dependent constant C. Therefore:

    KL <= C^2 * ||delta_W||_F^2 * ||x||^2 / (2 * eta_t^2)

Since ||delta_W||_F^2 is bounded by r * (sum of squared LoRA params), the KL is
bounded by a function that scales linearly with rank r.

**Concrete numbers for our architecture:**

- d_model = 3840, rank r = 8, alpha = 16, init_b_std = 0.01
- Per-layer update lives in an 8-dimensional subspace of R^{3840 x 3840}
- The full parameter space has 3840^2 = 14.7M dimensions per layer
- The LoRA subspace has 8 * (3840 + 3840) = 61.4K dimensions per layer
- Ratio: 0.42% of the full parameter space

**Comparison to PPO clipping:**

PPO clipping constrains the probability ratio rho_t = pi_new / pi_old to
[1 - eps, 1 + eps]. This bounds the magnitude of the policy change but places
no constraint on its direction. After clipping, the next gradient step can push
in any direction in the full parameter space.

LoRA constrains both magnitude AND direction. The policy update is restricted to
the column space of the LoRA matrices. This is a much tighter constraint:

- PPO clip at eps = 1e-4 constrains one scalar (the probability ratio) per step.
- LoRA rank r = 8 constrains the entire weight update to an 8-dimensional
  subspace per layer, for all steps simultaneously.

**The relationship:**

PPO clip epsilon ~ f(rank, alpha, init_b_std, lr). As rank -> full rank, the
LoRA constraint vanishes and explicit clipping becomes necessary. As rank -> 1,
the policy update is essentially one-dimensional and no additional trust region
is needed. LoRA rank is a trust region parameter that operates in weight space
rather than probability-ratio space.

This means we can use a more relaxed clip range (or use DDPO_SF / pure
REINFORCE with no clipping at all) when training with low-rank adapters. The
LoRA rank already prevents the catastrophic policy drift that clipping was
designed to prevent.

---

## 5. Errors Found in Current Implementation

### policy_step.py

- **Missing imports (NameError at runtime).** `step_log_ratio` and `unbiased_kl`
  are called at lines 176-178 but not imported. The import block (lines 34-37)
  only imports `step_log_prob` and `logsnr_step_weight` from `ddreinforce`. This
  is a hard crash on the first gradient accumulation call.

- **KL term produces infinity.** Even if the imports were fixed, the KL penalty
  computes `unbiased_kl(log_ratio_ref)` where `log_ratio_ref` contains a
  `1/(2*eta_t^2)` factor. For near-clean steps where eta_t ~ 1e-4, this
  produces log-ratios of order 1e8, and `exp(1e8)` is inf. The KL term then
  propagates inf into the loss and produces nan gradients. This is the root
  cause of the `kl=inf, grad=nan` crash.

- **KL penalty is not in the DDPO spec.** The paper does not use KL
  regularization. The `beta * w_t * kl_t.mean()` term at line 181 was
  imported from GRPO (DeepSeek-V3), where KL regularization is applied to
  autoregressive token-level policies with well-behaved categorical
  distributions. Gaussian transition kernels with tiny variance do not have
  well-behaved KL -- the penalty term is numerically degenerate by construction.

- **logSNR step weighting applied to policy gradient.** Line 180:
  `policy_loss = -advantage * w_t * lp_theta.mean()`. The `w_t` factor comes
  from `logsnr_step_weight(sigma_t)`, which is a BTRM training concept for
  controlling reward model emphasis across noise levels. The DDPO paper uses
  uniform step weighting (sum over all T with weight 1).

- **eta_t floor too small.** Line 132: `if eta_t < 1e-12: continue`. Should
  be 1e-6. At eta_t = 1e-7, the denominator 2*eta_t^2 = 2e-14. In float32,
  squared differences divided by 2e-14 overflow to inf. The floor must be set
  high enough that 1/(2*eta_t^2) is representable without overflow.

- **multiplier parameter creates inconsistency.** Line 138:
  `timestep = sigma_t * multiplier`. The model sees sigma_t * multiplier as
  its timestep conditioning, but the velocity computation (line 153:
  `d_theta = (x_t - denoised_theta) / sigma_t`) uses raw sigma_t as the
  denominator. If multiplier != 1.0, the model predicts a denoised image
  conditioned on the wrong noise level, and the velocity is computed with the
  right noise level, creating a mismatch. Currently always 1.0 but is a
  latent bug waiting for someone to pass multiplier=0.5.

### ddreinforce.py

- The `ddreinforce_loss()` function (line 477) applies `w_t` step weights via
  the `step_weights` parameter. If callers populate this from
  `logsnr_step_weight`, the same logSNR-weighting error propagates. The
  function itself is correct -- the error is in what callers pass for
  `step_weights`. For DDPO-correct behavior, all step weights should be 1.0.

- The `ddreinforce_loss()` and `ddgrpo_loss()` functions both include KL penalty
  terms (`beta * kl.mean() * w_t`). Per the paper, these should not exist.

### run_ddgrpo.py

- **Checkpoints only every 25 iterations** (line 66: `CHECKPOINT_INTERVAL = 25`).
  No optimizer state is saved. No iteration index is persisted. A crash at
  iteration 24 loses all 24 iterations of gradient computation.

- **No adapter save after every gradient step.** The script relies on the server
  holding adapter state in GPU memory. If the server crashes, all adapter
  progress since the last checkpoint is lost.

---

## 6. Persistence Requirements

The policy optimization script MUST checkpoint after every gradient step:

1. **Adapter weights** (safetensors). One file per iteration. ~500KB at rank 8.
2. **Optimizer state** (safetensors or torch.save). AdamW state for LoRA params.
   ~1.5MB at rank 8 (two momentum buffers per parameter).
3. **Current iteration index.** A single integer in a JSON sidecar or embedded
   in the JSONL metrics stream.
4. **Per-iteration metrics** (JSONL, flushed per line). Already partially
   implemented -- `run_ddgrpo.py` writes metrics JSONL at line 403. But the
   file is opened in append mode without checking for existing content or
   deduplicating on resume.

An earthquake, UPS failure, or spot instance preemption should lose at most ONE
iteration's worth of gradients. The current script buffers adapter weights in
server GPU memory and writes them to disk every 25 iterations. This violates
the streaming-writes-for-spot-instances requirement from CLAUDE.md.

### Pattern to follow

`src_ii/incremental_save.py` provides the correct primitives:

- **TrainingCurveWriter**: Append-only JSONL with per-line flush. Each write is
  durable within one syscall of computation. Handles crash recovery by skipping
  malformed trailing lines on reload.

- **PeriodicSaver**: Interval-gated save trigger with explicit flush. Use with
  interval=1 for adapter weights (every iteration) or interval=5 for optimizer
  state (every 5 iterations -- optimizer state is larger but less critical since
  AdamW recovers quickly from cold-started momentum).

- **atomic_json_save**: Write-to-temp-then-rename for JSON metadata. Prevents
  partial writes from corrupting the iteration index or run config.

### Resume protocol

On startup, the script should:

1. Check for existing JSONL metrics file. Count completed iterations.
2. Load the latest adapter checkpoint (if any) into the server.
3. Load the latest optimizer state (if any) into the optimizer.
4. Resume from iteration N+1 where N is the last completed iteration.

This is the same pattern used by `src_ii/dataset_resumption.py` for trajectory
generation and `src_ii/btrm_training.py` for BTRM training loops.

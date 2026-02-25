# Training/Eval Budget Metaoptimization for RLHF Policy Gradient Methods

## The Question

Given a fixed compute budget C (in GPU-seconds, FLOPS, or dollars), how
should you allocate between:

1. **Training iterations** (weight updates that actually modify the policy)
2. **Per-iteration batch size** (rollouts per step, reducing gradient noise)
3. **Evaluation** (scoring the policy on held-out prompts without backprop)

The naive answer is "maximize batch size per iteration to minimize gradient
noise." This is wrong. The correct answer depends on three cost asymmetries
that are specific to diffusion policy optimization but rhyme with broader
RLHF scaling questions.

## Cost Structure (Measured, RTX 4090, DDGRPO v2/v2b)

| Phase | Cost (s) | % of iteration | Scales with |
|-------|----------|----------------|-------------|
| Rollout (forward, SDE solve) | 30 | 12% | B × K × N_STEPS |
| Scoring (BTRM forward, no_grad) | 0.3 | 0.1% | B × K |
| Gradient accumulation (backward) | 230 | 88% | B × K × N_GRAD_STEPS |

The backward:forward ratio is **6:1**. Every trajectory you add to the batch
costs 6x more in gradient compute than it cost to generate. This ratio is
structural — it comes from differentiating through all 19 denoising steps of
the SDE, each requiring a full DiT forward pass under activation checkpointing.
Sparse step sampling (v1: 5/30 steps) reduces this ratio but also reduces
gradient signal proportionally; v2 abandoned it.

Scoring cost is negligible. The BTRM forward is a single compiled forward pass
at sigma=0 with no graph. At 0.1% of iteration cost, you could score 1000x
more images than you train on without affecting the training budget. This is
the key asymmetry.

## The Three Budgets

### Budget A: Intervention Strength (training iterations × learning rate)

The policy doesn't improve by being measured. It improves by having its weights
updated. Each optimizer step applies a delta to ~14.7M LoRA parameters. The
magnitude of this delta is `lr × clipped_gradient`. The *direction* is
determined by the REINFORCE loss over the batch.

At lr=2e-5 (v2), the per-iteration weight delta has norm ~0.1-0.15. At
lr=6e-5 (v2b), ~0.1-0.17. Sign agreement between successive deltas is
65-89%, meaning the optimizer is mostly pushing in the same direction but
with 10-35% of parameters disagreeing per step.

To move the policy meaningfully, you need many iterations. A single step at
any LR doesn't change the output distribution enough to be detectable above
the seed noise floor (within-group std = 0.18). You need the *cumulative*
effect of many coherent steps.

**Implication**: At fixed compute budget, reducing B×K to get more iterations
increases intervention strength, even if each iteration has noisier gradients.

### Budget B: Per-Iteration Signal Quality (batch size × K)

Larger batches reduce gradient variance. With K rollouts per prompt, the
advantage estimate has variance proportional to 1/K. With B prompts, you
cover more of the prompt distribution per iteration.

But gradient noise isn't just statistical noise that averages away. The
REINFORCE gradient for diffusion policies has a specific structure: it's a
sum over denoising steps, each with its own sigma-dependent noise scale.
Steps near sigma=0 (clean images) have enormous `1/eta^2` weighting; steps
at high sigma have near-zero signal. The gradient is *dominated* by a few
clean-regime steps regardless of batch size.

Additionally, our variance decomposition shows:
- **72% of reward variance is between-prompt** (the BTRM has fixed preferences)
- **28% is within-prompt** (seed + resolution + training noise)

Increasing K (more rollouts per prompt) only reduces the 28% component.
Increasing B (more prompts) averages over the 72% component but doesn't
reduce it — you're averaging over a *real* signal (prompt-specific quality
differences), not noise.

**Implication**: Increasing batch size has rapidly diminishing returns.
Doubling B×K from 6 to 12 halves gradient variance but doubles iteration
cost. The same compute buys either 50 iterations at B×K=12 or 100 iterations
at B×K=6. The 100-iteration run has more intervention strength.

### Budget C: Confirmatory Power (eval rollouts on held-out prompts)

Here's where the cost asymmetry matters most. Training a trajectory costs
~270s (30s forward + 230s backward + 0.3s scoring). Evaluating a trajectory
costs ~30.3s (30s forward + 0.3s scoring). **Eval is 9x cheaper than train.**

The quality of a policy is not "how high is the reward on the 12 prompts it
was trained on." It's "how does it perform on the distribution of prompts it
will encounter at deployment." This is distributional generalization, and
you cannot measure it without an eval budget.

Our power analysis shows:

| Detectable delta | Rollouts per prompt needed | At K=3, groups needed |
|-----------------|---------------------------|----------------------|
| 0.25 (current) | 13 | 4 |
| 0.10 | 83 | 28 |
| 0.05 | 329 | 110 |

At current training scale (B=2, K=3, 50 iters), we get ~12.5 rollouts per
prompt per half-run. We can detect reward changes of ±0.25 — barely enough
to distinguish "portrait" from "cityscape" prompt families, not enough to
measure whether training improves any specific prompt.

But if we redirect compute from batch size to eval, the arithmetic is
different. One iteration of gradient compute (230s backward) buys
**7.6 eval trajectories** (230s / 30.3s). Ten iterations of gradient compute
redirected to eval buys **76 eval trajectories per prompt** if spread across
12 prompts, which detects reward changes of ±0.11.

## The Metaoptimization

Define:
- `T` = total compute budget (GPU-seconds)
- `c_train` = cost per training trajectory (270s: forward + backward + score)
- `c_eval` = cost per eval trajectory (30.3s: forward + score)
- `n_iter` = number of training iterations
- `b` = batch size per iteration (B × K)
- `n_eval` = number of eval trajectories
- `n_prompts` = number of distinct prompts (training + held-out)

Budget constraint: `n_iter × b × c_train + n_eval × c_eval = T`

The policy's actual quality after training depends on:
1. **Intervention strength**: roughly `f(n_iter, lr, grad_quality(b))`
2. **We can't observe this directly** — we observe reward, which is a noisy
   proxy

Our *confidence* in whether the policy improved depends on:
3. **Eval sample size**: `n_eval / n_prompts` per prompt

The optimization is NOT `max(reward)` subject to `budget = T`. It's:

```
max P(policy actually improved | observed metrics)
subject to: n_iter × b × c_train + n_eval × c_eval = T
```

This is a different objective. It values *knowing* that you improved over
*having improved by a larger but unmeasured amount*.

## Why This Is Not the Chinchilla Problem

The Chinchilla scaling law asks: given compute C, what's the optimal
model size N and dataset size D? It's answered by: `N* ∝ C^0.5`,
`D* ∝ C^0.5` (roughly). The loss is a smooth function of (N, D) and
you're minimizing it.

The RLHF budget problem is different because:

1. **The reward is not the loss.** The loss is REINFORCE; the reward is what
   the BTRM outputs. They're connected by the REINFORCE gradient formula
   but not by a smooth scaling law. More compute doesn't smoothly reduce
   reward estimation error — it reduces *gradient noise*, which may or may
   not translate to better policy quality depending on the reward landscape's
   curvature.

2. **Eval is asymmetrically cheap.** In LLM pretraining, eval costs the same
   as train per token (one forward pass, no backward). In diffusion RLHF,
   eval is 9x cheaper because training requires differentiating through 19
   denoising steps. This asymmetry means the optimal eval budget is
   proportionally *larger* than in pretraining.

3. **The intervention is discrete.** You don't get half a weight update.
   Each optimizer step is a discrete event that either moves the policy in
   a useful direction or doesn't. Below some minimum iteration count,
   the policy hasn't changed enough to be distinguishable from initialization
   regardless of how precisely you measure it.

4. **Generalization is the metric, not training reward.** A policy that
   scores +0.5 on "portrait" prompts and -0.7 on "cityscape" prompts has
   average reward -0.1 but has *learned something*. Whether that something
   generalizes to unseen prompts requires eval on unseen prompts.

## Practical Regimes

### Regime 1: Exploration (current, 1× 4090)

Budget: ~4 GPU-hours per 50-iteration run.

Optimal allocation: **minimize batch size, maximize iterations, eval is
a post-hoc separate job.**

At B=2, K=3 we spend 88% of compute on backward passes. Each backward
pass produces a gradient update. We can't detect per-prompt improvement
but we can track sign agreement (a reward-independent metric) and aggregate
reward trends. The goal is not "measure whether we improved" but "iterate
the weights enough times that improvement is plausible, then check with
a separate eval."

Eval at this scale should be a separate script that loads the final adapter,
generates images for 50+ prompts at multiple resolutions, and scores them.
Cost: 50 prompts × 4 seeds × 30s = 100 minutes. This is 42% of the training
budget but provides definitive per-prompt quality measurement.

### Regime 2: Confirmation (2-8× H100)

Budget: ~8-32 GPU-hours per run.

Optimal allocation: **moderate batch size (B=4-8, K=4), interleave eval
checkpoints.**

At this scale, you can afford to score held-out prompts every N iterations
without materially reducing training iterations. Eval checkpoint every 10
iterations: 10 × 30 eval trajectories = 300 trajectories, cost = 9000s on
one GPU while the other 7 train. The eval GPU is 97% idle at current
per-trajectory scoring cost (0.3s). This is absurdly underutilized.

Better: the eval GPU runs continuous rollout generation for held-out prompts,
accumulating a running reward estimate that gets more precise as training
progresses. The training GPUs never pause. The eval budget is essentially
*free* because scoring is 0.1% of compute and the eval GPU would otherwise
be waiting for gradient synchronization.

### Regime 3: Production (16+ H100, TPU pods)

Budget: 100+ GPU-hours.

Optimal allocation: **the batch size question dissolves.**

At B=24, K=8, every iteration covers all 12 training prompts twice with 8
rollouts each. Per-prompt reward signal is statistically significant within
a single iteration (24 rollouts/prompt, detectable delta = 0.19). You no
longer need to choose between training and eval — you can eval at full
precision every iteration for free.

The real question at this scale is: **how many prompts should you train on?**
12 is clearly too few. With 192 rollouts per iteration, you should be
sampling from 100+ prompts to avoid overfitting the policy to BTRM
preferences on a narrow prompt set. The 72% between-prompt variance
becomes a feature (more signal) rather than a confound (more noise) when
you have enough prompts to estimate the prompt-conditional reward function.

## The Odd Implication

The optimal budget allocation is not continuous. There are phase transitions:

1. **Below ~20 iterations**: the policy hasn't moved enough to be
   distinguishable from initialization. All batch size spent here is wasted.
   Minimum viable run length comes first.

2. **At 20-100 iterations**: each additional iteration has high marginal
   value (the policy is in the regime where sign agreement is building and
   the optimizer is finding coherent directions). Batch size trades off
   against iterations roughly 1:1 in value.

3. **Above ~100 iterations (at current LR)**: sign agreement plateaus
   near 90%+. Additional iterations may be pushing along directions that
   are converging. The marginal value of iterations decreases. Batch size
   (for gradient quality) and eval (for confirmation) become relatively
   more valuable.

4. **The eval transition**: there exists a threshold iteration count where
   the expected information value of one eval trajectory exceeds the
   expected policy improvement from one training trajectory. Below this
   threshold, train. Above it, eval. At 9:1 cost asymmetry, this threshold
   is surprisingly early — if each training trajectory improves the policy
   by less than 1/9th of the eval trajectory's information value, you
   should be evaluating instead.

The sign agreement tracker gives us a direct measurement of where we are
in this progression. When sign agreement is climbing (iterations 1-20 in
v2b), we're in the "more iterations" regime. When it plateaus or begins
to oscillate, we're approaching the eval transition.

## Numbers for Our Codebase

At current v2b configuration (B=2, K=3, lr=6e-5, 50 iters):

| Allocation | GPU-seconds | Iterations | Eval rollouts | Detectable delta |
|-----------|-------------|------------|---------------|-----------------|
| All train | 13,500 | 50 | 0 | unmeasurable |
| 90/10 train/eval | 12,150 | 45 | 45/prompt | 0.135 |
| 80/20 train/eval | 10,800 | 40 | 89/prompt | 0.096 |
| 70/30 train/eval | 9,450 | 35 | 134/prompt | 0.078 |
| 50/50 train/eval | 6,750 | 25 | 223/prompt | 0.061 |

At the current within-prompt std of 0.23, a 20% eval allocation buys
detectable-delta < 0.10. This means: run 40 training iterations, then
spend the remaining compute scoring the trained policy on all 12 prompts
at diverse resolutions and seeds. You'll be able to say, with 80% power,
whether each prompt improved or degraded by more than 0.10 reward units.

The 50 training iterations at 0% eval gives you a policy that *might* be
better but you can't confirm it. The 40 training iterations at 20% eval
gives you a policy that's *almost as trained* and you *know* whether it
worked. Five lost iterations are cheap; unfounded confidence is expensive.

## Coda: What Sign Agreement Tells Us About This

Sign agreement is the only metric in our current setup that doesn't
require per-prompt stratification, doesn't require eval budget, and
directly measures "are the weight updates coherent?"

At 89% sign agreement (v2b iter 24), the optimizer is spending its entire
budget pushing 89% of parameters in a consistent direction. The remaining
11% are bouncing — these are either noise or subspace rotation (converged
dimensions being replaced by newly-active dimensions).

If sign agreement starts to decline under the higher LR, that's evidence
the optimizer is in the "eval transition" regime: additional training
iterations are rotating through subspaces rather than making progress, and
the compute would be better spent measuring what the current policy does.

If sign agreement continues climbing toward 95%+, the optimizer is still
finding coherent directions and more iterations are warranted. The eval
budget should be spent *after* sign agreement plateaus, not during the
climb.

This makes sign agreement a cheap, online proxy for the train/eval
allocation decision. No separate eval pass needed — just watch the
number and redirect compute when it stops climbing.

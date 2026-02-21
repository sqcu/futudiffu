# r_theta Learning Rate Sweep Analysis

## Setup

Two learning rate probes trained the BTRM reward model's r_theta LoRA adapter
for 100 macrobatches each on the same 50-trajectory dataset:

- **lr=1e-02** (high): aggressive learning rate
- **lr=1e-03** (low): conservative learning rate

Both used identical hyperparameters otherwise: rank-8 LoRA, alpha=16,
batch_size=32, logsq_weight=0.1, Bradley-Terry pairwise ranking loss with
logsquare regularizer. The loss function is `bt_loss + 0.1 * logsq_loss`.

## Loss Dynamics: lr=1e-03 Wins Unambiguously

The down-and-to-the-left phase-space plots tell a clean story. In these plots,
the X-axis is the current EMA loss level and the Y-axis is the instantaneous
rate of descent (d_loss/d_step). A probe that is simultaneously lower on X
(lower loss) and more negative on Y (still descending) dominates.

**lr=1e-03 dominates lr=1e-02 on both axes, at both EMA smoothing levels.**

| Metric | lr=1e-02 | lr=1e-03 |
|--------|----------|----------|
| EMA(0.1) final loss | 0.681 | 0.342 |
| EMA(0.3) final loss | 0.743 | 0.341 |
| Mean d_loss/d_step (0.1) | +0.0017 | -0.0042 |
| Mean d_loss/d_step (0.3) | +0.0023 | -0.0042 |

The lr=1e-02 probe has a *positive* mean descent rate, meaning its EMA-smoothed
loss is, on average, drifting upward over 100 steps. It descends early (reaching
a minimum around step 47-50), then oscillates at a high loss floor (~0.5-0.7)
for the remaining 50 steps with no further improvement.

The lr=1e-03 probe maintains a *negative* descent rate throughout, reaching its
EMA minimum at step 94 and still descending at step 99. It has not yet
converged. This is the most important finding: lr=1e-03 is still making
progress at step 100.

## Why lr=1e-02 Plateaus

The loss curve for lr=1e-02 shows violent oscillation. Raw loss swings from
0.08 to 2.45 in the space of a few steps (e.g., step 57: 0.19, step 58: 2.45).
The EMA never settles below ~0.5 because the optimizer overshoots minima faster
than it converges to them. The gradient norm early spike at step 1 (316.4) is a
classic sign of a learning rate too high for the loss landscape curvature.

In contrast, lr=1e-03 has lower raw variance. Its occasional large losses
(step 59: 1.42, step 60: 1.05) are less extreme and are followed by recovery.
The EMA descends steadily.

## Gradient Norm Comparison

Both probes exhibit gradient norm spikes, but in different patterns:

- **lr=1e-02**: One giant spike at step 1 (316), then rapid decay to
  single-digit norms by step 30. The grad norm settles to ~1-4 for the last
  30 steps, with occasional bumps to 25-28. This low-magnitude gradient
  combined with high learning rate explains the oscillation: each step is
  large in parameter space relative to the local curvature.

- **lr=1e-03**: Recurrent spikes throughout training (steps 19: 294, 30: 285,
  42: 231, 80: 235, 68: 99, 79: 81, etc.). These spikes do NOT cause the
  loss to blow up, because the learning rate is low enough that even a
  large gradient norm produces a bounded parameter update. The grad norm
  remains elevated (mean=30.2 vs 9.8 for 1e-02) because the model has not
  yet settled into a flat minimum; it is still navigating high-curvature
  regions of the loss landscape. This is consistent with active learning.

## The Step-Time Anomaly at lr=1e-03

The step time plot reveals a dramatic discontinuity for lr=1e-03 starting
at step 66:

- Steps 0-65: ~4.5-4.6s per step (consistent with lr=1e-02)
- Steps 66-99: ~65-70s per step (15.3x slowdown)

This is almost certainly a **torch.compile recompilation event**. At step 66,
the accumulated weight changes from 66 gradient updates push some tensor
through a shape or value boundary that triggers a guard failure in the compiled
graph, causing a full recompilation. The lr=1e-02 probe does not hit this
because its larger learning rate causes the optimizer to settle into a
parameter regime earlier (smaller effective updates after early steps).

Total wall-clock cost: lr=1e-02 took 572s (9.5 min), lr=1e-03 took 2277s
(38 min). The 4x difference is almost entirely attributable to the
recompilation regime in the last 34 steps. Without it, lr=1e-03 would
have taken ~450s. This is an infrastructure problem, not a training problem;
the compile guards need investigation.

## Accuracy

Both probes achieve 95% pinkify accuracy in the last 20 steps (pinkify is
the easier discrimination task: SDPA vs SageAttention INT8 QK).

The difference is on **thisnotthat** (step count discrimination: 30 vs 8-22):

- lr=1e-02: 45% accuracy (last 20 steps) -- essentially random
- lr=1e-03: 80% accuracy (last 20 steps) -- learning the harder task

This is the most practically important result. The reward model needs to
discriminate on both heads to produce useful training signal for policy
optimization. lr=1e-02 cannot learn the harder head.

## Recommendations

1. **Use lr=1e-03 (or explore 3e-03 as a middle ground).** lr=1e-02 is
   too aggressive for this loss landscape. lr=1e-03 has not converged at
   100 steps, suggesting running to 200-300 steps would yield further
   improvement.

2. **Investigate the compile recompilation.** The 15x slowdown at step 66
   for lr=1e-03 is a fixed cost that can likely be eliminated by
   pre-warming the relevant compiled graph shapes or by adjusting compile
   guards. This would bring lr=1e-03's wall-clock time down to parity with
   lr=1e-02.

3. **Extend the sweep.** lr=1e-03 is still descending at step 100. Running
   to 300+ steps would reveal whether it converges to a loss floor or
   continues improving. The lr=1e-04 probe (present in the sweep config but
   with no output) should also be run.

4. **Gradient clipping.** The lr=1e-03 probe has 6 gradient norm spikes
   above 100. Clipping at 50 or 100 would limit the damage from worst-case
   batches without affecting the typical gradient step (median grad norm is
   ~8-10).

## Data Artifacts

All analysis outputs are persisted to `rtheta_sweep_output/plots/`:

- `loss_vs_step.png`: Raw + EMA loss curves for both probes
- `grad_norm_vs_step.png`: Gradient norm time series
- `time_vs_step.png`: Step time showing the recompilation anomaly
- `down_and_left_alpha01.png`: Phase-space plot (EMA alpha=0.1)
- `down_and_left_alpha03.png`: Phase-space plot (EMA alpha=0.3)
- `summary.json`: Machine-readable statistics for downstream consumption

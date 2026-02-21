# Directive: Remove the Logsquare Regularizer from BTRM Training

**Date:** 2026-02-18
**Author:** Root session
**For:** Subagent assigned to refactor BTRM loss computation

---

## The Problem

The BTRM training loss in this project contains a logsquare regularizer term
that was ported from a different project (`dialogue_yoinker`) where it had a
specific semantic purpose. That purpose does not exist in this project. The
regularizer is actively competing with the pairwise ranking objective, causing
BT loss to barely descend during training (1.3% over 30 steps while the
regularizer term dominates the gradient budget).

## Where the Regularizer Came From

The logsquare regularizer originates in `~/dialogue_yoinker/scripts/train_btrm.py`,
a **text corpus membership classifier**. In that system:

- **Positive samples** are texts that belong to a target corpus (game dialogues
  from Oblivion, Fallout NV, Skyrim)
- **Negative samples** are out-of-domain texts in three tiers (related corpora,
  fiction, webscrape)
- The model outputs a scalar score per text, interpreted as **membership
  probability**
- The logsquare regularizer `log(r² + eps).mean()` is applied to **positive
  scores only**, anchoring them toward |r| ≈ 1
- This creates a calibrated reward scale: positive = score near 1, negative =
  score near 0, with the BT loss handling the ranking

The regularizer made sense there because:
1. There IS a target magnitude (r ≈ 1 for members)
2. It's applied asymmetrically (positives only)
3. Scores have a semantic interpretation (membership probability)

## Why It Doesn't Apply Here

In futudiffu, the BTRM trains on **pairwise image preferences** from pixel-space
reward functions (PINKIFY, THISNOTTHAT). The training data is:

- Two images from different trajectories
- A preference label: image A is pinker than image B (+1), or not (-1)
- Neither image "belongs" to anything — there are no positives and negatives in
  the membership sense, only relative preferences

The scores have no target magnitude. There is no "correct" absolute value for
a pinkify score — only the relative ordering matters. A model that outputs
scores of (0.001, 0.002) for a preferred pair is exactly as correct as one
that outputs (5.0, 6.0).

### What went wrong in the port

Two changes compounded the semantic mismatch:

1. **Applied to ALL scores instead of positive only.** The original applied the
   regularizer to preferred-sample scores, anchoring the "winner" toward |r|≈1.
   The current code concatenates both images' scores and regularizes them
   together. This means the regularizer is actively compressing the score gap
   that the BT loss is trying to widen.

2. **The asymmetry was lost.** Even if applied to preferred scores only, the
   concept of "this score should be near 1" has no semantic basis when scoring
   images for relative pinkness. There is no "membership" to calibrate against.

### The observed effect

Over 30 training steps:
- BT loss: 0.714 → 0.705 (1.3% reduction — almost no ranking improvement)
- logsq_loss: -0.62 → -4.26 (becoming more negative = scores shrinking)
- Total loss: 0.683 → 0.607 (11.1% reduction — mostly driven by regularizer)

The optimizer is spending its gradient budget satisfying the regularizer
(pushing scores toward |r|=1) rather than improving pairwise rankings.

## The Redundancy with soft_tanh_cap

The score unembedder already applies `soft_tanh_cap(logit_cap=10.0)` to all
outputs. This bounds scores to approximately [-10, +10] with a smooth ceiling.
The problem the regularizer was designed to solve in the original system —
unbounded score explosion — cannot happen here because the architectural
constraint already prevents it.

Two magnitude-control mechanisms on the same output is redundant. The tanh cap
is the correct one for a pairwise ranking model (bounds scores without imposing
a target magnitude). The logsquare regularizer is the correct one for a
membership classifier (anchors scores to a calibrated scale). Using both means
the model is simultaneously told "scores can be anywhere in [-10, 10]" (tanh)
and "scores should be near ±1" (logsquare). These constraints conflict.

## What to Do

### Remove

1. Remove the `logsquare_regularizer()` function from wherever it's defined
   (check `src/futudiffu/btrm.py` and `src_ii/btrm_training.py`)
2. Remove the `logsquare_weight` parameter from all training functions
3. Remove the logsq term from the loss computation
4. Remove logsq_loss from metrics logging
5. The total loss becomes just the BT loss: `-log(sigmoid(s_preferred - s_rejected))`

### Keep

1. `soft_tanh_cap(10.0)` in the ScoreUnembedder — this is the correct
   magnitude bound for a pairwise ranking model
2. The BT loss itself — this is the correct objective
3. Any grad clipping — this addresses optimization stability, not score
   magnitude

### Verify

After removing the logsquare term, re-run 30 macrobatches of differentiable
training and check:
- BT loss should descend more aggressively (no longer competing with regularizer)
- Pairwise agreement on the score comparison test should improve
- Gradient norms may change (the regularizer was contributing gradients)

## Reading List for the Implementing Subagent

1. This document (you're reading it)
2. `src_ii/btrm_training.py` — find the loss computation, remove logsquare
3. `src/futudiffu/btrm.py` — check if `logsquare_regularizer` is defined here
4. `scripts_ii/train_pinkify_differentiable.py` — the training script that
   will be re-run after the fix
5. `pinkify_thisnotthat_output/differentiable_run_v2/training_metrics.jsonl` —
   baseline metrics to compare against

## Do NOT

- Add a replacement regularizer. The tanh cap is sufficient.
- Change the BT loss formula. It's correct.
- Modify the ScoreUnembedder architecture. soft_tanh_cap stays.
- Touch anything in `src/futudiffu/` (frozen).

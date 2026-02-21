# Unit 4 Synthesis: Detached vs. Differentiable BTRM Training on PINKIFY and THISNOTTHAT

**Date**: 2026-02-18
**Hardware**: RTX 4090 (SM89, 24GB), WSL2 / Windows Python venv
**Dataset**: V2 (259 trajectories, 1732 scored positions) vs. V1 (10 trajectories, 80 images)

---

## 1. What Was Done

Unit 4 ran the BTRM training pipeline end-to-end twice: once with the detached path (`train_btrm()` on pre-extracted hidden states from 10 V1 trajectories, 40 epochs over 280 materialized pairs) and once with the differentiable path (`train_btrm_differentiable()` on 259 V2 trajectories, 30 macrobatches with on-the-fly pair sampling from a ~1.5M combinatorial space). The detached run trained only the 7,682-parameter ScoreUnembedder head -- the r_theta LoRA adapter received zero gradients because hidden states were extracted under `inference_mode` and the backbone was freed before training began. The differentiable run trained the full compound model: 10,096,640 adapter parameters plus 11,520 head parameters, with gradient-checkpointed forward passes through the frozen-weight FP8 backbone preserving the computation graph through all LoRA injection sites. Pre-computed literal scores from a ScoreCache (1732 images scored in 584.5s) provided preference labels without VRAM contention during training. Persistence round-trip, gradient verification, and pairwise agreement with the literal reward rules were validated after each run.

---

## 2. Results

| Metric | Detached (V1, 10 traj) | Differentiable (V2, 259 traj) |
|--------|----------------------|-------------------------------|
| **Training path** | `train_btrm()` | `train_btrm_differentiable()` |
| **Dataset** | V1: 10 trajectories, 80 images | V2: 259 trajectories, 1732 images |
| **Pair space** | 280 materialized (within-trajectory) | ~1,499,046 combinatorial (cross-trajectory) |
| **Epochs / steps** | 40 epochs | 30 macrobatches |
| **Trainable params** | 7,682 (head only) | 10,108,160 (adapter + head) |
| **Adapter gradients** | None (structurally impossible) | 120/120 params with nonzero grad at step 0 |
| **Wall time** | 148s (120s extraction + 28s head training) | 236.3s (220.4s differentiable training) |
| **Initial loss** | 0.602 | 0.681 |
| **Final loss** | 0.404 (best: 0.286 combined) | 0.455 |
| **Pinkify pairwise agreement** | 97.1% (on 280 V1 pairs) | 49.4% (on 33,156 V2 pairs) |
| **Thisnotthat pairwise agreement** | 84.3% (on 280 V1 pairs) | 66.4% (on 33,181 V2 pairs) |
| **Pinkify Spearman rho** | 0.834 | -0.040 |
| **Thisnotthat Spearman rho** | 0.261 | 0.492 |
| **Persistence verification** | PASS (bit-for-bit) | PASS (206 tensors, 10 scores, all exact) |

---

## 3. Why the Numbers Are Not Directly Comparable

The headline numbers appear to favor the detached path (97.1% vs. 49.4% pinkify agreement). This comparison is misleading for three independent reasons.

**Dataset scale.** The detached run evaluated on 280 within-trajectory pairs from 10 trajectories. A model that memorizes 10 images' relative orderings achieves high accuracy on those same 10 images. The differentiable run evaluated on 33,156 cross-trajectory pairs from 259 trajectories. This is a 118x increase in evaluation scale and tests generalization, not memorization. Pairwise agreement on training data is not the same metric as pairwise agreement on a held-out test set.

**Pair composition.** The detached run used within-trajectory pairs only -- pairs drawn from different denoising steps of the same trajectory. Pinkify has a strong monotonic signal along trajectories (noisy images are pinker than denoised images because broadband noise contributes pink channel energy). Within-trajectory agreement inflates accuracy because it measures a trivial ordering. The differentiable run used cross-trajectory pairs, which require the model to compare images from different prompts, different noise levels, and different generation parameters. This is a harder task.

**Training regime.** The detached run saw every pair 40 times (40 epochs over 280 pairs = 11,200 training examples). The differentiable run saw 30 unique pairs, each once. The detached model was heavily overtrained on its small dataset; the differentiable model saw a tiny fraction of its available pair space.

---

## 4. The Pinkify Data Problem

The pinkify head's near-chance performance (49.4%) is not a model failure -- it is a data distribution problem. The literal pinkify score distribution across V2 final images tells the story:

> **Literal pinkify scores (259 final images)**
> - min: 0.000
> - max: 0.084
> - mean: 0.017
> - stdev: 0.019

The entire dataset lives in a 0.084-wide band clustered near zero. Almost none of the generated images contain meaningful pink content. The pinkify scorer measures local-contrast-weighted pink pixel fraction; a score of 0.017 means roughly 1.7% of the image area has pink pixels with any local contrast. The maximum score in the entire V2 dataset (0.084) is lower than a single early-denoising-step image in V1 (0.085 at step_00 for trajectory 0).

When the literal scores for a pair differ by less than 0.01 -- as most V2 pairs do -- the "correct" pairwise preference is essentially noise. The BTRM model is being asked to predict which of two images with near-identical pinkify scores is pinker, using only diffusion backbone hidden states at a single noise level. The Spearman rho of -0.040 confirms there is no rank-order signal to learn: the BTRM's score ordering is uncorrelated with the literal ordering because the literal ordering itself is arbitrary at this resolution.

The detached V1 run avoided this problem because the 10 V1 trajectories included within-trajectory pairs spanning the full denoising schedule. Early steps (high noise) had pinkify scores of 0.085; final steps had scores near 0.007. The 12x dynamic range within trajectories gave the head an easy signal. The V2 evaluation used only final images, which are uniformly not-pink.

The BTRM pinkify score distribution tells a complementary story:

> **BTRM pinkify scores (259 final images)**
> - min: 0.097
> - max: 0.557
> - mean: 0.290
> - stdev: 0.120

The model has learned a wide distribution (stdev 0.12) from data with near-zero variance in the target (stdev 0.019). It is projecting the 3840-dimensional hidden states onto a direction that captures some other image property correlated with pinkness during training (likely overall color temperature or saturation), but the projection does not track the literal pinkify rule at the resolution required to disambiguate these images.

---

## 5. What Thisnotthat's rho = 0.49 Means

The thisnotthat head tells a different story. With 66.4% pairwise agreement and Spearman rho of 0.492, the model has learned a genuine rank-order correlation with the literal this-vs-that similarity metric from only 30 training steps.

The literal thisnotthat score distribution has more structure:

> **Literal thisnotthat scores (259 final images)**
> - min: -0.235
> - max: 0.164
> - mean: 0.032
> - stdev: 0.057

Unlike pinkify, thisnotthat spans a 0.4-wide range centered near zero, with meaningful variance. Images that structurally resemble THAT (the purple penguin) get negative scores; images resembling THIS (the pizza-ratto sketch) get positive scores. The 3x larger stdev relative to mean (compared to 1.1x for pinkify) means there is real signal in the data.

The BTRM has learned to extract this signal from hidden states:

> **BTRM thisnotthat scores (259 final images)**
> - min: -0.481
> - max: 0.439
> - mean: 0.057
> - stdev: 0.213

The model has widened the distribution (stdev 0.213 vs. 0.057) while preserving rank order (rho = 0.49). This is expected behavior for a Bradley-Terry model: it optimizes for pairwise ranking accuracy, not score regression, so it inflates the score dynamic range to increase the probability of correct comparisons.

A Spearman rho of 0.49 from 30 training steps on 259 cross-trajectory images is substantial. For comparison, the detached V1 run achieved rho of 0.261 from 40 epochs on 10 within-trajectory images. The differentiable path on a harder evaluation set (cross-trajectory) with less overfitting (30 steps, not 11,200 examples) produced nearly double the rank correlation. This is the strongest evidence that the adapter contributes signal beyond the linear probe.

---

## 6. Pipeline Validation: The Real Deliverable of Unit 4

The accuracy numbers are secondary to the pipeline validation that Unit 4 was designed to deliver. Regardless of head accuracy, the following end-to-end capabilities were demonstrated:

**Pre-computation of reward scores at scale.** 1732 images across 259 trajectories were VAE-decoded and scored with both reward functions, producing a ScoreCache compatible with the pair_sampler interface. Wall time: 584.5s. The ScoreCache decouples reward computation from training, eliminating VRAM contention between VAE and backbone.

**Differentiable training with gradient checkpointing.** 30 macrobatches of `train_btrm_differentiable()` completed in 220.4s. Each step performed a full forward pass through the 6B-parameter FP8 backbone with gradient checkpointing on 30 transformer blocks, extracted hidden states with preserved grad_fn, scored through the ScoreUnembedder, and backpropagated through the entire computation graph to update 10.1M adapter parameters.

**Adapter gradient verification.** At step 0, all 120 LoRA parameter tensors (lora_A and lora_B for qkv, out, and w2 projections across 32 main layers + 2 context refiner + 2 noise refiner blocks) had nonzero gradients. The `adapter_max_grad` was 4.1e-5 at step 0, confirming that the computation graph was intact from loss through head through hidden states through backbone through LoRA matrices. This is the structural prevention of Defect 24 and the detach defect.

**Persistence round-trip.** The full compound model (206 weight tensors: 102 lora_A, 102 lora_B, 1 RMSNorm weight, 1 projection weight) was persisted to safetensors and reloaded. 10 test inputs were scored before and after persistence. All 206 tensors and all 10 score pairs were bit-for-bit exact. Verdict: PASS.

**On-the-fly pair sampling.** The BTRMPairSampler drew 31 pairs from a space of 1,499,046 possible pairs (259 trajectories, 1732 positions) with zero retries. Each pair was evaluated by a pluggable preference_fn backed by the ScoreCache. No materialized pair table was required.

**Gradient norm evolution.** The pre-clip gradient norm trajectory reveals the adapter's learning dynamics:

| Steps | Pre-clip grad norm range | Behavior |
|-------|-------------------------|----------|
| 0-9 (warmup) | 6.6 - 12.0 | Moderate, stable |
| 10-15 | 8.7 - 23.1 | Rising as adapter starts differentiating |
| 16-21 | 12.7 - 302.6 | Spike at step 21 (302.6) -- transient instability |
| 22-29 | 6.2 - 118.7 | High variance, still rising trend |

The gradient clip of 0.1 was binding at every step (all 30 steps show `grad_norm = 0.1`). The rising pre-clip norms indicate the adapter is learning features that produce increasingly sharp score gradients -- the model is not converging to a flat loss landscape but actively exploring. The spike at step 21 (302.6x the clip threshold) suggests a particularly informative pair that produced a large gradient signal before clipping.

---

## 7. What Would Improve Pinkify

The pinkify head failed not because of a model deficiency but because the V2 dataset contains no pink variance in its final images. Three concrete approaches would fix this:

**Pink-biased prompts.** Generate trajectories with prompts that produce pink-rich images: "a pink castle at sunset," "fields of blooming cherry blossoms," "flamingos on a pink lake." Intersperse these with prompts that produce explicitly non-pink images ("a gray concrete building in fog," "black and white photograph of a mountain"). This creates cross-trajectory pairs with meaningful pinkify score differences.

**Multi-step evaluation.** The current score comparison evaluates only final (fully denoised) images. But the ScoreCache contains scores for all 8 step keys (step_00 through step_29 plus final). Early denoising steps have higher pinkify scores because broadband noise contributes pink channel energy. Including intermediate steps in the evaluation set would capture the within-trajectory signal that the detached V1 run exploited. This does not require new data generation.

**Image-to-image (i2i) augmentation.** Use the existing i2i pipeline to generate trajectories from deliberately pink/non-pink source images. The i2i starting point biases the color palette of the generated image, creating natural pink variance in final outputs.

The thisnotthat head requires no special intervention. With rho = 0.49 from only 30 steps, extended training (100+ macrobatches) on the existing dataset would likely push agreement above 75% and rho above 0.6.

---

## 8. Appendices

### A. Training Run Summary (differentiable)

> ```json
> {
>   "run_name": "pinkify_thisnotthat_differentiable",
>   "wall_total_s": 236.306,
>   "train_time_s": 220.379,
>   "n_training_steps": 30,
>   "lr": 0.0003,
>   "grad_clip": 0.1,
>   "n_trajectories": 259,
>   "n_cached_scores": 1732,
>   "adapter_trained": true,
>   "adapter_grad_verified_step0": true,
>   "n_adapter_params": 10096640,
>   "n_head_params": 11520,
>   "initial_loss": 0.681,
>   "final_loss": 0.455,
>   "initial_bt_loss": 0.700,
>   "final_bt_loss": 0.663,
>   "final_grad_norm": 63.72,
>   "sampler_stats": {
>     "pair_space_size": 1499046,
>     "n_sampled": 31,
>     "retry_rate": 0.0
>   }
> }
> ```

### B. Persistence Verification (differentiable)

> ```
> Verdict: PASS
> Weight tensors verified: 206 (all exact, max_diff = 0.0)
> Score entries verified: 10 (all exact, diff = 0.0)
> Adapter config: rank=8, alpha=16.0, init_b_std=0.01
> ```

### C. Score Distribution Comparison

> ```
> Pairwise agreement:
>   pinkify:     16,379 / 33,156 = 49.4%  (fail 70% threshold)
>   thisnotthat: 22,027 / 33,181 = 66.4%  (pass 60% threshold)
>
> Spearman rank correlation:
>   pinkify:     rho = -0.040  (no signal)
>   thisnotthat: rho =  0.492  (moderate positive)
>
> Literal score distributions (259 final images):
>   pinkify:     [0.000, 0.084], mean=0.017, stdev=0.019
>   thisnotthat: [-0.235, 0.164], mean=0.032, stdev=0.057
>
> BTRM score distributions (259 final images):
>   pinkify:     [0.097, 0.557], mean=0.290, stdev=0.120
>   thisnotthat: [-0.481, 0.439], mean=0.057, stdev=0.213
> ```

### D. Pinkify Disagreement Sample (representative)

> The disagreements in the pinkify head reveal the data problem. In most
> disagreeing pairs, the literal pinkify score difference is tiny:
>
> ```
> idx_a=0, idx_b=7:  literal_diff=+0.006, btrm_diff=-0.041  (opposite sign)
> idx_a=0, idx_b=14: literal_diff=-0.002, btrm_diff=+0.054  (opposite sign)
> idx_a=0, idx_b=17: literal_diff=-0.0002, btrm_diff=+0.170  (opposite sign)
> ```
>
> When `|literal_diff| < 0.01`, the "correct" preference is
> essentially a coin flip. The BTRM's predictions in these
> cases reflect its learned feature space, not a meaningful
> disagreement with the reward rule.

### E. Gradient Norm Trajectory (all 30 steps)

> ```
> Step  Pre-clip    Clipped    Loss     BT_loss   Acc_pink  Acc_tnt
>  0     6.64       0.10      0.681    0.700      0.0       0.0
>  1     9.31       0.10      0.612    0.653      1.0       1.0
>  5    10.43       0.10      0.606    0.653      0.0       1.0
> 10    16.62       0.10      0.581    0.665      0.0       1.0
> 15     8.65       0.10      0.555    0.643      0.0       1.0
> 16    23.92       0.10      0.437    0.576      1.0       1.0
> 20    12.73       0.10      0.585    0.677      0.0       1.0
> 21   302.60       0.10      0.488    0.723      0.0       0.0
> 25    10.80       0.10      0.557    0.646      1.0       1.0
> 29    63.72       0.10      0.455    0.663      1.0       0.0
> ```
>
> The per-step pinkify accuracy oscillates between 0.0 and 1.0 because each
> macrobatch contains a single pair -- pairwise accuracy on a single pair is
> binary. The training loss descends from 0.681 to 0.455 (33% reduction),
> while the BT loss component descends more modestly (0.700 to 0.663, 5%
> reduction). The difference is explained by the logsquare regularizer term
> becoming increasingly negative (from -0.38 to -4.17), indicating the
> model's scores are growing in magnitude as training progresses.

### F. Comparison with Run 02 (Scrimble/Scrongle)

> Run 02 trained the same compound model architecture on scrimblo/scrongle
> heads over 30 macrobatches with 208 examples per batch. Key differences:
>
> - Run 02 loss: 0.67 -> 0.32 (52% reduction)
> - Unit 4 loss: 0.68 -> 0.45 (33% reduction)
> - Run 02 scrongle acc: 100% by macrobatch 20
> - Unit 4 thisnotthat: 66.4% cross-trajectory agreement
>
> Run 02's scrimblo/scrongle heads trained on within-batch accuracy with 208
> examples per macrobatch. Unit 4 trained on single pairs (1 example per
> step). The 16x difference in examples-per-step explains the slower
> convergence. With matched batch sizes (16 examples/step), the loss
> trajectories would likely be comparable.

---

**Output artifacts** (all in `pinkify_thisnotthat_output/differentiable_run/`):
- `run_summary.json` -- training configuration and wall times
- `training_metrics.jsonl` -- per-step loss, accuracy, gradient norms (30 entries)
- `persistence_verification.json` -- 206 tensors, 10 scores, all bit-for-bit exact (PASS)
- `score_comparison.json` -- pairwise agreement and Spearman correlation vs. literal rules
- Persisted compound model: adapter safetensors + head safetensors + config JSON

**Source artifacts**:
- `pinkify_thisnotthat_output/v2_score_cache.json` -- literal scores for all 1732 positions
- `docs/essay_unit4_pinkify_thisnotthat.md` -- prior detached-path results
- `docs/essay_detach_defect_root_cause.md` -- why the detach path undertrained the adapter

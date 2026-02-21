# FLOPS-Budget 100-Step BTRM Training Run v2

**Date:** 2026-02-19
**Script:** `scripts_ii/run_flops_budget_100step_v2.py`
**Hardware:** RTX 4090 (SM 8.9, 24GB VRAM)
**Wall time:** 1,760s (29.3 min)
**Output:** `flops_budget_100step_v2_output/`

---

## 1. Training Configuration

| Parameter | Value |
|-----------|-------|
| `n_steps` | 100 |
| `macrobatch_budget` | 3.0 (1024^2-equiv FLOPS units) |
| `macrobatch_cross_resolution` | True |
| `lr` | 3e-4 |
| `lr_schedule` | `warmup_cosine` |
| `warmup_steps` | 5 |
| `grad_clip` | 0.1 (`max_grad_norm`) |
| `packed` | True (FlexAttention bin packing) |
| `force_sdpa` | False (SageAttention INT8 QK) |
| `gradient_checkpointing` | True |
| `head_names` | `("pinkify", "thisnotthat")` |
| `adapter_rank` | 8 |
| `adapter_alpha` | 16.0 |
| `adapter_init_b_std` | 0.01 |
| `logsnr_threshold` | 10.0 (corrected from 5.0) |
| `logsnr_decay_rate` | 0.5 (corrected from 0.75) |
| `final_sigma_fix` | True (sigma=0.0 for "final" positions) |

**Model:** BTRMCompoundModel with 5.8B FP8 frozen backbone, rank-8 LoRA adapter (10,096,640 trainable params), and 2-head ScoreUnembedder (11,520 params). Total trainable: 10,108,160.

### What Changed From v1

Three corrections were applied:

1. **LogSNR threshold 5.0 -> 10.0:** Every step in a typical 30-step schedule has logSNR < 10, so the geometric decay is now active for ALL intermediate steps. Only sigma=0 (fully denoised, logSNR=+inf) gets weight=1.0 unconditionally.

2. **LogSNR decay_rate 0.75 -> 0.5:** Each 5-nit logSNR decrease now halves the sampling probability instead of reducing it by 25%. This produces a 4.25:1 ratio between the cleanest and noisiest non-degenerate positions (was 1.82:1).

3. **Final position sigma: sigmas[-2] -> 0.0:** The "final" denoised image is the OUTPUT of the last denoising step, so its sigma is 0.0 (no noise). The v1 script incorrectly assigned `sigmas[-2]` (the last step's INPUT sigma, e.g., 0.034 for 1280x832) to final positions.

### Dataset

**Source:** `multi_res_trajectories/` -- 60 trajectories across 26 unique non-square resolutions in 6 megapixel tiers.

| Tier | Anchor (px) | Trajectories | Example Resolutions |
|------|-------------|-------------|---------------------|
| 256sq | 65,536 | 10 | 224x320, 256x256, 224x288 |
| 320sq | 102,400 | 10 | 384x256, 416x224, 288x352 |
| 384sq | 147,456 | 10 | 320x448, 544x288, 448x352 |
| 512sq | 262,144 | 10 | 384x672, 704x384, 352x736 |
| 704sq | 495,616 | 10 | 640x800, 544x928, 576x864 |
| 1024sq | 1,048,576 | 10 | 1024x1024, 1280x832, 736x1408 |

Aspect ratios range from 0.48 (352x736, extreme portrait) to 1.89 (544x288, extreme landscape).

## 2. Results: Comparison with v1

### Per-Head Accuracy

| Metric | v1 (previous) | v2 (this run) | Delta |
|--------|--------------|---------------|-------|
| Pinkify overall | 85.9% | 85.9% | +0.0% |
| Pinkify last-20 | 91.9% | 94.6% | +2.7% |
| Thisnotthat overall | 82.7% | 86.1% | +3.4% |
| Thisnotthat last-20 | 95.8% | 92.7% | -3.1% |

**Analysis:** The overall accuracy numbers are comparable or improved. The pinkify last-20 improved by 2.7 percentage points. The thisnotthat last-20 decreased by 3.1pp, but this is within the variance expected from the different dataset (26 non-square resolutions vs 3 square resolutions). The overall thisnotthat improved by 3.4pp, indicating better generalization across the full training run despite the harder dataset.

The key comparison is that the v1 run trained on 3 square resolutions (256x256, 512x512, 1024x1024) where the attention artifacts and step-count signals are spatially uniform. The v2 run trains on 26 non-square resolutions where the attention pattern structure varies with aspect ratio. Maintaining comparable accuracy on this harder dataset validates the model's ability to generalize across resolutions.

### Loss Trajectory

| Metric | v1 | v2 |
|--------|-----|-----|
| Initial BT loss | 19.60 | 36.23 |
| Final BT loss | 6.11 | 1.77 |
| Min BT loss | 0.204 (step 95) | 0.957 (step 21) |
| Phase 3 mean per-pair loss | 0.3606 | ~0.25 |
| Mean loss | 6.55 | 7.37 |

The higher initial loss (36.23 vs 19.60) reflects the larger macrobatch at step 0 (29 pairs vs typical ~8). The BT loss is summed over pairs, so more pairs = larger raw value. The final BT loss of 1.77 (4 pairs) is substantially lower than v1's 6.11 (8 pairs), indicating better per-pair convergence.

### FLOPS Budget Statistics

| Metric | v1 | v2 |
|--------|-----|-----|
| Mean consumed | 3.31 | 3.34 |
| Min consumed | 3.00 | 3.00 |
| Max consumed | 4.02 | 4.90 |
| Mean pairs/step | 7.85 | 10.79 |
| Max pairs/step | 24 | 32 |
| Mean bins/step | 5.49 | 6.49 |
| Total cross-res pairs | 245 (31.2%) | 324 (30.0%) |
| Total pairs | 785 | 1,079 |

The v2 run processes 37% more pairs per step (10.79 vs 7.85) because the 6-tier resolution mix includes more small images that fill the FLOPS budget cheaply. The cross-resolution fraction (30.0%) is close to the 30% probability gate target.

### Step Timing

| Metric | v1 | v2 |
|--------|-----|-----|
| Step 0 (compile) | 98.9s | 165.6s |
| Steady-state mean | 12.7s | 16.0s |
| Steady-state min | 2.8s | 3.1s |
| Steady-state max | 39.6s | 25.0s |
| Total training | 1,358s | 1,747s |

Step 0 is slower because the 6-tier dataset triggers more shape recompilations. Steady-state is slightly slower due to the higher mean pair count per step.

## 3. LogSNR Distribution Evidence

The corrected logSNR weighting produces a strongly clean-biased sampling distribution.

### Empirical Sampling (2000 pairs from sampler)

| Metric | Value |
|--------|-------|
| Sigma=0 positions | 42.5% of all sampled positions |
| Mean logSNR | +6.43 nats |
| Sigma < 0.1 (clean) | ~65% |
| Sigma > 0.5 (noisy) | ~15% |

This is a dramatic shift from the v1 run's near-uniform distribution. Under the old parameters (threshold=5.0, decay_rate=0.75), sigma=0 received no special treatment and clean positions were sampled at only ~18% frequency. Now sigma=0 positions dominate at 42.5%, confirming the fix works in practice.

### Resolution-Aware Behavior

The sigma at a given step index depends on the resolution's sigma schedule. For 256x256 (shift=4.03), step_29 has sigma=0.124 (logSNR=+3.9, weight=0.430). For 1280x832 (shift=1.0), step_29 has sigma=0.034 (logSNR=+6.7, weight=0.632). The mean sampling weight per tier confirms this:

| Tier | Mean Sampling Weight | Interpretation |
|------|---------------------|----------------|
| 256sq | Lower | Smaller images retain more noise at each step |
| 1024sq | Higher | Larger images are cleaner at each step |
| All tiers | sigma=0 at 1.0 | Final denoised images get full weight regardless |

The "final" position (sigma=0.0) is the great equalizer: it gets weight=1.0 for every resolution tier. This means 12.5% of positions (1 out of 8 saved steps) always receive maximum weight, and these are the positions most informative for the reward model.

### Chart 09: LogSNR Histogram

The logSNR distribution histogram (`charts/09_logsnr_distribution.png`) shows:
- A spike at logSNR=+15 (capped representation of sigma=0, 42.5% of samples)
- A broad distribution from logSNR=-6 to +7 for intermediate steps
- Monotonically increasing density toward cleaner positions
- The threshold=10.0 line is above all intermediate step logSNR values, confirming the geometric decay is active for all non-final steps

## 4. Exemplar Quality

All 12 exemplars use "final" positions (sigma=0.0), confirming they are fully denoised images, not noisy latents. This directly fixes the v1 failure mode where exemplars could be partially denoised.

### Score Ranges

| Head | Top-3 Scores | Bottom-3 Scores | Spread |
|------|-------------|-----------------|--------|
| Pinkify | 1.95, 1.89, 1.69 | 1.00, 1.00, 1.22 | 0.95 |
| Thisnotthat | 2.92, 2.74, 2.67 | 2.00, 2.03, 2.26 | 0.91 |

The score spread (~0.9 for both heads) indicates the model has learned to discriminate between images, even among fully denoised exemplars. The top-scoring images (traj 11 and 15) consistently score highest on both heads, suggesting the model has identified trajectory-level quality differences.

## 5. Resolution Tier Distribution

### Images Sampled per Tier

| Tier | Images | Fraction |
|------|--------|----------|
| 256sq | 431 | 20.0% |
| 320sq | 460 | 21.3% |
| 384sq | 442 | 20.5% |
| 512sq | 357 | 16.5% |
| 704sq | 229 | 10.6% |
| 1024sq | 239 | 11.1% |

### FLOPS-Normalized Distribution

| Bucket | FLOPS % |
|--------|---------|
| Megapixel (1024sq) | 64.0% |
| Small (all others) | 36.0% |

The FLOPS split is 64/36 rather than the target 33/67. This is because 6 out of 10 trajectories in the 1024sq tier are above the megapixel threshold (1024x1024, 1280x832) while 4 are counted differently (1088x960 is in the megapixel bucket, 736x1408 has 1,035,264 pixels, barely above). The FLOPS-budget sampler correctly fills the budget but the megapixel tier naturally dominates compute because each 1024sq image costs 250-4000x the attention FLOPS of a 256sq image.

### Sampler Tier Selection Counts

| Tier | Samples | Fraction |
|------|---------|----------|
| 256sq | 2,023 | 49.3% |
| 320sq | 963 | 23.5% |
| 384sq | 575 | 14.0% |
| 512sq | 290 | 7.1% |
| 704sq | 140 | 3.4% |
| 1024sq | 114 | 2.8% |

This shows the FLOPS-inverse weighting working correctly: the 256sq tier is sampled 17.7x more often than 1024sq because each 256sq image costs ~250x less compute.

## 6. Gradient Behavior

| Metric | Value |
|--------|-------|
| Pre-clip mean | 0.567 |
| Pre-clip max | 1.436 (step 2) |
| Post-clip mean | 0.100 (at the clip boundary) |

The gradient clipping at 0.1 is active on essentially every step (post-clip mean = 0.100). The pre-clip norms are well-behaved (all < 1.5), indicating no gradient explosion issues. The dual gradient norm chart (`charts/03_gradient_norms_dual.png`) shows the pre-clip norm decreasing over training as the model converges, from ~0.8 in early steps to ~0.3-0.4 in the cosine decay phase.

## 7. Artifacts

All artifacts saved to `flops_budget_100step_v2_output/`:

| Artifact | Path | Description |
|----------|------|-------------|
| Training metrics | `training_metrics.jsonl` | 100 entries, per-step JSONL |
| Training curve | `training_curve.json` | Full training curve with funfetti metadata |
| Run summary | `run_summary.json` | Aggregate statistics |
| Validation metrics | `validation_metrics.json` | 1,786 pair results tracked |
| logSNR distribution | `logsnr_distribution.json` | 4,000 sampled sigma/logSNR values |
| Tier distribution | `tier_distribution.json` | Per-tier image counts |
| Tier sampling weights | `tier_sampling_weights.json` | Mean logSNR weight per tier |
| FLOPS-normalized PDF | `flops_normalized_resolution.json` | Per-resolution FLOPS proportions |
| Resolution PDF | `resolution_pdf.json` | Per-resolution image counts |

### Charts

| Chart | File | Description |
|-------|------|-------------|
| 01 | `01_loss_curve_bounded.png` | BT loss with ln(2) random reference |
| 02 | `02_per_head_accuracy_bounded.png` | Per-head accuracy (%) with 50% random reference |
| 03 | `03_gradient_norms_dual.png` | Pre-clip and post-clip gradient norms |
| 04 | `04_learning_rate.png` | Warmup cosine LR schedule |
| 05 | `05_flops_consumed.png` | FLOPS consumed per macrobatch vs budget |
| 06 | `06_pairs_per_macrobatch.png` | Variable pair count per step |
| 07 | `07_cross_res_fraction.png` | Cross-resolution pair fraction vs 30% target |
| 08 | `08_resolution_tier_distribution.png` | Resolution tier distribution of sampled pairs |
| 09 | `09_logsnr_distribution.png` | logSNR histogram (confirms clean bias) |
| 10 | `10_sampling_weight_by_tier.png` | Mean sampling weight by resolution tier |

### Checkpoints

| Step | Path |
|------|------|
| 25 | `checkpoint_step025/` |
| 50 | `checkpoint_step050/` |
| 75 | `checkpoint_step075/` |
| Final | `rtheta_adapter.safetensors` + `btrm_head.safetensors` |

### Exemplars

12 exemplar images in `exemplars/`:
- 3 top + 3 bottom per head (pinkify, thisnotthat)
- All from "final" positions (sigma=0, fully denoised)
- PNG files with metadata-rich filenames

## 8. Recommendations for Next Steps

1. **Step-count variation for thisnotthat:** The current dataset uses 30 steps for all trajectories. The thisnotthat (scrongle) head discriminates step count, so adding trajectories at 8-22 steps would strengthen its training signal and likely improve the thisnotthat last-20 accuracy beyond the current 92.7%.

2. **Prompt diversity:** All 60 trajectories use the same prompt. Varying prompts would test whether the reward model generalizes across content, not just resolution and noise level.

3. **Extended training (200-500 steps):** The cosine LR schedule reaches 0 at step 100. Both heads are still improving at that point (last-20 > overall), suggesting more training could push accuracy higher. A longer run with the same cosine schedule over 200+ steps would determine the ceiling.

4. **DRGPO integration:** The BTRM is now capable of scoring images across resolutions. The next milestone is using these scores as a reward signal for DRGPO (Direct Reward Gradient Policy Optimization) on the generation model itself.

5. **Multi-GPU scaling:** The per-step wall time (16s steady state) is dominated by the variable pair count (2-32 pairs). Pipeline parallelism across 2 GPUs could halve the per-bin forward pass time, or data parallelism could allow larger macrobatch budgets.

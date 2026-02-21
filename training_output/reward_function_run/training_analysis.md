# Training Analysis: reward_function_validated_training

**Date:** 2026-02-20
**Steps completed:** 150
**Total training time:** 3106.6s (51.8 min)
**Wall time:** 3226.3s (53.8 min)

## Run Configuration

- **mode**: reward_function_validated_training
- **preference_source**: reward_manifest (pinkify_score_gpu + thisnotthat_score_gpu)
- **n_steps**: 150
- **macrobatch_budget**: 3.0
- **macrobatch_cross_resolution**: True
- **lr**: 0.0003
- **lr_schedule**: warmup_cosine
- **grad_clip**: 0.1
- **warmup_steps**: 5
- **clean_fraction**: 0.8
- **dataset**: F:\dox\repos\ai\futudiffu\multi_res_trajectories
- **n_trajectories**: 420
- **n_unique_resolutions**: 84
- **n_unique_prompts**: 12
- **eval_interval**: 10
- **eval_count**: 16

## Loss Trajectory

| Metric | Value |
|--------|-------|
| Initial BT loss | 23.1283 |
| Final BT loss | 6.7351 |
| Minimum BT loss | 1.939902 (step 24) |
| Maximum BT loss | 48.0076 (step 26) |
| EMA(0.1) best | 10.1040 (step 63) |
| Mean loss | 13.4241 |
| Std loss | 7.5655 |

## Per-Head Accuracy

| Head | Overall | Last 20 Steps |
|------|---------|-----------|
| pinkify | 83% | 81% |
| thisnotthat | 58% | 61% |

## Gradient Norm Analysis

| Metric | Value |
|--------|-------|
| Mean | 0.460 |
| Max | 1.054 (step 14) |
| Min | 0.237713 |
| Steps with norm > 10 | 0 |

## Learning Rate

| Metric | Value |
|--------|-------|
| Initial LR | 6.00e-05 |
| Peak LR | 3.00e-04 |
| Final LR | 0.00e+00 |

## Step Timing

| Metric | Value |
|--------|-------|
| Step 0 (compilation) | 91.0s |
| Mean steady-state (steps 1+) | 20.2s |
| Total training time | 3106.6s (51.8 min) |

## Charts

Generated in `charts/`:
- `01_loss_curve.png` -- BT loss raw scatter + EMA(0.1)
- `02_per_head_accuracy.png` -- Per-head accuracy with running average
- `03_gradient_norms.png` -- Pre-clip gradient norm (log scale)
- `04_learning_rate.png` -- LR schedule
- `05_step_timing.png` -- Seconds per step

## Funfetti Diagnostic Charts

- `06_resolution_pdf.png` -- Empirical PDF of resolution scale trained on (image count)
- `07_aspect_ratio_pdf.png` -- Aspect ratio PDF (W/H)
- `08_metrics_by_resolution.png` -- Loss distribution by resolution bucket
- `09_microbatch_pairs.png` -- Pairs per microbatch across training
- `10_context_length.png` -- Context length per microbatch
- `11_flops_normalized_resolution.png` -- FLOPS-normalized resolution PDF (validates 33/67 split)

### Resolution Summary

| Resolution (pixels) | Count | Proportion |
|---------------------|-------|------------|
| 61,440 | 59 | 1.6% |
| 64,512 | 426 | 11.9% |
| 65,536 | 114 | 3.2% |
| 67,584 | 136 | 3.8% |
| 71,680 | 91 | 2.5% |
| 93,184 | 48 | 1.3% |
| 98,304 | 204 | 5.7% |
| 100,352 | 155 | 4.3% |
| 101,376 | 183 | 5.1% |
| 102,400 | 59 | 1.6% |
| 106,496 | 92 | 2.6% |
| 110,592 | 36 | 1.0% |
| 138,240 | 52 | 1.4% |
| 143,360 | 180 | 5.0% |
| 146,432 | 29 | 0.8% |
| 147,456 | 254 | 7.1% |
| 153,600 | 42 | 1.2% |
| 156,672 | 85 | 2.4% |
| 157,696 | 53 | 1.5% |
| 258,048 | 217 | 6.0% |
| 259,072 | 37 | 1.0% |
| 261,120 | 76 | 2.1% |
| 262,144 | 28 | 0.8% |
| 266,240 | 74 | 2.1% |
| 270,336 | 70 | 1.9% |
| 272,384 | 76 | 2.1% |
| 276,480 | 15 | 0.4% |
| 479,232 | 13 | 0.4% |
| 486,400 | 16 | 0.4% |
| 487,424 | 23 | 0.6% |
| 491,520 | 57 | 1.6% |
| 494,592 | 27 | 0.8% |
| 495,616 | 20 | 0.6% |
| 497,664 | 76 | 2.1% |
| 504,832 | 25 | 0.7% |
| 505,856 | 36 | 1.0% |
| 507,904 | 6 | 0.2% |
| 512,000 | 42 | 1.2% |
| 1,022,976 | 1 | 0.0% |
| 1,024,000 | 4 | 0.1% |
| 1,032,192 | 15 | 0.4% |
| 1,036,288 | 26 | 0.7% |
| 1,038,336 | 8 | 0.2% |
| 1,039,360 | 4 | 0.1% |
| 1,044,480 | 12 | 0.3% |
| 1,047,552 | 5 | 0.1% |
| 1,048,576 | 40 | 1.1% |
| 1,049,600 | 25 | 0.7% |
| 1,050,624 | 39 | 1.1% |
| 1,056,768 | 37 | 1.0% |
| 1,059,840 | 22 | 0.6% |
| 1,060,864 | 27 | 0.8% |
| 1,064,960 | 93 | 2.6% |

**Total pairs processed:** 1795
**Total NFEs:** 3590

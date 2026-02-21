# PINKIFY-Validated BTRM Training Run

## 1. Executive Summary

A 150-step BTRM training run on the multi-resolution trajectory dataset (420 trajectories, 84 unique resolutions, 12 prompts) with 80/20 clean-biased sampling was executed with periodic PINKIFY holdout validation every 10 steps. The model learned a strong noise-vs-clean discriminator (77-80% pairwise accuracy on both heads) but **did not achieve a non-vacuous pinkify ranking** -- it never passed all 5 PINKIFY constraints simultaneously. The ranking the model converges to is `C < E < D < B < A < F`, which preserves only 2 of 5 constraints (D ~ E and max(D,E) < F), while the ground truth is `A < B < C < D ~ E < F`.

This is a clean negative result that identifies a structural gap: the training signal (noise-vs-clean preference from sigma comparisons) does not contain the information needed to learn the pinkify ranking (which discriminates quantization artifacts, not noise levels).

## 2. Ground Truth PINKIFY Scores (Pixel-Space)

The `pinkify_score_gpu` function was evaluated against 6 artisanally constructed challenge images with known ranking `A < B < C, D ~ E, {A,B,C} < {D,E} < F`:

| Label | Score     | Notes                            |
|-------|-----------|----------------------------------|
| A     | 0.000000  | Least pink                       |
| B     | 0.007070  | Slightly more pink               |
| C     | 0.007594  | B < C margin = 0.000524 (tight)  |
| D     | 0.008987  | Moderately pink                  |
| E     | 0.008987  | D = E (identical scores)         |
| F     | 0.046157  | Most pink (5x gap over D/E)      |

All 5 constraints pass in pixel space. Notable: the B < C gap is only 0.000524 -- barely 7% of the A-to-B gap. The D = E equality is exact. The F outlier is 5x above the D/E cluster.

## 3. Step-by-Step PINKIFY Validation Results

### 3.1 Constraint Evolution Table

| Step | A<B  | B<C  | max(ABC)<min(DE) | D~E  | max(DE)<F | Total | Rank Order          |
|------|------|------|------------------|------|-----------|-------|---------------------|
| -1   | PASS | FAIL | FAIL             | PASS | PASS      | 3/5   | E < C < A < D < B < F |
| 0    | PASS | FAIL | FAIL             | PASS | PASS      | 3/5   | C < E < A < D < B < F |
| 10   | PASS | FAIL | FAIL             | PASS | PASS      | 3/5   | C < E < A < D < B < F |
| 20   | PASS | FAIL | FAIL             | PASS | PASS      | 3/5   | C < E < A < D < B < F |
| 30   | PASS | FAIL | FAIL             | PASS | PASS      | 3/5   | C < E < D < A < B < F |
| 40   | PASS | FAIL | FAIL             | PASS | PASS      | 3/5   | C < E < D < A < B < F |
| 50   | PASS | FAIL | FAIL             | PASS | PASS      | 3/5   | C < E < D < A < B < F |
| 60   | PASS | FAIL | FAIL             | PASS | PASS      | 3/5   | E < C < D < A < B < F |
| 70   | FAIL | FAIL | FAIL             | PASS | PASS      | 2/5   | E < C < D < B < A < F |
| 80   | FAIL | FAIL | FAIL             | PASS | PASS      | 2/5   | C < E < D < B < A < F |
| 90   | FAIL | FAIL | FAIL             | PASS | PASS      | 2/5   | C < E < D < B < A < F |
| 100  | FAIL | FAIL | FAIL             | PASS | PASS      | 2/5   | C < E < D < B < A < F |
| 110  | FAIL | FAIL | FAIL             | PASS | PASS      | 2/5   | C < E < D < B < A < F |
| 120  | FAIL | FAIL | FAIL             | PASS | PASS      | 2/5   | C < E < D < B < A < F |
| 130  | FAIL | FAIL | FAIL             | PASS | PASS      | 2/5   | C < E < D < B < A < F |
| 140  | FAIL | FAIL | FAIL             | PASS | PASS      | 2/5   | C < E < D < B < A < F |
| 149  | FAIL | FAIL | FAIL             | PASS | PASS      | 2/5   | C < E < D < B < A < F |

### 3.2 Key Observations

**Phase 1 (steps -1 to 60): 3/5 constraints, A < B preserved.** The untrained model already passes A < B, D ~ E, and max(D,E) < F. During the first 60 steps, the model maintains the A < B ordering while all scores shift upward monotonically. The A-B gap starts at 0.014 and shrinks to 0.003 by step 50.

**Phase 2 (steps 70+): A < B flips, degrades to 2/5.** At step 70, the A < B constraint flips: A overtakes B by 0.007 and this gap widens to 0.024 by step 149. The model has stably converged to the ordering `C < E < D < B < A < F`.

**Invariants preserved throughout:** D ~ E (relative difference always under 6%) and max(D,E) < F (gap grows from 0.13 to 0.24). These two constraints are the easiest: D and E are nearly identical in pixel space, and F is a clear outlier.

**Invariant never achieved:** B < C is violated at every measurement point. In the model's internal representation, B consistently scores higher than C -- the OPPOSITE of the ground truth. The B-C inversion gap grows from 0.022 at step -1 to 0.163 at step 149.

### 3.3 Score Dynamics

All 6 PINKIFY scores increase monotonically throughout training:

| Step | A       | B       | C       | D       | E       | F       |
|------|---------|---------|---------|---------|---------|---------|
| -1   | -0.097  | -0.084  | -0.106  | -0.090  | -0.109  | 0.036   |
| 50   | 0.895   | 0.898   | 0.833   | 0.870   | 0.836   | 1.081   |
| 100  | 1.462   | 1.429   | 1.299   | 1.371   | 1.303   | 1.564   |
| 149  | 1.585   | 1.561   | 1.398   | 1.479   | 1.415   | 1.715   |

The model learns a monotonic global shift upward (likely from the sigma=0 clean-image bias in training), but the RELATIVE ordering of {A, B, C} vs {D, E} is driven by latent-space features the model is picking up that do not correspond to the pixel-space pinkify signal.

## 4. Training Dynamics

### 4.1 Loss and Accuracy

- **Initial loss:** 7.08 (random chance for BT loss is ln(2) = 0.69; the higher initial loss reflects the macro-batch normalization)
- **Final loss:** 4.53
- **Minimum loss:** 0.00 at step 20 (degenerate: all pairs were ties)
- **Pinkify accuracy:** 77.3% overall, 80.0% last-20
- **Thisnotthat accuracy:** 75.8% overall, 80.5% last-20
- **Step time:** 10.9s/step (FLOPS-budget packed, SageAttention INT8 QK)
- **Wall time:** 28.1 minutes (including model loading, TE encoding, VAE encoding)

### 4.2 Clean-Biased Sampling Verification

- **Configured:** clean_fraction=0.8
- **Measured:** 80.9% of sampled positions had sigma=0
- **LogSNR histogram:** 2922/3614 positions at sigma=0, remaining 692 distributed across logSNR bins with concentration at [2,5) nats (387 positions)
- **Tier coverage:** all 6 tiers populated with 70 trajectories each, sampling counts range from 151 (megapixel) to 439 (65536-pixel)

### 4.3 Model Configuration

- 420 trajectories, 84 unique resolutions, 12 unique prompts
- Adapter: rtheta rank=8, alpha=16.0, init_b_std=0.01
- Total trainable: 10,096,640 adapter + 11,520 head = 10,108,160 parameters
- LR: 3e-4 with warmup(5) + cosine decay
- Gradient clipping: 0.1
- Macrobatch budget: 3.0 FLOPS units
- Checkpoints at steps 25, 50, 75, 100, 125

## 5. Analysis: Why the Model Fails to Learn PINKIFY

### 5.1 The Training Signal Does Not Contain PINKIFY Information

The preference function used for training is:

```python
def preference_fn(pair):
    # Cleaner image (lower sigma) wins on ALL heads
    if sigma_a < sigma_b: return +1 for all heads
    if sigma_b < sigma_a: return -1 for all heads
```

This is a **noise-level discriminator**, not a pinkify discriminator. The model learns "cleaner images score higher" -- which is correct for the step_quality (thisnotthat) head but carries zero information about pink-channel saturation. The PINKIFY challenge images are all at sigma=0 (fully denoised), so the noise-vs-clean preference function cannot distinguish between them.

### 5.2 What the Model Actually Learned

The model learned a non-trivial internal representation that orders the challenge images by some latent-space feature correlated with noise level but NOT with pinkify score. Specifically:

- **C and E consistently score lowest.** These may have latent-space features that the model associates with "more noisy" (higher sigma).
- **A scores higher than B after step 70.** The A < B inversion suggests the model is responding to a latent-space feature that anti-correlates with the ground-truth pinkify ranking for these specific images.
- **F always scores highest.** F is a strong outlier in pixel space (highest pinkify score by 5x), and whatever latent feature the model learned also makes F an outlier. This is likely coincidental rather than causal.

### 5.3 The Gap: Missing Pinkify-Specific Preference Labels

For the BTRM model to learn pinkify ranking, it needs training pairs where the preference label reflects PINKIFY-SPECIFIC quality, not generic noise level. Concretely:

1. **Pairs of fully-denoised images** (sigma=0 for both) where one has more quantization artifacts (SageAttention INT8 QK) and the other has fewer (SDPA).
2. **Preference labels** that say "the less-artifacted image wins on the pinkify head."

The current training dataset has trajectories generated with different backends, but the preference function does not discriminate between backends -- it only looks at sigma. The pinkify head receives the same gradient signal as the thisnotthat head, so they converge to the same discriminator.

### 5.4 Relationship to Prior Runs

This result is qualitatively similar to prior FLOPS-budget training runs: strong noise-vs-clean discrimination (~80% accuracy), but no evidence of head specialization. The pinkify head and thisnotthat head learn nearly identical functions because they receive nearly identical gradient signals. Clean-biased sampling (80/20) and multi-resolution diversity did not change this outcome -- the bottleneck is the preference function, not the sampling distribution.

## 6. Conclusions and Next Steps

1. **The training infrastructure works correctly.** 150-step training with periodic PINKIFY validation, clean-biased sampling, multi-resolution FLOPS-budget packing, and incremental persistence all functioned as designed. The 0.85s per PINKIFY evaluation (6 cached latent forward passes) is negligible overhead.

2. **The PINKIFY ranking is non-vacuous in pixel space** (all 5 constraints pass) but **the BTRM model cannot learn it from noise-level preferences alone.** This is the expected result: pinkify discriminates quantization artifacts, which exist only in pixel space and only between images rendered with different attention backends. A preference function based on sigma cannot provide this signal.

3. **To achieve a non-vacuous BTRM pinkify ranking, the training pipeline needs:**
   - Trajectory pairs where both images are at sigma=0 but differ in attention backend (SDPA vs SageAttention)
   - A preference function that runs `pinkify_score_gpu` on VAE-decoded images to assign per-head preferences
   - OR: pre-computed pinkify preference labels stored in the dataset metadata

4. **The D ~ E and max(D,E) < F constraints are trivially satisfied** because D/E are nearly identical and F is a clear outlier. These constraints would pass for almost any monotonic scoring function.

5. **The B < C constraint is the hardest** -- a 0.000524 gap in pixel space. Even with the correct training signal, this may require many more steps or higher rank to resolve.

## Appendix A: PINKIFY Validation Log Excerpts

### Initial evaluation (step -1, untrained):
```json
{
  "scores": {"A": -0.097, "B": -0.084, "C": -0.106, "D": -0.090, "E": -0.109, "F": 0.036},
  "rank_order": ["E", "C", "A", "D", "B", "F"],
  "constraints_passed": 3
}
```

### Step 50 (peak A<B preservation):
```json
{
  "scores": {"A": 0.895, "B": 0.898, "C": 0.833, "D": 0.870, "E": 0.836, "F": 1.081},
  "rank_order": ["C", "E", "D", "A", "B", "F"],
  "constraints_passed": 3,
  "A_B_gap": 0.002
}
```

### Step 70 (A<B inversion):
```json
{
  "scores": {"A": 1.115, "B": 1.109, "C": 1.011, "D": 1.062, "E": 1.005, "F": 1.278},
  "rank_order": ["E", "C", "D", "B", "A", "F"],
  "constraints_passed": 2,
  "A_B_gap": -0.007
}
```

### Final evaluation (step 149):
```json
{
  "scores": {"A": 1.585, "B": 1.561, "C": 1.398, "D": 1.479, "E": 1.415, "F": 1.715},
  "rank_order": ["C", "E", "D", "B", "A", "F"],
  "constraints_passed": 2,
  "A_B_gap": -0.024
}
```

## Appendix B: Training Metrics Summary

```
Run: pinkify_validated_training
Steps: 150, Time: 28.1 min (10.9s/step)
Dataset: 420 trajectories, 84 resolutions, 12 prompts
Clean fraction: 80.9% (target 80%)
Macrobatch budget: 3.0 FLOPS units
Pair space: 5,643,120 possible pairs
Pairs sampled: 1,795 (0.03% coverage)

Loss: 7.08 -> 4.53 (min 0.00 at step 20)
Pinkify accuracy: 77.3% overall, 80.0% last-20
Thisnotthat accuracy: 75.8% overall, 80.5% last-20

PINKIFY evaluations: 17 (step -1, 0, 10, 20, ..., 149)
PINKIFY best: 3/5 constraints (steps -1 through 60)
PINKIFY final: 2/5 constraints (A<B lost at step 70)
PINKIFY never all-pass: first_all_pass_step = null
```

## Appendix C: File Manifest

| File | Description |
|------|-------------|
| `training_output/pinkify_validation_run/training_curve.jsonl` | 150 per-step training metrics (JSONL, incremental) |
| `training_output/pinkify_validation_run/pinkify_validation_log.jsonl` | 17 PINKIFY evaluations (JSONL, incremental) |
| `training_output/pinkify_validation_run/pinkify_validation_log.json` | Structured PINKIFY validation summary |
| `training_output/pinkify_validation_run/pinkify_ground_truth.json` | Pixel-space ground truth scores |
| `training_output/pinkify_validation_run/run_summary.json` | Run configuration and summary statistics |
| `training_output/pinkify_validation_run/validation_metrics.json` | Welford tracker (1,200 pair results) |
| `training_output/pinkify_validation_run/rtheta_adapter.safetensors` | Final adapter weights |
| `training_output/pinkify_validation_run/btrm_head.safetensors` | Final head weights |
| `training_output/pinkify_validation_run/checkpoint_step{025,050,075,100,125}/` | Intermediate checkpoints |
| `scripts_ii/run_pinkify_validated_training.py` | Training script |

# Training Data Distributions for r_theta BTRM Sweep

Analysis of image resolution, sigma (noise level), and log-SNR distributions
in the training data used by `scripts_ii/sweep_rtheta_lr.py`.

## 1. Image Resolution Distribution

**Finding: The r_theta sweep uses only 1280x832 images. All 10 training
trajectories are identical resolution.**

The sweep script sets `N_TRAJECTORIES = 10` and uses trajectories 0-9 from
`btrm_dataset/manifest.json`. All 10 are t2i (text-to-image) trajectories
generated at the default Z-Image output resolution:

| Trajectories | Pixel Size | Latent Shape     | Type | n_steps | Precision |
|-------------|-----------|------------------|------|---------|-----------|
| 0-9 (sdpa)  | 1280x832  | (1,16,104,160)   | t2i  | 30      | sdpa      |

The full dataset contains substantially more diversity that is **not** used
in the sweep:

| Traj Range | Pixel Size     | Latent Shape     | Type | n_steps  | Precision |
|-----------|---------------|------------------|------|----------|-----------|
| 0-9       | 1280x832      | (1,16,104,160)   | t2i  | 30       | sdpa      |
| 10-19     | 1280x832      | (1,16,104,160)   | t2i  | 30       | sage      |
| 20-29     | 1280x832      | (1,16,104,160)   | t2i  | 8-22     | sdpa      |
| 30-39     | 1280x832      | (1,16,104,160)   | t2i  | 8-21     | sage      |
| 40,44,47  | 832x1280      | (1,16,160,104)   | i2i  | 30       | mixed     |
| 41,42,46  | 496x544       | (1,16,68,62)     | i2i  | 30       | mixed     |
| 43,45,48,49| 512x512      | (1,16,64,64)     | i2i  | 30       | mixed     |

Key observations:
- **Zero resolution diversity in training.** All gradient updates see the
  same sequence length (4160 patches after patch_size=2).
- The model has never seen 512x512 or 496x544 latents during r_theta training.
- The i2i trajectories (40-49) add denoise < 1.0, meaning their sigma
  schedules start partway through, not from sigma=1.0.


## 2. Sigma Distribution in Training

### 2.1 The 30-Step Sigma Schedule

The CONST noise model with `shift=1.0, multiplier=1.0` produces a linear-ish
sigma schedule. For 30 steps:

| Step | Sigma  | log-SNR  |
|------|--------|----------|
| 0    | 1.000  | -inf     |
| 1    | 0.967  | -6.76    |
| 2    | 0.934  | -5.30    |
| 3    | 0.900  | -4.39    |
| 4    | 0.867  | -3.75    |
| 5    | 0.834  | -3.23    |
| ...  | ...    | ...      |
| 14   | 0.534  | -0.27    |
| 15   | 0.500  | 0.00     |
| ...  | ...    | ...      |
| 24   | 0.200  | +2.77    |
| ...  | ...    | ...      |
| 29   | 0.034  | +6.69    |
| 30   | 0.000  | +inf     |

The sigma spacing is approximately linear (~0.033 per step), but the log-SNR
spacing is highly non-uniform. The schedule spends more steps at high sigma
(high noise) than at low sigma (low noise).

### 2.2 Sparse Step Sampling

The trajectory dataset saves latents at steps 0, 4, 9, 14, 19, 24, 29, plus
"final" (which maps to `sigmas[-2]` = step 29's sigma). This gives **7 unique
sigma values** (step_29 and final are duplicates):

| Step Key | Sigma  | log-SNR   | Physical Meaning                    |
|----------|--------|-----------|--------------------------------------|
| step_00  | 1.000  | -inf      | Pure noise, zero signal              |
| step_04  | 0.867  | -3.75     | Heavy noise, faint structure         |
| step_09  | 0.700  | -1.69     | Noise-dominant, coarse layout        |
| step_14  | 0.534  | -0.27     | Near-equal noise/signal              |
| step_19  | 0.367  | +1.09     | Signal-dominant, mid-detail          |
| step_24  | 0.200  | +2.77     | Low noise, fine detail emerging      |
| step_29  | 0.034  | +6.69     | Near-clean, only fine grain noise    |
| final    | 0.034  | +6.69     | Same as step_29 (duplicate)          |

### 2.3 Distribution Across Training Images

With 10 trajectories x 8 step positions = **80 images** total:

| Sigma  | Count | Fraction |
|--------|-------|----------|
| 1.000  | 10    | 12.5%    |
| 0.867  | 10    | 12.5%    |
| 0.700  | 10    | 12.5%    |
| 0.534  | 10    | 12.5%    |
| 0.367  | 10    | 12.5%    |
| 0.200  | 10    | 12.5%    |
| 0.034  | 20    | 25.0%    |

The distribution is nearly uniform across unique sigma values, except sigma=0.034
gets double weight due to the step_29/final duplication. This is an artifact of
the trajectory saving scheme, not a deliberate design choice.


## 3. Log-SNR Analysis

### 3.1 Definition

For the CONST noise model:
- Noised latent: `x_t = sigma * noise + (1 - sigma) * x_0`
- Signal coefficient: `(1 - sigma)`
- Noise coefficient: `sigma`
- SNR = `(1 - sigma)^2 / sigma^2`
- **log-SNR = 2 * ln((1 - sigma) / sigma)**

### 3.2 Distribution of Training Data by Log-SNR Bin

| Log-SNR Bin | Count | Fraction | Physical Regime                     |
|-------------|-------|----------|--------------------------------------|
| > 10        | 0     | 0.0%     | Near-perfect (not represented)       |
| 5 to 10     | 20    | 25.0%    | Near-clean                           |
| 0 to 5      | 20    | 25.0%    | Signal-dominant                      |
| -5 to 0     | 30    | 37.5%    | Noise-dominant                       |
| < -5        | 10    | 12.5%    | Pure noise (sigma=1.0, log-SNR=-inf) |

**The training data is biased toward noise-dominant regimes.** 50% of images
have log-SNR < 0 (noise dominates signal), while only 25% have log-SNR > 5
(near-clean).

### 3.3 Gap Analysis in Log-SNR Space

The sparse step sampling creates uneven coverage in log-SNR space:

| Between Steps  | Log-SNR Gap |
|---------------|-------------|
| step_04 -> 09 | 2.05        |
| step_09 -> 14 | 1.42        |
| step_14 -> 19 | 1.36        |
| step_19 -> 24 | 1.68        |
| step_24 -> 29 | **3.92**    |

The largest gap is between step_24 (sigma=0.200, log-SNR=+2.77) and step_29
(sigma=0.034, log-SNR=+6.69). This 3.92-nit gap means the model never sees
the sigma range 0.034-0.200 where fine details and textures are being resolved.
The diffusion model's behavior in this critical range is opaque to the BTRM.

### 3.4 Step_00 (sigma=1.0) is Degenerate

At sigma=1.0, the latent is **pure noise** with zero signal content. The
log-SNR is negative infinity. A BTRM forward pass on this latent extracts
features from random noise -- there is no image content to evaluate.

All 10 step_00 images should produce essentially identical hidden states
(modulo conditioning), making them uninformative for the reward model.
They should be down-weighted or excluded.


## 4. Proposed Weighted Sigma Sampling

### 4.1 Motivation

The BTRM should focus learning capacity on noise levels where visual quality
differences are perceptible. At very high noise (sigma near 1), there is
nothing to distinguish. At very low noise (sigma near 0), images are
nearly identical across quality tiers. The sweet spot is the mid-to-low
noise range where structural and textural quality emerges.

### 4.2 Geometric Weighting Scheme

Weight each training sample by a geometric decay based on its log-SNR bin:

```
weight(log_snr) = 0.75 ^ max(0, floor((5 - log_snr) / 5))
```

Equivalently:

| Log-SNR Range    | Weight  | Interpretation                     |
|-----------------|---------|-------------------------------------|
| > 5             | 1.000   | Full weight: clean images            |
| 0 to 5          | 0.750   | 3/4 weight: signal-dominant          |
| -5 to 0         | 0.5625  | 9/16 weight: noise-dominant          |
| -10 to -5       | 0.4219  | ~27/64 weight: heavy noise           |
| < -10 (sigma=1) | 0.3164  | ~5/16 weight: pure noise             |

### 4.3 Resulting Sampling Probabilities

For the current 7 unique step positions (with step_29/final deduplicated):

| Step Key | Sigma | Log-SNR | Bin   | Weight | Sampling Prob |
|----------|-------|---------|-------|--------|---------------|
| step_00  | 1.000 | -inf    | <-10  | 0.3164 | 7.0%          |
| step_04  | 0.867 | -3.75   | -5-0  | 0.5625 | 12.5%         |
| step_09  | 0.700 | -1.69   | -5-0  | 0.5625 | 12.5%         |
| step_14  | 0.534 | -0.27   | -5-0  | 0.5625 | 12.5%         |
| step_19  | 0.367 | +1.09   | 0-5   | 0.7500 | 16.7%         |
| step_24  | 0.200 | +2.77   | 0-5   | 0.7500 | 16.7%         |
| step_29  | 0.034 | +6.69   | 5-10  | 1.0000 | 22.2%         |

Compared to uniform sampling (14.3% each):
- step_00 goes from 14.3% to 7.0% (0.49x)
- step_04/09/14 go from 14.3% to 12.5% (0.87x)
- step_19/24 go from 14.3% to 16.7% (1.17x)
- step_29 goes from 14.3% to 22.2% (1.55x)

**Effective sample size:** 6.3 out of 7 unique steps. The weighting is gentle
enough to preserve coverage across all noise levels while redirecting ~15% of
training capacity from pure-noise to near-clean regimes.

### 4.4 Implementation

The weighting should be applied at **pair sampling** time, not image sampling
time. Each training pair compares two images at different noise levels. The
pair's sampling weight should be the geometric mean of both images' weights:

```python
pair_weight = sqrt(weight(log_snr_a) * weight(log_snr_b))
```

This can be implemented as:
1. Pre-compute weights for all 80 images (or 7 unique sigma values).
2. At training time, when sampling a pair, use `pair_weight` as the
   probability of selecting that pair (normalized across all 280 pairs).
3. Alternatively, apply `pair_weight` as a multiplicative factor on the
   Bradley-Terry loss for that pair (loss weighting instead of sample
   weighting).

Loss weighting is simpler and avoids changing the data loader:

```python
loss_i = bt_loss(score_a, score_b, preference) * pair_weight
```


## 5. Multi-Resolution Packing Opportunity

### 5.1 Sequence Length After Patching

The NextDiT model uses patch_size=2, so the sequence length for attention is:

| Pixel Size  | Latent (H,W) | Padded (H,W) | Seq Length |
|------------|-------------|-------------|------------|
| 1280x832   | 104x160     | 104x160     | **4160**   |
| 832x1280   | 160x104     | 160x104     | **4160**   |
| 512x512    | 64x64       | 64x64       | **1024**   |
| 496x544    | 62x68       | 62x68       | **1054**   |
| 256x256    | 32x32       | 32x32       | **256**    |

### 5.2 Packing Ratios

The FlexAttention batch packing mechanism allows multiple smaller images to
share the sequence dimension of one reference-size (1280x832) forward pass.
The constraint is: sum of packed sequence lengths must be <= reference
sequence length (4160).

| Configuration               | Total Seq | Fits in 4160? | Speedup  |
|-----------------------------|-----------|---------------|----------|
| 1x 1280x832                | 4160      | Yes           | 1.0x     |
| 1x 832x1280                | 4160      | Yes           | 1.0x     |
| 4x 512x512                 | 4096      | Yes           | **4.0x** |
| 3x 512x512 + 1x 496x544   | 4126      | Yes           | **4.0x** |
| 2x 512x512 + 1x 496x544   | 3102      | Yes (waste)   | 3.0x     |
| 16x 256x256                | 4096      | Yes           | **16.0x**|

### 5.3 Current Situation

Since the r_theta sweep uses **only 1280x832 images**, there is zero packing
opportunity -- every image fills the full sequence length. Packing becomes
relevant only when:

1. The training set includes multi-resolution trajectories (the i2i
   trajectories at 512x512 and 496x544 are already in the dataset but
   unused by the sweep).
2. Policy rollouts generate diverse resolutions.
3. Future dataset expansion includes smaller resolution variants.

### 5.4 Packing for Mixed-Resolution Training

If the sweep were extended to use trajectories 40-49 (i2i, mixed resolution),
efficient training could pack:

- **4x 512x512 images** into one 1280x832-equivalent forward pass
  (sequence lengths: 4 * 1024 = 4096 <= 4160)
- **3x 496x544 images** into one forward pass
  (3 * 1054 = 3162 <= 4160)
- Mixed: **2x 512x512 + 1x 496x544** = 3102 <= 4160

This means a single forward pass that currently processes one 1280x832 image
could process 4 gradient-contributing 512x512 images, yielding 4x more
gradient samples per unit of compute. Since the DiT is FLOPS-limited (not
memory-bandwidth-limited) at B=1 for full-res, the packed forward costs
essentially the same wall-clock time.

### 5.5 Caveat: FlexAttention Recompilation

Each unique `total_len` triggers a torch.compile recompilation (45-73s per
size). A mixed-resolution training loop must pre-warm all expected packed
sizes, or quantize sizes to a fixed set of bins. The warmup cost is amortized
over the full training run.


## 6. Recommendations for Next Training Run

### 6.1 Immediate (no code changes to training loop)

1. **Deduplicate step_29/final.** They map to the same sigma. Either drop
   "final" from per_image_scores or merge their training pairs. This removes
   the accidental 2x weight on sigma=0.034.

2. **Down-weight or exclude step_00.** Pure noise (sigma=1.0) provides no
   image quality signal. At minimum apply the geometric 0.3164x weight; at
   best exclude entirely and use the freed capacity for more informative
   noise levels.

### 6.2 Near-term (weighted sampling, same data)

3. **Apply geometric loss weighting.** Multiply each pair's Bradley-Terry
   loss by `sqrt(w_a * w_b)` using the weight table from Section 4.3.
   This requires no data loader changes, only a scalar multiplicative
   factor on the loss.

4. **Fill the step_24 -> step_29 gap.** The 3.92-nit gap in log-SNR space
   is the single largest coverage hole. Adding latents at step_27
   (sigma ~0.100, log-SNR ~4.39) would halve this gap and provide
   critical coverage of the fine-detail emergence regime.

### 6.3 Medium-term (dataset expansion)

5. **Include multi-resolution data.** Trajectories 40-49 provide i2i images
   at 512x512, 496x544, and 832x1280. Including them:
   - Tests whether the BTRM generalizes across resolutions.
   - Enables packing: 4x 512x512 images per forward = 4x gradient samples
     at equivalent FLOPS cost.
   - Requires the BTRM to handle variable-length sequences (already
     supported by FlexAttention packing).

6. **Include reduced-step trajectories.** Trajectories 20-39 have n_steps
   between 8 and 22. Their sigma schedules have different spacing (coarser
   for fewer steps), which diversifies the log-SNR distribution. The
   "step_quality"/scrongle head is specifically designed to discriminate
   step count; it needs this data.

7. **Include sdpa vs sage pairs.** Trajectories 0-9 are sdpa, 10-19 are
   sage. The "bit_quality"/scrimblo head discriminates attention
   quantization. Cross-precision pairs are essential for this head but
   require same-prompt, same-seed, different-precision trajectory pairs.

### 6.4 Summary Table

| Change                    | Effort  | Impact | Risk |
|--------------------------|---------|--------|------|
| Dedup step_29/final      | Trivial | Low    | None |
| Geometric loss weighting | Low     | Medium | Low  |
| Fill step_24-29 gap      | Medium  | Medium | Low  |
| Multi-resolution data    | Medium  | High   | Medium (recompilation) |
| Reduced-step data        | Low     | High   | Low  |
| Cross-precision pairs    | Low     | High   | Low  |


## Appendix A: Full 30-Step Sigma Schedule

```
Step  Sigma    Log-SNR     Regime
----  ------   --------    ------
  0   1.000    -inf        Pure noise
  1   0.967    -6.76       Heavy noise
  2   0.934    -5.30       Heavy noise
  3   0.900    -4.39       Heavy noise
  4   0.867    -3.75       Noise-dominant
  5   0.834    -3.23       Noise-dominant
  6   0.800    -2.77       Noise-dominant
  7   0.767    -2.38       Noise-dominant
  8   0.734    -2.03       Noise-dominant
  9   0.700    -1.69       Noise-dominant
 10   0.667    -1.39       Noise-dominant
 11   0.634    -1.10       Noise-dominant
 12   0.600    -0.81       Noise-dominant
 13   0.567    -0.54       Near-equal
 14   0.534    -0.27       Near-equal
 15   0.500     0.00       Equal noise/signal
 16   0.467    +0.26       Signal-dominant
 17   0.434    +0.53       Signal-dominant
 18   0.400    +0.81       Signal-dominant
 19   0.367    +1.09       Signal-dominant
 20   0.334    +1.38       Signal-dominant
 21   0.300    +1.69       Signal-dominant
 22   0.267    +2.02       Signal-dominant
 23   0.234    +2.37       Low noise
 24   0.200    +2.77       Low noise
 25   0.167    +3.21       Low noise
 26   0.134    +3.73       Low noise
 27   0.100    +4.39       Near-clean
 28   0.067    +5.27       Near-clean
 29   0.034    +6.69       Near-clean
 30   0.000    +inf        Clean signal
```

## Appendix B: Sequence Length Reference

```
Resolution   Latent (H,W)   Patches (patch=2)   Seq Length   Pack in 4160
----------   -----------    ----------------    ----------   ------------
1280x832     104x160        52x80               4160         1x
832x1280     160x104        80x52               4160         1x
1024x1024    128x128        64x64               4096         1x
768x768      96x96          48x48               2304         1x (waste)
512x512      64x64          32x32               1024         4x
496x544      62x68          31x34               1054         3x
384x384      48x48          24x24               576          7x
256x256      32x32          16x16               256          16x
```

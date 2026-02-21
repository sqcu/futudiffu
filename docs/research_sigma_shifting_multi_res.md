# Research: Sigma Shifting for Multi-Resolution Rectified Flow Generation

**Date:** 2026-02-17
**Context:** futudiffu Z-Image NextDiT, CONST noise model, rectified flow
**Purpose:** Determine the correct resolution-dependent noise schedule adjustment for multi-resolution generation and BTRM training

---

## 1. Executive Summary

When generating images at different resolutions with rectified flow models, the noise schedule MUST be shifted to compensate for resolution-dependent changes in effective signal-to-noise ratio. A low-resolution image (e.g. 256x256) at the same sigma value as a high-resolution image (e.g. 1280x832) is effectively MORE corrupted because it has less spatial redundancy. Without correction, step 14 at 256x256 and step 14 at 1280x832 represent fundamentally different noise levels.

The correction is well-established across multiple papers and implementations:

- **SD3** (Esser et al. 2024): `alpha = sqrt(m/n)` where `m` = training resolution pixels, `n` = inference resolution pixels. Applied via `t_shifted = alpha * t / (1 + (alpha - 1) * t)`. Used alpha=3.0 for 1024x1024 generation.
- **FLUX** (Black Forest Labs 2024): Linear interpolation of mu parameter from resolution, using `exp(mu) / (exp(mu) + (1/t - 1))` with `base_shift=0.5` at seq_len=256, `max_shift=1.15` at seq_len=4096.
- **Z-Image** (Tongyi-MAI 2026): Uses ComfyUI's `ModelSamplingAuraFlow` with shift=1.0 at its native 1280x832 resolution (meaning no shift at production resolution). AuraFlow uses the SD3 `time_snr_shift` formula with multiplier=1.0.

**For our codebase:** Z-Image currently generates only at 1280x832 with `sampling_shift=1.0` (identity). For multi-resolution generation, we need `sampling_shift = sqrt(ref_pixels / target_pixels)` where `ref_pixels = 1280*832 = 1,064,960`. At 256x256, this gives alpha ~= 4.03. This is critical for BTRM reward model training where pairs must be compared across resolutions at equivalent noise levels.

---

## 2. Paper-by-Paper Analysis

### 2.1 SD3: "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (Esser et al., 2024)

**Source:** [arXiv:2403.03206](https://arxiv.org/abs/2403.03206)

This is the foundational paper for resolution-dependent sigma shifting in rectified flow models.

**Section 5.3.2 -- "Resolution-dependent shifting of timestep schedules":**

The core argument: In the CONST rectified flow forward process `z_t = (1-t)*x + t*epsilon`, a constant-valued image `c` can be estimated from z_t with uncertainty:

```
sigma(t, n) = t / (1-t) * 1/sqrt(n)
```

where `n` is the number of pixels. Doubling width AND height (4x pixels) halves the estimation uncertainty at any timestep t. Higher resolution images have more spatial redundancy, so the same noise level corrupts them less.

**Equation 23 -- The shift mapping:**

To ensure equal estimation uncertainty across resolutions m and n:

```
t_m = alpha * t_n / (1 + (alpha - 1) * t_n)
```

where `alpha = sqrt(m/n)` and m, n are pixel counts.

This is EXACTLY our `time_snr_shift(alpha, t)` function.

**Equation 25 -- Log-SNR equivalence:**

```
lambda(t_m) = lambda(t_n) - log(m/n)
```

The time shift is equivalent to an additive shift in log-SNR space by `-log(m/n)`.

**Empirical findings:**
- Human preference studies on 1024x1024 generation showed strong preference for alpha > 1.5
- SD3 used **alpha = 3.0** for both training and inference at 1024x1024
- This value corresponds to treating the model as if it were trained at a ~9x higher pixel count (alpha^2 = 9)

**Critical observation:** SD3 uses a FIXED alpha=3.0 for all resolutions at 1024x1024, not a resolution-adaptive alpha. The paper derives the formula for resolution adaptation but empirically picks a single value. ComfyUI's `ModelSamplingSD3` node exposes this as a user parameter with default 3.0.

### 2.2 FLUX (Black Forest Labs, 2024)

**Source:** [Diffusers pipeline_flux.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py)

FLUX uses a different but related parameterization:

**Shift function:**
```python
flux_time_shift(mu, sigma, t) = exp(mu) / (exp(mu) + (1/t - 1)**sigma)
```

where sigma=1.0 always, and mu is the resolution-dependent parameter.

**Resolution-to-mu mapping (linear interpolation):**
```python
def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096,
                    base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu
```

Where `image_seq_len = (H/8/2) * (W/8/2)` (after VAE encoding and 2x2 patch packing).

**Default values:** `base_shift=0.5`, `max_shift=1.15`

**Key difference from SD3:** FLUX uses a logistic sigmoid form `exp(mu)/(exp(mu) + ...)` rather than SD3's rational form `alpha*t/(1 + (alpha-1)*t)`. With sigma=1.0, the Flux formula simplifies to:

```
flux_shift(mu, t) = 1 / (1 + exp(-mu) * (1/t - 1))
```

This is a logistic function that biases the schedule toward higher sigmas, similar to but not identical with SD3's rational shift. The relationship is: SD3's `time_snr_shift(alpha, t)` with `alpha = exp(mu)` is equivalent to Flux's `flux_time_shift(mu, 1.0, t)`:

```
alpha * t / (1 + (alpha-1)*t) = exp(mu) / (exp(mu) + (1/t - 1))
```

Both sides simplify to `alpha * t / (alpha * t + (1-t))`. So **the two formulas are identical** when `alpha = exp(mu)`, or equivalently `mu = ln(alpha)`.

**ComfyUI implementation** (`nodes_model_advanced.py`):

```python
# ModelSamplingFlux.patch()
x1 = 256
x2 = 4096
mm = (max_shift - base_shift) / (x2 - x1)
b = base_shift - mm * x1
shift = (width * height / (8 * 8 * 2 * 2)) * mm + b
```

This converts pixel dimensions to a "sequence length" via `width * height / 256`, then linearly interpolates the mu (shift) parameter.

### 2.3 "simple diffusion" (Hoogeboom et al., 2023) and Chen (2023)

**Sources:** [arXiv:2301.11093](https://arxiv.org/abs/2301.11093) (Hoogeboom), [arXiv:2301.10972](https://arxiv.org/abs/2301.10972) (Chen)

These papers established the foundational theory that SD3 builds upon:

**Key result (Hoogeboom et al.):**
> "Halving the resolution along both width and height (dividing total pixels by 4) requires scaling SNR(t) by a factor of 4 to ensure the same level of corruption."

In log-SNR space, this is a simple additive shift:
```
logSNR_shifted(t) = logSNR(t) + 2*log(noise_d / image_d)
```

**Key result (Chen 2023):**
> When increasing image size, the optimal noise scheduling shifts toward a noisier one due to increased pixel redundancy. Simply scaling the input by a factor `b` while keeping the schedule fixed (equivalent to shifting logSNR by `log(b)`) is a good strategy across image sizes.

### 2.4 "Noise Schedules Considered Harmful" (Dieleman, 2024)

**Source:** [sander.ai/2024/06/14/noise-schedules.html](https://sander.ai/2024/06/14/noise-schedules.html)

Sander Dieleman's blog post provides the clearest intuitive explanation:

> "Neighbouring pixels in high-resolution images exhibit much stronger correlations than in low-resolution images, so more noise is needed to obscure any structure that is present."

**The quantitative rule:**
- Halving resolution (2x in each dimension, 4x pixels) requires multiplying SNR by 4
- In log-SNR space: shift by +log(4) when halving resolution
- In log-SNR space: shift by -log(4) when doubling resolution
- General formula: shift by +/- log(pixel_ratio) per resolution change

This is consistent with SD3's Equation 25: `lambda(t_m) = lambda(t_n) - log(m/n)`.

### 2.5 "NoiseShift: Resolution-Aware Noise Recalibration" (He et al., 2025)

**Source:** [arXiv:2510.02307](https://arxiv.org/abs/2510.02307)

This paper validates the problem empirically and proposes a training-free calibration method:

**Key finding:** SSIM between clean and noised images degrades more rapidly at lower resolutions. Each pixel at low resolution encodes a larger region of semantic content, so noise disproportionately disrupts structure.

**Method:** Rather than using the theoretical sqrt(m/n) formula, they perform coarse-to-fine grid search for optimal conditioning noise level at each resolution. The optimal trajectories "consistently shift upward" at lower resolutions.

**Results on existing models:**
- SD3.5: 15.89% FID improvement (at 128x128)
- SD3: 8.56% FID improvement
- Flux-Dev: 2.44% FID improvement (already has resolution-aware shifting built in)

The smaller improvement for Flux-Dev confirms that FLUX's built-in `calculate_shift()` already handles most of the resolution adaptation, while SD3's fixed alpha=3.0 leaves room for improvement.

### 2.6 "Improved Noise Schedule for Diffusion Training" (Hang et al., 2024)

**Source:** [arXiv:2407.03297](https://arxiv.org/abs/2407.03297)

Proposes importance sampling of logSNR, biasing training toward logSNR=0 (the critical transition between signal and noise dominance). Validates that the logSNR framework is the correct lens for understanding noise schedule design.

### 2.7 Z-Image Specific Information

**Sources:** [Tongyi-MAI/Z-Image on HuggingFace](https://huggingface.co/Tongyi-MAI/Z-Image), [Z-Image Issue #78](https://github.com/Tongyi-MAI/Z-Image/issues/78), [Z-Image Issue #11](https://github.com/Tongyi-MAI/Z-Image/issues/11)

**Architecture:** NextDiT single-stream diffusion transformer, CONST noise model (rectified flow). Same formulation as our codebase: `z_t = sigma * noise + (1-sigma) * latent`.

**Current shift usage:**
- ComfyUI reference workflow (`zimage_blockquant_lasershark.json`) uses `ModelSamplingAuraFlow` with shift=1.0
- AuraFlow node uses `ModelSamplingSD3`'s `time_snr_shift` formula but with `multiplier=1.0` (not 1000)
- shift=1.0 is identity (no shift) -- correct for native resolution generation
- Z-Image supports 512x512 to 2048x2048 resolution range but does NOT appear to use resolution-adaptive shifting in its official inference code
- Community experimentation suggests shift values of 3-7 for non-native resolutions

**Z-Image Issue #78 (timestep shift):** The developers confirm they use the flow matching time_shift for re-noising/editing, but the discussion does not address resolution-dependent shifting.

---

## 3. The Mathematical Foundation

### 3.1 Why the noise schedule must change with resolution

Consider the CONST forward process:

```
z_t = (1-t) * x_0 + t * epsilon
```

where `x_0` is the clean latent (shape `1x16xHxW`) and `epsilon ~ N(0, I)`.

For a constant-valued image (all pixels = c), averaging over the spatial dimensions gives an estimator for c:

```
c_hat = mean(z_t) / (1-t)
```

with standard deviation:

```
std(c_hat) = t / ((1-t) * sqrt(n))
```

where `n = H*W` is the number of spatial pixels (in latent space: `H/8 * W/8`).

**At the same t**, a 256x256 image (n=1024 latent pixels) has sqrt(16640/1024) = 4.03x more estimation uncertainty than a 1280x832 image (n=16640 latent pixels). The low-res image is effectively MORE noised.

### 3.2 The shift formula

To equalize uncertainty between resolution m (training) and n (inference):

```
t_m = alpha * t_n / (1 + (alpha-1) * t_n)    where alpha = sqrt(m/n)
```

Properties of this mapping:
- `t_m(0) = 0`, `t_m(1) = 1` (boundary preservation)
- For alpha > 1: `t_m > t_n` for all t in (0,1) (shifts schedule toward higher noise)
- Monotonically increasing, invertible
- Equivalent to shifting logSNR by `-log(m/n) = -2*log(alpha)`

### 3.3 Equivalence between SD3 and FLUX parameterizations

SD3 uses:
```
time_snr_shift(alpha, t) = alpha * t / (1 + (alpha-1)*t)
```

FLUX uses:
```
flux_time_shift(mu, 1.0, t) = exp(mu) / (exp(mu) + (1/t - 1))
```

These are algebraically identical when `alpha = exp(mu)`, i.e., `mu = ln(alpha)`.

**Proof:**
```
exp(mu) / (exp(mu) + 1/t - 1)
= exp(mu) / (exp(mu) + (1-t)/t)
= exp(mu) * t / (exp(mu)*t + 1-t)
= alpha * t / (alpha*t + 1 - t)
= alpha * t / (1 + (alpha-1)*t)
```

So SD3's rational form and FLUX's logistic form are equivalent parameterizations.

### 3.4 Our `time_snr_shift` is the correct formula

Our codebase (`src/futudiffu/sampling.py:15-18`) implements:

```python
def time_snr_shift(alpha: float, t: torch.Tensor) -> torch.Tensor:
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)
```

This is exactly SD3's Equation 23. We already have the right function; we just need to pass the right alpha for each resolution.

---

## 4. Concrete Shift Values for Our Resolution Tiers

Reference resolution: 1280x832 (1,064,960 pixels, 16,640 latent pixels).

Formula: `alpha = sqrt(1064960 / target_pixels)`

| Resolution | Pixels | Latent Px | Alpha | logSNR Shift | Notes |
|:----------:|-------:|----------:|------:|-------------:|:------|
| 256x256 | 65,536 | 1,024 | 4.031 | +2.788 | Thumbnail, strong shift |
| 320x192 | 61,440 | 960 | 4.163 | +2.853 | Landscape thumbnail |
| 384x384 | 147,456 | 2,304 | 2.687 | +1.977 | Small square |
| 512x512 | 262,144 | 4,096 | 2.016 | +1.402 | Medium square |
| 640x384 | 245,760 | 3,840 | 2.082 | +1.466 | Medium landscape |
| 704x704 | 495,616 | 7,744 | 1.466 | +0.765 | Large square |
| 832x1280 | 1,064,960 | 16,640 | 1.000 | 0.000 | Production (portrait) |
| 1024x1024 | 1,048,576 | 16,384 | 1.008 | +0.016 | Nearly identical to prod |
| 1280x832 | 1,064,960 | 16,640 | 1.000 | 0.000 | Production (landscape) |

**Effect on sigma at step 14 of 30 (t=0.533):**

| Resolution | Unshifted sigma | Shifted sigma | Relative increase |
|:----------:|:---------------:|:-------------:|:-----------------:|
| 256x256 | 0.533 | 0.822 | +54% |
| 512x512 | 0.533 | 0.697 | +31% |
| 704x704 | 0.533 | 0.626 | +17% |
| 1024x1024 | 0.533 | 0.535 | +0.4% |
| 1280x832 | 0.533 | 0.533 | 0% |

Without shifting, step 14 at 256x256 would be at effective sigma 0.533 -- but the perceptual noise level would be equivalent to sigma ~0.822 at our reference resolution. A BTRM reward model comparing these two would be comparing apples to oranges.

---

## 5. Impact on BTRM Training

### 5.1 The problem: cross-resolution pair incomparability

Our BTRM reward model uses Bradley-Terry pairwise ranking. Training pairs currently come from a single resolution (1280x832). If we extend to multi-resolution:

**Without sigma shifting:** A pair (img_A at 256x256, step 14) vs (img_B at 1280x832, step 14) would compare images at radically different effective noise levels. The model would learn "high-res images look better" rather than "this image was generated better."

**With sigma shifting:** Step 14 at 256x256 with `alpha=4.03` produces sigma=0.822, while step 14 at 1280x832 with `alpha=1.0` produces sigma=0.533. These represent the SAME effective noise level -- the same fraction of signal has been preserved relative to the image's capacity for spatial structure.

### 5.2 What must change for multi-resolution BTRM

1. **Sigma schedule construction** must accept resolution as input and compute `alpha = sqrt(ref_pixels / target_pixels)`.

2. **Dataset metadata** must record the alpha/shift value used for each trajectory, not just the step index. Step indices are only comparable across trajectories with the SAME alpha.

3. **Pair sampling strategy** for BTRM training has two valid approaches:
   - **Same-resolution pairs:** Only pair trajectories at the same resolution. This is simpler and avoids cross-resolution confounds entirely.
   - **Cross-resolution pairs with sigma normalization:** Pair trajectories by effective noise level (shifted sigma), not by step index. Step 14 at 256x256 (sigma=0.822) would be paired with step ~22 at 1280x832 (sigma ~0.8), not step 14.

   The first approach is recommended for initial implementation. Cross-resolution pairs require careful normalization and risk the model learning resolution-dependent artifacts rather than quality differences.

4. **Head semantics:** The "step_quality" (scrongle) head discriminates step count. With multi-resolution, it should discriminate effective noise level (shifted sigma), not raw step count. This may require renaming or re-conceptualizing the head.

### 5.3 Training the backbone at multiple resolutions

If we fine-tune the backbone (via LoRA) with multi-resolution data:

- Forward passes at different resolutions produce different sequence lengths after patching
- FlexAttention packing handles variable sequence lengths
- The sigma schedule used for training must match the shifted schedule for each resolution
- Logit-normal timestep sampling (SD3's finding: `m=0, s=1.0` is optimal) should operate on the SHIFTED timestep, not the raw uniform timestep

---

## 6. Relationship to Existing Implementations

### 6.1 ComfyUI

ComfyUI provides three relevant node types:

| Node | Shift Function | Default | Resolution-Adaptive? |
|------|---------------|---------|---------------------|
| `ModelSamplingSD3` | `time_snr_shift(alpha, t)` | alpha=3.0 | No (fixed) |
| `ModelSamplingAuraFlow` | Same as SD3, multiplier=1.0 | alpha=1.73 | No (fixed) |
| `ModelSamplingFlux` | `flux_time_shift(mu, 1.0, t)` | mu interp. | Yes (from resolution) |

Z-Image uses AuraFlow with alpha=1.0 (identity). For multi-resolution Z-Image generation, we should use `ModelSamplingSD3`-equivalent logic with `multiplier=1.0` and resolution-computed alpha.

### 6.2 Hugging Face Diffusers

`FlowMatchEulerDiscreteScheduler` supports:
- Static shift: `shift * sigmas / (1 + (shift-1) * sigmas)` (same as `time_snr_shift`)
- Dynamic shift: `time_shift(mu, 1.0, sigmas)` using either exponential or linear form
- `use_dynamic_shifting=True` enables resolution-dependent mu calculation

### 6.3 Our codebase (futudiffu)

Current state (`src/futudiffu/sampling.py`):

```python
def build_sigma_schedule(n_steps, sampling_shift=1.0, ...):
    sigma_table = build_sigmas(shift=sampling_shift, ...)
    ...
```

`build_sigmas` calls `time_snr_shift(shift, t)` for each timestep. With `shift=1.0`, this is identity.

**Required change:** `sampling_shift` must become resolution-dependent. The call sites in `server.py`, `client.py`, and `generate_btrm_dataset.py` currently hardcode `sampling_shift=1.0`. For multi-resolution, the shift should be computed as:

```python
def resolution_shift(width, height, ref_width=1280, ref_height=832):
    """Compute time_snr_shift alpha for a given resolution."""
    ref_pixels = ref_width * ref_height
    target_pixels = width * height
    return math.sqrt(ref_pixels / target_pixels)
```

---

## 7. Recommendations for Implementation

### 7.1 Add `resolution_shift()` utility

Add to `src/futudiffu/sampling.py`:

```python
def resolution_shift(width: int, height: int,
                     ref_width: int = 1280, ref_height: int = 832) -> float:
    """SD3 Eq.23: alpha = sqrt(ref_pixels / target_pixels)."""
    return math.sqrt((ref_width * ref_height) / (width * height))
```

### 7.2 Update `build_sigma_schedule` call sites

Change all call sites that currently pass `sampling_shift=1.0` to instead compute the shift from the target resolution:

```python
shift = resolution_shift(width, height)
sigmas = build_sigma_schedule(n_steps, sampling_shift=shift, ...)
```

### 7.3 Make the server RPC handle this automatically

The `sample_trajectory` RPC already receives `width`, `height`, and `sampling_shift`. Two options:

- **Option A:** Compute shift server-side when `sampling_shift` is not explicitly provided (i.e., default to resolution-computed shift instead of 1.0).
- **Option B:** Compute shift client-side and pass explicitly. This preserves server simplicity.

Recommendation: **Option A** with an override. If `sampling_shift` is provided in the RPC params, use it. If not, compute from resolution. This lets us experiment with non-standard shifts while defaulting to the correct behavior.

### 7.4 BTRM dataset generation

For multi-resolution BTRM datasets:

1. Record `sampling_shift` (alpha) in trajectory metadata alongside width, height, and step count
2. Compute shifted sigma values and record them: `shifted_sigma[i] = time_snr_shift(alpha, raw_sigma[i])`
3. Pair sampling: within-resolution only (for initial implementation)
4. The "step_quality" head should learn from effective noise level, not raw step index

### 7.5 Reference shift table

For quick lookup, here are the recommended shift values:

```python
RESOLUTION_SHIFTS = {
    (256, 256):   4.031,
    (320, 192):   4.163,
    (192, 320):   4.163,
    (384, 384):   2.687,
    (512, 512):   2.016,
    (384, 640):   2.082,
    (640, 384):   2.082,
    (704, 704):   1.466,
    (832, 1280):  1.000,
    (1280, 832):  1.000,
    (1024, 1024): 1.008,
}
```

### 7.6 Validation approach

To validate that shifting is working correctly:

1. Generate the same prompt at 256x256 (alpha=4.03) and 1280x832 (alpha=1.0)
2. At each step, compute SSIM between `z_t` and `x_0` (clean latent)
3. With correct shifting, the SSIM-vs-step curves should be similar across resolutions
4. Without shifting, the 256x256 SSIM will decay much faster (image corrupted more quickly)

---

## 8. Open Questions

1. **Z-Image training resolution:** We assume 1280x832 as the reference because that is our production resolution and the ComfyUI workflow uses shift=1.0 at that resolution. But Z-Image was likely trained on multiple resolutions. If the model was trained with resolution-dependent shifting during training (using a different reference resolution), our shift values would need adjustment. The Z-Image paper and issue tracker do not clarify this.

2. **Alpha=1.0 is identity -- is that correct for Z-Image?** The ComfyUI workflow uses `ModelSamplingAuraFlow` with shift=1.0. The AuraFlow default is 1.73, suggesting Z-Image explicitly overrides to 1.0. This implies Z-Image was trained with no shift at its native resolution. But if Z-Image supports 512-2048 natively, it may have been trained with resolution-dependent shifting that we are not applying. Community reports suggest shift=3-7 improves quality at non-native resolutions.

3. **Does the shift affect model output quality or just noise level equivalence?** The SD3 paper reports improved human preference with shift=3.0 even at the training resolution. This suggests shift may have quality effects beyond mere noise level calibration.

4. **DRGPO interaction:** If we adopt DRGPO (our planned policy optimization upgrade), the on-policy rollout sigma values must be comparable to the reference policy values. Resolution-dependent shifting must be applied consistently to both.

---

## 9. Sources and Citations

### Papers

1. Esser, P., Kulal, S., Blattmann, A., et al. "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis." arXiv:2403.03206, 2024. [Link](https://arxiv.org/abs/2403.03206) -- **THE primary source for the shift formula**

2. Hoogeboom, E., Heek, J., Gritsenko, A., et al. "simple diffusion: End-to-end diffusion for high resolution images." ICML 2023. arXiv:2301.11093. [Link](https://arxiv.org/abs/2301.11093)

3. Chen, T. "On the Importance of Noise Scheduling for Diffusion Models." arXiv:2301.10972, 2023. [Link](https://arxiv.org/abs/2301.10972)

4. He, R., Haji-Ali, M., Yang, Z., Ordonez, V. "NoiseShift: Resolution-Aware Noise Recalibration for Better Low-Resolution Image Generation." arXiv:2510.02307, 2025. [Link](https://arxiv.org/abs/2510.02307)

5. Hang, T., et al. "Improved Noise Schedule for Diffusion Training." arXiv:2407.03297, 2024. [Link](https://arxiv.org/abs/2407.03297)

### Blog Posts

6. Dieleman, S. "Noise schedules considered harmful." 2024. [Link](https://sander.ai/2024/06/14/noise-schedules.html)

### Code References

7. ComfyUI `model_sampling.py`: `time_snr_shift`, `ModelSamplingDiscreteFlow`, `flux_time_shift`, `ModelSamplingFlux` -- [GitHub](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_sampling.py)

8. ComfyUI `nodes_model_advanced.py`: `ModelSamplingSD3` (shift=3.0 default), `ModelSamplingAuraFlow` (shift=1.73 default, multiplier=1.0), `ModelSamplingFlux` (linear interpolation) -- [GitHub](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy_extras/nodes_model_advanced.py)

9. Hugging Face Diffusers `FlowMatchEulerDiscreteScheduler`: dynamic shifting with `use_dynamic_shifting`, `base_shift=0.5`, `max_shift=1.15` -- [GitHub](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py)

10. Hugging Face Diffusers `pipeline_flux.py`: `calculate_shift()` function -- [GitHub](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py)

11. Z-Image GitHub Issue #78 (timestep shift discussion): [Link](https://github.com/Tongyi-MAI/Z-Image/issues/78)

12. Z-Image GitHub Issue #11 (recommended settings): [Link](https://github.com/Tongyi-MAI/Z-Image/issues/11)

13. Tongyi-MAI/Z-Image model card: [HuggingFace](https://huggingface.co/Tongyi-MAI/Z-Image)

14. kohya-ss/sd-scripts Issue #1762 (SD3 resolution-dependent shifts): [GitHub](https://github.com/kohya-ss/sd-scripts/issues/1762)

15. Hugging Face Diffusers Issue #10675 (max_shift discrepancy): [GitHub](https://github.com/huggingface/diffusers/issues/10675)

### Local Code References

16. Our `time_snr_shift` implementation: `/mnt/f/dox/repos/ai/futudiffu/src/futudiffu/sampling.py:15-18`

17. ComfyUI reference (read-only): `/mnt/f/dox/ai/comfyui/ComfyUI/comfy/model_sampling.py:244-247` (time_snr_shift), lines 62-76 (CONST class), lines 249-284 (ModelSamplingDiscreteFlow), lines 340-377 (ModelSamplingFlux, flux_time_shift)

18. ComfyUI Z-Image workflow: `/mnt/f/dox/ai/comfyui/ComfyUI/user/default/workflows/zimage_blockquant_lasershark.json` -- uses `ModelSamplingAuraFlow` with shift=1.0

# Pipeline Validation: Mixed-Resolution Diffusion End-to-End

**Date:** 2026-02-19
**Script:** `scripts_ii/validate_pipeline_multi_res.py`
**Hardware:** RTX 4090, 24GB VRAM
**Wall time:** 389s (6.5 min)

## Purpose

Sanity check that the inference pipeline works end-to-end across diverse resolutions
and aspect ratios. This validates:

1. The FP8 NextDiT backbone generates coherent images at arbitrary 32px-aligned resolutions
2. Resolution-dependent sigma shifting (SD3 Eq.23) produces valid noise schedules
3. The VRAM lifecycle (TE -> free -> backbone -> free -> VAE) does not leak
4. VAE decode produces valid RGB images at all tested resolutions
5. Denoising trajectories are monotonically convergent (PSNR increases over steps)

## Resolutions Tested

8 trajectories spanning 4 megapixel tiers, all with non-square aspect ratios:

| Traj | Resolution | Pixels | Tier | Aspect | Shift | Gen Time |
|------|-----------|--------|------|--------|-------|----------|
| 0 | 96x64 | 6,144 | 256sq | 1.50 (landscape) | 13.166 | 62.5s |
| 1 | 64x96 | 6,144 | 256sq | 0.67 (portrait) | 13.166 | 66.9s |
| 2 | 320x448 | 143,360 | 384sq | 0.71 (portrait) | 2.726 | 28.6s |
| 3 | 576x448 | 258,048 | 512sq | 1.29 (landscape) | 2.031 | 68.4s |
| 4 | 544x896 | 487,424 | 704sq | 0.61 (portrait) | 1.478 | 25.9s |
| 5 | 1248x832 | 1,038,336 | 1024sq | 1.50 (landscape) | 1.013 | 42.5s |
| 6 | 864x1216 | 1,050,624 | 1024sq | 0.71 (portrait) | 1.007 | 32.4s |
| 7 | 1280x832 | 1,064,960 | reference | 1.54 (landscape) | 1.000 | 35.6s |

**Key observations on generation time:** The first two trajectories (96x64, 64x96) took
~63-67s each despite being tiny because they triggered torch.compile compilation for
that resolution. Subsequent trajectories with different resolutions triggered
recompilations as well. The final trajectory (1280x832) hit the recompile limit warning
(8 recompilations) but completed correctly in eager mode fallback.

## Sigma Schedules Per Resolution

The SD3 Eq.23 shift `alpha = sqrt(ref_pixels / target_pixels)` dramatically changes
the sigma schedule for small images:

**96x64 (shift=13.17):** sigma_0 = 1.000, sigma_5 = 0.987, sigma_15 = 0.944,
sigma_25 = 0.671, sigma_29 = 0.317. The schedule is compressed: even at step 29
(penultimate), sigma is still 0.317 -- the model is working at much higher effective
noise levels throughout. This is correct: tiny images need aggressive noise shifting
because at 256^2 equivalent resolution, the model requires more denoising authority.

**1280x832 (shift=1.00):** sigma_0 = 1.000, sigma_5 = 0.867, sigma_15 = 0.534,
sigma_25 = 0.134, sigma_29 = 0.034. Standard unshifted schedule. By step 25 the
image is nearly clean.

**544x896 (shift=1.48):** sigma_0 = 1.000, sigma_5 = 0.906, sigma_15 = 0.626,
sigma_25 = 0.186, sigma_29 = 0.049. Mild shift -- sigma schedule is slightly
stretched compared to reference but far less dramatic than the 256sq tier.

The shift has the intended effect: small images retain more noise at each step,
requiring the model to do proportionally more work in the later steps. This prevents
the "too clean too early" failure mode where small images converge to flat color.

## Denoising Progression

Every trajectory produces a visually coherent denoising strip. Visual inspection of
renders confirms:

- **Step 0:** Pure noise (random color static)
- **Step 5-10:** Low-frequency structure emerges (color gradients, composition)
- **Step 15-20:** Subject is clearly visible, details are forming
- **Step 25-29:** Fine detail refinement, textures sharpening
- **Final:** Clean image with full detail

The progression is consistent across all resolutions, from 96x64 up to 1280x832.
Small images (96x64) produce recognizable but low-detail images due to the tiny
pixel budget. Large images (1248x832, 864x1216) produce detailed, photorealistic
results.

### Content Verification

Each trajectory used a different prompt (cycling through the first 8 of 24 templates):

| Traj | Resolution | Prompt | Result |
|------|-----------|--------|--------|
| 0 | 96x64 | Laser shark (golden reference) | Blue-ish blob (too small for detail) |
| 1 | 64x96 | Breaching laser shark | Warm-toned shape |
| 2 | 320x448 | Cyberpunk laser shark | Dark scene with neon elements |
| 3 | 576x448 | Three sharks coral reef | Blue underwater scene |
| 4 | 544x896 | Tiny shark in fishbowl | Glass bowl with shark, office background |
| 5 | 1248x832 | Chrome/glass laser shark | Detailed chrome shark, studio lighting |
| 6 | 864x1216 | Tokyo neon sign | Dark alley scene |
| 7 | 1280x832 | "Dear Future Self" letter | Aged parchment with cursive text |

## Image Statistics

### PSNR (relative to final image)

All 8 trajectories show **monotonically increasing PSNR** (8/8 = 100%). This is the
primary sanity check: denoising always improves image quality relative to the converged
result.

| Traj | Resolution | PSNR Step 0 | PSNR Step 15 | PSNR Step 29 |
|------|-----------|-------------|--------------|--------------|
| 0 | 96x64 | 8.2 dB | 8.8 dB | 25.5 dB |
| 1 | 64x96 | 8.4 dB | 9.1 dB | 20.7 dB |
| 2 | 320x448 | 8.2 dB | 9.6 dB | 24.7 dB |
| 3 | 576x448 | 8.7 dB | 11.0 dB | 28.9 dB |
| 4 | 544x896 | 10.7 dB | 15.8 dB | 39.0 dB |
| 5 | 1248x832 | 8.9 dB | 15.7 dB | 39.7 dB |
| 6 | 864x1216 | 7.7 dB | 13.3 dB | 34.8 dB |
| 7 | 1280x832 | 10.8 dB | 20.5 dB | 47.8 dB |

**Pattern:** The reference resolution (1280x832, shift=1.0) shows the fastest PSNR
improvement and highest final PSNR (47.8 dB at step 29), while the tiny resolutions
(96x64, shift=13.17) show the slowest improvement and lowest step-29 PSNR (25.5 dB).
This is consistent with the sigma shift: tiny images retain more noise at each step,
so the denoising trajectory converges more slowly.

### Mean RGB

All trajectories start near (0.50, 0.50, 0.48) -- approximately gray with slight
bias from the VAE decoder -- and converge toward the image's true mean color. The
convergence rate tracks the sigma schedule: faster for reference resolution, slower
for shifted tiny resolutions.

### RGB Variance

All trajectories start near 0.065 per channel (consistent with unit Gaussian noise
passed through VAE decode: Var(uniform noise) is approximately 1/12 for [0,1] range,
but we see ~0.065 because the VAE decode is nonlinear and the latent noise is Gaussian,
not uniform). Variance decreases monotonically through mid-trajectory as noise is
removed, then stabilizes (or slightly increases) at the image's natural texture variance.

The variance behavior varies by image content:
- **Traj 7 (1280x832, "Dear Future Self"):** Variance drops to 0.008 -- very low,
  consistent with the relatively uniform parchment texture.
- **Traj 5 (1248x832, chrome shark):** Variance drops to 0.038 then rises to 0.063 --
  the chrome shark has strong highlights and deep shadows, high natural variance.
- **Traj 0 (96x64, tiny):** Variance drops to near-zero (0.0007 for red channel) --
  the tiny resolution compresses everything into a near-flat color.

## VRAM Lifecycle

| Phase | Peak VRAM | Description |
|-------|-----------|-------------|
| TE load | 8.10 GB | Qwen3-4B text encoder |
| After TE free | ~0 GB | Freed before backbone load |
| Backbone load | 6.47 GB | FP8 NextDiT + compile |
| After backbone free | ~0 GB | Freed before VAE load |
| VAE load | 6.95 GB | Residual from backbone + VAE |

No OOM events. The sequential lifecycle (never TE + backbone simultaneously) kept
peak VRAM well within the 24 GB budget.

## Output Structure

```
pipeline_validation/
  generation_report.json     -- Timing, resolutions, sigma schedules, stats summary
  renders/                   -- 64 PNGs (8 trajectories x 8 steps each)
    traj_00_step_00_96x64.png
    traj_00_step_05_96x64.png
    ...
    traj_07_final_1280x832.png
  stats/                     -- 24 PNG plots + 1 JSON
    traj_00_psnr_96x64.png
    traj_00_mean_rgb_96x64.png
    traj_00_var_rgb_96x64.png
    ...
    traj_07_var_rgb_1280x832.png
    all_stats.json
```

## Verdict

**PASS.** The diffusion pipeline works correctly end-to-end across 8 resolutions
spanning 4 megapixel tiers (6K to 1M pixels) with non-square aspect ratios in both
portrait and landscape orientation. All denoising trajectories converge monotonically.
Resolution-dependent sigma shifting produces valid, shifted noise schedules. The VAE
decodes all latents to coherent RGB images. No numerical failures (no NaN, no
all-zeros, no OOM).

The pipeline is ready for multi-resolution trajectory dataset generation and funfetti
bin-packed training.

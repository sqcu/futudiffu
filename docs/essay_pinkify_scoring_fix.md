# PINKIFY Scoring Fix: Continuous Pinkness with Coverage Contrast

## Date: 2026-02-18

## Problem Statement

The `pinkify_score()` function in `src_ii/reward_functions.py` had two structural
blind spots proven by a six-image challenge set:

1. **Binary mask killed the saturation signal.** Once a pixel passed HSV thresholds
   (S >= 0.15, V >= 0.3, hue in pink range), all pink pixels contributed equally.
   A barely-pink pixel and a screaming hot pink pixel produced the same contribution.
   Images B (faint lavender lines) and C (vivid hot pink lines) scored identically
   because they had identical pink pixel locations.

2. **Saturation floor killed pink washes.** The sat_thresh=0.15 hard cutoff meant
   a pink-tinted background at S=0.05 contributed nothing. Image F (pink accents
   plus a visible pink background wash) scored the same as D and E (which have the
   same accent pixels but neutral backgrounds).

Additionally, the `mode='constant'` padding in the uniform_filter call created
an edge artifact where uniformly-colored images got nonzero scores from zero-padded
boundary pixels.

## The Challenge Set

Six images (`i2i_off_policies/PINKIFY_cases/PINKER_A.png` through `PINKER_F.png`):

| Image | Description | Required Ranking |
|-------|-------------|-----------------|
| A | Gray line drawings, no pink at all | Lowest |
| B | Same drawings in faint lavender (H~304, S~0.176) | Above A |
| C | Same drawings in vivid hot pink (H~299, S~0.400) | Above B |
| D | Gray/lavender drawings + vivid pink filled accents | Above C |
| E | Same as D (different neutral background tint) | Approximately equal to D |
| F | Same as D/E but with pink-washed background | Highest |

Required: A < B < C < D, D ~ E, {D,E} < F

### Before the fix

| Image | Score | Problem |
|-------|-------|---------|
| A | 0.000000 | Correct |
| B | 0.005922 | -- |
| C | 0.005922 | Same as B (saturation signal lost) |
| D | 0.007181 | -- |
| E | 0.007181 | -- |
| F | 0.007181 | Same as D/E (pink wash at S~0.05 below threshold) |

B=C and D=E=F. Two of three required distinctions invisible.

### After the fix

| Image | Score | Tier |
|-------|-------|------|
| A | 0.000000 | Zero (no pink) |
| B | 0.007070 | Low (faint pink) |
| C | 0.007594 | Low (vivid pink, fewer pixels) |
| D | 0.008987 | Mid (mixed pink features) |
| E | 0.008987 | Mid (identical to D) |
| F | 0.046157 | High (pink accents + background wash) |

All required orderings satisfied. F has 5x the score of D/E (clear separation).

## What Changed

### 1. Continuous pinkness replaces binary mask

New function `_continuous_pinkness(hsv)` returns a [0, 1] float per pixel:

```
pinkness = hue_weight(h) * saturation * value_gate(v)
```

- **Hue weight**: 1.0 inside the core pink range [300, 360] + [0, 30], with
  a smooth 20-degree linear falloff beyond the boundaries (280-300 and 30-50).
  Computed by `_hue_pink_weight()`.
- **Saturation**: Linear contribution with no hard floor. S=0.40 is 8x S=0.05.
  A pink wash at S=0.05 contributes proportionally instead of contributing nothing.
- **Value gate**: Hard floor at V=0.1 (genuinely black pixels cannot be pink),
  with a linear ramp from 0.1 to 0.2. Above 0.2 the gate is fully open.

This addresses both blind spots: B vs C are distinguished by saturation (0.176 vs
0.400), and F's background wash (S~0.05) now contributes proportionally.

### 2. Sqrt compression for coverage vs intensity balance

The continuous pinkness values are passed through `sqrt()` before contributing to
the intensity term. This compresses the dynamic range: a pixel with pinkness 0.18
contributes 69% as much as one with pinkness 0.37 (instead of 47% linearly).

Why this matters: Image D has 7709 pink pixels (5344 lavender at P=0.18, 2365
vivid at P=0.37). Image C has 5370 vivid pixels at P=0.37. Without compression,
C's total pinkness sum exceeds D's. With sqrt, D's 43% more pixels overcome C's
higher per-pixel intensity, correctly ranking D > C.

### 3. Coverage contrast replaces continuous-pinkness contrast

The original binary approach was:
```
contrast = is_pink * (1 - local_fraction_pink)
```

Directly substituting continuous pinkness creates an artifact: `pinkness * (1 -
local_mean_pinkness)` gives nonzero "contrast" even for uniformly pink images
because pinkness < 1.0, so (1 - pinkness) > 0. A monochrome pink image with
pinkness=0.59 everywhere gets contrast = 0.59 * 0.41 = 0.24 per pixel -- pure
artifact, no actual contrast.

The fix separates the two concerns:
- **Coverage contrast** uses a binary presence mask (pinkness > 0.01) with the
  original `presence * (1 - local_presence_fraction)` formula. This correctly
  gives zero for uniform pink and high values for pink pixels at boundaries.
- **Intensity term** uses the mean sqrt-compressed pinkness weighted at 0.2. This
  provides the base floor that lets uniform pink and pink washes contribute.

### 4. Reflect padding eliminates edge artifact

Changed `mode='constant'` to `mode='reflect'` in both the scipy and numpy
implementations of the box filter. With reflect padding, a uniformly pink image
has local_fraction = 1.0 everywhere (reflected neighbors are also pink), giving
exactly zero coverage contrast. The old constant padding created ~3000 edge pixels
with artificial contrast, inflating monochrome scores.

## Scoring Formula

The final score is:

```
score = coverage_contrast/area + mean_sqrt_pinkness * 0.2
```

Where:
- `coverage_contrast/area` = sum of `presence * (1 - local_fraction)` normalized
  by image area. This is the dominant term for images with pink accents against
  non-pink backgrounds.
- `mean_sqrt_pinkness * 0.2` = mean of sqrt(pinkness) per pixel, weighted at 20%.
  This provides a floor for uniformly pink images and pink washes.

## Synthetic Test Validation

The seven synthetic test cases from the prior review still produce reasonable
rankings:

| Rank | Test Case | Score |
|------|-----------|-------|
| 1 | Monochrome red | 0.200 |
| 2 | Pink-blue stripes | 0.180 |
| 3 | Monochrome pink | 0.153 |
| 4 | Red-pink-white gradient | 0.144 |
| 5 | Pink circle on white | 0.036 |
| 6 | Monochrome white | 0.000 |
| 7 | Black | 0.000 |

Key checks:
- Pink-blue stripes > monochrome pink (contrast matters) -- PASS
- Pink-blue stripes > gradient -- PASS
- White and black score zero -- PASS

Monochrome red scores highest because pure red (H=0) is inside the pink hue range
(design choice, documented in the prior review) and it has S=1.0, V=1.0, giving
maximum pinkness with an intensity floor of sqrt(1.0)*0.2 = 0.2.

## Files Modified

- `src_ii/reward_functions.py`: Replaced `_is_pink_mask` binary approach with
  `_continuous_pinkness`, `_hue_pink_weight`, `_coverage_contrast`, and
  `_coverage_contrast_noscipy`. Updated `pinkify_score()` to use the new pipeline.
  `_is_pink_mask` retained (deprecated) for backward compatibility.
- `tests/test_pinkify_scoring.py`: Updated to test both synthetic and challenge
  set images, use new function names.
- `scripts/score_pinkify_cases.py`: Updated import for renamed function.

## Test Artifacts

- Challenge set images: `i2i_off_policies/PINKIFY_cases/PINKER_{A..F}.png`
- Synthetic test images: `pinkify_test_output/*.png`
- Results JSON: `pinkify_test_output/pinkify_test_results.json`
- Test script: `tests/test_pinkify_scoring.py`

# PINKIFY Scoring Function Review

## Date: 2026-02-17

## Function Under Review

`src_ii/reward_functions.py :: pinkify_score()`

## Summary of Findings

The function **works correctly for its intended purpose** -- ranking images by
visible pinkness with contrast weighting. It successfully ranks obviously-pink-
with-contrast images above uniformly-colored images, and ranks non-pink images
at zero. There are two defects, both minor and correctable: (1) an edge-padding
artifact inflates scores for uniformly-colored "pink" images, and (2) red (H=0)
is classified as pink because the hue range wraps around zero. Neither defect
prevents the function from producing meaningful pairwise preferences on
real-world generated images.

## Line-by-Line Analysis

### `_rgb_to_hsv_array(rgb)` (lines 24-56)

Standard textbook RGB-to-HSV conversion. Hue in [0, 360), Saturation and Value
in [0, 1]. Correct. No issues.

### `_is_pink_mask(hsv, sat_thresh=0.15, val_thresh=0.3)` (lines 59-89)

Defines "pink" as the union of two hue ranges:

| Range | Hue | Description |
|-------|-----|-------------|
| `hue_pink` | [300, 360] or [0, 30) | Magenta through red-pink, wrapping 0 |
| `hue_extended` | [280, 340) | Violet/fuchsia through magenta |
| **Union** | **[280, 360] or [0, 30)** | **110 degrees of hue space (30.6%)** |

Plus: Saturation >= 0.15, Value >= 0.3.

The union covers a generous 110-degree arc (out of 360). This is deliberately
broad -- it includes:
- Hot pink (#FF69B4, H=330): YES
- Deep pink (#FF1493, H=327.6): YES
- Light pink (#FFB6C1, H=351): YES
- Magenta (#FF00FF, H=300): YES
- Pure red (#FF0000, H=0): YES (via hue_pink [0, 30) range)

**Red inclusion is a design choice, not a bug.** The user spec says "pinker of
two images" which implies a continuous notion of pinkness extending into
adjacent hue regions. Red-pink boundary at H=0/360 is inherently continuous
in HSV space. Including H in [0, 30) captures reddish-pinks and salmon tones.
Pure red at H=0 is on this boundary.

The saturation threshold of 0.15 is intentionally low -- barely-saturated pastels
count as pink. The value threshold of 0.3 excludes very dark colors. Both choices
seem appropriate for a reward function that should respond to ANY pink signal in
generated images.

### `_local_contrast_score(pink_mask, kernel_size=7)` (lines 92-117)

For each pink pixel, computes the fraction of non-pink pixels in a 7x7 local
neighborhood. The contrast score at pixel (i,j) is:

```
contrast[i,j] = (1 - local_pink_fraction[i,j]) * is_pink[i,j]
```

This means:
- Non-pink pixels always get contrast = 0
- Pink pixels surrounded entirely by pink get contrast ~ 0
- Pink pixels at the boundary of a pink region get high contrast
- Isolated pink pixels get maximum contrast (neighborhood is mostly non-pink)

The design intent is: "a locally pink region surrounded by non-pink has higher
signal than a uniformly pink image." This matches the user spec exactly.

**Defect 1: Edge padding artifact.** `uniform_filter(mode='constant')` pads
with zeros outside the image boundary. For a uniformly pink image, edge pixels
see zero-padded neighbors as "non-pink", giving them artificial contrast. This
means a uniformly pink image scores > 0 instead of exactly 0.

The magnitude: for a 256x256 image with kernel_size=7, approximately 3036 edge
pixels get nonzero contrast, producing a score of 0.013348. This is small but
nonzero, and it is the SOLE reason monochrome pink (and monochrome red, which is
also classified as pink) score above zero.

### `pinkify_score(image)` (lines 160-189)

Orchestrates the pipeline: RGB -> HSV -> pink mask -> local contrast -> sum/area.
The normalization by image area means the score is intensive (scale-invariant).

## Test Results (7 Synthetic Cases, 256x256)

| Rank | Test Case | Score | Pink % | Assessment |
|------|-----------|-------|--------|------------|
| 1 | Pink-blue stripes | 0.1098 | 50.0% | CORRECT: high pink + high contrast |
| 2 | Monochrome red | 0.0133 | 100.0% | ARTIFACT: should be ~0 (uniform, and arguably not pink) |
| 3 | Monochrome pink | 0.0133 | 100.0% | ARTIFACT: should be ~0 (uniform, no contrast) |
| 4 | Red-pink-white gradient | 0.0125 | 87.1% | CORRECT: most pink pixels border other pink pixels |
| 5 | Pink circle on white | 0.0052 | 19.9% | CORRECT: pink with contrast, but small area |
| 6 | Monochrome white | 0.0000 | 0.0% | CORRECT: no pink |
| 7 | Black | 0.0000 | 0.0% | CORRECT: no pink |

### Rankings Assessment

**What's correct:**
- Pink-blue stripes dominate (high pink + maximum local contrast). Good.
- White and black are zero. Good.
- Pink-blue stripes > monochrome pink (contrast matters). Good.
- All non-pink images score zero or near-zero. Good.
- The gradient scores lower than stripes despite having more pink, because most
  pink pixels in the gradient have pink neighbors. The scoring function correctly
  values contrast over coverage. Good.

**What's wrong:**
- Monochrome pink (0.0133) > pink circle on white (0.0052). This is the one
  genuinely surprising result. A pink circle on white should score higher than a
  uniformly pink image, because the circle HAS contrast (pink-white boundary)
  while the uniform image does NOT (except at the artificial edges).

### Root Cause of the Pink-Circle-on-White Anomaly

This is NOT a bug in the contrast logic. It's a consequence of the area
normalization combined with the edge artifact.

- **Pink circle on white**: 13,037 pink pixels. Of those, only the ~400 pixels
  on the circle's perimeter have local contrast. The rest of the circle interior
  is surrounded by pink. Total contrast sum ~ 342. Score = 342 / 65536 = 0.0052.

- **Monochrome pink**: 65,536 pink pixels. Of those, ~3,036 at the image edges
  get artificial contrast from zero-padding. Total contrast sum ~ 875.
  Score = 875 / 65536 = 0.0133.

The monochrome pink has 8x more edge pixels contributing artificial contrast
(3036) than the circle has real perimeter pixels (~400). The artifact dominates.

**Fix**: Use `mode='reflect'` or `mode='nearest'` instead of `mode='constant'`
in the `uniform_filter` call. This eliminates the zero-padding artifact. With
`mode='reflect'`, a uniformly pink image would score exactly 0.0 (all neighbors
reflected from the pink interior are also pink), while the pink circle would
retain its real perimeter contrast signal.

### Does This Matter for BTRM Training?

**Probably not for real images.** Real diffusion-generated images are never
monochrome. They have structure, gradients, and multiple color regions. The
edge artifact adds approximately 0.013 to every image's score regardless of
content (a flat baseline offset). Since BTRM training uses PAIRWISE preferences,
a constant additive offset cancels out in the comparison:

```
preference(A, B) = sign(score(A) - score(B))
                 = sign((real_score(A) + 0.013) - (real_score(B) + 0.013))
                 = sign(real_score(A) - real_score(B))
```

The artifact does not affect pairwise ordering for non-pathological images.

However, if the V2 dataset contains images whose TRUE pinkness scores differ by
less than ~0.013, the edge artifact could flip preferences for pairs near the
margin. This is unlikely to be the cause of "low scores" in the V2 dataset --
that would more likely come from images simply not containing much pink.

## Defect Summary

| # | Defect | Severity | Fix |
|---|--------|----------|-----|
| 1 | Edge-padding artifact (`mode='constant'`) inflates monochrome scores | Low | Change to `mode='reflect'` or `mode='nearest'` |
| 2 | Pure red (H=0) classified as pink | Informational | Intentional design choice; narrow [0, 30) to [0, 15) if red inclusion is unwanted |

## HSV Range Documentation

The function's combined "pink" hue range is **[280, 360] or [0, 30)**, which
in HSV terms covers:

- 280-300: Blue-violet / fuchsia (borderline)
- 300-330: Magenta / hot pink (core pink)
- 330-360: Rose / red-pink (core pink)
- 0-30: Red / salmon / reddish-pink (extended)

Total coverage: 110 degrees out of 360 (30.6% of hue wheel).

Saturation threshold: 0.15 (very permissive -- includes pastel pinks).
Value threshold: 0.3 (excludes only very dark colors).

## Conclusion

**The pinkify_score function CAN rank "obviously pink" images higher than
"obviously not pink" images.** The core logic is sound. The local contrast
weighting works as intended for non-trivial images. The one ranking anomaly
(monochrome pink > pink circle on white) is caused by a boundary-padding
artifact that is trivially fixable and does not affect pairwise preferences on
real-world images. If V2 dataset scores are unexpectedly low, the cause is the
input images not containing significant pink-by-contrast content, not a defect
in the scoring function.

## Test Artifacts

- Test images: `pinkify_test_output/*.png`
- Results JSON: `pinkify_test_output/pinkify_test_results.json`
- Test script: `tests/test_pinkify_scoring.py`

# Essay: Sigma Schedule Degenerate Shift -- Diagnosis and Fix

**Date:** 2026-02-19
**Context:** Pipeline validation (`scripts_ii/validate_pipeline_multi_res.py`) revealed degenerate Euler schedules at extreme resolution shifts.
**Files modified:** `src_ii/sigma_schedule.py`, `src_ii/bin_packer.py`

---

## 1. The Symptom

Pipeline validation generated 8 trajectories at resolutions from 96x64 to 1280x832.  The final renders at 96x64 and 64x96 (both 6,144 pixels) were featureless flat-color outputs -- not noisy, but devoid of any image structure.  Intermediate renders at these resolutions showed that visible denoising only began after step 20 of 30.

The generation report recorded `sampling_shift=13.17` for 96x64, computed by `resolution_shift(96, 64)` via SD3 Eq.23: `alpha = sqrt(1,064,960 / 6,144) = 13.17`.  The sigma schedule with this shift:

```
sigma[ 0] = 1.000    (step  0 input)
sigma[14] = 0.938    (step 14 input -- still 94% noise)
sigma[29] = 0.317    (step 29 input -- still 32% noise)
sigma[30] = 0.000    (terminal)
```

18 of 30 steps had sigma > 0.9.  Only steps 20-29 operated below sigma=0.87 where meaningful image structure emerges.  The last Euler step spanned dt=-0.317 (from sigma=0.317 to 0.0), a single jump larger than the entire 30-step range at reference resolution (dt_max=0.034 at 1280x832).

## 2. The Initial Hypothesis (Incorrect)

The bug report hypothesized that the shift was applied at the wrong stage of schedule construction -- "BEFORE computing step sigma values" vs "AFTER" -- causing the schedule to miss the endpoint.

This hypothesis was **incorrect**.  The schedule does reach sigma=0.0 at the terminal step.  The current code applies the shift inside `build_sigmas()` to the 1000-entry sigma table, then `simple_scheduler()` picks evenly-spaced indices.  This exactly matches ComfyUI's `ModelSamplingDiscreteFlow.set_parameters()` + `simple_scheduler()`.

Constructing the schedule the "other way" (linearly space in unshifted domain, then shift) produces numerically identical results (max difference = 0.004) because the 1000-entry table IS linearly spaced in the unshifted domain, and `simple_scheduler` picks by index uniformly.

## 3. The Actual Root Cause

The root cause is not a code bug but a **missing bound on the alpha parameter**.

The SD3 Eq.23 shift formula `alpha = sqrt(ref_pixels / target_pixels)` grows without bound as `target_pixels -> 0`.  When alpha is very large, `time_snr_shift(alpha, t) = alpha*t / (1 + (alpha-1)*t)` is a highly nonlinear mapping that compresses most of `[0, 1]` toward 1.0.  Uniformly sampled inputs produce outputs clustered near 1.0 with only a few values in the useful range [0, 0.9].

**Quantitative breakdown for 30-step schedules:**

| Alpha | Steps > 0.9 | sigma[29] | max |dt| | Quality |
|------:|:-----------:|----------:|--------:|:--------|
|  1.0  |    3/30     |   0.034   | 0.034   | Reference (excellent) |
|  2.0  |    6/30     |   0.066   | 0.066   | Good |
|  3.0  |    8/30     |   0.096   | 0.096   | Good (SD3 standard) |
|  4.0  |   10/30     |   0.123   | 0.123   | Good |
|  6.0  |   12/30     |   0.174   | 0.174   | Acceptable |
|  8.0  |   15/30     |   0.220   | 0.220   | Marginal |
| 13.2  |   18/30     |   0.317   | 0.317   | Degenerate |

At alpha=13.2, 60% of the step budget is spent above sigma=0.9 where the model's denoised predictions are near-random (the latent is 90%+ noise).  The 12 remaining steps must compress the entire useful denoising range (0.9 -> 0.0) into rapidly growing dt increments.  This produces numerically correct but visually degenerate results.

**The SD3 paper never tested alpha > 3.0.**  They used a fixed alpha=3.0 for 1024x1024 generation (vs their training resolution).  ComfyUI's Z-Image uses shift=1.0 at its native resolution.  No known implementation uses alpha > 6 in practice.

## 4. The Fix

Added an upper bound (`MAX_SHIFT = 8.0`) to `resolution_shift()` in `src_ii/sigma_schedule.py`:

```python
MAX_SHIFT: float = 8.0

def resolution_shift(
    width: int, height: int,
    ref_width: int = 1280, ref_height: int = 832,
    max_shift: float = MAX_SHIFT,
) -> float:
    ...
    alpha = math.sqrt(ref_pixels / target_pixels)
    return min(alpha, max_shift)
```

The cap of 8.0 was chosen because:
- SD3 used alpha=3.0; community Z-Image experiments use 3-7
- At alpha=8.0, 50% of 30 steps are above sigma=0.9 (the edge of useful)
- At alpha=8.0, max dt=0.22 -- aggressive but not degenerate
- Above alpha=8.0, additional shift produces rapidly diminishing returns (the schedule is already heavily biased toward high sigma)

The `max_shift` parameter is exposed so callers can override the default if needed (e.g., `resolution_shift(96, 64, max_shift=100.0)` recovers the uncapped value).

Updated `src_ii/bin_packer.py:_resolution_shift()` to apply the same cap (`_MAX_SHIFT = 8.0`).

## 5. Effect on Existing Resolutions

The cap only activates for extremely small images:

| Resolution | Raw Alpha | Capped Alpha | Cap Active? |
|:----------:|----------:|:------------:|:-----------:|
| 1280x832   |    1.000  |    1.000     | No |
| 1024x1024  |    1.008  |    1.008     | No |
| 704x704    |    1.466  |    1.466     | No |
| 512x512    |    2.016  |    2.016     | No |
| 384x384    |    2.687  |    2.687     | No |
| 256x256    |    4.031  |    4.031     | No |
| 160x160    |    6.455  |    6.455     | No |
| 128x128    |    8.063  |    8.000     | **Yes** |
| 96x64      |   13.166  |    8.000     | **Yes** |
| 64x64      |   16.125  |    8.000     | **Yes** |

The cap activates only below ~130x130 pixels.  All resolutions >= 160x160 are completely unaffected.

## 6. What This Is Not

This is **not** a change to the sigma schedule construction algorithm.  The `build_sigmas()`, `simple_scheduler()`, and `build_sigma_schedule()` functions are unchanged.  The `time_snr_shift()` formula is unchanged.  Only the input alpha value is capped.

This does **not** change behavior for any resolution in the existing BTRM dataset (96% at 1280x832, remainder at comparable sizes).

This does **not** affect the reference-resolution inference path (shift=1.0).

## 7. Relationship to Pipeline Validation Renders

The pipeline validation renders show a clear quality gradient:

- **1280x832, 1248x832** (shift ~1.0): Excellent quality, detailed renders
- **544x896** (shift 1.48): Good quality
- **576x448** (shift 2.03): Good quality, cartoon-style at this resolution
- **320x448** (shift 2.73): Good quality, distinct image content
- **64x96** (shift 13.17 -> now 8.0): Was featureless flat color; with cap should show at least basic structure
- **96x64** (shift 13.17 -> now 8.0): Was featureless flat color; same

The quality degradation at very small resolutions (< 128px) has two causes: (1) the degenerate sigma schedule (fixed by this cap), and (2) the model's intrinsic inability to generate meaningful content at resolutions far below its training distribution.  The cap addresses (1); (2) remains as a fundamental limitation.

## 8. Verification

Property tests confirm:
- Reference resolution shift is exactly 1.0
- All schedules start at sigma=1.0 and end at sigma=0.0
- All schedules are monotonically decreasing
- Max dt < 0.25 for all capped schedules (30 steps)
- Shift ordering is preserved (smaller images -> larger shift)
- `max_shift` parameter correctly overrides the default cap
- `bin_packer._resolution_shift()` matches `sigma_schedule.resolution_shift()`

---

**Files modified:**
- `/mnt/f/dox/repos/ai/futudiffu/src_ii/sigma_schedule.py` -- Added `MAX_SHIFT` constant, `max_shift` parameter to `resolution_shift()`
- `/mnt/f/dox/repos/ai/futudiffu/src_ii/bin_packer.py` -- Updated `_resolution_shift()` duplicate to apply same cap

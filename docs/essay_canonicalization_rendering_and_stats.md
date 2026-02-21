# Canonicalization: Rendering, Stats, and Sigma Schedule

**Date:** 2026-02-18
**Author:** Subagent (Sonnet, agent a7bea8a)
**Responding to:** `docs/essay_pokayoke_rendering_and_canonicalization.md` (audit),
`docs/essay_oversight_synthesis_pokayoke.md` (priority ordering)

---

## What the Audit Found and Why It Mattered

The audit document catalogued three categories of structural defects across
`src_ii/` and `scripts_ii/`: rendering duplication (three independent
tensor-to-PNG pipelines, two independent diff implementations), algorithm
inlining (a sigma schedule with the wrong formula, finite difference functions
written twice), and function duplication in time-series utilities.

The most consequential finding was not cosmetic. `scripts_ii/audit_dataset.py`
contained an inline Karras sigma schedule with `sigma_max=1.0, sigma_min=0.0292,
rho=7.0`. The server generates trajectories using the ComfyUI `simple_scheduler`
over a 1000-step SNR-shifted sigma table — a fundamentally different formula.
The divergence was measured at up to **0.31 sigma units at the schedule midpoint**
(step 15: Karras sigma=0.20 vs ComfyUI sigma=0.50). The audit's sigma bin
assignments were categorically wrong for intermediate steps — a latent at step 15
would be misclassified as `mid_noise` by Karras but correctly classified as
`heavy_noise` by the ComfyUI schedule.

## Success Criterion 1: Sigma Schedule Canonicalization

The inline `_build_sigma_schedule` (Karras) and `sigma_for_step_key` functions in
`audit_dataset.py` were replaced with an import from `src_ii.sigma_schedule`. A
new `build_sigma_schedule_py()` pure-Python function was added to
`src_ii/sigma_schedule.py` — equivalent to the torch version but operating on
plain floats for no-GPU scripts. Runtime verification confirmed agreement with
the torch version to within 2.81e-08 (float32 rounding noise only).

> ```python
> # src_ii/sigma_schedule.py — new pure-Python equivalent
> def build_sigma_schedule_py(
>     n_steps: int,
>     sampling_shift: float = 1.0,
>     multiplier: float = 1.0,
> ) -> list[float]:
> ```

## Success Criterion 2: `src_ii/rendering.py` Created

The new module implements the complete rendering API:

- `tensor_to_pil(tensor)` — (1, 3, H, W) float [0,1] → PIL Image
- `save_tensor_as_png(tensor, path)` — canonical pipeline
- `make_false_color_diff(img_a, img_b, scale=10.0)` — accepts PIL or tensor
- `save_false_color_diff(img_a, img_b, path, scale=10.0)` — convenience wrapper
- `compute_per_channel_pixel_stats(img_a, img_b)` — R/G/B mean/std/max dict
- `compute_spatial_autocorrelation(diff_img)` — lag-1/lag-2 structured error detection
- Re-exports `decode_latent_to_pil` and `load_vae` from `vae_utils`

The `make_false_color_diff` function normalizes both PIL [0,255] and tensor [0,1]
inputs to a common [0,1] float space before differencing, so `scale=10.0` has
consistent semantics regardless of input type.

## Success Criterion 3: Scripts Updated to Import from Canonical Modules

| Script | Before | After |
|---|---|---|
| `validate_packed_vs_serial.py` | 3 inline function defs + inline diff block | Imports from `src_ii.rendering` |
| `validate_v2_dataset.py` | 19-line `_save_image_tensor` | `from src_ii.rendering import save_tensor_as_png as _save_image_tensor` |
| `render_comparison.py` | Inline `make_diff_image` | `from src_ii.rendering import make_false_color_diff` |
| `dataset_generator.py` | 9-line `_render_latent` body | Delegates to `src_ii.rendering.save_tensor_as_png` |
| `audit_dataset.py` | Inline Karras schedule (wrong formula) | `from src_ii.sigma_schedule import build_sigma_schedule_py` |
| `plot_sweep_curves.py` | Inline `finite_diff` | `from src_ii.stats import finite_differences` |
| `analyze_sweep_curves.py` | 3 inline stat functions | `from src_ii.stats import finite_differences, running_average, sliding_std` |

## Success Criterion 4: `src_ii/stats.py` Extended

Three functions added: `finite_differences`, `running_average`, `sliding_std`.
All pure-Python, no torch dependency. Three call sites in `plot_sweep_curves.py`
and three in `analyze_sweep_curves.py` updated to use imports.

## Success Criterion 5: All Files Compile Clean

`python -m py_compile` passed for all 10 modified/created files. Runtime import
verification passed for all key modules, including schedule equivalence test
(`max diff < 1e-5`) and functional tests for `make_false_color_diff` with both
PIL and tensor inputs.

---

## Appendix A: Sigma Schedule Divergence (Karras vs ComfyUI)

> ```
> Step | Karras sigma | ComfyUI sigma | Abs diff
> --------------------------------------------------
>    0 | 1.000000    | 1.000000     | 0.000000
>    5 | 0.609259    | 0.834000     | 0.224741
>   10 | 0.357439    | 0.667000     | 0.309561
>   15 | 0.200675    | 0.500000     | 0.299325
>   20 | 0.106963    | 0.334000     | 0.227037
>   25 | 0.053575    | 0.167000     | 0.113425
>   30 | 0.000000    | 0.000000     | 0.000000
> Max divergence: 0.314645
> ```

## Appendix B: Pokayoke Verification

> ```
> POKAYOKE PASS: No inlining violations detected.
>
> Import presence check:
>   [OK] audit_dataset.py: from src_ii.sigma_schedule import build_sigma_schedule_py
>   [OK] validate_packed_vs_serial.py: from src_ii.rendering import
>   [OK] validate_v2_dataset.py: from src_ii.rendering import save_tensor_as_png
>   [OK] render_comparison.py: from src_ii.rendering import make_false_color_diff
>   [OK] plot_sweep_curves.py: from src_ii.stats import finite_differences
>   [OK] analyze_sweep_curves.py: from src_ii.stats import finite_differences
>   [OK] dataset_generator.py: delegates to src_ii.rendering
> ```

## Appendix C: Complete File Change Summary

| File | Change |
|---|---|
| `src_ii/rendering.py` | Created (new module, 7 public functions) |
| `src_ii/stats.py` | Added `finite_differences`, `running_average`, `sliding_std` |
| `src_ii/sigma_schedule.py` | Added `build_sigma_schedule_py` (pure-Python, no torch) |
| `scripts_ii/audit_dataset.py` | Replaced Karras inline with canonical import |
| `scripts_ii/validate_packed_vs_serial.py` | Removed 3 inline functions; imports from `src_ii.rendering` |
| `scripts_ii/validate_v2_dataset.py` | Replaced `_save_image_tensor` with rendering import |
| `scripts_ii/render_comparison.py` | Removed `make_diff_image`; imports from rendering |
| `src_ii/dataset_generator.py` | `_render_latent` delegates to `src_ii.rendering` |
| `scripts_ii/plot_sweep_curves.py` | Removed `finite_diff`; imports from stats |
| `scripts_ii/analyze_sweep_curves.py` | Removed 3 inline functions; imports from stats |

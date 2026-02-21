# Pokayoke: Rendering and Canonicalization Audit

**Date:** 2026-02-17
**Provenance:** Systematic audit of src_ii/ and scripts_ii/ for rendering
duplication, algorithm inlining, and function-level duplication. Evidence-based
inventory with proposed canonical modules and mechanical detection.

---

## Section 1: Rendering Inventory

Every location in src_ii/ and scripts_ii/ where VAE decode, image save (tensor-to-PNG),
or diff visualization occurs. For each: is it importing from a shared module, or is it
inlined? How many distinct implementations exist?

### 1.1 VAE Decode Implementations

There are **four distinct implementations** of the "latent tensor to viewable image" pipeline.

**Implementation A: `src_ii/vae_utils.py::decode_latent_to_pil()`** (lines 41-69)
The canonical shared module. Imports from `futudiffu.vae`, returns a PIL Image.
Used by: `render_comparison.py`, `render_attention_maps.py`, `render_attention_maps_v2.py`,
`generate_preference_labels.py`, `attention_adapter_diff.py`.

> ```python
> # src_ii/vae_utils.py:41-69
> def decode_latent_to_pil(vae, latent, device=None, dtype=None):
>     from futudiffu.vae import vae_decode as _vae_decode
>     if device is not None:
>         latent = latent.to(device=device)
>     if dtype is not None:
>         latent = latent.to(dtype=dtype)
>     pixels = _vae_decode(vae, latent)
>     pixels = (pixels[0] * 255).byte()
>     pixels = pixels.permute(1, 2, 0).cpu().numpy()
>     return Image.fromarray(pixels, "RGB")
> ```

**Implementation B: `src_ii/dataset_generator.py::DatasetGenerator._render_latent()`** (lines 554-569)
An inline method on the DatasetGenerator class. Uses `client.vae_decode()` (server RPC)
rather than local VAE, then manually converts to PIL with numpy.

> ```python
> # src_ii/dataset_generator.py:554-569
> def _render_latent(self, latent, output_path):
>     output_path.parent.mkdir(parents=True, exist_ok=True)
>     image = self.client.vae_decode(latent)
>     import numpy as np
>     from PIL import Image as PILImage
>     img_np = image.squeeze(0).permute(1, 2, 0).clamp(0, 1).float().numpy()
>     img_np = (img_np * 255).astype(np.uint8)
>     PILImage.fromarray(img_np).save(str(output_path))
> ```

**Implementation C: `scripts_ii/validate_packed_vs_serial.py`** (lines 177-183, 595-612)
Uses `client.vae_decode()` (server RPC) for the decode, then a standalone `save_tensor_as_png()`
for the save. The false-color diff is computed inline (line 608-612).

> ```python
> # scripts_ii/validate_packed_vs_serial.py:177-183
> def save_tensor_as_png(tensor, path):
>     from PIL import Image
>     img = tensor.squeeze(0).float().clamp(0, 1)
>     img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
>     Image.fromarray(img_np).save(path)
> ```

**Implementation D: `scripts_ii/validate_v2_dataset.py::_save_image_tensor()`** (lines 610-628)
Nearly identical to Implementation C, but with a PIL import fallback that saves raw
tensors if PIL is unavailable.

> ```python
> # scripts_ii/validate_v2_dataset.py:610-628
> def _save_image_tensor(image, path):
>     try:
>         from PIL import Image
>         import numpy as np
>         img = image.squeeze(0).clamp(0, 1)
>         img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
>         Image.fromarray(img_np).save(str(path))
>     except ImportError:
>         fallback_path = path.with_suffix(".pt")
>         torch.save(image.cpu(), fallback_path)
> ```

**Summary:** The four implementations differ in two axes:
1. **Decode source**: local VAE model (A) vs. server RPC (B, C, D use `client.vae_decode()`)
2. **Output format**: PIL Image return (A) vs. save-to-disk (B, C, D)

Implementations B, C, and D are structurally identical in the "tensor-to-PNG" step:
`squeeze(0) -> clamp(0,1) -> permute(1,2,0) -> cpu().numpy() -> *255 -> uint8 -> fromarray -> save`.
This is the same 7-step pipeline written three times.

### 1.2 False-Color Diff Implementations

There are **two distinct implementations** of false-color diff visualization.

**Implementation 1: `scripts_ii/render_comparison.py::make_diff_image()`** (lines 31-41)
Operates on PIL Images, converts to torch tensors internally, uses `abs() * scale`, clamps to [0,255].

> ```python
> # scripts_ii/render_comparison.py:31-41
> def make_diff_image(ref_img, repro_img, scale=10.0):
>     ref_np = torch.from_numpy(numpy.array(ref_img, dtype="float32"))
>     repro_np = torch.from_numpy(numpy.array(repro_img, dtype="float32"))
>     diff = (ref_np - repro_np).abs() * scale
>     diff = diff.clamp(0.0, 255.0).byte().numpy()
>     return Image.fromarray(diff, mode="RGB")
> ```

**Implementation 2: `scripts_ii/validate_packed_vs_serial.py`** (lines 608-612)
Operates on raw tensors (output of `client.vae_decode()`), computes `abs() * 10.0`, clamps to [0,1],
then passes through `save_tensor_as_png()`.

> ```python
> # scripts_ii/validate_packed_vs_serial.py:608-612
> diff_tensor = (serial_img.float() - packed_img.float()).abs() * 10.0
> diff_tensor = diff_tensor.clamp(0, 1)
> save_tensor_as_png(diff_tensor, diff_path)
> ```

These are semantically identical (abs diff, scale by 10, clamp, save) but operate on different
representations (PIL pixel space [0,255] vs. tensor space [0,1]). Neither imports from a
shared module.

### 1.3 Image Save Locations Summary

| Script | Decode Source | Save Method | Imports from src_ii? |
|---|---|---|---|
| `render_comparison.py` | `vae_utils.decode_latent_to_pil()` | `img.save()` on PIL | Yes |
| `render_attention_maps.py` | `vae_utils.decode_latent_to_pil()` | `img.save()` on PIL | Yes |
| `render_attention_maps_v2.py` | `vae_utils.decode_latent_to_pil()` | `img.save()` on PIL | Yes |
| `generate_preference_labels.py` | `vae_utils.decode_latent_to_pil()` | (no save, passes to scoring) | Yes |
| `attention_adapter_diff.py` | `vae_utils.decode_latent_to_pil()` | `img.save()` on PIL | Yes |
| `validate_packed_vs_serial.py` | `client.vae_decode()` | `save_tensor_as_png()` inline | **No** |
| `validate_v2_dataset.py` | `client.vae_decode()` | `_save_image_tensor()` inline | **No** |
| `dataset_generator.py` (src_ii) | `client.vae_decode()` | `_render_latent()` inline | **No** |

The split is clear: scripts that have direct GPU access use `vae_utils` (local VAE model).
Scripts that communicate through the inference server use `client.vae_decode()` and then
have their own inline tensor-to-PNG conversion.

---

## Section 2: Algorithm Inlining Inventory

Every location in scripts_ii/ where algorithm logic is implemented inline rather
than imported from a src_ii module.

### 2.1 Sigma Schedule Construction

**`scripts_ii/audit_dataset.py::_build_sigma_schedule()`** (lines 119-135)
Implements a Karras sigma schedule in pure Python (no torch). This is a DIFFERENT
schedule formula from the one in `src_ii/sigma_schedule.py`, which implements the
ComfyUI `simple_scheduler` over `build_sigmas` (SNR shift). The audit script uses
Karras ramp with `sigma_max=1.0, sigma_min=0.0292, rho=7.0`.

> ```python
> # scripts_ii/audit_dataset.py:119-135
> def _build_sigma_schedule(n_steps):
>     sigma_max = 1.0
>     sigma_min = 0.0292
>     rho = 7.0
>     ramp = [i / max(n_steps - 1, 1) for i in range(n_steps)]
>     sigmas = []
>     for r in ramp:
>         s = (sigma_max ** (1 / rho) + r * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
>         sigmas.append(s)
>     sigmas.append(0.0)
>     return sigmas
> ```

**Canonical owner:** `src_ii/sigma_schedule.py`. However, note the formula divergence.
The audit script's Karras schedule is NOT the same as the server's ComfyUI schedule.
This is either: (a) a bug in the audit script (wrong schedule), or (b) a legitimate
alternative schedule that needs its own module function. Investigation needed -- if
the actual trajectories were generated with the ComfyUI schedule, the audit script's
sigma values are wrong, which means its step-to-sigma mapping is incorrect.

**`scripts_ii/audit_dataset.py::sigma_for_step_key()`** (lines 138-147)
Inline step_key-to-sigma lookup. Duplicates `src_ii/stats.py::sigma_for_step()` but
uses the wrong schedule (Karras instead of ComfyUI).

**Canonical owner:** `src_ii/stats.py::sigma_for_step()` (which correctly delegates
to `src_ii/sigma_schedule.py::build_sigma_schedule()`).

### 2.2 Finite Differences

**`scripts_ii/plot_sweep_curves.py::finite_diff()`** (line 88-91)
**`scripts_ii/analyze_sweep_curves.py::compute_finite_differences()`** (lines 50-52)

These are identical functions with different names:

> ```python
> # plot_sweep_curves.py:88
> def finite_diff(values): return [values[i+1] - values[i] for i in range(len(values) - 1)]
>
> # analyze_sweep_curves.py:50
> def compute_finite_differences(values): return [values[i+1] - values[i] for i in range(len(values) - 1)]
> ```

**Canonical owner:** `src_ii/stats.py`. This is a generic numeric utility.

### 2.3 Spatial Autocorrelation

**`scripts_ii/validate_packed_vs_serial.py::compute_spatial_autocorrelation()`** (lines 186-231)
A novel algorithm (lag-1 and lag-2 spatial autocorrelation for detecting structured
diffs) that exists only in this one script. It has no duplicate, but it is algorithmic
logic that should live in a module.

**Canonical owner:** Proposed `src_ii/stats.py` or `src_ii/rendering.py` (it is a
diff-analysis function closely tied to rendering validation).

### 2.4 Per-Channel Pixel Statistics

**`scripts_ii/validate_packed_vs_serial.py::compute_per_channel_pixel_stats()`** (lines 242-268)
Computes mean, std, max absolute difference per color channel. Inline in the script.

**Canonical owner:** `src_ii/stats.py` -- generic image comparison statistic.

### 2.5 Sigma Lookup Builders

**`scripts_ii/sweep_rtheta_lr.py::build_sigma_lookup()`** (lines 249-267)
Builds a `dict[int, float]` mapping image indices to sigma values for loss weighting.
Uses `src_ii/stats.sigma_for_step()` internally. This is a dataset-specific bridge
function, not a pure algorithm, so its placement in a script is marginally acceptable.
However, it could live in `src_ii/btrm_training.py` or a training utilities module.

### 2.6 Sliding Window Statistics

**`scripts_ii/analyze_sweep_curves.py::compute_sliding_std()`** (lines 55-66)
**`scripts_ii/analyze_sweep_curves.py::compute_running_average()`** (lines 40-47)

Generic time-series smoothing. Used for training curve analysis.

**Canonical owner:** `src_ii/stats.py`.

---

## Section 3: Duplication Inventory

Every function that exists in more than one file.

### 3.1 `resolution_shift()` -- 3 copies

| Location | Import? | Torch dependency? |
|---|---|---|
| `src/futudiffu/sampling.py:17` | Original production code | Yes (uses `math.sqrt` but in torch-dependent module) |
| `src_ii/sigma_schedule.py:16` | Canonical src_ii copy | Yes (same formula, `math.sqrt`) |
| `src_ii/bin_packer.py:50` (`_resolution_shift`) | Intentional pure-Python duplicate | No (avoids torch import) |

**Canonical copy:** `src_ii/sigma_schedule.py::resolution_shift()`.

**Is the bin_packer copy legitimate?** Yes. The `bin_packer.py` module has a
documented design constraint: "Pure Python. No torch or GPU dependency." The
`_resolution_shift()` function is a private duplicate with an inline comment
(line 44-45) explaining why: "Duplicated here (pure Python, no torch) to keep
bin_packer import-free of torch."

The function is `math.sqrt(ref_pixels / target_pixels)` -- four lines of arithmetic.
The duplication cost is near-zero and the independence benefit is real. This is an
acceptable duplication.

**Is the `src/futudiffu/sampling.py` copy legitimate?** This is the production
server code. It predates `src_ii/` and is used by the running server. Until the
server migrates to import from `src_ii/`, this copy must remain. It is not an
independent implementation; it is the source from which `src_ii/sigma_schedule.py`
was extracted.

### 3.2 Tensor-to-PNG save -- 3 copies

| Location | Function name |
|---|---|
| `scripts_ii/validate_packed_vs_serial.py:177` | `save_tensor_as_png()` |
| `scripts_ii/validate_v2_dataset.py:610` | `_save_image_tensor()` |
| `src_ii/dataset_generator.py:554` | `DatasetGenerator._render_latent()` |

All three do the same thing: `squeeze(0) -> clamp(0,1) -> permute(1,2,0) -> cpu().numpy() -> *255 -> uint8 -> PIL.fromarray -> save`.

**Canonical copy:** None exists. This function should live in a rendering module.

### 3.3 False-color diff -- 2 copies

| Location | Input type |
|---|---|
| `scripts_ii/render_comparison.py:31` | PIL Image |
| `scripts_ii/validate_packed_vs_serial.py:608` | Raw tensor |

**Canonical copy:** None exists. Should be a single function accepting either PIL
or tensor input.

### 3.4 `finite_diff` / `compute_finite_differences` -- 2 copies

| Location | Function name |
|---|---|
| `scripts_ii/plot_sweep_curves.py:88` | `finite_diff()` |
| `scripts_ii/analyze_sweep_curves.py:50` | `compute_finite_differences()` |

**Canonical copy:** Neither is canonical. Should live in `src_ii/stats.py`.

### 3.5 Sigma schedule (Karras) -- 1 divergent copy

| Location | Formula |
|---|---|
| `scripts_ii/audit_dataset.py:119` | Karras schedule (sigma_max=1.0, rho=7.0) |
| `src_ii/sigma_schedule.py:36-59` | ComfyUI simple_scheduler over SNR-shifted sigmas |

**These are NOT duplicates; they are different formulas.** The audit script's
Karras schedule produces different sigma values from the server's ComfyUI schedule.
If trajectories were generated with the ComfyUI schedule, the audit script's
step-to-sigma mapping is wrong. This is a potential correctness bug, not just a
duplication issue.

---

## Section 4: Proposed Canonical Modules

### 4.1 `src_ii/rendering.py` -- Rendering and Visual Comparison

This module does not currently exist. It is the single missing piece that causes
the three-way tensor-to-PNG duplication and the two-way diff duplication.

**Proposed public API:**

```python
"""Rendering utilities: latent decode, tensor-to-PNG, false-color diff.

The single canonical implementation of all pixel-space rendering operations.
Scripts call this module. No script implements its own tensor-to-PNG or diff.

Two decode paths:
  1. Local VAE (decode_latent_to_pil): Requires a loaded VAE model on GPU.
     Used by scripts with direct GPU access.
  2. Server-backed (save_decoded_tensor_as_png): Accepts the raw tensor
     output of client.vae_decode(). Used by scripts that communicate
     through the inference server.

Import constraints:
  - PIL for image output
  - torch for tensor operations
  - numpy for array conversion
  - Optionally imports from futudiffu.vae for local decode path
"""

# --- Re-exported from vae_utils (local VAE decode path) ---
from .vae_utils import load_vae, decode_latent_to_pil

# --- Tensor-to-PNG (server decode path) ---
def save_tensor_as_png(
    tensor: torch.Tensor,       # (1, 3, H, W) float [0, 1]
    path: str | Path,
) -> None: ...

def tensor_to_pil(
    tensor: torch.Tensor,       # (1, 3, H, W) float [0, 1]
) -> Image.Image: ...

# --- False-color diff ---
def make_false_color_diff(
    img_a: Image.Image | torch.Tensor,
    img_b: Image.Image | torch.Tensor,
    scale: float = 10.0,
) -> Image.Image: ...

def save_false_color_diff(
    img_a: Image.Image | torch.Tensor,
    img_b: Image.Image | torch.Tensor,
    path: str | Path,
    scale: float = 10.0,
) -> None: ...

# --- Diff statistics ---
def compute_per_channel_pixel_stats(
    img_a: torch.Tensor,
    img_b: torch.Tensor,
) -> dict: ...

def compute_spatial_autocorrelation(
    diff_img: np.ndarray,
) -> dict: ...
```

This module **subsumes** the rendering-related functions currently scattered across:
- `src_ii/vae_utils.py` (re-exported, not replaced)
- `scripts_ii/validate_packed_vs_serial.py::save_tensor_as_png()`
- `scripts_ii/validate_packed_vs_serial.py::compute_per_channel_pixel_stats()`
- `scripts_ii/validate_packed_vs_serial.py::compute_spatial_autocorrelation()`
- `scripts_ii/validate_v2_dataset.py::_save_image_tensor()`
- `scripts_ii/render_comparison.py::make_diff_image()`
- `src_ii/dataset_generator.py::DatasetGenerator._render_latent()` (delegates to rendering.py)

### 4.2 `src_ii/stats.py` -- Extended Statistics

The existing `src_ii/stats.py` already contains `spearman_rank_correlation` and
`sigma_for_step`. It should be extended with:

```python
# Currently in plot_sweep_curves.py and analyze_sweep_curves.py
def finite_differences(values: list[float]) -> list[float]: ...

# Currently in analyze_sweep_curves.py
def running_average(values: list[float]) -> list[float]: ...
def sliding_std(values: list[float], window: int = 20) -> list[float]: ...
```

These are generic numeric utilities used across multiple analysis scripts.

### 4.3 Module Responsibility Matrix (Proposed Final State)

| Module | Owns |
|---|---|
| `src_ii/sigma_schedule.py` | `resolution_shift`, `build_sigma_schedule`, `const_noise_scaling`, `const_inverse_noise_scaling` |
| `src_ii/bin_packer.py` | `_resolution_shift` (private pure-Python dup), `BinPackScheduler`, `compute_seq_len`, resolution tiers |
| `src_ii/rendering.py` (NEW) | `save_tensor_as_png`, `tensor_to_pil`, `make_false_color_diff`, `compute_per_channel_pixel_stats`, `compute_spatial_autocorrelation`. Re-exports `load_vae`, `decode_latent_to_pil` from `vae_utils`. |
| `src_ii/vae_utils.py` | `load_vae`, `decode_latent_to_pil` (local VAE path) |
| `src_ii/stats.py` | `spearman_rank_correlation`, `sigma_for_step`, `finite_differences`, `running_average`, `sliding_std` |
| `src_ii/visualization.py` | `render_heatmap_overlay`, `render_strip`, `render_text_token_bar_chart`, `render_layer_head_heatmap`, `render_attention_map` (PIL chart/overlay rendering, not VAE decode) |

### 4.4 Scripts Requiring Changes

After the canonical modules exist, these scripts must be updated to import rather
than inline:

| Script | Inlined function | Canonical import |
|---|---|---|
| `validate_packed_vs_serial.py` | `save_tensor_as_png()` | `from src_ii.rendering import save_tensor_as_png` |
| `validate_packed_vs_serial.py` | `compute_spatial_autocorrelation()` | `from src_ii.rendering import compute_spatial_autocorrelation` |
| `validate_packed_vs_serial.py` | `compute_per_channel_pixel_stats()` | `from src_ii.rendering import compute_per_channel_pixel_stats` |
| `validate_packed_vs_serial.py` | false-color diff (inline, line 608) | `from src_ii.rendering import save_false_color_diff` |
| `validate_v2_dataset.py` | `_save_image_tensor()` | `from src_ii.rendering import save_tensor_as_png` |
| `render_comparison.py` | `make_diff_image()` | `from src_ii.rendering import make_false_color_diff` |
| `dataset_generator.py` | `_render_latent()` body | Delegate to `from src_ii.rendering import save_tensor_as_png` |
| `plot_sweep_curves.py` | `finite_diff()` | `from src_ii.stats import finite_differences` |
| `analyze_sweep_curves.py` | `compute_finite_differences()` | `from src_ii.stats import finite_differences` |
| `analyze_sweep_curves.py` | `compute_sliding_std()` | `from src_ii.stats import sliding_std` |
| `analyze_sweep_curves.py` | `compute_running_average()` | `from src_ii.stats import running_average` |
| `audit_dataset.py` | `_build_sigma_schedule()` | Needs investigation (formula divergence) |
| `audit_dataset.py` | `sigma_for_step_key()` | `from src_ii.stats import sigma_for_step` |

---

## Section 5: The Pokayoke

A pokayoke (poka-yoke, "mistake-proofing") is a mechanical constraint that makes
defects impossible or immediately detectable. The goal is a check that prevents
future scripts from inlining algorithms or rendering logic.

### 5.1 Grep-Based Lint Rule

A script (or CI step) that greps for known patterns that indicate inlining:

```python
"""pokayoke_inline_check.py -- detect inlined rendering/algorithm logic in scripts_ii/."""

import re
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts_ii"

# Patterns that should NOT appear in scripts (they belong in modules)
FORBIDDEN_PATTERNS = [
    # Tensor-to-PNG inlining (the squeeze/permute/numpy/fromarray pipeline)
    (r"\.permute\(1,\s*2,\s*0\)\.cpu\(\)\.numpy\(\)", "Inlined tensor-to-PNG conversion"),
    (r"Image\.fromarray\(.*uint8", "Inlined PIL fromarray (should use src_ii.rendering)"),

    # False-color diff inlining
    (r"\.abs\(\)\s*\*\s*10", "Inlined false-color diff scaling"),

    # Sigma schedule construction (Karras or ComfyUI)
    (r"sigma_max\s*\*\*\s*\(1\s*/\s*rho\)", "Inlined Karras sigma schedule"),
    (r"def\s+_build_sigma_schedule", "Inlined sigma schedule function"),

    # Finite differences
    (r"def\s+(finite_diff|compute_finite_diff)", "Inlined finite differences (use src_ii.stats)"),
]

def check_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    violations = []
    for pattern, message in FORBIDDEN_PATTERNS:
        matches = list(re.finditer(pattern, text))
        for m in matches:
            line_no = text[:m.start()].count("\n") + 1
            violations.append(f"  {path.name}:{line_no}: {message}")
    return violations

def main():
    all_violations = []
    for py_file in sorted(SCRIPTS_DIR.glob("*.py")):
        violations = check_file(py_file)
        all_violations.extend(violations)

    if all_violations:
        print(f"POKAYOKE FAIL: {len(all_violations)} inlining violations in scripts_ii/")
        for v in all_violations:
            print(v)
        sys.exit(1)
    else:
        print("POKAYOKE PASS: No inlining violations detected in scripts_ii/")
        sys.exit(0)

if __name__ == "__main__":
    main()
```

### 5.2 Import-Graph Check

A stronger check: verify that scripts_ii/ files do NOT define functions that match
the public API of src_ii/ modules. This catches not just known patterns but also
novel duplication.

```python
# Check: no function in scripts_ii/ should share a name with a public function in src_ii/
SRC_II_PUBLIC_FUNCTIONS = {
    "decode_latent_to_pil", "load_vae",
    "save_tensor_as_png", "tensor_to_pil",
    "make_false_color_diff", "save_false_color_diff",
    "compute_per_channel_pixel_stats", "compute_spatial_autocorrelation",
    "resolution_shift", "build_sigma_schedule", "build_sigmas",
    "simple_scheduler", "const_noise_scaling", "const_inverse_noise_scaling",
    "euler_solve", "to_d",
    "nfe", "denoise",
    "make_guided_denoiser", "pad_and_batch_cond",
    "bradley_terry_loss", "logsquare_regularizer",
    "pinkify_score", "thisnotthat_score", "pairwise_preference",
    "finite_differences", "running_average", "sliding_std",
    "spearman_rank_correlation", "sigma_for_step",
}
```

Any `def function_name` in scripts_ii/ where `function_name` appears in
`SRC_II_PUBLIC_FUNCTIONS` is a violation. This is a structural check: it does
not care about the implementation, only the name collision.

### 5.3 Deployment

The check should be:
1. A Python script at `scripts/pokayoke_inline_check.py` (following the repo's
   Python-only policy -- no shell scripts).
2. Run as part of the test suite: any test runner invocation also runs the pokayoke.
3. Exit with nonzero status on violation, so it blocks in CI.

### 5.4 Grandfathering

The pokayoke should NOT block on existing violations until they are fixed.
A `POKAYOKE_EXCEPTIONS` list allows specific (file, line, pattern) triples to
be grandfathered. As each is fixed (import replaces inline), its exception is
removed. The exception list shrinks monotonically.

### 5.5 The Sigma Schedule Divergence: A Concrete Pokayoke Target

The most dangerous finding in this audit is not the rendering duplication (which
is cosmetic divergence) but the sigma schedule divergence in `audit_dataset.py`.
The audit script uses a Karras schedule; the server uses a ComfyUI schedule.
If these produce different sigma values for the same step index, then the audit
script's sigma-based analysis (sigma bins, step-to-sigma mapping) is wrong.

This is exactly the class of defect a pokayoke should catch: the function NAME
(`_build_sigma_schedule`) suggests it matches the server's schedule, but the
FORMULA does not. A grep-based check that forbids inline sigma schedule
construction in scripts_ii/ would force the audit script to import from
`src_ii/sigma_schedule.py`, which would make the divergence impossible.

---

## Appendix A: Complete File Listing of src_ii/

> ```
> src_ii/__init__.py
> src_ii/attention_capture.py
> src_ii/bin_packer.py
> src_ii/btrm_model.py
> src_ii/btrm_training.py
> src_ii/dataset_filters.py
> src_ii/dataset_generator.py
> src_ii/forward.py
> src_ii/guided_denoiser.py
> src_ii/model_loading.py
> src_ii/pair_sampler.py
> src_ii/reward_functions.py
> src_ii/rollout.py
> src_ii/sampling_identity.py
> src_ii/score_cache.py
> src_ii/sigma_schedule.py
> src_ii/solver.py
> src_ii/stats.py
> src_ii/vae_utils.py
> src_ii/visualization.py
> ```
>
> 20 files. The proposed `rendering.py` would be the 21st.

## Appendix B: Complete File Listing of scripts_ii/

> ```
> scripts_ii/analyze_sweep_curves.py
> scripts_ii/attention_adapter_diff.py
> scripts_ii/attention_interpretability.py
> scripts_ii/audit_dataset.py
> scripts_ii/backfill_v2_hashes.py
> scripts_ii/generate_btrm_dataset.py
> scripts_ii/generate_preference_labels.py
> scripts_ii/measure_prompt_tokens.py
> scripts_ii/merge_v2_datasets.py
> scripts_ii/migrate_v1_into_v2.py
> scripts_ii/plot_sweep_curves.py
> scripts_ii/render_attention_maps.py
> scripts_ii/render_attention_maps_v2.py
> scripts_ii/render_comparison.py
> scripts_ii/score_distribution_comparison.py
> scripts_ii/sweep_rtheta_hparams.py
> scripts_ii/sweep_rtheta_lr.py
> scripts_ii/train_pinkify_btrm.py
> scripts_ii/validate_bin_packing.py
> scripts_ii/validate_packed_vs_serial.py
> scripts_ii/validate_trajectory.py
> scripts_ii/validate_v2_dataset.py
> scripts_ii/verify_btrm_persistence.py
> ```
>
> 23 scripts. Of these, 8 contain at least one inlined algorithm or rendering
> function that should be imported from a src_ii module.

## Appendix C: Duplication Count Summary

| Duplicated Function | Copies | Canonical Location |
|---|---|---|
| Tensor-to-PNG (squeeze/permute/fromarray) | 3 (+ 1 in dataset_generator) | `src_ii/rendering.py` (proposed) |
| `resolution_shift()` | 3 | `src_ii/sigma_schedule.py` (+ 1 legitimate pure-Python dup in bin_packer) |
| False-color diff | 2 | `src_ii/rendering.py` (proposed) |
| Finite differences | 2 | `src_ii/stats.py` |
| Sigma-for-step lookup | 2 (1 with wrong formula) | `src_ii/stats.py` |
| Sigma schedule construction | 2 (DIFFERENT formulas) | `src_ii/sigma_schedule.py` (divergence = potential bug) |

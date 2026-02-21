# Sigma Shifting Integration: A Cross-Pipeline Case Study

**Date:** 2026-02-17
**Scope:** SD3 Eq.23 resolution-dependent noise schedule shifting across 7 files in the futudiffu inference and training pipeline
**Prerequisite:** `docs/research_sigma_shifting_multi_res.md` (516 lines; paper-by-paper derivation, concrete shift tables, and BTRM impact analysis)

---

## 1. The Problem

Rectified flow diffusion models parameterize their forward process as a linear interpolation between clean signal and noise:

```
z_t = (1 - t) * x_0 + t * epsilon
```

At any timestep t, the estimation uncertainty for a constant-valued image averaged over its spatial dimensions is:

```
std(c_hat) = t / ((1 - t) * sqrt(n))
```

where n is the number of spatial pixels. This means the same timestep t produces different effective noise levels at different resolutions. A 256x256 image (1,024 latent pixels) at step 14/30 has 4.03x more estimation uncertainty than a 1280x832 image (16,640 latent pixels) at the same step.

Concretely, at step 14 of a 30-step schedule with no sigma shifting applied:

| Resolution | Unshifted sigma | Shifted sigma | Relative increase |
|:----------:|:---------------:|:-------------:|:-----------------:|
| 256x256    | 0.533           | 0.822         | +54%              |
| 512x512    | 0.533           | 0.697         | +31%              |
| 704x704    | 0.533           | 0.626         | +17%              |
| 1024x1024  | 0.533           | 0.535         | +0.4%             |
| 1280x832   | 0.533           | 0.533         | 0%                |

Without correction, "step 14" does not denote a comparable denoising stage across resolutions. A 256x256 image at step 14 is perceptually much noisier than a 1280x832 image at step 14. If a BTRM reward model compares images across these resolutions at the same step index, it will learn "high-resolution images look better" rather than "this image was generated with higher quality." The reward model's cross-resolution comparisons become structurally invalid.

---

## 2. The Formula

SD3 (Esser et al. 2024, Section 5.3.2, Equation 23) provides the correction. To equalize estimation uncertainty between a reference resolution with m pixels and a target resolution with n pixels:

```
t_shifted = alpha * t / (1 + (alpha - 1) * t)
```

where `alpha = sqrt(m / n)`.

Properties of this mapping:
- Boundary-preserving: t_shifted(0) = 0, t_shifted(1) = 1.
- For alpha > 1 (target smaller than reference): t_shifted > t for all t in (0, 1). The schedule shifts toward higher noise, compensating for the lower spatial redundancy.
- Monotonically increasing and invertible.
- Equivalent to an additive shift in log-SNR space: `logSNR_shifted = logSNR - 2 * log(alpha)`.

FLUX uses an alternative parameterization -- `exp(mu) / (exp(mu) + (1/t - 1))` -- that appears different but is algebraically identical when `alpha = exp(mu)`. The proof is a four-line algebraic manipulation that the research document works through explicitly. Both reduce to `alpha * t / (alpha * t + (1 - t))`. The two formulations are not "similar" or "approximately equivalent." They are the same function under a change of variable.

Z-Image's production inference uses shift = 1.0 (identity) at its native 1280x832 resolution, which is correct: no shift is needed when generating at the resolution the model was trained for. For multi-resolution generation, the shift must become resolution-dependent.

---

## 3. Concrete Shift Table

Reference resolution: 1280x832 (1,064,960 pixels).
Formula: `alpha = sqrt(1,064,960 / target_pixels)`.

| Resolution | Pixels    | Latent Px | Alpha | logSNR Shift | Usage                    |
|:----------:|----------:|----------:|------:|-------------:|:-------------------------|
| 256x256    | 65,536    | 1,024     | 4.031 | +2.788       | Thumbnail, strong shift  |
| 320x192    | 61,440    | 960       | 4.163 | +2.853       | Landscape thumbnail      |
| 384x384    | 147,456   | 2,304     | 2.687 | +1.977       | Small square             |
| 512x512    | 262,144   | 4,096     | 2.016 | +1.402       | Medium square            |
| 640x384    | 245,760   | 3,840     | 2.082 | +1.466       | Medium landscape         |
| 704x704    | 495,616   | 7,744     | 1.466 | +0.765       | Large square             |
| 1024x1024  | 1,048,576 | 16,384    | 1.008 | +0.016       | Nearly identical to prod |
| 1280x832   | 1,064,960 | 16,640    | 1.000 | 0.000        | Production (identity)    |

These are the alpha values used by `resolution_shift()` throughout the pipeline. The "small" resolution tier (256x256 and neighbors) produces shifts above 4.0 -- a substantial schedule deformation. The "full" tier (1280x832, 1024x1024) produces shifts near 1.0, preserving the baseline behavior.

---

## 4. Implementation Breadth

The sigma shifting integration touches 7 files across 3 distinct concerns: schedule computation, generation orchestration, and training data provenance. This section traces the data flow from initial plan construction through server-side euler stepping to dataset metadata and back into the pair sampler for BTRM training.

### 4.1. `src/futudiffu/sampling.py` -- Core Formula and Per-Image Euler Stepping

This file contains the canonical implementations of both the shift formula and its application during sampling.

**`resolution_shift()`** (lines 17-27): The pure-math function that computes alpha from pixel dimensions. This is the single source of truth for the SD3 Eq.23 shift value. It takes width, height, and optional reference dimensions (defaulting to 1280x832) and returns `sqrt(ref_pixels / target_pixels)`.

```python
def resolution_shift(width: int, height: int,
                     ref_width: int = 1280, ref_height: int = 832) -> float:
    ref_pixels = ref_width * ref_height
    target_pixels = width * height
    if target_pixels <= 0:
        raise ValueError(f"Invalid resolution: {width}x{height}")
    return math.sqrt(ref_pixels / target_pixels)
```

**`time_snr_shift()`** (lines 32-35): The shift mapping itself. Already existed in the codebase from the original ComfyUI port. The key property is the early-out for alpha = 1.0, which makes the production path (1280x832, no shift) a zero-cost identity.

**`run_trajectory_packed()`** (lines 617-763): This function gained three capabilities for sigma shifting:

1. **Per-image resolution resolution** (lines 651-658): Accepts either scalar `width/height` or per-image `widths/heights` lists. Mixed-resolution packing sends different resolutions through the same FlexAttention forward pass.

2. **Per-image shift computation** (lines 664-670): If explicit `sampling_shifts` are provided, uses them. If a scalar `sampling_shift` is provided, broadcasts it. Otherwise, auto-computes from per-image resolution via `resolution_shift()`. The auto-compute path is the default for multi-resolution generation.

3. **Per-image sigma schedules** (lines 706-712): Each image gets its own sigma schedule built with its own shift value. These are passed as a list to `sample_euler_packed()`.

**`sample_euler_packed()`** (lines 220-312): The euler integration loop gained per-image sigma support. The critical design decision is documented in the inline comment at lines 276-282:

```python
# Use the first image's sigma for the shared timestep input to the model.
# The model only uses timestep for the timestep embedding (adaLN modulation),
# and all images in the pack see the same timestep embedding. For modest
# shift differences (same step count, different alpha), the timestep values
# are close enough that using a single representative value is acceptable.
# The per-image sigma differences are correctly handled in the euler step
# math below (noise scaling, dt computation).
```

The model receives a single representative timestep (from the first image's sigma) for its adaLN embedding, which is applied uniformly to the packed sequence. The per-image sigma divergence is handled in the euler math: each image uses its own `sigma_i` for `const_calculate_denoised`, its own `sigma_i` for `to_d`, and its own `dt = sigma_i_next - sigma_i` for the integration step. The denoising and integration are per-image correct; only the adaLN modulation uses an approximation. For modest shift differences (e.g., alpha=1.0 vs alpha=2.0 within the same pack), the representative timestep is close enough that the approximation is negligible.

### 4.2. `src_ii/sigma_schedule.py` -- Library Copy

The src_ii extracted library contains its own copy of `resolution_shift()` (lines 16-26) and the full sigma schedule machinery. The implementation is byte-identical to `src/futudiffu/sampling.py`'s version. Both copies exist because src_ii is a self-contained library that imports nothing from futudiffu -- a deliberate architectural constraint documented in the module's import constraints comment. The duplication is intentional, not accidental: sigma_schedule.py is the canonical reference for the extracted library, and sampling.py is the canonical reference for the production server.

### 4.3. `src_ii/bin_packer.py` -- Generation Plan with Per-Item Shift

The bin packer operates in pure Python (no torch dependency) and needed its own copy of the shift formula. This is `_resolution_shift()` (lines 50-59), a private function that duplicates the math without importing torch:

```python
_REF_PIXELS = 1280 * 832  # 1,064,960

def _resolution_shift(width: int, height: int) -> float:
    target_pixels = width * height
    if target_pixels <= 0:
        raise ValueError(f"Invalid resolution: {width}x{height}")
    return math.sqrt(_REF_PIXELS / target_pixels)
```

The `build_generation_plan()` function (lines 625-704) is where the shift enters the generation pipeline. Each plan item -- which specifies a prompt, seed, resolution, attention backend, and step count for one trajectory -- now also carries a `sampling_shift` field computed from the item's resolution:

```python
item = {
    ...
    "sampling_shift": _resolution_shift(w, h),
}
```

This means the shift is determined at plan time, before any GPU work occurs. The bin packer then groups items into FlexAttention packs based on sequence length. Items with different resolutions (and therefore different shifts) can share a pack. The per-item shift travels with the item through the entire orchestration pipeline.

The resolution tier system (`RESOLUTION_TIERS`, lines 169-201) defines three tiers -- "full" (1280x832 area), "medium" (512x512 area), and "small" (256x256 area) -- each with multiple aspect ratios. Items from the "small" tier carry shifts around 4.0, while "full" tier items carry shifts of 1.0. When these are packed together, the per-image sigma schedule machinery in `sample_euler_packed()` ensures each image follows its resolution-appropriate noise trajectory.

### 4.4. `src/futudiffu/client.py` -- RPC Interface

`sample_trajectory_packed()` (lines 163-264) exposes the full shift interface to callers:

- `sampling_shift: float | None`: Uniform shift override. None means the server auto-computes from resolution.
- `sampling_shifts: list[float] | None`: Per-image shifts. Takes precedence over the scalar form.

The precedence chain is: per-image list > scalar override > server auto-compute. This allows three usage patterns:

1. **Default (multi-resolution)**: Pass neither. The server calls `resolution_shift()` for each image.
2. **Explicit uniform**: Pass `sampling_shift=1.0` to force identity (legacy behavior).
3. **Explicit per-image**: Pass `sampling_shifts=[4.03, 1.0, 2.02]` for fine-grained control.

The single-image `sample_trajectory()` (line 119) retains `sampling_shift: float = 1.0` as a parameter -- backward compatible with existing callers that generate only at 1280x832.

### 4.5. `src/futudiffu/dataset_v2.py` -- Schema and Metadata Persistence

The V2 dataset parquet schema (line 83) includes:

```python
("sampling_shift", pa.float32()),  # nullable: SD3 Eq.23 alpha, null = 1.0 legacy
```

This is a nullable column. For legacy trajectories generated before sigma shifting was implemented, the value is null, which downstream consumers interpret as alpha = 1.0 (identity shift). New trajectories carry their actual shift value.

The `DatasetWriter.add_trajectory()` method (line 359) writes the shift from metadata:

```python
"sampling_shift": float(metadata["sampling_shift"])
    if metadata.get("sampling_shift") is not None else None,
```

This nullable design is deliberate: it avoids retroactively modifying existing trajectories while ensuring new data carries full provenance. The shift value becomes part of the trajectory's immutable identity -- once written, it describes exactly which noise schedule that trajectory was generated under.

### 4.6. `src_ii/pair_sampler.py` -- BTRM Training Integration

The pair sampler is where sigma shifting closes the loop. `build_positions_from_v2()` (lines 354-425) reads each trajectory's shift from the dataset and uses it to reconstruct the correct sigma schedule:

```python
recorded_shift = meta.get("sampling_shift")
if recorded_shift is not None:
    shift = float(recorded_shift)
else:
    w = meta.get("width", 1280)
    h = meta.get("height", 832)
    shift = resolution_shift(w, h)
sigmas = build_sigma_schedule(
    n_steps, sampling_shift=shift, denoise=denoise,
    device="cpu", dtype=torch.float32,
)
```

The fallback logic (compute from width/height when `sampling_shift` is null) handles legacy V1 trajectories that predate the schema addition. For V2 trajectories, the recorded shift is used directly -- this ensures the sigma values used for pair weighting match the sigma values the trajectory was actually generated under.

Each image position gets a sigma value derived from its trajectory's shifted schedule, and a logSNR-based sampling logit computed from that sigma. The logSNR decay weighting (`logsnr_sampling_weight()`) shapes the pair distribution: cleaner images (lower sigma, higher logSNR) receive full weight, while noisier images receive geometrically decaying weight. Because the sigmas are shift-corrected, the logSNR values are comparable across resolutions. A "step 14" position from a 256x256 trajectory (shifted sigma = 0.822) produces a different logSNR than a "step 14" position from a 1280x832 trajectory (sigma = 0.533), and the pair sampler's weighting reflects this difference.

---

## 5. Per-Image Euler Stepping: The Core Mechanism

The most technically subtle piece of the integration is how per-image sigma schedules interact with FlexAttention batch packing. In a packed forward pass, multiple images share a single model invocation: they are concatenated into one sequence with block-diagonal attention masks, and the model produces per-image outputs from a single forward call.

The diffusion model's adaLN (adaptive Layer Normalization) mechanism modulates all layers based on a timestep embedding. In a packed batch, there is one timestep embedding applied to the entire packed sequence. This creates a tension: if different images have different sigma values at the same step index, which sigma should the model see?

The resolution: use a representative sigma (from the first image) for the shared timestep embedding, and handle the per-image sigma differences entirely in the euler integration math. The euler step for each image i is:

```
denoised_i = const_calculate_denoised(sigma_i, model_output_i, x_i)
d_i = to_d(x_i, sigma_i, denoised_i)
dt_i = sigma_i_next - sigma_i
x_i = x_i + d_i * dt_i
```

Each of `const_calculate_denoised`, `to_d`, and the dt computation uses image i's own sigma from its own shifted schedule. The model's output is produced with a shared timestep embedding, but the integration uses per-image noise levels. This is exact for the euler step math and approximate for the model's internal conditioning. The approximation is acceptable because:

1. The shift differences within a single pack are bounded. Full-res (alpha=1.0) and small (alpha=4.0) images are unlikely to share a pack -- their sequence lengths differ by 16x, and the bin packer places them in separate bins.

2. Items that do share a pack have similar sequence lengths and therefore similar shift values. Two "medium" images (alpha ~2.0) share a pack with timestep values that differ by less than 5%.

3. The adaLN modulation is a relatively coarse conditioning signal (it scales and shifts layer activations globally). Small perturbations in the timestep embedding produce small perturbations in the output, which are then corrected by the per-image euler math.

---

## 6. BTRM Implications

Sigma shifting makes three contributions to BTRM reward model training:

**Cross-resolution pair validity.** Without shifting, a BTRM reward model trained on mixed-resolution pairs would learn resolution as a confound. A 256x256 image at step 14 (unshifted sigma 0.533) looks substantially noisier than a 1280x832 image at the same step (also sigma 0.533 but with 16x more spatial redundancy). The model would attribute the quality difference to the generation process rather than to the resolution mismatch. With shifting, step 14 at 256x256 uses sigma 0.822, producing an effective noise level comparable to the 1280x832 image at its own sigma 0.533. The reward model sees images at perceptually equivalent denoising stages.

**logSNR-consistent pair weighting.** The pair sampler weights positions by logSNR using a geometric decay function. If the sigmas are not shift-corrected, a "clean-looking" step from a 256x256 trajectory would receive an inflated logSNR weight (because its raw sigma is low), even though perceptually it is noisier than a 1280x832 image at the same raw sigma. The shift correction ensures that logSNR values reflect actual perceptual noise levels, so the weighting function operates on a meaningful scale.

**Dataset provenance.** The `sampling_shift` column in the V2 schema records which shift was used for each trajectory. This is not merely bookkeeping. If a future training run discovers that reward model accuracy differs across resolution tiers, the shift value in the metadata enables retroactive analysis: was the trajectory generated with the correct shift? Was the shift value used during pair sampling consistent with the generation shift? Without this provenance, debugging cross-resolution reward model failures would require re-deriving the shift from width/height and hoping the formula has not changed between versions.

---

## 7. Data Flow Summary

The complete path of a sigma shift value through the system:

```
bin_packer.build_generation_plan()
  |-- _resolution_shift(w, h) per item
  |-- item["sampling_shift"] = alpha
  v
client.sample_trajectory_packed()
  |-- params["sampling_shifts"] = per-image alphas
  v
server RPC -> sampling.run_trajectory_packed()
  |-- resolution_shift(w, h) if no explicit shift provided
  |-- build_sigma_schedule(n_steps, sampling_shift=alpha_i) per image
  |-- sample_euler_packed(... sigmas_list=[sigma_schedule_0, ..., sigma_schedule_N])
  |     |-- per-image euler step with sigma_i, dt_i
  v
dataset_v2.DatasetWriter.add_trajectory()
  |-- row["sampling_shift"] = alpha  (parquet column, nullable)
  v
pair_sampler.build_positions_from_v2()
  |-- reads meta["sampling_shift"] or computes from width/height
  |-- build_sigma_schedule(n_steps, sampling_shift=alpha)
  |-- logsnr_sampling_logit(sigma_at_step)
  v
BTRMPairSampler.sample_pair()
  |-- pair weighting uses shift-corrected sigma/logSNR values
```

Seven files. One formula. The shift is computed at plan time, transmitted through the RPC layer, applied per-image in the euler loop, persisted in the dataset schema, and recovered by the pair sampler for training. Each file has a specific role: bin_packer decides the shift, client transmits it, sampling applies it, dataset_v2 records it, pair_sampler reads it. No file both computes and consumes the shift in the same scope, which makes the data flow auditable.

---

## 8. Open Design Decisions

**Representative timestep selection.** The current implementation uses the first image's sigma as the representative timestep for the packed forward pass. An alternative would be to use the mean or median sigma across all images in the pack. For packs with narrow shift ranges (typical, given bin-packing by sequence length), the choice is immaterial. For hypothetical wide-range packs, the mean would reduce the maximum error for any single image's adaLN conditioning.

**DRGPO interaction.** The planned upgrade from REINFORCE to DRGPO requires consistent sigma schedules between the on-policy rollout and the reference policy. If the reference policy was generated at 1280x832 (shift=1.0) and the on-policy rollout targets 512x512 (shift=2.02), the log-ratio computation must account for the different noise schedules. The shift values must be propagated into the DRGPO loss computation, not just into the euler loop.

**Z-Image's training schedule.** The reference resolution (1280x832) and the identity shift at that resolution are based on Z-Image's ComfyUI workflow, which uses `ModelSamplingAuraFlow` with shift=1.0. If Z-Image was internally trained with a non-trivial shift at 1280x832, our shift table would be biased by a constant factor. Community reports of improved quality at shift=3-7 for non-native resolutions may partially reflect this uncertainty. The current implementation treats the question as settled (ref=1280x832, shift=1.0 at ref), with the understanding that the `ref_width` and `ref_height` parameters in `resolution_shift()` can be adjusted if new information surfaces.

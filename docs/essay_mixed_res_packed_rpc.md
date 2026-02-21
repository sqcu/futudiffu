# Mixed-Resolution Packed Inference: RPC Layer, Sigma Schedules, and the context_refiner RoPE Bug

**Date**: 2026-02-17
**Scope**: `src/futudiffu/sampling.py`, `src/futudiffu/server.py`, `src/futudiffu/client.py`,
`src/futudiffu/diffusion_model.py`, `src_ii/pair_sampler.py`

---

## Background

The packed inference path (`forward_packed` + FlexAttention) was introduced to amortize
compilation overhead across multiple images in a single forward pass. The initial implementation
assumed uniform resolution: all N images in a pack had the same width, height, and therefore the
same sigma schedule shift. That assumption held while datasets contained only reference-resolution
(1280x832) images.

As the BTRM dataset grew to include trajectories at varied resolutions -- and as the policy
optimization loop needed to generate images at arbitrary sizes for ablation studies -- the
uniform-resolution constraint became a hard limit. This document records three changes that
lifted it:

1. Per-image `widths`/`heights` in the `sample_trajectory_packed` RPC with automatic
   per-image sigma schedule construction.
2. A crash-producing bug in `prepare_packed_state` where the context_refiner RoPE was built
   with B=1 but applied to a CFG-batched (B=2) caption tensor.
3. A default-parameter change in `BTRMPairSampler` that enforces cross-trajectory-only pair
   sampling.

---

## Mixed-Resolution RPC

### The problem with scalar width/height

The original `run_trajectory_packed` in `sampling.py` extracted a single `width` and `height`
from the params dict, replicated them to every image, and built a single sigma schedule shared
by all:

```python
# Old behavior (implicit uniform resolution)
w = params["width"]
h = params["height"]
shift = resolution_shift(w, h)  # one shift for all
sigmas = build_sigma_schedule(n_steps, sampling_shift=shift, ...)
```

For a pack containing a 1280x832 image and a 512x512 image, this was wrong in three distinct ways:
- The noise tensor for image 2 would be the wrong shape (generated at 1280x832 dimensions).
- The sigma schedule shift for image 2 would be 1.0 (reference) instead of ~2.5 (smaller
  resolution), producing the wrong denoising trajectory.
- The per-step euler update would apply the wrong `dt` for image 2's noise level.

### The fix: per-image resolution resolution

`run_trajectory_packed` now resolves dimensions in priority order: per-image lists, then scalar
fallback:

```python
if "widths" in params and "heights" in params:
    widths = params["widths"]
    heights = params["heights"]
else:
    w = params["width"]
    h = params["height"]
    widths = [w] * n_images
    heights = [h] * n_images
```

Per-image sigma schedule shifts follow the same priority logic:

```python
if "sampling_shifts" in params:
    sampling_shifts = params["sampling_shifts"]
elif "sampling_shift" in params:
    shift_val = params["sampling_shift"]
    sampling_shifts = [shift_val] * n_images
else:
    sampling_shifts = [resolution_shift(widths[i], heights[i]) for i in range(n_images)]
```

The `resolution_shift` function implements SD3 Equation 23:
`alpha = sqrt(ref_pixels / target_pixels)`. For a 512x512 image relative to the 1280x832
reference, `alpha = sqrt(1064960 / 262144) ~= 2.015`. This shift biases the schedule toward
higher starting noise, which is correct for smaller images that have a different effective SNR
trajectory.

Per-image noise generation follows directly:

```python
for i in range(n_images):
    lh, lw = latent_dims[i]  # heights[i]//8, widths[i]//8
    gen = torch.Generator(device=device).manual_seed(seeds[i])
    noise = torch.randn(1, 16, lh, lw, ...)
    x_list.append(noise)
```

Per-image sigma schedules:

```python
sigmas_list = []
for i in range(n_images):
    s = build_sigma_schedule(
        n_steps, sampling_shift=sampling_shifts[i], ...
    )
    sigmas_list.append(s)
```

And per-image inverse noise scaling at the end uses each image's own final sigma:

```python
for img_i in range(n_images):
    x_list[img_i] = const_inverse_noise_scaling(
        sigmas_list[img_i][-1], x_list[img_i])
```

### The shared timestep approximation

`sample_euler_packed` (in `sampling.py`) uses the first image's sigma as the representative
timestep value passed to the model's adaLN modulation:

```python
sigma_representative = per_image_sigmas[0][step_i]
timestep = sigma_representative * multiplier
```

The per-image sigma differences only affect the Euler update math (noise scaling and dt
computation), not the model's internal computation. For modest shift differences -- images at
comparable resolutions with the same step count -- the timestep embedding difference is small
and using a representative value is acceptable. Per-image sigma values are then applied
correctly in the euler step:

```python
for img_i in range(n_images):
    sigma_i = per_image_sigmas[img_i][step_i]
    sigma_i_next = per_image_sigmas[img_i][step_i + 1]
    # ... CFG combination ...
    d = to_d(x_list[img_i], sigma_i, denoised)
    dt = sigma_i_next - sigma_i
    x_list[img_i] = x_list[img_i] + d * dt
```

---

## The context_refiner RoPE Bug

### Where the crash came from

`prepare_packed_state` pre-computes the constant-across-steps state for a packed batch: caption
embeddings refined through `context_refiner`, packing layout, and RoPE frequencies. The calling
convention for `context_list` is N tensors each of shape `(B, seq_i, cap_feat_dim)` where B=2
for CFG (pos+neg concatenated). The caption embedder outputs the same B dimension.

`prepare_rope_cache` constructs RoPE tensors with B=1 regardless of the calling batch size:

```python
rope_cache = self.prepare_rope_cache(H, W, cap_embedded_len, device)
cap_freqs_cis = rope_cache['cap_freqs_cis']
# cap_freqs_cis.shape[0] == 1 here
```

The `context_refiner` layers then receive `cap_embedded` with shape `(2, seq, dim)` but
`cap_freqs_cis` with shape `(1, ...)`. The attention kernel attempts to broadcast or index-match
these dimensions and crashes with:

```
RuntimeError: shape '[64, 64, 2, 2]' invalid for input size 8192
```

The number 8192 is `1 * seq_len * head_dim / 2` (complex pairs for one batch element). The
kernel expected `2 * seq_len * head_dim / 2` (for B=2). The error message expresses this as a
reshape failure: trying to view `(1, seq, heads, hdim/2, 2)` as `(2, seq, heads, hdim/2, 2)`.

### Why the non-packed path did not have this bug

The non-packed `forward` method already had an explicit expansion before calling context_refiner:

```python
# In forward() (non-packed path), from the Socratic dialogue in the codebase:
cap_freqs_cis = rope_cache['cap_freqs_cis'].expand(bsz, -1, -1, -1, -1, -1)
```

`prepare_packed_state` was written later as an extraction of the packed-specific setup and the
expansion was not ported over. This is a copy-paste gap, not a logic error.

### The fix

Three lines added after rope cache lookup:

```python
rope_cache = self.prepare_rope_cache(H, W, cap_embedded_len, device)
cap_freqs_cis = rope_cache['cap_freqs_cis']

# Expand RoPE for CFG batch dimension (B=2 for pos+neg)
bsz = cap_embedded.shape[0]
if bsz > 1 and cap_freqs_cis.shape[0] == 1:
    cap_freqs_cis = cap_freqs_cis.expand(bsz, -1, -1, -1, -1, -1)
```

The guard `bsz > 1 and cap_freqs_cis.shape[0] == 1` handles both the CFG case (B=2, needs
expansion) and any future non-CFG use of `prepare_packed_state` (B=1, no expansion needed).
The `expand` call is zero-copy (creates a view with broadcasted strides), so there is no memory
overhead.

### Why this only surfaced with the mixed-resolution change

The crash existed before mixed-resolution support but only triggered when `prepare_packed_state`
was called with CFG-batched (B=2) caption conditionings. The original uniform-resolution test
path happened to pass single-batch conditionings through a different code path. Once per-image
conditionings were being constructed correctly as `(2, seq_i, cap_feat_dim)` for CFG, every
call to `prepare_packed_state` hit the expansion guard.

---

## Cross-Trajectory Pairs: `allow_intra_trajectory=False`

### What intra-trajectory pairs are

A BTRM pair is (image_a, image_b) where the model is trained to prefer one over the other. An
intra-trajectory pair draws both images from the same generation run: for example, step_04 and
step_29 from traj_000042. These are different stages of the same denoising sequence, not
different generation outcomes.

### Why they are wrong for BTRM reward training

The BTRM reward model's job is to discriminate quality differences between generation outcomes,
not denoising stages. Head 0 ("bit_quality", scrimblo) is trained to discriminate attention
quantization quality (SDPA vs INT8 QK). Head 1 ("step_quality", scrongle) discriminates
generation quality at different step counts (30 vs 8-22). Both tasks compare generation outcomes
under different conditions.

An intra-trajectory pair (step_04 vs step_29 from the same run) does not test either of these.
It tests noise level: the step_04 latent is noisier than step_29 because it is earlier in the
denoising process. A reward model trained on such pairs learns to prefer "less noisy latents,"
which is a trivially correct answer at inference time (all outputs are fully denoised) but
provides no useful training signal for ranking generations by quality.

The correct pair is: "two fully-denoised outputs from different trajectories, with known
quality differences." Cross-trajectory pairs (step_29 from traj_000042 vs step_29 from
traj_000017) compare image quality under the same noise level, which is the actual reward
signal.

### The default change in `BTRMPairSampler`

```python
class BTRMPairSampler:
    def __init__(
        self,
        positions: list[_ImagePosition],
        allow_inter_trajectory: bool = True,
        allow_intra_trajectory: bool = False,  # was True
        ...
    ):
```

Changing the default from `True` to `False` means callers that do not explicitly set this
parameter get the correct behavior without needing to know about the intra/inter distinction.
The parameter is preserved for experimental use (e.g. studying what a reward model trained on
intra-trajectory pairs actually learns), but the production path cannot accidentally fall into
the wrong mode.

### `build_positions_from_v2` reading resolution metadata

`build_positions_from_v2` reads resolution metadata from each trajectory's meta record to
compute the correct sigma schedule for sigma value lookup:

```python
recorded_shift = meta.get("sampling_shift")
if recorded_shift is not None:
    shift = float(recorded_shift)
else:
    w = meta.get("width", 1280)
    h = meta.get("height", 832)
    shift = resolution_shift(w, h)
sigmas = build_sigma_schedule(
    n_steps, sampling_shift=shift, denoise=denoise, ...
)
```

This matters for accurate `sigma_val` assignment to each step position. When sigma values are
wrong, the logSNR-based sampling weights are wrong, which biases the pair distribution toward
steps that do not represent the intended noise range.

---

## Integration Surface: Full Chain of Changes

The mixed-resolution support propagates through five layers. Reading bottom-up:

### 1. Model forward: `diffusion_model.py:prepare_packed_state` (lines 1046-1052)

- Builds per-image RoPE from each image's `(H, W)`.
- Now expands `cap_freqs_cis` to the caption batch size before passing to `context_refiner`.
- No interface change; the fix is internal.

### 2. Sampling orchestration: `sampling.py:run_trajectory_packed`

- Accepts either `width`/`height` (scalar, backward-compatible) or `widths`/`heights` (lists).
- Resolves per-image sigma shifts with three-way priority: explicit `sampling_shifts` list,
  scalar `sampling_shift`, or auto-computed via SD3 Eq.23.
- Generates per-image noise at per-image latent dimensions.
- Calls `build_sigma_schedule` once per image.
- Passes `sigmas_list` (list of tensors) to `sample_euler_packed`.

### 3. Euler sampler: `sampling.py:sample_euler_packed`

- Accepts `sigmas_list` as either a single tensor (broadcasts to all images) or a list of N
  tensors.
- Uses representative sigma for the shared timestep embedding.
- Applies per-image `(sigma_i, sigma_i_next)` in the euler update for each image independently.
- Applies per-image final sigma in `const_inverse_noise_scaling`.

### 4. Server dispatch: `server.py:handle_sample_trajectory_packed`

- No changes needed. Delegates to `run_trajectory_packed`, which now accepts the extended
  params format. The server handler is a thin pass-through.

### 5. Client API: `client.py:InferenceClient.sample_trajectory_packed`

New parameters added to the public API:

```python
def sample_trajectory_packed(
    self,
    pos_conds: list[torch.Tensor],
    neg_cond: torch.Tensor,
    seeds: list[int],
    n_steps: int,
    cfg: float = 4.0,
    width: int | None = None,       # scalar (backward compat)
    height: int | None = None,      # scalar (backward compat)
    widths: list[int] | None = None,    # NEW: per-image
    heights: list[int] | None = None,   # NEW: per-image
    sampling_shift: float | None = None,       # scalar override
    sampling_shifts: list[float] | None = None, # NEW: per-image
    ...
) -> list[dict[str, torch.Tensor]]:
```

The client serializes these into the RPC params dict, where per-image lists take precedence
over scalar values, which take precedence over the default resolution fallback:

```python
if widths is not None and heights is not None:
    req_params["widths"] = widths
    req_params["heights"] = heights
elif width is not None and height is not None:
    req_params["width"] = width
    req_params["height"] = height
else:
    req_params["width"] = 1280
    req_params["height"] = 832
```

Sigma shift follows the same three-level pattern.

---

## Backward Compatibility

Existing callers that pass `width`/`height` as scalars continue to work without modification.
The server-side resolver in `run_trajectory_packed` handles the scalar-to-list promotion. The
client-side fallback to `width=1280, height=832` preserves the previous default behavior for
callers that pass neither scalar nor list forms.

The context_refiner RoPE fix is transparent to all callers; it corrects behavior rather than
changing interface.

The `allow_intra_trajectory=False` default is a behavioral change for any existing
`BTRMPairSampler` constructor calls that did not explicitly set `allow_intra_trajectory`. In
practice this is a correction: no training code should have been relying on intra-trajectory
pairs for meaningful reward signal.

---

## Summary

Three changes that cooperate to make mixed-resolution packed inference correct:

| Layer | Change | Why |
|---|---|---|
| `diffusion_model.py:prepare_packed_state` | Expand `cap_freqs_cis` to caption batch size | CFG passes B=2 captions; RoPE cache is B=1 |
| `sampling.py:run_trajectory_packed` | Per-image widths/heights/sigmas | Different resolutions require different noise shapes and sigma schedules |
| `sampling.py:sample_euler_packed` | Per-image sigma list | Correct `dt` and noise scaling per image in Euler step |
| `server.py` | No change | Thin pass-through; change is transparent |
| `client.py:sample_trajectory_packed` | New `widths`/`heights`/`sampling_shifts` params | Expose per-image resolution at the user-facing API |
| `src_ii/pair_sampler.py:BTRMPairSampler` | `allow_intra_trajectory=False` default | Intra-trajectory pairs compare denoising stages, not generation quality |
| `src_ii/pair_sampler.py:build_positions_from_v2` | Read resolution from trajectory metadata | Accurate sigma values for logSNR-based sampling weights |

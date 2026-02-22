# Triumphant Future Reduction Ops

K-tuple scatter-gather for guided diffusion sampling.
Module: `src_ii/triumphant_future_reduction_ops.py`.

## Core idea

A guidance spec is a list of `(conditioning, resolution, scale)` tuples.
Entry 0 is always the base — the trajectory being integrated. Remaining entries
are guidance queries with signed scales: positive = attractive, negative = repulsive.

```python
spec = [
    (base_cond,   (1024, 1024),  1.0),   # base
    (shrimp_cond, (1024, 1024),  3.0),   # +attractive
    (typo_cond,   (1024, 1024),  2.0),   # +attractive
    (base_cond,   (512, 512),   -2.0),   # -repulsive (low-res)
    (base_cond,   (256, 256),   -1.5),   # -repulsive (lower-res)
    (banana_cond, (1024, 1024), -4.0),   # -repulsive (different text)
]
```

Nothing in the implementation mentions the number 6 (or 2, or 1). K is a parameter.

## Functions

### `noise_field(max_h, max_w, seed, device, dtype) -> Tensor`

One PRNG call at the maximum latent resolution. Returns `(1, 16, max_h, max_w)`.
All guidance queries share noise from this field via `aperture()`.

### `aperture(master, h, w) -> Tensor`

Center-crop `(h, w)` from the master noise field. Smaller queries see the
center of the same noise — ensures coherent structure across resolutions.

### `scatter(x_base, spec) -> list[Tensor]`

Map the current base latent into K query latents:
- Same resolution as base: clone x_base.
- Different resolution: bilinear interpolate x_base to the query's latent size.

### `gather(denoised_list, spec) -> Tensor`

Reduce K denoised estimates to one guided estimate:
```
result = base + sum(scale_i * (upsample(guide_i) - base))
```
Different-resolution branches are bilinearly upsampled to base resolution before
computing the residual.

### `denoise_all(x_list, fields, sigmas) -> list[Tensor]`

CONST denoising: `denoised_k = x_k - field_k * sigma_k` for each branch.

### `euler_step(x_base, guided, sigma, sigma_next) -> Tensor`

One Euler integration step:
```
d = (x_base - guided) / sigma
x_next = x_base + d * (sigma_next - sigma)
```

### `build_per_image_sigmas(spec, n_steps, device, dtype) -> list[Tensor]`

Build per-branch sigma schedules. Each branch's resolution determines its
sigma shift via `resolution_shift()`.

## Factory helpers

### `cfg1(cond, res)`

Single-entry spec. Gather is identity — no guidance.

### `cfg2(pos, neg, res, scale)`

Standard classifier-free guidance. Two entries:
- `(pos, res, 1.0)` — base (positive conditioning)
- `(neg, res, -(scale - 1))` — repulsive from unconditional

Algebraic equivalence with standard CFG:
```
gather = pos + (1-cfg)*(neg - pos) = cfg*pos + (1-cfg)*neg = neg + cfg*(pos - neg)
```

### `cfg6(base, shrimp, typo, banana, base_res, lr1, lr2, scales)`

6-entry spec with mixed conditioning and mixed resolution. `scales` is a 6-tuple
of signed weights. Typical configuration:
```python
scales = (1.0, 3.0, 2.0, -2.0, -1.5, -4.0)
```

## The Euler loop

```python
sigma_schedules = build_per_image_sigmas(spec, n_steps, device, dtype)
master = noise_field(max_lh, max_lw, seed, device, dtype)
x_list = [aperture(master, res_h//8, res_w//8) * sigma_schedules[k][0]
          for k, (_, (res_w, res_h), _) in enumerate(spec)]

for step_i in range(n_steps):
    fields, scores = packed_forward(model, x_list, timesteps, ...)
    sigmas_k = [sigma_schedules[k][step_i] for k in range(len(spec))]
    denoised = denoise_all(x_list, fields, sigmas_k)
    guided = gather(denoised, spec)
    x_base = euler_step(x_list[0], guided, sigmas_k[0], sigma_schedules[0][step_i+1])
    x_list = scatter(x_base, spec)
```

15 lines. Never mentions CFG.

## Defects this abstraction fixes

### Shared adaLN (`zimage_model.py`)

Old: `t_shared = 1 - timesteps_list[0]` uses image 0's timestep for all images.
Fixed: per-image timestep embeddings for noise_refiner. Main-layer per-token
adaLN requires JointTransformerBlock rewrite in src_ii (pending).

### B=2 CFG batching (`rollout.py`)

Old: B=2 with `pos_k[:1]`/`neg_k[1:]` splits and `.expand(B, ...)` copies.
Fixed: B=1 always. Each guidance query is a separate image in the packed batch.
The B dimension is reserved for actual data parallelism.

## Spherical reduction with post-gain

### The problem with naive summation

`gather` sums `scale_i * (guide_i - base)`. Two problems:

1. **Double-counting**: correlated directions amplify shared components.
   "Shrimp detail" + "typography clarity" both push toward high spatial
   frequency — the shared HF component gets counted twice.

2. **Magnitude accumulation**: K directions produce a push K× stronger
   than any individual direction. The "power level" of the conditioning
   effect grows with K, even when directions are independent.

### The polar decomposition

Each conditioning defines a **direction** from base (what the guidance
pushes toward or away from) and a **magnitude** (how strong the conditioning
effect is — the "power level"). These are independent quantities and should
be controlled independently.

`gather_residual_gain` implements this polar decomposition:

- **Direction** (angular): per_scales rotate the combined guidance vector
  on the unit hypersphere. Positive scales attract toward a direction,
  negative scales repel. The ratio of scales controls the rotation angle.

- **Magnitude** (radial): gain × radius. The radius is the |scale|-weighted
  mean of raw residual norms — preserving the "power level" of conditioning
  without amplifying it when K grows.

### `gather_residual_gain(denoised_list, spec, gain) -> Tensor`

```
raw_residuals = [(guide_i - base) for i in 1..K-1]
unit_dirs     = [r_i / ‖r_i‖ for r_i in raw_residuals]
direction     = normalize(Σ scale_i · unit_dirs[i])
radius        = Σ |scale_i| · ‖r_i‖  /  Σ |scale_i|
result        = base + gain · radius · direction
```

Per_scales control WHERE on the hypersphere. Gain × radius controls HOW FAR.

### K=2 collapse to standard CFG

One residual: `r = d_neg - d₀`, scale `s` (negative for repulsive).

```
direction = sign(s) · r̂ = -(d_neg - d₀)/‖...‖ = (d₀ - d_neg)/‖...‖
radius    = |s| · ‖r‖ / |s| = ‖r‖
result    = d₀ + gain · ‖r‖ · (d₀ - d_neg)/‖d₀ - d_neg‖
          = d₀ + gain · (d₀ - d_neg)
```

Set `gain = cfg - 1` → standard CFG. The scale magnitude cancels in the
n=1 case — only sign matters. Gain is NOT data-dependent.

### K=3: angular interpolation between guidance directions

Spec: `[(P, res, 1.0), (P+"shrimp", res, +α), (∅, res, -β)]`

```
r₁ = d_shrimp - d₀    (toward shrimp features)
r₂ = d_neg - d₀        (toward unconditional)

direction = normalize(α · r̂₁ - β · r̂₂)
          = normalize(α · toward_shrimp + β · away_from_uncond)
```

The ratio α/β is the rotation angle:
- `α >> β`: direction ≈ toward shrimp (all shrimp, minimal CFG)
- `α << β`: direction ≈ away from uncond (standard CFG, minimal shrimp)
- `α = β`: bisector of the two directions

The radius is the weighted mean: `(α·‖r₁‖ + β·‖r₂‖) / (α + β)`.
Two directions don't produce 2× the push — the radius is a mean, not a sum.

### Smooth K transition

Setting `α → 0` in the K=3 spec:
```
direction → sign(-β) · r̂₂ = (d₀ - d_neg)/‖...‖
radius    → (0·‖r₁‖ + β·‖r₂‖) / (0 + β) = ‖r₂‖
result    → d₀ + gain · (d₀ - d_neg)  =  K=2 result
```

No discontinuity. Zero-weight entries contribute nothing to direction or radius.
Compare with Löwdin, where adding a zero-weight entry normalizes the surviving
residual (breaking the K=2 equivalence).

### Why per_scales are angular weights, not magnitude weights

With `gather` (naive summation), doubling a scale doubles the residual magnitude.
With `gather_residual_gain`, doubling a scale rotates the combined direction
toward that entry without changing the radius.

This is the correct decomposition: "how much shrimp vs how much CFG" is a
DIRECTION question (angular). "How strong is the overall guidance" is a
MAGNITUDE question (gain × radius). These are independent controls.

To sweep shrimp influence relative to base CFG: fix gain, vary α/β.
To sweep overall guidance strength: fix α/β, vary gain.

### `cfg6_residual_gain(..., per_scales, gain) -> (spec, gain)`

Same arguments as `cfg6` except the scales tuple is called `per_scales`
and there's an additional `gain` parameter. Returns `(spec, gain)` — the
spec is the same format as `cfg6`, used with `gather_residual_gain`
instead of `gather`.

```python
spec, gain = cfg6_residual_gain(
    base, shrimp, typo, banana,
    base_res=(1024, 1024), lr1=(512, 512), lr2=(256, 256),
    per_scales=(1.0, 1.0, 1.0, -1.0, -1.0, -1.0),
    gain=7.0,
)
# per_scales: angular weights on the guidance hypersphere
#   positive = attractive (rotate toward), negative = repulsive (rotate away)
#   ratios control direction, not magnitude
# gain: radius multiplier (the "CFG scale" knob)
```

### The Euler loop with residual gain

```python
for step_i in range(n_steps):
    fields, scores = packed_forward(model, x_list, timesteps, ...)
    denoised = denoise_all(x_list, fields, sigmas_k)
    guided = gather_residual_gain(denoised, spec, gain)   # <-- only this line changes
    x_base = euler_step(x_list[0], guided, ...)
    x_list = scatter(x_base, spec)
```

### 6-tuple example: per_scales as angular sweep

The 6-tuple spec with residual gain:
```
direction = normalize(
    1.0 · r̂_shrimp          # attract toward shrimp features
  + 1.0 · r̂_typo            # attract toward typography features
  - 1.0 · r̂_lowres_512      # repel from mid-res blur
  - 1.0 · r̂_lowres_256      # repel from low-res blur
  - 1.0 · r̂_banana          # repel from banana content
)
```

To emphasize shrimp 3× over typography while keeping everything else equal:
`per_scales = (1.0, 3.0, 1.0, -1.0, -1.0, -1.0)`. This rotates the
combined direction toward the shrimp axis without changing the radius.

To sweep overall strength: vary `gain` from 1.0 (subtle) to 15.0 (aggressive)
while keeping per_scales fixed.

## Verification checklist

- `cfg1(cond)`: K=1, gather returns base unchanged
- `cfg2(pos, neg, 7.0)`: algebraically equivalent to `uncond + 7*(cond - uncond)`
- `cfg6(...)`: 6 queries packed, produces coherent image (visual check)
- `aperture(master, small)` center == `aperture(master, large)` center (noise coherence)
- Per-image sigmas: different resolutions produce different schedules

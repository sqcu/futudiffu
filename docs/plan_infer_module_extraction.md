# Plan: Extract Inference Modules from demonstrate_rtheta_policy.py

## Problem

`scripts_ii/demonstrate_rtheta_policy.py` (534 lines) inlines algorithmic code
that should live in importable modules per `docs/user_iron_mandates.md`:

> SCHEDULING CODE NEVER INLINES ALGORITHMIC CODE.
> DIVIDING A VALUE OR INDEX BY `2` IS ALGORITHMIC CODE.
> ALGORITHMIC CODE LIVES IN SINGULAR IMPORTED MODULES.

The script is a scheduling script: it orchestrates model lifecycle phases,
runs N trajectories, saves outputs. Every computation and rendering decision
inside it is algorithmic code that must be extracted.

## Goal

1. Extract algorithmic code into `src_ii/infer/` modules.
2. Rewrite `demonstrate_rtheta_policy.py` as a skinny runner (~80-100 lines).
3. Write `validate_policy_intervention.py` as a second skinny runner importing
   the same modules. The 3-way policy comparison is just the 2-way rtheta demo
   generalized to N configs.

## Algorithmic Code Identified in demonstrate_rtheta_policy.py

### 1. Text Encoding Phase (lines 72-96)
**Decision**: Load TE, encode prompts, free TE.
**Algorithm**: `create_tokenizer()`, `load_text_encoder()`, `encode_prompt()`,
VRAM cleanup.
**Already exists**: `futudiffu.text_encoder` has the primitives.
**Extract to**: `src_ii/infer/text_encoding.py`
- `encode_prompts(prompts, te_path, device, dtype) -> dict[str, Tensor]`
- Handles load/encode/free lifecycle. Returns CPU tensors.

### 2. Model Loading + Adapter Setup (lines 100-142)
**Decision**: Load ZImageRLAIF, install LoRA, load BTRM, compile.
**Algorithm**: Model construction, adapter config, weight loading, compilation.
**Already exists**: `zimage_model.load_zimage_rlaif`, `multi_lora.install_multi_lora`,
`btrm_lifecycle.load_btrm`, `attention_srcii.patch_sage_for_compile`.
**Extract to**: `src_ii/infer/model_setup.py`
- `load_and_prepare_model(fp8_path, adapter_configs, btrm_dir, adapter_loads, device, dtype) -> (compiled, raw, head_names)`
- `adapter_configs`: list of {"name", "rank", "alpha"} dicts
- `adapter_loads`: list of {"name", "path", "loader"} — loader is a callable
  (load_btrm for BTRM, load_adapter or custom for policy)
- Handles the entire load → install → load weights → patch → compile pipeline.

### 3. Euler ODE Stepping (lines 225-236)
**Decision**: `denoised = x - field * sigma; d = (x - denoised) / sigma; x += d * dt`
**Algorithm**: First-order Euler integration of the probability flow ODE.
**Already exists**: Nowhere as an importable function! Inlined in multiple scripts.
**Extract to**: `src_ii/infer/euler.py`
- `euler_step(x, field, sigma_i, sigma_next) -> Tensor`
- Pure tensor math. ~8 lines.

### 4. LogSNR Computation (line 214-215)
**Decision**: `logsnr = 2.0 * math.log((1.0 - s) / s)` with clamping.
**Algorithm**: Convert sigma to logSNR.
**Already exists**: Inlined in multiple places (demonstrate_rtheta_policy.py,
sigma_schedule.py's compute_logsnr_uniform_steps, pair_sampler.py).
**Extract to**: `src_ii/infer/euler.py` (alongside euler_step)
- `sigma_to_logsnr(sigma: float) -> float`
- Clamp sigma to [0.001, 0.999] before computing.

### 5. Multi-Config Trajectory Sampling (lines 146-252)
**Decision**: For each prompt, run N forward passes per step across configs.
**Algorithm**: Initialize noise, prepare plan, loop steps × configs, record scores.
**Extract to**: `src_ii/infer/trajectory.py`
- `sample_trajectory(model, plan, sigmas, x_init, adapter_scales, n_steps) -> (final_latent, step_scores)`
- Single-config, single-trajectory. The scheduling script loops over configs/prompts/seeds.
- `step_scores`: list of `{"step", "sigma", "logsnr", "scores"}` dicts.

### 6. Score Chart Rendering (lines 283-396)
**Decision**: Pure PIL chart with lines, grid, legend, derivative bars.
**Algorithm**: Coordinate mapping, line drawing, scale computation.
**Already exists**: `src_ii/visualization.py` exists but may not have this exact chart.
**Extract to**: `src_ii/infer/charts.py`
- `draw_score_chart(logsnrs, named_series, head_name, chart_w, chart_h) -> PIL.Image`
- `named_series`: dict[str, {"values": list[float], "color": tuple[int,int,int]}]
- N-way generalization of the 2-series chart. Works for 2 or 3 or N series.

### 7. Composite Image Building (lines 399-475)
**Decision**: Paste images + charts into a single composite PNG.
**Algorithm**: Layout computation, scaling, text rendering.
**Extract to**: `src_ii/infer/composites.py`
- `build_comparison_composite(image_panels, charts, title, labels) -> PIL.Image`
- Generic N-panel comparison composite.

### 8. Policy Adapter Loading with Key Remapping
**Decision**: DDGRPO policy checkpoints have `_orig_mod.` in keys (from torch.compile)
and adapter name `policy_pinkify`. Must remap to target adapter name on uncompiled model.
**Already exists**: `multi_lora.load_adapter` handles old `.adapters.` format but NOT
`_orig_mod.` format.
**Extract to**: `src_ii/infer/model_setup.py`
- `load_policy_adapter(model, target_name, ckpt_path, source_name) -> int`
- Strips `_orig_mod.`, remaps adapter name, copies tensors.

### 9. Diff Analysis (new for policy intervention)
**Decision**: Mean diff latent across seeds, covariance eigenspectrum, effective rank.
**Algorithm**: Tensor averaging, covariance matrix, eigvalsh, effective rank formula.
**Already exists**: `rendering.py` has `compute_spatial_autocorrelation`,
`compute_per_channel_pixel_stats`, `make_false_color_diff`. But latent-space
covariance analysis is new.
**Extract to**: `src_ii/infer/diff_analysis.py`
- `compute_latent_covariance(diff_latent) -> {"effective_rank", "eigenvalues"}`
- `compute_mean_diff_latent(latents_a, latents_b) -> Tensor` (across seeds)
- Re-exports from rendering.py: `make_false_color_diff`, `compute_spatial_autocorrelation`

## New Module Map

```
src_ii/infer/
    __init__.py          # empty
    text_encoding.py     # encode_prompts()
    model_setup.py       # load_and_prepare_model(), load_policy_adapter()
    euler.py             # euler_step(), sigma_to_logsnr()
    trajectory.py        # sample_trajectory()
    charts.py            # draw_score_chart()
    composites.py        # build_comparison_composite()
    diff_analysis.py     # latent covariance, mean diff, re-exports from rendering
```

## Skinny Runner Pattern

After extraction, `demonstrate_rtheta_policy.py` becomes:

```python
# Constants: paths, prompts, seeds, resolution, n_steps
# main():
#   conds = encode_prompts(PROMPTS, TE_PATH, DEVICE, DTYPE)
#   model, raw, heads = load_and_prepare_model(FP8_PATH, [...], BTRM_DIR, [...], DEVICE, DTYPE)
#   for slug, seed, config in product(prompts, seeds, configs):
#       lat, scores = sample_trajectory(model, plan, sigmas, x_init, scales, N_STEPS)
#   # free backbone, load VAE, decode, free VAE
#   images = {slug: decode_latent_to_pil(vae, lat, ...) for ...}
#   # render composites
#   for slug in prompts:
#       chart = draw_score_chart(logsnrs, series, head_name)
#       composite = build_comparison_composite(panels, charts, title, labels)
#       composite.save(...)
#   # write JSONL + manifest
```

And `validate_policy_intervention.py` becomes the same pattern with:
- 3 adapter configs instead of 1
- Multiple resolutions with per-resolution sigma schedules
- Multiple seeds with mean-diff analysis
- Cross-resolution composites

Both scripts are ~80-120 lines of pure scheduling: constants, loops, I/O.

## Execution Order

1. Create `src_ii/infer/` directory and `__init__.py`
2. Write `euler.py` (smallest, no dependencies)
3. Write `text_encoding.py` (wraps existing futudiffu.text_encoder)
4. Write `model_setup.py` (wraps existing src_ii modules)
5. Write `trajectory.py` (imports euler.py + forward_packed)
6. Write `charts.py` (pure PIL, no src_ii deps)
7. Write `composites.py` (pure PIL)
8. Write `diff_analysis.py` (imports rendering.py)
9. Rewrite `demonstrate_rtheta_policy.py` as skinny runner
10. Write `validate_policy_intervention.py` as skinny runner

## What NOT To Do

- Do NOT duplicate any algorithm that already exists in src_ii/.
  sigma_schedule.py, rendering.py, forward_packed.py, multi_lora.py,
  btrm_lifecycle.py, vae_utils.py are all still imported.
- Do NOT create "wrapper" modules that just re-export. Each new module
  adds genuinely new code (euler stepping, chart rendering, composite building)
  that was previously inlined.
- Do NOT add features beyond what the two scripts need. The modules serve
  exactly two consumers. If a third consumer appears later, generalize then.

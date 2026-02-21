# Essay: Multi-Resolution BTRM Dataset Generation

**Date:** 2026-02-19
**Script:** `scripts_ii/generate_multi_res_trajectories.py`
**Hardware:** RTX 4090, 24GB VRAM
**Wall time:** 763.9s (12.7 min)
**Output:** `multi_res_trajectories/`

---

## 1. Purpose

The existing BTRM V2 dataset (`btrm_dataset_v2/`) has 259 trajectories but 96% are
1280x832. This makes FLOPS-weighted sampling degenerate (every trajectory is the same
cost), bin packing trivial (every item fills a bin alone), and resolution-dependent
sigma shifting untested.

This generation run produces 60 trajectories spanning all 6 megapixel anchor tiers
with algorithmically sampled non-square aspect ratios. It is specifically designed to
exercise:

- **FLOPS-weighted pair sampling** with genuinely different per-pair costs
- **Bin packing** that actually packs multiple images per bin
- **Resolution-dependent sigma shifting** (SD3 Eq.23) across a 16x pixel budget range
- **Cross-resolution BTRM pairs** where the pair sampler must handle different latent shapes

## 2. Resolution Distribution

### Tier Summary

| Tier | Anchor (px) | Shift Range | Traj/Backend | Total | Unique Resolutions |
|------|-------------|-------------|--------------|-------|--------------------|
| 256sq | 65,536 | 3.85-4.03 | 5 | 10 | 3 |
| 320sq | 102,400 | 3.11-3.57 | 5 | 10 | 5 |
| 384sq | 147,456 | 2.60-2.93 | 5 | 10 | 4 |
| 512sq | 262,144 | 1.96-2.10 | 5 | 10 | 5 |
| 704sq | 495,616 | 1.44-1.47 | 5 | 10 | 5 |
| 1024sq | 1,048,576 | 1.00-1.01 | 5 | 10 | 4 |
| **Total** | | | | **60** | **26** |

### Resolution Details

Each resolution was sampled by `sample_random_resolution(budget_pixels, rng)` from
`src_ii/resolution_sampling.py`. The RNG (seed=777) produces log-uniform aspect ratios
in [0.5, 2.0], quantized to 32px alignment.

**256sq tier** (65,536 px):
- 224x320 (AR=0.70, portrait), 256x256 (AR=1.00, square), 224x288 (AR=0.78, portrait x3)

**320sq tier** (102,400 px):
- 384x256 (AR=1.50), 416x224 (AR=1.86), 416x256 (AR=1.62), 352x288 (AR=1.22), 288x352 (AR=0.82)

**384sq tier** (147,456 px):
- 320x448 (AR=0.71), 544x288 (AR=1.89), 512x288 (AR=1.78 x2), 448x352 (AR=1.27)

**512sq tier** (262,144 px):
- 384x672 (AR=0.57), 608x448 (AR=1.36), 704x384 (AR=1.83), 352x736 (AR=0.48), 576x448 (AR=1.29)

**704sq tier** (495,616 px):
- 640x800 (AR=0.80), 736x672 (AR=1.10), 544x928 (AR=0.59), 800x640 (AR=1.25), 576x864 (AR=0.67)

**1024sq tier** (1,048,576 px):
- 1024x1024 (AR=1.00), 1280x832 (AR=1.54 x2), 1088x960 (AR=1.13), 736x1408 (AR=0.52)

Aspect ratios range from 0.48 (352x736, extreme portrait) to 1.89 (544x288, extreme
landscape). Both portrait and landscape orientations are well represented in every tier.

## 3. Sigma Schedule Behavior Across Resolutions

The SD3 Eq.23 resolution shift `alpha = sqrt(ref_pixels / target_pixels)` dramatically
changes the sigma schedule for small images. The shift is capped at MAX_SHIFT=8.0 per
the fix documented in `docs/essay_sigma_schedule_fix.md`.

### Representative Sigma Schedules

**224x320 (256sq, shift=3.854):**
```
step_00: 1.000  step_04: 0.961  step_09: 0.898  step_14: 0.816
step_19: 0.691  step_24: 0.490  step_29: 0.120  final: 0.000
```

**384x672 (512sq, shift=2.027):**
```
step_00: 1.000  step_04: 0.945  step_09: 0.859  step_14: 0.742
step_19: 0.584  step_24: 0.371  step_29: 0.079  final: 0.000
```

**640x800 (704sq, shift=1.442):**
```
step_00: 1.000  step_04: 0.918  step_09: 0.797  step_14: 0.641
step_19: 0.449  step_24: 0.246  step_29: 0.046  final: 0.000
```

**1280x832 (1024sq, shift=1.000):**
```
step_00: 1.000  step_04: 0.867  step_09: 0.699  step_14: 0.533
step_19: 0.334  step_24: 0.166  step_29: 0.033  final: 0.000
```

**Key observation:** At step 29 (penultimate), sigma ranges from 0.120 (256sq) down to
0.033 (1024sq). The 256sq images retain 3.6x more noise at the same step index. This
is correct -- smaller images need the shift to prevent "too clean too early" convergence.

The shift also affects logSNR weighting in the pair sampler. At step 14:
- 256sq: sigma=0.816, logSNR = 2*ln(0.184/0.816) = -2.96
- 1024sq: sigma=0.533, logSNR = 2*ln(0.467/0.533) = -0.26

So 1024sq step_14 positions get higher sampling weight than 256sq step_14 positions,
which is correct: the 1024sq image at step 14 is cleaner and more informative for the
reward model.

## 4. Per-Step Sigma Values

Each trajectory's metadata includes the full sigma schedule for its 8 saved positions
(steps 0, 4, 9, 14, 19, 24, 29, and final). These are recorded in two locations:

1. **Sidecar file:** `multi_res_trajectories/step_sigmas.json` maps trajectory index
   to `{step_key: sigma_value}` for all 60 trajectories.

2. **In-memory metadata:** The `step_sigmas` field in each trajectory's metadata dict
   is passed through the generation pipeline.

The "final" position always has sigma=0.0 (the fully denoised image). This is critical:
sigma=0 positions get FULL weight (logSNR=+inf) in the pair sampler's geometric decay
function. The pair sampler (`src_ii/pair_sampler.py:build_positions_from_v2`) can
reconstruct sigmas from `n_steps` and `sampling_shift` stored in the parquet index, but
the sidecar file provides explicit documentation and cross-validation.

## 5. Dataset Format

The dataset uses the V2 format (parquet index + safetensors blobs) from
`src/futudiffu/dataset_v2.py`.

### Structure

```
multi_res_trajectories/
  index.parquet              # 60 rows, one per trajectory
  blobs/
    blob_000.safetensors     # 2.6 MB (256sq tier, 10 trajectories)
    blob_001.safetensors     # 3.9 MB (320sq tier)
    blob_002.safetensors     # 5.8 MB (384sq tier)
    blob_003.safetensors     # 11 MB (512sq tier)
    blob_004.safetensors     # 20 MB (704sq tier)
    blob_005.safetensors     # 41 MB (1024sq tier)
  step_sigmas.json           # 13 KB, per-step sigma values
  generation_report.json     # 7 KB, timing and verification
  renders/                   # 60 PNGs, VAE-decoded final images
    mr_224x320_sdpa_p0_s400000.png
    mr_224x320_sage_p0_s400001.png
    ...
    mr_736x1408_sage_p0_s400059.png
```

Total dataset size: ~82 MB (blobs) + renders.

### Per-Trajectory Metadata (Parquet)

Each trajectory row contains:
- `traj_id`: Sequential integer (0-59)
- `prompt`, `prompt_idx`, `seed`, `cfg`: Generation parameters
- `width`, `height`: Pixel resolution
- `n_steps`: 30 (all trajectories)
- `attention_backend`: "sdpa" or "sage"
- `sampling_shift`: Resolution-dependent SD3 Eq.23 alpha
- `step_indices`: [0, 4, 9, 14, 19, 24, 29]
- `has_final`: True (all trajectories)
- `resolution_tier`: "256sq", "320sq", ..., "1024sq" (in metadata)

### Per-Trajectory Tensors (Safetensors)

Each trajectory stores 8 tensors:
- `step_00` through `step_29`: (16, H/8, W/8) bfloat16 latents at sparse steps
- `final`: (16, H/8, W/8) bfloat16 fully denoised latent

Latent shapes scale with resolution:
- 256sq: (16, 32-40, 28-40) depending on aspect ratio
- 1024sq: (16, 104-176, 92-160) depending on aspect ratio

## 6. Attention Backend Handling

Both SDPA and SageAttention INT8 QK backends are used:
- **SDPA** (30 trajectories): PyTorch scaled dot-product attention, BF16 precision.
  Gold standard. Used for `is_gold=True` marking.
- **SageAttention** (30 trajectories): INT8 QK quantization + BF16 PV accumulation.
  Introduces quantization artifacts that the BTRM scrimble head learns to detect.

The attention backend is switched globally via `futudiffu.attention.set_attention_backend()`
before each backend's generation batch. SageAttention is configured with
`qk_quant="int8"`, `pv_quant="bf16"` matching the production inference configuration.

## 7. Bin Packing Analysis

The 60 trajectories pack into 22 bins (REFERENCE_TOTAL_LEN=4224):
- **10 single-item bins** (1024sq images fill a bin alone)
- **12 multi-item bins** (smaller images pack together)
- **Overall utilization: 94.6%**
- **Sparse compute ratio: 0.60** (40% of attention computation is cross-image masking overhead)

### Bin Size Distribution

| Items/Bin | Count | Example |
|-----------|-------|---------|
| 1 | 10 | 1280x832 (4224/4224 = 100%) |
| 2 | 5 | 704sq pairs (4160/4224 = 98%) |
| 3 | 1 | 256sq + 320sq mix |
| 4 | 3 | 384sq-512sq mix |
| 6 | 1 | 320sq-384sq mix (4192/4224 = 99%) |
| 8 | 1 | 256sq-320sq mix (4192/4224 = 99%) |
| 11 | 1 | Tiny images (4224/4224 = 100%) |

The 11-item bin achieves 100% utilization -- 11 small images perfectly filling a single
FlexAttention forward pass. This is the exact scenario funfetti batching was designed
for: one GPU forward pass replaces 11 sequential passes.

## 8. Timing

| Phase | Time (s) | Description |
|-------|----------|-------------|
| TE encode | 6.4 | 10 prompts + negative |
| Model load | 3.4 | FP8 weights + compile wrapper |
| Generation | ~740 | 60 trajectories across 30 resolutions |
| Persist | ~1 | V2 dataset write (6 blobs) |
| VAE render | 7.7 | 60 final images decoded to PNG |
| Verification | ~4 | Shape checks, validity, readback |
| **Total** | **763.9** | **12.7 minutes** |

### Per-Tier Generation Timing

| Tier | Avg Time/Traj | Notes |
|------|--------------|-------|
| 256sq | ~5s | Fast inference, but first resolution triggers torch.compile (~48s) |
| 320sq | ~4-5s | Small images, fast after compile |
| 384sq | ~4-5s | Small images |
| 512sq | ~4.5s | Medium images |
| 704sq | ~9-24s | Large images, new shape recompiles |
| 1024sq | ~20-41s | Full resolution, recompile on first occurrence |

torch.compile recompilation is triggered by each new tensor shape. After 8
recompilations, PyTorch falls back to eager mode, so later resolutions run
without compilation benefit but without compilation overhead.

## 9. Sample Renders

Visual inspection confirms coherent image generation across all resolutions:

- **1280x832** (1024sq, landscape): Detailed laser shark illustration with legible text
- **736x1408** (1024sq, extreme portrait): Tall composition with text overlays
- **640x800** (704sq, portrait): Blue-toned shark with visible text rendering
- **384x672** (512sq, portrait): "LARGE LANGUAGE MODEL" text with colorful shark
- **416x224** (320sq, extreme landscape): "QWEN-3-4B" text on blue background
- **256x256** (256sq, square): Low-detail but recognizable shark silhouette

Quality degrades gracefully with resolution: 1024sq images are photorealistic,
512-704sq images are cartoon-like but coherent, 256-320sq images show basic structure
and color. This is expected and correct -- the model's capacity to generate detail is
fundamentally limited by pixel budget.

## 10. Compatibility

The dataset is fully compatible with the existing training pipeline:

1. **DatasetReader** reads the parquet index and lazy-loads tensors from blobs.
2. **build_positions_from_v2()** indexes all (trajectory, step) positions with
   correct resolution-dependent sigma reconstruction.
3. **BTRMPairSampler** forms pairs across trajectories with FLOPS-weighted allocation
   and logSNR-weighted step selection.
4. **BinPackScheduler** packs mixed-resolution pairs into FlexAttention bins.
5. **score_differentiable_packed()** processes packed forward passes with per-image
   sigma schedules.

The `sampling_shift` value in each trajectory's parquet row ensures the pair sampler
reconstructs the correct sigma schedule. The `step_sigmas.json` sidecar provides
cross-validation.

## 11. Limitations and Future Work

1. **Single prompt per trajectory:** All trajectories use prompt 0 (the "enormous laser
   shark" golden reference). A production dataset should vary prompts to exercise the
   text encoder's content diversity. This was a deliberate simplification for the
   initial multi-resolution exercise.

2. **No step-count variation:** All trajectories use 30 steps. For scrongle head
   training (step-count discrimination), trajectories at 8-22 steps are needed. These
   can be generated in a follow-up run using the same resolution plan.

3. **Resolution clustering at small tiers:** The 256sq tier has only 3 unique
   resolutions (out of 5 sampled) due to the coarse 32px grid at low pixel budgets.
   This is a fundamental discretization effect, not a sampling bug.

## 12. Bug Fix: Final Position Sigma in Pair Sampler

During validation, discovered that `build_positions_from_v2()` in
`src_ii/pair_sampler.py` assigned `sigmas[-2]` to "final" positions. For a 30-step
schedule, `sigmas[-2]` = `sigmas[29]` = the last step's INPUT sigma (e.g., 0.034 at
reference resolution, 0.120 at 256sq). But the "final" image is the OUTPUT of the last
step -- it has sigma=0.0 (fully denoised).

**Before fix:** Final positions got sigma~0.12 (256sq) to sigma~0.034 (1024sq), giving
sampling weight ~0.63-0.92 instead of 1.0.

**After fix:** Final positions correctly get sigma=0.0, logSNR=+inf, weight=1.0.

This is a meaningful correctness fix: 60 out of 480 positions (12.5%) were being
slightly underweighted. In training, this would bias the gradient toward penultimate-step
comparisons at the expense of clean-vs-clean comparisons, subtly degrading the reward
model's ability to distinguish quality at the final denoising step.

---

**Files modified:**
- `src_ii/rollout.py` -- Added `step_sigmas` dict to returned metadata (per-step sigma values)
- `src_ii/pair_sampler.py` -- Fixed "final" position sigma from `sigmas[-2]` to 0.0
- `scripts_ii/generate_multi_res_trajectories.py` -- Added attention backend switching,
  sigma metadata propagation, sidecar file writing, reduced to 5 per tier

**Files created:**
- `multi_res_trajectories/` -- Complete V2 dataset (60 trajectories, 82 MB)
- `multi_res_trajectories/step_sigmas.json` -- Per-step sigma sidecar
- `multi_res_trajectories/renders/` -- 60 VAE-decoded PNGs
- `docs/essay_multi_res_dataset_generation.md` -- This essay

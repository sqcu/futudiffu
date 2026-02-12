# futudiffu — Handoff Document

## What This Is

A bug-for-bug, float-for-float reimplementation of one specific ComfyUI Z-Image
workflow as a standalone Python package. The goal is an "emergency exfiltration
port" — exact replication first, redesign later.

This is NOT a redesign, NOT a "better ComfyUI," NOT an architecture project.
Copy ComfyUI's behavior faithfully, including its bugs.

## Current State (as of handoff)

### Done
- All modules written and import-checked:
  - `sampling.py` — CONST model, sigma schedule, euler sampler, Flux latent format
  - `attention.py` — SDPA, two RoPE styles (Flux/Lumina + Llama/Qwen), RMSNorm
  - `text_encoder.py` — Qwen3-4B architecture, tokenizer, chat template
  - `diffusion_model.py` — NextDiT Z-Image variant
  - `vae.py` — AutoencoderKL decode-only path
  - `fp8_kernels.py` — Triton FP8 GEMM kernels (from QuantOps-reference)
  - `fp8.py` — FP8Linear wrapper, blockwise quantization
  - `generate.py` — end-to-end pipeline + CLI
  - `tokenizer/` — Qwen2 tokenizer data files
- `bootstrap.py` working: can run `uv sync`, import checks, arbitrary code
- `pyproject.toml` configured with hatchling build, triton as optional extra

### NOT Done
- **torch is CPU-only** (`torch 2.10.0+cpu cuda=False`). Must be fixed before any
  GPU testing. See "Fixing torch CUDA" below.
- **No model loading tested** — weight key mapping in load_* functions is untested
- **No inference tested** — no forward pass has ever run
- **No validation done** — none of the 6 validation criteria have been checked
- **FP8 text encoder path** in generate.py has a dead placeholder (`sd = torch.load if False else None`)
- **Weight prefix stripping** in load functions may have bugs

## Cross-Platform Execution Environment

This is a Windows Python venv accessed from WSL2.

### Key facts
- **venv Python**: `/mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe`
- This is a **Windows Python 3.12.8** — it sees Windows paths, not WSL paths
- When passing paths to this Python (in -c scripts or as arguments), use Windows paths:
  `F:\dox\repos\ai\futudiffu\...` NOT `/mnt/f/dox/repos/ai/futudiffu/...`
- BUT: the WSL path works for the *executable itself* (the shebang/ELF bridge handles it)

### Bootstrap pattern
```bash
# Run bootstrap commands:
/mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe 'F:\dox\repos\ai\futudiffu\bootstrap.py' sync
/mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe 'F:\dox\repos\ai\futudiffu\bootstrap.py' check
/mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe 'F:\dox\repos\ai\futudiffu\bootstrap.py' test-sampling
/mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe 'F:\dox\repos\ai\futudiffu\bootstrap.py' run "import torch; print(torch.cuda.is_available())"
```

### Running arbitrary futudiffu code
```bash
/mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe -c "import sys; sys.path.insert(0, r'F:\dox\repos\ai\futudiffu\src'); <your code here>"
```

### uv sync
User initializes `uv sync` from Windows PowerShell. After that, bootstrap.py can
re-sync if needed. Don't run bare `uv sync` from WSL — the venv is Windows-native.
The user is responsible for the initial venv creation and for adding the CUDA torch
index (see below).

## Fixing torch CUDA

The current torch is CPU-only because default PyPI wheels are CPU. To get CUDA torch,
`pyproject.toml` needs a PyTorch CUDA wheel index. This requires adding to pyproject.toml:

```toml
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"

[tool.uv.sources]
torch = [{ index = "pytorch-cu126" }]
```

Then the user re-runs `uv sync --extra triton-windows` from PowerShell.

**IMPORTANT**: The exact CUDA wheel version (cu124, cu126, cu128) depends on the
user's CUDA toolkit. The system has CUDA 12.8 on PATH. Check what torch CUDA builds
are available. Don't guess — ask the user or check the PyTorch wheel index.

## ComfyUI as Read-Only Reference

You may READ from the ComfyUI repository to verify correctness, trace code paths,
and understand behavior. You must NOT WRITE to ComfyUI (no new files, no edits,
no feature additions). The one exception: you may run existing ComfyUI test/dump
scripts to capture intermediate tensors for validation on the futudiffu side.

### ComfyUI paths
- **Root**: `/mnt/f/dox/ai/comfyui/ComfyUI/` (double-nested, this is correct)
- **Architecture review**: `/mnt/f/dox/ai/comfyui/COMFYUI_ARCHITECTURE_REVIEW.md`
- **QuantOps reference** (with kernel fixes): `/mnt/f/dox/ai/comfyui/QuantOps-reference/`

### ComfyUI source files you'll need to reference
- `comfy/model_sampling.py` — CONST, ModelSamplingDiscreteFlow, time_snr_shift
- `comfy/k_diffusion/sampling.py` — sample_euler, to_d
- `comfy/samplers.py` — simple_scheduler, CFGGuider, calc_cond_batch
- `comfy/latent_formats.py` — Flux latent format
- `comfy/ldm/lumina/model.py` — NextDiT (Z-Image diffusion model)
- `comfy/ldm/flux/math.py` — RoPE implementation
- `comfy/ldm/flux/layers.py` — EmbedND, timestep_embedding
- `comfy/ldm/modules/diffusionmodules/mmdit.py` — TimestepEmbedder
- `comfy/ldm/modules/attention.py` — attention backends
- `comfy/ldm/common_dit.py` — pad_to_patch_size
- `comfy/text_encoders/z_image.py` — ZImageTEModel, chat template
- `comfy/text_encoders/llama.py` — Qwen3_4B architecture
- `comfy/rmsnorm.py` — rms_norm kernel
- `comfy/ldm/models/autoencoder.py` — AutoencoderKL
- `comfy/ldm/modules/diffusionmodules/model.py` — Decoder, ResnetBlock, AttnBlock
- `comfy/sd.py` — VAE loading, detect_te_model
- `QuantOps-reference/kernels/fp8_kernels.py` — fixed FP8 GEMM Triton kernels

### Model weight files
| File | WSL Path | Size |
|---|---|---|
| Diffusion (BF16) | `.../models/diffusion_models/z_image_bf16.safetensors` | 12G |
| Diffusion (FP8 blockwise) | `.../models/diffusion_models/z_image_fp8_blockwise.safetensors` | 5.8G |
| Text encoder (BF16) | `.../models/text_encoders/qwen_3_4b.safetensors` | 7.5G |
| Text encoder (FP8 blockwise) | `.../models/text_encoders/qwen_3_4b_fp8_blockwise.safetensors` | 3.8G |
| VAE | `.../models/vae/zimage.safetensors` | 160M |

Base path: `/mnt/f/dox/ai/comfyui/ComfyUI/models/`

Windows path equivalent: `F:\dox\ai\comfyui\ComfyUI\models\...`

### Reference workflow
`/mnt/f/dox/ai/comfyui/ComfyUI/user/default/workflows/zimage_blockquant_lasershark.json`

## Two Test Configurations

### Config B: "Golden" (test this first)
- diffusion_model: `z_image_fp8_blockwise.safetensors` (FP8 blockwise bs=128)
- text_encoder: `qwen_3_4b.safetensors` (BF16 full precision)
- vae: `zimage.safetensors`
- prompt: the "ahem *ting ting*... enormous laser sharks" text (from workflow JSON)
- negative_prompt: ""
- seed: 91849188298864, steps: 30, cfg: 4.0
- sampler: euler, scheduler: simple
- width: 1280, height: 832
- sampling_shift: 1.0, multiplier: 1.0, denoise: 1.0
- Expected: correct laser shark imagery

### Config A: "Defective" (test second)
- Same as Config B except text_encoder: `qwen_3_4b_fp8_blockwise.safetensors`
- Expected: kaleidoscopic artifacts (BAD but REPRODUCIBLE)

## Validation Strategy

Validation is done by comparing futudiffu outputs against ComfyUI's actual intermediate
tensors at each pipeline stage. To get ComfyUI's tensors, you can write a small Python
script that hooks into ComfyUI's pipeline and dumps tensors to .pt files, then load
those on the futudiffu side for comparison. Alternatively, add torch.save() calls at
key points in ComfyUI's code to capture tensors during a run.

### Stages to validate (in order)
1. **Sigma schedule** — `build_sigmas` + `simple_scheduler` output. Pure math, no GPU needed.
2. **Noise tensor** — `torch.randn` with `manual_seed(seed)`. Must match bitwise.
3. **Text encoder hidden states** — positive and negative. MSE < 1e-6.
4. **Single euler step** — one denoising iteration. MSE < 1e-6.
5. **Full 30-step pipeline** — final latent before VAE. MSE < 1e-6.
6. **VAE decode** — final image tensor.
7. **Config A replication** — FP8 text encoder produces same distortion (cosine sim > 0.99).

### Validation criteria
- Config B (golden): MSE < 1e-6 on final latent vs ComfyUI output
- Config A (defective): cosine sim > 0.99 with ComfyUI's defective output
- Per-step latents: MSE < 1e-6 at each of 30 euler steps
- Sigma schedule: exact bitwise match (float32)
- Noise tensor: exact bitwise match (same seed, same randn)

## Pipeline Detail

### Pipeline steps (from ComfyUI source, verified)
1. Text encode: wrap prompt in `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n` → tokenize → encode → layer_idx=-2 hidden states (B, seq, 2560)
2. Text encode negative: same template around ""
3. Empty latent: `zeros(1, 16, height//8, width//8)` = `zeros(1, 16, 104, 160)`
4. Noise: `torch.randn(1, 16, 104, 160)` with `manual_seed(91849188298864)`
5. Sigmas: `simple_scheduler(build_sigmas(shift=1.0, multiplier=1000.0), 30)` → 31 values
6. CONST noise_scaling: `x = sigma * noise + (1 - sigma) * latent`
7. Euler loop x30: CFG with scale=4.0, `d = (x - denoised) / sigma`, `dt = sigma_next - sigma`, `x = x + d * dt`
8. CONST inverse_noise_scaling: `x / (1 - sigma_last)` (no-op when sigma_last=0)
9. Flux process_out: `(latent / 0.3611) + 0.1159`
10. VAE decode → `clamp((decoded + 1) / 2, 0, 1)`
11. `np.clip(image * 255, 0, 255).astype(np.uint8)`

### Diffusion model: NextDiT Z-Image
- dim=3840, n_heads=30, n_kv_heads=30 (full MHA, NOT GQA)
- axes_dims=[32,48,48], axes_lens=[1536,512,512], rope_theta=256.0
- ffn_dim_multiplier=8.0/3.0, z_image_modulation=True, time_scale=1000.0
- in_channels=16, patch_size=2, n_refiner_layers=2
- Forward: `t = 1.0 - timesteps`, `t_embed = t_embedder(t * 1000.0)`
- Patchify → context_refiner → main layers → final_layer
- **Returns `-img`** (negation! critical for CONST.calculate_denoised to work)
- pad_tokens_multiple=32

### Text encoder: Qwen3-4B
- hidden=2560, 36 layers, 32 attn heads, 8 kv heads (GQA 4:1)
- intermediate=9728, vocab=151936, head_dim=128
- rope_theta=1000000, rms_norm_eps=1e-6, mlp_activation=silu
- q_norm="gemma3", k_norm="gemma3"
- Output: layer_idx=-2 (second to last hidden state)

### VAE: AutoencoderKL
- Flux latent format: scale_factor=0.3611, shift_factor=0.1159
- latent_channels=16, 8x downscale
- Decode path only (no encode needed)

### CONST model type
- `calculate_input(sigma, noise) = noise` (identity)
- `calculate_denoised(sigma, output, input) = input - output * sigma`
- `noise_scaling(sigma, noise, latent) = sigma * noise + (1-sigma) * latent`
- `inverse_noise_scaling(sigma, latent) = latent / (1-sigma)`

### ModelSamplingDiscreteFlow
- `sigma(timestep) = time_snr_shift(shift, timestep / multiplier)`
- `timestep(sigma) = sigma * multiplier`
- With shift=1.0, multiplier=1.0: identity (sigma = t)
- `time_snr_shift(alpha, t) = alpha * t / (1 + (alpha-1) * t)` — identity when alpha=1

### simple_scheduler
- `ss = len(sigmas) / steps` = 1000/30 ~ 33.333
- `sigs[i] = sigmas[-(1 + int(i * ss))]` for i in range(30)
- Appends 0.0 → 31 values total

### Euler sampler
- `to_d(x, sigma, denoised) = (x - denoised) / sigma`
- `dt = sigmas[i+1] - sigmas[i]` (negative, stepping down)
- `x = x + d * dt`
- No s_churn (0 by default)

## Known Potential Issues

1. **Weight key prefixes**: ComfyUI state dicts may have `model.diffusion_model.` or
   `diffusion_model.` prefixes that need stripping. The load functions have prefix
   stripping but it's untested against actual files.

2. **FP8 state dict format**: FP8 blockwise safetensors have per-layer `.weight` in
   float8_e4m3fn and `.weight_scale` in float32 with shape `(ceil(out/128), ceil(in/128))`.
   The `.comfy_quant` metadata key indicates the quantization format.

3. **RoPE implementations**: There are TWO different RoPE styles in this project:
   - Flux/Lumina RoPE (2x2 rotation matrices) — used by NextDiT diffusion model
   - Llama/Qwen RoPE (half-rotate) — used by Qwen3-4B text encoder
   These are NOT interchangeable. `attention.py` has both.

4. **NextDiT returns `-img`**: The diffusion model's forward pass negates the output.
   This is intentional and required for `CONST.calculate_denoised` to produce the
   correct result: `input - (-img) * sigma = input + img * sigma`.

5. **generate.py FP8 text encoder placeholder**: Line 99 has dead code
   (`sd = torch.load if False else None`). The FP8 text encoder path needs cleanup.

6. **Noise generation**: torch.randn with manual_seed must be on CUDA device to match
   ComfyUI's behavior. CPU randn and CUDA randn produce different sequences for the
   same seed.

## Hardware

- RTX 4090 class (SM 8.9, Ada Lovelace)
- Native FP8 tensor core support (float8_e4m3fn)
- CUDA 12.8 on PATH
- MSVC Build Tools 2022 + Windows SDK 10 (for Triton compilation)
- triton-windows-3.6.0.post25

## User Preferences (MUST follow)

- No CPU implementations ever — CUDA only
- No shell scripts — Python only
- No conda (crashes the shell)
- No emojis
- `python` and `python3` commands are redirected to `uv run` by shell hygiene hooks
- `pip` is blocked
- Use the `.venv/Scripts/python.exe` directly to bypass all of the above
- Prefer "defect legibilizing" patches over "defect concealing" workarounds
- `uv` with `pyproject.toml` for all environments

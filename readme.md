## futudiffu

Standalone Z-Image diffusion pipeline. Bug-for-bug ComfyUI port with performance
optimizations (unified BF16, CFG batching, torch.compile, FP8 blockwise quantized
diffusion model, SageAttention Triton kernels).

### Single image generation

```bash
.venv/Scripts/python.exe generate.py \
    --prompt "your prompt here" \
    --seed 12345 \
    --steps 30 \
    --cfg 4.0 \
    --width 1280 --height 832 \
    --diffusion-model "F:\path\to\z_image_fp8_blockwise.safetensors" \
    --text-encoder "F:\path\to\qwen_3_4b.safetensors" \
    --vae "F:\path\to\zimage.safetensors"
```

---

### Inference server

Long-running process that owns the GPU. Loads models on demand, manages VRAM
lifecycle (TE and diffusion are mutually exclusive, VAE coexists with either).
All inference goes through here -- dataset generation, rendering, interactive
eval, future RL rollout collection.

```bash
# Start the server:
.venv/Scripts/python.exe -m futudiffu.server --port 5555 \
    --fp8-diff "F:\path\to\z_image_fp8_blockwise.safetensors" \
    --te "F:\path\to\qwen_3_4b.safetensors" \
    --vae "F:\path\to\zimage.safetensors"
```

RPC methods: `encode_prompt`, `sample_trajectory` (t2i and i2i), `vae_encode`,
`vae_decode`, `warmup`, `status`, `free`. Communication via ZeroMQ multipart
frames (JSON envelope + raw tensor bytes, no pickle).

---

### BTRM dataset generation

Generates paired diffusion trajectory data for training BTRM (Bradley-Terry
Reward Model) heads that discriminate:
- **scrongle**: step count degradation (30-step gold vs fewer-step variants)
- **scrimble**: precision artifacts (BF16 gold vs FP8 quantized variants)

Each trajectory saves per-step latents at configurable checkpoints, decodeable
by the VAE for visual QA or downstream reward model training.

The generator is a pure scheduling client -- no model loading, no GPU code.
It talks to the inference server for all compute.

#### Dataset composition

| Family | Description | Default count |
|--------|-------------|---------------|
| t2i | Text-to-image from gaussian noise. 24 prompt templates x sampled seeds x step schedules x precisions. | 1908 |
| i2i | Image-to-image from 11 off-policy reference images. Forward-noised to sampled denoise strengths (0.3-0.7), denoised with object label + sampled style transform. | 396 |
| **Total** | | **2304** |

#### Quick start

```bash
# Start the server (separate terminal):
.venv/Scripts/python.exe -m futudiffu.server --port 5555 \
    --fp8-diff "F:\...\z_image_fp8_blockwise.safetensors" \
    --te "F:\...\qwen_3_4b.safetensors" \
    --vae "F:\...\zimage.safetensors"

# Preview the schedule without generating:
.venv/Scripts/python.exe generate_btrm_dataset.py --dry-run

# Generate with inline counts (interruptible with Ctrl-C):
.venv/Scripts/python.exe generate_btrm_dataset.py \
    --t2i 40 --i2i 10 --render 6

# Generate from a schedule file:
.venv/Scripts/python.exe generate_btrm_dataset.py \
    --schedule schedule.json

# Resume after interruption (re-run the same command):
.venv/Scripts/python.exe generate_btrm_dataset.py \
    --t2i 40 --i2i 10 --render 6
```

#### Schedule format

```json
[
  {"type": "t2i", "count": 10, "precision": "sdpa", "steps": 30, "render": 3},
  {"type": "t2i", "count": 10, "precision": "sage", "steps": [8, 22]},
  {"type": "i2i", "count": 5,  "precision": "sdpa", "render": 2}
]
```

Each batch: `type` (t2i/i2i), `count`, `precision` (sdpa/sage), `steps` (int
or [min, max] range), `render` (how many to VAE-decode inline), `denoise`
(i2i only, float or [lo, hi]).

#### Interruptibility and resume

SIGINT (Ctrl-C) finishes the current trajectory, saves the manifest, and exits.
A second Ctrl-C force-quits. Re-running the same command skips completed
trajectories automatically (RNG stays synchronized across resume).

---

### Rendering trajectories

Retroactively VAE-decode latents to PNG for any subset of an existing dataset.
Connects to the same inference server.

```bash
# Render all i2i trajectories (final + every checkpoint step):
.venv/Scripts/python.exe render_trajectories.py \
    --dataset-dir btrm_dataset --type i2i --steps all

# Render only final frames for everything:
.venv/Scripts/python.exe render_trajectories.py \
    --dataset-dir btrm_dataset

# Specific checkpoint steps:
.venv/Scripts/python.exe render_trajectories.py \
    --dataset-dir btrm_dataset --steps 0 14 29 final

# Only trajectories that haven't been rendered yet:
.venv/Scripts/python.exe render_trajectories.py \
    --dataset-dir btrm_dataset --missing-only

# Specific trajectory indices:
.venv/Scripts/python.exe render_trajectories.py \
    --dataset-dir btrm_dataset --traj 4 5 12 --steps all
```

Filters compose: `--type i2i --missing-only --steps all` renders all checkpoint
steps for i2i trajectories that don't already have a `final.png`.

#### Output structure

```
btrm_dataset/
    manifest.json           # Full plan + completed trajectory records
    generation.log          # Timestamped progress milestones
    latents/
        traj_0000/          # One directory per trajectory
            step_00.pt      # Latent checkpoint at step 0
            step_14.pt      # Mid-trajectory
            step_29.pt      # Late trajectory
            final.pt        # After inverse_noise_scaling
            meta.json       # {seed, prompt_idx, n_steps, precision, type, ...}
        traj_0001/
            ...
    renders/
        traj_0000/          # VAE-decoded PNGs for visual QA
            step_00.png
            step_14.png
            step_29.png
            final.png
        ...
```

#### i2i off-policy images

The `i2i_off_policies/` directory contains 11 reference images with hand-labeled
object descriptions. Each image is paired with sampled "transformative labels"
(style/medium transforms like "rendered as a detailed oil painting") that modify
the visual treatment while preserving the spatial composition.

Images are used at native resolution (center-cropped to nearest multiple of 16
for VAE alignment). No resampling — any interpolation kernel imposes spectral
assumptions that misrepresent non-photographic sources (pixel art, line art,
dithered images). Source dimensions range from 256x256 to 832x1280.

Denoise strengths control structure preservation. Z-Image's flow-matching
(CONST model) has a very strong structural prior — denoise below 0.75 produces
negligible transformation regardless of CFG scale. Effective range:
- **0.75**: Subtle changes, source composition clearly dominant
- **0.85**: Moderate restyling, recognizable composition with visible style shift
- **0.95**: Strong transformation, source silhouette as loose guide only

#### Pair construction

Training pairs for the BTRM heads are constructed post-hoc from the trajectory
pool using `btrm_dataset.build_training_pairs()`:

- **scrongle pairs**: same (seed, prompt), gold=30 steps vs variant=fewer steps
- **scrimble pairs**: same (seed, prompt, steps), gold=BF16 vs variant=FP8

Mixed variants (fewer steps AND FP8) contribute to both heads as
"doubly degraded" negatives.

---

### Architecture

```
src/futudiffu/
    server.py            # Inference server (ZeroMQ, model lifecycle, all GPU ops)
    client.py            # InferenceClient (thin ZeroMQ wrapper)
    protocol.py          # Serialization: JSON envelope + raw tensor bytes
    generate.py          # Single-image generation pipeline + CLI
    sampling.py          # CONST model, sigma schedule, euler sampler
    attention.py         # SDPA + SageAttention dispatch, RoPE (Flux + Llama)
    text_encoder.py      # Qwen3-4B, tokenizer, chat template
    diffusion_model.py   # NextDiT Z-Image (30 layers, 30 heads, dim=3840)
    vae.py               # AutoencoderKL encode + decode, Flux latent format
    fp8.py               # FP8Linear, blockwise quantization
    fp8_kernels.py       # Triton FP8 GEMM kernels
    sage_kernels.py      # SageAttention Triton kernels (FP8/INT8 QK, BF16 PV)
    sage_attention.py    # SageAttention Python API + SM dispatch
    btrm_dataset.py      # Dataset config, plan builder, pair construction
generate_btrm_dataset.py # BTRM dataset client (schedule-driven, server-backed)
render_trajectories.py   # Retroactive VAE decode of existing trajectories
bootstrap.py             # Dependency management, import checks, quick tests
```

### Execution environment

Windows Python venv accessed from WSL2. The `.venv/Scripts/python.exe` binary
is a Windows executable invoked via WSL's PE/COFF interop layer. All path
arguments to Python scripts must use Windows paths (`F:\...`).

```bash
# Check CUDA availability:
.venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available())"

# Run bootstrap checks:
.venv/Scripts/python.exe bootstrap.py check
```

Hardware: RTX 4090 (SM 8.9, Ada Lovelace), CUDA 12.8, torch 2.10.0+cu128.

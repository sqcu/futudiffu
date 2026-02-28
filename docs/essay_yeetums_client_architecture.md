# Yeetums Client Architecture and the src → src_ii Migration

## What This Document Covers

The migration of inference, training orchestration, and interactive evaluation
from standalone scripts to a two-tier HTTP architecture: a GPU inference server
(`src_ii/server.py`) and a torch-free BFF web client (`src_ii/client_yeetums/`).

## The Two Servers

### GPU Inference Server (`src_ii/server.py`, port 8787)

The FastAPI server that owns the GPU. Responsibilities:

- Model loading: FP8 checkpoint, text encoder, VAE, fuse ops, torch.compile
- Multi-LoRA lifecycle: allocate, init, load/save adapters, set scales
- BTRM lifecycle: load reward adapter + score head, scoring endpoints
- Inference queue: enqueue → TE encode → diffusion → SSE progress → result PNG
- Training orchestrator: background BTRM/DDGRPO/policy-intervention runs with SSE
- VAE encode/decode endpoints for image ↔ latent conversion

Launched via `scripts_ii/launch_server.py`. Binds to `0.0.0.0:8787`.

### BFF Client (`src_ii/client_yeetums/`, port 8001)

A torch-free FastAPI application that serves the web UI and proxies to the GPU
server. **Never imports torch.** All tensor data flows as opaque bytes through
`InferenceBridge`.

Launched via `scripts_ii/launch_yeetums.py`. Binds to `0.0.0.0:8001`.

## Package Layout

```
src_ii/client_yeetums/
├── __init__.py              # re-exports create_app
├── app.py                   # FastAPI app factory, all routes (~850 lines)
├── bridge.py                # InferenceBridge: httpx client to GPU server
├── gallery.py               # Gallery: PNG + JSONL metadata on disk
├── models.py                # Pydantic request/response models
├── presets/                  # Config preset JSON files (file-based discovery)
│   ├── inference_default.json
│   ├── policy_intervention.json
│   ├── reward_intervention.json
│   ├── btrm_training.json
│   └── ddgrpo_training.json
└── static/                  # Vanilla JS frontend
    ├── index.html
    ├── style.css
    ├── app.js               # Boot, status polling, module init
    ├── config_flow.js        # Config editor, preset dropdown, distributional controls
    ├── config_geometry.js    # Resolution controls, aspect ratio slider
    ├── generate.js           # Batch generation dispatch, SSE streaming, job tracking
    ├── gallery.js            # Gallery grid, image preview, metadata display
    ├── output_config.js      # Output format controls
    └── arrows.js             # Resolved-config copy-back arrows
```

## Data Flow: Inference Generation

```
Browser (config_flow.js)
  → POST /api/batch_generate   [BFF: resolve distributional config k times]
  → POST /enqueue              [GPU server: TE encode + queue]
  ← SSE /api/stream/{job_id}   [BFF proxies GPU server's SSE]
  → GET /result/{job_id}       [BFF fetches PNG from GPU server]
  → gallery.add()              [BFF saves to disk, emits gallery_ready SSE]
  ← gallery_ready event        [Browser adds image to gallery grid]
```

## Data Flow: Policy Intervention

```
Browser (preset dropdown → policy_intervention.json)
  → POST /api/batch_generate   [BFF: detects type=policy_intervention]
  → POST /training/start       [GPU server: TrainingOrchestrator._run_policy_intervention]
  ← SSE /api/train/stream/{id} [BFF proxies training SSE]
     Phase 1: TE encode prompts
     Phase 2: Load model, install multi-LoRA, load BTRM + optional policy adapter
     Phase 3: For each (prompt, seed): sample ref trajectory + policy trajectory
     Phase 4: VAE decode, false-color diff, composite assembly
     Phase 5: artifact_ready events → BFF fetches PNGs → gallery
  ← gallery_ready events       [Browser shows ref, policy, diff, composite, score chart]
```

## Config Type Routing

The `type` field in a config JSON determines how `POST /api/batch_generate`
handles it:

| type | Route | Handler |
|------|-------|---------|
| `inference` (default) | Queue-based generation | `InferenceQueue` |
| `policy_intervention` | `TrainingOrchestrator._run_policy_intervention` | A/B sampling + composites |
| `btrm` | `TrainingOrchestrator._run_btrm_training` | Bradley-Terry reward model training |
| `ddgrpo` | `TrainingOrchestrator._run_ddgrpo` | REINFORCE policy optimization |
| `validate` | `TrainingOrchestrator._run_validation` | Pinkify/TNT/decorrelation checks |

All training types share the same BFF→server pathway: `bridge.start_training_run(config)`
→ `POST /training/start` → SSE streaming → artifact interception → gallery.

## Preset Config Discovery

Implemented as filesystem scanning, not a database:

```
GET /api/config/presets → scans src_ii/client_yeetums/presets/*.json
Returns: { presets: [{ name: "policy_intervention", config: {...} }, ...] }
```

Adding a new run type to the UI = dropping a `.json` file into `presets/`.
Frontend populates a `<select>` dropdown dynamically from the API response.
Selecting a preset replaces the JSON editor contents; the user can then modify
fields before hitting Generate.

### Current Presets

- **inference_default**: Standard text-to-image config with distributional seed range
- **policy_intervention**: DDGRPO policy adapter A/B eval with composites and diffs
- **reward_intervention**: r_theta adapter only (no policy checkpoint) — measures
  how the reward model's own adapter perturbs generation
- **btrm_training**: BTRM reward model training with multi-resolution dataset
- **ddgrpo_training**: REINFORCE policy optimization against trained BTRM

## What Moved from src/scripts to src_ii

### Server Infrastructure

| Old | New | What Changed |
|-----|-----|-------------|
| `src/futudiffu/server.py` (ZMQ) | `src_ii/server.py` (FastAPI) | Dead ZMQ → async HTTP + SSE. Queue-based inference. Training orchestration. |
| `src/futudiffu/client.py` (ZMQ) | `src_ii/client_yeetums/bridge.py` | ZMQ REQ/REP → httpx. Torch-free. Opaque tensor bytes. |
| `scripts/launch_server.py` | `scripts_ii/launch_server.py` | Points to src_ii server. Graceful shutdown. |
| N/A | `scripts_ii/launch_yeetums.py` | New: BFF launcher for web UI. |
| N/A | `src_ii/training_orchestrator.py` | New: server-side background run manager. |

### Model and LoRA Lifecycle

| Old | New | What Changed |
|-----|-----|-------------|
| `src/futudiffu/model_manager.py` | `src_ii/zimage_model.py` + `src_ii/btrm_lifecycle.py` | Monolithic manager → separate model + lifecycle free functions. |
| `src/futudiffu/lora.py` | `src_ii/multi_lora.py` | Single adapter → multi-adapter sparse routing. `_strip_orig_mod` for torch.compile. `add_adapter()` for incremental installation. |
| `src/futudiffu/btrm.py` | Deleted. `src_ii/btrm_lifecycle.py` | BTRMCompoundModel class → free functions. Score head is now inside ZImageRLAIF. |

### Training Scripts

| Old | New | What Changed |
|-----|-----|-------------|
| `scripts/train.py` | `src_ii/btrm_training.py` (library) | Script → importable function. Server-invocable via training orchestrator. |
| `scripts_ii/run_ddgrpo_v2.py` | `src_ii/ddreinforce.py` + `src_ii/policy_step.py` | Script → library modules. Policy intervention via orchestrator preset. |
| `scripts_ii/validate_policy_intervention.py` | `src_ii/training_orchestrator.py:_run_policy_intervention` | Standalone script → server-side orchestrated run with SSE + gallery output. |
| `scripts_ii/demonstrate_rtheta_policy.py` | Still exists, but superseded by reward_intervention preset | GUI equivalent via preset config. |

### Inference Pipeline

| Old | New | What Changed |
|-----|-----|-------------|
| `src/futudiffu/sampling.py` | `src_ii/infer/trajectory.py` + `src_ii/inference_sampling.py` | Sampling uses packed forward. sigma-shift per resolution. |
| `src/futudiffu/text_encoder.py` | `src_ii/infer/text_encoding.py` | Lifecycle-managed TE (load → encode → free). |
| `src/futudiffu/generate.py` | Queue in `src_ii/inference_queue.py` | Blocking generate → async queue with SSE progress. |
| `src/futudiffu/diffusion_model.py` | `src_ii/zimage_model.py` | Standalone diffusion → unified diffusion+scoring (ZImageRLAIF). |

## torch.compile and MultiLoRA

`torch.compile` wraps every module in `OptimizedModule`, adding `._orig_mod.`
to all paths returned by `named_modules()`. This breaks multi-LoRA operations
that construct keys from module paths (load/save adapter, install on target
modules, get adapter params).

The fix is `_strip_orig_mod()` at `multi_lora.py:327`:

```python
def _strip_orig_mod(name: str) -> str:
    return name.replace("._orig_mod", "")
```

Applied in six locations across `multi_lora.py` and `infer/model_setup.py`.

Additionally, `install_multi_lora` now detects already-wrapped `MultiLoRALinear`
modules and calls `add_adapter()` instead of re-wrapping. This is required for
policy intervention, which installs the BTRM adapter first, then the policy
adapter second.

Adapter scale tensors must be shaped `(n_images, n_adapters)` to match the
number of installed adapters. The training orchestrator builds these conditionally:

```python
# 2 adapters: [rtheta, policy_pinkify]
scales_ref    = [[1.0, 0.0]]   # reward on, policy off
scales_policy = [[1.0, 1.0]]   # both on

# 1 adapter: [rtheta] (reward-only intervention)
scales_ref    = [[0.0]]        # adapter off
scales_policy = [[1.0]]        # adapter on
```

## Validation Status (2026-02-27)

- Reward intervention (no policy checkpoint): GPU validated, composites + diffs in gallery
- Policy intervention (DDGRPO adapter): GPU validated, composites + diffs in gallery
- Inference generation: Working through preset + distributional configs
- BTRM training: Config preset exists, orchestrator path implemented
- DDGRPO training: Config preset exists, orchestrator path implemented

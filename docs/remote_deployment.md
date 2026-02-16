# Remote Deployment Guide

Deploy futudiffu to a spot H100 (or local RTX 4090) and validate the full
pipeline from zero to training-ready in a single deterministic script.

## Prerequisites

- Python 3.11+ with CUDA support
- NVIDIA GPU(s) with SM 8.9+ (RTX 4090) or SM 9.0 (H100)
- HuggingFace token (one of: `HF_TOKEN` env var, `.supersekrit` file, `huggingface-cli login`)
- ~15 GB disk for models (5.8G FP8 diff + 7.5G TE + 160M VAE + BF16 source)

## Quick Start

```bash
# One-liner: download models, quantize, launch servers, validate everything
python scripts/launch_remote.py --model-dir ./models --n-gpus 1

# Quick mode (skip kernel tests + gen stub, ~10 min)
python scripts/launch_remote.py --model-dir ./models --quick

# Multi-GPU
python scripts/launch_remote.py --model-dir ./models --n-gpus 2
```

## Phase Table

| Phase | What | Validates | Duration (4090) | Duration (H100) | On Fail |
|-------|------|-----------|------------------|------------------|---------|
| 0 | Environment check | Python, CUDA, packages, HF token | ~10s | ~10s | ABORT |
| 1 | Model bootstrap | Download from HF + FP8 quantize | 5-10 min | 3-5 min | ABORT |
| 2 | Kernel smoke tests | Triton compilation on target SM | 2-3 min | 1-2 min | WARN |
| 3 | Server launch | N server processes, port polling | 1-2 min | 1-2 min | ABORT |
| 4 | Pipeline validation | Canonical trajectory vs reference | 2-3 min | 1-2 min | ABORT |
| 5 | Gen stub | Multi-GPU generation + merge | 1-2 min/GPU | 1-2 min/GPU | WARN |
| 6 | BTRM stub | Head injection, loss, LoRA weights | 2-3 min | 1-2 min | WARN |
| 7 | Policy stub | Rollout gen, gradient, optimizer step | 3-5 min | 2-3 min | WARN |
| 8 | Summary | Timing table + suggested commands | instant | instant | always |

Total: ~15-25 min (4090), ~10-18 min (H100). Add ~5-10 min if models need downloading.

## Expected Cross-Architecture Behavior

When running on a different GPU architecture than the reference trajectory was
captured on (e.g., reference from RTX 4090 SM89, tested on H100 SM90):

- **Text encoder**: cos >= 0.99 expected (deterministic, no arch-dependent kernels)
- **Final latent**: cos 0.90-0.99 expected (FP8 GEMM rounding differs between SM89/SM90)
- **VAE output**: tracks final latent divergence

Thresholds in Phase 4:
- cos < 0.90 = **FAIL** (something is fundamentally broken)
- cos 0.90-0.99 = **WARN** (expected cross-arch, proceed)
- cos >= 0.99 = **PASS** (same-arch match)

## Proceeding to Real Training

After launch_remote.py completes successfully:

```bash
# Phase 8 prints a suggested command. Typical real run:
python scripts/train.py \
    --ports 5555 5556 \
    --dataset-dir btrm_dataset \
    --output-dir training_run_001 \
    --btrm-macrobatches 30 \
    --btrm-batch-size 32 \
    --policy-iterations 50 \
    --policy-group-size 4 \
    --render-every 8

# Or generate a fresh dataset first:
python scripts/generate_btrm_dataset.py \
    --t2i 40 \
    --server tcp://localhost:5555 \
    --output-dir btrm_dataset_fresh \
    --dataset-format v2
```

## Multi-GPU Operation

The launcher starts N independent servers, one per GPU:

```
GPU 0: port 5555, CUDA_VISIBLE_DEVICES=0
GPU 1: port 5556, CUDA_VISIBLE_DEVICES=1
...
```

Training uses `MultiGPUClient` which dispatches:
- **Rollout generation**: round-robin across all GPUs
- **Training RPCs** (BTRM step, policy grad, optimizer): primary GPU only (port 0)
- **TE encoding**: any GPU (first available)

Dataset generation (`--gpu-id`) writes per-GPU staging dirs, then
`merge_staged_datasets.py` unifies them with globally unique trajectory IDs.

## Spot Instance Resumability

What survives preemption:
- Downloaded models in `--model-dir` (Phase 1 has skip-if-present logic)
- FP8 quantized model (same)
- BTRM dataset on disk (if committed before preemption)
- `launch_report.json` from previous runs

What does NOT survive:
- Server processes (must relaunch)
- torch.compile caches (will rewarm, ~1-2 min)
- In-flight training state (LoRA weights are in server VRAM)
- Dataset v2 WIP blobs (buffers sealed only on writer close)

Resume strategy:
```bash
# Re-run the same command. Skip-if-present handles models.
python scripts/launch_remote.py --model-dir ./models --n-gpus 2

# After validation passes, resume training from last checkpoint:
python scripts/train.py \
    --ports 5555 5556 \
    --dataset-dir btrm_dataset \
    --output-dir training_run_001 \
    --resume \
    --skip-btrm \
    --policy-iterations 50
```

Save LoRA checkpoints frequently (`--checkpoint-every 5`). On preemption,
the last checkpoint is your recovery point.

## Cost Estimates (Prime Intellect Spot)

At $1/GPU-hr for H100:

| Activity | GPUs | Time | Cost |
|----------|------|------|------|
| Launch + validate | 2 | ~15 min | $0.50 |
| Generate 50 trajectories | 2 | ~25 min | $0.85 |
| BTRM training (30 macrobatches) | 1 | ~15 min | $0.25 |
| Policy optimization (50 iters) | 2 | ~2 hr | $4.00 |
| **Full pipeline from zero** | **2** | **~3 hr** | **~$6.00** |

Amortized over multiple runs: models download once, quantize once.
Subsequent launches with `--skip-download --skip-quantize` save ~$0.20.

## CLI Reference

```
python scripts/launch_remote.py [OPTIONS]

Options:
  --model-dir PATH         Model download/storage directory (default: ./models)
  --n-gpus N               Number of GPUs, 0=auto-detect (default: 0)
  --base-port PORT         Starting port for servers (default: 5555)
  --skip-download          Skip HF model download
  --skip-quantize          Skip BF16 -> FP8 quantization
  --skip-kernel-test       Skip kernel smoke tests (Phase 2)
  --skip-validation        Skip reference trajectory comparison (Phase 4)
  --skip-gen-stub          Skip dataset generation stub (Phase 5)
  --skip-btrm-stub         Skip BTRM training stub (Phase 6)
  --skip-policy-stub       Skip policy training stub (Phase 7)
  --dataset-dir PATH       BTRM dataset directory (default: btrm_dataset)
  --output-dir PATH        Validation output directory (default: remote_validation)
  --quick                  Alias: --skip-kernel-test --skip-gen-stub
```

## Troubleshooting

**Phase 0 fails: "CUDA not available"**
- Check `nvidia-smi` works
- Verify torch was installed with CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`

**Phase 1 fails: download errors**
- Verify HF token has access to `Comfy-Org/z_image`
- Check network connectivity: `curl -I https://huggingface.co`
- Pre-download manually and use `--skip-download`

**Phase 2 fails: kernel compilation errors**
- First run on a new GPU arch always recompiles Triton kernels
- Check `triton` version matches PyTorch: `python -c "import triton; print(triton.__version__)"`
- Phase 2 failure is WARN, not ABORT. Phase 4 is the true correctness gate.

**Phase 3 fails: server timeout**
- Check GPU VRAM: model needs ~8 GB minimum
- Check server logs: `cat remote_validation/server_gpu0.log`
- Try with fewer GPUs: `--n-gpus 1`

**Phase 4 fails: cos < 0.90**
- This indicates a fundamental issue (wrong model, broken kernels)
- Check diff images in `remote_validation/pipeline/`
- Compare `launch_report.json` stats against known-good values

**Phase 6/7 warns: training stubs fail**
- Check that `btrm_dataset/` has trajectory data
- Server logs may show OOM or RPC errors
- These phases are WARN, not ABORT — you can investigate and re-run training manually

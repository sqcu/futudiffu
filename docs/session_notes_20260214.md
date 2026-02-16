# Session Notes — 2026-02-14 (late night)

Notes dump of user-specified objectives, reframings, and actions initiated
since the "going to bed soon" inflection point. Written pre-autocompaction
to preserve context.

---

## User reframings (these override earlier assumptions)

### 1. HuggingFace upload is mid-hoc, not post-hoc
Dataset upload to HF happens DURING the rented node session, not after.
The node is a trajectory factory with an upload daemon running alongside
training. Nothing valuable accumulates only on the ephemeral node.

### 2. Continuous sync, not emergency flush
Instead of SIGTERM handlers for crash dumps, continuously rsync adapters
+ metrics to home machine. Adapters are ~10MB — trivial bandwidth. If
the node vanishes, you've lost the last few iterations, not the run.

### 3. Scale reality check
At $4-10 budget, preemption is unlikely. Runs are short (1-3 hours).
The architecture should be "politely close process and sync remaining
data" rather than "panic dump in 30-second preemption window."

### 4. Names lie, trajectories don't
File naming (e.g. "is ae.safetensors the same as zimage.safetensors?")
is resolved by operational equivalence: if the downloaded-and-requantized
model produces the same laser shark trajectory as the reference, the
files are correct. A string cannot tell you whether the tensors attached
to its filename match a reference implementation.

### 5. Reference trajectories belong in git
54MB of .pt files is not "a lot for git." C programmers put 800MB of
decision trees in repos. Our tensors are small and high information
content per MB. `stream_futudiffu/` is now tracked (un-gitignored)
as the canonical FP8 reference trajectory.

### 6. RTX PRO 6000 Blackwell is BAD for this workload
FP8 dense peak is only 504 TFLOPS (lower than 4090's 660). Blackwell's
tensor core gains went into FP4 and structured sparsity, not dense FP8.
Per-step is ~625ms vs 707ms on 4090 — barely faster. Worst cost-efficiency
of all rental options. H100 is the correct rental target.

### 7. Hybrid local + cloud is the budget play
Generate 2,304 trajectories overnight on local 4090 (free), rent 2xH100
for ~30 minutes of training only ($2.25). Uses all the data, stays under
budget.

---

## Actions completed this session

### Code written

| File | What | Status |
|------|------|--------|
| `train.py` | Production training script (diverse prompts, phased, JSONL metrics, checkpoint saves) | Written by agent, parses clean |
| `pack_trajectories.py` | Trajectory .pt → safetensors + JSONL packer. Verified round-trip on all 50 trajectories | Written + tested |
| `remote_tmux.py` | tmux session manager (launch/attach/watch/list/logs/kill/setup) | Written by agent |
| `remote_node_bootstrap.py` | HF download + FP8 quantize + trajectory validation | Written by agent, parses clean |

### Docs written

| File | What |
|------|------|
| `docs/remote_training_plan.md` | Operational plan + GPU perf analysis (Piece 6 appended with H100/RTX PRO 6000/4090 comparison) |
| `docs/multi_gpu_scaling.md` | N-GPU architecture design (trajectory gen trivially parallel, training stays single-GPU) |
| `docs/h100_optimization_opportunities.md` | 4 optimization topics: torch._scaled_mm, SM90 Triton configs, skip grad checkpointing, no-offload mode |
| `docs/remote_sessions.md` | tmux wrapper usage guide |

### .gitignore changes

- `bench_renders/` — UN-ignored (deliberate comparison images for documentation)
- `stream_futudiffu/` — UN-ignored (canonical FP8 reference trajectory, 54MB)
- `*.safetensors` — blanket ignored (model weights, lora checkpoints)
- Various dirs ignored: `btrm_dataset/renders/`, `i2i_off_policies/`, `stream_comfyui/`, `stream_futudiffu_bf16/`, `stream_futudiffu_f16te/`, `stream_compat_bf16/`, `lora_dumps_test/`, `lora_roundtrip_test/`, `throughput_study/`, `throughput_study_v2/`

---

## Actions in progress

### Bootstrap validation (agent aae7594, background)
Running `remote_node_bootstrap.py` locally:
1. Downloading 3 files from Comfy-Org/z_image (~20GB)
2. Quantizing BF16 diffusion model to FP8 blockwise
3. Starting validation server, generating canonical laser shark trajectory
4. Comparing per-step against `stream_futudiffu/` reference
**Still running** — downloads are the bottleneck.

---

## Scoped problems described but NOT yet attempted

### 1. torch._scaled_mm dispatch for H100
PyTorch PR #158037 (July 2025) added BlockWise128x128 support to
`torch._scaled_mm` on SM90+. Could replace our custom Triton FP8 GEMM
kernels with CUTLASS-backed built-in. Runtime arch dispatch (SM89→Triton,
SM90→_scaled_mm) would let same codebase run optimally on both.
**Documented in** `docs/h100_optimization_opportunities.md`, not implemented.

### 2. SM90 Triton kernel tuning
Larger tile configs (BLOCK_M=256, BLOCK_N=256, num_stages=6; BLOCK_M=128,
BLOCK_N=512, num_stages=5) become legal with H100's 228KB shared memory.
Moot if _scaled_mm replaces our kernels. **Documented, not implemented.**

### 3. Skip gradient checkpointing on H100
Per-block activations ~1.6GB at B=2, 30 blocks = ~48GB, fits in H100's
~66GB headroom. Saves ~30 block recomputes per backward pass. Add a
`--no-checkpoint` flag to training_utils.py.
**Documented, not implemented.**

### 4. No-offload mode on H100
All models (TE 7.5GB + diff 5.8GB + VAE 0.16GB = 13.5GB) fit in 80GB.
Load once, never offload. Eliminates lifecycle transitions + repeated
torch.compile warmups. Add `--no-offload` flag to server.py.
**Documented, not implemented.**

### 5. MultiGPUClient wrapper
~100-150 lines wrapping N InferenceClient instances. Trajectory gen
dispatched round-robin via ThreadPoolExecutor. Training RPCs route to
primary server. **Designed in** `docs/multi_gpu_scaling.md`, not written.

### 6. HuggingFace dataset upload integration
Upload daemon / script for pushing trajectory safetensors to a private HF
dataset repo as they're generated. **Mentioned in planning, not written.**

### 7. Vendored quantization in fp8.py
The bootstrap script has a self-contained quantize function. Could/should
be promoted to `fp8.py` as a proper `quantize_fp8_blockwise()` alongside
the existing `dequantize_fp8_blockwise()`. **Not done yet.**

---

## Key H100 performance numbers (from Piece 6)

| GPU | FP8 dense TFLOPS | Est. utilization | Effective TFLOPS | Per-step (ms) | Per-traj (s) |
|-----|-----------------|-----------------|-----------------|--------------|-------------|
| RTX 4090 (SM89) | 660 | 40% (measured) | 264 | 707 | 22.2 |
| H100 SXM (SM90) | 1,979 | 50-60% (est.) | ~1,088 | ~235 | ~7.5 |
| RTX PRO 6000 (SM100) | 504 | 45-55% (est.) | ~250 | ~625 | ~19.5 |

## Cost estimates

| Config | Time | Cost |
|--------|------|------|
| Local 4090 (gen only) | 14hr | $0 |
| 2xH100 (train only, pre-gen'd data) | 30min | ~$2.25 |
| 2xH100 (gen 500 + train) | 55min | ~$4.14 |
| 2xH100 (gen 2304 + train) | 2.85hr | ~$12.83 |
| 8xH100 (gen 2304 + train) | 1.1hr | ~$19.80 |

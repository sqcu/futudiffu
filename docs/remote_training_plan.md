# Remote Training Run: Planning Notes

Operational plan for running BTRM training + policy optimization on a rented
GPU node (spot 2xH100, or 8xH100, or 2xRTX PRO 6000 — whatever's cheap
tomorrow). Budget: $4-10 total. Duration: 1-4 hours.

---

## What we have right now

### Dataset
- 50 trajectories on disk (151MB), 8 sparse checkpoints each (~4.3MB/traj)
- 24 t2i prompt templates, 11 i2i source images, 10 step schedules, 2 attn backends
- Full generation plan produces ~2,304 trajectories (~10GB)
- `generate_btrm_dataset.py` is a working schedule-driven client with resume support
- Per-trajectory format: `traj_NNNNNN/{step_00.pt, step_04.pt, ..., final.pt, meta.json}`

### Training pipeline
- Server/client split over ZMQ, all GPU state on server
- 5 training RPCs: `inject_btrm_head`, `score_btrm`, `train_btrm_step`,
  `accumulate_policy_gradients`, `policy_optimizer_step`
- BTRM training verified on 50 trajectories: loss 1.43→1.27, scrimble 50%→83%
- Policy optimization verified: gradient flow through 120/204 LoRA params,
  6.07GB baseline VRAM, +0.20GB during backward

### Monitoring
- `progress.py` has unicode bar, throughput tracking, ETA, latency percentiles
- Wired into `generate_btrm_dataset.py` only — not into training loop
- Training scripts print per-step metrics to stdout but no structured logging

### Missing
- No continuous sync / upload infrastructure
- Policy rollouts use single hardcoded prompt
- No production training script (only smoke tests)
- No HuggingFace integration
- No safetensors packing for trajectory archives

---

## The session on the remote node

### Timeline sketch

```
t=0        ssh in, clone repo, install deps, download model weights
t=5min     start server, verify status RPC responds
t=7min     generate trajectories (if expanding beyond current 50)
           MEANWHILE: sync daemon starts uploading completed trajectories
t=1-2hr    trajectory generation complete (or enough for training)
t=~2hr     BTRM training (10-50 macrobatches depending on dataset size)
           adapters synced home every N steps
t=~2.5hr   policy optimization (10-50 iterations with diverse prompts)
           adapters synced home every N iterations
t=~3hr     final sync, verify all artifacts landed, release node
```

The key insight: **trajectory generation, training, and upload are three
concurrent streams**. The node is never idle — if training is waiting on
anything, trajectory generation or upload fills the gap.

### What runs concurrently

1. **Inference server** (`server.py`) — owns GPU, handles all RPCs
2. **Foreground task** — either `generate_btrm_dataset.py` or `train.py`
3. **Sync daemon** — background process watching output dirs, uploading
   new artifacts to home machine + HF as they appear

---

## Piece 1: Production training script (`train.py`)

Not a smoke test. A real training loop with:

### Prompt diversity

Encode all 24 prompts + negative prompt once at startup. Per policy iteration,
sample K prompts uniformly (or weighted — laser sharks could get 2x weight
since that's where we have the most BTRM signal). Each rollout in a group
uses the same prompt (so advantages are comparable within a group), but
different groups use different prompts.

```
for iteration in range(n_iterations):
    prompt_idx = rng.choice(24)
    pos_cond = cached_conds[prompt_idx]
    for k in range(group_size):
        rollout = sample_trajectory(pos_cond, neg_cond, seed=...)
        ...
```

The BTRM was trained on all 24 prompts, so scoring rollouts from any of them
is in-distribution. Policy gradients from diverse prompts prevent the LoRA
from overfitting to one composition style.

### Phased execution

```
Phase 0 (optional): Generate more trajectories if dataset is small
Phase 1: BTRM training from trajectory pool
Phase 2: Policy optimization with live rollouts
Phase 3: Final adapter dump + eval renders
```

Each phase is independently skippable (via CLI flags or config). A resumed
run can jump straight to phase 2 if BTRM is already trained.

### Checkpoint cadence

Adapters are ~3.5MB per LoRA + ~30KB BTRM head. Save every N iterations:
- BTRM phase: every 5 macrobatches (or every 30s, whichever is more frequent)
- Policy phase: every 3 iterations

Each checkpoint is a self-contained {rtheta, ptheta, btrm_head} bundle in
safetensors format, timestamped. The sync daemon picks these up and ships
them home.

### Metrics logging

Structured JSONL to a logfile, one line per training step:

```json
{"phase": "btrm", "step": 7, "loss": 1.31, "acc_scrimble": 0.75, "acc_scrongle": 0.80, "dt_s": 2.1, "vram_gb": 6.2}
{"phase": "policy", "iter": 3, "mean_reward": 0.042, "grad_norm": 1.2e-3, "prompt_idx": 14, "dt_s": 58.3}
```

Plus a human-readable summary line to stderr using the same `progress.py`
milestone format. Tail the logfile from another terminal for real-time
monitoring; parse the JSONL post-hoc for analysis.

### Config

A flat dataclass or dict, not YAML. Passed as CLI args or a JSON file:

```
--btrm-macrobatches 30
--btrm-batch-size 32
--btrm-lr 1e-3
--policy-iterations 50
--policy-group-size 4
--policy-rollout-steps 10
--policy-sparse-steps 5
--policy-lr 1e-4
--checkpoint-every 5
--prompt-weights uniform  (or "laser-heavy")
--dataset-dir /path/to/btrm_dataset
--output-dir /path/to/run_output
```

---

## Piece 2: Continuous sync

### Design: dumb rsync loop

A shell one-liner running in a `tmux` pane alongside the training:

```bash
while true; do
    rsync -avz --include='*.safetensors' --include='*.json' --include='*.jsonl' \
        /scratch/run_output/ user@home:~/futudiffu_runs/run_001/
    sleep 30
done
```

This handles adapters, metrics JSONL, and checkpoint manifests. No custom
code needed — rsync is idempotent and incremental.

### HuggingFace upload for trajectory data

Trajectories are larger (~4MB each) and accumulate over hours. These go to
a private HF dataset repo rather than the home machine:

```python
# Pseudocode for the upload daemon
from huggingface_hub import HfApi
api = HfApi()

while True:
    new_trajs = find_unuploaded_trajectories()
    for traj_dir in new_trajs:
        pack_to_safetensors(traj_dir)  # 8 checkpoints → 1 file
        api.upload_file(...)
        mark_uploaded(traj_dir)
    sleep(60)
```

Or even simpler: `huggingface-cli upload` in a cron-style loop. The point
is that trajectory data leaves the ephemeral node as fast as it's created.
If the node dies, the worst case is losing the last minute's worth of
trajectories — the rest are already on HF.

### What gets synced where

| Artifact | Size | Destination | Cadence |
|----------|------|-------------|---------|
| LoRA adapters (rtheta, ptheta) | ~7MB | home + HF | every N steps |
| BTRM head | ~30KB | home + HF | with adapters |
| Metrics JSONL | ~KB | home | every 30s |
| Trajectories (safetensors) | ~4MB each | HF | as completed |
| Eval renders (PNG) | ~2MB each | home | end of run |

---

## Piece 3: Trajectory packing format

### Current: directory of .pt files

```
traj_000042/
    step_00.pt    534KB
    step_04.pt    534KB
    ...
    final.pt      534KB
    meta.json     ~1KB
```

9 files per trajectory. 50 trajectories = 450 files. 2,304 trajectories =
~21,000 files. This is fine for local disk but annoying for HF (slow
uploads, slow listing).

### Target: safetensors archive + JSONL index

Pack each trajectory into a single safetensors file:

```python
from safetensors.torch import save_file

tensors = {
    "step_00": load("step_00.pt"),
    "step_04": load("step_04.pt"),
    ...
    "final": load("final.pt"),
}
metadata = {"seed": "12345", "prompt_idx": "7", "n_steps": "30", ...}
save_file(tensors, "traj_000042.safetensors", metadata=metadata)
```

Safetensors metadata is string→string only, so numeric values get stringified.
That's fine — the JSONL manifest carries typed metadata:

```
manifest.jsonl  (one line per trajectory)
traj_000000.safetensors
traj_000001.safetensors
...
```

Each JSONL line:

```json
{"traj_id": 0, "file": "traj_000000.safetensors", "type": "t2i", "seed": 12345, "prompt_idx": 7, "prompt": "...", "n_steps": 30, "precision": "sdpa", "checkpoints": ["step_00", "step_04", "step_09", "step_14", "step_19", "step_24", "step_29", "final"]}
```

This is the natural format for HuggingFace Datasets hosting. One safetensors
file per trajectory, one JSONL manifest for the whole dataset. `datasets`
library can load the JSONL directly; safetensors can be memory-mapped for
random access during training.

### Sharding for large datasets

If the dataset grows beyond ~1000 trajectories, shard into ~500MB chunks:

```
shard_000.tar    (traj_000000 through traj_000099)
shard_001.tar
...
manifest.jsonl   (complete index, references shard files)
```

But at our current scale (2,304 trajectories, ~10GB), individual files are
fine. Sharding is a future concern.

---

## Piece 4: Dataset variety (is it enough?)

### Current coverage

| Axis | Count | Notes |
|------|-------|-------|
| t2i prompts | 24 | Good spread: animals, text, scenes, textures, styles |
| i2i sources | 11 | Diverse media: pixel art, line art, photo, painting |
| Step counts | 10 | [4,6,8,10,12,14,16,18,20,22] + 30 gold |
| Seeds | 100 | Per-prompt |
| Attn backends | 2 | sdpa (gold) vs sage (INT8 QK) |
| i2i denoise | 3 | [0.75, 0.85, 0.95] |

### Is this enough for a $5 training run?

Yes. The bottleneck isn't dataset variety — it's compute time. With 2,304
trajectories we have:

- **~4,000+ scrongle pairs** (30-step vs reduced, matched by prompt)
- **~2,000+ scrimble pairs** (sdpa vs sage, matched by prompt+steps)
- **~400 i2i pairs** (cross-denoise, cross-backend)

That's more than enough for a BTRM head with 2 outputs and ~15K parameters
to converge. The 50-trajectory smoke test already showed learning (loss
1.43→1.27, scrimble accuracy 50%→83%) — 2,304 trajectories is 46x more data.

The real question is whether 24 prompts is enough for the *policy* to
generalize. For a proof-of-concept run proving the pipeline works end-to-end
and produces a measurably different adapter — yes. For a production adapter
that improves arbitrary user prompts — probably want 100-500 diverse prompts,
drawn from a real prompt distribution (ShareGPT, DiffusionDB, etc.).

That's a dataset v2 concern, not a "tomorrow's run" concern.

---

## Piece 5: What the trained adapters are for

Worth stating explicitly so the run has a clear success criterion.

### rtheta (BTRM backbone adapter)

Rank-8 LoRA on layers 28-29 of NextDiT. Trained by BTRM loss to make the
backbone's hidden states at those layers discriminative for:
- **scrimble** (head 0): "was this generated with full-precision attention?"
- **scrongle** (head 1): "was this generated with enough denoising steps?"

This adapter + the BTRM head together form a **learned quality metric** that
runs inside the diffusion model's forward pass. It's a cheap critic — no
separate model needed, just 2 extra layers of LoRA + a linear head.

### ptheta (policy adapter)

Rank-8 LoRA on all 30 layers. Trained by REINFORCE against the BTRM scores
to produce outputs that score higher on the quality metric. The policy
adapter's job is to compensate for the quality degradation introduced by
FP8 quantization and reduced step counts — making the quantized model's
outputs closer to what the full-precision, full-step model would produce.

### Success criteria for the run

1. BTRM loss converges (decreasing trend, not just noise)
2. Both heads show >70% accuracy on held-out pairs
3. Policy adapter produces measurably different outputs (cos < 0.99 vs baseline)
4. Policy adapter's outputs score higher on BTRM than baseline (even slightly)
5. All artifacts (adapters, head, metrics) synced to home machine and HF

Criteria 1-4 are verifiable from the metrics JSONL. Criterion 5 is
operational. A run that achieves 1-3 is a success even if 4 is marginal
(policy optimization with K=2-4 and 10-50 iterations is stochastic — a
consistent positive trend is the goal, not a guarantee).

---

## Priority order for tomorrow's session

1. **`train.py`** — production training script with diverse prompts, phased
   execution, checkpoint saves, metrics JSONL. This is the thing that runs
   on the node.

2. **Trajectory packer** — `pack_trajectories.py` that converts the current
   directory-of-pt-files format into safetensors + JSONL. Small script, maybe
   50 lines. Run once on the existing 50 trajectories to validate the format,
   then use it as the upload unit for the sync daemon.

3. **Sync setup** — could be as simple as documenting the rsync one-liner and
   the `huggingface-cli upload` invocation. No custom daemon code needed for
   a 3-hour run — a tmux pane with a while-sleep loop is fine.

4. **Expanded dataset generation** — run `generate_btrm_dataset.py` with a
   larger schedule. This is mostly "start it and wait" — the script and
   server already work. The question is whether to generate on the local
   4090 (slower, free) or on the rented node (faster, costs money).

5. **Post-run eval** — `generate_policy_eval.py` already exists for rendering
   comparison images. After training, generate a grid of before/after renders
   across diverse prompts. These go in `bench_renders/` as documentation.

---

## Piece 6: GPU performance analysis and timeline estimates

### Hardware comparison table

All FP8 TFLOPS figures are **dense** (no structured sparsity). Our workload does
not use 2:4 sparsity, so sparse TFLOPS are irrelevant.

| Spec | RTX 4090 (SM89, Ada) | H100 SXM (SM90, Hopper) | RTX PRO 6000 (SM100, Blackwell) |
|------|---------------------|------------------------|-------------------------------|
| FP8 tensor TFLOPS (dense) | 660 | 1,979 | 504 |
| FP8 tensor TFLOPS (sparse) | 1,321 | 3,958 | 1,008 |
| FP16/BF16 tensor TFLOPS | 330 | 989 | 252 |
| Memory | 24 GB GDDR6X | 80 GB HBM3 | 96 GB GDDR7 |
| Memory bandwidth | 1,008 GB/s | 3,350 GB/s | 1,792 GB/s |
| TDP | 450 W | 700 W | 600 W |
| Tensor cores | 512 (4th gen) | 528 (4th gen) | 752 (5th gen) |
| Architecture features | -- | TMA, wgmma, async copy | TMA, FP4/FP6, 2nd-gen FP8 TE |

Sources for these numbers:
- RTX 4090: NVIDIA Ada Lovelace datasheet
- H100 SXM: NVIDIA H100 datasheet (1,979 dense = 3,958 sparse / 2)
- RTX PRO 6000: WareDB / BIZON spec aggregation (503.80 dense confirmed)

### FP8 GEMM utilization analysis

On the RTX 4090 we measured **~40% utilization** of 660 TFLOPS peak for our FP8
GEMMs (250-295 effective TFLOPS). This was confirmed by cuBLAS benchmarks hitting
the same ceiling -- it is an inherent SM89 limitation at our GEMM shapes, not a
kernel deficiency.

**H100 utilization estimate: 50-60%** (educated estimate, not measured)

Rationale for higher utilization on H100:
- **TMA (Tensor Memory Accelerator)**: H100's hardware TMA unit handles address
  generation and data movement with a single thread, eliminating the register
  pressure and thread occupancy overhead that plagues small-tile loads on SM89.
  Our GEMM tiles (M~128-256, N~128-256, K~128 blockwise) benefit directly.
- **wgmma instruction**: Hopper's warp-group MMA instruction operates on larger
  fragments than SM89's mma.sync, reducing instruction issue overhead per FLOP.
- **HBM3 bandwidth (3,350 GB/s)**: For shapes where our GEMMs are partially
  bandwidth-bound (e.g., the smaller adaLN projections, LoRA matmuls), the 3.3x
  bandwidth increase removes the bottleneck entirely. The large GEMMs (3840x3840,
  3840x10240) are compute-bound on both architectures, so bandwidth helps less
  there but TMA/wgmma help more.
- **Larger L2 cache (50 MB vs 72 MB)**: Better tile reuse for multi-head attention
  GEMMs where K is reused across heads.
- Published benchmarks show Triton FP8 kernels on H100 achieving 70-80% of peak
  for large GEMMs. Our shapes are smaller, so 50-60% is a conservative estimate.

Estimated effective TFLOPS: 1,979 * 0.55 = **~1,088 TFLOPS** (vs 264 on 4090).
That is a **~4.1x effective speedup** on compute-bound FP8 GEMMs.

**RTX PRO 6000 utilization estimate: 45-55%** (speculative, no SM100 benchmarks yet)

The RTX PRO 6000 is an unusual case. Its FP8 dense peak (504 TFLOPS) is actually
**lower** than the RTX 4090 (660 TFLOPS). This is because Blackwell's tensor
core improvements focus on FP4 and structured sparsity, not raw FP8 dense
throughput. However, the 5th-gen tensor cores and GDDR7 bandwidth (1,792 GB/s,
1.78x over 4090) may improve utilization percentage. Estimated effective: ~250
TFLOPS -- roughly comparable to the 4090, not faster.

The RTX PRO 6000's real advantages for us would be:
- 96 GB VRAM (4x the 4090) -- can hold model + full trajectory in memory
- 1,792 GB/s bandwidth -- helps bandwidth-bound ops (attention, elementwise)
- FP4 tensor cores -- irrelevant for us today, but future quantization path

### Per-operation H100 speedup estimates

| Operation | 4090 time | Bottleneck | BW ratio | Compute ratio | H100 estimate | Speedup |
|-----------|-----------|-----------|----------|---------------|---------------|---------|
| FP8 GEMM (large, 3840x10240) | ~dominant | compute | 3.3x | 4.1x | /4.1 | 4.1x |
| FP8 GEMM (small, adaLN 3840x32) | ~small | bandwidth | 3.3x | 4.1x | /3.0 | 3.0x |
| Attention SDPA (N=4288, D=128) | 6.54ms | bandwidth | 3.3x | -- | ~2.2ms | 3.0x |
| SageAttention INT8 (same shape) | 4.84ms | bandwidth | 3.3x | -- | ~1.7ms | 2.8x |
| RMSNorm + modulate (fused Triton) | ~small | bandwidth | 3.3x | -- | /3.0 | 3.0x |
| RoPE (fused with QKV postprocess) | ~small | bandwidth | 3.3x | -- | /3.0 | 3.0x |
| Python/dispatch overhead | ~fixed | CPU | 1x | 1x | ~same | 1.0x |
| torch.compile graph replay | ~low | launch | 1x | 1x | ~same | 1.0x |

The overall forward pass is dominated by FP8 GEMMs (~70% of time) with attention
as the second contributor (~15%). The remaining ~15% is elementwise/overhead.

### Forward pass time estimates

**RTX 4090 baseline** (measured): **707 ms** per step (B=2, FP8, all fusions, compiled)

**H100 SXM estimate**:
- FP8 GEMMs (70% = 495ms): 495 / 4.1 = 121ms
- Attention (15% = 106ms): 106 / 3.0 = 35ms
- Elementwise + overhead (15% = 106ms): 106 / 2.0 = 53ms (partial speedup, some is fixed)
- **Total: ~209ms per step** (3.4x overall speedup)
- Conservative estimate accounting for Amdahl's law on fixed overhead: **~235ms**
- Optimistic estimate if TMA eliminates more dispatch overhead: **~200ms**

**RTX PRO 6000 estimate**:
- FP8 GEMMs (70% = 495ms): 495 / 1.0 = 495ms (comparable to 4090)
- Attention (15% = 106ms): 106 / 1.8 = 59ms
- Elementwise + overhead (15% = 106ms): 106 / 1.5 = 71ms
- **Total: ~625ms per step** (1.13x overall speedup)
- The RTX PRO 6000 is **not meaningfully faster** than the 4090 for our workload.
  Its FP8 dense throughput is lower and bandwidth is only 1.78x higher.

### Per-trajectory time estimates

30 euler steps per trajectory. Add text encoder + VAE overhead (~1s on 4090).

| GPU | Per-step (ms) | 30 steps (s) | TE+VAE (s) | Per-trajectory (s) |
|-----|--------------|-------------|-----------|-------------------|
| RTX 4090 | 707 | 21.2 | 1.0 | **~22.2** |
| H100 SXM | 235 | 7.1 | 0.4 | **~7.5** |
| RTX PRO 6000 | 625 | 18.8 | 0.7 | **~19.5** |

### What our existing optimizations buy us on H100

Every architectural optimization we built for the 4090 becomes **more** impactful
on H100, because the faster the raw hardware, the larger the fraction of total
time consumed by overhead -- and our optimizations specifically target overhead.

**Kernel fusion (fused_kernels.py, fp8_kernels.py)**:
On H100, individual Triton kernels run 3-4x faster, but kernel launch overhead
stays roughly constant (~5-10us per launch). A chain of 6 unfused kernels that
takes 60us of launch overhead on 4090 still takes 60us on H100, but the compute
portion drops from 500us to 125us -- so launch overhead rises from 11% to 32% of
total time. Fusing those 6 kernels into 2 eliminates 40us of launch overhead,
which on H100 is a 21% savings vs 7% on 4090. The fusion ROI is **3x higher** on
faster hardware.

**FP8 FFN chain (fp8_silu_gate_quant -> fp8_gemm_blockwise)**:
Skipping the BF16 intermediate materialization between SiLU-gate and w2 GEMM
saves a round-trip to memory. On 4090 at 1,008 GB/s, writing+reading a
(B, seq, 10240) BF16 intermediate costs ~0.4ms. On H100 at 3,350 GB/s, the
same I/O costs ~0.12ms -- but there are 30 layers, so the cumulative savings
are 0.12 * 30 = 3.6ms on H100 vs 12ms on 4090. The absolute savings are smaller
but the fractional savings relative to the now-faster forward pass are similar.
More importantly: on H100 the GEMMs finish faster but the bandwidth-bound
materialization does NOT speed up as much, so without fusion the FFN chain becomes
more bandwidth-bottlenecked. The fusion prevents this regression.

**Zero graph breaks + torch.compile**:
With zero graph breaks, torch.compile captures the entire forward pass as a
single CUDA graph. Graph replay eliminates all Python dispatch overhead and
CUDA API calls during the 30-step loop. On 4090 this saved ~11ms/step (707ms
compiled vs 718ms uncompiled). On H100, the same dispatch overhead is unchanged
but the compute portion shrinks by 3.4x, so dispatch would be a larger fraction
of the uncompiled cost. The compiled savings could be ~15-20ms/step on H100 --
the difference between 235ms and 255ms, which is a 8% savings vs 1.5% on 4090.

**CFG batching (B=2)**:
Runs positive and negative conditioning in a single forward pass. On H100,
this avoids a second kernel launch sequence (whose fixed overhead is proportionally
more expensive). The batch dimension also helps H100's wgmma instruction achieve
higher utilization since it operates on larger fragments.

**RoPE caching + TE early exit**:
These avoid redundant compute. The savings are small in absolute terms but
compound across 30 steps and become proportionally larger as other ops get faster.

### Updated timeline: 8xH100 SXM

```
t=0         ssh in, clone, install deps, download weights       (~5 min)
            5.8GB FP8 diff + 7.5GB BF16 TE + 0.16GB VAE = 13.5GB
            at 1 Gbps inter-DC: ~2 min; from object storage: <30s

t=5min      start 8 inference servers (1 per GPU), verify RPCs  (~2 min)
            torch.compile warmup: ~45s per server (parallel)

t=7min      generate 2,304 trajectories across 8 GPUs           (~36 min)
            8 GPUs * (1 traj / 7.5s) = 1.07 traj/s
            2,304 / 1.07 = ~2,153s = ~36 min
            sync daemon uploading completed trajectories throughout

t=43min     BTRM training on primary GPU                        (~2 min)
            216 examples (from 2,304 trajs), 7 macrobatches
            Each macrobatch: ~4s on H100 (vs ~20s on 4090)

t=45min     policy optimization, 50 iterations                  (~17 min)
            per iteration: 4 rollouts * 10 steps * 235ms = ~9.4s rollout
            + checkpointed fwd/bwd ~5s + overhead ~2s = ~17s/iter
            50 * 17s = 850s = ~14 min
            add 3 min margin for recompilations and prompt encoding

t=62min     final sync + eval renders (10 diverse prompts)      (~3 min)

t=65min     done. ~1.1 hours total.
```

**Cost estimate (8xH100 SXM)**:
- Spot price: ~$2.00-2.50/GPU-hr (Hyperstack, Vast.ai, Cudo range as of Feb 2026)
- 8 GPUs * 1.1 hours * $2.25/GPU-hr = **~$19.80**
- This is above the original $4-10 budget. See 2xH100 below.

### Updated timeline: 2xH100 SXM

```
t=0         ssh in, clone, install deps, download weights       (~5 min)

t=5min      start 2 inference servers, verify RPCs              (~2 min)

t=7min      generate 2,304 trajectories across 2 GPUs           (~2.4 hr)
            2 GPUs * (1 traj / 7.5s) = 0.27 traj/s
            2,304 / 0.27 = ~8,533s = ~142 min
            sync daemon running throughout

t=149min    BTRM training on primary GPU                        (~2 min)

t=151min    policy optimization, 50 iterations                  (~17 min)
            (same as 8xH100, only uses 1 GPU)

t=168min    final sync + eval renders                           (~3 min)

t=171min    done. ~2.85 hours total.
```

**Cost estimate (2xH100 SXM)**:
- 2 GPUs * 2.85 hours * $2.25/GPU-hr = **~$12.83**
- Still over budget. But trajectory generation dominates -- with the existing
  50 trajectories as seed data, we could generate only 500 new trajectories
  and still have a meaningful training run:
  - 500 trajs / 0.27 = ~31 min generation
  - Total run: ~55 min
  - Cost: 2 * 0.92 * $2.25 = **~$4.14** -- within budget

### Updated timeline: 2xRTX PRO 6000 Blackwell

The RTX PRO 6000 is a surprising disappointment for our specific workload.

Key facts:
- FP8 dense: 504 TFLOPS (lower than 4090's 660 TFLOPS)
- Bandwidth: 1,792 GB/s (1.78x 4090)
- VRAM: 96 GB (massive, but we only need ~6 GB)
- Per-step estimate: ~625ms (vs 707ms on 4090, vs 235ms on H100)
- Per-trajectory: ~19.5s (barely faster than 4090's 22.2s)

The RTX PRO 6000's strengths (huge VRAM, FP4 support, high bandwidth) do not
align with our workload's bottleneck (FP8 compute throughput on medium-sized
GEMMs). The 5th-gen tensor cores are optimized for FP4 and sparse workloads,
not dense FP8. At ~$1.80/GPU-hr (Hyperstack), it is cheaper than H100 per
GPU-hour, but the per-trajectory cost is worse because it is so much slower.

```
t=0         ssh in, clone, install deps, download weights       (~5 min)

t=5min      start 2 inference servers, verify RPCs              (~2 min)

t=7min      generate 2,304 trajectories across 2 GPUs           (~6.3 hr)
            2 GPUs * (1 traj / 19.5s) = 0.103 traj/s
            2,304 / 0.103 = ~22,369s = ~373 min

t=380min    BTRM training                                       (~3 min)

t=383min    policy optimization, 50 iterations                  (~30 min)
            per iteration: 4 rollouts * 10 steps * 625ms = 25s rollout
            + fwd/bwd ~10s + overhead ~3s = ~38s/iter
            50 * 38s = 1,900s = ~32 min

t=415min    final sync + eval renders                           (~3 min)

t=418min    done. ~7 hours total.
```

**Cost estimate (2xRTX PRO 6000)**:
- 2 GPUs * 7 hours * $1.80/GPU-hr = **~$25.20**
- Worst value of all three options despite lowest per-GPU-hour price.
- The 4090 at home (free) is a better choice than renting RTX PRO 6000s.

### Cost-efficiency comparison

| Configuration | Total time | Cost | Cost per 1K trajectories |
|---------------|-----------|------|--------------------------|
| 1x RTX 4090 (local, free) | ~14.2 hr | $0 | $0 |
| 2x RTX PRO 6000 (spot) | ~7.0 hr | ~$25.20 | ~$10.94 |
| 2x H100 SXM (spot) | ~2.85 hr | ~$12.83 | ~$5.57 |
| 8x H100 SXM (spot) | ~1.1 hr | ~$19.80 | ~$8.60 |
| 2x H100 SXM (500 trajs) | ~0.9 hr | ~$4.14 | ~$8.28 |

The clear winner is **2xH100 SXM with a reduced trajectory count** (500-800
trajectories instead of 2,304). This fits the $4-10 budget while still
providing 10-16x more training data than the current 50-trajectory smoke test.

Alternatively, generating trajectories on the local 4090 overnight (free, ~14
hours for 2,304) and then renting 2xH100 for just the training phases (~20 min,
cost: ~$1.50) gives the best of both worlds.

### Recommendation

**Preferred plan: hybrid local + cloud**

1. Generate 2,304 trajectories overnight on local 4090 (free, ~14 hours)
2. Pack trajectories into safetensors + upload to HF (~30 min)
3. Rent 2xH100 SXM spot for ~30 minutes:
   - Download trajectories from HF (~2 min at DC bandwidth)
   - BTRM training (~2 min)
   - Policy optimization with 50 iterations (~17 min)
   - Sync artifacts home + eval renders (~3 min)
4. Total cloud cost: 2 * 0.5hr * $2.25 = **~$2.25**

This hits the budget constraint, uses all 2,304 trajectories, and avoids paying
cloud rates for the embarrassingly-parallel but time-consuming trajectory
generation phase.

**Fallback plan: all-cloud with 2xH100**

If overnight local generation is not practical (machine needed for other things),
rent 2xH100 SXM and generate 600-800 trajectories on the node before training.
Total time ~1.5 hours, cost ~$6.75. Still within budget, still enough data for a
meaningful training run.

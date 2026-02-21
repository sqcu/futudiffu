# futudiffu Architectural Review

A comprehensive analysis of the futudiffu codebase: a standalone Z-Image diffusion
pipeline that evolved from a bug-for-bug ComfyUI port into an inference server that
doubles as a policy environment for reinforcement learning.

---

## 1. The Server/Client Split as a Design Philosophy

The central architectural decision in futudiffu is the separation of GPU-owning
inference from all other concerns. This split is not just a deployment convenience --
it shapes every other design choice in the system and resolves an entire class of bugs
that plagued the earlier monolithic scripts.

### What lives on the server

The inference server (`/mnt/f/dox/repos/ai/futudiffu/src/futudiffu/server.py`,
1371 lines) owns all GPU state and all forward/backward computation. Specifically:

**Model weights and compiled graphs.** The `InferenceServer` class holds three model
slots (`_te_model`, `_diff_model`, `_vae_model`) plus two compiled entry points
(`_diff_compiled` for `forward()` and `_diff_compiled_packed` for
`forward_packed()`). The text encoder is compiled with `mode="default"` (line 124);
the diffusion model is compiled after LoRA injection and fusion (lines 215-220). All
`torch.compile` state, all Triton kernel caches, all CUDA graph replays live here.

**LoRA adapter state.** The server manages the full lifecycle of LoRA adapters through
three parallel data structures (lines 68-70):

```python
self._lora_configs: list[dict] = []        # Injection configs to replay on reload
self._lora_weights: dict[str, dict] = {}   # CPU weight snapshots per adapter
self._lora_scales: dict[str, float] = {}   # Last-set scale per adapter
```

This triple-bookkeeping enables LoRA persistence across model lifecycle swaps -- when
the server frees the diffusion model to load the text encoder, it snapshots LoRA
weights to CPU (line 76-85), and when the diffusion model reloads, it replays all
injections and restores weights from the CPU cache (lines 184-211). The compiled graph
is rebuilt after replay so the tracer sees the LoRA modules from the start.

**VRAM lifecycle management.** The server implements a phase system where the text
encoder (~7.5GB) and diffusion model (~8GB with compiled graphs) are mutually exclusive,
while the VAE (~320MB) can coexist with either. The `_ensure_te()` and
`_ensure_diffusion()` methods handle the swap: freeing the opposing model, loading
the requested one, and restoring any LoRA state that needs to survive the transition.
This is the critical constraint on a 24GB RTX 4090 -- there is not enough VRAM for
both the TE and the diffusion model simultaneously with room for activations.

**All forward and backward computation.** Every neural network evaluation runs on the
server, including:
- Text encoding (`handle_encode_prompt`, line 240)
- Diffusion sampling with full euler loops (`handle_sample_trajectory`, line 263)
- FlexAttention-packed multi-image sampling (`handle_sample_trajectory_packed`, line 418)
- VAE encode and decode (`handle_vae_encode`, line 607; `handle_vae_decode`, line 624)
- BTRM head management and scoring (`handle_inject_btrm_head`, `handle_score_btrm`, `handle_train_btrm_step`)
- Policy gradient accumulation and stepping (`handle_accumulate_policy_gradients`, `handle_policy_optimizer_step`)

The training RPCs keep all heavy state on the server. The BTRM head (~30KB) lives
alongside LoRA as a permanent GPU resident. `train_btrm_step` is a single atomic
RPC: the client sends labeled examples and gets back scalar metrics -- all
forward/backward/step happens server-side. `accumulate_policy_gradients` runs the
same checkpointed forward + log-ratio loss + backward computation but retains
gradients on the server instead of streaming them. The client calls it K times
(once per rollout), then calls `policy_optimizer_step` once to clip and step.

### What lives on the client

The client (`/mnt/f/dox/repos/ai/futudiffu/src/futudiffu/client.py`, 426 lines)
is deliberately thin. It holds:

**Scheduling logic.** The dataset generator (`generate_btrm_dataset.py`) decides
which prompts to encode, which seeds to use, which step counts and attention backends
to vary. It owns the RNG (a `random.Random` instance with a fixed seed for
reproducibility across resume), the schedule format, and the trajectory metadata.

**Pure scheduling logic.** The client is a pure scheduling process with no GPU state.
The BTRM head lives on the server (created via `inject_btrm_head`, scored via
`score_btrm`, trained via `train_btrm_step`). Policy optimizer state also lives on
the server (gradients accumulate via `accumulate_policy_gradients`, stepping via
`policy_optimizer_step`). The client decides WHAT to train and in what order, but
all neural network computation and optimizer state management happens server-side.

**Disk I/O and metadata.** Trajectory latents (each `(1, 16, 104, 160)` bf16,
~522KB) are saved to disk by the client. The manifest, per-trajectory `meta.json`,
and rendered PNGs are all client-side concerns.

### The coupling surface

The wire protocol (`/mnt/f/dox/repos/ai/futudiffu/src/futudiffu/protocol.py`, 168
lines) is JSON envelope + raw tensor bytes over ZeroMQ multipart frames. No pickle
anywhere. The JSON envelope carries the RPC method name, scalar parameters, and tensor
descriptors (name, shape, dtype). Tensors are serialized as contiguous byte buffers;
bfloat16 tensors are viewed as uint16 for byte-level transfer since numpy lacks a
bf16 dtype.

The server exposes 19 RPC methods:

| Category | Methods |
|----------|---------|
| Inference | `encode_prompt`, `sample_trajectory`, `sample_trajectory_packed`, `vae_encode`, `vae_decode` |
| Lifecycle | `warmup`, `warmup_packed`, `status`, `free` |
| LoRA | `inject_lora`, `update_lora_weights`, `set_adapter_config`, `get_lora_state_dict`, `dump_all_loras` |
| Training | `inject_btrm_head`, `score_btrm`, `train_btrm_step`, `accumulate_policy_gradients`, `policy_optimizer_step` |

What does NOT cross the wire: model weights (except LoRA deltas), compiled graph
state, CUDA context, attention backend configuration (set by method parameter, not
global state transfer), RoPE caches (recomputed server-side per request), optimizer
state, BTRM head weights, hidden state tensors, or gradient tensors. Only scalar
metrics (loss values, accuracy, grad norms) cross the wire for training RPCs.

### Comparison to vLLM

vLLM provides an inference server that clients query for text generation. futudiffu's
server goes beyond this in several ways:

**Training RPCs.** vLLM has no equivalent of `accumulate_policy_gradients` -- it is
inference-only. futudiffu's server runs backward passes through gradient-checkpointed
transformer layers, computes per-step REINFORCE log-ratio losses, and accumulates LoRA
gradients server-side. It also hosts the BTRM reward head and its optimizer. This
makes it a training server masquerading as an inference server.

**LoRA hot-swapping.** vLLM supports LoRA adapters but treats them as static
configurations. futudiffu supports in-place weight updates during a session
(`update_lora_weights` with `.data.copy_()`), per-batch per-adapter scale routing
(the `lora_scale` registered buffer), and multiple named adapters stacked on the same
linear layer. The server can run batch[0] with the policy LoRA active and batch[1]
with it disabled, in a single forward pass.

**Lifecycle persistence.** When the server swaps between TE and diffusion phases, it
snapshots LoRA weights to CPU and replays them on reload. This is not a feature vLLM
needs (it does not swap between different model types mid-session), but it is essential
for the BTRM training workflow where the client encodes prompts (requiring TE), then
generates rollouts (requiring diffusion + LoRA), then re-encodes (requiring TE again).

### The RNG discipline benefit

The `docs/btrm_ops_notes.md` file documents a subtle but important consequence of the
split: "The server/client split eliminates this class of bug." The bug in question was
RNG desynchronization during resume. When a preview pass and a generation pass must
consume identical RNG draws to stay synchronized, any stray `rng.randrange()` in one
path desynchronizes all downstream seeds. By giving the client exclusive ownership of
all RNG state and making the server stateless with respect to randomness (it receives
seeds as parameters), the entire class of RNG-desync bugs is eliminated by
construction.

---

## 2. The "Policy Environment" Concept

The project's animating idea is that there is no vLLM for diffusion models -- and what
if an inference server could also serve as a policy environment for reinforcement
learning? The server/client split is not just an engineering convenience but the
realization of this concept.

### The server as an environment

In RL terminology, the server is the environment and the client is the agent. The
environment accepts actions and produces observations:

**Actions the client can take:**
- `inject_lora`: Add a new adapter (structural change to the policy)
- `update_lora_weights`: Modify the policy's parameters (weight update)
- `set_adapter_config`: Change per-batch routing (which adapters are active)
- `sample_trajectory` with specific seed, steps, attention backend: Request a rollout

**Observations the server returns:**
- Trajectory latents at configurable checkpoints (`step_NN` tensors)
- BTRM scores as scalar metadata (`score_btrm`)
- Training metrics: loss, accuracy, grad norms (`train_btrm_step`, `accumulate_policy_gradients`, `policy_optimizer_step`)
- The final denoised latent

The server maintains state between calls (the loaded model with its LoRA adapters,
compiled graphs, attention backend configuration) just as an RL environment maintains
state between steps. The `warmup` RPC is the environment's `reset` -- it triggers
compilation so subsequent calls are fast.

### accumulate_policy_gradients + policy_optimizer_step: the training RPCs

The `handle_accumulate_policy_gradients` method implements the REINFORCE gradient
computation on the server side. Gradients stay on the server (accumulated in `.grad`)
rather than being streamed to the client:

1. **Enable gradients on target LoRA params.** The method takes an `adapter_name`
   parameter and calls `requires_grad_(True)` on exactly those parameters. Existing
   gradients are NOT zeroed -- they accumulate across calls (one per rollout).

2. **Set up concurrent pi+ref batching.** It calls
   `set_lora_scale(model, tensor([1.0, 0.0]), adapter_name=adapter_name)` so that
   batch element 0 runs with the policy LoRA active and batch element 1 runs with it
   disabled.

3. **Per-step forward with gradient checkpointing.** For each sparse step index,
   `forward_checkpointed()` runs the embedding and refiner layers in `no_grad`, then
   gradient-checkpoints each of the 30 main transformer layers individually.

4. **Log-ratio loss computation.** The Gaussian log-probability ratio is computed at
   each sparse step, with `ref_output` detached. Each step's loss is backpropagated
   immediately to free activations and bound VRAM usage.

5. **Gradients stay on server.** No tensor output. The method returns only
   `{total_log_ratio, n_steps}` as scalar metadata.

The client calls `accumulate_policy_gradients` K times (once per rollout with its
advantage weight), then calls `policy_optimizer_step` once. The step RPC clips
gradients, runs the optimizer (lazy-initialized from LoRA params on first call),
and zeros gradients for the next accumulation cycle. This two-phase pattern
eliminates all gradient streaming over the wire.

### The BTRM as reward signal

The Bradley-Terry Reward Model provides the reward signal for policy optimization.
The BTRM head (`btrm.py`) is a ~30KB projection that lives on the server alongside
LoRA:

```
hidden_states (B, N_tokens, 3840) -> mean_pool -> RMSNorm -> Linear(3840, N_heads) -> tanh_cap(10.0)
```

The server hosts the head (`inject_btrm_head`), scores with it (`score_btrm`), and
trains it (`train_btrm_step` -- a single atomic RPC that does backbone forward +
scoring + BT loss + backward + optimizer step, returning only scalar metrics).

Two named heads are defined: `bit_quality` (attention quantization discrimination)
and `step_quality` (step count discrimination). The naming convention from the
dataset is:
- **Scrimble pairs**: same prompt, same seed, same step count, different attention
  backend (SDPA vs SageAttention). The BTRM should score SDPA higher.
- **Scrongle pairs**: same prompt, same seed, same attention backend, different step
  count. The BTRM should score 30-step higher than reduced-step.

Training uses Bradley-Terry ranking loss: `-log_sigmoid(pos_score - neg_score)` plus
a logsquare regularizer `log(score^2 + eps)` that anchors scores toward `|r| ~ 1`.

### sample_trajectory with save_steps as environment rollouts

The `save_steps` parameter on `sample_trajectory` (server.py, line 307) controls
which intermediate euler step latents are returned alongside the final output. When
set to `None`, all 30 steps are saved. When set to a sparse list like
`[0, 4, 9, 14, 19, 24, 29]`, only those checkpoints are saved. This is the
mechanism that makes the server a rollout environment: the client can request a
trajectory and get back not just the endpoint but the intermediate states needed for
REINFORCE attribution.

The `sample_euler` callback mechanism (sampling.py, lines 98-99) fires after each
step, and the server's `save_callback` (server.py, lines 402-405) captures `x` at
the requested indices. These intermediate latents serve dual purposes: BTRM scoring
(the reward model evaluates hidden states at specific sigma values) and
`compute_policy_gradients` (which re-runs the model from these checkpoints with
gradients enabled).

### The backbone-head bridge (internal)

The server's `_run_backbone_hidden()` private method installs a forward hook on
`model.layers[-1]` (the last transformer block), runs a forward pass through
the raw (uncompiled) model, captures the hook output, removes the hook, and returns
the hidden states tensor. This is used internally by `score_btrm` and
`train_btrm_step` -- hidden states never leave the server. The BTRM head runs
on the same GPU, fed directly from the hook capture, eliminating the ~30MB per
sample transfer that the old `forward_hidden` RPC required.

---

## 3. What Was Actually Accomplished

### The BTRM dataset

The `btrm_dataset/` directory contains 50 trajectories with intermediate checkpoints,
organized into six schedule batches according to the manifest:

| Batch | Type | Attention | Step Count | Count | Trajectory IDs |
|-------|------|-----------|------------|-------|----------------|
| 0     | t2i  | SDPA      | 30         | 10    | 000000-000009  |
| 1     | t2i  | Sage      | 30         | 10    | 000010-000019  |
| 2     | t2i  | SDPA      | 10-22      | 10    | 000020-000029  |
| 3     | t2i  | Sage      | 8-21       | 10    | 000030-000039  |
| 4     | i2i  | SDPA      | 30         | 5     | 000040-000044  |
| 5     | i2i  | Sage      | 30         | 5     | 000045-000049  |

Each trajectory directory contains 7 intermediate latent checkpoints (`step_00.pt`
through `step_29.pt` at indices 0, 4, 9, 14, 19, 24, 29), a `final.pt`, and a
`meta.json` with full provenance (seed, prompt, step count, precision, batch index).
Each `.pt` file is a `(1, 16, 104, 160)` bf16 tensor (~522KB), representing the
latent state at 1280x832 resolution.

The dataset covers 13 unique prompt indices from the 24-template library, spanning
laser shark variants, text rendering challenges, complex scenes, and artistic styles.
The i2i trajectories use off-policy reference images from `i2i_off_policies/` at
native resolution with center-crop alignment.

This is small -- 50 trajectories from a planned 2304 -- but sufficient for the
smoke-testing purpose it serves. The `docs/e2e_training_test_plan.md` document
identifies 4 scrimble pairs and 8 scrongle pairs constructible from this data.

### The policy training that happened

The `smoke_test_e2e_training.py` script (402 lines) demonstrates the full server-based
training pipeline:

**Phase 1: BTRM head training (10 steps).** The client injects `rtheta` LoRA on
layers 28-29, generates paired noisy latents at varying sigma values, and sends them
to the server via `train_btrm_step`. The server runs backbone + BTRM head + BT loss +
backward + optimizer step atomically, returning only scalar metrics.

**Phase 2: Policy optimization (10 iterations).** The client freezes rtheta (scale=0),
injects `ptheta` on all layers, and runs the full REINFORCE loop: K=2 rollouts per
iteration via `sample_trajectory`, BTRM scoring via `score_btrm`, advantage
computation, gradient accumulation via `accumulate_policy_gradients`, and optimizer
step via `policy_optimizer_step`. All computation happens server-side.

The `bench_renders/` directory contains the visual outputs from different
configurations, including a `ptheta_lora.safetensors` file -- the saved policy
weights from a training run.

### The bench_renders: what was observed

The `bench_renders/` directory contains renders from multiple configurations:

- `sdpa.png`: Reference image using PyTorch SDPA (full precision BF16 attention)
- `fp8_qk_p_bf16_pv.png`: SageAttention with FP8 QK and BF16 PV matmuls
- `int8_qk_p_bf16_pv.png`: SageAttention with INT8 QK and BF16 PV
- `fp8_qk_p_fp8_pv.png`: SageAttention with FP8 for both QK and PV
- `policy_off.png`: Inference with ptheta LoRA injected but scale=0 (baseline)
- `policy_on.png`: Inference with ptheta LoRA active (trained policy)
- Multiple diff images (`diff_sdpa_vs_*.png`, `diff_*_vs_policy_on.png`)
- False-color diff visualizations at 5x, 10x, and 20x gain

The diff analysis, preserved in MEMORY.md, reveals:

**Quantization variants** (fp8+bf16, int8+bf16, fp8+fp8) preserve the same image
composition as the SDPA reference, with increasing noise floors. Mean pixel
differences range from 18 to 29 against the SDPA reference. The images show the same
laser shark, the same text, the same composition -- just with progressively more
grain.

**The policy render was structurally different** from ALL reference renders, with a
mean pixel difference of 35 against SDPA. Different composition, different text
placement, different contours. This is not noise on top of the reference -- it is a
genuinely different image from a different attractor basin.

**The policy render had LESS background grain** than the heavily-quantized (fp8+fp8)
variant. This suggests the policy learned something about attention quality that
generalizes beyond matching the reference. Rather than steering the quantized model
toward the SDPA reference output, the LoRA found a different solution with cleaner
features.

### The LoRA save/load roundtrip

The `lora_roundtrip_test/` directory contains two artifacts:
- `reference_outputs.pt`: Model outputs captured before save/load
- `rtheta_trained.safetensors`: Trained rtheta adapter weights

This verifies that the `lora_state_dict()` -> `save_file()` -> `load_file()` ->
`load_lora_state_dict()` pipeline preserves weights correctly. The save/load
infrastructure in `lora.py` (lines 381-429) uses safetensors format with keys like
`path.adapters.name.lora_A`, enabling clean round-tripping of named adapters.

### The crash dump mechanism

The `handle_dump_all_loras` RPC (server.py, lines 1204-1263) provides emergency state
preservation. It enumerates all adapter names on the diffusion model, saves each to a
separate safetensors file with a timestamp, and writes a JSON manifest. The docstring
explicitly says "Designed for crash recovery -- call this when the client is dying."
This addresses the operational reality of long training runs on a single GPU: if the
process is about to die, at least the current LoRA state can be preserved.

---

## 4. What the Renders Tell Us

The visual evidence in `bench_renders/` is the most concrete artifact of the project's
training experiments, and it tells a more interesting story than simple
convergence/divergence metrics would suggest.

### The quantization gradient is smooth

The progression from SDPA -> INT8+BF16 -> FP8+BF16 -> FP8+FP8 shows a smooth
degradation in visual quality. INT8 QK quantization (7-bit mantissa equivalent) is
nearly imperceptible; FP8 QK (3-bit mantissa) introduces subtle grain; FP8 PV adds
further noise. But in all cases, the image COMPOSITION is preserved. The laser shark
is the same laser shark, positioned the same way, with the same text. Quantization
affects the texture, not the structure.

This makes sense physically: attention quantization introduces noise in the softmax
weights, which manifests as per-pixel perturbation to the output. The noise is
approximately i.i.d. across spatial positions, so it affects local texture (grain,
sharpness) rather than global composition (layout, object identity).

### The policy found a different basin

The policy-on render is the outlier. Where quantization variants show the same
composition with added noise, the policy render shows a DIFFERENT composition with
LESS noise. The mean pixel difference of 35 (vs 18-29 for quantization variants)
combined with lower background grain suggests the LoRA did not learn to denoise toward
the SDPA reference -- it learned to steer the diffusion process into a different
attractor basin that the BTRM scores favorably.

This is consistent with the training objective. The BTRM head was trained on
Bradley-Terry preferences between SDPA and Sage hidden states. It learned to score
"SDPA-like" hidden states higher. But the LoRA, optimizing to maximize that score, is
not constrained to produce SDPA-matching outputs -- it is free to find any set of
hidden states that the BTRM scores highly. The result is a policy that produces clean
images (low grain, like SDPA) but with different compositions (because it found a
different maximum of the reward landscape).

This is a classic reward hacking signature, and the `docs/btrm_rl_architecture_review.md`
document explicitly identifies the missing KL penalty as the highest-priority gap:
"The policy loss `loss = -reward` is unbounded. Add a KL penalty." Without an anchor
to the reference distribution, the policy is free to drift into regions of output
space that score well but look nothing like what the base model would produce.

### Implications for the architecture

The renders validate that the server/client training pipeline works -- gradients flow
through the server's checkpointed backbone, the client's optimizer modifies LoRA
weights, the server applies them, and subsequent rollouts produce visibly different
outputs. The BTRM head successfully learns to distinguish SDPA from Sage hidden
states. And the policy optimization successfully modifies the model's behavior.

What the renders also show is that the reward model alone is insufficient to guide
the policy toward the desired behavior. The DRGRPO loss formulation in `policy_loss.py`
(lines 158-225) includes all the components needed for constrained optimization --
group advantages, clipped surrogate objectives, entropy bonuses, reference anchoring
-- but the smoke test used only `loss = -reward_bq`, the simplest possible objective.
The full machinery is implemented but was never deployed.

---

## 5. What is Missing / Next Steps

### The policy weights were lost

The `bench_renders/ptheta_lora.safetensors` file exists, but MEMORY.md notes that
"the policy weights were lost (trained but not saved before the LoRA scale bug was
discovered; the save/load infrastructure now exists)." The LoRA scale bug was that
setting scale to a per-batch tensor `[1.0, 0.0]` for concurrent pi+ref batching
caused a shape mismatch on reload. This was fixed by making
`set_lora_scale` handle shape changes via buffer reassignment (lora.py, line 313).
The save/load infrastructure is now solid -- the `lora_roundtrip_test/` artifacts
prove it -- but the specific policy training run that produced the interesting
renders would need to be re-run.

### FlexAttention batch packing: implemented but not integrated

FlexAttention batch packing is fully designed (`docs/flexattention_batch_packing.md`,
417 lines), implemented in `diffusion_model.py` (`PackingInfo` dataclass,
`build_packed_sequence()`, `build_packed_rope()`, `forward_packed()`), tested
(`test_flexattn_integration.py` with 9 comparisons across 5 test cases, all
cos > 0.999), and has a server RPC (`handle_sample_trajectory_packed`, server.py
line 418, and `handle_warmup_packed`, line 708).

The dataset generator (`generate_btrm_dataset.py`) uses packed batching for t2i
trajectories -- it accumulates pending trajectories with the same step count and
flushes them in packs of up to 4 via `client.sample_trajectory_packed()` (line 268).
However, packed batching is not yet used in the training pipeline
(`smoke_test_e2e_training.py` uses single-image `sample_trajectory`).

The key limitation noted in MEMORY.md: "Each new `total_len` triggers recompilation
(~45-73s). Production must bucket total_len values and pre-warm." For dataset
generation where all t2i images are the same resolution (1280x832), this is not an
issue. For mixed-resolution i2i or variable prompt lengths, total_len bucketing would
be needed.

### The scrongle reward head: trained but unused in policy optimization

The BTRM head has two named outputs -- `bit_quality` (scrimble, attention quality)
and `step_quality` (scrongle, step count). The dataset generation pipeline
(`btrm_dataset.py`) builds both scrimble and scrongle pairs, and the multi-head loss
function (`btrm.py:compute_multihead_loss`) supports per-head masking with negative
tier weighting. But all training scripts that actually ran used only
`scores[:, 0]` (bit_quality). The step_quality head exists as a defined projection
in the `ScoreUnembedder` module but received no training signal.

This matters because the two degradation modes are orthogonal: scrimble affects
texture (per-pixel noise from attention quantization), while scrongle affects
structure (under-denoised features from insufficient sampling steps). A policy
optimized against both heads would be trained to produce outputs that are both
texturally clean and structurally complete, regardless of the compute budget. Using
only the scrimble head means the policy only learns to compensate for attention
quantization artifacts.

### No paired policy-on vs policy-off evaluation dataset

The `generate_policy_eval.py` file exists but was written after the policy training
run whose weights were lost. Without the trained policy weights, a comparative
evaluation dataset cannot be generated. The intended workflow was:

1. Generate trajectories with ptheta LoRA active (policy-on)
2. Generate trajectories with ptheta LoRA at scale=0 (policy-off)
3. Pair them by seed/prompt for controlled comparison
4. Score with the BTRM to verify the policy improves the reward signal
5. Render to PNG for visual comparison

The infrastructure for this exists -- the server supports `set_adapter_config` to
toggle LoRA scale between generations, and the client can request trajectories with
the same seed but different adapter configurations. What is missing is a completed
training run with saved weights.

### The trainer module: consolidation in progress

The `trainer.py` module (726 lines) represents an attempt to consolidate the training
logic from three separate smoke test scripts into a single canonical implementation.
It defines `TrainConfig`, `TrainingState`, `setup_training()`, `train_btrm_epoch()`,
`rollout_group()`, `policy_step()`, and `train_loop()`. The `train_loop` function
implements both phases (BTRM training then policy optimization) with proper adapter
lifecycle management.

However, `trainer.py` imports from `btrm.py` (which was not among the files I was
asked to read but is referenced throughout), and it uses a local
`forward_checkpointed()` from `training_utils.py` rather than the server's
`accumulate_policy_gradients` RPC. This means `trainer.py` is designed for the
single-process path (load models locally, no server) while
`smoke_test_e2e_training.py` demonstrates the server-based path. The two paths have
not been unified.

### The multi-LoRA Triton kernel backward

The sparse multi-LoRA Triton kernel in `lora_kernels.py` has a registered autograd
backward (lines 406-454), but the `docs/btrm_rl_architecture_review.md` states
(line 213): "Backward is NOT implemented" with a `raise NotImplementedError`. Looking
at the actual code, the backward IS implemented (lines 412-452) using standard
PyTorch operations (not a Triton kernel), and it is registered via
`multi_lora_op.register_autograd(_backward, setup_context=_setup_context)` at line
454. The review document appears to be outdated on this point -- the backward was
added after the review was written.

The cat-based path in `LoRALinear.forward()` (lora.py, lines 156-196) remains the
production path for training, while the Triton sparse kernel is used for inference
(N > 1 adapters). For N=1, the fast path (line 167) is a simple pair of matmuls.

---

## 6. Structural Observations

### The progression from port to platform

The codebase tells a clear evolutionary story:

1. **Bug-for-bug port** (generate.py, sampling.py, attention.py): Exact ComfyUI
   replication with a `comfyui_compat` flag that enables the validated dtype flow
   (f16 TE, f32 sampling loop, CPU noise). This is the project's origin.

2. **Performance optimization** (fp8.py, fp8_kernels.py, fused_kernels.py,
   sage_attention.py): FP8 blockwise quantization, Triton GEMM kernels,
   SageAttention INT8/FP8 QK matmuls, fused elementwise kernels, pre-batched adaLN.
   All registered as `torch.library.custom_op` for zero graph breaks under
   `torch.compile`.

3. **Inference server** (server.py, client.py, protocol.py): The compute surface
   extracted into a long-running process. Model lifecycle management, LoRA
   persistence across swaps, ZeroMQ wire protocol.

4. **Training infrastructure** (lora.py, lora_kernels.py, policy_loss.py,
   training_utils.py, trainer.py): Multi-adapter LoRA with per-batch routing, DRGRPO
   loss formulation, gradient-checkpointed forward, sparse step sampling.

5. **Dataset generation** (btrm_dataset.py, generate_btrm_dataset.py): Schedule-driven
   trajectory collection with FlexAttention batch packing, interruptible resume,
   progress tracking.

6. **FlexAttention packing** (diffusion_model.py forward_packed, attention.py
   block_mask dispatch): Multi-image inference in a single forward pass for dataset
   generation throughput.

Each layer built on the previous one. The FP8 custom ops were needed for
`torch.compile`, which was needed for the inference server's performance, which was
needed for dataset generation throughput, which was needed for BTRM training data.

### The "registered buffer as graph input" pattern

A recurring design pattern throughout the codebase is the use of PyTorch registered
buffers as dynamically-valued graph inputs for `torch.compile`. The clearest example
is `LoRAAdapter.lora_scale` (lora.py, lines 79-82):

```python
self.register_buffer(
    "lora_scale",
    torch.ones(1, dtype=torch.bfloat16, device=device),
)
```

Because `lora_scale` is a registered buffer (not a parameter), `torch.compile` treats
it as a graph input operand. Same-shape value changes use `buf.copy_(scale)` (lora.py,
line 310) -- no recompilation needed. Shape changes (e.g., scalar to per-batch)
trigger one recompilation. This enables the concurrent pi+ref batching pattern:
`set_lora_scale(model, tensor([1.0, 0.0]))` makes batch[0] use the LoRA and batch[1]
skip it, all within the compiled CUDA graph.

This pattern is essential for the training pipeline's efficiency. Without it, every
scale change would require a `torch._dynamo.reset()` and full recompilation (~30s
for the 30-layer NextDiT). With it, scale toggling is a memory copy.

### The absence of abstractions

The codebase is notably light on abstraction. There is no base class for models, no
registry pattern for RPC methods, no configuration framework beyond dataclasses. The
server's dispatch table is a flat dictionary (server.py, lines 1269-1286). The
protocol is hand-rolled JSON + bytes. The LoRA injection walks the module tree with
string matching on path suffixes.

This is a deliberate choice, consistent with the CLAUDE.md instruction "Prefer
'defect legibilizing' patches over 'defect concealing' workarounds." Every bug in the
system is visible at the call site. There is no indirection to hide behind. When
`forward_checkpointed` runs the embedding in `no_grad` and the main layers with
`grad_ckpt`, you can see exactly which tensors have gradient connections and which
do not (training_utils.py, lines 247-314).

The cost is code duplication -- the CFG model function is defined separately in
`generate.py` (lines 266-325), `server.py` (lines 381-392), and
`training_utils.py` (lines 132-171). Each is slightly different (the generate.py
version handles compat mode; the server.py version uses the compiled model; the
training_utils.py version is for the single-process training path). A shared
abstraction would reduce duplication but add a layer of indirection that makes each
call site harder to debug independently.

### The DRGRPO loss: complete but underused

The `policy_loss.py` module (226 lines) implements a full DRGRPO (Diffusion-adapted
GRPO) loss formulation:

- `compute_group_advantages`: Z-score normalization within a group of K rollouts
- `compute_step_log_ratios`: Per-step Gaussian log-probability ratio between policy
  and reference, naturally sigma-weighted (noisier steps get less signal)
- `clipped_policy_loss`: Asymmetric clipping (DAPO-style, clip_low=0.2,
  clip_high=0.28) for the PPO surrogate
- `latent_entropy_bonus`: Anti-mode-collapse penalty from latent variance across
  rollouts
- `reference_anchor_loss`: MSE anchor between policy and reference outputs
- `drgrpo_diffusion_loss`: Full assembly of all components with configurable weights

This is a carefully designed loss function. The asymmetric clipping (less aggressive
at suppressing improvements than penalizing regressions) addresses a known failure
mode of symmetric PPO in diffusion settings. The sigma-weighted log-ratios naturally
downweight early noisy steps and upweight late clean steps. The entropy bonus prevents
mode collapse where all rollouts converge to the same output.

Yet none of this machinery was used in the actual training runs. The smoke tests used
`loss = -reward_bq` (direct reward maximization) or `loss = step_loss * advantage`
(vanilla REINFORCE). The DRGRPO formulation is fully implemented and tested
(`test_scrongle_scrimble.py` validates the loss components on synthetic data) but
awaits a production training run.

---

## 7. Summary

futudiffu is a project that began as a faithful ComfyUI port and evolved into
something more ambitious: a diffusion inference server that doubles as a policy
environment for reinforcement learning. The server/client split is the load-bearing
architectural decision, enabling LoRA weight updates during training sessions,
concurrent pi+ref batching for efficient REINFORCE gradient estimation, and VRAM
lifecycle management across mutually exclusive model phases.

The concrete artifacts -- 50 dataset trajectories, a trained BTRM head, bench renders
showing the policy found a different attractor basin, a crash dump mechanism for
emergency state preservation -- demonstrate that the pipeline works end-to-end. The
theoretical machinery -- DRGRPO loss with asymmetric clipping, multi-head BTRM with
negative tier weighting, FlexAttention batch packing, sparse multi-LoRA Triton
kernels -- is implemented and tested but largely unused in production.

The gap between what is implemented and what has been deployed is the project's most
salient characteristic. The codebase has more infrastructure than it has consumed. The
multi-head loss function exists but is never called with real data. The scrongle head
is defined but untrained. The DRGRPO loss components are tested but the full assembly
was never used for policy optimization. FlexAttention packing works for dataset
generation but not for training rollouts. The reference anchor loss is computed but
was not included in the policy objective that produced the bench renders.

This is not unusual for research codebases operating at the frontier of what a single
GPU can do. The constraint is not code but compute: each training run on a 4090
requires exclusive GPU access for minutes to hours, and the experimental space (reward
model architecture, policy loss formulation, attention backend, LoRA rank and
placement) is large. The codebase is prepared for experiments that have not yet been
run.

The most impactful next step, as identified by the project's own architecture review
(`docs/btrm_rl_architecture_review.md`), is adding a KL divergence penalty to the
policy loss. The per-batch LoRA routing already supports concurrent policy+reference
forward passes. The `reference_anchor_loss` function exists in `policy_loss.py`. The
infrastructure is complete -- it needs to be wired into the training loop and validated
against the reward hacking behavior observed in the bench renders.

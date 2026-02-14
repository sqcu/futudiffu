# BTRM + RL Architecture Review

Comprehensive review of the reinforcement learning and reward model architecture
in the futudiffu codebase. This document describes what EXISTS in code today,
identifies structural gaps, and provides concrete recommendations.

Last reviewed against codebase state: 2026-02-13.

---

## 1. Pseudo-BTRM Loss Definition

### 1.1 What Exists

The BTRM (Bradley-Terry Reward Model) loss is implemented in
`/mnt/f/dox/repos/ai/futudiffu/src/futudiffu/btrm.py`.

**Core BT loss** (lines 128-141):
```python
def bradley_terry_loss(pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
    return -F.logsigmoid(pos_scores - neg_scores).mean()
```
This is the standard Bradley-Terry ranking loss: `P(pos > neg) = sigmoid(pos - neg)`,
minimized as `-log_sigmoid(margin).mean()`. No modifications, no trust region
penalty, no KL term. It is a clean pairwise ranking loss.

**Logsquare regularizer** (lines 144-162):
```python
def logsquare_regularizer(scores: Tensor, eps: float = 1e-6) -> Tensor:
    return torch.log(scores ** 2 + eps).mean()
```
This regularizes positive scores toward `|r| ~ 1`. At `r=1`: `log(1) = 0`.
At `r=10`: `log(100) ~ 4.6` (penalized). The docstring explicitly states
"This is NOT MSE to batch mean." The regularizer serves a reward scale anchoring
role but is NOT a trust region constraint.

**Multi-head loss** (`compute_multihead_loss`, lines 165-261):
- Per-head masking via `pos_head_indices` (each positive belongs to exactly one head)
- Multiple negative tiers with configurable weights (DEFAULT_TIER_WEIGHTS):
  - `soft_neg`: 2.0x (close to full-precision, e.g., FP8+BF16)
  - `semi_firm_neg`: 1.0x (moderately degraded)
  - `furthest_neg`: 0.5x (heavily degraded)
- BT loss computed per tier, summed with weights
- Logsquare applied to positive scores only
- Averaged across active heads in the batch
- Total: `L = L_BT_weighted + lambda * L_logsquare`

### 1.2 Bounded Maximum Score (Soft Tanh Cap)

The `BTRMHead` (lines 65-121) applies a soft tanh cap:
```python
if self.logit_cap > 0:
    scores = self.logit_cap * torch.tanh(scores / self.logit_cap)
```
Default `logit_cap = 10.0`. Scores are smoothly bounded to `[-10, +10]`. This
provides:
- Bounded intra-class distances (scores cannot diverge to infinity)
- Smooth gradients everywhere (no hard clipping)
- A form of implicit trust region on the score space

This is the ONLY trust region mechanism in the system. There is no explicit KL
divergence penalty, no PPO clipping, no TRPO constraint.

### 1.3 What Is Missing

1. **No KL divergence penalty between policy and reference**: The policy loss
   in `smoke_test_btrm_v2.py` (line 607) and `smoke_test_btrm_policy.py`
   (line 566) is simply `loss = -reward_bq`. There is no
   `beta * KL(pi_theta || pi_ref)` term. Without this, the policy can drift
   arbitrarily far from the reference model in output distribution, potentially
   producing reward-hacked outputs that score high on the BTRM but look nothing
   like natural images.

2. **No PPO clipping or importance sampling**: The REINFORCE implementation in
   `train_qat.py` (lines 565-681) uses raw advantage weighting:
   `weighted_loss = step_loss * advantages[k]`. There is no clipped surrogate
   objective (`min(r * A, clip(r, 1-eps, 1+eps) * A)`). This is vanilla
   REINFORCE/GRPO, not PPO.

3. **No trust region on LoRA weight updates**: The optimizer is plain AdamW
   with gradient clipping (`clip_grad_norm_`, default 1.0). There is no
   proximal constraint on the LoRA weight delta per iteration.

4. **Logsquare regularizer is applied only during BTRM head training, not
   during policy optimization**: The policy loss functions in the smoke tests
   and `train_qat.py` do not include any regularizer on the BTRM scores during
   policy gradient steps.

---

## 2. Model Architecture: r_theta, p_theta, and Reference

### 2.1 The Three Roles

The codebase defines three conceptual models that share a single physical
NextDiT backbone:

**r_theta (Reward Model)**:
- Frozen NextDiT backbone + LoRA("rtheta", layers 28-29) + BTRMHead
- `BTRMHead`: `RMSNorm(3840) -> Linear(3840, N_heads) -> soft_tanh_cap`
- Default heads: `("bit_quality", "step_quality")`
- Captures hidden states from the last transformer block via forward hook
  (`BTRMWrapper._run_backbone`, `btrm.py` line 319)
- Scoring: `hidden_states -> mean_pool(dim=1) -> norm -> proj -> tanh_cap`
- LoRA on last 2 layers gives the reward model nonlinear capacity beyond the
  frozen backbone's representational bias

**p_theta (Policy)**:
- Same NextDiT backbone + LoRA("ptheta", all layers)
- Trainable: only ptheta LoRA A/B matrices
- Runs under quantized attention (SageAttention) to learn compensating
  corrections
- The policy objective is `loss = -BTRM_score(hidden_states_from_policy_forward)`

**Reference Model (pi_ref)**:
- The frozen NextDiT backbone with NO LoRA adapters active (all scales = 0)
- Used implicitly to generate SDPA "gold standard" trajectories
- NOT used explicitly as a KL anchor during policy optimization (this is a gap)

### 2.2 Physical Implementation

All three roles share one physical `NextDiT` model on GPU. The LoRA system
(`/mnt/f/dox/repos/ai/futudiffu/src/futudiffu/lora.py`) supports horizontal
multi-adapter stacking on the same `LoRALinear` wrapper. At any given time:

- Base weights: frozen, shared across all roles
- rtheta LoRA: frozen after BTRM training, scale=0 during policy forward
- ptheta LoRA: trainable during policy optimization, scale=1 during policy forward
- Per-batch routing via `set_lora_scale()` allows concurrent score+sample in
  one forward pass (demonstrated in `test_multilora_fused.py`, lines 231-293)

### 2.3 BTRMWrapper

`btrm.py` lines 268-394 define `BTRMWrapper`:
- Wraps a frozen NextDiT + BTRMHead
- Installs a forward hook on `model.layers[-1]` to capture pre-final-layer
  hidden states
- `score()` method: runs frozen backbone forward, captures hidden, passes to
  BTRMHead
- `train()` override: only head trains, backbone stays in eval

However, `BTRMWrapper` is NOT used in the smoke tests. The smoke tests
(`smoke_test_btrm_policy.py`, `smoke_test_btrm_v2.py`) use a manual
`HiddenCapture` class or `forward_checkpointed()` to extract hidden states.
`BTRMWrapper` exists as a clean API but the training scripts bypass it for
gradient checkpointing reasons.

---

## 3. Multi-LoRA Setup

### 3.1 Horizontal Fused Multi-LoRA (Cat-Based Path)

File: `/mnt/f/dox/repos/ai/futudiffu/src/futudiffu/lora.py`

**LoRALinear** (lines 85-195) wraps a base `nn.Linear` or `FP8Linear` with N
named adapter slots via `nn.ModuleDict`:

For N=1 (fast path, line 150):
```python
return base_out + adapters[0](x_bf16).to(base_out.dtype)
```

For N>1 (fused path, lines 153-195):
```python
A_cat = cat([A_1, ..., A_N], dim=0)      # gather A matrices
mid   = x @ A_cat.T                       # single matmul
mid   = mid * scale_vec                  # per-adapter per-batch scale
B_cat = cat([B_1, ..., B_N], dim=1)      # gather B matrices
out   = mid @ B_cat.T                    # single matmul (scatter + sum)
```

Two matmuls regardless of N adapters. The `scale_vec` supports per-batch
per-adapter gating: `scale=0` disables an adapter for a batch element without
graph mutation (CUDA-graph safe).

**inject_lora** (lines 202-263): Injects a named adapter into matching linear
layers. Supports:
- `layer_indices` parameter for sparse injection (e.g., only layers 28-29)
- Automatic wrapping: if target is already `LoRALinear`, adds a new adapter slot
- Target module matching by suffix (default: qkv, out, w1, w2, w3)

**State dict round-trip**: `lora_state_dict()` and `load_lora_state_dict()`
(lines 332-380) serialize/deserialize per-adapter weights with keys like
`path.adapters.name.lora_A`.

### 3.2 Triton Sparse Multi-LoRA Kernel

File: `/mnt/f/dox/repos/ai/futudiffu/src/futudiffu/lora_kernels.py`

A Triton kernel (`_multi_lora_kernel`, lines 55-155) that:
- Takes pre-stacked A/B matrices: `A_all (N, rank, in)`, `B_all (N, out, rank)`
- Per-batch per-adapter scale: `scale_all (B, N)` in float32
- Loops over adapters (unrolled at compile time, `N_ADAPTERS` is `tl.constexpr`)
- **Uniform branch**: `if scale != 0` is evaluated at the program level (all
  threads in a warp see the same scale), so zero-scale adapters are skipped
  with zero warp divergence
- Both A-projection and B-projection matmuls are skipped entirely for zero-scale
  adapters

Performance (from `test_multi_lora_kernel`, lines 566-636):
- 50% sparsity: 1.70x speedup over dense
- Registered as `torch.library.custom_op("futudiffu::multi_lora")` with
  `register_fake` for torch.compile compatibility

**Backward is NOT implemented** (lines 412-421):
```python
def _backward(ctx, grad_out):
    raise NotImplementedError(
        "multi_lora_op backward is not implemented. "
        "Use multi_lora_forward() directly in inference mode."
    )
```

This means the Triton sparse kernel CANNOT be used for training. Training uses
the cat-based path in `LoRALinear.forward()`, which is autograd-compatible via
standard PyTorch operations.

### 3.3 How Policy and Reward LoRAs Are Managed

From `smoke_test_btrm_v2.py` (lines 522-647):

1. **Phase 2**: `inject_lora(model, name="rtheta", layer_indices={28, 29})`
   - rtheta LoRA on last 2 layers only
   - Train with Bradley-Terry loss against SDPA vs Sage hidden states

2. **Phase 3**: After rtheta training:
   - `freeze_adapter(model, "rtheta")` -- set requires_grad=False
   - `set_lora_scale(model, tensor([0.0]), adapter_name="rtheta")` -- zero scale
   - `inject_lora(model, name="ptheta")` -- all layers
   - Layers 28-29 now have BOTH rtheta (frozen, scale=0) and ptheta (trainable, scale=1)
   - Layers 0-27 have only ptheta

This stacking is sound: rtheta contributes nothing to the forward pass (scale=0),
but its weights remain available for scoring when scale is toggled back to 1.0.

### 3.4 Concurrent Score+Sample Architecture

Demonstrated in `test_multilora_fused.py` (lines 231-293):
```python
set_lora_scale(model, tensor([1.0, 0.0]), adapter_name="policy")
set_lora_scale(model, tensor([0.0, 1.0]), adapter_name="rtheta")
x_batch = x.expand(2, -1, -1).clone()
out = model(x_batch)
sample_out = out[0]   # policy-adapted
score_out = out[1]    # rtheta-adapted
```

Batch element 0 sees only the policy adapter; batch element 1 sees only rtheta.
Single forward pass produces both sampling output and scoring output. This is
verified to produce different outputs and correct gradient flow.

---

## 4. RL Objective and Error Taxonomy

### 4.1 The Qualitative RL Objective

The goal is to train LoRA adapters so that FP8/INT8 quantized attention
(SageAttention) produces outputs matching full-precision SDPA. The BTRM provides
a differentiable reward signal measuring output quality.

**Two training modes exist in `train_qat.py`**:

**Direct mode** (lines 504-563):
```
loss = MSE(sage_with_lora_latent, sdpa_reference_latent)
```
Backpropagates through all checkpointed euler steps. Direct supervision, no RL.

**REINFORCE mode** (lines 565-681):
```
reward = cosine_sim(rollout_latent, reference_latent)
advantage = reward_k - mean(rewards)
loss = sparse_step_loss * advantage
```
GRPO-style: K rollouts per group, cosine similarity as reward, advantage
normalization via group mean. Only positive-advantage rollouts contribute
gradients. Sparse step sampling (3-5 of 30 steps) for FLOPS efficiency.

**BTRM-based policy optimization** (smoke tests only):
```
loss = -btrm_head(hidden_states)[:, 0]   # maximize bit_quality score
```
Direct gradient ascent on the BTRM score. No RL wrapper -- just differentiable
reward maximization through gradient checkpointed transformer layers.

### 4.2 Scrongle vs Scrimble Error Taxonomy

**Definitions** (from `btrm_dataset.py` docstring, lines 1-29, and
`test_scrongle_scrimble.py` docstring, lines 1-15):

- **Scrongle**: step-count artifacts. Too few sampling steps produce
  under-denoised images with visible noise patterns, blurry details, or
  incomplete structures. Paired as: same (seed, prompt), different step count.

- **Scrimble**: quantization artifacts. Lower-precision weights (FP8/INT8
  attention) introduce subtle distortions -- color shifts, texture degradation,
  edge artifacts. Paired as: same (seed, prompt, steps), different attention
  backend (SDPA vs Sage).

**Implementation status**:

The error taxonomy IS implemented in the dataset generation layer:

- `btrm_dataset.py` `build_training_pairs()` (lines 542-596) constructs
  explicit scrongle and scrimble pair lists from trajectory records
- `btrm_dataset.py` `BTRMDatasetConfig` defines the variation axes:
  8 step schedules (8-22), 2 attention backends (sdpa, sage)
- `test_scrongle_scrimble.py` validates a `BTRMHead` with two named heads
  `("scrongle", "scrimble")` can discriminate both degradation types on
  synthetic MiniModel data. Target: 75% per-head classification accuracy.

The taxonomy is NOT fully wired into the production training pipeline:

- `smoke_test_btrm_policy.py` uses `head_names=("bit_quality", "step_quality")`
  but only trains on bit_quality (SDPA vs Sage at the SAME step count). The
  step_quality head receives no training signal.
- `smoke_test_btrm_v2.py` also uses `("bit_quality", "step_quality")` and
  only trains the bit_quality head (BT loss on `scores[:, 0]`).
- The multi-head loss function `compute_multihead_loss()` with per-head masking
  and negative tiers EXISTS in `btrm.py` but is NOT CALLED by any training
  script. All training scripts use the raw `bradley_terry_loss()` directly.
- The negative tier hierarchy (soft_neg, semi_firm_neg, furthest_neg) is
  defined as `DEFAULT_TIER_WEIGHTS` in `btrm.py` but never instantiated with
  real data.

### 4.3 Dataset Generation Architecture

`generate_btrm_dataset.py` is a pure scheduling client that delegates all GPU
work to the inference server. It:

1. Collects all unique prompts and encodes them via server (phase 1)
2. Frees the TE on server, warms up diffusion model (phase 2)
3. Generates trajectories per-schedule-batch, saving latents + metadata (phase 3)
4. Optionally VAE-decodes selected trajectories for visual QA (phase 4)

The schedule format supports:
- t2i and i2i trajectory types
- Per-batch precision (sdpa vs sage)
- Per-batch step count (fixed or range)
- Render counts for visual QA

This is a data generation pipeline. It does NOT train any model. The generated
latent trajectories are meant to be consumed by a BTRM training loop that does
not yet exist in production form (only in smoke tests).

---

## 5. Inference Server + Train Server Architecture

### 5.1 What Exists

**Inference server** (`/mnt/f/dox/repos/ai/futudiffu/src/futudiffu/server.py`):
- ZeroMQ REP socket, single-threaded
- Manages model lifecycle: loads/frees TE, diffusion, VAE as needed
- RPC methods: `encode_prompt`, `sample_trajectory`, `vae_encode`, `vae_decode`,
  `warmup`, `status`, `free`
- Model groups are mutually exclusive for VRAM: TE (~7.5GB) vs diffusion (~8GB)
- The diffusion model is `torch.compile`'d and fused (`fuse_model()`)

**Inference client** (`/mnt/f/dox/repos/ai/futudiffu/src/futudiffu/client.py`):
- Thin ZeroMQ REQ wrapper
- Methods mirror server RPC: `encode_prompt()`, `sample_trajectory()`,
  `vae_encode()`, `vae_decode()`, `warmup()`, `status()`, `free()`

**Key limitation**: The server has NO RPC for LoRA weight updates. There is no
`load_lora`, `update_lora`, `set_lora_scale`, or `swap_adapter` method.

### 5.2 Can Iterative LoRA Weight Updates Work?

**Current answer: No, not without significant additions.**

The inference server (`server.py`) treats the diffusion model as immutable after
loading. It:
- Loads the model once in `_ensure_diffusion()` (line 115)
- Applies `fuse_model()` and `torch.compile()` (lines 157-159)
- Uses the compiled model for all subsequent requests

To support iterative LoRA weight updates, the following would be needed:

1. **New RPC method: `load_lora_weights`**
   - Accept a serialized LoRA state dict (adapter name + A/B tensors)
   - Call `inject_lora()` if first load, or `load_lora_state_dict()` for updates
   - `torch._dynamo.reset()` after weight update (compiled graph may cache
     weight tensor metadata)

2. **New RPC method: `set_adapter_config`**
   - Set per-adapter scales for routing
   - Toggle between inference mode (policy only) and scoring mode (rtheta only)

3. **torch.compile invalidation**:
   - The current server compiles with `mode="default"`. Changing LoRA weights
     in-place (`.data.copy_()`) should be transparent to CUDA graphs IF the
     tensors are not resized. The `load_lora_state_dict()` function in `lora.py`
     (line 375) does `param.data.copy_(tensor)`, which preserves tensor identity.
   - However, the FIRST `inject_lora()` call replaces `nn.Linear` with
     `LoRALinear`, which mutates the module tree. This MUST happen before
     `torch.compile()`, or a dynamo reset is required.

4. **Train server**: Does not exist. Training currently happens in standalone
   scripts (`train_qat.py`, `smoke_test_btrm_policy.py`, `smoke_test_btrm_v2.py`)
   that load models locally. There is no train-server <-> inference-server
   communication protocol.

### 5.3 Proposed Architecture for Iterative Training

The cleanest path (given existing code):

```
[Train Process]                      [Inference Server]
    |                                      |
    |-- encode_prompt() ------------------>|
    |<-- conditioning --------------------|
    |                                      |
    |-- sample_trajectory(policy_lora) --->|  # with policy LoRA active
    |<-- step latents + final ------------|
    |                                      |
    |-- [local] BTRM score latents         |
    |-- [local] compute advantage/loss     |
    |-- [local] backward through           |
    |          checkpointed steps          |
    |-- [local] optimizer.step()           |
    |                                      |
    |-- update_lora(new_weights) --------->|  # push updated weights
    |                                      |
    |  [repeat]                            |
```

But this requires the inference server to:
- Accept LoRA weight updates mid-session
- Apply them to the compiled model without full recompilation
- Support per-request attention backend switching (already exists)

An alternative: run training and inference in the same process (as the smoke
tests do). This avoids the serialization overhead but limits to single-GPU and
requires careful VRAM management.

---

## 6. Component-by-Component Status

### 6.1 Implemented and Tested

| Component | File | Status |
|-----------|------|--------|
| BTRMHead (multi-head scoring) | `btrm.py:65-121` | Implemented, tested in smoke tests |
| Bradley-Terry loss | `btrm.py:128-141` | Implemented, used in all training scripts |
| Logsquare regularizer | `btrm.py:144-162` | Implemented, used in smoke test BTRM training |
| Soft tanh score capping | `btrm.py:114-115` | Implemented, default logit_cap=10.0 |
| Multi-head loss with tiers | `btrm.py:165-261` | Implemented, NOT USED by any training script |
| BTRMWrapper | `btrm.py:268-394` | Implemented, NOT USED by training scripts |
| LoRALinear (multi-adapter) | `lora.py:85-195` | Implemented, tested |
| inject_lora (sparse layers) | `lora.py:202-263` | Implemented, tested |
| Per-batch LoRA scale routing | `lora.py:285-297` | Implemented, tested |
| Freeze/unfreeze adapters | `lora.py:314-325` | Implemented, tested |
| State dict save/load | `lora.py:332-380` | Implemented, tested |
| Triton sparse multi-LoRA | `lora_kernels.py:55-155` | Implemented, forward only (no backward) |
| Gradient checkpointing | `smoke_test_btrm_policy.py:378-465` | Implemented in smoke tests |
| Direct QAT mode | `train_qat.py:504-563` | Implemented |
| REINFORCE/GRPO mode | `train_qat.py:565-681` | Implemented |
| Euler with grad flow | `sampling.py:127` | Implemented (`sample_euler_train`) |
| Sparse step loss | `sampling.py:201` | Implemented (`sparse_step_loss`) |
| Dataset plan builder | `btrm_dataset.py:345-496` | Implemented |
| Pair construction | `btrm_dataset.py:542-596` | Implemented |
| Scrongle/scrimble test | `test_scrongle_scrimble.py` | Implemented on synthetic data |
| Inference server | `server.py` | Implemented, no LoRA support |
| Inference client | `client.py` | Implemented |
| Dataset generation driver | `generate_btrm_dataset.py` | Implemented, server-backed |

### 6.2 Not Implemented

| Component | Description | Priority |
|-----------|-------------|----------|
| KL divergence penalty | `beta * KL(pi_theta \|\| pi_ref)` in policy loss | HIGH -- prevents reward hacking |
| PPO clipping | Clipped surrogate objective | MEDIUM -- vanilla REINFORCE may suffice for this domain |
| Multi-head training pipeline | Uses `compute_multihead_loss` with real data | HIGH -- the function exists but is never called |
| Step_quality head training | Scrongle pairs with varying step counts | MEDIUM -- data generation supports it |
| LoRA weight update RPC | Server method to accept LoRA updates | HIGH -- needed for train/infer split |
| Train server | Separate process for training computation | LOW -- single-process works for 4090 |
| Reference model KL computation | Forward pass through base model for KL anchor | HIGH -- coupled with KL penalty |
| Triton multi-LoRA backward | Backward kernels for `_multi_lora_kernel` | LOW -- cat-based path works for training |
| Production BTRM training loop | Not a smoke test, uses full dataset | HIGH |
| Negative tier data assignment | Assigning real trajectories to soft/semi/furthest tiers | MEDIUM |

---

## 7. Concrete Recommendations

### 7.1 Add KL Divergence Penalty (Highest Priority)

The policy loss `loss = -reward` is unbounded. Add a KL penalty:

```python
# During policy forward, also run reference (no LoRA, scale=0):
#   batch[0] = policy (ptheta active), batch[1] = reference (all LoRA scale=0)
# Compute KL on output distributions (denoised latents or hidden states)
kl = F.kl_div(log_softmax(policy_hidden), softmax(ref_hidden), reduction='batchmean')
loss = -reward + beta * kl
```

The per-batch LoRA routing already supports this: set ptheta scale to
`[1.0, 0.0]` so batch[0] is policy and batch[1] is reference. Single forward
pass gives both.

### 7.2 Wire Up compute_multihead_loss

The multi-head loss with negative tiers already exists (`btrm.py:165-261`).
Create a training script that:
1. Loads trajectories from the BTRM dataset (generated by `generate_btrm_dataset.py`)
2. Assigns negative tiers based on attention backend and step count
3. Calls `compute_multihead_loss()` with proper `pos_head_indices`
4. Trains BOTH scrongle and scrimble heads simultaneously

### 7.3 Unify Training Scripts

There are currently THREE training implementations:
- `train_qat.py` (standalone, self-contained LoRA injection, no multi-adapter)
- `smoke_test_btrm_policy.py` (single-adapter LoRA + BTRM, no gradient checkpointing)
- `smoke_test_btrm_v2.py` (multi-adapter horizontal LoRA + BTRM, gradient checkpointing)

`train_qat.py` has its OWN `LoRALinear` class (lines 101-146) that is NOT the
same as `lora.py:LoRALinear`. It does not support multi-adapter or per-batch
scaling. This duplication should be eliminated.

Recommendation: a single `train.py` that:
- Uses `lora.py` for injection (not its own LoRA class)
- Supports both direct and REINFORCE modes
- Supports BTRM-based reward (not just MSE/cosine)
- Uses gradient checkpointing
- Supports the horizontal multi-adapter architecture for concurrent score+sample

### 7.4 Add LoRA Update RPC to Inference Server

For iterative training with server-backed inference:

```python
# In server.py, add:
def handle_update_lora(self, params, tensors):
    """Update LoRA adapter weights on the loaded diffusion model."""
    adapter_name = params["adapter_name"]
    # First injection: inject_lora, then compile
    # Subsequent: load_lora_state_dict (in-place, no recompile needed)
    load_lora_state_dict(self._diff_model, tensors)
    return pack_response("ok")
```

Weight updates via `.data.copy_()` do not invalidate torch.compile graphs
(tensor identity is preserved). The first injection requires a dynamo reset.

### 7.5 Consider Reward Model Validation

The BTRM head is trained on hidden states from specific sigma values. Its
generalization across the full sigma range is unvalidated. Recommendation:
after BTRM training, evaluate classification accuracy at ALL 30 sigma values
(not just the 5 probed during training) to verify the reward signal is reliable
across the denoising trajectory.

---

## 8. Summary of Gaps

**Structural gaps** (missing components that affect correctness):
1. No KL penalty in policy loss -- risk of reward hacking
2. Multi-head loss function exists but is never called with real training data
3. Step_quality head is defined but never trained
4. `train_qat.py` uses a duplicate LoRA implementation that diverges from `lora.py`

**Operational gaps** (missing components that affect deployment):
1. Inference server has no LoRA weight update mechanism
2. No production training loop (only smoke tests with 5-step probes)
3. No automated evaluation pipeline (BTRM accuracy vs human judgment)

**Architecture gaps** (design decisions not yet made):
1. Single-process vs distributed training (currently single-process only)
2. Reward model update frequency (train once vs iteratively with policy)
3. Whether to use the Triton sparse kernel for training (requires backward implementation)
4. FlexAttention batch packing integration with training (design doc exists, no implementation)

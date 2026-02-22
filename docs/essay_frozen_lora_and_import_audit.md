# Frozen Multi-LoRA Architecture & Import Audit

Research survey of the frozen `src/futudiffu/` multi-LoRA implementation and
a complete inventory of frozen imports from `src_ii/` and `scripts_ii/`.

Blueprint for Tasks #69 (src_ii multi-tenant LoRA) and #70 (import elimination).

---

## Section 1: Multi-LoRA Architecture in Frozen Codebase

### 1.1 Adapter Data Structure (`src/futudiffu/lora.py`)

#### LoRAAdapter (lines 40-91)

A single rank-r adapter stored as an `nn.Module`:

```
lora_A: nn.Parameter  -- (rank, in_features) BF16, kaiming_uniform init
lora_B: nn.Parameter  -- (out_features, rank) BF16, zero or N(0, std) init
lora_scale: buffer     -- (1,) or (B,) BF16, registered buffer
```

Key design decisions:

- **`lora_scale` is a registered buffer**, not a parameter (line 79). This is
  critical: `torch.compile` treats registered buffers as dynamic graph inputs.
  Changing the buffer value (same shape) uses `copy_()` with zero recompilation.
  Changing the buffer shape (e.g., from `(1,)` to `(B,)`) triggers exactly one
  recompilation.

- **Per-batch-element routing** is encoded in `lora_scale` shape. `(1,)` broadcasts
  uniformly to all batch elements. `(B,)` gives per-batch-element gating: e.g.,
  `[1, 0]` means LoRA active for batch[0], silent for batch[1]. This is how
  policy-vs-reference batching works in REINFORCE training (line 617 of trainer.py:
  `set_lora_scale(model, tensor([1.0, 0.0]), adapter_name="ptheta")`).

- **`scale = alpha / rank`** is pre-folded into the constant `self.scale` (line 64).
  The effective contribution is `lora_scale * (alpha/rank) * (x @ A^T) @ B^T`.

- **`init_b_std`** controls initialization of B. Zero-init (standard LoRA) means
  the adapter starts as identity. Nonzero (`init_b_std=0.01`) is required for
  policy adapters to avoid zero MSE gradient (MEMORY.md pattern).

#### LoRALinear (lines 94-196)

A wrapper around any `nn.Linear` or `FP8Linear` that holds N named adapter slots:

```python
class LoRALinear(nn.Module):
    base: nn.Module          # frozen nn.Linear or FP8Linear
    adapters: nn.ModuleDict  # {name: LoRAAdapter}
```

- **Base is always frozen** (lines 109-113): `requires_grad=False` for all parameters
  and buffers of the wrapped base module.

- **Attribute forwarding** (lines 136-143): `__getattr__` delegates to `self.base`
  for FP8-specific attributes (`block_size`, `weight_scale`, `output_dtype`,
  `_transposed`). This lets the fused FFN chain access FP8Linear internals
  through the LoRALinear wrapper transparently.

- **All adapter counts dispatch to the Triton kernel** (lines 167-196). Even N=1
  uses `multi_lora_op` rather than the standalone `LoRAAdapter.forward()` method.
  The rationale (lines 168-170): using the custom_op for N=1 replaces 2 visible
  matmul graph nodes with 1 opaque node, eliminating inductor GEMM analysis for
  LoRA paths entirely.

#### Forward Path (lines 158-196)

```
base_out = self.base(x)       # FP8 GEMM (frozen)
x_bf16 = x.to(bf16)
A_all = stack([a.lora_A ...])  # (N, R, IN)
B_all = stack([a.lora_B ...])  # (N, OUT, R)
scale_all = stack([a.lora_scale.expand(B) * a.scale ...])  # (B, N) float32
delta = multi_lora_op(x_3d, A_all, B_all, scale_all, n, R)
return base_out + delta
```

The `stack` + `multi_lora_op` pattern means:
- All adapters are evaluated in a single fused kernel call
- The kernel is opaque to torch.compile (custom_op)
- No graph breaks regardless of adapter count

### 1.2 Triton Kernel (`src/futudiffu/lora_kernels.py`)

#### Kernel Design (lines 55-155)

The `_multi_lora_kernel` is a Triton JIT kernel with these constexpr parameters:

```
N_ADAPTERS: tl.constexpr    -- adapter count, loop is fully unrolled
RANK: tl.constexpr           -- LoRA rank, determines inner matmul size
BLOCK_M: tl.constexpr = 64  -- seq tile
BLOCK_N: tl.constexpr = 64  -- output feature tile
BLOCK_K: tl.constexpr = 64  -- reduction tile for in_features
```

Grid: `(cdiv(seq_len, BLOCK_M), cdiv(out_features, BLOCK_N), batch_size)`

The kernel implements:
```
out[b, s, d] = sum_i (scale[b, i] * (x[b, s, :] @ A[i].T) @ B[i].T)[d]
```

**Sparse skip mechanism** (lines 110-150):

```python
for adapter_idx in range(N_ADAPTERS):   # unrolled at compile time
    s = tl.load(scale_batch_ptr + adapter_idx)  # scalar, uniform across warps
    if s != 0.0:                                 # UNIFORM branch, zero divergence
        # Phase 1: mid = x_tile @ A_i^T  (tiled reduction over in_features)
        # Phase 2: acc += s * mid @ B_i^T (output tile accumulation)
```

The key insight: `s` is loaded once per (batch_element, adapter) and is the same
for all threads in the program instance. The `if s != 0.0` branch is therefore
**uniform** -- all warps take the same branch. When an adapter's scale is zero for
a given batch element, both matmuls are skipped with zero warp divergence.

This is NOT dynamic sparsity. It is compile-time loop unrolling + runtime uniform
branching. The adapter loop is fully unrolled because `N_ADAPTERS` is `tl.constexpr`,
so each adapter gets its own code path in the compiled PTX. The scale check is a
runtime conditional but uniform across the SM.

**Rank padding** (lines 226-242 in wrapper): If rank < 16, A and B are zero-padded
to the next power of 2 >= 16 (minimum for `tl.dot` on SM89). Rank 8 pads to 16.

#### Custom Op Registration (lines 375-458)

Registered as `futudiffu::multi_lora` via `torch.library.custom_op`:

- **`register_fake`** (lines 392-404): Returns correctly shaped empty tensor for
  dynamo tracing.
- **`register_autograd`** (lines 406-454): Standard linear algebra backward:
  ```
  dB_i = s_i * M_i^T @ grad_out       (recompute M_i = x @ A_i^T)
  dA_i = s_i * (grad_out @ B_i)^T @ x
  dx   = Σ_i s_i * (grad_out @ B_i) @ A_i
  ```
  The backward is implemented in pure PyTorch (not a Triton kernel). It loops
  over adapters, recomputing intermediates. No gradient flows through `scale_all`.

- Fallback (lines 460-470): If custom_op registration fails (older torch), a plain
  function wrapping `multi_lora_forward` is provided.

### 1.3 Target Layer Selection (`src/futudiffu/lora.py` lines 203-232)

The `_find_lora_targets` function scans the model for injection candidates:

**Default targets** (lines 31-37):
```python
DEFAULT_TARGET_SUFFIXES = (
    "attention.qkv",
    "attention.out",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
)
```

This means adapters can be placed on **all 5 linear projections** in each
transformer block: QKV, output projection, and all 3 FFN projections.

**Layer filtering** (lines 220-227):
- If `target_modules` is None (default), candidates must be in `"layers."`,
  `"noise_refiner."`, or `"context_refiner."` -- i.e., all transformer blocks.
- If `layer_indices` is provided, only `layers.{idx}` blocks are targeted.
  For r_theta (BTRM), this is typically `{28, 29}` (last 2 of 32 layers).
  For p_theta (policy), this is all 32 layers.

**Layer scope by adapter type** (from `trainer.py` lines 721-833):
- **r_theta (BTRM reward)**: layers 28-29 only (10 adapters: 5 targets x 2 layers)
- **p_theta (policy)**: all layers (160+ adapters: 5 targets x 32 layers)

### 1.4 How the Diffusion Model Integrates LoRA

The diffusion model (`src/futudiffu/diffusion_model.py`) does NOT explicitly
reference LoRA. Integration happens through module replacement:

1. `inject_lora()` or `allocate_adapter()` walks the model tree via
   `named_modules()` and replaces matching `nn.Linear`/`FP8Linear` with
   `LoRALinear` wrappers.
2. The transformer block's `self.attention.qkv`, `self.attention.out`,
   `self.feed_forward.w1`, etc. are silently replaced.
3. When `layer.forward()` calls `self.attention.qkv(x)`, it now goes through
   `LoRALinear.forward()` which calls `self.base(x)` + `multi_lora_op(...)`.
4. All control (which adapters are active, per-batch routing) flows through
   the `lora_scale` buffer, which is a graph input to torch.compile.

This is clean -- the diffusion model code never imports or references LoRA
directly. No conditional branches in the transformer code.

---

## Section 2: Adapter Lifecycle and Hot-Swap Mechanics

### 2.1 Two-Phase Injection Protocol

The frozen codebase enforces a strict two-phase protocol:

**Phase 1: Allocation** (`allocate_adapter`, lines 235-280)
- Graph-mutating: wraps `Linear`/`FP8Linear` -> `LoRALinear`
- Must happen BEFORE `torch.compile`
- Creates adapter with zero B and scale=0 (silent)
- Modifies `nn.ModuleDict`, changes graph topology

**Phase 2: Weight Init** (`init_adapter_weights`, lines 283-321)
- Graph-invariant: safe after `torch.compile`
- Re-initializes A (kaiming) and B (zeros or normal)
- Sets scale (1.0 = active, 0.0 = silent)
- Only modifies tensor data and buffer values, not graph structure

The legacy `inject_lora()` combines both phases (lines 324-359) but is
deprecated in favor of the split API.

### 2.2 Scale Management

Three functions control adapter activation:

- **`set_lora_scale(model, scale_tensor, adapter_name)`** (lines 381-405):
  Sets the `lora_scale` buffer on all matching adapters. Same-shape updates
  use `copy_()` (no recompilation). Shape changes reassign the buffer (one
  recompilation).

- **`clear_lora_scale(model, adapter_name)`** (lines 408-424):
  Resets to broadcast `(1,)` with value 1.0 (all active, uniform).

- **Freeze/unfreeze** (lines 427-452): `requires_grad_(False/True)` on
  `lora_A` and `lora_B` parameters.

### 2.3 Concurrent Multi-Adapter Training

In `trainer.py` `train_loop()` (lines 703-906):

1. **Phase 1 (BTRM)**: Inject `rtheta` on layers 28-29. Train BTRM head +
   rtheta adapter jointly. Both are in the optimizer.

2. **Transition** (lines 820-823): Freeze rtheta, set scale=0. Freeze BTRM head.

3. **Phase 2 (Policy)**: Inject `ptheta` on ALL layers (on top of existing
   rtheta slots). Both adapters coexist in the same LoRALinear modules.
   ptheta has scale=1.0, rtheta has scale=0.0.

4. **Per-step routing** (line 617): During policy training, batch[0] gets
   ptheta active (scale=1.0), batch[1] gets ptheta off (scale=0.0) for
   reference comparison: `set_lora_scale(model, tensor([1.0, 0.0]))`.

### 2.4 Persistence and Hot-Reload (`model_manager.py`)

`ModelManager` handles adapter survival across model lifecycle swaps
(TE <-> diffusion):

- **`snapshot_lora_weights()`** (lines 69-78): Before destroying the diffusion
  model, extracts all LoRA state dicts to CPU.

- **`ensure_diffusion()`** (lines 154-241): When re-loading the diffusion model:
  1. Loads FP8 weights
  2. Fuses model
  3. Replays all `lora_configs` (re-injects LoRA adapters)
  4. Restores saved weights from CPU cache
  5. Restores scale settings
  6. Recompiles with all adapters in the graph

- **`lora_configs: list[dict]`** (line 53): Injection configs stored for replay.
  Contains adapter_name, rank, alpha, layer_indices, init_b_std.

- **`lora_weights: dict[str, dict[str, Tensor]]`** (line 54): CPU weight cache.

- **`lora_scales: dict[str, float | list[float]]`** (line 55): Last-set scale.

### 2.5 Emergency Dump (`lora.py` lines 551-624)

`dump_all_loras()` writes each adapter as a separate safetensors file plus a
JSON manifest. Optionally includes the BTRM head. Used for:
- Checkpoint saves during training
- Crash recovery
- Phase transitions

### 2.6 Production Training Flow (`scripts/train.py`)

The frozen `train.py` is a ZMQ client that orchestrates adapter lifecycle via
RPC calls to the server:

1. **Pre-allocate all adapter slots** (lines 896-908):
   ```python
   client.allocate_adapter("rtheta", rank=8, alpha=16.0, layer_indices=[28, 29])
   client.allocate_adapter("ptheta", rank=8, alpha=16.0)  # all layers
   ```

2. **Single compile** (lines 910-919): Warmup after all slots are allocated.
   No recompilation during training.

3. **Initialize rtheta** (line 922): `init_adapter_weights("rtheta", init_b_std=0.0, scale=1.0)`

4. **Phase 1 BTRM** (lines 930-985): Train using `client.train_btrm_step()`.
   Score trajectories from disk with BTRM head + rtheta adapter.

5. **Phase transition** (line 466-470 in `phase_policy`):
   ```python
   client.set_adapter_config("rtheta", frozen=True, scale=0.0)
   client.init_adapter_weights("ptheta", init_b_std=0.01, scale=1.0)
   ```

6. **Phase 2 Policy** (lines 592-608): Generate rollouts, compute advantages,
   accumulate REINFORCE gradients via `client.accumulate_policy_gradients()`,
   then `client.policy_optimizer_step()`.

7. **Phase 3 Final** (lines 711-721): `client.dump_all_loras()`.

### 2.7 Gradient Flow Architecture

The `training_utils.py` module provides two forward modes:

- **`forward_checkpointed()`** (lines 228-323): Embedding + refiners under
  `no_grad`, 30 main layers with per-block gradient checkpointing, returns
  `(-img, last_hidden)`. Used for BTRM training where LoRA gradients flow
  through the main layers.

- **`forward_no_grad()`** (lines 326-401): Entire forward under `no_grad`.
  Used for reference pass in REINFORCE.

- **`compute_reinforce_step()`** (lines 408-479): Split B=1 approach:
  1. Reference forward (scale=0, no_grad) -> `ref_denoised`
  2. Policy forward (scale=1, checkpointed) -> `pi_denoised`
  3. `log_ratio = -MSE(pi - ref) / (2 * sigma^2)`
  4. `loss = -advantage * log_ratio`
  5. `loss.backward()` -> gradients accumulate in LoRA params

---

## Section 3: Frozen Import Inventory

### 3.1 src_ii/ Imports from Frozen Code

| File | Line | Import | What It Provides | src_ii Equivalent? |
|------|------|--------|------------------|--------------------|
| `src_ii/btrm_model.py` | 34 | `from futudiffu.btrm import ScoreUnembedder, _RMSNorm` | BTRM scoring head + RMSNorm layer | **No** -- no src_ii copy |
| `src_ii/btrm_model.py` | 35-44 | `from futudiffu.lora import allocate_adapter, freeze_adapter, get_lora_params, init_adapter_weights, load_lora_state_dict, lora_state_dict, set_lora_scale, unfreeze_adapter` | Full LoRA lifecycle API (8 functions) | **No** |
| `src_ii/btrm_model.py` | 45 | `from futudiffu.training_utils import HiddenCapture` | Forward hook for hidden state extraction | **No** |
| `src_ii/btrm_model.py` | 359 | `from futudiffu.diffusion_model import pad_to_patch_size, pad_zimage` | Patchification utilities | **No** |
| `src_ii/btrm_model.py` | 556 | `from futudiffu.diffusion_model import (pad_to_patch_size, pad_zimage, build_packed_sequence, ...)` | Packing + patchification | **No** |
| `src_ii/btrm_model.py` | 663 | `from futudiffu.attention import set_attention_backend` | Global attention backend switch | **No** (src_ii/attention.py takes backend as param) |
| `src_ii/btrm_training.py` | 26 | `from futudiffu.btrm import bradley_terry_loss` | BT pairwise loss function | **No** |
| `src_ii/btrm_training.py` | 652 | `from futudiffu.vae import vae_decode as _vae_decode` | VAE decoder function | `src_ii/vae_utils.py` wraps same |
| `src_ii/attention.py` | 94 | `from futudiffu.sage_attention import sage_attn_masked_op` | Masked SageAttention Triton kernel | **No** -- kernel is in frozen |
| `src_ii/attention.py` | 105 | `from futudiffu.sage_attention import sage_attn_op` | Unmasked SageAttention Triton kernel | **No** -- kernel is in frozen |
| `src_ii/attention_capture.py` | 49 | `import futudiffu.attention as attn_mod` | Monkey-patches sdpa_attention for interception | **No** |
| `src_ii/attention_capture.py` | 133, 136 | `import futudiffu.attention as attn_mod; import futudiffu.diffusion_model as dm_mod` | Module-level monkey-patching | **No** |
| `src_ii/model_loading.py` | 52-58 | `from futudiffu.diffusion_model import _detect_cap_feat_dim, _detect_n_layers, _detect_qk_norm, _strip_diffusion_prefix, create_diffusion_model, fuse_model` | Model creation + FP8 injection | **No** |
| `src_ii/model_loading.py` | 60 | `from futudiffu.fp8 import replace_linear_with_fp8` | FP8 weight replacement | **No** |
| `src_ii/model_loading.py` | 117 | `from futudiffu.sage_attention import configure_sage` | SageAttention config | Wrapper in `src_ii/model_loading.py` itself |
| `src_ii/vae_utils.py` | 35 | `from futudiffu.vae import load_vae as _load_vae` | VAE model loading | **No** (wrapper exists, but delegates) |
| `src_ii/vae_utils.py` | 58 | `from futudiffu.vae import vae_decode as _vae_decode` | VAE decode function | **No** (wrapper exists, but delegates) |
| `src_ii/forward_packed.py` | 149, 192 | `from futudiffu.attention import set_attention_backend` | Global attention backend switch | **No** |
| `src_ii/dataset_catalog.py` | 626 | `from futudiffu.dataset_v2 import DatasetReader` | Parquet-backed dataset reader | **No** |
| `src_ii/dataset_generator.py` | 274 | `from futudiffu.dataset_v2 import DatasetWriter` | Streaming dataset writer | **No** |
| `src_ii/pinkify_validation.py` | 281 | `from futudiffu.vae import vae_encode` | VAE encoder function | **No** |
| `src_ii/server.py` | 750 | `from futudiffu.lora import lora_state_dict` | LoRA weight extraction | **No** |
| `src_ii/server.py` | 768 | `from futudiffu.text_encoder import create_tokenizer, load_text_encoder` | Text encoder loading | **No** |
| `src_ii/server.py` | 805-810 | `from futudiffu.lora import (allocate_adapter, init_adapter_weights, inject_lora, load_lora_state_dict, set_lora_scale, ...)` | Full LoRA lifecycle (6 functions) | **No** |
| `src_ii/server.py` | 851 | `from futudiffu.text_encoder import encode_prompt as _encode_prompt` | Prompt encoding | **No** |
| `src_ii/server.py` | 866-867 | `from futudiffu.attention import set_attention_backend; from futudiffu.sampling import run_trajectory` | Attention + sampling | **No** |
| `src_ii/server.py` | 882-883 | `from futudiffu.attention import set_attention_backend; from futudiffu.sampling import run_trajectory_packed` | Packed trajectory sampling | **No** |
| `src_ii/server.py` | 897, 907 | `from futudiffu.vae import vae_encode, vae_decode` | VAE encode/decode | Wrapper in `src_ii/vae_utils.py` |
| `src_ii/server.py` | 920-921 | `from futudiffu.attention import set_attention_backend; from futudiffu.sampling import warmup_diffusion` | Warmup + attention | **No** |
| `src_ii/server.py` | 931 | `from futudiffu.sampling import warmup_packed` | Packed warmup | **No** |
| `src_ii/server.py` | 941, 974, 990 | `from futudiffu.lora import allocate_adapter, init_adapter_weights, inject_lora` | LoRA injection | **No** |
| `src_ii/server.py` | 1032 | `from futudiffu.lora import load_lora_state_dict` | LoRA weight loading | **No** |
| `src_ii/server.py` | 1051-1056 | `from futudiffu.lora import (clear_lora_scale, freeze_adapter, set_lora_scale, unfreeze_adapter, lora_state_dict, dump_all_loras)` | Scale + freeze + dump | **No** |
| `src_ii/server.py` | 1098-1099 | `from futudiffu.btrm import ScoreUnembedder; from futudiffu.lora import get_lora_params` | BTRM head + LoRA params | **No** |
| `src_ii/server.py` | 1146-1149 | `from futudiffu.attention import set_attention_backend; from futudiffu.training_utils import run_backbone_hidden` | Backbone hidden extraction | **No** |
| `src_ii/server.py` | 1168-1169 | `from futudiffu.attention import set_attention_backend; from futudiffu.training_utils import train_btrm_step` | BTRM training step | **No** |
| `src_ii/server.py` | 1183 | `from futudiffu.training_utils import accumulate_policy_gradients` | REINFORCE gradient accumulation | **No** |
| `src_ii/server.py` | 1192 | `from futudiffu.training_utils import policy_optimizer_step` | Policy optimizer step | **No** |

### 3.2 scripts_ii/ Imports from Frozen Code

| File | Lines | Frozen Imports | What They Provide |
|------|-------|----------------|-------------------|
| `run_flops_budget_100step.py` | 100, 175 | `DatasetReader`, `create_tokenizer, load_text_encoder, encode_prompt` | Dataset + TE |
| `run_flops_budget_100step_v2.py` | 121, 207 | `DatasetReader`, `create_tokenizer, load_text_encoder, encode_prompt` | Dataset + TE |
| `run_funfetti_100step.py` | 92, 153 | `DatasetReader`, `create_tokenizer, load_text_encoder, encode_prompt` | Dataset + TE |
| `run_funfetti_stratified.py` | 101, 177 | `DatasetReader`, `create_tokenizer, load_text_encoder, encode_prompt` | Dataset + TE |
| `verify_pinkify_persistence.py` | 85-86, 148 | `DatasetReader`, `create_tokenizer, load_text_encoder, encode_prompt`, `lora_state_dict` | Dataset + TE + LoRA |
| `train_pinkify_btrm.py` | 92, 116 | `create_tokenizer, load_text_encoder, encode_prompt`, `make_rope_cache` | TE + sampling |
| `run_reward_validated_training.py` | 86, 134, 296, 339 | `vae_encode`, `DatasetReader`, `create_tokenizer, load_text_encoder, encode_prompt` | VAE + dataset + TE |
| `validate_pipeline_multi_res.py` | 108, 127 | `PROMPT_TEMPLATES`, `create_tokenizer, load_text_encoder, encode_prompt` | Dataset templates + TE |
| `sweep_rtheta_hparams.py` | 144, 181, 230, 626-629 | `create_tokenizer, load_text_encoder, encode_prompt`, `make_rope_cache`, `LoRALinear`, `attention + diffusion_model` modules | TE + sampling + LoRA + monkey-patching |
| `validate_trajectory.py` | 192, 219 | `create_tokenizer, load_text_encoder, encode_prompt`, `set_attention_backend` | TE + attention |
| `sweep_rtheta_lr.py` | 174, 208, 292, 707, 867-870 | `create_tokenizer, encode_prompt, load_text_encoder`, `make_rope_cache`, `LoRALinear`, `allocate_adapter, init_adapter_weights`, `attention + diffusion_model` modules | TE + sampling + LoRA lifecycle + monkey-patching |
| `test_funfetti_e2e.py` | 116, 231 | `DatasetReader`, `create_tokenizer, load_text_encoder, encode_prompt` | Dataset + TE |
| `run03_btrm_training.py` | 51-52 | `InferenceClient`, `DatasetReader, DatasetWriter` | ZMQ client + dataset |
| `verify_btrm_persistence.py` | 32 | `ScoreUnembedder` | BTRM head class |
| `run_pinkify_validated_training.py` | 95, 239, 286 | `vae_encode`, `DatasetReader`, `create_tokenizer, load_text_encoder, encode_prompt` | VAE + dataset + TE |
| `validate_packed_vs_serial.py` | 131, 243 | `pack_request, unpack_response`, `InferenceClient` | ZMQ protocol + client |
| `train_pinkify_differentiable.py` | 126, 332 | `DatasetReader`, `create_tokenizer, load_text_encoder, encode_prompt` | Dataset + TE |
| `validate_packed_scoring.py` | 89, 161, 294 | `DatasetReader`, `create_tokenizer, load_text_encoder, encode_prompt`, `set_attention_backend` | Dataset + TE + attention |
| `validate_v2_dataset.py` | 52, 456 | `INDEX_SCHEMA, DatasetReader`, `InferenceClient` | Dataset + ZMQ client |
| `generate_multi_res_trajectories.py` | 218, 353-354, 735, 895, 1150 | `create_tokenizer, load_text_encoder, encode_prompt`, `set_attention_backend, DatasetWriter`, `DatasetReader` | TE + attention + dataset |
| `attention_adapter_diff.py` | 139, 267, 298-299, 314-315 | `LoRALinear`, `create_tokenizer, load_text_encoder, encode_prompt`, `make_rope_cache, set_attention_backend`, `attention + diffusion_model` modules | LoRA + TE + sampling + monkey-patching |
| `merge_v2_datasets.py` | 35 | `INDEX_SCHEMA, _PARQUET_WRITE_KWARGS` | Dataset constants |
| `attention_interpretability.py` | 68, 93-94, 108-109 | `create_tokenizer, load_text_encoder, encode_prompt`, `make_rope_cache, set_attention_backend`, `attention + diffusion_model` modules | TE + sampling + monkey-patching |
| `regenerate_funfetti_charts.py` | 85 | `DatasetReader` | Dataset |
| `compare_pinkify_scores.py` | 197, 218 | `DatasetReader`, `create_tokenizer, load_text_encoder, encode_prompt` | Dataset + TE |
| `generate_btrm_dataset.py` | 65, 236 | `PROMPT_TEMPLATES`, `InferenceClient` | Dataset templates + ZMQ client |

### 3.3 Import Frequency Summary

Counting unique frozen modules imported across all `src_ii/` and `scripts_ii/` files:

| Frozen Module | Import Count | Description |
|---|---|---|
| `futudiffu.lora` | 12 files (31 import lines) | Full LoRA lifecycle -- the heaviest dependency |
| `futudiffu.text_encoder` | 14 files | `create_tokenizer`, `load_text_encoder`, `encode_prompt` |
| `futudiffu.attention` | 9 files | `set_attention_backend` (global mutable state) |
| `futudiffu.btrm` | 4 files | `ScoreUnembedder`, `_RMSNorm`, `bradley_terry_loss` |
| `futudiffu.dataset_v2` | 11 files | `DatasetReader`, `DatasetWriter`, `INDEX_SCHEMA` |
| `futudiffu.vae` | 5 files | `load_vae`, `vae_encode`, `vae_decode` |
| `futudiffu.sampling` | 8 files | `make_rope_cache`, `run_trajectory`, `warmup_diffusion`, etc. |
| `futudiffu.diffusion_model` | 6 files | `NextDiT`, `pad_to_patch_size`, `pad_zimage`, `create_diffusion_model`, etc. |
| `futudiffu.sage_attention` | 3 files | `configure_sage`, `sage_attn_op`, `sage_attn_masked_op` |
| `futudiffu.fp8` | 1 file | `replace_linear_with_fp8` |
| `futudiffu.client` | 3 files | `InferenceClient` (ZMQ) |
| `futudiffu.multi_gpu_client` | 0 src_ii (1 in scripts/) | `MultiGPUClient` (ZMQ) |
| `futudiffu.training_utils` | 3 files | `HiddenCapture`, `run_backbone_hidden`, `train_btrm_step`, etc. |
| `futudiffu.btrm_dataset` | 2 files | `PROMPT_TEMPLATES` |
| `futudiffu.protocol` | 1 file | `pack_request`, `unpack_response` (ZMQ) |

---

## Section 4: Dependency Graph

### 4.1 Import Dependency Layers

The frozen imports form a layered dependency graph. Modules must be replaced
bottom-up:

```
Layer 0 (leaf, no frozen deps):
  - sigma_schedule.py      -- pure math, DONE
  - solver.py              -- pure math, DONE
  - resolution_sampling.py -- pure Python, DONE
  - validation_metrics.py  -- pure Python, DONE
  - block_mask.py          -- pure torch, DONE
  - incremental_save.py    -- pure Python, DONE
  - flops_sampling.py      -- pure Python, DONE
  - bin_packer.py          -- pure Python, DONE
  - pair_sampler.py        -- pure Python, DONE (DatasetReader via comment only)
  - reward_functions.py    -- pure torch, DONE

Layer 1 (depends only on frozen kernels/primitives):
  - attention.py           -- needs sage_attn_op, sage_attn_masked_op
  - vae_utils.py           -- needs load_vae, vae_encode, vae_decode

Layer 2 (depends on frozen model architecture):
  - model_loading.py       -- needs diffusion_model.*, fp8.*, sage_attention.*
  - forward_packed.py      -- needs attention.set_attention_backend
  - attention_capture.py   -- needs futudiffu.attention module object for monkey-patching

Layer 3 (depends on frozen LoRA + BTRM):
  - btrm_model.py          -- needs lora.* (8 functions), btrm.ScoreUnembedder, training_utils.HiddenCapture
  - btrm_training.py       -- needs btrm.bradley_terry_loss, vae.vae_decode
  - dataset_catalog.py     -- needs dataset_v2.DatasetReader
  - dataset_generator.py   -- needs dataset_v2.DatasetWriter

Layer 4 (depends on everything):
  - server.py              -- needs lora.*, btrm.*, attention.*, sampling.*, vae.*, text_encoder.*, training_utils.*
```

### 4.2 Replacement Order

To minimize disruption, replace in this order:

1. **Pure functions** (can be copied verbatim):
   - `bradley_terry_loss` -- 5 lines of torch math
   - `pad_to_patch_size`, `pad_zimage` -- utility functions
   - `PROMPT_TEMPLATES` -- constant list
   - `INDEX_SCHEMA`, `_PARQUET_WRITE_KWARGS` -- constants
   - `HiddenCapture` -- 30-line forward hook class

2. **Self-contained modules** (need vendoring):
   - `ScoreUnembedder` + `_RMSNorm` -- ~100 lines, depends on `rms_norm` kernel
   - `DatasetReader` / `DatasetWriter` -- ~300 lines, parquet + safetensors

3. **Kernel wrappers** (need the Triton kernels):
   - `sage_attn_op`, `sage_attn_masked_op` -- wrappers around Triton custom_ops
   - `configure_sage` -- global config for SageAttention
   - `set_attention_backend` -- global mutable state (should become parameter-based dispatch)

4. **LoRA system** (biggest piece):
   - `LoRAAdapter`, `LoRALinear` -- core data structures
   - `lora_kernels.py` -- Triton kernel + custom_op
   - 8 lifecycle functions
   - State dict extraction/loading

5. **Model loading pipeline** (depends on model architecture staying stable):
   - `create_diffusion_model`, `_detect_*`, `_strip_diffusion_prefix`
   - `replace_linear_with_fp8` -- FP8 weight injection
   - `fuse_model` -- horizontal fusion passes

6. **Text encoder** (self-contained but large):
   - `create_tokenizer`, `load_text_encoder`, `encode_prompt`

7. **Sampling pipeline** (depends on model + attention + LoRA):
   - `make_rope_cache`, `run_trajectory`, `warmup_diffusion`, `run_trajectory_packed`

8. **Training utilities** (depends on everything):
   - `forward_checkpointed`, `forward_no_grad`, `compute_reinforce_step`
   - `run_backbone_hidden`, `train_btrm_step`
   - `accumulate_policy_gradients`, `policy_optimizer_step`

---

## Section 5: Recommendations for src_ii Multi-Tenant LoRA Design

### 5.1 What to Keep from the Frozen Design

The frozen LoRA system has several genuinely good architectural decisions:

1. **Registered buffer for scale**: `lora_scale` as a registered buffer is the
   correct approach for torch.compile compatibility. Keep this.

2. **Two-phase allocation/init**: The split between graph-mutating allocation
   and graph-invariant weight init is essential for compile efficiency. Keep this.

3. **Triton sparse kernel with uniform branching**: The zero-divergence skip
   mechanism is elegant and correct. The kernel should be migrated to src_ii.

4. **Custom_op with register_autograd**: Required for training. The backward
   implementation is straightforward (PyTorch ops) and correct.

5. **Per-batch-element scale tensor**: The `(B,)` scale shape for per-batch
   routing is the right abstraction for concurrent policy/reference batching.

### 5.2 What to Change

1. **Eliminate global mutable state**: The frozen code uses `set_attention_backend()`
   which sets a module-level global. `src_ii/attention.py` already takes `backend`
   as a parameter -- this pattern should extend to all dispatch points. No globals.

2. **Eliminate ModelManager as god object**: The lifecycle management in
   `model_manager.py` conflates model loading, LoRA replay, compilation, and
   VRAM management. In src_ii, these should be separate:
   - Model loading: `src_ii/model_loading.py` (already exists)
   - LoRA lifecycle: new `src_ii/lora.py`
   - Compilation: thin wrapper, not a class
   - VRAM lifecycle: caller responsibility (sequential load/free)

3. **src_ii LoRA should NOT import from frozen**: Currently `src_ii/btrm_model.py`
   imports 8 functions from `futudiffu.lora`. These must all be vendored into
   `src_ii/lora.py`. The Triton kernel should also move.

4. **Consider fusing adapter stacking into the kernel**: Currently, the Python
   forward path (LoRALinear.forward, lines 179-190) does `torch.stack` on every
   call to build `A_all`, `B_all`, `scale_all`. For a fixed adapter set, these
   could be pre-stacked and stored as persistent buffers, eliminating N+2 small
   allocations per forward call.

5. **Eliminate the monkey-patching pattern**: `src_ii/attention_capture.py`
   imports `futudiffu.attention` and `futudiffu.diffusion_model` as module
   objects for monkey-patching. This should be replaced with a proper hook or
   callback mechanism.

### 5.3 Vendoring Plan

The minimum viable vendoring for Tasks #69/#70:

**New file: `src_ii/lora.py`** (~600 lines):
- `LoRAAdapter` class (verbatim from frozen, 50 lines)
- `LoRALinear` class (verbatim, 100 lines)
- `allocate_adapter`, `init_adapter_weights`, `inject_lora` (top-level functions)
- `get_lora_params`, `set_lora_scale`, `clear_lora_scale`
- `freeze_adapter`, `unfreeze_adapter`
- `lora_state_dict`, `load_lora_state_dict`, `count_lora_params`
- `enumerate_adapters`, `dump_all_loras`

**New file: `src_ii/lora_kernels.py`** (~470 lines):
- `_multi_lora_kernel` Triton JIT (verbatim)
- `multi_lora_forward` Python wrapper (verbatim)
- `multi_lora_op` custom_op registration + autograd
- `multi_lora_forward_ref` reference implementation (for testing)

**New file: `src_ii/btrm_primitives.py`** (~80 lines):
- `_RMSNorm` class
- `ScoreUnembedder` class
- `bradley_terry_loss` function
- These currently live in `futudiffu.btrm` and are imported by 4 src_ii files.

**Modify: `src_ii/btrm_model.py`**:
- Change `from futudiffu.btrm import ScoreUnembedder, _RMSNorm` to `from .btrm_primitives import ...`
- Change `from futudiffu.lora import ...` to `from .lora import ...`
- Change `from futudiffu.training_utils import HiddenCapture` to local copy

**Modify: `src_ii/server.py`**:
- The heaviest importer (28 frozen import lines). Change all to src_ii imports.

### 5.4 Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Triton kernel behavioral drift during copy | High | Run `test_multi_lora_kernel()` against both versions |
| Custom_op namespace collision (`futudiffu::multi_lora`) | Medium | Use different namespace (`futudiffu_ii::multi_lora`) |
| FP8Linear attribute forwarding breaks | Medium | Test LoRALinear wrapper with fused FFN chain |
| Backward pass numerical divergence | Medium | Compare gradients from frozen vs src_ii on same inputs |
| ModelManager replay logic lost | Low | Not needed in src_ii (no lifecycle swaps in FastAPI) |
| `set_attention_backend` global state race | Low | Already parameter-based in src_ii/attention.py |

### 5.5 Testing Strategy

For the vendored LoRA:

1. **Kernel equivalence**: Run `test_multi_lora_kernel()` from both frozen and
   src_ii versions on the same GPU. All 7 tests must pass. Cosine > 0.999.

2. **Forward equivalence**: Load same FP8 model, inject same adapter (same seed),
   run same input. Compare outputs from frozen LoRALinear vs src_ii LoRALinear.
   Max abs diff < 1e-4 (BF16 precision).

3. **Backward equivalence**: Same setup, run backward. Compare grad norms on
   lora_A and lora_B from frozen vs src_ii. Relative error < 1%.

4. **Scale routing**: Verify that `set_lora_scale([1, 0])` produces batch[1]
   output identical to base model (no LoRA). Max abs diff = 0.

5. **Lifecycle**: allocate -> compile -> init -> train 10 steps -> save ->
   reload -> verify weights match. This is the minimal e2e test.

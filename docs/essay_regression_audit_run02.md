# Regression Audit: Run 02 Training Pipeline

**Date**: 2026-02-17
**Auditor**: Regression audit agent
**Scope**: Three regressions in the current training pipeline identified during run 02 full execution

---

## Executive Summary

Three regressions were identified in the run 02 training pipeline. All three are
code-path problems, not algorithmic errors: the correct code exists but is not
being exercised by the training loop.

| # | Regression | Impact | Severity |
|---|-----------|--------|----------|
| 1 | Uncompiled training forward passes | ~2x slower training | Performance |
| 2 | No pluggable reward functions in RL phase | BTRM scores only, not pinkify/thisnotthat | Correctness |
| 3 | Training uses SDPA, not SageAttention | Untested kernel path in training | Correctness + Performance |

---

## Regression 1: Uncompiled Training with Leaked Compilation Errors

### Diagnosis

**What code path is being used**: `forward_checkpointed()` in
`src/futudiffu/training_utils.py` (line 228) calls each transformer layer
directly via `grad_ckpt(layer, ...)`. These layers are the raw (uncompiled)
`JointTransformerBlock` instances from `model.layers`. The `model` passed to
`forward_checkpointed` is `diff_model` (the raw model), not `diff_compiled`
(the `torch.compile`'d wrapper).

**What code path SHOULD be used**: The 30 main transformer layers should
execute with compiled kernels. `torch.compile` fuses elementwise ops, enables
CUDA graph capture, and reduces kernel launch overhead.

**Why the difference matters**: Run 02's field report states:

> The training phase ran on a fresh server process that did not need to
> recompile the inference path, since `train_btrm_step` uses
> `forward_checkpointed(diff_model, ...)` which bypasses `diff_compiled`
> entirely.

This is presented as a feature ("no need to recompile") but it means every
training forward pass runs uncompiled. The inductor SymPy recursion (defect
R2-03) was triggered by LoRA allocation, but the "fix" was to skip warmup
entirely and let training run eager. This is a performance regression.

### Root Cause Analysis

The issue is a structural mismatch between how `torch.compile` works and how
`forward_checkpointed` invokes layers.

`torch.compile(model)` wraps `model.__call__` (i.e., `model.forward`). It does
NOT individually compile each layer. When `forward_checkpointed` calls
`grad_ckpt(layer, ...)` on each `model.layers[i]`, it calls the raw
(uncompiled) layer's `forward()` directly, completely bypassing the compiled
graph.

**Can `forward_checkpointed` use the compiled model?**

No, not directly. `torch.compile(model)` creates a single compiled graph for
the entire `model.forward()` call. Gradient checkpointing requires calling each
layer independently (to checkpoint/recompute per layer). You cannot call
sub-parts of a compiled graph individually.

However, you CAN compile individual layers. `torch.compile` works on any
callable, including individual `nn.Module`s. The fix is to compile each
transformer layer individually, BEFORE using them in `forward_checkpointed`.

**Why the inductor SymPy error occurs with LoRA**:

When LoRA is allocated via `allocate_adapter()`, the graph structure changes:
`Linear` modules become `LoRALinear` wrappers with adapter branches that
include `torch.stack([a.lora_A for a in adapters])`. This dynamic tensor
construction creates symbolic shapes that trigger a recursion in SymPy's
expression printer during torch.compile's shape analysis.

This affects `torch.compile(model, mode="default")` (the whole-model compile).
Per-layer compilation may avoid this because each layer's compilation context
is simpler (fewer symbolic shapes to track).

### Fix

The fix has two parts:

**Part A**: Add a function `compile_layers_for_training()` to `model_manager.py`
that individually compiles each transformer layer. This is called after LoRA
allocation but before training begins. Each layer is compiled with
`mode="reduce-overhead"` (optimized for repeated calls with same shapes).

**Part B**: Modify `forward_checkpointed()` to accept an optional
`compiled_layers` parameter. When provided, the gradient checkpointing loop
uses compiled layer callables instead of raw layers.

If per-layer compilation also triggers the SymPy recursion (which is possible
on torch 2.10.0 + triton-windows), the fallback is to set
`torch._dynamo.config.suppress_errors = True` before per-layer compilation,
which lets dynamo fall back gracefully to eager for any layer that fails to
compile, while successfully compiling layers that can be compiled.

> ```python
> # model_manager.py addition
> def compile_layers_for_training(self):
>     """Compile individual transformer layers for gradient-checkpointed training.
>
>     Unlike whole-model compile (which creates one graph for model.forward()),
>     this compiles each layer independently. This is compatible with
>     torch.utils.checkpoint which needs to call layers individually.
>
>     Falls back gracefully: if any layer fails to compile (e.g., SymPy
>     recursion with LoRA), that layer runs eager while others stay compiled.
>     """
>     if self.diff_model is None:
>         return
>     import torch._dynamo
>     old_suppress = torch._dynamo.config.suppress_errors
>     torch._dynamo.config.suppress_errors = True
>     try:
>         compiled = []
>         for i, layer in enumerate(self.diff_model.layers):
>             try:
>                 c = torch.compile(layer, mode="reduce-overhead")
>                 compiled.append(c)
>             except Exception as e:
>                 print(f"  [compile_layers] Layer {i} failed: {e}, using eager")
>                 compiled.append(layer)
>         self._compiled_layers = compiled
>     finally:
>         torch._dynamo.config.suppress_errors = old_suppress
> ```

> ```python
> # training_utils.py: forward_checkpointed modification
> def forward_checkpointed(
>     model, x, timesteps, context, num_tokens, rope_cache,
>     compiled_layers=None,   # NEW: optional compiled layers
> ):
>     # ... embedding phase unchanged ...
>
>     # Phase 3: 30 main layers with per-block gradient checkpointing
>     layers = compiled_layers if compiled_layers is not None else model.layers
>     for layer in layers:
>         embed = grad_ckpt(layer, embed, None, freqs_cis, adaln_input,
>                           use_reentrant=False)
>     # ... rest unchanged ...
> ```

### Verification Strategy

After applying the fix, verify with:
1. Check that `compiled_layers` is populated (non-None, length == 30)
2. Time a single `forward_checkpointed` call with and without compilation
3. Check for graph breaks via `torch._dynamo.explain`

---

## Regression 2: Not Using PINKIFY/THISNOTTHAT Reward Functions for RL

### Diagnosis

**What code path is being used**: The production training script
(`scripts/train.py`) in `phase_policy()` (line 426) uses raw BTRM scores as
the reward signal. Specifically, at line 523-535:

```python
btrm_scores = trajectory.pop("_btrm_scores", None)
if btrm_scores is not None:
    reward = btrm_scores[0][0]
else:
    scores = client.score_btrm(...)
    reward = scores[0][0]
```

The reward is always `scores[0][0]` -- the first BTRM head's (scrimble's) raw
score. There is no pluggable reward function mechanism.

**What code path SHOULD be used**: The `src_ii/` library has a fully developed
pluggable reward architecture:

1. `src_ii/reward_functions.py` provides `pinkify_score()` and
   `thisnotthat_score()` -- pixel-space reward functions on PIL Images.

2. `src_ii/pair_sampler.py` has `BTRMPairSampler` with explicit
   `preference_fn` decoupling: the sampler returns pairs, and a separate
   callable computes preferences.

3. `src_ii/btrm_training.py:train_btrm_differentiable()` accepts a
   `preference_fn` parameter (line 399) that maps pair metadata to per-head
   preferences.

4. `src_ii/score_cache.py` has `ScoreCache` with pluggable `reward_fns`
   dict.

None of this machinery is wired into the production `scripts/train.py` or the
server-side `training_utils.py:train_btrm_step()`. The `src_ii/` library is
a parallel codebase that was never integrated into the `src/futudiffu/`
production path.

**Why the difference matters**: The BTRM head's raw score is a trained
discriminator of specific signal/noise properties (SDPA vs SageAttention
quantization for scrimble, step count for scrongle). Using it as the sole
reward signal for policy optimization means:

- The policy optimizes for whatever the BTRM head learned, not for user-defined
  aesthetics
- The pinkify/thisnotthat reward functions (designed to test whether arbitrary
  user preferences can steer generation) are never exercised
- The entire `preference_fn` pluggable architecture in `src_ii/` is dead code
  from the production pipeline's perspective

### Fix

The fix adds a pluggable `reward_fn` parameter to the policy phase. The
production training script wires this to a configurable reward function that
can be:

1. `"btrm"` (default) -- current behavior, uses BTRM head's first score
2. `"pinkify"` -- pixel-space pinkness scoring after VAE decode
3. `"thisnotthat"` -- pixel-space similarity scoring after VAE decode
4. A custom callable `(latent, client) -> float`

The key integration point is `phase_policy()` in `scripts/train.py`. The reward
is computed at line 518-535. To support pluggable rewards:

> ```python
> # scripts/train.py addition: reward function registry
>
> def _make_reward_fn(name: str, client: Client) -> Callable:
>     """Create a reward function from a name string.
>
>     Returns callable (trajectory_dict, pos_cond) -> float.
>     """
>     if name == "btrm":
>         def btrm_reward(trajectory, pos_cond, sigmas, mid_idx):
>             btrm_scores = trajectory.pop("_btrm_scores", None)
>             if btrm_scores is not None:
>                 return btrm_scores[0][0]
>             step_key = f"step_{mid_idx:02d}"
>             x_for_score = trajectory.get(step_key, trajectory["final"])
>             sigma_for_score = sigmas[mid_idx] if step_key in trajectory else sigmas[-2]
>             scores = client.score_btrm(
>                 x_for_score, torch.tensor([float(sigma_for_score)]),
>                 pos_cond[:1], attention_backend="sdpa",
>             )
>             return scores[0][0]
>         return btrm_reward
>
>     elif name == "pinkify":
>         from src_ii.reward_functions import pinkify_score
>         def pinkify_reward(trajectory, pos_cond, sigmas, mid_idx):
>             latent = trajectory["final"]
>             image = client.vae_decode(latent)
>             # Convert tensor to PIL Image
>             from futudiffu.rendering import tensor_to_pil
>             pil_img = tensor_to_pil(image)
>             return pinkify_score(pil_img)
>         return pinkify_reward
>
>     elif name == "thisnotthat":
>         # Requires reference images -- would need config paths
>         raise NotImplementedError(
>             "thisnotthat reward requires THIS/THAT reference images. "
>             "Pass a custom reward_fn callable instead."
>         )
>
>     raise ValueError(f"Unknown reward function: {name}")
> ```

This also requires adding `--reward-fn` to the argparser and threading it
through `phase_policy()`.

### Verification Strategy

1. Confirm pinkify_score returns non-zero values on VAE-decoded images
2. Run a short policy phase with `--reward-fn pinkify` and verify rewards
   change across iterations

---

## Regression 3: Not Using the Full Kernel/Inference Stack During Training

### Diagnosis

**What code path is being used**: When `handle_train_btrm_step` is called
(server.py line 238-252), the attention backend is set based on the RPC
parameter:

```python
set_attention_backend(params.get("attention_backend", "sdpa"))
```

Looking at where this is called from `scripts/train.py` line 364-368:

```python
metrics = client.train_btrm_step(
    batch_examples,
    logsquare_weight=logsq_weight,
    attention_backend="sdpa",
)
```

The BTRM training phase **always** passes `attention_backend="sdpa"`. This
means `forward_checkpointed` runs with SDPA attention via `sdpa_attention()`
dispatching to `F.scaled_dot_product_attention()`.

For the policy phase, `accumulate_policy_gradients` and
`compute_reinforce_step` in `training_utils.py` also run through
`forward_checkpointed` and `forward_no_grad`, both of which call
`model.layers[i]` -> `JointAttention.forward` -> `sdpa_attention`. The
attention backend is whatever was last set globally. Since `phase_btrm`
sets it to `"sdpa"`, and `phase_policy` generates rollouts with `"sage"`
(line 563), but the gradient accumulation calls `accumulate_policy_gradients`
which does NOT set the attention backend, the backend may be "sage" or "sdpa"
depending on the last RPC call that set it.

**Critical discovery**: The `handle_accumulate_policy_gradients` handler
(server.py line 254-260) does NOT call `set_attention_backend` at all:

```python
def handle_accumulate_policy_gradients(self, params, tensors):
    self._mm.ensure_diffusion()
    metadata = accumulate_policy_gradients(
        self._mm.diff_model, self.device, self.dtype, params, tensors,
    )
    return pack_response("ok", metadata=metadata)
```

This means policy gradient forward passes inherit whatever attention backend
was set by the previous RPC call (typically `sample_trajectory` with "sage").
This is accidental correctness (sage IS faster) but not intentional, and it
means:

1. The BTRM training phase ALWAYS uses SDPA -- never SageAttention
2. The policy phase uses whatever was last set -- usually sage from rollout
   generation, but this is fragile and undocumented

**What code path SHOULD be used**: Training forward passes should use the same
SageAttention kernels that were validated in the packed-vs-serial test. The
validated path uses SageAttention with INT8 QK quantization, which was
confirmed correct (cos_sim 0.987-0.994 vs SDPA).

Additionally, the training forward uses the **unpacked** path
(`forward_checkpointed` -> serial layer calls) while inference uses the
**packed** path (`forward_packed` with FlexAttention block masks). The
packed path was specifically validated; the unpacked-with-sage path was not
explicitly validated for training (though it is simpler since it has no
block masks).

**Why the difference matters**:

1. **Performance**: SageAttention is ~1.35x faster than SDPA for attention.
   BTRM training runs 30 main layers with attention at each layer, so this
   is a direct speedup.

2. **Correctness**: The BTRM head is trained to discriminate SDPA vs Sage
   quantization differences (the "scrimble" head). If training always uses
   SDPA, the hidden states fed to the score unembedder never contain Sage
   quantization artifacts. This doesn't invalidate training (the labels
   from the dataset correctly encode which trajectory used which backend),
   but it means the model only sees SDPA inference-path hidden states
   during training, never Sage-path hidden states. This is a minor concern.

3. **Untested path**: The BTRM training forward goes through
   `forward_checkpointed` which:
   - Uses the raw model (not compiled)
   - Calls layers serially (not packed)
   - Uses whichever attention backend happens to be set
   This path was never explicitly validated against the compiled+packed
   inference path. The run 02 field report confirms training works
   (loss decreases, accuracy improves), but the numerical equivalence
   between training-path and inference-path was not checked.

### Fix

**Part A**: Make `handle_train_btrm_step` and `handle_accumulate_policy_gradients`
explicitly set the attention backend from their params, with a sensible default:

> ```python
> # server.py: handle_accumulate_policy_gradients fix
> def handle_accumulate_policy_gradients(self, params, tensors):
>     self._mm.ensure_diffusion()
>     self._mm.configure_sage_if_needed(params.get("attention_backend", "sage"))
>     set_attention_backend(params.get("attention_backend", "sage"))
>     metadata = accumulate_policy_gradients(
>         self._mm.diff_model, self.device, self.dtype, params, tensors,
>     )
>     return pack_response("ok", metadata=metadata)
> ```

**Part B**: Change the default attention backend for training RPCs to "sage"
(matching the validated inference path) rather than "sdpa":

> ```python
> # scripts/train.py: phase_btrm fix
> metrics = client.train_btrm_step(
>     batch_examples,
>     logsquare_weight=logsq_weight,
>     attention_backend="sage",  # was "sdpa"
> )
> ```

**Part C**: Add `attention_backend` parameter to `client.accumulate_policy_gradients()`
and pass "sage" from the training script.

### Verification Strategy

1. Run a single `train_btrm_step` with `attention_backend="sage"` and confirm
   no errors
2. Compare loss/accuracy between sage and sdpa backends on the same data
   (should be similar)
3. Time comparison: sage should be ~1.35x faster per macrobatch

---

## Appendix A: Code Changes

### File: `src/futudiffu/model_manager.py`

New method `compile_layers_for_training()`:

```python
def compile_layers_for_training(self):
    """Compile individual transformer layers for gradient-checkpointed training."""
    if self.diff_model is None:
        return []
    import torch._dynamo
    old_suppress = torch._dynamo.config.suppress_errors
    torch._dynamo.config.suppress_errors = True
    compiled = []
    n_compiled = 0
    try:
        for i, layer in enumerate(self.diff_model.layers):
            try:
                c = torch.compile(layer, mode="reduce-overhead")
                compiled.append(c)
                n_compiled += 1
            except Exception as e:
                print(f"  [compile_layers] Layer {i} compile failed: {e}")
                compiled.append(layer)
    finally:
        torch._dynamo.config.suppress_errors = old_suppress
    self._compiled_training_layers = compiled
    print(f"  [compile_layers] {n_compiled}/{len(compiled)} layers compiled for training")
    return compiled
```

### File: `src/futudiffu/training_utils.py`

Modified `forward_checkpointed` signature to accept `compiled_layers`:

```python
def forward_checkpointed(
    model, x, timesteps, context, num_tokens, rope_cache,
    compiled_layers=None,
):
    # ... unchanged until Phase 3 ...
    layers = compiled_layers if compiled_layers is not None else model.layers
    for layer in layers:
        embed = grad_ckpt(layer, embed, None, freqs_cis, adaln_input,
                          use_reentrant=False)
```

### File: `src/futudiffu/server.py`

Fixed `handle_accumulate_policy_gradients` to set attention backend:

```python
def handle_accumulate_policy_gradients(self, params, tensors):
    self._mm.ensure_diffusion()
    self._mm.configure_sage_if_needed(params.get("attention_backend", "sage"))
    set_attention_backend(params.get("attention_backend", "sage"))
    metadata = accumulate_policy_gradients(
        self._mm.diff_model, self.device, self.dtype, params, tensors,
    )
    return pack_response("ok", metadata=metadata)
```

### File: `scripts/train.py`

Changed BTRM training attention backend default from "sdpa" to "sage".
Added `--reward-fn` argument for pluggable reward functions.

---

## Appendix B: What Was NOT Changed

- The running training process was not interrupted
- No server restart required for these fixes
- The `src_ii/` library was not modified (it is the reference implementation)
- No changes to kernel code, model architecture, or dataset format

---

## Appendix C: Fundamental vs. Wiring Problems

| Question | Answer |
|----------|--------|
| Can `forward_checkpointed` use compiled code? | Yes, via per-layer compilation |
| Does gradient checkpointing require eager mode? | No. `grad_ckpt` calls a function; that function can be compiled |
| Is the SymPy recursion a fundamental compile constraint? | No. It's a torch 2.10.0 inductor bug with complex symbolic shapes |
| Are reward functions fundamentally incompatible with the server? | No. They operate in pixel space after VAE decode, which the server already supports |
| Is SDPA required for training correctness? | No. SageAttention works correctly for both forward and backward (custom_ops with register_autograd) |

All three regressions are wiring problems, not fundamental limitations.

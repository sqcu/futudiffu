# Spec: ZImageRLAIF Model Class

**Date**: 2026-02-20
**Supersedes**: The 5-way forward path mess documented in
`docs/review_btrm_model_forward_duplication.md`

## What this is

A single model class that IS the diffusion model. Not a wrapper, not a
coordinator, not a compound object. It returns `(diffusion_field, scores)`
from every forward call. Always. No branches. No modes.

## The two outputs

1. **diffusion_field**: `(B, C, H, W)` — the denoised image prediction.
   Same as current `NextDiT.forward()` returns (negated).

2. **scores**: `(B, n_heads)` — scalar scores from a linear projection
   against the same hidden state that final_layer uses. When head weights
   are zero-initialized (untrained), scores are `(0, 0, ...)`. When head
   weights are trained, scores are meaningful BTRM preference predictions.

## The head

```python
self.score_proj = nn.Linear(hidden_dim, n_score_heads, bias=False)
# zero-init: untrained model returns zeros, no graph break vs trained
nn.init.zeros_(self.score_proj.weight)
```

The head runs the same matmul whether weights are zeros or trained.
No `if has_head`. No `if training`. The projection is part of the model
definition. `torch.compile` sees one graph always.

When loading pretrained weights that include score_proj weights, they
load normally via `load_state_dict`. When loading weights that DON'T
include score_proj (legacy checkpoints), `strict=False` leaves the
zero-init in place. No special handling.

## The forward path

ONE forward. The packed forward with block masks. A single image is a
packed batch of size 1 with a trivial (all-ones) block mask.

```
Input: packed sequence (B, total_len, dim), block_mask, packed_rope, adaln_input

Phase 1: Embedding + refiners (per-image, then pack)
Phase 2: 30 main transformer layers (compiled, gradient-checkpointed when training)
Phase 3a: final_layer → unpatchify → diffusion_field
Phase 3b: mean_pool → score_proj → scores

Return: (diffusion_field, scores)
```

Phase 3a and 3b share the same hidden state (output of layers[-1]).
They are two independent projections from the same representation.
No hooks. No capture objects. No "extract hidden then score" indirection.

## What callers look like

```python
# Inference (sampling loop)
field, _ = model(x, timesteps, context, ...)
# or equivalently
field, scores = model(x, timesteps, context, ...)

# BTRM training (scoring pairs)
field, scores = model(packed_images, ...)
# scores has grad_fn through score_proj AND LoRA adapters

# Eval
with torch.no_grad():
    _, scores = model(latent, timestep, cond, ...)
```

No `extract_hidden()`. No `score_differentiable()`. No
`score_differentiable_packed()`. No `extract_hidden_differentiable()`.
One call signature. One return type.

## Gradient checkpointing

A model attribute, not a caller argument:

```python
model.gradient_checkpointing = True  # enables per-layer grad_ckpt
```

When True, each of the 30 main layers is wrapped in
`torch.utils.checkpoint`. This is set once during training setup.
The forward path checks `self.gradient_checkpointing` to decide
whether to wrap layer calls. This is a `register_buffer` bool or
a simple attribute that does NOT cause graph breaks (the branching
happens at trace time, not runtime — torch.compile resolves it
statically because the attribute doesn't change between calls).

## Compiled layers

Installed INTO `self.layers` directly. Not a side list. Not a
`_compiled_layers` attribute on a wrapper. The ModuleList entries
ARE the compiled wrappers. Both training and inference use the
same compiled layers. No "eval runs eager, training runs compiled"
split.

## LoRA adapters

Allocated on the model directly (as they already are via
`allocate_adapter()`). The model IS the thing that has adapters.
Not a wrapper that "owns" a backbone and "manages" adapters.

## What gets deleted

From `btrm_model.py`:
- `HiddenCapture` class (hook-based hidden extraction)
- `BTRMCompoundModel` class (wrapper/coordinator)
- `extract_hidden()` (inference hidden extraction)
- `extract_hidden_differentiable()` (100-line forward reimplementation)
- `score_differentiable()` (wrapper)
- `score_differentiable_packed()` (213-line packed forward reimplementation)
- `_compile_layers_for_training()` (side-list compilation)
- `_get_training_layers()` (compiled layer dispatch)

From `diffusion_model.py`:
- `forward()` — replaced by unified packed forward
- The distinction between `forward()` and `forward_packed()` — one method

What remains:
- `ScoreUnembedder` or equivalent (the `score_proj` linear + optional
  RMSNorm + soft_tanh_cap) — but as part of the model definition, not
  a separate class that gets "composed" externally
- `btrm_model.py` shrinks to: LoRA lifecycle helpers, optimizer
  construction, persist/load for adapter + head weights

## What does NOT change

- `src_ii/attention.py` — attention dispatch (sage/sage_masked/sdpa)
- `src_ii/block_mask.py` — uint8 block mask construction
- `src_ii/forward_packed.py` — packing orchestration (prepare + forward)
  This module calls the model's forward. It stays thin.
- `src_ii/lora.py` — LoRA allocation/init/freeze/unfreeze
- `src_ii/fp8.py` — FP8Linear, quantization
- `src_ii/fused_kernels.py` — Triton fused ops
- `src_ii/sage_attention.py` — SageAttention kernels
- `src_ii/btrm_training.py` — training loop (but simplified: calls
  `model(images)` instead of `model.score_differentiable_packed(images)`)

## Batched adaLN

The backbone's `_compute_adaln_params()` precomputes all adaLN modulation
params in one batched GEMM. This optimization stays. The unified forward
calls it. The current BTRM reimplementations skip it (passing
`precomputed_adaln=None` to every layer, causing 30 redundant GEMMs).
The unified forward fixes this for free.

## Per-image timesteps in packed training

Current `forward_packed()` uses one shared timestep for all images.
This is correct for inference (euler stepping) but wrong for BTRM
training (different sigmas per image in a pair).

The fix: `adaln_input` becomes per-token, not per-batch. Each image's
tokens in the packed sequence get their own adaln modulation. This
requires `_compute_adaln_params` to handle per-image timesteps
(compute N adaln_inputs, then scatter into the packed sequence layout).

This is a real problem to solve but it's VISIBLE now because there's
one forward path. The current code hides it behind a `pass` statement
and a TODO in a reimplemented forward that nobody reads.

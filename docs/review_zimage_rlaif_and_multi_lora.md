# Code Review: src_ii/zimage_model.py + src_ii/multi_lora.py

**Date**: 2026-02-21
**Spec**: `docs/spec_zimage_rlaif_model.md`
**Frozen reference**: `src/futudiffu/diffusion_model.py`

## Import Verification

All 16 imports from `futudiffu.diffusion_model` verified present with correct
signatures (classes, functions, dataclass attributes). `PackingInfo.document_id`,
`.cap_lens`, `.segments`, `.n_images` all exist. `JointTransformerBlock.forward()`
accepts both `block_mask` and `precomputed_adaln` kwargs.

## CRITICAL Issues

### 1. score_proj zero-init lost on meta device

`zimage_model.py:171`:
```python
nn.init.zeros_(self.score_proj.weight)
```

Model is created with `device="meta"` (line 714 in `create_zimage_rlaif`).
`nn.init.zeros_()` on a meta tensor is a no-op (no storage). After
`model.to(device)` in `load_zimage_rlaif` (line 797), the weight materializes
as **uninitialized garbage**, not zeros. An "untrained" model would return
random scores instead of zeros, violating the spec's "total functional
equivalence" guarantee.

**Fix**: Re-zero after materialization in `load_zimage_rlaif()`:
```python
model = model.to(device)
model.score_proj.weight.data.zero_()
```

### 2. _compute_scores gradient disconnection

`zimage_model.py:416`:
```python
scores = hidden.new_zeros(n_images, self.n_score_heads)
```

From MEMORY.md: "`Tensor.new_zeros()` does NOT inherit requires_grad."
This creates a leaf tensor with no grad_fn. The subsequent in-place
assignment `scores[i] = ...` (line 435) writes into a detached tensor.
Gradients will NOT flow through score_proj to the LoRA adapters.
BTRM training would see zero gradients for the score head.

**Fix**: Collect scores in a list and stack:
```python
score_list = []
for i in range(n_images):
    ...
    score_i = self.score_cap * torch.tanh(raw[0] / self.score_cap)
    score_list.append(score_i)
return torch.stack(score_list)
```

## MODERATE Issues

### 3. forward_packed.py interface mismatch

`forward_packed.py:152` calls `model.forward_packed(...)` but `ZImageRLAIF`
has only `forward()`. The orchestration module needs updating.

Also: `forward_packed.py` passes `timesteps` (singular tensor) but
`ZImageRLAIF.forward()` expects `timesteps_list` (list of tensors).

The spec says forward_packed.py "stays thin" and "calls the model's forward."
It currently calls `.forward_packed()` which doesn't exist on ZImageRLAIF.
Needs a minor update to call `.forward()` with adapted args.

### 4. forward return type changed

Old `forward_packed()` returned `list[Tensor]` (just diffusion fields).
New `forward()` returns `tuple[list[Tensor], Tensor]` (fields + scores).
All callers that unpack the old return need updating:
- `forward_packed.py:packed_forward()`
- `forward_packed.py:make_packed_model_fn()`
- Sample euler loops
- BTRM training loop

## MINOR Issues

### 5. Dead imports

`zimage_model.py:48` imports `attention` and `select_backend` from
`src_ii.attention` but neither is used. The model delegates to
`JointTransformerBlock` which handles attention routing internally
through `futudiffu.attention.sdpa_attention`. Remove dead imports.

### 6. Comment inaccuracy

Line 557 says "30 main transformer layers" but default `n_layers=32`.
The actual Z-Image architecture has 32 main layers.

### 7. configure_sage_attention not called

Neither `load_zimage_rlaif()` nor the model's forward path calls
`configure_sage_attention()`. The SageAttention kernels need
`qk_quant="int8"`, `pv_quant="bf16"` configured before first use.
This is caller responsibility but should be documented.

## multi_lora.py Review

### Correct

- Zero-init B matrices (lines 101-102): Fresh adapters are silent.
- Kaiming init A matrices (line 99): Correct for down-projection.
- Per-image sparse routing via token_to_image (lines 198-228).
- Dual routing context (explicit args + module attributes).
- install_multi_lora replaces modules in-place (line 364).
- Default targeting: only layers.* (main transformer), excluding
  embedders, refiners, final_layer, score_proj, score_norm.
- save_adapter / load_adapter via safetensors.
- adapter_summary statistics.
- init_adapter_b_weights with std=0.01 (matches known pattern).

### Potential concern

- `_find_target_linears` hasattr check (line 292-298) could match non-linear
  modules that have `weight` and `__call__` (e.g., RMSNormModule). The
  `_is_target_module` / exclude_patterns filters should prevent this in
  practice, but the detection heuristic is loose.

- LoRA adapter device/dtype in install_multi_lora (lines 348-361):
  After `replace_linear_with_fp8`, base modules may have FP8 weights.
  The code correctly uses `torch.bfloat16` for adapter weights regardless
  of base dtype. Good.

- In-place `adapter_out[active_idx] += contribution` (line 228): If two
  adapters have overlapping active indices, this accumulates correctly.

## Architecture Conformance

| Spec requirement | Status |
|---|---|
| Returns (diffusion_fields, scores) always | YES |
| Zero-init score head | **BROKEN** (meta device) |
| No graph breaks trained vs untrained | YES (same matmul always runs) |
| No extract_hidden, no HiddenCapture | YES |
| No BTRMCompoundModel | YES |
| One forward path | YES |
| gradient_checkpointing as attribute | YES |
| Batched adaLN precomputation | YES |
| Compiled layers in self.layers | YES (via torch.compile on whole model) |
| LoRA as MultiLoRALinear wrappers | YES |

## Summary

Two critical bugs must be fixed before any training run:
1. Score head zero-init must happen after materialization (1-line fix)
2. _compute_scores must use list+stack, not new_zeros+assign (5-line fix)

One moderate interface change needed:
3. forward_packed.py must call model.forward() instead of model.forward_packed()

These are all localized fixes. The overall architecture is correct.

# Code Review: btrm_model.py Forward Path Duplication

**Date**: 2026-02-20
**Files reviewed**: `src_ii/btrm_model.py`, `src_ii/diffusion_model.py`, `src_ii/forward_packed.py`, `src_ii/attention.py`

## Summary

`BTRMCompoundModel` contains two hand-rolled reimplementations of the backbone's
forward pass. These are not wrappers, adaptors, or thin shims — they are
full copies of the embedding → refiner → transformer layer loop, maintained
in parallel with the backbone's own `forward()` and `forward_packed()`.

## The four forward paths that exist today

| Method | Location | Lines | What it does |
|--------|----------|-------|-------------|
| `NextDiT.forward()` | diffusion_model.py:895 | 105 | Single-image inference forward. Batched adaLN, final_layer, unpatchify. |
| `NextDiT.forward_packed()` | diffusion_model.py:1098 | 138 | Multi-image packed inference forward. Per-image noise refiner, packed layers, final_layer, unpatchify. |
| `extract_hidden_differentiable()` | btrm_model.py:444 | 100 | **Reimplements** single-image forward for training. Skips final_layer. |
| `score_differentiable_packed()` | btrm_model.py:628 | 213 | **Reimplements** packed forward for training. Skips final_layer, adds per-image scoring. |

Total: ~450 lines of forward logic in btrm_model.py that duplicate ~240 lines
in diffusion_model.py. The duplication is not mechanical (copy-paste) — each
copy has evolved independently, introducing divergences.

## Concrete divergences between backbone and BTRM reimplementations

### 1. Missing batched adaLN (performance)

Backbone calls `_compute_adaln_params(adaln_input)` — one batched GEMM that
precomputes (scale_msa, gate_msa, scale_mlp, gate_mlp) for ALL 34 layers at once.
Result is passed as `precomputed_adaln` to each layer.

Both BTRM methods pass `precomputed_adaln=None`. Each of the 30 layers recomputes
its own adaLN modulation independently: 30 separate GEMMs instead of 1 batched GEMM.
This wastes compute and creates 30 sets of intermediate tensors.

### 2. Timestep handling in packed path (correctness)

`score_differentiable_packed()` lines 700-713: Only the FIRST image's timestep
embedding is used as `adaln_input` for ALL images in the packed batch. The code
contains a `pass` statement and a TODO comment acknowledging this is wrong.

`NextDiT.forward_packed()` has the same limitation (one shared `timesteps` tensor),
but at least that's designed for inference where all images share the same sigma
during euler stepping. For BTRM training, images in a pair have DIFFERENT sigmas.
The reimplementation inherited a design constraint from inference that doesn't
apply to training, and then made it worse by silently discarding per-image timesteps.

### 3. Compiled layers not used for eval (VRAM)

Compiled layers live in `self._compiled_layers` (a side list on BTRMCompoundModel).
The backbone's `self.layers` remains uncompiled.

- `extract_hidden()` → calls `self.backbone()` → uses uncompiled `self.layers` → **eager eval**
- `extract_hidden_differentiable()` → calls `_get_training_layers()` → uses compiled layers
- `score_differentiable_packed()` → calls `_get_training_layers()` → uses compiled layers

Every eval pass (initial eval, every-10-step validation callback) runs the full
30-layer backbone in eager mode. Eager mode uses ~14GB activations vs ~8GB compiled.
The eval VRAM spike may not fully reclaim due to CUDA memory fragmentation.

### 4. Attention backend configuration (correctness/VRAM)

`score_differentiable_packed()` correctly calls `set_attention_backend("sage")`
at line 797. But `extract_hidden_differentiable()` does NOT set the backend at all.
It relies on whatever global `_ATTENTION_BACKEND` happens to be set.

`extract_hidden()` (the eval path) also doesn't set the backend. If the first
eval runs before the first training step, `_ATTENTION_BACKEND` is still `"sdpa"`
(the default in attention.py:57). All eval attention goes through PyTorch SDPA
instead of SageAttention.

The training script (`run_reward_validated_training.py`) never calls
`configure_sage_attention()` or `set_attention_backend()` at the script level.
(I added these to the script earlier in this session, but the BTRM model itself
should not depend on external callers remembering to configure global state.)

### 5. SageAttention kernel configuration (performance)

Nobody calls `configure_sage(smooth_k=True, qk_quant="int8", pv_quant="bf16")`
in the training path. The default in `sage_attention.py:33` is `_SAGE_QK_QUANT = "fp8"`.
The canonical training config is INT8 QK + BF16 PV (documented in MEMORY.md).

## The correct architecture (what should exist instead)

The backbone already has the correct forward methods. The BTRM coordinator should
use them, not reimplement them.

**What training needs that inference doesn't:**
1. `torch.no_grad()` on embedding + refiners (frozen, no trainable params)
2. Gradient checkpointing on the 30 main layers
3. Compiled layers instead of raw layers
4. Stop after layers[-1], before final_layer (hidden state extraction)

**How to get these without reimplementing forward:**

Option A — backbone supports training mode:
- Add `return_hidden: bool = False` to `forward()` and `forward_packed()`
- When True, return after the 30 layers, skip final_layer + unpatchify
- Add `gradient_checkpointing: bool = False` flag
- Install compiled layers INTO `backbone.layers` (replace the ModuleList entries)
- BTRM coordinator becomes: configure backbone → call backbone → score hidden

Option B — factor out shared logic:
- Extract embedding + refiners into `_embed_and_refine()`
- Extract the 30-layer loop into `_run_main_layers(embed, freqs, adaln, ...)`
- `forward()` = `_embed_and_refine()` + `_run_main_layers()` + final_layer
- Training = `no_grad(_embed_and_refine())` + `grad_ckpt(_run_main_layers())` + score

Either way, `extract_hidden_differentiable()` and the 213-line packed
reimplementation in `score_differentiable_packed()` get deleted. The BTRM
coordinator goes back to coordinating.

## What `extract_hidden()` gets right

For contrast: `extract_hidden()` (lines 410-442) is correct. It calls
`self.backbone()` and captures hidden states via the HiddenCapture hook.
12 lines. No duplication. The hook fires on `layers[-1]` output, which is
exactly the hidden state needed for scoring. This is what the other methods
should look like.

## Impact on the VRAM regression

The user reported ~15GB peak in pre-refactor runs, now significantly higher.
Contributing factors from this review:

1. **Eager eval** (compiled layers not in backbone.layers): +6GB activation peak during every eval pass
2. **SDPA default** (no sage config): larger N×N attention intermediates
3. **30x redundant adaLN GEMMs** (no precomputed_adaln): minor but wasteful
4. **Step timing variance**: caused by torch.compile recompilation on each new
   total_len. Each unique packed sequence length triggers recompilation of all
   30 compiled layers. With 26 resolutions and variable pairing, many unique
   lengths appear. First encounter = ~90s (compilation), subsequent = ~12s (cached).

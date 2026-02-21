# Attention Dispatch and Packed Forward Path in src_ii/

**Date:** 2026-02-17
**Modules:** `src_ii/block_mask.py`, `src_ii/attention.py`, `src_ii/forward_packed.py`

---

## What Was Built

Three tightly coupled modules that replace the callable-mask-based packed
attention path with direct uint8 SageAttention masked kernels.

| Module | Responsibility |
|--------|---------------|
| `block_mask.py` | Constructs uint8 block-diagonal masks from segment lengths |
| `attention.py` | Routes Q, K, V to the correct kernel based on a string backend |
| `forward_packed.py` | Orchestrates mask construction + backend config + model forward |

Together they close the integration gap identified in
`docs/essay_synthesis_sage_integration_and_src_ii_gap.md`: call sites
constructed callable mask objects (wrong type) instead of uint8 tensors
(right type for Sage dispatch). These modules make uint8 the only mask
type that exists in the src_ii/ layer.

---

## Design Decisions

### 1. Delegation vs. Replacement for the Model Forward

The first design question: should `forward_packed.py` re-implement the packed
forward path (extracting model internals), or should it delegate to the
model's existing `forward_packed()` method?

**Decision: Delegate.**

The model's `forward_packed()` already threads `block_mask` through all
transformer layers without type-checking it. The model's internal
`sdpa_attention()` already handles uint8 block masks when the attention
backend is set to "sage" or "auto". The only missing piece was that callers
were constructing the wrong mask type. Fixing the mask construction at the
call site is simpler and less fragile than re-implementing 200+ lines of
patchification, refiner execution, packing, RoPE computation, and unpacking.

The src/ codebase is frozen, so we cannot modify `forward_packed()` to
accept only uint8. But we don't need to -- it already works with uint8.
We just need to ensure callers pass the right type, which is what
`prepare_packed_forward()` guarantees.

### 2. String-Based Backend Selection (Not Runtime Type Dispatch)

The attention dispatch in `src_ii/attention.py` uses a `Literal["sage",
"sage_masked", "sdpa"]` backend parameter, not runtime inspection of
tensor types or values.

**Why this matters for torch.compile:**

torch.compile traces Python code and specializes on values known at trace
time. A branch like `if backend == "sage_masked":` with a string literal
is resolved at trace time -- the other branches are dead code. A branch
like `if block_mask is not None:` on a tensor requires a guard that checks
at every invocation, potentially causing graph breaks.

The `select_backend()` helper converts the structural question (is this a
packed sequence?) into a string answer once, before the compiled region.
Inside the compiled forward, the string is a static constant.

### 3. Block Mask Construction: Outer Product Equality

The mask construction in `build_block_mask()` uses a two-step approach:

1. Build a per-token `doc_ids` array mapping token index to image index
2. Sample the document ID at each block boundary
3. Compute `(q_doc.unsqueeze(1) == kv_doc.unsqueeze(0)).to(torch.uint8)`

This is a pure tensor operation -- no Python loops over sequence positions.
The only Python loop is over the number of packed images (typically 2-6)
to fill the `doc_ids` array with segment assignments. This loop is
executed outside the compiled graph (mask construction happens before
the forward pass) and its iteration count is small and fixed.

**Block alignment correctness:** The mask samples document IDs at block
boundaries (multiples of BLOCK_M=128 for Q, BLOCK_N=64 for KV). This
is correct because `pad_tokens_multiple=32` divides both block sizes,
ensuring no block spans two images. The first token of each block
determines the entire block's document ownership.

### 4. Separation of Concerns: prepare vs. forward

`prepare_packed_forward()` computes everything that is constant across
euler steps:
- Refined caption embeddings (context_refiner output)
- Packing layout (segment offsets, document IDs)
- Packed RoPE frequencies
- uint8 block mask
- Backend selection

`packed_forward()` is called per step and does only:
- Set the attention backend (if `ensure_sage_backend=True`)
- Delegate to `model.forward_packed()`

This matches the existing pattern: `prepare_rope_cache()` computes
constants, `model.forward()` consumes them per step.

### 5. The `ensure_sage_backend` Flag

The model's internal `sdpa_attention()` checks a global `_ATTENTION_BACKEND`
variable. For uint8 masks to route to Sage, this must be "sage" or "auto".
The `ensure_sage_backend` parameter (default True) calls
`set_attention_backend("sage")` before each forward.

This is a global mutation, which is undesirable in general. But it matches
the existing pattern in the codebase (the server sets the backend once at
startup). The flag can be set to False when the caller knows the backend is
already configured.

### 6. No Monkey-Patching

An alternative design would monkey-patch the model's attention calls to use
`src_ii/attention.py` directly. This was rejected because:
- The model's `sdpa_attention()` already handles uint8 + sage correctly
- Monkey-patching breaks under torch.compile (patched function is not the
  compiled function)
- The mask type was the only bug; the dispatch was already correct

---

## Module Details

### `src_ii/block_mask.py`

Exports:
- `build_block_mask(segment_lengths, total_len, device)` -- core construction
- `build_block_mask_from_packing_info(packing_info, device)` -- convenience wrapper
- `BLOCK_M`, `BLOCK_N` -- tile size constants (128, 64)

The mask shape is `(n_q_blocks, n_kv_blocks)` -- 2D, shared across all
batch and head dimensions. The Sage kernels accept 2D masks and use
`stride_bm_z=0` to broadcast.

For a typical 2-image pack at 1280x832 + 512x512:
- total_len ~ 4192 + 1088 = 5280
- n_q_blocks = ceil(5280/128) = 42
- n_kv_blocks = ceil(5280/64) = 83
- Mask: 42 x 83 = 3486 bytes

This is negligible memory and constructed once per packing configuration.

### `src_ii/attention.py`

Exports:
- `attention(q, k, v, backend, block_mask, sm_scale)` -- unified dispatch
- `select_backend(is_packed, force_sdpa)` -- backend selection helper
- `AttentionBackend` -- Literal type alias

The dispatch is a simple if-chain on the backend string. Each branch
validates that block_mask is consistent with the backend (present for
"sage_masked", absent for others) and raises ValueError on mismatch.
This catches the exact class of bug that motivated this work: passing
a mask to an unmasked backend, which would silently compute cross-image
attention.

### `src_ii/forward_packed.py`

Exports:
- `prepare_packed_forward(model, context_list, img_sizes, cap_lens, device, force_sdpa)` -- prepare constants
- `packed_forward(model, x_list, timesteps, refined_caps, packing_info, block_mask, packed_rope, ensure_sage_backend)` -- per-step forward
- `make_packed_model_fn(model, refined_caps, packing_info, block_mask, packed_rope, cfg, multiplier, ensure_sage_backend)` -- create a callable for sample_euler_packed
- `prepare_and_build_mask(segment_lengths, total_len, device)` -- convenience re-export

---

## Lifecycle Axis Compliance

| Axis | Requirement | How Met |
|------|-------------|---------|
| 6: Activation checkpointing | Forward must support gradient retention | sage_attn_masked_op has register_autograd; model layers are standard nn.Module |
| 7: Rollout-training coupling | Inference path must be training-compatible | Same forward path works under torch.no_grad (inference) and with gradients (training) |
| 10: Sequence packing | Mask must isolate documents within packed sequence | uint8 block-diagonal mask with block sizes matching Sage kernel tiles |

---

## What Changed in Existing Files

- `src_ii/block_mask.py`: Docstring updated -- removed references to callable
  mask APIs to comply with naming constraints
- `src_ii/attention.py`: Docstring updated -- same reason

No modifications to any file in `src/futudiffu/` (frozen).

---

## Appendix A: Block Mask Construction

> ```python
> # From src_ii/block_mask.py
> def build_block_mask(
>     segment_lengths: list[int],
>     total_len: int | None = None,
>     device: torch.device | str = "cuda",
> ) -> torch.Tensor:
>     if total_len is None:
>         total_len = sum(segment_lengths)
>
>     n_q_blocks = _ceildiv(total_len, BLOCK_M)
>     n_kv_blocks = _ceildiv(total_len, BLOCK_N)
>
>     doc_ids = torch.empty(total_len, dtype=torch.int32, device=device)
>     offset = 0
>     for img_idx, seg_len in enumerate(segment_lengths):
>         doc_ids[offset:offset + seg_len] = img_idx
>         offset += seg_len
>
>     q_block_starts = torch.arange(n_q_blocks, device=device) * BLOCK_M
>     q_block_starts = q_block_starts.clamp(max=total_len - 1)
>     q_doc = doc_ids[q_block_starts]
>
>     kv_block_starts = torch.arange(n_kv_blocks, device=device) * BLOCK_N
>     kv_block_starts = kv_block_starts.clamp(max=total_len - 1)
>     kv_doc = doc_ids[kv_block_starts]
>
>     mask = (q_doc.unsqueeze(1) == kv_doc.unsqueeze(0)).to(torch.uint8)
>     return mask
> ```

## Appendix B: Attention Dispatch

> ```python
> # From src_ii/attention.py
> def attention(
>     q: Tensor, k: Tensor, v: Tensor,
>     backend: AttentionBackend = "sage",
>     block_mask: Tensor | None = None,
>     sm_scale: float | None = None,
> ) -> Tensor:
>     B, H, N, D = q.shape
>     if sm_scale is None:
>         sm_scale = 1.0 / math.sqrt(D)
>
>     if backend == "sage_masked":
>         from futudiffu.sage_attention import sage_attn_masked_op
>         out, _lse = sage_attn_masked_op(q, k, v, block_mask, sm_scale)
>         return out
>
>     if backend == "sage":
>         from futudiffu.sage_attention import sage_attn_op
>         out, _lse = sage_attn_op(q, k, v, sm_scale)
>         return out
>
>     if backend == "sdpa":
>         out = F.scaled_dot_product_attention(
>             q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
>         )
>         return out
> ```

## Appendix C: Packed Forward Orchestration

> ```python
> # From src_ii/forward_packed.py
> def prepare_packed_forward(model, context_list, img_sizes, cap_lens,
>                            device, force_sdpa=False):
>     refined_caps, packing_info, packed_rope = model.prepare_packed_state(
>         context_list, img_sizes, cap_lens, device,
>     )
>     block_mask = build_block_mask_from_packing_info(packing_info, device=device)
>     is_packed = packing_info.n_images > 1
>     backend = select_backend(is_packed=is_packed, force_sdpa=force_sdpa)
>     return {
>         'refined_caps': refined_caps,
>         'packing_info': packing_info,
>         'packed_rope': packed_rope,
>         'block_mask': block_mask,
>         'backend': backend,
>     }
>
> def packed_forward(model, x_list, timesteps, refined_caps, packing_info,
>                    block_mask, packed_rope, ensure_sage_backend=True):
>     if ensure_sage_backend:
>         from futudiffu.attention import set_attention_backend
>         set_attention_backend("sage")
>     return model.forward_packed(
>         x_list, timesteps, refined_caps,
>         packing_info, block_mask, packed_rope,
>     )
> ```

# SageAttention + Block Mask Integration Audit

This essay traces the current state of SageAttention and FlexAttention
dispatch within the packed multi-image forward path, identifies the
integration gap, and characterizes the work required to close it.

---

## 1. The Masked SageAttention Kernels Exist

Both FP8 and INT8 variants of the masked SageAttention kernel are fully
implemented in `src/futudiffu/sage_kernels.py`. They accept a
`block_mask_ptr` argument of shape `(BH, n_q_blocks, n_kv_blocks)` as
uint8, where 1 means "compute" and 0 means "skip". A `tl.constexpr`
flag `HAS_BLOCK_MASK` compiles away the masking logic when set to
`False`, ensuring zero overhead for the unpacked case.

FP8 QK variant (line 1171):

> ```python
> @triton.jit
> def _sage_attn_fwd_fp8qk_bf16pv_masked(
>     Q, K, V, Out,
>     block_mask_ptr,  # (BH, n_q_blocks, n_kv_blocks) uint8 or None
>     stride_z,        # stride of batch*head dim (= seq_len * D)
>     stride_bm_z,     # stride of block_mask batch dim (= n_q_blocks * n_kv_blocks)
>     seq_len,
>     n_kv_blocks,     # number of KV blocks (for block_mask row stride)
>     sm_scale_log2e,
>     FP8_MAX: tl.constexpr,
>     BLOCK_M: tl.constexpr,
>     BLOCK_N: tl.constexpr,
>     D: tl.constexpr,
>     HAS_BLOCK_MASK: tl.constexpr,
> ):
> ```

INT8 QK variant (line 1271):

> ```python
> @triton.jit
> def _sage_attn_fwd_int8qk_bf16pv_masked(
>     Q, K, V, Out,
>     block_mask_ptr,
>     stride_z,
>     stride_bm_z,
>     seq_len,
>     n_kv_blocks,
>     sm_scale_log2e,
>     BLOCK_M: tl.constexpr,
>     BLOCK_N: tl.constexpr,
>     D: tl.constexpr,
>     HAS_BLOCK_MASK: tl.constexpr,
> ):
> ```

Both kernels implement block-level mask lookup inside the KV inner loop.
When `HAS_BLOCK_MASK=True`, each iteration reads
`block_mask[pid_z, pid_m, n_idx]` and skips the entire K/V tile when
the entry is zero:

> ```python
> if HAS_BLOCK_MASK:
>     should_compute = tl.load(bm_base + pid_m * n_kv_blocks + n_idx)
>     do_block = should_compute != 0
>
> if do_block:
>     # ... quantize K, compute QK^T, online softmax, PV ...
> ```

A Python-level dispatcher in `src/futudiffu/sage_attention.py` wraps
these kernels as `sage_attn_forward_masked()` (line 183), handling
shape normalization, K-smoothing, and QK quant dispatch (FP8 vs INT8):

> ```python
> def sage_attn_forward_masked(
>     q: Tensor, k: Tensor, v: Tensor,
>     block_mask: Tensor,
>     sm_scale: float | None = None,
> ) -> Tensor:
> ```

Conclusion: The kernels exist, the API wrapper exists, and the
dispatch between FP8 and INT8 variants is handled.

---

## 2. The Dispatch Logic in `sdpa_attention()` Already Supports Masked Sage

The central attention dispatch function in `src/futudiffu/attention.py`
(line 37) has a three-tier dispatch that already routes to
`sage_attn_forward_masked` under certain conditions:

> ```python
> def sdpa_attention(
>     q: Tensor, k: Tensor, v: Tensor,
>     heads: int, mask: Tensor | None = None,
>     skip_reshape: bool = False,
>     block_mask=None,
> ) -> Tensor:
> ```

The dispatch logic (lines 72-94):

> ```python
> # Dispatch to SageAttention masked kernel when block_mask is a uint8 tensor
> # and the attention backend is sage or auto.
> if block_mask is not None and isinstance(block_mask, Tensor):
>     if block_mask.dtype == torch.uint8 and _ATTENTION_BACKEND in ("sage", "auto"):
>         try:
>             from .sage_attention import sage_attn_forward_masked
>             sm_scale = 1.0 / (dim_head ** 0.5)
>             out = sage_attn_forward_masked(q, k, v, block_mask, sm_scale)
>             out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
>             return out
>         except Exception:
>             if _ATTENTION_BACKEND == "sage":
>                 raise
>             # "auto" mode: fall through to FlexAttention
>
> # FlexAttention dispatch for packed batch inference.
> if block_mask is not None:
>     from torch.nn.attention.flex_attention import flex_attention
>     out = flex_attention(q, k, v, block_mask=block_mask)
>     out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
>     return out
> ```

The dispatch is type-driven:
- If `block_mask` is a `torch.Tensor` with `dtype=torch.uint8` AND the
  backend is `sage` or `auto`, it calls `sage_attn_forward_masked`.
- If `block_mask` is a FlexAttention `BlockMask` object (not a plain
  Tensor), it falls through to `flex_attention()`.
- If the backend is `sdpa`, all packed batches go to FlexAttention
  regardless.

This design means the plumbing inside `attention.py` is complete. The
problem is upstream: what type of `block_mask` actually arrives.

---

## 3. The Gap: `create_block_mask` Produces a `BlockMask`, Not a uint8 Tensor

The packed forward path creates the block mask in
`src/futudiffu/sampling.py` at two call sites: `run_packed_trajectories`
(line 728) and `warmup_packed_forward` (line 815). Both use
FlexAttention's `create_block_mask`:

> ```python
> from torch.nn.attention.flex_attention import create_block_mask
>
> block_mask = create_block_mask(
>     make_packing_mask_mod(packing_info.document_id),
>     B=2, H=None,
>     Q_LEN=packing_info.total_len,
>     KV_LEN=packing_info.total_len,
>     device=device,
> )
> ```

`create_block_mask` returns a `BlockMask` object (an opaque
FlexAttention type), NOT a `torch.Tensor` with `dtype=torch.uint8`.
When this `BlockMask` reaches `sdpa_attention`, it fails the
`isinstance(block_mask, Tensor)` check on line 74, so the SageAttention
masked path is NEVER entered. The FlexAttention path on line 90 catches
it instead.

This is the integration gap. The wiring exists at both ends:
- The kernel end (sage_kernels.py) accepts a uint8 block mask.
- The dispatch end (attention.py) routes uint8 block masks to Sage.
- But the construction site (sampling.py) builds a FlexAttention
  `BlockMask` instead of a uint8 tensor.

---

## 4. What Would Wiring Them In Require?

Three concrete changes are needed:

### 4a. A uint8 block mask constructor

A function that takes a `PackingInfo` (specifically its `document_id`
tensor) and produces a `(n_q_blocks, n_kv_blocks)` uint8 tensor where
entry `[i, j]` is 1 iff any query token in Q-block `i` and any
key token in KV-block `j` share the same `document_id`. This is the
same block-diagonal structure that `make_packing_mask_mod` describes,
but materialized as a flat uint8 tensor rather than a FlexAttention
closure.

Block sizes must match the SageAttention kernel constants: `BLOCK_M=128`
for Q blocks, `BLOCK_N=64` for KV blocks.

### 4b. Conditional construction at the two call sites in sampling.py

At lines 728 and 815, the code currently unconditionally calls
`create_block_mask`. This should branch: if the attention backend is
`sage` or `auto`, construct a uint8 tensor; if `sdpa`, use the existing
FlexAttention `create_block_mask`. Alternatively, always construct the
uint8 tensor and let `sdpa_attention`'s dispatch handle fallback to
FlexAttention (but this would require converting uint8 to BlockMask
for the SDPA path, which is non-trivial).

The cleaner design: always construct the uint8 tensor when the backend
supports Sage, and thread the attention backend decision into the mask
construction site rather than doing it implicitly at dispatch time.

### 4c. Expansion along the BH dimension

The SageAttention masked kernels expect the block mask to be either
`(n_q_blocks, n_kv_blocks)` (broadcast across all B*H) or
`(BH, n_q_blocks, n_kv_blocks)`. Since the document-level mask is
head-independent and the CFG batch (B=2) uses the same packing layout,
the 2D form `(n_q_blocks, n_kv_blocks)` with `stride_bm_z=0` is
sufficient. The existing `sage_attn_forward_masked` already handles this
case.

---

## 5. Is There a Duplicated Code Path?

Yes. There is currently a "FlexAttention for packed, SageAttention for
unpacked" bifurcation. It manifests as follows:

**Unpacked path** (single-image `forward()`): The `block_mask` parameter
is `None` at every layer. Attention dispatch in `sdpa_attention` falls
through to the third tier (line 99):

> ```python
> if mask is None and _ATTENTION_BACKEND in ("sage", "auto"):
>     from .sage_attention import sage_attn_op
>     sm_scale = 1.0 / (dim_head ** 0.5)
>     out, _lse = sage_attn_op(q, k, v, sm_scale)
> ```

This path uses SageAttention (with full backward pass via
`register_autograd`), supporting training.

**Packed path** (`forward_packed()`): A `BlockMask` arrives at every
layer. Attention dispatch hits the FlexAttention tier (line 90):

> ```python
> if block_mask is not None:
>     from torch.nn.attention.flex_attention import flex_attention
>     out = flex_attention(q, k, v, block_mask=block_mask)
> ```

This path cannot use SageAttention because the block mask is the wrong
type. It also loses the INT8/FP8 quantization benefits of Sage, instead
running FlexAttention in BF16 precision.

The result is that the same model, within the same process, uses two
different attention backends depending on whether the forward is packed.
The design doc (Section 8 of `docs/flexattention_batch_packing.md`)
explicitly codified this as intentional:

> "SageAttention custom Triton kernels implement dense attention with no
> masking support. **Incompatible with FlexAttention block masks.**
> Decision: Use FlexAttention for packed batches, keep SageAttention for
> unpacked single-image batches."

This was true at the time of writing. It is now false. The masked
SageAttention kernels (added after the design doc) render Section 8
obsolete. The incompatibility was accidental and has been resolved at the
kernel level, but the wiring has not been updated to reflect this.

---

## 6. What Does `src_ii/` Have?

The `src_ii/` library contains 20+ modules extracted from the main
codebase. Relevant to this audit:

- **`src_ii/forward.py`**: Contains `nfe()` and `denoise()`, the two
  lowest levels of the five-function decomposition. `nfe()` wraps
  `model.forward()` (the unpacked path only). There is no
  `nfe_packed()` or any reference to `forward_packed` in this module.

- **`src_ii/attention_capture.py`**: A monkey-patching module for
  interpretability. It patches `sdpa_attention` and always falls through
  to `F.scaled_dot_product_attention` (plain SDPA). It does not handle
  `block_mask` at all -- the patched function accepts the parameter but
  ignores it, always using the SDPA path.

- **`src_ii/bin_packer.py`**: Schedules mixed-resolution images into
  FlexAttention bins. References FlexAttention extensively in comments
  and design but contains no attention kernel code. It constructs plan
  items with `attention_backend` metadata but does not create block
  masks.

- **`src_ii/dataset_generator.py`**: Orchestrates the 7-phase generation
  pipeline. It passes `attention_backend` from plan items through to the
  client RPC but does not touch block mask construction.

- **`src_ii/model_loading.py`**: Contains `configure_sage_attention()`
  which sets the global `_SAGE_QK_QUANT` / `_SAGE_PV_QUANT` config.

There is **no attention dispatch logic in `src_ii/`**. The dispatch is
entirely in `src/futudiffu/attention.py`. The `src_ii/` rewrite has not
yet replicated or replaced the packed forward path.

---

## 7. Summary of Findings

| Question | Answer |
|----------|--------|
| Do masked Sage kernels exist? | Yes: `_sage_attn_fwd_fp8qk_bf16pv_masked` and `_sage_attn_fwd_int8qk_bf16pv_masked` |
| Are they wired into forward_packed? | No. The block mask is a FlexAttention `BlockMask`, not a uint8 tensor |
| What would wiring require? | (1) uint8 mask constructor, (2) conditional mask type at 2 call sites, (3) no kernel changes |
| Is there a duplicated code path? | Yes: FlexAttention for packed, SageAttention for unpacked |
| Does src_ii/ have attention dispatch? | No. All dispatch is in `src/futudiffu/attention.py` |

The gap is narrow: two call sites in `sampling.py` construct a
`BlockMask` instead of a `uint8` tensor. Everything downstream -- the
dispatch in `attention.py`, the API wrapper in `sage_attention.py`, and
the Triton kernels in `sage_kernels.py` -- is ready.

---

## 8. Implications for Integration Work

The integration work is a plumbing task, not a kernel task. The
delegation should include:

1. **Write `build_sage_block_mask(packing_info, BLOCK_M=128, BLOCK_N=64) -> torch.Tensor`**
   in `diffusion_model.py` alongside the existing `make_packing_mask_mod`.
   This produces a `(n_q_blocks, n_kv_blocks)` uint8 tensor from the
   `document_id`.

2. **Modify the two mask construction sites** in `sampling.py`
   (`run_packed_trajectories` line 728, `warmup_packed_forward` line 815)
   to conditionally build a uint8 tensor or a FlexAttention `BlockMask`
   based on `_ATTENTION_BACKEND`.

3. **Update `docs/flexattention_batch_packing.md` Section 8** to replace
   the "incompatible" claim with the current state: masked SageAttention
   kernels exist and are wired into `sdpa_attention()` via the uint8
   block mask type dispatch.

4. **Backward pass**: The masked kernels are forward-only (no LSE
   output). Training through the packed path with SageAttention would
   require masked variants of the LSE-producing kernels and their
   corresponding backward kernels (`_sage_attn_bwd_dkdv_masked`,
   `_sage_attn_bwd_dq_masked`). This is a kernel-level task and should
   be deferred unless training on packed batches is imminent.

5. **Test**: The S-S-S model (`tests/stubbed_skinny_shared.py`) can
   exercise `forward_packed` at low cost. A test that runs packed
   forward with both FlexAttention and masked SageAttention and
   compares outputs (expecting cos_sim > 0.99, not bitwise identity due
   to quantization) would validate the integration.

---

## Appendix A: Key File Paths

- Masked kernels: `/mnt/f/dox/repos/ai/futudiffu/src/futudiffu/sage_kernels.py` (lines 1159-1357)
- Kernel API wrapper: `/mnt/f/dox/repos/ai/futudiffu/src/futudiffu/sage_attention.py` (lines 183-268)
- Attention dispatch: `/mnt/f/dox/repos/ai/futudiffu/src/futudiffu/attention.py` (lines 37-119)
- Forward packed: `/mnt/f/dox/repos/ai/futudiffu/src/futudiffu/diffusion_model.py` (lines 1098-1235)
- Block mask construction: `/mnt/f/dox/repos/ai/futudiffu/src/futudiffu/sampling.py` (lines 728, 815)
- Outdated design doc: `/mnt/f/dox/repos/ai/futudiffu/docs/flexattention_batch_packing.md` (Section 8)
- S-S-S test model: `/mnt/f/dox/repos/ai/futudiffu/tests/stubbed_skinny_shared.py`
- src_ii forward: `/mnt/f/dox/repos/ai/futudiffu/src_ii/forward.py`
- src_ii attention capture: `/mnt/f/dox/repos/ai/futudiffu/src_ii/attention_capture.py`

## Appendix B: The Dispatch Decision Tree

```
sdpa_attention(q, k, v, heads, mask, block_mask)
  |
  |-- block_mask is Tensor AND dtype==uint8 AND backend in (sage, auto)?
  |     YES -> sage_attn_forward_masked(q, k, v, block_mask, sm_scale)
  |            [uses _sage_attn_fwd_{fp8,int8}qk_bf16pv_masked]
  |     NO  -> fall through
  |
  |-- block_mask is not None? (catches FlexAttention BlockMask objects)
  |     YES -> flex_attention(q, k, v, block_mask=block_mask)
  |     NO  -> fall through
  |
  |-- mask is None AND backend in (sage, auto)?
  |     YES -> sage_attn_op(q, k, v, sm_scale)
  |            [uses _sage_attn_fwd_{fp8,int8}qk_bf16pv with autograd]
  |     NO  -> fall through
  |
  |-- F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
```

Currently, packed forward always arrives at tier 2 (FlexAttention
BlockMask). The integration work routes it to tier 1 (uint8 tensor)
instead.

## Appendix C: Block Mask Tensor Shape Math

For a packed sequence of total_len tokens:
- `n_q_blocks = ceil(total_len / BLOCK_M)` where `BLOCK_M = 128`
- `n_kv_blocks = ceil(total_len / BLOCK_N)` where `BLOCK_N = 64`
- Block mask shape: `(n_q_blocks, n_kv_blocks)` uint8

Example: 2 images packed at 1280x832 (cap=32, img=4160, padded to 4192 each):
- `total_len = 2 * (32 + 4160) = 8384` (after per-segment padding)
- `n_q_blocks = ceil(8384 / 128) = 66`
- `n_kv_blocks = ceil(8384 / 64) = 132`
- Block mask: `(66, 132)` uint8 = 8,712 bytes

For 4 packed 256x256 images (total ~1152 tokens):
- `n_q_blocks = ceil(1152 / 128) = 9`
- `n_kv_blocks = ceil(1152 / 64) = 18`
- Block mask: `(9, 18)` uint8 = 162 bytes

The overhead is negligible.

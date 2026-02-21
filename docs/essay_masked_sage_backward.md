# Masked SageAttention Backward Kernels

## What Was Added

Three new Triton kernels in `sage_kernels.py` and three Python-level functions
plus a `torch.library.custom_op` registration in `sage_attention.py`. Together
these complete the missing quadrant: block-masked backward pass for
SageAttention, enabling end-to-end training through the packed forward path.

### Kernel Inventory (before and after)

| Kernel | Unmasked | Masked |
|--------|----------|--------|
| FP8 forward (no LSE) | `_sage_attn_fwd_fp8qk_bf16pv` | `_sage_attn_fwd_fp8qk_bf16pv_masked` |
| INT8 forward (no LSE) | `_sage_attn_fwd_int8qk_bf16pv` | `_sage_attn_fwd_int8qk_bf16pv_masked` |
| FP8 forward + LSE | `_sage_attn_fwd_fp8qk_bf16pv_lse` | -- (not needed) |
| INT8 forward + LSE | `_sage_attn_fwd_int8qk_bf16pv_lse` | **NEW:** `_sage_attn_fwd_int8qk_bf16pv_masked_lse` |
| D prepass | `_sage_attn_d_prepass` | (shared, no masking needed) |
| FP8 backward dK/dV | `_sage_attn_bwd_dkdv` | -- (INT8 only for masked) |
| FP8 backward dQ | `_sage_attn_bwd_dq` | -- (INT8 only for masked) |
| INT8 backward dK/dV | `_sage_attn_bwd_dkdv_int8` | **NEW:** `_sage_attn_bwd_dkdv_int8_masked` |
| INT8 backward dQ | `_sage_attn_bwd_dq_int8` | **NEW:** `_sage_attn_bwd_dq_int8_masked` |

### Python API

| Function | File | Purpose |
|----------|------|---------|
| `sage_attn_forward_masked_with_lse()` | `sage_attention.py` | Masked forward returning (out, lse) |
| `sage_attn_backward_masked()` | `sage_attention.py` | Masked backward returning (dq, dk, dv) |
| `sage_attn_masked_op` | `sage_attention.py` | `torch.library.custom_op` with `register_autograd` |

## Design Decisions

### Why INT8-only for the masked backward?

The masked forward already exists in both FP8 and INT8 variants. The backward
must recompute P using the exact same quantization path as the forward. Since
the backward kernels for masked attention are primarily needed for training
through FlexAttention batch packing, and the server runtime default is INT8 QK
(which has better accuracy -- 7-bit mantissa vs FP8's 3-bit), only the INT8
variant was implemented. Adding FP8 masked backward would be mechanical but is
not needed until someone trains through FP8 QK with block masks.

### MASK_BLOCK_M: cross-block-size mapping in dK/dV

The block mask is defined at the forward kernel's granularity: Q blocks of
BLOCK_M=128 tokens, KV blocks of BLOCK_N=64 tokens. The dQ backward kernel
uses the same outer Q block size (BLOCK_M=128), so mask indexing is trivial:
`mask[z, pid_m, n_idx]`.

The dK/dV backward kernel is different. Its outer loop is over KV blocks at
BLOCK_N=64 (matches forward), but its inner loop is over Q blocks at
BLOCK_M_BWD=64 (half the forward's 128). This means two consecutive backward
Q-block iterations map to the same forward Q-block mask entry. The kernel
accepts `MASK_BLOCK_M: tl.constexpr` (set to the forward's BLOCK_M=128) and
computes `mask_q_idx = m_start // MASK_BLOCK_M` to look up the correct entry.

This is correct because if mask[z, fwd_q, kv] == 0, then no Q token in that
128-token block should attend to any K token in that 64-token block. Both
64-token halves of the backward's inner Q loop get the same skip decision.

### D prepass is unmasked

The D prepass (`_sage_attn_d_prepass`) computes `Delta_i = rowsum(dO_i * O_i)`
for each query position. This depends only on the forward output O and the
upstream gradient dO, not on the Q-K attention pattern. The masking affected
which KV blocks contributed to O during forward, but that is already reflected
in the stored O values. No masking is needed in the D prepass.

### LSE handling for masked-out blocks

When a Q block has some KV blocks masked out, the LSE for that Q block reflects
only the KV blocks that were computed. The backward correctly skips the same
blocks -- it loads the same block mask and applies the same skip logic. The
recomputed P using `exp2(s - lse)` will match the forward's P exactly because
both forward and backward see the same set of active blocks and the same LSE.

For KV blocks that are masked out, the forward never accumulates them into m_i
or l_i, so they don't affect LSE. The backward skips them entirely, never
computing s for those blocks. P is never reconstructed for masked blocks, so
there is no risk of division by zero or stale LSE interaction.

### HAS_BLOCK_MASK compiles away

All three new kernels use `HAS_BLOCK_MASK: tl.constexpr`. When compiled with
`HAS_BLOCK_MASK=False`, the entire mask-checking code is eliminated at compile
time. The generated PTX is identical to the unmasked variants. This means
calling the masked kernels with `HAS_BLOCK_MASK=False` has zero overhead.

### Custom op registration

`sage_attn_masked_op` follows the exact pattern of the existing `sage_attn_op`:

1. Try `torch.library.custom_op` (torch >= 2.4)
2. Register `register_fake` returning `q.new_empty(shape)` (not `torch.empty_like`)
3. Register `register_autograd` wiring forward to backward
4. Fall back to `torch.autograd.Function` + `@torch.compiler.allow_in_graph`

The masked op takes 5 inputs (q, k, v, block_mask, sm_scale) and returns 2
outputs (out, lse). The backward returns 5 gradients (dq, dk, dv, None, None)
where None entries correspond to block_mask and sm_scale.

## Tests That Should Be Written

1. **Correctness vs unmasked (all-ones mask)**: Create a block mask of all 1s.
   Run both `sage_attn_masked_op` and `sage_attn_op` on the same inputs. The
   outputs should be bitwise identical (since the same INT8 code path runs).
   Run backward on both and verify dQ, dK, dV match.

2. **Correctness vs SDPA reference (block-diagonal mask)**: Create a packed
   sequence of 2 images with a block-diagonal mask. Run the masked forward and
   backward. Compare against running SDPA separately on each image and
   concatenating. The forward outputs should match within quantization tolerance
   (cos_sim > 0.99). The gradients should match similarly.

3. **Gradient finite-difference check**: For small inputs (B=1, H=2, N=256,
   D=128), compute numerical gradients via finite differences on the masked
   attention and compare against the Triton backward. This validates the
   softmax backward math with masking. Use a relaxed tolerance (1e-2) due to
   INT8 quantization noise.

4. **torch.compile compatibility**: Verify that `torch.compile(model)` where
   `model` calls `sage_attn_masked_op` produces zero graph breaks. Run both
   forward and backward under compile.

5. **Performance: masked vs unmasked**: Benchmark the masked backward with a
   50% density block mask (representative of 2-image packing). Verify that
   wall time is roughly 50% of the unmasked backward (confirming that tile
   skipping actually reduces work).

6. **Edge cases**: Empty mask (all zeros) should produce valid output (all
   zeros or NaN from 0/0 softmax, depending on convention). Single-block
   sequence. Sequence length not divisible by BLOCK_M or BLOCK_N.

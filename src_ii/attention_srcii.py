"""Compile-friendly sage attention ops for src_ii/.

The frozen sage_attention.py registers sage_attn_masked_op with a backward
function that calls raw Triton kernels. During AOT autograd tracing
(triggered by gradient_checkpointing + torch.compile), those kernels call
.data_ptr() on FakeTensors → crash.

This module defines replacement ops where the backward is itself a custom
op with register_fake. AOT autograd traces through the backward's fake
implementation (returns empty_like) instead of launching real Triton kernels.

src_ii/transformer.py imports sage_attn_masked_op from HERE, not from
frozen sage_attention.py.

The underlying kernel functions (sage_attn_forward_masked_with_lse,
sage_attn_backward_masked, etc.) are correct Triton implementations.
Only the torch.compile integration layer was broken.
"""

from __future__ import annotations

import torch
from torch import Tensor

from futudiffu.sage_attention import (
    sage_attn_forward_masked_with_lse,
    sage_attn_backward_masked,
    sage_attn_forward_with_lse,
    sage_attn_backward,
)

# ---------------------------------------------------------------------------
# Masked backward as opaque custom op
# ---------------------------------------------------------------------------

@torch.library.custom_op("futudiffu_ii::sage_bwd_masked", mutates_args=())
def _sage_bwd_masked(
    q: Tensor, k: Tensor, v: Tensor,
    out: Tensor, lse: Tensor, grad_out: Tensor,
    block_mask: Tensor, sm_scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    return sage_attn_backward_masked(q, k, v, out, lse, grad_out, block_mask, sm_scale)


@_sage_bwd_masked.register_fake
def _fake_bwd_masked(q, k, v, out, lse, grad_out, block_mask, sm_scale):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


# ---------------------------------------------------------------------------
# Masked forward op with compile-friendly backward
# ---------------------------------------------------------------------------

@torch.library.custom_op("futudiffu_ii::sage_attn_masked", mutates_args=())
def sage_attn_masked_op(
    q: Tensor, k: Tensor, v: Tensor, block_mask: Tensor, sm_scale: float,
) -> tuple[Tensor, Tensor]:
    return sage_attn_forward_masked_with_lse(q, k, v, block_mask, sm_scale)


@sage_attn_masked_op.register_fake
def _fake_fwd_masked(q, k, v, block_mask, sm_scale):
    B, H, N, D = q.shape
    return q.new_empty(B, H, N, D), q.new_empty(B * H, N, dtype=torch.float32)


def _masked_setup_context(ctx, inputs, output):
    q, k, v, block_mask, sm_scale = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse, block_mask)
    ctx.sm_scale = sm_scale


def _masked_backward(ctx, grad_out, grad_lse):
    q, k, v, out, lse, block_mask = ctx.saved_tensors
    dq, dk, dv = _sage_bwd_masked(
        q, k, v, out, lse, grad_out, block_mask, ctx.sm_scale,
    )
    return dq, dk, dv, None, None


sage_attn_masked_op.register_autograd(
    _masked_backward, setup_context=_masked_setup_context,
)


# ---------------------------------------------------------------------------
# Unmasked backward as opaque custom op
# ---------------------------------------------------------------------------

@torch.library.custom_op("futudiffu_ii::sage_bwd", mutates_args=())
def _sage_bwd(
    q: Tensor, k: Tensor, v: Tensor,
    out: Tensor, lse: Tensor, grad_out: Tensor,
    sm_scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    return sage_attn_backward(q, k, v, out, lse, grad_out, sm_scale)


@_sage_bwd.register_fake
def _fake_bwd(q, k, v, out, lse, grad_out, sm_scale):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


# ---------------------------------------------------------------------------
# Unmasked forward op with compile-friendly backward
# ---------------------------------------------------------------------------

@torch.library.custom_op("futudiffu_ii::sage_attn", mutates_args=())
def sage_attn_op(
    q: Tensor, k: Tensor, v: Tensor, sm_scale: float,
) -> tuple[Tensor, Tensor]:
    return sage_attn_forward_with_lse(q, k, v, sm_scale)


@sage_attn_op.register_fake
def _fake_fwd(q, k, v, sm_scale):
    B, H, N, D = q.shape
    return q.new_empty(B, H, N, D), q.new_empty(B * H, N, dtype=torch.float32)


def _setup_context(ctx, inputs, output):
    q, k, v, sm_scale = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse)
    ctx.sm_scale = sm_scale


def _backward(ctx, grad_out, grad_lse):
    q, k, v, out, lse = ctx.saved_tensors
    dq, dk, dv = _sage_bwd(q, k, v, out, lse, grad_out, ctx.sm_scale)
    return dq, dk, dv, None


sage_attn_op.register_autograd(_backward, setup_context=_setup_context)

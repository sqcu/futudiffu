"""Sage attention backward custom op wrapping for torch.compile.

The frozen sage_attention.py registers autograd backward functions that call
raw Triton kernels. torch.compile cannot trace these. This module wraps both
backward functions as opaque custom ops.

No sdpa_attention function. No monkey-patching of frozen attention/diffusion_model.
The branchless transformer in src_ii/transformer.py calls sage_attn_masked_op
directly — the forward dispatch problem is eliminated, not patched.

The backward monkey-patch on futudiffu.sage_attention IS valid because
_masked_backward resolves sage_attn_backward_masked via module globals
at call time (not captured via `from` import).

Usage:
    from src_ii.attention_srcii import patch_sage_for_compile
    patch_sage_for_compile()  # call once after model load, before any forward
"""

from __future__ import annotations

import torch
from torch import Tensor


def patch_sage_for_compile() -> None:
    """Patch sage attention backward functions as custom ops for torch.compile.

    Wraps both backward functions (masked and unmasked) as opaque custom ops
    with register_fake, then monkey-patches them into the sage_attention module.

    Call once after model load, before any compiled forward pass.
    Idempotent: checks the custom op registry instead of a module-level flag.
    """
    if hasattr(torch.ops, "futudiffu") and hasattr(torch.ops.futudiffu, "sage_bwd_masked"):
        return

    import futudiffu.sage_attention as _sa

    # Wrap masked backward
    _orig_bwd_masked = _sa.sage_attn_backward_masked

    @torch.library.custom_op("futudiffu::sage_bwd_masked", mutates_args=())
    def _sage_bwd_masked_op(
        q: Tensor, k: Tensor, v: Tensor,
        out: Tensor, lse: Tensor, grad_out: Tensor,
        block_mask: Tensor, sm_scale: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return _orig_bwd_masked(q, k, v, out, lse, grad_out, block_mask, sm_scale)

    @_sage_bwd_masked_op.register_fake
    def _fake_bwd_masked(q, k, v, out, lse, grad_out, block_mask, sm_scale):
        return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

    _sa.sage_attn_backward_masked = _sage_bwd_masked_op

    # Wrap unmasked backward
    _orig_bwd = _sa.sage_attn_backward

    @torch.library.custom_op("futudiffu::sage_bwd", mutates_args=())
    def _sage_bwd_op(
        q: Tensor, k: Tensor, v: Tensor,
        out: Tensor, lse: Tensor, grad_out: Tensor,
        sm_scale: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return _orig_bwd(q, k, v, out, lse, grad_out, sm_scale)

    @_sage_bwd_op.register_fake
    def _fake_bwd(q, k, v, out, lse, grad_out, sm_scale):
        return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

    _sa.sage_attn_backward = _sage_bwd_op

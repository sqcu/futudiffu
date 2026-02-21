"""Unified attention dispatch for the src_ii pipeline.

Routes attention computation to one of three backends:
  1. sage_attn_masked_op  -- packed sequences with uint8 block mask
  2. sage_attn_op         -- single images (no mask, fastest path)
  3. F.scaled_dot_product_attention -- CPU fallback, no Triton

The backend is selected by a string parameter, NOT by runtime inspection
of tensor types or values. This makes the dispatch resolvable at
torch.compile trace time with no graph breaks.

The custom ops (sage_attn_op, sage_attn_masked_op) are imported from
src/futudiffu/sage_attention.py. They have register_autograd, so
backward passes work through them without any additional wiring here.

Relationship to the other two modules:
  - block_mask.py constructs the uint8 mask tensor
  - attention.py (this) consumes the mask and dispatches to kernels
  - forward_packed.py orchestrates both within the NextDiT forward path

Import constraints:
  - IMPORTS from futudiffu.sage_attention: sage_attn_op, sage_attn_masked_op
  - torch and torch.nn.functional for SDPA fallback
  - No callable mask objects. uint8 tensors only.
"""

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

# Backend type for static dispatch (no runtime tensor inspection)
AttentionBackend = Literal["sage", "sage_masked", "sdpa"]


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    backend: AttentionBackend = "sage",
    block_mask: Tensor | None = None,
    sm_scale: float | None = None,
) -> Tensor:
    """Unified attention dispatch.

    Computes scaled dot-product attention via the specified backend.
    All inputs and outputs are in (B, H, N, D) layout (batch, heads,
    sequence, head_dim).

    Backend selection:
      - "sage_masked": Uses sage_attn_masked_op. Requires block_mask (uint8).
        This is the default for packed multi-image sequences. Supports
        both forward (inference) and backward (training through LoRA).
      - "sage": Uses sage_attn_op. No mask. This is the default for single
        images (no packing overhead). Supports forward and backward.
      - "sdpa": Uses torch.nn.functional.scaled_dot_product_attention.
        No mask support (causal=False, no attn_mask). Fallback for CPU
        or non-NVIDIA environments.

    The block_mask argument is ONLY consumed by "sage_masked". Passing a
    block_mask with backend="sage" or "sdpa" is an error (the mask would
    be silently ignored, which is a correctness bug for packed sequences).

    Args:
        q: (B, H, N, D) query tensor in BF16.
        k: (B, H, N, D) key tensor in BF16.
        v: (B, H, N, D) value tensor in BF16.
        backend: Attention backend to use.
        block_mask: uint8 tensor of shape (n_q_blocks, n_kv_blocks) or
            (B*H, n_q_blocks, n_kv_blocks). Required for "sage_masked",
            must be None for other backends.
        sm_scale: Softmax scale factor. If None, computed as 1/sqrt(D).

    Returns:
        (B, H, N, D) attention output.

    Raises:
        ValueError: If block_mask is provided with a non-masked backend,
            or if block_mask is missing with "sage_masked" backend.
    """
    B, H, N, D = q.shape

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    if backend == "sage_masked":
        if block_mask is None:
            raise ValueError(
                "sage_masked backend requires block_mask, got None. "
                "Use 'sage' backend for unmasked (single-image) attention."
            )
        from futudiffu.sage_attention import sage_attn_masked_op
        out, _lse = sage_attn_masked_op(q, k, v, block_mask, sm_scale)
        return out

    if backend == "sage":
        if block_mask is not None:
            raise ValueError(
                "sage backend does not accept block_mask (would silently "
                "ignore it, causing cross-image attention leakage). "
                "Use 'sage_masked' for packed sequences."
            )
        from futudiffu.sage_attention import sage_attn_op
        out, _lse = sage_attn_op(q, k, v, sm_scale)
        return out

    if backend == "sdpa":
        if block_mask is not None:
            raise ValueError(
                "sdpa backend does not accept block_mask. For packed "
                "sequences on CPU/non-NVIDIA, convert to a dense bool "
                "mask and pass as attn_mask."
            )
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
        )
        return out

    raise ValueError(f"Unknown attention backend: {backend!r}")


def select_backend(
    is_packed: bool,
    force_sdpa: bool = False,
) -> AttentionBackend:
    """Select the appropriate attention backend.

    Convenience function for callers that need to decide the backend once
    and pass it to multiple attention() calls (e.g., across transformer
    layers in a single forward pass).

    Args:
        is_packed: Whether the sequence is a packed multi-image batch.
        force_sdpa: Force SDPA fallback (e.g., for CPU testing).

    Returns:
        Backend string suitable for passing to attention().
    """
    if force_sdpa:
        return "sdpa"
    if is_packed:
        return "sage_masked"
    return "sage"

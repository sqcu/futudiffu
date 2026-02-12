"""SageAttention2 Python API.

Dispatcher, pre/post-processing, and fallback logic for the Triton attention
kernels defined in sage_kernels.py.

Usage:
    from futudiffu.sage_attention import sage_attn_forward, smoke_test_fp8_dot

    # Phase 0: validate hardware
    result = smoke_test_fp8_dot()
    print(f"FP8 dot cosine sim: {result['cosine_similarity']}")

    # Phase 1: fused attention (no grad)
    out = sage_attn_forward(q, k, v)  # (B, H, N, D) -> (B, H, N, D)

    # Phase 2: attention with backward pass (torch.compile compatible)
    out, lse = sage_attn_op(q, k, v, 1.0 / (128 ** 0.5))
    loss = out.sum()
    loss.backward()  # computes dQ, dK, dV via Triton backward kernels
"""

import math

import torch
from torch import Tensor

_LOG2E = math.log2(math.e)  # 1.4426950408889634

# --- Sage attention configuration ---
# Controlled via configure_sage(), checked by forward/backward at call time.

_SAGE_SMOOTH_K = True       # K-smoothing: subtract per-head channel mean before quantization
_SAGE_QK_QUANT = "fp8"      # "fp8" (Phase 1) or "int8" (Phase 2)
_SAGE_PV_QUANT = "bf16"     # "bf16" (Phase 1/2) or "fp8" (Phase 3, two-level accum)


def configure_sage(
    *,
    smooth_k: bool = True,
    qk_quant: str = "fp8",
    pv_quant: str = "bf16",
) -> None:
    """Configure SageAttention kernel variant selection.

    Args:
        smooth_k: Subtract per-head channel mean from K before quantization.
            Free accuracy improvement — softmax is shift-invariant.
        qk_quant: Quantization for QK^T matmul. "fp8" (3-bit mantissa, 660 TFLOPS)
            or "int8" (7-bit mantissa, 660 TOPS, better accuracy).
        pv_quant: Quantization for PV matmul. "bf16" (true FP32 accum) or
            "fp8" (FP22 accum with two-level accumulation, 2x PV throughput).
    """
    global _SAGE_SMOOTH_K, _SAGE_QK_QUANT, _SAGE_PV_QUANT
    if qk_quant not in ("fp8", "int8"):
        raise ValueError(f"Unknown qk_quant: {qk_quant!r}")
    if pv_quant not in ("bf16", "fp8"):
        raise ValueError(f"Unknown pv_quant: {pv_quant!r}")
    _SAGE_SMOOTH_K = smooth_k
    _SAGE_QK_QUANT = qk_quant
    _SAGE_PV_QUANT = pv_quant


def _smooth_k(k: Tensor) -> Tensor:
    """Subtract per-head channel mean from K (LASER-style preprocessing).

    K has shape (B, H, N, D). The mean across the sequence dimension (dim=2)
    is a rank-1 component that wastes FP8/INT8 dynamic range. Removing it
    is free because softmax(QK^T) = softmax(Q(K-mu)^T + Q*mu^T) and Q*mu^T
    is a per-query constant that cancels in the softmax normalization.
    """
    return k - k.mean(dim=2, keepdim=True)


def _check_sm_capability() -> tuple[int, int]:
    """Verify GPU has FP8 tensor core support (SM >= 8.9).

    Returns the (major, minor) compute capability.
    Raises RuntimeError if unsupported.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("SageAttention requires CUDA")
    cap = torch.cuda.get_device_capability()
    if cap < (8, 9):
        raise RuntimeError(
            f"SageAttention requires SM >= 8.9 (Ada Lovelace), got SM {cap[0]}.{cap[1]}"
        )
    return cap


def smoke_test_fp8_dot(n: int = 128) -> dict:
    """Phase 0: Validate FP8 dot product on this hardware.

    Creates two random (n, n) BF16 matrices, quantizes to FP8 E4M3 per-row
    inside a Triton kernel, computes the dot product, and compares against
    the BF16 PyTorch reference (a.float() @ b.float().T).

    Returns:
        dict with 'cosine_similarity' (target > 0.999) and 'max_error'.
    """
    _check_sm_capability()
    from .sage_kernels import _smoke_fp8_dot_kernel

    a = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
    c = torch.empty(n, n, dtype=torch.bfloat16, device="cuda")

    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    _smoke_fp8_dot_kernel[(1,)](a, b, c, N=n, FP8_MAX=fp8_max, num_warps=4)

    ref = a.float() @ b.float().T
    c_f32 = c.float()
    cos_sim = torch.nn.functional.cosine_similarity(
        c_f32.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)
    ).item()
    max_err = (c_f32 - ref).abs().max().item()

    return {"cosine_similarity": cos_sim, "max_error": max_err}


def sage_attn_forward(
    q: Tensor, k: Tensor, v: Tensor, sm_scale: float | None = None
) -> Tensor:
    """SageAttention forward pass (no LSE, no grad).

    Dispatches to FP8 or INT8 QK^T kernel based on _SAGE_QK_QUANT,
    with optional K-smoothing based on _SAGE_SMOOTH_K.

    Args:
        q, k, v: (B, H, N, D) tensors in BF16 on CUDA.
        sm_scale: Softmax scale factor (1/sqrt(D)). Computed from D if None.

    Returns:
        out: (B, H, N, D) tensor in BF16.
    """
    import triton

    B, H, N, D = q.shape
    assert D == 128, f"SageAttention currently requires head_dim=128, got {D}"
    assert q.dtype == torch.bfloat16, f"Expected BF16 input, got {q.dtype}"

    # K-smoothing: subtract per-head channel mean (free accuracy, shift-invariant)
    if _SAGE_SMOOTH_K:
        k = _smooth_k(k)

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    sm_scale_log2e = sm_scale * _LOG2E
    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    # Reshape to (B*H, N, D) and ensure contiguous
    q = q.reshape(B * H, N, D).contiguous()
    k = k.reshape(B * H, N, D).contiguous()
    v = v.reshape(B * H, N, D).contiguous()
    out = torch.empty_like(q)

    stride_z = N * D
    BLOCK_M = 128
    BLOCK_N = 64
    grid = (triton.cdiv(N, BLOCK_M), B * H)

    # Dispatch based on QK quantization type
    if _SAGE_QK_QUANT == "fp8":
        from .sage_kernels import _sage_attn_fwd_fp8qk_bf16pv
        _sage_attn_fwd_fp8qk_bf16pv[grid](
            q, k, v, out,
            stride_z, N, sm_scale_log2e,
            FP8_MAX=fp8_max, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
            num_warps=4, num_stages=1,
        )
    elif _SAGE_QK_QUANT == "int8":
        from .sage_kernels import _sage_attn_fwd_int8qk_bf16pv
        _sage_attn_fwd_int8qk_bf16pv[grid](
            q, k, v, out,
            stride_z, N, sm_scale_log2e,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
            num_warps=4, num_stages=1,
        )

    return out.reshape(B, H, N, D)


def sage_attn_forward_with_lse(
    q: Tensor, k: Tensor, v: Tensor, sm_scale: float | None = None
) -> tuple[Tensor, Tensor]:
    """SageAttention forward pass returning (output, LSE) for backward.

    Same computation as sage_attn_forward but additionally stores the
    log-sum-exp (LSE) per query position, needed for exact P recomputation
    in the backward pass.

    Args:
        q, k, v: (B, H, N, D) tensors in BF16 on CUDA.
        sm_scale: Softmax scale factor (1/sqrt(D)). Computed from D if None.

    Returns:
        out: (B, H, N, D) tensor in BF16.
        lse: (B*H, N) tensor in float32.
    """
    import triton

    B, H, N, D = q.shape
    assert D == 128, f"SageAttention currently requires head_dim=128, got {D}"
    assert q.dtype == torch.bfloat16, f"Expected BF16 input, got {q.dtype}"

    # K-smoothing: subtract per-head channel mean (free accuracy, shift-invariant)
    if _SAGE_SMOOTH_K:
        k = _smooth_k(k)

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    sm_scale_log2e = sm_scale * _LOG2E
    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    # Reshape to (B*H, N, D) and ensure contiguous
    q_flat = q.reshape(B * H, N, D).contiguous()
    k_flat = k.reshape(B * H, N, D).contiguous()
    v_flat = v.reshape(B * H, N, D).contiguous()
    out = torch.empty_like(q_flat)
    lse = torch.empty(B * H, N, dtype=torch.float32, device=q.device)

    stride_z = N * D
    BLOCK_M = 128
    BLOCK_N = 64
    grid = (triton.cdiv(N, BLOCK_M), B * H)

    # Dispatch based on QK and PV quantization type
    if _SAGE_PV_QUANT == "fp8":
        from .sage_kernels import _sage_attn_fwd_fp8qk_fp8pv_lse
        _sage_attn_fwd_fp8qk_fp8pv_lse[grid](
            q_flat, k_flat, v_flat, out, lse,
            stride_z, N, sm_scale_log2e,
            FP8_MAX=fp8_max, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
            num_warps=4, num_stages=1,
        )
    elif _SAGE_QK_QUANT == "int8":
        from .sage_kernels import _sage_attn_fwd_int8qk_bf16pv_lse
        _sage_attn_fwd_int8qk_bf16pv_lse[grid](
            q_flat, k_flat, v_flat, out, lse,
            stride_z, N, sm_scale_log2e,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
            num_warps=4, num_stages=1,
        )
    else:
        from .sage_kernels import _sage_attn_fwd_fp8qk_bf16pv_lse
        _sage_attn_fwd_fp8qk_bf16pv_lse[grid](
            q_flat, k_flat, v_flat, out, lse,
            stride_z, N, sm_scale_log2e,
            FP8_MAX=fp8_max, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
            num_warps=4, num_stages=1,
        )

    return out.reshape(B, H, N, D), lse


def sage_attn_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    lse: Tensor,
    dout: Tensor,
    sm_scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Backward pass: compute dQ, dK, dV.

    Uses saved LSE from forward to exactly recompute attention probabilities P
    via the same quantization path as forward (FP8 or INT8), then computes
    gradients. K-smoothing is re-applied if enabled (deterministic).

    Args:
        q, k, v: (B, H, N, D) tensors in BF16 on CUDA (same as forward input).
        out: (B, H, N, D) forward output in BF16.
        lse: (B*H, N) log-sum-exp from forward in float32.
        dout: (B, H, N, D) upstream gradient in BF16.
        sm_scale: 1/sqrt(D).

    Returns:
        dq, dk, dv: (B, H, N, D) gradients in BF16.
    """
    import triton

    B, H, N, D = q.shape
    BH = B * H

    # K-smoothing: must match forward's smoothing for correct P recomputation
    if _SAGE_SMOOTH_K:
        k = _smooth_k(k)

    sm_scale_log2e = sm_scale * _LOG2E
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    stride_z = N * D

    # Block sizes for backward kernels are smaller than forward to fit within
    # shared memory limits (~100KB on SM89). The dK/dV kernel has BLOCK_N as
    # the outer (program) dimension and BLOCK_M_BWD as the inner (Q loop),
    # while the dQ kernel has BLOCK_M as the outer and BLOCK_N_BWD as inner.
    BLOCK_M = 128     # outer block for D prepass and dQ kernel
    BLOCK_N_BWD = 64  # outer block for dK/dV kernel
    BLOCK_M_BWD = 64  # inner Q loop block for dK/dV kernel (reduced from 128)
    BLOCK_N_DQ = 64   # inner K/V loop block for dQ kernel

    # Reshape to (B*H, N, D) contiguous
    q_flat = q.reshape(BH, N, D).contiguous()
    k_flat = k.reshape(BH, N, D).contiguous()
    v_flat = v.reshape(BH, N, D).contiguous()
    out_flat = out.reshape(BH, N, D).contiguous()
    dout_flat = dout.reshape(BH, N, D).contiguous()

    # Step 1: D pre-pass — Delta_i = rowsum(dO_i * O_i)
    # (shared across all QK quant variants — no quantization involved)
    from .sage_kernels import _sage_attn_d_prepass
    delta = torch.empty(BH, N, dtype=torch.float32, device=q.device)
    grid_d = (triton.cdiv(N, BLOCK_M), BH)
    _sage_attn_d_prepass[grid_d](
        dout_flat, out_flat, delta,
        stride_z, N,
        D_HEAD=D, BLOCK_M=BLOCK_M,
        num_warps=4, num_stages=1,
    )

    # Steps 2-3: dK/dV and dQ kernels — dispatch based on QK quantization
    dk = torch.empty_like(q_flat)
    dv = torch.empty_like(q_flat)
    dq = torch.empty_like(q_flat)

    if _SAGE_QK_QUANT == "int8":
        from .sage_kernels import _sage_attn_bwd_dkdv_int8, _sage_attn_bwd_dq_int8

        grid_kv = (triton.cdiv(N, BLOCK_N_BWD), BH)
        _sage_attn_bwd_dkdv_int8[grid_kv](
            q_flat, k_flat, v_flat, dout_flat, out_flat, lse, delta,
            dk, dv,
            stride_z, N, sm_scale_log2e, sm_scale,
            BLOCK_M=BLOCK_M_BWD, BLOCK_N=BLOCK_N_BWD, D=D,
            num_warps=4, num_stages=1,
        )

        grid_q = (triton.cdiv(N, BLOCK_M), BH)
        _sage_attn_bwd_dq_int8[grid_q](
            q_flat, k_flat, v_flat, dout_flat, out_flat, lse, delta,
            dq,
            stride_z, N, sm_scale_log2e, sm_scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N_DQ, D=D,
            num_warps=4, num_stages=1,
        )
    else:
        from .sage_kernels import _sage_attn_bwd_dkdv, _sage_attn_bwd_dq

        grid_kv = (triton.cdiv(N, BLOCK_N_BWD), BH)
        _sage_attn_bwd_dkdv[grid_kv](
            q_flat, k_flat, v_flat, dout_flat, out_flat, lse, delta,
            dk, dv,
            stride_z, N, sm_scale_log2e, sm_scale,
            FP8_MAX=fp8_max,
            BLOCK_M=BLOCK_M_BWD, BLOCK_N=BLOCK_N_BWD, D=D,
            num_warps=4, num_stages=1,
        )

        grid_q = (triton.cdiv(N, BLOCK_M), BH)
        _sage_attn_bwd_dq[grid_q](
            q_flat, k_flat, v_flat, dout_flat, out_flat, lse, delta,
            dq,
            stride_z, N, sm_scale_log2e, sm_scale,
            FP8_MAX=fp8_max,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N_DQ, D=D,
            num_warps=4, num_stages=1,
        )

    return dq.reshape(B, H, N, D), dk.reshape(B, H, N, D), dv.reshape(B, H, N, D)


# =============================================================================
# Custom Op Registration for torch.compile compatibility
# =============================================================================

# Try the modern torch.library.custom_op API first (torch >= 2.4).
# Fall back to torch.autograd.Function + allow_in_graph for older versions.

_USE_CUSTOM_OP = False

try:
    # Probe for the custom_op decorator
    _custom_op_fn = torch.library.custom_op

    @torch.library.custom_op("futudiffu::sage_attn", mutates_args=())
    def sage_attn_op(q: Tensor, k: Tensor, v: Tensor, sm_scale: float) -> tuple[Tensor, Tensor]:
        """SageAttention forward: returns (output, lse) for backward."""
        return sage_attn_forward_with_lse(q, k, v, sm_scale)

    @sage_attn_op.register_fake
    def _sage_attn_op_fake(q: Tensor, k: Tensor, v: Tensor, sm_scale: float) -> tuple[Tensor, Tensor]:
        B, H, N, D = q.shape
        # Must return contiguous tensors — the real kernel reshapes to (BH, N, D)
        # contiguous then reshapes back, so output is always contiguous (B, H, N, D).
        # torch.empty_like(q) would preserve q's transposed strides and cause
        # CUDA graph stride assertion failures in torch.compile.
        return q.new_empty(B, H, N, D), q.new_empty(B * H, N, dtype=torch.float32)

    def _setup_context(ctx, inputs, output):
        q, k, v, sm_scale = inputs
        out, lse = output
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.sm_scale = sm_scale

    def _backward(ctx, grad_out, grad_lse):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = sage_attn_backward(q, k, v, out, lse, grad_out, ctx.sm_scale)
        return dq, dk, dv, None

    sage_attn_op.register_autograd(_backward, setup_context=_setup_context)
    _USE_CUSTOM_OP = True

except (AttributeError, TypeError):
    # Fall back to torch.autograd.Function with allow_in_graph for torch.compile
    pass


if not _USE_CUSTOM_OP:
    @torch.compiler.allow_in_graph
    class _SageAttnFunction(torch.autograd.Function):
        """SageAttention with autograd support, compatible with torch.compile."""

        @staticmethod
        def forward(ctx, q: Tensor, k: Tensor, v: Tensor, sm_scale: float) -> Tensor:
            out, lse = sage_attn_forward_with_lse(q, k, v, sm_scale)
            ctx.save_for_backward(q, k, v, out, lse)
            ctx.sm_scale = sm_scale
            return out

        @staticmethod
        def backward(ctx, grad_out: Tensor) -> tuple[Tensor, Tensor, Tensor, None]:
            q, k, v, out, lse = ctx.saved_tensors
            dq, dk, dv = sage_attn_backward(q, k, v, out, lse, grad_out, ctx.sm_scale)
            return dq, dk, dv, None

    def sage_attn_op(q: Tensor, k: Tensor, v: Tensor, sm_scale: float) -> tuple[Tensor, Tensor]:
        """SageAttention forward with autograd support (fallback path).

        Returns (output, lse) for API compatibility, but lse is detached.
        Backward is handled via autograd.Function.
        """
        out = _SageAttnFunction.apply(q, k, v, sm_scale)
        # Compute LSE separately for the return value (not part of autograd graph)
        with torch.no_grad():
            _, lse = sage_attn_forward_with_lse(q, k, v, sm_scale)
        return out, lse

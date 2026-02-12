"""
FP8 Blockwise/Rowwise Triton Kernels

Native FP8 matmul kernels for SM 8.9+ (Ada Lovelace, Hopper).
Uses tl.dot with FP8 inputs directly for native tensor core FP8 instructions.

Key kernels:
- fp8_gemm_blockwise: 2D blockwise FP8 matmul (weights have [N//bs, K//bs] scales)
- fp8_gemm_rowwise: Rowwise FP8 matmul (weights have [N] scales)
- fp8_addmm_blockwise: Fused matmul + bias (blockwise)
- fp8_addmm_rowwise: Fused matmul + bias (rowwise)
- fp8_gemm_quant: Fused matmul + output requantization to FP8
- fp8_gelu: Fused dequant -> GELU -> requant (no intermediate materialization)
- fp8_act_quant: Activation quantization to FP8 blockwise

torch._scaled_mm only supports scalar (tensorwise) scales, hence these custom kernels.
"""

import torch
import logging
from typing import Tuple

try:
    import triton
    import triton.language as tl
    from triton import Config

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False
    logging.info("FP8 kernels: Triton not available, will use dequantize fallback")


def _check_triton_available() -> bool:
    return _HAS_TRITON


if _HAS_TRITON:
    # ==============================================================================
    # FP8 Activation Quantization
    # ==============================================================================

    @triton.jit
    def fp8_act_quant_kernel(
        x_ptr,
        y_ptr,
        s_ptr,
        BLOCK_SIZE: tl.constexpr,
        FP8_MAX: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        x = tl.load(x_ptr + offs).to(tl.float32)
        amax = tl.max(tl.abs(x))

        scale = amax / FP8_MAX
        scale = tl.maximum(scale, 1e-12)

        quant_scale = FP8_MAX / tl.maximum(amax, 1e-12)
        y = x * quant_scale
        y = tl.minimum(tl.maximum(y, -FP8_MAX), FP8_MAX)

        tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty))
        tl.store(s_ptr + pid, scale)

    def fp8_act_quant(
        x: torch.Tensor, block_size: int = 128, dtype: torch.dtype = torch.float8_e4m3fn
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.is_contiguous(), "Input must be contiguous"
        assert x.size(-1) % block_size == 0

        fp8_max = torch.finfo(dtype).max
        y = torch.empty_like(x, dtype=dtype)
        s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)

        grid = (s.numel(),)
        fp8_act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size, FP8_MAX=fp8_max)
        return y, s

    # ==============================================================================
    # FP8 Blockwise GEMM — native FP8 tensor core matmul
    # ==============================================================================

    # Autotune configs: vary M tile and pipeline depth.
    # N and K tile sizes are pinned to input_block_size (constexpr) so scale
    # indexing is always 1:1 with quantization blocks.
    fp8_gemm_configs = [
        Config(
            {"BLOCK_SIZE_M": block_m},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for block_m in [64, 128, 256]
        for num_stages in [3, 4, 5]
        for num_warps in [4, 8]
    ]

    @triton.autotune(configs=fp8_gemm_configs, key=["M", "N", "K"])
    @triton.jit
    def fp8_gemm_blockwise_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        a_s_ptr,
        b_s_ptr,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        input_block_size: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        # N and K tile sizes are constexpr derived from input_block_size
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        """
        FP8 blockwise GEMM: C = A @ B.T

        Uses native FP8 tensor core instructions via tl.dot with FP8 operands.
        Scale indexing is 1:1 with quantization blocks because BLOCK_SIZE_N and
        BLOCK_SIZE_K are set equal to input_block_size.
        """
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        k_blocks = tl.cdiv(K, BLOCK_SIZE_K)

        offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]

        a_s_k_blocks = tl.cdiv(K, input_block_size)
        a_s_ptrs = a_s_ptr + offs_m * a_s_k_blocks

        b_s_k_blocks = tl.cdiv(K, input_block_size)
        b_s_base = b_s_ptr + pid_n * b_s_k_blocks

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k_idx in range(k_blocks):
            k_start = k_idx * BLOCK_SIZE_K

            mask_k = offs_k < K - k_start
            a_tile = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b_tile = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)

            # Native FP8 tensor core matmul — no upcast to FP32
            dot_result = tl.dot(a_tile, b_tile, out_dtype=tl.float32)

            k_scale_idx = k_start // input_block_size
            a_s = tl.load(a_s_ptrs + k_scale_idx)
            b_s = tl.load(b_s_base + k_scale_idx)

            accumulator += dot_result * a_s[:, None] * b_s

            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K

        c = accumulator.to(c_ptr.dtype.element_ty)
        offs_m_actual = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n_actual = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + offs_m_actual[:, None] * N + offs_n_actual[None, :]
        mask = (offs_m_actual[:, None] < M) & (offs_n_actual[None, :] < N)
        tl.store(c_ptrs, c, mask=mask)

    def fp8_gemm_blockwise(
        a: torch.Tensor,
        a_s: torch.Tensor,
        b: torch.Tensor,
        b_s: torch.Tensor,
        input_block_size: int = 128,
        output_dtype: torch.dtype = None,
    ) -> torch.Tensor:
        assert a.is_contiguous() and b.is_contiguous()
        assert a_s.is_contiguous() and b_s.is_contiguous()
        assert b.dim() == 2

        K = a.size(-1)
        M = a.numel() // K
        N = b.shape[0]
        batch_shape = a.shape[:-1]

        if output_dtype is None:
            output_dtype = torch.bfloat16

        c = a.new_empty(*batch_shape, N, dtype=output_dtype)

        def grid(META):
            return (
                triton.cdiv(M, META['BLOCK_SIZE_M']),
                triton.cdiv(N, input_block_size),
            )

        fp8_gemm_blockwise_kernel[grid](
            a, b, c, a_s, b_s,
            M, N, K, input_block_size,
            BLOCK_SIZE_N=input_block_size,
            BLOCK_SIZE_K=input_block_size,
        )
        return c

    # ==============================================================================
    # FP8 Blockwise GEMM + Bias (addmm)
    # ==============================================================================

    @triton.autotune(configs=fp8_gemm_configs, key=["M", "N", "K"])
    @triton.jit
    def fp8_addmm_blockwise_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        bias_ptr,
        a_s_ptr,
        b_s_ptr,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        input_block_size: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        k_blocks = tl.cdiv(K, BLOCK_SIZE_K)

        offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]

        a_s_k_blocks = tl.cdiv(K, input_block_size)
        a_s_ptrs = a_s_ptr + offs_m * a_s_k_blocks
        b_s_k_blocks = tl.cdiv(K, input_block_size)
        b_s_base = b_s_ptr + pid_n * b_s_k_blocks

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k_idx in range(k_blocks):
            k_start = k_idx * BLOCK_SIZE_K

            mask_k = offs_k < K - k_start
            a_tile = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b_tile = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)

            dot_result = tl.dot(a_tile, b_tile, out_dtype=tl.float32)

            k_scale_idx = k_start // input_block_size
            a_s = tl.load(a_s_ptrs + k_scale_idx)
            b_s = tl.load(b_s_base + k_scale_idx)

            accumulator += dot_result * a_s[:, None] * b_s

            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
            accumulator += bias[None, :]

        c = accumulator.to(c_ptr.dtype.element_ty)
        offs_m_actual = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n_actual = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + offs_m_actual[:, None] * N + offs_n_actual[None, :]
        mask = (offs_m_actual[:, None] < M) & (offs_n_actual[None, :] < N)
        tl.store(c_ptrs, c, mask=mask)

    def fp8_addmm_blockwise(
        a: torch.Tensor,
        a_s: torch.Tensor,
        b: torch.Tensor,
        b_s: torch.Tensor,
        bias: torch.Tensor = None,
        input_block_size: int = 128,
        output_dtype: torch.dtype = None,
    ) -> torch.Tensor:
        assert a.is_contiguous() and b.is_contiguous()
        assert a_s.is_contiguous() and b_s.is_contiguous()

        K = a.size(-1)
        M = a.numel() // K
        N = b.shape[0]
        batch_shape = a.shape[:-1]

        if output_dtype is None:
            output_dtype = torch.bfloat16

        c = a.new_empty(*batch_shape, N, dtype=output_dtype)

        has_bias = bias is not None
        if has_bias:
            assert bias.is_contiguous() and bias.dim() == 1 and bias.size(0) == N
            bias_ptr = bias
        else:
            bias_ptr = c

        def grid(META):
            return (
                triton.cdiv(M, META['BLOCK_SIZE_M']),
                triton.cdiv(N, input_block_size),
            )

        fp8_addmm_blockwise_kernel[grid](
            a, b, c, bias_ptr, a_s, b_s,
            M, N, K, input_block_size,
            BLOCK_SIZE_N=input_block_size,
            BLOCK_SIZE_K=input_block_size,
            HAS_BIAS=has_bias,
        )
        return c

    # ==============================================================================
    # FP8 Rowwise GEMM
    # ==============================================================================

    @triton.autotune(configs=fp8_gemm_configs, key=["M", "N", "K"])
    @triton.jit
    def fp8_gemm_rowwise_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        a_s_ptr,
        b_s_ptr,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        input_block_size: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        """
        FP8 rowwise GEMM. Weight scales are per-row [N], activation scales
        are blockwise [..., K//block_size].
        """
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        k_blocks = tl.cdiv(K, BLOCK_SIZE_K)

        offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]

        a_s_k_blocks = tl.cdiv(K, input_block_size)
        a_s_ptrs = a_s_ptr + offs_m * a_s_k_blocks

        # Per-row weight scales: load once for this N tile
        b_s = tl.load(b_s_ptr + offs_n)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k_idx in range(k_blocks):
            k_start = k_idx * BLOCK_SIZE_K

            mask_k = offs_k < K - k_start
            a_tile = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b_tile = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)

            dot_result = tl.dot(a_tile, b_tile, out_dtype=tl.float32)

            k_scale_idx = k_start // input_block_size
            a_s = tl.load(a_s_ptrs + k_scale_idx)

            accumulator += dot_result * a_s[:, None]

            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K

        # Apply per-row weight scales after full K accumulation
        accumulator *= b_s[None, :]

        c = accumulator.to(c_ptr.dtype.element_ty)
        offs_m_actual = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n_actual = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + offs_m_actual[:, None] * N + offs_n_actual[None, :]
        mask = (offs_m_actual[:, None] < M) & (offs_n_actual[None, :] < N)
        tl.store(c_ptrs, c, mask=mask)

    def fp8_gemm_rowwise(
        a: torch.Tensor,
        a_s: torch.Tensor,
        b: torch.Tensor,
        b_s: torch.Tensor,
        input_block_size: int = 128,
        output_dtype: torch.dtype = None,
    ) -> torch.Tensor:
        assert a.is_contiguous() and b.is_contiguous()
        assert a_s.is_contiguous() and b_s.is_contiguous()
        assert b_s.dim() == 1 and b_s.size(0) == b.size(0)

        K = a.size(-1)
        M = a.numel() // K
        N = b.shape[0]
        batch_shape = a.shape[:-1]

        if output_dtype is None:
            output_dtype = torch.bfloat16

        c = a.new_empty(*batch_shape, N, dtype=output_dtype)

        def grid(META):
            return (
                triton.cdiv(M, META['BLOCK_SIZE_M']),
                triton.cdiv(N, input_block_size),
            )

        fp8_gemm_rowwise_kernel[grid](
            a, b, c, a_s, b_s,
            M, N, K, input_block_size,
            BLOCK_SIZE_N=input_block_size,
            BLOCK_SIZE_K=input_block_size,
        )
        return c

    # ==============================================================================
    # FP8 Rowwise GEMM + Bias (addmm) — NEW
    # ==============================================================================

    @triton.autotune(configs=fp8_gemm_configs, key=["M", "N", "K"])
    @triton.jit
    def fp8_addmm_rowwise_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        bias_ptr,
        a_s_ptr,
        b_s_ptr,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        input_block_size: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        """Fused FP8 rowwise matmul + bias addition."""
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        k_blocks = tl.cdiv(K, BLOCK_SIZE_K)

        offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]

        a_s_k_blocks = tl.cdiv(K, input_block_size)
        a_s_ptrs = a_s_ptr + offs_m * a_s_k_blocks

        b_s = tl.load(b_s_ptr + offs_n)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k_idx in range(k_blocks):
            k_start = k_idx * BLOCK_SIZE_K

            mask_k = offs_k < K - k_start
            a_tile = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b_tile = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)

            dot_result = tl.dot(a_tile, b_tile, out_dtype=tl.float32)

            k_scale_idx = k_start // input_block_size
            a_s = tl.load(a_s_ptrs + k_scale_idx)

            accumulator += dot_result * a_s[:, None]

            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K

        accumulator *= b_s[None, :]

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
            accumulator += bias[None, :]

        c = accumulator.to(c_ptr.dtype.element_ty)
        offs_m_actual = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n_actual = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + offs_m_actual[:, None] * N + offs_n_actual[None, :]
        mask = (offs_m_actual[:, None] < M) & (offs_n_actual[None, :] < N)
        tl.store(c_ptrs, c, mask=mask)

    def fp8_addmm_rowwise(
        a: torch.Tensor,
        a_s: torch.Tensor,
        b: torch.Tensor,
        b_s: torch.Tensor,
        bias: torch.Tensor = None,
        input_block_size: int = 128,
        output_dtype: torch.dtype = None,
    ) -> torch.Tensor:
        assert a.is_contiguous() and b.is_contiguous()
        assert a_s.is_contiguous() and b_s.is_contiguous()
        assert b_s.dim() == 1 and b_s.size(0) == b.size(0)

        K = a.size(-1)
        M = a.numel() // K
        N = b.shape[0]
        batch_shape = a.shape[:-1]

        if output_dtype is None:
            output_dtype = torch.bfloat16

        c = a.new_empty(*batch_shape, N, dtype=output_dtype)

        has_bias = bias is not None
        if has_bias:
            assert bias.is_contiguous() and bias.dim() == 1 and bias.size(0) == N
            bias_ptr = bias
        else:
            bias_ptr = c

        def grid(META):
            return (
                triton.cdiv(M, META['BLOCK_SIZE_M']),
                triton.cdiv(N, input_block_size),
            )

        fp8_addmm_rowwise_kernel[grid](
            a, b, c, bias_ptr, a_s, b_s,
            M, N, K, input_block_size,
            BLOCK_SIZE_N=input_block_size,
            BLOCK_SIZE_K=input_block_size,
            HAS_BIAS=has_bias,
        )
        return c

    # ==============================================================================
    # FP8 Fused GEMM + Output Requantization — NEW
    # ==============================================================================

    @triton.heuristics({
        'NUM_BLOCKS': lambda args: args["BLOCK_SIZE_N"] // args["out_block_size"],
    })
    @triton.jit
    def fp8_gemm_quant_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        c_s_ptr,
        a_s_ptr,
        b_s_ptr,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        input_block_size: tl.constexpr,
        out_block_size: tl.constexpr,
        FP8_MAX: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
    ):
        """
        Fused FP8 matmul + output requantization to FP8.
        Avoids materializing full-precision intermediate.

        Output: FP8 tensor + per-block dequant scales.
        """
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        k_blocks = tl.cdiv(K, BLOCK_SIZE_K)

        offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]

        a_s_k_blocks = tl.cdiv(K, input_block_size)
        a_s_ptrs = a_s_ptr + offs_m * a_s_k_blocks
        b_s_base = b_s_ptr + pid_n * tl.cdiv(K, input_block_size)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k_idx in range(k_blocks):
            k_start = k_idx * BLOCK_SIZE_K

            mask_k = offs_k < K - k_start
            a_tile = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b_tile = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)

            dot_result = tl.dot(a_tile, b_tile, out_dtype=tl.float32)

            k_scale_idx = k_start // input_block_size
            a_s = tl.load(a_s_ptrs + k_scale_idx)
            b_s = tl.load(b_s_base + k_scale_idx)

            accumulator += dot_result * a_s[:, None] * b_s

            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K

        # Requantize output to FP8 with per-block scaling
        acc_reshaped = tl.reshape(accumulator, (BLOCK_SIZE_M, NUM_BLOCKS, out_block_size))

        block_max = tl.max(tl.abs(acc_reshaped), axis=2)
        dequant_scale = tl.maximum(block_max / FP8_MAX, 1e-12)

        quant_scale = FP8_MAX / tl.maximum(block_max, 1e-12)
        quant_scale_bc = tl.reshape(quant_scale, (BLOCK_SIZE_M, NUM_BLOCKS, 1))

        quantized = acc_reshaped * quant_scale_bc
        quantized = tl.minimum(tl.maximum(quantized, -FP8_MAX), FP8_MAX)
        quantized_fp8 = quantized.to(c_ptr.dtype.element_ty)
        quantized_fp8 = tl.reshape(quantized_fp8, (BLOCK_SIZE_M, BLOCK_SIZE_N))

        # Store quantized output
        offs_m_actual = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n_actual = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + offs_m_actual[:, None] * N + offs_n_actual[None, :]
        mask = (offs_m_actual[:, None] < M) & (offs_n_actual[None, :] < N)
        tl.store(c_ptrs, quantized_fp8, mask=mask)

        # Store dequant scales
        n_scale_stride = N // out_block_size
        offs_m_scale = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n_scale = pid_n * NUM_BLOCKS + tl.arange(0, NUM_BLOCKS)
        scale_ptrs = c_s_ptr + offs_m_scale[:, None] * n_scale_stride + offs_n_scale[None, :]
        scale_mask = (offs_m_scale[:, None] < M) & (offs_n_scale[None, :] < n_scale_stride)
        tl.store(scale_ptrs, dequant_scale, mask=scale_mask)

    def fp8_gemm_quant(
        a: torch.Tensor,
        a_s: torch.Tensor,
        b: torch.Tensor,
        b_s: torch.Tensor,
        input_block_size: int = 128,
        out_block_size: int = 128,
        out_dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fused FP8 matmul + output requantization. Returns (FP8 output, dequant scales)."""
        assert a.is_contiguous() and b.is_contiguous()
        assert a_s.is_contiguous() and b_s.is_contiguous()
        assert b.dim() == 2

        K = a.size(-1)
        M = a.numel() // K
        N = b.shape[0]
        batch_shape = a.shape[:-1]

        assert N % out_block_size == 0

        fp8_max = torch.finfo(out_dtype).max

        c = a.new_empty(*batch_shape, N, dtype=out_dtype)
        n_blocks = N // out_block_size
        c_s = a.new_empty(M, n_blocks, dtype=torch.float32)

        grid = (
            triton.cdiv(M, 128),
            triton.cdiv(N, input_block_size),
        )

        fp8_gemm_quant_kernel[grid](
            a, b, c, c_s, a_s, b_s,
            M, N, K, input_block_size, out_block_size, fp8_max,
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=input_block_size,
            BLOCK_SIZE_K=input_block_size,
        )

        if len(batch_shape) > 0:
            c_s = c_s.reshape(*batch_shape, n_blocks)

        return c, c_s

    # ==============================================================================
    # FP8 Fused GELU — NEW
    # Dequant -> GELU -> Requant in a single kernel, no intermediate write
    # ==============================================================================

    @triton.heuristics({
        'BLOCK_SN': lambda args: args["BLOCK_N"] // args["BLOCK_SIZE"],
    })
    @triton.jit
    def fp8_gelu_kernel(
        output_ptr,
        output_scale_ptr,
        input_ptr,
        input_scale_ptr,
        M,
        N: tl.constexpr,
        SN: tl.constexpr,
        FP8_MAX: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_SN: tl.constexpr,
    ):
        """
        Fused FP8 GELU: dequant(input) -> GELU -> requant(output)
        No intermediate full-precision tensor touches global memory.
        """
        pid = tl.program_id(0)
        NUM_BLOCK_N = tl.cdiv(N, BLOCK_N)
        pid_m = pid // NUM_BLOCK_N
        pid_n = pid % NUM_BLOCK_N

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Load FP8 input
        input_ptrs = input_ptr + offs_m[:, None] * N + offs_n[None, :]
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        input_data = tl.load(input_ptrs, mask=mask, other=0.0)

        # Load dequant scales
        offs_sn = pid_n * BLOCK_SN + tl.arange(0, BLOCK_SN)
        scale_ptrs = input_scale_ptr + offs_m[:, None] * SN + offs_sn[None, :]
        scale_mask = (offs_m[:, None] < M) & (offs_sn[None, :] < SN)
        input_scales = tl.load(scale_ptrs, mask=scale_mask, other=1.0)

        # Reshape for block-wise dequant
        input_data = tl.reshape(input_data.to(tl.float32), (BLOCK_M, BLOCK_SN, BLOCK_SIZE))
        input_scales = tl.reshape(input_scales, (BLOCK_M, BLOCK_SN, 1))

        # Dequantize
        x = input_data * input_scales

        # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        sqrt_2 = 1.41421356237
        gelu_out = x * 0.5 * (1.0 + tl.math.erf(x / sqrt_2))

        # Requantize to FP8
        abs_out = tl.abs(gelu_out)
        block_max = tl.max(abs_out, axis=2)
        dequant_scale = tl.maximum(block_max / FP8_MAX, 1e-12)

        quant_scale = FP8_MAX / tl.maximum(block_max, 1e-12)
        quant_scale_bc = tl.reshape(quant_scale, (BLOCK_M, BLOCK_SN, 1))

        quantized = gelu_out * quant_scale_bc
        quantized = tl.minimum(tl.maximum(quantized, -FP8_MAX), FP8_MAX)
        quantized_fp8 = quantized.to(output_ptr.dtype.element_ty)
        quantized_fp8 = tl.reshape(quantized_fp8, (BLOCK_M, BLOCK_N))

        # Store
        output_ptrs = output_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.store(output_ptrs, quantized_fp8, mask=mask)

        output_scale_ptrs = output_scale_ptr + offs_m[:, None] * SN + offs_sn[None, :]
        tl.store(output_scale_ptrs, dequant_scale, mask=scale_mask)

    def fp8_gelu(
        x: torch.Tensor,
        s_x: torch.Tensor,
        block_size: int = 128,
        dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fused FP8 GELU. Returns (FP8 output, dequant scales)."""
        assert x.is_contiguous() and s_x.is_contiguous()
        assert x.size(-1) % block_size == 0

        fp8_max = torch.finfo(dtype).max

        original_shape = x.shape
        batch_shape = original_shape[:-1]
        N = original_shape[-1]

        if x.dim() > 2:
            x = x.reshape(-1, N)
            s_x = s_x.reshape(-1, s_x.size(-1))

        M = x.size(0)
        SN = N // block_size
        kernel_block_n = max(128, block_size)
        if kernel_block_n % block_size != 0:
            kernel_block_n = block_size

        y = torch.empty_like(x, dtype=dtype)
        s_y = torch.empty_like(s_x, dtype=torch.float32)

        grid = (
            triton.cdiv(M, 128) * triton.cdiv(N, kernel_block_n),
        )

        fp8_gelu_kernel[grid](
            y, s_y, x, s_x,
            M, N, SN, fp8_max,
            BLOCK_SIZE=block_size,
            BLOCK_M=128,
            BLOCK_N=kernel_block_n,
        )

        if len(batch_shape) > 0:
            y = y.reshape(*batch_shape, N)
            s_y = s_y.reshape(*batch_shape, SN)

        return y, s_y

else:
    # Fallback stubs
    def fp8_act_quant(x, block_size=128, dtype=torch.float8_e4m3fn):
        raise RuntimeError("Triton not available for FP8 kernels")

    def fp8_gemm_blockwise(a, a_s, b, b_s, input_block_size=128, output_dtype=None):
        raise RuntimeError("Triton not available for FP8 kernels")

    def fp8_addmm_blockwise(a, a_s, b, b_s, bias=None, input_block_size=128, output_dtype=None):
        raise RuntimeError("Triton not available for FP8 kernels")

    def fp8_gemm_rowwise(a, a_s, b, b_s, input_block_size=128, output_dtype=None):
        raise RuntimeError("Triton not available for FP8 kernels")

    def fp8_addmm_rowwise(a, a_s, b, b_s, bias=None, input_block_size=128, output_dtype=None):
        raise RuntimeError("Triton not available for FP8 kernels")

    def fp8_gemm_quant(a, a_s, b, b_s, input_block_size=128, out_block_size=128, out_dtype=torch.float8_e4m3fn):
        raise RuntimeError("Triton not available for FP8 kernels")

    def fp8_gelu(x, s_x, block_size=128, dtype=torch.float8_e4m3fn):
        raise RuntimeError("Triton not available for FP8 kernels")

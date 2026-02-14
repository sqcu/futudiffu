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
- fp8_silu_gate_quant: Fused SiLU + gate + FP8 requant (BF16 in, FP8 out)
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

    # ==============================================================================
    # FP8 Fused SiLU + Gate + Requantization
    # Takes BF16 concatenated w1w3 output, splits, applies SiLU gating,
    # and quantizes to FP8 blockwise. For SwiGLU FFN: w2(silu(w1(x)) * w3(x))
    # where w1 and w3 are horizontally fused into a single (*, 2*N) GEMM output.
    # ==============================================================================

    @triton.jit
    def fp8_silu_gate_quant_kernel(
        input_ptr,
        output_ptr,
        output_scale_ptr,
        M,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        FP8_MAX: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """
        Fused SiLU + gate + FP8 requantization kernel.

        Input:  BF16 tensor of shape (M, 2*N) — first N cols are w1_out, last N are w3_out
        Output: FP8 tensor of shape (M, N) — silu(w1_out) * w3_out, quantized blockwise

        Each program handles a tile of BLOCK_M rows x BLOCK_N columns.
        BLOCK_N must equal BLOCK_SIZE for 1:1 scale indexing.
        """
        pid = tl.program_id(0)
        NUM_BLOCK_N = tl.cdiv(N, BLOCK_N)
        pid_m = pid // NUM_BLOCK_N
        pid_n = pid % NUM_BLOCK_N

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Input has stride 2*N per row
        input_stride = N * 2

        # Load w1_out from first half: input[m, n]
        w1_ptrs = input_ptr + offs_m[:, None] * input_stride + offs_n[None, :]
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        w1 = tl.load(w1_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Load w3_out from second half: input[m, N + n]
        w3_ptrs = input_ptr + offs_m[:, None] * input_stride + (N + offs_n[None, :])
        w3 = tl.load(w3_ptrs, mask=mask, other=0.0).to(tl.float32)

        # SiLU(w1) = w1 * sigmoid(w1) = w1 / (1 + exp(-w1))
        silu = w1 * (1.0 / (1.0 + tl.exp(-w1)))

        # Gate: silu(w1) * w3
        gated = silu * w3

        # Requantize to FP8 blockwise
        # Reshape to (BLOCK_M, 1, BLOCK_N) for per-block max over BLOCK_N dim
        # Since BLOCK_N == BLOCK_SIZE, each tile is exactly one quant block per row
        abs_gated = tl.abs(gated)
        # max over the column (BLOCK_N) dimension -> (BLOCK_M,)
        amax = tl.max(abs_gated, axis=1)

        # Dequant scale: stored so that dequant = fp8_val * scale
        dequant_scale = tl.maximum(amax / FP8_MAX, 1e-12)

        # Quant scale: fp8_val = bf16_val * quant_scale
        quant_scale = FP8_MAX / tl.maximum(amax, 1e-12)

        # Quantize
        quantized = gated * quant_scale[:, None]
        quantized = tl.minimum(tl.maximum(quantized, -FP8_MAX), FP8_MAX)
        quantized_fp8 = quantized.to(output_ptr.dtype.element_ty)

        # Store FP8 output (M, N)
        output_ptrs = output_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.store(output_ptrs, quantized_fp8, mask=mask)

        # Store dequant scales (M, N // BLOCK_SIZE)
        SN = N // BLOCK_SIZE
        offs_sn = pid_n  # one scale per BLOCK_N block
        scale_ptrs = output_scale_ptr + offs_m * SN + offs_sn
        scale_mask = offs_m < M
        tl.store(scale_ptrs, dequant_scale, mask=scale_mask)

    def fp8_silu_gate_quant(
        x: torch.Tensor,
        block_size: int = 128,
        dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fused SiLU + gate + FP8 requantization.

        Takes the concatenated w1w3 GEMM output (BF16), splits into two halves,
        applies SiLU gating (silu(w1_out) * w3_out), and quantizes to FP8 blockwise.

        Args:
            x: BF16 tensor of shape (*, 2*N) — concatenated w1 and w3 outputs
            block_size: FP8 quantization block size (default 128)
            dtype: FP8 output dtype (default float8_e4m3fn)

        Returns:
            (FP8 output of shape (*, N), dequant scales of shape (*, N // block_size))
        """
        assert x.is_contiguous(), "Input must be contiguous"
        total_cols = x.size(-1)
        assert total_cols % 2 == 0, f"Last dim must be even (got {total_cols})"
        N = total_cols // 2
        assert N % block_size == 0, f"N={N} must be divisible by block_size={block_size}"

        fp8_max = torch.finfo(dtype).max

        original_shape = x.shape
        batch_shape = original_shape[:-1]

        # Flatten to 2D: (M, 2*N)
        if x.dim() > 2:
            x = x.reshape(-1, total_cols)

        M = x.size(0)
        SN = N // block_size

        y = torch.empty(M, N, dtype=dtype, device=x.device)
        s_y = torch.empty(M, SN, dtype=torch.float32, device=x.device)

        BLOCK_M = 4
        BLOCK_N = block_size  # == BLOCK_SIZE for 1:1 scale indexing

        grid = (
            triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        )

        fp8_silu_gate_quant_kernel[grid](
            x, y, s_y,
            M, N, block_size, fp8_max,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        if len(batch_shape) > 1:
            y = y.reshape(*batch_shape, N)
            s_y = s_y.reshape(*batch_shape, SN)
        elif len(batch_shape) == 1:
            # Already (M, N) / (M, SN) which matches (batch, N) / (batch, SN)
            pass

        return y, s_y

    # ==============================================================================
    # V2 FP8 GEMM — Fused Activation Quantization + Coalesced Weight Access
    #
    # Improvements over v1:
    # 1. Accepts BF16 input A directly — quantizes to FP8 in-register (no separate
    #    act_quant kernel, eliminates one full GMEM roundtrip)
    # 2. Expects B pre-transposed to [K, N] layout for coalesced memory access
    #    (consecutive N elements are consecutive in memory)
    # 3. Tile sizes BLOCK_N is decoupled from input_block_size,
    #    allowing {128, 256} with multi-block scale indexing
    #
    # Architecture:
    #   The K loop always iterates in steps of input_block_size (128). This is the
    #   natural unit for FP8 blockwise quantization — each step gets one activation
    #   scale per row and one weight scale per (k_block, n_block) pair.
    #
    #   For BLOCK_N > input_block_size (e.g. 256 = 2*128), multiple weight scale
    #   values must be applied per dot product. We construct a (BLOCK_N,) scale
    #   vector with piecewise-constant values (one per 128-element sub-block) and
    #   broadcast-multiply against the dot result.
    # ==============================================================================

    def _smem_estimate_v2(bm, bn, ibs, ns):
        """Estimate shared memory bytes for v2 kernel configuration.
        A tile: BLOCK_M * input_block_size * 2 bytes (BF16)
        B tile: input_block_size * BLOCK_N * 1 byte  (FP8)

        The compiler double-buffers (ns stages), but A and B tiles may
        share pipeline slots. Use a heuristic: total = (A_tile + B_tile) * ns.
        SM89 supports up to 100KB (102400 bytes) per threadblock.
        We use 2x the limit to be permissive — Triton will validate at compile
        time and the autotuner will skip configs that fail compilation.
        """
        per_stage = bm * ibs * 2 + ibs * bn
        return per_stage * ns

    # SM89 has 100KB (102400 bytes) shared memory per threadblock.
    # K tile is always input_block_size=128 (not autotuned).
    # Use a generous filter (2x) — Triton rejects oversized configs at JIT time,
    # and the autotuner gracefully skips compilation failures.
    _IBS = 128  # input_block_size for smem estimation
    _SMEM_LIMIT = 102400 * 2  # generous: 200KB filter, actual 100KB enforced by HW
    fp8_gemm_v2_configs = [
        Config(
            {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn},
            num_stages=ns,
            num_warps=nw,
        )
        for bm in [64, 128, 256]
        for bn in [128, 256]
        for ns in [2, 3, 4]
        for nw in [4, 8]
        if _smem_estimate_v2(bm, bn, _IBS, ns) <= _SMEM_LIMIT
    ]

    @triton.autotune(configs=fp8_gemm_v2_configs, key=["M", "N", "K"])
    @triton.jit
    def fp8_gemm_v2_kernel(
        # A is BF16 input (fused quantization)
        a_ptr,          # (M, K) BF16 — NOT pre-quantized
        # B is FP8 weight, PRE-TRANSPOSED to [K, N]
        b_ptr,          # (K, N) FP8
        # Output
        c_ptr,          # (M, N) output dtype
        # Scales
        b_s_ptr,        # (K//bs, N//bs) float32 weight scales (transposed layout)
        # Dimensions
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        input_block_size: tl.constexpr,
        # Tile sizes (autotuned)
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        FP8_MAX: tl.constexpr,
    ):
        """
        FP8 blockwise GEMM v2: C = quantize(A) @ B  where B is [K, N] pre-transposed.

        Fuses activation quantization into the tile-loading code:
        - Loads BF16 A tile, quantizes per-sub-block (input_block_size elements along K)
        - Loads FP8 B tile with coalesced N-dimension access
        - Uses native FP8 tensor core dot products
        - Handles multi-block scale indexing when BLOCK_N > input_block_size
        """
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        # Total number of input_block_size steps across K
        k_steps = tl.cdiv(K, input_block_size)

        # Scale grid dimension for N
        n_scale_blocks = tl.cdiv(N, input_block_size)

        # Number of N sub-blocks covered by this N tile
        N_SUB: tl.constexpr = BLOCK_SIZE_N // input_block_size

        # Offsets — use modular wrapping for M to avoid masking loads
        offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        # Sub-block offsets for K dimension
        sb_offs = tl.arange(0, input_block_size)

        # Pre-compute row base addresses for A (loop-invariant)
        a_row_base = a_ptr + offs_m * K  # (BLOCK_SIZE_M,)

        # Pre-compute N sub-block index for weight scales (loop-invariant)
        n_base = pid_n * N_SUB  # first N scale block for this tile
        # For N_SUB > 1: which N sub-block does each column belong to?
        n_col_block = tl.arange(0, BLOCK_SIZE_N) // input_block_size  # (BLOCK_N,)

        # B scale base for this N tile
        b_s_n_base = b_s_ptr + n_base

        mask_n = offs_n < N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k_step in range(k_steps):
            k_start = k_step * input_block_size

            # --- Load A sub-block (BLOCK_M, input_block_size) as BF16 ---
            a_sub_ptrs = a_row_base[:, None] + (k_start + sb_offs[None, :])
            a_sub = tl.load(a_sub_ptrs)  # no mask needed: modular M, K divisible by bs
            a_f32 = a_sub.to(tl.float32)

            # Per-row amax and quantize to FP8 in register
            amax = tl.max(tl.abs(a_f32), axis=1)  # (BLOCK_SIZE_M,)
            amax = tl.maximum(amax, 1e-12)
            a_s = amax / FP8_MAX
            a_quant = a_f32 * (FP8_MAX / amax)[:, None]
            a_quant = tl.minimum(tl.maximum(a_quant, -FP8_MAX), FP8_MAX)
            a_fp8 = a_quant.to(b_ptr.dtype.element_ty)

            # --- Load B sub-block (input_block_size, BLOCK_N) — coalesced ---
            b_sub_ptrs = b_ptr + (k_start + sb_offs[:, None]) * N + offs_n[None, :]
            b_sub = tl.load(b_sub_ptrs, mask=mask_n[None, :], other=0.0)

            # --- FP8 tensor core dot ---
            dot_result = tl.dot(a_fp8, b_sub, out_dtype=tl.float32)

            # --- Apply dequantization scales ---
            b_s_k_offset = k_step * n_scale_blocks

            if N_SUB == 1:
                b_s_val = tl.load(b_s_n_base + b_s_k_offset)
                accumulator += dot_result * a_s[:, None] * b_s_val
            else:
                # Build (BLOCK_N,) scale vector from N_SUB scalar scales
                b_s_vec = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
                for ns_idx in range(N_SUB):
                    b_s_val = tl.load(b_s_n_base + b_s_k_offset + ns_idx)
                    b_s_vec += tl.where(n_col_block == ns_idx, b_s_val, 0.0)
                accumulator += dot_result * a_s[:, None] * b_s_vec[None, :]

        # Store output — mask needed for M boundary (modular wrapping loaded valid data)
        c = accumulator.to(c_ptr.dtype.element_ty)
        offs_m_actual = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        c_ptrs = c_ptr + offs_m_actual[:, None] * N + offs_n[None, :]
        mask = (offs_m_actual[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, c, mask=mask)

    def fp8_gemm_v2(
        a: torch.Tensor,        # (*, K) BF16 input
        b: torch.Tensor,        # (K, N) FP8 weight (pre-transposed)
        b_s: torch.Tensor,      # (K//bs, N//bs) float32 scales
        input_block_size: int = 128,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """FP8 GEMM v2: fused activation quantization + coalesced weight access.

        Args:
            a: BF16 input tensor of shape (*, K)
            b: FP8 weight tensor of shape (K, N), pre-transposed from original [N, K]
            b_s: float32 weight scales of shape (K//bs, N//bs), transposed layout
            input_block_size: quantization block size (default 128)
            output_dtype: output tensor dtype (default bfloat16)

        Returns:
            Output tensor of shape (*, N) in output_dtype
        """
        assert a.is_contiguous() and b.is_contiguous()
        assert b_s.is_contiguous()
        assert b.dim() == 2

        K = a.size(-1)
        M = a.numel() // K
        N = b.shape[1]  # B is [K, N]
        batch_shape = a.shape[:-1]

        assert b.shape[0] == K, f"B shape {b.shape} incompatible with K={K}"

        fp8_max = torch.finfo(torch.float8_e4m3fn).max

        c = a.new_empty(*batch_shape, N, dtype=output_dtype)

        def grid(META):
            return (
                triton.cdiv(M, META['BLOCK_SIZE_M']),
                triton.cdiv(N, META['BLOCK_SIZE_N']),
            )

        fp8_gemm_v2_kernel[grid](
            a, b, c, b_s,
            M, N, K, input_block_size,
            FP8_MAX=fp8_max,
        )
        return c

    # ==============================================================================
    # V2 FP8 GEMM + Bias (addmm) — same optimizations as fp8_gemm_v2 + bias epilogue
    # ==============================================================================

    @triton.autotune(configs=fp8_gemm_v2_configs, key=["M", "N", "K"])
    @triton.jit
    def fp8_addmm_v2_kernel(
        # A is BF16 input (fused quantization)
        a_ptr,          # (M, K) BF16
        # B is FP8 weight, PRE-TRANSPOSED to [K, N]
        b_ptr,          # (K, N) FP8
        # Output
        c_ptr,          # (M, N) output dtype
        # Bias
        bias_ptr,       # (N,) bias vector
        # Scales
        b_s_ptr,        # (K//bs, N//bs) float32 weight scales
        # Dimensions
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        input_block_size: tl.constexpr,
        # Tile sizes (autotuned)
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        FP8_MAX: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        """
        FP8 blockwise GEMM v2 + bias: C = quantize(A) @ B + bias
        Same as fp8_gemm_v2_kernel but adds bias in the epilogue.
        """
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        k_steps = tl.cdiv(K, input_block_size)
        n_scale_blocks = tl.cdiv(N, input_block_size)
        N_SUB: tl.constexpr = BLOCK_SIZE_N // input_block_size

        # Modular wrapping for M to avoid masking loads
        offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        sb_offs = tl.arange(0, input_block_size)

        # Pre-compute row base addresses and loop-invariant values
        a_row_base = a_ptr + offs_m * K
        n_base = pid_n * N_SUB
        n_col_block = tl.arange(0, BLOCK_SIZE_N) // input_block_size
        b_s_n_base = b_s_ptr + n_base

        mask_n = offs_n < N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k_step in range(k_steps):
            k_start = k_step * input_block_size

            # Load and quantize A sub-block
            a_sub_ptrs = a_row_base[:, None] + (k_start + sb_offs[None, :])
            a_sub = tl.load(a_sub_ptrs)
            a_f32 = a_sub.to(tl.float32)

            amax = tl.max(tl.abs(a_f32), axis=1)
            amax = tl.maximum(amax, 1e-12)
            a_s = amax / FP8_MAX
            a_quant = a_f32 * (FP8_MAX / amax)[:, None]
            a_quant = tl.minimum(tl.maximum(a_quant, -FP8_MAX), FP8_MAX)
            a_fp8 = a_quant.to(b_ptr.dtype.element_ty)

            # Load B sub-block (coalesced)
            b_sub_ptrs = b_ptr + (k_start + sb_offs[:, None]) * N + offs_n[None, :]
            b_sub = tl.load(b_sub_ptrs, mask=mask_n[None, :], other=0.0)

            # FP8 dot
            dot_result = tl.dot(a_fp8, b_sub, out_dtype=tl.float32)

            # Apply scales
            b_s_k_offset = k_step * n_scale_blocks
            if N_SUB == 1:
                b_s_val = tl.load(b_s_n_base + b_s_k_offset)
                accumulator += dot_result * a_s[:, None] * b_s_val
            else:
                b_s_vec = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
                for ns_idx in range(N_SUB):
                    b_s_val = tl.load(b_s_n_base + b_s_k_offset + ns_idx)
                    b_s_vec += tl.where(n_col_block == ns_idx, b_s_val, 0.0)
                accumulator += dot_result * a_s[:, None] * b_s_vec[None, :]

        # Bias epilogue
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
            accumulator += bias[None, :]

        # Store output
        c = accumulator.to(c_ptr.dtype.element_ty)
        offs_m_actual = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        c_ptrs = c_ptr + offs_m_actual[:, None] * N + offs_n[None, :]
        mask = (offs_m_actual[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, c, mask=mask)

    def fp8_addmm_v2(
        a: torch.Tensor,        # (*, K) BF16 input
        b: torch.Tensor,        # (K, N) FP8 weight (pre-transposed)
        b_s: torch.Tensor,      # (K//bs, N//bs) float32 scales
        bias: torch.Tensor = None,  # (N,) optional bias
        input_block_size: int = 128,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """FP8 GEMM v2 + bias: fused activation quantization + coalesced weight access + bias.

        Args:
            a: BF16 input tensor of shape (*, K)
            b: FP8 weight tensor of shape (K, N), pre-transposed from original [N, K]
            b_s: float32 weight scales of shape (K//bs, N//bs), transposed layout
            bias: optional bias vector of shape (N,)
            input_block_size: quantization block size (default 128)
            output_dtype: output tensor dtype (default bfloat16)

        Returns:
            Output tensor of shape (*, N) in output_dtype
        """
        assert a.is_contiguous() and b.is_contiguous()
        assert b_s.is_contiguous()
        assert b.dim() == 2

        K = a.size(-1)
        M = a.numel() // K
        N = b.shape[1]
        batch_shape = a.shape[:-1]

        assert b.shape[0] == K, f"B shape {b.shape} incompatible with K={K}"

        fp8_max = torch.finfo(torch.float8_e4m3fn).max

        c = a.new_empty(*batch_shape, N, dtype=output_dtype)

        has_bias = bias is not None
        if has_bias:
            assert bias.is_contiguous() and bias.dim() == 1 and bias.size(0) == N
            bias_ptr = bias
        else:
            bias_ptr = c  # dummy pointer, not accessed

        def grid(META):
            return (
                triton.cdiv(M, META['BLOCK_SIZE_M']),
                triton.cdiv(N, META['BLOCK_SIZE_N']),
            )

        fp8_addmm_v2_kernel[grid](
            a, b, c, bias_ptr, b_s,
            M, N, K, input_block_size,
            FP8_MAX=fp8_max,
            HAS_BIAS=has_bias,
        )
        return c

    # ==============================================================================
    # V1T: Transpose-only GEMM — coalesced B access, pre-quantized FP8 input
    # ==============================================================================
    # Combines the best of v1 (pre-quantized FP8 inputs, no prologue overhead)
    # with v2's coalesced B access (B stored as [K, N] instead of [N, K]).
    # Also uses enlarged tile configs for multi-block N/K indexing.

    fp8_gemm_v1t_configs = [
        Config(
            {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk},
            num_stages=ns,
            num_warps=nw,
        )
        for bm in [64, 128, 256]
        for bn in [128, 256]
        for bk in [128]
        for ns in [2, 3, 4]
        for nw in [4, 8]
        if _smem_estimate_v2(bm, bn, bk, ns) <= _SMEM_LIMIT
    ]

    @triton.autotune(configs=fp8_gemm_v1t_configs, key=["M", "N", "K"])
    @triton.jit
    def fp8_gemm_v1t_kernel(
        # A is pre-quantized FP8 (from fp8_act_quant)
        a_ptr,          # (M, K) FP8
        # B is FP8 weight, PRE-TRANSPOSED to [K, N]
        b_ptr,          # (K, N) FP8
        # Output
        c_ptr,          # (M, N) output dtype
        # Scales
        a_s_ptr,        # (M, K//bs) float32 activation scales (rowwise blockwise)
        b_s_ptr,        # (K//bs, N//bs) float32 weight scales (transposed layout)
        # Dimensions
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        input_block_size: tl.constexpr,
        # Tile sizes (autotuned)
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        """
        FP8 blockwise GEMM v1t: C = A @ B  where A is FP8, B is [K, N] pre-transposed FP8.

        Combines pre-quantized FP8 input (no in-register quantization overhead)
        with coalesced B memory access (consecutive threads read consecutive N addresses).
        Multi-block scale indexing supports BLOCK_N > input_block_size.
        """
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        k_blocks = tl.cdiv(K, BLOCK_SIZE_K)

        # Scale grid dimensions
        a_s_k_blocks = tl.cdiv(K, input_block_size)
        n_scale_blocks = tl.cdiv(N, input_block_size)

        # Number of sub-blocks per tile in each dimension
        N_SUB: tl.constexpr = BLOCK_SIZE_N // input_block_size
        K_SUB: tl.constexpr = BLOCK_SIZE_K // input_block_size

        # Offsets — modular wrapping for M
        offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        # A pointers: (BLOCK_M, BLOCK_K), A is [M, K] row-major FP8
        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]

        # B pointers: (BLOCK_K, BLOCK_N), B is [K, N] row-major FP8 (coalesced!)
        b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]

        # A scale base pointers: a_s is (M, K//bs), row-major
        a_s_base = a_s_ptr + offs_m * a_s_k_blocks  # (BLOCK_M,)

        # B scale: (K//bs, N//bs), row-major
        n_base = pid_n * N_SUB
        b_s_n_base = b_s_ptr + n_base

        # For multi-block N: which sub-block does each N column belong to?
        n_col_block = tl.arange(0, BLOCK_SIZE_N) // input_block_size  # (BLOCK_N,)

        mask_n = offs_n < N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k_idx in range(k_blocks):
            k_start = k_idx * BLOCK_SIZE_K

            # Load A tile (BLOCK_M, BLOCK_K) — FP8
            mask_k = offs_k < K - k_start
            a_tile = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)

            # Load B tile (BLOCK_K, BLOCK_N) — FP8, coalesced access
            b_tile = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)

            # FP8 tensor core dot product
            dot_result = tl.dot(a_tile, b_tile, out_dtype=tl.float32)

            # Apply dequantization scales — handle K_SUB x N_SUB sub-blocks
            for ks in range(K_SUB):
                k_scale_idx = (k_start // input_block_size) + ks

                # A scale for this K sub-block: (BLOCK_M,)
                a_s = tl.load(a_s_base + k_scale_idx)

                # B scale for this K sub-block
                b_s_k_offset = k_scale_idx * n_scale_blocks

                if N_SUB == 1:
                    b_s_val = tl.load(b_s_n_base + b_s_k_offset)
                    # This K sub-block's contribution range in dot_result
                    # When K_SUB=1, it's the entire dot_result
                    accumulator += dot_result * a_s[:, None] * b_s_val
                else:
                    b_s_vec = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
                    for ns_idx in range(N_SUB):
                        b_s_val = tl.load(b_s_n_base + b_s_k_offset + ns_idx)
                        b_s_vec += tl.where(n_col_block == ns_idx, b_s_val, 0.0)
                    accumulator += dot_result * a_s[:, None] * b_s_vec[None, :]

            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K * N

        # Store output
        c = accumulator.to(c_ptr.dtype.element_ty)
        offs_m_actual = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        c_ptrs = c_ptr + offs_m_actual[:, None] * N + offs_n[None, :]
        mask = (offs_m_actual[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, c, mask=mask)

    def fp8_gemm_v1t(
        a: torch.Tensor,        # (*, K) FP8 pre-quantized
        a_s: torch.Tensor,      # (M, K//bs) float32 activation scales
        b: torch.Tensor,        # (K, N) FP8 weight (pre-transposed)
        b_s: torch.Tensor,      # (K//bs, N//bs) float32 weight scales
        input_block_size: int = 128,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """FP8 GEMM v1t: pre-quantized FP8 input + coalesced pre-transposed weight access.

        Isolates the coalesced memory access benefit without fused act_quant overhead.

        Args:
            a: FP8 input tensor of shape (*, K) — output of fp8_act_quant
            a_s: float32 activation scales of shape (M, K//bs) — output of fp8_act_quant
            b: FP8 weight tensor of shape (K, N), pre-transposed from original [N, K]
            b_s: float32 weight scales of shape (K//bs, N//bs), transposed layout
            input_block_size: quantization block size (default 128)
            output_dtype: output tensor dtype (default bfloat16)

        Returns:
            Output tensor of shape (*, N) in output_dtype
        """
        assert a.is_contiguous() and b.is_contiguous()
        assert a_s.is_contiguous() and b_s.is_contiguous()
        assert b.dim() == 2

        K = a.size(-1)
        M = a.numel() // K
        N = b.shape[1]  # B is [K, N]
        batch_shape = a.shape[:-1]

        assert b.shape[0] == K, f"B shape {b.shape} incompatible with K={K}"

        c = a.new_empty(*batch_shape, N, dtype=output_dtype)

        def grid(META):
            return (
                triton.cdiv(M, META['BLOCK_SIZE_M']),
                triton.cdiv(N, META['BLOCK_SIZE_N']),
            )

        fp8_gemm_v1t_kernel[grid](
            a, b, c, a_s, b_s,
            M, N, K, input_block_size,
        )
        return c

    # ==============================================================================
    # V1T: FP8 GEMM + Bias (addmm) — coalesced B, pre-quantized FP8, with bias
    # ==============================================================================

    @triton.autotune(configs=fp8_gemm_v1t_configs, key=["M", "N", "K"])
    @triton.jit
    def fp8_addmm_v1t_kernel(
        a_ptr,          # (M, K) FP8
        b_ptr,          # (K, N) FP8
        c_ptr,          # (M, N) output dtype
        bias_ptr,       # (N,) bias vector
        a_s_ptr,        # (M, K//bs) float32
        b_s_ptr,        # (K//bs, N//bs) float32
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        input_block_size: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        """FP8 blockwise GEMM v1t + bias: C = A @ B + bias, coalesced B access."""
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
        a_s_k_blocks = tl.cdiv(K, input_block_size)
        n_scale_blocks = tl.cdiv(N, input_block_size)
        N_SUB: tl.constexpr = BLOCK_SIZE_N // input_block_size
        K_SUB: tl.constexpr = BLOCK_SIZE_K // input_block_size

        offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]

        a_s_base = a_s_ptr + offs_m * a_s_k_blocks
        n_base = pid_n * N_SUB
        b_s_n_base = b_s_ptr + n_base
        n_col_block = tl.arange(0, BLOCK_SIZE_N) // input_block_size

        mask_n = offs_n < N
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k_idx in range(k_blocks):
            k_start = k_idx * BLOCK_SIZE_K
            mask_k = offs_k < K - k_start
            a_tile = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b_tile = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)

            dot_result = tl.dot(a_tile, b_tile, out_dtype=tl.float32)

            for ks in range(K_SUB):
                k_scale_idx = (k_start // input_block_size) + ks
                a_s = tl.load(a_s_base + k_scale_idx)
                b_s_k_offset = k_scale_idx * n_scale_blocks

                if N_SUB == 1:
                    b_s_val = tl.load(b_s_n_base + b_s_k_offset)
                    accumulator += dot_result * a_s[:, None] * b_s_val
                else:
                    b_s_vec = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
                    for ns_idx in range(N_SUB):
                        b_s_val = tl.load(b_s_n_base + b_s_k_offset + ns_idx)
                        b_s_vec += tl.where(n_col_block == ns_idx, b_s_val, 0.0)
                    accumulator += dot_result * a_s[:, None] * b_s_vec[None, :]

            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K * N

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
            accumulator += bias[None, :]

        c = accumulator.to(c_ptr.dtype.element_ty)
        offs_m_actual = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        c_ptrs = c_ptr + offs_m_actual[:, None] * N + offs_n[None, :]
        mask = (offs_m_actual[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, c, mask=mask)

    def fp8_addmm_v1t(
        a: torch.Tensor,        # (*, K) FP8
        a_s: torch.Tensor,      # (M, K//bs) float32
        b: torch.Tensor,        # (K, N) FP8 (pre-transposed)
        b_s: torch.Tensor,      # (K//bs, N//bs) float32
        bias: torch.Tensor = None,
        input_block_size: int = 128,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """FP8 GEMM v1t + bias: pre-quantized FP8 + coalesced B + optional bias."""
        assert a.is_contiguous() and b.is_contiguous()
        assert a_s.is_contiguous() and b_s.is_contiguous()
        assert b.dim() == 2

        K = a.size(-1)
        M = a.numel() // K
        N = b.shape[1]
        batch_shape = a.shape[:-1]

        assert b.shape[0] == K, f"B shape {b.shape} incompatible with K={K}"

        c = a.new_empty(*batch_shape, N, dtype=output_dtype)

        has_bias = bias is not None
        if has_bias:
            assert bias.is_contiguous() and bias.dim() == 1 and bias.size(0) == N
            bias_ptr = bias
        else:
            bias_ptr = c  # dummy

        def grid(META):
            return (
                triton.cdiv(M, META['BLOCK_SIZE_M']),
                triton.cdiv(N, META['BLOCK_SIZE_N']),
            )

        fp8_addmm_v1t_kernel[grid](
            a, b, c, bias_ptr, a_s, b_s,
            M, N, K, input_block_size,
            HAS_BIAS=has_bias,
        )
        return c


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

    def fp8_silu_gate_quant(x, block_size=128, dtype=torch.float8_e4m3fn):
        """Pure-PyTorch fallback for fused SiLU + gate + FP8 requantization."""
        total_cols = x.size(-1)
        N = total_cols // 2
        w1_out = x[..., :N].float()
        w3_out = x[..., N:].float()
        gated = torch.nn.functional.silu(w1_out) * w3_out
        # Blockwise quantization
        flat = gated.reshape(-1, N)
        M = flat.size(0)
        SN = N // block_size
        flat_blocks = flat.reshape(M, SN, block_size)
        amax = flat_blocks.abs().amax(dim=-1)  # (M, SN)
        fp8_max = torch.finfo(dtype).max
        dequant_scale = (amax / fp8_max).clamp(min=1e-12)
        quant_scale = fp8_max / amax.clamp(min=1e-12)
        quantized = (flat_blocks * quant_scale.unsqueeze(-1)).clamp(-fp8_max, fp8_max)
        quantized = quantized.reshape(M, N).to(dtype)
        batch_shape = x.shape[:-1]
        if len(batch_shape) > 1:
            quantized = quantized.reshape(*batch_shape, N)
            dequant_scale = dequant_scale.reshape(*batch_shape, SN)
        return quantized, dequant_scale

    def fp8_gemm_v2(a, b, b_s, input_block_size=128, output_dtype=torch.bfloat16):
        raise RuntimeError("Triton not available for FP8 kernels")

    def fp8_addmm_v2(a, b, b_s, bias=None, input_block_size=128, output_dtype=torch.bfloat16):
        raise RuntimeError("Triton not available for FP8 kernels")

    def fp8_gemm_v1t(a, a_s, b, b_s, input_block_size=128, output_dtype=torch.bfloat16):
        raise RuntimeError("Triton not available for FP8 kernels")

    def fp8_addmm_v1t(a, a_s, b, b_s, bias=None, input_block_size=128, output_dtype=torch.bfloat16):
        raise RuntimeError("Triton not available for FP8 kernels")


# ==============================================================================
# Custom op registrations for torch.compile compatibility
# ==============================================================================
# These allow torch.compile(mode="reduce-overhead") to capture direct Triton
# kernel calls (fp8_silu_gate_quant, fp8_gemm_blockwise) into CUDA graphs.
# Used by FeedForward._forward_fused_chain which calls these functions directly
# (bypassing FP8Linear which already has its own custom_ops in fp8.py).
# ==============================================================================

# Dtype encoding for custom_op args (torch.dtype cannot be passed directly).
_DTYPE_TO_CODE_K = {
    torch.bfloat16: 0, torch.float16: 1, torch.float32: 2,
    torch.float8_e4m3fn: 3,
}
_CODE_TO_DTYPE_K = {v: k for k, v in _DTYPE_TO_CODE_K.items()}

_USE_FP8_KERNEL_CUSTOM_OP = False

try:
    _custom_op_fn = torch.library.custom_op

    # --- fp8_silu_gate_quant ---
    # Input: (*, 2*N) BF16. Output: (FP8 (*, N), float32 scales (*, N//block_size))

    @torch.library.custom_op("futudiffu::fp8_silu_gate_quant", mutates_args=())
    def fp8_silu_gate_quant_op(
        x: torch.Tensor,
        block_size: int,
        dtype_code: int = 3,  # 3 = float8_e4m3fn
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = _CODE_TO_DTYPE_K[dtype_code]
        return fp8_silu_gate_quant(x, block_size=block_size, dtype=dtype)

    @fp8_silu_gate_quant_op.register_fake
    def _fp8_silu_gate_quant_fake(
        x: torch.Tensor,
        block_size: int,
        dtype_code: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = _CODE_TO_DTYPE_K[dtype_code]
        total_cols = x.shape[-1]
        N = total_cols // 2
        SN = N // block_size
        batch_shape = x.shape[:-1]
        y = x.new_empty(*batch_shape, N, dtype=dtype)
        s_y = x.new_empty(*batch_shape, SN, dtype=torch.float32)
        return y, s_y

    # --- fp8_gemm_blockwise ---
    # Input: a (*, K) FP8, a_s (*, K//bs) f32, b (N, K) FP8, b_s (N//bs, K//bs) f32
    # Output: (*, N) in output_dtype

    @torch.library.custom_op("futudiffu::fp8_gemm_blockwise", mutates_args=())
    def fp8_gemm_blockwise_op(
        a: torch.Tensor,
        a_s: torch.Tensor,
        b: torch.Tensor,
        b_s: torch.Tensor,
        input_block_size: int,
        output_dtype_code: int,
    ) -> torch.Tensor:
        output_dtype = _CODE_TO_DTYPE_K[output_dtype_code]
        return fp8_gemm_blockwise(
            a, a_s, b, b_s,
            input_block_size=input_block_size,
            output_dtype=output_dtype,
        )

    @fp8_gemm_blockwise_op.register_fake
    def _fp8_gemm_blockwise_fake(
        a: torch.Tensor,
        a_s: torch.Tensor,
        b: torch.Tensor,
        b_s: torch.Tensor,
        input_block_size: int,
        output_dtype_code: int,
    ) -> torch.Tensor:
        output_dtype = _CODE_TO_DTYPE_K[output_dtype_code]
        N = b.shape[0]
        return a.new_empty(*a.shape[:-1], N, dtype=output_dtype)

    # --- fp8_gemm_v1t ---
    # Input: a (*, K) FP8, a_s (*, K//bs) f32, b (K, N) FP8, b_s (K//bs, N//bs) f32
    # Output: (*, N) in output_dtype

    @torch.library.custom_op("futudiffu::fp8_gemm_v1t", mutates_args=())
    def fp8_gemm_v1t_op(
        a: torch.Tensor,
        a_s: torch.Tensor,
        b: torch.Tensor,
        b_s: torch.Tensor,
        input_block_size: int,
        output_dtype_code: int,
    ) -> torch.Tensor:
        output_dtype = _CODE_TO_DTYPE_K[output_dtype_code]
        return fp8_gemm_v1t(
            a, a_s, b, b_s,
            input_block_size=input_block_size,
            output_dtype=output_dtype,
        )

    @fp8_gemm_v1t_op.register_fake
    def _fp8_gemm_v1t_fake(
        a: torch.Tensor,
        a_s: torch.Tensor,
        b: torch.Tensor,
        b_s: torch.Tensor,
        input_block_size: int,
        output_dtype_code: int,
    ) -> torch.Tensor:
        output_dtype = _CODE_TO_DTYPE_K[output_dtype_code]
        N = b.shape[1]  # weight is [K, N]
        return a.new_empty(*a.shape[:-1], N, dtype=output_dtype)

    # =================================================================
    # Autograd for FFN chain custom_ops (backward through fused chain)
    # =================================================================
    # fp8_silu_gate_quant: STE through FP8 quantization, analytical
    #   silu/gate derivatives.
    # fp8_gemm_blockwise/v1t: dequantize frozen weight for dx = dy @ W.
    # =================================================================

    # --- fp8_silu_gate_quant backward ---
    # Forward: gate = silu(w1) * w3 from x=[w1,w3], then FP8-quantize gate.
    # STE: treat quantization as identity in backward.

    def _fp8_silu_gate_quant_setup(ctx, inputs, output):
        x, block_size, dtype_code = inputs
        ctx.save_for_backward(x)

    def _fp8_silu_gate_quant_backward(ctx, d_gate_fp8, d_gate_scale):
        (x,) = ctx.saved_tensors
        # STE: cast FP8 gradient to input dtype
        d_gate = d_gate_fp8.to(x.dtype)
        N = x.shape[-1] // 2
        w1_out = x[..., :N]
        w3_out = x[..., N:]
        # silu(x) = x * sigmoid(x)
        # silu'(x) = sigmoid(x) * (1 + x - x * sigmoid(x))
        sig = torch.sigmoid(w1_out)
        d_w3 = d_gate * (w1_out * sig)  # d_gate * silu(w1)
        d_w1 = d_gate * w3_out * sig * (1 + w1_out * (1 - sig))
        return torch.cat([d_w1, d_w3], dim=-1), None, None

    fp8_silu_gate_quant_op.register_autograd(
        _fp8_silu_gate_quant_backward,
        setup_context=_fp8_silu_gate_quant_setup,
    )

    # --- fp8_gemm_blockwise backward ---
    # Forward: output = dequant(a, a_s) @ dequant(b, b_s)^T
    # Backward: d_a = d_output @ dequant(b, b_s) (STE for activation quant)

    def _fp8_gemm_bw_setup(ctx, inputs, output):
        a, a_s, b, b_s, input_block_size, output_dtype_code = inputs
        ctx.save_for_backward(b, b_s)
        ctx.block_size = b.shape[0] // b_s.shape[0]  # weight block_size

    def _fp8_gemm_bw_backward(ctx, d_output):
        b, b_s = ctx.saved_tensors
        from .fp8 import dequantize_fp8_blockwise
        # b [N, K] -> dequant [N, K] bf16
        w = dequantize_fp8_blockwise(b, b_s, ctx.block_size)
        # d_a = d_output @ W: [..., N] @ [N, K] -> [..., K]
        d_a = d_output @ w
        return d_a, None, None, None, None, None

    fp8_gemm_blockwise_op.register_autograd(
        _fp8_gemm_bw_backward, setup_context=_fp8_gemm_bw_setup,
    )

    # --- fp8_gemm_v1t backward ---
    # Forward: output = dequant(a, a_s) @ dequant(b, b_s)  [b is K,N]
    # Backward: d_a = d_output @ dequant(b, b_s)^T

    def _fp8_gemm_v1t_setup(ctx, inputs, output):
        a, a_s, b, b_s, input_block_size, output_dtype_code = inputs
        ctx.save_for_backward(b, b_s)
        ctx.block_size = b.shape[0] // b_s.shape[0]

    def _fp8_gemm_v1t_backward(ctx, d_output):
        b, b_s = ctx.saved_tensors
        from .fp8 import dequantize_fp8_blockwise
        # b [K, N] -> dequant [K, N] bf16 -> transpose [N, K]
        w = dequantize_fp8_blockwise(b, b_s, ctx.block_size)
        # d_a = d_output @ W^T: [..., N] @ [N, K] -> [..., K]
        d_a = d_output @ w.t()
        return d_a, None, None, None, None, None

    fp8_gemm_v1t_op.register_autograd(
        _fp8_gemm_v1t_backward, setup_context=_fp8_gemm_v1t_setup,
    )

    _USE_FP8_KERNEL_CUSTOM_OP = True

except (AttributeError, TypeError):
    _USE_FP8_KERNEL_CUSTOM_OP = False


# ==============================================================================
# Standalone test — correctness + benchmark of v1 vs v2 kernels
# ==============================================================================

if __name__ == "__main__":
    import time
    import math

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")

    FP8_DTYPE = torch.float8_e4m3fn
    FP8_MAX = torch.finfo(FP8_DTYPE).max
    BS = 128  # input_block_size

    def quantize_weights_blockwise(w_bf16, block_size=128):
        """Quantize a BF16 weight matrix to FP8 blockwise. Returns (w_fp8, scales).
        w_bf16: (N, K) -> w_fp8: (N, K), scales: (N//bs, K//bs)
        """
        N, K = w_bf16.shape
        n_blocks = math.ceil(N / block_size)
        k_blocks = math.ceil(K / block_size)

        # Pad if needed
        N_pad = n_blocks * block_size
        K_pad = k_blocks * block_size
        w_pad = torch.zeros(N_pad, K_pad, dtype=torch.float32, device=w_bf16.device)
        w_pad[:N, :K] = w_bf16.float()

        # Reshape into blocks
        w_blocks = w_pad.reshape(n_blocks, block_size, k_blocks, block_size)
        w_blocks = w_blocks.permute(0, 2, 1, 3)  # (n_blocks, k_blocks, bs, bs)

        # Per-block amax
        amax = w_blocks.abs().amax(dim=(2, 3))  # (n_blocks, k_blocks)
        amax = amax.clamp(min=1e-12)
        scales = amax / FP8_MAX  # dequant scales

        # Quantize
        quant_scale = FP8_MAX / amax
        w_quant = w_blocks * quant_scale[:, :, None, None]
        w_quant = w_quant.clamp(-FP8_MAX, FP8_MAX)
        w_quant = w_quant.permute(0, 2, 1, 3).reshape(N_pad, K_pad)
        w_fp8 = w_quant[:N, :K].to(FP8_DTYPE)

        return w_fp8, scales

    def run_v1(a_bf16, w_fp8, w_scales):
        """Run old kernel path: act_quant -> fp8_gemm_blockwise."""
        a_fp8, a_s = fp8_act_quant(a_bf16, block_size=BS)
        return fp8_gemm_blockwise(a_fp8, a_s, w_fp8, w_scales, input_block_size=BS)

    def run_v2(a_bf16, w_fp8_t, w_scales_t):
        """Run new kernel: fp8_gemm_v2 with pre-transposed weights."""
        return fp8_gemm_v2(a_bf16, w_fp8_t, w_scales_t, input_block_size=BS)

    def cosine_similarity(a, b):
        a_flat = a.flatten().float()
        b_flat = b.flatten().float()
        return torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()

    def benchmark(fn, warmup=10, repeat=20):
        """Benchmark with CUDA events, returns median time in ms."""
        # Warmup
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        times = []
        for _ in range(repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        times.sort()
        return times[len(times) // 2]  # median

    # Test shapes from the Z-Image model
    test_shapes = [
        ("QKV proj",     8576, 3840, 11520),
        ("Out proj",     8576, 3840, 3840),
        ("W1W3 fused",   8576, 3840, 20480),
        ("W2 proj",      8576, 10240, 3840),
        ("Small square", 1024, 3840, 3840),
    ]

    print("=" * 80)
    print("FP8 GEMM V2 Kernel Test: Correctness + Benchmark")
    print("=" * 80)
    print()

    for name, M, K, N in test_shapes:
        print(f"--- {name}: M={M}, K={K}, N={N} ---")

        # Create random BF16 activations and weights
        a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)

        # Quantize weights to FP8
        w_fp8, w_scales = quantize_weights_blockwise(w_bf16, block_size=BS)
        # w_fp8: (N, K), w_scales: (N//bs, K//bs)

        # Prepare v2 inputs: transpose weight and scales
        w_fp8_t = w_fp8.t().contiguous()   # (K, N)
        w_scales_t = w_scales.t().contiguous()  # (K//bs, N//bs)

        # --- Correctness ---
        out_v1 = run_v1(a_bf16, w_fp8, w_scales)
        out_v2 = run_v2(a_bf16, w_fp8_t, w_scales_t)

        cos_sim = cosine_similarity(out_v1, out_v2)
        max_diff = (out_v1.float() - out_v2.float()).abs().max().item()
        mean_diff = (out_v1.float() - out_v2.float()).abs().mean().item()

        print(f"  Cosine similarity: {cos_sim:.6f}")
        print(f"  Max abs diff:      {max_diff:.4f}")
        print(f"  Mean abs diff:     {mean_diff:.6f}")

        # Also compare against PyTorch reference (dequant weights -> bf16 matmul)
        # Dequantize weights for reference
        n_blocks = math.ceil(N / BS)
        k_blocks = math.ceil(K / BS)
        N_pad = n_blocks * BS
        K_pad = k_blocks * BS
        w_deq = torch.zeros(N_pad, K_pad, dtype=torch.float32, device=device)
        w_deq[:N, :K] = w_fp8.float()
        w_deq_blocks = w_deq.reshape(n_blocks, BS, k_blocks, BS).permute(0, 2, 1, 3)
        w_deq_blocks = w_deq_blocks * w_scales[:, :, None, None]
        w_deq = w_deq_blocks.permute(0, 2, 1, 3).reshape(N_pad, K_pad)[:N, :K]
        ref = a_bf16.float() @ w_deq.t()  # (M, N)

        cos_v1_ref = cosine_similarity(out_v1, ref)
        cos_v2_ref = cosine_similarity(out_v2, ref)
        print(f"  V1 vs reference:   cos={cos_v1_ref:.6f}")
        print(f"  V2 vs reference:   cos={cos_v2_ref:.6f}")

        # --- Benchmark ---
        t_v1 = benchmark(lambda: run_v1(a_bf16, w_fp8, w_scales))
        t_v2 = benchmark(lambda: run_v2(a_bf16, w_fp8_t, w_scales_t))

        # Compute TFLOPS
        flops = 2 * M * N * K  # standard GEMM FLOPs
        tflops_v1 = flops / (t_v1 * 1e-3) / 1e12
        tflops_v2 = flops / (t_v2 * 1e-3) / 1e12

        speedup = t_v1 / t_v2
        print(f"  V1 time: {t_v1:.3f} ms  ({tflops_v1:.1f} TFLOPS)")
        print(f"  V2 time: {t_v2:.3f} ms  ({tflops_v2:.1f} TFLOPS)")
        print(f"  Speedup: {speedup:.2f}x")
        print()

    # --- Benchmark addmm (with bias) ---
    print("=" * 80)
    print("FP8 ADDMM V2 Test (with bias)")
    print("=" * 80)
    print()

    M, K, N = 8576, 3840, 3840
    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    bias = torch.randn(N, dtype=torch.bfloat16, device=device)

    w_fp8, w_scales = quantize_weights_blockwise(w_bf16, block_size=BS)
    w_fp8_t = w_fp8.t().contiguous()
    w_scales_t = w_scales.t().contiguous()

    # V1 addmm
    a_fp8, a_s = fp8_act_quant(a_bf16, block_size=BS)
    out_v1_bias = fp8_addmm_blockwise(a_fp8, a_s, w_fp8, w_scales, bias=bias, input_block_size=BS)

    # V2 addmm
    out_v2_bias = fp8_addmm_v2(a_bf16, w_fp8_t, w_scales_t, bias=bias, input_block_size=BS)

    cos_sim = cosine_similarity(out_v1_bias, out_v2_bias)
    max_diff = (out_v1_bias.float() - out_v2_bias.float()).abs().max().item()
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Max abs diff:      {max_diff:.4f}")

    t_v1 = benchmark(lambda: fp8_addmm_blockwise(a_fp8, a_s, w_fp8, w_scales, bias=bias, input_block_size=BS))
    t_v2 = benchmark(lambda: fp8_addmm_v2(a_bf16, w_fp8_t, w_scales_t, bias=bias, input_block_size=BS))
    print(f"  V1 addmm time: {t_v1:.3f} ms")
    print(f"  V2 addmm time: {t_v2:.3f} ms")
    print(f"  Speedup: {t_v1/t_v2:.2f}x")
    print()

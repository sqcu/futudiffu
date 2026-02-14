"""FP8 quantization support for blockwise FP8 models.

Sources:
- QuantOps-reference/fp8_ops.py (HybridFP8Ops pattern)
- QuantOps-reference/quant_layouts/fp8_variants.py (BlockWiseFP8Layout)

This module provides:
1. FP8Linear: A drop-in nn.Linear replacement that stores weights in FP8
   with blockwise scales and uses Triton kernels for matmul.
2. load_fp8_state_dict: Loads a safetensors file with .comfy_quant metadata
   and creates FP8Linear layers in place of regular Linear layers.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file

from .fp8_kernels import (
    fp8_act_quant,
    fp8_addmm_blockwise,
    fp8_gemm_blockwise,
    fp8_addmm_v1t,
    fp8_gemm_v1t,
)


BLOCK_SIZE = 128

# Dtype encoding for custom_op args (torch.dtype cannot be passed directly).
_DTYPE_TO_CODE = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}
_CODE_TO_DTYPE = {0: torch.bfloat16, 1: torch.float16, 2: torch.float32}


# ---------------------------------------------------------------------------
# Custom ops for FP8 linear (torch >= 2.4)
# ---------------------------------------------------------------------------
# These allow torch.compile(mode="reduce-overhead") to capture FP8Linear calls
# into CUDA graphs, eliminating ~100ms of Python dispatch overhead per forward
# pass that the old @torch.compiler.disable workaround imposed.
# ---------------------------------------------------------------------------

_USE_FP8_CUSTOM_OP = False

try:
    # Probe for the custom_op decorator
    _custom_op_fn = torch.library.custom_op

    @torch.library.custom_op("futudiffu::fp8_linear", mutates_args=())
    def fp8_linear_op(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        block_size: int,
        output_dtype_code: int,
    ) -> torch.Tensor:
        """FP8 linear without bias: quantize activations + FP8 GEMM."""
        output_dtype = _CODE_TO_DTYPE[output_dtype_code]

        original_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1]).contiguous()
        x_fp8, x_scale = fp8_act_quant(x_flat, block_size=block_size)
        x_fp8 = x_fp8.reshape(original_shape[:-1] + (x_fp8.shape[-1],))
        x_scale = x_scale.reshape(original_shape[:-1] + (x_scale.shape[-1],))

        out = fp8_gemm_blockwise(
            x_fp8, x_scale, weight, weight_scale,
            input_block_size=block_size,
            output_dtype=output_dtype,
        )
        return out

    @fp8_linear_op.register_fake
    def _fp8_linear_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        block_size: int,
        output_dtype_code: int,
    ) -> torch.Tensor:
        output_dtype = _CODE_TO_DTYPE[output_dtype_code]
        N = weight.shape[0]
        # CRITICAL: use x.new_empty() with explicit shape, NOT torch.empty_like().
        # empty_like preserves non-contiguous strides which causes CUDA graph
        # assertion failures in torch.compile.
        return x.new_empty(*x.shape[:-1], N, dtype=output_dtype)

    @torch.library.custom_op("futudiffu::fp8_linear_bias", mutates_args=())
    def fp8_linear_bias_op(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor,
        block_size: int,
        output_dtype_code: int,
    ) -> torch.Tensor:
        """FP8 linear with bias: quantize activations + FP8 GEMM + bias."""
        output_dtype = _CODE_TO_DTYPE[output_dtype_code]

        original_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1]).contiguous()
        x_fp8, x_scale = fp8_act_quant(x_flat, block_size=block_size)
        x_fp8 = x_fp8.reshape(original_shape[:-1] + (x_fp8.shape[-1],))
        x_scale = x_scale.reshape(original_shape[:-1] + (x_scale.shape[-1],))

        out = fp8_addmm_blockwise(
            x_fp8, x_scale, weight, weight_scale,
            bias=bias, input_block_size=block_size,
            output_dtype=output_dtype,
        )
        return out

    @fp8_linear_bias_op.register_fake
    def _fp8_linear_bias_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor,
        block_size: int,
        output_dtype_code: int,
    ) -> torch.Tensor:
        output_dtype = _CODE_TO_DTYPE[output_dtype_code]
        N = weight.shape[0]
        return x.new_empty(*x.shape[:-1], N, dtype=output_dtype)

    # --- V1T custom_ops: transposed weight layout [K, N] ---

    @torch.library.custom_op("futudiffu::fp8_linear_v1t", mutates_args=())
    def fp8_linear_v1t_op(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        block_size: int,
        output_dtype_code: int,
    ) -> torch.Tensor:
        """FP8 linear with transposed weight [K,N]: quantize activations + v1t GEMM."""
        output_dtype = _CODE_TO_DTYPE[output_dtype_code]

        original_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1]).contiguous()
        x_fp8, x_scale = fp8_act_quant(x_flat, block_size=block_size)
        x_fp8 = x_fp8.reshape(original_shape[:-1] + (x_fp8.shape[-1],))
        x_scale = x_scale.reshape(original_shape[:-1] + (x_scale.shape[-1],))

        out = fp8_gemm_v1t(
            x_fp8, x_scale, weight, weight_scale,
            input_block_size=block_size,
            output_dtype=output_dtype,
        )
        return out

    @fp8_linear_v1t_op.register_fake
    def _fp8_linear_v1t_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        block_size: int,
        output_dtype_code: int,
    ) -> torch.Tensor:
        output_dtype = _CODE_TO_DTYPE[output_dtype_code]
        N = weight.shape[1]  # weight is [K, N]
        return x.new_empty(*x.shape[:-1], N, dtype=output_dtype)

    @torch.library.custom_op("futudiffu::fp8_linear_bias_v1t", mutates_args=())
    def fp8_linear_bias_v1t_op(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor,
        block_size: int,
        output_dtype_code: int,
    ) -> torch.Tensor:
        """FP8 linear with transposed weight [K,N] + bias."""
        output_dtype = _CODE_TO_DTYPE[output_dtype_code]

        original_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1]).contiguous()
        x_fp8, x_scale = fp8_act_quant(x_flat, block_size=block_size)
        x_fp8 = x_fp8.reshape(original_shape[:-1] + (x_fp8.shape[-1],))
        x_scale = x_scale.reshape(original_shape[:-1] + (x_scale.shape[-1],))

        out = fp8_addmm_v1t(
            x_fp8, x_scale, weight, weight_scale,
            bias=bias, input_block_size=block_size,
            output_dtype=output_dtype,
        )
        return out

    @fp8_linear_bias_v1t_op.register_fake
    def _fp8_linear_bias_v1t_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor,
        block_size: int,
        output_dtype_code: int,
    ) -> torch.Tensor:
        output_dtype = _CODE_TO_DTYPE[output_dtype_code]
        N = weight.shape[1]  # weight is [K, N]
        return x.new_empty(*x.shape[:-1], N, dtype=output_dtype)

    _USE_FP8_CUSTOM_OP = True

except (AttributeError, TypeError):
    _USE_FP8_CUSTOM_OP = False


class FP8Linear(nn.Module):
    """Linear layer with FP8 blockwise quantized weights.

    Weights are stored as FP8 (float8_e4m3fn) with 2D blockwise scales
    of shape [N//block_size, K//block_size].
    """

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        block_size: int = BLOCK_SIZE,
        output_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.register_buffer("weight", weight)  # (N, K) in FP8
        self.register_buffer("weight_scale", weight_scale)  # (N//bs, K//bs) in float32
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None
        self.block_size = block_size
        self.output_dtype = output_dtype
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self._transposed = False

    def transpose_weight(self) -> None:
        """Pre-transpose weight [N,K] -> [K,N] and scale [N//bs,K//bs] -> [K//bs,N//bs].

        This enables coalesced memory access in the GEMM kernel (v1t path).
        Must be called once after model loading, before inference.
        """
        if self._transposed:
            return
        # weight: (N, K) FP8 -> (K, N) FP8
        self.weight = nn.Parameter(
            self.weight.data.t().contiguous(), requires_grad=False,
        )
        # weight_scale: (N//bs, K//bs) float32 -> (K//bs, N//bs) float32
        self.weight_scale = nn.Parameter(
            self.weight_scale.data.t().contiguous(), requires_grad=False,
        )
        self._transposed = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype_code = _DTYPE_TO_CODE[self.output_dtype]

        if self._transposed:
            return self._forward_v1t(x, dtype_code)
        elif _USE_FP8_CUSTOM_OP:
            return self._forward_custom_op(x, dtype_code)
        else:
            return self._forward_eager(x)

    def _forward_custom_op(self, x: torch.Tensor, dtype_code: int) -> torch.Tensor:
        """Custom_op path for v1 (original [N,K] weight layout)."""
        if self.bias is not None:
            return fp8_linear_bias_op(
                x, self.weight, self.weight_scale,
                self.bias, self.block_size, dtype_code,
            )
        else:
            return fp8_linear_op(
                x, self.weight, self.weight_scale,
                self.block_size, dtype_code,
            )

    def _forward_v1t(self, x: torch.Tensor, dtype_code: int) -> torch.Tensor:
        """V1T path: pre-transposed [K,N] weight, coalesced access."""
        if _USE_FP8_CUSTOM_OP:
            if self.bias is not None:
                return fp8_linear_bias_v1t_op(
                    x, self.weight, self.weight_scale,
                    self.bias, self.block_size, dtype_code,
                )
            else:
                return fp8_linear_v1t_op(
                    x, self.weight, self.weight_scale,
                    self.block_size, dtype_code,
                )
        else:
            # Eager fallback for v1t
            original_shape = x.shape
            x_flat = x.reshape(-1, x.shape[-1]).contiguous()
            x_fp8, x_scale = fp8_act_quant(x_flat, block_size=self.block_size)
            x_fp8 = x_fp8.reshape(original_shape[:-1] + (x_fp8.shape[-1],))
            x_scale = x_scale.reshape(original_shape[:-1] + (x_scale.shape[-1],))

            if self.bias is not None:
                out = fp8_addmm_v1t(
                    x_fp8, x_scale, self.weight, self.weight_scale,
                    bias=self.bias, input_block_size=self.block_size,
                    output_dtype=self.output_dtype,
                )
            else:
                out = fp8_gemm_v1t(
                    x_fp8, x_scale, self.weight, self.weight_scale,
                    input_block_size=self.block_size,
                    output_dtype=self.output_dtype,
                )
            return out

    def _forward_eager(self, x: torch.Tensor) -> torch.Tensor:
        """Eager mode fallback for v1 (original [N,K] weight layout)."""
        original_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1]).contiguous()
        x_fp8, x_scale = fp8_act_quant(x_flat, block_size=self.block_size)
        x_fp8 = x_fp8.reshape(original_shape[:-1] + (x_fp8.shape[-1],))
        x_scale = x_scale.reshape(original_shape[:-1] + (x_scale.shape[-1],))

        if self.bias is not None:
            out = fp8_addmm_blockwise(
                x_fp8, x_scale, self.weight, self.weight_scale,
                bias=self.bias, input_block_size=self.block_size,
                output_dtype=self.output_dtype,
            )
        else:
            out = fp8_gemm_blockwise(
                x_fp8, x_scale, self.weight, self.weight_scale,
                input_block_size=self.block_size,
                output_dtype=self.output_dtype,
            )
        return out


def dequantize_fp8_blockwise(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP8 blockwise weight to full precision.

    Args:
        weight: (N, K) FP8 tensor.
        scale: (N//bs, K//bs) float32 tensor.
        block_size: Block size used for quantization.
        output_dtype: Output dtype.

    Returns:
        (N, K) dequantized tensor.
    """
    N, K = weight.shape
    weight_f32 = weight.to(torch.float32)

    # Reshape to blocks
    n_blocks = N // block_size
    k_blocks = K // block_size

    weight_blocks = weight_f32.reshape(n_blocks, block_size, k_blocks, block_size)
    scale_expanded = scale.reshape(n_blocks, 1, k_blocks, 1)

    dequantized = weight_blocks * scale_expanded
    return dequantized.reshape(N, K).to(output_dtype)


# ---------------------------------------------------------------------------
# Autograd for FP8 custom_ops (backward through frozen FP8 weights)
# ---------------------------------------------------------------------------
# Forward uses FP8 tensor cores; backward dequantizes to BF16 for dx = dy @ W.
# Only input gradients are computed (weights/scales are frozen buffers).
# The dequantized weight is a temporary (~150MB for largest layer) freed after
# each matmul. With per-block gradient checkpointing, only one layer's
# temporaries exist at a time.
# ---------------------------------------------------------------------------

if _USE_FP8_CUSTOM_OP:

    # --- fp8_linear [N,K] weight, no bias ---

    def _fp8_linear_setup(ctx, inputs, output):
        x, weight, weight_scale, block_size, output_dtype_code = inputs
        ctx.save_for_backward(weight, weight_scale)
        ctx.block_size = block_size

    def _fp8_linear_backward(ctx, grad_output):
        weight, weight_scale = ctx.saved_tensors
        # weight [N, K] -> dequant [N, K] bf16
        w = dequantize_fp8_blockwise(weight, weight_scale, ctx.block_size)
        # dx = grad @ W: [..., N] @ [N, K] -> [..., K]
        return grad_output @ w, None, None, None, None

    fp8_linear_op.register_autograd(
        _fp8_linear_backward, setup_context=_fp8_linear_setup,
    )

    # --- fp8_linear_bias [N,K] weight, with bias ---

    def _fp8_linear_bias_setup(ctx, inputs, output):
        x, weight, weight_scale, bias, block_size, output_dtype_code = inputs
        ctx.save_for_backward(weight, weight_scale)
        ctx.block_size = block_size

    def _fp8_linear_bias_backward(ctx, grad_output):
        weight, weight_scale = ctx.saved_tensors
        w = dequantize_fp8_blockwise(weight, weight_scale, ctx.block_size)
        return grad_output @ w, None, None, None, None, None

    fp8_linear_bias_op.register_autograd(
        _fp8_linear_bias_backward, setup_context=_fp8_linear_bias_setup,
    )

    # --- fp8_linear_v1t [K,N] weight, no bias ---

    def _fp8_linear_v1t_setup(ctx, inputs, output):
        x, weight, weight_scale, block_size, output_dtype_code = inputs
        ctx.save_for_backward(weight, weight_scale)
        ctx.block_size = block_size

    def _fp8_linear_v1t_backward(ctx, grad_output):
        weight, weight_scale = ctx.saved_tensors
        # weight [K, N] -> dequant [K, N] bf16 -> transpose [N, K]
        w = dequantize_fp8_blockwise(weight, weight_scale, ctx.block_size)
        # dx = grad @ W^T: [..., N] @ [N, K] -> [..., K]
        return grad_output @ w.t(), None, None, None, None

    fp8_linear_v1t_op.register_autograd(
        _fp8_linear_v1t_backward, setup_context=_fp8_linear_v1t_setup,
    )

    # --- fp8_linear_bias_v1t [K,N] weight, with bias ---

    def _fp8_linear_bias_v1t_setup(ctx, inputs, output):
        x, weight, weight_scale, bias, block_size, output_dtype_code = inputs
        ctx.save_for_backward(weight, weight_scale)
        ctx.block_size = block_size

    def _fp8_linear_bias_v1t_backward(ctx, grad_output):
        weight, weight_scale = ctx.saved_tensors
        w = dequantize_fp8_blockwise(weight, weight_scale, ctx.block_size)
        return grad_output @ w.t(), None, None, None, None, None

    fp8_linear_bias_v1t_op.register_autograd(
        _fp8_linear_bias_v1t_backward, setup_context=_fp8_linear_bias_v1t_setup,
    )


def replace_linear_with_fp8(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str = "",
    block_size: int = BLOCK_SIZE,
    output_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Walk a module tree and replace Linear layers with FP8Linear where
    the state_dict has FP8 weights with scales.

    The QuantOps convention is:
    - weight: (N, K) in float8_e4m3fn
    - weight_scale: (N//bs, K//bs) in float32
    - bias: (N,) in float32 (optional)
    """
    for name, child in list(module.named_children()):
        full_name = f"{prefix}{name}" if prefix else name
        weight_key = f"{full_name}.weight"
        scale_key = f"{full_name}.weight_scale"

        if isinstance(child, nn.Linear) and weight_key in state_dict:
            weight = state_dict[weight_key]
            if weight.dtype == torch.float8_e4m3fn and scale_key in state_dict:
                scale = state_dict[scale_key]
                bias = state_dict.get(f"{full_name}.bias", None)
                fp8_linear = FP8Linear(
                    weight=weight,
                    weight_scale=scale,
                    bias=bias,
                    block_size=block_size,
                    output_dtype=output_dtype,
                )
                setattr(module, name, fp8_linear)
                continue

        replace_linear_with_fp8(child, state_dict, f"{full_name}.", block_size, output_dtype)


def test_fp8_linear_custom_op():
    """Test that custom_op FP8Linear matches eager mode."""
    import time

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"_USE_FP8_CUSTOM_OP = {_USE_FP8_CUSTOM_OP}")

    # Create a simple FP8Linear with simulated FP8 weights.
    # Use dimensions matching the actual model (NextDiT hidden=3840).
    K, N = 3840, 10240

    # Simulate FP8 weights by quantizing random bf16 data
    w_bf16 = torch.randn(N, K, device=device, dtype=dtype)
    # Quantize row by row in blocks of 128
    w_flat = w_bf16.reshape(-1, 128)
    w_fp8, w_scale = fp8_act_quant(w_flat)
    w_fp8 = w_fp8.reshape(N, K)
    w_scale = w_scale.reshape(N, K // 128)

    linear = FP8Linear(w_fp8, w_scale, bias=None, block_size=128, output_dtype=dtype)
    linear = linear.to(device)

    x = torch.randn(2, 4288, K, device=device, dtype=dtype)

    # --- Test 1: eager vs custom_op produce same output ---
    out_eager = linear._forward_eager(x)
    out_custom = linear(x)

    cos = torch.nn.functional.cosine_similarity(
        out_eager.flatten().float(), out_custom.flatten().float(), dim=0,
    )
    print(f"Test 1 - Cosine similarity (eager vs custom_op): {cos:.10f}")
    assert cos > 0.9999, f"Eager vs custom_op too divergent: {cos}"
    print("  PASSED")

    # --- Test 2: torch.compile with reduce-overhead ---
    print("\nTest 2 - torch.compile(mode='reduce-overhead')...")
    linear_compiled = torch.compile(linear, mode="reduce-overhead")

    # Warmup (triggers Triton autotune + compilation + CUDA graph capture)
    for i in range(3):
        _ = linear_compiled(x)
        torch.cuda.synchronize()
        print(f"  Warmup {i+1}/3 done")

    out_compiled = linear_compiled(x)
    torch.cuda.synchronize()

    cos2 = torch.nn.functional.cosine_similarity(
        out_eager.flatten().float(), out_compiled.flatten().float(), dim=0,
    )
    print(f"  Cosine similarity (eager vs compiled): {cos2:.10f}")
    assert cos2 > 0.9999, f"Eager vs compiled too divergent: {cos2}"
    print("  PASSED")

    # --- Test 3: with bias ---
    print("\nTest 3 - FP8Linear with bias...")
    bias = torch.randn(N, device=device, dtype=dtype)
    linear_bias = FP8Linear(w_fp8, w_scale, bias=bias, block_size=128, output_dtype=dtype)
    linear_bias = linear_bias.to(device)

    out_eager_b = linear_bias._forward_eager(x)
    out_custom_b = linear_bias(x)

    cos3 = torch.nn.functional.cosine_similarity(
        out_eager_b.flatten().float(), out_custom_b.flatten().float(), dim=0,
    )
    print(f"  Cosine similarity (eager vs custom_op, bias): {cos3:.10f}")
    assert cos3 > 0.9999, f"Eager vs custom_op (bias) too divergent: {cos3}"
    print("  PASSED")

    # --- Test 4: Timing comparison ---
    print("\nTest 4 - Timing (eager vs compiled)...")
    torch.cuda.synchronize()

    # Time eager
    t0 = time.perf_counter()
    for _ in range(5):
        _ = linear._forward_eager(x)
    torch.cuda.synchronize()
    t_eager = (time.perf_counter() - t0) / 5

    # Time compiled (already warmed up)
    t0 = time.perf_counter()
    for _ in range(5):
        _ = linear_compiled(x)
    torch.cuda.synchronize()
    t_compiled = (time.perf_counter() - t0) / 5

    print(f"  Eager:    {t_eager*1000:.2f} ms")
    print(f"  Compiled: {t_compiled*1000:.2f} ms")
    print(f"  Speedup:  {t_eager/t_compiled:.2f}x")

    print("\nAll tests passed!")

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
)


BLOCK_SIZE = 128


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

    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize activations to FP8 blockwise
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

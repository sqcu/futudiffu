"""Sparse multi-LoRA Triton kernel — skip zero-gated adapters without warp divergence.

Computes the LoRA delta for N named adapters sharing a base linear layer:

    out[b, s, d] = sum_i (scale[b, i] * (x[b, s, :] @ A[i].T) @ B[i].T)[d]

The scale tensor is loaded once per (batch_element, adapter) — all threads in a
program instance see the same value, so the `if scale != 0` branch is uniform
(zero warp divergence). When an adapter's scale is zero for a given batch element,
both the A-projection and B-projection matmuls for that adapter are skipped entirely.

Pre-concatenated adapter weights:
    A_all: (N_ADAPTERS, rank, in_features) contiguous BF16
    B_all: (N_ADAPTERS, out_features, rank) contiguous BF16
    scale_all: (batch_size, N_ADAPTERS) contiguous float32  (alpha/rank pre-folded)
    x: (batch_size, seq_len, in_features) contiguous BF16
    out: (batch_size, seq_len, out_features) contiguous BF16 — LoRA delta

Grid: (cdiv(seq_len, BLOCK_M), cdiv(out_features, BLOCK_N), batch_size)
    Fully static, CUDA-graph safe.

N_ADAPTERS and RANK are tl.constexpr (loop is unrolled at compile time).

Usage:
    delta = multi_lora_forward(x, A_all, B_all, scale_all, out_features)
    y = base_linear(x) + delta

    # Or via custom_op (torch.compile compatible):
    delta = multi_lora_op(x, A_all, B_all, scale_all, N_ADAPTERS, RANK)
"""

from __future__ import annotations

import time
from typing import Optional

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


if _HAS_TRITON:

    # =========================================================================
    # Triton kernel: sparse multi-LoRA with conditional adapter skip
    # =========================================================================

    @triton.jit
    def _multi_lora_kernel(
        # Pointers
        x_ptr,          # (B, S, IN) bf16
        A_ptr,          # (N_ADAPTERS, RANK, IN) bf16
        B_ptr,          # (N_ADAPTERS, OUT, RANK) bf16
        scale_ptr,      # (B, N_ADAPTERS) float32
        out_ptr,        # (B, S, OUT) bf16
        # Dimensions (runtime)
        seq_len,
        in_features,
        out_features,
        # Strides for x: (B, S, IN) row-major
        stride_x_b,     # = S * IN
        stride_x_s,     # = IN
        # Strides for A: (N_ADAPTERS, RANK, IN) row-major
        stride_A_n,     # = RANK * IN
        stride_A_r,     # = IN
        # Strides for B: (N_ADAPTERS, OUT, RANK) row-major
        stride_B_n,     # = OUT * RANK
        stride_B_o,     # = RANK
        # Strides for out: (B, S, OUT) row-major
        stride_out_b,   # = S * OUT
        stride_out_s,   # = OUT
        # Constexpr tile sizes and adapter config
        N_ADAPTERS: tl.constexpr,
        RANK: tl.constexpr,
        BLOCK_M: tl.constexpr,     # seq tile
        BLOCK_N: tl.constexpr,     # output feature tile
        BLOCK_K: tl.constexpr,     # reduction tile for in_features
    ):
        # Program indices
        pid_m = tl.program_id(0)   # seq tile index
        pid_n = tl.program_id(1)   # output feature tile index
        pid_b = tl.program_id(2)   # batch index

        # Offsets for this tile
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # (BLOCK_N,)
        offs_r = tl.arange(0, RANK)                        # (RANK,)

        # Mask for seq and output bounds
        mask_m = offs_m < seq_len
        mask_n = offs_n < out_features

        # Accumulator for final output: (BLOCK_M, BLOCK_N) in float32
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Base pointer for x[batch]: x_ptr + pid_b * stride_x_b
        x_batch_ptr = x_ptr + pid_b * stride_x_b

        # Base pointer for scale[batch]: scale_ptr + pid_b * N_ADAPTERS
        scale_batch_ptr = scale_ptr + pid_b * N_ADAPTERS

        # Loop over adapters (unrolled at compile time since N_ADAPTERS is constexpr)
        for adapter_idx in range(N_ADAPTERS):
            # Load scale for this (batch, adapter) — scalar, uniform across all threads
            s = tl.load(scale_batch_ptr + adapter_idx)

            # Uniform branch: all threads in this program see the same scale value.
            # If scale is zero, skip both matmuls entirely — no warp divergence.
            if s != 0.0:
                # ---- Phase 1: mid = x_tile @ A_i^T -> (BLOCK_M, RANK) ----
                # A_i has shape (RANK, in_features), we want x @ A_i^T
                # which is (BLOCK_M, in_features) @ (in_features, RANK)
                mid = tl.zeros((BLOCK_M, RANK), dtype=tl.float32)

                A_adapter_ptr = A_ptr + adapter_idx * stride_A_n

                for k_start in range(0, in_features, BLOCK_K):
                    offs_k = k_start + tl.arange(0, BLOCK_K)
                    mask_k = offs_k < in_features

                    # x_tile: (BLOCK_M, BLOCK_K)
                    x_ptrs = x_batch_ptr + offs_m[:, None] * stride_x_s + offs_k[None, :]
                    x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

                    # a_tile: (RANK, BLOCK_K) — load A_i[r, k]
                    a_ptrs = A_adapter_ptr + offs_r[:, None] * stride_A_r + offs_k[None, :]
                    a_tile = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)

                    # mid += x_tile @ a_tile^T = (BLOCK_M, BLOCK_K) @ (BLOCK_K, RANK)
                    mid += tl.dot(x_tile, tl.trans(a_tile), out_dtype=tl.float32)

                # ---- Phase 2: acc += s * mid @ B_i_block^T -> (BLOCK_M, BLOCK_N) ----
                # B_i has shape (out_features, RANK). We only need the output tile:
                # B_i[offs_n, :] which is (BLOCK_N, RANK)
                B_adapter_ptr = B_ptr + adapter_idx * stride_B_n
                b_ptrs = B_adapter_ptr + offs_n[:, None] * stride_B_o + offs_r[None, :]
                b_tile = tl.load(b_ptrs, mask=mask_n[:, None], other=0.0)

                # mid_bf16 @ b_tile^T = (BLOCK_M, RANK) @ (RANK, BLOCK_N)
                mid_bf16 = mid.to(tl.bfloat16)
                contrib = tl.dot(mid_bf16, tl.trans(b_tile), out_dtype=tl.float32)

                acc += s * contrib

        # Store output: (BLOCK_M, BLOCK_N) -> out[batch, seq_block, out_block]
        out_batch_ptr = out_ptr + pid_b * stride_out_b
        out_ptrs = out_batch_ptr + offs_m[:, None] * stride_out_s + offs_n[None, :]
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


# =============================================================================
# Python wrapper
# =============================================================================

def multi_lora_forward(
    x: Tensor,
    A_all: Tensor,
    B_all: Tensor,
    scale_all: Tensor,
    *,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 64,
    num_warps: int = 4,
    num_stages: int = 1,
) -> Tensor:
    """Compute sparse multi-LoRA delta via Triton kernel.

    Args:
        x: Input activations, (B, seq_len, in_features) contiguous BF16.
        A_all: Stacked A matrices, (N_ADAPTERS, rank, in_features) contiguous BF16.
        B_all: Stacked B matrices, (N_ADAPTERS, out_features, rank) contiguous BF16.
        scale_all: Per-batch per-adapter scales, (B, N_ADAPTERS) contiguous float32.
            Includes alpha/rank pre-folded. Zero means skip.
        BLOCK_M: Tile size along seq dimension.
        BLOCK_N: Tile size along output feature dimension.
        BLOCK_K: Tile size for in_features reduction.
        num_warps: Warps per program instance.
        num_stages: Pipeline stages.

    Returns:
        LoRA delta tensor, (B, seq_len, out_features) BF16.
    """
    if not _HAS_TRITON:
        raise RuntimeError("Triton is required for multi_lora_forward")

    # Validate shapes
    assert x.ndim == 3, f"x must be 3D (B, S, IN), got {x.ndim}D"
    assert A_all.ndim == 3, f"A_all must be 3D (N, R, IN), got {A_all.ndim}D"
    assert B_all.ndim == 3, f"B_all must be 3D (N, OUT, R), got {B_all.ndim}D"
    assert scale_all.ndim == 2, f"scale_all must be 2D (B, N), got {scale_all.ndim}D"

    B, seq_len, in_features = x.shape
    N_ADAPTERS, rank, in_features_A = A_all.shape
    N_ADAPTERS_B, out_features, rank_B = B_all.shape

    assert in_features == in_features_A, (
        f"in_features mismatch: x has {in_features}, A has {in_features_A}")
    assert N_ADAPTERS == N_ADAPTERS_B, (
        f"N_ADAPTERS mismatch: A has {N_ADAPTERS}, B has {N_ADAPTERS_B}")
    assert rank == rank_B, (
        f"rank mismatch: A has {rank}, B has {rank_B}")
    assert scale_all.shape == (B, N_ADAPTERS), (
        f"scale_all shape mismatch: expected ({B}, {N_ADAPTERS}), got {scale_all.shape}")

    # Dtype checks
    assert x.dtype == torch.bfloat16, f"x must be BF16, got {x.dtype}"
    assert A_all.dtype == torch.bfloat16, f"A_all must be BF16, got {A_all.dtype}"
    assert B_all.dtype == torch.bfloat16, f"B_all must be BF16, got {B_all.dtype}"
    assert scale_all.dtype == torch.float32, f"scale_all must be float32, got {scale_all.dtype}"

    # Contiguity
    assert x.is_contiguous(), "x must be contiguous"
    assert A_all.is_contiguous(), "A_all must be contiguous"
    assert B_all.is_contiguous(), "B_all must be contiguous"
    assert scale_all.is_contiguous(), "scale_all must be contiguous"

    # Rank must be a power of 2 for tl.dot (minimum 16 on SM89)
    # If rank < 16, we pad A and B to 16
    padded_rank = max(16, 1 << (rank - 1).bit_length())  # next power of 2, min 16
    if padded_rank != rank:
        # Pad A: (N, rank, IN) -> (N, padded_rank, IN)
        A_padded = torch.zeros(
            N_ADAPTERS, padded_rank, in_features,
            dtype=torch.bfloat16, device=x.device)
        A_padded[:, :rank, :] = A_all
        A_all = A_padded

        # Pad B: (N, OUT, rank) -> (N, OUT, padded_rank)
        B_padded = torch.zeros(
            N_ADAPTERS, out_features, padded_rank,
            dtype=torch.bfloat16, device=x.device)
        B_padded[:, :, :rank] = B_all
        B_all = B_padded
        rank = padded_rank

    # Allocate output
    out = torch.empty(B, seq_len, out_features, dtype=torch.bfloat16, device=x.device)

    # Grid: (ceil(seq_len/BLOCK_M), ceil(out_features/BLOCK_N), batch_size)
    grid = (
        triton.cdiv(seq_len, BLOCK_M),
        triton.cdiv(out_features, BLOCK_N),
        B,
    )

    _multi_lora_kernel[grid](
        x, A_all, B_all, scale_all, out,
        seq_len, in_features, out_features,
        # x strides
        x.stride(0), x.stride(1),
        # A strides
        A_all.stride(0), A_all.stride(1),
        # B strides
        B_all.stride(0), B_all.stride(1),
        # out strides
        out.stride(0), out.stride(1),
        # Constexpr
        N_ADAPTERS=N_ADAPTERS,
        RANK=rank,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out


# =============================================================================
# PyTorch reference implementation (for testing)
# =============================================================================

def multi_lora_forward_ref(
    x: Tensor,
    A_all: Tensor,
    B_all: Tensor,
    scale_all: Tensor,
) -> Tensor:
    """Reference PyTorch implementation of multi-LoRA forward.

    Args:
        x: (B, S, IN) BF16
        A_all: (N, R, IN) BF16
        B_all: (N, OUT, R) BF16
        scale_all: (B, N) float32

    Returns:
        (B, S, OUT) BF16
    """
    B, S, IN = x.shape
    N, R, _ = A_all.shape
    _, OUT, _ = B_all.shape

    out = torch.zeros(B, S, OUT, dtype=torch.bfloat16, device=x.device)

    for i in range(N):
        # (B, S, IN) @ (IN, R) -> (B, S, R)
        mid = x @ A_all[i].t()
        # (B, S, R) @ (R, OUT) -> (B, S, OUT)
        lora_out = mid @ B_all[i].t()
        # scale: (B, 1, 1)
        s = scale_all[:, i].to(torch.bfloat16).unsqueeze(-1).unsqueeze(-1)
        out = out + s * lora_out

    return out


# =============================================================================
# Cat-based approach (for benchmarking baseline)
# =============================================================================

def multi_lora_forward_cat(
    x: Tensor,
    A_all: Tensor,
    B_all: Tensor,
    scale_all: Tensor,
) -> Tensor:
    """Cat-based multi-LoRA: 2 dense matmuls, no sparsity.

    Benchmarking baseline only — NOT used in production.
    All adapters are concatenated and computed together regardless of scale.

    Args:
        x: (B, S, IN) BF16
        A_all: (N, R, IN) BF16
        B_all: (N, OUT, R) BF16
        scale_all: (B, N) float32

    Returns:
        (B, S, OUT) BF16
    """
    N, R, IN = A_all.shape
    _, OUT, _ = B_all.shape
    B_batch = x.shape[0]

    # Cat A: (N*R, IN)
    A_cat = A_all.reshape(N * R, IN)

    # x @ A_cat^T -> (B, S, N*R)
    mid = x @ A_cat.t()

    # Apply per-adapter per-batch scales
    # mid has shape (B, S, N*R), reshape to (B, S, N, R)
    mid = mid.view(B_batch, -1, N, R)
    # scale_all: (B, N) -> (B, 1, N, 1)
    s = scale_all.to(torch.bfloat16).unsqueeze(1).unsqueeze(-1)
    mid = mid * s
    mid = mid.view(B_batch, -1, N * R)

    # Cat B: (OUT, N*R) — need to interleave correctly
    # B_all is (N, OUT, R). For the cat approach, we cat along rank dim:
    # B_cat = cat along dim=2 -> (OUT, N*R) but we need to transpose per-adapter
    # Actually: B_cat[o, n*R:(n+1)*R] = B_all[n, o, :]
    B_cat = B_all.permute(1, 0, 2).reshape(OUT, N * R)

    # mid @ B_cat^T -> (B, S, OUT)
    out = mid @ B_cat.t()

    return out


# =============================================================================
# Custom Op Registration for torch.compile compatibility
# =============================================================================

_USE_CUSTOM_OP = False

try:
    _custom_op_fn = torch.library.custom_op

    @torch.library.custom_op("futudiffu::multi_lora", mutates_args=())
    def multi_lora_op(
        x: Tensor,
        A_all: Tensor,
        B_all: Tensor,
        scale_all: Tensor,
        N_ADAPTERS: int,
        RANK: int,
    ) -> Tensor:
        """Multi-LoRA forward via Triton kernel (custom op for torch.compile)."""
        return multi_lora_forward(x, A_all, B_all, scale_all)

    @multi_lora_op.register_fake
    def _multi_lora_op_fake(
        x: Tensor,
        A_all: Tensor,
        B_all: Tensor,
        scale_all: Tensor,
        N_ADAPTERS: int,
        RANK: int,
    ) -> Tensor:
        B, S, IN = x.shape
        OUT = A_all.shape[0]  # This is N_ADAPTERS, not OUT
        OUT = B_all.shape[1]  # out_features
        return x.new_empty(B, S, OUT)

    def _setup_context(ctx, inputs, output):
        x, A_all, B_all, scale_all, N_ADAPTERS, RANK = inputs
        ctx.save_for_backward(x, A_all, B_all, scale_all)
        ctx.N_ADAPTERS = N_ADAPTERS
        ctx.RANK = RANK

    def _backward(ctx, grad_out):
        """Backward for multi-LoRA.

        Forward: out = Σ_i s_i * (x @ A_i^T) @ B_i^T
        Backward (standard linear algebra, PyTorch ops — not a Triton kernel):
          dB_i = s_i * M_i^T @ grad_out    (recompute M_i = x @ A_i^T)
          dA_i = s_i * (grad_out @ B_i)^T @ x
          dx   = Σ_i s_i * (grad_out @ B_i) @ A_i
        """
        x, A_all, B_all, scale_all = ctx.saved_tensors
        N = ctx.N_ADAPTERS
        B, S, OUT = grad_out.shape

        grad_x = torch.zeros_like(x)
        grad_A = torch.zeros_like(A_all)
        grad_B = torch.zeros_like(B_all)

        for i in range(N):
            # (B, 1, 1) broadcast scale
            s = scale_all[:, i].to(x.dtype).unsqueeze(-1).unsqueeze(-1)

            # Recompute intermediate: M_i = x @ A_i^T  (B, S, R)
            M_i = x @ A_all[i].t()

            # Scaled gradient through output: dL_i = s * grad_out  (B, S, OUT)
            dL_i = s * grad_out

            # dB_i: (B, OUT, S) @ (B, S, R) -> (B, OUT, R) -> sum over B
            grad_B[i] = (dL_i.transpose(-2, -1) @ M_i).sum(0)

            # dM_i = dL_i @ B_i  (B, S, R)
            dM_i = dL_i @ B_all[i]

            # dA_i: (B, R, S) @ (B, S, IN) -> (B, R, IN) -> sum over B
            grad_A[i] = (dM_i.transpose(-2, -1) @ x).sum(0)

            # dx contribution
            grad_x = grad_x + dM_i @ A_all[i]

        # No gradients for scale_all, N_ADAPTERS, RANK
        return grad_x, grad_A, grad_B, None, None, None

    multi_lora_op.register_autograd(_backward, setup_context=_setup_context)
    _USE_CUSTOM_OP = True

except (AttributeError, TypeError):
    pass

if not _USE_CUSTOM_OP:
    def multi_lora_op(
        x: Tensor,
        A_all: Tensor,
        B_all: Tensor,
        scale_all: Tensor,
        N_ADAPTERS: int,
        RANK: int,
    ) -> Tensor:
        """Fallback multi-LoRA forward (no custom op)."""
        return multi_lora_forward(x, A_all, B_all, scale_all)


# =============================================================================
# Test
# =============================================================================

def test_multi_lora_kernel():
    """Comprehensive test for the sparse multi-LoRA Triton kernel."""
    import torch.nn.functional as F

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")

    print("=" * 70)
    print("Sparse Multi-LoRA Triton Kernel Tests")
    print("=" * 70)

    # --- Test 1: Correctness vs PyTorch reference ---
    print("\n--- Test 1: Correctness (4 adapters, rank=8) ---")

    B, S, IN, OUT = 4, 128, 512, 256
    N_ADAPTERS = 4
    RANK = 8

    torch.manual_seed(42)
    x = torch.randn(B, S, IN, dtype=torch.bfloat16, device=device)
    A_all = torch.randn(N_ADAPTERS, RANK, IN, dtype=torch.bfloat16, device=device) * 0.01
    B_all = torch.randn(N_ADAPTERS, OUT, RANK, dtype=torch.bfloat16, device=device) * 0.01
    # Mixed scales: some nonzero, some might be zero
    scale_all = torch.tensor([
        [1.0, 0.5, 0.0, 2.0],
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ], dtype=torch.float32, device=device)

    out_triton = multi_lora_forward(x, A_all, B_all, scale_all)
    out_ref = multi_lora_forward_ref(x, A_all, B_all, scale_all)

    cos = F.cosine_similarity(
        out_triton.float().flatten().unsqueeze(0),
        out_ref.float().flatten().unsqueeze(0),
    ).item()
    mse = (out_triton.float() - out_ref.float()).pow(2).mean().item()
    max_err = (out_triton.float() - out_ref.float()).abs().max().item()

    print(f"  Cosine similarity: {cos:.6f}")
    print(f"  MSE:               {mse:.2e}")
    print(f"  Max error:         {max_err:.2e}")
    assert cos > 0.999, f"Cosine similarity too low: {cos}"
    print("  PASSED")

    # --- Test 2: Batch 3 (all scales zero) should produce zero output ---
    print("\n--- Test 2: All-zero scales produce zero output ---")
    out_batch3 = out_triton[3]  # scale_all[3] = [0, 0, 0, 0]
    max_val = out_batch3.abs().max().item()
    print(f"  Max absolute value in zero-scale batch: {max_val:.2e}")
    assert max_val == 0.0, f"Expected zero output for all-zero scales, got max={max_val}"
    print("  PASSED")

    # --- Test 3: Zero-scale adapter contributes nothing ---
    print("\n--- Test 3: Zero-scale adapter == excluded adapter ---")

    # Compute with adapter 2 having scale=0 for all batches
    scale_only_012 = scale_all.clone()
    scale_only_012[:, 2] = 0.0  # already zero for batch 0,1; set for batch 2,3

    out_no_adapter2 = multi_lora_forward(x, A_all, B_all, scale_only_012)

    # Compute reference excluding adapter 2 entirely
    A_no2 = torch.stack([A_all[0], A_all[1], A_all[3]])
    B_no2 = torch.stack([B_all[0], B_all[1], B_all[3]])
    scale_no2 = torch.stack([
        scale_only_012[:, 0],
        scale_only_012[:, 1],
        scale_only_012[:, 3],
    ], dim=1)
    out_ref_no2 = multi_lora_forward_ref(x, A_no2, B_no2, scale_no2)

    cos_no2 = F.cosine_similarity(
        out_no_adapter2.float().flatten().unsqueeze(0),
        out_ref_no2.float().flatten().unsqueeze(0),
    ).item()
    print(f"  Cosine similarity (kernel w/ zero scale vs ref w/o adapter): {cos_no2:.6f}")
    assert cos_no2 > 0.999, f"Cosine similarity too low: {cos_no2}"
    print("  PASSED")

    # --- Test 4: Single adapter correctness ---
    print("\n--- Test 4: Single adapter (N=1, rank=16) ---")

    RANK_1 = 16
    A_single = torch.randn(1, RANK_1, IN, dtype=torch.bfloat16, device=device) * 0.01
    B_single = torch.randn(1, OUT, RANK_1, dtype=torch.bfloat16, device=device) * 0.01
    scale_single = torch.ones(B, 1, dtype=torch.float32, device=device) * 0.75

    out_t1 = multi_lora_forward(x, A_single, B_single, scale_single)
    out_r1 = multi_lora_forward_ref(x, A_single, B_single, scale_single)

    cos_1 = F.cosine_similarity(
        out_t1.float().flatten().unsqueeze(0),
        out_r1.float().flatten().unsqueeze(0),
    ).item()
    print(f"  Cosine similarity: {cos_1:.6f}")
    assert cos_1 > 0.999, f"Cosine similarity too low: {cos_1}"
    print("  PASSED")

    # --- Test 5: Larger rank (rank=4, padded to 16) ---
    print("\n--- Test 5: Small rank (rank=4, padded to 16) ---")

    RANK_SMALL = 4
    A_small = torch.randn(N_ADAPTERS, RANK_SMALL, IN, dtype=torch.bfloat16, device=device) * 0.01
    B_small = torch.randn(N_ADAPTERS, OUT, RANK_SMALL, dtype=torch.bfloat16, device=device) * 0.01
    scale_small = torch.ones(B, N_ADAPTERS, dtype=torch.float32, device=device)

    out_ts = multi_lora_forward(x, A_small, B_small, scale_small)
    out_rs = multi_lora_forward_ref(x, A_small, B_small, scale_small)

    cos_s = F.cosine_similarity(
        out_ts.float().flatten().unsqueeze(0),
        out_rs.float().flatten().unsqueeze(0),
    ).item()
    print(f"  Cosine similarity: {cos_s:.6f}")
    assert cos_s > 0.999, f"Cosine similarity too low: {cos_s}"
    print("  PASSED")

    # --- Test 6: Benchmark: Triton vs cat-based ---
    print("\n--- Test 6: Benchmark ---")

    B_bench, S_bench, IN_bench, OUT_bench = 2, 256, 2560, 2560
    N_bench = 4
    R_bench = 16

    x_bench = torch.randn(B_bench, S_bench, IN_bench, dtype=torch.bfloat16, device=device)
    A_bench = torch.randn(N_bench, R_bench, IN_bench, dtype=torch.bfloat16, device=device) * 0.01
    B_bench_t = torch.randn(N_bench, OUT_bench, R_bench, dtype=torch.bfloat16, device=device) * 0.01

    # Dense scales (all active)
    scale_dense = torch.ones(B_bench, N_bench, dtype=torch.float32, device=device)

    # Sparse scales (2 of 4 adapters active per batch)
    scale_sparse = torch.tensor([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
    ], dtype=torch.float32, device=device)

    # Warmup
    for _ in range(5):
        _ = multi_lora_forward(x_bench, A_bench, B_bench_t, scale_dense)
        _ = multi_lora_forward_cat(x_bench, A_bench, B_bench_t, scale_dense)
    torch.cuda.synchronize()

    N_ITER = 50

    # Benchmark Triton (dense)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        _ = multi_lora_forward(x_bench, A_bench, B_bench_t, scale_dense)
    torch.cuda.synchronize()
    t_triton_dense = (time.perf_counter() - t0) / N_ITER * 1000

    # Benchmark Triton (sparse)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        _ = multi_lora_forward(x_bench, A_bench, B_bench_t, scale_sparse)
    torch.cuda.synchronize()
    t_triton_sparse = (time.perf_counter() - t0) / N_ITER * 1000

    # Benchmark cat-based (dense — cat doesn't benefit from sparsity)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        _ = multi_lora_forward_cat(x_bench, A_bench, B_bench_t, scale_dense)
    torch.cuda.synchronize()
    t_cat_dense = (time.perf_counter() - t0) / N_ITER * 1000

    # Benchmark reference (dense)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        _ = multi_lora_forward_ref(x_bench, A_bench, B_bench_t, scale_dense)
    torch.cuda.synchronize()
    t_ref_dense = (time.perf_counter() - t0) / N_ITER * 1000

    print(f"  Problem: B={B_bench}, S={S_bench}, IN={IN_bench}, OUT={OUT_bench}, "
          f"N={N_bench}, R={R_bench}")
    print(f"  Triton (dense, 4/4 active):  {t_triton_dense:.3f} ms")
    print(f"  Triton (sparse, 2/4 active): {t_triton_sparse:.3f} ms")
    print(f"  Cat-based (dense):           {t_cat_dense:.3f} ms")
    print(f"  PyTorch loop ref (dense):    {t_ref_dense:.3f} ms")

    if t_triton_sparse < t_triton_dense:
        speedup = t_triton_dense / t_triton_sparse
        print(f"  Sparse skip speedup:         {speedup:.2f}x")
    else:
        print(f"  Sparse skip overhead:        {t_triton_sparse / t_triton_dense:.2f}x (compile overhead may dominate)")

    # --- Test 7: custom_op registration ---
    print("\n--- Test 7: custom_op registration ---")
    if _USE_CUSTOM_OP:
        out_op = multi_lora_op(x, A_all, B_all, scale_all, N_ADAPTERS, RANK)
        cos_op = F.cosine_similarity(
            out_op.float().flatten().unsqueeze(0),
            out_ref.float().flatten().unsqueeze(0),
        ).item()
        print(f"  custom_op cosine vs reference: {cos_op:.6f}")
        assert cos_op > 0.999, f"custom_op cosine too low: {cos_op}"
        print(f"  custom_op registered: True")
    else:
        print(f"  custom_op registered: False (fallback path)")
    print("  PASSED")

    print("\n" + "=" * 70)
    print("All tests passed.")
    print("=" * 70)


if __name__ == "__main__":
    test_multi_lora_kernel()

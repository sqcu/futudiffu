"""Fused elementwise Triton kernels for NextDiT JointTransformerBlock.

Each JointTransformerBlock launches ~12 separate elementwise kernels for
norm, modulation, gating, and residual operations. These fused kernels
reduce that to fewer launches by combining:

Pattern A: RMSNorm + adaLN modulation (pre-attention and pre-FFN)
  normed = rms_norm(x, weight, eps) * (1 + scale.unsqueeze(1))

Pattern B: RMSNorm + tanh gate + residual add (post-attention and post-FFN)
  result = residual + tanh(gate.unsqueeze(1)) * rms_norm(x, weight, eps)

Both kernels process one row (one token position) per program instance.
Grid = (B * seq,). Accumulation for RMSNorm variance is always float32.
Input/output can be bfloat16.
"""

import torch
import logging

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice as tl_libdevice

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False
    logging.info("Fused kernels: Triton not available, will use PyTorch fallback")


if _HAS_TRITON:

    # =========================================================================
    # Pattern A: Fused RMSNorm + adaLN modulation
    #
    #   normed = rms_norm(x, weight, eps)
    #   output = normed * (1 + scale)
    #
    # where scale is (B, dim) broadcast over the seq dimension.
    # =========================================================================

    @triton.jit
    def fused_rms_norm_modulate_kernel(
        # Pointers
        x_ptr,          # (B * seq, dim) input
        weight_ptr,     # (dim,) RMSNorm affine weight
        scale_ptr,      # (B, dim) adaLN scale
        out_ptr,        # (B * seq, dim) output
        # Dimensions
        dim,            # actual hidden dim (e.g. 3840)
        seq_len,        # sequence length, for computing batch index
        # RMSNorm epsilon
        eps,
        # Block size (next power of 2 >= dim)
        BLOCK_DIM: tl.constexpr,
    ):
        # One program per row = one token position
        row = tl.program_id(0)
        batch_idx = row // seq_len

        # Column offsets within this row
        cols = tl.arange(0, BLOCK_DIM)
        mask = cols < dim

        # --- Load x row into float32 for accumulation ---
        x_offset = row * dim + cols
        x = tl.load(x_ptr + x_offset, mask=mask, other=0.0).to(tl.float32)

        # --- RMSNorm: rsqrt(mean(x^2) + eps) ---
        x_sq = x * x
        mean_sq = tl.sum(x_sq, axis=0) / dim
        rrms = 1.0 / tl.sqrt(mean_sq + eps)
        normed = x * rrms

        # --- Apply elementwise affine weight ---
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        normed = normed * w

        # --- adaLN modulation: normed * (1 + scale) ---
        scale_offset = batch_idx * dim + cols
        s = tl.load(scale_ptr + scale_offset, mask=mask, other=0.0).to(tl.float32)
        result = normed * (1.0 + s)

        # --- Store output (cast back to input dtype) ---
        tl.store(out_ptr + x_offset, result.to(out_ptr.dtype.element_ty), mask=mask)


    # =========================================================================
    # Pattern B: Fused RMSNorm + tanh gate + residual add
    #
    #   normed = rms_norm(x, weight, eps)
    #   gated = tanh(gate) * normed        (gate broadcast over seq)
    #   result = residual + gated
    #
    # where gate is (B, dim) broadcast over the seq dimension.
    # =========================================================================

    @triton.jit
    def fused_rms_norm_gate_residual_kernel(
        # Pointers
        x_ptr,          # (B * seq, dim) input (attention/FFN output)
        weight_ptr,     # (dim,) RMSNorm affine weight
        gate_ptr,       # (B, dim) gate values
        residual_ptr,   # (B * seq, dim) residual tensor
        out_ptr,        # (B * seq, dim) output
        # Dimensions
        dim,            # actual hidden dim (e.g. 3840)
        seq_len,        # sequence length, for computing batch index
        # RMSNorm epsilon
        eps,
        # Block size (next power of 2 >= dim)
        BLOCK_DIM: tl.constexpr,
    ):
        # One program per row = one token position
        row = tl.program_id(0)
        batch_idx = row // seq_len

        # Column offsets within this row
        cols = tl.arange(0, BLOCK_DIM)
        mask = cols < dim

        # --- Load x row into float32 for accumulation ---
        x_offset = row * dim + cols
        x = tl.load(x_ptr + x_offset, mask=mask, other=0.0).to(tl.float32)

        # --- RMSNorm: rsqrt(mean(x^2) + eps) ---
        x_sq = x * x
        mean_sq = tl.sum(x_sq, axis=0) / dim
        rrms = 1.0 / tl.sqrt(mean_sq + eps)
        normed = x * rrms

        # --- Apply elementwise affine weight ---
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        normed = normed * w

        # --- tanh(gate) * normed ---
        gate_offset = batch_idx * dim + cols
        g = tl.load(gate_ptr + gate_offset, mask=mask, other=0.0).to(tl.float32)
        g_tanh = tl_libdevice.tanh(g)
        gated = g_tanh * normed

        # --- Residual add ---
        r = tl.load(residual_ptr + x_offset, mask=mask, other=0.0).to(tl.float32)
        result = r + gated

        # --- Store output (cast back to input dtype) ---
        tl.store(out_ptr + x_offset, result.to(out_ptr.dtype.element_ty), mask=mask)


    # =========================================================================
    # Python wrappers
    # =========================================================================

    def _fused_rms_norm_modulate_triton(
        x: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Fused RMSNorm + adaLN modulation.

        Computes: rms_norm(x, weight, eps) * (1 + scale.unsqueeze(1))

        Args:
            x: (B, seq, dim) input tensor.
            weight: (dim,) RMSNorm elementwise affine weight.
            scale: (B, dim) adaLN scale values.
            eps: RMSNorm epsilon.

        Returns:
            (B, seq, dim) modulated output.
        """
        assert x.ndim == 3, f"Expected x to be 3D (B, seq, dim), got {x.ndim}D"
        assert x.is_contiguous(), "Input x must be contiguous"
        assert weight.is_contiguous(), "Weight must be contiguous"
        assert scale.is_contiguous(), "Scale must be contiguous"

        B, seq_len, dim = x.shape
        assert weight.shape == (dim,), f"Weight shape mismatch: {weight.shape} vs ({dim},)"
        assert scale.shape == (B, dim), f"Scale shape mismatch: {scale.shape} vs ({B}, {dim})"

        # Flatten to 2D for kernel: (B * seq, dim)
        x_flat = x.reshape(B * seq_len, dim)
        out = torch.empty_like(x_flat)

        # Next power of 2 >= dim
        BLOCK_DIM = triton.next_power_of_2(dim)

        grid = (B * seq_len,)
        fused_rms_norm_modulate_kernel[grid](
            x_flat, weight, scale, out,
            dim, seq_len, eps,
            BLOCK_DIM=BLOCK_DIM,
        )

        return out.reshape(B, seq_len, dim)


    def _fused_rms_norm_gate_residual_triton(
        x: torch.Tensor,
        weight: torch.Tensor,
        gate: torch.Tensor,
        residual: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Fused RMSNorm + tanh gate + residual add.

        Computes: residual + tanh(gate.unsqueeze(1)) * rms_norm(x, weight, eps)

        Args:
            x: (B, seq, dim) input tensor (attention or FFN output).
            weight: (dim,) RMSNorm elementwise affine weight.
            gate: (B, dim) gate values from adaLN.
            residual: (B, seq, dim) residual tensor.
            eps: RMSNorm epsilon.

        Returns:
            (B, seq, dim) output tensor.
        """
        assert x.ndim == 3, f"Expected x to be 3D (B, seq, dim), got {x.ndim}D"
        assert x.is_contiguous(), "Input x must be contiguous"
        assert weight.is_contiguous(), "Weight must be contiguous"
        assert gate.is_contiguous(), "Gate must be contiguous"
        assert residual.is_contiguous(), "Residual must be contiguous"

        B, seq_len, dim = x.shape
        assert weight.shape == (dim,), f"Weight shape mismatch: {weight.shape} vs ({dim},)"
        assert gate.shape == (B, dim), f"Gate shape mismatch: {gate.shape} vs ({B}, {dim})"
        assert residual.shape == x.shape, f"Residual shape mismatch: {residual.shape} vs {x.shape}"

        # Flatten to 2D for kernel: (B * seq, dim)
        x_flat = x.reshape(B * seq_len, dim)
        residual_flat = residual.reshape(B * seq_len, dim)
        out = torch.empty_like(x_flat)

        # Next power of 2 >= dim
        BLOCK_DIM = triton.next_power_of_2(dim)

        grid = (B * seq_len,)
        fused_rms_norm_gate_residual_kernel[grid](
            x_flat, weight, gate, residual_flat, out,
            dim, seq_len, eps,
            BLOCK_DIM=BLOCK_DIM,
        )

        return out.reshape(B, seq_len, dim)


    # =========================================================================
    # Pattern C: Fused QKV split + per-head RMSNorm + Flux RoPE + transpose
    #
    # Replaces 6 bandwidth-bound ops between QKV GEMM and SDPA attention:
    #   1. torch.split (Q, K, V)
    #   2. reshape to (B, seq, heads, head_dim)
    #   3. q_norm (RMSNorm over head_dim)
    #   4. k_norm (RMSNorm over head_dim)
    #   5. apply_rope_flux (2x2 rotation per pair)
    #   6. movedim(1, 2) to (B, heads, seq, head_dim) layout
    #
    # One program per (token_position, head). Grid = (B * seq, n_heads).
    # Each program:
    #   - Loads 128 elements of Q, K, V from the 11520-wide QKV row
    #   - RMSNorm Q and K (f32 accumulation, 128-element reduction)
    #   - Loads 64 RoPE 2x2 rotation matrices for this token position
    #   - Applies rotation to Q and K
    #   - Writes Q, K, V in transposed (B, heads, seq, head_dim) layout
    # =========================================================================

    @triton.jit
    def fused_qkv_postprocess_kernel(
        # Input
        qkv_ptr,            # (B * seq, 3 * dim) BF16 — contiguous QKV GEMM output
        # Norm weights
        q_norm_w_ptr,        # (head_dim,) float32 — q_norm learnable weight
        k_norm_w_ptr,        # (head_dim,) float32 — k_norm learnable weight
        # RoPE frequencies
        freqs_ptr,           # (B * seq, n_pairs, 2, 2) float32 — precomputed rotation matrices
        # Outputs (transposed layout: B, heads, seq, head_dim)
        out_q_ptr,           # (B, n_heads, seq, head_dim) BF16
        out_k_ptr,           # (B, n_heads, seq, head_dim) BF16
        out_v_ptr,           # (B, n_heads, seq, head_dim) BF16
        # Dimensions
        seq_len,             # sequence length
        n_heads,             # number of attention heads (30)
        head_dim,            # per-head dimension (128)
        dim,                 # n_heads * head_dim = total hidden dim (3840)
        n_pairs,             # head_dim // 2 = number of rotation pairs (64)
        # RMSNorm epsilon
        eps,
        # Block size for head_dim (must be power of 2 >= head_dim)
        BLOCK_HD: tl.constexpr,
        # Block size for RoPE pairs (must be power of 2 >= n_pairs)
        BLOCK_PAIRS: tl.constexpr,
    ):
        # Grid: (B * seq_len, n_heads)
        row = tl.program_id(0)       # token position in flattened (B * seq)
        head = tl.program_id(1)      # head index

        # Column offsets within one head
        cols = tl.arange(0, BLOCK_HD)
        mask = cols < head_dim

        # --- Compute input offsets into the (B*seq, 3*dim) QKV tensor ---
        qkv_row_base = row * dim * 3
        q_offset = qkv_row_base + head * head_dim + cols
        k_offset = qkv_row_base + dim + head * head_dim + cols
        v_offset = qkv_row_base + dim * 2 + head * head_dim + cols

        # --- Load Q, K, V for this head (cast to f32 for norm) ---
        q = tl.load(qkv_ptr + q_offset, mask=mask, other=0.0).to(tl.float32)
        k = tl.load(qkv_ptr + k_offset, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(qkv_ptr + v_offset, mask=mask, other=0.0)  # V stays in input dtype

        # --- RMSNorm Q: compute scalar rrms = rsqrt(mean(q^2) + eps) ---
        q_sq = q * q
        q_mean_sq = tl.sum(q_sq, axis=0) / head_dim
        q_rrms = 1.0 / tl.sqrt(q_mean_sq + eps)

        # --- RMSNorm K: compute scalar rrms = rsqrt(mean(k^2) + eps) ---
        k_sq = k * k
        k_mean_sq = tl.sum(k_sq, axis=0) / head_dim
        k_rrms = 1.0 / tl.sqrt(k_mean_sq + eps)

        # --- Apply Flux RoPE: 2x2 rotation per pair of elements ---
        # freqs layout: (B*seq, n_pairs, 2, 2), row-major.
        # For pair p: freqs[row, p, i, j] at offset row * n_pairs * 4 + p * 4 + i * 2 + j
        # freqs[p, 0, 0] = cos,  freqs[p, 0, 1] = -sin
        # freqs[p, 1, 0] = sin,  freqs[p, 1, 1] = cos
        pair_ids = tl.arange(0, BLOCK_PAIRS)
        pair_mask = pair_ids < n_pairs
        freqs_base = row * n_pairs * 4 + pair_ids * 4

        cos_val = tl.load(freqs_ptr + freqs_base + 0, mask=pair_mask, other=1.0)    # [p, 0, 0] = cos
        neg_sin = tl.load(freqs_ptr + freqs_base + 1, mask=pair_mask, other=0.0)    # [p, 0, 1] = -sin
        sin_val = tl.load(freqs_ptr + freqs_base + 2, mask=pair_mask, other=0.0)    # [p, 1, 0] = sin
        # cos2 = cos again at [p, 1, 1], but we already have cos_val

        # Extract even/odd elements for RoPE pair-wise rotation.
        # We need normed Q/K split into even/odd positions, but Triton can't
        # gather from register vectors. Instead, reload raw Q/K at even/odd
        # offsets (L1/L2 hit from the first load) and recompute the norm
        # using the already-computed scalar rrms values.
        even_ids = pair_ids * 2
        odd_ids = pair_ids * 2 + 1
        even_mask = even_ids < head_dim
        odd_mask = odd_ids < head_dim

        q_even = tl.load(qkv_ptr + qkv_row_base + head * head_dim + even_ids, mask=even_mask, other=0.0).to(tl.float32)
        q_odd = tl.load(qkv_ptr + qkv_row_base + head * head_dim + odd_ids, mask=odd_mask, other=0.0).to(tl.float32)
        k_even = tl.load(qkv_ptr + qkv_row_base + dim + head * head_dim + even_ids, mask=even_mask, other=0.0).to(tl.float32)
        k_odd = tl.load(qkv_ptr + qkv_row_base + dim + head * head_dim + odd_ids, mask=odd_mask, other=0.0).to(tl.float32)

        q_w_even = tl.load(q_norm_w_ptr + even_ids, mask=even_mask, other=1.0).to(tl.float32)
        q_w_odd = tl.load(q_norm_w_ptr + odd_ids, mask=odd_mask, other=1.0).to(tl.float32)
        k_w_even = tl.load(k_norm_w_ptr + even_ids, mask=even_mask, other=1.0).to(tl.float32)
        k_w_odd = tl.load(k_norm_w_ptr + odd_ids, mask=odd_mask, other=1.0).to(tl.float32)

        q_normed_even = q_even * q_rrms * q_w_even
        q_normed_odd = q_odd * q_rrms * q_w_odd
        k_normed_even = k_even * k_rrms * k_w_even
        k_normed_odd = k_odd * k_rrms * k_w_odd

        # Apply 2x2 rotation
        q_rot_even = cos_val * q_normed_even + neg_sin * q_normed_odd
        q_rot_odd = sin_val * q_normed_even + cos_val * q_normed_odd
        k_rot_even = cos_val * k_normed_even + neg_sin * k_normed_odd
        k_rot_odd = sin_val * k_normed_even + cos_val * k_normed_odd

        # --- Compute output offsets (transposed: B, heads, seq, head_dim) ---
        batch_idx = row // seq_len
        seq_idx = row % seq_len
        # Output stride: out[b, h, s, d] = out_ptr[b * n_heads * seq * hd + h * seq * hd + s * hd + d]
        out_base = batch_idx * (n_heads * seq_len * head_dim) + head * (seq_len * head_dim) + seq_idx * head_dim

        # --- Store Q (interleave even/odd from rotated pairs) ---
        tl.store(out_q_ptr + out_base + even_ids, q_rot_even.to(out_q_ptr.dtype.element_ty), mask=even_mask)
        tl.store(out_q_ptr + out_base + odd_ids, q_rot_odd.to(out_q_ptr.dtype.element_ty), mask=odd_mask)

        # --- Store K (interleave even/odd from rotated pairs) ---
        tl.store(out_k_ptr + out_base + even_ids, k_rot_even.to(out_k_ptr.dtype.element_ty), mask=even_mask)
        tl.store(out_k_ptr + out_base + odd_ids, k_rot_odd.to(out_k_ptr.dtype.element_ty), mask=odd_mask)

        # --- Store V (no norm, no rope, just copy with transpose) ---
        tl.store(out_v_ptr + out_base + cols, v.to(out_v_ptr.dtype.element_ty), mask=mask)


    def _fused_qkv_postprocess_triton(
        qkv: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        freqs_cis: torch.Tensor,
        n_heads: int,
        head_dim: int,
        norm_eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fused QKV split + per-head RMSNorm + Flux RoPE + transpose.

        Replaces 6 separate bandwidth-bound operations with a single kernel launch.
        Input is the raw QKV GEMM output; outputs are Q, K, V in SDPA-ready layout.

        Args:
            qkv: (B, seq, 3*dim) BF16 — raw output of the QKV linear projection.
            q_norm_weight: (head_dim,) — RMSNorm weight for Q normalization.
            k_norm_weight: (head_dim,) — RMSNorm weight for K normalization.
            freqs_cis: (B, seq, 1, n_pairs, 2, 2) float32 — Flux RoPE rotation matrices.
                       The dim-2 size-1 broadcasts over heads; n_pairs = head_dim // 2.
            n_heads: Number of attention heads (e.g. 30).
            head_dim: Per-head dimension (e.g. 128).
            norm_eps: RMSNorm epsilon (default 1e-5).

        Returns:
            Tuple of (Q, K, V), each (B, n_heads, seq, head_dim) in input dtype.
        """
        assert qkv.ndim == 3, f"Expected qkv to be 3D (B, seq, 3*dim), got {qkv.ndim}D"
        assert qkv.is_contiguous(), "QKV must be contiguous"

        B, seq_len, total_dim = qkv.shape
        dim = n_heads * head_dim
        assert total_dim == 3 * dim, f"QKV last dim {total_dim} != 3 * {dim}"

        n_pairs = head_dim // 2
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"

        # Validate norm weights
        assert q_norm_weight.shape == (head_dim,), f"q_norm shape {q_norm_weight.shape} != ({head_dim},)"
        assert k_norm_weight.shape == (head_dim,), f"k_norm shape {k_norm_weight.shape} != ({head_dim},)"

        # Handle freqs_cis shape: expected (B, seq, 1, n_pairs, 2, 2)
        # Squeeze the broadcast dim and flatten to (B*seq, n_pairs, 2, 2)
        if freqs_cis.ndim == 6:
            # (B, seq, 1, n_pairs, 2, 2) -> squeeze dim 2 -> (B, seq, n_pairs, 2, 2)
            assert freqs_cis.shape[2] == 1, f"Expected freqs_cis dim 2 to be 1, got {freqs_cis.shape[2]}"
            freqs_flat = freqs_cis.squeeze(2).reshape(B * seq_len, n_pairs, 2, 2).contiguous()
        elif freqs_cis.ndim == 5:
            # Already (B, seq, n_pairs, 2, 2)
            freqs_flat = freqs_cis.reshape(B * seq_len, n_pairs, 2, 2).contiguous()
        else:
            raise ValueError(f"Unexpected freqs_cis ndim: {freqs_cis.ndim}, shape: {freqs_cis.shape}")

        assert freqs_flat.shape == (B * seq_len, n_pairs, 2, 2), \
            f"freqs_flat shape {freqs_flat.shape} != ({B * seq_len}, {n_pairs}, 2, 2)"

        # Flatten QKV to (B*seq, 3*dim) for kernel
        qkv_flat = qkv.reshape(B * seq_len, total_dim)

        # Allocate outputs in transposed layout: (B, n_heads, seq, head_dim)
        out_q = torch.empty(B, n_heads, seq_len, head_dim, dtype=qkv.dtype, device=qkv.device)
        out_k = torch.empty(B, n_heads, seq_len, head_dim, dtype=qkv.dtype, device=qkv.device)
        out_v = torch.empty(B, n_heads, seq_len, head_dim, dtype=qkv.dtype, device=qkv.device)

        # Ensure norm weights are float32 and contiguous
        q_norm_f32 = q_norm_weight.to(dtype=torch.float32).contiguous()
        k_norm_f32 = k_norm_weight.to(dtype=torch.float32).contiguous()

        BLOCK_HD = triton.next_power_of_2(head_dim)
        BLOCK_PAIRS = triton.next_power_of_2(n_pairs)

        grid = (B * seq_len, n_heads)
        fused_qkv_postprocess_kernel[grid](
            qkv_flat,
            q_norm_f32, k_norm_f32,
            freqs_flat,
            out_q, out_k, out_v,
            seq_len, n_heads, head_dim, dim, n_pairs,
            norm_eps,
            BLOCK_HD=BLOCK_HD,
            BLOCK_PAIRS=BLOCK_PAIRS,
        )

        return out_q, out_k, out_v


else:
    # =========================================================================
    # PyTorch fallback implementations (no Triton)
    # =========================================================================

    def _rms_norm_fallback(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """Pure-PyTorch RMSNorm."""
        x_f32 = x.float()
        rrms = torch.rsqrt(torch.mean(x_f32 ** 2, dim=-1, keepdim=True) + eps)
        normed = x_f32 * rrms
        if weight is not None:
            normed = normed * weight.float()
        return normed.to(x.dtype)


    def _fused_rms_norm_modulate_fallback(
        x: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Fallback: RMSNorm + adaLN modulation in PyTorch."""
        normed = _rms_norm_fallback(x, weight, eps)
        return normed * (1 + scale.unsqueeze(1))


    def _fused_rms_norm_gate_residual_fallback(
        x: torch.Tensor,
        weight: torch.Tensor,
        gate: torch.Tensor,
        residual: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Fallback: RMSNorm + tanh gate + residual add in PyTorch."""
        normed = _rms_norm_fallback(x, weight, eps)
        gated = gate.unsqueeze(1).tanh() * normed
        return residual + gated


    def _fused_qkv_postprocess_fallback(
        qkv: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        freqs_cis: torch.Tensor,
        n_heads: int,
        head_dim: int,
        norm_eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fallback: QKV split + RMSNorm + Flux RoPE + transpose in PyTorch."""
        from .attention import apply_rope_flux

        B, seq_len, total_dim = qkv.shape
        dim = n_heads * head_dim
        xq, xk, xv = torch.split(qkv, [dim, dim, dim], dim=-1)
        xq = xq.view(B, seq_len, n_heads, head_dim)
        xk = xk.view(B, seq_len, n_heads, head_dim)
        xv = xv.view(B, seq_len, n_heads, head_dim)
        xq = _rms_norm_fallback(xq, q_norm_weight, norm_eps)
        xk = _rms_norm_fallback(xk, k_norm_weight, norm_eps)
        xq, xk = apply_rope_flux(xq, xk, freqs_cis)
        xq = xq.movedim(1, 2)
        xk = xk.movedim(1, 2)
        xv = xv.movedim(1, 2)
        return xq, xk, xv


# =========================================================================
# Custom op registrations for torch.compile compatibility
# =========================================================================
# These allow torch.compile(mode="reduce-overhead") to capture fused kernel
# calls into CUDA graphs, eliminating graph breaks that the old
# @torch.compiler.disable workaround imposed.
# =========================================================================

_USE_FUSED_CUSTOM_OP = False

# Pick the right implementation based on Triton availability
if _HAS_TRITON:
    _rms_norm_modulate_impl = _fused_rms_norm_modulate_triton
    _rms_norm_gate_residual_impl = _fused_rms_norm_gate_residual_triton
    _qkv_postprocess_impl = _fused_qkv_postprocess_triton
else:
    _rms_norm_modulate_impl = _fused_rms_norm_modulate_fallback
    _rms_norm_gate_residual_impl = _fused_rms_norm_gate_residual_fallback
    _qkv_postprocess_impl = _fused_qkv_postprocess_fallback

try:
    _custom_op_fn = torch.library.custom_op

    # --- fused_rms_norm_modulate ---

    @torch.library.custom_op("futudiffu::fused_rms_norm_modulate", mutates_args=())
    def fused_rms_norm_modulate(
        x: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        return _rms_norm_modulate_impl(x, weight, scale, eps)

    @fused_rms_norm_modulate.register_fake
    def _fused_rms_norm_modulate_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        return x.new_empty(*x.shape)

    # --- fused_rms_norm_gate_residual ---

    @torch.library.custom_op("futudiffu::fused_rms_norm_gate_residual", mutates_args=())
    def fused_rms_norm_gate_residual(
        x: torch.Tensor,
        weight: torch.Tensor,
        gate: torch.Tensor,
        residual: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        return _rms_norm_gate_residual_impl(x, weight, gate, residual, eps)

    @fused_rms_norm_gate_residual.register_fake
    def _fused_rms_norm_gate_residual_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        gate: torch.Tensor,
        residual: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        return x.new_empty(*x.shape)

    # --- fused_qkv_postprocess ---
    # Returns a tuple of 3 tensors: (Q, K, V) each (B, n_heads, seq, head_dim).
    # freqs_cis is passed as a tensor; n_heads, head_dim, norm_eps are primitives.

    @torch.library.custom_op("futudiffu::fused_qkv_postprocess", mutates_args=())
    def fused_qkv_postprocess(
        qkv: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        freqs_cis: torch.Tensor,
        n_heads: int,
        head_dim: int,
        norm_eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _qkv_postprocess_impl(
            qkv, q_norm_weight, k_norm_weight, freqs_cis,
            n_heads, head_dim, norm_eps,
        )

    @fused_qkv_postprocess.register_fake
    def _fused_qkv_postprocess_fake(
        qkv: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        freqs_cis: torch.Tensor,
        n_heads: int,
        head_dim: int,
        norm_eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = qkv.shape[0]
        seq_len = qkv.shape[1]
        out_q = qkv.new_empty(B, n_heads, seq_len, head_dim)
        out_k = qkv.new_empty(B, n_heads, seq_len, head_dim)
        out_v = qkv.new_empty(B, n_heads, seq_len, head_dim)
        return out_q, out_k, out_v

    # =================================================================
    # Autograd registrations for backward pass (LoRA policy training)
    # =================================================================
    # Analytical backward with float32 accumulation for RMSNorm.
    # Only input gradients (dx) are computed; norm weights, adaLN
    # params, and RoPE freqs are frozen.
    # =================================================================

    # --- fused_rms_norm_modulate backward ---
    # Forward: y = rms_norm(x, w, eps) * (1 + scale)
    # Backward: dx = rrms * (g - x * rrms^2 * mean(g * x))
    #   where g = grad * (1 + scale) * w

    def _rms_norm_modulate_setup(ctx, inputs, output):
        x, weight, scale, eps = inputs
        ctx.save_for_backward(x, weight, scale)
        ctx.eps = eps

    def _rms_norm_modulate_backward(ctx, grad_output):
        x, weight, scale = ctx.saved_tensors
        eps = ctx.eps
        D = x.shape[-1]
        m = 1 + scale.unsqueeze(1)  # [B, 1, dim]
        g = (grad_output * m * weight).float()  # [B, seq, dim]
        x_f32 = x.float()
        var = (x_f32 * x_f32).mean(dim=-1, keepdim=True)
        rrms = torch.rsqrt(var + eps)
        c = (g * x_f32).sum(dim=-1, keepdim=True) / D
        dx = (rrms * (g - x_f32 * rrms * rrms * c)).to(x.dtype)
        return dx, None, None, None

    fused_rms_norm_modulate.register_autograd(
        _rms_norm_modulate_backward, setup_context=_rms_norm_modulate_setup,
    )

    # --- fused_rms_norm_gate_residual backward ---
    # Forward: y = residual + tanh(gate) * rms_norm(x, w, eps)
    # Backward: d_residual = grad (skip connection pass-through)
    #           dx via RMSNorm backward on gated gradient

    def _rms_norm_gate_residual_setup(ctx, inputs, output):
        x, weight, gate, residual, eps = inputs
        ctx.save_for_backward(x, weight, gate)
        ctx.eps = eps

    def _rms_norm_gate_residual_backward(ctx, grad_output):
        x, weight, gate = ctx.saved_tensors
        eps = ctx.eps
        D = x.shape[-1]
        # Gradient through tanh gate
        gate_tanh = gate.unsqueeze(1).tanh()  # [B, 1, dim]
        g = (grad_output * gate_tanh * weight).float()  # [B, seq, dim]
        # RMSNorm backward
        x_f32 = x.float()
        var = (x_f32 * x_f32).mean(dim=-1, keepdim=True)
        rrms = torch.rsqrt(var + eps)
        c = (g * x_f32).sum(dim=-1, keepdim=True) / D
        dx = (rrms * (g - x_f32 * rrms * rrms * c)).to(x.dtype)
        # d_residual = grad_output (pass-through)
        return dx, None, None, grad_output, None

    fused_rms_norm_gate_residual.register_autograd(
        _rms_norm_gate_residual_backward,
        setup_context=_rms_norm_gate_residual_setup,
    )

    # --- fused_qkv_postprocess backward ---
    # Forward: split QKV, per-head RMSNorm Q/K, Flux RoPE, transpose
    # Backward: un-transpose, inverse RoPE, RMSNorm backward, concat

    def _qkv_postprocess_setup(ctx, inputs, output):
        qkv, q_norm_w, k_norm_w, freqs_cis, n_heads, head_dim, norm_eps = inputs
        ctx.save_for_backward(qkv, q_norm_w, k_norm_w, freqs_cis)
        ctx.n_heads = n_heads
        ctx.head_dim = head_dim
        ctx.norm_eps = norm_eps

    def _qkv_postprocess_backward(ctx, grad_q, grad_k, grad_v):
        qkv, q_norm_w, k_norm_w, freqs_cis = ctx.saved_tensors
        n_heads = ctx.n_heads
        head_dim = ctx.head_dim
        norm_eps = ctx.norm_eps

        B, seq_len, total_dim = qkv.shape
        dim = n_heads * head_dim
        n_pairs = head_dim // 2

        # 1. Un-transpose: (B, heads, seq, hd) -> (B, seq, heads, hd)
        dq = grad_q.movedim(1, 2).contiguous()
        dk = grad_k.movedim(1, 2).contiguous()
        dv = grad_v.movedim(1, 2).contiguous()

        # 2. Inverse RoPE: transpose of [[cos,-sin],[sin,cos]]
        if freqs_cis.ndim == 6:
            freqs = freqs_cis.squeeze(2)  # (B, seq, n_pairs, 2, 2)
        else:
            freqs = freqs_cis
        cos_val = freqs[:, :, :, 0, 0].unsqueeze(2)  # (B, seq, 1, n_pairs)
        sin_val = freqs[:, :, :, 1, 0].unsqueeze(2)  # (B, seq, 1, n_pairs)

        # Q inverse rotation
        dq_pairs = dq.reshape(B, seq_len, n_heads, n_pairs, 2)
        dq_e, dq_o = dq_pairs[..., 0], dq_pairs[..., 1]
        dq_pre = torch.stack([
            cos_val * dq_e + sin_val * dq_o,
            -sin_val * dq_e + cos_val * dq_o,
        ], dim=-1).reshape(B, seq_len, n_heads, head_dim)

        # K inverse rotation
        dk_pairs = dk.reshape(B, seq_len, n_heads, n_pairs, 2)
        dk_e, dk_o = dk_pairs[..., 0], dk_pairs[..., 1]
        dk_pre = torch.stack([
            cos_val * dk_e + sin_val * dk_o,
            -sin_val * dk_e + cos_val * dk_o,
        ], dim=-1).reshape(B, seq_len, n_heads, head_dim)

        # 3. Per-head RMSNorm backward for Q
        q_raw = qkv[:, :, :dim].reshape(B, seq_len, n_heads, head_dim)
        q_f32 = q_raw.float()
        q_var = (q_f32 * q_f32).mean(dim=-1, keepdim=True)
        q_rrms = torch.rsqrt(q_var + norm_eps)
        g_q = (dq_pre * q_norm_w).float()
        c_q = (g_q * q_f32).sum(dim=-1, keepdim=True) / head_dim
        dq_raw = (q_rrms * (g_q - q_f32 * q_rrms * q_rrms * c_q)).to(qkv.dtype)

        # 4. Per-head RMSNorm backward for K
        k_raw = qkv[:, :, dim:2 * dim].reshape(B, seq_len, n_heads, head_dim)
        k_f32 = k_raw.float()
        k_var = (k_f32 * k_f32).mean(dim=-1, keepdim=True)
        k_rrms = torch.rsqrt(k_var + norm_eps)
        g_k = (dk_pre * k_norm_w).float()
        c_k = (g_k * k_f32).sum(dim=-1, keepdim=True) / head_dim
        dk_raw = (k_rrms * (g_k - k_f32 * k_rrms * k_rrms * c_k)).to(qkv.dtype)

        # 5. Concat back to QKV layout
        dqkv = torch.cat([
            dq_raw.reshape(B, seq_len, dim),
            dk_raw.reshape(B, seq_len, dim),
            dv.reshape(B, seq_len, dim),
        ], dim=-1)

        return dqkv, None, None, None, None, None, None

    fused_qkv_postprocess.register_autograd(
        _qkv_postprocess_backward, setup_context=_qkv_postprocess_setup,
    )

    _USE_FUSED_CUSTOM_OP = True

except (AttributeError, TypeError):
    # torch.library.custom_op not available (old torch version).
    # Fall back to direct calls (no torch.compile graph capture support).
    fused_rms_norm_modulate = _rms_norm_modulate_impl
    fused_rms_norm_gate_residual = _rms_norm_gate_residual_impl
    fused_qkv_postprocess = _qkv_postprocess_impl


def _qkv_postprocess_reference(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    n_heads: int,
    head_dim: int,
    norm_eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference implementation for testing: exact unfused path from JointAttention.forward.

    Uses the same operations as the unfused code path, matching the PyTorch
    semantics exactly (float32 RMSNorm accumulation, Flux 2x2 rotation).
    """
    from .attention import apply_rope_flux

    B, seq_len, total_dim = qkv.shape
    dim = n_heads * head_dim

    # 1. Split
    xq, xk, xv = torch.split(qkv, [dim, dim, dim], dim=-1)

    # 2. Reshape to (B, seq, heads, head_dim)
    xq = xq.view(B, seq_len, n_heads, head_dim)
    xk = xk.view(B, seq_len, n_heads, head_dim)
    xv = xv.view(B, seq_len, n_heads, head_dim)

    # 3. Per-head RMSNorm (matching RMSNormModule behavior via rms_norm)
    from .attention import rms_norm
    xq = rms_norm(xq, q_norm_weight, norm_eps)
    xk = rms_norm(xk, k_norm_weight, norm_eps)

    # 4. Apply Flux RoPE
    xq, xk = apply_rope_flux(xq, xk, freqs_cis)

    # 5. Transpose to (B, heads, seq, head_dim)
    xq = xq.movedim(1, 2)
    xk = xk.movedim(1, 2)
    xv = xv.movedim(1, 2)

    return xq, xk, xv


def test_fused_qkv_postprocess():
    """Test the fused QKV postprocess kernel against the unfused reference.

    Creates random inputs matching the Z-Image model dimensions (B=2, seq=4288,
    n_heads=30, head_dim=128, dim=3840), runs both fused and unfused paths,
    and compares Q, K, V outputs via cosine similarity and max absolute error.
    Also prints timing comparison.
    """
    import time

    if not _HAS_TRITON:
        print("SKIP: Triton not available")
        return

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Z-Image dimensions
    B = 2
    seq_len = 4288
    n_heads = 30
    head_dim = 128
    dim = n_heads * head_dim  # 3840
    n_pairs = head_dim // 2    # 64
    norm_eps = 1e-5

    print(f"Test config: B={B}, seq={seq_len}, n_heads={n_heads}, head_dim={head_dim}, dim={dim}")

    # Create random inputs
    torch.manual_seed(42)
    qkv = torch.randn(B, seq_len, 3 * dim, dtype=dtype, device=device)
    q_norm_weight = torch.randn(head_dim, dtype=torch.float32, device=device) * 0.1 + 1.0
    k_norm_weight = torch.randn(head_dim, dtype=torch.float32, device=device) * 0.1 + 1.0

    # Create RoPE frequencies with the same shape as the real pipeline:
    # EmbedND produces (B, 1, seq, n_pairs, 2, 2), movedim(1,2) -> (B, seq, 1, n_pairs, 2, 2)
    # Build realistic rotation matrices: [[cos, -sin], [sin, cos]]
    angles = torch.randn(B, seq_len, n_pairs, dtype=torch.float32, device=device) * 0.5
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)
    freqs_cis = torch.zeros(B, seq_len, 1, n_pairs, 2, 2, dtype=torch.float32, device=device)
    freqs_cis[:, :, 0, :, 0, 0] = cos_vals
    freqs_cis[:, :, 0, :, 0, 1] = -sin_vals
    freqs_cis[:, :, 0, :, 1, 0] = sin_vals
    freqs_cis[:, :, 0, :, 1, 1] = cos_vals

    print("Running reference (unfused) path...")
    ref_q, ref_k, ref_v = _qkv_postprocess_reference(
        qkv, q_norm_weight, k_norm_weight, freqs_cis, n_heads, head_dim, norm_eps,
    )

    print("Running fused kernel path...")
    fused_q, fused_k, fused_v = fused_qkv_postprocess(
        qkv, q_norm_weight, k_norm_weight, freqs_cis, n_heads, head_dim, norm_eps,
    )

    # Compare outputs
    def compare(name, ref, fused):
        ref_f = ref.float()
        fused_f = fused.float()
        max_err = (ref_f - fused_f).abs().max().item()
        # Per-head cosine similarity
        ref_flat = ref_f.reshape(B, n_heads, -1)
        fused_flat = fused_f.reshape(B, n_heads, -1)
        cos_sim = torch.nn.functional.cosine_similarity(ref_flat, fused_flat, dim=-1)
        min_cos = cos_sim.min().item()
        mean_cos = cos_sim.mean().item()
        print(f"  {name}: max_err={max_err:.6e}, cos_sim min={min_cos:.6f} mean={mean_cos:.6f}")
        return min_cos

    print("\nAccuracy comparison:")
    q_cos = compare("Q", ref_q, fused_q)
    k_cos = compare("K", ref_k, fused_k)
    v_cos = compare("V", ref_v, fused_v)

    # Timing comparison
    torch.cuda.synchronize()

    n_warmup = 5
    n_iters = 20

    # Warmup fused
    for _ in range(n_warmup):
        fused_qkv_postprocess(qkv, q_norm_weight, k_norm_weight, freqs_cis, n_heads, head_dim, norm_eps)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        fused_qkv_postprocess(qkv, q_norm_weight, k_norm_weight, freqs_cis, n_heads, head_dim, norm_eps)
    torch.cuda.synchronize()
    fused_time = (time.perf_counter() - t0) / n_iters * 1000

    # Warmup reference
    for _ in range(n_warmup):
        _qkv_postprocess_reference(qkv, q_norm_weight, k_norm_weight, freqs_cis, n_heads, head_dim, norm_eps)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _qkv_postprocess_reference(qkv, q_norm_weight, k_norm_weight, freqs_cis, n_heads, head_dim, norm_eps)
    torch.cuda.synchronize()
    ref_time = (time.perf_counter() - t0) / n_iters * 1000

    print(f"\nTiming ({n_iters} iterations):")
    print(f"  Reference (unfused): {ref_time:.2f} ms")
    print(f"  Fused kernel:        {fused_time:.2f} ms")
    print(f"  Speedup:             {ref_time / fused_time:.2f}x")

    # Pass/fail
    all_pass = q_cos > 0.999 and k_cos > 0.999 and v_cos > 0.999
    print(f"\nResult: {'PASS' if all_pass else 'FAIL'}")
    if not all_pass:
        print("  FAIL: cosine similarity below 0.999 threshold")
    return all_pass

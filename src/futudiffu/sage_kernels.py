"""SageAttention2 Triton kernel definitions.

Fused flash-attention kernels with FP8/INT8 QK^T matmul for 2x tensor core
throughput on SM89+ (Ada Lovelace). Implements online softmax with BF16 PV
accumulation.

Kernel variants:
- _sage_attn_fwd_fp8qk_bf16pv: FP8 E4M3 QK^T + BF16 PV (Phase 1, no LSE)
- _sage_attn_fwd_fp8qk_bf16pv_lse: Same but also stores LSE for backward pass
- _sage_attn_fwd_fp8qk_fp8pv_lse: FP8 QK^T + FP8 PV with per-column V quant (Phase 1d)
- _sage_attn_d_prepass: D = rowsum(dO * O) pre-pass for backward
- _sage_attn_bwd_dkdv: Backward kernel A: compute dK, dV
- _sage_attn_bwd_dq: Backward kernel B: compute dQ
- _smoke_fp8_dot_kernel: Standalone FP8 dot product validation (Phase 0)

Design notes:
- Q quantized to FP8 per-row (per query token), loaded once per program
- K loaded transposed via pointer arithmetic (D, BLOCK_N), quantized per-column
  (per key token). Avoids tl.trans on FP8 data.
- sm_scale * log2(e) fused into Q dequantization scale so softmax can use
  exp2 (single PTX instruction ex2.approx.f32) instead of exp.
- V stays BF16 for PV matmul to avoid FP22 accumulator precision loss
  (fp8qk_bf16pv variants). The fp8qk_fp8pv variant quantizes V per-column
  for ~2x throughput with slightly reduced accuracy.
- Online softmax (flash-attention-2 style) for memory-efficient streaming.
- Backward kernels recompute P via the same FP8/INT8 quantization path as
  forward, using saved LSE for exact softmax reconstruction. All gradient
  matmuls are BF16 (not FP8).
"""

import logging

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False
    logging.info("SageAttention: Triton not available")


if _HAS_TRITON:
    # =========================================================================
    # Phase 0: FP8 Dot Product Smoke Test
    # =========================================================================

    @triton.jit
    def _smoke_fp8_dot_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        N: tl.constexpr,
        FP8_MAX: tl.constexpr,
    ):
        """Load two (N, N) BF16 tiles, quantize to FP8 per-row, dot, dequant.

        Single thread block, no tiling. Used only for hardware validation.
        """
        offs_r = tl.arange(0, N)
        offs_c = tl.arange(0, N)

        # Load (N, N) tiles
        a = tl.load(a_ptr + offs_r[:, None] * N + offs_c[None, :]).to(tl.float32)
        b = tl.load(b_ptr + offs_r[:, None] * N + offs_c[None, :]).to(tl.float32)

        # Per-row quantization: A -> FP8
        a_abs_max = tl.max(tl.abs(a), axis=1)
        a_inv_scale = FP8_MAX / tl.maximum(a_abs_max, 1e-12)
        a_fp8 = (a * a_inv_scale[:, None]).to(tl.float8e4nv)
        a_descale = a_abs_max / FP8_MAX

        # Per-row quantization: B -> FP8
        b_abs_max = tl.max(tl.abs(b), axis=1)
        b_inv_scale = FP8_MAX / tl.maximum(b_abs_max, 1e-12)
        b_fp8 = (b * b_inv_scale[:, None]).to(tl.float8e4nv)
        b_descale = b_abs_max / FP8_MAX

        # FP8 dot: A @ B^T via transposed B load
        # B is (N, N) row-major. B^T[d, n] = B[n, d].
        # We loaded B as (N, N) and want A_fp8 @ B_fp8^T.
        # Use tl.trans for the smoke test since it's a simple validation.
        c_fp8 = tl.dot(a_fp8, tl.trans(b_fp8), out_dtype=tl.float32)

        # Dequantize: multiply by row scales
        c = c_fp8 * a_descale[:, None] * b_descale[None, :]

        # Store
        tl.store(c_ptr + offs_r[:, None] * N + offs_c[None, :], c.to(tl.bfloat16))

    # =========================================================================
    # Phase 1: FP8 QK + BF16 PV Fused Flash Attention
    # =========================================================================

    @triton.jit
    def _sage_attn_fwd_fp8qk_bf16pv(
        Q, K, V, Out,
        stride_z,    # stride of batch*head dimension (= seq_len * D)
        seq_len,     # total sequence length (queries = keys = values)
        sm_scale_log2e,  # log2(e) / sqrt(D), fused into Q descale
        FP8_MAX: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        D: tl.constexpr,
    ):
        """Fused flash-attention with FP8 QK^T and BF16 PV.

        Grid: (ceil(seq_len / BLOCK_M), batch * heads)

        Memory layout: Q, K, V, Out are all (B*H, N, D) contiguous.
        stride_z = N * D, stride_n = D (implicit), stride_d = 1 (implicit).

        Algorithm:
        1. Load Q block, quantize to FP8 per-row
        2. For each K/V block:
           a. Load K transposed (D, BLOCK_N), quantize to FP8 per-column
           b. FP8 dot: S = Q_fp8 @ K_fp8_T, dequant with fused sm_scale*log2e
           c. Online softmax: exp2 update of running max, sum, output accumulator
           d. BF16 PV: accumulate P @ V in float32
        3. Normalize output by softmax denominator, store as BF16
        """
        pid_m = tl.program_id(0)
        pid_z = tl.program_id(1)

        # Base pointers for this batch*head
        q_base = Q + pid_z * stride_z
        k_base = K + pid_z * stride_z
        v_base = V + pid_z * stride_z
        o_base = Out + pid_z * stride_z

        # Q block offsets
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        q_mask = offs_m < seq_len

        # Load Q tile (BLOCK_M, D) and quantize to FP8 per-row
        q_ptrs = q_base + offs_m[:, None] * D + offs_d[None, :]
        q_f32 = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

        q_abs_max = tl.max(tl.abs(q_f32), axis=1)  # (BLOCK_M,)
        q_inv_scale = FP8_MAX / tl.maximum(q_abs_max, 1e-12)
        q_fp8 = (q_f32 * q_inv_scale[:, None]).to(tl.float8e4nv)
        # Fuse sm_scale * log2(e) into Q dequant scale
        q_descale = (q_abs_max / FP8_MAX) * sm_scale_log2e  # (BLOCK_M,)

        # Online softmax accumulators
        m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        o_acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        # Inner loop over K/V blocks
        n_blocks = tl.cdiv(seq_len, BLOCK_N)
        for n_idx in range(n_blocks):
            n_start = n_idx * BLOCK_N
            offs_n = n_start + tl.arange(0, BLOCK_N)
            n_mask = offs_n < seq_len

            # Load K transposed: (D, BLOCK_N) via pointer arithmetic
            # K[n, d] stored at k_base + n * D + d
            # K_T[d, n] = K[n, d] -> read with offs_d on rows, offs_n on cols
            k_ptrs = k_base + offs_n[None, :] * D + offs_d[:, None]
            k_f32 = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0).to(tl.float32)

            # Per-column (= per key token) FP8 quantization
            k_abs_max = tl.max(tl.abs(k_f32), axis=0)  # (BLOCK_N,)
            k_inv_scale = FP8_MAX / tl.maximum(k_abs_max, 1e-12)
            k_fp8 = (k_f32 * k_inv_scale[None, :]).to(tl.float8e4nv)  # (D, BLOCK_N)
            k_descale = k_abs_max / FP8_MAX  # (BLOCK_N,)

            # FP8 QK^T: (BLOCK_M, D) @ (D, BLOCK_N) = (BLOCK_M, BLOCK_N)
            s = tl.dot(q_fp8, k_fp8, out_dtype=tl.float32)
            # Dequantize + apply sm_scale * log2(e)
            s = s * q_descale[:, None] * k_descale[None, :]

            # Mask out-of-bounds K positions
            s = tl.where(n_mask[None, :], s, float('-inf'))

            # Online softmax update (exp2 since we pre-multiplied log2(e))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(s - m_new[:, None])

            l_i = alpha * l_i + tl.sum(p, axis=1)
            o_acc = o_acc * alpha[:, None]

            # BF16 PV: (BLOCK_M, BLOCK_N) @ (BLOCK_N, D) = (BLOCK_M, D)
            v_ptrs = v_base + offs_n[:, None] * D + offs_d[None, :]
            v_tile = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
            o_acc += tl.dot(p.to(tl.bfloat16), v_tile, out_dtype=tl.float32)

            m_i = m_new

        # Normalize by softmax denominator
        o_acc = o_acc / l_i[:, None]

        # Store output as BF16
        o_ptrs = o_base + offs_m[:, None] * D + offs_d[None, :]
        tl.store(o_ptrs, o_acc.to(tl.bfloat16), mask=q_mask[:, None])

    # =========================================================================
    # Phase 1b: FP8 QK + BF16 PV Fused Flash Attention with LSE output
    # =========================================================================

    @triton.jit
    def _sage_attn_fwd_fp8qk_bf16pv_lse(
        Q, K, V, Out, LSE,
        stride_z,    # stride of batch*head dimension (= seq_len * D)
        seq_len,     # total sequence length (queries = keys = values)
        sm_scale_log2e,  # log2(e) / sqrt(D), fused into Q descale
        FP8_MAX: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        D: tl.constexpr,
    ):
        """Fused flash-attention with FP8 QK^T and BF16 PV, also storing LSE.

        Same as _sage_attn_fwd_fp8qk_bf16pv but additionally stores the
        log-sum-exp (LSE) per query position for use in the backward pass.

        LSE layout: (B*H, N) contiguous float32.
        LSE[z, m] = m_i + log2(l_i) after the full softmax reduction.

        Grid: (ceil(seq_len / BLOCK_M), batch * heads)
        """
        pid_m = tl.program_id(0)
        pid_z = tl.program_id(1)

        # Base pointers for this batch*head
        q_base = Q + pid_z * stride_z
        k_base = K + pid_z * stride_z
        v_base = V + pid_z * stride_z
        o_base = Out + pid_z * stride_z

        # Q block offsets
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        q_mask = offs_m < seq_len

        # Load Q tile (BLOCK_M, D) and quantize to FP8 per-row
        q_ptrs = q_base + offs_m[:, None] * D + offs_d[None, :]
        q_f32 = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

        q_abs_max = tl.max(tl.abs(q_f32), axis=1)  # (BLOCK_M,)
        q_inv_scale = FP8_MAX / tl.maximum(q_abs_max, 1e-12)
        q_fp8 = (q_f32 * q_inv_scale[:, None]).to(tl.float8e4nv)
        # Fuse sm_scale * log2(e) into Q dequant scale
        q_descale = (q_abs_max / FP8_MAX) * sm_scale_log2e  # (BLOCK_M,)

        # Online softmax accumulators
        m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        o_acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        # Inner loop over K/V blocks
        n_blocks = tl.cdiv(seq_len, BLOCK_N)
        for n_idx in range(n_blocks):
            n_start = n_idx * BLOCK_N
            offs_n = n_start + tl.arange(0, BLOCK_N)
            n_mask = offs_n < seq_len

            # Load K transposed: (D, BLOCK_N) via pointer arithmetic
            k_ptrs = k_base + offs_n[None, :] * D + offs_d[:, None]
            k_f32 = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0).to(tl.float32)

            # Per-column (= per key token) FP8 quantization
            k_abs_max = tl.max(tl.abs(k_f32), axis=0)  # (BLOCK_N,)
            k_inv_scale = FP8_MAX / tl.maximum(k_abs_max, 1e-12)
            k_fp8 = (k_f32 * k_inv_scale[None, :]).to(tl.float8e4nv)  # (D, BLOCK_N)
            k_descale = k_abs_max / FP8_MAX  # (BLOCK_N,)

            # FP8 QK^T: (BLOCK_M, D) @ (D, BLOCK_N) = (BLOCK_M, BLOCK_N)
            s = tl.dot(q_fp8, k_fp8, out_dtype=tl.float32)
            # Dequantize + apply sm_scale * log2(e)
            s = s * q_descale[:, None] * k_descale[None, :]

            # Mask out-of-bounds K positions
            s = tl.where(n_mask[None, :], s, float('-inf'))

            # Online softmax update (exp2 since we pre-multiplied log2(e))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(s - m_new[:, None])

            l_i = alpha * l_i + tl.sum(p, axis=1)
            o_acc = o_acc * alpha[:, None]

            # BF16 PV: (BLOCK_M, BLOCK_N) @ (BLOCK_N, D) = (BLOCK_M, D)
            v_ptrs = v_base + offs_n[:, None] * D + offs_d[None, :]
            v_tile = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
            o_acc += tl.dot(p.to(tl.bfloat16), v_tile, out_dtype=tl.float32)

            m_i = m_new

        # Compute LSE = m_i + log2(l_i) before normalization
        lse_i = m_i + tl.math.log2(l_i)  # (BLOCK_M,) float32
        lse_ptrs = LSE + pid_z * seq_len + offs_m
        tl.store(lse_ptrs, lse_i, mask=q_mask)

        # Normalize by softmax denominator
        o_acc = o_acc / l_i[:, None]

        # Store output as BF16
        o_ptrs = o_base + offs_m[:, None] * D + offs_d[None, :]
        tl.store(o_ptrs, o_acc.to(tl.bfloat16), mask=q_mask[:, None])

    # =========================================================================
    # Backward: D pre-pass — D_i = sum_j(dO_ij * O_ij)
    # =========================================================================

    @triton.jit
    def _sage_attn_d_prepass(
        dO, O, Delta,  # pointers: dO (B*H, N, D), O (B*H, N, D), Delta (B*H, N)
        stride_z,       # B*H stride (= N * D)
        seq_len,
        D_HEAD: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Compute D_i = rowsum(dO_i * O_i) for each query position.

        Grid: (ceil(seq_len / BLOCK_M), B*H)
        D is (B*H, N) in float32.
        """
        pid_m = tl.program_id(0)
        pid_z = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D_HEAD)
        mask = offs_m < seq_len

        base = pid_z * stride_z
        do_ptrs = dO + base + offs_m[:, None] * D_HEAD + offs_d[None, :]
        o_ptrs = O + base + offs_m[:, None] * D_HEAD + offs_d[None, :]

        do_vals = tl.load(do_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)
        o_vals = tl.load(o_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)
        delta = tl.sum(do_vals * o_vals, axis=1)  # (BLOCK_M,)

        delta_ptrs = Delta + pid_z * seq_len + offs_m
        tl.store(delta_ptrs, delta, mask=mask)

    # =========================================================================
    # Backward Kernel A: dK, dV
    # =========================================================================

    @triton.jit
    def _sage_attn_bwd_dkdv(
        Q, K, V, dO, O, LSE, Delta,  # inputs
        dK, dV,                        # outputs
        stride_z, seq_len,
        sm_scale_log2e,  # log2(e) / sqrt(D), for P recomputation
        sm_scale,        # 1 / sqrt(D), for dK scaling
        FP8_MAX: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        D: tl.constexpr,
    ):
        """Backward pass computing dK and dV.

        Outer loop over K/V blocks (one per program), inner loop over Q blocks.
        P is recomputed via the same FP8 quantization path as the forward pass.

        Grid: (ceil(seq_len / BLOCK_N), B*H)
        """
        pid_n = tl.program_id(0)  # which K/V block
        pid_z = tl.program_id(1)  # which batch*head

        # Base pointers for this batch*head
        q_base = Q + pid_z * stride_z
        k_base = K + pid_z * stride_z
        v_base = V + pid_z * stride_z
        do_base = dO + pid_z * stride_z

        # K/V block offsets
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)
        n_mask = offs_n < seq_len

        # Load K transposed (D, BLOCK_N) and quantize to FP8 per-column (same as forward)
        k_ptrs_T = k_base + offs_n[None, :] * D + offs_d[:, None]
        k_f32 = tl.load(k_ptrs_T, mask=n_mask[None, :], other=0.0).to(tl.float32)
        k_abs_max = tl.max(tl.abs(k_f32), axis=0)  # (BLOCK_N,)
        k_inv_scale = FP8_MAX / tl.maximum(k_abs_max, 1e-12)
        k_fp8 = (k_f32 * k_inv_scale[None, :]).to(tl.float8e4nv)  # (D, BLOCK_N)
        k_descale = k_abs_max / FP8_MAX  # (BLOCK_N,)

        # Load V block (BLOCK_N, D) transposed for dP computation: (D, BLOCK_N)
        v_ptrs_T = v_base + offs_n[None, :] * D + offs_d[:, None]
        v_T = tl.load(v_ptrs_T, mask=n_mask[None, :], other=0.0)  # (D, BLOCK_N) BF16

        # Accumulators for dK and dV
        dk_acc = tl.zeros([BLOCK_N, D], dtype=tl.float32)
        dv_acc = tl.zeros([BLOCK_N, D], dtype=tl.float32)

        # Inner loop over Q blocks
        m_blocks = tl.cdiv(seq_len, BLOCK_M)
        for m_idx in range(m_blocks):
            m_start = m_idx * BLOCK_M
            offs_m = m_start + tl.arange(0, BLOCK_M)
            q_mask = offs_m < seq_len

            # Load Q block (BLOCK_M, D) and quantize to FP8 per-row (same as forward)
            q_ptrs = q_base + offs_m[:, None] * D + offs_d[None, :]
            q_f32 = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

            q_abs_max = tl.max(tl.abs(q_f32), axis=1)  # (BLOCK_M,)
            q_inv_scale = FP8_MAX / tl.maximum(q_abs_max, 1e-12)
            q_fp8 = (q_f32 * q_inv_scale[:, None]).to(tl.float8e4nv)
            q_descale = (q_abs_max / FP8_MAX) * sm_scale_log2e  # (BLOCK_M,)

            # Recompute P: FP8 QK^T (BLOCK_M, D) @ (D, BLOCK_N) = (BLOCK_M, BLOCK_N)
            s = tl.dot(q_fp8, k_fp8, out_dtype=tl.float32)
            s = s * q_descale[:, None] * k_descale[None, :]
            s = tl.where(n_mask[None, :], s, float('-inf'))

            # Reconstruct P from saved LSE
            lse_ptrs = LSE + pid_z * seq_len + offs_m
            lse_i = tl.load(lse_ptrs, mask=q_mask, other=0.0)  # (BLOCK_M,)
            p = tl.math.exp2(s - lse_i[:, None])  # (BLOCK_M, BLOCK_N)

            # Mask out-of-bounds query positions in P
            p = tl.where(q_mask[:, None], p, 0.0)

            # Load dO block (BLOCK_M, D)
            do_ptrs = do_base + offs_m[:, None] * D + offs_d[None, :]
            do_block = tl.load(do_ptrs, mask=q_mask[:, None], other=0.0)  # BF16

            # Load Delta (D_i) for this Q block
            delta_ptrs = Delta + pid_z * seq_len + offs_m
            delta_i = tl.load(delta_ptrs, mask=q_mask, other=0.0)  # (BLOCK_M,) float32

            # dP = dO @ V^T: (BLOCK_M, D) @ (D, BLOCK_N) = (BLOCK_M, BLOCK_N)
            dp = tl.dot(do_block, v_T, out_dtype=tl.float32)  # (BLOCK_M, BLOCK_N)

            # Softmax backward: ds = P * (dP - D_i)
            ds = p * (dp - delta_i[:, None])  # (BLOCK_M, BLOCK_N)

            # Scale ds by sm_scale for dK accumulation
            ds_scaled = ds * sm_scale  # (BLOCK_M, BLOCK_N)

            # Accumulate dV: dV += P^T @ dO -> (BLOCK_N, BLOCK_M) @ (BLOCK_M, D)
            dv_acc += tl.dot(tl.trans(p.to(tl.bfloat16)), do_block, out_dtype=tl.float32)

            # Accumulate dK: dK += ds_scaled^T @ Q -> (BLOCK_N, BLOCK_M) @ (BLOCK_M, D)
            # Load Q in BF16 for the matmul
            q_bf16 = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)  # BF16
            dk_acc += tl.dot(tl.trans(ds_scaled.to(tl.bfloat16)), q_bf16, out_dtype=tl.float32)

        # Store dK and dV as BF16
        dk_ptrs = dK + pid_z * stride_z + offs_n[:, None] * D + offs_d[None, :]
        dv_ptrs = dV + pid_z * stride_z + offs_n[:, None] * D + offs_d[None, :]
        tl.store(dk_ptrs, dk_acc.to(tl.bfloat16), mask=n_mask[:, None])
        tl.store(dv_ptrs, dv_acc.to(tl.bfloat16), mask=n_mask[:, None])

    # =========================================================================
    # Backward Kernel B: dQ
    # =========================================================================

    @triton.jit
    def _sage_attn_bwd_dq(
        Q, K, V, dO, O, LSE, Delta,  # inputs
        dQ,                            # output
        stride_z, seq_len,
        sm_scale_log2e,  # log2(e) / sqrt(D), for P recomputation
        sm_scale,        # 1 / sqrt(D), for dQ scaling
        FP8_MAX: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        D: tl.constexpr,
    ):
        """Backward pass computing dQ.

        Outer loop over Q blocks (one per program), inner loop over K/V blocks.
        Structurally similar to the forward kernel.

        Grid: (ceil(seq_len / BLOCK_M), B*H)
        """
        pid_m = tl.program_id(0)  # which Q block
        pid_z = tl.program_id(1)  # which batch*head

        # Base pointers for this batch*head
        q_base = Q + pid_z * stride_z
        k_base = K + pid_z * stride_z
        v_base = V + pid_z * stride_z
        do_base = dO + pid_z * stride_z

        # Q block offsets
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        q_mask = offs_m < seq_len

        # Load Q block (BLOCK_M, D) and quantize to FP8 per-row (same as forward)
        q_ptrs = q_base + offs_m[:, None] * D + offs_d[None, :]
        q_f32 = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

        q_abs_max = tl.max(tl.abs(q_f32), axis=1)  # (BLOCK_M,)
        q_inv_scale = FP8_MAX / tl.maximum(q_abs_max, 1e-12)
        q_fp8 = (q_f32 * q_inv_scale[:, None]).to(tl.float8e4nv)
        q_descale = (q_abs_max / FP8_MAX) * sm_scale_log2e  # (BLOCK_M,)

        # Load dO block (BLOCK_M, D)
        do_ptrs = do_base + offs_m[:, None] * D + offs_d[None, :]
        do_block = tl.load(do_ptrs, mask=q_mask[:, None], other=0.0)  # BF16

        # Load LSE and Delta for this Q block
        lse_ptrs = LSE + pid_z * seq_len + offs_m
        lse_i = tl.load(lse_ptrs, mask=q_mask, other=0.0)  # (BLOCK_M,) float32

        delta_ptrs = Delta + pid_z * seq_len + offs_m
        delta_i = tl.load(delta_ptrs, mask=q_mask, other=0.0)  # (BLOCK_M,) float32

        # Accumulator for dQ
        dq_acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        # Inner loop over K/V blocks
        n_blocks = tl.cdiv(seq_len, BLOCK_N)
        for n_idx in range(n_blocks):
            n_start = n_idx * BLOCK_N
            offs_n = n_start + tl.arange(0, BLOCK_N)
            n_mask = offs_n < seq_len

            # Load K transposed (D, BLOCK_N) and quantize to FP8 per-column (same as forward)
            k_ptrs_T = k_base + offs_n[None, :] * D + offs_d[:, None]
            k_f32 = tl.load(k_ptrs_T, mask=n_mask[None, :], other=0.0).to(tl.float32)

            k_abs_max = tl.max(tl.abs(k_f32), axis=0)  # (BLOCK_N,)
            k_inv_scale = FP8_MAX / tl.maximum(k_abs_max, 1e-12)
            k_fp8 = (k_f32 * k_inv_scale[None, :]).to(tl.float8e4nv)  # (D, BLOCK_N)
            k_descale = k_abs_max / FP8_MAX  # (BLOCK_N,)

            # Recompute P: FP8 QK^T (BLOCK_M, D) @ (D, BLOCK_N) = (BLOCK_M, BLOCK_N)
            s = tl.dot(q_fp8, k_fp8, out_dtype=tl.float32)
            s = s * q_descale[:, None] * k_descale[None, :]
            s = tl.where(n_mask[None, :], s, float('-inf'))

            # Reconstruct P from saved LSE
            p = tl.math.exp2(s - lse_i[:, None])  # (BLOCK_M, BLOCK_N)

            # Load V transposed (D, BLOCK_N) for dP computation
            v_ptrs_T = v_base + offs_n[None, :] * D + offs_d[:, None]
            v_T = tl.load(v_ptrs_T, mask=n_mask[None, :], other=0.0)  # (D, BLOCK_N) BF16

            # dP = dO @ V^T: (BLOCK_M, D) @ (D, BLOCK_N) = (BLOCK_M, BLOCK_N)
            dp = tl.dot(do_block, v_T, out_dtype=tl.float32)

            # Softmax backward: ds = P * (dP - D_i)
            ds = p * (dp - delta_i[:, None])  # (BLOCK_M, BLOCK_N)

            # dQ += ds @ K * sm_scale: (BLOCK_M, BLOCK_N) @ (BLOCK_N, D) = (BLOCK_M, D)
            # Load K in normal (BLOCK_N, D) layout for this matmul
            k_ptrs_normal = k_base + offs_n[:, None] * D + offs_d[None, :]
            k_block = tl.load(k_ptrs_normal, mask=n_mask[:, None], other=0.0)  # BF16
            dq_acc += tl.dot(ds.to(tl.bfloat16), k_block, out_dtype=tl.float32)

        # Scale by sm_scale and store dQ as BF16
        dq_acc = dq_acc * sm_scale
        dq_ptrs = dQ + pid_z * stride_z + offs_m[:, None] * D + offs_d[None, :]
        tl.store(dq_ptrs, dq_acc.to(tl.bfloat16), mask=q_mask[:, None])

    # =========================================================================
    # INT8 QK^T Kernels: Symmetric per-token INT8 quantization
    #
    # INT8 has 7 mantissa-equivalent bits vs FP8 E4M3's 3, giving ~16x better
    # quantization accuracy at the same TOPS on SM89 tensor cores. The dot
    # product uses tl.dot(int8, int8) with float32 accumulation, followed by
    # per-row/per-column dequantization.
    #
    # Quantization scheme (symmetric, per-token):
    #   abs_max = max(|x|) across the quantization axis
    #   inv_scale = 127.0 / max(abs_max, 1e-12)
    #   x_int8 = (x * inv_scale).to(int8)       # truncates toward zero
    #   descale = abs_max / 127.0                # dequantization factor
    #
    # Q is quantized per-row (per query token, axis=1).
    # K is loaded transposed (D, BLOCK_N) and quantized per-column (per key
    # token, axis=0 of the transposed layout).
    #
    # The .to(tl.int8) cast truncates toward zero, not round-to-nearest.
    # Maximum per-element error is 0.5/127 ~ 0.4% of abs_max — acceptable.
    # =========================================================================

    @triton.jit
    def _sage_attn_fwd_int8qk_bf16pv(
        Q, K, V, Out,
        stride_z,    # stride of batch*head dimension (= seq_len * D)
        seq_len,     # total sequence length (queries = keys = values)
        sm_scale_log2e,  # log2(e) / sqrt(D), fused into Q descale
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        D: tl.constexpr,
    ):
        """Fused flash-attention with INT8 QK^T and BF16 PV (no LSE output).

        Grid: (ceil(seq_len / BLOCK_M), batch * heads)

        Memory layout: Q, K, V, Out are all (B*H, N, D) contiguous.
        stride_z = N * D, stride_n = D (implicit), stride_d = 1 (implicit).

        Algorithm:
        1. Load Q block, quantize to INT8 per-row (symmetric)
        2. For each K/V block:
           a. Load K transposed (D, BLOCK_N), quantize to INT8 per-column
           b. INT8 dot: S = Q_int8 @ K_int8_T, dequant with fused sm_scale*log2e
           c. Online softmax: exp2 update of running max, sum, output accumulator
           d. BF16 PV: accumulate P @ V in float32
        3. Normalize output by softmax denominator, store as BF16
        """
        pid_m = tl.program_id(0)
        pid_z = tl.program_id(1)

        # Base pointers for this batch*head
        q_base = Q + pid_z * stride_z
        k_base = K + pid_z * stride_z
        v_base = V + pid_z * stride_z
        o_base = Out + pid_z * stride_z

        # Q block offsets
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        q_mask = offs_m < seq_len

        # Load Q tile (BLOCK_M, D) and quantize to INT8 per-row
        q_ptrs = q_base + offs_m[:, None] * D + offs_d[None, :]
        q_f32 = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

        q_abs_max = tl.max(tl.abs(q_f32), axis=1)  # (BLOCK_M,)
        q_inv_scale = 127.0 / tl.maximum(q_abs_max, 1e-12)
        q_int8 = (q_f32 * q_inv_scale[:, None]).to(tl.int8)  # truncates toward zero
        # Fuse sm_scale * log2(e) into Q dequant scale
        q_descale = (q_abs_max / 127.0) * sm_scale_log2e  # (BLOCK_M,)

        # Online softmax accumulators
        m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        o_acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        # Inner loop over K/V blocks
        n_blocks = tl.cdiv(seq_len, BLOCK_N)
        for n_idx in range(n_blocks):
            n_start = n_idx * BLOCK_N
            offs_n = n_start + tl.arange(0, BLOCK_N)
            n_mask = offs_n < seq_len

            # Load K transposed: (D, BLOCK_N) via pointer arithmetic
            # K[n, d] stored at k_base + n * D + d
            # K_T[d, n] = K[n, d] -> read with offs_d on rows, offs_n on cols
            k_ptrs = k_base + offs_n[None, :] * D + offs_d[:, None]
            k_f32 = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0).to(tl.float32)

            # Per-column (= per key token) INT8 quantization
            k_abs_max = tl.max(tl.abs(k_f32), axis=0)  # (BLOCK_N,)
            k_inv_scale = 127.0 / tl.maximum(k_abs_max, 1e-12)
            k_int8 = (k_f32 * k_inv_scale[None, :]).to(tl.int8)  # (D, BLOCK_N)
            k_descale = k_abs_max / 127.0  # (BLOCK_N,)

            # INT8 QK^T: (BLOCK_M, D) @ (D, BLOCK_N) = (BLOCK_M, BLOCK_N)
            s = tl.dot(q_int8, k_int8, out_dtype=tl.float32)
            # Dequantize + apply sm_scale * log2(e)
            s = s * q_descale[:, None] * k_descale[None, :]

            # Mask out-of-bounds K positions
            s = tl.where(n_mask[None, :], s, float('-inf'))

            # Online softmax update (exp2 since we pre-multiplied log2(e))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(s - m_new[:, None])

            l_i = alpha * l_i + tl.sum(p, axis=1)
            o_acc = o_acc * alpha[:, None]

            # BF16 PV: (BLOCK_M, BLOCK_N) @ (BLOCK_N, D) = (BLOCK_M, D)
            v_ptrs = v_base + offs_n[:, None] * D + offs_d[None, :]
            v_tile = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
            o_acc += tl.dot(p.to(tl.bfloat16), v_tile, out_dtype=tl.float32)

            m_i = m_new

        # Normalize by softmax denominator
        o_acc = o_acc / l_i[:, None]

        # Store output as BF16
        o_ptrs = o_base + offs_m[:, None] * D + offs_d[None, :]
        tl.store(o_ptrs, o_acc.to(tl.bfloat16), mask=q_mask[:, None])

    # =========================================================================
    # INT8 QK + BF16 PV Fused Flash Attention with LSE output
    # =========================================================================

    @triton.jit
    def _sage_attn_fwd_int8qk_bf16pv_lse(
        Q, K, V, Out, LSE,
        stride_z,    # stride of batch*head dimension (= seq_len * D)
        seq_len,     # total sequence length (queries = keys = values)
        sm_scale_log2e,  # log2(e) / sqrt(D), fused into Q descale
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        D: tl.constexpr,
    ):
        """Fused flash-attention with INT8 QK^T and BF16 PV, also storing LSE.

        Same as _sage_attn_fwd_int8qk_bf16pv but additionally stores the
        log-sum-exp (LSE) per query position for use in the backward pass.

        LSE layout: (B*H, N) contiguous float32.
        LSE[z, m] = m_i + log2(l_i) after the full softmax reduction.

        Grid: (ceil(seq_len / BLOCK_M), batch * heads)
        """
        pid_m = tl.program_id(0)
        pid_z = tl.program_id(1)

        # Base pointers for this batch*head
        q_base = Q + pid_z * stride_z
        k_base = K + pid_z * stride_z
        v_base = V + pid_z * stride_z
        o_base = Out + pid_z * stride_z

        # Q block offsets
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        q_mask = offs_m < seq_len

        # Load Q tile (BLOCK_M, D) and quantize to INT8 per-row
        q_ptrs = q_base + offs_m[:, None] * D + offs_d[None, :]
        q_f32 = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

        q_abs_max = tl.max(tl.abs(q_f32), axis=1)  # (BLOCK_M,)
        q_inv_scale = 127.0 / tl.maximum(q_abs_max, 1e-12)
        q_int8 = (q_f32 * q_inv_scale[:, None]).to(tl.int8)
        # Fuse sm_scale * log2(e) into Q dequant scale
        q_descale = (q_abs_max / 127.0) * sm_scale_log2e  # (BLOCK_M,)

        # Online softmax accumulators
        m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        o_acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        # Inner loop over K/V blocks
        n_blocks = tl.cdiv(seq_len, BLOCK_N)
        for n_idx in range(n_blocks):
            n_start = n_idx * BLOCK_N
            offs_n = n_start + tl.arange(0, BLOCK_N)
            n_mask = offs_n < seq_len

            # Load K transposed: (D, BLOCK_N) via pointer arithmetic
            k_ptrs = k_base + offs_n[None, :] * D + offs_d[:, None]
            k_f32 = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0).to(tl.float32)

            # Per-column (= per key token) INT8 quantization
            k_abs_max = tl.max(tl.abs(k_f32), axis=0)  # (BLOCK_N,)
            k_inv_scale = 127.0 / tl.maximum(k_abs_max, 1e-12)
            k_int8 = (k_f32 * k_inv_scale[None, :]).to(tl.int8)  # (D, BLOCK_N)
            k_descale = k_abs_max / 127.0  # (BLOCK_N,)

            # INT8 QK^T: (BLOCK_M, D) @ (D, BLOCK_N) = (BLOCK_M, BLOCK_N)
            s = tl.dot(q_int8, k_int8, out_dtype=tl.float32)
            # Dequantize + apply sm_scale * log2(e)
            s = s * q_descale[:, None] * k_descale[None, :]

            # Mask out-of-bounds K positions
            s = tl.where(n_mask[None, :], s, float('-inf'))

            # Online softmax update (exp2 since we pre-multiplied log2(e))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(s - m_new[:, None])

            l_i = alpha * l_i + tl.sum(p, axis=1)
            o_acc = o_acc * alpha[:, None]

            # BF16 PV: (BLOCK_M, BLOCK_N) @ (BLOCK_N, D) = (BLOCK_M, D)
            v_ptrs = v_base + offs_n[:, None] * D + offs_d[None, :]
            v_tile = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
            o_acc += tl.dot(p.to(tl.bfloat16), v_tile, out_dtype=tl.float32)

            m_i = m_new

        # Compute LSE = m_i + log2(l_i) before normalization
        lse_i = m_i + tl.math.log2(l_i)  # (BLOCK_M,) float32
        lse_ptrs = LSE + pid_z * seq_len + offs_m
        tl.store(lse_ptrs, lse_i, mask=q_mask)

        # Normalize by softmax denominator
        o_acc = o_acc / l_i[:, None]

        # Store output as BF16
        o_ptrs = o_base + offs_m[:, None] * D + offs_d[None, :]
        tl.store(o_ptrs, o_acc.to(tl.bfloat16), mask=q_mask[:, None])

    # =========================================================================
    # Backward Kernel A (INT8): dK, dV with INT8 P recomputation
    # =========================================================================

    @triton.jit
    def _sage_attn_bwd_dkdv_int8(
        Q, K, V, dO, O, LSE, Delta,  # inputs
        dK, dV,                        # outputs
        stride_z, seq_len,
        sm_scale_log2e,  # log2(e) / sqrt(D), for P recomputation
        sm_scale,        # 1 / sqrt(D), for dK scaling
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        D: tl.constexpr,
    ):
        """Backward pass computing dK and dV with INT8 P recomputation.

        Outer loop over K/V blocks (one per program), inner loop over Q blocks.
        P is recomputed via the same INT8 quantization path as the forward pass.
        All gradient matmuls (dP, dV, dK) are in BF16.

        Grid: (ceil(seq_len / BLOCK_N), B*H)
        """
        pid_n = tl.program_id(0)  # which K/V block
        pid_z = tl.program_id(1)  # which batch*head

        # Base pointers for this batch*head
        q_base = Q + pid_z * stride_z
        k_base = K + pid_z * stride_z
        v_base = V + pid_z * stride_z
        do_base = dO + pid_z * stride_z

        # K/V block offsets
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)
        n_mask = offs_n < seq_len

        # Load K transposed (D, BLOCK_N) and quantize to INT8 per-column (same as forward)
        k_ptrs_T = k_base + offs_n[None, :] * D + offs_d[:, None]
        k_f32 = tl.load(k_ptrs_T, mask=n_mask[None, :], other=0.0).to(tl.float32)
        k_abs_max = tl.max(tl.abs(k_f32), axis=0)  # (BLOCK_N,)
        k_inv_scale = 127.0 / tl.maximum(k_abs_max, 1e-12)
        k_int8 = (k_f32 * k_inv_scale[None, :]).to(tl.int8)  # (D, BLOCK_N)
        k_descale = k_abs_max / 127.0  # (BLOCK_N,)

        # Load V block (BLOCK_N, D) transposed for dP computation: (D, BLOCK_N)
        v_ptrs_T = v_base + offs_n[None, :] * D + offs_d[:, None]
        v_T = tl.load(v_ptrs_T, mask=n_mask[None, :], other=0.0)  # (D, BLOCK_N) BF16

        # Accumulators for dK and dV
        dk_acc = tl.zeros([BLOCK_N, D], dtype=tl.float32)
        dv_acc = tl.zeros([BLOCK_N, D], dtype=tl.float32)

        # Inner loop over Q blocks
        m_blocks = tl.cdiv(seq_len, BLOCK_M)
        for m_idx in range(m_blocks):
            m_start = m_idx * BLOCK_M
            offs_m = m_start + tl.arange(0, BLOCK_M)
            q_mask = offs_m < seq_len

            # Load Q block (BLOCK_M, D) and quantize to INT8 per-row (same as forward)
            q_ptrs = q_base + offs_m[:, None] * D + offs_d[None, :]
            q_f32 = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

            q_abs_max = tl.max(tl.abs(q_f32), axis=1)  # (BLOCK_M,)
            q_inv_scale = 127.0 / tl.maximum(q_abs_max, 1e-12)
            q_int8 = (q_f32 * q_inv_scale[:, None]).to(tl.int8)
            q_descale = (q_abs_max / 127.0) * sm_scale_log2e  # (BLOCK_M,)

            # Recompute P: INT8 QK^T (BLOCK_M, D) @ (D, BLOCK_N) = (BLOCK_M, BLOCK_N)
            s = tl.dot(q_int8, k_int8, out_dtype=tl.float32)
            s = s * q_descale[:, None] * k_descale[None, :]
            s = tl.where(n_mask[None, :], s, float('-inf'))

            # Reconstruct P from saved LSE
            lse_ptrs = LSE + pid_z * seq_len + offs_m
            lse_i = tl.load(lse_ptrs, mask=q_mask, other=0.0)  # (BLOCK_M,)
            p = tl.math.exp2(s - lse_i[:, None])  # (BLOCK_M, BLOCK_N)

            # Mask out-of-bounds query positions in P
            p = tl.where(q_mask[:, None], p, 0.0)

            # Load dO block (BLOCK_M, D)
            do_ptrs = do_base + offs_m[:, None] * D + offs_d[None, :]
            do_block = tl.load(do_ptrs, mask=q_mask[:, None], other=0.0)  # BF16

            # Load Delta (D_i) for this Q block
            delta_ptrs = Delta + pid_z * seq_len + offs_m
            delta_i = tl.load(delta_ptrs, mask=q_mask, other=0.0)  # (BLOCK_M,) float32

            # dP = dO @ V^T: (BLOCK_M, D) @ (D, BLOCK_N) = (BLOCK_M, BLOCK_N)
            dp = tl.dot(do_block, v_T, out_dtype=tl.float32)  # (BLOCK_M, BLOCK_N)

            # Softmax backward: ds = P * (dP - D_i)
            ds = p * (dp - delta_i[:, None])  # (BLOCK_M, BLOCK_N)

            # Scale ds by sm_scale for dK accumulation
            ds_scaled = ds * sm_scale  # (BLOCK_M, BLOCK_N)

            # Accumulate dV: dV += P^T @ dO -> (BLOCK_N, BLOCK_M) @ (BLOCK_M, D)
            dv_acc += tl.dot(tl.trans(p.to(tl.bfloat16)), do_block, out_dtype=tl.float32)

            # Accumulate dK: dK += ds_scaled^T @ Q -> (BLOCK_N, BLOCK_M) @ (BLOCK_M, D)
            # Load Q in BF16 for the matmul
            q_bf16 = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)  # BF16
            dk_acc += tl.dot(tl.trans(ds_scaled.to(tl.bfloat16)), q_bf16, out_dtype=tl.float32)

        # Store dK and dV as BF16
        dk_ptrs = dK + pid_z * stride_z + offs_n[:, None] * D + offs_d[None, :]
        dv_ptrs = dV + pid_z * stride_z + offs_n[:, None] * D + offs_d[None, :]
        tl.store(dk_ptrs, dk_acc.to(tl.bfloat16), mask=n_mask[:, None])
        tl.store(dv_ptrs, dv_acc.to(tl.bfloat16), mask=n_mask[:, None])

    # =========================================================================
    # Backward Kernel B (INT8): dQ with INT8 P recomputation
    # =========================================================================

    @triton.jit
    def _sage_attn_bwd_dq_int8(
        Q, K, V, dO, O, LSE, Delta,  # inputs
        dQ,                            # output
        stride_z, seq_len,
        sm_scale_log2e,  # log2(e) / sqrt(D), for P recomputation
        sm_scale,        # 1 / sqrt(D), for dQ scaling
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        D: tl.constexpr,
    ):
        """Backward pass computing dQ with INT8 P recomputation.

        Outer loop over Q blocks (one per program), inner loop over K/V blocks.
        Structurally similar to the forward kernel.

        Grid: (ceil(seq_len / BLOCK_M), B*H)
        """
        pid_m = tl.program_id(0)  # which Q block
        pid_z = tl.program_id(1)  # which batch*head

        # Base pointers for this batch*head
        q_base = Q + pid_z * stride_z
        k_base = K + pid_z * stride_z
        v_base = V + pid_z * stride_z
        do_base = dO + pid_z * stride_z

        # Q block offsets
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        q_mask = offs_m < seq_len

        # Load Q block (BLOCK_M, D) and quantize to INT8 per-row (same as forward)
        q_ptrs = q_base + offs_m[:, None] * D + offs_d[None, :]
        q_f32 = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

        q_abs_max = tl.max(tl.abs(q_f32), axis=1)  # (BLOCK_M,)
        q_inv_scale = 127.0 / tl.maximum(q_abs_max, 1e-12)
        q_int8 = (q_f32 * q_inv_scale[:, None]).to(tl.int8)
        q_descale = (q_abs_max / 127.0) * sm_scale_log2e  # (BLOCK_M,)

        # Load dO block (BLOCK_M, D)
        do_ptrs = do_base + offs_m[:, None] * D + offs_d[None, :]
        do_block = tl.load(do_ptrs, mask=q_mask[:, None], other=0.0)  # BF16

        # Load LSE and Delta for this Q block
        lse_ptrs = LSE + pid_z * seq_len + offs_m
        lse_i = tl.load(lse_ptrs, mask=q_mask, other=0.0)  # (BLOCK_M,) float32

        delta_ptrs = Delta + pid_z * seq_len + offs_m
        delta_i = tl.load(delta_ptrs, mask=q_mask, other=0.0)  # (BLOCK_M,) float32

        # Accumulator for dQ
        dq_acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        # Inner loop over K/V blocks
        n_blocks = tl.cdiv(seq_len, BLOCK_N)
        for n_idx in range(n_blocks):
            n_start = n_idx * BLOCK_N
            offs_n = n_start + tl.arange(0, BLOCK_N)
            n_mask = offs_n < seq_len

            # Load K transposed (D, BLOCK_N) and quantize to INT8 per-column (same as forward)
            k_ptrs_T = k_base + offs_n[None, :] * D + offs_d[:, None]
            k_f32 = tl.load(k_ptrs_T, mask=n_mask[None, :], other=0.0).to(tl.float32)

            k_abs_max = tl.max(tl.abs(k_f32), axis=0)  # (BLOCK_N,)
            k_inv_scale = 127.0 / tl.maximum(k_abs_max, 1e-12)
            k_int8 = (k_f32 * k_inv_scale[None, :]).to(tl.int8)  # (D, BLOCK_N)
            k_descale = k_abs_max / 127.0  # (BLOCK_N,)

            # Recompute P: INT8 QK^T (BLOCK_M, D) @ (D, BLOCK_N) = (BLOCK_M, BLOCK_N)
            s = tl.dot(q_int8, k_int8, out_dtype=tl.float32)
            s = s * q_descale[:, None] * k_descale[None, :]
            s = tl.where(n_mask[None, :], s, float('-inf'))

            # Reconstruct P from saved LSE
            p = tl.math.exp2(s - lse_i[:, None])  # (BLOCK_M, BLOCK_N)

            # Load V transposed (D, BLOCK_N) for dP computation
            v_ptrs_T = v_base + offs_n[None, :] * D + offs_d[:, None]
            v_T = tl.load(v_ptrs_T, mask=n_mask[None, :], other=0.0)  # (D, BLOCK_N) BF16

            # dP = dO @ V^T: (BLOCK_M, D) @ (D, BLOCK_N) = (BLOCK_M, BLOCK_N)
            dp = tl.dot(do_block, v_T, out_dtype=tl.float32)

            # Softmax backward: ds = P * (dP - D_i)
            ds = p * (dp - delta_i[:, None])  # (BLOCK_M, BLOCK_N)

            # dQ += ds @ K * sm_scale: (BLOCK_M, BLOCK_N) @ (BLOCK_N, D) = (BLOCK_M, D)
            # Load K in normal (BLOCK_N, D) layout for this matmul
            k_ptrs_normal = k_base + offs_n[:, None] * D + offs_d[None, :]
            k_block = tl.load(k_ptrs_normal, mask=n_mask[:, None], other=0.0)  # BF16
            dq_acc += tl.dot(ds.to(tl.bfloat16), k_block, out_dtype=tl.float32)

        # Scale by sm_scale and store dQ as BF16
        dq_acc = dq_acc * sm_scale
        dq_ptrs = dQ + pid_z * stride_z + offs_m[:, None] * D + offs_d[None, :]
        tl.store(dq_ptrs, dq_acc.to(tl.bfloat16), mask=q_mask[:, None])

    # =========================================================================
    # Phase 1d: FP8 QK + FP8 PV Fused Flash Attention with LSE output
    #
    # Both QK^T and PV matmuls run on FP8 tensor cores (660 TFLOPS on SM89),
    # giving ~2x total attention throughput vs the FP8 QK + BF16 PV variant.
    #
    # P is quantized per-row (per query position) and V is quantized per-column
    # (per feature dimension d). Per-column V quantization is required so that
    # the scale factor is constant across the reduction dimension n, allowing
    # clean factorization after the dot product:
    #
    #   pv_true[m,d] = p_scale[m] * v_scale[d] * dot(P_fp8, V_fp8)[m,d]
    #
    # Two-level accumulation is inherent: SM89's FP8 tensor cores use an FP22
    # internal accumulator within each tl.dot call (reduction over BLOCK_N=64,
    # i.e. 4 MMA tiles), while the outer loop accumulation (o_acc += pv) is
    # true FP32. No explicit flushing is needed for these block sizes.
    # =========================================================================

    @triton.jit
    def _sage_attn_fwd_fp8qk_fp8pv_lse(
        Q, K, V, Out, LSE,
        stride_z,    # stride of batch*head dimension (= seq_len * D)
        seq_len,     # total sequence length (queries = keys = values)
        sm_scale_log2e,  # log2(e) / sqrt(D), fused into Q descale
        FP8_MAX: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        D: tl.constexpr,
    ):
        """Fused flash-attention with FP8 QK^T and FP8 PV, also storing LSE.

        Same as _sage_attn_fwd_fp8qk_bf16pv_lse but additionally quantizes
        P (attention weights) per-row and V per-column (per feature dimension)
        to FP8 for the PV matmul. This uses FP8 tensor cores for both QK^T
        and PV, giving ~2x total attention throughput vs BF16 PV.

        The SM89 FP8 tensor cores use an FP22 internal accumulator. For the
        PV matmul, the reduction dimension is BLOCK_N (typically 64), meaning
        only 4 MMA tiles (64/16=4) accumulate in FP22 per tl.dot call. The
        outer accumulation across N-blocks is true FP32 (o_acc += pv), so
        this is inherently two-level: FP22 within each dot, FP32 across dots.

        Dequantization math for PV:
            pv_true[m,d] = sum_n(P[m,n] * V[n,d])
                         = sum_n(P_fp8[m,n]*p_scale[m] * V_fp8[n,d]*v_scale[d])
                         = p_scale[m] * v_scale[d] * sum_n(P_fp8[m,n] * V_fp8[n,d])
                         = p_scale[m] * v_scale[d] * tl.dot(P_fp8, V_fp8)[m,d]
        Per-column V quantization (v_scale[d]) factors cleanly out of the sum
        because the scale is constant across the reduction dimension n.

        LSE layout: (B*H, N) contiguous float32.
        LSE[z, m] = m_i + log2(l_i) after the full softmax reduction.

        Grid: (ceil(seq_len / BLOCK_M), batch * heads)
        """
        pid_m = tl.program_id(0)
        pid_z = tl.program_id(1)

        # Base pointers for this batch*head
        q_base = Q + pid_z * stride_z
        k_base = K + pid_z * stride_z
        v_base = V + pid_z * stride_z
        o_base = Out + pid_z * stride_z

        # Q block offsets
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        q_mask = offs_m < seq_len

        # Load Q tile (BLOCK_M, D) and quantize to FP8 per-row
        q_ptrs = q_base + offs_m[:, None] * D + offs_d[None, :]
        q_f32 = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

        q_abs_max = tl.max(tl.abs(q_f32), axis=1)  # (BLOCK_M,)
        q_inv_scale = FP8_MAX / tl.maximum(q_abs_max, 1e-12)
        q_fp8 = (q_f32 * q_inv_scale[:, None]).to(tl.float8e4nv)
        # Fuse sm_scale * log2(e) into Q dequant scale
        q_descale = (q_abs_max / FP8_MAX) * sm_scale_log2e  # (BLOCK_M,)

        # Online softmax accumulators
        m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        o_acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        # Inner loop over K/V blocks
        n_blocks = tl.cdiv(seq_len, BLOCK_N)
        for n_idx in range(n_blocks):
            n_start = n_idx * BLOCK_N
            offs_n = n_start + tl.arange(0, BLOCK_N)
            n_mask = offs_n < seq_len

            # Load K transposed: (D, BLOCK_N) via pointer arithmetic
            k_ptrs = k_base + offs_n[None, :] * D + offs_d[:, None]
            k_f32 = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0).to(tl.float32)

            # Per-column (= per key token) FP8 quantization
            k_abs_max = tl.max(tl.abs(k_f32), axis=0)  # (BLOCK_N,)
            k_inv_scale = FP8_MAX / tl.maximum(k_abs_max, 1e-12)
            k_fp8 = (k_f32 * k_inv_scale[None, :]).to(tl.float8e4nv)  # (D, BLOCK_N)
            k_descale = k_abs_max / FP8_MAX  # (BLOCK_N,)

            # FP8 QK^T: (BLOCK_M, D) @ (D, BLOCK_N) = (BLOCK_M, BLOCK_N)
            s = tl.dot(q_fp8, k_fp8, out_dtype=tl.float32)
            # Dequantize + apply sm_scale * log2(e)
            s = s * q_descale[:, None] * k_descale[None, :]

            # Mask out-of-bounds K positions
            s = tl.where(n_mask[None, :], s, float('-inf'))

            # Online softmax update (exp2 since we pre-multiplied log2(e))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(s - m_new[:, None])

            l_i = alpha * l_i + tl.sum(p, axis=1)
            o_acc = o_acc * alpha[:, None]

            # FP8 PV: quantize P per-row and V per-column, then FP8 dot
            # P (BLOCK_M, BLOCK_N) float32 -> FP8 per-row
            p_abs_max_pv = tl.max(tl.abs(p), axis=1)  # (BLOCK_M,)
            p_inv_scale_pv = FP8_MAX / tl.maximum(p_abs_max_pv, 1e-12)
            p_fp8 = (p * p_inv_scale_pv[:, None]).to(tl.float8e4nv)
            p_descale_pv = p_abs_max_pv / FP8_MAX  # (BLOCK_M,)

            # V (BLOCK_N, D) -> load and quantize per-column (per feature dim)
            v_ptrs = v_base + offs_n[:, None] * D + offs_d[None, :]
            v_f32 = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
            v_abs_max = tl.max(tl.abs(v_f32), axis=0)  # (D,) per-column
            v_inv_scale = FP8_MAX / tl.maximum(v_abs_max, 1e-12)
            v_fp8 = (v_f32 * v_inv_scale[None, :]).to(tl.float8e4nv)  # (BLOCK_N, D)
            v_descale = v_abs_max / FP8_MAX  # (D,)

            # FP8 dot: (BLOCK_M, BLOCK_N) @ (BLOCK_N, D) = (BLOCK_M, D)
            pv = tl.dot(p_fp8, v_fp8, out_dtype=tl.float32)
            # Dequantize: p_scale[m] * v_scale[d]
            pv = pv * p_descale_pv[:, None] * v_descale[None, :]
            o_acc += pv

            m_i = m_new

        # Compute LSE = m_i + log2(l_i) before normalization
        lse_i = m_i + tl.math.log2(l_i)  # (BLOCK_M,) float32
        lse_ptrs = LSE + pid_z * seq_len + offs_m
        tl.store(lse_ptrs, lse_i, mask=q_mask)

        # Normalize by softmax denominator
        o_acc = o_acc / l_i[:, None]

        # Store output as BF16
        o_ptrs = o_base + offs_m[:, None] * D + offs_d[None, :]
        tl.store(o_ptrs, o_acc.to(tl.bfloat16), mask=q_mask[:, None])

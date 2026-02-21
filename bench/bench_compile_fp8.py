"""'This Better Frickin Compile' benchmark.

Minimal DNN using our actual FP8 blockwise linears + 3D RoPE.
Two ResNet-like layers, hdim=256, residual_dim=64, 8x8 image, 32-token text.
Measures: compile time, 256 NFEs wall clock.

Usage:
    .venv/Scripts/python.exe -u bench/bench_compile_fp8.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
import torch.nn as nn
from futudiffu.fp8 import FP8Linear, quantize_fp8_blockwise, BLOCK_SIZE
from futudiffu.attention import rope_embed, apply_rope_flux

HDIM = 256
RESIDUAL_DIM = 64
IMG_H, IMG_W = 8, 8
IMG_SEQ = IMG_H * IMG_W  # 64
TEXT_SEQ = 32
TOTAL_SEQ = IMG_SEQ + TEXT_SEQ  # 96
N_HEADS = 4
HEAD_DIM = HDIM // N_HEADS  # 64
ROPE_DIM = HEAD_DIM  # each head gets full rope


def make_fp8_linear(in_f, out_f, bias=False):
    """Create an FP8Linear from random BF16 weights."""
    # Pad to block_size if needed
    pad_out = ((out_f + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    pad_in = ((in_f + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    w_full = torch.randn(pad_out, pad_in, dtype=torch.bfloat16) * 0.02
    qw, qs = quantize_fp8_blockwise(w_full)
    # Slice back to actual dims if we padded
    qw = qw[:out_f, :in_f].contiguous().cuda()
    qs_padded = qs  # scales are for padded dims
    # Recompute scales for the actual weight
    qw_full, qs_full = quantize_fp8_blockwise(w_full[:out_f, :in_f].float()
                                               if out_f % BLOCK_SIZE == 0 and in_f % BLOCK_SIZE == 0
                                               else w_full)
    if out_f % BLOCK_SIZE == 0 and in_f % BLOCK_SIZE == 0:
        qw, qs = qw_full.cuda(), qs_full.cuda()
    else:
        # Fallback: just use regular linear for non-aligned dims
        raise ValueError(f"Dims must be multiples of {BLOCK_SIZE}: got ({out_f}, {in_f})")

    b = torch.zeros(out_f, dtype=torch.bfloat16).cuda() if bias else None
    linear = FP8Linear(qw, qs, bias=b, block_size=BLOCK_SIZE, output_dtype=torch.bfloat16)
    linear.transpose_weight()
    return linear


class ResBlockFP8(nn.Module):
    """ResNet block using FP8 blockwise linears.

    x -> RMSNorm -> FP8Linear(hdim, hdim) -> SiLU -> FP8Linear(hdim, hdim) -> + x
    """

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.RMSNorm(dim, elementwise_affine=True).cuda().to(torch.bfloat16)
        self.lin1 = make_fp8_linear(dim, dim)
        self.lin2 = make_fp8_linear(dim, dim)

    def forward(self, x):
        h = self.norm(x)
        h = self.lin1(h)
        h = torch.nn.functional.silu(h)
        h = self.lin2(h)
        return x + h


class TinyDiT(nn.Module):
    """Minimal DiT-like model: project -> 2x ResBlock with attention + RoPE -> project out."""

    def __init__(self):
        super().__init__()
        # Input projection: residual_dim -> hdim (bf16, non-aligned dims)
        self.proj_in = nn.Linear(RESIDUAL_DIM, HDIM, bias=False, dtype=torch.bfloat16, device="cuda")
        # Two ResNet blocks
        self.block1 = ResBlockFP8(HDIM)
        self.block2 = ResBlockFP8(HDIM)
        # QKV for self-attention (standard bf16 — small enough)
        self.qkv = nn.Linear(HDIM, 3 * HDIM, bias=False, dtype=torch.bfloat16, device="cuda")
        self.out_proj = nn.Linear(HDIM, HDIM, bias=False, dtype=torch.bfloat16, device="cuda")
        # Output projection: hdim -> residual_dim (bf16, non-aligned dims)
        self.proj_out = nn.Linear(HDIM, RESIDUAL_DIM, bias=False, dtype=torch.bfloat16, device="cuda")
        # Timestep embedding (simple)
        self.t_embed = nn.Sequential(
            nn.Linear(1, HDIM, dtype=torch.bfloat16, device="cuda"),
            nn.SiLU(),
            nn.Linear(HDIM, HDIM, dtype=torch.bfloat16, device="cuda"),
        )
        # Register buffer for rope cache (avoid recompilation from Python branching)
        self._build_rope_cache()

    def _build_rope_cache(self):
        """Build 3D RoPE: 2D grid for image tokens + 1D for text tokens."""
        # Image: 2D grid positions
        ys = torch.arange(IMG_H, device="cuda").float()
        xs = torch.arange(IMG_W, device="cuda").float()
        grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1)  # (H, W, 2)
        img_pos = grid.reshape(IMG_SEQ, 2)  # (64, 2)
        # Pad to 3D (y, x, 0)
        img_pos_3d = torch.cat([img_pos, torch.zeros(IMG_SEQ, 1, device="cuda")], dim=-1)

        # Text: 1D positions
        txt_pos = torch.arange(TEXT_SEQ, device="cuda").float()
        txt_pos_3d = torch.stack([
            torch.zeros(TEXT_SEQ, device="cuda"),
            torch.zeros(TEXT_SEQ, device="cuda"),
            txt_pos,
        ], dim=-1)  # (32, 3)

        # Combine
        all_pos = torch.cat([img_pos_3d, txt_pos_3d], dim=0)  # (96, 3)

        # Build rope for each axis, concat
        rope_dim_per_axis = HEAD_DIM // 3
        # Pad last axis to make dims work out (64 / 3 = 21.33... -> 22, 22, 20)
        dims = [rope_dim_per_axis + 1, rope_dim_per_axis + 1, HEAD_DIM - 2 * (rope_dim_per_axis + 1)]

        rope_parts = []
        for axis in range(3):
            pos_1d = all_pos[:, axis].unsqueeze(0)  # (1, 96)
            r = rope_embed(pos_1d, dims[axis], theta=10000.0)  # (1, 96, dim//2, 2, 2)
            rope_parts.append(r)

        # Concat along the dim axis
        rope = torch.cat(rope_parts, dim=2)  # (1, 96, HEAD_DIM//2, 2, 2)
        self.register_buffer("rope", rope)

    def forward(self, x, t):
        """
        x: (B, TOTAL_SEQ, RESIDUAL_DIM) bf16
        t: (B,) float — timestep/sigma
        """
        B = x.shape[0]
        # Project in
        h = self.proj_in(x)  # (B, 96, HDIM)

        # Add timestep embedding (broadcast over seq)
        t_emb = self.t_embed(t.unsqueeze(-1))  # (B, HDIM)
        h = h + t_emb.unsqueeze(1)

        # Block 1
        h = self.block1(h)

        # Self-attention with RoPE
        qkv = self.qkv(h)  # (B, 96, 3*HDIM)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, TOTAL_SEQ, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.reshape(B, TOTAL_SEQ, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.reshape(B, TOTAL_SEQ, N_HEADS, HEAD_DIM).transpose(1, 2)

        # Apply RoPE
        rope_expanded = self.rope.expand(B, -1, -1, -1, -1)
        q_flat = q.reshape(B, N_HEADS * TOTAL_SEQ, HEAD_DIM)
        k_flat = k.reshape(B, N_HEADS * TOTAL_SEQ, HEAD_DIM)
        # Apply per-head (rope is same for all heads)
        rope_per_head = rope_expanded.repeat(1, N_HEADS, 1, 1, 1)
        q_roped, k_roped = apply_rope_flux(q_flat, k_flat, rope_per_head)
        q = q_roped.reshape(B, N_HEADS, TOTAL_SEQ, HEAD_DIM)
        k = k_roped.reshape(B, N_HEADS, TOTAL_SEQ, HEAD_DIM)

        # SDPA
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, TOTAL_SEQ, HDIM)
        h = h + self.out_proj(attn_out)

        # Block 2
        h = self.block2(h)

        # Project out
        return self.proj_out(h)  # (B, 96, RESIDUAL_DIM)


def main():
    torch.manual_seed(42)
    device = torch.device("cuda")

    print("=== 'This Better Frickin Compile' Benchmark ===")
    print(f"  Model: 2x ResBlock(FP8 {HDIM}d) + RoPE attention")
    print(f"  Input: B=2, seq={TOTAL_SEQ} (img {IMG_H}x{IMG_W} + txt {TEXT_SEQ}), dim={RESIDUAL_DIM}")
    print(f"  Attention: {N_HEADS} heads x {HEAD_DIM}d, 3D RoPE")
    print(f"  FP8 block size: {BLOCK_SIZE}")
    print()

    # Build model
    print("Building model...", end="", flush=True)
    t0 = time.perf_counter()
    model = TinyDiT()
    model.eval()
    build_time = time.perf_counter() - t0
    print(f" {build_time:.2f}s")

    n_params = sum(p.numel() for p in model.parameters())
    n_buffers = sum(b.numel() for b in model.buffers())
    print(f"  Params: {n_params:,}  Buffers: {n_buffers:,}")
    print()

    # Test inputs (B=2 to match CFG batching)
    x = torch.randn(2, TOTAL_SEQ, RESIDUAL_DIM, dtype=torch.bfloat16, device=device)
    t = torch.tensor([1.0, 0.5], dtype=torch.bfloat16, device=device)

    # Eager warmup
    print("Eager forward (sanity check)...", end="", flush=True)
    with torch.no_grad():
        out = model(x, t)
    torch.cuda.synchronize()
    print(f" output shape={out.shape}, dtype={out.dtype}")
    print()

    # === COMPILE ===
    print("torch.compile(fullgraph=True)...", end="", flush=True)
    t0 = time.perf_counter()
    compiled = torch.compile(model, fullgraph=True)

    # First forward triggers actual compilation
    with torch.no_grad():
        out_compiled = compiled(x, t)
    torch.cuda.synchronize()
    compile_time = time.perf_counter() - t0
    print(f" {compile_time:.2f}s")

    # Verify compiled matches eager
    cos = torch.nn.functional.cosine_similarity(
        out.flatten().float(), out_compiled.flatten().float(), dim=0
    )
    print(f"  Compiled vs eager cosine: {cos:.8f}")
    print()

    # === BENCHMARK: 256 NFEs ===
    N_NFES = 256
    # Warmup compiled path (3 extra forwards)
    with torch.no_grad():
        for _ in range(3):
            compiled(x, t)
    torch.cuda.synchronize()

    print(f"Benchmarking {N_NFES} NFEs...", end="", flush=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    with torch.no_grad():
        for i in range(N_NFES):
            out = compiled(x, t)

    end_event.record()
    torch.cuda.synchronize()
    wall_time = time.perf_counter() - t0
    cuda_time_ms = start_event.elapsed_time(end_event)
    print(f" done")
    print()

    per_nfe_wall_ms = (wall_time * 1000) / N_NFES
    per_nfe_cuda_ms = cuda_time_ms / N_NFES

    print("=== RESULTS ===")
    print(f"  Build time:    {build_time:.2f}s")
    print(f"  Compile time:  {compile_time:.2f}s")
    print(f"  {N_NFES} NFEs wall:  {wall_time:.3f}s  ({per_nfe_wall_ms:.2f}ms/NFE)")
    print(f"  {N_NFES} NFEs CUDA:  {cuda_time_ms:.1f}ms  ({per_nfe_cuda_ms:.2f}ms/NFE)")
    print()

    # Save results
    results = {
        "model": "TinyDiT_2xResBlock_FP8",
        "hdim": HDIM,
        "residual_dim": RESIDUAL_DIM,
        "img_hw": [IMG_H, IMG_W],
        "text_seq": TEXT_SEQ,
        "total_seq": TOTAL_SEQ,
        "n_heads": N_HEADS,
        "head_dim": HEAD_DIM,
        "batch_size": 2,
        "fp8_block_size": BLOCK_SIZE,
        "n_params": n_params,
        "n_buffers": n_buffers,
        "build_time_s": round(build_time, 3),
        "compile_time_s": round(compile_time, 3),
        "n_nfes": N_NFES,
        "wall_time_s": round(wall_time, 4),
        "cuda_time_ms": round(cuda_time_ms, 1),
        "per_nfe_wall_ms": round(per_nfe_wall_ms, 3),
        "per_nfe_cuda_ms": round(per_nfe_cuda_ms, 3),
        "compiled_vs_eager_cos": round(cos.item(), 8),
    }
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "bench_compile_output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"results_{time.strftime('%Y%m%d_%H%M%S')}.json")
    # Use Windows path for writing
    win_out_dir = out_dir.replace("/mnt/f/", "F:\\").replace("/", "\\")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {out_path}")


if __name__ == "__main__":
    main()

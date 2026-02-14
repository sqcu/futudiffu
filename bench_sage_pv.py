"""Benchmark: SageAttention kernel variants vs PyTorch SDPA.

Tests B=2, H=30, N=4288, D=128 (actual NextDiT Z-Image diffusion model shapes).

Variants:
  1. PyTorch SDPA (BF16 baseline)
  2. Sage FP8 QK + BF16 PV
  3. Sage INT8 QK + BF16 PV
  4. Sage FP8 QK + FP8 PV
"""

import sys
sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

import math
import torch
import torch.nn.functional as F

from futudiffu.sage_attention import (
    configure_sage,
    sage_attn_forward,
    sage_attn_forward_with_lse,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
B, H, N, D = 2, 30, 4288, 128
WARMUP = 5
ITERS = 20
STEPS_PER_IMAGE = 34  # 30 euler steps * 2 (cfg batching) + some extra = ~34 attn calls per step? Actually 34 is stated.

device = "cuda"
dtype = torch.bfloat16

print(f"Benchmark: SageAttention kernel variants")
print(f"Shape: B={B}, H={H}, N={N}, D={D}")
print(f"Device: {torch.cuda.get_device_name()}")
print(f"Warmup: {WARMUP}, Timed iters: {ITERS}")
print(f"Extrapolation: {STEPS_PER_IMAGE} calls per denoising loop")
print()

# Create random inputs (same for all variants)
torch.manual_seed(42)
q = torch.randn(B, H, N, D, device=device, dtype=dtype)
k = torch.randn(B, H, N, D, device=device, dtype=dtype)
v = torch.randn(B, H, N, D, device=device, dtype=dtype)
sm_scale = 1.0 / math.sqrt(D)

# ---------------------------------------------------------------------------
# Variant 1: PyTorch SDPA (BF16 baseline reference)
# ---------------------------------------------------------------------------
print("Running variant 1: PyTorch SDPA (BF16 baseline)...")
# Warmup
for _ in range(WARMUP):
    ref = F.scaled_dot_product_attention(q, k, v, scale=sm_scale)
torch.cuda.synchronize()

# Time
start_evt = torch.cuda.Event(enable_timing=True)
end_evt = torch.cuda.Event(enable_timing=True)
start_evt.record()
for _ in range(ITERS):
    ref = F.scaled_dot_product_attention(q, k, v, scale=sm_scale)
end_evt.record()
torch.cuda.synchronize()
sdpa_ms = start_evt.elapsed_time(end_evt) / ITERS
sdpa_ref = ref.clone()
print(f"  {sdpa_ms:.3f} ms/call")

# ---------------------------------------------------------------------------
# Variant 2: Sage FP8 QK + BF16 PV
# ---------------------------------------------------------------------------
print("Running variant 2: Sage FP8 QK + BF16 PV...")
configure_sage(smooth_k=True, qk_quant="fp8", pv_quant="bf16")
# Warmup (includes Triton JIT compilation)
for _ in range(WARMUP):
    out2 = sage_attn_forward(q, k, v, sm_scale)
torch.cuda.synchronize()

start_evt = torch.cuda.Event(enable_timing=True)
end_evt = torch.cuda.Event(enable_timing=True)
start_evt.record()
for _ in range(ITERS):
    out2 = sage_attn_forward(q, k, v, sm_scale)
end_evt.record()
torch.cuda.synchronize()
fp8qk_bf16pv_ms = start_evt.elapsed_time(end_evt) / ITERS
fp8qk_bf16pv_out = out2.clone()
print(f"  {fp8qk_bf16pv_ms:.3f} ms/call")

# ---------------------------------------------------------------------------
# Variant 3: Sage INT8 QK + BF16 PV
# ---------------------------------------------------------------------------
print("Running variant 3: Sage INT8 QK + BF16 PV...")
configure_sage(smooth_k=True, qk_quant="int8", pv_quant="bf16")
# Warmup
for _ in range(WARMUP):
    out3 = sage_attn_forward(q, k, v, sm_scale)
torch.cuda.synchronize()

start_evt = torch.cuda.Event(enable_timing=True)
end_evt = torch.cuda.Event(enable_timing=True)
start_evt.record()
for _ in range(ITERS):
    out3 = sage_attn_forward(q, k, v, sm_scale)
end_evt.record()
torch.cuda.synchronize()
int8qk_bf16pv_ms = start_evt.elapsed_time(end_evt) / ITERS
int8qk_bf16pv_out = out3.clone()
print(f"  {int8qk_bf16pv_ms:.3f} ms/call")

# ---------------------------------------------------------------------------
# Variant 4: Sage FP8 QK + FP8 PV (via sage_attn_forward_with_lse)
# ---------------------------------------------------------------------------
print("Running variant 4: Sage FP8 QK + FP8 PV...")
configure_sage(smooth_k=True, qk_quant="fp8", pv_quant="fp8")
# Warmup
for _ in range(WARMUP):
    out4, lse4 = sage_attn_forward_with_lse(q, k, v, sm_scale)
torch.cuda.synchronize()

start_evt = torch.cuda.Event(enable_timing=True)
end_evt = torch.cuda.Event(enable_timing=True)
start_evt.record()
for _ in range(ITERS):
    out4, lse4 = sage_attn_forward_with_lse(q, k, v, sm_scale)
end_evt.record()
torch.cuda.synchronize()
fp8qk_fp8pv_ms = start_evt.elapsed_time(end_evt) / ITERS
fp8qk_fp8pv_out = out4.clone()
print(f"  {fp8qk_fp8pv_ms:.3f} ms/call")

# ---------------------------------------------------------------------------
# Cosine similarity vs SDPA reference
# ---------------------------------------------------------------------------
def cos_sim(a, b):
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()

cos_sdpa = 1.0  # self-reference
cos_fp8_bf16 = cos_sim(fp8qk_bf16pv_out, sdpa_ref)
cos_int8_bf16 = cos_sim(int8qk_bf16pv_out, sdpa_ref)
cos_fp8_fp8 = cos_sim(fp8qk_fp8pv_out, sdpa_ref)

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
print()
print("=" * 90)
print(f"{'Variant':<30} {'ms/call':>8} {'x34 calls':>12} {'Speedup':>10} {'cos_sim':>10}")
print(f"{'':30} {'':>8} {'(ms)':>12} {'vs SDPA':>10} {'vs SDPA':>10}")
print("-" * 90)

variants = [
    ("1. SDPA (BF16 baseline)",   sdpa_ms,           cos_sdpa),
    ("2. FP8 QK + BF16 PV",      fp8qk_bf16pv_ms,   cos_fp8_bf16),
    ("3. INT8 QK + BF16 PV",     int8qk_bf16pv_ms,   cos_int8_bf16),
    ("4. FP8 QK + FP8 PV",       fp8qk_fp8pv_ms,     cos_fp8_fp8),
]

for name, ms, cos in variants:
    total_ms = ms * STEPS_PER_IMAGE
    speedup = sdpa_ms / ms
    print(f"{name:<30} {ms:>8.3f} {total_ms:>12.1f} {speedup:>9.2f}x {cos:>10.6f}")

print("=" * 90)
print()

# Memory info
mem_alloc = torch.cuda.max_memory_allocated() / (1024**3)
mem_reserved = torch.cuda.max_memory_reserved() / (1024**3)
print(f"Peak CUDA memory: {mem_alloc:.2f} GiB allocated, {mem_reserved:.2f} GiB reserved")

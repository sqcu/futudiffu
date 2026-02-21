"""Benchmark: trace the EXACT production server warmup code path with a tiny model.

Reproduces the server's ensure_diffusion() -> torch.compile -> forward_packed()
flow using the S-S-S (Stubbed-Skinny-Shared) model. If the pathology is in the
warmup FLOW (not model SIZE), this will still be slow.

## What this tests

The production server hung for 20+ minutes during warmup. A standalone TinyDiT
compiled in 5.56s. The difference: the TinyDiT benchmark compiled a simple
model.forward() with fullgraph=True, while the production server:

1. Installs a forward_hook (HiddenCapture) BEFORE torch.compile
2. Compiles model.forward_packed (NOT model.forward) with mode="default"
   (NOT fullgraph=True) -- allows graph breaks
3. Runs warmup_packed_single which creates FlexAttention block_mask,
   packed RoPE, etc.
4. The first forward_packed call triggers compilation with:
   - 30+ distinct Python branches (fused elementwise, fused QKV, fused chain)
   - FlexAttention's flex_attention() call inside each of 30 attention layers
   - Multiple custom_ops (FP8 GEMM, SiLU+gate+requant, RMSNorm fusions)
   - A forward_hook that mode="default" must trace through

## BUGS FOUND DURING BENCHMARK DEVELOPMENT

### Bug 1: fused_qkv_postprocess incompatible with forward_packed rope shape
The fused_qkv_postprocess kernel expects freqs_cis shape (B, seq, 1, n_pairs, 2, 2)
but forward_packed passes packed_rope shape (1, total_len, 1, 64, 2, 2). The kernel's
assertion `freqs_cis.shape[2] == 1` passes because the packed rope's dim-2 IS 1, but
the kernel then indexes freqs_ptr as (B * seq, n_pairs, 2, 2), which is incompatible
with the packed rope layout where seq is in dim-1 (not dim-2).

### Bug 2: _pad_for_packed_single rope padding assumes wrong tensor layout
The code indexes packed_rope[:, :, :natural_cap_len, :, :, :] assuming the tensor
has shape (1, n_axes, total_len, 1, dim, 2) per its docstring. But the ACTUAL shape
from build_packed_rope is (1, total_len, 1, 64, 2, 2). This means:
- dim 2 is the broadcast dim (size 1), NOT the sequence dim
- Slicing [:, :, :natural_cap_len, ...] on dim 2 (size 1) always returns the full rope
- The padding operation corrupts the rope by inserting zeros in the wrong dimension
- rope_img ends up empty (size 0 in dim 2), and the cat produces a wrong-shaped tensor
This bug manifests as a RuntimeError during torch.compile fake tensor propagation.

## Architecture of the test

Phase 0: Load S-S-S model (real FP8 weights, tiny dims)
Phase 1: fuse_model() -- same as production (with fused QKV disabled for packed path)
Phase 2: Install HiddenCapture -- same as production
Phase 3: torch.compile(model.forward_packed, mode="default") -- same as production
Phase 4: Manually-constructed warmup (same ops as warmup_packed_single but bypassing
         the broken _pad_for_packed_single rope padding)
Phase 5-8: Controls isolating hook, FlexAttention, and compile mode contributions

Usage:
    PYTHONUNBUFFERED=1 .venv/Scripts/python.exe -u bench/bench_warmup_flow.py
"""

import json
import os
import sys
import time

# Setup paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tests"))

import torch
torch.set_float32_matmul_precision("high")


def manual_warmup_forward_packed(
    diff_compiled_packed, diff_model, device, dtype,
    latent_h=32, latent_w=32, cap_len=32,
):
    """Warmup forward_packed with manually-constructed inputs.

    Bypasses warmup_packed_single and _pad_for_packed_single to avoid the rope
    padding bug (Bug 2). Uses the same functions that warmup_packed_single calls
    (prepare_packed_state, create_block_mask) but with matching cap_len so
    extra_padding = 0.

    Returns elapsed seconds (includes compilation time).
    """
    from futudiffu.diffusion_model import make_packing_mask_mod
    from torch.nn.attention.flex_attention import create_block_mask

    pH = pW = diff_model.patch_size
    padded_h = latent_h + ((-latent_h) % pH)
    padded_w = latent_w + ((-latent_w) % pW)

    # Dummy conditioning -- cap_len matches what prepare_packed_state will produce
    dummy_cond = torch.zeros(2, cap_len, 2560, device=device, dtype=dtype)

    with torch.inference_mode():
        refined_caps, packing_info, packed_rope = \
            diff_model.prepare_packed_state(
                [dummy_cond],
                [(padded_h, padded_w)],
                [cap_len],
                device,
            )

        block_mask = create_block_mask(
            make_packing_mask_mod(packing_info.document_id),
            B=2, H=None,
            Q_LEN=packing_info.total_len,
            KV_LEN=packing_info.total_len,
            device=device,
        )

        x_list = [
            torch.randn(2, 16, latent_h, latent_w, device=device, dtype=dtype)
        ]
        t_batch = torch.tensor([0.5, 0.5], device=device, dtype=dtype)

        t0 = time.perf_counter()
        diff_compiled_packed(
            x_list, t_batch, refined_caps,
            packing_info, block_mask, packed_rope,
        )
        torch.cuda.synchronize()
        return time.perf_counter() - t0


# --- Phase 0: Load S-S-S model ---
print("=" * 70)
print("BENCH: Server Warmup Flow (S-S-S model)")
print("=" * 70)
print()

print("[Phase 0] Loading S-S-S model (real FP8 weights, tiny dims)...")
t0 = time.perf_counter()
from stubbed_skinny_shared import load_sss_model, SSS_DIM, SSS_CAP_FEAT_DIM
model = load_sss_model(device="cuda")
load_time = time.perf_counter() - t0
print(f"  Loaded in {load_time:.2f}s")
print(f"  dim={model.dim}, n_heads={model.n_heads}, n_layers={len(model.layers)}")
n_params = sum(p.numel() for p in model.parameters())
total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"  Params: {n_params:,} ({total_bytes / 1e6:.1f} MB)")
print()

# --- Phase 1: fuse_model() (same as production ensure_diffusion) ---
print("[Phase 1] fuse_model() -- same as production...")
t0 = time.perf_counter()
from futudiffu.diffusion_model import fuse_model
fuse_model(model)
fuse_time = time.perf_counter() - t0
print(f"  Fused in {fuse_time:.3f}s")

# IMPORTANT: Disable fused QKV for forward_packed path (Bug 1).
# The fused_qkv_postprocess kernel expects freqs_cis shape (B, seq, 1, n_pairs, 2, 2)
# but forward_packed passes packed_rope with incompatible layout.
# The production server likely also has this bug -- forward_packed is the ONLY
# compiled forward path, and fuse_model() enables _use_fused_qkv on all attention
# modules. Either (a) the production server crashes here too, or (b) the
# non-fused fallback path in JointAttention.forward() is taken by accident.
from futudiffu.diffusion_model import JointAttention
for m in model.modules():
    if isinstance(m, JointAttention):
        m._use_fused_qkv = False
print("  NOTE: Disabled fused QKV (Bug 1: incompatible with packed rope shape)")

# Check what fusions are active
n_fused_elem = sum(1 for m in model.modules()
                   if getattr(m, '_use_fused_elementwise', False))
n_fused_qkv = sum(1 for m in model.modules()
                  if getattr(m, '_use_fused_qkv', False))
n_fused_chain = sum(1 for m in model.modules()
                    if getattr(m, '_fused_chain', False))
has_adaln_cache = hasattr(model, '_adaln_W')
print(f"  Fused elementwise blocks: {n_fused_elem}")
print(f"  Fused QKV blocks: {n_fused_qkv}")
print(f"  Fused FP8 chain blocks: {n_fused_chain}")
print(f"  Batched adaLN cache: {has_adaln_cache}")
print()

# --- Phase 2: Install HiddenCapture (same as production) ---
print("[Phase 2] Install HiddenCapture hook (BEFORE compile, same as production)...")
from futudiffu.training_utils import HiddenCapture
hidden_capture = HiddenCapture(model)
hidden_capture.install()
print(f"  Hook installed on: {model.layers[-1].__class__.__name__} (last main layer)")
print()

# --- Phase 3: torch.compile (EXACT same call as production) ---
print("[Phase 3] torch.compile(model.forward_packed, mode='default') -- same as production...")
print("  NOTE: mode='default' allows graph breaks (hooks survive)")
print("  Calling torch.compile (does NOT compile yet -- lazy)...", end="", flush=True)
t0 = time.perf_counter()
torch._dynamo.reset()
diff_compiled_packed = torch.compile(model.forward_packed, mode="default")
compile_wrap_time = time.perf_counter() - t0
print(f" {compile_wrap_time:.3f}s")
print()

# --- Phase 4: Manual warmup (same code path, bypassing rope padding bug) ---
print("[Phase 4] Manual warmup (forward_packed via prepare_packed_state)...")
print("  Same as warmup_packed_single but without broken _pad_for_packed_single")
print("  This is where the actual compilation happens (first forward pass)")

# Use a small resolution to isolate compilation time from compute time.
WIDTH = 256
HEIGHT = 256
LATENT_H = HEIGHT // 8
LATENT_W = WIDTH // 8

device = torch.device("cuda")
dtype = torch.bfloat16

t0 = time.perf_counter()
warmup_elapsed = manual_warmup_forward_packed(
    diff_compiled_packed, model,
    device, dtype,
    latent_h=LATENT_H, latent_w=LATENT_W, cap_len=32,
)
total_phase4 = time.perf_counter() - t0
print(f"  forward_packed compile + first call: {warmup_elapsed:.2f}s")
print(f"  Total Phase 4 wall clock:           {total_phase4:.2f}s")
print()

# Verify the hook still fires after compilation
print("  Verifying HiddenCapture hook fires after compile...", end="", flush=True)
captured = hidden_capture.captured
if captured is not None:
    print(f" YES (shape={captured.shape})")
    hidden_capture.captured = None
else:
    print(" NO (hook did NOT fire -- graph break may have excluded it)")
print()

# --- Phase 4b: Second forward (should be instant -- already compiled) ---
print("[Phase 4b] Second forward (should reuse compiled graph)...")
t0 = time.perf_counter()
warmup2_elapsed = manual_warmup_forward_packed(
    diff_compiled_packed, model,
    device, dtype,
    latent_h=LATENT_H, latent_w=LATENT_W, cap_len=32,
)
total_phase4b = time.perf_counter() - t0
print(f"  Second call: {warmup2_elapsed:.4f}s (should be << Phase 4)")
print()

# --- Phase 5: Control - compile forward_packed with fullgraph=True ---
print("[Phase 5] CONTROL: torch.compile(forward_packed, fullgraph=True)...")
print("  If this is MUCH faster than Phase 4, graph breaks are the pathology.")

torch._dynamo.reset()
hidden_capture_2 = HiddenCapture(model)
hidden_capture_2.install()

try:
    diff_compiled_fullgraph = torch.compile(model.forward_packed, fullgraph=True)

    t0 = time.perf_counter()
    warmup_fg_elapsed = manual_warmup_forward_packed(
        diff_compiled_fullgraph, model,
        device, dtype,
        latent_h=LATENT_H, latent_w=LATENT_W, cap_len=32,
    )
    total_phase5 = time.perf_counter() - t0
    fullgraph_error = None
    print(f"  fullgraph compile + warmup: {total_phase5:.2f}s")
except Exception as e:
    fullgraph_error = str(e)[:200]
    total_phase5 = None
    warmup_fg_elapsed = None
    print(f"  fullgraph=True FAILED: {fullgraph_error}")
    print("  (Expected: hooks, flex_attention, or data-dependent control flow)")
print()

hidden_capture_2.remove()

# --- Phase 6: Control - compile forward_packed WITHOUT hook ---
print("[Phase 6] CONTROL: compile forward_packed WITHOUT HiddenCapture hook...")
print("  If this is much faster than Phase 4, the hook is the pathology.")

torch._dynamo.reset()
hidden_capture.remove()

diff_compiled_nohook = torch.compile(model.forward_packed, mode="default")

t0 = time.perf_counter()
warmup_nohook_elapsed = manual_warmup_forward_packed(
    diff_compiled_nohook, model,
    device, dtype,
    latent_h=LATENT_H, latent_w=LATENT_W, cap_len=32,
)
total_phase6 = time.perf_counter() - t0
print(f"  No-hook compile + warmup: {total_phase6:.2f}s")
print()

# --- Phase 7: Control - compile with "reduce-overhead" mode ---
print("[Phase 7] CONTROL: torch.compile(forward_packed, mode='reduce-overhead')...")
print("  If this is much slower, reduce-overhead overhead is pathological at scale.")

torch._dynamo.reset()

try:
    diff_compiled_ro = torch.compile(model.forward_packed, mode="reduce-overhead")

    t0 = time.perf_counter()
    warmup_ro_elapsed = manual_warmup_forward_packed(
        diff_compiled_ro, model,
        device, dtype,
        latent_h=LATENT_H, latent_w=LATENT_W, cap_len=32,
    )
    total_phase7 = time.perf_counter() - t0
    ro_error = None
    print(f"  reduce-overhead compile + warmup: {total_phase7:.2f}s")
except Exception as e:
    ro_error = str(e)[:200]
    total_phase7 = None
    warmup_ro_elapsed = None
    print(f"  reduce-overhead FAILED: {ro_error}")
print()

# --- Phase 8: Compile just model.forward() (not forward_packed) for comparison ---
print("[Phase 8] CONTROL: compile model.forward() instead of forward_packed...")
print("  If much faster, forward_packed's packing/FlexAttention is the pathology.")

torch._dynamo.reset()

diff_compiled_forward = torch.compile(model.forward, mode="default")

# Create simple inputs for model.forward()
B = 2  # CFG batch
x_simple = torch.randn(B, 16, LATENT_H, LATENT_W, device=device, dtype=dtype)
t_simple = torch.tensor([0.5, 0.5], device=device, dtype=dtype)
ctx_simple = torch.randn(B, 32, SSS_CAP_FEAT_DIM, device=device, dtype=dtype)

t0 = time.perf_counter()
with torch.inference_mode():
    out_simple = diff_compiled_forward(x_simple, t_simple, ctx_simple, num_tokens=32)
torch.cuda.synchronize()
total_phase8 = time.perf_counter() - t0
print(f"  model.forward() compile + warmup: {total_phase8:.2f}s")
print(f"  Output shape: {out_simple.shape}")
print()

# --- Results summary ---
print("=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print()
print(f"  Model load:                        {load_time:.2f}s")
print(f"  fuse_model():                      {fuse_time:.3f}s")
print()
print(f"  Phase 4: Production warmup flow    {total_phase4:.2f}s  <-- THE NUMBER")
print(f"    (mode='default', hook, forward_packed, FlexAttention)")
print(f"  Phase 4b: Second call (cached)     {total_phase4b:.4f}s")
print()
print(f"  Phase 5: fullgraph=True            ", end="")
if total_phase5 is not None:
    print(f"{total_phase5:.2f}s")
else:
    print(f"FAILED ({fullgraph_error[:60]}...)")
print(f"  Phase 6: No hook                   {total_phase6:.2f}s")
print(f"  Phase 7: reduce-overhead           ", end="")
if total_phase7 is not None:
    print(f"{total_phase7:.2f}s")
else:
    print(f"FAILED ({ro_error[:60]}...)")
print(f"  Phase 8: model.forward() only      {total_phase8:.2f}s")
print()

# Diagnosis
print("DIAGNOSIS:")
print()
print("  BUGS FOUND (blocking correct execution of production warmup path):")
print("  1. fused_qkv_postprocess kernel is incompatible with forward_packed rope shape")
print("  2. _pad_for_packed_single assumes wrong rope tensor layout (dim order)")
print("     Both bugs are in source files -- requires fixes in diffusion_model.py")
print("     and sampling.py (or fused_kernels.py). See docstring for details.")
print()

if total_phase4 > 60:
    print("  TIMING: PATHOLOGICAL (>60s on tiny model) -- problem is in the FLOW")
    if total_phase6 < total_phase4 * 0.5:
        print("  --> HiddenCapture hook is a major contributor")
    if total_phase5 is not None and total_phase5 < total_phase4 * 0.5:
        print("  --> Graph breaks from mode='default' are a major contributor")
    if total_phase8 < total_phase4 * 0.5:
        print("  --> forward_packed / FlexAttention is a major contributor")
elif total_phase4 > 15:
    print("  TIMING: MODERATELY SLOW (15-60s) -- some overhead from flow")
    if total_phase6 < total_phase4 * 0.7:
        print("  --> Hook adds measurable overhead")
    if total_phase8 < total_phase4 * 0.7:
        print("  --> FlexAttention packing adds measurable overhead")
else:
    print("  TIMING: FAST (<15s) -- problem is likely pure model SIZE")
    print("  Production: 32 layers x 3840d x 30 heads = ~15x more inductor work")

ratio_p4_p8 = total_phase4 / total_phase8 if total_phase8 > 0.01 else float('inf')
print(f"\n  forward_packed / forward ratio: {ratio_p4_p8:.1f}x")
print(f"  (>3x means FlexAttention packing is significant overhead)")

# Save results
results = {
    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    "model": "S-S-S (Stubbed-Skinny-Shared)",
    "model_dim": SSS_DIM,
    "model_n_layers": len(model.layers),
    "model_n_heads": model.n_heads,
    "model_params": n_params,
    "model_bytes": total_bytes,
    "warmup_resolution": f"{WIDTH}x{HEIGHT}",
    "fusions": {
        "fused_elementwise": n_fused_elem,
        "fused_qkv": n_fused_qkv,
        "fused_chain": n_fused_chain,
        "batched_adaln": has_adaln_cache,
    },
    "bugs_found": [
        "fused_qkv_postprocess incompatible with forward_packed rope shape",
        "_pad_for_packed_single assumes wrong rope tensor layout (dim 1 is total_len, not n_axes)",
    ],
    "timings_seconds": {
        "load": round(load_time, 3),
        "fuse": round(fuse_time, 4),
        "phase4_production_flow": round(total_phase4, 3),
        "phase4b_cached": round(total_phase4b, 4),
        "phase5_fullgraph": round(total_phase5, 3) if total_phase5 is not None else None,
        "phase5_error": fullgraph_error,
        "phase6_no_hook": round(total_phase6, 3),
        "phase7_reduce_overhead": round(total_phase7, 3) if total_phase7 is not None else None,
        "phase7_error": ro_error if total_phase7 is None else None,
        "phase8_forward_only": round(total_phase8, 3),
    },
    "cuda_device": torch.cuda.get_device_name(0),
    "torch_version": torch.__version__,
}

out_dir = os.path.join(REPO_ROOT, "bench_warmup_output")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"results_{time.strftime('%Y%m%d_%H%M%S')}.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved: {out_path}")

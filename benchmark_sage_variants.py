"""Benchmark all SageAttention variants: SDPA, FP8+BF16, INT8+BF16, FP8+FP8.

Runs warmup + timed generation for each variant, saves output images,
and reports NFE/min, cosine similarity, and PSNR vs SDPA baseline.
"""

import os
import shutil
import sys
import time

# NOTE: Do NOT clear the torchinductor disk cache — it stores compiled code
# and dramatically speeds up warmup. The CUDA graph contamination issue is
# from Dynamo's in-process cache, fixed by torch._dynamo.reset() between variants.

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

import numpy as np
import torch
from PIL import Image

from futudiffu.generate import GenerateConfig, generate

PROMPT = (
    'ahem.\n*ting ting ting ting ting*\nthe query model for this is a LARGE '
    'LANGUAGE MODEL, specifically QWEN-3-4B, a GENERAL PURPOSE SEMANTIC PARSER '
    'which is able to WRITE SENTENCES AT A TIME when they are participating in '
    'dialogue. however, in this situation, they are being used as a hidden state '
    'generator to steer an *image generation model*, z-image.\n\nqwen-3-4b, draw '
    'me an "enormous laser shark for the sega saturn".'
)

COMMON = dict(
    diffusion_model_path=r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors",
    text_encoder_path=r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors",
    vae_path=r"F:\dox\ai\comfyui\ComfyUI\models\vae\zimage.safetensors",
    tokenizer_path=r"F:\dox\repos\ai\futudiffu\src\futudiffu\tokenizer",
    prompt=PROMPT,
    negative_prompt="",
    seed=91849188298864,
    steps=30,
    cfg=4.0,
    width=1280,
    height=832,
    fp8_diffusion=True,
    dtype="bfloat16",
)

VARIANTS = [
    ("SDPA", dict(attention_backend="sdpa")),
    ("FP8 QK + BF16 PV", dict(
        attention_backend="sage",
        sage_qk_quant="fp8",
        sage_pv_quant="bf16",
        sage_smooth_k=True,
    )),
    ("INT8 QK + BF16 PV", dict(
        attention_backend="sage",
        sage_qk_quant="int8",
        sage_pv_quant="bf16",
        sage_smooth_k=True,
    )),
    ("FP8 QK + FP8 PV", dict(
        attention_backend="sage",
        sage_qk_quant="fp8",
        sage_pv_quant="fp8",
        sage_smooth_k=True,
    )),
]

OUT_DIR = r"F:\dox\repos\ai\futudiffu\bench_renders"
os.makedirs(OUT_DIR, exist_ok=True)

results = {}

for name, overrides in VARIANTS:
    # Reset Dynamo compilation cache between variants to prevent CUDA graph
    # contamination. Without this, torch.compile(mode="reduce-overhead") can
    # reuse CUDA graphs from a previous variant's capture, replaying the wrong
    # Triton kernels (since the custom_op dispatch is opaque to Dynamo guards).
    torch._dynamo.reset()

    print(f"\n{'=' * 70}")
    print(f"  VARIANT: {name}")
    print(f"{'=' * 70}")

    cfg = GenerateConfig(**{**COMMON, **overrides})

    # Warmup run (triggers torch.compile + CUDA graph capture)
    print(f"\n--- Warmup run ---")
    t0 = time.perf_counter()
    _ = generate(cfg)
    torch.cuda.synchronize()
    warmup_time = time.perf_counter() - t0
    print(f"  Warmup: {warmup_time:.1f}s")

    # DO NOT reset between warmup and timed — we WANT to reuse the CUDA graphs
    # captured during warmup. The reset between variants (above) ensures each
    # variant captures its own CUDA graphs with the correct kernels.

    # Timed run (reuses CUDA graphs from warmup — fast replay)
    print(f"\n--- Timed run ---")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    img = generate(cfg)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    nfe_min = 30.0 / elapsed * 60.0
    results[name] = {"image": img, "time": elapsed, "nfe_min": nfe_min, "warmup": warmup_time}

    # Save individual image
    safe_name = name.replace(" ", "_").replace("+", "p").lower()
    fname = os.path.join(OUT_DIR, f"{safe_name}.png")
    Image.fromarray(img[0]).save(fname)
    print(f"  Saved: {fname}")
    print(f"  Time: {elapsed:.1f}s | NFE/min: {nfe_min:.1f}")

# --- Compute metrics vs SDPA baseline ---
baseline = results["SDPA"]["image"].astype(np.float32) / 255.0
baseline_flat = baseline.flatten()
baseline_norm = np.linalg.norm(baseline_flat)

print(f"\n\n{'=' * 70}")
print(f"  BENCHMARK RESULTS — SageAttention Variant Comparison")
print(f"{'=' * 70}")
print(f"  Config: FP8 diffusion, BF16 TE, seed=91849188298864, 30 steps, CFG=4.0")
print(f"  Resolution: 1280x832, Model: Z-Image NextDiT")
print(f"{'=' * 70}")
print()
print(f"{'Variant':<22} {'Time':>7} {'NFE/min':>8} {'Speedup':>8} {'Cos Sim':>10} {'PSNR':>8}")
print(f"{'-' * 22} {'-' * 7} {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 8}")

sdpa_nfe = results["SDPA"]["nfe_min"]

for name, res in results.items():
    img_f = res["image"].astype(np.float32) / 255.0
    img_flat = img_f.flatten()

    # Cosine similarity
    cos = np.dot(baseline_flat, img_flat) / (baseline_norm * np.linalg.norm(img_flat))

    # PSNR
    mse = np.mean((baseline - img_f) ** 2)
    psnr = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)

    speedup = res["nfe_min"] / sdpa_nfe

    res["cos_sim"] = cos
    res["psnr"] = psnr
    res["speedup"] = speedup

    psnr_str = f"{psnr:.1f}dB" if psnr != float("inf") else "inf"
    cos_str = f"{cos:.6f}" if cos < 1.0 else "1.000000"

    print(f"{name:<22} {res['time']:>6.1f}s {res['nfe_min']:>7.1f} {speedup:>7.2f}x {cos_str:>10} {psnr_str:>8}")

# Warmup times
print()
print(f"{'Variant':<22} {'Warmup':>8} (includes torch.compile)")
print(f"{'-' * 22} {'-' * 8}")
for name, res in results.items():
    print(f"{name:<22} {res['warmup']:>7.1f}s")

# Create side-by-side comparison strip
print(f"\nCreating comparison strip...")
images = []
for name, res in results.items():
    images.append(res["image"][0])  # (H, W, 3)

strip = np.concatenate(images, axis=1)  # horizontal concat
strip_path = os.path.join(OUT_DIR, "comparison_strip.png")
Image.fromarray(strip).save(strip_path)
print(f"Saved comparison strip: {strip_path}")

print("\nDone!")

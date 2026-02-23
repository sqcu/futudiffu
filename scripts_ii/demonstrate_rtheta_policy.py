r"""Demonstrate r_theta LoRA adapter effect on diffusion field and BTRM scores.

For each prompt:
  - Euler-sample at adapter_scales=[0.0] (reference) and [1.0] (r_theta active)
  - Record BTRM scores (both heads) at every step
  - VAE decode final latents to pixelspace
  - Plot d_reward/d_logSNR alongside pixelspace renders

Batch structure: at each Euler step, two entries (scale=0 and scale=1) are packed
into a single forward call. Same sigma, same conditioning, same noisy latent.
The ONLY difference is the LoRA adapter contribution.

Output: validation_renders/rtheta_policy_demo/
  {prompt_slug}_ref.png, {prompt_slug}_rtheta.png
  {prompt_slug}_composite.png  -- side-by-side images + d_reward/d_logSNR chart
  scores_per_step.jsonl
  manifest.json

Usage:
    .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\demonstrate_rtheta_policy.py
"""

from __future__ import annotations

import gc
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FP8_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
TE_PATH  = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"
VAE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\vae\zimage.safetensors"
BTRM_DIR = REPO_ROOT / "training_output" / "reward_function_run_tnt_v2"

N_STEPS = 30
SEED    = 42

DEVICE = torch.device("cuda")
DTYPE  = torch.bfloat16

OUTPUT_DIR = REPO_ROOT / "validation_renders" / "rtheta_policy_demo"

# Resolution: 1280x832 is the native reference. Use it for clean comparison.
RES_W, RES_H = 1280, 832

# Prompts to demonstrate. Diverse enough to see different adapter effects.
PROMPTS = [
    ("shrimp_field", 'qwen-3-4b, draw me "pink shrimp with crisp typography in a banana field".'),
    ("laser_shark",  "enormous laser sharks, photorealistic, dramatic lighting, ocean scene"),
    ("portrait",     "portrait of an old fisherman, weathered face, golden hour, shallow depth of field"),
]

ADAPTER_NAME = "rtheta"


def _log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Phase 1: Text encoder
# ---------------------------------------------------------------------------

def phase1_encode(prompts, device, dtype) -> dict[str, torch.Tensor]:
    """Encode all prompts, return CPU tensors. Free TE after."""
    _log("\n" + "=" * 60)
    _log("  PHASE 1: TEXT ENCODER")
    _log("=" * 60)

    from futudiffu.text_encoder import create_tokenizer, load_text_encoder, encode_prompt

    t0 = time.perf_counter()
    tokenizer = create_tokenizer()
    te = load_text_encoder(TE_PATH, device=device, dtype=dtype)
    _log(f"  VRAM after TE load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    conds = {}
    for slug, prompt in prompts:
        cond = encode_prompt(te, tokenizer, prompt, device=device)
        conds[slug] = cond.cpu()
        _log(f"  '{slug}': shape {tuple(cond.shape)}, cap_len={cond.shape[1]}")

    del te, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    _log(f"  TE freed. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    _log(f"  Phase 1: {time.perf_counter() - t0:.1f}s")
    return conds


# ---------------------------------------------------------------------------
# Phase 2: Load model + r_theta adapter
# ---------------------------------------------------------------------------

def phase2_load_model(device, dtype):
    """Load ZImageRLAIF, install LoRA, load r_theta weights + BTRM head, compile."""
    _log("\n" + "=" * 60)
    _log("  PHASE 2: LOAD MODEL + R_THETA ADAPTER")
    _log("=" * 60)

    from src_ii.zimage_model import load_zimage_rlaif
    from src_ii.attention_srcii import patch_sage_for_compile
    from src_ii.multi_lora import install_multi_lora, MultiLoRALinear
    from safetensors.torch import load_file

    t0 = time.perf_counter()

    # Load model WITHOUT compile (need to install LoRA first)
    _, raw_model = load_zimage_rlaif(
        FP8_PATH, device=device, dtype=dtype,
        compile_model=False, fuse=True,
    )
    _log(f"  Model loaded + fused: {time.perf_counter() - t0:.1f}s")
    _log(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Read adapter config
    config_path = BTRM_DIR / "btrm_compound_config.json"
    with open(config_path) as f:
        config = json.load(f)
    rank = config["adapter_rank"]
    alpha = config["adapter_alpha"]
    head_names = config["head_names"]
    _log(f"  Adapter config: rank={rank}, alpha={alpha}, heads={head_names}")

    # Install LoRA wrappers (before compile)
    adapter_configs = [{"name": ADAPTER_NAME, "rank": rank, "alpha": alpha}]
    wrappers = install_multi_lora(raw_model, adapter_configs)
    _log(f"  Installed {len(wrappers)} MultiLoRALinear wrappers")

    # Load adapter weights with key format remapping.
    # Saved format: {path}.adapters.{name}.lora_{AB}
    # Expected format: {path}.lora_{AB}.{name}
    adapter_sd = load_file(str(BTRM_DIR / "rtheta_adapter.safetensors"))

    # Remap keys: "X.adapters.rtheta.lora_A" -> "X.lora_A.rtheta"
    remapped = {}
    for key, tensor in adapter_sd.items():
        # Pattern: ...adapters.{adapter_name}.lora_{A|B}
        parts = key.split(".")
        if "adapters" in parts:
            idx = parts.index("adapters")
            # parts[idx] = "adapters", parts[idx+1] = adapter_name, parts[idx+2] = "lora_X"
            adapter_n = parts[idx + 1]
            lora_ab = parts[idx + 2]
            new_key = ".".join(parts[:idx]) + f".{lora_ab}.{adapter_n}"
            remapped[new_key] = tensor
        else:
            remapped[key] = tensor

    loaded = 0
    for name, module in raw_model.named_modules():
        if isinstance(module, MultiLoRALinear):
            a_key = f"{name}.lora_A.{ADAPTER_NAME}"
            b_key = f"{name}.lora_B.{ADAPTER_NAME}"
            if a_key in remapped and ADAPTER_NAME in module.lora_A:
                module.lora_A[ADAPTER_NAME].data.copy_(remapped[a_key])
                loaded += 1
            if b_key in remapped and ADAPTER_NAME in module.lora_B:
                module.lora_B[ADAPTER_NAME].data.copy_(remapped[b_key])
                loaded += 1
    _log(f"  Loaded {loaded} adapter tensors (remapped key format)")

    # Load BTRM head weights with key remapping.
    # Saved: norm.weight, proj.weight -> score_norm.weight, score_proj.weight
    head_sd = load_file(str(BTRM_DIR / "btrm_head.safetensors"))
    head_remap = {"norm.weight": "score_norm", "proj.weight": "score_proj"}
    for old_key, tensor in head_sd.items():
        if old_key == "norm.weight":
            raw_model.score_norm.weight.data.copy_(tensor.to(raw_model.score_norm.weight.device))
            _log(f"  Loaded score_norm.weight {tuple(tensor.shape)}")
        elif old_key == "proj.weight":
            raw_model.score_proj.weight.data.copy_(tensor.to(raw_model.score_proj.weight.device))
            _log(f"  Loaded score_proj.weight {tuple(tensor.shape)}")
        else:
            _log(f"  WARNING: unknown head key {old_key}")

    # Patch sage backward for compile
    patch_sage_for_compile()

    # Compile
    raw_model.eval()
    compiled = torch.compile(raw_model, mode="default")
    _log(f"  torch.compile done")
    _log(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    _log(f"  Phase 2: {time.perf_counter() - t0:.1f}s")

    return compiled, raw_model, head_names


# ---------------------------------------------------------------------------
# Phase 3: Dual-scale Euler sampling
# ---------------------------------------------------------------------------

def phase3_dual_sampling(model, conds, head_names, device, dtype):
    """For each prompt, Euler-sample at scale=0 and scale=1, record scores.

    At 1280x832 each image is ~4192 tokens, which fills REFERENCE_TOTAL_LEN=4224.
    So two images don't fit in one packed forward. We run serial: two separate
    forward calls per step (one per scale), each containing a single image.
    The packing plan is reused across all 30 steps (constant conditioning/resolution).
    """
    _log("\n" + "=" * 60)
    _log("  PHASE 3: DUAL-SCALE EULER SAMPLING")
    _log("=" * 60)

    from src_ii.forward_packed import prepare_packed_forward, packed_forward
    from src_ii.sigma_schedule import build_sigma_schedule, resolution_shift

    alpha = resolution_shift(RES_W, RES_H)
    sigmas = build_sigma_schedule(N_STEPS, sampling_shift=alpha, device=device, dtype=dtype)
    _log(f"  Resolution {RES_W}x{RES_H}, alpha={alpha:.4f}, {N_STEPS} steps")

    lh, lw = RES_H // 8, RES_W // 8

    all_results = {}

    for slug, cond_cpu in conds.items():
        _log(f"\n  --- {slug} ---")
        t0 = time.perf_counter()

        cond = cond_cpu.to(device)
        cap_len = cond.shape[1]

        # One packing plan for single-image forwards (reused for both scales)
        plan = prepare_packed_forward(
            model, [cond], [(lh, lw)], [cap_len], device,
        )

        # Initialize identical noise for both trajectories
        gen = torch.Generator(device=device).manual_seed(SEED)
        x_ref    = sigmas[0] * torch.randn(1, 16, lh, lw, dtype=dtype, device=device, generator=gen)
        x_rtheta = x_ref.clone()

        # adapter_scales: (1, 1) tensors for each scale
        scales_ref    = torch.tensor([[0.0]], device=device)  # no adapter
        scales_rtheta = torch.tensor([[1.0]], device=device)  # full r_theta

        step_scores = []

        with torch.no_grad():
            for step_i in range(N_STEPS):
                sigma_i = sigmas[step_i]
                sigma_next = sigmas[step_i + 1]
                ts = sigma_i.reshape(1)

                # Forward 1: reference (scale=0, adapter has no effect)
                fields_ref, scores_ref_t = packed_forward(
                    model, [x_ref], [ts],
                    plan["refined_caps"], plan["packing_info"],
                    plan["block_mask"], plan["packed_rope"],
                    adapter_scales=scales_ref,
                )

                # Forward 2: r_theta (scale=1, adapter active)
                fields_rtheta, scores_rtheta_t = packed_forward(
                    model, [x_rtheta], [ts],
                    plan["refined_caps"], plan["packing_info"],
                    plan["block_mask"], plan["packed_rope"],
                    adapter_scales=scales_rtheta,
                )

                field_ref = fields_ref[0]
                field_rtheta = fields_rtheta[0]
                scores_ref = scores_ref_t[0].cpu().tolist()
                scores_rtheta = scores_rtheta_t[0].cpu().tolist()

                # Compute logSNR for this step
                s = max(0.001, min(0.999, float(sigma_i)))
                logsnr = 2.0 * math.log((1.0 - s) / s)

                step_scores.append({
                    "step": step_i,
                    "sigma": float(sigma_i),
                    "logsnr": logsnr,
                    "scores_ref": scores_ref,
                    "scores_rtheta": scores_rtheta,
                })

                # Euler step: denoised = x - field * sigma
                denoised_ref = x_ref - field_ref * sigma_i
                denoised_rtheta = x_rtheta - field_rtheta * sigma_i

                if sigma_next > 0:
                    d_ref = (x_ref - denoised_ref) / sigma_i
                    x_ref = x_ref + d_ref * (sigma_next - sigma_i)

                    d_rtheta = (x_rtheta - denoised_rtheta) / sigma_i
                    x_rtheta = x_rtheta + d_rtheta * (sigma_next - sigma_i)
                else:
                    # Final step: jump to denoised
                    x_ref = denoised_ref
                    x_rtheta = denoised_rtheta

                if (step_i + 1) % 10 == 0:
                    _log(f"    step {step_i+1}/{N_STEPS} — "
                         f"ref_scores={[f'{v:.4f}' for v in scores_ref]}, "
                         f"rtheta_scores={[f'{v:.4f}' for v in scores_rtheta]}")

        elapsed = time.perf_counter() - t0
        _log(f"  {slug}: {elapsed:.1f}s ({elapsed/N_STEPS:.2f}s/step)")

        all_results[slug] = {
            "latent_ref": x_ref.cpu(),
            "latent_rtheta": x_rtheta.cpu(),
            "step_scores": step_scores,
        }

    return all_results, head_names


# ---------------------------------------------------------------------------
# Phase 4: VAE decode
# ---------------------------------------------------------------------------

def phase4_decode(all_results, device, dtype):
    """VAE decode all latents to PIL images."""
    _log("\n" + "=" * 60)
    _log("  PHASE 4: VAE DECODE")
    _log("=" * 60)

    from src_ii.vae_utils import load_vae, decode_latent_to_pil

    t0 = time.perf_counter()
    vae = load_vae(VAE_PATH, device=device, dtype=dtype)
    _log(f"  VAE loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    images = {}
    for slug, result in all_results.items():
        img_ref = decode_latent_to_pil(vae, result["latent_ref"], device=device, dtype=dtype)
        img_rtheta = decode_latent_to_pil(vae, result["latent_rtheta"], device=device, dtype=dtype)
        images[slug] = {"ref": img_ref, "rtheta": img_rtheta}
        _log(f"  {slug}: decoded ref {img_ref.size} + rtheta {img_rtheta.size}")

    del vae
    gc.collect()
    torch.cuda.empty_cache()
    _log(f"  VAE freed. Phase 4: {time.perf_counter() - t0:.1f}s")
    return images


# ---------------------------------------------------------------------------
# Phase 5: Composite renders
# ---------------------------------------------------------------------------

def _draw_chart(
    logsnrs: list[float],
    scores_ref: list[float],
    scores_rtheta: list[float],
    head_name: str,
    chart_w: int = 800,
    chart_h: int = 400,
) -> "Image.Image":
    """Draw a score vs logSNR chart using pure PIL (no matplotlib)."""
    from PIL import Image, ImageDraw

    margin_l, margin_r, margin_t, margin_b = 70, 20, 40, 50
    plot_w = chart_w - margin_l - margin_r
    plot_h = chart_h - margin_t - margin_b

    img = Image.new("RGB", (chart_w, chart_h), "white")
    draw = ImageDraw.Draw(img)

    # Data bounds
    all_scores = scores_ref + scores_rtheta
    x_min, x_max = min(logsnrs), max(logsnrs)
    y_min, y_max = min(all_scores), max(all_scores)
    y_pad = max(0.05, (y_max - y_min) * 0.15)
    y_min -= y_pad
    y_max += y_pad
    x_pad = max(0.1, (x_max - x_min) * 0.05)
    x_min -= x_pad
    x_max += x_pad

    def to_px(xv, yv):
        px = margin_l + int((xv - x_min) / (x_max - x_min) * plot_w)
        py = margin_t + int((1 - (yv - y_min) / (y_max - y_min)) * plot_h)
        return px, py

    # Grid lines
    for i in range(5):
        frac = i / 4
        yv = y_min + frac * (y_max - y_min)
        _, py = to_px(x_min, yv)
        draw.line([(margin_l, py), (margin_l + plot_w, py)], fill=(220, 220, 220))
        draw.text((5, py - 6), f"{yv:.3f}", fill="black")

    # Axes
    draw.rectangle([margin_l, margin_t, margin_l + plot_w, margin_t + plot_h],
                    outline="black")

    # Plot lines
    def draw_line(values, color):
        points = [to_px(logsnrs[i], values[i]) for i in range(len(logsnrs))]
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill=color, width=2)
        for p in points:
            draw.ellipse([p[0]-3, p[1]-3, p[0]+3, p[1]+3], fill=color)

    draw_line(scores_ref, (50, 50, 200))      # blue
    draw_line(scores_rtheta, (200, 50, 50))    # red

    # d_reward/d_logSNR as bars (background)
    if len(logsnrs) > 1:
        d_logsnr = [logsnrs[i+1] - logsnrs[i] for i in range(len(logsnrs)-1)]
        derivs_ref = []
        derivs_rtheta = []
        for i in range(len(d_logsnr)):
            dl = d_logsnr[i]
            if abs(dl) > 1e-6:
                derivs_ref.append((scores_ref[i+1] - scores_ref[i]) / dl)
                derivs_rtheta.append((scores_rtheta[i+1] - scores_rtheta[i]) / dl)
            else:
                derivs_ref.append(0.0)
                derivs_rtheta.append(0.0)

        if derivs_ref:
            d_min = min(min(derivs_ref), min(derivs_rtheta))
            d_max = max(max(derivs_ref), max(derivs_rtheta))
            d_range = max(abs(d_min), abs(d_max), 0.01)
            zero_y = margin_t + plot_h // 2  # zero line for derivatives

            for i in range(len(derivs_ref)):
                mid_x = (logsnrs[i] + logsnrs[i+1]) / 2
                px, _ = to_px(mid_x, 0)
                bar_w = max(2, plot_w // (len(derivs_ref) * 3))

                # ref bar (blue, left of center)
                bar_h_ref = int(abs(derivs_ref[i]) / d_range * (plot_h // 4))
                if derivs_ref[i] >= 0:
                    draw.rectangle([px - bar_w - 1, zero_y - bar_h_ref, px - 1, zero_y],
                                    fill=(50, 50, 200, 60), outline=None)
                else:
                    draw.rectangle([px - bar_w - 1, zero_y, px - 1, zero_y + bar_h_ref],
                                    fill=(50, 50, 200, 60), outline=None)

                # rtheta bar (red, right of center)
                bar_h_rth = int(abs(derivs_rtheta[i]) / d_range * (plot_h // 4))
                if derivs_rtheta[i] >= 0:
                    draw.rectangle([px + 1, zero_y - bar_h_rth, px + bar_w + 1, zero_y],
                                    fill=(200, 50, 50, 60), outline=None)
                else:
                    draw.rectangle([px + 1, zero_y, px + bar_w + 1, zero_y + bar_h_rth],
                                    fill=(200, 50, 50, 60), outline=None)

    # Labels
    draw.text((chart_w // 2 - 80, 5), f"{head_name}: score vs logSNR", fill="black")
    draw.text((chart_w // 2 - 60, chart_h - 18), "logSNR (noisy -> clean ->)", fill="gray")

    # Legend
    draw.rectangle([margin_l + 10, margin_t + 5, margin_l + 25, margin_t + 15],
                    fill=(50, 50, 200))
    draw.text((margin_l + 30, margin_t + 3), "ref (scale=0)", fill=(50, 50, 200))
    draw.rectangle([margin_l + 10, margin_t + 20, margin_l + 25, margin_t + 30],
                    fill=(200, 50, 50))
    draw.text((margin_l + 30, margin_t + 18), "r_theta (scale=1)", fill=(200, 50, 50))

    # Final score annotations
    final_ref = scores_ref[-1]
    final_rtheta = scores_rtheta[-1]
    draw.text((chart_w - margin_r - 180, margin_t + 5),
              f"final ref={final_ref:.4f}", fill=(50, 50, 200))
    draw.text((chart_w - margin_r - 180, margin_t + 20),
              f"final r_theta={final_rtheta:.4f}", fill=(200, 50, 50))
    delta = final_rtheta - final_ref
    draw.text((chart_w - margin_r - 180, margin_t + 35),
              f"delta={delta:+.4f}", fill="black")

    return img


def phase5_render(all_results, images, head_names, output_dir):
    """Build composite figures: pixelspace renders + d_reward/d_logSNR charts."""
    _log("\n" + "=" * 60)
    _log("  PHASE 5: COMPOSITE RENDERS")
    _log("=" * 60)

    from PIL import Image, ImageDraw

    output_dir.mkdir(parents=True, exist_ok=True)
    scores_log = []

    for slug, result in all_results.items():
        step_scores = result["step_scores"]
        img_ref = images[slug]["ref"]
        img_rtheta = images[slug]["rtheta"]

        # Save individual images
        img_ref.save(output_dir / f"{slug}_ref.png")
        img_rtheta.save(output_dir / f"{slug}_rtheta.png")

        # Extract score trajectories
        logsnrs = [s["logsnr"] for s in step_scores]
        n_heads = len(step_scores[0]["scores_ref"])

        # Build charts for each head
        charts = []
        for head_idx in range(n_heads):
            head_name = head_names[head_idx] if head_idx < len(head_names) else f"head_{head_idx}"
            scores_ref_h = [s["scores_ref"][head_idx] for s in step_scores]
            scores_rtheta_h = [s["scores_rtheta"][head_idx] for s in step_scores]
            chart = _draw_chart(logsnrs, scores_ref_h, scores_rtheta_h, head_name)
            charts.append(chart)

        # Composite: [ref_image | rtheta_image | charts stacked vertically]
        # Scale images to fit alongside charts
        img_h = img_ref.size[1]
        img_w = img_ref.size[0]
        chart_total_h = sum(c.size[1] for c in charts)
        # Scale images to match chart column height
        target_h = max(chart_total_h, img_h)
        scale = target_h / img_h
        scaled_w = int(img_w * scale)
        scaled_h = int(img_h * scale)

        img_ref_scaled = img_ref.resize((scaled_w, scaled_h), Image.LANCZOS)
        img_rtheta_scaled = img_rtheta.resize((scaled_w, scaled_h), Image.LANCZOS)

        chart_w = charts[0].size[0]
        # Stack charts vertically, scale to match image height
        chart_col = Image.new("RGB", (chart_w, chart_total_h), "white")
        y_off = 0
        for c in charts:
            chart_col.paste(c, (0, y_off))
            y_off += c.size[1]
        chart_col_scaled = chart_col.resize(
            (chart_w, scaled_h), Image.LANCZOS,
        )

        # Build composite
        total_w = scaled_w * 2 + chart_col_scaled.size[0] + 10  # 10px gaps
        composite = Image.new("RGB", (total_w, scaled_h + 30), "white")

        # Title
        draw = ImageDraw.Draw(composite)
        draw.text((10, 5), f"{slug} — r_theta adapter policy demonstration", fill="black")
        draw.text((scaled_w // 2 - 40, 15), "Reference (scale=0)", fill=(50, 50, 200))
        draw.text((scaled_w + scaled_w // 2 - 40, 15), "r_theta (scale=1)", fill=(200, 50, 50))

        composite.paste(img_ref_scaled, (0, 30))
        composite.paste(img_rtheta_scaled, (scaled_w + 5, 30))
        composite.paste(chart_col_scaled, (scaled_w * 2 + 10, 30))

        composite_path = output_dir / f"{slug}_composite.png"
        composite.save(str(composite_path))
        _log(f"  {slug}: composite {composite.size[0]}x{composite.size[1]} -> {composite_path.name}")

        # Append to scores log
        for entry in step_scores:
            scores_log.append({"slug": slug, **entry})

    # Write scores JSONL
    scores_path = output_dir / "scores_per_step.jsonl"
    with open(scores_path, "w") as f:
        for entry in scores_log:
            f.write(json.dumps(entry) + "\n")
    _log(f"  Scores log: {scores_path.name} ({len(scores_log)} entries)")

    return scores_log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    _log(f"r_theta LoRA adapter policy demonstration — "
         f"{datetime.now(timezone.utc).isoformat()}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    t_total = time.perf_counter()

    # Phase 1: Text encoder
    conds = phase1_encode(PROMPTS, DEVICE, DTYPE)

    # Phase 2: Load model + r_theta
    model, raw_model, head_names = phase2_load_model(DEVICE, DTYPE)

    # Phase 3: Dual-scale sampling
    all_results, head_names = phase3_dual_sampling(model, conds, head_names, DEVICE, DTYPE)

    # Free backbone before VAE
    del model, raw_model
    gc.collect()
    torch.cuda.empty_cache()
    _log(f"  Backbone freed. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Phase 4: VAE decode
    images = phase4_decode(all_results, DEVICE, DTYPE)

    # Phase 5: Composite renders
    scores_log = phase5_render(all_results, images, head_names, OUTPUT_DIR)

    # Write manifest
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompts": {slug: prompt for slug, prompt in PROMPTS},
        "resolution": f"{RES_W}x{RES_H}",
        "n_steps": N_STEPS,
        "seed": SEED,
        "adapter_name": ADAPTER_NAME,
        "btrm_dir": str(BTRM_DIR),
        "head_names": head_names,
        "n_prompts": len(PROMPTS),
        "output_files": [
            f"{slug}_ref.png" for slug, _ in PROMPTS
        ] + [
            f"{slug}_rtheta.png" for slug, _ in PROMPTS
        ] + [
            f"{slug}_composite.png" for slug, _ in PROMPTS
        ] + ["scores_per_step.jsonl"],
        "total_elapsed_s": time.perf_counter() - t_total,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    _log(f"\n{'=' * 60}")
    _log(f"  DONE — {time.perf_counter() - t_total:.1f}s total")
    _log(f"  Output: {OUTPUT_DIR}")
    _log(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

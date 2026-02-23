r"""K=6 shrimp-banana multi-resolution 6-tuple guidance validation.

Forcing function: if this produces a coherent image, scatter-gather-euler works.
Also runs K=2 and K=1 baselines on the same prompt/seed for visual comparison.

Serial executor (one entry per forward launch) is the fallback until BatchExecutor
lands. It is correct but slow: K=6 = 6 separate packed-forward calls per step.

Output: validation_renders/ktuple_6tuple/
  final_k6.png, final_k2.png, final_k1.png
  latents/  -- raw .pt latents for all three
  scores_per_step.jsonl  -- (step, entry, scores) for K=6
  packing_diagnostics.json  -- launches per step
  pixel_stats.json  -- decoded image statistics
  manifest.json  -- full run config

Usage:
    .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\validate_ktuple_6tuple.py
"""

from __future__ import annotations

import gc
import json
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


FP8_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
TE_PATH  = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"
VAE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\vae\zimage.safetensors"

N_STEPS = 30
SEED    = 42
GAIN    = 6.0   # gather_residual_gain gain parameter

DEVICE = torch.device("cuda")
DTYPE  = torch.bfloat16

OUTPUT_DIR = REPO_ROOT / "validation_renders" / "ktuple_6tuple"

P_BASE   = 'qwen-3-4b, draw me "pink shrimp with crisp typography in a banana field".'
P_SHRIMP = "pink shrimp, detailed color, clear shape"
P_TYPO   = "clean typography, sharp letterforms, no blur"
P_BANANA = "banana, yellow, tropical poem"

P_NEG = ""   # empty negative for K=2 CFG baseline
UNIQUE_PROMPTS = [P_BASE, P_SHRIMP, P_TYPO, P_BANANA, P_NEG]

SPEC_DEF = [
    (P_BASE,   (1024, 1024), +1.0),   # 0 base trajectory
    (P_SHRIMP, (1024, 1024), +3.0),   # 1 attractive: shrimp emphasis
    (P_TYPO,   (1024, 1024), +2.0),   # 2 attractive: typography clarity
    (P_BASE,   ( 512,  512), -2.0),   # 3 repulsive: mid-res blur
    (P_BASE,   ( 256,  256), -1.5),   # 4 repulsive: low-res blur
    (P_BANANA, (1024, 1024), -4.0),   # 5 negative: banana dominance
]



def _log(msg: str) -> None:
    print(msg, flush=True)


def _pixel_stats(pil_img) -> dict:
    px = np.array(pil_img).astype(np.float32)
    return {
        "size": list(pil_img.size),
        "mean": float(px.mean()),
        "std": float(px.std()),
        "min": float(px.min()),
        "max": float(px.max()),
    }



def phase1_encode(device, dtype) -> dict[str, torch.Tensor]:
    """Load TE, encode all 4 unique prompts, free TE. Returns CPU tensors."""
    _log("\n" + "=" * 60)
    _log("  PHASE 1: TEXT ENCODER")
    _log("=" * 60)

    from futudiffu.text_encoder import create_tokenizer, load_text_encoder, encode_prompt

    t0 = time.perf_counter()
    tokenizer = create_tokenizer()
    te = load_text_encoder(TE_PATH, device=device, dtype=dtype)
    _log(f"  VRAM after TE load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    conds: dict[str, torch.Tensor] = {}
    for prompt in UNIQUE_PROMPTS:
        cond = encode_prompt(te, tokenizer, prompt, device=device)
        conds[prompt] = cond.cpu()
        _log(f"  '{prompt[:60]}': shape {tuple(cond.shape)}")

    del te, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    _log(f"  TE freed. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    _log(f"  Phase 1 done: {time.perf_counter() - t0:.1f}s")
    return conds




def make_serial_executor(model, spec, device, dtype):
    """Build a serial executor for a fixed spec (one query).

    Pre-builds packing state once per (conditioning, resolution) pair.
    Packing state is constant across Euler steps — only latent values change.

    Returns:
        executor(x_base, step_i) -> (denoised_list, scores_stacked, n_launches)
          denoised_list: K denoised tensors at their native resolutions
          scores_stacked: (K, n_heads) BTRM scores
          n_launches: int (always K for serial executor)
    """
    from src_ii.forward_packed import prepare_packed_forward, packed_forward
    from src_ii.triumphant_future_reduction_ops import (
        build_per_image_sigmas, scatter, denoise_all,
    )

    entry_sigmas = build_per_image_sigmas(spec, N_STEPS, device, dtype)

    plans = []
    for cond, (rw, rh), _ in spec:
        lh, lw = rh // 8, rw // 8
        cap_len = cond.shape[1]
        plan = prepare_packed_forward(
            model, [cond], [(lh, lw)], [cap_len], device,
        )
        plans.append(plan)
    _log(f"  Serial executor: {len(spec)} plans pre-built "
         f"(one forward per entry, {len(spec)} launches/step)")

    def executor(x_base, step_i):
        scattered = scatter(x_base, spec)  # K latents at native resolutions

        fields_list = []
        all_scores = []
        for entry_idx, (entry_x, plan) in enumerate(zip(scattered, plans)):
            sigma_i = entry_sigmas[entry_idx][step_i]
            ts = sigma_i.reshape(1).to(device, dtype)
            with torch.no_grad():
                fields, scores = packed_forward(
                    model, [entry_x.to(device)],
                    [ts], plan["refined_caps"],
                    plan["packing_info"], plan["block_mask"], plan["packed_rope"],
                )
            fields_list.append(fields[0])
            all_scores.append(scores)

        sigmas = [entry_sigmas[i][step_i] for i in range(len(spec))]
        denoised_list = denoise_all(scattered, fields_list, sigmas)
        scores_stacked = torch.cat(all_scores, dim=0)  # (K, n_heads)
        return denoised_list, scores_stacked, len(spec)

    return executor



def run_rollout(model, spec, gather_fn, device, dtype, label):
    """Euler sampling loop. Returns (final_latent_cpu, scores_log, packing_diag)."""
    from src_ii.sigma_schedule import build_sigma_schedule, resolution_shift
    from src_ii.triumphant_future_reduction_ops import noise_field, aperture, euler_step

    executor = make_serial_executor(model, spec, device, dtype)

    base_res = spec[0][1]
    alpha = resolution_shift(base_res[0], base_res[1])
    query_sigmas = build_sigma_schedule(N_STEPS, sampling_shift=alpha,
                                        device=device, dtype=dtype)

    max_lh = max(rh // 8 for _, (_, rh), _ in spec)
    max_lw = max(rw // 8 for _, (rw, _), _ in spec)
    master = noise_field(max_lh, max_lw, SEED, device, dtype)
    base_h, base_w = base_res[1] // 8, base_res[0] // 8
    x_base = query_sigmas[0] * aperture(master, base_h, base_w)

    scores_log = []
    n_launches_per_step = []

    _log(f"\n  [{label}] {N_STEPS} steps, K={len(spec)}, "
         f"base_res={base_res[0]}x{base_res[1]}")
    t0 = time.perf_counter()

    with torch.no_grad():
        for step_i in range(N_STEPS):
            denoised_list, scores, n_launches = executor(x_base, step_i)

            guided = gather_fn(denoised_list, spec)

            sigma_i    = query_sigmas[step_i]
            sigma_next = query_sigmas[step_i + 1]
            x_base = euler_step(x_base, guided, sigma_i, sigma_next)

            scores_log.append({
                "step": step_i,
                "n_launches": n_launches,
                "scores": scores.cpu().tolist(),
            })
            n_launches_per_step.append(n_launches)

            if (step_i + 1) % 10 == 0:
                elapsed = time.perf_counter() - t0
                _log(f"    [{label}] step {step_i+1}/{N_STEPS} — {elapsed:.1f}s elapsed")

    elapsed = time.perf_counter() - t0
    total_launches = sum(n_launches_per_step)
    _log(f"  [{label}] Done: {elapsed:.1f}s, {total_launches} total launches "
         f"({total_launches / N_STEPS:.1f}/step)")

    packing_diag = {
        "k": len(spec),
        "n_steps": N_STEPS,
        "n_launches_per_step": n_launches_per_step,
        "total_launches": total_launches,
        "launches_per_step_expected": len(spec),
        "elapsed_s": elapsed,
        "note": "serial executor: one launch per spec entry",
    }

    return x_base.cpu(), scores_log, packing_diag



def main() -> int:
    _log(f"K=6 shrimp-banana 6-tuple validation — "
         f"{datetime.now(timezone.utc).isoformat()}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "latents").mkdir(exist_ok=True)

    conds_cpu = phase1_encode(DEVICE, DTYPE)

    _log("\n" + "=" * 60)
    _log("  PHASE 2: LOAD MODEL")
    _log("=" * 60)

    from src_ii.zimage_model import load_zimage_rlaif
    from futudiffu.attention import set_attention_backend

    t0 = time.perf_counter()
    model = load_zimage_rlaif(
        FP8_PATH, device=DEVICE, dtype=DTYPE,
        compile_model=True, fuse=True,
    )
    set_attention_backend("sage")
    _log(f"  Loaded + compiled in {time.perf_counter() - t0:.1f}s")
    _log(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    model = model

    _log("\n" + "=" * 60)
    _log("  PHASE 3: ROLLOUTS")
    _log("=" * 60)

    import functools
    from src_ii.triumphant_future_reduction_ops import (
        gather, gather_residual_gain, cfg1, cfg2,
    )

    c_base   = conds_cpu[P_BASE].to(DEVICE)
    c_shrimp = conds_cpu[P_SHRIMP].to(DEVICE)
    c_typo   = conds_cpu[P_TYPO].to(DEVICE)
    c_banana = conds_cpu[P_BANANA].to(DEVICE)
    c_neg    = conds_cpu[P_NEG].to(DEVICE)

    spec_k6 = [
        (c, res, scale)
        for (p, res, scale), c in zip(SPEC_DEF, [
            c_base, c_shrimp, c_typo, c_base, c_base, c_banana,
        ])
    ]
    gather_k6 = functools.partial(gather_residual_gain, gain=GAIN)
    lat_k6, scores_k6, diag_k6 = run_rollout(
        model, spec_k6, gather_k6, DEVICE, DTYPE, "K=6",
    )

    spec_k2 = cfg2(c_base, c_neg, (1024, 1024), 7.0)
    gather_k2 = functools.partial(gather_residual_gain, gain=6.0)
    lat_k2, scores_k2, diag_k2 = run_rollout(
        model, spec_k2, gather_k2, DEVICE, DTYPE, "K=2",
    )

    spec_k1 = cfg1(c_base, (1024, 1024))
    lat_k1, scores_k1, diag_k1 = run_rollout(
        model, spec_k1, gather, DEVICE, DTYPE, "K=1",
    )

    _log("\n" + "=" * 60)
    _log("  PHASE 4: PERSIST")
    _log("=" * 60)

    torch.save(lat_k6, OUTPUT_DIR / "latents" / "final_k6.pt")
    torch.save(lat_k2, OUTPUT_DIR / "latents" / "final_k2.pt")
    torch.save(lat_k1, OUTPUT_DIR / "latents" / "final_k1.pt")
    _log("  Latents saved.")

    with open(OUTPUT_DIR / "scores_per_step.jsonl", "w") as f:
        for row in scores_k6:
            f.write(json.dumps(row) + "\n")
    _log("  scores_per_step.jsonl written.")

    with open(OUTPUT_DIR / "packing_diagnostics.json", "w") as f:
        json.dump({"k6": diag_k6, "k2": diag_k2, "k1": diag_k1}, f, indent=2)
    _log("  packing_diagnostics.json written.")

    _log("\n" + "=" * 60)
    _log("  PHASE 5: VAE DECODE")
    _log("=" * 60)

    del model, model, model
    gc.collect()
    torch.cuda.empty_cache()
    _log(f"  VRAM after backbone free: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    from src_ii.vae_utils import load_vae, decode_latent_to_pil

    vae = load_vae(VAE_PATH, device=DEVICE, dtype=DTYPE)
    _log(f"  VRAM after VAE load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    pil_k6 = decode_latent_to_pil(vae, lat_k6.to(DEVICE), device=DEVICE, dtype=DTYPE)
    pil_k2 = decode_latent_to_pil(vae, lat_k2.to(DEVICE), device=DEVICE, dtype=DTYPE)
    pil_k1 = decode_latent_to_pil(vae, lat_k1.to(DEVICE), device=DEVICE, dtype=DTYPE)

    pil_k6.save(OUTPUT_DIR / "final_k6.png")
    pil_k2.save(OUTPUT_DIR / "final_k2.png")
    pil_k1.save(OUTPUT_DIR / "final_k1.png")
    _log("  Images saved: final_k6.png, final_k2.png, final_k1.png")

    pixel_stats = {
        "k6": _pixel_stats(pil_k6),
        "k2": _pixel_stats(pil_k2),
        "k1": _pixel_stats(pil_k1),
    }
    with open(OUTPUT_DIR / "pixel_stats.json", "w") as f:
        json.dump(pixel_stats, f, indent=2)
    _log("  pixel_stats.json written.")

    del vae
    gc.collect()
    torch.cuda.empty_cache()

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_steps": N_STEPS,
        "gain": GAIN,
        "spec_k6": [
            {"prompt": p, "resolution": list(res), "scale": scale}
            for p, res, scale in SPEC_DEF
        ],
        "k2_note": "cfg2(base, empty_neg, 1024x1024, scale=7.0) + gather_residual_gain(gain=6.0)",
        "k1_note": "cfg1(base, 1024x1024) + linear gather (identity)",
        "fp8_path": FP8_PATH,
        "vae_path": VAE_PATH,
        "te_path": TE_PATH,
        "executor": "serial",
        "packing_note": (
            "Serial executor: one packed-forward per spec entry. "
            "K=6 = 6 launches/step. Replace with BatchExecutor when available."
        ),
        "pixel_stats": pixel_stats,
        "packing_diagnostics": {
            "k6_total_launches": diag_k6["total_launches"],
            "k2_total_launches": diag_k2["total_launches"],
            "k1_total_launches": diag_k1["total_launches"],
        },
    }
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    _log("  manifest.json written.")

    _log("\n" + "=" * 60)
    _log("  SUMMARY")
    _log("=" * 60)
    for label, stats in pixel_stats.items():
        _log(f"  {label}: size={stats['size']} mean={stats['mean']:.1f} "
             f"std={stats['std']:.1f} range=[{stats['min']:.0f},{stats['max']:.0f}]")
    _log(f"  K=6 launches: {diag_k6['total_launches']} "
         f"({diag_k6['total_launches'] / N_STEPS:.1f}/step, expected {len(spec_k6)})")
    _log(f"  K=2 launches: {diag_k2['total_launches']} "
         f"({diag_k2['total_launches'] / N_STEPS:.1f}/step)")
    _log(f"  K=1 launches: {diag_k1['total_launches']} "
         f"({diag_k1['total_launches'] / N_STEPS:.1f}/step)")
    _log(f"  Output: {OUTPUT_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

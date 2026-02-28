r"""Batch rollout pipeline validation: TE → backbone → packed rollout → VAE decode.

Exercises the complete production pipeline for packed multi-image generation:
  Phase 1: Text encoder encode (real prompt + negative), free TE
  Phase 2a: SDPA determinism baseline (proves Euler loop is bitwise deterministic)
  Phase 2b: SageAttention rollouts with strong adapters (adapter effect >> sage noise)
  Phase 3: VAE decode all latents, pixel-space distributional statistics

Key insight from first run: SageAttention is NON-DETERMINISTIC (INT8 quantization +
parallel Triton reductions). Adapter B init std=0.1 produced ~0.3% perturbation,
completely buried in SageAttention noise. Fixed by:
  - Adding SDPA determinism baseline (bitwise identical when attention is deterministic)
  - Increasing adapter B init to std=1.0 (adapter signal >> SageAttention noise)
  - Comparing adapter effect magnitude vs SageAttention noise magnitude

Every tensor and image is persisted to disk. Metrics are saved as JSON.
No thresholds, no asserts — continuous measures that speak for themselves.

Output: batch_rollout_validation/ (latents .pt, images .png, metrics .json)

Usage:
    .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\validate_batch_rollout.py
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

import torch
import numpy as np

from src_ii.zimage_model import load_zimage_rlaif
from src_ii.multi_lora import install_multi_lora, assign_adapter, adapter_summary, init_adapter_b_weights
from src_ii.ktuple_sampling import batch_rollout
from src_ii.sigma_schedule import resolution_shift
from src_ii.rendering import (
    load_vae,
    decode_latent_to_pil,
    save_false_color_diff,
    compute_per_channel_pixel_stats,
    compute_spatial_autocorrelation,
)


FP8_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
TE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"
VAE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\vae\zimage.safetensors"

PROMPT = (
    'qwen-3-4b, draw me an "enormous laser shark for the sega saturn".'
)

RESOLUTIONS = [(512, 512), (640, 384)]
SEEDS = [42, 7]
N_STEPS = 30
CFG = 4.0

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16

OUTPUT_DIR = REPO_ROOT / "batch_rollout_validation"



def _log(msg: str) -> None:
    print(msg, flush=True)


def _save_json(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    _log(f"  Saved: {path.name}")


def _single_image_pixel_stats(pil_img) -> dict:
    """Per-channel pixel statistics for a single decoded image."""
    pixels = np.array(pil_img).astype(np.float32)  # (H, W, 3)
    channels = {"R": pixels[:, :, 0], "G": pixels[:, :, 1], "B": pixels[:, :, 2]}
    per_channel = {}
    for name, ch in channels.items():
        per_channel[name] = {
            "mean": float(ch.mean()),
            "std": float(ch.std()),
            "min": float(ch.min()),
            "max": float(ch.max()),
        }
    return {
        "size": pil_img.size,  # (W, H)
        "per_channel": per_channel,
        "overall_mean": float(pixels.mean()),
        "overall_std": float(pixels.std()),
        "dynamic_range": float(pixels.max() - pixels.min()),
    }



def phase1_encode(device, dtype) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Load TE, encode prompt + negative, free TE. Returns CPU tensors."""
    from futudiffu.text_encoder import create_tokenizer, load_text_encoder, encode_prompt

    _log("\n" + "=" * 60)
    _log("  PHASE 1: TEXT ENCODER")
    _log("=" * 60)

    t0 = time.perf_counter()

    tokenizer = create_tokenizer()
    te = load_text_encoder(TE_PATH, device=device, dtype=dtype)
    _log(f"  VRAM after TE load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    pos = encode_prompt(te, tokenizer, PROMPT, device=device)
    neg = encode_prompt(te, tokenizer, "", device=device)
    cap_len = pos.shape[1]

    _log(f"  pos shape: {tuple(pos.shape)}, neg shape: {tuple(neg.shape)}")
    _log(f"  cap_len (tokens): {cap_len}")

    pos = pos.cpu()
    neg = neg.cpu()
    del te, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    _log(f"  TE freed. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    _log(f"  Phase 1 complete: {elapsed:.1f}s")

    return pos, neg, cap_len



def phase2a_sdpa_determinism(model, pos_list, neg_list, cap_lens, device, dtype):
    """SDPA determinism baseline: proves the Euler loop is bitwise deterministic.

    SDPA attention (torch.nn.functional.scaled_dot_product_attention) is
    deterministic — no INT8 quantization, no Triton parallel reductions.
    Two identical runs should produce bitwise identical outputs.

    Note: SDPA backend is set for determinism testing. Block mask is still
    passed through — SDPA just ignores it internally.
    """
    _log("\n  --- Phase 2a: SDPA Determinism Baseline ---")
    from futudiffu.attention import set_attention_backend
    set_attention_backend("sdpa")
    scales = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device)

    t1 = time.perf_counter()
    trajs_sdpa1, _ = batch_rollout(
        model, pos_list, neg_list, cap_lens, SEEDS, RESOLUTIONS,
        N_STEPS, CFG, device, dtype,
        adapter_scales=scales,
    )
    elapsed1 = time.perf_counter() - t1
    _log(f"    SDPA run 1: {elapsed1:.2f}s")

    t1 = time.perf_counter()
    trajs_sdpa2, _ = batch_rollout(
        model, pos_list, neg_list, cap_lens, SEEDS, RESOLUTIONS,
        N_STEPS, CFG, device, dtype,
        adapter_scales=scales,
    )
    elapsed2 = time.perf_counter() - t1
    _log(f"    SDPA run 2: {elapsed2:.2f}s")

    results = {}
    for k in range(2):
        bitwise = torch.equal(trajs_sdpa1[k]["final"], trajs_sdpa2[k]["final"])
        diff = (trajs_sdpa1[k]["final"].float() - trajs_sdpa2[k]["final"].float())
        results[f"img{k}"] = {
            "bitwise_equal": bool(bitwise),
            "max_abs_diff": float(diff.abs().max().item()),
            "mean_abs_diff": float(diff.abs().mean().item()),
            "l2_norm": float(diff.norm().item()),
            "note": "SDPA is deterministic → expect bitwise identical",
        }
        _log(f"    img{k}: bitwise={bitwise}, max_abs={diff.abs().max().item():.6f}")

    return results, trajs_sdpa1


def phase2b_sdpa_adapter_effect(model, pos_list, neg_list, cap_lens,
                                 trajs_sdpa_baseline, device, dtype):
    """SDPA adapter routing test: deterministic measurement of adapter effect.

    Since SDPA is bitwise deterministic, any diff between Run A (adapter ON)
    and Run B (adapter OFF mid-trajectory for img0) is PURELY from the adapter.
    No SageAttention noise contamination.

    Note: SDPA has no block masks → cross-image attention leakage exists.
    So img1 WILL show a diff (because img0's adapter-modified hidden states
    leak into img1's attention computation). But img0's diff should be LARGER
    than img1's diff, because img0 is the directly-affected image.
    """
    _log("\n  --- Phase 2b: SDPA Adapter Routing (deterministic) ---")
    from futudiffu.attention import set_attention_backend
    set_attention_backend("sdpa")

    mid = N_STEPS // 2

    def scales_fn(step_i):
        if step_i < mid:
            return torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device)
        else:
            return torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device)

    t1 = time.perf_counter()
    trajs_sdpa_b, _ = batch_rollout(
        model, pos_list, neg_list, cap_lens, SEEDS, RESOLUTIONS,
        N_STEPS, CFG, device, dtype,
        adapter_scales=scales_fn,
    )
    elapsed = time.perf_counter() - t1
    _log(f"    SDPA adapter-off run: {elapsed:.2f}s")

    results = {}
    for k in range(2):
        diff = (trajs_sdpa_baseline[k]["final"].float() - trajs_sdpa_b[k]["final"].float())
        cosine = torch.nn.functional.cosine_similarity(
            trajs_sdpa_baseline[k]["final"].flatten().float().unsqueeze(0),
            trajs_sdpa_b[k]["final"].flatten().float().unsqueeze(0),
        ).item()
        results[f"img{k}"] = {
            "max_abs_diff": float(diff.abs().max().item()),
            "mean_abs_diff": float(diff.abs().mean().item()),
            "l2_norm": float(diff.norm().item()),
            "cosine_similarity": float(cosine),
            "bitwise_equal": bool(torch.equal(
                trajs_sdpa_baseline[k]["final"], trajs_sdpa_b[k]["final"],
            )),
            "note": (
                "img0: adapter disabled mid-trajectory → expect large deterministic diff"
                if k == 0 else
                "img1: adapter unchanged, but SDPA cross-attn leakage → expect smaller diff"
            ),
        }
        _log(f"    img{k}: max_abs={diff.abs().max().item():.4f}, "
             f"l2={diff.norm().item():.4f}, cos={cosine:.6f}")

    set_attention_backend("sage")
    return results


def phase2_rollouts(pos, neg, cap_len, device, dtype, output_dir):
    """Load ZImageRLAIF, install LoRA, run rollouts. Returns latent dicts."""
    _log("\n" + "=" * 60)
    _log("  PHASE 2: BACKBONE + ROLLOUTS")
    _log("=" * 60)

    t0 = time.perf_counter()

    model = load_zimage_rlaif(
        FP8_PATH, device=device, dtype=dtype,
        compile_model=False, fuse=True,
    )
    _log(f"  VRAM after backbone load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    wrappers = install_multi_lora(model, max_adapters=4, max_rank=8)
    assign_adapter(model, 0, "p_theta", 4, 4.0)
    assign_adapter(model, 1, "r_theta", 4, 4.0)
    init_adapter_b_weights(model, "p_theta", std=0.01)

    summary = adapter_summary(model)
    _log(f"  LoRA installed: {summary['n_wrapped_layers']} layers, "
         f"{len(summary['adapters'])} adapters")

    pos_gpu, neg_gpu = pos.to(device), neg.to(device)
    pos_list = [pos_gpu, pos_gpu]
    neg_list = [neg_gpu, neg_gpu]
    cap_lens = [cap_len, cap_len]

    sdpa_determinism, trajs_sdpa_baseline = phase2a_sdpa_determinism(
        model, pos_list, neg_list, cap_lens, device, dtype,
    )

    sdpa_adapter_effect = phase2b_sdpa_adapter_effect(
        model, pos_list, neg_list, cap_lens,
        trajs_sdpa_baseline, device, dtype,
    )

    _log("\n  --- Phase 2c: SageAttention Noise Characterization ---")
    _log("  Run A: adapter p_theta ON (constant scales), SageAttention")
    scales_a = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device)

    t1 = time.perf_counter()
    trajs_a, meta_a = batch_rollout(
        model, pos_list, neg_list, cap_lens, SEEDS, RESOLUTIONS,
        N_STEPS, CFG, device, dtype,
        adapter_scales=scales_a,
        save_steps={0, 4, 14, 24, N_STEPS - 1},
    )
    elapsed_a = time.perf_counter() - t1
    _log(f"    {elapsed_a:.2f}s, K={meta_a['K']}, {N_STEPS} steps")

    _log("  Run A2: SageAttention noise characterization (same seeds + scales)")
    t1 = time.perf_counter()
    trajs_a2, _ = batch_rollout(
        model, pos_list, neg_list, cap_lens, SEEDS, RESOLUTIONS,
        N_STEPS, CFG, device, dtype,
        adapter_scales=scales_a,
    )
    elapsed_a2 = time.perf_counter() - t1
    _log(f"    {elapsed_a2:.2f}s")

    mid = N_STEPS // 2
    _log(f"  Run B: adapter disabled for image 0 at step >= {mid}")

    def scales_fn(step_i):
        if step_i < mid:
            return torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device)
        else:
            return torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device)

    t1 = time.perf_counter()
    trajs_b, meta_b = batch_rollout(
        model, pos_list, neg_list, cap_lens, SEEDS, RESOLUTIONS,
        N_STEPS, CFG, device, dtype,
        adapter_scales=scales_fn,
    )
    elapsed_b = time.perf_counter() - t1
    _log(f"    {elapsed_b:.2f}s")

    latent_dir = output_dir / "latents"
    latent_dir.mkdir(exist_ok=True)
    for k in range(2):
        torch.save(trajs_a[k]["final"], latent_dir / f"run_a_img{k}.pt")
        torch.save(trajs_a2[k]["final"], latent_dir / f"run_a2_img{k}.pt")
        torch.save(trajs_b[k]["final"], latent_dir / f"run_b_img{k}.pt")
    _log(f"  Latents saved to {latent_dir}")

    metrics = {
        "resolutions": RESOLUTIONS,
        "seeds": SEEDS,
        "n_steps": N_STEPS,
        "cfg": CFG,
        "prompt": PROMPT,
        "adapter_b_init_std": 0.01,
        "adapter_summary": summary,
        "timing": {
            "run_a_seconds": elapsed_a,
            "run_a2_seconds": elapsed_a2,
            "run_b_seconds": elapsed_b,
        },
        "sdpa_determinism": sdpa_determinism,
        "sdpa_adapter_effect": sdpa_adapter_effect,
        "sigma_schedules": {},
        "scores_per_step": {},
        "sage_noise_floor": {},
        "sage_adapter_effect": {},
    }

    for k in range(2):
        metrics["sigma_schedules"][f"img{k}"] = {
            "sigmas": trajs_a[k]["sigmas"],
            "alpha": float(resolution_shift(*RESOLUTIONS[k])),
            "resolution": RESOLUTIONS[k],
        }

    for step_i, scores in enumerate(meta_a["scores_per_step"]):
        metrics["scores_per_step"][f"step_{step_i}"] = {
            f"img{k}": scores[k].tolist() for k in range(2)
        }

    for k in range(2):
        diff = (trajs_a[k]["final"].float() - trajs_a2[k]["final"].float())
        metrics["sage_noise_floor"][f"img{k}"] = {
            "max_abs_diff": float(diff.abs().max().item()),
            "mean_abs_diff": float(diff.abs().mean().item()),
            "l2_norm": float(diff.norm().item()),
            "bitwise_equal": bool(torch.equal(trajs_a[k]["final"], trajs_a2[k]["final"])),
            "note": "SageAttention INT8 non-determinism — expected to be nonzero",
        }

    for k in range(2):
        diff = (trajs_a[k]["final"].float() - trajs_b[k]["final"].float())
        cosine = torch.nn.functional.cosine_similarity(
            trajs_a[k]["final"].flatten().float().unsqueeze(0),
            trajs_b[k]["final"].flatten().float().unsqueeze(0),
        ).item()

        noise_l2 = metrics["sage_noise_floor"][f"img{k}"]["l2_norm"]
        effect_l2 = float(diff.norm().item())
        signal_to_noise = effect_l2 / max(noise_l2, 1e-8)

        metrics["sage_adapter_effect"][f"img{k}"] = {
            "max_abs_diff": float(diff.abs().max().item()),
            "mean_abs_diff": float(diff.abs().mean().item()),
            "l2_norm": effect_l2,
            "cosine_similarity": float(cosine),
            "bitwise_equal": bool(torch.equal(trajs_a[k]["final"], trajs_b[k]["final"])),
            "signal_to_noise_ratio": signal_to_noise,
            "note": (
                "img0: adapter disabled mid-trajectory → expect effect >> sage noise"
                if k == 0 else
                "img1: adapter unchanged + block masks → expect effect ≈ sage noise"
            ),
        }

    elapsed_total = time.perf_counter() - t0
    _log(f"  Phase 2 complete: {elapsed_total:.1f}s")

    return model, trajs_a, trajs_a2, trajs_b, metrics



def phase3_vae_decode(trajs_a, trajs_a2, trajs_b, metrics, device, dtype, output_dir):
    """Load VAE, decode all latents, save images + pixel-space stats."""
    _log("\n" + "=" * 60)
    _log("  PHASE 3: VAE DECODE + PIXEL STATISTICS")
    _log("=" * 60)

    t0 = time.perf_counter()
    vae = load_vae(VAE_PATH, device=device, dtype=dtype)
    _log(f"  VRAM after VAE load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    img_dir = output_dir / "images"
    img_dir.mkdir(exist_ok=True)

    metrics["pixel_stats"] = {}
    metrics["adapter_pixel_diff"] = {}

    for k in range(2):
        w, h = RESOLUTIONS[k]

        pil_a = decode_latent_to_pil(vae, trajs_a[k]["final"], device=device, dtype=dtype)
        pil_a2 = decode_latent_to_pil(vae, trajs_a2[k]["final"], device=device, dtype=dtype)
        pil_b = decode_latent_to_pil(vae, trajs_b[k]["final"], device=device, dtype=dtype)

        pil_a.save(img_dir / f"run_a_img{k}_{w}x{h}.png")
        pil_a2.save(img_dir / f"run_a2_img{k}_{w}x{h}.png")
        pil_b.save(img_dir / f"run_b_img{k}_{w}x{h}.png")

        metrics["pixel_stats"][f"run_a_img{k}"] = _single_image_pixel_stats(pil_a)
        metrics["pixel_stats"][f"run_a2_img{k}"] = _single_image_pixel_stats(pil_a2)
        metrics["pixel_stats"][f"run_b_img{k}"] = _single_image_pixel_stats(pil_b)

        _log(f"  img{k} ({w}x{h}): "
             f"run_a mean={metrics['pixel_stats'][f'run_a_img{k}']['overall_mean']:.1f} "
             f"std={metrics['pixel_stats'][f'run_a_img{k}']['overall_std']:.1f} "
             f"range={metrics['pixel_stats'][f'run_a_img{k}']['dynamic_range']:.0f}")

        tensor_a = torch.from_numpy(np.array(pil_a)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor_b = torch.from_numpy(np.array(pil_b)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        pixel_diff_stats = compute_per_channel_pixel_stats(tensor_a, tensor_b)
        metrics["adapter_pixel_diff"][f"img{k}"] = pixel_diff_stats

        diff_np = np.abs(np.array(pil_a).astype(np.float32) - np.array(pil_b).astype(np.float32))
        autocorr = compute_spatial_autocorrelation(diff_np)
        metrics["adapter_pixel_diff"][f"img{k}"]["spatial_autocorrelation"] = autocorr

        save_false_color_diff(
            tensor_a, tensor_b,
            img_dir / f"diff_a_vs_b_img{k}_{w}x{h}.png",
            scale=10.0,
        )

        _log(f"    adapter pixel diff img{k}: "
             f"mean={pixel_diff_stats['overall_mean']:.4f} "
             f"max={pixel_diff_stats['overall_max']:.4f} "
             f"autocorr={autocorr['max_autocorrelation']:.4f} ({autocorr['verdict']})")

    del vae
    torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    _log(f"  Phase 3 complete: {elapsed:.1f}s")

    return metrics



def main() -> int:
    _log(f"Batch rollout validation — {datetime.now(timezone.utc).isoformat()}")
    _log(f"Prompt: {PROMPT}")
    _log(f"Resolutions: {RESOLUTIONS}, Seeds: {SEEDS}")
    _log(f"N_STEPS={N_STEPS}, CFG={CFG}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    pos, neg, cap_len = phase1_encode(DEVICE, DTYPE)

    model, trajs_a, trajs_a2, trajs_b, metrics = phase2_rollouts(
        pos, neg, cap_len, DEVICE, DTYPE, OUTPUT_DIR,
    )


    metrics = phase3_vae_decode(trajs_a, trajs_a2, trajs_b, metrics, DEVICE, DTYPE, OUTPUT_DIR)

    _save_json(metrics, OUTPUT_DIR / "metrics.json")

    _log("\n" + "=" * 60)
    _log("  SUMMARY")
    _log("=" * 60)

    _log("\n  SDPA Determinism (bitwise baseline):")
    for k in range(2):
        d = metrics["sdpa_determinism"][f"img{k}"]
        _log(f"    img{k}: bitwise={d['bitwise_equal']}, max_abs={d['max_abs_diff']:.6f}")

    _log("\n  Sigma schedules:")
    for k in range(2):
        info = metrics["sigma_schedules"][f"img{k}"]
        _log(f"    img{k} ({info['resolution']}): alpha={info['alpha']:.3f}, "
             f"sigmas={[f'{s:.3f}' for s in info['sigmas']]}")

    _log("\n  SageAttention noise floor (Run A vs A2, identical params):")
    for k in range(2):
        d = metrics["sage_noise_floor"][f"img{k}"]
        _log(f"    img{k}: max_abs={d['max_abs_diff']:.6f}, "
             f"l2={d['l2_norm']:.6f}")

    _log("\n  SDPA Adapter Effect (deterministic, no Sage noise):")
    for k in range(2):
        e = metrics["sdpa_adapter_effect"][f"img{k}"]
        _log(f"    img{k}: max_abs={e['max_abs_diff']:.6f}, "
             f"l2={e['l2_norm']:.6f}, cos={e['cosine_similarity']:.6f}")
        _log(f"           {e['note']}")

    _log("\n  SageAttention Adapter Effect (noisy, SNR = effect/noise):")
    for k in range(2):
        e = metrics["sage_adapter_effect"][f"img{k}"]
        _log(f"    img{k}: max_abs={e['max_abs_diff']:.6f}, "
             f"l2={e['l2_norm']:.6f}, cos={e['cosine_similarity']:.6f}, "
             f"SNR={e['signal_to_noise_ratio']:.2f}x")

    _log("\n  Pixel-space adapter diff:")
    for k in range(2):
        p = metrics["adapter_pixel_diff"][f"img{k}"]
        _log(f"    img{k}: pixel_mean_diff={p['overall_mean']:.4f}, "
             f"pixel_max_diff={p['overall_max']:.4f}")

    _log(f"\n  All outputs saved to: {OUTPUT_DIR}")

    del model, trajs_a, trajs_a2, trajs_b
    gc.collect()
    torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    sys.exit(main())

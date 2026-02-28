r"""Validate BatchExecutor (bin-packed) vs serial (one-entry-per-launch) for K=6 6-tuple.

Proves that bin-packed multi-entry FlexAttention launches produce numerically
identical results to serial execution within the known divergence floor
(~0.0625 max_abs per step from block mask quantization).

5 phases:
  1. TE encode (shared prompts)
  2. Load model + compile
  3. Comparison runs (step-by-step + full trajectory)
  4. Persist diagnostics (JSONL, JSON)
  5. VAE decode comparison images

Usage:
    .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\validate_packed_vs_serial_ktuple.py
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
import torch.nn.functional as F


FP8_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
TE_PATH  = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"
VAE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\vae\zimage.safetensors"

N_STEPS = 10
SEED    = 42

DEVICE = torch.device("cuda")
DTYPE  = torch.bfloat16

OUTPUT_DIR = REPO_ROOT / "validation_renders" / "packed_vs_serial_ktuple"

MAX_ABS_THRESHOLD = 0.2

P_BASE   = 'qwen-3-4b, draw me "pink shrimp with crisp typography in a banana field".'
P_SHRIMP = "pink shrimp, detailed color, clear shape"
P_TYPO   = "clean typography, sharp letterforms, no blur"
P_BANANA = "banana, yellow, tropical poem"
P_NEG    = ""
UNIQUE_PROMPTS = [P_BASE, P_SHRIMP, P_TYPO, P_BANANA, P_NEG]

ENTRY_DEFS = [
    (P_BASE,   (1024, 1024)),  # entry 0: base
    (P_SHRIMP, (1024, 1024)),  # entry 1: shrimp emphasis
    (P_TYPO,   (1024, 1024)),  # entry 2: typography
    (P_BASE,   ( 512,  512)),  # entry 3: mid-res
    (P_BASE,   ( 256,  256)),  # entry 4: low-res
    (P_BANANA, (1024, 1024)),  # entry 5: banana
]



def _log(msg: str) -> None:
    print(msg, flush=True)



def phase1_encode(device, dtype) -> dict[str, torch.Tensor]:
    """Load TE, encode all unique prompts, free TE. Returns CPU tensors."""
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



def _build_serial_plans(model, entries_gpu, device):
    """Pre-build packing plans for serial execution (one entry per plan).

    Plans depend only on (cond_shape, latent_shape) which are constant across
    Euler steps, so we build them once and reuse.
    """
    from src_ii.forward_packed import prepare_packed_forward

    plans = []
    for entry in entries_gpu:
        cond = entry["cond"]
        cap_len = cond.shape[1]
        lh, lw = entry["x"].shape[2], entry["x"].shape[3]
        plan = prepare_packed_forward(
            model, [cond], [(lh, lw)], [cap_len], device,
        )
        plans.append(plan)
    return plans


def _run_serial_one_step(model, entries_gpu, plans, sigma, device):
    """Run one forward pass per entry (serial). Returns list of (denoised, scores).

    Uses pre-built plans from _build_serial_plans() to avoid recomputing
    packing state every step.
    """
    from src_ii.forward_packed import packed_forward
    from src_ii.triumphant_future_reduction_ops import denoise_all

    denoised_list = []
    scores_list = []

    for entry, plan in zip(entries_gpu, plans):
        x = entry["x"]
        ts = torch.tensor([sigma], device=device, dtype=torch.float32)
        with torch.no_grad():
            fields, scores = packed_forward(
                model, [x], [ts],
                plan["refined_caps"], plan["packing_info"],
                plan["block_mask"], plan["packed_rope"],
            )

        sigma_t = torch.tensor(sigma, device=device, dtype=x.dtype)
        denoised = [d for d in denoise_all([x], fields, [sigma_t])][0]
        denoised_list.append(denoised)
        scores_list.append(scores[0] if scores is not None else None)

    return denoised_list, scores_list


def _run_packed_one_step(executor, x_base, conds_gpu, sigma, device):
    """Run one forward pass via BatchExecutor. Returns list of (denoised, scores)."""
    forks = []
    for i, (prompt_key, (rw, rh)) in enumerate(ENTRY_DEFS):
        fork = {"entry_id": f"e{i}"}
        if prompt_key != P_BASE:
            fork["cond"] = conds_gpu[prompt_key]
        if (rw, rh) != (1024, 1024):
            fork["resolution"] = (rw, rh)
        forks.append(fork)

    query = {
        "query_id": "q0",
        "base_latent": x_base,
        "base_cond": conds_gpu[P_BASE],
        "base_cap_len": conds_gpu[P_BASE].shape[1],
        "base_resolution": (1024, 1024),
        "sigma": float(sigma),
        "forks": forks,
    }

    results = executor.execute([query])

    result_by_id = {r["entry_id"]: r for r in results}
    denoised_list = []
    scores_list = []
    for i in range(len(ENTRY_DEFS)):
        r = result_by_id[f"e{i}"]
        denoised_list.append(r["denoised"].to(device))
        scores_list.append(r["scores"])

    return denoised_list, scores_list


def _build_serial_entries(x_base, conds_gpu, device):
    """Build per-entry latents matching BatchExecutor._scatter() logic."""
    from src_ii.triumphant_future_reduction_ops import latent_padded

    entries = []
    base_lh, base_lw = x_base.shape[2], x_base.shape[3]

    for prompt_key, (rw, rh) in ENTRY_DEFS:
        lh, lw = latent_padded(rw, rh)
        if lh == base_lh and lw == base_lw:
            x = x_base.clone()
        else:
            x = F.interpolate(
                x_base, size=(lh, lw),
                mode="bilinear", align_corners=False,
            )
        entries.append({
            "x": x.to(device),
            "cond": conds_gpu[prompt_key],
        })

    return entries



def phase3_compare(model, conds_cpu, device, dtype):
    """Run step-by-step and full-trajectory comparisons."""
    _log("\n" + "=" * 60)
    _log("  PHASE 3: COMPARISON RUNS")
    _log("=" * 60)

    from src_ii.batch_executor import BatchExecutor
    from src_ii.sigma_schedule import build_sigma_schedule, resolution_shift
    from src_ii.triumphant_future_reduction_ops import (
        noise_field, aperture, euler_step, latent_padded,
    )

    conds_gpu = {k: v.to(device) for k, v in conds_cpu.items()}

    alpha = resolution_shift(1024, 1024)
    sigmas = build_sigma_schedule(N_STEPS, sampling_shift=alpha,
                                  device=device, dtype=dtype)
    _log(f"  Sigma schedule ({N_STEPS} steps): "
         f"[{float(sigmas[0]):.4f} ... {float(sigmas[-1]):.4f}]")

    executor = BatchExecutor(model, device)

    _log(f"  Entry definitions:")
    for i, (p, (rw, rh)) in enumerate(ENTRY_DEFS):
        lh, lw = latent_padded(rw, rh)
        _log(f"    e{i}: {rw}x{rh} -> latent {lh}x{lw}, "
             f"prompt='{p[:40]}...'")

    base_h, base_w = 1024 // 8, 1024 // 8  # 128x128
    master = noise_field(base_h, base_w, SEED, device, dtype)
    x_base_init = sigmas[0] * aperture(master, base_h, base_w)

    _log("\n  --- Part A: Step-by-step forward pass comparison ---")

    step_divergences = []
    x_base = x_base_init.clone()

    serial_entries_init = _build_serial_entries(x_base, conds_gpu, device)
    serial_plans = _build_serial_plans(model, serial_entries_init, device)
    _log(f"  Serial plans pre-built: {len(serial_plans)} entries")

    t0 = time.perf_counter()

    for step_i in range(N_STEPS):
        step_t0 = time.perf_counter()
        sigma_i = float(sigmas[step_i])
        sigma_next = float(sigmas[step_i + 1])

        serial_entries = _build_serial_entries(x_base, conds_gpu, device)
        denoised_serial, scores_serial = _run_serial_one_step(
            model, serial_entries, serial_plans, sigma_i, device,
        )
        serial_ms = (time.perf_counter() - step_t0) * 1000
        vram_after_serial = torch.cuda.memory_allocated() / 1e9

        packed_t0 = time.perf_counter()
        denoised_packed, scores_packed = _run_packed_one_step(
            executor, x_base, conds_gpu, sigma_i, device,
        )
        packed_ms = (time.perf_counter() - packed_t0) * 1000
        vram_after_packed = torch.cuda.memory_allocated() / 1e9


        _log(f"    step {step_i}: serial={serial_ms:.0f}ms packed={packed_ms:.0f}ms "
            f"VRAM={vram_after_packed:.2f}GB")

        step_record = {"step": step_i, "sigma": sigma_i, "entries": []}

        for entry_idx in range(len(ENTRY_DEFS)):
            d_ser = denoised_serial[entry_idx]
            d_pack = denoised_packed[entry_idx]

            assert d_ser.shape == d_pack.shape, (
                f"Shape mismatch at step {step_i} entry {entry_idx}: "
                f"serial={d_ser.shape} packed={d_pack.shape}"
            )

            diff = (d_ser - d_pack).float()
            max_abs = float(diff.abs().max())
            mean_abs = float(diff.abs().mean())

            s_flat = d_ser.flatten().float()
            p_flat = d_pack.flatten().float()
            cos_sim = float(F.cosine_similarity(s_flat.unsqueeze(0),
                                                 p_flat.unsqueeze(0)))

            entry_record = {
                "entry_id": f"e{entry_idx}",
                "resolution": list(ENTRY_DEFS[entry_idx][1]),
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "cosine_similarity": cos_sim,
                "serial_norm": float(s_flat.norm()),
                "packed_norm": float(p_flat.norm()),
            }
            step_record["entries"].append(entry_record)

        step_divergences.append(step_record)

        max_abs_all = max(e["max_abs"] for e in step_record["entries"])
        mean_abs_all = sum(e["mean_abs"] for e in step_record["entries"]) / len(step_record["entries"])
        min_cos = min(e["cosine_similarity"] for e in step_record["entries"])
        _log(f"    step {step_i:2d} sigma={sigma_i:.4f}  "
             f"max_abs={max_abs_all:.6f}  mean_abs={mean_abs_all:.6f}  "
             f"min_cos={min_cos:.8f}")

        guided = denoised_serial[0]  # Use entry 0 (base cond, base res) as the trajectory
        x_base = euler_step(x_base, guided,
                            torch.tensor(sigma_i, device=device, dtype=dtype),
                            torch.tensor(sigma_next, device=device, dtype=dtype))

    elapsed_a = time.perf_counter() - t0
    _log(f"  Part A done: {elapsed_a:.1f}s")

    _log("\n  --- Part B: Full trajectory comparison (independent Euler) ---")

    x_serial = x_base_init.clone()
    x_packed = x_base_init.clone()

    traj_divergences = []
    t1 = time.perf_counter()

    for step_i in range(N_STEPS):
        sigma_i = float(sigmas[step_i])
        sigma_next = float(sigmas[step_i + 1])

        serial_entries = _build_serial_entries(x_serial, conds_gpu, device)
        denoised_serial_b, _ = _run_serial_one_step(
            model, serial_entries, serial_plans, sigma_i, device,
        )

        denoised_packed_b, _ = _run_packed_one_step(
            executor, x_packed, conds_gpu, sigma_i, device,
        )

        x_serial = euler_step(
            x_serial, denoised_serial_b[0],
            torch.tensor(sigma_i, device=device, dtype=dtype),
            torch.tensor(sigma_next, device=device, dtype=dtype),
        )
        x_packed = euler_step(
            x_packed, denoised_packed_b[0],
            torch.tensor(sigma_i, device=device, dtype=dtype),
            torch.tensor(sigma_next, device=device, dtype=dtype),
        )

        traj_diff = (x_serial - x_packed).float()
        traj_max_abs = float(traj_diff.abs().max())
        traj_mean_abs = float(traj_diff.abs().mean())
        cos = float(F.cosine_similarity(
            x_serial.flatten().float().unsqueeze(0),
            x_packed.flatten().float().unsqueeze(0),
        ))

        traj_divergences.append({
            "step": step_i,
            "sigma": sigma_i,
            "traj_max_abs": traj_max_abs,
            "traj_mean_abs": traj_mean_abs,
            "traj_cosine": cos,
        })

        _log(f"    step {step_i:2d} sigma={sigma_i:.4f}  "
             f"traj_max_abs={traj_max_abs:.6f}  traj_cos={cos:.8f}")

    elapsed_b = time.perf_counter() - t1
    _log(f"  Part B done: {elapsed_b:.1f}s")

    packing_plans = {}
    for key, plan in executor._plan_cache.items():
        bins = plan["bins"]
        packing_plans[key] = {
            "n_bins": len(bins),
            "bins": [
                {
                    "entry_indices": b,
                    "entry_ids": [f"e{i}" for i in b],
                    "resolutions": [list(ENTRY_DEFS[i][1]) for i in b],
                }
                for b in bins
            ],
        }

    return (
        step_divergences,
        traj_divergences,
        packing_plans,
        x_serial.cpu(),
        x_packed.cpu(),
        elapsed_a + elapsed_b,
    )



def phase4_persist(step_divs, traj_divs, packing_plans, elapsed):
    """Write JSONL and JSON diagnostics."""
    _log("\n" + "=" * 60)
    _log("  PHASE 4: PERSIST DIAGNOSTICS")
    _log("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = OUTPUT_DIR / "divergence_per_step.jsonl"
    with open(jsonl_path, "w") as f:
        for record in step_divs:
            f.write(json.dumps(record) + "\n")
    _log(f"  {jsonl_path.name}: {len(step_divs)} records")

    traj_path = OUTPUT_DIR / "trajectory_divergence.jsonl"
    with open(traj_path, "w") as f:
        for record in traj_divs:
            f.write(json.dumps(record) + "\n")
    _log(f"  {traj_path.name}: {len(traj_divs)} records")

    plans_path = OUTPUT_DIR / "packing_plans.json"
    with open(plans_path, "w") as f:
        json.dump(packing_plans, f, indent=2)
    _log(f"  {plans_path.name}: {len(packing_plans)} plan(s)")

    all_max_abs = []
    for record in step_divs:
        for e in record["entries"]:
            all_max_abs.append(e["max_abs"])

    overall_max = max(all_max_abs) if all_max_abs else 0.0
    overall_mean = sum(all_max_abs) / len(all_max_abs) if all_max_abs else 0.0

    traj_final_max = traj_divs[-1]["traj_max_abs"] if traj_divs else 0.0
    traj_final_cos = traj_divs[-1]["traj_cosine"] if traj_divs else 1.0

    passed = overall_max < MAX_ABS_THRESHOLD

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_steps": N_STEPS,
        "n_entries": len(ENTRY_DEFS),
        "threshold": MAX_ABS_THRESHOLD,
        "step_comparison": {
            "overall_max_abs": overall_max,
            "overall_mean_max_abs": overall_mean,
            "per_step_max": [
                max(e["max_abs"] for e in r["entries"]) for r in step_divs
            ],
        },
        "trajectory_comparison": {
            "final_max_abs": traj_final_max,
            "final_cosine": traj_final_cos,
            "per_step_max_abs": [r["traj_max_abs"] for r in traj_divs],
        },
        "packing_plans": packing_plans,
        "elapsed_s": elapsed,
        "pass": passed,
        "verdict": "PASS" if passed else "FAIL",
    }

    summary_path = OUTPUT_DIR / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"  {summary_path.name}: verdict={summary['verdict']}")

    return passed, overall_max, traj_final_max, traj_final_cos



def phase5_vae_decode(x_serial_cpu, x_packed_cpu, device, dtype):
    """Decode serial and packed final latents, save side-by-side."""
    _log("\n" + "=" * 60)
    _log("  PHASE 5: VAE DECODE COMPARISON")
    _log("=" * 60)

    from src_ii.vae_utils import load_vae, decode_latent_to_pil

    vae = load_vae(VAE_PATH, device=device, dtype=dtype)
    _log(f"  VRAM after VAE load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    img_serial = decode_latent_to_pil(vae, x_serial_cpu.to(device),
                                       device=device, dtype=dtype)
    img_packed = decode_latent_to_pil(vae, x_packed_cpu.to(device),
                                       device=device, dtype=dtype)

    img_serial.save(OUTPUT_DIR / "final_serial.png")
    img_packed.save(OUTPUT_DIR / "final_packed.png")
    _log(f"  final_serial.png: {img_serial.size}")
    _log(f"  final_packed.png: {img_packed.size}")

    from PIL import Image
    w, h = img_serial.size
    composite = Image.new("RGB", (w * 2 + 10, h), (40, 40, 40))
    composite.paste(img_serial, (0, 0))
    composite.paste(img_packed, (w + 10, 0))
    composite.save(OUTPUT_DIR / "side_by_side.png")
    _log(f"  side_by_side.png: {composite.size}")

    del vae
    gc.collect()
    torch.cuda.empty_cache()



def main() -> int:
    _log(f"Packed vs Serial K-tuple validation -- {datetime.now(timezone.utc).isoformat()}")
    _log(f"  N_STEPS={N_STEPS}, SEED={SEED}, threshold={MAX_ABS_THRESHOLD}")
    _log(f"  Entries: {len(ENTRY_DEFS)}")
    for i, (p, (rw, rh)) in enumerate(ENTRY_DEFS):
        _log(f"    e{i}: {rw}x{rh} '{p[:50]}'")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    conds_cpu = phase1_encode(DEVICE, DTYPE)

    _log("\n" + "=" * 60)
    _log("  PHASE 2: LOAD MODEL")
    _log("=" * 60)

    from src_ii.zimage_model import load_zimage_rlaif
    import src_ii.attention_srcii  # noqa: F401  -- trigger op registration

    t0 = time.perf_counter()
    model = load_zimage_rlaif(
        FP8_PATH, device=DEVICE, dtype=DTYPE,
        compile_model=True, fuse=True,
    )
    _log(f"  Loaded + compiled in {time.perf_counter() - t0:.1f}s")
    _log(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    model = model

    (
        step_divs,
        traj_divs,
        packing_plans,
        x_serial_cpu,
        x_packed_cpu,
        elapsed,
    ) = phase3_compare(model, conds_cpu, DEVICE, DTYPE)

    passed, overall_max, traj_final_max, traj_final_cos = phase4_persist(
        step_divs, traj_divs, packing_plans, elapsed,
    )

    del model, model, model
    gc.collect()
    torch.cuda.empty_cache()
    _log(f"  VRAM after backbone free: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    phase5_vae_decode(x_serial_cpu, x_packed_cpu, DEVICE, DTYPE)

    _log("\n" + "=" * 60)
    _log("  FINAL VERDICT")
    _log("=" * 60)
    _log(f"  Step-by-step max divergence: {overall_max:.6f}")
    _log(f"  Trajectory final max_abs:    {traj_final_max:.6f}")
    _log(f"  Trajectory final cosine:     {traj_final_cos:.8f}")
    _log(f"  Threshold:                   {MAX_ABS_THRESHOLD}")
    _log(f"  Output:                      {OUTPUT_DIR}")

    if passed:
        _log(f"\n  PASS -- max_abs {overall_max:.6f} < {MAX_ABS_THRESHOLD}")
    else:
        _log(f"\n  FAIL -- max_abs {overall_max:.6f} >= {MAX_ABS_THRESHOLD}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

r"""Demonstrate r_theta LoRA adapter effect on diffusion field and BTRM scores.

For each prompt:
  - Euler-sample at adapter_scales=[[0.0]] (reference) and [[1.0]] (r_theta active)
  - Record BTRM scores (both heads) at every step via sample_trajectory()
  - VAE decode final latents to pixelspace
  - Plot score vs logSNR charts alongside pixelspace renders

Algorithmic responsibilities delegated to src_ii/infer/ modules:
  - encode_prompts         (text_encoding.py) -- TE load/encode/free lifecycle
  - load_and_prepare_model (model_setup.py)   -- model load, LoRA install, compile
  - sample_trajectory      (trajectory.py)    -- Euler ODE loop + score recording
  - draw_score_chart       (charts.py)        -- PIL chart rendering
  - build_comparison_composite (composites.py)-- layout and panel assembly

This script is pure scheduling: constants, phase headers, loops, I/O.

Output: validation_renders/rtheta_policy_demo/
  {slug}_ref.png, {slug}_rtheta.png, {slug}_composite.png
  scores_per_step.jsonl, manifest.json

Usage:
    .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\demonstrate_rtheta_policy.py
"""
from __future__ import annotations
import gc, json, sys, time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from src_ii.infer.text_encoding  import encode_prompts
from src_ii.infer.model_setup    import load_and_prepare_model
from src_ii.inference_sampling   import run_trajectory_packed
from src_ii.infer.charts         import draw_score_chart
from src_ii.infer.composites     import build_comparison_composite
from src_ii.sigma_schedule       import sigma_to_logsnr
from src_ii.vae_utils            import load_vae, decode_latent_to_pil

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FP8_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
TE_PATH  = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"
VAE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\vae\zimage.safetensors"
BTRM_DIR = REPO_ROOT / "training_output" / "reward_function_run_tnt_v2"
N_STEPS  = 30
SEED     = 42
DEVICE   = torch.device("cuda")
DTYPE    = torch.bfloat16
OUTPUT_DIR = REPO_ROOT / "validation_renders" / "rtheta_policy_demo"
RES_W, RES_H = 1280, 832
PROMPTS = [
    ("shrimp_field", 'qwen-3-4b, draw me "pink shrimp with crisp typography in a banana field".'),
    ("laser_shark",  "enormous laser sharks, photorealistic, dramatic lighting, ocean scene"),
    ("portrait",     "portrait of an old fisherman, weathered face, golden hour, shallow depth of field"),
]
ADAPTER_NAME = "rtheta"


def main() -> int:
    def hdr(msg): print(f"\n{'='*60}\n  {msg}\n{'='*60}", flush=True)

    print(f"r_theta policy demo — {datetime.now(timezone.utc).isoformat()}", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.perf_counter()

    # Phase 1: Text Encoding
    hdr("PHASE 1: TEXT ENCODER")
    conds = encode_prompts(PROMPTS, TE_PATH, DEVICE, DTYPE)

    # Phase 2: Load Model + r_theta Adapter
    hdr("PHASE 2: LOAD MODEL + R_THETA ADAPTER")
    with open(BTRM_DIR / "btrm_compound_config.json") as f:
        btrm_cfg = json.load(f)
    adapter_configs = [{"name": ADAPTER_NAME,
                        "rank": btrm_cfg["adapter_rank"],
                        "alpha": btrm_cfg["adapter_alpha"]}]
    model, raw_model, head_names = load_and_prepare_model(
        FP8_PATH, adapter_configs, BTRM_DIR, ADAPTER_NAME, device=DEVICE, dtype=DTYPE,
    )

    # Phase 3: Dual-Scale Euler Sampling (via authoritative CFG path)
    hdr("PHASE 3: DUAL-SCALE EULER SAMPLING")
    scales_ref    = torch.tensor([[0.0]], device=DEVICE)
    scales_rtheta = torch.tensor([[1.0]], device=DEVICE)

    # Encode empty negative prompt for CFG
    from src_ii.infer.text_encoding import encode_prompts as _ep
    neg_cond = _ep([("_neg", "")], TE_PATH, DEVICE, DTYPE)["_neg"]

    all_results: dict[str, dict] = {}
    for slug, cond_cpu in conds.items():
        print(f"\n  --- {slug} ---", flush=True)
        t0 = time.perf_counter()
        cond = cond_cpu.to(DEVICE)
        params = {
            "n_images": 1, "seeds": [SEED], "n_steps": N_STEPS,
            "cfg": 4.0, "multiplier": 1.0, "denoise": 1.0,
            "width": RES_W, "height": RES_H,
        }
        tensors = {"neg_cond": neg_cond, "pos_cond_0": cond}

        result_ref, meta_ref = run_trajectory_packed(
            model, DEVICE, DTYPE, params, tensors,
            adapter_scales=scales_ref,
        )
        result_rth, meta_rth = run_trajectory_packed(
            model, DEVICE, DTYPE, params, tensors,
            adapter_scales=scales_rtheta,
        )
        elapsed = time.perf_counter() - t0
        print(f"  {slug}: {elapsed:.1f}s ({elapsed/N_STEPS:.2f}s/step)", flush=True)
        all_results[slug] = {
            "latent_ref": result_ref["final_0"], "latent_rtheta": result_rth["final_0"],
            "recs_ref": meta_ref.get("step_scores", []),
            "recs_rtheta": meta_rth.get("step_scores", []),
        }

    del model, raw_model
    gc.collect(); torch.cuda.empty_cache()
    print(f"  Backbone freed. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    # Phase 4: VAE Decode
    hdr("PHASE 4: VAE DECODE")
    t0  = time.perf_counter()
    vae = load_vae(VAE_PATH, device=DEVICE, dtype=DTYPE)
    print(f"  VAE loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
    images: dict[str, dict] = {}
    for slug, result in all_results.items():
        img_ref    = decode_latent_to_pil(vae, result["latent_ref"],    device=DEVICE, dtype=DTYPE)
        img_rtheta = decode_latent_to_pil(vae, result["latent_rtheta"], device=DEVICE, dtype=DTYPE)
        images[slug] = {"ref": img_ref, "rtheta": img_rtheta}
        print(f"  {slug}: ref {img_ref.size}  rtheta {img_rtheta.size}", flush=True)
    del vae
    gc.collect(); torch.cuda.empty_cache()
    print(f"  VAE freed. Phase 4: {time.perf_counter()-t0:.1f}s", flush=True)

    # Phase 5: Charts, Composites, JSONL, Manifest
    hdr("PHASE 5: COMPOSITE RENDERS")
    scores_log: list[dict] = []
    for slug, result in all_results.items():
        recs_ref    = result["recs_ref"]
        recs_rtheta = result["recs_rtheta"]
        img_ref     = images[slug]["ref"]
        img_rtheta  = images[slug]["rtheta"]
        img_ref.save(OUTPUT_DIR / f"{slug}_ref.png")
        img_rtheta.save(OUTPUT_DIR / f"{slug}_rtheta.png")

        logsnrs = [sigma_to_logsnr(r["sigma"]) for r in recs_ref]
        n_heads = len(recs_ref[0]["scores"]) if recs_ref else 0
        charts  = []
        for head_idx in range(n_heads):
            head_name = head_names[head_idx] if head_idx < len(head_names) else f"head_{head_idx}"
            named_series = {
                "ref":    {"values": [r["scores"][head_idx] for r in recs_ref],    "color": (50, 50, 200)},
                "rtheta": {"values": [r["scores"][head_idx] for r in recs_rtheta], "color": (200, 50, 50)},
            }
            charts.append(draw_score_chart(logsnrs, named_series, head_name))

        composite = build_comparison_composite(
            image_panels=[img_ref, img_rtheta],
            panel_labels=["ref (scale=0)", "r_theta (scale=1)"],
            charts=charts,
            title=f"{slug} — r_theta adapter policy demonstration",
        )
        composite_path = OUTPUT_DIR / f"{slug}_composite.png"
        composite.save(str(composite_path))
        print(f"  {slug}: composite {composite.size[0]}x{composite.size[1]} -> {composite_path.name}", flush=True)

        for step_i, (r_ref, r_rth) in enumerate(zip(recs_ref, recs_rtheta)):
            sigma = r_ref["sigma"]
            scores_log.append({"slug": slug, "step": step_i,
                                "sigma": sigma, "logsnr": sigma_to_logsnr(sigma),
                                "scores_ref": r_ref["scores"], "scores_rtheta": r_rth["scores"]})

    scores_path = OUTPUT_DIR / "scores_per_step.jsonl"
    with open(scores_path, "w") as f:
        for entry in scores_log:
            f.write(json.dumps(entry) + "\n")
    print(f"  Scores log: {scores_path.name} ({len(scores_log)} entries)", flush=True)

    manifest = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "prompts":         {slug: prompt for slug, prompt in PROMPTS},
        "resolution":      f"{RES_W}x{RES_H}",
        "n_steps":         N_STEPS,
        "seed":            SEED,
        "adapter_name":    ADAPTER_NAME,
        "btrm_dir":        str(BTRM_DIR),
        "head_names":      head_names,
        "n_prompts":       len(PROMPTS),
        "output_files": (
            [f"{slug}_ref.png"       for slug, _ in PROMPTS] +
            [f"{slug}_rtheta.png"    for slug, _ in PROMPTS] +
            [f"{slug}_composite.png" for slug, _ in PROMPTS] +
            ["scores_per_step.jsonl"]
        ),
        "total_elapsed_s": time.perf_counter() - t_total,
    }
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}\n  DONE — {time.perf_counter()-t_total:.1f}s total\n  Output: {OUTPUT_DIR}\n{'='*60}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

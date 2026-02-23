"""Validate src_ii/ rollout against stored reference trajectories.

Loads the FP8 diffusion model (via src_ii.zimage_model.load_zimage_rlaif),
picks a reference trajectory from btrm_dataset/, runs the src_ii rollout
with the same inputs (seed, prompt conditioning, sigmas, cfg), and writes
comparison tensors to disk.

Does NOT assert pass/fail. Writes persistent files only.
The output directory contains:
  - reproduced step tensors (step_NN.pt, final.pt)
  - diff tensors (diff_step_NN.pt, diff_final.pt) -- element-wise difference
  - comparison_stats.json -- per-step L2 norm of difference, max abs diff, etc.

Execution environment:
  Windows Python from WSL2:
    /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe \\
        F:\\dox\\repos\\ai\\futudiffu\\scripts_ii\\validate_trajectory.py

Usage:
  python validate_trajectory.py [--traj-idx 0] [--output-dir validation_output_ii]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path, PureWindowsPath

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch


def wsl_to_win(p: str) -> str:
    """Convert WSL path to Windows path for arguments."""
    if p.startswith("/mnt/"):
        parts = p.split("/")
        drive = parts[2].upper()
        rest = "\\".join(parts[3:])
        return f"{drive}:\\{rest}"
    return p


def win_to_wsl(p: str) -> str:
    """Convert Windows path to WSL path for file access."""
    if len(p) >= 3 and p[1] == ":" and p[2] == "\\":
        drive = p[0].lower()
        rest = p[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return p


def load_reference_trajectory(traj_dir_wsl: str) -> dict:
    """Load all step_NN.pt and final.pt from a trajectory directory.

    Args:
        traj_dir_wsl: WSL path to trajectory directory.

    Returns:
        Dict mapping key names to tensors: {"step_00": tensor, ..., "final": tensor}
    """
    traj_path = Path(traj_dir_wsl)
    tensors = {}

    for pt_file in sorted(traj_path.glob("step_*.pt")):
        key = pt_file.stem  # e.g. "step_00"
        tensors[key] = torch.load(str(pt_file), weights_only=True)

    final_path = traj_path / "final.pt"
    if final_path.exists():
        tensors["final"] = torch.load(str(final_path), weights_only=True)

    return tensors


def main():
    parser = argparse.ArgumentParser(description="Validate src_ii rollout against reference trajectory")
    parser.add_argument("--traj-idx", type=int, default=0,
                        help="Trajectory index in btrm_dataset manifest (default: 0)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: validation_output_ii/)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip torch.compile (faster startup, slower inference)")
    parser.add_argument("--no-sage", action="store_true",
                        help="Skip SageAttention configuration (use SDPA)")
    parser.add_argument("--fp8-path", type=str, default=None,
                        help="Path to FP8 diffusion model safetensors")
    parser.add_argument("--te-path", type=str, default=None,
                        help="Path to text encoder safetensors (for encoding prompt)")
    parser.add_argument("--tokenizer-path", type=str, default=None,
                        help="Path to tokenizer directory")
    args = parser.parse_args()

    if args.fp8_path:
        fp8_path = args.fp8_path
    else:
        fp8_candidates = [
            Path(r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"),
            Path(r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_e4m3fn.safetensors"),
        ]
        fp8_path = None
        for c in fp8_candidates:
            if c.exists():
                fp8_path = str(c)
                break
        if fp8_path is None:
            print("ERROR: Could not find FP8 diffusion model. Use --fp8-path.")
            sys.exit(1)

    if args.te_path:
        te_path = args.te_path
    else:
        te_path = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"

    if args.tokenizer_path:
        tokenizer_path = args.tokenizer_path
    else:
        tokenizer_path = str(REPO_ROOT / "src" / "futudiffu" / "tokenizer")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = REPO_ROOT / "validation_output_ii"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = REPO_ROOT / "btrm_dataset" / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    records = manifest["records"]
    if args.traj_idx >= len(records):
        print(f"ERROR: traj_idx {args.traj_idx} out of range (0..{len(records)-1})")
        sys.exit(1)

    record = records[args.traj_idx]
    print(f"\n=== Validating trajectory {args.traj_idx} ===")
    print(f"  Type: {record['type']}")
    print(f"  Seed: {record['seed']}")
    print(f"  Prompt: {record['prompt'][:80]}...")
    print(f"  Steps: {record['n_steps']}")
    print(f"  Precision: {record['precision']}")

    if record["type"] != "t2i":
        print(f"WARNING: Trajectory type is '{record['type']}', not 't2i'.")
        print("  i2i validation requires VAE encoding of source images.")
        print("  Proceeding anyway, but results may not match if clean_latent differs.")

    traj_dir = REPO_ROOT / "btrm_dataset" / "latents" / f"traj_{args.traj_idx:06d}"
    print(f"  Trajectory dir: {traj_dir}")

    ref_tensors = load_reference_trajectory(str(traj_dir))
    print(f"  Reference tensors loaded: {sorted(ref_tensors.keys())}")

    ref_final = ref_tensors.get("final")
    if ref_final is None:
        print("ERROR: No final.pt in reference trajectory")
        sys.exit(1)

    _, _, latent_h, latent_w = ref_final.shape
    img_height = latent_h * 8
    img_width = latent_w * 8
    print(f"  Image size: {img_width}x{img_height} (latent {latent_w}x{latent_h})")

    print(f"\n--- Phase 1: Encoding prompt ---")
    device = torch.device("cuda")
    dtype = torch.bfloat16

    from futudiffu.text_encoder import create_tokenizer, load_text_encoder, encode_prompt

    tokenizer = create_tokenizer(tokenizer_path)
    te_model = load_text_encoder(te_path, device=device, dtype=dtype)
    te_compiled = torch.compile(te_model, mode="default")

    pos_cond = encode_prompt(te_compiled, tokenizer, record["prompt"], device=device)

    neg_cond = encode_prompt(te_compiled, tokenizer, "", device=device)

    print(f"  pos_cond shape: {pos_cond.shape}")
    print(f"  neg_cond shape: {neg_cond.shape}")

    torch.save(pos_cond.cpu(), output_dir / "pos_cond.pt")
    torch.save(neg_cond.cpu(), output_dir / "neg_cond.pt")

    del te_model, te_compiled
    torch.cuda.empty_cache()

    print(f"\n--- Phase 2: Loading diffusion model ---")
    from src_ii.zimage_model import load_zimage_rlaif

    attention_backend = record["precision"]  # "sdpa" or "sage"
    if args.no_sage and attention_backend == "sage":
        print("  WARNING: --no-sage flag set but trajectory used sage precision.")
        print("  Using SDPA instead. Results WILL differ from reference.")
        attention_backend = "sdpa"

    use_sage = (attention_backend == "sage")
    print(f"  Attention backend: {attention_backend}")

    diff_model = load_zimage_rlaif(
        fp8_path,
        device=device,
        dtype=dtype,
        compile_model=not args.no_compile,
        use_sage=use_sage,
    )

    print(f"\n--- Phase 3: Running src_ii rollout ---")
    from src_ii.rollout import rollout

    ref_step_indices = set()
    for key in ref_tensors:
        if key.startswith("step_"):
            idx = int(key.split("_")[1])
            ref_step_indices.add(idx)

    cfg = 4.0
    sampling_shift = 1.0
    multiplier = 1.0
    denoise_strength = record.get("denoise", 1.0)

    t0 = time.perf_counter()
    result_tensors, metadata = rollout(
        model=diff_model,
        pos_cond=pos_cond,
        neg_cond=neg_cond,
        seed=record["seed"],
        n_steps=record["n_steps"],
        cfg=cfg,
        width=img_width,
        height=img_height,
        device=device,
        dtype=dtype,
        sampling_shift=sampling_shift,
        multiplier=multiplier,
        denoise=denoise_strength,
        save_steps=ref_step_indices,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Rollout completed in {elapsed:.1f}s")
    print(f"  Result keys: {sorted(result_tensors.keys())}")

    print(f"\n--- Phase 4: Writing comparison outputs ---")
    traj_output_dir = output_dir / f"traj_{args.traj_idx:06d}"
    traj_output_dir.mkdir(parents=True, exist_ok=True)

    comparison_stats = {}

    for key in sorted(set(ref_tensors.keys()) | set(result_tensors.keys())):
        ref_t = ref_tensors.get(key)
        repro_t = result_tensors.get(key)

        if ref_t is None:
            print(f"  {key}: MISSING in reference")
            comparison_stats[key] = {"status": "missing_in_reference"}
            continue
        if repro_t is None:
            print(f"  {key}: MISSING in reproduced")
            comparison_stats[key] = {"status": "missing_in_reproduced"}
            continue

        ref_f32 = ref_t.float()
        repro_f32 = repro_t.float()
        diff = repro_f32 - ref_f32

        l2_norm = diff.norm().item()
        max_abs = diff.abs().max().item()
        mean_abs = diff.abs().mean().item()
        ref_norm = ref_f32.norm().item()
        relative_l2 = l2_norm / (ref_norm + 1e-10)

        stats = {
            "l2_norm": l2_norm,
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "ref_l2_norm": ref_norm,
            "relative_l2": relative_l2,
            "shape": list(ref_t.shape),
            "dtype": str(ref_t.dtype),
        }
        comparison_stats[key] = stats
        print(f"  {key}: L2={l2_norm:.6f}, max_abs={max_abs:.6f}, rel_L2={relative_l2:.6f}")

        torch.save(repro_t, traj_output_dir / f"{key}.pt")
        torch.save(diff.to(torch.bfloat16), traj_output_dir / f"diff_{key}.pt")

    stats_path = traj_output_dir / "comparison_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "traj_idx": args.traj_idx,
            "record": record,
            "generation_params": {
                "cfg": cfg,
                "sampling_shift": sampling_shift,
                "multiplier": multiplier,
                "denoise": denoise_strength,
                "width": img_width,
                "height": img_height,
            },
            "elapsed_seconds": elapsed,
            "comparison": comparison_stats,
        }, f, indent=2)

    print(f"\n  Output written to: {traj_output_dir}")
    print(f"  Stats written to: {stats_path}")
    print(f"\n=== Validation complete ===")


if __name__ == "__main__":
    main()

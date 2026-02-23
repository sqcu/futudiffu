r"""Compare BTRM model rankings against literal reward function rankings.

Loads the trained compound model from pinkify_thisnotthat_output/differentiable_run/
(or a directory specified via --model-dir), scores a test set of at least 100 images
at sigma=0 (step "final"), then computes:

  - Pairwise agreement percentage vs literal rules for both heads
  - Spearman rank correlation vs literal rules for both heads

All results are written to:
  pinkify_thisnotthat_output/differentiable_run/score_comparison.json
  (or the path specified via --output)

Execution:
  PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe \
      F:\dox\repos\ai\futudiffu\scripts_ii\compare_pinkify_scores.py \
      [--model-dir <path>] [--output <path>]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src_ii"))

import torch


FP8_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
TOKENIZER_PATH = str(REPO_ROOT / "src" / "futudiffu" / "tokenizer")
TE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"

V2_DATASET_DIR = REPO_ROOT / "btrm_dataset_v2"
SCORE_CACHE_PATH = REPO_ROOT / "pinkify_thisnotthat_output" / "v2_score_cache.json"
_DEFAULT_COMPOUND_MODEL_DIR = REPO_ROOT / "pinkify_thisnotthat_output" / "differentiable_run"

HEAD_NAMES = ("pinkify", "thisnotthat")

N_IMAGES = 259


def spearman_rank_correlation(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation. Available in src_ii.stats but inlined here
    to ensure the script is self-contained if the module path is tricky."""
    import numpy as np
    n = len(x)
    if n < 2:
        return 0.0
    sorted_x = sorted(range(n), key=lambda i: x[i])
    sorted_y = sorted(range(n), key=lambda i: y[i])
    rx = [0.0] * n
    ry = [0.0] * n
    for r, idx in enumerate(sorted_x):
        rx[idx] = r + 1
    for r, idx in enumerate(sorted_y):
        ry[idx] = r + 1
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    rho = 1 - (6 * d2) / (n * (n * n - 1))
    return float(rho)


def compute_pairwise_agreement(
    btrm_scores: list[float],
    literal_scores: list[float],
) -> tuple[int, int, list[dict]]:
    """Compare all N*(N-1)/2 pairs between BTRM and literal rankings.

    Returns:
        (n_agree, n_total_nontie, disagreement_details[:20])
    """
    n = len(btrm_scores)
    n_agree = 0
    n_total_nontie = 0
    disagreements = []

    for i in range(n):
        for j in range(i + 1, n):
            lit_diff = literal_scores[i] - literal_scores[j]
            btrm_diff = btrm_scores[i] - btrm_scores[j]

            if lit_diff == 0.0:
                continue

            n_total_nontie += 1
            lit_pref = 1 if lit_diff > 0 else -1
            btrm_pref = 1 if btrm_diff > 0 else -1

            if lit_pref == btrm_pref:
                n_agree += 1
            else:
                if len(disagreements) < 20:
                    disagreements.append({
                        "idx_a": i,
                        "idx_b": j,
                        "literal_pref": lit_pref,
                        "btrm_pref": btrm_pref,
                        "literal_diff": lit_diff,
                        "btrm_diff": btrm_diff,
                    })

    return n_agree, n_total_nontie, disagreements


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare BTRM model rankings against literal reward function rankings."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help=(
            "Path to trained compound model directory (contains btrm_compound_config.json, "
            "rtheta_adapter.safetensors, btrm_head.safetensors). "
            f"Default: {_DEFAULT_COMPOUND_MODEL_DIR}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path to write score_comparison.json. "
            "Default: <model_dir>/score_comparison.json"
        ),
    )
    args = parser.parse_args()

    COMPOUND_MODEL_DIR = args.model_dir if args.model_dir is not None else _DEFAULT_COMPOUND_MODEL_DIR
    COMPOUND_MODEL_DIR = COMPOUND_MODEL_DIR.resolve()
    OUTPUT_PATH = args.output if args.output is not None else (COMPOUND_MODEL_DIR / "score_comparison.json")
    OUTPUT_PATH = OUTPUT_PATH.resolve()

    wall_start = time.perf_counter()
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("=" * 60)
    print("  compare_pinkify_scores.py")
    print(f"  BTRM vs Literal Rule Comparison")
    print(f"  Model dir: {COMPOUND_MODEL_DIR}")
    print(f"  Output:    {OUTPUT_PATH}")
    print("=" * 60)

    print("\n[Phase 1] Loading literal scores from JSON...")
    import json as _json

    with open(str(SCORE_CACHE_PATH)) as _f:
        _raw_cache = _json.load(_f)
    print(f"  Loaded {len(_raw_cache)} cached score entries")

    final_entries = []
    for cache_key, scores in _raw_cache.items():
        if cache_key.endswith(":final"):
            traj_id_str = cache_key.split(":")[0]
            try:
                traj_id = int(traj_id_str)
            except ValueError:
                continue
            final_entries.append({
                "traj_id": traj_id,
                "step_key": "final",
                "literal_pinkify": scores.get("pinkify", 0.0),
                "literal_thisnotthat": scores.get("thisnotthat", 0.0),
            })

    final_entries.sort(key=lambda e: e["traj_id"])
    print(f"  Found {len(final_entries)} trajectories with 'final' step in cache")

    if len(final_entries) < 100:
        print(f"  WARNING: Only {len(final_entries)} entries -- expected >= 100")

    print("\n[Phase 2] Loading V2 dataset...")
    from futudiffu.dataset_v2 import DatasetReader

    reader = DatasetReader(str(V2_DATASET_DIR))
    print(f"  Dataset: {len(reader)} trajectories")

    valid_entries = []
    for e in final_entries:
        if e["traj_id"] in reader:
            valid_entries.append(e)

    print(f"  Valid entries (in V2 dataset): {len(valid_entries)}")

    if len(valid_entries) < 100:
        print(f"  ERROR: Only {len(valid_entries)} valid entries (need >= 100)")
        return 1

    print("\n[Phase 3] Encoding prompts with text encoder...")
    from futudiffu.text_encoder import create_tokenizer, load_text_encoder, encode_prompt

    tokenizer = create_tokenizer(TOKENIZER_PATH)
    te_model = load_text_encoder(TE_PATH, device=device, dtype=dtype)

    prompt_cache: dict[str, torch.Tensor] = {}
    for e in valid_entries:
        meta, _ = reader[e["traj_id"]]
        prompt = meta.get("prompt", "")
        if prompt and prompt not in prompt_cache:
            cond = encode_prompt(te_model, tokenizer, prompt, device=device)
            prompt_cache[prompt] = cond.cpu()

    del te_model, tokenizer
    torch.cuda.empty_cache()
    print(f"  Encoded {len(prompt_cache)} unique prompts. TE freed.")
    print(f"  VRAM after TE free: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\n[Phase 4] Loading backbone and compound BTRM model...")
    from src_ii.zimage_model import load_zimage_rlaif
    from src_ii.btrm_lifecycle import load_btrm, score_serial
    from src_ii.multi_lora import install_multi_lora
    from src_ii.sigma_schedule import build_sigma_schedule, resolution_shift

    raw_model = load_zimage_rlaif(
        FP8_PATH, device=device, dtype=dtype,
        compile_model=False, fuse=True,
    )
    print(f"  VRAM after backbone: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    install_multi_lora(raw_model, [{"name": "rtheta", "rank": 8, "alpha": 16.0}])
    load_btrm(raw_model, "rtheta", str(COMPOUND_MODEL_DIR))
    raw_model.gradient_checkpointing = False
    raw_model.eval()
    print(f"  Loaded compound BTRM model from {COMPOUND_MODEL_DIR}")
    print(f"  VRAM after compound model: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print(f"\n[Phase 5] Scoring {len(valid_entries)} images with BTRM model...")
    print("  (using sigma=0 / 'final' step for cleanest signal)")

    _meta_cache: dict[int, tuple] = {}

    def get_meta(traj_id: int):
        if traj_id not in _meta_cache:
            meta, accessor = reader[traj_id]
            _meta_cache[traj_id] = (meta, accessor)
        return _meta_cache[traj_id]

    def load_latent_fn(traj_id: int, step_key: str):
        meta, accessor = get_meta(traj_id)
        latent = accessor[step_key].to(device=device, dtype=dtype)
        if latent.dim() == 3:
            latent = latent.unsqueeze(0)  # (1, C, H, W)

        n_steps = meta.get("n_steps", 30)
        denoise_val = meta.get("denoise") or 1.0
        recorded_shift = meta.get("sampling_shift")
        if recorded_shift is not None:
            shift = float(recorded_shift)
        else:
            w = meta.get("width", 1280)
            h = meta.get("height", 832)
            shift = resolution_shift(w, h)

        sigmas = build_sigma_schedule(
            n_steps, sampling_shift=shift, denoise=denoise_val,
            device="cpu", dtype=torch.float32,
        )

        if step_key == "final":
            sigma_val = float(sigmas[-2].item()) if len(sigmas) > 1 else 0.01
        else:
            idx = int(step_key.split("_")[1])
            sigma_val = float(sigmas[idx].item()) if idx < len(sigmas) else 0.01

        timestep = torch.tensor([sigma_val], device=device, dtype=dtype)

        prompt = meta.get("prompt", "")
        cond = prompt_cache.get(prompt)
        if cond is None:
            raise ValueError(f"No encoded prompt for traj {traj_id}")
        cond = cond.to(device=device, dtype=dtype)

        num_tokens = cond.shape[1]

        return latent, timestep, cond, num_tokens

    scored_entries = []
    n_total = len(valid_entries)

    for i, entry in enumerate(valid_entries):
        traj_id = entry["traj_id"]
        step_key = entry["step_key"]

        try:
            lat, ts, cond, nt = load_latent_fn(traj_id, step_key)

            with torch.no_grad():
                scores = score_serial(raw_model, lat, ts, cond, nt, gradient_checkpointing=False)

            scored_entries.append({
                "traj_id": traj_id,
                "step_key": step_key,
                "literal_pinkify": entry["literal_pinkify"],
                "literal_thisnotthat": entry["literal_thisnotthat"],
                "btrm_pinkify": float(scores[0, 0].item()),
                "btrm_thisnotthat": float(scores[0, 1].item()),
            })

            del lat, ts, cond, scores
            torch.cuda.empty_cache()

            if (i + 1) % 50 == 0 or (i + 1) == n_total:
                print(f"  Scored {i + 1}/{n_total} images... "
                      f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        except torch.cuda.OutOfMemoryError:
            print(f"  OOM on traj {traj_id} step {step_key} -- skipping")
            torch.cuda.empty_cache()
            continue
        except Exception as exc:
            print(f"  Error on traj {traj_id}: {exc!r} -- skipping")
            continue

    print(f"\n  Successfully scored {len(scored_entries)}/{n_total} images")

    if len(scored_entries) < 100:
        print(f"  ERROR: Only {len(scored_entries)} scored (need >= 100)")
        return 1

    print("\n[Phase 6] Computing pairwise agreement and Spearman correlation...")

    btrm_pink = [e["btrm_pinkify"] for e in scored_entries]
    btrm_tnt = [e["btrm_thisnotthat"] for e in scored_entries]
    lit_pink = [e["literal_pinkify"] for e in scored_entries]
    lit_tnt = [e["literal_thisnotthat"] for e in scored_entries]

    agree_pink, ntotal_pink, disagree_pink = compute_pairwise_agreement(btrm_pink, lit_pink)
    agree_tnt, ntotal_tnt, disagree_tnt = compute_pairwise_agreement(btrm_tnt, lit_tnt)

    pink_agree_pct = agree_pink / max(ntotal_pink, 1)
    tnt_agree_pct = agree_tnt / max(ntotal_tnt, 1)

    try:
        from src_ii.stats import spearman_rank_correlation as stats_spearman
        rho_pink = stats_spearman(lit_pink, btrm_pink)
        rho_tnt = stats_spearman(lit_tnt, btrm_tnt)
        spearman_source = "src_ii.stats"
    except ImportError:
        rho_pink = spearman_rank_correlation(lit_pink, btrm_pink)
        rho_tnt = spearman_rank_correlation(lit_tnt, btrm_tnt)
        spearman_source = "inline"

    print(f"\n  === Pairwise Agreement ===")
    print(f"  Pinkify:     {agree_pink}/{ntotal_pink} = {pink_agree_pct:.1%}")
    print(f"  ThisNotThat: {agree_tnt}/{ntotal_tnt} = {tnt_agree_pct:.1%}")
    print(f"\n  === Spearman Rank Correlation (via {spearman_source}) ===")
    print(f"  Pinkify rho:     {rho_pink:.4f}")
    print(f"  ThisNotThat rho: {rho_tnt:.4f}")

    print(f"\n  === Rubric Check ===")
    pink_pass = pink_agree_pct >= 0.70
    tnt_pass = tnt_agree_pct >= 0.60
    n_scored_pass = len(scored_entries) >= 100
    print(f"  [{'PASS' if n_scored_pass else 'FAIL'}] >= 100 images scored: {len(scored_entries)}")
    print(f"  [{'PASS' if pink_pass else 'FAIL'}] Pinkify agreement >= 70%: {pink_agree_pct:.1%}")
    print(f"  [{'PASS' if tnt_pass else 'FAIL'}] ThisNotThat agreement >= 60%: {tnt_agree_pct:.1%}")

    print(f"\n[Phase 7] Writing results to {OUTPUT_PATH}...")

    def _stats(vals: list[float]) -> dict:
        import statistics
        if not vals:
            return {}
        return {
            "min": min(vals),
            "max": max(vals),
            "mean": statistics.mean(vals),
            "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }

    wall_total = time.perf_counter() - wall_start

    output = {
        "metadata": {
            "n_images_scored": len(scored_entries),
            "n_images_attempted": n_total,
            "compound_model_dir": str(COMPOUND_MODEL_DIR),
            "score_cache_path": str(SCORE_CACHE_PATH),
            "v2_dataset_dir": str(V2_DATASET_DIR),
            "step_key_used": "final",
            "head_names": list(HEAD_NAMES),
            "wall_time_s": wall_total,
        },
        "pairwise_agreement": {
            "pinkify": {
                "n_agree": agree_pink,
                "n_total_nontie": ntotal_pink,
                "agreement_pct": pink_agree_pct,
                "pass_70pct": pink_pass,
                "disagreements_sample": disagree_pink,
            },
            "thisnotthat": {
                "n_agree": agree_tnt,
                "n_total_nontie": ntotal_tnt,
                "agreement_pct": tnt_agree_pct,
                "pass_60pct": tnt_pass,
                "disagreements_sample": disagree_tnt,
            },
        },
        "spearman_correlation": {
            "pinkify": rho_pink,
            "thisnotthat": rho_tnt,
            "source": spearman_source,
        },
        "score_distributions": {
            "btrm_pinkify": _stats(btrm_pink),
            "literal_pinkify": _stats(lit_pink),
            "btrm_thisnotthat": _stats(btrm_tnt),
            "literal_thisnotthat": _stats(lit_tnt),
        },
        "per_image": scored_entries,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(str(OUTPUT_PATH), "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Written: {OUTPUT_PATH}")
    print(f"\n  Wall time: {wall_total:.1f}s ({wall_total / 60:.1f} min)")

    reader.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

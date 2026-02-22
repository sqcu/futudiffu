r"""Verify bit-for-bit persistence round-trip for PINKIFY/THISNOTTHAT BTRM compound model.

Loads the trained compound model from pinkify_thisnotthat_output/differentiable_run/,
re-runs forward passes for each test entry in pre_persist_scores.json, and verifies
that the reloaded model produces bit-for-bit identical outputs to the pre-persist scores.

Also compares raw weight tensors from the saved safetensors files against the loaded
model parameters (zero max_diff required for both adapter and head weights).

Execution:
  PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe \
      F:\dox\repos\ai\futudiffu\scripts_ii\verify_pinkify_persistence.py

Output:
  pinkify_thisnotthat_output/differentiable_run/persistence_verification.json
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src_ii"))

import torch
from safetensors.torch import load_file

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FP8_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
TE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"
TOKENIZER_PATH = str(REPO_ROOT / "src" / "futudiffu" / "tokenizer")

V2_DATASET_DIR = REPO_ROOT / "btrm_dataset_v2"
RUN_DIR = REPO_ROOT / "pinkify_thisnotthat_output" / "differentiable_run"
PRE_PERSIST_SCORES_PATH = RUN_DIR / "pre_persist_scores.json"
OUTPUT_PATH = RUN_DIR / "persistence_verification.json"


def main():
    t0 = time.perf_counter()
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("=" * 60)
    print("  PINKIFY/THISNOTTHAT Persistence Verification")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Phase 1: Load pre-persist scores (ground truth)
    # -----------------------------------------------------------------------
    print("\n[Phase 1] Loading pre-persist scores...")
    with open(str(PRE_PERSIST_SCORES_PATH)) as f:
        pre_persist_scores = json.load(f)
    print(f"  {len(pre_persist_scores)} test entries loaded")
    for e in pre_persist_scores[:3]:
        print(f"    traj {e['traj_id']:3d}/{e['step_key']}: "
              f"pinkify={e['score_pinkify']:.6f}, "
              f"thisnotthat={e['score_thisnotthat']:.6f}")

    # -----------------------------------------------------------------------
    # Phase 2: Load compound config
    # -----------------------------------------------------------------------
    print("\n[Phase 2] Loading compound config...")
    config_path = RUN_DIR / "btrm_compound_config.json"
    with open(str(config_path)) as f:
        config = json.load(f)
    head_names = tuple(config["head_names"])
    print(f"  Config: {config}")

    # -----------------------------------------------------------------------
    # Phase 3: Encode prompts (text encoder)
    # -----------------------------------------------------------------------
    print("\n[Phase 3] Loading V2 dataset and encoding prompts...")
    from futudiffu.dataset_v2 import DatasetReader
    from futudiffu.text_encoder import create_tokenizer, load_text_encoder, encode_prompt

    reader = DatasetReader(str(V2_DATASET_DIR))
    traj_ids_all = reader._table.column("traj_id").to_pylist()
    print(f"  V2 dataset: {len(reader)} trajectories")

    # We only need prompts for the 10 test trajectories
    test_traj_ids = [e["traj_id"] for e in pre_persist_scores]

    tokenizer = create_tokenizer(TOKENIZER_PATH)
    te_model = load_text_encoder(TE_PATH, device=device, dtype=dtype)

    prompt_cache = {}
    for traj_id in test_traj_ids:
        if traj_id in traj_ids_all:
            meta, _ = reader[traj_id]
            prompt = meta.get("prompt", "")
            if prompt and prompt not in prompt_cache:
                cond = encode_prompt(te_model, tokenizer, prompt, device=device)
                prompt_cache[prompt] = cond.cpu()

    del te_model, tokenizer
    torch.cuda.empty_cache()
    print(f"  Encoded {len(prompt_cache)} unique prompts. "
          f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # -----------------------------------------------------------------------
    # Phase 4: Load FP8 backbone
    # -----------------------------------------------------------------------
    print("\n[Phase 4] Loading FP8 backbone (no compile for verification)...")
    from src_ii.zimage_model import load_zimage_rlaif
    from src_ii.btrm_lifecycle import load_btrm, score_serial
    from src_ii.multi_lora import install_multi_lora, get_adapter_params

    _, raw_model = load_zimage_rlaif(
        FP8_PATH, device=device, dtype=dtype,
        compile_model=False, fuse=True,
    )
    print(f"  VRAM after backbone: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # -----------------------------------------------------------------------
    # Phase 5: Load compound model from persisted weights
    # -----------------------------------------------------------------------
    print("\n[Phase 5] Loading BTRM model from persisted weights...")

    install_multi_lora(raw_model, [{"name": "rtheta", "rank": 8, "alpha": 16.0}])
    load_btrm(raw_model, "rtheta", str(RUN_DIR))
    raw_model.gradient_checkpointing = False
    raw_model.eval()
    print(f"  Compound model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # -----------------------------------------------------------------------
    # Phase 6: Compare weight tensors (raw safetensors vs model parameters)
    # -----------------------------------------------------------------------
    print("\n[Phase 6] Weight tensor comparison...")
    adapter_path = RUN_DIR / "rtheta_adapter.safetensors"
    head_path = RUN_DIR / "btrm_head.safetensors"

    saved_adapter_sd = load_file(str(adapter_path))
    saved_head_sd = load_file(str(head_path))

    loaded_adapter_sd = {k: v.data for k, v in get_adapter_params(raw_model, config["adapter_name"]).items()}
    loaded_head_sd = {}
    for name, param in raw_model.named_parameters():
        if "score_proj" in name or "score_norm" in name:
            loaded_head_sd[name] = param.data

    weight_results = {}
    all_weights_exact = True

    print("  Adapter weights:")
    for key in sorted(saved_adapter_sd.keys()):
        saved_t = saved_adapter_sd[key].float()
        if key in loaded_adapter_sd:
            loaded_t = loaded_adapter_sd[key].cpu().float()
            exact = torch.equal(saved_t, loaded_t)
            max_diff = (saved_t - loaded_t).abs().max().item()
        else:
            exact = False
            max_diff = float("nan")
            print(f"    WARNING: key '{key}' not found in loaded adapter!")

        weight_results[f"adapter/{key}"] = {
            "exact": exact,
            "max_diff": max_diff,
            "shape": list(saved_adapter_sd[key].shape),
            "dtype": str(saved_adapter_sd[key].dtype),
        }
        if not exact:
            all_weights_exact = False
        status = "OK" if exact else "MISMATCH"
        print(f"    [{status}] {key}: shape={list(saved_adapter_sd[key].shape)}, "
              f"max_diff={max_diff:.2e}")

    print("  Head weights:")
    for key in sorted(saved_head_sd.keys()):
        saved_t = saved_head_sd[key].float()
        if key in loaded_head_sd:
            loaded_t = loaded_head_sd[key].cpu().float()
            exact = torch.equal(saved_t, loaded_t)
            max_diff = (saved_t - loaded_t).abs().max().item()
        else:
            exact = False
            max_diff = float("nan")
            print(f"    WARNING: key '{key}' not found in loaded head!")

        weight_results[f"head/{key}"] = {
            "exact": exact,
            "max_diff": max_diff,
            "shape": list(saved_head_sd[key].shape),
            "dtype": str(saved_head_sd[key].dtype),
        }
        if not exact:
            all_weights_exact = False
        status = "OK" if exact else "MISMATCH"
        print(f"    [{status}] {key}: shape={list(saved_head_sd[key].shape)}, "
              f"max_diff={max_diff:.2e}")

    n_weight_tensors = len(weight_results)
    n_exact_weights = sum(1 for v in weight_results.values() if v["exact"])
    print(f"  Weight summary: {n_exact_weights}/{n_weight_tensors} tensors exact")

    # -----------------------------------------------------------------------
    # Phase 7: Re-run forward passes for each test entry
    # -----------------------------------------------------------------------
    print("\n[Phase 7] Re-running forward passes for test entries...")
    from src_ii.sigma_schedule import build_sigma_schedule, resolution_shift

    score_results = []
    all_scores_exact = True

    _meta_cache = {}

    def get_meta(traj_id):
        if traj_id not in _meta_cache:
            meta, accessor = reader[traj_id]
            _meta_cache[traj_id] = (meta, accessor)
        return _meta_cache[traj_id]

    for entry in pre_persist_scores:
        traj_id = entry["traj_id"]
        step_key = entry["step_key"]
        pre_pinkify = entry["score_pinkify"]
        pre_thisnotthat = entry["score_thisnotthat"]

        meta, accessor = get_meta(traj_id)

        # Load latent
        latent = accessor[step_key].to(device=device, dtype=dtype)

        # Compute sigma for this step (same logic as train_pinkify_differentiable.py)
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
            step_idx = int(step_key.split("_")[1])
            sigma_val = float(sigmas[step_idx].item()) if step_idx < len(sigmas) else 0.01

        timestep = torch.tensor([sigma_val], device=device, dtype=dtype)

        # Get conditioning
        prompt = meta.get("prompt", "")
        cond_cpu = prompt_cache.get(prompt)
        if cond_cpu is None:
            print(f"  ERROR: No cached prompt for traj {traj_id}!")
            continue
        cond = cond_cpu.to(device=device, dtype=dtype)

        num_tokens = cond.shape[1]

        # Score via loaded model (inference path, same as training script)
        with torch.no_grad():
            scores = score_serial(raw_model, latent, timestep, cond, num_tokens,
                                  gradient_checkpointing=False)

        post_pinkify = float(scores[0, 0].item())
        post_thisnotthat = float(scores[0, 1].item())

        # Bit-for-bit comparison via float32 bit patterns
        pre_pinkify_t = torch.tensor(pre_pinkify, dtype=torch.float32)
        pre_thisnotthat_t = torch.tensor(pre_thisnotthat, dtype=torch.float32)
        post_pinkify_t = torch.tensor(post_pinkify, dtype=torch.float32)
        post_thisnotthat_t = torch.tensor(post_thisnotthat, dtype=torch.float32)

        exact_pinkify = torch.equal(pre_pinkify_t, post_pinkify_t)
        exact_thisnotthat = torch.equal(pre_thisnotthat_t, post_thisnotthat_t)
        diff_pinkify = abs(pre_pinkify - post_pinkify)
        diff_thisnotthat = abs(pre_thisnotthat - post_thisnotthat)

        entry_exact = exact_pinkify and exact_thisnotthat
        if not entry_exact:
            all_scores_exact = False

        status = "OK" if entry_exact else "MISMATCH"
        print(f"  [{status}] traj {traj_id:3d}/{step_key}:")
        print(f"    pinkify:     pre={pre_pinkify:.8f} post={post_pinkify:.8f} "
              f"diff={diff_pinkify:.2e} exact={exact_pinkify}")
        print(f"    thisnotthat: pre={pre_thisnotthat:.8f} post={post_thisnotthat:.8f} "
              f"diff={diff_thisnotthat:.2e} exact={exact_thisnotthat}")

        result = {
            "traj_id": traj_id,
            "step_key": step_key,
            "pinkify": {
                "pre_persist": pre_pinkify,
                "post_persist": post_pinkify,
                "diff": diff_pinkify,
                "exact": exact_pinkify,
            },
            "thisnotthat": {
                "pre_persist": pre_thisnotthat,
                "post_persist": post_thisnotthat,
                "diff": diff_thisnotthat,
                "exact": exact_thisnotthat,
            },
            "entry_exact": entry_exact,
        }
        score_results.append(result)

        del latent, cond, rope_cache, scores, timestep
        torch.cuda.empty_cache()

    n_exact_scores = sum(1 for r in score_results if r["entry_exact"])
    print(f"\n  Score summary: {n_exact_scores}/{len(score_results)} entries exact")

    # -----------------------------------------------------------------------
    # Phase 8: Verdict + persist results
    # -----------------------------------------------------------------------
    elapsed = time.perf_counter() - t0
    verdict = "PASS" if (all_weights_exact and all_scores_exact) else "FAIL"

    verification = {
        "verdict": verdict,
        "all_weights_exact": all_weights_exact,
        "all_scores_exact": all_scores_exact,
        "n_test_entries": len(pre_persist_scores),
        "n_exact_scores": n_exact_scores,
        "n_weight_tensors": n_weight_tensors,
        "n_exact_weights": n_exact_weights,
        "score_results": score_results,
        "weight_results": weight_results,
        "elapsed_s": elapsed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(RUN_DIR),
        "config": config,
    }

    with open(str(OUTPUT_PATH), "w") as f:
        json.dump(verification, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  VERDICT: {verdict}")
    print(f"  Weights: {n_exact_weights}/{n_weight_tensors} tensors bit-for-bit exact")
    print(f"  Scores:  {n_exact_scores}/{len(score_results)} entries bit-for-bit exact")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Output:  {OUTPUT_PATH}")
    print("=" * 60)

    reader.close()

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

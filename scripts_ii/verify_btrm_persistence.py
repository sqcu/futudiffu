r"""Verify BTRM compound model persistence: load saved head+adapter and compare.

Loads the persisted compound model (adapter + head), scores sample hidden
states, and compares to the pre-persist scores saved during training.

Uses ScoreUnembedder.forward() for scoring instead of manual norm+proj+cap inlining.

Execution:
  PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe \
      F:\dox\repos\ai\futudiffu\scripts_ii\verify_btrm_persistence.py

Output:
  pinkify_thisnotthat_output/persistence_verification.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from safetensors.torch import load_file

from futudiffu.btrm import ScoreUnembedder

OUTPUT_DIR = REPO_ROOT / "pinkify_thisnotthat_output"
HEAD_NAMES = ("pinkify", "thisnotthat")


def main():
    t0 = time.perf_counter()
    device = torch.device("cuda")

    # --- Load pre-persist scores ---
    print("Loading pre-persist scores...")
    pre_persist_scores = torch.load(str(OUTPUT_DIR / "pre_persist_scores.pt"), weights_only=True)
    print(f"  Shape: {pre_persist_scores.shape}")
    print(f"  dtype: {pre_persist_scores.dtype}")

    # --- Load head config ---
    # Try compound config first, fall back to legacy head_config
    config_path = OUTPUT_DIR / "btrm_compound_config.json"
    if not config_path.exists():
        config_path = OUTPUT_DIR / "head_config.json"
    with open(config_path) as f:
        head_config = json.load(f)
    print(f"  Config: {head_config}")

    # --- Load persisted head ---
    print("\nLoading persisted head...")
    head = ScoreUnembedder(
        hidden_dim=head_config["hidden_dim"],
        head_names=tuple(head_config["head_names"]),
        logit_cap=head_config["logit_cap"],
    )

    # Try compound head path first, fall back to legacy
    head_path = OUTPUT_DIR / "btrm_head.safetensors"
    if not head_path.exists():
        head_path = OUTPUT_DIR / "trained_head.safetensors"
    state_dict = load_file(str(head_path))
    head.load_state_dict(state_dict, assign=True)
    head = head.to(device=device, dtype=torch.float32)
    head.eval()
    print(f"  Head loaded from {head_path}")

    # --- Score sample hidden states using head.forward() (not manual inline) ---
    print("\nScoring sample hidden states...")
    sample_0 = torch.load(str(OUTPUT_DIR / "hidden_sample_0.pt"), weights_only=True)
    sample_last = torch.load(str(OUTPUT_DIR / "hidden_sample_last.pt"), weights_only=True)

    with torch.no_grad():
        h0 = sample_0.to(device=device, dtype=torch.float32)
        score_0 = head(h0)  # head.forward handles mean-pool + norm + proj + cap

        h_last = sample_last.to(device=device, dtype=torch.float32)
        score_last = head(h_last)

    pre_score_0 = pre_persist_scores[0].to(device=device)
    pre_score_last = pre_persist_scores[-1].to(device=device)

    post_score_0 = score_0.squeeze(0)
    post_score_last = score_last.squeeze(0)

    exact_0 = torch.equal(pre_score_0, post_score_0)
    exact_last = torch.equal(pre_score_last, post_score_last)
    diff_0 = (pre_score_0 - post_score_0).abs()
    diff_last = (pre_score_last - post_score_last).abs()

    print(f"\n  Sample 0:")
    print(f"    Pre-persist:  {pre_score_0.tolist()}")
    print(f"    Post-persist: {post_score_0.tolist()}")
    print(f"    Diff:         {diff_0.tolist()}")
    print(f"    Bit-for-bit:  {exact_0}")

    print(f"\n  Sample last:")
    print(f"    Pre-persist:  {pre_score_last.tolist()}")
    print(f"    Post-persist: {post_score_last.tolist()}")
    print(f"    Diff:         {diff_last.tolist()}")
    print(f"    Bit-for-bit:  {exact_last}")

    # Verify weight equality
    print("\n  Checking weight equality...")
    pre_weights = load_file(str(head_path))
    loaded_weights = {k: v for k, v in head.state_dict().items()}

    weight_comparison = {}
    all_weights_exact = True
    for key in pre_weights:
        if key in loaded_weights:
            pre_w = pre_weights[key]
            post_w = loaded_weights[key].cpu()
            exact = torch.equal(pre_w, post_w)
            max_diff = (pre_w.float() - post_w.float()).abs().max().item()
            weight_comparison[key] = {
                "exact": exact, "max_diff": max_diff,
                "shape": list(pre_w.shape), "dtype": str(pre_w.dtype),
            }
            if not exact:
                all_weights_exact = False
            print(f"    {key}: exact={exact}, max_diff={max_diff:.2e}")

    # Check adapter persistence if available
    adapter_path = OUTPUT_DIR / "rtheta_adapter.safetensors"
    adapter_exists = adapter_path.exists()
    if adapter_exists:
        adapter_sd = load_file(str(adapter_path))
        n_adapter_params = sum(v.numel() for v in adapter_sd.values())
        any_nonzero = any(v.abs().max().item() > 0 for v in adapter_sd.values())
        print(f"\n  Adapter check: {n_adapter_params} params, any_nonzero={any_nonzero}")
    else:
        print(f"\n  No adapter file found at {adapter_path}")

    # --- Write verification results ---
    verification = {
        "sample_0": {
            "pre_persist": pre_score_0.cpu().tolist(),
            "post_persist": post_score_0.cpu().tolist(),
            "diff": diff_0.cpu().tolist(),
            "bit_for_bit": exact_0,
        },
        "sample_last": {
            "pre_persist": pre_score_last.cpu().tolist(),
            "post_persist": post_score_last.cpu().tolist(),
            "diff": diff_last.cpu().tolist(),
            "bit_for_bit": exact_last,
        },
        "weight_comparison": weight_comparison,
        "all_weights_exact": all_weights_exact,
        "all_scores_exact": exact_0 and exact_last,
        "adapter_persisted": adapter_exists,
        "verdict": "PASS" if (exact_0 and exact_last and all_weights_exact) else "FAIL",
    }

    out_path = OUTPUT_DIR / "persistence_verification.json"
    with open(out_path, "w") as f:
        json.dump(verification, f, indent=2)

    elapsed = time.perf_counter() - t0
    print(f"\n=== Verification complete in {elapsed:.1f}s ===")
    print(f"  Verdict: {verification['verdict']}")
    print(f"  Results: {out_path}")


if __name__ == "__main__":
    main()

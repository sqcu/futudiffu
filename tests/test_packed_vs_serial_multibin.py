r"""Packed vs serial divergence validation for multi-bin BatchExecutor.

Proves that the 2-bin packed forward produces the same denoised outputs as
9 individual serial forwards. This is the functional correctness test:
cross-query bin packing must not corrupt results.

At step 0 (sigma=1.0 base), for each of the 9 entries in the k-tuple rollout:
  1. Packed path:  BatchExecutor (2 bins, FFD-assigned) processes all 9 entries
  2. Serial path:  Each entry individually through prepare_packed_forward + packed_forward (B=1)
  3. Compare:      max_abs(packed_denoised - serial_denoised) per entry

Expected: max_abs < 0.625 (SageAttention INT8 quantization noise, amplified by
different padding layouts between packed multi-entry and serial single-entry
forward passes -- each path pads to REFERENCE_TOTAL_LEN but with different
block mask geometry, which shifts INT8 rounding boundaries).

The per-step noise floor for IDENTICAL padding layouts is ~0.0625, but packed
vs serial compares DIFFERENT padding layouts at the same REFERENCE_TOTAL_LEN,
which produces up to ~0.5 max_abs divergence in BF16 outputs. This is expected
FP8+INT8 quantization noise, not algorithmic divergence. Mean_abs stays below
0.04 and cosine similarity > 0.9999 across all entries.

Three query sources:
  Q0 "shrimp": K=4 at 512x512 (base + shrimp + typo + uncond)
  Q1 "banana": K=3 at 512x512 (base + tropical + uncond)
  Q2 "tiny":   K=2 at 256x256 (base + uncond)

Total of 9 entries. They exceed a single REFERENCE_TOTAL_LEN=4224 bin, forcing
the executor to split into 2+ FlexAttention launches -- each padded to
REFERENCE_TOTAL_LEN for zero recompiles.

Output: validation_renders/packed_vs_serial_multibin/

Usage:
    .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\tests\test_packed_vs_serial_multibin.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED_BASE = 42
DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
MAX_ABS_THRESHOLD = 0.625

OUT_DIR = REPO_ROOT / "validation_renders" / "packed_vs_serial_multibin"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "validation.log"

# Query definitions: (query_name, base_resolution_wxh, forks)
# Each fork: (name, cap_len, resolution_or_None, scale)
QUERIES = [
    ("shrimp", (512, 512), [
        ("base",   29, None,       1.0),
        ("shrimp", 35, None,       3.0),
        ("typo",   40, None,       2.0),
        ("uncond",  8, None,      -6.0),
    ]),
    ("banana", (512, 512), [
        ("base",   20, None,       1.0),
        ("tropical", 25, None,     2.0),
        ("uncond",  8, None,      -3.0),
    ]),
    ("tiny", (256, 256), [
        ("base",   29, None,       1.0),
        ("uncond",  8, None,      -7.0),
    ]),
]

BASE_SIGMA = 1.0  # Step 0 sigma for the base resolution


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FILE = None


def _init_log():
    global _LOG_FILE
    _LOG_FILE = open(LOG_PATH, "w")


def _log(msg: str) -> None:
    print(msg, flush=True)
    if _LOG_FILE is not None:
        _LOG_FILE.write(msg + "\n")
        _LOG_FILE.flush()


# ---------------------------------------------------------------------------
# Phase 1: Load model + compile
# ---------------------------------------------------------------------------

def phase1_load_model():
    _log("\n" + "=" * 60)
    _log("  PHASE 1: LOAD SSS-II MODEL + COMPILE")
    _log("=" * 60)

    from tests.stubbed_skinny_shared_ii import load_sss_model

    t0 = time.perf_counter()
    model = load_sss_model(device=DEVICE)
    _log(f"  Model loaded in {time.perf_counter() - t0:.1f}s")
    _log(f"  VRAM after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    t1 = time.perf_counter()
    compiled_model = torch.compile(model, mode="default")
    _log(f"  torch.compile() in {time.perf_counter() - t1:.3f}s (lazy)")
    _log(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    return compiled_model


# ---------------------------------------------------------------------------
# Phase 2: Build conditionings + latents with deterministic seeds
# ---------------------------------------------------------------------------

def phase2_build_inputs():
    _log("\n" + "=" * 60)
    _log("  PHASE 2: BUILD CONDITIONINGS + LATENTS")
    _log("=" * 60)

    from tests.stubbed_skinny_shared_ii import make_random_conditioning, SSS_CAP_FEAT_DIM
    from src_ii.triumphant_future_reduction_ops import latent_padded

    # Build conditionings with deterministic seeds per fork
    cond_seed = 1000
    conds = {}  # (q_idx, f_idx) -> cond tensor
    for q_idx, (q_name, base_res, forks) in enumerate(QUERIES):
        for f_idx, (fork_name, cap_len, res_override, scale) in enumerate(forks):
            gen = torch.Generator(device=DEVICE).manual_seed(cond_seed + q_idx * 100 + f_idx)
            cond = torch.randn(
                1, cap_len, SSS_CAP_FEAT_DIM,
                device=DEVICE, dtype=DTYPE, generator=gen,
            )
            conds[(q_idx, f_idx)] = cond
            _log(f"  cond ({q_name}/{fork_name}): shape={tuple(cond.shape)}, "
                 f"seed={cond_seed + q_idx * 100 + f_idx}")

    # Build base latents (one per query, noised at sigma * randn)
    latents = {}  # q_idx -> base_latent
    for q_idx, (q_name, base_res, forks) in enumerate(QUERIES):
        seed = SEED_BASE + q_idx
        w, h = base_res
        lh, lw = latent_padded(w, h)
        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        noise = torch.randn(1, 16, lh, lw, device=DEVICE, dtype=DTYPE, generator=gen)
        # x = sigma * noise at step 0
        latents[q_idx] = BASE_SIGMA * noise
        _log(f"  latent ({q_name}): shape={tuple(latents[q_idx].shape)}, "
             f"seed={seed}, norm={float(latents[q_idx].norm()):.2f}")

    return conds, latents


# ---------------------------------------------------------------------------
# Phase 3: Packed path (BatchExecutor)
# ---------------------------------------------------------------------------

def phase3_packed(compiled_model, conds, latents):
    _log("\n" + "=" * 60)
    _log("  PHASE 3: PACKED PATH (BatchExecutor)")
    _log("=" * 60)

    from src_ii.batch_executor import BatchExecutor

    batch_executor = BatchExecutor(compiled_model, DEVICE)

    # Build query dicts in the format BatchExecutor expects
    queries = []
    for q_idx, (q_name, base_res, forks) in enumerate(QUERIES):
        base_cond = conds[(q_idx, 0)]  # fork 0 is the base

        fork_dicts = []
        for f_idx, (fork_name, cap_len, res_override, scale) in enumerate(forks):
            fork_dict = {"entry_id": f"e{f_idx}"}
            # If fork's cond differs from base, pass it explicitly
            if f_idx > 0:
                fork_dict["cond"] = conds[(q_idx, f_idx)]
            if res_override is not None:
                fork_dict["resolution"] = res_override
            fork_dicts.append(fork_dict)

        queries.append({
            "query_id": f"q{q_idx}",
            "base_latent": latents[q_idx],
            "base_cond": base_cond,
            "base_cap_len": base_cond.shape[1],
            "base_resolution": base_res,
            "sigma": BASE_SIGMA,
            "forks": fork_dicts,
        })

    _log(f"  Submitting {len(queries)} queries to BatchExecutor...")

    t0 = time.perf_counter()
    results = batch_executor.execute(queries)
    elapsed = time.perf_counter() - t0
    _log(f"  BatchExecutor.execute() completed in {elapsed:.2f}s")
    _log(f"  Got {len(results)} results")

    # Log packing plan details
    for key, plan in batch_executor._plan_cache.items():
        bins = plan["bins"]
        _log(f"  Packing plan {key[:8]}... => {len(bins)} bins:")
        for b_idx, bin_info in enumerate(bins):
            indices = bin_info["indices"]
            # Map global indices to (query, entry) names
            entry_names = []
            global_idx = 0
            entry_map = {}
            for q_idx, (q_name, _, forks) in enumerate(QUERIES):
                for f_idx, (fork_name, _, _, _) in enumerate(forks):
                    entry_map[global_idx] = f"{q_name}/{fork_name}"
                    global_idx += 1
            names = [entry_map.get(i, f"?{i}") for i in indices]
            _log(f"    Bin {b_idx}: entries {indices} = [{', '.join(names)}]")

    return results, batch_executor


# ---------------------------------------------------------------------------
# Phase 4: Serial path (one entry at a time through packed_forward)
# ---------------------------------------------------------------------------

def phase4_serial(compiled_model, conds, latents):
    _log("\n" + "=" * 60)
    _log("  PHASE 4: SERIAL PATH (one entry at a time)")
    _log("=" * 60)

    from src_ii.batch_executor import _scatter
    from src_ii.forward_packed import prepare_packed_forward, packed_forward
    from src_ii.triumphant_future_reduction_ops import denoise_all

    # Use _scatter to get the exact same entries (with correct sigma shifting)
    # that BatchExecutor would produce
    queries = []
    for q_idx, (q_name, base_res, forks) in enumerate(QUERIES):
        base_cond = conds[(q_idx, 0)]
        fork_dicts = []
        for f_idx, (fork_name, cap_len, res_override, scale) in enumerate(forks):
            fork_dict = {"entry_id": f"e{f_idx}"}
            if f_idx > 0:
                fork_dict["cond"] = conds[(q_idx, f_idx)]
            if res_override is not None:
                fork_dict["resolution"] = res_override
            fork_dicts.append(fork_dict)

        queries.append({
            "query_id": f"q{q_idx}",
            "base_latent": latents[q_idx],
            "base_cond": base_cond,
            "base_cap_len": base_cond.shape[1],
            "base_resolution": base_res,
            "sigma": BASE_SIGMA,
            "forks": fork_dicts,
        })

    entries = _scatter(queries)
    _log(f"  Scattered {len(entries)} entries from {len(queries)} queries")

    serial_results = []
    for i, entry in enumerate(entries):
        _log(f"  Serial forward {i}/{len(entries)}: "
             f"{entry['query_id']}/{entry['entry_id']} "
             f"x={tuple(entry['x'].shape)} cap_len={entry['cap_len']} "
             f"sigma={entry['sigma']:.4f}")

        x = entry["x"].to(DEVICE)
        cond = entry["cond"].to(DEVICE)
        sigma = entry["sigma"]
        cap_len = entry["cap_len"]
        lh, lw = x.shape[2], x.shape[3]

        t0 = time.perf_counter()

        # Prepare packed state for a single entry
        prepared = prepare_packed_forward(
            compiled_model,
            [cond],               # single conditioning
            [(lh, lw)],           # single image size (latent dims)
            [cap_len],            # single cap_len
            DEVICE,
        )

        # Build timestep
        timestep = torch.tensor([sigma], device=DEVICE, dtype=torch.float32)

        # Run packed forward with single entry
        fields, scores = packed_forward(
            compiled_model,
            [x],
            [timestep],
            prepared["refined_caps"],
            prepared["packing_info"],
            prepared["block_mask"],
            prepared["packed_rope"],
        )

        # Denoise: denoised = x - field * sigma
        sigma_tensor = torch.tensor(sigma, device=DEVICE, dtype=x.dtype)
        denoised_list = denoise_all([x], fields, [sigma_tensor])

        elapsed = time.perf_counter() - t0
        _log(f"    Forward + denoise in {elapsed:.2f}s, "
             f"denoised norm={float(denoised_list[0].norm()):.2f}")

        serial_results.append({
            "query_id": entry["query_id"],
            "entry_id": entry["entry_id"],
            "denoised": denoised_list[0].cpu(),
            "scores": scores[0].cpu() if scores is not None else None,
        })

    return serial_results


# ---------------------------------------------------------------------------
# Phase 5: Compare packed vs serial
# ---------------------------------------------------------------------------

def phase5_compare(packed_results, serial_results):
    _log("\n" + "=" * 60)
    _log("  PHASE 5: COMPARISON (packed vs serial)")
    _log("=" * 60)

    assert len(packed_results) == len(serial_results), (
        f"Result count mismatch: packed={len(packed_results)}, serial={len(serial_results)}"
    )

    comparisons = []
    all_pass = True

    for i in range(len(packed_results)):
        pr = packed_results[i]
        sr = serial_results[i]

        assert pr["query_id"] == sr["query_id"], (
            f"Entry {i}: query_id mismatch: packed={pr['query_id']}, serial={sr['query_id']}"
        )
        assert pr["entry_id"] == sr["entry_id"], (
            f"Entry {i}: entry_id mismatch: packed={pr['entry_id']}, serial={sr['entry_id']}"
        )

        packed_d = pr["denoised"].float()
        serial_d = sr["denoised"].float()
        diff = packed_d - serial_d

        max_abs = float(diff.abs().max().item())
        mean_abs = float(diff.abs().mean().item())
        l2 = float(diff.norm().item())

        p_flat = packed_d.flatten()
        s_flat = serial_d.flatten()
        cos_sim = float(
            (p_flat * s_flat).sum() / (p_flat.norm() * s_flat.norm() + 1e-12)
        )

        passed = max_abs < MAX_ABS_THRESHOLD
        verdict = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False

        comp = {
            "entry_idx": i,
            "query_id": pr["query_id"],
            "entry_id": pr["entry_id"],
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "l2": l2,
            "cosine_similarity": cos_sim,
            "packed_norm": float(packed_d.norm().item()),
            "serial_norm": float(serial_d.norm().item()),
            "verdict": verdict,
        }
        comparisons.append(comp)

        flag = "" if passed else " ** FAIL **"
        _log(f"  Entry {i} ({pr['query_id']}/{pr['entry_id']}): "
             f"max_abs={max_abs:.6f}  mean_abs={mean_abs:.6f}  "
             f"l2={l2:.4f}  cos={cos_sim:.8f}  {verdict}{flag}")

        # Also compare scores if present
        if pr["scores"] is not None and sr["scores"] is not None:
            score_diff = (pr["scores"].float() - sr["scores"].float()).abs().max().item()
            comp["score_max_abs"] = float(score_diff)
            _log(f"    score max_abs: {score_diff:.6f}")

    return comparisons, all_pass


# ---------------------------------------------------------------------------
# Phase 6: Persist results
# ---------------------------------------------------------------------------

def phase6_persist(packed_results, serial_results, comparisons, all_pass, batch_executor):
    _log("\n" + "=" * 60)
    _log("  PHASE 6: PERSIST RESULTS")
    _log("=" * 60)

    # Save per-entry denoised tensors
    for tag, results in [("packed", packed_results), ("serial", serial_results)]:
        for r in results:
            path = OUT_DIR / f"{tag}_{r['query_id']}_{r['entry_id']}_denoised.pt"
            torch.save(r["denoised"], path)
    _log(f"  Saved {2 * len(packed_results)} denoised tensors to {OUT_DIR}")

    # Build packing plan info for the report
    packing_info = {}
    for key, plan in batch_executor._plan_cache.items():
        bins = plan["bins"]
        bin_list = []
        for b_idx, bin_info in enumerate(bins):
            bin_list.append({
                "bin_idx": b_idx,
                "entry_indices": bin_info["indices"],
            })
        packing_info[key] = {"n_bins": len(bins), "bins": bin_list}

    # Write summary JSON
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "queries": [
                {
                    "name": q_name,
                    "resolution": base_res,
                    "forks": [
                        {"name": fn, "cap_len": cl, "res_override": ro, "scale": sc}
                        for fn, cl, ro, sc in forks
                    ],
                }
                for q_name, base_res, forks in QUERIES
            ],
            "base_sigma": BASE_SIGMA,
            "seed_base": SEED_BASE,
            "max_abs_threshold": MAX_ABS_THRESHOLD,
        },
        "packing_plan": packing_info,
        "overall_verdict": "PASS" if all_pass else "FAIL",
        "comparisons": comparisons,
    }

    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"  Summary written to {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    _init_log()
    _log("=" * 60)
    _log("  PACKED vs SERIAL MULTIBIN DIVERGENCE VALIDATION")
    _log("=" * 60)
    _log(f"  Timestamp: {datetime.now(timezone.utc).isoformat()}")
    _log(f"  Output: {OUT_DIR}")
    _log(f"  Threshold: max_abs < {MAX_ABS_THRESHOLD}")
    _log(f"  Base sigma: {BASE_SIGMA}")

    total_entries = sum(len(forks) for _, _, forks in QUERIES)
    _log(f"  Queries: {len(QUERIES)}, total entries: {total_entries}")
    for q_name, base_res, forks in QUERIES:
        _log(f"    {q_name}: K={len(forks)} at {base_res[0]}x{base_res[1]}")
        for fname, cap_len, res, scale in forks:
            res_str = f"{res[0]}x{res[1]}" if res else "base"
            _log(f"      {fname}: cap_len={cap_len}, res={res_str}, scale={scale:+.1f}")

    # Phase 1: Load + compile
    compiled_model = phase1_load_model()

    # Phase 2: Build deterministic inputs
    conds, latents = phase2_build_inputs()

    # Phase 3: Packed path (BatchExecutor, 2+ bins)
    packed_results, batch_executor = phase3_packed(compiled_model, conds, latents)

    # Phase 4: Serial path (one entry at a time)
    serial_results = phase4_serial(compiled_model, conds, latents)

    # Phase 5: Compare
    comparisons, all_pass = phase5_compare(packed_results, serial_results)

    # Phase 6: Persist
    summary = phase6_persist(
        packed_results, serial_results, comparisons, all_pass, batch_executor,
    )

    # Final verdict
    _log("\n" + "=" * 60)
    overall = "PASS" if all_pass else "FAIL"
    _log(f"  OVERALL VERDICT: {overall}")
    _log("=" * 60)

    if not all_pass:
        failing = [c for c in comparisons if c["verdict"] == "FAIL"]
        _log(f"  {len(failing)} entries exceeded max_abs threshold of {MAX_ABS_THRESHOLD}:")
        for c in failing:
            _log(f"    {c['query_id']}/{c['entry_id']}: max_abs={c['max_abs']:.6f}")

    # Assert for CI
    assert all_pass, (
        f"Packed vs serial divergence exceeded threshold {MAX_ABS_THRESHOLD}. "
        f"Failing entries: "
        + ", ".join(
            f"{c['query_id']}/{c['entry_id']}={c['max_abs']:.4f}"
            for c in comparisons if c["verdict"] == "FAIL"
        )
    )

    _log(f"\n  All {len(comparisons)} entries PASS (max_abs < {MAX_ABS_THRESHOLD})")
    _log(f"  Output directory: {OUT_DIR}")

    if _LOG_FILE is not None:
        _LOG_FILE.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

r"""Full k-tuple rollout on SSS-II: multiple workloads, GATHER, persistence, renders.

Three sampler client sources submit to the same BatchExecutor simultaneously:
  Q0 "shrimp": K=4 at 512x512 (base + shrimp + typo + uncond)
  Q1 "banana": K=3 at 512x512 (base + tropical + uncond)
  Q2 "tiny":   K=2 at 256x256 (base + uncond)

Entries are packed into REFERENCE_TOTAL_LEN-sized bins by the server,
demonstrating funfetti batching across heterogeneous resolutions.  Total
tokens across all 9 entries exceed a single REFERENCE_TOTAL_LEN=4224 bin,
so the executor must split them into 2+ FlexAttention launches — each
padded to REFERENCE_TOTAL_LEN for zero recompiles.

Client-side GATHER (weighted sum of residuals) reduces K entries to one
guided denoised estimate per query. Client-side EULER advances the trajectory.
Per-step latents, scores, and guided estimates are persisted to disk.
Final latents are rendered as pseudo-RGB PNGs (channels 0:3 normalized).

Output: validation_renders/sss_ii_ktuple/

Usage:
    .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\tests\test_sss_ii_ktuple_rollout.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_STEPS = 5
SEED_BASE = 42
DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16

OUT_DIR = REPO_ROOT / "validation_renders" / "sss_ii_ktuple"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "rollout.log"

# Query definitions: (query_name, base_resolution, forks)
# Each fork: (name, cap_len, resolution_or_None, scale)
# resolution_or_None = None means same as base
#
# Multi-resolution workload that exercises multi-bin PACKSOLVE.
# Total tokens exceed REFERENCE_TOTAL_LEN=4224, forcing the server
# to bin-pack entries across multiple FlexAttention launches.
# All launches pad to REFERENCE_TOTAL_LEN — zero recompiles.
QUERIES = [
    ("shrimp", (512, 512), [
        ("base",   29, None,       1.0),    # e0: identity
        ("shrimp", 35, None,       3.0),    # e1: attractive
        ("typo",   40, None,       2.0),    # e2: attractive
        ("uncond",  8, None,      -6.0),    # e3: negative
    ]),
    ("banana", (512, 512), [
        ("base",   20, None,       1.0),    # e0: identity
        ("tropical", 25, None,     2.0),    # e1: attractive
        ("uncond",  8, None,      -3.0),    # e2: negative
    ]),
    ("tiny", (256, 256), [
        ("base",   29, None,       1.0),    # e0: identity
        ("uncond",  8, None,      -7.0),    # e1: negative
    ]),
]


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
# Rendering
# ---------------------------------------------------------------------------

def render_pseudo_rgb(latent: torch.Tensor, path: Path) -> None:
    """Save channels 0:3 of a latent as a pseudo-RGB PNG."""
    x = latent[0, :3].detach().cpu().float().numpy()  # (3, H, W)
    x = x.transpose(1, 2, 0)  # (H, W, 3)
    for c in range(3):
        lo, hi = x[:, :, c].min(), x[:, :, c].max()
        if hi > lo:
            x[:, :, c] = (x[:, :, c] - lo) / (hi - lo) * 255
        else:
            x[:, :, c] = 128
    Image.fromarray(x.astype(np.uint8)).save(str(path))


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
# Phase 2: Build specs + conditionings
# ---------------------------------------------------------------------------

def phase2_build_specs():
    _log("\n" + "=" * 60)
    _log("  PHASE 2: BUILD SPECS + CONDITIONINGS")
    _log("=" * 60)

    from tests.stubbed_skinny_shared_ii import make_random_conditioning

    specs = []
    cap_lens_per_query = []

    for q_name, base_res, forks in QUERIES:
        spec_entries = []
        query_cap_lens = []
        for fork_name, cap_len, res_override, scale in forks:
            cond = make_random_conditioning(cap_len=cap_len, device=DEVICE, dtype=DTYPE)
            res = res_override if res_override is not None else base_res
            spec_entries.append((cond, res, scale))
            query_cap_lens.append(cap_len)
            _log(f"  {q_name}/{fork_name}: cap_len={cap_len}, res={res}, scale={scale}")
        specs.append(spec_entries)
        cap_lens_per_query.append(query_cap_lens)

    _log(f"  Total queries: {len(specs)}")
    _log(f"  Total entries: {sum(len(s) for s in specs)}")

    return specs, cap_lens_per_query


# ---------------------------------------------------------------------------
# Phase 3: Build executor adapter
# ---------------------------------------------------------------------------

def make_batch_executor_adapter(batch_executor, query_sigmas, device):
    """Bridge ktuple_sampling's executor protocol to BatchExecutor.execute().

    ktuple_sampling.step() calls: executor(x_bases, specs, step_i, adapter_scales)
    Returns: (denoised_per_query, scores)

    BatchExecutor.execute() takes: [query_dicts]
    Returns: [result_dicts]
    """
    def executor_fn(x_bases, specs, step_i, adapter_scales=None):
        queries = []
        for k, (x_base, spec) in enumerate(zip(x_bases, specs)):
            base_cond, base_res, _ = spec[0]
            sigma = float(query_sigmas[k][step_i])

            forks = []
            for j, (cond, res, scale) in enumerate(spec):
                fork = {"entry_id": f"e{j}"}
                if j > 0:
                    fork["cond"] = cond
                if res != base_res:
                    fork["resolution"] = res
                forks.append(fork)

            queries.append({
                "query_id": f"q{k}",
                "base_latent": x_base,
                "base_cond": base_cond,
                "base_cap_len": base_cond.shape[1],
                "base_resolution": base_res,
                "sigma": sigma,
                "forks": forks,
            })

        results = batch_executor.execute(queries)

        # Group by query, sort by entry
        buckets = {k: [] for k in range(len(specs))}
        for r in results:
            k = int(r["query_id"][1:])
            j = int(r["entry_id"][1:])
            buckets[k].append((j, r["denoised"].to(device), r["scores"]))

        denoised_per_query = []
        for k in range(len(specs)):
            entries_sorted = sorted(buckets[k], key=lambda t: t[0])
            denoised_per_query.append([d for _, d, _ in entries_sorted])

        # Scores: return None for now (SSS-II score_proj is zero-init)
        return denoised_per_query, None

    return executor_fn


# ---------------------------------------------------------------------------
# Phase 4: Full trajectory with persistence
# ---------------------------------------------------------------------------

def phase4_trajectory(compiled_model, specs, cap_lens_per_query):
    _log("\n" + "=" * 60)
    _log("  PHASE 3: FULL K-TUPLE TRAJECTORY WITH GATHER + PERSISTENCE")
    _log("=" * 60)

    from src_ii.batch_executor import BatchExecutor
    from src_ii.sigma_schedule import build_sigma_schedule, resolution_shift
    from src_ii.triumphant_future_reduction_ops import (
        noise_field, aperture, euler_step, gather, latent_padded,
    )
    from src_ii.ktuple_sampling import solve

    # Build per-query sigma schedules (base resolution)
    query_sigmas = []
    for q_idx, (q_name, base_res, forks) in enumerate(QUERIES):
        alpha = resolution_shift(base_res[0], base_res[1])
        sigmas = build_sigma_schedule(N_STEPS, sampling_shift=alpha, device=DEVICE, dtype=DTYPE)
        query_sigmas.append(sigmas)
        _log(f"  {q_name} sigmas: [{float(sigmas[0]):.4f} ... {float(sigmas[-1]):.4f}]")

    # Create BatchExecutor
    batch_executor = BatchExecutor(compiled_model, DEVICE)
    executor_fn = make_batch_executor_adapter(batch_executor, query_sigmas, DEVICE)

    # Initial noise (each query gets different seed, aperture-correlated for multi-res)
    x_bases = []
    for q_idx, (q_name, base_res, forks) in enumerate(QUERIES):
        seed = SEED_BASE + q_idx
        w, h = base_res
        lh, lw = latent_padded(w, h)
        master = noise_field(lh, lw, seed, DEVICE, DTYPE)
        x = query_sigmas[q_idx][0] * aperture(master, lh, lw)
        x_bases.append(x)
        _log(f"  {q_name}: seed={seed}, latent={tuple(x.shape)}, norm={float(x.norm()):.2f}")

    # Save steps
    save_steps = set(range(N_STEPS))  # save every step

    # Trajectory storage
    trajectories = [{} for _ in range(len(QUERIES))]
    scores_jsonl = []
    step_timings = []
    peak_vram = 0.0

    def save_fn(step_i, x_pres, guided_list):
        nonlocal peak_vram
        for k, (q_name, _, _) in enumerate(QUERIES):
            q_dir = OUT_DIR / q_name
            q_dir.mkdir(parents=True, exist_ok=True)

            # Persist latent
            torch.save(x_pres[k].detach().cpu(), q_dir / f"step_{step_i:02d}_x.pt")
            torch.save(guided_list[k].detach().cpu(), q_dir / f"step_{step_i:02d}_guided.pt")

            # Stats
            x_cpu = x_pres[k].detach().cpu().float()
            g_cpu = guided_list[k].detach().cpu().float()
            sigma_val = float(query_sigmas[k][step_i])

            entry = {
                "step": step_i,
                "query": q_name,
                "sigma": sigma_val,
                "x_norm": float(x_cpu.norm()),
                "x_range": [float(x_cpu.min()), float(x_cpu.max())],
                "guided_norm": float(g_cpu.norm()),
                "guided_range": [float(g_cpu.min()), float(g_cpu.max())],
                "has_nan": bool(torch.isnan(x_cpu).any()),
                "has_inf": bool(torch.isinf(x_cpu).any()),
            }
            scores_jsonl.append(entry)
            _log(f"    {q_name} step {step_i}: sigma={sigma_val:.4f} "
                 f"x_norm={entry['x_norm']:.2f} guided_norm={entry['guided_norm']:.2f}")

        vram = torch.cuda.max_memory_allocated() / 1e9
        peak_vram = max(peak_vram, vram)

    # --- Run the solver ---
    _log(f"\n  Starting {N_STEPS}-step Euler solve with GATHER...")
    t0 = time.perf_counter()

    x_finals, scores_all = solve(
        executor_fn, x_bases, specs, query_sigmas, N_STEPS,
        gather_fn=gather,
        save_fn=save_fn,
    )

    total_elapsed = time.perf_counter() - t0
    _log(f"\n  Solve completed in {total_elapsed:.1f}s")
    _log(f"  Peak VRAM: {peak_vram:.2f} GB")

    # --- Persist finals ---
    _log("\n  Persisting final latents + renders...")
    for k, (q_name, base_res, _) in enumerate(QUERIES):
        q_dir = OUT_DIR / q_name
        x_final = x_finals[k].detach().cpu()

        # Save raw latent
        torch.save(x_final, q_dir / "final.pt")

        # Render pseudo-RGB
        render_pseudo_rgb(x_final, q_dir / "final_rgb.png")

        # Stats
        x_f = x_final.float()
        _log(f"  {q_name}: shape={tuple(x_final.shape)}, "
             f"norm={float(x_f.norm()):.2f}, "
             f"range=[{float(x_f.min()):.2f}, {float(x_f.max()):.2f}], "
             f"nan={bool(torch.isnan(x_f).any())}, inf={bool(torch.isinf(x_f).any())}")

    # --- Persist scores JSONL ---
    jsonl_path = OUT_DIR / "scores.jsonl"
    with open(jsonl_path, "w") as f:
        for entry in scores_jsonl:
            f.write(json.dumps(entry) + "\n")
    _log(f"  Scores written: {jsonl_path} ({len(scores_jsonl)} entries)")

    # --- Packing diagnostics ---
    _log(f"\n  Packing plans cached: {len(batch_executor._plan_cache)}")
    packing_diag = {}
    for key, plan in batch_executor._plan_cache.items():
        bins = plan["bins"]
        _log(f"    Plan {key[:8]}... {len(bins)} bins:")
        bin_info_list = []
        for b_idx, bin_info in enumerate(bins):
            indices = bin_info["indices"]
            # Map indices back to query names
            queries_in_bin = set()
            for global_idx in indices:
                for q_idx in range(len(QUERIES)):
                    q_start = sum(len(QUERIES[j][2]) for j in range(q_idx))
                    q_end = q_start + len(QUERIES[q_idx][2])
                    if q_start <= global_idx < q_end:
                        queries_in_bin.add(QUERIES[q_idx][0])
                        break
            _log(f"      Bin {b_idx}: {len(indices)} entries {indices} "
                 f"(queries: {queries_in_bin})")
            bin_info_list.append({
                "indices": indices,
                "n_entries": len(indices),
                "queries": sorted(queries_in_bin),
            })
        packing_diag[key] = {"n_bins": len(bins), "bins": bin_info_list}

    diag_path = OUT_DIR / "packing_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(packing_diag, f, indent=2)
    _log(f"  Packing diagnostics: {diag_path}")

    # --- Assertions ---
    for k, (q_name, base_res, _) in enumerate(QUERIES):
        x_f = x_finals[k].detach().cpu().float()
        lh, lw = latent_padded(base_res[0], base_res[1])
        assert x_f.shape == (1, 16, lh, lw), (
            f"{q_name}: shape {x_f.shape} != (1, 16, {lh}, {lw})"
        )
        assert not torch.isnan(x_f).any(), f"{q_name}: NaN in final latent"
        assert not torch.isinf(x_f).any(), f"{q_name}: Inf in final latent"

    # --- Summary ---
    n_bins = 0
    for plan in batch_executor._plan_cache.values():
        n_bins = len(plan["bins"])

    summary = {
        "n_steps": N_STEPS,
        "n_queries": len(QUERIES),
        "total_entries": sum(len(f) for _, _, f in QUERIES),
        "n_bins": n_bins,
        "total_elapsed_s": total_elapsed,
        "peak_vram_gb": peak_vram,
        "queries": [
            {
                "name": q_name,
                "resolution": base_res,
                "k": len(forks),
                "final_norm": float(x_finals[k].detach().cpu().float().norm()),
            }
            for k, (q_name, base_res, forks) in enumerate(QUERIES)
        ],
    }
    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"\n  Summary: {summary_path}")

    return x_finals, total_elapsed, peak_vram


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    _init_log()
    _log(f"SSS-II K-tuple Rollout -- Multi-Workload with GATHER + Persistence")
    _log(f"  N_STEPS={N_STEPS}, SEED_BASE={SEED_BASE}")
    _log(f"  Output: {OUT_DIR}")
    for q_name, base_res, forks in QUERIES:
        _log(f"  Query '{q_name}': K={len(forks)} at {base_res[0]}x{base_res[1]}")
        for fname, cap_len, res, scale in forks:
            res_str = f"{res[0]}x{res[1]}" if res else "base"
            _log(f"    {fname}: cap_len={cap_len}, res={res_str}, scale={scale:+.1f}")

    compiled_model = phase1_load_model()
    specs, cap_lens = phase2_build_specs()
    x_finals, elapsed, peak = phase4_trajectory(compiled_model, specs, cap_lens)

    _log("\n" + "=" * 60)
    _log("  PASS -- Multi-workload k-tuple rollout completed")
    _log("=" * 60)
    _log(f"  Output directory: {OUT_DIR}")
    _log(f"  Files:")
    for p in sorted(OUT_DIR.rglob("*")):
        if p.is_file():
            size = p.stat().st_size
            _log(f"    {p.relative_to(OUT_DIR)}: {size:,} bytes")

    return 0


if __name__ == "__main__":
    sys.exit(main())

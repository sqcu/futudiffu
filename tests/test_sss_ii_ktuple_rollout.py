r"""Full k-tuple rollout on SSS-II: multiple workloads, GATHER, persistence, renders,
and packed-vs-serial divergence validation.

Three sampler client sources submit to the same BatchExecutor simultaneously:
  Q0 "shrimp": K=4 at 512x512 (base + shrimp + typo + uncond)
  Q1 "banana": K=3 at 512x512 (base + tropical + uncond)
  Q2 "tiny":   K=2 at 256x256 (base + uncond)

Entries are packed into REFERENCE_TOTAL_LEN-sized bins by the server,
demonstrating funfetti batching across heterogeneous resolutions.  Total
tokens across all 9 entries exceed a single REFERENCE_TOTAL_LEN=4224 bin,
so the executor must split them into 2+ FlexAttention launches — each
padded to REFERENCE_TOTAL_LEN for zero recompiles.

Phases 1-4: K-tuple trajectory
  Client-side GATHER (weighted sum of residuals) reduces K entries to one
  guided denoised estimate per query. Client-side EULER advances the trajectory.
  Per-step latents, scores, and guided estimates are persisted to disk.
  Final latents are rendered as pseudo-RGB PNGs (channels 0:3 normalized).

Phase 5: Packed vs serial divergence validation
  At step 0 (sigma=1.0), runs all 9 entries through both packed (BatchExecutor,
  2+ bins) and serial (prepare_packed_forward + packed_forward, B=1) paths.
  Compares per-entry max_abs, mean_abs, l2, cosine similarity.
  Threshold: max_abs < 0.625 (SageAttention INT8 quantization noise).
  Reuses the SAME compiled model from phase 1 — zero redundant compilation.

Output: validation_renders/sss_ii_ktuple/

Usage:
    .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\tests\test_sss_ii_ktuple_rollout.py
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
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_STEPS = 5
SEED_BASE = 42
DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
BASE_SIGMA = 1.0  # Step 0 sigma for packed-vs-serial comparison
MAX_ABS_THRESHOLD = 0.625  # SageAttention INT8 quantization noise ceiling

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
# Phase 5: Packed vs serial divergence validation
# ---------------------------------------------------------------------------

def phase5_packed_vs_serial(compiled_model):
    """Validate that packed multi-bin forward matches serial per-entry forward.

    At step 0 (BASE_SIGMA), runs the same 9 entries through:
      (a) BatchExecutor (packed, 2+ bins)
      (b) Serial path (each entry individually via prepare_packed_forward + packed_forward)
    Compares per-entry: max_abs, mean_abs, l2, cosine similarity.
    Threshold: max_abs < MAX_ABS_THRESHOLD (0.625).

    Uses deterministic seeded conditionings so packed and serial get identical inputs.
    """
    _log("\n" + "=" * 60)
    _log("  PHASE 5: PACKED vs SERIAL DIVERGENCE VALIDATION")
    _log("=" * 60)
    _log(f"  Threshold: max_abs < {MAX_ABS_THRESHOLD}")
    _log(f"  Base sigma: {BASE_SIGMA}")

    from tests.stubbed_skinny_shared_ii import SSS_CAP_FEAT_DIM
    from src_ii.batch_executor import BatchExecutor, _scatter
    from src_ii.forward_packed import prepare_packed_forward, packed_forward
    from src_ii.triumphant_future_reduction_ops import denoise_all, latent_padded

    # --- Build deterministic conditionings (seeded per fork) ---
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

    # --- Build base latents (one per query, seeded) ---
    latents = {}  # q_idx -> base_latent
    for q_idx, (q_name, base_res, forks) in enumerate(QUERIES):
        seed = SEED_BASE + q_idx
        w, h = base_res
        lh, lw = latent_padded(w, h)
        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        noise = torch.randn(1, 16, lh, lw, device=DEVICE, dtype=DTYPE, generator=gen)
        latents[q_idx] = BASE_SIGMA * noise
        _log(f"  latent ({q_name}): shape={tuple(latents[q_idx].shape)}, "
             f"seed={seed}, norm={float(latents[q_idx].norm()):.2f}")

    # --- Helper: build query dicts from conds/latents ---
    def _build_query_dicts():
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
        return queries

    # --- Packed path (BatchExecutor) ---
    _log("\n  Running PACKED path (BatchExecutor)...")
    batch_executor = BatchExecutor(compiled_model, DEVICE)
    query_dicts = _build_query_dicts()

    t0 = time.perf_counter()
    packed_results = batch_executor.execute(query_dicts)
    packed_elapsed = time.perf_counter() - t0
    _log(f"  BatchExecutor.execute() completed in {packed_elapsed:.2f}s, "
         f"{len(packed_results)} results")

    # Log packing plan
    for key, plan in batch_executor._plan_cache.items():
        bins = plan["bins"]
        _log(f"  Packing plan {key[:8]}... => {len(bins)} bins:")
        entry_map = {}
        global_idx = 0
        for q_idx, (q_name, _, forks) in enumerate(QUERIES):
            for f_idx, (fork_name, _, _, _) in enumerate(forks):
                entry_map[global_idx] = f"{q_name}/{fork_name}"
                global_idx += 1
        for b_idx, bin_info in enumerate(bins):
            indices = bin_info["indices"]
            names = [entry_map.get(i, f"?{i}") for i in indices]
            _log(f"    Bin {b_idx}: entries {indices} = [{', '.join(names)}]")

    # --- Serial path (one entry at a time) ---
    _log("\n  Running SERIAL path (one entry at a time)...")
    query_dicts_serial = _build_query_dicts()
    entries = _scatter(query_dicts_serial)
    _log(f"  Scattered {len(entries)} entries from {len(QUERIES)} queries")

    serial_results = []
    for i, entry in enumerate(entries):
        _log(f"  Serial forward {i}/{len(entries)}: "
             f"{entry['query_id']}/{entry['entry_id']} "
             f"x={tuple(entry['x'].shape)} cap_len={entry['cap_len']} "
             f"sigma={entry['sigma']:.4f}")

        x = entry["x"].to(DEVICE)
        cond_entry = entry["cond"].to(DEVICE)
        sigma = entry["sigma"]
        cap_len = entry["cap_len"]
        lh, lw = x.shape[2], x.shape[3]

        t0_s = time.perf_counter()

        prepared = prepare_packed_forward(
            compiled_model,
            [cond_entry],
            [(lh, lw)],
            [cap_len],
            DEVICE,
        )

        timestep = torch.tensor([sigma], device=DEVICE, dtype=torch.float32)

        fields, scores = packed_forward(
            compiled_model,
            [x],
            [timestep],
            prepared["refined_caps"],
            prepared["packing_info"],
            prepared["block_mask"],
            prepared["packed_rope"],
        )

        sigma_tensor = torch.tensor(sigma, device=DEVICE, dtype=x.dtype)
        denoised_list = denoise_all([x], fields, [sigma_tensor])

        elapsed_s = time.perf_counter() - t0_s
        _log(f"    Forward + denoise in {elapsed_s:.2f}s, "
             f"denoised norm={float(denoised_list[0].norm()):.2f}")

        serial_results.append({
            "query_id": entry["query_id"],
            "entry_id": entry["entry_id"],
            "denoised": denoised_list[0].cpu(),
            "scores": scores[0].cpu() if scores is not None else None,
        })

    # --- Comparison ---
    _log("\n  Comparing packed vs serial...")
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

        # Compare scores if present
        if pr["scores"] is not None and sr["scores"] is not None:
            score_diff = (pr["scores"].float() - sr["scores"].float()).abs().max().item()
            comp["score_max_abs"] = float(score_diff)

        comparisons.append(comp)

        flag = "" if passed else " ** FAIL **"
        _log(f"  Entry {i} ({pr['query_id']}/{pr['entry_id']}): "
             f"max_abs={max_abs:.6f}  mean_abs={mean_abs:.6f}  "
             f"l2={l2:.4f}  cos={cos_sim:.8f}  {verdict}{flag}")
        if "score_max_abs" in comp:
            _log(f"    score max_abs: {comp['score_max_abs']:.6f}")

    # --- Persist results ---
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

    result_path = OUT_DIR / "packed_vs_serial.json"
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"\n  Results written to {result_path}")

    # --- Verdict ---
    if all_pass:
        _log(f"\n  All {len(comparisons)} entries PASS (max_abs < {MAX_ABS_THRESHOLD})")
    else:
        failing = [c for c in comparisons if c["verdict"] == "FAIL"]
        _log(f"\n  {len(failing)} entries FAILED (max_abs >= {MAX_ABS_THRESHOLD}):")
        for c in failing:
            _log(f"    {c['query_id']}/{c['entry_id']}: max_abs={c['max_abs']:.6f}")

    return all_pass, comparisons


# ---------------------------------------------------------------------------
# Phase 6: LoRA-bound packed vs serial (exercises adapter_scales != None branch)
# ---------------------------------------------------------------------------

# Per-query adapter scales for phase 6.
# Different scales per query exercises per-image routing in cross-query bins.
# shrimp: full adapter, banana: half, tiny: zero (adapter disabled).
LORA_ADAPTER_SCALES = {
    "shrimp": 1.0,
    "banana": 0.5,
    "tiny": 0.0,
}
LORA_RANK = 8
LORA_ALPHA = 16.0
LORA_ADAPTER_NAME = "rtheta"
LORA_INIT_B_STD = 0.01


def phase6_lora_packed_vs_serial():
    """Validate packed vs serial with LoRA adapters installed.

    Loads a FRESH SSS-II model (separate from phases 1-5), installs LoRA
    adapters, compiles, initializes with nonzero B, then runs the same
    packed-vs-serial validation with per-query adapter_scales.

    Lifecycle: load -> fuse -> install_multi_lora -> compile -> init_B -> test.
    This is the only correct ordering for torch.compile + LoRA.
    """
    _log("\n" + "=" * 60)
    _log("  PHASE 6: LORA-BOUND PACKED vs SERIAL VALIDATION")
    _log("=" * 60)
    _log(f"  Adapter: {LORA_ADAPTER_NAME}, rank={LORA_RANK}, alpha={LORA_ALPHA}")
    _log(f"  Per-query scales: {LORA_ADAPTER_SCALES}")

    from tests.stubbed_skinny_shared_ii import load_sss_model, SSS_CAP_FEAT_DIM
    from src_ii.multi_lora import install_multi_lora, assign_adapter, init_adapter_b_weights
    from src_ii.batch_executor import BatchExecutor, _scatter
    from src_ii.forward_packed import prepare_packed_forward, packed_forward
    from src_ii.triumphant_future_reduction_ops import denoise_all, latent_padded

    # --- Load fresh model (fused, NOT compiled) ---
    t0 = time.perf_counter()
    model = load_sss_model(device=DEVICE)
    _log(f"  Model loaded in {time.perf_counter() - t0:.1f}s")

    # --- Install LoRA adapters ---
    wrappers = install_multi_lora(model, max_adapters=3, max_rank=LORA_RANK)
    assign_adapter(model, 0, LORA_ADAPTER_NAME, LORA_RANK, LORA_ALPHA)
    _log(f"  LoRA installed: {len(wrappers)} modules wrapped")

    # --- Compile ---
    t1 = time.perf_counter()
    compiled_model = torch.compile(model, mode="default")
    _log(f"  torch.compile() in {time.perf_counter() - t1:.3f}s (lazy)")

    # --- Init B weights AFTER compile (so adapter has nonzero signal) ---
    n_inited = init_adapter_b_weights(model, LORA_ADAPTER_NAME, std=LORA_INIT_B_STD)
    _log(f"  Initialized {n_inited} B matrices with std={LORA_INIT_B_STD}")

    # --- Init score head with small random values (so scores are nonzero) ---
    torch.manual_seed(999)
    model.score_proj.weight.data.normal_(std=0.01)
    # score_norm is RMSNormModule(elementwise_affine=False) — no learnable weight
    _log(f"  Score head initialized (score_proj std=0.01)")

    _log(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # --- Build deterministic inputs (same seeds as phase 5) ---
    cond_seed = 1000
    conds = {}
    for q_idx, (q_name, base_res, forks) in enumerate(QUERIES):
        for f_idx, (fork_name, cap_len, res_override, scale) in enumerate(forks):
            gen = torch.Generator(device=DEVICE).manual_seed(cond_seed + q_idx * 100 + f_idx)
            cond = torch.randn(1, cap_len, SSS_CAP_FEAT_DIM, device=DEVICE, dtype=DTYPE, generator=gen)
            conds[(q_idx, f_idx)] = cond

    latents = {}
    for q_idx, (q_name, base_res, forks) in enumerate(QUERIES):
        seed = SEED_BASE + q_idx
        w, h = base_res
        lh, lw = latent_padded(w, h)
        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        noise = torch.randn(1, 16, lh, lw, device=DEVICE, dtype=DTYPE, generator=gen)
        latents[q_idx] = BASE_SIGMA * noise

    # --- Build adapter_scales per query ---
    # (max_adapters,) tensor — slot 0 is "rtheta", slots 1-2 empty
    query_adapter_scales = {}
    for q_name, _, _ in QUERIES:
        s = LORA_ADAPTER_SCALES[q_name]
        query_adapter_scales[q_name] = torch.tensor([s, 0.0, 0.0], dtype=torch.float32, device=DEVICE)

    # --- PACKED path ---
    _log(f"\n  Running PACKED path (with adapter_scales)...")
    batch_executor = BatchExecutor(compiled_model, DEVICE)

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
            "adapter_scales": query_adapter_scales[q_name],
        })

    t2 = time.perf_counter()
    packed_results = batch_executor.execute(queries)
    packed_elapsed = time.perf_counter() - t2
    _log(f"  BatchExecutor.execute() completed in {packed_elapsed:.2f}s, {len(packed_results)} results")

    # Log packing plan
    for key, plan in batch_executor._plan_cache.items():
        bins = plan["bins"]
        _log(f"  Packing plan {key[:8]}... => {len(bins)} bins:")
        entry_map = {}
        global_idx = 0
        for q_idx, (q_name, _, forks) in enumerate(QUERIES):
            for f_idx, (fork_name, _, _, _) in enumerate(forks):
                entry_map[global_idx] = f"{q_name}/{fork_name}"
                global_idx += 1
        for b_idx, bin_info in enumerate(bins):
            names = [entry_map.get(i, f"?{i}") for i in bin_info["indices"]]
            _log(f"    Bin {b_idx}: entries {bin_info['indices']} = [{', '.join(names)}]")

    # --- SERIAL path ---
    _log(f"\n  Running SERIAL path (with adapter_scales)...")
    entries = _scatter(queries)

    serial_results = []
    for i, entry in enumerate(entries):
        x = entry["x"].to(DEVICE)
        cond = entry["cond"].to(DEVICE)
        sigma = entry["sigma"]
        cap_len = entry["cap_len"]
        lh, lw = x.shape[2], x.shape[3]
        entry_adapter_scales = entry.get("adapter_scales")
        if entry_adapter_scales is not None:
            # Serial: single entry, so adapter_scales is (1, max_adapters)
            entry_adapter_scales = entry_adapter_scales.unsqueeze(0).to(DEVICE)

        t3 = time.perf_counter()
        prepared = prepare_packed_forward(
            compiled_model, [cond], [(lh, lw)], [cap_len], DEVICE,
        )
        timestep = torch.tensor([sigma], device=DEVICE, dtype=torch.float32)
        fields, scores = packed_forward(
            compiled_model, [x], [timestep],
            prepared["refined_caps"], prepared["packing_info"],
            prepared["block_mask"], prepared["packed_rope"],
            adapter_scales=entry_adapter_scales,
        )
        sigma_tensor = torch.tensor(sigma, device=DEVICE, dtype=x.dtype)
        denoised_list = denoise_all([x], fields, [sigma_tensor])
        serial_elapsed = time.perf_counter() - t3

        _scales = entry.get("adapter_scales")
        _s0 = float(_scales[0]) if _scales is not None else 0.0
        _log(f"  Serial {i}/9: {entry['query_id']}/{entry['entry_id']} "
             f"scale[0]={_s0:.1f} "
             f"in {serial_elapsed:.2f}s, "
             f"denoised_norm={float(denoised_list[0].detach().norm()):.2f}")

        serial_results.append({
            "query_id": entry["query_id"],
            "entry_id": entry["entry_id"],
            "denoised": denoised_list[0].detach().cpu(),
            "scores": scores[0].detach().cpu() if scores is not None else None,
        })

    # --- COMPARE ---
    _log(f"\n  Comparing packed vs serial (with LoRA)...")
    all_pass = True
    comparisons = []
    for i in range(len(packed_results)):
        pr = packed_results[i]
        sr = serial_results[i]

        packed_d = pr["denoised"].float()
        serial_d = sr["denoised"].float()
        diff = packed_d - serial_d

        max_abs = float(diff.abs().max())
        mean_abs = float(diff.abs().mean())
        l2 = float(diff.norm())

        p_flat = packed_d.flatten()
        s_flat = serial_d.flatten()
        cos_sim = float((p_flat * s_flat).sum() / (p_flat.norm() * s_flat.norm() + 1e-12))

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
            "verdict": verdict,
        }

        # Compare scores (should be nonzero now)
        if pr["scores"] is not None and sr["scores"] is not None:
            score_diff = (pr["scores"].float() - sr["scores"].float()).abs().max().item()
            comp["score_max_abs"] = float(score_diff)
            packed_score = pr["scores"].float().tolist()
            serial_score = sr["scores"].float().tolist()
            _log(f"  Entry {i} ({pr['query_id']}/{pr['entry_id']}): "
                 f"max_abs={max_abs:.6f}  cos={cos_sim:.8f}  {verdict}  "
                 f"scores_packed={[f'{s:.4f}' for s in packed_score]}  "
                 f"scores_serial={[f'{s:.4f}' for s in serial_score]}  "
                 f"score_diff={score_diff:.6f}")
        else:
            _log(f"  Entry {i} ({pr['query_id']}/{pr['entry_id']}): "
                 f"max_abs={max_abs:.6f}  cos={cos_sim:.8f}  {verdict}")

        comparisons.append(comp)

    # --- Persist ---
    result_path = OUT_DIR / "packed_vs_serial_lora.json"
    with open(result_path, "w") as f:
        json.dump({
            "adapter": LORA_ADAPTER_NAME,
            "rank": LORA_RANK, "alpha": LORA_ALPHA,
            "per_query_scales": {k: v for k, v in LORA_ADAPTER_SCALES.items()},
            "comparisons": comparisons,
            "overall_verdict": "PASS" if all_pass else "FAIL",
        }, f, indent=2)
    _log(f"\n  Results written to {result_path}")

    if all_pass:
        _log(f"  All {len(comparisons)} entries PASS (max_abs < {MAX_ABS_THRESHOLD})")
    else:
        failing = [c for c in comparisons if c["verdict"] == "FAIL"]
        _log(f"  {len(failing)} entries FAILED:")
        for c in failing:
            _log(f"    {c['query_id']}/{c['entry_id']}: max_abs={c['max_abs']:.6f}")

    return all_pass, comparisons


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

    # Phase 5: Packed vs serial divergence validation (reuses compiled_model, no LoRA)
    pvs_pass, pvs_comparisons = phase5_packed_vs_serial(compiled_model)

    # Phase 6: LoRA-bound packed vs serial (fresh model with adapters installed)
    lora_pass, lora_comparisons = phase6_lora_packed_vs_serial()

    # --- Final verdicts ---
    _log("\n" + "=" * 60)
    _log("  VERDICTS")
    _log("=" * 60)

    trajectory_pass = True  # phases 1-4 assert on failure; reaching here means pass
    _log(f"  Trajectory (phases 1-4):      PASS")

    pvs_verdict = "PASS" if pvs_pass else "FAIL"
    _log(f"  Packed vs serial (phase 5):   {pvs_verdict}")
    if not pvs_pass:
        failing = [c for c in pvs_comparisons if c["verdict"] == "FAIL"]
        for c in failing:
            _log(f"    {c['query_id']}/{c['entry_id']}: max_abs={c['max_abs']:.6f}")

    lora_verdict = "PASS" if lora_pass else "FAIL"
    _log(f"  LoRA-bound pvs (phase 6):     {lora_verdict}")
    if not lora_pass:
        failing = [c for c in lora_comparisons if c["verdict"] == "FAIL"]
        for c in failing:
            _log(f"    {c['query_id']}/{c['entry_id']}: max_abs={c['max_abs']:.6f}")

    _log("\n" + "=" * 60)
    overall = "PASS" if (trajectory_pass and pvs_pass and lora_pass) else "FAIL"
    _log(f"  OVERALL: {overall}")
    _log("=" * 60)
    _log(f"  Output directory: {OUT_DIR}")
    _log(f"  Files:")
    for p in sorted(OUT_DIR.rglob("*")):
        if p.is_file():
            size = p.stat().st_size
            _log(f"    {p.relative_to(OUT_DIR)}: {size:,} bytes")

    if _LOG_FILE is not None:
        _LOG_FILE.close()

    assert pvs_pass, (
        f"Packed vs serial divergence exceeded threshold {MAX_ABS_THRESHOLD}. "
        f"Failing entries: "
        + ", ".join(
            f"{c['query_id']}/{c['entry_id']}={c['max_abs']:.4f}"
            for c in pvs_comparisons if c["verdict"] == "FAIL"
        )
    )

    assert lora_pass, (
        f"LoRA-bound packed vs serial exceeded threshold {MAX_ABS_THRESHOLD}. "
        f"Failing entries: "
        + ", ".join(
            f"{c['query_id']}/{c['entry_id']}={c['max_abs']:.4f}"
            for c in lora_comparisons if c["verdict"] == "FAIL"
        )
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

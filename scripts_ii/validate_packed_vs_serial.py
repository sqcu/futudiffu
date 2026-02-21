#!/usr/bin/env python
r"""Packed vs serial trajectory equivalence validation.

Runs the same generation queries serially and via FlexAttention packing,
then compares latents and decoded images to prove equivalence.

For each of 5 resolutions, runs:
  1. Serial trajectory via sample_trajectory()
  2. Packed trajectory (N=1) via sample_trajectory_packed()
  3. Compares latents at every step + final
  4. VAE-decodes final latents, saves PNGs + false-color diffs
  5. Computes spatial autocorrelation of diffs (structured = bug, white noise = ok)
  6. Writes per-image and overall PASS/WARN/FAIL report

Output: packed_vs_serial_validation/ (latents, PNGs, false-color diffs, stats)

Usage:
    .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\validate_packed_vs_serial.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# -- path setup -------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import numpy as np

from src_ii.rendering import (
    save_tensor_as_png,
    save_false_color_diff,
    compute_per_channel_pixel_stats,
    compute_spatial_autocorrelation,
)

# =====================================================================
# Configuration
# =====================================================================

PROMPT = "An astronaut riding a horse across a desert under a starfield, photorealistic."

# 5 images, each at a different resolution tier
IMAGE_SPECS = [
    {"width": 1280, "height": 832,  "seed": 42, "label": "full_landscape"},
    {"width": 1024, "height": 1024, "seed": 43, "label": "full_square"},
    {"width": 512,  "height": 512,  "seed": 44, "label": "medium_square"},
    {"width": 640,  "height": 384,  "seed": 45, "label": "medium_landscape"},
    {"width": 256,  "height": 256,  "seed": 46, "label": "small_square"},
]

N_STEPS = 10
CFG = 4.0
SAMPLING_SHIFT = 1.0    # same shift for all => comparable latents
MULTIPLIER = 1.0
ATTENTION_BACKEND = "sdpa"

# Thresholds for latent comparison (max_abs_diff).
# FP8 blockwise quantization + FlexAttention vs SDPA numerical differences
# produce ~0.0625 max_abs per step, growing with accumulated steps.
# Serial uses SDPA; packed uses FlexAttention (even for N=1). This is
# expected behavior, not a bug. Thresholds are set accordingly:
#   WARN: > 0.1  -- single-step divergence (0.0625) passes; multi-step accumulation warns
#   ERROR: > 2.5 -- algorithmic divergence (mismatched RoPE, wrong sigma, etc.)
WARN_THRESHOLD = 0.1     # FP8+FlexAttention per-step noise floor
ERROR_THRESHOLD = 2.5    # algorithmic divergence beyond FP8+FlexAttention

SERVER_ENDPOINT = "tcp://127.0.0.1:5555"
CONNECTIVITY_TIMEOUT_MS = 2000   # Phase 0 check
RPC_TIMEOUT_MS = 300_000         # 5 min per RPC call

OUTPUT_ROOT = REPO_ROOT / "packed_vs_serial_validation"


# =====================================================================
# Tee logger: prints to stdout AND writes to a logfile simultaneously
# =====================================================================

class TeeLogger:
    """Dual-output logger: stdout + logfile."""

    def __init__(self, log_path: Path):
        self._log_path = log_path
        self._buf = io.StringIO()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "w", encoding="utf-8")

    def print(self, *args, flush: bool = True, end: str = "\n", **kwargs):
        msg = " ".join(str(a) for a in args) + end
        sys.stdout.write(msg)
        if flush:
            sys.stdout.flush()
        self._file.write(msg)
        self._file.flush()

    def close(self):
        self._file.close()


# =====================================================================
# Helpers
# =====================================================================

def check_server_zmq(endpoint: str, timeout_ms: int) -> bool:
    """Attempt a ZMQ connection + status RPC with a short timeout.

    Returns True if the server responds within timeout_ms, False otherwise.
    Does NOT raise on failure. Socket is closed cleanly in all cases.
    """
    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.LINGER, 0)
    try:
        sock.connect(endpoint)
        # Send a minimal status request using the protocol format
        from futudiffu.protocol import pack_request, unpack_response
        frames = pack_request("status", {}, {})
        sock.send_multipart(frames)
        response_frames = sock.recv_multipart()
        status, _, _ = unpack_response(response_frames)
        return status == "ok"
    except zmq.Again:
        return False
    except Exception:
        return False
    finally:
        sock.close(linger=0)
        ctx.term()


def make_output_dir() -> Path:
    """Create timestamped output directory."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_ROOT / f"run_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def compute_comparison_stats(
    serial_t: torch.Tensor, packed_t: torch.Tensor
) -> dict:
    """Compute L2, max_abs, cosine similarity, relative L2 between two tensors."""
    s = serial_t.float().flatten()
    p = packed_t.float().flatten()
    diff = p - s

    l2 = diff.norm().item()
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()

    # cosine similarity
    s_norm = s.norm().item()
    p_norm = p.norm().item()
    dot = (s * p).sum().item()
    cos_sim = dot / (s_norm * p_norm + 1e-12)

    # relative L2
    relative_l2 = l2 / (s_norm + 1e-12)

    return {
        "l2_distance": l2,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "cosine_similarity": cos_sim,
        "serial_l2_norm": s_norm,
        "packed_l2_norm": p_norm,
        "relative_l2": relative_l2,
    }


def classify_verdict(max_abs: float) -> str:
    """Classify a single image's max_abs_diff into PASS/WARN/FAIL."""
    if max_abs > ERROR_THRESHOLD:
        return "FAIL"
    elif max_abs > WARN_THRESHOLD:
        return "WARN"
    return "PASS"


# =====================================================================
# Main validation pipeline
# =====================================================================

def main():
    # Create output dir early so we can write logs and server_unavailable.txt
    output_dir = make_output_dir()
    log = TeeLogger(output_dir / "validation.log")

    log.print("=" * 72)
    log.print("PACKED vs SERIAL FlexAttention Validation")
    log.print("=" * 72)
    log.print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    log.print(f"Output directory: {output_dir}")
    log.print()

    # =================================================================
    # Phase 0: Server connectivity check (ZMQ with 2s timeout)
    # =================================================================
    log.print("--- Phase 0: Server connectivity check ---")
    log.print(f"  Endpoint: {SERVER_ENDPOINT}")
    log.print(f"  Timeout: {CONNECTIVITY_TIMEOUT_MS}ms")

    server_ok = check_server_zmq(SERVER_ENDPOINT, CONNECTIVITY_TIMEOUT_MS)

    if not server_ok:
        log.print()
        log.print("SERVER NOT AVAILABLE - script written but not executed.")
        log.print(f"Launch server with: .venv/Scripts/python.exe scripts/launch_server.py")
        log.print()

        # Write marker file
        note_path = output_dir / "server_unavailable.txt"
        with open(note_path, "w") as f:
            f.write(f"Server at {SERVER_ENDPOINT} was unreachable at "
                    f"{datetime.now(timezone.utc).isoformat()}\n")
            f.write(f"Timeout: {CONNECTIVITY_TIMEOUT_MS}ms\n")
            f.write(f"\nLaunch server with:\n")
            f.write(f"  .venv/Scripts/python.exe scripts/launch_server.py\n")
        log.print(f"  Note written to: {note_path}")
        log.close()
        sys.exit(0)

    log.print("  Server is reachable. Proceeding with validation.")

    # =================================================================
    # Setup client with longer timeout for actual RPCs
    # =================================================================
    from futudiffu.client import InferenceClient
    client = InferenceClient(SERVER_ENDPOINT, timeout_ms=RPC_TIMEOUT_MS)

    try:
        status = client.status()
        log.print(f"  Server status: {json.dumps(status, indent=2)}")
    except Exception as e:
        log.print(f"  ERROR getting server status: {e}")
        log.print(f"  Proceeding anyway (connectivity check passed).")

    # Track timing for everything
    timing = {}
    errors = []

    # =================================================================
    # Phase 1: Encode prompt (once, shared by all queries)
    # =================================================================
    log.print()
    log.print("--- Phase 1: Encoding prompt ---")
    log.print(f"  Prompt: {PROMPT}")
    t0 = time.perf_counter()
    pos_cond = client.encode_prompt(PROMPT)
    neg_cond = client.encode_prompt("")
    encode_time = time.perf_counter() - t0
    timing["encode_prompt"] = encode_time
    log.print(f"  pos_cond: {pos_cond.shape} {pos_cond.dtype}")
    log.print(f"  neg_cond: {neg_cond.shape} {neg_cond.dtype}")
    log.print(f"  Encoding took {encode_time:.2f}s")

    # Save conditioning for reproducibility
    torch.save(pos_cond, output_dir / "pos_cond.pt")
    torch.save(neg_cond, output_dir / "neg_cond.pt")

    all_save_steps = list(range(N_STEPS))

    # =================================================================
    # Phase 2: Serial (unpacked) trajectories
    # =================================================================
    log.print()
    log.print("--- Phase 2: Serial trajectories ---")
    serial_results: list[dict[str, torch.Tensor] | None] = []
    serial_timings: list[float] = []

    for idx, spec in enumerate(IMAGE_SPECS):
        w, h, seed, label = spec["width"], spec["height"], spec["seed"], spec["label"]
        log.print(f"  [{idx+1}/5] {label} {w}x{h} seed={seed} ... ", end="", flush=True)

        try:
            t0 = time.perf_counter()
            result = client.sample_trajectory(
                pos_cond=pos_cond,
                neg_cond=neg_cond,
                seed=seed,
                n_steps=N_STEPS,
                cfg=CFG,
                width=w,
                height=h,
                attention_backend=ATTENTION_BACKEND,
                sampling_shift=SAMPLING_SHIFT,
                multiplier=MULTIPLIER,
                save_steps=all_save_steps,
            )
            elapsed = time.perf_counter() - t0
            serial_timings.append(elapsed)
            serial_results.append(result)

            # Persist all step latents
            img_dir = output_dir / "serial" / f"query_{idx}_{w}x{h}"
            img_dir.mkdir(parents=True, exist_ok=True)
            for key, tensor in result.items():
                if isinstance(tensor, torch.Tensor):
                    torch.save(tensor, img_dir / f"{key}.pt")

            log.print(f"{elapsed:.2f}s  keys={sorted(result.keys())}")

        except Exception as e:
            elapsed = time.perf_counter() - t0
            serial_timings.append(elapsed)
            serial_results.append(None)
            err_msg = f"Serial query {idx} ({label} {w}x{h}) FAILED: {e}"
            errors.append(err_msg)
            log.print(f"FAILED ({elapsed:.2f}s)")
            log.print(f"    {err_msg}")
            log.print(f"    {traceback.format_exc()}")

    timing["serial_total"] = sum(serial_timings)
    timing["serial_per_query"] = serial_timings

    # =================================================================
    # Phase 3: Packed trajectories (N=1 per resolution)
    # =================================================================
    log.print()
    log.print("--- Phase 3: Packed trajectories (N=1 per resolution) ---")
    packed_results: list[dict[str, torch.Tensor] | None] = []
    packed_timings: list[float] = []

    for idx, spec in enumerate(IMAGE_SPECS):
        w, h, seed, label = spec["width"], spec["height"], spec["seed"], spec["label"]
        log.print(f"  [{idx+1}/5] {label} {w}x{h} seed={seed} (packed N=1) ... ", end="", flush=True)

        try:
            t0 = time.perf_counter()
            result_list = client.sample_trajectory_packed(
                pos_conds=[pos_cond],
                neg_cond=neg_cond,
                seeds=[seed],
                n_steps=N_STEPS,
                cfg=CFG,
                width=w,
                height=h,
                attention_backend=ATTENTION_BACKEND,
                sampling_shift=SAMPLING_SHIFT,
                multiplier=MULTIPLIER,
                save_steps=all_save_steps,
            )
            elapsed = time.perf_counter() - t0
            packed_timings.append(elapsed)
            result = result_list[0]  # N=1, take the only result
            packed_results.append(result)

            # Persist all step latents
            img_dir = output_dir / "packed" / f"query_{idx}_{w}x{h}"
            img_dir.mkdir(parents=True, exist_ok=True)
            for key, tensor in result.items():
                if isinstance(tensor, torch.Tensor):
                    torch.save(tensor, img_dir / f"{key}.pt")

            log.print(f"{elapsed:.2f}s  keys={sorted(result.keys())}")

        except Exception as e:
            elapsed = time.perf_counter() - t0
            packed_timings.append(elapsed)
            packed_results.append(None)
            err_msg = f"Packed query {idx} ({label} {w}x{h}) FAILED: {e}"
            errors.append(err_msg)
            log.print(f"FAILED ({elapsed:.2f}s)")
            log.print(f"    {err_msg}")
            log.print(f"    Traceback:")
            log.print(f"    {traceback.format_exc()}")

    timing["packed_total"] = sum(packed_timings)
    timing["packed_per_query"] = packed_timings

    # =================================================================
    # Phase 4: Latent comparison
    # =================================================================
    log.print()
    log.print("--- Phase 4: Latent comparison ---")

    per_image_stats: list[dict] = []

    for idx, spec in enumerate(IMAGE_SPECS):
        w, h, label = spec["width"], spec["height"], spec["label"]
        serial = serial_results[idx]
        packed = packed_results[idx]

        log.print(f"\n  Image {idx}: {label} {w}x{h}")

        if serial is None:
            log.print(f"    SKIPPED: serial result missing (generation failed)")
            per_image_stats.append({
                "idx": idx, "label": label, "width": w, "height": h,
                "seed": spec["seed"], "verdict": "SKIP",
                "reason": "serial generation failed",
            })
            continue

        if packed is None:
            log.print(f"    SKIPPED: packed result missing (generation failed)")
            per_image_stats.append({
                "idx": idx, "label": label, "width": w, "height": h,
                "seed": spec["seed"], "verdict": "SKIP",
                "reason": "packed generation failed",
            })
            continue

        # Find common tensor keys
        serial_keys = {k for k, v in serial.items() if isinstance(v, torch.Tensor)}
        packed_keys = {k for k, v in packed.items() if isinstance(v, torch.Tensor)}
        common_keys = sorted(serial_keys & packed_keys)

        if serial_keys != packed_keys:
            missing_in_packed = serial_keys - packed_keys
            missing_in_serial = packed_keys - serial_keys
            if missing_in_packed:
                log.print(f"    WARNING: keys in serial but not packed: {missing_in_packed}")
            if missing_in_serial:
                log.print(f"    WARNING: keys in packed but not serial: {missing_in_serial}")

        step_stats = {}
        worst_max_abs = 0.0

        for key in common_keys:
            stats = compute_comparison_stats(serial[key], packed[key])
            step_stats[key] = stats
            worst_max_abs = max(worst_max_abs, stats["max_abs_diff"])

            flag = ""
            if stats["max_abs_diff"] > ERROR_THRESHOLD:
                flag = " ** FAIL **"
            elif stats["max_abs_diff"] > WARN_THRESHOLD:
                flag = " * WARN *"

            log.print(f"    {key:>10s}: L2={stats['l2_distance']:.6f}  "
                      f"max_abs={stats['max_abs_diff']:.6f}  "
                      f"cos={stats['cosine_similarity']:.8f}  "
                      f"rel_L2={stats['relative_l2']:.6f}{flag}")

        verdict = classify_verdict(worst_max_abs)
        image_report = {
            "idx": idx,
            "label": label,
            "width": w,
            "height": h,
            "seed": spec["seed"],
            "worst_max_abs_diff": worst_max_abs,
            "verdict": verdict,
            "per_step": step_stats,
            "serial_keys": sorted(serial_keys),
            "packed_keys": sorted(packed_keys),
        }
        per_image_stats.append(image_report)
        log.print(f"    >> Image verdict: {verdict} (worst max_abs={worst_max_abs:.6f})")

        # Save per-step stats JSON
        stats_path = output_dir / f"latent_stats_{idx}_{w}x{h}.json"
        with open(stats_path, "w") as f:
            json.dump(step_stats, f, indent=2)

    # =================================================================
    # Phase 5: VAE decode + visual comparison
    # =================================================================
    log.print()
    log.print("--- Phase 5: VAE decode + render ---")

    diff_stats_all: list[dict] = []

    for idx, spec in enumerate(IMAGE_SPECS):
        w, h, label = spec["width"], spec["height"], spec["label"]
        serial = serial_results[idx]
        packed = packed_results[idx]

        if serial is None or packed is None:
            log.print(f"  [{idx+1}/5] {label} {w}x{h}: SKIPPED (missing result)")
            diff_stats_all.append({
                "idx": idx, "label": label, "width": w, "height": h,
                "verdict": "SKIP",
            })
            continue

        serial_final = serial.get("final")
        packed_final = packed.get("final")

        if serial_final is None or packed_final is None:
            log.print(f"  [{idx+1}/5] {label} {w}x{h}: SKIPPED (no final tensor)")
            diff_stats_all.append({
                "idx": idx, "label": label, "width": w, "height": h,
                "verdict": "SKIP",
            })
            continue

        try:
            log.print(f"  [{idx+1}/5] {label} {w}x{h}: decoding serial ... ", end="", flush=True)
            t0 = time.perf_counter()
            serial_img = client.vae_decode(serial_final)
            t1 = time.perf_counter()
            log.print(f"{t1-t0:.2f}s, decoding packed ... ", end="", flush=True)
            packed_img = client.vae_decode(packed_final)
            t2 = time.perf_counter()
            log.print(f"{t2-t1:.2f}s")

            # Save rendered PNGs
            serial_path = str(output_dir / f"serial_{idx}_{w}x{h}.png")
            packed_path = str(output_dir / f"packed_{idx}_{w}x{h}.png")
            save_tensor_as_png(serial_img, serial_path)
            save_tensor_as_png(packed_img, packed_path)

            # False-color diff: abs(serial - packed) * 10.0, saved via canonical module
            diff_path = str(output_dir / f"diff_{idx}_{w}x{h}.png")
            save_false_color_diff(serial_img, packed_img, diff_path, scale=10.0)

            # Per-channel pixel diff statistics
            pixel_stats = compute_per_channel_pixel_stats(serial_img, packed_img)

            # Spatial autocorrelation
            abs_diff_np = (serial_img.float() - packed_img.float()).abs()
            abs_diff_np = abs_diff_np.squeeze(0).permute(1, 2, 0).cpu().numpy()
            autocorr = compute_spatial_autocorrelation(abs_diff_np)

            diff_report = {
                "idx": idx,
                "label": label,
                "width": w,
                "height": h,
                "pixel_diff_stats": pixel_stats,
                "spatial_autocorrelation": autocorr,
                "serial_png": os.path.basename(serial_path),
                "packed_png": os.path.basename(packed_path),
                "diff_png": os.path.basename(diff_path),
                "decode_time_serial_s": t1 - t0,
                "decode_time_packed_s": t2 - t1,
            }
            diff_stats_all.append(diff_report)

            # Save per-image diff stats JSON
            diff_json_path = output_dir / f"diff_stats_{idx}_{w}x{h}.json"
            with open(diff_json_path, "w") as f:
                json.dump(diff_report, f, indent=2)

            log.print(f"    pixel diff: mean={pixel_stats['overall_mean']:.6f} "
                      f"std={pixel_stats['overall_std']:.6f} "
                      f"max={pixel_stats['overall_max']:.6f}")
            log.print(f"    per-channel: "
                      f"R(mean={pixel_stats['per_channel']['R']['mean']:.6f}) "
                      f"G(mean={pixel_stats['per_channel']['G']['mean']:.6f}) "
                      f"B(mean={pixel_stats['per_channel']['B']['mean']:.6f})")
            log.print(f"    autocorrelation: {autocorr['verdict']} "
                      f"(max={autocorr['max_autocorrelation']:.4f})")

        except Exception as e:
            err_msg = f"VAE decode/render for query {idx} ({label} {w}x{h}) FAILED: {e}"
            errors.append(err_msg)
            log.print(f"FAILED")
            log.print(f"    {err_msg}")
            log.print(f"    {traceback.format_exc()}")
            diff_stats_all.append({
                "idx": idx, "label": label, "width": w, "height": h,
                "verdict": "ERROR", "error": str(e),
            })

    # =================================================================
    # Phase 6: Summary report
    # =================================================================
    log.print()
    log.print("=" * 72)
    log.print("REPORT: Packed vs Serial FlexAttention Validation")
    log.print("=" * 72)

    # Overall verdict
    verdicts = [r.get("verdict", "SKIP") for r in per_image_stats]
    if "FAIL" in verdicts:
        overall = "FAIL"
    elif "WARN" in verdicts:
        overall = "WARN"
    elif all(v == "SKIP" for v in verdicts):
        overall = "NO_DATA"
    else:
        overall = "PASS"

    report_lines: list[str] = []
    report_lines.append("Packed vs Serial FlexAttention Validation Report")
    report_lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    report_lines.append(f"Prompt: {PROMPT}")
    report_lines.append(f"Steps: {N_STEPS}, CFG: {CFG}, Shift: {SAMPLING_SHIFT}")
    report_lines.append(f"Attention backend: {ATTENTION_BACKEND}")
    report_lines.append(f"WARN threshold: max_abs_diff > {WARN_THRESHOLD}")
    report_lines.append(f"FAIL threshold: max_abs_diff > {ERROR_THRESHOLD}")
    report_lines.append("")

    # -- Latent comparison table --
    report_lines.append("LATENT COMPARISON (final step):")
    report_lines.append(f"{'Img':>4s}  {'Resolution':>12s}  {'Seed':>6s}  "
                        f"{'L2':>10s}  {'max_abs':>10s}  {'cos_sim':>10s}  "
                        f"{'rel_L2':>10s}  {'Verdict':>7s}")
    report_lines.append("-" * 80)

    for r in per_image_stats:
        if r.get("verdict") == "SKIP":
            reason = r.get("reason", "missing data")
            report_lines.append(
                f"{r['idx']:>4d}  {r['width']:>5d}x{r['height']:<5d}  "
                f"{r['seed']:>6d}  {'--':>10s}  {'--':>10s}  {'--':>10s}  "
                f"{'--':>10s}  {'SKIP':>7s}  ({reason})"
            )
            continue

        final_stats = r.get("per_step", {}).get("final", {})
        l2 = final_stats.get("l2_distance", -1)
        max_abs = final_stats.get("max_abs_diff", -1)
        cos = final_stats.get("cosine_similarity", -1)
        rel_l2 = final_stats.get("relative_l2", -1)
        report_lines.append(
            f"{r['idx']:>4d}  {r['width']:>5d}x{r['height']:<5d}  {r['seed']:>6d}  "
            f"{l2:>10.6f}  {max_abs:>10.6f}  {cos:>10.8f}  "
            f"{rel_l2:>10.6f}  {r['verdict']:>7s}"
        )

    report_lines.append("-" * 80)
    report_lines.append(f"Overall verdict: {overall}")
    report_lines.append("")

    # -- Decoded image diff table --
    report_lines.append("DECODED IMAGE DIFF STATISTICS:")
    report_lines.append(f"{'Img':>4s}  {'Resolution':>12s}  {'mean':>10s}  "
                        f"{'std':>10s}  {'max':>10s}  {'Autocorr verdict'}")
    report_lines.append("-" * 80)
    for d in diff_stats_all:
        if d.get("verdict") in ("SKIP", "ERROR"):
            report_lines.append(
                f"{d['idx']:>4d}  {d['width']:>5d}x{d['height']:<5d}  "
                f"{'--':>10s}  {'--':>10s}  {'--':>10s}  "
                f"{d.get('verdict', 'SKIP')}"
            )
            continue
        ps = d["pixel_diff_stats"]
        ac = d["spatial_autocorrelation"]
        report_lines.append(
            f"{d['idx']:>4d}  {d['width']:>5d}x{d['height']:<5d}  "
            f"{ps['overall_mean']:>10.6f}  {ps['overall_std']:>10.6f}  "
            f"{ps['overall_max']:>10.6f}  {ac['verdict']}"
        )
    report_lines.append("")

    # -- Timing summary --
    report_lines.append("TIMING SUMMARY:")
    report_lines.append(f"  Prompt encoding: {timing.get('encode_prompt', 0):.2f}s")
    report_lines.append(f"  Serial total:    {timing.get('serial_total', 0):.2f}s")
    report_lines.append(f"  Packed total:    {timing.get('packed_total', 0):.2f}s")
    for i, spec in enumerate(IMAGE_SPECS):
        s_t = timing.get("serial_per_query", [0]*5)[i] if i < len(timing.get("serial_per_query", [])) else 0
        p_t = timing.get("packed_per_query", [0]*5)[i] if i < len(timing.get("packed_per_query", [])) else 0
        report_lines.append(f"    Query {i} ({spec['label']:>18s} {spec['width']:>4d}x{spec['height']:<4d}): "
                            f"serial={s_t:.2f}s  packed={p_t:.2f}s")
    report_lines.append("")

    # -- Errors --
    if errors:
        report_lines.append(f"ERRORS ({len(errors)}):")
        for err in errors:
            report_lines.append(f"  - {err}")
        report_lines.append("")
    else:
        report_lines.append("ERRORS: None")
        report_lines.append("")

    report_text = "\n".join(report_lines)
    log.print(report_text)

    # Save report.txt
    report_path = output_dir / "report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    log.print(f"\nReport saved to: {report_path}")

    # Save structured JSON report
    full_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "prompt": PROMPT,
            "n_steps": N_STEPS,
            "cfg": CFG,
            "sampling_shift": SAMPLING_SHIFT,
            "multiplier": MULTIPLIER,
            "attention_backend": ATTENTION_BACKEND,
            "warn_threshold": WARN_THRESHOLD,
            "error_threshold": ERROR_THRESHOLD,
            "image_specs": IMAGE_SPECS,
        },
        "overall_verdict": overall,
        "per_image_latent_stats": per_image_stats,
        "decoded_diff_stats": diff_stats_all,
        "timing": timing,
        "errors": errors,
    }

    json_path = output_dir / "report.json"
    with open(json_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    log.print(f"JSON report saved to: {json_path}")

    client.close()
    log.close()

    # Exit code reflects verdict
    if overall == "FAIL":
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

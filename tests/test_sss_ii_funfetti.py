r"""SSS-II compiled funfetti 6-tuple packed forward test.

Loads the stubbed-skinny-shared ZImageRLAIF model, compiles it, builds a
6-tuple entry set at mixed resolutions, bin-packs them into forward passes
via FlexAttention block masks at REFERENCE_TOTAL_LEN=4224, and verifies
the output shapes.

6-tuple entries:
  e0: 1024x1024, base prompt
  e1: 1024x1024, shrimp emphasis prompt
  e2: 1024x1024, typography emphasis prompt
  e3:  512x512,  base prompt (mid-res)
  e4:  256x256,  base prompt (low-res)
  e5: 1024x1024, banana prompt

The 6-tuple cannot fit in a single REFERENCE_TOTAL_LEN=4224 packed
sequence (four 1024x1024 images alone need ~16384 img tokens). Entries
are bin-packed via inference_packing.pack_for_inference() into multiple
bins, each running one compiled packed forward pass. This is the
canonical "funfetti batch" pattern.

Usage:
    .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\tests\test_sss_ii_funfetti.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch

from tests.stubbed_skinny_shared_ii import (
    load_sss_model,
    make_random_conditioning,
    SSS_N_SCORE_HEADS,
)
from src_ii.forward_packed import prepare_packed_forward, packed_forward
from src_ii.inference_packing import compute_entry_seq_len, pack_for_inference
from src_ii.bin_packer import REFERENCE_TOTAL_LEN

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16

# 6-tuple entry definitions: (prompt_key, (W, H))
P_BASE = "base"
P_SHRIMP = "shrimp"
P_TYPO = "typo"
P_BANANA = "banana"
P_NEG = "neg"

UNIQUE_PROMPTS = [P_BASE, P_SHRIMP, P_TYPO, P_BANANA, P_NEG]

ENTRY_DEFS = [
    (P_BASE,   (1024, 1024)),  # e0: base
    (P_SHRIMP, (1024, 1024)),  # e1: shrimp emphasis
    (P_TYPO,   (1024, 1024)),  # e2: typography emphasis
    (P_BASE,   ( 512,  512)),  # e3: mid-res
    (P_BASE,   ( 256,  256)),  # e4: low-res
    (P_BANANA, (1024, 1024)),  # e5: banana
]

SIGMA = 0.5  # arbitrary mid-schedule sigma


def _log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    _log("=" * 60)
    _log("SSS-II COMPILED FUNFETTI 6-TUPLE PACKED FORWARD TEST")
    _log("=" * 60)

    # ------------------------------------------------------------------
    # Phase 1: Create random conditionings (one per unique prompt)
    # ------------------------------------------------------------------
    _log("\n--- Phase 1: Random conditionings ---")
    conds: dict[str, torch.Tensor] = {}
    for prompt_key in UNIQUE_PROMPTS:
        conds[prompt_key] = make_random_conditioning(
            cap_len=29, device=DEVICE, dtype=DTYPE,
        )
        _log(f"  {prompt_key}: shape {tuple(conds[prompt_key].shape)}")

    # ------------------------------------------------------------------
    # Phase 2: Load SSS-II model + compile
    # ------------------------------------------------------------------
    _log("\n--- Phase 2: Load SSS-II model ---")
    t0 = time.perf_counter()
    model = load_sss_model(device=DEVICE)
    load_time = time.perf_counter() - t0
    _log(f"  Loaded in {load_time:.1f}s")
    _log(f"  VRAM after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    _log("  Compiling with torch.compile...")
    t1 = time.perf_counter()
    compiled_model = torch.compile(model, mode="default")
    compile_time = time.perf_counter() - t1
    _log(f"  torch.compile() returned in {compile_time:.1f}s")

    # ------------------------------------------------------------------
    # Phase 3: Bin-pack entries + prepare packing state per bin
    # ------------------------------------------------------------------
    _log("\n--- Phase 3: Bin-pack + prepare packing state ---")

    # Build per-entry metadata
    entries: list[dict] = []
    for i, (prompt_key, (rw, rh)) in enumerate(ENTRY_DEFS):
        cond = conds[prompt_key]
        lh, lw = rh // 8, rw // 8
        cap_len = cond.shape[1]
        seq_len = compute_entry_seq_len(lh, lw, cap_len)
        entries.append({
            "idx": i,
            "prompt_key": prompt_key,
            "rw": rw, "rh": rh,
            "lh": lh, "lw": lw,
            "cond": cond,
            "cap_len": cap_len,
            "seq_len": seq_len,
        })
        _log(f"  e{i}: {rw}x{rh} -> latent {lh}x{lw}, "
             f"cap_len={cap_len}, seq_len={seq_len}, prompt={prompt_key}")

    # Bin-pack with FFD
    entry_seq_lens = [e["seq_len"] for e in entries]
    bins = pack_for_inference(entry_seq_lens, max_total_len=REFERENCE_TOTAL_LEN)
    _log(f"\n  Bin packing result: {len(bins)} bins (REFERENCE_TOTAL_LEN={REFERENCE_TOTAL_LEN})")
    for b_idx, bin_indices in enumerate(bins):
        bin_total = sum(entry_seq_lens[i] for i in bin_indices)
        entry_strs = [f"e{i}({entries[i]['rw']}x{entries[i]['rh']})" for i in bin_indices]
        _log(f"    bin {b_idx}: {entry_strs}, total_seq={bin_total}/{REFERENCE_TOTAL_LEN}")

    # Prepare packing state per bin
    bin_plans: list[dict] = []
    t2 = time.perf_counter()
    for b_idx, bin_indices in enumerate(bins):
        context_list = [entries[i]["cond"] for i in bin_indices]
        img_sizes = [(entries[i]["lh"], entries[i]["lw"]) for i in bin_indices]
        cap_lens_bin = [entries[i]["cap_len"] for i in bin_indices]

        plan = prepare_packed_forward(
            compiled_model, context_list, img_sizes, cap_lens_bin, DEVICE,
        )
        bin_plans.append(plan)
        _log(f"    bin {b_idx} prepared: total_len={plan['packing_info'].total_len}, "
             f"block_mask={tuple(plan['block_mask'].shape)}")

    prep_time = time.perf_counter() - t2
    _log(f"  All bins prepared in {prep_time:.1f}s")
    _log(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ------------------------------------------------------------------
    # Phase 4: Run packed forward per bin, collect results
    # ------------------------------------------------------------------
    _log("\n--- Phase 4: Packed forward passes ---")

    # Pre-create random latents for all entries
    all_latents: list[torch.Tensor] = []
    for e in entries:
        all_latents.append(
            torch.randn(1, 16, e["lh"], e["lw"], device=DEVICE, dtype=DTYPE)
        )

    # Collect results indexed by entry
    all_fields: dict[int, torch.Tensor] = {}
    all_scores: dict[int, torch.Tensor] = {}

    torch.cuda.synchronize()
    t3 = time.perf_counter()

    for b_idx, bin_indices in enumerate(bins):
        plan = bin_plans[b_idx]

        x_list = [all_latents[i] for i in bin_indices]
        timesteps = [
            torch.tensor([SIGMA], device=DEVICE, dtype=torch.float32)
            for _ in bin_indices
        ]

        _log(f"  bin {b_idx}: running packed_forward with {len(x_list)} entries...")

        with torch.no_grad():
            fields, scores = packed_forward(
                compiled_model,
                x_list,
                timesteps,
                plan["refined_caps"],
                plan["packing_info"],
                plan["block_mask"],
                plan["packed_rope"],
            )

        # Map results back to global entry indices
        for local_idx, global_idx in enumerate(bin_indices):
            all_fields[global_idx] = fields[local_idx]
            all_scores[global_idx] = scores[local_idx]

        _log(f"    bin {b_idx}: done, {len(fields)} fields, scores shape {tuple(scores.shape)}")

    torch.cuda.synchronize()
    fwd_time = time.perf_counter() - t3
    _log(f"  All bins completed in {fwd_time * 1000:.0f}ms")
    _log(f"  VRAM after forward: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    _log(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # ------------------------------------------------------------------
    # Phase 5: Assertions
    # ------------------------------------------------------------------
    _log("\n--- Phase 5: Assertions ---")

    # Should have results for all 6 entries
    assert len(all_fields) == len(ENTRY_DEFS), (
        f"Expected {len(ENTRY_DEFS)} fields, got {len(all_fields)}"
    )
    assert len(all_scores) == len(ENTRY_DEFS), (
        f"Expected {len(ENTRY_DEFS)} scores, got {len(all_scores)}"
    )

    # Check each field's spatial shape matches input
    for i, (_, (rw, rh)) in enumerate(ENTRY_DEFS):
        field = all_fields[i]
        expected_h, expected_w = rh // 8, rw // 8
        actual_shape = tuple(field.shape)
        assert field.shape[0] == 1, f"e{i}: batch dim is {field.shape[0]}, expected 1"
        assert field.shape[1] == 16, f"e{i}: channels is {field.shape[1]}, expected 16"
        assert field.shape[2] == expected_h, (
            f"e{i}: height is {field.shape[2]}, expected {expected_h}"
        )
        assert field.shape[3] == expected_w, (
            f"e{i}: width is {field.shape[3]}, expected {expected_w}"
        )
        _log(f"  e{i}: field shape {actual_shape} -- OK")

    # Check scores shape per entry
    for i in range(len(ENTRY_DEFS)):
        score_i = all_scores[i]
        assert score_i.shape == (SSS_N_SCORE_HEADS,), (
            f"e{i}: score shape {tuple(score_i.shape)}, "
            f"expected ({SSS_N_SCORE_HEADS},)"
        )

    # Stack all scores for summary
    scores_stacked = torch.stack([all_scores[i] for i in range(len(ENTRY_DEFS))])
    assert scores_stacked.shape == (len(ENTRY_DEFS), SSS_N_SCORE_HEADS), (
        f"stacked scores shape {tuple(scores_stacked.shape)}, "
        f"expected ({len(ENTRY_DEFS)}, {SSS_N_SCORE_HEADS})"
    )
    _log(f"  scores shape (stacked): {tuple(scores_stacked.shape)} -- OK")

    score_max = float(scores_stacked.abs().max())
    _log(f"  scores max abs: {score_max:.6f} (expected ~0)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _log("\n" + "=" * 60)
    _log("  ALL ASSERTIONS PASSED")
    _log("=" * 60)
    _log(f"  Model load:       {load_time:.1f}s")
    _log(f"  Compile:          {compile_time:.1f}s")
    _log(f"  Packing prep:     {prep_time:.1f}s")
    _log(f"  Forward passes:   {fwd_time * 1000:.0f}ms ({len(bins)} bins)")
    _log(f"  Peak VRAM:        {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    _log(f"  Fields:           {len(all_fields)} tensors")
    _log(f"  Scores:           {tuple(scores_stacked.shape)}")
    for b_idx, bin_indices in enumerate(bins):
        _log(f"  Bin {b_idx}: entries {bin_indices}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

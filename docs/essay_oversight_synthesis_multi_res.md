# Oversight Synthesis: The Multi-Resolution Pipeline

**Date:** 2026-02-17
**Author:** Root session (Opus oversight)
**Responding to:**
- `docs/essay_bin_packing_text_aware.md` (bin packing text-aware fix)
- `docs/essay_sigma_shifting_integration.md` (sigma shifting cross-pipeline integration)

---

## What Happened

Three independent defects in the multi-resolution pipeline were discovered
and fixed in a single session. Each defect was invisible in the
single-resolution case (1280x832) and only surfaced when the bin packer
started scheduling heterogeneous batches. The defects are:

1. **Text token accounting** — the bin packer ignored caption tokens,
   silently over-packing small images by up to 23%.
2. **Missing sigma shifting** — all resolutions used the same noise
   schedule, making cross-resolution BTRM comparisons structurally invalid.
3. **CFG batch expansion** — the packed path's context_refiner received
   B=1 RoPE for B=2 CFG tensors, crashing every packed forward pass.

Defects 1 and 2 were found by audit. Defect 3 was found by the validation
script (serial succeeded, packed crashed). All three share a root cause:
the packed path was developed and tested at a single resolution, so
multi-resolution invariants were never exercised.

## Cross-Cutting Observation: Abstraction Boundaries

The bin packing essay identifies the core structural issue: "the bin packer
and the FlexAttention kernel were developed at different levels of
abstraction." The packer counted patch tokens. The kernel concatenated text
and image tokens. Neither had a cross-check.

The sigma shifting essay reveals the same pattern at a different boundary:
the euler loop used a global sigma schedule while the bin packer scheduled
items at different resolutions. The schedule and the packer agreed on step
count but disagreed on what that step count meant physically.

The CFG expansion bug is the third instance: `forward()` handled CFG batch
expansion; `prepare_packed_state()` did not. Two code paths that should
have shared invariants instead had independent, subtly different
implementations.

**Pattern:** Every time two subsystems agree on an interface (sequence
length, step index, batch dimension) but disagree on what the interface
*means*, a silent correctness defect exists. The defect is invisible until
the shared assumption is violated by a new usage pattern (multi-resolution
packing, in this case).

## The Sparse Compute Insight

The bin packing essay's most important contribution is not the fix but the
reframing of "utilization." Dense utilization (% of capacity filled) is
the wrong metric for block-sparse attention. The correct metric is the
sparse compute ratio:

```
sparse_ratio = sum(s_i^2) / capacity^2
```

At 98.5% dense utilization (13x 256x256 in a 4224-capacity bin), the
sparse ratio is 0.075 — meaning 92.5% of the attention FLOPS are skipped.
The 1.5% "wasted" capacity contributes nothing to cost. The 19% density
reduction (16→13 items) is entirely illusory because the old packing was
invalid (123% of capacity) and the new packing's underfill is free.

This reframing matters beyond bin packing. Any future analysis of packing
efficiency — including BTRM training batch construction — must use sparse
compute ratio, not dense utilization, to reason about throughput.

## The Sigma Shifting Data Flow

The sigma shifting essay traces the shift value through 7 files and 5
distinct lifecycle stages: plan → transmit → apply → persist → recover.
The key design property is that no file both computes and consumes the
shift in the same scope. This makes the data flow auditable: you can read
any single file and determine whether it is producing or consuming the
shift value, and verify the handoff at each boundary.

The open design decision about representative timestep selection is worth
flagging. The current choice (first image's sigma) is correct for
homogeneous packs but becomes questionable if the bin packer ever mixes
"full" and "small" tiers in the same pack. Currently the sequence-length
disparity (4160 vs 320 tokens) prevents this from happening — they
physically cannot share a bin. But if a future "medium" tier has items
with shift ~2.0 and shift ~1.5 in the same pack, the mean might be
preferable. This should be revisited if packing heterogeneity increases.

## The BTRM Validity Chain

Combining the two essays reveals the full chain of BTRM validity
requirements for multi-resolution training:

1. **Correct packing** (essay 1): Items fit within reference capacity, so
   FlexAttention masks are valid and no cross-image attention leakage
   occurs.
2. **Correct sigma** (essay 2): Each image follows its
   resolution-appropriate noise schedule, so "step 14" means the same
   perceptual denoising stage regardless of resolution.
3. **Correct pairing** (not yet essayed, pending agent): Cross-trajectory
   pairs only, with logSNR weighting derived from shift-corrected sigmas.
4. **Correct provenance** (essay 2, section 4.5): The shift value is
   persisted in V2 metadata so the pair sampler can reconstruct the
   exact sigma schedule used during generation.

If any link in this chain breaks, the reward model learns a confound
instead of a quality signal. Text token over-packing corrupts the forward
pass. Missing sigma shifting biases cross-resolution comparisons.
Intra-trajectory pairs confuse denoising stage with generation quality.
Missing provenance makes the pair sampler guess the shift instead of
reading it.

## What Remains

- **Validation re-run**: The freqs_cis fix is applied but the server has
  stale code. A Sonnet agent is restarting the server and re-running the
  validation script. Expected outcome: packed outputs match serial outputs
  with cos_sim > 0.999 for all resolutions.
- **Mixed-res RPC essay**: Third essay (server/client/pair changes) is
  being written by a Sonnet agent.
- **End-to-end test**: Once validation passes, the full chain (generate
  mixed-resolution batch → persist to V2 → pair sample → verify
  shift-corrected logSNR) should be exercised as a regression test.

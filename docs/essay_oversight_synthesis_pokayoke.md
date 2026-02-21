# Oversight Synthesis: Pokayoke Audit and the Karras Divergence

**Date:** 2026-02-18
**Author:** Root session (Opus oversight)
**Responding to:** `docs/essay_pokayoke_rendering_and_canonicalization.md`

---

## What the Audit Found

The pokayoke audit inventoried src_ii/ (20 modules) and scripts_ii/
(23 scripts) for rendering duplication, algorithm inlining, and
function-level code duplication. The numbers:

- **4** distinct tensor-to-PNG implementations (1 canonical, 3 inline)
- **2** false-color diff implementations (0 canonical, 2 inline)
- **3** copies of `resolution_shift()` (1 legitimate pure-Python dup)
- **2** sigma schedule implementations using **different formulas**
- **8** scripts containing inlined algorithms that should be imports
- **13** specific script-to-module migrations identified

## The Dangerous Finding

The rendering duplication (3 copies of squeeze/permute/fromarray) is
cosmetic. It wastes developer attention but does not produce wrong
results. The same pipeline written three times produces the same PNG
three times.

The sigma schedule divergence is different. `audit_dataset.py` uses a
Karras schedule (`sigma_max^(1/rho) + r * (sigma_min^(1/rho) -
sigma_max^(1/rho)))^rho` with rho=7.0). The production server and
all generation code use a ComfyUI `simple_scheduler` over SNR-shifted
sigmas. These are different formulas. They produce different sigma
values at the same step index.

The audit script's job is to validate the BTRM dataset — to check
that sigma values, step indices, and trajectory metadata are
consistent. If the audit script maps step indices to sigmas using the
wrong formula, every sigma-based check it performs is comparing
against the wrong reference. The audit passes when it should fail, or
fails when it should pass, because its ground truth is wrong.

This is the most insidious class of bug: a correctness error in the
validation tool itself. The trajectories are generated correctly (they
use the ComfyUI schedule via the server). The audit claims to verify
them but verifies against a Karras schedule instead. The audit
"passes" and everyone moves on. The divergence is silent.

**This is precisely what the pokayoke is designed to prevent.** If
`audit_dataset.py` were forced to import `build_sigma_schedule` from
`src_ii/sigma_schedule.py`, it would be structurally impossible for
the audit to use a different formula. The divergence exists because
the script inlines its own schedule instead of importing the canonical
one.

## The Rendering Module Gap

The audit identifies a clean architectural gap: `src_ii/rendering.py`
does not exist. The proposed API is well-scoped:

```
save_tensor_as_png(tensor, path)          — the 7-step pipeline, once
tensor_to_pil(tensor)                     — same thing, returns PIL
make_false_color_diff(a, b, scale=10.0)   — accepts PIL or tensor
compute_per_channel_pixel_stats(a, b)     — R/G/B mean/std/max
compute_spatial_autocorrelation(diff)     — lag-1, lag-2 structured error detection
```

This module would eliminate 3 inline tensor-to-PNG copies, 2 inline
diff implementations, and 2 inline statistics functions from
scripts_ii/. The migration is mechanical: replace the inline function
body with an import.

The split between "local VAE decode" (vae_utils, for GPU-local
scripts) and "server-backed decode" (client.vae_decode, for RPC
scripts) is correctly preserved. The rendering module handles
everything AFTER the decode — the tensor-to-pixel path that is
currently duplicated.

## The Pokayoke Mechanism

The audit proposes a grep-based lint script with two layers:

1. **Pattern detection**: Forbidden regex patterns in scripts_ii/
   (e.g., `.permute(1, 2, 0).cpu().numpy()` → "inlined tensor-to-PNG").
   Catches known duplication patterns.

2. **Name collision detection**: No function defined in scripts_ii/
   should share a name with a public function in src_ii/. Catches
   novel duplication even if the implementation doesn't match known
   patterns.

The grandfathering mechanism (exception list that shrinks
monotonically) is the right approach: it doesn't block on existing
violations, but it prevents new ones. Each migration removes an
exception. The exception list is itself an inventory of remaining
tech debt.

## Priority Order

Based on the audit's findings, the migration priority is:

1. **Fix the Karras divergence in audit_dataset.py.** This is a
   correctness bug in a validation tool. Highest priority because it
   undermines trust in dataset quality checks.

2. **Create `src_ii/rendering.py`.** This is the missing module that
   enables all the tensor-to-PNG and diff migrations.

3. **Migrate the 3 tensor-to-PNG copies.** Mechanical replacement.

4. **Migrate the 2 false-color diff copies.** Mechanical.

5. **Migrate finite_differences and sliding window stats to
   src_ii/stats.py.** Mechanical.

6. **Deploy the pokayoke lint script.** Prevents regression.

Items 1-5 are the refactoring. Item 6 is the pokayoke that keeps
the refactoring permanent. Without item 6, the duplication will
reappear the next time someone writes a script in a hurry.

## Broader Observation

The audit demonstrates the value of the reading-list-and-rubric
delegation pattern. The subagent was given:

- A reading list (10 items)
- A rubric (5 sections the essay must contain)
- Total autonomy in how to explore

It returned an evidence-based essay with block-quoted code excerpts,
concrete inventories, a proposed module API, and a deployable
pokayoke script. It also found a correctness bug (the Karras
divergence) that was not in the task description and would not have
been found by a narrowly-specified "fix the rendering duplication"
task.

This is the argument for rubrics over specifications: a rubric that
says "inventory all sigma schedule implementations" will find formula
divergences. A specification that says "move save_tensor_as_png to a
module" will not.

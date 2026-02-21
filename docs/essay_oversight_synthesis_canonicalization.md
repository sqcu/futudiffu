# Oversight Synthesis: Canonicalization and the Pokayoke Ratchet

**Date:** 2026-02-18
**Author:** Root session (Opus oversight)
**Responding to:**
- `docs/essay_pokayoke_rendering_and_canonicalization.md` (audit)
- `docs/essay_oversight_synthesis_pokayoke.md` (priority ordering)
- `docs/essay_canonicalization_rendering_and_stats.md` (implementation)

---

## What Was Accomplished

The canonicalization work executed the priority ordering from the pokayoke
synthesis essay. All six items completed:

1. **Karras sigma divergence fixed.** The audit script's inline Karras
   schedule was replaced with the canonical ComfyUI simple_scheduler via
   a new `build_sigma_schedule_py()` pure-Python function. The divergence
   was measured at up to **0.31 sigma units** at the schedule midpoint —
   not a minor numerical discrepancy but a categorical misclassification
   of denoising stage for every intermediate step.

2. **`src_ii/rendering.py` created.** Seven public functions covering the
   complete tensor-to-pixel pipeline: `tensor_to_pil`, `save_tensor_as_png`,
   `make_false_color_diff`, `save_false_color_diff`,
   `compute_per_channel_pixel_stats`, `compute_spatial_autocorrelation`,
   plus re-exports of `decode_latent_to_pil` and `load_vae`.

3. **Three tensor-to-PNG copies migrated.** From `validate_packed_vs_serial.py`,
   `validate_v2_dataset.py`, and `dataset_generator.py`.

4. **Two false-color diff copies migrated.** From `validate_packed_vs_serial.py`
   and `render_comparison.py`.

5. **Finite differences and time-series utilities migrated to `src_ii/stats.py`.**
   From `plot_sweep_curves.py` and `analyze_sweep_curves.py`.

6. **Pokayoke lint script deployed and exception list shrunk.** From 13
   grandfathered exceptions (10 inline + 3 name-collision) down to **1**
   (a false positive on `.permute(1,2,0).cpu().numpy()` used for
   autocorrelation preprocessing, not tensor-to-PNG).

## The Ratchet Effect

The pokayoke's exception list is the project's ratchet mechanism. The
number 13→1 is the concrete measure of canonicalization progress. The one
remaining exception documents itself: the comment explains what pattern
fires, why it's not a true violation, and what would remove it (either
refining the regex or changing the autocorrelation preprocessing path).

More importantly, any *new* violation will now be caught immediately. If
a future subagent writes a script in `scripts_ii/` that inlines a
tensor-to-PNG pipeline, a sigma schedule, or a finite differences
function, the pokayoke will fail. The violation must either be fixed
(import the canonical module) or grandfathered (add an exception with a
comment explaining the debt). The ratchet only turns one way.

This is precisely the structural guarantee that was missing when the same
rendering-as-measurement directive was given 4+ times across sessions.
The directive was textual — it lived in conversation history and docs.
Now it is mechanical — it lives in a script that can be run by CI, by
subagents, or by future root sessions that have never read the original
conversations.

## The Karras Divergence as Case Study

The sigma schedule bug deserves attention because it illustrates the
difference between cosmetic duplication and semantic duplication.

The three tensor-to-PNG copies were cosmetic: the same algorithm written
three times produces the same PNG three times. Fixing them reduces
maintenance burden but does not change correctness.

The Karras schedule was semantic: the audit script used a *different*
formula than the production server. Every sigma-based check the audit
performed was comparing against wrong reference values. The audit
"passed" for 50+ trajectories while being structurally incapable of
detecting the class of error it was designed to find.

This is the strongest argument for canonical modules: they make semantic
divergence *structurally impossible*. If `audit_dataset.py` imports
`build_sigma_schedule_py` from `src_ii.sigma_schedule`, it cannot use a
different formula. The pokayoke prevents it from defining an inline
replacement. The formula is a fact that exists in exactly one place.

## Cross-Cutting Observation: Pure-Python Variants

The canonicalization agent made an important design decision:
`build_sigma_schedule_py()` is a pure-Python port of the torch-based
`build_sigma_schedule()`, with agreement to within 2.81e-08. This
addresses a real constraint — audit scripts should not require CUDA —
without creating divergence risk, because the pure-Python function lives
in the same module as the torch version. If the torch version's formula
changes, the pure-Python version is in the same file and can be updated
atomically.

This pattern generalizes: any canonical module that serves both GPU and
CPU scripts should provide both tensor and scalar variants in the same
file. The alternative — letting CPU scripts inline a "simplified" version
of the formula — is exactly what produced the Karras divergence.

## What Remains

The canonicalization work addressed the `scripts_ii/` → `src_ii/`
boundary. The original `src/futudiffu/` codebase has its own duplication
(3 copies of `resolution_shift()`, sampling.py vs sigma_schedule.py
formula overlap). These are tracked by the pokayoke audit essay but were
not in scope for this work.

The rendering module is now canonical for post-decode operations. The
next integration target is ensuring that every script that generates
rollout latents *also* renders them through this module. The
rendering-is-measurement principle now has its canonical implementation;
what remains is enforcing its use at every generation site.

# Synthesis: SageAttention Integration and the src_ii/ Attention Gap

**Date:** 2026-02-17
**Author:** Root session
**Provenance:** Synthesis of `docs/essay_sage_integration_audit.md` against
the src/ freeze policy and lifecycle specifications

---

## The Narrow Gap and the Wide Implication

The audit confirms the SageAttention integration gap is mechanically narrow:
two call sites in `sampling.py` construct a FlexAttention `BlockMask` instead
of a `uint8` tensor. The kernel end accepts uint8. The dispatch end routes
uint8 to Sage. The construction site builds the wrong type.

But the implication is wide. src/ is frozen. The fix cannot go into
`src/futudiffu/sampling.py`. And `src_ii/` has **no attention dispatch logic
at all** — no packed forward path, no block mask construction, no attention
backend selection. The audit found that `src_ii/forward.py` wraps only the
unpacked `model.forward()`, and `src_ii/attention_capture.py` always falls
through to plain SDPA.

This means the SageAttention integration is not a patch. It is a *design
requirement for the src_ii/ attention module.* The correct move is to build
the attention dispatch in src_ii/ correctly from the start, with unified
masked SageAttention as the default path for both packed and unpacked cases,
rather than replicating the bifurcated dispatch from src/ and then fixing it.

---

## The Backward Pass Question

The audit surfaces a critical constraint: the masked SageAttention kernels
are **forward-only**. They do not produce the LSE (log-sum-exp) output
required for the backward pass. Training through the packed path with
SageAttention would require:

1. Masked LSE-producing forward kernels
2. Masked backward kernels (`_sage_attn_bwd_dkdv_masked`, `_sage_attn_bwd_dq_masked`)

This intersects lifecycle axis 7 (rollout-training coupling): if the same
forward path is used for both inference rollouts and training forward passes,
and training requires gradients through attention, then the attention kernel
must support backward. The unpacked SageAttention kernels already have full
backward pass support via `register_autograd`. The masked variants do not.

**Decision needed:** Is training through the packed path imminent? If yes,
masked backward kernels are blocking. If the packed path is inference-only
(rollout generation for BTRM dataset construction), then forward-only masked
Sage is sufficient and backward can be deferred.

Given the current training pipeline (BTRM reward model training uses
gradient-checkpointed forward through the backbone), and that packed forward
is used for multi-image dataset generation (inference) while training uses
single-image forward (with gradients), the answer is likely: **forward-only
masked Sage is sufficient for now.** Training through packed batches would be
a future optimization for multi-image BTRM training macrobatches.

---

## What src_ii/ Needs (Ordered)

1. **`src_ii/attention.py`** — Unified attention dispatch. Takes Q, K, V,
   optional block mask (always uint8 or None), backend config. Routes to
   masked Sage, unmasked Sage, or SDPA. No FlexAttention. No type-driven
   dispatch between `BlockMask` and `Tensor`. One type, one path.

2. **`src_ii/block_mask.py`** — uint8 block mask construction from packing
   info. `build_sage_block_mask(document_id, total_len, BLOCK_M=128,
   BLOCK_N=64) -> torch.Tensor`. Tiny module, ~30 lines.

3. **`src_ii/forward_packed.py`** — Packed forward path that uses the above.
   This is distinct from `src_ii/forward.py` (unpacked) because the outer
   specifications differ on axis 10 (sequence packing).

4. **Masked backward kernels** — In progress. The pattern is mechanical:
   add `HAS_BLOCK_MASK: tl.constexpr` and block mask lookup to the existing
   `_sage_attn_bwd_dkdv` and `_sage_attn_bwd_dq` kernels, same as was done
   for the forward kernels. Plus: masked forward variant that outputs LSE
   (needed by backward). STE for quantized activations inherited from
   existing unmasked backward. This is a kernel task, not an architectural
   decision. Dispatched to subagent.

---

## Section 8 Update

The audit confirms Section 8 of `docs/flexattention_batch_packing.md` is
obsolete. It should be updated to reflect:

- Masked SageAttention kernels exist (added after the design doc)
- The incompatibility was accidental and is resolved at the kernel level
- The integration gap is in block mask construction, not in the kernels
- The src_ii/ rewrite should build unified masked-Sage as the default

This update is deferred until the src_ii/ attention module is implemented,
at which point the design doc should be rewritten to reflect the actual
architecture rather than patched incrementally.

---

## Cross-Cutting Observation

The pattern here — "kernels exist but aren't wired in because a design doc
said they couldn't" — is an instance of a broader failure mode: **stale
design docs becoming normative.** Section 8 was correct when written. The
kernels were added. The doc was not updated. A subsequent agent read the doc,
not the kernels, and concluded integration was impossible.

The countermeasure is the pokayoke pattern: mechanical checks that detect
divergence between docs and code. For attention dispatch specifically, a test
that runs the same packed forward through both FlexAttention and masked Sage
and compares outputs would have surfaced the integration possibility. The
S-S-S model exists for exactly this kind of fast GPU test.

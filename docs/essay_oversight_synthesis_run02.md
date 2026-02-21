# Oversight Synthesis: Run 02 — Mixed-Resolution BTRM Training

**Date:** 2026-02-18
**Author:** Root session (Opus oversight)
**Responding to:** `docs/essay_run02_mixed_res.md` (subagent field report)

---

## What Run 02 Proved

Run 02 is the first end-to-end validation that the corrected multi-resolution
pipeline produces meaningful reward model signal. The stack of five prerequisites
has been discharged:

1. **BTRM training with improved optimizer** — loss 0.67→0.32 in 30 macrobatches,
   rtheta LoRA receiving gradients from macrobatch 1 (defect 24 confirmed fixed).
2. **Wider pair selection with sampling weighting** — cross-trajectory pairs,
   logSNR geometric weighting in the pair sampler.
3. **Multi-resolution data** — 4 specification tiers including 512x512 with
   sigma shift 2.016, all persisted in V2 format with `sampling_shift` metadata.
4. **Batch packing** — first 6 generation bins packed 1280x832 + 512x512 pairs
   into single FlexAttention kernel calls.
5. **E2E validation** — the packed path produced trajectories that were
   successfully loaded, scored, and trained on.

## The Three Live Defects

Three bugs were discovered and fixed during the run. Each is characteristic of
a different failure mode:

**R2-01 (Windows path delimiter)** is a cross-platform portability defect.
`traj_ref.split(":", 2)` fails on Windows drive letters (`F:`). The fix uses
`rfind(":")` to extract the integer traj_id from the end. This is the same
class of defect as run 1's #14 (Windows Python path expansion breaks SSH).
The pattern: any code that splits on `:` will fail on Windows paths. Future
code review should flag `:` splitting as suspicious.

**R2-02 (CUDA OOM without gradient checkpointing)** is a VRAM lifecycle defect.
Running 30 transformer layers under `enable_grad()` without checkpointing stores
all activations simultaneously (~51GB on 1280x832). The fix uses
`forward_checkpointed()` which recomputes each layer's activations during
backward. This is the correct long-term solution, not a workaround — gradient
checkpointing is standard practice for training models that exceed activation
memory budgets. The 4090's 24GB budget makes this mandatory for full-resolution
training.

**R2-03 (Inductor SymPy recursion)** is a torch.compile interaction defect.
LoRA's `torch.stack` operations create symbolic shape dependencies that trigger
a SymPy printer recursion in torch 2.10.0's inductor. The workaround — bypassing
`diff_compiled` during training via `forward_checkpointed` — is correct because
training should use eager mode with gradient checkpointing anyway. The compiled
inference path remains available for generation and scoring. This defect is
torch-version-specific and may resolve in a future torch release.

## Training Dynamics: What the Numbers Say

The scrongle head (step count discrimination: 30-step vs 10-step) reaches 100%
accuracy by macrobatch 20. The scrimble head (quantization discrimination: SDPA
vs SageAttention INT8) is noisier, reaching 93% at mb 20 but oscillating. This
asymmetry is expected and informative:

- **Scrongle is a low-frequency signal.** Fewer denoising steps produce visibly
  different images — lost detail, artifacts, incomplete denoising. The backbone's
  features at any layer can distinguish these.
- **Scrimble is a high-frequency signal.** INT8 quantization noise is subtle
  (cos_sim 0.9997 between SDPA and SageAttention outputs). Discriminating this
  requires many more examples to isolate the tiny feature-space displacement
  from random variation.

The gradient norm growth (9→70→29) is expected for LoRA with zero-initialized B.
As lora_B updates away from zero, the gradient path strengthens multiplicatively
(gradients flow through A·B, so larger B = larger gradients). The clip at 0.1
prevents this from causing divergence, but the growing norms suggest the learning
rate could be reduced in later stages or a warmup-then-decay schedule used.

## Dataset Scale Assessment

Run 02 used 30 trajectories (208 examples). The existing unified V2 dataset
contains 259 trajectories (~1800+ examples) from prior generation sessions,
all at 1280x832 single-resolution. These were not used in run 02's training.

The next training iteration should merge the datasets: 259 existing + 30 new
= 289 trajectories. This provides:
- More scrimble signal (existing data includes both SDPA and SageAttention runs)
- More scrongle signal (existing data includes various step counts)
- Cross-resolution pairs (new 512x512 data paired against existing 1280x832)

The 289-trajectory pool supports ~1800+ examples for pairing, which is
significantly more than the 208 used in run 02.

## What Remains in the Stack

The user's stated task stack was 5 items deep. All 5 are now exercised. The
natural continuation is:

1. **BTRM training on full dataset** — merge run 02 data with existing V2
   dataset, train for 100-200 macrobatches instead of 30.
2. **Policy optimization (REINFORCE)** — use the trained BTRM as reward signal
   to update ptheta. This requires the trained rtheta + score unembedder from
   the BTRM phase.
3. **Evaluation** — compare run 02 policy outputs against run 01 baseline.
   Render images, compute reward scores, assess whether the multi-resolution
   training data improved the reward model's discriminative power.

The render-at-checkpoint gap (the agent notes `btrm_renders/` is empty) should
be addressed: every BTRM checkpoint should VAE-decode its highest and lowest
scored latents for visual inspection. This is the rendering-is-measurement
principle applied to reward model training.

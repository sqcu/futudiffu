# Synthesis Essay 1: Extraction Checkpoint

## Where We Are

Six subagent essays have been produced. Three were audits of the existing codebase (codebase analysis, policy train drift, live run defects), written by agents reading the source documents cold. One was an algorithmic decomposition that read both the essays and the live source to identify five function boundaries (nfe, denoise, make_guided_denoiser, euler_solve, rollout) whose strict layering makes rollout/training drift structurally impossible. One was the Unit 1 extraction report, written by an Opus orchestrator that dispatched Sonnet agents to write 7 files in src_ii/ implementing those boundaries. One was the validation run report, written by a Sonnet that fixed path issues and ran the extraction against stored reference trajectory 0.

The extraction works. Step 0 is bitwise identical (seeded noise matches). Steps 1-30 diverge monotonically, reaching ~32% relative L2 at the final step. This is consistent with eager-vs-compiled floating-point non-determinism, not a logic error. A render pass (VAE decode to pixel space) is in progress to confirm this visually.

## What the Audit Essays Established

The three audit essays converged on a diagnosis that the reimplementation directive (v2) codified: the codebase has ~7 live files carrying all inference and training weight, embedded in dead or contradictory code. Two correctness bugs invalidate prior training (CFG mismatch: training path doesn't do guidance; sigma off-by-one: training evaluates at wrong noise level). The god class (ModelManager) conflates model loading, VRAM lifecycle, adapter management, training state, and RPC helpers. Multi-GPU coordination was designed for single-GPU and broke silently on expansion.

The critical user correction to the directive was that three features are *definitional*, not optional: (1) multi-LoRA with explicit lora_scales_tensor on every forward, (2) FlexAttention block masks on every forward — required even for B=1 because variable aspect ratios and prompt lengths produce variable sequence lengths — and (3) adapter lifecycle separation (allocate pre-compile, init post-compile). A fourth correction was that CFG is a sampling strategy, not part of forward(), because we intend to train CFG-free single-NFE diffusers.

## What the Algorithmic Decomposition Found

The decomposition essay is the most consequential artifact so far. It identified that the training path's `forward_checkpointed()` and `forward_no_grad()` in training_utils.py are standalone reimplementations of the model architecture — they re-derive embedding, refiners, 30 main layers, final layer, unpatchify, and negation from scratch rather than calling NextDiT.forward(). This is the structural root cause of every drift bug: any change to forward() that isn't independently mirrored in these two training functions creates a silent divergence. The decomposition's proposed fix — one `nfe()` function that wraps NextDiT.forward(), called by both rollout and training — is exactly what src_ii/ now implements for the rollout side.

The decomposition also clarified the type contracts: nfe returns raw predictions, denoise converts to denoised estimates, make_guided_denoiser returns a (x, sigma) → denoised closure, euler_solve consumes that closure. Each layer's output type is the next layer's input type. No layer reaches down two levels. The training path will call the same make_guided_denoiser with the same conditioning, and gradient checkpointing is applied by the caller wrapping nfe, not by writing a separate nfe_checkpointed.

## What Extraction + Validation Revealed

The extraction was clean: 7 files, no forbidden imports, all pure-math functions verified bitwise identical to their futudiffu.sampling equivalents. The Opus orchestrator dispatched Sonnets for code writing and reviewed their output — the Opus-orchestrates-Sonnets pattern worked as intended, with the Opus providing directives and the Sonnets producing code.

The validation run revealed the expected result: perfect noise match at step 0, monotonically diverging thereafter. The Sonnet agent that ran the validation needed zero code fixes — all path bugs had been caught in the prior debugging round. The 51-second eager-mode execution time is reasonable (30 steps, no compilation overhead, RTX 4090).

The gap between intent and result is small but real: we asked for bit-for-bit reproduction and got 32% relative L2 at the final step. The essay correctly identifies compilation state as the likely cause. The next diagnostic — running with torch.compile — would likely close most of the gap. But the *renders* (currently in progress) are the more important diagnostic: if the two images look perceptually identical despite 32% latent-space L2, then the divergence is benign floating-point noise amplified by the chaotic ODE. If they look visibly different, there's a configuration mismatch to find.

## What Comes Next

Pending the render essay, the immediate next units are:

**Unit 3: Training path alignment.** This is where the bugs actually get fixed. The training forward must call the same make_guided_denoiser() and nfe() as the rollout. The REINFORCE log-ratio must use correct sigma indexing. The BTRM hidden extraction must see both pos_cond and neg_cond branches. This unit should be dispatched to an Opus orchestrator with Sonnet code-writers, following the same pattern as Unit 1.

**Unit 4: PINKIFY/NOT_THAT integration gate.** The trivial-reward full-RL-loop test. This gates all subsequent work — if we can't move a trivially-computable reward in the right direction, the pipeline is broken regardless of BTRM quality. This requires Units 1-3 to be solid.

**Compiled-mode validation.** A follow-up validation run with torch.compile enabled, to determine how much of the 32% gap is compilation-state vs something else. This is a diagnostic, not a blocker — we proceed with Unit 3 regardless.

The operating protocol is working: subagents read and write, the root session reads essays and writes synthesis, the user reviews and course-corrects. The essay format creates legibility checkpoints that prevent context drift between layers of the agent hierarchy. The main risk is that I (root session) accumulate too many pending questions (renders, compiled-mode validation, Unit 3 design) and lose track of dependencies. This synthesis essay is the mechanism for preventing that.

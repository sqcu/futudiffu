# Reimplementation Directive v2: Rubrics, Not Solutions

## Errata from v1

v1 contained four errors that would have propagated into every subsequent work unit:

1. **FlexAttention is not "batch packing."** FlexAttention with block masks is required for EVERY forward call, including B=1, because this is a single-stream diffuser where prompt conditioning tokens are concatenated into the self-attention context. Different aspect ratios produce different sequence lengths; different prompts produce different token counts. FlexAttention masks pad any (image_tokens + prompt_tokens) to fit the compiled kernel's total_ctx without recompilation. That you can ALSO bin-pack multiple queries into one kernel call is incidental. The fundamental necessity is: 12,000 aspect ratios within a megapixel budget, or even 2 aspect ratios, or even 1 aspect ratio with 2 different prompt lengths, all require flex attention masking to share a single compiled forward. The v1 misunderstanding ("FlexAttention = batch packing multiple images") is the exact misconception that produced the branching, alternate, bloated plural code paths in the old codebase. There is one way to call forward(). It uses FlexAttention.

2. **CFG is not part of forward().** CFG is a sampling strategy: call forward() twice with different conditioning, linearly combine the results. Forward() takes (x, sigma, conditioning, lora_scales) and returns a prediction. The v1 directive baked CFG into the forward primitive ("CFG as part of every conditioned forward"). This is wrong because: (a) we intend to train CFG-free single-NFE diffusers, (b) the BTRM reward model evaluates individual forward outputs without CFG combination, (c) baking sampling strategy into the model call is the exact kind of conflation that caused the rollout/training drift. CFG lives in the sampler, not in the model.

3. **v1 prescribed solutions instead of rubrics.** It named files (engine.py, training_state.py, forward_cfg.py), prescribed class structures, cargo-culted a ZeroMQ server, and specified APIs. A directive should define the QUESTIONS subagents must answer and the INVARIANTS their answers must satisfy, not the implementation. The implementation is what subagents discover.

4. **v1 proposed unit tests.** Unit tests are forbidden. All validation is end-to-end: scripts that hit real services, map input tensors to output tensors, produce persistent files on disk, and allow ex post facto statistical comparison against previous runs. No mocks, no assert pass/fail, no bespoke imitation of imaginary programs.

## Synthesis of the Three Audits (Unchanged from v1)

The three audits converge on a single structural diagnosis: the codebase has a **live spine** of approximately 7 files embedded within dead or contradictory code. Two correctness bugs invalidate prior training runs (CFG mismatch, sigma off-by-one). The god class (ModelManager) conflates every concern. Multi-GPU coordination was designed for single-GPU and broke silently. Incremental fixes would touch 5+ files and leave the structural conditions. The choice is discontinuity.

## What Is Definitional

1. **Multi-LoRA with explicit lora_scales_tensor.** Every forward pass receives a lora_scales_tensor controlling per-batch, per-adapter contribution. This is the representation that makes one compiled model serve as policy, reference, and reward model depending on which scales are nonzero.

2. **FlexAttention block masks on every forward.** This is how one compiled kernel serves all (aspect_ratio, prompt_length) combinations without recompilation. It is required for a single image. It is required for a single aspect ratio with two different prompts. It is the only way to call forward(). The incidental ability to pack multiple queries is a consequence, not the purpose.

3. **Adapter lifecycle separation.** allocate_adapter (graph mutation, pre-compile, idempotent) and init_adapter_weights (parameter fill, post-compile, graph-invariant) are separate operations because torch.compile requires graph stability. 15 minutes of recompilation per violation.

4. **CFG is a sampling strategy, not a model property.** Forward() does not know about CFG. The sampler may call forward() once (CFG-free) or twice (with CFG). The training path for REINFORCE may or may not use CFG depending on what the rollout used. The function boundary between "one model evaluation" and "a sampling strategy that composes model evaluations" must be absolute.

## The Reimplementation Method

### Reproduction First, Restructuring Second

The documents describe reference tensor trajectories — stored latent checkpoints from rollouts that were run on the existing code. The reimplementation proceeds by:

1. **Lift the minimal scope** of functions and modules out of `src/` into `src_ii/` that bit-for-bit reproduce a reference tensor trajectory.
2. **Validate reproduction** by running the lifted code against stored reference trajectories and comparing output tensors numerically.
3. **Only then restructure** the lifted code to eliminate the structural conditions that caused the bugs.

This is not "rewrite from specification." This is "extract the working subset, prove it works, then clean it up." The extraction is informed by the essays (which identify what's live and what's dead), but the reproduction is verified against real tensors.

### No Assumed Architecture

v1 prescribed ZeroMQ, specific file names, specific class hierarchies. v2 does not. The questions to answer are:

- What is the minimal set of functions that, composed, produce a reference trajectory from (noise, prompt, sigma_schedule)?
- What is the minimal set of functions that, composed, compute a REINFORCE policy gradient from a stored trajectory?
- What is the minimal set of functions that, composed, score a trajectory with the BTRM head?
- How do these three sets overlap? (They should overlap maximally — drift bugs come from non-overlap.)
- What transport/dispatch layer, if any, is needed? (This is a question, not an assumption.)

### The Algorithmic Decomposition (Pending)

A subagent is currently analyzing the relationship between four concepts:
- **forward()**: one NFE, model(x, sigma, conditioning, lora_scales) → prediction
- **CFG reduction**: call forward() twice, combine — a sampling strategy
- **sampling function**: ODE/SDE stepper, iterates forward() or CFG(forward()) across sigma schedule
- **autoregressive rollout**: PRNG → noise → sampling → trajectory → image

The essay from this subagent will inform how the function boundaries are drawn. Until it returns, the work units below are rubrics (questions to answer), not implementations.

## Work Units as Rubrics

### Unit 0: Algorithmic Decomposition
**Question**: What are the 3-5 function boundaries that make rollout/training drift impossible, support both CFG and CFG-free sampling, and correctly layer every performance optimization (FlexAttention, multi-LoRA, FP8 GEMM, SageAttention, fused kernels) into a single compiled forward path?
**Status**: Subagent in progress.
**Output**: `docs/essay_algorithmic_decomposition.md`

### Unit 1: Minimal Extraction for Trajectory Reproduction
**Question**: What is the smallest subset of functions from the live spine that, lifted into `src_ii/`, reproduces a reference tensor trajectory bit-for-bit? This subagent reads the algorithmic decomposition essay and the three audit essays, then reads ONLY the specific functions identified as "live" to extract them.
**Depends on**: Unit 0
**Output**: Code in `src_ii/`, plus an essay documenting what was extracted, what was left behind, and what the function call graph looks like.
**Validation**: Run the extracted code against a stored reference trajectory. Persistent output: tensor files showing element-wise difference (should be zero or ULP-level).

### Unit 2: Reproduction Validation Script
**Question**: Does the extracted code in src_ii/ produce identical output to the reference trajectories?
**Depends on**: Unit 1
**Output**: A script in `scripts_ii/` that loads reference trajectory data, runs the src_ii/ code with the same inputs, and writes comparison tensors to disk. No assert pass/fail — the script produces files; a human or subsequent subagent reads the files.
**Validation**: Persistent tensor comparison files on disk.

### Unit 3: Training Path Alignment
**Question**: Does the training forward path (REINFORCE log-ratio, BTRM hidden extraction) in src_ii/ call the exact same forward() as the sampling path? Is the sigma indexing correct? Is the conditioning setup identical?
**Depends on**: Units 1, 2 (reproduction confirmed)
**Output**: Training-path code in src_ii/ that reuses the sampling path's forward(), plus an essay on how drift is eliminated by construction.
**Validation**: Script that runs both paths on the same input and writes both outputs to disk for comparison. Persistent files.

### Unit 4: PINKIFY/NOT_THAT Integration Gate
**Question**: Can we run a complete RL loop — generate rollout → score with trivially-computable reward → compute policy gradient → update adapter weights → generate another rollout → verify reward improves — using ONLY src_ii/ code?
**Depends on**: Unit 3
**Output**: A script that executes the full loop and writes per-iteration metrics + trajectory tensors to disk.
**Validation**: The persistent metrics file shows reward improving across iterations. If it doesn't, the pipeline is broken.

### Unit 5: Structural Cleanup
**Question**: Now that the extracted code reproduces trajectories and passes the PINKIFY gate, what restructuring eliminates the conditions that caused the 25 defects from the live run? This unit reads the live run defects essay and restructures the extracted code.
**Depends on**: Unit 4 (PINKIFY passes)
**Output**: Restructured src_ii/ code, plus an essay on what structural changes were made and which defect classes they prevent.
**Validation**: Re-run the PINKIFY gate and the reproduction validation. Both must still pass. Persistent output files for comparison against pre-restructuring runs.

## Validation Philosophy

All validation is end-to-end. A "test" is a script that:
1. Takes input tensors (or generates them from a PRNG seed)
2. Calls real code through real services (if services exist) or through direct function calls
3. Writes output tensors and metrics to persistent files on disk
4. Does NOT assert pass/fail — it produces data

Success or failure is determined by reading the output files, either by a human or by a subsequent subagent that writes a comparison essay. This means:
- Every run is comparable to every previous run
- Regressions are detected by statistical comparison, not by Boolean assertions
- The evidence base accumulates over time and survives environment changes
- A "test" that was written but never run is not a test; it is a hypothesis

## Subagent Operating Protocol (Unchanged)

Subagents receive rubrics, reading lists, and success criteria. They return implementations and 5-paragraph essays. The root session reads only essays and writes synthesis essays every 2-3 rounds. The root session does not read source files from src/ or src_ii/.

## What This Is

This is reproduction-first reimplementation. Extract the live code, prove it reproduces reference outputs, then restructure it so the 25 classes of discovered defects cannot recur. The essays are the map. The reference trajectories are the ground truth. The persistent output files are the evidence.

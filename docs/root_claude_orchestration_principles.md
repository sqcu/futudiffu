# Root Claude Orchestration Principles

**Date:** 2026-02-18
**Author:** Root session
**Provenance:** Direct user correction after observing repeated violations
of delegation structure across 4+ sessions

---

## The Two Roles

There are exactly two roles in this system. They have different tools,
different information diets, and different failure modes.

### Root Session Claude

**Has:** Full conversation history. User intent. The reason tasks exist.
Knowledge of which essays have been written, which defects are open, which
agents succeeded and which failed and why. The oversight document
(`docs/user_re_oversight.md`). The lifecycle specification
(`docs/user_dataflow_and_lifecycle.md`).

**Does not have (by policy):** Direct code reads. The root session does
not read source files. It does not trace shape mismatches through 5
files. It does not grep for function definitions. Every byte of
implementation detail that enters the root session's context is one fewer
byte available for understanding *why the work is being done.*

**Failure mode:** Context drowning. The root session reads a 400-line
file to "understand the issue," loses the thread of what 3 agents are
doing concurrently, and spends the next 10 messages recovering awareness
instead of synthesizing results. This has happened multiple times.

**Job:** Assign reading lists and rubrics. Receive essays. Write synthesis
essays. Maintain awareness of the project's trajectory. Know *why*
coding tasks were launched. Delegate *how* they are accomplished.

### Subagent (Opus overseeing Sonnets)

**Has:** Tools. File access. Bash. The ability to read, grep, edit,
compile, run. Domain knowledge from its prompt (the reading list and
rubric). Flexible authority to make decisions within the rubric's scope.

**Does not have:** Conversation history beyond its prompt. The user's
intent beyond what the root session communicates. Knowledge of what other
agents are doing.

**Failure mode:** Solving the wrong problem correctly. An agent with a
tight task specification will execute the specification even if the
specification is wrong. An agent with a reading list and rubric will
read the material, form its own understanding, and make decisions that
align with the rubric's *intent* rather than just its *letter.*

**Job:** Read assigned materials. Understand the codebase through its
own exploration. Make implementation decisions. Write code. Run tests.
Return an essay documenting what it did, what it found, and what
decisions it made, with block-quoted appendices for evidence.

---

## Reading Lists and Rubrics, Not Task Specifications

The root session does not write task specifications. It writes:

**A reading list** — files and documents the subagent must read before
acting. These establish context. Example:

> Read: `docs/user_dataflow_and_lifecycle_rollup.md` (the 10 axes),
> `src_ii/bin_packer.py`, `src_ii/dataset_generator.py`,
> `scripts_ii/validate_packed_vs_serial.py`.

**A rubric for success** — properties that the result must satisfy,
stated as invariants rather than as implementation steps. Example:

> Rubric: Every function that computes rollout latents must also render
> them to file via VAE decode. "Compute then discard" is a defect.
> Rendering is measurement. Latent statistics without rendered images
> are not validation.

What the root session does NOT write:

- "Open file X. Find function Y. Change line Z."
- "First do A, then B, then C."
- Step-by-step implementation plans with line numbers.

The subagent has file access. It can find things. Telling it where to
look (reading list) and what counts as success (rubric) is sufficient.
Telling it exactly what to type is micromanagement that produces brittle
results and wastes the root session's context on implementation details
it should not be tracking.

---

## The Essay as Return Value

Subagents return essays, not summaries. The distinction:

**A summary** is "I changed X in file Y to fix Z." It tells the root
session what happened but not why, and provides no evidence.

**An essay** is a structured document with:

1. **k paragraphs of analysis** — what the subagent found, what
   decisions it made, and why. Written as prose that a future reader
   (including a future root session after autocompaction) can use to
   reconstruct the reasoning.

2. **Block-quoted appendices of unlimited length** — code excerpts,
   test output, error messages, diff fragments. These are evidence.
   They are separated from the analysis so the root session can read
   the k paragraphs without being forced to consume the appendices.

The essay is persisted to `docs/` as a markdown file. It is not
transient. It is not a chat message. It survives autocompaction,
session restarts, and context limits. The root session reads essays
and writes synthesis essays in response. The synthesis essays capture
cross-cutting observations that no single subagent could make.

---

## Rendering Is Measurement

This principle has been stated by the user 4+ times across multiple
sessions:

> Image rendering is the measurement of end-to-end validation.
> Image statistics are what we index on.

Corollaries:

1. **Every script that produces rollout latents must also VAE-decode
   and render them to PNG.** The VAE is cheap (~200ms per image on a
   4090) compared to N forward evaluations of the diffuser (~7-120s
   depending on resolution and step count). There is no performance
   argument for skipping the decode.

2. **Latent statistics (L2, cosine similarity, max absolute diff) are
   auxiliary, not primary.** They are useful for automated pass/fail
   thresholds. They are not useful for understanding whether the
   generation is correct. A human looks at images. A reward model
   looks at images. Latent norms are a proxy that can be arbitrarily
   misleading.

3. **"Compute then discard" is a defect.** If a script generates
   latents and computes statistics on them without saving the latents
   and rendering them, the script has a structural defect. The
   intermediate data is the evidence base. Discarding it means the
   result cannot be audited, compared, or reproduced.

4. **Rendering belongs in a canonical module, not in each script.**
   If 3 scripts each contain their own VAE decode + save logic, that
   is 3 opportunities for subtle divergence. The canonical
   implementation lives in a module (`src_ii/rendering.py` or
   equivalent). Scripts call the module. The module handles decode,
   color space, file format, metadata embedding, and false-color
   diff generation.

This last point — canonicalization in modules — generalizes beyond
rendering.

---

## The src_ii Canonicalization Principle

`src_ii/` is the extracted library. `scripts_ii/` is the thin
orchestration layer. The boundary between them is:

**Module (src_ii):** Implements an algorithm. Has no CLI. Has no
argparse. Has no print statements except through logging. Is imported
by scripts and by other modules. Is the single source of truth for
its algorithm. Is tested.

**Script (scripts_ii):** Parses arguments. Calls modules. Handles
I/O (reading configs, writing output files). Contains NO algorithm
implementation. If a script contains a for-loop that implements
euler stepping, or a function that computes sigma schedules, or
inline VAE decode logic — that code should be in a module.

Violations of this boundary:

- **Inlined algorithms in scripts**: A script that reimplements bin
  packing instead of importing `src_ii.bin_packer`.
- **Wrapper indirection**: A module that wraps another module's
  function with no added value, creating a call chain that obscures
  the actual implementation.
- **Pathological duplication**: The same function appearing in
  `src/futudiffu/sampling.py`, `src_ii/sigma_schedule.py`, and
  `src_ii/bin_packer.py` because each needs the formula but none
  imports from a shared location.
- **Manager classes that own algorithms**: A class called
  `XManager` that contains both lifecycle management (loading,
  unloading, caching) and algorithm implementation (forward pass,
  sampling). These should be separated: the manager calls the
  algorithm module.

The refactoring agenda is not "move code from src/ to src_ii/." It
is: identify canonical algorithms, implement each one once in a
module, eliminate all other copies, and verify that scripts are thin
orchestration with no inlined logic.

---

## What Was Missing From Prior Documentation

The prior documents (`user_re_oversight.md`,
`user_dataflow_and_lifecycle.md`, `essay_*`) established:

- The 10 outer specifications for lifecycle correctness
- The oversight pattern (root doesn't read code)
- Technical case studies for specific integrations

What they did not establish, and what this document adds:

1. **The reading-list-and-rubric delegation pattern.** Prior
   subagent dispatches used detailed task specifications ("read file
   X, change Y, run Z"). This is the wrong abstraction level for
   the root session.

2. **The essay-as-return-value format.** Prior subagents returned
   summaries or tool-call results. The expectation is structured
   essays with block-quoted appendices, persisted to docs/.

3. **The rendering-is-measurement principle as a mandatory
   invariant.** Prior docs mentioned rendering but did not
   establish it as a hard constraint on every script that produces
   latents.

4. **The src_ii module/script boundary as an enforceable rule.**
   The refactoring was described but the boundary was not defined
   precisely enough to detect violations mechanically.

These four additions are intended to be durable. They apply to
every future session, every subagent dispatch, and every code
review. They are the operating rules of this orchestration system.

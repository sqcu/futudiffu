# Synthesis Essay 2: Structural Corrections

## The State After Unit 4

Nine subagent essays have been produced since the start of this session. The first three (audits) established the diagnosis. The fourth (algorithmic decomposition) identified the five function boundaries. The fifth and sixth (extraction + validation) proved that src_ii/ can reproduce reference trajectories to perceptual equivalence. The seventh (PINKIFY/THISNOTTHAT) proved the BTRM pipeline end-to-end: reward functions, preference labels, head training, persistence, scoring. The eighth (renders) confirmed the trajectory divergence is edge-positional floating-point noise, not semantic. The ninth (fake-porting audit) found two structural problems: the BTRM head trained without an adapter (reproducing defect 24), and 8 instances of algorithmic code inlined in scripts instead of imported from modules.

The extraction checkpoint (synthesis 1) was optimistic: "the extraction works." It does — the five function boundaries are clean, the rollout reproduces, the BTRM head trains. But the audit revealed that the *scripts* built on top of the extraction don't respect the same discipline. They inline algorithms, duplicate functions, and — most critically — allow the same class of structural defect (forgetting to bind an adapter) that the refactoring was supposed to eliminate.

## Two Corrections In Flight

The Opus orchestrator currently running addresses both problems simultaneously:

**Correction 1: BTRM compound model.** A new `src_ii/btrm_model.py` that enforces adapter+head coupling at construction time. You cannot create a BTRM model without allocating an r_theta adapter. The optimizer always includes both adapter and head parameters. Persist and load handle both as one artifact. This makes defect 24 — "adapter never trained because optimizer only had head params" — structurally unrepresentable. The fix is not a check or an assertion; it is a type-level constraint.

This connects to the user's broader point: the refactoring exists to make classes of errors impossible, not to produce diffs that look right in GitHub. If the new code allows the same errors as the old code, it's a fake port regardless of how clean the function signatures are.

**Correction 2: Algorithm extraction.** Every inlined algorithm in scripts_ii/ gets pulled into a named, importable function in src_ii/. The scripts become thin orchestration: parse args, call functions, write outputs. The extractions include: BTRM scoring (4 copies → 1 import), Bradley-Terry training loop (inlined → function), VAE decode-to-PIL (3 copies → 1 import), AttentionCapture (script-local class → reusable module), Spearman correlation (reimplemented → function), visualization rendering (170 lines → module).

After this round, src_ii/ should contain approximately 13 files: the original 7 (forward, guided_denoiser, solver, sigma_schedule, rollout, model_loading, reward_functions) plus 6 new extractions (btrm_model, btrm_training, vae_utils, attention_capture, stats, visualization). Each file has one role. Each script imports instead of inlining.

## What The Audit Pattern Reveals

The fake-porting audit was the most valuable intervention in this session so far, despite producing no code. It established a meta-pattern: every round of subagent code-writing should be followed by an audit round that checks whether the new code actually respects the principles it was supposed to embody. Subagents write code that works — the BTRM head trained, the attention maps rendered, the scores computed. But "works" is not the same as "is correctly structured." A script that inlines 65 lines of Bradley-Terry loss works. A script that imports `train_btrm()` from src_ii works AND prevents the next developer (or subagent) from creating a sixth copy.

The audit also caught the OOM: the attention script materialized the full attention matrix in float32 instead of streaming per-head at FP8. This is a consequence of subagents optimizing for correctness without memory awareness. The fix (per-head streaming) was applied, but the lesson is that VRAM constraints should be in the rubric, not discovered by the audit.

Going forward, every subagent directive should include: (1) what to build, (2) what to import (not reimplement), (3) VRAM budget (the model is 6GB, workspace should not exceed 10GB total), (4) the persistent output files expected.

## What Comes After This Round

Once the compound model and extraction are done, the outstanding work units are:

1. **Re-run attention interpretability** with the compound BTRM model (adapter bound, not just head). This time the attention diff between scale=0 and scale=1 will show what the trained adapter actually changes, not just what the frozen backbone encodes.

2. **Unit 3: Training path alignment.** The REINFORCE forward calls the same make_guided_denoiser() and nfe() as rollout. Correct sigma indexing. The BTRM compound model provides the reward scoring. This is the last piece before the full RL loop.

3. **Compiled-mode validation.** Re-run the trajectory reproduction with torch.compile to see if the 32% latent L2 collapses. Diagnostic, not blocker.

4. **Synthesis essay 3** after the compound model essay arrives, covering the full state of src_ii/ as a library.

## The Operating Protocol Is Working (With One Amendment)

The essay-as-return-type protocol is effective: subagents read directives, produce code and essays, the root session reads essays and writes synthesis. The user reviews and course-corrects. The amendment from this round: every code-writing dispatch should be followed by an audit dispatch before the root session declares a unit complete. The audit catches fake-porting, inlined algorithms, VRAM oversubscription, and structural defects that functional correctness alone does not reveal. The cost is one additional subagent round per unit. The benefit is that the new code actually embodies the principles it was supposed to, rather than just passing its immediate tests.

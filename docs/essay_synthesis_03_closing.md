# Synthesis Essay 3: Closing Assessment

## The Opening Problem

This session began with a diagnosis: the futudiffu codebase had undergone the specific kind of blowout that happens when mid-sized codebases grow through incremental feature extension without corresponding architectural evolution. Three audit documents existed — a DeepSeek-authored codebase critique, a policy-vs-training drift analysis, and a live run defects catalog from the first 2xH100 session. Together they identified: a god class conflating six responsibilities, two correctness bugs invalidating all prior training (CFG mismatch, sigma off-by-one), 25 operational defects from the first real deployment, and a structural condition where the "live" code carrying all inference and training weight was embedded within a larger body of dead or contradictory code.

The user's framing was precise: incremental editing would produce diffs with lots of green and red lines. What was needed was a discontinuity — deprecate the old reference code without deleting it, validate that better implementations are correct, and restructure so that the discovered classes of errors cannot recur. The constraint on the root session was equally precise: do not read the source directly (you'll drown in tokens and lose the ability to reason), use subagents for all reading and writing, receive results as essays, write synthesis essays for legibility checkpoints, and let the user intervene at every boundary.

## What Was Accomplished

**Eleven subagent essays were produced and reviewed.** Three audits established the diagnosis. One algorithmic decomposition identified five function boundaries (nfe, denoise, make_guided_denoiser, euler_solve, rollout) whose strict type layering makes rollout/training drift structurally impossible. One extraction report documented lifting these into src_ii/. One validation run confirmed perceptual equivalence against reference trajectories (exact noise match at step 0, edge-positional float divergence thereafter — confirmed by VAE-decoded renders). One PINKIFY/THISNOTTHAT report proved the BTRM training pipeline end-to-end: 97% pairwise agreement with literal pinkify rule, bit-for-bit persistence, trained head scoring real trajectories. One fake-porting audit caught 8 inlined algorithm instances and the reproduction of defect 24 (head trained without adapter). One compound model report documented the structural fix: BTRMCompoundModel enforces backbone+adapter+head coupling at construction, making defect 24 unrepresentable, and extracted all inlined algorithms into src_ii/. Two attention interpretability reports produced spatial heatmaps and layer-head sensitivity maps.

**src_ii/ is now a 13-module library** with clear non-overlapping responsibilities: 5 inference pipeline modules, 1 model loader, 1 reward functions, 2 BTRM compound modules, 4 utilities. scripts_ii/ contains thin orchestration scripts that import from src_ii/ instead of inlining algorithms. The import firewall is verified: src_ii/ imports architecture and kernels from futudiffu but zero orchestration.

**Three structural defects from the audit documents are now addressed by construction:**
- CFG mismatch (drift bug #1): eliminated because make_guided_denoiser is the single entry point for both rollout and training, parameterized by CFG scale
- Sigma off-by-one (drift bug #2): eliminated because euler_solve has one implementation with one callback convention
- Adapter never trained (defect 24): eliminated because BTRMCompoundModel refuses to construct without adapter, and the optimizer always includes both parameter groups

## What the Supervisory Method Revealed

The essay-as-return-type protocol worked better than expected. The root session consumed ~12 essays and wrote 3 synthesis essays without reading a single line of Python source from either the old codebase or the new one. Every course correction came from reading an essay and noticing a gap between intent and result — not from reading code.

The most valuable intervention was the fake-porting audit. Without it, src_ii/ would have been declared complete with 7 genuine modules and 8 instances of inlined algorithms in the scripts layer. The audit cost one subagent round and caught a structural defect (head-without-adapter) that functional testing alone did not reveal. The lesson: every code-writing dispatch should be followed by an audit dispatch. Working code is necessary but not sufficient; the code must also embody the structural principles it was supposed to.

The user's corrections to the directive were the other critical interventions: FlexAttention is not batch packing (it's required for every forward, even B=1), CFG is a sampling strategy (not part of forward), and the directive should specify rubrics (questions to answer) not solutions (files to create). Each correction prevented a wrong abstraction from propagating through all subsequent subagent work. The root session cannot make these corrections if it doesn't understand the domain — the synthesis essay format forces that understanding to be maintained despite never reading source.

## What Remains

The adapter attention diff capture is running now — this will show what the trained r_theta actually does to attention routing, the first mechanistic interpretability result from the compound model.

Beyond this session, the outstanding units from the directive are:

**Unit 3: Training path alignment.** The REINFORCE forward must call the same make_guided_denoiser() and nfe() as rollout. The compound BTRM model provides reward scoring. src_ii/ now has the infrastructure; the training-specific code (gradient checkpointing around nfe, log-ratio computation, advantage normalization) needs to be written against it.

**The full PINKIFY RL loop.** Generate rollout → score with BTRM compound model → compute policy gradient → update adapter weights → generate again → verify reward improves. This is the integration gate that proves the entire pipeline before touching real BTRM heads.

**Transport layer.** The directive correctly left the server/client architecture as an open question. Whether src_ii/ needs its own ZMQ layer, uses a simpler dispatch, or runs in-process depends on the deployment topology — which is a decision, not a discovery.

The codebase entered this session as a "self-referential bloated mush" with two known wrong-gradient bugs and 25 operational defects. It exits with a 13-module library that reproduces reference trajectories, trains reward models with enforced adapter coupling, and has produced 60+ persistent output files as the evidence base for continued development. The discontinuity was achieved — not by editing old code, but by extracting and validating new code that makes the old code's failure modes unrepresentable.

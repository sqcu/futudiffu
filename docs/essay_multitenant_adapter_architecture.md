# Multi-Tenant Adapter Architecture: A Review of Nonstandard-for-2023 Design Patterns

**Date**: 2026-02-27
**Context**: Written after a root cause analysis of VRAM oversubscription in the DDGRPO
training orchestrator, which was caused by treating BTRM scoring as a separate blocking
serial phase instead of scheduling it as work items through the same executor pipeline
used by everything else.

---

## 1. One Model, Many Effective Models

The ZImageRLAIF model is physically one set of weights on one GPU. But through
multi-tenant LoRA adapter routing, it is logically many effective models simultaneously.
A `MultiLoRALinear` layer computes `base(x) + sum(scale_j * (x @ A_j^T @ B_j^T))` where
the scales are per-image in a packed batch. Two images in the same packed forward pass
can see completely different effective models — one might be running the policy adapter
while the other runs the reward model adapter — because their adapter_scales differ.

This is not a convenience optimization. It is the architectural primitive that makes
on-policy RL over diffusion models fit on a single accelerator. Without it, policy
rollouts, reward scoring, and reference model evaluation would each require separate
model instances or sequential weight-swapping. With it, they are batch entries in the
same forward pass, distinguished only by their scale vectors.

The scale vector is a semantic declaration. `[policy=1, btrm=0]` means: the diffusion
field output is a valid policy denoising and the score head output is meaningless
(the hidden states feeding the score head were shaped by the policy adapter, not the
reward adapter that was trained to produce those scores). `[policy=0, btrm=1]` means:
the diffusion field is an off-policy reference-plus-reward output, but the score head
output is a valid learned BTRM reward prediction. `[0, 0]` is the base reference model —
valid for KL divergence anchoring. `[1, 1]` is an out-of-distribution chimera that is
on-policy for neither adapter's training distribution and should never be scheduled.

These are not hardcoded bitstrings. They come from a model setup config that maps adapter
indices to semantic roles. The training loop must record which adapter index means what
in its run metadata, because a checkpoint with adapter indices `[0=rtheta, 1=policy_pinkify]`
is semantically different from `[0=policy_pinkify, 1=rtheta]` even if the weight files
are identical.

## 2. The Forward Pass Returns Everything; Adapter Scales Determine What's Valid

The model's `forward()` always returns `(diffusion_fields, scores)`. This is a kernel
uniformity decision: the score head is a 20-line projection (RMSNorm → Linear → soft_tanh_cap)
applied to the same hidden state that `final_layer` consumes. Running it costs <0.1% of
the transformer FLOPS. Branching on "do we need scores this call?" would save nothing
computationally but would create a graph break that splits the compiled kernel, doubling
compilation time and doubling compiled cache memory.

The consequence: every forward pass produces scores, but not every forward pass produces
*valid* scores. Validity is determined by the adapter_scales that were active during the
forward. A rollout step with `adapter_scales=[1, 0]` (policy on, BTRM off) produces
valid policy denoising and arbitrary scores. The scores exist as tensors — they're just
not semantically meaningful BTRM reward predictions because the hidden states were shaped
by policy weights, not the reward model weights that were trained to interpret those
hidden states.

This means "score the rollout finals" is not "run the score head" (which already happened
during rollout). It is "run a forward pass on the clean final latent with
`adapter_scales=[0, 1]` so that the BTRM adapter shapes the hidden states correctly for
the trained score projection." It is a real NFE, not a free readout.

## 3. Scatter-Gather Is the Only Calling Convention

The iron mandate: there is never more than one way to call a resource. All model
evaluations — inference rollout steps, BTRM reward scoring, reference model anchoring,
gradient accumulation replay — are entries submitted to the BatchExecutor as queries
with fork specs. The executor bins them by sequence length into REFERENCE_TOTAL_LEN-sized
packs, runs each bin as a single compiled forward pass with FlexAttention block masks
preventing cross-image attention leakage, and returns tagged results.

There is no serial `for rollout in rollouts: score(rollout)` loop. There is
`executor.execute(scoring_queries)` where `scoring_queries` is a list of dicts, each
with a `base_latent`, a `sigma` (0.0 for clean finals), a `conditioning`, and an
`adapter_scales` vector. The executor decides how many forward passes this costs based
on bin packing. For 2 scoring queries at 512×512, they might fit in a single bin (one
forward pass). For 8 scoring queries at 1024×1024, they might need 4 bins. The caller
doesn't know and doesn't care.

A separate `score_serial()` call — which creates a throwaway BatchExecutor, builds its own
plan cache, runs one entry per forward pass, and operates in a different torch.compile
context — violates this convention three times. It duplicates the scheduling infrastructure,
bypasses the bin-packing efficiency, and creates a gratuitous compiled graph variant
that consumes ~700MB of GPU memory permanently.

## 4. Adapter Scales Are a Gradient Gate

When `adapter_scales[j] = 0.0` for adapter j during a forward pass, the contribution
of adapter j to the output is `0.0 * (x @ A_j^T @ B_j^T) = 0`. The chain rule through
the multiplication by zero means `d(loss)/d(A_j) = 0` and `d(loss)/d(B_j) = 0`. The
adapter receives exactly zero gradient. This is not a side effect — it is the mechanism
by which adapters are selectively trained.

In DDGRPO, the gradient accumulation step replays trajectory checkpoints through the
model to compute `∇_θ log π_θ(x_{t-1} | x_t)`. Only the policy adapter should receive
gradients. The adapter_scales for this forward must be `[policy=1, btrm=0]`. If both
scales were 1.0, the BTRM adapter would receive gradients from the policy objective —
gradients that push the reward model weights toward generating higher-reward images
rather than toward predicting rewards accurately. This silently corrupts the reward
model that the policy is optimizing against.

Conversely, during BTRM training (supervised Bradley-Terry loss on preference pairs),
the adapter_scales must be `[policy=0, btrm=1]` so that only the reward adapter receives
gradients. If the policy adapter's scale were nonzero, it would receive gradients from
the BTRM loss, which would push it toward producing images that match the preference
labels rather than images that maximize the learned reward.

The scale vector is not a hyperparameter to tune. It is a logical specification of which
parameters participate in which optimization objective. Getting it wrong doesn't produce
a training error — it produces a silently corrupted model that passes all shape checks
and dtype checks while learning the wrong thing.

## 5. Work Scheduling vs. Control Flow

The difference between a correct and an incorrect DDGRPO implementation is not in the
math (the REINFORCE gradient estimator, the advantage normalization, the log-probability
computation). It is in the work scheduling.

An incorrect implementation treats "generate rollouts," "score rollouts," and "accumulate
gradients" as three sequential phases with different calling conventions, different
executors, and different compilation contexts. Each phase loads work into the GPU
differently, causing torch.compile to generate separate compiled kernels for each phase.
The compiled cache grows by one entry per phase per transformer block. Activation
retention policies differ between phases (inference_mode for rollouts, grad-enabled
without checkpointing for scoring, grad-enabled with checkpointing for gradient
accumulation). The peak memory is the sum of the compiled cache for all phases plus the
worst-case activation retention from whichever phase is most wasteful.

A correct implementation treats all three as work items with different adapter_scales
submitted to the same executor. Rollout steps are queries with `[1, 0]` at the current
sigma. Scoring queries are entries with `[0, 1]` at sigma=0 on clean finals. Gradient
accumulation replays are entries with `[1, 0]` through the same executor (but now with
gradients enabled and checkpointing on). The executor bins them, the model runs them,
the results come back tagged. The scheduling code never inlines algorithmic code. The
scoring query doesn't know it's "scoring" — it's a forward pass with specific inputs
and scales, and the caller interprets the output.

This is the content of the iron mandate that scheduling code never inlines algorithmic
code, and the 6-tuple spec's requirement that SCATTER maps over an arbitrary branch set
and GATHER reduces it. The branch count doesn't appear in the code. Neither does the
distinction between "rollout step" and "reward scoring" — they are both entries in a
batch, differentiated by their metadata (sigma, adapter_scales, conditioning), not by
which code path they take through the system.

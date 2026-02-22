# K-Tuple Scheduling Algorithm Notes

Working notes from interactive dialogue, 2026-02-21. Not a spec — a record
of reasoning toward a spec.

## The problem

A diffusion sampling loop must schedule K guidance queries per Euler step
through a model service. K varies per query (K=1 for uncond, K=2 for
standard cfg, K=6+ for multi-direction guidance). The model service
executes packed FlexAttention forwards with finite context length.

Previous code (`rollout.py`, then `ktuple_sampling.py`) conflated the
sampling loop with the model's execution internals — calling
`prepare_packed_forward` and `packed_forward` directly, building block
masks, managing RoPE caches. This is wrong because the sampling loop is
a **client** to a model service, not an owner of the model.

## Why the sampling loop is a client, not an owner

The acc:23 case makes this clear. 23 accelerators each have the model
loaded. The sampling loop doesn't own 23 models — it has a work queue
and submits entries to a service that knows how to execute them.

In the acc:1 case the decomposition is identical. The model is still a
service. They happen to be colocated. The data flow doesn't change.

Consequence: the sampling loop should not know about block masks, RoPE,
packing_info, bin-packing strategy, or how entries get dispatched to
accelerators. It submits work and receives results.

## NFEs vs FlexAttention launches

An NFE (neural function evaluation) is one denoising of one latent with
one conditioning. For a query with K=6 across 100 steps, that's 600 NFEs.

A FlexAttention launch is one packed forward pass. Multiple NFEs can share
a launch if their entries fit in the context budget. 10 small images at
256x256 might share one launch. One large image at 1024x1024 fills an
entire launch by itself.

NFE count is determined by the sampling spec (K * n_steps * n_queries).
Launch count is determined by bin-packing. They are independent quantities.

## The 6-tuple context length problem

A 1024x1024 image is ~4608 tokens (4096 latent + 512 text). The reference
context length is 4224. A single full-res image barely fits. Two full-res
images (K=2 cfg) need ~9216 tokens — that's already 2 launches minimum.
K=6 at full resolution is 6 launches per step.

`ktuple_sampling.py` calls `prepare_packed_forward` with all K entries and
`packed_forward` once per step. This is only correct when all entries fit
in one launch — which is the K=2 same-resolution case that was the only
tested configuration.

## What the client submits

The client does not submit tensors. The client submits **fork-and-mutate
functions** alongside **literal tensors**.

A query with K=3 cfg is:
```
literals: (text_tensor, image_latent)
forks:
  entry_0: identity()              -- base, scale=1.0
  entry_1: swapcond("a shrimpy painting"), scale=+3.0
  entry_2: swapcond(""), scale=-6.0
```

The service receives fork descriptors, applies them to the literals,
bins the resulting entries, executes, and returns results tagged by
(query_id, entry_id). The client never materializes the 66 entries.
SCATTER is service-side.

Mutation functions have no limit to their scope — they are our way of
describing the ability to script or program arbitrary tensor operations
on a forked input. Everything not explicitly changed by a mutation
function is identity-broadcast from the forked input. Adapter scales
are identity-broadcast if no mutation function changes them. A mutation
function *could* change them.

## What the client receives

Results tagged by (query_id, entry_id). All scores returned — every
entry in every launch produces a score, and the client decides which
are meaningful.

The client has no way of knowing what literal kernel calls produced
the tensors it gets back. Bins, launches, accelerator assignment,
block masks — all invisible.

Results don't need to arrive all at once. The client only needs to
know when it has all K results for a given query to GATHER. Different
queries can GATHER independently as their results stream in.

## Correct function composition

### Per-step, from the client's perspective

```
submit(fork_specs, literals, step_i)   -- 12 queries -> service
results = await(tagged_results)         -- 66 tagged (field, score) back
guided  = GATHER(results, specs)        -- 66 results -> 12 guided
base_latents = EULER(...)               -- client-side ODE step
```

SCATTER, PACKSOLVE, EXECUTE are all service-side. The client does
GATHER and EULER.

### Per-step, from the service's perspective

```
entries = SCATTER(literals, fork_specs) -- apply mutations -> 66 entries
bins    = PACKSOLVE(entries)            -- 66 entries -> N bins
results = EXECUTE(bins, model)          -- N launches -> 66 (field, score)
return tagged(results)                  -- tag by (query_id, entry_id)
```

### What changes with accelerator count

Nothing in the composition. PACKSOLVE targets N ~ acc_count for
parallelism. EXECUTE dispatches bins across accelerators. The
client-service interface is unchanged.

### What is constant across steps

Fork specs, resolutions, conditionings, text lengths, bin assignments.
Only latent tensor values change per step. The service can establish a
plan on the first submission and reuse packing layout, block masks,
RoPE, and compiled graphs for subsequent steps.

## Tensor lifecycles

| Tensor | Lifetime | Owner |
|--------|----------|-------|
| Base latents | Persistent across steps | Client |
| Text conditionings | Constant, submitted once | Literals (plan) |
| Scattered entries | Ephemeral, one step | Service |
| Fields/scores | Ephemeral, one step | Service -> Client |
| Guided estimates | Ephemeral, one step | Client |
| Model weights | Immutable | Service |

The only mutable state the client holds across steps is the base latents.

## Client-service boundary

**Client (sampling loop) owns:**
- Work queue (queries with fork specs)
- Base latents (persistent mutable state)
- GATHER (reduce K results to guided estimate)
- EULER (ODE step from guided estimate)
- Trajectory persistence
- Sigma schedules (EULER is client-side, sigmas are EULER's concern)

**Service (model executor) owns:**
- Model weights, compilation, backend selection
- SCATTER (apply fork-and-mutate to literals)
- PACKSOLVE (bin-packing strategy)
- EXECUTE (packed forward dispatch)
- Block masks, RoPE, packing_info
- Per-entry sigma derivation from resolution metadata
- Multi-GPU data parallel orchestration (if acc > 1)
- Hardware-dominated algorithmic details: tile sizes for block mask
  construction, virtualized batch layout, tensor core alignment

The service is ignorant of *why* clients submit work. It is a staid
integrator over submitted work items. It doesn't know about CFG,
guidance scales, or sampling strategies. It receives fork specs,
applies them, packs the results into efficient kernel launches
determined by hardware constraints (tensor core tile sizes, context
length budgets, accelerator count), and returns tagged results.

## What ktuple_sampling.py gets wrong

1. Calls `prepare_packed_forward` — reaches across the boundary.
2. Calls `packed_forward` directly — bypasses scheduling.
3. Assumes all entries fit in one launch — breaks at K>2 for large images.
4. Does SCATTER client-side (materializes all K entries as tensors).
5. Threads block_mask, packing_info, RoPE through the sampling loop.
6. Had `force_sdpa`/`ensure_sage` — backend is service's concern.
   (Deleted this session.)

## What ktuple_sampling.py gets right

1. The gather-then-euler decomposition.
2. Spec tuples `(conditioning, resolution, scale)` as work description
   (close to fork-and-mutate, but not quite — it sends tensors not
   mutation functions).
3. The `solve` loop structure (step iteration with save_fn).
4. `gather_fn` as a pluggable reduction strategy.

## What needs to happen

The sampling client submits fork-and-mutate specs to a batch endpoint
on the existing FastAPI server. The server does SCATTER, PACKSOLVE,
EXECUTE. The client does GATHER, EULER.

`ktuple_sampling.py` should become a pure client: it knows about specs,
GATHER, EULER, and the solve loop. It does not import `forward_packed`,
`block_mask`, or any model internals.

The server gets a `/batch_forward` or equivalent endpoint that accepts
fork specs + literals and returns tagged (field, score) results.

## Implementation decomposition

### Layer A: Server batch forward endpoint (fork specs from day one)

The server accepts fork specs + literal tensors, not materialized entries.
SCATTER is server-side from the start. The endpoint:

- Receives: list of (literal_latent, literal_cond, fork_specs[]) per query
- Applies fork-and-mutate to produce entries
- Packs entries into launches (NEW packing strategy — the existing
  bin_packer.py was built for BTRM training pairs, different constraints)
- Executes launches
- Returns: tagged (query_id, entry_id, field, score) results

The packing strategy for inference scheduling is different from BTRM pair
packing. BTRM packing optimizes for training throughput (FLOPS budget,
cross-resolution pairs). Inference packing optimizes for latency (minimize
launches per step) or parallelism (target launches = acc_count). New
packing code needed.

### Layer B: Sampling client rewrite

`ktuple_sampling.py` becomes a pure client:
- Imports: triumphant_future_reduction_ops (gather, euler_step, denoise_all)
- Imports: sigma_schedule (build schedules, resolution_shift)
- Does NOT import: forward_packed, block_mask, model internals
- Submits fork specs per step to server endpoint
- Receives tagged results
- Does GATHER, EULER, trajectory persistence

### Layer C: Validation (no unit tests)

Every intermediate tensor persisted to disk. Validation is continuous
measures saved as JSON + images, not pass/fail asserts.

Three integration cases:

**K=1**: Single image, no guidance. `cfg1(cond, res)`. Produces an image.
Pixel statistics in-distribution. Baseline.

**K=2**: Standard cfg. Same seed/prompt/cfg as known-good generation.
Output within SageAttention noise floor of known-good. Also: K=2 with
`gather_residual_gain(gain=6.0)` must approximately equal standard cfg
with scale=7.0 (mathematical identity for K=2).

**K=6**: The forcing function. Shrimp-banana multi-resolution 6-tuple
with `gather_residual_gain`:
- Entry 0: base prompt, 1024x1024, scale=1.0 (base trajectory)
- Entry 1: shrimp emphasis, 1024x1024, scale=+3.0 (attractive)
- Entry 2: typography emphasis, 1024x1024, scale=+2.0 (attractive)
- Entry 3: base prompt, 512x512, scale=-2.0 (repulsive mid-res)
- Entry 4: base prompt, 256x256, scale=-1.5 (repulsive low-res)
- Entry 5: banana poem, 1024x1024, scale=-4.0 (negative)

This CANNOT work with current code (6 full-res entries don't fit in
one launch). If it produces a coherent image with visible shrimp
emphasis and sharper detail than K=2, the implementation works.

Validation measures per case:
- VAE-decoded final image (persisted PNG)
- BTRM scores per entry per step (JSONL)
- Packing diagnostics: launches per step, bin utilization (from server)
- Pixel statistics: mean, std, histogram
- Cross-K comparison: K=6 vs K=2 vs K=1 on same prompt/seed

## Current misimplementations

### Wrong boundary: client swallows service

`ktuple_sampling.py` does SCATTER → PACKSOLVE → EXECUTE → GATHER → EULER
all in one process, one call stack, one module. There is no client-service
boundary. The client imports `prepare_packed_forward` to build block masks
and RoPE, which are hardware-dominated service internals.

`btrm_lifecycle.py`'s `score_packed` does the same: imports
`pad_to_patch_size`, computes image sizes, calls `prepare_packed_forward`,
calls `model()` directly. A client function that has swallowed the service.

`forward_packed.py` is positioned as if it were the service layer but it's
just `model()` with extra steps. It doesn't schedule, doesn't bin-pack,
doesn't manage work queues.

### Wrong boundary: server swallows client

`dataset_generator.py` talks to `server.py` via HTTP but submits complete
generation requests. The server owns the entire sampling loop — Euler steps,
sigma schedules, CFG. The client-service boundary is inverted: the server
does what the sampling client should do.

### No batch interface

`server.py` has per-request endpoints. One generation at a time. No batch
submission. The bin packer exists (`bin_packer.py`) but nothing in the
serving path calls it — it was built for BTRM training pair construction.

### Summary

The boundary is wrong in two opposite ways depending on code path:
- ktuple_sampling, btrm_lifecycle: client IS the service (no boundary)
- dataset_generator → server: server IS the client (boundary inverted)

## What must change

### Server: batch forward endpoint

One endpoint: "here are N fork specs with literals, return N tagged
(field, score) results." One step's worth of NFEs. The server applies
mutations, bin-packs, executes across however many accelerators it has,
returns tagged results. It doesn't know about Euler steps, sigma
schedules, or CFG scales.

### Server: plan/session reuse

On first submission, the server sees fork specs and establishes a packing
plan (bin assignments, block masks, RoPE, compiled graphs). On subsequent
steps with the same fork specs, it reuses the plan and only receives
updated latent values. This is the "constant across steps" optimization.

### Client: pure sampling loop

The sampling client holds base latents, submits fork specs per step,
receives tagged results, does GATHER and EULER. It imports
`triumphant_future_reduction_ops` for gather/euler/denoise. It does NOT
import `forward_packed`, `block_mask`, `prepare_packed_forward`, or any
model internals.

### forward_packed.py: internal to server

Stops being a public interface. Becomes an implementation detail of the
server's EXECUTE step. No external callers import it.

## What is fundamentally different from published image generation code

Every public diffusion inference codebase — ComfyUI, diffusers, A1111,
InvokeAI — treats each image as an independent serial job. CFG is
hardcoded as "run pos and neg, subtract." The model forward is a black
box called once per image per step.

### FlexAttention block-masked packing

Multiple images share one transformer forward with zero cross-image
attention leakage. Not trivial B>1 batching — sequence packing with
compile-time block-diagonal masks through custom Triton kernels. The
closest analogue is vLLM's paged attention for LLM serving: same
problem (variable-length sequences sharing one forward), different
domain. The masked SageAttention kernels have full backward pass
support — packing works for training, not just inference.

### Score head inside the forward pass

ZImageRLAIF returns (diffusion_fields, scores) from every forward.
No separate scoring pipeline. The BTRM reward signal is a free rider
on every NFE. No published code integrates reward modeling into the
denoising forward.

### One path parameterized by K

There is no `if do_cfg:`. K=1 (no guidance) is `[(cond, res, 1.0)]` —
one entry through the same scatter-gather-euler path. K=2 (standard cfg)
is two entries. K=6 is six entries. The service sees fork specs and
entry counts. It doesn't know what "CFG" means.

Published codebases branch on whether CFG is enabled. We don't branch.
cfg1, cfg2, cfg6 are different parameterizations of the same function.

### Autograd through guided sampling (RL / LCM)

REINFORCE over a guided trajectory needs the log-probability of the
trajectory under the policy. The trajectory was generated under guided
denoising — the gather step combined K denoised estimates into the
actual step taken. Scatter and gather must be differentiable or you
can't compute the policy gradient through the guided trajectory.

Scatter is clone/interpolate. Gather is weighted sum of residuals.
Both are standard differentiable tensor ops. Autograd works for any K
with zero additional implementation — it's a consequence of the ops
being branchless tensor math, not control flow.

The service boundary holds for training: the packed forward with masked
Sage has register_autograd on all custom ops, so returned fields carry
grad_fn. The client's GATHER and EULER are pure differentiable tensor
math. The full trajectory from noise to final latent is differentiable
through arbitrary K-tuple specs without the service knowing gradients
will flow through its results.

### Custom FP8 + INT8 precision pipeline with full autograd

23 custom Triton kernels. FP8 weights, INT8 SageAttention QK, FP32
accumulation, BF16 I/O. register_autograd on all 12 custom ops. Not
"call torch.float8" — a hand-written precision pipeline where every
kernel knows its dtypes and the backward pass is analytically correct
through quantized attention.

### Accelerators as a resource pool

The design targets acc:23 the same way vLLM targets multi-GPU serving.
The service schedules work across N accelerators by FLOPS budget, not
"each GPU gets one image." This is serving infrastructure.

### The responsibility surface

We own the kernels, the packing, the masking, the precision pipeline,
the backward pass, the reward model integration, and the scheduling.
Published code outsources all of this to PyTorch defaults.

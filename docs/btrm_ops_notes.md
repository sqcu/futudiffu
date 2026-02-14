# BTRM Dataset Generation: Operational Notes

Observations from implementing `generate_btrm_dataset.py`. Each section describes
a real regression that was introduced and fixed during development.

## torch.compile wraps, doesn't copy

We tried compiling the diffusion model twice (once per attention backend), expecting
`torch.compile` to snapshot the global `_ATTENTION_BACKEND` into each compiled graph.
It doesn't -- the global is read at call time, not trace time. A single compile plus
a runtime `set_attention_backend()` call before each forward pass is correct.
Compiling twice wastes ~2GB VRAM on a duplicate graph cache that dispatches to the
same underlying parameters.

## Compile only models that run many times

`torch.compile` has fixed overhead from Triton autotune, graph capture, and code
generation. For models that run O(1) times per session (VAE decode for rendering a
few trajectories), the overhead exceeds savings. The diffusion model runs 30 steps
per trajectory across hundreds of trajectories, so compiling it pays off. The VAE
does not.

## Encode the negative prompt once

The first version of the TE encoding loop re-encoded the negative prompt (always
`""`) for every positive prompt, producing 24 identical tensors. Since the negative
is constant, encoding once and sharing the reference saves both time and memory.
Under `torch.inference_mode()`, outputs are already non-grad -- calling
`.detach().clone()` on them allocates a full copy for no reason.

## Model lifetime: load, use, free

On a 24GB GPU the budget is roughly: FP8 diffusion ~8GB (weights + compiled
graphs), BF16 text encoder ~7.5GB, VAE ~320MB. Loading all three simultaneously
leaves too little room for activations. The correct pattern is sequential phases:
load TE, encode all prompts, free TE; load diffusion, run all sampling, free
diffusion; load VAE if renders are needed.

## Always use FP8 when available

The FP8 diffusion model (5.8GB on disk) fits in VRAM alongside the text encoder
during the handoff between phases. The BF16 model (12GB) does not. Loading BF16
when FP8 is available is always wrong unless explicitly benchmarking precision
differences.

## fuse_model() before torch.compile()

`fuse_model` mutates the model in-place (replaces modules with fused variants,
pre-batches adaLN, sets up QKV postprocess). These mutations must be visible to the
Dynamo tracer. Compiling first and fusing after means the compiled graph captures
the unfused ops, missing all fusion speedups.

## RNG determinism across resume

When a "preview" pass mirrors the generation RNG sequence (to collect unique prompts
before encoding), both passes must consume exactly the same draws per trajectory. A
stray `rng.randrange()` call in one path but not the other desynchronizes all
downstream seeds, breaking resume determinism. The server/client split eliminates
this class of bug: the client owns all RNG, the server consumes none.

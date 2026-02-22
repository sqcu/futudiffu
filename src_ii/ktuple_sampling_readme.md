# ktuple_sampling

K-tuple guided Euler sampling. Pure client — zero model internals.

SCATTER, PACKSOLVE, and EXECUTE are server-side. This module does
GATHER and EULER only.

---

## Executor interface

Every function in this module takes an `executor` as first argument:

```python
executor(
    x_bases: list[Tensor],           # K base latents, one per query
    specs: list[list[tuple]],        # K specs, each a list of (cond, (w,h), scale)
    step_i: int,                     # current step index
    adapter_scales: Tensor | None,   # (K, n_adapters) or None
) -> (
    denoised_per_query: list[list[Tensor]],  # K lists, each with per-entry denoised
    scores: Tensor | None,                    # BTRM scores from model, or None
)
```

The executor applies fork-and-mutate (SCATTER) to base latents using the
spec tuples, packs entries into launches, executes, runs denoise_all on
its side, and returns denoised estimates per entry. The client never sees
scattered latents, block masks, packing_info, or RoPE.

The server is free to reuse packing plans, compiled graphs, and block masks
across steps — the spec structure is constant, only latent values change.

---

## Four functions

**`step(executor, x_bases, specs, query_sigmas, step_i, adapter_scales, gather_fn)`**
`-> (x_next, scores, guided_list)`

Submits one step to the executor, receives denoised estimates per query,
does GATHER and EULER. `query_sigmas` is one (n_steps+1,) sigma schedule
per query (base resolution). `gather_fn(denoised_list, spec) -> Tensor`
or None for default linear gather. Use
`functools.partial(gather_residual_gain, gain=7.0)` for spherical reduction.

**`solve(executor, x_bases, specs, query_sigmas, n_steps, adapter_scales, gather_fn, save_fn)`**
`-> (x_final, scores_all)`

The Euler loop. Resolves callable adapter_scales per step. Calls
`save_fn(step_i, x_pre, guided)` after each step if provided.
Returns x_final (list of K latents) and scores_all (list of per-step
score tensors, may contain None if executor returns no scores).

**`batch_rollout(executor, pos_conds, neg_conds, cap_lens, seeds, resolutions, n_steps, cfg, ...)`**
`-> (trajectories, metadata)`

Convenience wrapper. Builds specs from pos/neg conds + cfg scalar,
builds query sigma schedules, inits noise, runs solve, packages trajectories.
`cfg != 1.0` uses cfg2; `cfg == 1.0` uses cfg1 (single entry, no guidance).
Returns list of K trajectory dicts and a metadata dict.

**`spec_rollout(executor, spec, cap_lens, seed, n_steps, ...)`**
`-> (trajectory, metadata)`

Pre-built spec rollout. Uses master noise field + aperture for correlated
multi-resolution initialization: the 256x256 branch sees the center crop
of the same noise that the 1024x1024 branch sees. Step 0 uses aperture
init; subsequent steps use executor-side SCATTER from the base latent.

---

## Query sigma convention

`query_sigmas[k]` is the sigma schedule for query k at its **base resolution**
(spec entry 0). It is used only for EULER. The server derives per-entry sigmas
from spec resolutions itself; those are not exposed to the client.

In `spec_rollout`, entry_sigmas from `build_per_image_sigmas` are computed
client-side to initialize the noise magnitude (step 0 sigma). Only entry 0's
schedule is used as query_sigmas. The server handles per-entry sigma derivation
for entries 1..K-1.

---

## Sign convention

- Server: field = -img (model output sign), `denoised = x - field * sigma`
- GATHER: `base + sum(scale_i * (denoised_i - base))`
- EULER: `x + ((x - guided) / sigma) * dt`
- Negative scale = repulsive residual (push away from that branch's denoised)

---

## adapter_scales convention

- `None`: no adapter routing
- `Tensor (K, n_adapters)`: one row per query, passed to executor as-is
- `Callable(step_i) -> Tensor | None`: resolved per step in `solve()`

The executor decides how to expand per-query scales to per-entry scales
(e.g. broadcast each row to all entries in that query's spec).

---

## Forbidden imports (not in this file)

- `src_ii.forward_packed` — server-internal
- `src_ii.block_mask` — server-internal
- `src_ii.attention` — server-internal
- `futudiffu.*` — frozen
- Any model class

For testing, write a trivial executor in the test script that calls
`forward_packed` directly. That import belongs in the test, not here.

---

## Spec tuple format

`spec: list[tuple[Tensor, tuple[int,int], float]]`

Each entry: `(conditioning_tensor, (width_px, height_px), scale)`

- Entry 0: base trajectory (scale = 1.0 by convention)
- Entries 1+: guidance branches (positive scale = attractive, negative = repulsive)

Factory helpers in `triumphant_future_reduction_ops`:
- `cfg1(cond, res)` — single entry, no guidance
- `cfg2(pos, neg, res, scale)` — standard CFG
- `cfg6(base, shrimp, typo, banana, base_res, lr1, lr2, scales)` — 6-tuple

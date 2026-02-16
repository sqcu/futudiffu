# server.py Refactor Plan

Prioritized from code review. Each item should be checked off by the refactor agent
and validated by the test agent. Skip item 16 (print-based logging).

## CRITICAL

### C1: Inlined Euler loop in `handle_sample_trajectory_packed`
- **Location**: `server.py:349-393`
- **Problem**: Packed handler reimplements the full CFG + denoising + euler step
  loop inline. `handle_sample_trajectory` correctly delegates to `sample_euler`
  in `sampling.py`, but the packed handler has its own copy of the same math.
  Any change to sampling (bug fix, new sampler) must be applied in two places.
- **Fix**: Create `sample_euler_packed` in `sampling.py` that takes the packed
  forward callable, list of x tensors, sigmas, cfg, and a save callback. The
  server handler calls it the same way the unpacked handler calls `sample_euler`.
- **Cross-ref**: `sampling.py:make_cfg_model_fn`, `sampling.py:sample_euler`,
  `sampling.py:const_calculate_denoised`, `sampling.py:to_d`
- **Status**: [x]

### C2: `_run_backbone_hidden` reimplements `HiddenCapture`
- **Location**: `server.py:711-757`
- **Problem**: Third copy of the forward-hook-on-last-block hidden state capture
  pattern. Same logic exists in `training_utils.py:HiddenCapture` (lines 70-98)
  and `btrm.py:BTRMWrapper._run_backbone` (lines 339-370).
- **Fix**: Import and use `HiddenCapture` from `training_utils.py`. The method
  becomes ~10 lines: set attention backend, move tensors, create HiddenCapture,
  install, forward, get, remove.
- **Cross-ref**: `training_utils.py:70-98`, `btrm.py:339-370`
- **Status**: [x]

## HIGH

### H1: Inlined conditioning padding in packed handler
- **Location**: `server.py:273-286`
- **Problem**: Reimplements per-image pos/neg conditioning padding that already
  exists as `pad_and_batch_cond` in `sampling.py:107-129`.
- **Fix**: Call `pad_and_batch_cond(pos_i, neg_cond)` in the loop body instead
  of manually padding with `F.pad`.
- **Cross-ref**: `sampling.py:pad_and_batch_cond`
- **Status**: [x]

### H2: Packed handler is 155 lines
- **Location**: `server.py:213-394`
- **Problem**: Far too large for a thin dispatch handler. Should be ~30 lines
  after extracting sampling into sampling.py (C1), padding into existing
  function (H1), and trajectory state prep into a shared helper (M1).
- **Fix**: This is the aggregate result of C1 + H1 + M1. After those are done,
  verify the handler is under 40 lines.
- **Depends on**: C1, H1, M1
- **Status**: [x]

### H3: `handle_train_btrm_step` inlines loss computation (86 lines)
- **Location**: `server.py:841-946`
- **Problem**: Reimplements a simpler version of `compute_multihead_loss` from
  `btrm.py:186-282` instead of calling it. The per-head split, Bradley-Terry
  all-pairs, logsquare regularizer, and accuracy computation are all duplicated.
- **Fix**: Call `compute_multihead_loss` from `btrm.py`, or extract the common
  loss logic into a shared function. Handler becomes: forward examples, stack
  scores, call loss function, backward, step, return metrics.
- **Cross-ref**: `btrm.py:compute_multihead_loss`
- **Status**: [x]

### H4: `handle_accumulate_policy_gradients` is 105 lines
- **Location**: `server.py:948-1052`
- **Problem**: REINFORCE math (concurrent-batch LoRA pi/ref, per-step forward,
  MSE log-ratio, advantage-weighted backward) is trapped inside an RPC handler.
  Untestable without a full ZMQ server.
- **Fix**: Extract the REINFORCE log-ratio computation into `training_utils.py`
  (e.g., `compute_reinforce_log_ratio(model, x_t, sigma, conditioning,
  rope_cache, multiplier)` returning `(log_ratio, loss_contribution)`). Handler
  sets up LoRA scale, calls the function per step, cleans up.
- **Cross-ref**: `training_utils.py`
- **Status**: [x]

### H5: Dead code in packed handler -- load then delete clean_latent
- **Location**: `server.py:301-307`
- **Problem**: Loads `clean_latent_{i}` from tensors, assigns to `clean`, then
  immediately `del clean`. The same tensor is re-loaded 15 lines later at
  318-322. The first load is wasted work from an incomplete refactor.
- **Fix**: Delete lines 301-307 entirely.
- **Status**: [x]

## MEDIUM

### M1: Shared trajectory state preparation
- **Location**: `server.py:123-187` (unpacked) and `server.py:233-310` (packed)
- **Problem**: Both handlers duplicate parameter extraction, tensor device moves,
  RoPE cache setup, noise generation, sigma schedule construction, and initial
  latent preparation. ~60 lines of near-identical boilerplate.
- **Fix**: Extract `prepare_trajectory_state(params, tensors, model_manager)`
  returning `(x, sigmas, rope_cache, steps_to_save, ...)`. Both handlers call it.
- **Status**: [x] (overlap reduced by C1, H1, H5; remaining shared lines are
  minimal -- packed handler has inherent packing-specific complexity)

### M2: `inference_mode` vs `no_grad` inconsistency
- **Location**: `server.py:747` vs `server.py:835`
- **Problem**: `_run_backbone_hidden` uses `torch.inference_mode()`, but the
  BTRM training handler expects returned hidden states to flow through the
  gradient graph. Works only because backbone is frozen and only head params
  need gradients. Fragile if backbone unfreezing is ever needed.
- **Fix**: Add `requires_grad` parameter to `_run_backbone_hidden` (or to the
  HiddenCapture usage after C2 is done). Training handler passes True, scoring
  handler passes False.
- **Depends on**: C2
- **Status**: [x]

### M3: RoPE cache computation duplicated 6 times
- **Location**: `server.py:157-161`, `290-291`, `461-465`, `506-507`, `732-737`, `1002-1005`
- **Problem**: Same 4-line RoPE setup pattern appears 6 times.
- **Fix**: Extract `self._make_rope_cache(H, W, num_tokens)` method on
  `InferenceServer`. One definition, six call sites.
- **Status**: [x]

### M4: `set_adapter_config(frozen=False)` silently does nothing
- **Location**: `server.py:696-698`
- **Problem**: Only `frozen=True` is handled. `frozen=False` falls through.
- **Fix**: Either implement the unfreeze path or validate that `frozen` is only
  `True` or `None` and raise on `False`.
- **Status**: [x]

### M5: Scale-clearing logic unreachable from client
- **Location**: `server.py:691-694`
- **Problem**: Server handles `{"scale": null}` to clear, but client never sends
  it (client omits `scale` key when `scale is None`).
- **Fix**: Add `clear_scale: bool = False` to client's `set_adapter_config`.
  When True, pass `"scale": None` explicitly. Or remove the dead server-side
  code path.
- **Status**: [x]

### M6: `handle_dump_all_loras` has 40 lines of file I/O
- **Location**: `server.py:1140-1216`
- **Problem**: File I/O, timestamping, JSON manifest, BTRM head dumping are
  all non-RPC concerns embedded in a handler.
- **Fix**: Extract dump logic into a utility function (e.g., in `lora.py` or
  new `persistence.py`). Handler becomes a thin wrapper.
- **Status**: [x]

## LOW

### L1: Packed handler doesn't accept `attention_backend`
- **Location**: `server.py:213-394` vs `client.py:151-217`
- **Problem**: Unpacked handler configures attention backend. Packed handler
  does not -- uses whatever was last set. Implicit and fragile.
- **Fix**: Add `attention_backend` to packed handler and client method.
- **Status**: [x]

### L2: `torch.nn.functional` import only used for duplicated padding
- **Location**: `server.py:22`
- **Problem**: `import torch.nn.functional as F` only used in packed handler
  for `F.pad` (the duplicated conditioning padding from H1).
- **Fix**: Remove after H1 is done (padding calls replaced with library fn).
- **Depends on**: H1
- **Status**: [x]

### L3: `handle_free("bogus")` returns success
- **Location**: `server.py:1122-1138`
- **Problem**: Unrecognized model name silently succeeds.
- **Fix**: Add `else` clause returning error response.
- **Status**: [x]

### L4: Warmup hardcodes 832x1280
- **Location**: `server.py:455`
- **Problem**: `latent_h, latent_w = 104, 160` hardcoded. Different resolution
  workloads still get cold-compiled on first real inference.
- **Fix**: Accept optional `width`/`height` in warmup RPC, defaulting to
  832x1280. Update client to pass them.
- **Status**: [x]

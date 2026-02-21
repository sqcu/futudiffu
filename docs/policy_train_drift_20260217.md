# Policy vs Training Implementation Drift Analysis

_2026-02-17. Source: automated code audit of sampling.py, training_utils.py, train.py, client.py, server.py._

## Executive Summary

Five divergences between the rollout (data generation) path and the training
(REINFORCE + BTRM) path. Two are correctness bugs that produce wrong gradients.

| # | Divergence | Severity | Status |
|---|-----------|----------|--------|
| 1 | CFG vs no-CFG in REINFORCE | **CRITICAL** | Open |
| 2 | Checkpoint is x_{t+1}, evaluated at sigma_t | MEDIUM | Open |
| 3 | BTRM head never sees neg_cond | LOW-MEDIUM | Open |
| 4 | score_btrm uses pos_cond only | LOW | Open (consistent with #3) |
| 5 | Context refiner sees different B=2 content | LOW | Open |

---

## 1. CFG vs No-CFG in REINFORCE (CRITICAL)

### Rollout path (sampling.py `sample_euler_packed`, L174-242)

```
x_cfg = [x_i.expand(2, -1, -1, -1)]   # B=2
# refined_caps[0] = refined(pos_cond), refined_caps[1] = refined(neg_cond)
outputs = packed_forward_fn(x_cfg, t_batch, refined_caps, ...)
out_cond, out_uncond = outputs[img_i].chunk(2, dim=0)
denoised_cond = calculate_denoised(sigma, out_cond, x)
denoised_uncond = calculate_denoised(sigma, out_uncond, x)
denoised = denoised_uncond + cfg * (denoised_cond - denoised_uncond)  # CFG
```

- `refined_caps` built from `pad_and_batch_cond(pos_cond, neg_cond)` -- proper CFG conditioning
- The euler step uses CFG-combined denoised prediction
- The saved checkpoint x_{t+1} is shaped by CFG

### Training path (training_utils.py `compute_reinforce_step`, L297-377)

```
# _prepare_packed_single_state calls:
pad_and_batch_cond(conditioning, conditioning)  # pos_cond DUPLICATED
# Both batch elements see identical positive conditioning
outputs = diff_compiled_packed([x_cfg], t_batch, refined_caps, ...)
return outputs[0][:1]  # Take batch element 0 only; NO CFG computation
```

- `refined_caps[0]` = `refined_caps[1]` = refined(pos_cond)
- No CFG combination: raw model(x_t, sigma, pos_cond) output
- Client (`train.py` L601-607) sends only `pos_cond[:1]`, never neg_cond

### Impact

The REINFORCE log-ratio `log(pi/ref)` is:
```
diff = pi_denoised - ref_denoised
log_ratio = -MSE(diff) / (2 * sigma^2)
```

Both `pi_denoised` and `ref_denoised` are computed WITHOUT CFG, on a latent
that was generated WITH CFG. The gradient optimizes the policy to match the
reference in the pos_cond-only regime, not in the CFG regime that actually
produced the trajectories and rewards.

### Fix

1. Pass `neg_cond` and `cfg` through the RPC chain to `compute_reinforce_step`
2. Build proper CFG conditioning: `pad_and_batch_cond(pos_cond, neg_cond)`
3. After model forward, compute CFG: `denoised = uncond + cfg * (cond - uncond)`
4. Use CFG-combined denoised for log-ratio

---

## 2. Checkpoint is x_{t+1}, Evaluated at sigma_t (MEDIUM)

### Cause

`sample_euler_packed` callback fires AFTER the euler step (L237-240):
```python
x_list[img_i] = x_list[img_i] + d * dt   # x_{t+1}
...
callback({'i': step_i, 'x_list': x_list})  # saves x_{t+1} as step_NN
```

`compute_reinforce_step` evaluates `model(checkpoint[step_i], sigmas[step_i])`,
which is `model(x_{t+1}, sigma_t)`. But x_{t+1} corresponds to noise level
sigma_{t+1}, not sigma_t.

### Impact

sigma_t > sigma_{t+1}, so the denominator `2*sigma_t^2` in the log-ratio is
systematically too large, underweighting the gradient signal. For sparse
steps with large sigma gaps, this is non-trivial.

### Fix (pick one)

**A.** Move callback before euler step (saves x_t, dataset format change).
**B.** Use `sigmas[step_idx + 1]` in `compute_reinforce_step` (minimal code change).

---

## 3. BTRM Head Never Sees neg_cond (LOW-MEDIUM)

BTRM training (`train_btrm_step`, `run_backbone_hidden_packed`) always receives
pos_cond only. The empty-string conditioning (neg_cond) is a real in-distribution
input to the model. The neg_cond hidden states carry quality-discriminative
information (unconditional prediction quality also degrades with attention
quantization and step count).

### Fix

Include neg_cond as separate training examples with the SAME quality labels.
This doubles training data per trajectory at zero additional generation cost.

---

## 4. score_btrm Uses pos_cond Only (LOW)

Consistent with #3 (BTRM training also uses pos_cond only). But the reward
is computed on a model running without CFG, on a latent produced WITH CFG.

---

## 5. Context Refiner Sees Different B=2 Content (LOW)

Rollout: context_refiner processes [pos_cond, neg_cond] as B=2.
Training: context_refiner processes [pos_cond, pos_cond] as B=2.

Standard self-attention is per-sequence (no cross-batch attention), so this
should have zero effect unless batch normalization or similar operators exist.

---

## Implementation Priority

1. Fix #1 (CFG in REINFORCE) -- critical correctness bug
2. Fix #2 (sigma index) -- medium correctness bug, 1-line fix
3. Fix #3 (neg_cond in BTRM) -- quality improvement, doubles data

### Critical Files

- `src/futudiffu/training_utils.py` -- compute_reinforce_step, _prepare_packed_single_state
- `src/futudiffu/sampling.py` -- sample_euler_packed callback position
- `scripts/train.py` -- client-side: pass neg_cond + cfg to accumulate_policy_gradients
- `src/futudiffu/client.py` -- RPC interface: add neg_cond + cfg params
- `src/futudiffu/server.py` -- RPC dispatch: thread new params through

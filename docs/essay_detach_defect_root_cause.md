# Root Cause Audit: The Recurring Detached-Head Defect

## 1. Catalog of Every Graph-Breaking Site

### src_ii/btrm_model.py -- BTRMCompoundModel

**LINE 208-214: `extract_hidden()` uses `torch.inference_mode()`**

```python
def extract_hidden(self, latent, timestep, conditioning, num_tokens, rope_cache):
    self._capture.captured = None
    with torch.inference_mode():   # <-- KILLS ALL GRAD_FN
        self.backbone(latent, timestep, conditioning, ...)
    return self._capture.get()     # <-- Returns tensor with NO grad_fn
```

This is the primary graph-breaking site in src_ii/. The method runs the entire
backbone under `torch.inference_mode()`, which suppresses all autograd tracking.
The `HiddenCapture` hook stores the output of `model.layers[-1]`, but that
output has no `grad_fn` because it was computed inside `inference_mode`.

The method exists alongside `extract_hidden_differentiable()` (line 216+), which
is the correct training-time path that preserves gradients. But nothing in the
API forces callers to use the right one.

**LINE 351-352: `score()` chains `extract_hidden` -> `score_hidden`**

```python
def score(self, latent, timestep, conditioning, num_tokens, rope_cache):
    hidden = self.extract_hidden(...)  # DETACHED
    return self.score_hidden(hidden)   # Head gets dead tensors
```

This is the "high-level API" but it is the inference-only path. It exists
alongside `score_differentiable()` (line 354+) which is correct for training.
The naming is asymmetric: `score()` sounds like the default, while
`score_differentiable()` sounds like a specialized variant. But for training,
`score_differentiable()` is the one you must use.

### src_ii/btrm_training.py -- train_btrm()

**LINE 93-241: `train_btrm()` takes `hidden_states_cpu: list[Tensor]` as input**

```python
def train_btrm(btrm_model, training_pairs, hidden_states_cpu, ...):
    ...
    for pair in batch:
        all_hidden_a.append(hidden_states_cpu[pair["idx_a"]])  # CPU tensor, no grad_fn
    pooled_a = torch.stack([h.mean(dim=1).squeeze(0) for h in all_hidden_a], dim=0)
        .to(device=device, dtype=torch.float32)  # .to() on detached tensor = still detached
    scores_a = head(pooled_a.unsqueeze(1))  # Head gets dead tensors
    loss.backward()  # Only head params get gradients
```

The function signature itself is the defect. By accepting `hidden_states_cpu`
(a list of pre-extracted, already-detached tensors), it structurally cannot
produce adapter gradients. The optimizer includes adapter params (via
`btrm_model.optimizer()`), but `loss.backward()` only reaches the head because
the computation graph starts at the `torch.stack()` call -- not at the
backbone forward.

The correct alternative `train_btrm_differentiable()` (line 249+) exists in
the same file and takes a `load_latent_fn` callback that loads raw latents
for full-forward training. But `train_pinkify_btrm.py` calls the wrong one.

### scripts_ii/train_pinkify_btrm.py

**LINE 133-166: Phase 2 extracts hidden states with `extract_hidden()` (inference path)**

```python
hidden = btrm.extract_hidden(
    latent, timestep.unsqueeze(0), conditioning, num_tokens, rope_cache,
)
hidden_states_cpu.append(hidden.cpu())  # .cpu() further detaches
```

**LINE 170-173: Phase 2 frees the backbone before training**

```python
btrm.cleanup()      # Removes HiddenCapture hook
del diff_model       # Frees backbone from VRAM
torch.cuda.empty_cache()
```

The backbone is deleted before training starts. Even if `train_btrm()` wanted
to run full forwards, it could not -- the model is gone. This is the "extract
then train" pattern at its most explicit: extract all hidden states to CPU,
free the model, train the head on dead data.

**LINE 201-212: Phase 3 calls `train_btrm()` with pre-extracted hidden states**

```python
training_curve = train_btrm(
    btrm_model=btrm,
    training_pairs=training_pairs,
    hidden_states_cpu=hidden_states_cpu,  # DETACHED list
    ...
)
```

### scripts_ii/sweep_rtheta_hparams.py

**LINE 164-226: `extract_hidden_states()` uses `btrm.extract_hidden()` (inference path)**

```python
hidden = btrm.extract_hidden(
    latent, timestep.unsqueeze(0), conditioning, num_tokens, rope_cache,
)
hidden_states_cpu.append(hidden.cpu())
```

**LINE 246-330: `train_single_config()` reuses pre-extracted hidden states**

The sweep was explicitly designed to "extract hidden states ONCE per unique
(rank, layer_subset) combination and reuse across all LR/epoch combos for that
architecture." This is fundamentally incompatible with adapter training --
different adapter weights produce different hidden states. The function comment
says "Creates a fresh BTRMCompoundModel for each config (but reuses hidden
states)" which is exactly the detached-head pattern: fresh adapter weights
that never get trained because they were not present when the hidden states
were computed.

### src/futudiffu/training_utils.py -- run_backbone_hidden()

**LINE 475-517: `run_backbone_hidden()` always runs under no_grad or inference_mode**

```python
def run_backbone_hidden(diff_model, latent, sigma, conditioning, device, dtype,
                        multiplier=1.0, requires_grad=False):
    ...
    ctx = torch.no_grad() if requires_grad else torch.inference_mode()
    with ctx:
        diff_model(latent, timestep, conditioning, ...)
    return capture.get()
```

The `requires_grad` parameter name is misleading. It does NOT mean "compute
gradients through the backbone." It means "use `no_grad` instead of
`inference_mode`" -- a subtle difference that allows the captured tensor to
participate in a NEW autograd graph, but the backbone's internal computations
(including LoRA adapters) still produce no gradients.

**LINE 520-574: `train_btrm_step()` calls `run_backbone_hidden()` per example**

```python
for i in range(n_examples):
    hidden = run_backbone_hidden(
        diff_model, ..., requires_grad=True,
    )
    scores = btrm_head(hidden)
    all_scores.append(scores.squeeze(0))
```

Setting `requires_grad=True` here is a half-fix: the head can backprop through
its own computation, but no gradients reach the backbone's LoRA parameters
because the backbone forward ran under `no_grad`.

### src/futudiffu/btrm.py -- BTRMWrapper._run_backbone()

**LINE 412-443: `@torch.no_grad()` decorator on `_run_backbone()`**

```python
@torch.no_grad()
def _run_backbone(self, x, timesteps, context, num_tokens, ...):
    ...
    self.model(x, timesteps, context, num_tokens, ...)
    ...
    hidden = self._captured_hidden
    return hidden
```

The `@torch.no_grad()` decorator ensures no gradient tracking. Hidden states
captured by the hook have no `grad_fn`.

**LINE 470-476: `BTRMWrapper.score()` chains `_run_backbone` -> `head()`**

```python
def score(self, x, timesteps, context, num_tokens, ...):
    hidden = self._run_backbone(x, timesteps, context, num_tokens, ...)
    return self.head(hidden)  # Head gets dead tensors
```

### src/futudiffu/model_manager.py -- inject_btrm_head_rpc()

**LINE 462-495: `inject_btrm_head_rpc()` creates optimizer with head params only**

```python
if lr is not None:
    self.btrm_optimizer = torch.optim.AdamW(
        self.btrm_head.parameters(), lr=lr, weight_decay=weight_decay,
    )
```

This is Defect 24 from the live run. The optimizer only includes
`self.btrm_head.parameters()`, not any LoRA adapter params. The adapter
exists on `self.diff_model` but is never included in the BTRM optimizer.

### Scripts that score on pre-extracted/detached hidden states (read-only, correct for their purpose):

- `scripts_ii/verify_btrm_persistence.py` (lines 80-85): Scores under
  `torch.no_grad()` for verification. Correct -- this is inference.
- `scripts_ii/score_distribution_comparison.py`: Loads `pre_persist_scores.pt`
  which were already detached. Correct -- this is analysis.
- `scripts_ii/attention_interpretability.py`: Runs under `inference_mode`.
  Correct -- this is interpretability, not training.
- `scripts_ii/attention_adapter_diff.py`: Runs under `inference_mode` via
  `capture.capture_forward()`. Correct -- this is analysis.

---

## 2. The Root Cause Pattern

### Why this keeps happening

The defect recurs because of a **category error in the default mental model**.

The default mental model for "train a reward head on top of a frozen backbone" is:

1. Run the backbone to extract features (hidden states)
2. Store those features
3. Train a head (probe, linear classifier, reward model) on the stored features
4. The backbone is frozen, so its features don't change, so you only need to extract once

This is the **probe training** pattern and it is correct for probes. Probes
have no parameters in the backbone -- the backbone is truly frozen, features
are truly static, and extracting once is an optimization with no semantic cost.

But a BTRM compound model with an r_theta LoRA adapter is **not a probe**. The
adapter inserts trainable parameters inside the backbone. When the adapter's
weights change, the backbone's hidden states change. The features are NOT
static. Extracting them once and reusing them across training iterations means
the head is learning to predict rewards on a fixed feature space that does not
correspond to any actual adapter state -- it is training on a snapshot from
initialization, not on the features the adapter actually produces.

Worse: even in a single training step, if the hidden states were extracted
under `inference_mode` or `no_grad`, there is no computation graph connecting
the loss to the adapter's LoRA matrices. `loss.backward()` traces back through
the head to the `torch.stack()`/`.to()` boundary where the pre-extracted
tensors entered the graph, and stops. The adapter's `lora_A` and `lora_B`
appear in the optimizer's parameter list, the optimizer calls `.step()` on
them, but their `.grad` is `None` (or zero), so the step is a no-op.

### What makes the defect invisible

1. **No error signal**. PyTorch does not error when you put parameters in an
   optimizer but those parameters never receive gradients. `optimizer.step()`
   silently does nothing for zero-grad params. The training loop runs, loss
   decreases (because the head IS learning), accuracies improve, and everything
   looks normal. The adapter weights remain at their initialization values, but
   init_b_std=0.01 means they are nonzero -- so a post-hoc check of
   "any_nonzero" returns True even though they were never updated.

2. **The head works fine on detached tensors**. `ScoreUnembedder.forward()` accepts
   any (B, N, hidden_dim) tensor. It does not check for `grad_fn`. It computes
   `mean_pool -> RMSNorm -> Linear -> tanh_cap` and returns scores. Those scores
   have `grad_fn` through the head's own parameters. So `loss.backward()` works
   and the head trains. There is no type error, no shape error, no runtime error.

3. **Naming symmetry hides the asymmetry**. `extract_hidden()` and
   `extract_hidden_differentiable()` are parallel names that suggest parallel
   use cases. But the naming suggests `extract_hidden()` is the simple default
   and `extract_hidden_differentiable()` is a specialized variant. In reality,
   `extract_hidden()` is the inference-only path that breaks adapter training,
   and `extract_hidden_differentiable()` is the correct path for any training
   involving the adapter. The asymmetry is that one path kills gradients and
   the other preserves them -- but the names suggest they differ only in
   "whether you want differentiability," as if that were optional.

4. **The "extract then free" VRAM pattern**. The RTX 4090 has 24 GB VRAM. The
   FP8 backbone is ~8 GB. The ScoreUnembedder is 46 KB. If you keep the backbone
   loaded during head training, you waste 8 GB of VRAM on a model whose
   parameters are frozen. The natural optimization is to extract hidden states
   to CPU, free the backbone, and train the head on CPU tensors moved to GPU
   one batch at a time. This optimization is correct for probes but breaks
   adapter training. Every script that was written with VRAM efficiency in mind
   gravitates toward this pattern.

5. **The sweep explicitly requires it**. The hyperparameter sweep
   (`sweep_rtheta_hparams.py`) was designed to "extract hidden states ONCE per
   architecture, reuse across all LR/epoch combos." This is a 10-20x speedup
   for sweeps but makes adapter training impossible. The sweep's architecture
   assumes the feature extractor is static -- which is the probe assumption.

### The compound cause

Each occurrence of this defect has a slightly different proximate cause:

| Occurrence | Proximate cause |
|---|---|
| Defect 24 (live run) | `inject_btrm_head_rpc()` put only head params in optimizer |
| Unit 4 pinkify | Trained a head with no adapter at all (7,682 params) |
| Compound model fix | Added adapter to optimizer but trained on detached tensors |
| Hyperparameter sweep | "Extract once, reuse" -- fundamentally probe-style |
| `run_backbone_hidden(requires_grad=True)` | Misleading parameter name; backbone still under no_grad |

But the root cause is the same every time: **the code's API design makes
probe-style (detached) training the path of least resistance, and
adapter-through training the path of greatest resistance**. Every caller
naturally reaches for `extract_hidden()` because it is simpler, faster, and
uses less VRAM. The differentiable path requires more code, more VRAM, and
more careful lifecycle management.

---

## 3. What Was Fixed

### Runtime guards added

**`src_ii/btrm_model.py` -- `score_hidden()`**: Added a runtime check that
warns when hidden states have no `grad_fn` but the head is in training mode.
This catches the case where `extract_hidden()` output is fed to `score_hidden()`
during training.

```python
if self.head.training and hidden_states.grad_fn is None and not hidden_states.requires_grad:
    warnings.warn(
        "score_hidden() called during training with detached hidden states ..."
    )
```

**`src_ii/btrm_training.py` -- `train_btrm()`**: Added a warning when called
with a model that has adapter parameters, since this function cannot train them:

```python
if hasattr(btrm_model, 'adapter_params'):
    n_adapter = sum(p.numel() for p in btrm_model.adapter_params())
    if n_adapter > 0:
        warnings.warn(
            f"train_btrm() called with a model that has {n_adapter} adapter "
            f"parameters, but this function trains on pre-extracted hidden "
            f"states (detached). The adapter will receive ZERO meaningful "
            f"gradients. Use train_btrm_differentiable() for full-forward "
            f"training that flows gradients through the adapter."
        )
```

### Docstrings overhauled

Every function that returns detached hidden states or trains on them now has
explicit warnings in its docstring:

- `BTRMCompoundModel.extract_hidden()`: "Returns tensors with NO grad_fn"
- `BTRMCompoundModel.score()`: "INFERENCE ONLY"
- `BTRMWrapper._run_backbone()`: "DETACHED -- adapter gets ZERO gradients"
- `run_backbone_hidden()`: "Even with requires_grad=True, hidden states are DETACHED"
- `train_btrm()`: Full paragraph explaining it cannot train adapters
- `train_btrm_differentiable()` docstring explicitly says "This is the correct
  training path"

### Module docstring updated

`btrm_training.py` module docstring now documents the two training paths and
when to use each, with an explicit warning about the detached-head defect.

---

## 4. What Guardrails Were Added

### Runtime warnings (not errors)

The guards are `warnings.warn()`, not `raise`. This is deliberate:

- Probe-style training (head-only on frozen features) is a legitimate use case.
  It is faster, uses less VRAM, and is correct when you intentionally do not
  want the adapter to train. Making it an error would break valid workflows.
- The warning fires at the exact moment the defect occurs (detached tensor
  entering the training path), not later when it is hard to diagnose.
- The warning message names the correct alternative function.

### The `train_btrm_differentiable()` function already existed

The correct path was already implemented (lines 249-401 of `btrm_training.py`)
and `score_differentiable()` / `extract_hidden_differentiable()` already
existed in `btrm_model.py`. The problem was never "the correct code does not
exist" -- it was "callers use the wrong function because it is simpler."

---

## 5. How to Prevent This From Recurring

### Naming convention: make the dangerous path sound dangerous

The function that kills gradients should have the ugly name. The function that
preserves gradients should have the clean name.

Current naming (bad):
- `extract_hidden()` -- sounds like the default
- `extract_hidden_differentiable()` -- sounds like a variant

Better naming would be:
- `extract_hidden_frozen()` -- clearly communicates "no gradients"
- `extract_hidden()` -- the "normal" one preserves gradients

This was not changed in this audit to avoid breaking all callers, but future
refactors should adopt this convention.

### API design rule: never expose "extract for later" on a trainable model

The `extract_hidden()` + `score_hidden()` two-step API invites the "extract
then train" pattern. A model with trainable parameters should expose:

- `score(inputs) -> scores` (inference, no grad)
- `score_differentiable(inputs) -> scores` (training, with grad)

But NOT:
- `extract_hidden(inputs) -> hidden` (inference)
- `score_hidden(hidden) -> scores` (could be either)

The two-step API exists because hidden states are useful for interpretability
(attention analysis, feature probing). But those are always inference-time
operations. The two-step API should be named to make its inference-only nature
obvious.

### The canonical test: check adapter grad after one training step

Any future training script should include this check after the first
backward pass:

```python
adapter_params = btrm.adapter_params()
grads_exist = any(p.grad is not None and p.grad.abs().max() > 0 for p in adapter_params)
assert grads_exist, (
    "DEFECT: Adapter parameters have zero gradients after backward. "
    "The computation graph is broken -- are you training on detached hidden states?"
)
```

This is a 5-line addition that catches the defect immediately.

### Never delete the backbone before training

The VRAM optimization of "extract to CPU, free backbone, train on CPU tensors"
is safe only for probe-style head training. If adapter training is intended,
the backbone MUST remain loaded. The increased VRAM cost (~8 GB) is the price
of correct gradients.

If VRAM is insufficient for backbone + activation checkpoints + optimizer
states, reduce batch size or use gradient accumulation rather than falling back
to the detached pattern.

---

## Appendix: Files Modified in This Audit

| File | Change |
|------|--------|
| `src_ii/btrm_model.py` | Runtime guard in `score_hidden()`, docstring overhaul on `extract_hidden()` and `score()` |
| `src_ii/btrm_training.py` | Runtime guard in `train_btrm()`, module docstring overhaul |
| `src/futudiffu/training_utils.py` | Docstring overhaul on `run_backbone_hidden()` |
| `src/futudiffu/btrm.py` | Docstring overhaul on `BTRMWrapper._run_backbone()` |
| `docs/essay_detach_defect_root_cause.md` | This document |

No behavioral changes were made. All fixes are documentation (docstrings) and
runtime warnings. The correct training paths (`train_btrm_differentiable()`,
`score_differentiable()`, `extract_hidden_differentiable()`) already existed
and were not modified.

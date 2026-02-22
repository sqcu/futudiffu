# btrm_model.py: What's Dead, What's Missing

**Date**: 2026-02-21
**Context**: ZImageRLAIF + multi_lora.py now exist. What in btrm_model.py is obsolete?

## Answer: the entire file

Every line of BTRMCompoundModel is replaced, defective, or needs reimplementation.

## Category 1: DEAD — replaced by ZImageRLAIF.forward()

The unified forward returns `(diffusion_fields, scores)`. There is no hidden
state extraction step. The score head is a linear projection inside the model.
This kills:

| Function | Lines | Why dead |
|----------|-------|----------|
| `extract_hidden()` | 294-326 | HiddenCapture hook → model() returns scores directly |
| `extract_hidden_differentiable()` | 328-427 | 100-line forward reimplementation. THE disease. |
| `score_hidden()` | 429-453 | No hidden states to score externally |
| `score()` | 455-479 | extract_hidden + score_hidden = model() |
| `score_differentiable()` | 481-510 | extract_hidden_diff + score_hidden = model() |
| `score_differentiable_packed()` | 512-724 | 213-line packed forward reimpl = model() |
| `_compile_layers_for_training()` | 172-206 | Side-list compilation → torch.compile(model) |
| `_get_training_layers()` | 208-216 | Dispatch compiled vs raw → one set of layers |
| `HiddenCapture` import | 45 | No hooks. Score head is inside the model. |
| `ScoreUnembedder` import | 34 | Replaced by score_norm + score_proj + tanh cap inside model |

## Category 2: DEAD — replaced by multi_lora.py

The old single-adapter LoRA from futudiffu.lora is replaced by multi-tenant
MultiLoRALinear with per-image sparse routing.

| Function/Import | Lines | Replacement |
|-----------------|-------|-------------|
| `allocate_adapter()` | 99-105 | `install_multi_lora()` |
| `init_adapter_weights()` | 108-113 | `init_adapter_b_weights()` |
| `get_lora_params()` | 220 | `get_adapter_params()` |
| `set_lora_scale()` | 288-292 | `adapter_scales` tensor arg to forward() |
| `lora_state_dict()` | 741 | `save_adapter()` |
| `load_lora_state_dict()` | 837 | `load_adapter()` |
| `freeze_adapter()` | 874 | `freeze_base_params()` |
| `unfreeze_adapter()` | 868 | just set requires_grad on adapter params |
| `adapter_params()` | 218-220 | `get_adapter_params()` |
| `head_params()` | 222-224 | `model.score_proj.parameters()` + `model.score_norm.parameters()` |
| `all_trainable_params()` | 226-232 | iterate `model.parameters()` with `requires_grad` filter |

## Category 3: DEAD — replaced by ZImageRLAIF being a real nn.Module

BTRMCompoundModel is a coordinator (NOT an nn.Module) that wraps a backbone,
an adapter, and a head. ZImageRLAIF IS the model. No wrapping. No coordination.

| Function | Lines | Why dead |
|----------|-------|----------|
| `__init__` | 72-161 | The model IS constructed with score head built-in |
| `train_mode()` | 848-868 | `model.train()` + `model.gradient_checkpointing = True` |
| `eval_mode()` | 870-874 | `model.eval()` |
| `cleanup()` | 876-878 | No hooks to remove |
| `is_compiled` | 168-170 | Compilation state is just whether you called torch.compile |
| `device` property | 163-165 | Standard nn.Module pattern |

## Category 4: SALVAGEABLE — but must be reimplemented

These serve a real purpose but their BTRMCompoundModel implementation is wrong.

### Optimizer construction

The idea: "one function that creates an optimizer over BOTH adapter and score
head params, making it structurally impossible to forget one." Still needed.

New implementation (free function, not a method on a coordinator):
```python
def make_training_optimizer(model, adapter_name, lr, ...):
    adapter_ps = list(get_adapter_params(model, adapter_name).values())
    head_ps = [model.score_proj.weight, *model.score_norm.parameters()]
    return AdamW(adapter_ps + head_ps, lr=lr)
```

### Persist/load

Save and load adapter weights + score head weights. New implementation uses
`multi_lora.save_adapter()` / `load_adapter()` for adapters, and standard
state_dict for score_proj + score_norm.

### Muon heterogeneous optimizer

Muon for LoRA matrices, AdamW for score head. The concept is valid.
Implementation just uses multi_lora's param collection instead of futudiffu.lora's.

## What is concretely MISSING for policy optimization

With ZImageRLAIF + multi_lora.py, we have the model. What we don't have:

### 1. Training loop wiring (currently nothing calls ZImageRLAIF)

The training script still uses BTRMCompoundModel. Needs rewrite to:
```python
model = load_zimage_rlaif(checkpoint_path)
install_multi_lora(model, [
    {"name": "rtheta", "rank": 8, "alpha": 16},    # reward adapter
    {"name": "ptheta", "rank": 8, "alpha": 16},     # policy adapter
])
freeze_base_params(model)
unfreeze_score_head(model)
```

### 2. Optimizer construction helpers (trivial, ~30 lines)

As described above. Free functions, not a class.

### 3. Persist/load helpers (trivial, ~40 lines)

Save: adapter weights + score_proj + score_norm + config JSON.
Load: reverse.

### 4. DDGRPO policy loss (task #68, non-trivial)

The actual RL algorithm. Advantage estimation from scores, policy gradient
through p_theta adapter. This is the real remaining work.

### 5. Two-phase training loop

Phase A (reward model): Train rtheta + score head on preference pairs.
Phase B (policy optimization): Generate with ptheta, score with rtheta,
update ptheta via policy gradient.

adapter_scales controls which adapter is active per-image:
- Scoring: `adapter_scales = [[1.0, 0.0], [1.0, 0.0]]` (rtheta on, ptheta off)
- Generation: `adapter_scales = [[0.0, 1.0], [0.0, 1.0]]` (ptheta on, rtheta off)

### 6. Generation with adapter routing

Euler loop that uses packed_forward() with adapter_scales to control which
adapter influences which images during sampling.

## Verdict on btrm_model.py

Delete the entire BTRMCompoundModel class. Replace with ~100 lines of free
functions for optimizer construction and persist/load. The 879-line file
becomes ~100 lines. The 213-line score_differentiable_packed() and the
100-line extract_hidden_differentiable() (the two worst offenders) are
gone entirely — their functionality is a single call to model().

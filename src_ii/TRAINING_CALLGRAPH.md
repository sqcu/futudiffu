# BTRM / RLAIF Training Pipeline Callgraph

Generated 2026-02-25. Covers the complete call chain from training entry point
scripts through to individual forward passes and loss computation.

---

## 1. Training Entry Point Scripts

### 1a. `scripts_ii/run_reward_validated_training.py`

The most complete training script. Uses reward-function-derived preference labels
(pinkify_score_gpu + thisnotthat_score_gpu) instead of sigma-based preferences.
Includes PINKIFY and TNT holdout validation at every EVAL_INTERVAL steps.

**src_ii/ imports:**
| Module | Symbols |
|--------|---------|
| `src_ii.tnt_validation` | `TNT_CHALLENGE_LABELS`, `_TNT_FILENAMES`, `validate_tnt_ranking`, `_check_tnt_ranking`, `_load_tnt_challenge_images` |
| `src_ii.btrm_lifecycle` | `score_serial`, `setup_btrm_training`, `persist_btrm` |
| `src_ii.pair_sampler` | `BTRMPairSampler`, `build_positions_from_v2` |
| `src_ii.flops_sampling` | `compute_flops_sampling_weights_from_positions` |
| `src_ii.vae_utils` | `load_vae` |
| `src_ii.pinkify_validation` | `validate_pinkify_ranking`, `_check_ranking`, `_load_challenge_images` |
| `src_ii.cross_head_decorrelation` | `measure_cross_head_from_pixel_scores`, `measure_cross_head_decorrelation` |
| `src_ii.reward_functions` | `pinkify_score_gpu`, `thisnotthat_score_gpu`, `_pil_to_tensor` |
| `src_ii.zimage_model` | `load_zimage_rlaif` |
| `src_ii.multi_lora` | `get_adapter_params` |
| `src_ii.sigma_schedule` | `build_sigma_schedule`, `resolution_shift` |
| `src_ii.btrm_training` | `train_btrm_differentiable` |
| `src_ii.training_artifacts` | `TrainingArtifacts` |
| `src_ii.incremental_save` | `TrainingCurveWriter` |

**futudiffu/ imports:**
| Module | Symbols |
|--------|---------|
| `futudiffu.dataset_v2` | `DatasetReader` |
| `futudiffu.text_encoder` | `create_tokenizer`, `load_text_encoder`, `encode_prompt` |
| `futudiffu.vae` | `vae_encode` |

**Key call in `main()`:** Line 693:
```python
training_curve = train_btrm_differentiable(
    model=raw_model,
    pair_sampler=sampler,
    load_latent_fn=load_latent_fn,
    n_steps=N_STEPS,
    lr=LR,
    head_names=HEAD_NAMES,
    pref_keys=PREF_KEYS,
    gradient_checkpointing=True,
    max_grad_norm=GRAD_CLIP,
    packed=True,
    macrobatch_budget=MACROBATCH_BUDGET,
    reward_manifest=reward_manifest,
    vae=vae,
    callback=validation_callback,
    ...
)
```

---

### 1b. `scripts_ii/run_flops_budget_100step_v2.py`

FLOPS-budget training with multi-resolution dataset. Uses sigma-based preferences.
Uses composable `src_ii.training_setup` helpers.

**src_ii/ imports:**
| Module | Symbols |
|--------|---------|
| `src_ii.model_paths` | `FP8_PATH`, `TE_PATH`, `VAE_PATH`, `TOKENIZER_PATH` |
| `src_ii.pair_sampler` | `BTRMPairSampler`, `build_positions_from_v2`, `logsnr_sampling_weight`, `logsnr_sampling_logit` |
| `src_ii.flops_sampling` | `compute_flops_sampling_weights_from_positions`, `summarize_flops_weights`, `_attention_flops_ratio`, `_MEGAPIXEL_THRESHOLD` |
| `src_ii.training_setup` | `encode_training_prompts`, `load_training_backbone` |
| `src_ii.btrm_lifecycle` | `persist_btrm` |
| `src_ii.multi_lora` | `get_adapter_params` |
| `src_ii.dataset_io` | `make_load_latent_fn` |
| `src_ii.btrm_training` | `train_btrm_differentiable` |
| `src_ii.training_artifacts` | `TrainingArtifacts`, `PILChart`, `_ema`, `_running_average` |
| `src_ii.incremental_save` | `TrainingCurveWriter` |
| `src_ii.resolution_sampling` | `assign_budget_tier`, `ANCHOR_LABELS` |
| `src_ii.exemplar_renderer` | `render_exemplars_from_model` |

**Key call in `main()`:** Line 262:
```python
training_curve = train_btrm_differentiable(
    model=raw_model,
    pair_sampler=sampler,
    load_latent_fn=load_latent_fn,
    preference_fn=preference_fn,   # sigma-based
    n_steps=N_STEPS,
    packed=True,
    macrobatch_budget=MACROBATCH_BUDGET,
    ...
)
```

---

### 1c. `scripts_ii/run_pinkify_validated_training.py`

PINKIFY-validated training with sigma-based preferences. Evaluates PINKIFY holdout
ranking every 10 steps.

**src_ii/ imports:**
| Module | Symbols |
|--------|---------|
| `src_ii.model_paths` | `FP8_PATH`, `TE_PATH`, `VAE_PATH`, `TOKENIZER_PATH` |
| `src_ii.training_setup` | `build_dataset_positions`, `encode_training_prompts`, `load_training_backbone` |
| `src_ii.btrm_lifecycle` | `persist_btrm`, `score_serial` |
| `src_ii.pinkify_validation` | `validate_pinkify_ranking`, `_check_ranking` |
| `src_ii.multi_lora` | `get_adapter_params` |
| `src_ii.dataset_io` | `make_load_latent_fn` |
| `src_ii.btrm_training` | `train_btrm_differentiable` |
| `src_ii.training_artifacts` | `TrainingArtifacts` |
| `src_ii.incremental_save` | `TrainingCurveWriter` |
| `src_ii.vae_utils` | `load_vae` |

**Key call in `main()`:** Line 399:
```python
training_curve = train_btrm_differentiable(
    model=raw_model,
    pair_sampler=sampler,
    load_latent_fn=load_latent_fn,
    preference_fn=preference_fn,   # sigma-based
    packed=True,
    macrobatch_budget=MACROBATCH_BUDGET,
    callback=pinkify_validation_callback,
    ...
)
```

---

### 1d. `scripts_ii/run03_btrm_training.py`

Earliest training script. Two-phase: generation + training. Uses ZMQ
`InferenceClient` for generation (legacy). Sigma-based scrimble/scrongle preferences.

**src_ii/ imports:**
| Module | Symbols |
|--------|---------|
| `src_ii.bin_packer` | `REFERENCE_SEQ_LEN`, `REFERENCE_TOTAL_LEN`, `BinPackScheduler`, `build_generation_plan`, `compute_seq_len` |
| `src_ii.zimage_model` | `load_zimage_rlaif` |
| `src_ii.btrm_lifecycle` | `setup_btrm_training`, `persist_btrm`, `get_all_trainable_params` |
| `src_ii.multi_lora` | `get_adapter_params` |
| `src_ii.btrm_training` | `train_btrm_differentiable` |
| `src_ii.dataset_generator` | `DatasetGenerationConfig`, `DatasetGenerator` |
| `src_ii.pair_sampler` | `BTRMPairSampler`, `build_positions_from_v2` |
| `src_ii.rendering` | `save_tensor_as_png` |
| `src_ii.sigma_schedule` | `build_sigma_schedule`, `resolution_shift` |
| `src_ii.dataset_filters` | `filter_training_trajectories` |

**Key call in `phase2_train()`:** Line 492:
```python
training_curve = train_btrm_differentiable(
    model=raw_model,
    pair_sampler=sampler,
    load_latent_fn=load_latent_fn,
    preference_fn=preference_fn,
    n_steps=BTRM_N_STEPS,
    callback=training_callback,
    ...
)
```

---

## 2. Complete Function Call Chain for a Training Step

All four entry point scripts converge on the same core call:
`train_btrm_differentiable()` in `src_ii/btrm_training.py`.

### 2.1 FLOPS-Budget Macrobatch Path (primary path, `packed=True` + `macrobatch_budget` set)

```
train_btrm_differentiable()                      [btrm_training.py L396-L1613]
  |
  +-- make_training_optimizer()                   [btrm_lifecycle.py L113-L166]
  |     +-- get_adapter_params()                  [multi_lora.py L489-L511]
  |     +-- torch.optim.AdamW() or Muon()
  |
  +-- SequentialLR(LinearLR + CosineAnnealingLR)  [torch LR schedulers]
  |
  +-- FOR EACH STEP (L809):
  |   |
  |   +-- optimizer.zero_grad()
  |   |
  |   +-- [FLOPS-BUDGET PATH, L843-L1117]
  |   |   |
  |   |   +-- pair_sampler.sample_macrobatch()    [pair_sampler.py]
  |   |   |     Returns list of PairSpec objects
  |   |   |
  |   |   +-- preference_fn(pair_dict)            [per-script closure or reward_manifest]
  |   |   |     When reward_manifest is set:
  |   |   |       +-- load_latent_fn(key_a), load_latent_fn(key_b)
  |   |   |       +-- futudiffu.vae.vae_decode() for both images
  |   |   |       +-- reward_manifest[head_name](pixel_tensor) for each head
  |   |   |       +-- Returns {pref_key: +1/-1/0}
  |   |   |
  |   |   +-- load_latent_fn(key) for each image  [dataset_io.py or script closure]
  |   |   |     +-- reader[traj_id]                [futudiffu.dataset_v2.DatasetReader]
  |   |   |     +-- sigma_schedule.build_sigma_schedule()
  |   |   |     +-- sigma_schedule.resolution_shift()
  |   |   |     Returns: (latent, timestep, conditioning, num_tokens, None)
  |   |   |
  |   |   +-- BinPackScheduler().pack()            [bin_packer.py]
  |   |   |     +-- compute_effective_seq_len()    [bin_packer.py]
  |   |   |     Returns: list of bins, each bin = list of image items
  |   |   |
  |   |   +-- FOR EACH BIN (L996):
  |   |   |   |
  |   |   |   +-- score_packed()                   [btrm_lifecycle.py L179-L239]
  |   |   |   |     +-- BatchExecutor(model, device)  [batch_executor.py L29-L65]
  |   |   |   |     +-- executor.execute(queries)  [batch_executor.py L47-L61]
  |   |   |   |           +-- _scatter(queries)        [batch_executor.py L72-L125]
  |   |   |   |           +-- _build_plan()            [batch_executor.py L140-L168]
  |   |   |   |           |     +-- pack_for_inference()  [inference_packing.py]
  |   |   |   |           |     +-- prepare_packed_forward()  [forward_packed.py L76-L106]
  |   |   |   |           |           +-- model.prepare_packed_state()  [zimage_model.py L312-L393]
  |   |   |   |           |           |     +-- cap_embedder()
  |   |   |   |           |           |     +-- context_refiner layers
  |   |   |   |           |           |     +-- build_packed_sequence()  [transformer.py]
  |   |   |   |           |           |     +-- build_packed_rope()     [transformer.py]
  |   |   |   |           |           +-- _pad_plan_to_fixed_len()      [forward_packed.py L27-L73]
  |   |   |   |           |           +-- build_block_mask_from_packing_info()  [block_mask.py]
  |   |   |   |           |
  |   |   |   |           +-- _execute_plan()          [batch_executor.py L175-L229]
  |   |   |   |                 +-- packed_forward()   [forward_packed.py L109-L133]
  |   |   |   |                 |     +-- model()      [zimage_model.py L630-L713]
  |   |   |   |                 |           SEE SECTION 3 FOR FULL FORWARD DETAILS
  |   |   |   |                 +-- denoise_all()      [triumphant_future_reduction_ops.py]
  |   |   |   |
  |   |   |   +-- BT LOSS COMPUTATION (per-pair, per-head)  [btrm_training.py L1013-L1065]
  |   |   |   |     SEE SECTION 4 FOR EXACT BT LOSS CODE
  |   |   |   |
  |   |   |   +-- partial_loss.backward(retain_graph=not all_pairs_done)  [L1082]
  |   |   |   |
  |   |   |   +-- Detach images whose pairs are all processed  [L1089-L1096]
  |   |   |
  |   |   +-- ValidationMetrics.update()           [validation_metrics.py]
  |   |
  |   +-- torch.nn.utils.clip_grad_norm_()         [L1455]
  |   +-- optimizer.step()                          [L1461]
  |   +-- scheduler.step()                          [L1462]
  |   +-- curve_writer.write_step()                [incremental_save.py]
  |   +-- callback(step, entry)                     [script-provided]
  |   +-- artifacts.log_step() + save_checkpoint()  [training_artifacts.py]
```

### 2.2 Legacy Fixed-Pair-Count Packed Path (`packed=True`, no `macrobatch_budget`)

```
train_btrm_differentiable()
  +-- FOR EACH STEP:
      +-- FOR EACH micro IN grad_accum_steps:
          |
          +-- Sample pairs_per_pack pairs
          +-- Load all 2*pairs_per_pack images via load_latent_fn()
          +-- BinPackScheduler().pack() to produce bins
          +-- FOR EACH BIN:
          |     +-- score_packed() -> BatchExecutor -> packed_forward -> model()
          |
          +-- Reassemble scores, compute BT loss across all K pairs  [L1252-L1302]
          +-- (loss / grad_accum_steps).backward()
```

### 2.3 Serial Path (`packed=False`)

```
train_btrm_differentiable()
  +-- FOR EACH STEP:
      +-- FOR EACH micro IN grad_accum_steps:
          |
          +-- Sample 1 pair (from sampler or training_pairs list)
          +-- load_latent_fn(key_a), load_latent_fn(key_b)
          +-- score_serial(model, lat_a, ...)  [btrm_lifecycle.py L242-L270]
          |     +-- score_packed(model, [(lat_a, ts_a, cond_a, nt_a)])
          |           (wraps single image as packed batch of 1)
          +-- score_serial(model, lat_b, ...)
          +-- BT loss across heads  [L1366-L1412]
          +-- (loss / grad_accum_steps).backward()
```

---

## 3. Model Forward Pass Details

`ZImageRLAIF.forward()` at `src_ii/zimage_model.py` L630-L713.

```
model(x_list, timesteps_list, refined_caps, packing_info, block_mask, packed_rope)
  |
  +-- Phase 1: _preprocess()  [L452-L595, @torch.compiler.disable, EAGER]
  |     +-- TimestepEmbedder(1 - sigma * 1000)            [L496]
  |     +-- Per-token adaLN:
  |     |     +-- _compute_adaln_params(all_adaln_inputs)  [L500, L233-L252]
  |     |           +-- F.linear(adaln_input, _adaln_W, _adaln_B)
  |     |           +-- Split into per-block tuples of (scale_msa, gate_msa, scale_mlp, gate_mlp)
  |     |     +-- Scatter per-image params to per-token using document_id  [L510-L516]
  |     +-- Per-image patchify:
  |     |     +-- pad_to_patch_size(x_i)                   [transformer.py]
  |     |     +-- x_embedder (Linear, patch_size^2 * 16 -> 3840)
  |     |     +-- pad_zimage() for token alignment         [transformer.py]
  |     +-- Per-image noise_refiner (2 JointTransformerBlocks)  [L560-L566]
  |     +-- Pack all cap + img tokens into single sequence  [L572-L578]
  |     +-- Pad to REFERENCE_TOTAL_LEN                     [L583-L585]
  |     Returns: (packed, adaln_input, adaln_params, rope, token_to_image, original_hw)
  |
  +-- Phase 2: 30 main JointTransformerBlock layers  [L687-L702, COMPILED]
  |     FOR EACH layer IN self.layers:
  |       if gradient_checkpointing:
  |         grad_checkpoint(_layer_forward, layer, packed, rope, ...)
  |       else:
  |         _layer_forward(layer, packed, rope, ...)
  |
  |     _layer_forward()  [L715-L743]
  |       +-- layer(packed, None, rope,
  |                 adaln_input, precomputed_adaln,
  |                 block_mask, adapter_scales, token_to_image)
  |
  |     JointTransformerBlock.forward()  [transformer.py]
  |       +-- RMSNorm + adaLN modulate (scale_msa, gate_msa)
  |       +-- JointAttention:
  |       |     +-- qkv = MultiLoRALinear.forward(x, adapter_scales, token_to_image)
  |       |     |     +-- base(x)  [FP8Linear or nn.Linear]
  |       |     |     +-- LoRA delta: x @ A^T @ B^T * (alpha/rank) * scale
  |       |     |     +-- base_output + adapter_output
  |       |     +-- QK head-norm + RoPE
  |       |     +-- SageAttention or SDPA (with block_mask)
  |       |     +-- out_proj = MultiLoRALinear.forward(attn_output)
  |       +-- Residual + gate_msa
  |       +-- RMSNorm + adaLN modulate (scale_mlp, gate_mlp)
  |       +-- FeedForward:
  |       |     +-- w1 = MultiLoRALinear(x)  [gate projection]
  |       |     +-- w3 = MultiLoRALinear(x)  [up projection]
  |       |     +-- SiLU(w1) * w3   (or fused FP8 SiLU+gate+requant kernel)
  |       |     +-- w2 = MultiLoRALinear(intermediate)  [down projection]
  |       +-- Residual + gate_mlp
  |
  +-- Phase 3a: Score head  [L705]
  |     _compute_scores(packed, packing_info)  [L399-L445]
  |       FOR EACH image i:
  |         +-- Extract segment from hidden state (text_start to img_start + img_len)
  |         +-- Mean pool over tokens -> (1, dim=3840)
  |         +-- score_norm (RMSNorm, dim=3840)
  |         +-- score_proj (Linear, 3840 -> n_score_heads=2, no bias, zero-init)
  |         +-- Soft tanh cap: cap * tanh(raw / cap), cap=10.0
  |       +-- torch.stack(score_list) -> (n_images, n_score_heads)
  |
  +-- Phase 3b: FinalLayer  [L708]
  |     +-- final_layer(packed, adaln_input)
  |     +-- Modulate + Linear + unpatchify projection
  |
  +-- Phase 4: _postprocess()  [L711, @torch.compiler.disable, EAGER]
  |     +-- unpack_and_unpatchify()
  |     +-- Crop to original sizes
  |     +-- Negate: -results[i]  (model returns -field by convention)
  |
  Returns: (diffusion_fields: list[Tensor], scores: Tensor(n_images, n_score_heads))
```

---

## 4. Bradley-Terry Loss Computation (EXACT CODE)

### 4.1 In FLOPS-Budget Macrobatch Path

The BT loss for packed FLOPS-budget training is computed inline at
**`src_ii/btrm_training.py` lines 1010-1065**:

```python
# After scoring this bin, compute loss for any pairs that
# now have BOTH images scored and haven't been processed yet.
bin_bt = torch.zeros((), device=device)
bin_active = 0

for k, pd in enumerate(macro_pairs):
    if pair_processed[k]:
        continue
    idx_a, idx_b = pair_image_indices[k]
    if idx_a not in img_idx_to_score or idx_b not in img_idx_to_score:
        continue

    # Both images are now scored -- compute pairwise BT loss
    pair_processed[k] = True
    scores_a_k = img_idx_to_score[idx_a]
    scores_b_k = img_idx_to_score[idx_b]

    for head_idx, (name, pref_key) in enumerate(zip(head_names, pref_keys)):
        pref = pd[pref_key]
        if pref == 0:
            continue

        if pref > 0:
            pos_s = scores_a_k[head_idx]
            neg_s = scores_b_k[head_idx]
        else:
            pos_s = scores_b_k[head_idx]
            neg_s = scores_a_k[head_idx]

        bt = -F.logsigmoid(pos_s - neg_s)           # <-- THE BT LOSS (line 1037)
        bin_bt = bin_bt + bt
        bin_active += 1
        active_heads += 1
```

The partial loss is then backward'd at **line 1079**:
```python
if bin_active > 0:
    partial_loss = bin_bt / _norm_denom              # normalized by pre-counted active heads
    if partial_loss.requires_grad or partial_loss.grad_fn is not None:
        _all_pairs_done = all(pair_processed)
        partial_loss.backward(retain_graph=not _all_pairs_done)
    total_bt_val += bin_bt.item()
```

`_norm_denom` is pre-counted at **lines 971-975**:
```python
_precount_active_heads = 0
for pd in macro_pairs:
    for pref_key in pref_keys:
        if pd.get(pref_key, 0) != 0:
            _precount_active_heads += 1
_norm_denom = max(_precount_active_heads, 1)
```

### 4.2 In Legacy Packed Path

Same BT loss formula at **lines 1259-1273**:
```python
for head_idx, (name, pref_key) in enumerate(zip(head_names, pref_keys)):
    pref = pair[pref_key]
    if pref == 0:
        continue
    if pref > 0:
        pos_s = scores_a_k[head_idx]
        neg_s = scores_b_k[head_idx]
    else:
        pos_s = scores_b_k[head_idx]
        neg_s = scores_a_k[head_idx]

    bt = -F.logsigmoid(pos_s - neg_s)               # <-- THE BT LOSS (line 1271)
    total_bt = total_bt + bt
    active_heads += 1
```

Normalized at **line 1302**: `total_bt = total_bt / active_heads`

### 4.3 In Serial Path

Same formula at **lines 1369-1383**:
```python
bt = -F.logsigmoid(pos_s - neg_s)                   # <-- THE BT LOSS (line 1381)
total_bt = total_bt + bt
active_heads += 1
```

Normalized at **line 1410**: `total_bt = total_bt / active_heads`

### 4.4 In `compute_pairwise_bt_loss()` helper (used only by `train_btrm()` head-only path)

At **lines 126-180**, delegates to `futudiffu.btrm.bradley_terry_loss()`:
```python
bt_loss = bradley_terry_loss(pos_scores, neg_scores)
```

Which is defined in **`src/futudiffu/btrm.py` line 155**:
```python
def bradley_terry_loss(pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
    return -F.logsigmoid(pos_scores - neg_scores).mean()
```

**Key difference**: The batched `bradley_terry_loss()` uses `.mean()` over all pairs.
The differentiable training loop accumulates individual `-F.logsigmoid(pos - neg)`
terms and normalizes by `active_heads` (number of non-tied head-pair contributions).

---

## 5. Module Dependency Graph (Breadth-First)

### Tier 1: Entry point scripts import from src_ii/
```
scripts_ii/run_reward_validated_training.py
  --> src_ii/btrm_training.py
  --> src_ii/btrm_lifecycle.py
  --> src_ii/zimage_model.py
  --> src_ii/multi_lora.py
  --> src_ii/pair_sampler.py
  --> src_ii/flops_sampling.py
  --> src_ii/sigma_schedule.py
  --> src_ii/vae_utils.py
  --> src_ii/pinkify_validation.py
  --> src_ii/tnt_validation.py
  --> src_ii/cross_head_decorrelation.py
  --> src_ii/reward_functions.py
  --> src_ii/training_artifacts.py
  --> src_ii/incremental_save.py

scripts_ii/run_flops_budget_100step_v2.py
  --> src_ii/btrm_training.py
  --> src_ii/training_setup.py
  --> src_ii/btrm_lifecycle.py
  --> src_ii/multi_lora.py
  --> src_ii/model_paths.py
  --> src_ii/pair_sampler.py
  --> src_ii/flops_sampling.py
  --> src_ii/dataset_io.py
  --> src_ii/training_artifacts.py
  --> src_ii/incremental_save.py
  --> src_ii/resolution_sampling.py
  --> src_ii/exemplar_renderer.py

scripts_ii/run_pinkify_validated_training.py
  --> src_ii/btrm_training.py
  --> src_ii/training_setup.py
  --> src_ii/btrm_lifecycle.py
  --> src_ii/multi_lora.py
  --> src_ii/model_paths.py
  --> src_ii/pinkify_validation.py
  --> src_ii/dataset_io.py
  --> src_ii/vae_utils.py
  --> src_ii/training_artifacts.py
  --> src_ii/incremental_save.py

scripts_ii/run03_btrm_training.py
  --> src_ii/btrm_training.py
  --> src_ii/btrm_lifecycle.py
  --> src_ii/zimage_model.py
  --> src_ii/multi_lora.py
  --> src_ii/bin_packer.py
  --> src_ii/pair_sampler.py
  --> src_ii/sigma_schedule.py
  --> src_ii/rendering.py
  --> src_ii/dataset_generator.py
  --> src_ii/dataset_filters.py
```

### Tier 2: Core training modules import from src_ii/

```
src_ii/btrm_training.py
  --> futudiffu.btrm (bradley_terry_loss)
  --> src_ii/btrm_lifecycle.py (make_training_optimizer, get_all_trainable_params, score_packed, score_serial)
  --> src_ii/validation_metrics.py (ValidationMetrics, PairResult)
  --> src_ii/incremental_save.py (TrainingCurveWriter, PeriodicSaver, atomic_json_save)
  --> src_ii/pair_sampler.py (logsnr_sampling_weight)
  --> src_ii/bin_packer.py (BinPackScheduler, compute_effective_seq_len)

src_ii/btrm_lifecycle.py
  --> src_ii/multi_lora.py (install_multi_lora, freeze_base_params, unfreeze_score_head, init_adapter_b_weights, get_adapter_params, save_adapter, load_adapter)
  --> src_ii/batch_executor.py (BatchExecutor)  [lazy import in score_packed()]

src_ii/batch_executor.py
  --> src_ii/forward_packed.py (prepare_packed_forward, packed_forward)
  --> src_ii/bin_packer.py (REFERENCE_TOTAL_LEN)
  --> src_ii/inference_packing.py (pack_for_inference, compute_entry_seq_len)
  --> src_ii/sigma_schedule.py (resolution_shift)
  --> src_ii/triumphant_future_reduction_ops.py (denoise_all, latent_padded)

src_ii/forward_packed.py
  --> src_ii/block_mask.py (build_block_mask, build_block_mask_from_packing_info)
  --> src_ii/bin_packer.py (REFERENCE_TOTAL_LEN)
```

### Tier 3: Model modules import from src_ii/

```
src_ii/zimage_model.py
  --> src_ii/transformer.py (EmbedND, FinalLayer, JointTransformerBlock, PackingInfo, RMSNormModule, TimestepEmbedder, build_packed_rope, build_packed_sequence, build_trivial_mask, pad_to_patch_size, pad_zimage, unpack_and_unpatchify, _detect_*, _strip_diffusion_prefix, fuse_model)
  --> futudiffu.fp8 (replace_linear_with_fp8, FP8Linear, dequantize_fp8_blockwise)
  --> futudiffu.sage_attention (configure_sage)

src_ii/multi_lora.py
  --> (no src_ii/ imports, pure module)

src_ii/training_setup.py
  --> futudiffu.text_encoder (create_tokenizer, load_text_encoder, encode_prompt)
  --> futudiffu.dataset_v2 (DatasetReader)
  --> src_ii/pair_sampler.py (BTRMPairSampler, build_positions_from_v2)
  --> src_ii/flops_sampling.py (compute_flops_sampling_weights_from_positions)
  --> src_ii/zimage_model.py (load_zimage_rlaif)
  --> src_ii/btrm_lifecycle.py (setup_btrm_training, load_btrm)
  --> src_ii/multi_lora.py (get_adapter_params)
  --> src_ii/dataset_io.py (make_load_latent_fn)  [not directly, but designed to pair]

src_ii/dataset_io.py
  --> src_ii/sigma_schedule.py (build_sigma_schedule, resolution_shift)
  --> futudiffu.vae (vae_decode)
```

---

## 6. Key Architectural Facts

1. **Single forward path**: `ZImageRLAIF.forward()` ALWAYS returns `(diffusion_fields, scores)`. There is no inference-only mode. A single image is a packed batch of 1.

2. **Score head location**: Scores are computed from `layers[-1]` output BEFORE `final_layer`. The score_proj is zero-initialized, so untrained models return scores of zero.

3. **Loss normalization**: In FLOPS-budget mode, loss is normalized by `_precount_active_heads` (total non-tied head-pair contributions across the entire macrobatch). In legacy modes, normalized by `active_heads` per microbatch. This ensures gradient scale is consistent per-signal regardless of batch composition.

4. **Per-bin backward**: In FLOPS-budget mode, gradients are accumulated per-bin (not all-at-once). Each bin's partial loss is backward'd immediately after scoring, which caps GPU memory at O(max_active_graphs) instead of O(total_bins * graph_size). `retain_graph=True` is used when cross-bin pairs still need partner scores.

5. **No logsquare regularizer**: The BT loss is pure `-F.logsigmoid(pos - neg)`. The soft_tanh_cap(10.0) in the score head bounds magnitudes instead.

6. **MultiLoRALinear routing**: adapter_scales (n_images, n_adapters) + token_to_image (total_len,) enables per-image adapter contribution in packed batches. When adapter_scales is None, only the base linear runs.

7. **Compiled vs eager boundary**: `_preprocess()` and `_postprocess()` are `@torch.compiler.disable`. The 30 main layers + score head + final_layer run inside the compiled graph.

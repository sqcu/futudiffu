# Unit 4: PINKIFY and THISNOTTHAT -- Exploration Briefing for Root Claude

**Date**: 2026-02-17
**Author**: Exploration agent (Opus 4.6)
**Purpose**: Research briefing to inform implementation rubrics for coding subagents

---

## 1. Spec Summary

The user's specification (verbatim in `docs/unit4_user_spec_verbatim.md`, elaborated in `docs/directive_unit4_pinkify_thisnotthat.md`) defines two *trivially computable* pairwise preference rules that serve as end-to-end tests of the BTRM training pipeline. The rules are deliberately simple so that the test validates the *pipeline machinery* (adapter training, persistence, bit-for-bit reload, scoring distributions), not the quality of the reward signal itself.

### PINKIFY (Head 0)

Given two images, the pinker one wins. "Pinkness" is defined in pixel space using HSV color analysis with local contrast: a pink pixel's contribution is proportional to the fraction of non-pink pixels in its local neighborhood. A uniformly pink image scores lower than one with vivid pink regions against non-pink surroundings. Score is normalized by image area.

### THISNOTTHAT (Head 1)

Given an image, is it more similar to THIS (`i2i_off_policies/pizza-ratto.png`, a black-and-white line drawing of a pizza rat) or THAT (`i2i_off_policies/offhand_pleometric.png`, a purple penguin with red hat)? The scoring function resizes both references to the judged image's dimensions and computes the average of (cosine similarity difference) and (structural similarity difference). Positive score means more like THIS; negative means more like THAT.

### The broader purpose

Both preferences *could* be consumed as direct rewards for RL, but the Unit 4 spec explicitly scopes to training a BTRM model against them -- no RL, no policy gradients, no REINFORCE. The deliverable is a demonstrated end-to-end pipeline: generate preference labels from real trajectory data, train BTRM (backbone + r_theta adapter + score unembedder), persist the trained model, reload it, verify bit-for-bit equivalence, and show that the BTRM's rankings agree with the literal rules. If this works, the BTRM pipeline is proven before touching REINFORCE.

---

## 2. Codebase Readiness

### What already exists and works

The exploration reveals that **Unit 4 has already been substantially implemented in a prior session**. The following artifacts exist on disk:

**Reward functions** (`src_ii/reward_functions.py`, 312 lines): Both `pinkify_score()` and `thisnotthat_score()` are fully implemented, tested, and documented. The pinkify scorer uses integral-image-based local contrast (no scipy dependency). The thisnotthat scorer uses cosine + structural similarity. Both are pure CPU functions operating on PIL Images with no GPU or model dependencies. A `pairwise_preference()` helper converts scores to +1/-1/0 preferences.

**Score cache** (`src_ii/score_cache.py`, 235 lines): A JSON-backed per-image score cache mapping `(traj_id, step_key) -> {head_name: score}`. Supports both pre-scored mode (load existing cache) and live scoring mode (VAE-decode + score on first access). Already integrated with reward_functions.

**Preference label generation** (`scripts_ii/generate_preference_labels.py`): Complete script that loads V1 trajectory latents, VAE-decodes, scores with both reward functions, generates all-pairs preferences within each trajectory, and persists to JSON.

**BTRM compound model** (`src_ii/btrm_model.py`, 571 lines): `BTRMCompoundModel` is fully implemented with:
- Allocation of r_theta LoRA adapter on the backbone
- ScoreUnembedder creation (RMSNorm + Linear + tanh_cap)
- Both detached (`extract_hidden`) and differentiable (`extract_hidden_differentiable`) forward paths
- Integrated hidden capture hook
- `optimizer()` method that structurally prevents Defect 24 (always includes both adapter + head params)
- `persist()` / `load()` for the compound triple (adapter + head + config)
- Support for both AdamW and Muon (heterogeneous) optimizers

**Training loops** (`src_ii/btrm_training.py`, 740 lines): Two training functions:
- `train_btrm()`: Detached path on pre-extracted hidden states. Fast but adapter gets zero meaningful gradients (explicitly warned in docstring and detached-head guard).
- `train_btrm_differentiable()`: Full forward through the 6B backbone per step with gradient checkpointing. Correct for adapter training.

Both support the pair_sampler + preference_fn interface (on-the-fly pairs from combinatorial space) as well as the materialized pair table interface.

**Pair sampler** (`src_ii/pair_sampler.py`, 501 lines): `BTRMPairSampler` with logSNR-weighted two-stage sampling. Reward-function-agnostic (preferences come from a pluggable `preference_fn`, not hardcoded).

**BTRM training script** (`scripts_ii/train_pinkify_btrm.py`): Three-phase script that encodes prompts, extracts hidden states, and trains the compound model. Already run to completion.

**Existing output** (`pinkify_thisnotthat_output/`): Contains preference labels, training curves, persisted compound model (adapter + head + config), pre-persist scores, and persistence verification results -- all PASS.

**Existing essay** (`docs/essay_unit4_pinkify_thisnotthat.md`): Comprehensive results document covering reward functions, preference label statistics, training results, persistence verification, and BTRM-vs-literal agreement (97.1% pinkify, 84.3% thisnotthat).

### What exists in src_ii/ that Unit 4 can build on

| Module | Status | Relevance |
|--------|--------|-----------|
| `reward_functions.py` | Complete | Primary reward functions for both heads |
| `btrm_model.py` | Complete | BTRMCompoundModel with all training/persistence infrastructure |
| `btrm_training.py` | Complete | Both detached and differentiable training loops |
| `pair_sampler.py` | Complete | On-the-fly pair sampling with logSNR weighting |
| `score_cache.py` | Complete | Per-image score cache for reward function outputs |
| `dataset_filters.py` | Complete | Deprecation filters for training data selection |
| `rendering.py` | Complete | Tensor-to-PNG, false-color diff, diff statistics |
| `vae_utils.py` | Complete | VAE load/decode utilities |
| `attention.py` | Complete | Unified attention dispatch (sage/sage_masked/sdpa) |
| `block_mask.py` | Complete | uint8 block mask construction |
| `forward_packed.py` | Complete | Packed forward orchestration |
| `dataset_generator.py` | Complete | 7-phase generation pipeline |
| `model_loading.py` | Complete | FP8 diffusion model loading |
| `sigma_schedule.py` | Complete | Sigma schedule construction with resolution shift |
| `bin_packer.py` | Complete | FFD bin packing for mixed-resolution batches |
| `stats.py` | Complete | Spearman correlation, sigma_for_step, etc. |

### What is missing or needs extension

1. **Full-forward differentiable training with pinkify/thisnotthat heads**: The existing `train_pinkify_btrm.py` uses `train_btrm()` (the detached path). The BTRM Training Policy in CLAUDE.md mandates that adapter training use the differentiable forward path. A script using `train_btrm_differentiable()` with the pinkify/thisnotthat preference function exists as `run03_btrm_training.py` (which was designed for scrimblo/scrongle but uses the same machinery).

2. **Integration with V2 datasets**: The existing pinkify scripts use V1 data (10 trajectories from `btrm_dataset/`). The V2 dataset (`btrm_dataset_v2/`) has 259+ trajectories. Training on V2 data with the `BTRMPairSampler` + `preference_fn` interface is the scalable path.

3. **Preference function for pair_sampler**: The `train_btrm_differentiable()` function accepts a `preference_fn` callable. For pinkify/thisnotthat, this function needs to: (a) load both latents, (b) VAE-decode them, (c) score with `pinkify_score()` and `thisnotthat_score()`, (d) return preferences. This requires a VAE to be available during training, which conflicts with backbone VRAM on a 24GB GPU unless: the preference function uses pre-computed scores from the ScoreCache, OR the preferences are pre-computed before backbone loading.

4. **Score distribution comparison script**: `scripts_ii/score_distribution_comparison.py` exists and was already run (output in `pinkify_thisnotthat_output/score_distribution_comparison.json`).

5. **Persistence verification script**: `scripts_ii/verify_btrm_persistence.py` exists and was already run (output shows PASS).

---

## 3. ScoreUnembedder Architecture

### Current architecture (`src/futudiffu/btrm.py`)

```
Hidden states (B, N_tokens, 3840) from final transformer block
  -> mean pool over token dim -> (B, 3840)
  -> _RMSNorm(3840, eps=1e-6, learnable weight) -> (B, 3840)
  -> nn.Linear(3840, N_heads, bias=False) -> (B, N_heads)
  -> soft_tanh_cap(logit_cap=10.0) -> (B, N_heads)
```

Total parameters for N_heads=2: 3840 (RMSNorm weight) + 2*3840 (projection) = 11,520 parameters.

### How many heads does it have?

The ScoreUnembedder accepts `head_names` as a constructor argument. The default in `btrm.py` is `("bit_quality", "step_quality")` (the original scrimblo/scrongle heads). The BTRMCompoundModel in `src_ii/btrm_model.py` defaults to `("pinkify", "thisnotthat")`. The number of heads is dynamic -- set at construction time.

### How would PINKIFY and THISNOTTHAT be added?

**They are already their own heads on a separate unembedder.** The existing implementation creates a BTRMCompoundModel with `head_names=("pinkify", "thisnotthat")`. This is a completely separate compound model from the scrimblo/scrongle one used in Run 01 and Run 02.

This is the correct design: PINKIFY and THISNOTTHAT train against different preference rules than scrimblo/scrongle, so they need:
- Their own r_theta LoRA adapter (different gradient signal)
- Their own ScoreUnembedder (different head count and meaning)
- Their own optimizer state

If someone wanted all four heads (scrimblo, scrongle, pinkify, thisnotthat) in a single compound model, the architecture supports it -- ScoreUnembedder accepts any number of head names. But this would require training against all four preference functions simultaneously, with shared adapter parameters, which is a different (and harder) training objective. The spec does not require this.

---

## 4. Training Pipeline Readiness

### Can the existing pipeline handle PINKIFY/THISNOTTHAT?

**Yes, completely.** Both training paths (`train_btrm` and `train_btrm_differentiable`) accept arbitrary `head_names` and `pref_keys`. The preference computation is decoupled from pair selection -- the `BTRMPairSampler` returns pair metadata without preferences, and a pluggable `preference_fn` evaluates the reward function.

### Preference function for each head

**PINKIFY**: `pinkify_score(vae_decode(latent_a)) > pinkify_score(vae_decode(latent_b))` -> pref = +1 (A wins)

**THISNOTTHAT**: `thisnotthat_score(vae_decode(latent_a), this_ref, that_ref) > thisnotthat_score(vae_decode(latent_b), this_ref, that_ref)` -> pref = +1 (A wins)

The `reward_functions.py` module already implements both score functions and the `pairwise_preference()` helper that converts score differences to +1/-1/0 labels.

### The VRAM problem for live preference computation

During `train_btrm_differentiable()`, the backbone is loaded for full-forward passes (~6GB FP8 weights + activations). VAE decode requires the VAE (~160MB). On a 24GB GPU, both can coexist in principle, but the training loop's gradient checkpointed forward pass peaks at ~18GB, leaving only ~6GB for VAE decode of a 1280x832 image (which needs ~2GB). This is tight but feasible.

However, the existing approach avoids this problem entirely: **pre-compute all preference scores before backbone loading**. The `generate_preference_labels.py` script does this as a separate phase (load VAE, decode all latents, score, save labels, free VAE). Then the training script loads the labels from disk. This phased approach is correct and eliminates VRAM contention.

For the `pair_sampler + preference_fn` interface (on-the-fly pairs from the full combinatorial space), the preference function can use the `ScoreCache`:

```python
def make_preference_fn(score_cache):
    def preference_fn(pair_meta):
        scores_a = score_cache.get_scores(pair_meta["traj_a"], pair_meta["step_a"])
        scores_b = score_cache.get_scores(pair_meta["traj_b"], pair_meta["step_b"])
        return {
            "pinkify_pref": score_cache.preference(scores_a, scores_b, "pinkify"),
            "thisnotthat_pref": score_cache.preference(scores_a, scores_b, "thisnotthat"),
        }
    return preference_fn
```

This requires pre-populating the ScoreCache with all image scores (one-time VAE decode phase), after which preferences are computed in pure Python with zero GPU cost.

### Loss function

The existing `compute_pairwise_bt_loss()` in `btrm_training.py` computes Bradley-Terry pairwise loss for any single head. The training loops iterate over all heads, sum losses, and average across active heads. This is correct for multi-head BTRM with arbitrary head semantics.

### Optimizer setup

`BTRMCompoundModel.optimizer()` creates an optimizer over ALL trainable parameters (adapter + head). Supports both AdamW and Muon (heterogeneous: Muon for LoRA matrices, AdamW for ScoreUnembedder). The Defect 24 prevention is structural -- impossible to create an optimizer without adapter params.

---

## 5. Reference Image Handling

### How are reference images provided to the model?

**They are NOT provided to the model.** This is a critical architectural point that the spec makes clear but could be easily misunderstood.

The reference images (pizza-ratto.png, offhand_pleometric.png) are used exclusively by the `thisnotthat_score()` reward function, which operates in **pixel space** on decoded PIL images. The reward function is:

1. VAE-decode the latent to get a PIL Image
2. Resize THIS and THAT references to match the decoded image dimensions
3. Compute similarity in pixel space (cosine + structural)
4. Return a scalar score

The diffusion model backbone never sees the reference images. The BTRM model learns to predict, from hidden states at arbitrary noise levels, which of two images will score higher when decoded and compared to the references. The "knowledge" of what THIS and THAT look like is encoded implicitly in the learned adapter and score unembedder weights, not as an explicit input.

### Reference image status

Both reference images exist on disk:
- THIS: `/mnt/f/dox/repos/ai/futudiffu/i2i_off_policies/pizza-ratto.png` (11,347 bytes, sketch)
- THAT: `/mnt/f/dox/repos/ai/futudiffu/i2i_off_policies/offhand_pleometric.png` (21,521 bytes, purple penguin)

Both were confirmed present and used successfully in the existing preference label generation run.

### No ambiguity

The spec is clear: "if we resize and rescale the THIS image and THAT image to the image we are comparing, the image is THISSER if the similarity of the THIS image and the judged image is higher." This is pixel-space comparison after VAE decode. No CLIP embeddings, no latent-space similarity, no conditional injection of reference images into the model. The reference images are consumed by the reward function, not the model.

---

## 6. Blockers and Risks

### Already resolved

1. **Defect 24 (adapter never trained)**: Prevented structurally by BTRMCompoundModel. All training paths include adapter params in the optimizer.

2. **Reward function implementation**: Both `pinkify_score()` and `thisnotthat_score()` are implemented and tested.

3. **Reference images**: Present on disk and verified.

4. **Persistence/reload**: Verified bit-for-bit (persistence_verification.json shows PASS).

5. **Training pipeline**: Both detached and differentiable paths work.

### Current gaps requiring attention

1. **The existing training used the detached path**: `train_pinkify_btrm.py` calls `train_btrm()`, which trains on pre-extracted hidden states. Per the BTRM Training Policy, this is acceptable as a "fast hyperparameter sweep" or "probe-style training" but is NOT the correct path for training the adapter. The r_theta adapter gets zero meaningful gradients through detached hidden states. A differentiable training run using `train_btrm_differentiable()` should be executed to properly train the adapter.

2. **Small dataset**: The existing run used only 10 trajectories (280 pairs) from the V1 dataset. The V2 dataset has 259+ trajectories. Training on the larger dataset would provide more meaningful results, especially for the thisnotthat head (which had lower agreement, 84.3% vs 97.1% for pinkify).

3. **No V2-integrated preference scoring**: The existing `generate_preference_labels.py` reads from V1 (`btrm_dataset/`). A V2-integrated version that reads from `btrm_dataset_v2/` using `DatasetReader` and populates a `ScoreCache` would be needed for training with the pair_sampler interface.

4. **VRAM constraints for differentiable training on RTX 4090**: The differentiable forward path with gradient checkpointing peaks at ~18GB for a single 1280x832 image. Training pairs where both images are full-resolution require two sequential forward passes per optimizer step. This works (demonstrated in Run 02) but is slow (~10-35s per macrobatch). Mixed-resolution images (512x512) would be faster but the existing pinkify preference labels are for full-resolution images only.

5. **Inductor SymPy recursion with LoRA**: Known bug (Defect R2-03). `torch.compile` fails with LoRA adapters present. The differentiable training path uses `forward_checkpointed()` which bypasses `diff_compiled`, so this is not a blocker for training, but it means BTRM scoring during training cannot use the compiled inference path.

### Non-blockers

- scipy is optional for `pinkify_score()` (pure numpy fallback implemented)
- The BTRMCompoundModel defaults to `("pinkify", "thisnotthat")` head names, matching the Unit 4 spec exactly
- All necessary modules (`reward_functions`, `btrm_model`, `btrm_training`, `pair_sampler`, `score_cache`, `vae_utils`, `rendering`) exist in `src_ii/`

---

## 7. Suggested Decomposition

Given that the foundational work is already done, Unit 4 can be decomposed into tasks that upgrade the existing implementation to full compliance with the BTRM Training Policy (differentiable forward, adapter actually trained) and scale to the V2 dataset.

### Task 1: V2-Integrated Preference Score Pre-computation

**What**: Script that reads all trajectories from `btrm_dataset_v2/`, VAE-decodes each latent, scores with both reward functions, and populates a `ScoreCache` JSON file. This replaces the V1-only `generate_preference_labels.py` for the V2 dataset.

**Reading list**:
- `src_ii/score_cache.py` (ScoreCache API)
- `src_ii/reward_functions.py` (pinkify_score, thisnotthat_score)
- `src_ii/vae_utils.py` (load_vae, decode_latent_to_pil)
- `src/futudiffu/dataset_v2.py` (DatasetReader)
- `docs/dataset_v2_spec.md` (V2 dataset schema)

**Rubric**:
- Reads from `btrm_dataset_v2/` using DatasetReader
- Scores every (traj_id, step_key) position for both heads
- Writes to a ScoreCache-compatible JSON file
- Output persisted to `pinkify_thisnotthat_output/v2_score_cache.json`
- Tolerates partial completion (can resume from existing cache)
- Wall time logged

**Estimated scope**: ~150 lines of script code, 1-2 hours of GPU time for VAE decode of all ~1800 images.

### Task 2: Differentiable BTRM Training with Pinkify/TNT Heads

**What**: Script that trains a BTRMCompoundModel using `train_btrm_differentiable()` with the pinkify/thisnotthat preference function. Uses the V2 dataset and pair_sampler interface. This is the critical task: it produces an adapter that has actually been trained with gradients flowing through the LoRA matrices.

**Reading list**:
- `src_ii/btrm_model.py` (BTRMCompoundModel)
- `src_ii/btrm_training.py` (train_btrm_differentiable, preference_fn interface)
- `src_ii/pair_sampler.py` (BTRMPairSampler, build_positions_from_v2)
- `src_ii/score_cache.py` (ScoreCache for preference_fn)
- `scripts_ii/run03_btrm_training.py` (reference implementation for differentiable training)
- `docs/claude_BTRM_training_policy.md` (mandatory policy)
- `docs/essay_run02_mixed_res.md` (Run 02 results for context)

**Rubric**:
- Creates BTRMCompoundModel with `head_names=("pinkify", "thisnotthat")`
- Uses `train_btrm_differentiable()` with `pair_sampler` + `preference_fn`
- preference_fn reads from the ScoreCache (Task 1 output)
- load_latent_fn loads from V2 dataset
- Gradient checkpointing enabled (required for 24GB VRAM)
- rtheta adapter receives nonzero gradients from macrobatch 1 (verify `pre_clip_grad_norm > 0`)
- At least 50 optimizer steps
- Training metrics (loss, accuracy, grad_norm) persisted to JSONL
- Compound model persisted at end (adapter + head + config)
- Output to `pinkify_thisnotthat_output/differentiable_run/`

**Estimated scope**: ~200 lines of script code, ~30-60 minutes of training time.

### Task 3: Persistence Round-Trip Verification

**What**: Load the compound model from Task 2, score a set of test inputs, compare to pre-persist scores, verify bit-for-bit equivalence.

**Reading list**:
- `src_ii/btrm_model.py` (BTRMCompoundModel.load())
- `scripts_ii/verify_btrm_persistence.py` (reference implementation)
- `pinkify_thisnotthat_output/persistence_verification.json` (existing pass for detached-trained model)

**Rubric**:
- Loads compound model from Task 2 output directory
- Scores at least 10 test inputs (latents from V2 dataset)
- Compares output tensors to pre-persist scores saved during Task 2
- All scores bit-for-bit identical
- Weight comparison shows zero max_diff for all tensors
- Verification result persisted to JSON with verdict PASS/FAIL
- Output to `pinkify_thisnotthat_output/differentiable_run/persistence_verification.json`

**Estimated scope**: ~100 lines, ~5 minutes runtime.

### Task 4: Score Distribution Comparison (BTRM vs. Literal Rules)

**What**: Score a test set with both the trained BTRM head and the literal rule functions. Compare rankings. This is the ultimate validation: does the BTRM model, trained through the full pipeline, actually agree with the trivially-computable rules it was trained against?

**Reading list**:
- `src_ii/reward_functions.py` (literal rule scores)
- `src_ii/score_cache.py` (pre-computed literal scores)
- `src_ii/stats.py` (spearman_rank_correlation)
- `scripts_ii/score_distribution_comparison.py` (reference implementation)

**Rubric**:
- Uses the compound model from Task 2/3
- Scores at least 100 test images with the BTRM head
- Computes pairwise agreement with literal rules for both heads
- Computes Spearman rank correlation for both heads
- Pinkify pairwise agreement >= 90%
- Thisnotthat pairwise agreement >= 75% (lower bound is realistic for hidden-state-based prediction of pixel-space similarity)
- All results persisted to JSON
- Output to `pinkify_thisnotthat_output/differentiable_run/score_comparison.json`

**Estimated scope**: ~150 lines, ~15 minutes runtime (backbone forward passes for BTRM scoring).

### Task 5: Essay Synthesis

**What**: After Tasks 1-4 complete, write a synthesis essay comparing the detached-path results (already exist) with the differentiable-path results (Task 2-4). Key questions: Does the adapter make a measurable difference? Is the differentiable-trained model more accurate than the detached-trained probe? How do gradient norms evolve?

**Reading list**:
- `docs/essay_unit4_pinkify_thisnotthat.md` (existing detached-path results)
- Task 2-4 outputs
- `docs/essay_run02_mixed_res.md` (Run 02 for comparison)
- `docs/essay_detach_defect_root_cause.md` (why the detach path undertrained the adapter)

**Rubric**:
- Compares detached vs. differentiable training quantitatively
- Reports accuracy improvement (if any) from adapter training
- Reports gradient norm trajectories
- Discusses whether the adapter contributes signal beyond the linear probe
- All claims backed by persisted numeric artifacts
- Output to `docs/essay_unit4_differentiable_training.md`

**Estimated scope**: ~500-word essay.

---

## Summary

Unit 4 is in a mature state. The reward functions, BTRM training infrastructure, preference label generation, persistence/verification, and score distribution comparison all exist and have been run successfully. The critical gap is that the existing training used the **detached path** (`train_btrm`), which violates the BTRM Training Policy by giving the adapter zero meaningful gradients. The decomposition above addresses this by executing the **differentiable path** (`train_btrm_differentiable`) on the larger V2 dataset, then verifying and comparing results.

The codebase is fully ready for this upgrade. No new modules need to be created from scratch. All five tasks operate on existing src_ii/ infrastructure with script-level orchestration only.

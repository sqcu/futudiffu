# Pair Constraint Audit: Removing Unnecessary Indexing Hierarchies

## The Principle

Every image in the universe has a PINKIFY score. Every image has a THISNOTTHAT score. Any two images can form a valid pairwise comparison.

- **PINKIFY** discriminates attention quantization artifacts (SageAttention INT8 QK vs SDPA). This is a property of the rendering backend, not the image content, prompt, or resolution. A 256x256 cat rendered with SDPA can be compared to a 1280x832 landscape rendered with SageAttention.
- **THISNOTTHAT** discriminates step count (30-step vs 8-22 step). Also a property of the generation process metadata, not the content.

The pair sampler should restrict which images can be paired together ONLY by the requirement that we know the ground-truth preference ordering (which image should score higher on each head). Content, prompt, and resolution are irrelevant to this determination.

## Constraints Audited

### 1. `prompt_idx` field on `_ImagePosition`

**Location:** `src_ii/pair_sampler.py`, `_ImagePosition.__slots__`

**What it did:** Stored the prompt index for each image position. Used by `_prompt_traj_ids`, `_traj_prompt`, and `_cross_res_prompts` to build a prompt-based trajectory grouping used exclusively for cross-resolution pair construction.

**Classification:** UNNECESSARY for pairing logic. Retained on `_ImagePosition` for informational/logging purposes but no longer used for any pairing decision.

**Action:** Field retained (harmless metadata). All pairing logic that consumed it has been removed.

### 2. `_prompt_traj_ids: dict[int, list[int]]`

**Location:** `src_ii/pair_sampler.py`, `BTRMPairSampler.__init__`

**What it did:** Mapped `prompt_idx -> list[traj_id]` so that cross-resolution pairs could be restricted to trajectories sharing the same prompt.

**Classification:** UNNECESSARY. The premise that cross-resolution pairs must share a prompt is wrong. Pinkify and thisnotthat scores are universal -- they do not depend on image content.

**Action:** REMOVED. The entire block (lines 547-556 in the original) has been deleted.

### 3. `_traj_prompt: dict[int, int]`

**Location:** `src_ii/pair_sampler.py`, `BTRMPairSampler.__init__`

**What it did:** Inverse mapping from `traj_id -> prompt_idx`. Only existed to support `_prompt_traj_ids` lookups.

**Classification:** UNNECESSARY. Dependent on the removed `_prompt_traj_ids`.

**Action:** REMOVED.

### 4. `_cross_res_prompts: list[int]`

**Location:** `src_ii/pair_sampler.py`, `BTRMPairSampler.__init__`

**What it did:** Identified which prompts had trajectories in multiple resolution tiers. This was used as a gate: cross-resolution pairs could ONLY be formed between trajectories sharing one of these prompts.

**Classification:** UNNECESSARY. The concept that cross-resolution requires same-prompt is the fundamental error this audit addresses. If the dataset has trajectories in multiple tiers, cross-resolution pairs can be formed between ANY trajectory in tier X and ANY trajectory in tier Y.

**Action:** REMOVED. The entire block (lines 565-574 in the original) has been deleted.

### 5. Cross-resolution logic in `_sample_pair_spec()`

**Location:** `src_ii/pair_sampler.py`, `_sample_pair_spec()`, lines 836-860 in the original

**What it did:** When `allow_cross_resolution=True`, checked whether:
- `self._cross_res_prompts` was non-empty
- `pos_a.prompt_idx >= 0`
- `pos_a.prompt_idx in self._prompt_traj_ids`
- Random gate: 30% probability

Then found trajectories for the SAME prompt in OTHER tiers, and sampled image B from those specific trajectories.

**Classification:** UNNECESSARY constraint (the prompt matching). The 30% probability gate is FINE -- it controls the mix of same-tier vs cross-tier pairs, which affects FLOPS budget allocation.

**Action:** REPLACED with unconstrained cross-tier sampling. The new logic:
1. Check if `allow_cross_resolution=True` and multiple tiers are populated
2. With 30% probability, sample image B from a randomly chosen OTHER tier
3. No prompt matching -- any trajectory in the other tier is eligible

**Before (prompt-constrained):**
```python
if (
    allow_cross_resolution
    and self._cross_res_prompts
    and pos_a.prompt_idx >= 0
    and pos_a.prompt_idx in self._prompt_traj_ids
    and self._rng.random() < 0.30
):
    prompt_tids = self._prompt_traj_ids[pos_a.prompt_idx]
    anchor_a = assign_budget_tier(pos_a.width, pos_a.height)
    other_tier_tids = [
        tid for tid in prompt_tids
        if assign_budget_tier(...) != anchor_a
    ]
    if other_tier_tids:
        traj_b = self._rng.choice(other_tier_tids)
        ...
```

**After (unconstrained):**
```python
if (
    allow_cross_resolution
    and len(populated) > 1
    and self._rng.random() < 0.30
):
    anchor_a = assign_budget_tier(pos_a.width, pos_a.height)
    other_tiers = [a for a in populated if a != anchor_a]
    if other_tiers:
        other_tier = self._rng.choice(other_tiers)
        pos_b = self._pick_position(tier_anchor=other_tier)
```

### 6. Same-position rejection

**Location:** `src_ii/pair_sampler.py`, `sample_pair()` and `_sample_pair_spec()`

**What it does:** Rejects pairs where `pos_a.traj_id == pos_b.traj_id AND pos_a.step_key == pos_b.step_key`.

**Classification:** NECESSARY. An image compared to itself has zero information content. This is the only structural constraint on pair formation that should exist.

### 7. `allow_inter_trajectory` / `allow_intra_trajectory`

**Location:** `src_ii/pair_sampler.py`, `BTRMPairSampler.__init__` and sampling methods

**What it does:** Controls whether pairs can come from different trajectories (inter) or the same trajectory (intra).

**Classification:** ACCEPTABLE. These are explicit caller-configurable controls, not hidden restrictions. Intra-trajectory pairs have correlated noise schedules, which may be useful or harmful depending on the experimental design. The caller makes this decision, not the sampler.

**Action:** RETAINED. No change.

### 8. `tier_anchor` constraint in `sample_pair()`

**Location:** `src_ii/pair_sampler.py`, `sample_pair()` method

**What it does:** When `tier_anchor` is passed, BOTH images are sampled from the specified tier.

**Classification:** PARTIALLY NECESSARY. The stratified batch sampler (`sample_stratified_batch`) relies on tier-constrained pairs for FLOPS allocation accounting. However, this means `sample_pair(tier_anchor=X)` produces same-tier pairs only, which is a hidden same-resolution constraint.

**Action:** RETAINED for backward compatibility. The FLOPS-budget path (`sample_macrobatch`) uses `_sample_pair_spec()` which now supports cross-resolution pairs. The `sample_pair()` method's tier constraint is only used by the legacy stratified path.

### 9. `resolution_bucket` parameter in `sample_pair()`

**Location:** `src_ii/pair_sampler.py`, `sample_pair()` method

**What it does:** Legacy binary bucket constraint ("megapixel" or "small"). Both images are sampled from the specified bucket.

**Classification:** LEGACY. Same issue as `tier_anchor` -- forces same-resolution pairs. Kept for backward compatibility but not used by any new code path.

**Action:** RETAINED for backward compatibility.

## Constraints in `btrm_training.py`

### 10. Resolution handling in the training loop

**What the code does:** Tracks `width_a, height_a, width_b, height_b` independently for each image in a pair. The bin packer handles mixed sizes. `PairSpec.to_pair_dict()` includes per-image resolution metadata.

**Classification:** NO CONSTRAINT. The training loop already correctly handles cross-resolution pairs. Both the packed path and the serial path pass per-image latents to the model independently.

**Action:** No change needed.

### 11. Single backward over all bins (FLOPS-budget path)

**Location:** `src_ii/btrm_training.py`, FLOPS-budget macrobatch path (formerly lines 803-884)

**What it did:** Scored ALL bins (J forward passes), accumulated scores in `img_idx_to_score`, computed loss over ALL pairs, then called `loss.backward()` once. This kept ALL J computation graphs in GPU memory simultaneously.

**Classification:** BUG. With J=3-8 bins, each containing a packed forward through a 5.8B parameter model with gradient checkpointing, this requires `J * per_bin_graph_size` VRAM for the backward pass. For J=5, this is 5x the VRAM of a single forward.

**Action:** FIXED with per-bin gradient accumulation. The new approach:

1. **Pre-count `active_heads`** from pair metadata (no computation graph needed). This gives a constant normalization denominator `N` that does not change as bins are processed.
2. **Build image-to-pair membership map:** for each image, track which pairs it participates in. This is needed for safe detaching.
3. **For each bin:** run forward, check which pairs now have BOTH images scored, compute BT loss for those pairs, backward `bin_bt / N` immediately.
4. **Smart detach:** after backward, only detach images whose ALL pairs have been processed. Images with unprocessed cross-bin pairs keep their computation graph alive until the partner bin is scored and the pair loss is backward'd.
5. **Memory profile:** at most 2 bins' computation graphs alive simultaneously (current bin + one prior bin with an unmatched cross-bin partner). Down from J bins.

The gradient equivalence argument: `backward(sum(bt_i) / N)` produces the same gradients as `sum(backward(bt_i / N))` because backward distributes over addition and scalar division is linear. The pre-counted `N = active_heads` is constant across all bins.

**Cross-bin pair correctness:** A pair where image A is in bin 1 and image B is in bin 3 requires both bins' computation graphs alive when the pair loss is backward'd. The smart detach logic ensures image A's graph survives until bin 3 is processed and the pair loss computed. After the pair loss backward, both images are detachable. This is correct because BT loss `bt = -logsigmoid(s_A - s_B)` has gradients flowing through both `s_A` and `s_B`, and both scores' computation graphs must be alive for `backward()` to produce gradients for both bins' adapter parameters.

## Constraints in Other Files

### `flops_sampling.py`

**Checked for:** Unnecessary bucketing constraints that restrict pair formation.

**Finding:** No pairing constraints. This module computes per-trajectory sampling weights for FLOPS-aware training. It classifies trajectories into resolution tiers for weight computation only. The weights influence which trajectories are SAMPLED, but do not restrict which images can be PAIRED.

**Action:** No change.

### `btrm_model.py` (`score_differentiable_packed()`)

**Checked for:** Assumptions that all images in a batch have the same resolution.

**Finding:** No such assumption. The method accepts `images: list[tuple[Tensor, Tensor, Tensor, int]]` where each image can have a different `(H, W)`. It builds per-image embeddings, constructs a packed sequence with block-diagonal attention masks, and unpacks hidden states per-image for scoring. Cross-resolution batches are structurally supported.

**Action:** No change.

### `validation_metrics.py` (`PairResult`)

**Checked for:** Whether it tracks both images' resolutions correctly for cross-resolution pairs.

**Finding:** Yes. `PairResult` has independent `width_a, height_a, width_b, height_b` fields. The `ValidationMetrics.update()` method computes resolution buckets for BOTH images independently and records the pair result under both buckets. Cross-resolution pairs are handled correctly by construction.

**Action:** No change.

### `preference_fn`

**Checked for:** Whether the preference function works for cross-resolution pairs.

**Finding:** The preference function is caller-supplied (passed to `train_btrm_differentiable()`). For pinkify, preference comes from the backend label (sage vs sdpa) in trajectory metadata. For thisnotthat, preference comes from step count in trajectory metadata. Both are per-trajectory properties independent of resolution. A well-implemented `preference_fn` handles cross-resolution pairs correctly because it only reads `traj_a`, `step_a`, `traj_b`, `step_b` metadata, never resolution fields.

**Action:** No change to the training loop. The preference_fn contract is correct.

## Summary of Changes

| File | Change | Lines affected |
|------|--------|----------------|
| `src_ii/pair_sampler.py` | Removed `_prompt_traj_ids` construction | ~15 lines deleted |
| `src_ii/pair_sampler.py` | Removed `_traj_prompt` construction | ~4 lines deleted |
| `src_ii/pair_sampler.py` | Removed `_cross_res_prompts` construction | ~9 lines deleted |
| `src_ii/pair_sampler.py` | Simplified `_sample_pair_spec()` cross-res logic | ~20 lines replaced with ~10 |
| `src_ii/pair_sampler.py` | Updated module docstring | Clarified universal pairing principle |
| `src_ii/pair_sampler.py` | Updated class docstring | Clarified no-constraint design |
| `src_ii/btrm_training.py` | Per-bin gradient accumulation in FLOPS-budget path | ~80 lines rewritten |
| `src_ii/btrm_training.py` | Updated `macrobatch_cross_resolution` docstring | 2 lines |

## Remaining Constraints (All Necessary)

After these changes, the pair sampler has exactly THREE constraints:

1. **Same-position rejection:** An image cannot be paired with itself.
2. **`allow_inter_trajectory` / `allow_intra_trajectory`:** Explicit caller controls for whether pairs can cross trajectory boundaries. Default: inter=True, intra=False.
3. **FLOPS budget / tier allocation:** The stratified and FLOPS-budget sampling paths control the resolution MIX of sampled pairs, but do not restrict which specific images can be paired.

None of these restrict pair formation based on content, prompt, or resolution matching. Any image can be paired with any other image.

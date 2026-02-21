# Exemplar Selection Fix: Duplicate Images Across Heads

## The Bug

After the `flops_budget_100step_v2` training run, both the pinkify and thisnotthat
scoring heads selected the **same images** as their top-3 and bottom-3 exemplars:

| Category   | pinkify traj_id | thisnotthat traj_id |
|------------|-----------------|---------------------|
| top 0      | 11              | 11                  |
| top 1      | 15              | 15                  |
| top 2      | 1               | 1                   |
| bottom 0   | 5               | 4                   |
| bottom 1   | 4               | 5                   |
| bottom 2   | 3               | 3                   |

The scores were numerically different (pinkify top-0: 1.951, thisnotthat top-0:
2.915), but the **ranking** was identical. The bottom sets contained the same
trajectories in near-identical order (only traj 4 and 5 swapped).

## Root Cause Analysis

The bug had **two contributing causes**, neither of which is in the model weights:

### Cause 1: Identical Training Signal (Not Fixed Here)

The `preference_fn` in `run_flops_budget_100step_v2.py` computed the same
preference for both heads:

```python
def preference_fn(pair: dict) -> dict:
    prefs = {}
    for pref_key in PREF_KEYS:  # ("pinkify_pref", "thisnotthat_pref")
        # SAME sigma comparison for both heads
        if sigma_a < sigma_b - 0.001:
            prefs[pref_key] = 1
        elif sigma_b < sigma_a - 0.001:
            prefs[pref_key] = -1
        else:
            prefs[pref_key] = 0
    return prefs
```

Both heads were trained on the identical "lower sigma wins" signal. The only
thing differentiating them was the random initialization of their projection
columns in `nn.Linear(3840, 2, bias=False)`. Over 100 training steps, both
columns learned highly correlated projections, producing the same ranking.

This is a **training configuration issue**, not a code bug. The trained weights
are valid -- both heads correctly learn to prefer lower-sigma images. A future
training run should use distinct preference functions (e.g., pinkify uses
VAE-decoded pinkify_score, thisnotthat uses VAE-decoded thisnotthat_score).

### Cause 2: No Deduplication in Exemplar Selection (Fixed)

The `render_exemplars()` function iterated over heads independently but did
not track which images had already been selected. With correlated rankings,
each head's `sort -> take top-K/bottom-K` produced the same image set.

### Cause 3: Homogeneous Scoring Pool (Fixed)

Phase 6 of the training script scored only 18 "final" positions (all at
sigma=0.0). With every scored image at the same sigma, the only ranking
signal was the per-trajectory hidden representation quality. Including
images at diverse sigma levels (step_29, step_15, step_04) gives the
heads a wider distribution to rank, increasing the chance of divergent
rankings even with correlated training signals.

## The Fix

### 1. Deduplication in `render_exemplars()` (`src_ii/exemplar_renderer.py`)

Added `deduplicate_across_heads: bool = True` parameter (default True).
When enabled, images selected for an earlier head's top-K or bottom-K are
excluded from later heads' candidate pools:

```python
used_keys_top: set[str] = set()
used_keys_bottom: set[str] = set()

for head in head_names:
    # Select top-K, skipping already-used keys
    top_k_items = []
    for img_key, score_val in reversed(head_scores):
        if deduplicate_across_heads and img_key in used_keys_top:
            continue
        top_k_items.append((img_key, score_val))
        if len(top_k_items) >= top_k:
            break
    # ... same for bottom-K ...
    used_keys_top.update(img_key for img_key, _ in top_k_items)
```

Top and bottom selections are deduplicated independently (a top-image for
head A can still be a bottom-image for head B, which is informative).

### 2. Rank Correlation Diagnostics

Added `_compute_rank_correlation()` which computes Spearman's rho between
each pair of heads' score rankings. The result is:
- Printed as a WARNING when rho > 0.9
- Included in `exemplars_manifest.json` under `rank_correlations`
- Allows automated flagging of correlated heads

### 3. All Scores Saved to Disk

`all_scores.json` is now written alongside the manifest, containing every
image's per-head score. This enables post-hoc analysis without re-scoring.

### 4. Diverse Scoring Pool in Training Script

Phase 6 of `run_flops_budget_100step_v2.py` now scores images at four
sigma levels:
- 12 "final" positions (sigma=0.0)
- 6 "step_29" positions (near-clean sigma)
- 6 "step_15" positions (moderate sigma)
- 6 "step_04" positions (noisy sigma)

Total: up to 30 images scored (was: 18 at sigma=0 only).

## Verification

Six pure-Python unit tests in `tests/test_exemplar_dedup.py`:

1. **test_rank_correlation_computation** -- Verifies Spearman rho = 1.0 for
   perfectly correlated scores, -1.0 for anticorrelated, ~0 for uncorrelated.

2. **test_correlated_heads_without_dedup** -- Confirms that without dedup,
   correlated heads produce identical top/bottom sets (the original bug).

3. **test_correlated_heads_with_dedup** -- Confirms that with dedup, correlated
   heads produce non-overlapping top/bottom sets. Head A gets its true top-3;
   head B gets the next-best-3.

4. **test_anticorrelated_heads_no_dedup_needed** -- Verifies that dedup is a
   no-op when heads naturally produce different rankings.

5. **test_manifest_includes_diagnostics** -- Checks that `deduplicated`,
   `rank_correlations`, and `n_images_scored` fields appear in the manifest.

6. **test_all_scores_saved_to_disk** -- Confirms `all_scores.json` is written.

All 6 tests pass.

## What Was NOT Changed

- **Model weights**: The trained BTRM compound model is unchanged. Both heads
  are valid; they just happen to have learned correlated rankings due to
  identical training signals.

- **Training loop**: `btrm_training.py` is unchanged. The preference_fn
  duplication is a caller-side issue in the training script, not a library bug.

- **ScoreUnembedder**: The `nn.Linear(3840, 2)` projection correctly produces
  per-head scores. The indexing `score_tensor[0, head_idx]` in
  `render_exemplars_from_model()` was already correct.

## Files Modified

- `src_ii/exemplar_renderer.py` -- Added deduplication, rank correlation
  diagnostics, all_scores.json dump
- `scripts_ii/run_flops_budget_100step_v2.py` -- Phase 6 now scores images
  at diverse sigma levels (4 layers instead of 1)
- `tests/test_exemplar_dedup.py` -- 6 new unit tests (new file)

# Validation Covariance Infrastructure for BTRM Training

**Date:** 2026-02-18
**Author:** Layer 4 subagent
**Module:** `src_ii/validation_metrics.py`
**Provenance:** Layer 4 of funfetti batching integration (from `docs/root_claude_funfetti_batching_state_of_affairs.md`, Section 5)

---

## 1. What This Module Does and Why It Matters

The funfetti batching hypothesis is that training a BTRM reward model on
mixed-resolution packed batches (33% megapixel FLOPS, 67% sub-megapixel)
achieves equivalent or better accuracy-per-NFE compared to monotonic
megapixel sampling. To evaluate this hypothesis, we need per-resolution
accuracy tracking: "how well does the model discriminate at 512x512
given that it was also trained on 256x256 in the same macrobatch?"

The existing training loop (`train_btrm_differentiable()` in
`src_ii/btrm_training.py`) tracks per-head accuracy as simple running
counters, producing per-step JSONL entries like:

```json
{
  "step": 42,
  "loss": 0.45,
  "accuracy_pinkify": 1.0,
  "accuracy_thisnotthat": 0.0,
  "pair_weight": 0.87,
  ...
}
```

This tells us nothing about resolution-specific performance. A model that
achieves 90% accuracy on 1280x832 and 40% on 256x256 reports the same
overall accuracy as one that achieves 65% on both. The former has a
resolution bias problem; the latter is well-calibrated. The existing
metrics cannot distinguish these cases.

`ValidationMetrics` is a multi-indexed accuracy and loss tracker that
records per-pair results indexed by:

- **Resolution bucket** (megapixel-quantized: <0.1, 0.1-0.2, 0.2-0.4, 0.4-0.8, 0.8-1.2, >=1.2 MP)
- **LogSNR bucket** (sigma-derived noise level: very noisy, noisy, moderate, clean-ish, near-clean, clean)
- **Head name** (pinkify, thisnotthat, or any future scoring head)
- **Aspect ratio** (portrait, square, landscape)
- **Trajectory source** (original_v1, gpu_rollout, policy_rollout)

For each index combination, it tracks running mean, variance, and count
of both accuracy and loss, plus the running covariance between accuracy
and loss. All statistics use Welford's online algorithm -- no raw values
are accumulated in memory.

---

## 2. The Bucketing Scheme

### Resolution Buckets (by megapixels)

| Bucket | Range | Example resolutions |
|--------|-------|-------------------|
| < 0.1 MP | 0-65K pixels | 256x256 (65K) |
| 0.1-0.2 MP | 65K-200K | 320x320 (102K) |
| 0.2-0.4 MP | 200K-400K | 384x384 (147K), 512x512 (262K) |
| 0.4-0.8 MP | 400K-800K | 704x704 (496K) |
| 0.8-1.2 MP | 800K-1.2M | 1024x1024 (1049K), 1280x832 (1064K) |
| >= 1.2 MP | 1.2M+ | Large images |

Bucketing by megapixels rather than exact (W, H) pairs collapses aspect
ratio variants into the same compute-cost tier. A 1280x832 image and an
832x1280 image cost the same FLOPS and should be compared at the same
resolution tier.

### LogSNR Buckets

The spec prescribes `logSNR = -2 * log(sigma)` as the bucketing axis. This
is the standard signal-to-noise ratio definition for variance-exploding
diffusion, where larger logSNR means cleaner images.

| Bucket | logSNR range | Sigma range | Interpretation |
|--------|-------------|-------------|----------------|
| very noisy | < -2 | > ~2.7 | Dominated by noise (sigma > 1 in some schedules) |
| noisy | [-2, 0) | ~1-2.7 | Heavy noise |
| moderate | [0, 2) | ~0.37-1 | Balanced signal/noise |
| clean-ish | [2, 5) | ~0.08-0.37 | Signal dominates |
| near-clean | [5, inf) | < 0.08 | Nearly clean |
| clean | sigma = 0 | 0 | Final denoised image |

For the typical 30-step schedule with sigma in [0, ~0.97], most images land
in the moderate-to-near-clean range. The "very noisy" and "noisy" buckets
are relevant for aggressive sigma schedules or noise-augmented training.

**Note on formula choice:** The pair_sampler uses `logSNR = 2 * ln((1-sigma)/sigma)`
(the CONST noise model's specific formula), while the bucketing spec
prescribes `logSNR = -2 * log(sigma)` (the standard VE definition). These
are different functions but both monotonically decreasing in sigma.
The bucketing uses the spec's formula because it is the standard one from
the diffusion literature and produces well-separated buckets for the
sigma range encountered in practice.

### Aspect Ratio Buckets

| Bucket | W/H range | Examples |
|--------|-----------|---------|
| portrait | < 0.8 | 832x1280 (0.65) |
| square | 0.8-1.2 | 512x512 (1.0), 1024x1024 (1.0) |
| landscape | >= 1.2 | 1280x832 (1.54) |

Aspect ratio bucketing is coarse because it exists primarily for
cross-tabulation with resolution (e.g., "does the model struggle with
portrait images at 704x704?"), not as a primary analysis axis.

---

## 3. The Running Statistics Algorithm

### Welford's Online Algorithm

For each index combination, mean and variance are tracked using Welford's
(1962) one-pass algorithm. The state is three numbers: `(count, mean, m2)`
where `m2 = sum((xi - mean)^2)` is the sum of squared deviations from the
current running mean.

```
update(x):
    count += 1
    delta = x - mean
    mean += delta / count
    delta2 = x - mean  # using UPDATED mean
    m2 += delta * delta2

variance = m2 / count  # population variance
```

This is numerically stable because it never computes `sum(x^2)` minus
`sum(x)^2`, which suffers from catastrophic cancellation when the mean
is large relative to the variance. For accuracy values (0 or 1) and
loss values (~0.3-0.7), the numerical benefit is modest, but the
algorithm costs nothing extra and is the correct choice.

### Welford's Merge (Chan et al., 1979)

Two accumulators can be merged (for parallel or checkpoint-resume scenarios):

```
combined_count = count_A + count_B
delta = mean_B - mean_A
combined_mean = (mean_A * count_A + mean_B * count_B) / combined_count
combined_m2 = m2_A + m2_B + delta^2 * count_A * count_B / combined_count
```

### Online Covariance (Pebay, 2008)

The accuracy-loss covariance uses the one-pass co-moment formula:

```
update(x, y):
    count += 1
    dx = x - mean_x
    mean_x += dx / count
    dy = y - mean_y
    mean_y += dy / count
    co_moment += dx * (y - mean_y)  # note: uses NEW mean_y

covariance = co_moment / count
```

The covariance tells us whether low-loss pairs correspond to high-accuracy
pairs within each bucket. A strongly negative covariance (accuracy up, loss
down) indicates the model is learning a coherent signal. A near-zero
covariance despite good accuracy suggests the loss landscape is noisy at
that resolution or noise level.

---

## 4. The Serialization Format

### In-memory State

The tracker maintains 9 dictionaries of `_MetricsCell` objects:

1. `_global` -- one cell for all pairs
2. `_by_head` -- keyed by head name
3. `_by_resolution` -- keyed by resolution bucket label
4. `_by_logsnr` -- keyed by logSNR bucket label
5. `_by_aspect` -- keyed by aspect ratio bucket label
6. `_by_source` -- keyed by trajectory source label
7. `_by_head_resolution` -- keyed by `"{head}|{resolution}"`
8. `_by_head_logsnr` -- keyed by `"{head}|{logsnr}"`
9. `_by_resolution_logsnr` -- keyed by `"{resolution}|{logsnr}"`

Each `_MetricsCell` contains 5 Welford accumulators (accuracy, loss,
preferred score, rejected score) and 1 covariance accumulator
(accuracy-loss covariance). Total state per cell: 5 * 3 + 4 = 19 floats.

### JSON Persistence

The full state serializes to a JSON file via `save_json()`:

```json
{
  "version": 1,
  "n_updates": 100,
  "global": {
    "accuracy": {"count": 100, "mean": 0.66, "m2": 22.44},
    "loss": {"count": 100, "mean": 0.45, "m2": 2.08},
    "acc_loss_cov": {"count": 100, "mean_x": 0.66, "mean_y": 0.45, "co_moment": -0.12},
    "score_preferred": {"count": 100, "mean": 2.5, "m2": 0.83},
    "score_rejected": {"count": 100, "mean": 0.75, "m2": 0.62}
  },
  "by_head": {
    "pinkify": { ... },
    "thisnotthat": { ... }
  },
  "by_resolution": {
    "0.2-0.4 MP": { ... },
    "0.8-1.2 MP": { ... }
  },
  ...
}
```

The file is written atomically (temp file + `os.replace()`) to prevent
corruption if the process is interrupted mid-write. Restoration is via
`ValidationMetrics.load_json()`, which is a lossless round-trip:
`from_dict(to_dict())` recovers the exact accumulator state.

### JSONL Summary

The `summary()` method returns a flat dict suitable for appending to the
existing training metrics JSONL:

```json
{
  "n_updates": 100,
  "global_accuracy": 0.66,
  "global_loss": 0.45,
  "global_count": 100,
  "val_accuracy_pinkify": 0.70,
  "val_loss_pinkify": 0.42,
  "val_n_pinkify": 50,
  "val_accuracy_thisnotthat": 0.62,
  "val_n_thisnotthat": 50,
  "val_accuracy_res_0p2-0p4_MP": 0.64,
  "val_n_res_0p2-0p4_MP": 25,
  "val_accuracy_res_0p8-1p2_MP": 0.68,
  "val_n_res_0p8-1p2_MP": 75,
  ...
}
```

The key naming convention uses sanitized bucket labels (`p` for `.`,
underscores for spaces) to produce valid JSON keys. The `val_` prefix
distinguishes validation metrics from the existing training metrics
(`accuracy_pinkify`, `loss`, etc.).

---

## 5. Integration with the Training Loop

The tracker is a passive observer. It does not modify the training loop's
control flow, loss computation, or optimizer behavior. Integration is
three lines:

```python
# At training start:
from src_ii.validation_metrics import ValidationMetrics, PairResult
tracker = ValidationMetrics()

# After each pair is scored (inside the microbatch loop):
tracker.update(PairResult(
    head_name=head_name,
    correct=correct,
    loss_contribution=bt_loss_item,
    score_preferred=pos_score_item,
    score_rejected=neg_score_item,
    width_a=width_a, height_a=height_a, sigma_a=sigma_a,
    width_b=width_b, height_b=height_b, sigma_b=sigma_b,
    source_a=source_a, source_b=source_b,
    traj_a=traj_a, step_a=step_a,
    traj_b=traj_b, step_b=step_b,
))

# At log_interval or end of training:
entry.update(tracker.summary())  # merges into the JSONL entry
tracker.save_json(output_dir / "validation_metrics.json")
```

The `PairResult` requires all metadata as plain Python types (no tensors).
The training loop calls `.item()` on any tensor values before constructing
the `PairResult`.

The width/height and source metadata come from the dataset. When using the
`pair_sampler`, the pair dict contains `traj_a`, `step_a`, `sigma_a`, etc.
The width/height can be looked up from the dataset reader's metadata.
When this metadata is not available (e.g., legacy training pairs), the
defaults (1280x832, source="unknown") are used.

### Where the metadata comes from

In the `pair_sampler` path:
- `sigma_a`, `sigma_b`: directly from `pair_sampler.sample_pair()`
- `traj_a`, `step_a`, `traj_b`, `step_b`: from the pair dict
- `width_a`, `height_a`, `width_b`, `height_b`: from dataset metadata lookup
- `source_a`, `source_b`: from dataset `run_name` column

In the `training_pairs` path:
- Resolution is uniform (all 1280x832 in the current V2 dataset)
- Sigma comes from `sigma_lookup_fn`
- Source is "unknown" unless the training script populates it

---

## 6. Example Output

After 100 updates with mixed resolutions, heads, and sigma values, the
resolution breakdown table looks like this:

```
Bucket                                      Count   Accuracy    Acc Std       Loss   Loss Std Acc-Loss Cov
----------------------------------------------------------------------------------------------------
0.2-0.4 MP                                     25     0.6400     0.4800     0.4600     0.1442    -0.000000
0.8-1.2 MP                                     95     0.6632     0.4726     0.4500     0.1440     0.001211
< 0.1 MP                                       20     0.6500     0.4770     0.4625     0.1442    -0.004375
----------------------------------------------------------------------------------------------------
TOTAL                                         100     0.6600     0.4737     0.4525     0.1443     0.000000
```

Cross-indexed by head and resolution:

```
Bucket                                      Count   Accuracy    Acc Std       Loss   Loss Std Acc-Loss Cov
----------------------------------------------------------------------------------------------------
0.2-0.4 MP                                     25     0.6400     0.4800     0.4600     0.1442    -0.000000
0.8-1.2 MP                                     45     0.6667     0.4714     0.4500     0.1438     0.000667
< 0.1 MP                                       10     0.6000     0.4899     0.4750     0.1436    -0.000000
----------------------------------------------------------------------------------------------------
TOTAL                                         100     0.6600     0.4737     0.4525     0.1443     0.000000
```

These tables are generated by `format_table()` and can be printed at
log intervals during training for real-time monitoring.

### Reading the tables

The key columns for the funfetti evaluation:

1. **Count**: How many pair results contributed to this bucket. Highly
   uneven counts (e.g., 95 for 0.8-1.2 MP vs 5 for <0.1 MP) indicate
   the current sampling PDF does not target resolution diversity.

2. **Accuracy Mean**: The mean accuracy for this bucket. If funfetti
   training is working, all resolution buckets should converge to
   similar accuracy (no resolution bias).

3. **Acc Std**: The standard deviation of accuracy within the bucket.
   A high std indicates the model's performance is inconsistent for
   this resolution tier.

4. **Acc-Loss Cov**: The covariance between accuracy and loss. Should
   be negative (accuracy goes up as loss goes down). A near-zero
   covariance suggests the loss signal is noisy for this bucket --
   the model is not learning a coherent discrimination signal at
   this resolution.

---

## 7. Design Decisions

### Why not torch tensors?

The tracker uses plain Python floats and dicts, not torch tensors. This is
intentional:

1. The tracker accumulates statistics over thousands of steps. Keeping
   torch tensors alive for statistics would waste GPU memory on a task
   that is inherently CPU-bound (three additions per update).

2. The tracker must serialize to JSON. JSON serialization of torch tensors
   requires `.item()` conversion anyway.

3. The tracker has no compute-heavy operations. Welford's update is 5
   floating-point operations per accumulator per update. There is no
   performance argument for GPU acceleration.

### Why per-pair indexing, not per-image?

Each pair contributes to the index buckets of **both** images. If image A
is 512x512 and image B is 1024x1024, the pair's accuracy is recorded under
both "0.2-0.4 MP" and "0.8-1.2 MP" resolution buckets. This means a single
update can increment multiple cells.

The alternative -- tracking per-image statistics -- would require separate
accuracy tracking for the preferred and rejected images, which does not
map cleanly onto the pairwise BT loss. The question "how accurate is the
model at 512x512?" means "when a pair includes a 512x512 image, how often
does the model correctly identify the winner?" This is naturally a
per-pair, per-image-in-pair metric.

### Why not a database?

SQLite or DuckDB would provide arbitrary query flexibility. The tracker
uses flat dicts because:

1. The number of unique index combinations is bounded (~200-500 cells).
   There is no scalability argument for a database.

2. JSON serialization is trivially portable. The output can be consumed
   by any analysis tool without a database dependency.

3. The module has zero external dependencies (pure Python + stdlib).
   Adding a database would be the first non-stdlib dependency.

### Why pre-computed cross-indices?

The tracker pre-computes 3 cross-indexed stores (head x resolution,
head x logsnr, resolution x logsnr) rather than supporting arbitrary
multi-key queries. This is because:

1. These 3 cross-indices answer the primary questions: "does funfetti
   training improve accuracy at 1024x1024 for pinkify?" (head x resolution),
   "does the model learn differently at high vs low noise?" (head x logsnr),
   "does resolution affect noise-level sensitivity?" (resolution x logsnr).

2. Arbitrary N-way cross-indexing would require storing a cell for every
   observed combination, which could grow combinatorially. The 3 chosen
   cross-indices are the most analytically valuable.

3. If additional cross-indices are needed, they can be added by following
   the same pattern: one dict, one line in `update()`, one restore in
   `from_dict()`.

---

## 8. Verification

The module was tested against 8 test cases:

1. **WelfordAccumulator correctness**: Mean and variance on [2,4,4,4,5,5,7,9] match analytical values (mean=5.0, var=4.0).
2. **Welford merge**: Split the data into two halves, accumulate separately, merge. Result matches single-pass accumulation.
3. **WelfordCovariance correctness**: Positive linear correlation (y=2x) gives cov=4.0. Negative correlation (y=-2x+12) gives cov=-4.0.
4. **Resolution bucketing**: All 7 reference resolutions from the spec (256x256 through 2048x2048) land in the correct buckets.
5. **LogSNR bucketing**: sigma=0 -> clean, sigma=0.034 -> near-clean, sigma=0.5 -> moderate, sigma=1.0 -> very noisy.
6. **Aspect bucketing**: 1280x832 -> landscape, 512x512 -> square, 832x1280 -> portrait.
7. **Full round-trip**: 100 mixed-index updates, JSON serialization, deserialization, exact bit-for-bit match.
8. **Covariance sign**: Perfect inverse correlation (accuracy=1 when loss=0.1, accuracy=0 when loss=0.9) produces negative covariance (-0.2).

All tests pass under the Windows Python venv via WSL2.

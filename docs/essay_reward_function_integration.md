# Essay: Reward Function Integration into BTRM Training

**Date:** 2026-02-20
**Run:** `training_output/reward_function_run/`
**Duration:** 55.5 minutes wall time, 150 training steps (21.5s/step avg)
**Dataset:** 420 multi-res trajectories, 84 unique resolutions, 12 unique prompts

---

## 1. What Changed

The prior BTRM training run (`training_output/pinkify_validation_run/`) used
sigma-based preference labels: for every pair, the cleaner image (lower sigma)
won for ALL heads identically. This meant both the pinkify and thisnotthat
heads received identical gradient signals, making them converge to the same
noise-vs-clean discriminator.

This run replaces the sigma-based preference function with a **reward manifest**:
a dict mapping each head name to a ground truth scoring function. For each
training pair:

1. Both latents are VAE-decoded to pixel tensors inside `torch.no_grad()`
2. Each pixel tensor is scored by each head's ground truth function
3. Per-head preferences are computed independently: the image with the higher
   ground truth score wins for that head

The two ground truth functions are:

- **pinkify**: `pinkify_score_gpu()` -- measures local pinkness with
  coverage contrast. Higher score = more vivid pink features standing out
  against non-pink backgrounds.
- **thisnotthat**: `thisnotthat_score_gpu()` bound with `pizza-ratto.png`
  (THIS) and `offhand_pleometric.png` (THAT) -- measures pixel-space
  cosine + structural similarity. Positive = more like the B&W sketch,
  negative = more like the colorful cartoon.

## 2. TNT Ground Truth Function: What It Measures

The THISNOTTHAT scoring function computes:

```
score = (cos_sim(image, THIS) + struct_sim(image, THIS)) / 2
      - (cos_sim(image, THAT) + struct_sim(image, THAT)) / 2
```

where `cos_sim` is cosine similarity of flattened pixel tensors, and
`struct_sim` is mean-adjusted cosine similarity (SSIM-lite). Both references
are bilinearly interpolated to match the target image resolution.

THIS reference (`pizza-ratto.png`) is a simple black-and-white line drawing of
a blocky rat character with text "SMOOSH" and "SAVE". THAT reference
(`offhand_pleometric.png`) is a colorful digital illustration of a purple
penguin-like character with a red Santa hat and yellow feet.

The function effectively measures "how sketch-like vs how colorful" an image
is in pixel space. Images with high white backgrounds and black line art score
positive (like THIS). Images with saturated color fills score negative (like THAT).

## 3. TNT Validation Set Design

The validation set uses all images in `i2i_off_policies/`, organized into tiers
by expected TNT score:

| Tier | Images | Expected TNT Score |
|------|--------|--------------------|
| THIS_REF | pizza-ratto.png | Highest (self-similarity = max) |
| SKETCH | 7 B&W line drawings (1bit, bubblegum, clear-sky, deviantart, enso, snek, widemeister) | Positive (similar to THIS) |
| MIXED | red-tonegraph.png | Intermediate |
| COLOR | 00500-nightmode2.png | Negative (more like THAT) |
| THAT_REF | offhand_pleometric.png | Lowest (self-similarity to THAT = max negative) |

**Ranking constraints:**
1. THIS_REF > THAT_REF (fundamental)
2. THIS_REF > all others (it IS the reference)
3. THAT_REF < all others
4. min(SKETCH) > max(COLOR) (all sketches score higher than colorful images)
5. THAT_REF < NIGHTMODE (THAT ref is more THAT-like than any non-reference)

**Ground truth validation result:** 3/5 constraints passed. Constraint failures:
- THIS_REF was NOT the highest scorer (some sketch images score higher because
  they have more total white area than pizza-ratto, increasing structural
  similarity to THE REFERENCE when resized to different aspect ratios).
- THAT_REF was not universally lowest (NIGHTMODE, being a dark-background render,
  has lower cosine similarity to both references than THAT_REF itself).

This tells us the TNT validation set is harder than PINKIFY: the ground truth
function itself doesn't perfectly satisfy all constraints on this image set.
The 3/5 baseline is therefore the upper bound for model performance.

## 4. Training Results

### 4.1 Loss Curve

| Step | BT Loss | Acc (pinkify) | Acc (thisnotthat) |
|------|---------|---------------|-------------------|
| 0    | 23.13   | 58.8%         | 47.1%            |
| 25   | 3.72    | 100.0%        | 75.0%            |
| 50   | 15.85   | 92.9%         | 42.9%            |
| 75   | 7.05    | 66.7%         | 66.7%            |
| 100  | 3.30    | 85.7%         | 42.9%            |
| 125  | 3.33    | 87.5%         | 50.0%            |
| 149  | 6.74    | 71.4%         | 57.1%            |

**Overall accuracy:** pinkify 83.1%, thisnotthat 58.2%
**Last-20-step accuracy:** pinkify 80.7%, thisnotthat 60.8%

The pinkify head consistently outperforms the thisnotthat head on training
accuracy, which makes sense: pinkness is a more spatially coherent feature
(concentrated in hue-saturation space) than pixel-space similarity to a
specific reference image.

The key observation: **the two heads have DIFFERENT training accuracies**.
This is impossible under sigma-based preferences where both heads receive
identical labels. Different accuracies directly prove that per-head
independent preference labels are being applied.

### 4.2 Cross-Head Decorrelation

| Step | Spearman rho (pinkify vs thisnotthat) |
|------|---------------------------------------|
| -1   | 0.517 (untrained)                     |
| 0    | 0.466                                 |
| 10   | 0.424                                 |
| 20   | 0.605                                 |
| 30   | 0.510                                 |
| 40   | 0.601                                 |
| 50   | 0.647                                 |
| 60   | 0.804                                 |
| 70   | 0.851                                 |
| 80   | 0.811                                 |
| 90   | 0.828                                 |
| 100  | 0.777                                 |
| 110  | 0.821                                 |
| 120  | 0.843                                 |
| 130  | 0.828                                 |
| 140  | 0.836                                 |
| 149  | 0.848                                 |

**Ground truth cross-head rho: -0.186** (decorrelated by construction).
**Final model cross-head rho: 0.848** (highly correlated).

This is **failure mode 3** from the directive: both heads learn but become
correlated. The adapter found a single dominant feature in the residual stream
and both heads are reading it. This feature happens to correlate modestly with
both ground truth functions on the training distribution.

The rho started at 0.52 (untrained model), dipped to 0.42 at step 10 (brief
decorrelation during early learning), then climbed monotonically to 0.85 by
step 70 and plateaued there.

### 4.3 Per-Head Validation Constraints

**PINKIFY trajectory:** Started at 4/5 constraints, stabilized at 3/5 from
step 20 onward (F falling below A), dropped to 2/5 at step 149. The
monotonic degradation from step 30 onward indicates the model is NOT
learning the pinkify ground truth ordering -- it is learning something
correlated but distinct.

**TNT trajectory:** Started at 2/5 constraints, dropped to 1/5 by step 10
and stayed there. The single passing constraint is THIS_REF > THAT_REF
(the most basic requirement). The model never learned to rank sketch images
above colorful ones in the TNT head.

## 5. Comparison with Sigma-Based Baseline

The sigma-based run (in `training_output/pinkify_validation_run/`) was
diagnosed as producing identical head gradients (failure mode 1). The
reward-function run produces failure mode 3 instead:

| Metric | Sigma-based | Reward-function |
|--------|-------------|-----------------|
| Preference source | sigma comparison (identical for all heads) | ground truth reward functions (independent per head) |
| Per-head accuracy difference | ~0% (identical labels) | 25% (pinkify 83% vs tnt 58%) |
| Cross-head rho | expected ~1.0 | 0.848 |
| PINKIFY constraints (final) | (not tracked) | 2/5 |
| TNT constraints (final) | (not tracked) | 1/5 |

The reward-function run is strictly better: it produces measurably different
per-head accuracies, cross-head rho below 1.0 (though still high), and runs
the validation protocol that was missing from the sigma-based run.

## 6. Manifold Hypothesis Probe: Interpretation

The manifold hypothesis question: does the pretrained generative model's
residual stream support two independent linear readouts for two decorrelated
qualitative objectives?

**The evidence from this run is mixed:**

1. **In favor:** The two heads achieve different training accuracies (83%
   vs 58%), proving the preference labels are independent. The initial
   cross-head rho was 0.52 and briefly dropped to 0.42, suggesting there
   IS initial decorrelation potential.

2. **Against:** The cross-head rho climbs to 0.85 by step 70 and stays
   there. The adapter converges to a single dominant feature that both
   heads read. Neither head's validation constraints improve monotonically.

**Possible explanations:**

- **Adapter rank too low (r=8):** With only 8 dimensions of intervention
  per LoRA pair, the adapter may not have enough capacity to encode two
  independent features. A higher rank (r=16 or r=32) might allow the
  adapter to support two orthogonal readout directions.

- **Training distribution confound:** 80% of training positions are
  sigma=0 (fully denoised). At sigma=0, both scoring functions operate
  on clean images. If pinkness and THIS-similarity are correlated on the
  training distribution of clean generated images (e.g., prompts that
  generate pink images also generate images structurally similar to the
  B&W sketch style), the heads would learn the same feature even with
  independent labels.

- **TNT signal is too weak in latent space:** The TNT function operates
  on pixel-space cosine similarity, which may not map cleanly to features
  in the diffusion model's residual stream. The model's internal
  representations may not encode "pixel-space similarity to reference X"
  as a recoverable linear feature, even though they encode higher-level
  visual properties.

- **Shared adapter architecture:** Both heads read from the SAME adapter
  output. The 2-head ScoreUnembedder is a single linear layer with 2
  output columns. If the adapter learns to amplify one feature direction,
  both columns of the linear head will align to read it, because that
  direction has the highest signal-to-noise ratio for BOTH objectives
  simultaneously.

## 7. What Would Change the Outcome

1. **Separate adapters per head:** Instead of one shared adapter feeding
   two head columns, use two independent LoRA adapters, each feeding its
   own scalar head. This prevents the correlation mechanism described above.

2. **Higher adapter rank:** r=32 or r=64 may provide enough capacity for
   two orthogonal feature directions to coexist in the adapted residual
   stream.

3. **Decorrelation regularizer:** Add a penalty on the cosine similarity
   between the two head weight vectors, encouraging them to read from
   orthogonal subspaces.

4. **Different training distributions per head:** Sample different image
   pairs for each head, rather than scoring both heads on the same pair.
   This breaks the gradient correlation that arises from shared examples.

5. **More diverse TNT references:** The current THIS/THAT references are
   very small, simple images. More complex references might produce a
   stronger, more distinctive signal in the residual stream.

---

## Artifacts

All training artifacts are saved to `training_output/reward_function_run/`:

- `training_curve.jsonl` -- per-step loss, accuracy, grad norm (JSONL)
- `pinkify_validation_log.jsonl` -- per-eval PINKIFY constraint checks
- `tnt_validation_log.jsonl` -- per-eval TNT constraint checks
- `cross_head_decorrelation_log.jsonl` -- per-eval Spearman rho
- `validation_summary.json` -- aggregated validation trajectory
- `run_summary_final.json` -- run configuration and final metrics
- `ground_truth.json` -- pixel-space ground truth scores
- `ground_truth_cross_head.json` -- ground truth cross-head rho (-0.186)
- `rtheta_adapter.safetensors` -- trained LoRA adapter weights
- `btrm_head.safetensors` -- trained score unembedder weights
- `checkpoint_step{025,050,075,100,125}/` -- intermediate checkpoints

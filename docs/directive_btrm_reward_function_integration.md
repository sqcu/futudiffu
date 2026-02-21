# Directive: BTRM Reward Function Integration

**Date:** 2026-02-20
**Author:** Root session
**Status:** Active directive. Supersedes any prior training scripts that use
sigma-based preference labels.
**Audience:** Any agent modifying `btrm_training.py`, training scripts in
`scripts_ii/`, or the pair sampling pipeline.

---

## 1. Research Context

This is not function approximation. The BTRM low-rank adapter is a probe
into the pretrained generative model's residual stream. The research
question is:

> Do the activations of a pretrained image generation model already
> contain feature detectors sufficient to support arbitrary qualitative
> objective functions, extractable via low-rank intervention on the
> residual stream alone?

The LoRA adapter has no access to the underlying model weights. It
intervenes only on activations (the residual stream). The final linear
head reads two scalar outputs from these adapted activations. If training
succeeds — both heads learn their respective ground truth functions AND
remain decorrelated with each other — it is evidence that:

1. The pretrained model's representations are rich enough to encode
   these qualitative distinctions.
2. The necessary "circuits" are already present and merely need to be
   pointed at.
3. Two decorrelated objectives can be projected simultaneously from
   the same residual stream without interference.

This is a modest claim in the geometry of deep learning (that a
high-dimensional residual stream supports multiple independent linear
readouts), but a bold claim in alignment (that pretrained models
contain the representations needed to evaluate arbitrary human-legible
quality axes, recoverable by cheap PEFT probes rather than expensive
SAE decompositions).

The two-head design is the strong form of the test. A single head
proving correlation with one ground truth function could be coincidence
or overfitting. Two heads, decorrelated, each tracking a different
ground truth, is much harder to dismiss.

### Why image generation models that also write text

The methodology generalizes. A model architecture that handles both
visual and textual tokens makes the manifold hypothesis claim harder
to refute — the representations must be general enough to support
multimodal reasoning, and qualitative visual objectives are a strict
subset of that generality.

---

## 2. Training Data Flow Contract

The training pipeline for BTRM reward model heads is:

```
For each training pair (latent_a, latent_b):
    1. VAE-decode latent_a → pixel_a   (torch.no_grad, not in training graph)
    2. VAE-decode latent_b → pixel_b   (torch.no_grad, not in training graph)
    3. For each head h in manifest:
        score_a_h = manifest[h](pixel_a)   # ground truth scoring function
        score_b_h = manifest[h](pixel_b)   # ground truth scoring function
        preference_h = +1 if score_a_h > score_b_h else -1
    4. Forward both latents through backbone + adapter → hidden states
    5. Project hidden states through linear head → (scalar_a_h, scalar_b_h) per head
    6. BT loss per head: -log(sigmoid(preference_h * (scalar_a_h - scalar_b_h)))
    7. Aggregate loss across heads, normalize by active_heads
```

### Critical distinctions

- **Steps 1-3 are outside the training graph.** VAE decode and reward
  function evaluation happen in `torch.no_grad()`. They produce
  preference LABELS, not gradients. The gradient flows only through
  steps 4-6 (backbone forward → adapter → head → loss).

- **Each head gets INDEPENDENT preference labels.** For a given pair,
  the pinkify head might prefer image A (because A is pinker) while
  the thisnotthat head prefers image B (because B has more of whatever
  TNT measures). This per-head independence is what enables
  decorrelated learning.

- **The reward functions are the sole source of preference labels.**
  Not sigma. Not attention backend. Not step count. Not any metadata
  field. The model learns to predict relative magnitude of the actual
  ground truth scoring functions from latent representations alone.

- **Sigma still matters for SAMPLING** (which positions to draw from
  trajectories) but NOT for labeling. The clean-biased sampling
  (80% sigma=0) controls which training examples the model sees.
  The reward function scores control what the model learns from them.

### VAE lifecycle during training

The VAE (~160 MB) is loaded before the training loop begins (after the
text encoder is freed) and remains loaded throughout training. It is
used only for decode (latent → pixel) in `torch.no_grad()` mode. It is
not part of the training graph and receives no gradients.

---

## 3. Reward Function Manifest

The training loop accepts a manifest: a dict mapping head name strings
to callable scoring functions. The manifest is the ONLY place where
head-specific knowledge lives. The training loop is generic.

```python
reward_manifest = {
    "pinkify": pinkify_score_gpu,       # from src_ii.reward_functions
    "thisnotthat": thisnotthat_score_gpu, # from src_ii.reward_functions
}
```

### Scoring function interface contract

Each scoring function in the manifest must satisfy:

```python
def score_fn(pixel_tensor: torch.Tensor) -> torch.Tensor:
    """Score a single decoded image.

    Args:
        pixel_tensor: (3, H, W) float32 tensor in [0, 1] range.
            This is the VAE-decoded output, NOT a latent.

    Returns:
        Scalar tensor (0-dimensional) with the score.
        Higher = more of whatever quality this function measures.
    """
```

### Adding a third head

To add a new reward head (e.g., a sharpness detector):

1. Implement `sharpness_score_gpu()` in `src_ii/reward_functions.py`
   satisfying the interface contract above.
2. Add `"sharpness": sharpness_score_gpu` to the manifest dict.
3. Add a third output to the linear head (N_heads=3).
4. Design a validation set for the sharpness ground truth function.

No changes to the training loop, loss computation, or pair sampling.

---

## 4. Validation Protocol

Three measurements at each evaluation checkpoint. All three are
required. Omitting any one makes the other two uninterpretable.

### 4.1 Per-Head Ground Truth Correlation

For each head, score the head's validation set with both the BTRM
model and the ground truth function. Compute Spearman rank correlation
between the two score vectors.

- **PINKIFY**: Validation set is `i2i_off_policies/PINKIFY_cases/`.
  Ground truth function is `pinkify_score_gpu`. Constraint checker
  is `validate_pinkify_ranking()` in `src_ii/pinkify_validation.py`.
  Known ranking: `A < B < C, D ~ E, {A,B,C} < {D,E} < F`.

- **THISNOTTHAT**: Validation set and ranking constraints exist in
  `i2i_off_policies/` (the directory structure and ground truth
  function `thisnotthat_score_gpu` together determine the validation
  protocol — the implementing agent should read both to design the
  constraint checker, following the same pattern as PINKIFY).

**What we want to see:** Spearman rho increasing over training steps
(or at least not decreasing). We do NOT expect total ordering match.
The validation set validates the ground truth function, not the reward
model. Monotonic improvement in correlation is evidence the training
signal is wired correctly.

**What indicates a bug:** Spearman rho at or near zero throughout
training (the model isn't learning the function at all), or rho
decreasing over training (the model is learning something
anti-correlated with the ground truth).

### 4.2 Cross-Head Decorrelation

Score a shared set of diverse images (e.g., a random sample of decoded
training images, or the union of both validation sets) with BOTH heads.
Compute Spearman rho between the pinkify head scores and the
thisnotthat head scores.

**What we want to see:** Low correlation (|rho| < 0.5) that stays low
throughout training. The two heads should learn different things.

**What indicates a bug:** Correlation approaching 1.0 means both heads
are learning the same function. This is exactly the failure mode of
sigma-based preference labels: both heads receive identical gradients,
so they converge to the same discriminator. Cross-head correlation
~1.0 is the smoking gun for miswired preference labels.

**What indicates success:** Low or zero cross-head correlation
combined with each head showing increasing correlation with its own
ground truth function. This is the strong evidence for the manifold
hypothesis: the residual stream supports two independent readouts.

### 4.3 Per-Head Validation Constraint Pass Rate

The discrete constraint checks (A < B, B < C, etc.) provide a
human-readable summary of which ordering relationships the model has
learned. These are coarser than Spearman rho but more interpretable
for diagnosing specific failures.

Track the number of constraints passed per head at each checkpoint.
Plot over training steps. Monotonic increase (or stable plateau at
the maximum) is good. Non-monotonic behavior (constraints passing
then failing) suggests training instability or insufficient
regularization.

---

## 5. What "Success" and "Failure" Look Like

### Success

Both heads show increasing Spearman correlation with their respective
ground truth functions over training. Cross-head correlation stays low.
Per-head constraint pass rates increase monotonically or plateau.

This means: the pretrained model's residual stream encodes features
that the low-rank adapter can extract to predict two independent
qualitative objective functions. The manifold hypothesis holds for
these objectives.

### Failure Mode 1: Identical Head Gradients

Both heads learn the same function. Cross-head correlation ~1.0.
Neither head correlates with its ground truth.

**Diagnosis:** The preference labels are not per-head. Most likely
the reward functions are not being called and some proxy (sigma,
backend, step count) is being used for all heads identically.

**This is what the run on 2026-02-20 produced.** It was caused by
sigma-based preference labels instead of reward-function-based labels.

### Failure Mode 2: One Head Learns, the Other Doesn't

Pinkify correlation increases but TNT stays flat (or vice versa).
Cross-head correlation is moderate.

**Diagnosis:** One reward function is providing a learnable signal
and the other isn't. Check: (a) is the non-learning head's reward
function actually being called? (b) does the reward function produce
sufficient variance across the training distribution? (c) is the
signal present in the latent representation at all?

### Failure Mode 3: Both Heads Learn but Become Correlated

Both heads show increasing ground-truth correlation, but cross-head
correlation also increases toward 1.0.

**Diagnosis:** The adapter found ONE feature in the residual stream
and both heads are reading it. The feature happens to correlate with
both ground truth functions on the training distribution. This is the
hardest failure to diagnose — it could mean the ground truth functions
aren't sufficiently decorrelated on the training data, or the adapter
rank is too low to support two independent readouts.

### Failure Mode 4: Perfect Ordering Match in Few Steps

The BTRM model achieves total ordering match on a validation set
designed to validate the ground truth function (not the model) within
a handful of training steps.

**Diagnosis:** Something has gone wrong. Possible causes: label
leakage (validation images somehow in the training set), the model
memorizing rather than learning, or the validation set being too
easy (all constraints satisfied by any monotonic function).

---

## 6. Implementation Notes

### VAE decode batching

For a macrobatch of K pairs (2K images), VAE decode can be batched.
At ~200ms per image on RTX 4090, a macrobatch of 8 pairs (16 images)
costs ~3.2s for decode. Training steps are currently ~11s, so this
adds ~30% overhead. This is acceptable — the alternative (not having
correct preference labels) makes training worthless.

### Reward function caching

For sigma=0 positions that appear multiple times across training
(same trajectory, same step), the decoded pixel image and its reward
scores are deterministic. A cache keyed on (trajectory_id, step_key)
could eliminate redundant VAE decodes and reward computations. This
is an optimization, not a requirement — correctness first.

### Gradient isolation

The VAE decode and reward function evaluation MUST be inside
`torch.no_grad()`. They produce labels, not training signal. If
gradients accidentally flow through the VAE decode into the reward
functions, the model will learn to game the scoring functions rather
than learn the underlying feature detectors. This would be a
catastrophic but silent bug.

### The manifest lives in the training script, not the training loop

The training loop (`btrm_training.py`) accepts a `reward_manifest`
parameter. It does not import reward functions directly. The training
script (`scripts_ii/run_*.py`) constructs the manifest by importing
the functions and passing them in. This keeps the training loop
generic and testable.

---

## 7. Relationship to Prior Documents

- **`docs/claude_BTRM_training_policy.md`**: Covers BT loss mechanics,
  normalization, regularization policy (no logsquare). This directive
  covers the preference label source, which that document does not
  address.

- **`docs/user_dataflow_and_lifecycle_rollup.md`**: The 10 outer
  specifications. This directive adds an 11th: "preference labels
  derive from ground truth scoring functions, never from metadata
  proxies."

- **`docs/essay_pinkify_validated_training_run.md`**: Documents the
  2026-02-20 run that diagnosed the sigma-based preference bug.
  This directive ensures the bug cannot recur.

- **`src_ii/pinkify_validation.py`**: The PINKIFY validation
  implementation. The TNT validation should follow the same pattern
  (same module or a parallel one).

---

## 8. Checklist for Implementing Agents

Before modifying any training code, verify:

- [ ] The reward manifest maps each head name to an imported scoring
      function from `src_ii/reward_functions.py`.
- [ ] The training loop calls `manifest[head](decoded_pixels)` for
      each head and each image in the pair, inside `torch.no_grad()`.
- [ ] Preference labels are per-head (each head can prefer a
      different image in the same pair).
- [ ] The VAE is loaded before training and used for decode only.
- [ ] Validation at each checkpoint includes: per-head ground truth
      correlation, cross-head decorrelation, and per-head constraint
      pass rates.
- [ ] No preference label is derived from sigma, backend, step count,
      or any other metadata field. The reward functions are the sole
      source.
- [ ] The training curve JSONL includes per-head accuracy computed
      against reward-function-derived labels (not sigma-derived).
- [ ] Cross-head Spearman rho is logged at each evaluation step.

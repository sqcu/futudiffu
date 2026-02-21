# Synthesis: Session 2026-02-20

**Author:** Root session
**Scope:** Streaming persistence, reward function integration, compilation
regression, loss normalization, manifold hypothesis probe

---

## 1. What Happened

This session began with a newly grown multi-resolution dataset (420
trajectories, 84 unique resolutions, 12 prompts) and a set of
recently-built modules (logSNR-uniform step capture, clean-biased
sampling, PINKIFY validation, dataset catalog). The user identified
two architectural problems and one research execution problem:

1. **Dataset generation was not crash-safe.** 38.7 minutes of GPU
   compute buffered in RAM before any disk write. Incompatible with
   preemptible spot instances, which are the physics-optimal deployment
   for FP8 TFLOPS-bound workloads.

2. **Training metrics were ephemeral.** The training curve, validation
   metrics, and run summary were written only at end-of-run. A crash
   at step 99 of 100 lost all per-step data.

3. **The BTRM training pipeline used sigma-based preference labels**
   instead of the actual ground truth reward functions. Both heads
   received identical gradients, making the two-head design
   meaningless.

Each of these was addressed, and each fix exposed a further defect,
producing a cascade of corrections that ultimately reached the
research question the whole system was built to answer.

---

## 2. Streaming Persistence (Crash Safety)

### Dataset Generation

`scripts_ii/generate_multi_res_trajectories.py` was restructured from
"compute all → write all" to "compute one → write one → next." A new
module `src_ii/dataset_resumption.py` provides deterministic plan
generation, parquet-index-based completion detection, and incremental
sidecar updates. Re-running a generation script with the same
parameters skips already-generated trajectories. Maximum unwritten
compute: ~80 seconds (10 trajectories × 8s/trajectory between
flushes).

### Training Metrics

`src_ii/incremental_save.py` provides `TrainingCurveWriter` (JSONL
append per step, ~10us overhead) and `PeriodicSaver` (interval-gated
auto-save for ValidationMetrics). The training loop now accepts these
as optional parameters. `atomic_json_save()` handles write-to-temp-
then-rename for all JSON persistence.

**37 new tests (21 resumption + 16 incremental save), all passing.**

The user's architectural position is clear: all generation and
training code must be interruptible and resumable at ~5 minute
granularity. This is not a preference but a consequence of the
compute economics — FP8 TFLOPS-bound workloads run optimally on
cheap interruptible spot instances.

---

## 3. Reward Function Integration

### The Directive

The session produced `docs/directive_btrm_reward_function_integration.md`,
which captures the full research intent and implementation contract
for BTRM training. Key points:

- The LoRA adapter is a **probe into the pretrained model's residual
  stream**, not a function approximator. The research question is
  whether pretrained activations already contain the feature detectors
  for arbitrary qualitative objective functions.

- The two-head design (PINKIFY + TNT) is the strong form of the
  manifold hypothesis test: two decorrelated objectives projected
  simultaneously from the same residual stream.

- Preference labels derive from **ground truth scoring functions**
  (`pinkify_score_gpu`, `thisnotthat_score_gpu`), never from sigma,
  backend, step count, or any metadata proxy. Each head gets
  independent labels.

- Three validation measurements at each checkpoint: per-head ground
  truth correlation, cross-head decorrelation, per-head constraint
  pass rate.

- The PINKIFY validation set validates the ground truth function, not
  the reward model. The reward model is not expected to match the
  total ordering of the validation set. If it did, something is wrong.

### The Wiring Fix

`btrm_training.py` now accepts a `reward_manifest: dict[str, Callable]`
parameter. When provided, both latents in each pair are VAE-decoded to
pixel tensors (inside `torch.inference_mode()`), scored by each head's
ground truth function, and per-head preference labels are derived from
relative scores. The VAE (~160MB) is loaded before training and used
only for decode.

A TNT validation module (`src_ii/tnt_validation.py`) and cross-head
decorrelation measurement (`src_ii/cross_head_decorrelation.py`) were
built alongside the integration.

### The Baseline (Sigma-Based, Run 1)

150 steps, 28 minutes. Both heads achieved ~80% accuracy but with
identical rankings. Cross-head correlation was not measured (both heads
received identical labels). PINKIFY validation: 3/5 → 2/5 constraints
over training. This confirmed: sigma-based preferences produce a noise
discriminator, not a reward model.

### The Reward-Function Run (Run 2)

150 steps, 55.5 minutes (2x slower due to compilation regression —
see section 4). Key results:

- **Pinkify accuracy: 83.1%, TNT accuracy: 58.2%.** The 25-point gap
  between heads proves independent per-head preference labels are
  working. This was impossible under sigma-based preferences.

- **Cross-head rho: 0.52 → 0.42 → 0.85.** Initial decorrelation at
  step 10, then monotonic convergence to high correlation. This is
  Failure Mode 3 from the directive: both heads learn but become
  correlated. The shared rank-8 adapter found one dominant feature
  and both heads aligned to read it.

- **Ground truth cross-head rho: -0.186.** The ground truth functions
  are decorrelated by construction, but the model converges to 0.85.

This is the first run where the measurement infrastructure actually
works end-to-end: independent preference labels, per-head validation,
cross-head decorrelation tracking, incremental persistence. The result
(failure mode 3) is interpretable and actionable.

---

## 4. The Compilation Regression

### Discovery

The reward-function run was 2x slower than the sigma-based run (21.5s
vs 10.9s/step) and used ~2GB more peak VRAM. A memory audit found
`compile_model=False` in the new training script.

### Root Cause Chain

1. `load_fp8_diffusion_model()` offered only whole-model compile or
   nothing. Whole-model compile is incompatible with per-block
   gradient checkpointing.

2. The frozen `src/futudiffu/model_manager.py` had a
   `compile_layers_for_training()` method that compiled each layer
   independently — compatible with gradient checkpointing. This
   pattern was never replicated in `src_ii/`.

3. A misleading comment in the earliest training script ("No
   torch.compile for training") propagated by copy-paste to every
   subsequent script.

4. The directive document didn't mention compilation.

5. Result: **every BTRM training run in the project's history ran
   all 30 transformer layers in eager mode.**

### Structural Fix

Per-layer compilation is now built into `BTRMCompoundModel.__init__`
with `compile_layers=True` as the default. A runtime `UserWarning`
fires if training starts without compilation. All 18
`BTRMCompoundModel` call sites across `scripts_ii/` verified.

---

## 5. The Loss Normalization Non-Bug

The reward-function run's reported loss (23.13 at step 0) appeared
3.3x higher than the sigma-based run (7.08). Investigation revealed:

- The reported metric was `bt_loss` (raw sum of all BT terms), not
  `loss` (per-term normalized).

- The reward-function run has 0% ties → 34 active terms per
  macrobatch. The sigma-based run has 65% ties → ~10 active terms.

- Per-term normalized loss: 0.680 (reward-function) vs 0.708
  (sigma-based). Both are ~ln(2) = 0.693, exactly correct for
  random initialization.

The gradient computation was correct all along. The reporting metric
was the raw sum, making cross-run comparison misleading. A fix to
migrate all reporting to per-term normalized loss is in progress.

The bonus finding: the sigma-based run was discarding **65% of its
training pairs** as ties (both images at sigma=0, identical sigma →
pref=0 on both heads). The reward-function path uses 100% of pairs,
making it ~3x more sample-efficient per macrobatch.

---

## 6. Manifold Hypothesis Probe: Current State

The research question: does the pretrained generative model's residual
stream support two independent linear readouts for two decorrelated
qualitative objectives?

**Evidence for:** The two heads achieve different training accuracies
(83% vs 58%), proving independent preference labels reach the model.
Cross-head rho briefly dips at step 10, suggesting initial
decorrelation potential exists.

**Evidence against:** Cross-head rho converges to 0.85. The rank-8
shared adapter collapses to one dominant feature direction that both
heads read.

**What this does NOT tell us:** Whether the residual stream lacks the
representations, or whether the shared adapter architecture prevents
them from being expressed. The shared adapter is a known bottleneck —
both heads compete for the same rank-8 intervention. Separate adapters
per head, or a higher-rank shared adapter, would disambiguate.

**What the next run should test:** The cheapest intervention is a
decorrelation regularizer on the two head weight vectors (penalize
their cosine similarity). If that breaks the correlation without
killing per-head accuracy, the representations were always there and
the architecture just needed a nudge. If it kills accuracy, the
rank-8 adapter genuinely can't support two orthogonal feature
directions and higher rank is needed.

---

## 7. Modules Created This Session

| Module | Purpose |
|--------|---------|
| `src_ii/dataset_resumption.py` | Trajectory identity, plan persistence, resume detection |
| `src_ii/incremental_save.py` | TrainingCurveWriter (JSONL), PeriodicSaver, atomic_json_save |
| `src_ii/tnt_validation.py` | TNT holdout validation (parallel to pinkify_validation.py) |
| `src_ii/cross_head_decorrelation.py` | Spearman rho between heads on shared image set |

## 8. Directives Written This Session

| Document | Purpose |
|----------|---------|
| `docs/directive_btrm_reward_function_integration.md` | Training contract: reward manifest, validation protocol, failure mode taxonomy |

## 9. Defects Found and Fixed This Session

| Defect | Proximate Cause | Root Cause | Fix |
|--------|----------------|------------|-----|
| 38 min unwritten compute | Phase 2/3 separation in gen script | "Compute-first, write-last" architecture | Streaming write-through + resume detection |
| Ephemeral training metrics | End-of-run write only | No incremental persistence module | JSONL append per step + PeriodicSaver |
| Identical head gradients | Sigma-based preference function | No reward manifest pattern in training loop | `reward_manifest` parameter + directive |
| 2x training slowdown + 2GB VRAM | `compile_model=False` in new script | Per-layer compilation never replicated from frozen src/ | `compile_layers=True` default in BTRMCompoundModel |
| Incomparable loss across runs | Summary reports `bt_loss` (raw sum) | No convention for which metric is primary | Migration to per-term normalized `loss` (in progress) |

## 10. What Remains

1. **Compilation-enabled reward-function training run.** The prior run
   was uncompiled. Re-running with `compile_layers=True` should halve
   wall time and reduce VRAM by ~2GB.

2. **Decorrelation regularizer.** The cheapest test of whether the
   residual stream supports two independent readouts.

3. **`inference_mode` migration.** Multiple paths use `no_grad` where
   `inference_mode` is correct (VAE, reward functions). Not a
   correctness issue but an efficiency one.

4. **Step-count variation for thisnotthat.** All trajectories use 30
   steps. If TNT's ground truth function is sensitive to step count
   (it may not be — TNT measures pixel-space similarity to reference
   images), varied step counts would increase training signal diversity.

5. **Extended training (500+ steps).** 150 steps with 420 trajectories
   covers only 0.03% of the 5.6M pair space. The model may need much
   longer exposure to express decorrelated features.

# Technical Debt Survey: src_ii/ and scripts_ii/

Survey date: 2026-02-25. Research only, no code modified.

---

## Category 1: Inlined Algorithms

### Finding 1.1 — `train_btrm_differentiable` God Function

**File:** `/mnt/f/dox/repos/ai/futudiffu/src_ii/btrm_training.py`
**Line range:** 396–1614
**Size:** 1,219 lines including docstring, 980 lines of code body

`train_btrm_differentiable` is the largest function in the codebase. It contains
three structurally near-identical execution paths in a deeply nested if/else tree,
making the function impossible to unit-test and difficult to extend without
introducing path-specific bugs.

#### Three inlined code paths

The function's step loop contains a top-level branch on `_use_flops_budget`
(line 833), and within the `else` arm a nested branch on `packed` (line 1137).
The result is three distinct execution paths:

**Path A: FLOPS-budget macrobatch** (lines 843–1117)
- Sample macro_pair_specs via `pair_sampler.sample_macrobatch()`
- Load all latents, build image list
- Bin-pack all images into bins
- Per-bin forward with partial backward (retain_graph logic)
- Pre-counted `_precount_active_heads` normalization

**Path B: Legacy fixed-count packed** (lines 1136–1323, packed=True branch)
- Sample K pairs per micro-batch
- Load all 2K latents
- Bin-pack into bins
- Score each bin, reassemble, compute BT loss once
- Post-hoc `active_heads` normalization (different from Path A)

**Path C: Serial** (lines 1325–1451, packed=False branch)
- Sample one pair per micro-batch
- `score_serial()` on each image separately
- BT loss on a single pair

#### Triplicated BT loss + metrics block

The BT loss computation (pref routing, `bt = -F.logsigmoid(pos_s - neg_s)`,
accuracy tracking, `val_tracker.update(PairResult(...))`) appears three times:

- Lines 1025–1065 (Path A, per-bin inner loop)
- Lines 1259–1299 (Path B, packed post-hoc loop)
- Lines 1369–1407 (Path C, serial inner loop)

Each instance is 35–40 lines of near-identical code differing only in:
- Variable names (`pd` vs `pair`)
- Index arithmetic (`idx_a/idx_b` vs `2*k / 2*k+1` vs scalar)
- `source_a` / `source_b` string literals
- Whether `width_a`/`height_a` are pulled from `image_resolutions` or omitted

These blocks should be a single helper function, e.g.:

```python
def _score_pair_bt_loss(
    pos_s, neg_s, name, pref_key, pair, image_resolutions, idx_a, idx_b,
    source_tag, accum_accs, accum_acc_counts, val_tracker
) -> tuple[Tensor, int]:
    ...
```

#### Extractable sub-algorithms

The following logic blocks inside the function body are self-contained algorithms
mixed with the outer loop:

1. **Reward-manifest preference closure builder** (lines 651–712): Constructs
   `_reward_manifest_preference_fn` inline. This exact pattern was extracted to
   `src_ii/dataset_io.py::make_reward_manifest_preference_fn()`, but the copy in
   `btrm_training.py` was not removed. Two implementations now exist.

2. **LR scheduler construction** (lines 765–787): `warmup_cosine` vs `warmup_only`
   branching. Already simple, but repeated in sweep scripts.

3. **Macrobatch collection + bin-packing setup** (lines 858–915 in Path A,
   lines 1143–1204 in Path B): Load latents, compute resolutions, call
   `BinPackScheduler().pack()`. The two paths duplicate the
   `pack_items.append({"img_idx", "seq_len", "width", "height"})` loop.

4. **Funfetti metadata aggregation** (lines 1493–1518): Post-hoc assembly of
   per-step `funfetti` dict from `step_microbatch_meta`. Extractable as a pure
   function.

#### Docstring length

The function docstring alone runs from line 441 to 633 (193 lines), describing
42 parameters. This indicates the function's interface has grown beyond what a
single abstraction can own cleanly.

---

### Finding 1.2 — `sweep_rtheta_lr.py` Inlines Full Training Setup

**File:** `/mnt/f/dox/repos/ai/futudiffu/scripts_ii/sweep_rtheta_lr.py`
**Size:** 904 lines
**Inlined phases:** encode_all_prompts (lines 160–190), build_latent_loader
(lines 193–235), build_sigma_lookup (lines 238–256), vram_report (lines 46–62)

`sweep_rtheta_lr.py` predates `src_ii/training_setup.py` and `src_ii/dataset_io.py`
and replicates their logic directly inside the script. It has its own
`encode_all_prompts()` function (line 160) that does the same
load-TE → encode → del-TE → empty_cache lifecycle as
`training_setup.encode_training_prompts()`. The `build_latent_loader()` function
(line 193) constructs a `load_latent` closure that mirrors `dataset_io.make_load_latent_fn()`,
but reads from the older v1 dataset format (flat `.pt` files at
`btrm_dataset/latents/traj_XXXXXX/`) rather than the V2 DatasetReader API.

It also contains its own local `vram_report()` helper (line 46) not shared with
any other module.

---

### Finding 1.3 — `generate_multi_res_trajectories.py` Inlines Resolution Plan + Sigma Logic

**File:** `/mnt/f/dox/repos/ai/futudiffu/scripts_ii/generate_multi_res_trajectories.py`
**Size:** 1,266 lines

`_build_resolution_plan()` (line 83) produces the 6-tier resolution list
(256x256, 320x320, 384x384, 512x512, 704x704, 1024x1024) with per-tier step
counts and backend assignments. This logic is specific to this script but the
6-tier anchor list appears in other places (flops_sampling.py, pair_sampler.py).

`_sigma_to_logsnr()` (line 138) and `_get_sparse_steps()` (line 144) are
local reimplementations of functions already available in
`src_ii/sigma_schedule.py`. The logSNR conversion in particular appears in
`pair_sampler.py::logsnr_sampling_weight()` as well.

`phase1_encode_prompts()` (lines 180–215) inlines the TE lifecycle:
load → encode → del → empty_cache. Should call
`training_setup.encode_training_prompts()`.

---

### Finding 1.4 — `run_reward_validated_training.py` Has an Inlined `encode_pinkify_challenge_latents` Block

**File:** `/mnt/f/dox/repos/ai/futudiffu/scripts_ii/run_reward_validated_training.py`
**Size:** 907 lines
**Lines:** 71–160

`encode_pinkify_challenge_latents()` (lines 71–117) and
`encode_tnt_challenge_latents()` (lines 119–158) load all challenge images from
disk, VAE-encode them, and return a label-to-latent dict. This is called once
per training run to precompute the holdout evaluation cache.

`run_pinkify_validated_training.py` has a counterpart `encode_pinkify_challenge_latents()`
starting at line 135 that performs the same operation. Neither calls a shared
helper. The VAE-encode-from-dir pattern also appears in
`pinkify_validation.py::_load_pinkify_challenge_images()` (which decodes to pixels
rather than latents, but the filesystem scan logic is the same).

---

## Category 2: Duplication across modules and scripts

### Finding 2.1 — `load_latent_fn` Closures: 5 Inline Definitions vs 1 Factory

**Canonical implementation:** `src_ii/dataset_io.py::make_load_latent_fn()` (lines 21–91)

**Inline duplicates (not migrated):**

| File | Line | Notes |
|------|------|-------|
| `scripts_ii/compare_pinkify_scores.py` | 248 | Different signature: `(traj_id, step_key)` not a tuple key |
| `scripts_ii/diagnose_nan_step0.py` | 109 | Uses V2 reader; near-identical to canonical |
| `scripts_ii/run03_btrm_training.py` | 417 | Uses V2 reader; near-identical to canonical |
| `scripts_ii/run_reward_validated_training.py` | 542 | Uses V2 reader; near-identical to canonical |
| `scripts_ii/test_funfetti_e2e.py` | 250 | Has an `if USE_MULTI_RES` branch; the True-branch is near-identical to canonical, but contains the old `sigmas[-2]` bug for the final step (should be `0.0`) |

**Scripts already migrated to `make_load_latent_fn`:**
`diagnose_nan_minimal.py`, `diagnose_nan_training.py`,
`run_flops_budget_100step_v2.py`, `run_pinkify_validated_training.py`,
`validate_refactor_2step.py`

The canonical implementation in `dataset_io.py` contains the correct final-step
sigma fix (`sigma_val = 0.0` when `step_key == "final"`). The non-migrated copy
in `test_funfetti_e2e.py` (line 276) still uses `sigmas[-2]` for the
`USE_MULTI_RES` path, which is the documented `essay_logsnr_step_weighting_fix.md`
bug.

---

### Finding 2.2 — Model Path Constants: 16 Scripts Hardcode vs 6 Import from `model_paths`

**Canonical module:** `src_ii/model_paths.py` (lines 17–22)

```python
FP8_PATH = str(COMFYUI_ROOT / "models" / "diffusion_models" / "z_image_fp8_blockwise.safetensors")
TE_PATH  = str(COMFYUI_ROOT / "models" / "text_encoders" / "qwen_3_4b.safetensors")
VAE_PATH = str(COMFYUI_ROOT / "models" / "vae" / "zimage.safetensors")
TOKENIZER_PATH = str(REPO_ROOT / "src" / "futudiffu" / "tokenizer")
```

**Scripts that import from `model_paths` (correct):**
`diagnose_nan_minimal.py`, `diagnose_nan_training.py`, `run_ddgrpo_v2.py`,
`run_flops_budget_100step_v2.py`, `run_pinkify_validated_training.py`,
`validate_refactor_2step.py`

**Scripts that hardcode path constants (non-conforming):**

| File | Lines | Constants redefined |
|------|-------|-------------------|
| `compare_pinkify_scores.py` | 38–40 | FP8_PATH, TOKENIZER_PATH, TE_PATH |
| `constructive_nan_debug.py` | 113–114, 193–194, 315–316 | FP8_PATH, TE_PATH, TOKENIZER_PATH (3 separate inline definitions in 3 functions) |
| `demonstrate_rtheta_policy.py` | 49–51 | FP8_PATH, TE_PATH, VAE_PATH |
| `diagnose_block_alignment.py` | 24–25 | FP8_PATH, TE_PATH |
| `diagnose_nan_step0.py` | 28–30 | FP8_PATH, TE_PATH, TOKENIZER_PATH |
| `generate_multi_res_trajectories.py` | 62, 64–65 | FP8_WEIGHTS (alternate name), VAE_WEIGHTS, TOKENIZER_PATH |
| `generate_preference_labels.py` | 38 | VAE_PATH |
| `render_attention_maps.py` | 42 | VAE_PATH |
| `render_attention_maps_v2.py` | 42 | VAE_PATH |
| `render_comparison.py` | 22 | VAE_PATH |
| `run03_btrm_training.py` | 65–66 | FP8_WEIGHTS (alternate name), VAE_WEIGHTS |
| `run_reward_validated_training.py` | 42–45 | FP8_PATH, TE_PATH, VAE_PATH, TOKENIZER_PATH |
| `sweep_ktuple_gain.py` | 36–38 | FP8_PATH, TE_PATH, VAE_PATH |
| `sweep_rtheta_hparams.py` | 42–44 | FP8_PATH, TE_PATH, TOKENIZER_PATH |
| `sweep_rtheta_lr.py` | 65–67 | FP8_PATH, TE_PATH, TOKENIZER_PATH |
| `test_funfetti_e2e.py` | 44–46 | FP8_PATH, TE_PATH, TOKENIZER_PATH |

Additionally, `constructive_nan_debug.py` redefines its path constants three
separate times inside three different function bodies (lines 113–114, 193–194,
315–316) rather than once at module scope, creating the worst case in the
codebase: duplicated hardcoded strings within a single file.

---

### Finding 2.3 — Text Encoder Lifecycle: 8 Inline Copies vs 1 Canonical Function

**Canonical function:** `src_ii/training_setup.py::encode_training_prompts()` (lines 32–68)

Pattern: `create_tokenizer → load_text_encoder → loop encode_prompt → del te_model, tokenizer → torch.cuda.empty_cache()`

**Inline duplicates:**

| File | Lines | Notes |
|------|-------|-------|
| `compare_pinkify_scores.py` | 200–213 | Full lifecycle inlined |
| `constructive_nan_debug.py` | 214–215, 322–323 | Inlined twice in two separate functions |
| `diagnose_nan_step0.py` | 71–72, cleanup at 88 | TE and model loads interleaved |
| `generate_multi_res_trajectories.py` | 190–215 (phase1_encode_prompts) | Extracted to a named function inside the script but not into training_setup |
| `run_reward_validated_training.py` | 309–325 | Full lifecycle inlined in main() |
| `sweep_rtheta_hparams.py` | 142–155 | Full lifecycle inlined |
| `sweep_rtheta_lr.py` | 160–190 (encode_all_prompts) | Extracted to a named function inside the script; reads from v1 manifest format |
| `test_funfetti_e2e.py` | 205–224 | Full lifecycle inlined |

**Scripts already using `encode_training_prompts` (correct):**
`diagnose_nan_minimal.py`, `diagnose_nan_training.py`,
`run_flops_budget_100step_v2.py`, `run_pinkify_validated_training.py`,
`validate_refactor_2step.py`

The lifecycle is safety-critical: loading TE and backbone simultaneously uses
~16 GB VRAM and will OOM on a 24 GB card. Every inline copy is a potential
regression point where a future edit could reorder the loads.

---

### Finding 2.4 — Model Loading + Adapter Setup: 5 Inline Sequences vs 1 Canonical Function

**Canonical function:** `src_ii/training_setup.py::load_training_backbone()` (lines 124–200)

Pattern: `load_zimage_rlaif → setup_btrm_training → (optionally) load_btrm checkpoint`

**Inline duplicates (not migrated):**

| File | Lines | Differences from canonical |
|------|-------|---------------------------|
| `run03_btrm_training.py` | 376–395 | Calls `setup_btrm_training(raw_model)` with no kwargs; manually prints param counts; does not use training_setup |
| `run_reward_validated_training.py` | 442–455 | Calls `load_zimage_rlaif + setup_btrm_training` inline; manually prints VRAM |
| `sweep_rtheta_hparams.py` | 336–340, 574–621 | Two separate model load sites; second site (line 574) uses `install_multi_lora` directly instead of `setup_btrm_training` |
| `sweep_rtheta_lr.py` | 653–670, 795–820 | Two separate model load sites; first (line 653) uses `allocate_adapter` from `futudiffu.lora` (legacy API); second (line 795) uses `install_multi_lora` |
| `test_funfetti_e2e.py` | 236–240, 353, 415 | Model loaded once (line 236); `setup_btrm_training` called twice (lines 353 and 415) on the same model |

`sweep_rtheta_lr.py` is the worst case: it calls `allocate_adapter` from the
frozen `futudiffu.lora` (src/ API) in one code path (line 662–663), violating the
src/ freeze policy. The second code path at line 795 uses the correct
`src_ii/multi_lora.py::install_multi_lora`.

**Scripts already using `load_training_backbone` (correct):**
`diagnose_nan_minimal.py`, `diagnose_nan_training.py`,
`run_flops_budget_100step_v2.py`, `run_pinkify_validated_training.py`,
`validate_refactor_2step.py`

---

### Finding 2.5 — Pair Sampler Construction: 5 Inline Sequences vs 1 Canonical Function

**Canonical function:** `src_ii/training_setup.py::build_dataset_positions()` (lines 71–121)

Pattern: `DatasetReader → build_positions_from_v2 → compute_flops_sampling_weights_from_positions → BTRMPairSampler(...)`

**Inline duplicates (not migrated):**

| File | Lines | Differences |
|------|-------|-------------|
| `run03_btrm_training.py` | 285–333 | Multi-blob merge loop with `global_id_offset`; does not use flops weights; `allow_intra_trajectory=False` |
| `run_reward_validated_training.py` | 270–303 | Standard single-reader path; uses flops weights |
| `sweep_rtheta_hparams.py` | (no pair sampler; uses materialized pair table) | N/A |
| `test_funfetti_e2e.py` | 156–198 | Has `USE_MULTI_RES` branch; when True, near-identical to canonical |
| `run_flops_budget_100step_v2.py` | 109–175 | Extended version with `summarize_flops_weights` diagnostics and macrobatch smoke test |

`run03_btrm_training.py` has the most divergent construction: it merges multiple
dataset blobs by remapping `traj_id` with a `global_id_offset` (line 322) to
avoid collisions. This multi-blob merging is not covered by
`build_dataset_positions()`.

**Scripts already using `build_dataset_positions` (correct):**
`run_pinkify_validated_training.py`, `diagnose_nan_minimal.py`,
`diagnose_nan_training.py`, `validate_refactor_2step.py`

---

### Finding 2.6 — Reward Manifest Preference Function: Duplicate Implementation

**Two locations with near-identical code:**

1. `src_ii/dataset_io.py::make_reward_manifest_preference_fn()` (lines 94–145)
2. `src_ii/btrm_training.py::_reward_manifest_preference_fn` closure (lines 669–712,
   constructed inline inside `train_btrm_differentiable`)

Both functions:
- Accept `pair` dict with `traj_a/step_a/traj_b/step_b` keys
- Load latents for both images via `load_latent_fn`
- VAE-decode to pixel tensors
- Score each pixel tensor with each head's reward function
- Return `{pref_key: +1/-1/0}` dict

The `btrm_training.py` copy was the original. `dataset_io.py` was created to
extract this pattern for reuse but the original was not removed, leaving two
implementations that can diverge silently.

---

### Finding 2.7 — `_get_v2_meta` Cache Closure: 3 Inline Duplicates

Pattern: per-trajectory metadata cache to avoid re-reading parquet on each
latent load call.

```python
_v2_meta_cache = {}

def _get_v2_meta(traj_id):
    if traj_id not in _v2_meta_cache:
        meta, accessor = reader[traj_id]
        _v2_meta_cache[traj_id] = (meta, accessor)
    return _v2_meta_cache[traj_id]
```

Appears in:
- `run_reward_validated_training.py` lines 534–540
- `test_funfetti_e2e.py` lines 242–248

The canonical `dataset_io.py::make_load_latent_fn()` internalizes this cache
as `_meta_cache` (lines 46–51) — it is not a separate exported function. Scripts
that inline `load_latent_fn` also inline the cache. Migrating to
`make_load_latent_fn` eliminates the cache duplication automatically.

---

## Summary Table

| Finding | Category | Severity | Files affected |
|---------|----------|----------|----------------|
| 1.1 `train_btrm_differentiable` God function | Inlined algorithm | High | `btrm_training.py` |
| 1.2 `sweep_rtheta_lr` training setup inlining | Inlined algorithm | Medium | `sweep_rtheta_lr.py` |
| 1.3 `generate_multi_res_trajectories` sigma + phase inlining | Inlined algorithm | Medium | `generate_multi_res_trajectories.py` |
| 1.4 `encode_*_challenge_latents` duplication | Inlined algorithm | Low | `run_reward_validated_training.py`, `run_pinkify_validated_training.py` |
| 2.1 `load_latent_fn` closures | Duplication | High | 5 scripts |
| 2.2 Model path constants | Duplication | Medium | 16 scripts |
| 2.3 Text encoder lifecycle | Duplication | High | 8 scripts |
| 2.4 Model loading + adapter setup | Duplication | Medium | 5 scripts |
| 2.5 Pair sampler construction | Duplication | Medium | 5 scripts |
| 2.6 Reward manifest preference fn | Duplication | Medium | `dataset_io.py`, `btrm_training.py` |
| 2.7 `_get_v2_meta` cache closure | Duplication | Low | 2 scripts |

## Extraction Status

The following canonical modules already exist for these patterns. The debt is
in scripts that have not yet migrated to them:

- `src_ii/model_paths.py` — model path constants
- `src_ii/training_setup.py` — `encode_training_prompts`, `build_dataset_positions`, `load_training_backbone`
- `src_ii/dataset_io.py` — `make_load_latent_fn`, `make_reward_manifest_preference_fn`

The internal triplication of the BT loss loop inside `train_btrm_differentiable`
has no canonical extraction yet and is the highest-priority internal refactor
target. The three paths (FLOPS-budget, packed legacy, serial) share 35–40 lines
of identical per-pair BT loss + accuracy + `val_tracker.update` code that should
be a helper function.

---

## Category 3: Neglected Obvious Abstractions

Survey date: 2026-02-25.

---

### Finding 3.1 — "Score N cached latents" Loop: 3 Inline Definitions

**Pattern:** iterate over a dict/list of labelled latents, call `score_serial` on
each, collect `{label: float}` score dictionaries, then run ranking or correlation
checks. This is a repeated loop-body that appears in three different locations.

**Occurrences:**

| File | Lines | Context |
|------|-------|---------|
| `src_ii/pinkify_validation.py` | 280–304 | PINKIFY holdout: loops over `images.items()`, VAE-encodes each PIL, calls `score_serial` at `sigma=0`, collects `{A,B,C,D,E,F}: score` dict. Function: `validate_btrm_pinkify_ranking()`. |
| `src_ii/cross_head_decorrelation.py` | 120–138 | Cross-head Spearman: loops over `latent_cache.keys()`, calls `score_serial` on pre-cached latents, collects per-head score dicts. Function: `measure_cross_head_decorrelation()`. |
| `src_ii/exemplar_renderer.py` | 314–338 | Exemplar selection: loops over `sample_keys`, calls `score_serial` on each, collects `{key: {head: score}}` for top/bottom-K selection. Function: `score_and_render_exemplars()`. |

Each occurrence:
1. Calls `model.eval()` and wraps in `torch.no_grad()`
2. Calls `score_serial(model, latent, timestep, conditioning, num_tokens, gradient_checkpointing=False)`
3. Extracts `float(score_tensor[0, head_idx].item())` per head
4. Returns a dict keyed by image label

The three differ only in: (a) whether latents are pre-cached or freshly VAE-encoded,
(b) the key structure (str label vs (traj_id, step_key) tuple), and (c) what
downstream analysis is run on the collected scores.

**Missing abstraction:**

```python
def score_latent_cache(
    model,
    latent_cache: dict[str, tuple[Tensor, Tensor, Tensor, int]],
    head_names: list[str],
    device: torch.device | None = None,
) -> dict[str, dict[str, float]]:
    """Score a dict of pre-loaded latents. Returns {label: {head_name: score}}."""
```

This would be the common kernel. The three callers differ only in the pre-loading
step (VAE encode vs pass-through) and post-scoring step (ranking check vs correlation
vs top-K sort), neither of which belongs in the scoring loop itself.

---

### Finding 3.2 — Checkpoint Persistence: `persist_btrm()` Is Correctly Canonical, But Training Callbacks Are Not

**Canonical function:** `src_ii/btrm_lifecycle.py::persist_btrm()` (lines 273–330)

`persist_btrm()` saves three files atomically: `rtheta_adapter.safetensors`,
`btrm_head.safetensors`, and `btrm_compound_config.json`. All active training
scripts call it. This is a success — the checkpoint persistence itself is not
duplicated.

**What IS duplicated:** the checkpoint scheduling logic and the
per-step JSONL callback that wraps the checkpoint call.

In `scripts_ii/run03_btrm_training.py` (lines 476–490):

```python
def training_callback(step, entry):
    record = {"phase": "btrm", "step": step, **entry, ...}
    metrics_file.write(json.dumps(record, default=str) + "\n")
    metrics_file.flush()
    if (step + 1) % BTRM_CHECKPOINT_INTERVAL == 0:
        ckpt_dir = OUTPUT_DIR / f"btrm_ckpt_{step + 1:04d}"
        persist_btrm(raw_model, "rtheta", str(ckpt_dir))
```

In `scripts_ii/run_reward_validated_training.py` (lines 603–640) and
`scripts_ii/run_pinkify_validated_training.py` (lines 346–380): the scripts
pass `curve_writer`, `checkpoint_fn`, and `checkpoint_steps` directly to
`train_btrm_differentiable()`. These are the newer pattern. The `run03` script
predates `TrainingCurveWriter` integration and has not been migrated.

The `run03` callback writes a raw `open(...).write(json.dumps(...))` JSONL rather
than using `TrainingCurveWriter`. It also wraps `persist_btrm` inside the callback
rather than passing `checkpoint_fn`. This is a single-script regression point, not
a systemic failure (other scripts use the right path), but the script is still
in active use for multi-blob training.

---

### Finding 3.3 — Training Loop Boilerplate: Inline vs Delegated

**Pattern:** `optimizer.zero_grad()` / `loss.backward()` / `clip_grad_norm_()` /
`optimizer.step()` / `scheduler.step()` as an inline 5-step sequence.

**`train_btrm_differentiable` (lines 809–1461):** The canonical training function.
The outer step loop calls `optimizer.zero_grad()` at line 812, accumulates
gradients across bins/microbatches, clips at line 1455, then steps at lines 1461–1462.
This is the correctly decomposed version — the training loop owns the gradient
accumulation structure and the step functions are inlined only within this one
canonical function.

**`scripts_ii/constructive_nan_debug.py` (lines 232–285 and 363–430):**
Two standalone training loops inlining the full 5-step sequence independently.
This is a debug script and these loops are not reusable, so the duplication is
lower priority. However, both copies use `max_norm=0.1` consistent with the
canonical training, suggesting they were hand-synced rather than derived from
a shared callable.

**`scripts_ii/run_ddgrpo_v2.py` (lines 745–750):** The DDGRPO loop calls
`policy_optimizer.zero_grad()` inline before `accumulate_and_step()`, which
handles `clip_grad_norm_` and `optimizer.step()` internally via
`policy_optimizer_step()` in `src_ii/policy_step.py`. This is correctly split:
`zero_grad` is owned by the outer loop (because the caller controls when
accumulation resets), and the step itself is delegated. No abstraction gap here.

**Summary:** The 5-step boilerplate is NOT widely duplicated across real training
scripts. `btrm_training.py` is the canonical implementation, the other scripts
correctly delegate to it, and `constructive_nan_debug.py` is the only non-canonical
location (and is explicitly a debug tool, not a training pathway).

---

### Finding 3.4 — JSONL Logging: 6 Raw Writers vs TrainingCurveWriter

**Canonical class:** `src_ii/incremental_save.py::TrainingCurveWriter`

`TrainingCurveWriter` handles: file open with optional append mode, `json.dumps`
per entry, immediate flush, and close. It is the correct abstraction for any
JSONL log that must survive process crashes.

**Scripts using TrainingCurveWriter (correct):**
`run_pinkify_validated_training.py` (line 354), `run_reward_validated_training.py`
(line 615), `run_ddgrpo_v2.py` (line 646), `run_flops_budget_100step_v2.py`
(line 258), `validate_ddgrpo_e2e.py` (line 537).

**Scripts using raw `f.write(json.dumps(...) + "\n")` (non-conforming):**

| File | Lines | Context |
|------|-------|---------|
| `scripts_ii/run03_btrm_training.py` | 484 | Per-step training metrics, inside `training_callback` |
| `scripts_ii/run_pinkify_validated_training.py` | 186 | Per-checkpoint pinkify validation log (separate from training curve) |
| `scripts_ii/run_reward_validated_training.py` | 232 | Per-checkpoint combined validation log |
| `scripts_ii/sweep_rtheta_lr.py` | 314 | Per-epoch sweep metrics |
| `scripts_ii/demonstrate_rtheta_policy.py` | 173 | Per-step policy demo scores |
| `scripts_ii/validate_policy_intervention.py` | 550 | Per-step intervention scores |
| `scripts_ii/validate_packed_vs_serial_ktuple.py` | 416, 422 | Two separate per-record writes in same function |
| `scripts_ii/validate_ktuple_6tuple.py` | 312 | Per-row scores |

Note that `run_pinkify_validated_training.py` and `run_reward_validated_training.py`
appear in BOTH lists: they use `TrainingCurveWriter` for the training curve but
raw writes for the separate validation log. Both define local helper functions
(`append_pinkify_log` at line 183, `append_validation_log` at line 229) that
wrap the raw write. These helpers are script-local and not shared with each other,
despite being structurally identical single-write wrappers.

The risk of raw writers is atomicity: a crash between `write()` and `flush()` can
leave a partial JSON line. `TrainingCurveWriter` flushes after every write.
The six raw-write sites all call `flush()` manually or rely on Python's
line-buffered mode — the actual crash risk is low but the inconsistency
means future edits to any of these files may silently drop the flush.

---

## Category 4: Multi-Accelerator Scatter-Gather

Survey date: 2026-02-25.

---

### Finding 4.1 — `BatchExecutor` Is Single-Device by Construction

**File:** `/mnt/f/dox/repos/ai/futudiffu/src_ii/batch_executor.py`
**Lines:** 41–61 (constructor and `execute` method)

`BatchExecutor.__init__` takes one `model` and one `device` (line 41–44):

```python
def __init__(self, model, device: torch.device, max_total_len: int = REFERENCE_TOTAL_LEN):
    self.model = model
    self.device = device
```

The `_execute_plan` function (line 175) moves all tensors to `self.device` and calls
`packed_forward(model, ...)` which calls `model(...)` directly. There is no concept
of routing different bins to different devices — all bins execute on the same device
against the same model object.

**Multi-GPU fix:** The minimum change to dispatch bins across devices is to make
`BatchExecutor` accept a list of `(model_replica, device)` pairs and a bin-routing
policy (e.g., round-robin). `_execute_plan` would partition bins across replicas
and collect results. Because bins are independent (no cross-bin tensor sharing
except in the BTRM training retain_graph path), the scatter and gather are both
trivially parallelizable. Gradient synchronization would require `allreduce` of
parameter gradients after all bins complete, which is the standard DDP pattern.

---

### Finding 4.2 — `forward_packed.py` Has No Device Assumption of Its Own

**File:** `/mnt/f/dox/repos/ai/futudiffu/src_ii/forward_packed.py`
**Lines:** 109–133 (`packed_forward`)

`packed_forward` is a pure function: it takes `model` as an argument and has no
stored device state. It calls `model(x_list, ...)` and the model controls device
placement. The function itself is multi-GPU clean — the constraint is that all
tensors in `x_list` and the model must be on the same device, which is enforced
by the caller.

`prepare_packed_forward` (lines 76–106) is similarly clean: it calls
`model.prepare_packed_state()` and builds tensors on whatever device the model
is on. No hardcoded `"cuda"` or `torch.device("cuda")` calls in this file.

**Multi-GPU fix:** none needed in this file specifically. The device constraint
comes from its callers (`BatchExecutor`, `score_packed`), not from `forward_packed`.

---

### Finding 4.3 — `btrm_training.py` Gradient Loop Is Architecturally Single-Device

**File:** `/mnt/f/dox/repos/ai/futudiffu/src_ii/btrm_training.py`
**Lines:** 809–1461 (`train_btrm_differentiable` step loop)
**Key constraint:** line 982 — `device = next(model.parameters()).device`

The gradient accumulation loop resolves the model's device dynamically:

```python
device = next(model.parameters()).device
```

This is a single device. The entire per-bin scoring and backward path runs on one
device, and the optimizer step at lines 1461–1462 updates parameters on that device.

**Data-parallel BTRM (minimum change):** Wrap the model in
`torch.nn.parallel.DistributedDataParallel` (DDP). Because `score_packed` already
handles batching, DDP's gradient `allreduce` would fire after each `backward()`
call. The macrobatch structure (one optimizer step spanning multiple bins) is
compatible with DDP — the `allreduce` happens at `optimizer.step()` boundaries,
not per-bin. The training loop itself requires no structural changes: the same
`train_btrm_differentiable()` function runs on each GPU with a different
`pair_sampler` slice, and DDP handles gradient synchronization transparently.

**What would NOT work without code changes:** the `retain_graph=not _all_pairs_done`
logic in the FLOPS-budget path (lines 1078–1082). DDP's `allreduce` must fire
before the next backward, but `retain_graph=True` defers freeing the graph
across bins. With DDP, you must ensure the last `backward()` per optimizer step
(the one where `retain_graph=False`) corresponds to the `allreduce` barrier.
The existing structure satisfies this: `retain_graph=False` is set only when
`all(pair_processed)`, which is the final backward in each macrobatch. DDP's
default `find_unused_parameters=False` and `gradient_as_bucket_view=True` are
compatible with this pattern.

---

### Finding 4.4 — `ddreinforce.py` Is Multi-GPU Clean (Pure Math Module)

**File:** `/mnt/f/dox/repos/ai/futudiffu/src_ii/ddreinforce.py`
**Lines:** 1–601

`ddreinforce.py` contains only pure tensor math functions with no `torch.device`
calls and no model references (enforced by the file's import constraints). All
functions accept tensors and return tensors; device placement is the caller's
responsibility.

**Multi-GPU relevance:** `group_advantages()`, `step_log_prob()`, `unbiased_kl()`,
`ddreinforce_loss()`, and `ddgrpo_loss()` are all callable on any device without
modification.

---

### Finding 4.5 — `run_ddgrpo_v2.py` Has Three Hard Serialization Points

**File:** `/mnt/f/dox/repos/ai/futudiffu/scripts_ii/run_ddgrpo_v2.py`
**Lines with per-group serial execution:**

**Serialization Point 1 — Rollout generation (lines 689–699):**
```python
for b, group in enumerate(groups):
    trajs = generate_sde_trajectories(...)  # sequential per group
    all_trajectories.extend(trajs)
```
The comment at line 685–686 explicitly acknowledges this:
```
# Each group generates K trajectories sequentially
# (groups could be parallelized on multi-GPU, but sequential on 1 GPU)
```
Each group of K trajectories for one prompt is generated serially. For B=2, K=2
(the current config), this is 4 sequential 20-step rollouts. For B=4, K=4 (v1
target), this would be 16 sequential rollouts — the bottleneck grows as B*K.

**Serialization Point 2 — Gradient accumulation (lines 745–751):**
```python
policy_optimizer.zero_grad()
grad_result = accumulate_and_step(
    executor, model, groups, all_trajectories, all_advantages, ...
)
```
`accumulate_and_step` calls `accumulate_reinforce_gradients` with
`microbatch_size=GRAD_MICROBATCH=1`, meaning one trajectory per backward() call.
This serializes gradient computation across all B*K trajectories.

**Serialization Point 3 — Scoring all finals (lines 703–713):**
```python
all_scores = score_finals(executor, all_trajectories, cond_map, res_map, ...)
```
`score_finals` issues all B*K scoring queries to `executor.batch_exec.execute()`
in a single call (line 371). Because `BatchExecutor` bins queries by packing and
runs each bin serially, this is still serial on the single device. With multiple
bins, the scoring is also sequential.

**Pipeline parallelism (minimum change for DDGRPO):**

The data flow is: generate → score → compute advantages → backprop → step. Steps
1 and 4 require GPU compute. Steps 2 and 3 are also GPU but much cheaper. Steps
1 and 4 are structurally independent across prompts: the gradient for prompt b=0
does not depend on the trajectory for prompt b=1 (within one iteration; advantages
do depend on intra-group K trajectories, so groups must complete together).

The minimum change for data-parallel rollout + backprop:
- Replicate the model to GPU0 and GPU1 (DDP or manual `model.to("cuda:N")`)
- Assign groups to devices: GPU0 handles `groups[:B//2]`, GPU1 handles `groups[B//2:]`
- Each device generates its rollouts, scores them, and accumulates gradients
- `dist.all_reduce` on gradient tensors before `optimizer.step()`

This avoids pipeline parallelism (which would require splitting the model across
devices) and instead uses data parallelism (each GPU sees a different subset of
prompt groups). Because the model fits in 24 GB on a single 4090 (8 GB for
weights + ~8 GB for activations during backprop), a 2xH100 or 2xRTX Pro 6000
configuration would double throughput without model splitting.

For pipeline parallelism specifically (GPU0 generates while GPU1 backprops from
the PREVIOUS iteration's trajectories):
- The existing per-iteration structure uses `del all_trajectories` at the end
  of each iteration (line 809) and does not buffer trajectories between iterations
- Making this async requires changing the loop structure to produce trajectories
  N iterations ahead of backprop, which is the DDPO off-policy reuse pattern —
  a significant algorithmic change, not a minimum-change refactor

---

### Finding 4.6 — `policy_step.py` `accumulate_reinforce_gradients` Is Single-Device

**File:** `/mnt/f/dox/repos/ai/futudiffu/src_ii/policy_step.py`
**Line 159:** `device = query_sigmas[0].device`

The gradient accumulation loop resolves device from the first query sigma:
```python
device = query_sigmas[0].device
```

All per-step backward calls run on this device. With DDP, this function would
run identically on each GPU replica with a different trajectory subset. No
structural change is needed to this function for data-parallel training.

The `microbatch_size=1` default (set by `GRAD_MICROBATCH=1` in
`run_ddgrpo_v2.py:78`) means one trajectory per executor call and one `backward()`
per trajectory-step pair. This is the most memory-conservative setting but also
the slowest. On 80 GB H100, `microbatch_size=K` (process all K trajectories for
one step simultaneously) would be viable and would reduce executor round-trips
from `N_STEPS * B * K` to `N_STEPS * B`.

---

### Finding 4.7 — `btrm_lifecycle.py::score_packed` Creates a Transient `BatchExecutor` Per Call

**File:** `/mnt/f/dox/repos/ai/futudiffu/src_ii/btrm_lifecycle.py`
**Lines:** 208–229

```python
executor = BatchExecutor(model, device=device)
results = executor.execute(queries)
```

A new `BatchExecutor` is created on every `score_packed` call, discarding the
plan cache. For the training loop in `train_btrm_differentiable`, which calls
`score_packed` thousands of times with the same image resolution structure,
this means the packing plan is recomputed from scratch on every step.

The plan cache in `BatchExecutor` (line 45: `self._plan_cache`) is designed to
amortize packing cost across repeated identical structures. By constructing a
transient executor, the cache is never populated. The `BinPackScheduler.pack()`
call itself is pure Python and fast (~microseconds), so this is not a correctness
issue, but it is the primary reason `score_packed` cannot be upgraded to a
multi-GPU dispatch: the executor object has no persistent routing state.

**Multi-GPU fix:** `btrm_lifecycle.py::score_packed` should accept an optional
pre-built `BatchExecutor` instance (or executor pool). The training scripts that
call `train_btrm_differentiable` already own the model lifecycle; they could
pre-allocate executor(s) at startup and pass them through. This would also
restore the plan cache amortization benefit.

---

### Multi-GPU Summary

| Bottleneck | Location | What Blocks Multi-GPU | Minimum Fix |
|-----------|----------|-----------------------|-------------|
| Single model replica | `BatchExecutor.__init__` line 41 | One `model` arg, one `device` | Accept list of `(model, device)` pairs, round-robin bins |
| Serial rollout generation | `run_ddgrpo_v2.py` line 689 | `for b in groups: generate_sde_trajectories(...)` sequential | Assign `groups[:B//2]` to GPU0, `groups[B//2:]` to GPU1; `allreduce` gradients before step |
| Serial backprop | `policy_step.py` line 159 | `device = query_sigmas[0].device` (single device) | DDP wrap + per-device trajectory slicing |
| Transient executor | `btrm_lifecycle.py` line 228 | `BatchExecutor` created anew per `score_packed` call | Pass persistent executor from caller; enables multi-GPU routing |
| Retain-graph across bins | `btrm_training.py` line 1082 | `retain_graph=True` defers graph free across bins | Compatible with DDP if `allreduce` fires at the final `backward()` (when `all(pair_processed)` = True), which the existing code already ensures |

The minimum data-parallel BTRM training change (not pipeline-parallel) requires
three coordinated edits: (1) DDP-wrap the model in the calling script, (2) split
the `pair_sampler` across ranks so each rank sees different pairs, and (3)
ensure `train_btrm_differentiable`'s optimizer step calls `dist.barrier()` after
`optimizer.step()`. The function body itself is DDP-compatible.

The minimum pipeline-parallel DDGRPO change requires restructuring the iteration
loop to overlap generation and backpropagation across iterations, which is the
DDPO off-policy trajectory reuse pattern and represents a significant algorithmic
change beyond a refactor.

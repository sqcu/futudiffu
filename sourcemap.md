# sourcemap.md — What Every Module Does and Why Scripts Are Dying

## The Thesis

Every GPU operation — inference, BTRM reward training, DDGRPO policy optimization,
policy intervention A/B — flows through one server process, one compiled model instance,
one forward path (`forward_packed.py`), one bin-packing pipeline (`batch_executor.py`),
one adapter routing system (`multi_lora.py`).

Scripts that used to load models locally, wire their own training loops, and invent
bespoke forward paths are being replaced by JSON configs submitted to
`POST /training/start`. The `TrainingOrchestrator` parses the config, selects a
`run_type`, and executes through the same modules that inference uses. Variant
implementations die because there is no second code path to put them on.

The bottleneck is intentional. The question "does the packed forward match the serial
forward" is answered once, in `execute_single_bin()`, not re-answered in every script.

---

## System Architecture (Two Processes, One GPU)

### On "Browser"

"Browser" in the diagrams below means "anything that sends HTTP to the BFF."
The BFF is a torch-free FastAPI server. Its routes accept JSON, return JSON or
SSE event streams. The browser UI is one client. `curl` is another. A coding
agent with `httpx` or `requests` is another. The BFF exists so that launching
a training run, generating an image, or querying adapter state is a single
HTTP call with a JSON body — not "import 14 modules, load the model, wire a
forward loop." If you are a coding agent and you want to run a 5-step BTRM
training benchmark, you `POST /api/batch_generate` with `{"type": "btrm",
"n_steps": 5, ...}` and stream SSE events until `"type": "complete"`. If you
want to generate an image, you `POST /api/batch_generate` with
`{"type": "inference", "prompt": "...", ...}` and poll `/api/stream/{job_id}`.
The presets in `src_ii/client_yeetums/presets/*.json` are example configs.

```
Client (browser, curl, coding agent, httpx)
  │
  │ POST /api/batch_generate {type: "btrm", ...}
  ▼
BFF (port 8001)                          ← torch-free, FastAPI + httpx
  │  src_ii/client_yeetums/app.py           routes, preset discovery, SSE proxy
  │  src_ii/client_yeetums/bridge.py        InferenceBridge (httpx → GPU server)
  │  src_ii/client_yeetums/gallery.py       PNG + JSONL on disk
  │  src_ii/client_yeetums/models.py        Pydantic request/response types
  │  src_ii/config_distributions.py         collapse distributional configs to scalars
  │  src_ii/resolution_sampling.py          (W,H) from megapixels + aspect ratio
  │
  │ POST /training/start {run_type, ...config}
  ▼
GPU Server (port 8787)                   ← owns the GPU, runs everything
  │  src_ii/server.py                       FastAPI, GPUModelBackend, route dispatch
  │  src_ii/training_orchestrator.py        background run manager, SSE events
  │  src_ii/inference_queue.py              batched inference with drain window
  │
  ▼
  Model lifecycle: load ZImageRLAIF → install_multi_lora → torch.compile → serve
  Forward path:    BatchExecutor.execute → execute_single_bin → packed_forward → model()
```

---

## Config-Driven Dispatch

`POST /training/start` receives a JSON body with `run_type`:

| run_type | Orchestrator method | What it does |
|----------|-------------------|--------------|
| `btrm` | `_run_btrm()` | FLOPS-budget macrobatch BTRM reward training |
| `ddgrpo` | `_run_ddgrpo()` | On-policy rollouts → REINFORCE gradient accumulation |
| `policy_intervention` | `_run_policy_intervention()` | A/B comparison: ref adapter vs policy adapter |

Inference (the "Generate" button) routes through `POST /enqueue` → `InferenceQueue` →
`run_trajectory_packed()` → same `BatchExecutor` → same `packed_forward()`.

There is no mode switch on the model. `ZImageRLAIF.forward()` ALWAYS returns
`(diffusion_fields, scores)`. The only variation is `adapter_scales` (which
adapters are active) and whether `torch.inference_mode()` wraps the call.

---

## Central Source Modules

### The Forward Path (every GPU operation flows through these)

| Module | Purpose | Leaf? |
|--------|---------|-------|
| `zimage_model.py` | ZImageRLAIF: unified diffusion + scoring model. `forward()` → `(fields, scores)`. Score head = RMSNorm → Linear → soft_tanh_cap, computed from `layers[-1]` output before `final_layer`. | No |
| `transformer.py` | Branchless NextDiT blocks: `JointTransformerBlock`, `FinalLayer`, `EmbedND`, RoPE, packed sequence construction. | No |
| `attention_srcii.py` | Unified attention dispatch: sage / sage_masked / sdpa. Triton kernel selection. | Leaf |
| `forward_packed.py` | `prepare_packed_forward()` builds block mask + RoPE + pads to REFERENCE_TOTAL_LEN. `packed_forward()` runs model, unpacks per-image results. THE forward path. | No |
| `block_mask.py` | uint8 block-diagonal mask construction for FlexAttention. | Leaf |
| `batch_executor.py` | SCATTER → PACKSOLVE → EXECUTE pipeline. `execute_single_bin()` is the shared inner loop for both single-device and multi-device paths. Plan caching by structure hash. | No |
| `accelerator_pool.py` | K-device dispatch wrapping BatchExecutor. FLOPS-balanced greedy bin assignment. Gradient gathering across replicas. K=1 is the default. | No |
| `bin_packer.py` | Pure Python FFD bin packer. `REFERENCE_TOTAL_LEN=4224`. `compute_effective_seq_len()` = pad32(cap) + pad32(img). | Leaf |
| `inference_packing.py` | `pack_for_inference()`: FFD over entry sequence lengths. `compute_entry_seq_len()`. | Leaf |
| `triumphant_future_reduction_ops.py` | `denoise_all()`, `cfg1()`, `cfg2()`, `gather()`, `euler_step()`, `latent_padded()`. Fused elementwise ops. | Leaf |

### Adapter Routing

| Module | Purpose | Leaf? |
|--------|---------|-------|
| `multi_lora.py` | `MultiLoRALinear`: pre-allocated slots, branchless `_compute_adapter_delta()`, per-token sparse routing via `token_to_image`. Full lifecycle: `install` → `assign` → `freeze` → `train` → `save` → `load`. | Leaf |

### Training — BTRM Reward Model

| Module | Purpose | Leaf? |
|--------|---------|-------|
| `btrm_training.py` | `train_btrm_differentiable()`: 33-param single entry point. FLOPS-budget macrobatch sampling, bin-packed forward, per-bin immediate backward, gradient clipping, optimizer step. ~770 lines. | No |
| `btrm_lifecycle.py` | Free functions replacing deleted BTRMCompoundModel. `setup_btrm_training()`, `persist_btrm()`, `load_btrm()`, `score_packed()`, `make_training_optimizer()`. | No |
| `pair_sampler.py` | `BTRMPairSampler`: on-the-fly pair sampling over ~1.6M combinatorial space. `sample_macrobatch(budget_units)` with 6-tier FLOPS allocation. Clean-biased step selection (80% sigma=0). | No |
| `flops_sampling.py` | `flops_units()`, `pair_flops_units()`. 6-tier resolution bucketing. Per-trajectory FLOPS sampling weights. | No |
| `sigma_schedule.py` | `resolution_shift()` (SD3 Eq.23), `build_sigma_schedule()`, `compute_logsnr_uniform_steps()`, `const_noise_scaling()`. Pure math. | Leaf |
| `resolution_sampling.py` | Continuous resolution from megapixel budgets. `MEGAPIXEL_ANCHORS` (6 tiers), `assign_budget_tier()`, log-uniform aspect ratios, 32px grid snap. | Leaf |

### Training — DDGRPO Policy Optimization

| Module | Purpose | Leaf? |
|--------|---------|-------|
| `ddreinforce.py` | Pure math: `group_advantages()`, `step_log_prob()`, `clipped_surrogate_loss()`, `compute_eta_schedule()`, `euler_sde_generate()`, `recompute_step_log_prob()`, `ddreinforce_loss()`, `ddgrpo_loss()`. | Leaf |
| `policy_step.py` | `accumulate_reinforce_gradients()`: step-first gradient accumulation with micro-batching. `policy_optimizer_step()`: clip + step. `SignAgreementTracker`. | No |

### Training — Inference Path (rollouts)

| Module | Purpose | Leaf? |
|--------|---------|-------|
| `inference_sampling.py` | `run_trajectory_packed()`: builds ktuple specs, creates `ExecutorAdapter`, calls `ktuple_sampling.solve()` which loops Euler steps through `BatchExecutor`. | No |
| `ktuple_sampling.py` | `solve()`: K-tuple guidance with per-step GATHER reduction. | No |
| `inference_queue.py` | `InferenceQueue`: batched inference with configurable drain window. Groups by n_steps, bin-packs across jobs. | No |

### Reward Functions

| Module | Purpose | Leaf? |
|--------|---------|-------|
| `reward_functions.py` | Pixel-space: `pinkify_score_gpu()` (HSV pink hue + coverage contrast), `thisnotthat_score_gpu()` (FrFT spectral discriminant). | No |
| `frft.py` | `frft_descriptor()`, `frft_discriminant_score()`. Batched real-arithmetic FrFT. 0.7ms eager. | Leaf |
| `pinkify_validation.py` | 6-image holdout calibration: `validate_btrm_pinkify_ranking()`. | No |
| `tnt_validation.py` | `validate_tnt_ranking()`. | No |
| `cross_head_decorrelation.py` | `measure_cross_head_decorrelation()`. | No |

### Data Loading

| Module | Purpose | Leaf? |
|--------|---------|-------|
| `dataset_io.py` | `make_load_latent_fn()`: closure returning `(latent, sigma, cond, num_tokens, None)`. `make_reward_manifest_preference_fn()`: ground-truth preference from pixel-space reward. | No |
| `training_setup.py` | `encode_training_prompts()`: load TE → encode → free TE. `build_dataset_positions()`: V2 dataset → BTRMPairSampler. `load_training_backbone()`: ZImageRLAIF + LoRA + optimizer. | No |
| `dataset_catalog.py` | V2 dataset identity + SHA-256 integrity. Named splits. JSON + parquet. | Leaf |
| `dataset_resumption.py` | Trajectory identity matching for generation plan resumption. | Leaf |
| `model_paths.py` | `FP8_PATH`, `TE_PATH`, `VAE_PATH`, `TOKENIZER_PATH`. 23 lines. | Leaf |

### Persistence

| Module | Purpose | Leaf? |
|--------|---------|-------|
| `incremental_save.py` | `TrainingCurveWriter` (flush-per-line JSONL), `PeriodicSaver`, `atomic_json_save`. | Leaf |
| `training_artifacts.py` | `TrainingArtifacts`: streaming JSONL metrics, checkpoint saving (delegates to `btrm_lifecycle`), PIL chart rendering, markdown analysis. | No |
| `training_resume.py` | `detect_resume()`, `load_optimizer_state()`, `save_resume_state()`. | Leaf |
| `validation_metrics.py` | Multi-indexed Welford tracker. `PairResult`, `ValidationMetrics`. Resolution/logSNR/head bucketing. | Leaf |

### Visualization (policy intervention + training charts)

| Module | Purpose | Leaf? |
|--------|---------|-------|
| `infer/charts.py` | PIL-only line/scatter charts for training curves. | Leaf |
| `infer/composites.py` | Side-by-side comparison composites (ref vs policy). | Leaf |
| `infer/diff_analysis.py` | False-color diff images between ref and policy outputs. | Leaf |
| `infer/model_setup.py` | `load_and_prepare_model()`, `load_policy_adapter()`. Compiled-model safe. | No |
| `vae_utils.py` | `load_vae()`, `decode_latent_to_pil()`. | Leaf |

---

## Module Dependency Graph

```
                    training_orchestrator
                   /         |          \
                  v          v           v
            _run_btrm    _run_ddgrpo    _run_policy_intervention
               |            |    \            |
               v            v     v           v
        btrm_training   policy_step  ddreinforce   infer/*
           / | | \          |
          v  v v  v         v
  btrm_lifecycle  pair_sampler  batch_executor ◄── accelerator_pool
      |     |        |   \         / | \
      v     v        v    v       v  v  v
  multi_lora  flops_sampling  forward_packed  inference_packing
   (LEAF)        |                / \              (LEAF)
                 v               v   v
          resolution_sampling  block_mask  bin_packer
             (LEAF)            (LEAF)      (LEAF)

  inference_queue ──► batch_executor
       |
       v
  bin_packer

  inference_sampling ──► ktuple_sampling
       |                      |
       v                      v
  batch_executor        triumphant_future_reduction_ops
                              (LEAF)
```

Leaf modules (no src_ii/ imports): `multi_lora`, `sigma_schedule`, `bin_packer`,
`block_mask`, `resolution_sampling`, `validation_metrics`, `incremental_save`,
`dataset_catalog`, `dataset_resumption`, `training_resume`, `model_paths`,
`attention_srcii`, `inference_packing`, `frft`, `ddreinforce`,
`triumphant_future_reduction_ops`, `vae_utils`, `infer/charts`, `infer/composites`,
`infer/diff_analysis`.

---

## End-to-End: Start BTRM Training

```
Client POST /api/batch_generate {type: "btrm", head_names, pref_keys, n_steps, ...}
  │
  ▼
BFF routes by type="btrm" → bridge.start_training_run(config)
  │  POST /training/start {run_type: "btrm", ...}
  ▼
GPU Server → training_orchestrator.start_btrm_run(config, backend)
  │  Spawns background thread: _run_btrm()
  │
  │  1. DatasetReader(dataset_path)
  │  2. build_positions_from_v2() → BTRMPairSampler
  │  3. backend.encode_prompt() for each unique prompt    [loads TE, frees after]
  │  4. make_load_latent_fn(reader, prompt_cache)
  │  5. backend._ensure_diffusion()                       [loads backbone + compile]
  │  6. setup_btrm_training(model, adapter, rank, alpha)  [LoRA + freeze + optimizer]
  │
  │  7. train_btrm_differentiable():
  │     FOR each step:
  │       pair_sampler.sample_macrobatch(budget_units=3.0)
  │       load_latent_fn() for each image
  │       _bin_pack_images() → BinPackScheduler
  │       FOR each bin:
  │         score_packed() → BatchExecutor.execute()
  │           → execute_single_bin()
  │             → packed_forward(model, x_list, timesteps, ...)
  │               → model.forward()  →  (fields, scores)
  │             → denoise_all()
  │         BT loss + immediate backward
  │       optimizer.step()
  │       callback → _publish("step", metrics) → SSE → BFF → browser
  │
  │  8. persist_btrm() → save adapter + score head
  │  9. _publish("complete")
  ▼
BFF proxies SSE events → client (browser, agent, curl) receives progress + gallery URLs
```

## End-to-End: Generate Image

```
Client POST /api/batch_generate {type: "inference", prompt, cfg, resolution, k, ...}
  │
  ▼
BFF: for i in range(k):
  │  resolve_generation_config() → collapse distributions
  │  sample_resolution() → (W, H)
  │  bridge.enqueue_generation() → POST /enqueue → InferenceQueue
  │
  ▼
InferenceQueue._worker_loop():
  drain batch → group by n_steps → BinPackScheduler
  _execute_batch():
    _encode_prompts()       → backend.encode_prompt()     [TE phase]
    _run_packed_trajectory() → run_trajectory_packed()     [diffusion phase]
      BatchExecutor + ExecutorAdapter
      ktuple_sampling.solve():
        FOR each of 30 steps:
          ExecutorAdapter → BatchExecutor.execute()
            → execute_single_bin() → packed_forward() → model()
          gather() → euler_step()
    _decode_finals()        → backend.vae_decode_png()    [VAE phase]
  │
  ▼
BFF: intercepts "complete" SSE → fetches PNG → saves to gallery → emits "gallery_ready"
Client: receives gallery_ready event with image URL → fetch or display
```

---

## What Survived the Purge

40 scripts and tests were deleted: 20 one-shot validation scripts (validated a specific
refactor, never needed again), 20 scripts subsumed by config-driven server launch
(their parameters now live in `src_ii/client_yeetums/presets/*.json` or as fields in
a `POST /training/start` body).

### scripts_ii/ (22 files)

**Infrastructure (5):**
- `launch_server.py` — GPU server entry point (uvicorn)
- `launch_yeetums.py` — BFF entry point (uvicorn, port 8001)
- `restart_server.py` — atomic kill + relaunch + health poll
- `node_heartbeat.py` — remote spot instance observer (tmux-resident)
- `observe_remote.py` — local aggregator for remote observer protocol

**Production regression tests (2):**
- `validate_batch_rollout.py` — full inference E2E (TE → packed rollout → VAE)
- `validate_packed_vs_serial_ktuple.py` — BatchExecutor bin-packed vs serial for K=6

**Dataset/offline tooling (15):**
- `analyze_sweep_curves.py` — post-hoc training curve analysis (EMA, derivatives)
- `audit_dataset.py` — V2 dataset integrity scan
- `backfill_v2_hashes.py` — identity hash backfill into parquet index
- `compare_pinkify_scores.py` — BTRM vs literal rule Spearman correlation
- `compress_node_logs.py` — remote node log compression
- `generate_preference_labels.py` — VAE decode + pixel-space reward → pairwise prefs
- `inspect_datasets.py` — CLI dataset catalog inspector
- `measure_prompt_tokens.py` — token length statistics for unique prompts
- `merge_v2_datasets.py` — merge per-GPU V2 splits into unified dataset
- `migrate_v1_into_v2.py` — one-time V1 → V2 format migration
- `plot_sweep_curves.py` — PIL-based sweep comparison plots
- `render_attention_maps.py` — attention heatmap visualization from captured stats
- `render_comparison.py` — trajectory comparison renders (per-step diff strips)
- `score_distribution_comparison.py` — score distribution offline analysis
- `validate_v2_dataset.py` — 5-phase V2 dataset validation

### tests/ (13 files)

All import exclusively from `src_ii/`. None import from frozen `src/futudiffu/`.
None use ZMQ. All run without a server.

- `stubbed_skinny_shared_ii.py` — SSS-II ZImageRLAIF fixture (shared helper)
- `test_clean_biased_sampling.py` — 80/20 clean-biased step selection
- `test_dataset_catalog.py` — catalog register/verify/split lifecycle
- `test_dataset_resumption.py` — plan identity matching + resume detection
- `test_exemplar_dedup.py` — cross-head deduplication in exemplar renderer
- `test_fastapi_server.py` — HTTP route tests with mock backend
- `test_incremental_save.py` — JSONL writer + atomic save + periodic saver
- `test_logsnr_step_weighting.py` — logSNR sampling weight correctness
- `test_logsnr_uniform_steps.py` — logSNR-uniform step selection
- `test_multi_lora_compiled.py` — MultiLoRA through torch.compile (GPU)
- `test_pinkify_validation.py` — holdout ranking calibration
- `test_sss_ii_funfetti.py` — compiled funfetti packed forward (GPU)
- `test_sss_ii_ktuple_rollout.py` — full K-tuple rollout + packed-vs-serial (GPU)

---

## Gaps in the Server Bottleneck

Two capabilities were provided by deleted scripts and lack server-side equivalents.
They are the next orchestrator routes to implement:

1. **Dataset generation** (was `generate_btrm_dataset.py`): needs
   `run_type="dataset_generation"` on TrainingOrchestrator. Rollout generation +
   V2 persistence + resume detection. The relay client pattern was already right;
   the server route was missing.

2. **Parameter sweeps** (was `sweep_ktuple_gain.py`): run N configs sharing one
   compiled model. Needs a sweep orchestrator that accepts a config matrix and
   streams per-config results as SSE events.

---

## Leaf Module Quick Reference

These modules have zero src_ii/ imports. They are the foundation layer.

| Module | Lines | What it computes |
|--------|-------|-----------------|
| `model_paths.py` | 23 | 4 filesystem paths |
| `bin_packer.py` | ~200 | FFD bin packing, `REFERENCE_TOTAL_LEN=4224` |
| `block_mask.py` | ~80 | uint8 block-diagonal attention masks |
| `sigma_schedule.py` | ~180 | Sigma tables, resolution shift, logSNR |
| `resolution_sampling.py` | ~120 | Continuous (W,H) from megapixel budget |
| `multi_lora.py` | ~890 | Pre-allocated LoRA slots, branchless routing |
| `validation_metrics.py` | ~300 | Multi-indexed Welford running stats |
| `incremental_save.py` | ~100 | JSONL writer, atomic JSON, periodic saver |
| `dataset_catalog.py` | ~350 | V2 dataset registry + SHA-256 integrity |
| `dataset_resumption.py` | ~150 | Generation plan resume detection |
| `training_resume.py` | ~120 | Optimizer state save/load |
| `ddreinforce.py` | ~480 | REINFORCE/GRPO/PPO pure math |
| `frft.py` | ~150 | Fractional Fourier transform descriptors |
| `attention_srcii.py` | ~200 | Triton kernel dispatch for attention |
| `inference_packing.py` | ~80 | FFD over entry sequence lengths |
| `triumphant_future_reduction_ops.py` | ~300 | Denoise, CFG, gather, Euler step |
| `vae_utils.py` | ~80 | VAE load + decode to PIL |

# HEAD Diff Audit for Training Pipeline Files

Generated 2026-02-25. Compares working tree against git HEAD (`27a44ff`) for
every file listed in `src_ii/TRAINING_CALLGRAPH.md` plus `src/futudiffu/`
modules imported by the training pipeline.

---

## Files with NON-WHITESPACE changes

### 1. `src_ii/btrm_lifecycle.py` (5 lines added)

Added a docstring clarification to `score_packed()` explaining that
`adapter_scales` is intentionally not passed during BTRM training.

```diff
diff --git a/src_ii/btrm_lifecycle.py b/src_ii/btrm_lifecycle.py
index 7eb791e..5a8034d 100644
--- a/src_ii/btrm_lifecycle.py
+++ b/src_ii/btrm_lifecycle.py
@@ -188,6 +188,11 @@ def score_packed(
     each bin as a packed forward, and returns tagged scores. This function
     collects scores in submission order.

+    BTRM training: adapter_scales is intentionally NOT passed. The backward
+    flows only through score_norm + score_proj (the score head), not through
+    the full backbone. LoRA adapters are trained during policy optimization,
+    not during BTRM reward model training.
+
     Args:
         model: ZImageRLAIF model (compiled or raw).
         images: List of (latent, timestep, conditioning, num_tokens) tuples.
```

### 2. `scripts_ii/run_reward_validated_training.py` (8 lines changed, net -4)

Changed OUTPUT_DIR, reduced N_STEPS from 150 to 100, reduced CHECKPOINT_STEPS,
and changed `load_latent_fn` return signature to include 5th element (`None`).

```diff
diff --git a/scripts_ii/run_reward_validated_training.py b/scripts_ii/run_reward_validated_training.py
index d770e4a..8ddabec 100644
--- a/scripts_ii/run_reward_validated_training.py
+++ b/scripts_ii/run_reward_validated_training.py
@@ -45,18 +45,18 @@ VAE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\vae\zimage.safetensors"
 TOKENIZER_PATH = str(REPO_ROOT / "src" / "futudiffu" / "tokenizer")

 DATASET_DIR = REPO_ROOT / "multi_res_trajectories"
-OUTPUT_DIR = REPO_ROOT / "training_output" / "reward_function_run_tnt_v2"
+OUTPUT_DIR = REPO_ROOT / "training_output" / "reward_function_run_head_test"
 PINKIFY_CHALLENGE_DIR = REPO_ROOT / "i2i_off_policies" / "PINKIFY_cases"
 TNT_CHALLENGE_DIR = REPO_ROOT / "i2i_off_policies"

-N_STEPS = 150
+N_STEPS = 100
 MACROBATCH_BUDGET = 3.0
 MACROBATCH_CROSS_RES = True
 LR = 3e-4
 GRAD_CLIP = 0.1
 WARMUP_STEPS = 5
 LR_SCHEDULE = "warmup_cosine"
-CHECKPOINT_STEPS = [25, 50, 75, 100, 125]
+CHECKPOINT_STEPS = [25, 50, 75]
 CLEAN_FRACTION = 0.8

 HEAD_NAMES = ("pinkify", "thisnotthat")
@@ -575,7 +575,7 @@ def main():

         num_tokens = cond.shape[1]

-        return latent, timestep, cond, num_tokens
+        return latent, timestep, cond, num_tokens, None

     print("\n" + "=" * 60)
     print("  Building reward manifest")
```

### 3. `scripts_ii/run_flops_budget_100step_v2.py` (104 lines changed: +31 added, -104 removed)

Replaced inline path constants with `from src_ii.model_paths import ...`.
Replaced inline prompt encoding with `encode_training_prompts()`.
Replaced inline backbone loading with `load_training_backbone()`.
Replaced inline `load_latent_fn` closure with `make_load_latent_fn()`.
Changed test load assertion to expect 5-tuple return.

```diff
diff --git a/scripts_ii/run_flops_budget_100step_v2.py b/scripts_ii/run_flops_budget_100step_v2.py
index 857ced3..3897ba6 100644
--- a/scripts_ii/run_flops_budget_100step_v2.py
+++ b/scripts_ii/run_flops_budget_100step_v2.py
@@ -37,10 +37,7 @@ sys.path.insert(0, str(REPO_ROOT / "src"))
 import torch


-FP8_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
-TE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"
-VAE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\vae\zimage.safetensors"
-TOKENIZER_PATH = str(REPO_ROOT / "src" / "futudiffu" / "tokenizer")
+from src_ii.model_paths import FP8_PATH, TE_PATH, VAE_PATH, TOKENIZER_PATH

 DATASET_DIR = REPO_ROOT / "multi_res_trajectories"
 OUTPUT_DIR = REPO_ROOT / "flops_budget_100step_v2_output"
@@ -185,108 +182,35 @@ def main():
     print("  Phase 2: Encoding prompts")
     print("=" * 60)

-    from futudiffu.text_encoder import create_tokenizer, load_text_encoder, encode_prompt
-
-    tokenizer = create_tokenizer(TOKENIZER_PATH)
-    te_model = load_text_encoder(TE_PATH, device=device, dtype=dtype)
-
-    prompt_cache = {}
-    for idx in traj_ids:
-        meta, _ = reader[idx]
-        prompt = meta.get("prompt", "")
-        if prompt and prompt not in prompt_cache:
-            cond = encode_prompt(te_model, tokenizer, prompt, device=device)
-            prompt_cache[prompt] = cond.cpu()
+    from src_ii.training_setup import encode_training_prompts

+    prompt_cache = encode_training_prompts(
+        reader, traj_ids, TOKENIZER_PATH, TE_PATH, device=device, dtype=dtype,
+    )
     n_prompts = len(prompt_cache)
-    print(f"  Encoded {n_prompts} unique prompts")
-
-    del te_model, tokenizer
-    torch.cuda.empty_cache()
-    vram_after_te_free = torch.cuda.memory_allocated() / 1e9
-    print(f"  TE freed. VRAM: {vram_after_te_free:.2f} GB")

     print("\n" + "=" * 60)
     print("  Phase 3: Loading backbone + creating BTRM compound model")
     print("=" * 60)

-    from src_ii.zimage_model import load_zimage_rlaif
-    from src_ii.btrm_lifecycle import setup_btrm_training, persist_btrm
-    from src_ii.multi_lora import get_adapter_params
-    from src_ii.sigma_schedule import build_sigma_schedule, resolution_shift
+    from src_ii.training_setup import load_training_backbone
+    from src_ii.btrm_lifecycle import persist_btrm

-    raw_model = load_zimage_rlaif(
-        FP8_PATH, device=device, dtype=dtype,
-        compile_model=False, fuse=True,
-    )
-    vram_after_backbone = torch.cuda.memory_allocated() / 1e9
-    print(f"  VRAM after backbone: {vram_after_backbone:.2f} GB")
-
-    optimizer = setup_btrm_training(
-        raw_model,
-        adapter_name="rtheta",
-        adapter_rank=8,
-        adapter_alpha=16.0,
-        adapter_init_b_std=0.01,
+    raw_model, optimizer, head_names_loaded = load_training_backbone(
+        FP8_PATH, device=device, dtype=dtype, lr=LR,
     )

+    from src_ii.multi_lora import get_adapter_params
     n_adapter = sum(p.numel() for p in get_adapter_params(raw_model, "rtheta").values())
     n_head = sum(p.numel() for p in raw_model.score_proj.parameters()) + \
              sum(p.numel() for p in raw_model.score_norm.parameters())
-    print(f"  Adapter params: {n_adapter:,}")
-    print(f"  Head params: {n_head:,}")
-    print(f"  Total trainable: {n_adapter + n_head:,}")
-    vram_after_btrm = torch.cuda.memory_allocated() / 1e9
-    print(f"  VRAM after BTRM: {vram_after_btrm:.2f} GB")
-
-    _v2_meta_cache = {}
-
-    def _get_v2_meta(traj_id):
-        if traj_id not in _v2_meta_cache:
-            meta, accessor = reader[traj_id]
-            _v2_meta_cache[traj_id] = (meta, accessor)
-        return _v2_meta_cache[traj_id]
-
-    def load_latent_fn(key):
-        traj_id, step_key = key
-        meta, accessor = _get_v2_meta(traj_id)
-        latent = accessor[step_key].to(device=device, dtype=dtype)
-        if latent.dim() == 3:
-            latent = latent.unsqueeze(0)
-
-        n_steps_traj = meta.get("n_steps", 30)
-        w = meta.get("width", 1280)
-        h = meta.get("height", 832)
-        recorded_shift = meta.get("sampling_shift")
-        if recorded_shift is not None:
-            shift = float(recorded_shift)
-        else:
-            shift = resolution_shift(w, h)
-
-        sigmas = build_sigma_schedule(
-            n_steps_traj, sampling_shift=shift, device="cpu", dtype=torch.float32,
-        )
-
-        if step_key == "final":
-            sigma_val = 0.0
-        else:
-            step_idx = int(step_key.split("_")[1])
-            sigma_val = float(sigmas[step_idx].item()) if step_idx < len(sigmas) else 0.01
-
-        timestep = torch.tensor([sigma_val], device=device, dtype=dtype)
-
-        prompt = meta.get("prompt", "")
-        cond = prompt_cache.get(prompt)
-        if cond is None:
-            raise ValueError(f"No cached prompt for traj {traj_id}: '{prompt[:60]}...'")
-        cond = cond.to(device=device, dtype=dtype)
-
-        num_tokens = cond.shape[1]

-        return latent, timestep, cond, num_tokens
+    from src_ii.dataset_io import make_load_latent_fn
+    load_latent_fn = make_load_latent_fn(reader, prompt_cache, device=device, dtype=dtype)

+    # Verify final sigma = 0.0 (critical correctness check)
     test_key = (traj_ids[0], positions[0].step_key)
-    lat, ts, cond, nt = load_latent_fn(test_key)
+    lat, ts, cond, nt, _ = load_latent_fn(test_key)
     print(f"  Test load: latent={lat.shape}, timestep={ts.shape}, cond={cond.shape}")

     test_final_key = (traj_ids[0], "final")
```

### 4. `scripts_ii/run_pinkify_validated_training.py` (131 lines changed: +26 added, -131 removed)

Replaced inline path constants with `from src_ii.model_paths import ...`.
Removed `import math`. Replaced inline dataset loading with `build_dataset_positions()`.
Replaced inline prompt encoding with `encode_training_prompts()`.
Replaced inline backbone loading with `load_training_backbone()`.
Replaced inline `load_latent_fn` closure with `make_load_latent_fn()`.

```diff
diff --git a/scripts_ii/run_pinkify_validated_training.py b/scripts_ii/run_pinkify_validated_training.py
index 340a054..3154128 100644
--- a/scripts_ii/run_pinkify_validated_training.py
+++ b/scripts_ii/run_pinkify_validated_training.py
@@ -21,7 +21,6 @@ Execution:
 from __future__ import annotations

 import json
-import math
 import sys
 import time
 from datetime import datetime, timezone
@@ -36,10 +35,7 @@ sys.path.insert(0, str(REPO_ROOT / "src"))
 import torch


-FP8_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
-TE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"
-VAE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\vae\zimage.safetensors"
-TOKENIZER_PATH = str(REPO_ROOT / "src" / "futudiffu" / "tokenizer")
+from src_ii.model_paths import FP8_PATH, TE_PATH, VAE_PATH, TOKENIZER_PATH

 DATASET_DIR = REPO_ROOT / "multi_res_trajectories"
 OUTPUT_DIR = REPO_ROOT / "training_output" / "pinkify_validation_run"
@@ -219,66 +215,32 @@ def main():
     print("  Phase 1: Loading multi-res V2 dataset")
     print("=" * 60)

-    from futudiffu.dataset_v2 import DatasetReader
-    from src_ii.pair_sampler import BTRMPairSampler, build_positions_from_v2
-    from src_ii.flops_sampling import compute_flops_sampling_weights_from_positions
+    from src_ii.training_setup import build_dataset_positions

-    reader = DatasetReader(str(DATASET_DIR))
-    n_available = len(reader)
-    print(f"  Dataset: {n_available} trajectories")
+    reader, positions, sampler, traj_ids = build_dataset_positions(
+        DATASET_DIR, clean_fraction=CLEAN_FRACTION,
+    )

-    if n_available < 10:
-        print(f"  ERROR: Need at least 10 trajectories, have {n_available}")
+    if len(traj_ids) < 10:
+        print(f"  ERROR: Need at least 10 trajectories, have {len(traj_ids)}")
         return 1

-    traj_ids = list(range(n_available))
-    positions = build_positions_from_v2(reader, traj_ids=traj_ids)
-    print(f"  Positions: {len(positions)} across {len(traj_ids)} trajectories")
-
     res_dist = {}
     for pos in positions:
         key = f"{pos.width}x{pos.height}"
         res_dist[key] = res_dist.get(key, 0) + 1
     n_unique_res = len(res_dist)
-    print(f"  Unique resolutions: {n_unique_res}")
-
-    flops_weights = compute_flops_sampling_weights_from_positions(positions)
-
-    sampler = BTRMPairSampler(
-        positions=positions,
-        allow_inter_trajectory=True,
-        allow_intra_trajectory=True,
-        rng_seed=42,
-        flops_weights=flops_weights,
-        clean_fraction=CLEAN_FRACTION,
-    )
-    print(f"  Pair space: {sampler.pair_space_size:,} possible pairs")
-    print(f"  Clean fraction: {CLEAN_FRACTION}")
-    print(f"  Populated tiers: {sampler.populated_tiers}")

     print("\n" + "=" * 60)
     print("  Phase 2: Encoding prompts")
     print("=" * 60)

-    from futudiffu.text_encoder import create_tokenizer, load_text_encoder, encode_prompt
-
-    tokenizer = create_tokenizer(TOKENIZER_PATH)
-    te_model = load_text_encoder(TE_PATH, device=device, dtype=dtype)
-
-    prompt_cache = {}
-    for idx in traj_ids:
-        meta, _ = reader[idx]
-        prompt = meta.get("prompt", "")
-        if prompt and prompt not in prompt_cache:
-            cond = encode_prompt(te_model, tokenizer, prompt, device=device)
-            prompt_cache[prompt] = cond.cpu()
+    from src_ii.training_setup import encode_training_prompts

+    prompt_cache = encode_training_prompts(
+        reader, traj_ids, TOKENIZER_PATH, TE_PATH, device=device, dtype=dtype,
+    )
     n_prompts = len(prompt_cache)
-    print(f"  Encoded {n_prompts} unique prompts")
-
-    del te_model, tokenizer
-    torch.cuda.empty_cache()
-    print(f"  TE freed. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

     print("\n" + "=" * 60)
     print("  Phase 3: Encoding PINKIFY challenge images to latents")
@@ -318,31 +280,17 @@ def main():
     print("  Phase 4: Loading backbone + creating BTRM compound model")
     print("=" * 60)

-    from src_ii.zimage_model import load_zimage_rlaif
-    from src_ii.btrm_lifecycle import setup_btrm_training, persist_btrm, score_serial
-    from src_ii.multi_lora import get_adapter_params
-    from src_ii.sigma_schedule import build_sigma_schedule, resolution_shift
+    from src_ii.training_setup import load_training_backbone
+    from src_ii.btrm_lifecycle import persist_btrm

-    raw_model = load_zimage_rlaif(
-        FP8_PATH, device=device, dtype=dtype,
-        compile_model=False, fuse=True,
-    )
-    print(f"  VRAM after backbone: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
-
-    optimizer = setup_btrm_training(
-        raw_model,
-        adapter_name="rtheta",
-        adapter_rank=8,
-        adapter_alpha=16.0,
-        adapter_init_b_std=0.01,
+    raw_model, optimizer, head_names_loaded = load_training_backbone(
+        FP8_PATH, device=device, dtype=dtype, lr=LR,
     )

+    from src_ii.multi_lora import get_adapter_params
     n_adapter = sum(p.numel() for p in get_adapter_params(raw_model, "rtheta").values())
     n_head = sum(p.numel() for p in raw_model.score_proj.parameters()) + \
              sum(p.numel() for p in raw_model.score_norm.parameters())
-    print(f"  Adapter params: {n_adapter:,}")
-    print(f"  Head params: {n_head:,}")
-    print(f"  Total trainable: {n_adapter + n_head:,}")

     print("\n" + "=" * 60)
     print("  Phase 4b: Initial PINKIFY evaluation (before training)")
@@ -369,51 +317,8 @@ def main():
         status = "PASS" if check["passed"] else "FAIL"
         print(f"    [{status}] {check['name']}")

-    _v2_meta_cache = {}
-
-    def _get_v2_meta(traj_id):
-        if traj_id not in _v2_meta_cache:
-            meta, accessor = reader[traj_id]
-            _v2_meta_cache[traj_id] = (meta, accessor)
-        return _v2_meta_cache[traj_id]
-
-    def load_latent_fn(key):
-        traj_id, step_key = key
-        meta, accessor = _get_v2_meta(traj_id)
-        latent = accessor[step_key].to(device=device, dtype=dtype)
-        if latent.dim() == 3:
-            latent = latent.unsqueeze(0)
-
-        n_steps_traj = meta.get("n_steps", 30)
-        w = meta.get("width", 1280)
-        h = meta.get("height", 832)
-        recorded_shift = meta.get("sampling_shift")
-        if recorded_shift is not None:
-            shift = float(recorded_shift)
-        else:
-            shift = resolution_shift(w, h)
-
-        sigmas = build_sigma_schedule(
-            n_steps_traj, sampling_shift=shift, device="cpu", dtype=torch.float32,
-        )
-
-        if step_key == "final":
-            sigma_val = 0.0
-        else:
-            step_idx = int(step_key.split("_")[1])
-            sigma_val = float(sigmas[step_idx].item()) if step_idx < len(sigmas) else 0.01
-
-        timestep = torch.tensor([sigma_val], device=device, dtype=dtype)
-
-        prompt = meta.get("prompt", "")
-        cond = prompt_cache.get(prompt)
-        if cond is None:
-            raise ValueError(f"No cached prompt for traj {traj_id}: '{prompt[:60]}...'")
-        cond = cond.to(device=device, dtype=dtype)
-
-        num_tokens = cond.shape[1]
-
-        return latent, timestep, cond, num_tokens
+    from src_ii.dataset_io import make_load_latent_fn
+    load_latent_fn = make_load_latent_fn(reader, prompt_cache, device=device, dtype=dtype)

     def preference_fn(pair: dict) -> dict:
         """Deterministic preference: cleaner image (lower sigma) wins."""
```

---

## NEW files (not in HEAD)

These files exist on disk in `src_ii/` but are not tracked by git at HEAD:

| File | Lines | Description |
|------|-------|-------------|
| `src_ii/TRAINING_CALLGRAPH.md` | 609 | Training pipeline callgraph documentation |
| `src_ii/dataset_io.py` | 145 | `make_load_latent_fn()` factory, consolidated from script closures |
| `src_ii/model_paths.py` | 22 | Centralized model weight paths (FP8_PATH, TE_PATH, etc.) |
| `src_ii/training_setup.py` | 200 | `build_dataset_positions()`, `encode_training_prompts()`, `load_training_backbone()` |
| `src_ii/training_resume.py` | 118 | Training checkpoint resume logic |
| `src_ii/distributed.py` | 97 | Multi-GPU / distributed training utilities |

---

## Clean files (whitespace-only or no changes)

### src_ii/ training pipeline modules

- `src_ii/btrm_training.py` -- clean
- `src_ii/zimage_model.py` -- clean
- `src_ii/multi_lora.py` -- clean
- `src_ii/pair_sampler.py` -- clean
- `src_ii/flops_sampling.py` -- clean
- `src_ii/sigma_schedule.py` -- clean
- `src_ii/vae_utils.py` -- clean
- `src_ii/pinkify_validation.py` -- clean
- `src_ii/tnt_validation.py` -- clean
- `src_ii/cross_head_decorrelation.py` -- clean
- `src_ii/reward_functions.py` -- clean
- `src_ii/training_artifacts.py` -- clean
- `src_ii/incremental_save.py` -- clean
- `src_ii/batch_executor.py` -- clean
- `src_ii/forward_packed.py` -- clean
- `src_ii/bin_packer.py` -- clean
- `src_ii/inference_packing.py` -- clean
- `src_ii/triumphant_future_reduction_ops.py` -- clean
- `src_ii/block_mask.py` -- clean
- `src_ii/validation_metrics.py` -- clean
- `src_ii/transformer.py` -- clean
- `src_ii/resolution_sampling.py` -- clean
- `src_ii/exemplar_renderer.py` -- clean
- `src_ii/rendering.py` -- clean
- `src_ii/dataset_generator.py` -- clean
- `src_ii/dataset_filters.py` -- clean
- `src_ii/dataset_catalog.py` -- clean
- `src_ii/dataset_resumption.py` -- clean
- `src_ii/attention_srcii.py` -- clean
- `src_ii/stats.py` -- clean
- `src_ii/sampling_identity.py` -- clean
- `src_ii/policy_step.py` -- clean
- `src_ii/ddreinforce.py` -- clean
- `src_ii/ktuple_sampling.py` -- clean
- `src_ii/visualization.py` -- clean
- `src_ii/http_client.py` -- clean
- `src_ii/server.py` -- clean
- `src_ii/server_models.py` -- clean
- `src_ii/frft.py` -- clean
- `src_ii/infer/charts.py` -- clean
- `src_ii/infer/composites.py` -- clean
- `src_ii/infer/diff_analysis.py` -- clean
- `src_ii/infer/euler.py` -- clean
- `src_ii/infer/model_setup.py` -- clean
- `src_ii/infer/text_encoding.py` -- clean
- `src_ii/infer/trajectory.py` -- clean

### scripts_ii/ entry point scripts

- `scripts_ii/run03_btrm_training.py` -- clean

### src/futudiffu/ imported modules

- `src/futudiffu/__init__.py` -- clean
- `src/futudiffu/btrm.py` -- clean
- `src/futudiffu/dataset_v2.py` -- clean
- `src/futudiffu/text_encoder.py` -- clean
- `src/futudiffu/vae.py` -- clean
- `src/futudiffu/fp8.py` -- clean
- `src/futudiffu/diffusion_model.py` -- clean
- `src/futudiffu/sage_attention.py` -- clean
- `src/futudiffu/sage_kernels.py` -- clean
- `src/futudiffu/tokenizer/merges.txt` -- clean
- `src/futudiffu/tokenizer/tokenizer_config.json` -- clean
- `src/futudiffu/tokenizer/vocab.json` -- clean

---

## Summary

| Category | Count |
|----------|-------|
| Files with non-whitespace changes | 4 |
| New files (untracked) | 6 |
| Clean files | 57 |
| **Total audited** | **67** |

The 4 changed files represent a refactoring pattern: consolidating duplicated
inline code (path constants, prompt encoding, backbone loading, `load_latent_fn`
closures) from the three training entry scripts into shared modules
(`model_paths.py`, `training_setup.py`, `dataset_io.py`). The only non-refactor
change is the `score_packed()` docstring addition in `btrm_lifecycle.py` and
the hyperparameter tweaks in `run_reward_validated_training.py` (N_STEPS 150->100,
output dir rename, 5-tuple return signature).

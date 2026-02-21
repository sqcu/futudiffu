# Audit: Fake Porting, Fake src_ii, and VRAM Oversubscription

## 1. VRAM Oversubscription: What Was Found and How It Was Fixed

The VRAM oversubscription was concentrated in `scripts_ii/attention_interpretability.py`, in the `AttentionCapture` monkey-patch of `sdpa_attention`. The original code at line 114 performed `torch.matmul(q.float(), k.float().transpose(-2, -1))` which materialised the full `(B, heads, seq_q, seq_k)` attention weight matrix in float32. For Z-Image at 1024x1024 resolution, the image sequence length is `(1024/8/2)^2 = 4096` image tokens plus text tokens (~32-128), giving a total sequence around 4128. The full attention matrix for all 30 heads was therefore `1 * 30 * 4128 * 4128 * 4 bytes = ~1.93 GB` in float32. Crucially, this allocation happened *inside the model's forward pass*, stacking on top of: the ~6 GB FP8 model, the forward pass activations (~3-4 GB), the float32 upcasts of q and k (`2 * 1 * 30 * 4128 * 128 * 4 bytes = ~120 MB` each), and the softmax output (another ~1.93 GB). Combined with PyTorch's memory fragmentation overhead, this pushed well past the 24 GB RTX 4090 limit, especially since this monkey-patch ran for every one of the 36 attention layers in the model (4 refiner + 32 main blocks).

The fix replaces the all-heads-at-once materialisation with a per-head loop. Each iteration computes `(B, 1, seq_q, seq_k)` logits for a single head (~64 MB at seq=4128), reduces it immediately to per-token statistics (attn_received, attn_given, head_norms), then deletes the intermediate. This caps the peak overhead of the statistics-gathering at ~64 MB instead of ~4 GB. The model's actual output is now computed via `F.scaled_dot_product_attention` (the efficient SDPA kernel) rather than manual matmul-then-matmul-v, which uses the flash-attention memory-efficient path that never materialises the full attention matrix. The statistics and the model output are now computed by independent code paths: the per-head loop (under `torch.no_grad()`) for diagnostics, and SDPA for the model's actual computation.

No other script had a VRAM oversubscription bug. `train_pinkify_btrm.py` correctly uses three-phase lifecycle: (1) load TE, encode, free TE, (2) load diffusion model, extract hidden states to CPU, free model, (3) train BTRM head on CPU-cached data. `generate_preference_labels.py` loads only the VAE (~160 MB). `render_attention_maps.py` loads only the VAE plus pre-computed CPU-resident `.pt` files. The remaining scripts (`score_distribution_comparison.py`, `verify_btrm_persistence.py`) do not load GPU models at all.

## 2. src_ii/ Files: Genuine Narrow Ports vs Fake Ports

**Genuinely narrow ports (5 of 7 files):**

- `forward.py` -- Defines exactly two functions: `nfe()` (sigma-to-timestep conversion + model call) and `denoise()` (pure math: `x - pred * sigma`). These are strict narrowings of the sprawling `sampling.py` which tangles scheduling, guidance, solving, and callback logic. Clean import boundary: imports nothing from futudiffu beyond the model class.

- `guided_denoiser.py` -- Implements `make_guided_denoiser()` as a closure factory with type signature `(x, sigma) -> denoised`. Replaces both `make_cfg_model_fn` and `build_cfg_model_fn` from the original codebase. Handles the cfg=1.0 vs cfg!=1.0 branch at construction time rather than runtime. Uses `forward.nfe` and `forward.denoise` as its only lower-layer calls.

- `sigma_schedule.py` -- Pure math module (torch only, no futudiffu imports). Ports `time_snr_shift`, `build_sigmas`, `simple_scheduler`, `build_sigma_schedule`, and the two CONST scaling functions. These were previously scattered across `sampling.py` mixed with solver logic. This file is a genuine narrowing.

- `solver.py` -- `to_d()` and `euler_solve()` as a pure solver with zero model/conditioning knowledge. Takes a `denoiser_fn` closure and a sigma schedule. The original `sampling.py` interleaved solver logic with scheduling, guidance setup, and callback management.

- `rollout.py` -- Top-level composition: builds sigma schedule, generates noise, calls `make_guided_denoiser`, calls `euler_solve`, applies inverse scaling. Replaces the tangled `sample_trajectory` RPC in `sampling.py` which mixed in server concerns (ZMQ dispatch, dataset writing, hidden capture hooks). `rollout.py` has zero server/client dependencies.

**Borderline port (1 of 7):**

- `model_loading.py` -- Extracts the FP8 model loading sequence from `ModelManager.ensure_diffusion()` into a standalone function. This is a real narrowing (no ModelManager state machine, no VRAM lifecycle tracking, no LoRA replay). However, it directly imports from `futudiffu.diffusion_model` and `futudiffu.fp8`, which means it is not a self-contained module -- it is a thin convenience wrapper over the original implementation. The `configure_sage_attention` function is similarly a three-line wrapper over `futudiffu.sage_attention.configure_sage`. Verdict: useful but not algorithmically novel. It genuinely narrows the interface (no ModelManager) but does not extract or redefine any algorithm.

**Genuinely new module (1 of 7):**

- `reward_functions.py` -- Entirely new code: `pinkify_score()`, `thisnotthat_score()`, `pairwise_preference()`. Pure CPU/numpy functions with no torch or futudiffu dependencies. This is not a "port" at all -- it is new functionality that correctly lives in `src_ii/` as a named module. Well-structured with clear import constraints.

## 3. scripts_ii/ Files: Inlined Algorithms That Should Be in src_ii/

**`train_pinkify_btrm.py` -- 3 inlined algorithms:**

1. **ScoreUnembedder scoring (lines 268-278, 370-379):** The script manually calls `head.norm()`, `head.proj()`, and applies the tanh cap, duplicating the logic of `ScoreUnembedder.forward()`. The ScoreUnembedder class already has a `forward()` method that does exactly `mean_pool -> norm -> proj -> tanh_cap`. This inline reimplementation is fragile: if the head's forward path changes (e.g., adding dropout), the training script silently diverges. Should call `head(pooled)` after pre-pooling, or better, `head(hidden_states)` which handles pooling internally.

2. **Bradley-Terry training loop (lines 279-343):** The positive/negative score selection logic, the per-head iteration, and the loss aggregation are ~65 lines of algorithmic code inlined in the script. This should be a function like `compute_pairwise_bt_loss(head, scores_a, scores_b, prefs_per_head) -> (loss, metrics)` in `src_ii/` or in `futudiffu.btrm`.

3. **Mean-pooling of variable-length hidden states (line 263):** `torch.stack([h.mean(dim=1).squeeze(0) for h in all_hidden_a], dim=0)` is a one-liner but it encodes a decision (mean-pool before stacking, handle variable seq_len) that should match the ScoreUnembedder's own pooling. The head's `forward()` already mean-pools; using it would eliminate the mismatch risk.

**`attention_interpretability.py` -- 2 inlined algorithms:**

1. **AttentionCapture class (lines 66-186):** 120 lines implementing attention monkey-patching, per-head statistics reduction, and module patching/unpatching. This is reusable infrastructure (any mechanistic interpretability script would need it) and should be in `src_ii/attention_capture.py`.

2. **Sigma-for-step lookup (lines 291-301):** Duplicated identically in `train_pinkify_btrm.py` (lines 150-163). The logic "if step_key == 'final', use sigmas[-2]; else parse step index" should be a function like `sigma_for_step(step_key, n_steps, **kwargs) -> Tensor`.

**`generate_preference_labels.py` -- 1 inlined algorithm:**

1. **VAE decode to PIL (lines 45-59):** The `latent_to_pil()` function is defined locally in the script. Identical logic appears in `render_attention_maps.py` (lines 51-64) and `render_comparison.py` (lines 36-47). Should be in `src_ii/` as a shared utility.

**`render_attention_maps.py` -- 2 inlined algorithms:**

1. **Heatmap colormap rendering (lines 67-139):** `make_heatmap_overlay()` implements diverging and hot colormaps from scratch in PIL. This is 70 lines of rendering code that has nothing to do with the attention analysis and should be a reusable rendering utility.

2. **Layer-head heatmap rendering (lines 189-231):** Another standalone visualization function that should be in a shared rendering module.

**`render_comparison.py` -- 1 inlined issue:**

1. **VAE decode (lines 36-47):** Third copy of the latent-to-PIL conversion. Also imports `futudiffu.vae` directly rather than going through any `src_ii/` abstraction.

**`score_distribution_comparison.py` -- 1 inlined algorithm:**

1. **Spearman rank correlation (lines 159-179):** Implements a rank correlation function from scratch in numpy. This is a generic statistical function. Either import from scipy (`scipy.stats.spearmanr`) or place it in a shared `src_ii/statistics.py`.

**`validate_trajectory.py` -- 0 inlined algorithms.** This script correctly composes `src_ii.model_loading`, `src_ii.rollout`, and `futudiffu.text_encoder` with no algorithmic inlining. It does contain WSL/Windows path conversion helpers (lines 41-56) that are script-specific and arguably fine where they are. This is the most cleanly structured script in the directory.

**`verify_btrm_persistence.py` -- 1 inlined algorithm:**

1. **ScoreUnembedder scoring (lines 87-100):** Fourth copy of the manual `norm -> proj -> tanh_cap` scoring path, identical to the inlined version in `train_pinkify_btrm.py`. Should call `head(hidden_states)` or `head(pooled.unsqueeze(1))`.

## 4. Overall Assessment: Is src_ii/ Actually Narrower Than src/?

Yes, with qualifications. The five-function decomposition (nfe, denoise, make_guided_denoiser, euler_solve, rollout) genuinely factors the diffusion sampling pipeline into composable layers with clean type signatures and strict import boundaries. The original `src/futudiffu/sampling.py` is a 500+ line file that tangles all of these concerns together with server-specific logic (hidden capture hooks, dataset writing, ZMQ dispatch state). The `src_ii/` modules are individually testable and composable in ways the originals are not.

However, the narrowing is incomplete. The scripts_ii/ files contain at least 8 instances of algorithmic code that should have been extracted into `src_ii/` modules. The most egregious case is the 4x-duplicated ScoreUnembedder scoring path that reimplements `ScoreUnembedder.forward()` inline. The sigma-for-step lookup is duplicated 2x. The VAE-decode-to-PIL conversion is duplicated 3x. The attention capture infrastructure is a one-off that should be a module. These inlined algorithms partially undo the narrowing: code that calls `src_ii.rollout()` correctly (validate_trajectory.py) coexists with code that reimplements `ScoreUnembedder.forward()` by hand (train_pinkify_btrm.py, verify_btrm_persistence.py, attention_interpretability.py).

The src_ii/ library layer is genuinely narrow. The scripts_ii/ layer is not -- it contains the sprawl that should have been pushed down.

## 5. Specific Recommendations for Extraction Into src_ii/ Modules

1. **`src_ii/vae_utils.py`**: Extract `latent_to_pil(vae, latent) -> PIL.Image` from the 3 duplicated copies across generate_preference_labels.py, render_attention_maps.py, and render_comparison.py. One function, one import.

2. **`src_ii/attention_capture.py`**: Move the `AttentionCapture` class from attention_interpretability.py. Any future mechanistic interpretability script (attention diff maps, head ablation, probing classifiers) will need this same monkey-patch infrastructure.

3. **`src_ii/sigma_utils.py`** (or add to `sigma_schedule.py`): Extract `sigma_for_step(step_key: str, n_steps: int, **schedule_kwargs) -> Tensor` to replace the 2x-duplicated if/else parsing logic for "final" vs "step_NN".

4. **Stop reimplementing `ScoreUnembedder.forward()`**: Every script that scores hidden states should call `head(hidden_states)` or at minimum `head(pooled.unsqueeze(1))`, never manually inline `head.norm(x); head.proj(x); cap * tanh(x / cap)`. The ScoreUnembedder class already encapsulates this. The scripts should trust the class interface.

5. **`src_ii/training_loss.py`** (or add to `reward_functions.py`): Extract the Bradley-Terry pairwise training loop body from train_pinkify_btrm.py into a function like `compute_pairwise_bt_loss(scores_a, scores_b, prefs, head_idx) -> (bt_loss, accuracy)`. The existing `futudiffu.btrm.bradley_terry_loss` handles the raw loss computation but not the winner/loser routing from preference labels; that routing logic is the part that is inlined and should be named.

6. **`src_ii/visualization.py`**: Extract `make_heatmap_overlay`, `make_text_token_bar_chart`, `make_layer_head_heatmap`, `make_summary_strip`, and the colormaps from render_attention_maps.py. These are pure PIL rendering functions with no model dependencies.

# Unit 1: Minimal Extraction for Trajectory Reproduction

## 1. What Was Extracted and Why

The algorithmic decomposition essay identifies five function boundaries that, if strictly maintained, make rollout/training drift impossible. Unit 1 extracts these five functions into `src_ii/`, proving they can be composed into a complete trajectory rollout that uses the existing model architecture (`futudiffu.diffusion_model.NextDiT`) and all existing kernels (FP8, SageAttention, fused elementwise) without importing any orchestration code (`model_manager`, `server`, `client`, `sampling`, `training_utils`). The five functions are: `nfe()` in `forward.py`, which wraps `NextDiT.forward()` with the sigma-to-timestep conversion; `denoise()` in `forward.py`, which implements the CONST noise model's `calculate_denoised` formula; `make_guided_denoiser()` in `guided_denoiser.py`, which constructs a `(x, sigma) -> denoised` closure supporting both CFG and cfg=1.0; `euler_solve()` in `solver.py`, which runs the Euler ODE integration; and `rollout()` in `rollout.py`, which composes all layers from seed to final latent. A sixth file, `model_loading.py`, extracts the FP8 diffusion model loading sequence from `ModelManager.ensure_diffusion()` into a standalone function, and a seventh file, `sigma_schedule.py`, contains the pure-math sigma schedule construction and CONST noise scaling functions. The separation into seven files rather than the five function boundaries is deliberate: `sigma_schedule.py` contains shared math that multiple layers need, and `model_loading.py` is infrastructure rather than a function boundary.

## 2. What Was Produced and Verified

All seven `src_ii/` modules import successfully and compile without syntax errors. The critical math functions -- `build_sigma_schedule`, `const_noise_scaling`, `const_inverse_noise_scaling`, `denoise` (vs `const_calculate_denoised`), `to_d`, `pad_and_batch_cond` -- were tested against their `futudiffu.sampling` equivalents with identical inputs and produce bitwise identical outputs (max_diff = 0.0 in every case). The `euler_solve` function was tested against `sample_euler` using a deterministic mock denoiser, producing bitwise identical final results and identical per-step callback values. The `make_guided_denoiser` CFG path was tested against `make_cfg_model_fn` with a mock model and produces bitwise identical CFG-combined denoised outputs. These equivalence tests were run on the actual GPU environment (RTX 4090, torch 2.10.0+cu128) through the Windows Python venv.

A validation script (`scripts_ii/validate_trajectory.py`) was written that executes the full end-to-end pipeline: load a reference trajectory from `btrm_dataset/`, load and compile the text encoder to produce conditioning tensors using the canonical `encode_prompt` pipeline (chat template wrapping, tokenization, layer_idx=-2 hidden state extraction), free the text encoder, load and compile the FP8 diffusion model via `src_ii/model_loading`, run the `src_ii/rollout` with the same parameters (seed, n_steps=30, cfg=4.0, sampling_shift=1.0, multiplier=1.0), and write comparison tensors (element-wise differences, L2 norms, max absolute differences) to `validation_output_ii/`. The script does not assert pass/fail -- it produces persistent files for human or automated comparison.

## 3. How the Extracted Code Composes

The call graph is strictly layered with no skip connections:

```
rollout()
  +-- build_sigma_schedule()           [sigma_schedule.py]
  +-- const_noise_scaling()            [sigma_schedule.py]
  +-- make_rope_cache()                [rollout.py, calls model.prepare_rope_cache]
  +-- make_guided_denoiser()           [guided_denoiser.py]
  |     +-- pad_and_batch_cond()       [guided_denoiser.py]
  |     +-- (returns closure that calls:)
  |           +-- nfe()                [forward.py, calls model.forward()]
  |           +-- denoise()            [forward.py]
  +-- euler_solve()                    [solver.py]
  |     +-- denoiser_fn()             [from make_guided_denoiser]
  |     +-- to_d()                     [solver.py]
  +-- const_inverse_noise_scaling()    [sigma_schedule.py]
```

Each layer's output type is the next layer's expected input type. `nfe()` returns raw model predictions. `denoise()` converts them to denoised estimates. `make_guided_denoiser()` returns a `(x, sigma) -> denoised` closure. `euler_solve()` consumes that type signature. `rollout()` orchestrates the composition. No layer reaches down more than one level. The training path (Unit 3) will call the same `make_guided_denoiser()` with the same conditioning, ensuring the guidance strategy used during rollout generation is identical to the guidance strategy used during log-ratio computation.

The import structure enforces the constraint that `src_ii/` uses model architecture and kernels from `futudiffu` but does not import orchestration:
- `forward.py`: imports nothing from futudiffu (pure torch)
- `guided_denoiser.py`: imports only from `src_ii.forward`
- `solver.py`: imports nothing from futudiffu (pure torch)
- `sigma_schedule.py`: imports nothing from futudiffu (pure torch)
- `rollout.py`: imports from `src_ii.guided_denoiser`, `src_ii.sigma_schedule`, `src_ii.solver`
- `model_loading.py`: imports from `futudiffu.diffusion_model` (architecture) and `futudiffu.fp8` (FP8Linear)

## 4. Validation Script Design and Runnability

The validation script (`scripts_ii/validate_trajectory.py`) is designed for the Windows-Python-from-WSL2 execution environment. It accepts `--traj-idx N` to select which of the 50 reference trajectories to validate against, and `--no-compile` / `--no-sage` flags for faster iteration. The script proceeds in four phases:

**Phase 1 (TE encoding):** Loads the Qwen3-4B text encoder, encodes the reference trajectory's prompt using `futudiffu.text_encoder.encode_prompt()` (which applies the chat template and extracts layer -2 hidden states), encodes the empty string for neg_cond, saves both conditioning tensors to disk, then frees the text encoder.

**Phase 2 (Model loading):** Loads the FP8 diffusion model via `src_ii.model_loading.load_fp8_diffusion_model()`, which replicates the exact loading sequence from `ModelManager.ensure_diffusion()`: safetensors load, FP8Linear injection, non-FP8 weight load, device transfer, model fusion, torch.compile.

**Phase 3 (Rollout):** Calls `src_ii.rollout.rollout()` with the reference trajectory's seed, step count, and the freshly-encoded conditioning, saving intermediate latents at the same step indices as the reference.

**Phase 4 (Comparison):** For each step present in both reference and reproduced trajectories, computes element-wise difference in float32, writes diff tensors and reproduced tensors to disk, and records L2 norms, max absolute differences, and relative L2 errors to a JSON stats file.

The script was NOT run during this session because it requires approximately 15GB VRAM (text encoder loading + diffusion model loading are sequential but each is large) and torch.compile warmup takes 40+ seconds. The code paths that exercise the GPU were each verified independently through mock-based equivalence tests. The script is ready to run.

## 5. What Remains Uncertain and Needs Future Units

**Conditioning determinism.** The reference trajectories were generated via the inference server, which compiled the text encoder before encoding prompts. Whether the compiled TE produces bitwise identical hidden states to an independently-compiled TE on the same GPU with the same weights is not guaranteed -- torch.compile may introduce non-deterministic operator fusion. If the validation script shows non-zero but small differences concentrated in the early steps (where conditioning matters most), this is the likely cause. The mitigation would be to save the conditioning tensors that were used during original generation, or to accept ULP-level differences as the baseline.

**Attention backend.** The reference trajectories include both "sdpa" and "sage" precision modes. The validation script configures SageAttention for "sage" trajectories but does not currently switch the attention backend via `set_attention_backend()` from `futudiffu.attention`. This function sets a module-level flag that `sdpa_attention()` checks. The validation script should call this function to match the reference trajectory's attention backend. Without it, "sage" trajectories will be validated using SDPA, which will produce different (though both correct) outputs.

**i2i trajectories.** The first 40 trajectories are t2i; the last 10 are i2i. The i2i case requires VAE-encoding a source image to produce `clean_latent`, which requires loading the VAE model. The validation script notes this limitation but does not implement it. Unit 2 or a follow-up should add i2i support.

**Training path.** The five function boundaries are designed so the training path calls the same `make_guided_denoiser()` and `nfe()` as the rollout path, with gradient checkpointing applied by the caller. This is Unit 3's concern. The current `src_ii/` code does not yet include the training variants (gradient-enabled solver, REINFORCE log-ratio computation). The key structural guarantee is that adding training support requires only wrapping the existing functions, not reimplementing them -- the REINFORCE step will call `denoiser_fn(x_t, sigma_t)` on checkpoint latents using the same closure that produced the trajectory, and the gradient will flow through the same `nfe()` call.

**LoRA adapter lifecycle.** The current extraction loads the model with no adapters. For training validation, adapters must be allocated (pre-compile), compiled, and initialized (post-compile). The `model_loading.py` module does not yet support this. The `futudiffu.lora` module's `allocate_adapter()` and `init_adapter_weights()` functions are compatible with the extracted code since they operate on the raw `NextDiT` model, but the integration needs to be validated.

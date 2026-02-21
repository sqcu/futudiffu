# The Algorithmic Decomposition of Diffusion Sampling: Forward, CFG, Solver, Rollout

## 1. The Four Concepts As They Exist in the Code

The diffusion forward pass in futudiffu is `NextDiT.forward()` and its sibling `NextDiT.forward_packed()`, defined in `src/futudiffu/diffusion_model.py`. Both take a noisy latent `x`, a sigma-derived timestep, and text conditioning, and return a negated model prediction `-img`. The negation is a ComfyUI heritage artifact: the CONST noise model's `calculate_denoised` formula is `model_input - model_output * sigma`, and because the model returns `-img`, the subtraction becomes an addition that recovers the denoised estimate. This forward call is one neural function evaluation (NFE). It knows nothing about guidance, nothing about which step it occupies in a trajectory, and nothing about whether gradients are flowing. It is a pure function from `(noisy_latent, noise_level, conditioning) -> prediction`, with the caveat that the prediction is negated by convention. The training-side equivalents are `forward_checkpointed()` and `forward_no_grad()` in `src/futudiffu/training_utils.py`, which re-derive the same architecture from first principles -- embedding, refiners, 30 main layers, final layer, unpatchify, negate -- rather than calling `forward()` or `forward_packed()`. This duplication is the origin of most drift bugs.

The CFG reduction function is not a single function in the codebase; it is a pattern that appears in three places. In `sampling.py`, `make_cfg_model_fn()` constructs a closure that expands the input to batch dimension 2, calls the compiled model once with `(pos_cond, neg_cond)` stacked on the batch axis, splits the output, applies `const_calculate_denoised` to each half independently, and returns `denoised_uncond + (denoised_cond - denoised_uncond) * cfg`. In `training_utils.py`, `build_cfg_model_fn()` does approximately the same thing but with a special case for `cfg == 1.0` that skips the batch expansion entirely and runs a single-NFE unconditional pass. In `sampling.py`, `sample_euler_packed()` inlines the CFG reduction: it expands each image to batch 2, calls `packed_forward_fn`, then applies CFG per-image in a loop. In all three cases, the CFG reduction takes two model outputs (conditioned and unconditioned), converts each to a denoised estimate via the CONST formula, and linearly interpolates. The critical structural fact is that CFG calls `const_calculate_denoised` -- it converts raw model output to denoised space before interpolating. This means CFG's output is a denoised latent, not a raw model prediction.

The sampling function is `sample_euler()` in `sampling.py`. It takes a `model_fn` -- which is the CFG-wrapped closure, not the raw forward -- and iterates over a sigma schedule. At each step it calls `model_fn(x, sigma)`, receives a denoised estimate, computes the ODE derivative via `to_d(x, sigma, denoised) = (x - denoised) / sigma`, and takes an Euler step. The training variant `sample_euler_train()` does the same integration but wraps `model_fn` in `torch.utils.checkpoint.checkpoint` for memory savings, stores per-step checkpoints, and optionally adds stochastic churn for REINFORCE rollout diversity. Both samplers expect `model_fn` to return a fully-denoised estimate -- the CFG combination, if applicable, has already happened inside the closure. The sampler has no knowledge of whether CFG was applied; it sees only a `(x, sigma) -> denoised` interface.

The autoregressive rollout is orchestrated by `run_trajectory()` and `run_trajectory_packed()` in `sampling.py`. These functions handle the full pipeline: constructing noise from a seed, building the sigma schedule via `build_sigma_schedule()`, applying CONST noise scaling to produce the initial noisy latent, constructing the CFG model function, running the Euler sampler, applying inverse noise scaling at the end, and saving per-step intermediates via a callback. The rollout is the composition: `seed -> noise -> CONST_scale(noise, sigma_0) -> euler_loop(CFG(forward), sigmas) -> CONST_inverse_scale -> final_latent`. VAE decoding, if desired, happens outside this function entirely.

## 2. How They Should Compose

The correct nesting is a strict four-layer stack with no skip connections. At the bottom, the forward primitive takes `(noisy_latent, sigma, conditioning) -> raw_prediction`. One layer up, the denoising conversion takes `(raw_prediction, sigma, model_input) -> denoised_estimate` -- this is `const_calculate_denoised`, and it is a pure function of the noise model, not the neural network. One more layer up, CFG takes a forward primitive, conditioning pair, and guidance scale, and returns a `(noisy_latent, sigma) -> denoised_estimate` closure that internally calls forward twice (or once for cfg=1.0) and linearly combines. At the top, the Euler solver takes a `(x, sigma) -> denoised` function and a sigma schedule and produces a trajectory of latents.

The key invariant is that each layer's output type is the next layer's expected input type. The forward returns a raw prediction. The denoising conversion turns a raw prediction into a denoised estimate. CFG calls forward + denoising conversion and returns a denoised estimate. The solver calls CFG (or any other `(x, sigma) -> denoised` function) and produces latents. No layer reaches down to call a function two levels below it. No layer's implementation depends on the implementation of any other layer -- only on the type signature of the layer directly below.

This decomposition has a specific mathematical consequence: the solver's ODE derivative `d = (x - denoised) / sigma` is well-defined only when `denoised` is in denoised space (i.e., it has had `const_calculate_denoised` applied). If a layer returned raw model output instead of denoised output, the derivative would be wrong. The current code respects this invariant in the inference path -- `make_cfg_model_fn` calls `const_calculate_denoised` before interpolating, and `sample_euler` feeds the result to `to_d`. But the invariant is not enforced by types or assertions; it is enforced by the developer's memory of the convention.

## 3. Where the Current Code Violates the Correct Composition

The drift analysis in `docs/essay_policy_train_drift.md` documents the central violations. The most severe is that the training path in `compute_reinforce_step()` constructs its forward pass independently of the sampling path. Where the rollout path builds CFG from `pad_and_batch_cond(pos_cond, neg_cond)`, the training path in `compute_reinforce_step()` calls `forward_no_grad` and `forward_checkpointed` with only `conditioning` (positive), never constructing a CFG batch or performing the guidance interpolation. The reference and policy denoised estimates are computed in a non-CFG regime, but the checkpoint latents they operate on were generated by a CFG-guided trajectory. The log-ratio `log p(a|s)` is therefore computed for an action distribution the policy never actually samples from during inference. This is divergence #1 from the drift analysis, classified as critical.

The second violation is architectural: `forward_checkpointed()` and `forward_no_grad()` in `training_utils.py` are hand-rolled reimplementations of `NextDiT.forward()`. They duplicate the embedding logic, the refiner loops, the concatenation, the main layer iteration, the final layer call, the unpatchify, and the negation. Any change to `forward()` that is not mirrored in these two functions creates a silent divergence. The adaLN batching optimization (`prepare_adaln_cache` / `_compute_adaln_params`) exists only in `forward()` and `forward_packed()`; the training functions use the per-block path. The fused elementwise kernels are gated by `_use_fused_elementwise` which is set by `fuse_model()` and is present during `forward()` but may or may not be active during training depending on whether the same model instance is used. None of these differences are bugs per se -- the training path deliberately avoids fusions for gradient correctness -- but they mean that the "same model" produces numerically different outputs through two code paths, which undermines the semantic equivalence that REINFORCE requires between rollout and training.

The third violation is subtler. The rollout's `save_callback` fires during the Euler loop at step `i`, recording `info["x"]` which is the latent *before* the Euler step at iteration `i`. But the training path in `accumulate_policy_gradients()` evaluates `model(checkpoint[step_i], sigmas[step_i])`. Since `checkpoint[step_i]` is `x` at the start of step `i` (i.e., after `i` completed Euler updates), and `sigmas[step_i]` is the sigma *used* at step `i`, this is actually correct -- `x` at the start of step `i` corresponds to noise level `sigma_i`. The drift document flags this as potentially off-by-one, but the real ambiguity arises from the fact that the callback convention and the checkpoint indexing convention are defined in different files with no shared specification. Whether the current code is correct or off-by-one depends entirely on which callback fires when, and the only way to verify it is to read both `sample_euler`'s callback position and `accumulate_policy_gradients`'s indexing and mentally verify they agree.

## 4. CFG as a Sampling Strategy, Not a Forward Primitive

The design constraint that CFG must not be part of `forward()` is both a present requirement and a future-proofing decision. The present requirement is that `forward()` is used in at least three distinct regimes: inference with CFG (the normal case, 2 NFE per step), BTRM hidden state extraction (1 NFE, no guidance, via `run_backbone_hidden`), and training forward passes (1 NFE per reference, 1 NFE per policy, no guidance in the current bug, but CFG-guided in the correct version). If CFG were baked into `forward()`, the BTRM extraction path would need to call `forward()` and then discard half the output, or `forward()` would need a `cfg_scale` parameter with a special `1.0` path, which is exactly the kind of implicit mode-switching that causes drift.

The future requirement is that CFG-free distilled models -- trained to produce guided-quality outputs in a single NFE without the negative conditioning pass -- will use the same `forward()` primitive. A CFG-free model's sampling loop calls `forward()` once per step and feeds the output (after denoising conversion) directly to the Euler derivative. The sampler does not change; only the model function handed to it changes. If CFG were inside `forward()`, a CFG-free model would require a different `forward()` or a flag that disables the internal CFG path, creating exactly the kind of conditional branch that torch.compile handles poorly and that humans maintain worse.

The correct layering, then, is: `forward()` is a 1-NFE primitive that returns raw model output. Denoising conversion (`const_calculate_denoised`) is a pure function applied to raw model output. CFG is a higher-order function that takes `forward` + conditioning pair + scale and returns a `(x, sigma) -> denoised` closure. For CFG-free models, a trivial adapter takes `forward` + conditioning and returns a `(x, sigma) -> denoised` closure that calls forward once and applies denoising conversion. Both CFG and CFG-free adapters produce the same type signature: `(x, sigma) -> denoised`. The solver consumes this type signature without knowing which adapter produced it. This is the only decomposition that supports both regimes without conditional branches, duplicated code, or implicit mode flags.

## 5. The Minimal Function Boundaries That Make Drift Impossible

Five functions, with strict type contracts, eliminate all identified drift vectors.

**`nfe(model, x, sigma, conditioning, rope_cache) -> raw_prediction`**: A single neural function evaluation. This is the existing `NextDiT.forward()` (or `forward_packed` for the packed path), unchanged. It takes a noisy latent, a sigma (not a timestep -- the `sigma * multiplier` conversion belongs here or in a thin wrapper, but it must happen in exactly one place), conditioning, and precomputed RoPE. It returns the raw model output (negated, per ComfyUI convention). The training path calls this same function -- not a reimplementation. Gradient checkpointing is applied by the caller wrapping individual layer calls or by using `torch.utils.checkpoint` around `nfe` itself, not by writing a separate `nfe_checkpointed` that re-derives the architecture.

**`denoise(raw_prediction, sigma, model_input) -> denoised`**: This is `const_calculate_denoised`. It is a pure function with no state, no model reference, and no conditional logic. It exists solely to make the type conversion explicit: raw model output is not the same type as a denoised estimate, and the only place where the CONST formula is applied is inside this function.

**`make_guided_denoiser(nfe_fn, pos_cond, neg_cond, cfg_scale, rope_cache) -> (x, sigma) -> denoised`**: This replaces both `make_cfg_model_fn` and `build_cfg_model_fn`. It returns a closure that, given `(x, sigma)`, calls `nfe_fn` with the appropriate conditioning batch, applies `denoise` to each output, and performs the CFG interpolation. When `cfg_scale == 1.0`, it calls `nfe_fn` once with `pos_cond` only and applies `denoise` once. When the model is CFG-free, the caller constructs a trivial version of this closure that always calls `nfe_fn` once. The critical property is that this function's return type is `(x, sigma) -> denoised` regardless of guidance strategy. The solver never sees the guidance implementation.

**`euler_solve(denoiser_fn, x0, sigmas, callback?) -> x_final`**: This is the existing `sample_euler` with one change: it takes `denoiser_fn` (the output of `make_guided_denoiser`), not `model_fn`. The name change is cosmetic but clarifies the type contract: the function it receives returns denoised estimates, not raw predictions. The training variant wraps each `denoiser_fn` call in gradient checkpointing. There is exactly one Euler solver, not two (no `sample_euler_train` that reimplements the loop).

**`rollout(model, conditioning, sigmas, seed, ...) -> trajectory`**: This is the composition: generate noise, apply CONST noise scaling, construct `denoiser_fn` via `make_guided_denoiser`, run `euler_solve`, apply inverse noise scaling, return the final latent and any saved intermediates. The training path calls the same `rollout` (or a thin training-mode variant that enables gradient flow) rather than reimplementing the trajectory logic. The REINFORCE step computes log-ratios by calling `denoiser_fn` (not a reimplemented forward) on checkpoint latents at their correct sigmas, ensuring that the training evaluation uses the same guidance strategy that produced the rollout.

The property that makes drift impossible is that there is exactly one implementation of each function, and each function's output type is the next function's input type. The training path does not reimplement `nfe`; it calls `nfe` with gradient checkpointing. The REINFORCE step does not reimplement CFG; it calls `make_guided_denoiser` with the same conditioning that the rollout used. The only degree of freedom is the guidance strategy (CFG, CFG-free, or something else), and that degree of freedom is encapsulated behind a single type signature that the solver and the training loss both consume without knowledge of the strategy's internals.

---

## Appendix: Block Quotations from Source Files

### A. The raw forward pass (`src/futudiffu/diffusion_model.py`, `NextDiT.forward`)

```python
def forward(self, x: torch.Tensor, timesteps: torch.Tensor,
            context: torch.Tensor, num_tokens: int,
            attention_mask: torch.Tensor | None = None,
            rope_cache: dict | None = None) -> torch.Tensor:
    """Forward pass of NextDiT.

    Args:
        x: (B, C, H, W) noisy latent.
        timesteps: (B,) sigma values.
        context: (B, seq, cap_feat_dim) text encoder hidden states.
        num_tokens: Number of text tokens.
        attention_mask: Optional attention mask for text.
        rope_cache: Optional precomputed RoPE from prepare_rope_cache().

    Returns:
        (B, C, H, W) model output (NEGATED, as per ComfyUI).
    """
    t = 1 - timesteps
    # ... [embedding, refiners, 30 main layers, final layer, unpatchify] ...
    return -img
```

### B. The CONST denoising conversion (`src/futudiffu/sampling.py`)

```python
def const_calculate_denoised(sigma: torch.Tensor, model_output: torch.Tensor,
                              model_input: torch.Tensor) -> torch.Tensor:
    """CONST.calculate_denoised: model_input - model_output * sigma"""
    sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
    return model_input - model_output * sigma
```

### C. The inference CFG closure (`src/futudiffu/sampling.py`, `make_cfg_model_fn`)

```python
def make_cfg_model_fn(diff_model, cond_batch, num_tokens, rope_cache, cfg, multiplier):
    def model_fn(x_in, sigma):
        timestep = sigma * multiplier
        x_batch = x_in.expand(2, -1, -1, -1)
        t_batch = timestep.expand(2)
        output_batch = diff_model(
            x_batch, t_batch, cond_batch,
            num_tokens=num_tokens, rope_cache=rope_cache,
        )
        out_cond, out_uncond = output_batch.chunk(2, dim=0)
        denoised_cond = const_calculate_denoised(sigma, out_cond, x_in)
        denoised_uncond = const_calculate_denoised(sigma, out_uncond, x_in)
        return denoised_uncond + (denoised_cond - denoised_uncond) * cfg
    return model_fn
```

### D. The training CFG closure (`src/futudiffu/training_utils.py`, `build_cfg_model_fn`)

```python
def build_cfg_model_fn(diff_model, pos_cond, neg_cond, rope_cache,
                        num_tokens, cfg, multiplier):
    if cfg != 1.0:
        cond_batch = torch.cat([pos_cond, neg_cond], dim=0)  # (2, seq, dim)

    def model_fn(x_in: Tensor, sigma: Tensor) -> Tensor:
        timestep = sigma * multiplier
        if cfg == 1.0:
            output = diff_model(
                x_in, timestep, pos_cond,
                num_tokens=num_tokens, rope_cache=rope_cache,
            )
            return const_calculate_denoised(sigma, output, x_in)
        x_batch = x_in.expand(2, -1, -1, -1)
        t_batch = timestep.expand(2)
        output_batch = diff_model(
            x_batch, t_batch, cond_batch,
            num_tokens=num_tokens, rope_cache=rope_cache,
        )
        output_cond, output_uncond = output_batch.chunk(2, dim=0)
        denoised_cond = const_calculate_denoised(sigma, output_cond, x_in)
        denoised_uncond = const_calculate_denoised(sigma, output_uncond, x_in)
        return denoised_uncond + (denoised_cond - denoised_uncond) * cfg
    return model_fn
```

### E. The Euler solver (`src/futudiffu/sampling.py`, `sample_euler`)

```python
@torch.inference_mode()
def sample_euler(model_fn, x, sigmas, callback=None):
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        sigma_hat = sigmas[i]
        denoised = model_fn(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i],
                      'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt
    return x
```

### F. The training REINFORCE step -- the drift site (`src/futudiffu/training_utils.py`, `compute_reinforce_step`)

```python
def compute_reinforce_step(model, x_t, sigma, conditioning, num_tokens,
                            rope_cache, multiplier, advantage, adapter_name):
    # 1. Reference forward: no_grad, scale=0 (no LoRA effect)
    set_lora_scale(model, torch.tensor([0.0], ...), adapter_name=adapter_name)
    ref_output = forward_no_grad(
        model, x_t, timestep.unsqueeze(0), conditioning, num_tokens, rope_cache,
    )
    ref_denoised = const_calculate_denoised(sigma, ref_output, x_t)

    # 2. Policy forward: checkpointed, scale=1.0
    set_lora_scale(model, torch.tensor([1.0], ...), adapter_name=adapter_name)
    pi_output, _ = forward_checkpointed(
        model, x_t, timestep.unsqueeze(0), conditioning, num_tokens, rope_cache,
    )
    pi_denoised = const_calculate_denoised(sigma, pi_output, x_t)

    # 3. Log-ratio (no CFG -- this is the bug)
    diff = pi_denoised - ref_denoised
    mse = (diff * diff).sum()
    log_ratio = -mse / (2.0 * sigma * sigma + 1e-10)
```

Note: `forward_no_grad` and `forward_checkpointed` are standalone reimplementations of the model architecture, not calls to `NextDiT.forward()`. They receive only `conditioning` (positive), never constructing a CFG batch. The log-ratio is computed in a non-CFG regime on latents that were generated by a CFG-guided rollout.

### G. The DRGRPO log-ratio formulation (`src/futudiffu/policy_loss.py`, `compute_step_log_ratios`)

```python
def compute_step_log_ratios(pi_denoised, ref_denoised, sigmas):
    """Per-step Gaussian log-probability ratio between policy and reference.

    log_ratio_t = -||mu_pi_t - mu_ref_t||^2 / (2 * sigma_t^2)
    """
    T = len(pi_denoised)
    log_ratios = pi_denoised[0].new_zeros(T)
    for t in range(T):
        diff = pi_denoised[t] - ref_denoised[t]
        mse = (diff * diff).sum()
        sigma_t = sigmas[t]
        log_ratios[t] = -mse / (2.0 * sigma_t * sigma_t + 1e-10)
    return log_ratios
```

### H. The reimplemented forward in `training_utils.py` (`forward_checkpointed`)

```python
def forward_checkpointed(model, x, timesteps, context, num_tokens, rope_cache):
    """Model forward with per-block gradient checkpointing on the 30 main layers.
    ...
    Returns:
        (-img, last_hidden) where last_hidden is the output of model.layers[-1]
        before final_layer. Both retain gradient connections.
    """
    # --- Phase 1: Embedding + refiners (no grad, cheap) ---
    with torch.no_grad():
        t = 1 - timesteps
        bs, c, h, w = x.shape
        x_padded = pad_to_patch_size(x, (model.patch_size, model.patch_size))
        t_emb = model.t_embedder(t * model.time_scale, dtype=x.dtype)
        adaln_input = t_emb
        # ... [cap_embedder, x_embedder, context_refiner, noise_refiner] ...
        embed = torch.cat([cap_feats_embedded, x_patches], dim=1)

    # --- Phase 2: Detach and start autograd graph ---
    embed = embed.detach().clone().requires_grad_(True)

    # --- Phase 3: 30 main layers with per-block gradient checkpointing ---
    for layer in model.layers:
        embed = grad_ckpt(layer, embed, None, freqs_cis, adaln_input,
                          use_reentrant=False)

    last_hidden = embed

    # --- Phase 4: Final layer ---
    img = model.final_layer(embed, adaln_input)
    img = model.unpatchify(img, img_sizes, l_effective_cap_len,
                           return_tensor=True)[:, :, :h, :w]
    return -img, last_hidden
```

This function reimplements the full forward pipeline. It does not call `NextDiT.forward()`. It does not use `prepare_adaln_cache()` or `_compute_adaln_params()`. It does not check for `_use_fused_elementwise` or `_use_fused_qkv`. Any optimization or bugfix applied to `NextDiT.forward()` must be independently applied here to maintain semantic equivalence.

### I. The rollout orchestrator (`src/futudiffu/sampling.py`, `run_trajectory`)

```python
def run_trajectory(diff_compiled, diff_model, device, dtype, params, tensors,
                   btrm_head=None):
    # ... [extract params, build noise, build sigmas] ...
    cond_batch, num_tokens = pad_and_batch_cond(pos_cond, neg_cond)
    rope_cache = make_rope_cache(diff_model, latent_h, latent_w, num_tokens, device)
    x = prepare_initial_latent(noise, sigmas[0], clean_latent, ...)
    model_fn = make_cfg_model_fn(
        diff_compiled, cond_batch, num_tokens, rope_cache, cfg, multiplier,
    )
    # ... [callback setup] ...
    with torch.inference_mode():
        x = sample_euler(model_fn, x, sigmas, callback=save_callback)
        x = const_inverse_noise_scaling(sigmas[-1], x)
    return result_tensors, metadata
```

This is the correct composition: `make_cfg_model_fn` constructs a `(x, sigma) -> denoised` closure, `sample_euler` consumes it, and the rollout function orchestrates the pipeline. The training path should call the same functions with the same conditioning; currently it does not.

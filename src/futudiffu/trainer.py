"""Unified training module for DRGRPO policy optimization + BTRM training.

Consolidates training logic from train_qat.py, smoke_test_btrm_policy.py,
and smoke_test_btrm_v2.py into a single canonical module.

Architecture:
  TrainConfig -- all hyperparameters
  TrainingState -- model, head, optimizer, caches
  setup_training() -- loads models, injects LoRA, builds caches
  train_btrm_epoch() -- one epoch of BTRM dual-head training
  rollout_group() -- K rollouts with BTRM scoring
  policy_step() -- single DRGRPO update with reference anchoring
  train_loop() -- outer loop: BTRM phase then policy phase
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .btrm import BTRMHead, bradley_terry_loss, compute_multihead_loss, logsquare_regularizer
from .lora import (
    freeze_adapter,
    get_lora_params,
    inject_lora,
    lora_state_dict,
    load_lora_state_dict,
    set_lora_scale,
    clear_lora_scale,
)
from .policy_loss import (
    compute_group_advantages,
    compute_step_log_ratios,
    drgrpo_diffusion_loss,
    reference_anchor_loss,
)
from .training_utils import (
    HiddenCapture,
    build_cfg_model_fn,
    forward_checkpointed,
    log_section,
    prepare_conditioning,
    prepare_latent_state,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LASER_SHARK_PROMPT = (
    'ahem.\n*ting ting ting ting ting*\n'
    'the query model for this is a LARGE LANGUAGE MODEL, specifically QWEN-3-4B, '
    'a GENERAL PURPOSE SEMANTIC PARSER which is able to WRITE SENTENCES AT A TIME '
    'when they are participating in dialogue. however, in this situation, they are '
    'being used as a hidden state generator to steer an *image generation model*, '
    'z-image.\n\nqwen-3-4b, draw me an "enormous laser shark for the sega saturn".'
)


@dataclass
class TrainConfig:
    """Configuration for unified training."""

    # Model paths
    diffusion_model_path: str = ""
    text_encoder_path: str = ""
    vae_path: str = ""
    tokenizer_path: str | None = None

    # Image params
    width: int = 1280
    height: int = 832
    steps: int = 30
    cfg: float = 4.0
    sampling_shift: float = 1.0
    multiplier: float = 1.0

    # Prompt
    prompt: str = LASER_SHARK_PROMPT
    negative_prompt: str = ""

    # LoRA config
    lora_rank: int = 8
    lora_alpha: float = 16.0
    rtheta_layer_indices: set[int] = field(default_factory=lambda: {28, 29})

    # BTRM config
    btrm_lr: float = 1e-3
    btrm_epochs: int = 3
    btrm_logsq_weight: float = 0.1
    btrm_head_names: tuple[str, ...] = ("bit_quality", "step_quality")

    # DRGRPO policy config
    policy_lr: float = 1e-4
    policy_iterations: int = 20
    grad_clip: float = 1.0
    group_size: int = 4
    sparse_steps: int = 5
    s_churn: float = 0.0

    # DRGRPO loss params
    clip_low: float = 0.2
    clip_high: float = 0.28
    lambda_ent: float = 0.01
    lambda_anchor: float = 1e-4

    # Sage attention config
    sage_smooth_k: bool = True
    sage_qk_quant: str = "fp8"
    sage_pv_quant: str = "bf16"

    # Device/dtype
    device: str = "cuda"
    dtype: str = "bfloat16"
    fp8_diffusion: bool = True
    fp8_block_size: int = 128

    # Output
    save_dir: str = "training_output"

    # Server-based training
    use_server: bool = False
    server_endpoint: str = "tcp://localhost:5555"


def _get_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[name]


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------

@dataclass
class TrainingState:
    """Mutable training state."""
    model: nn.Module
    btrm_head: BTRMHead
    pos_cond: Tensor
    neg_cond: Tensor
    num_tokens: int
    rope_cache: dict
    sigmas: Tensor
    latent_h: int
    latent_w: int
    device: torch.device
    dtype: torch.dtype
    iteration: int = 0


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_training(config: TrainConfig) -> TrainingState:
    """Load models, encode prompt, precompute caches.

    Returns a TrainingState ready for train_btrm_epoch / policy_step.
    The diffusion model is loaded but LoRA is NOT injected yet --
    that happens in the BTRM or policy training phases.
    """
    device = torch.device(config.device)
    dtype = _get_dtype(config.dtype)

    # --- Text encoder: encode then free ---
    print("[setup] Loading text encoder...")
    from .text_encoder import create_tokenizer, encode_prompt, load_text_encoder

    tokenizer = create_tokenizer(config.tokenizer_path)
    te_model = load_text_encoder(config.text_encoder_path, device=device, dtype=dtype)
    te_model = torch.compile(te_model, mode="default")

    print("[setup] Encoding prompts...")
    pos_cond = encode_prompt(te_model, tokenizer, config.prompt, device=device)
    neg_cond = encode_prompt(te_model, tokenizer, config.negative_prompt, device=device)
    print(f"  pos: {pos_cond.shape}, neg: {neg_cond.shape}")

    del te_model, tokenizer
    torch.cuda.empty_cache()

    # Clone to escape inference_mode tensors
    pos_cond = pos_cond.clone()
    neg_cond = neg_cond.clone()

    # Pad conditioning
    pos_cond, neg_cond, num_tokens = prepare_conditioning(pos_cond, neg_cond)

    # --- Diffusion model ---
    print("[setup] Loading diffusion model...")
    if config.fp8_diffusion:
        from .diffusion_model import (
            _detect_cap_feat_dim,
            _detect_n_layers,
            _detect_qk_norm,
            _strip_diffusion_prefix,
            create_diffusion_model,
        )
        from .fp8 import replace_linear_with_fp8
        from safetensors.torch import load_file

        diff_sd = load_file(config.diffusion_model_path, device=str(device))
        remapped = _strip_diffusion_prefix(diff_sd)
        del diff_sd

        n_layers = _detect_n_layers(remapped.keys())
        cap_feat_dim = _detect_cap_feat_dim(remapped)
        qk_norm = _detect_qk_norm(remapped.keys())
        model = create_diffusion_model(
            dtype=dtype, n_layers=n_layers,
            cap_feat_dim=cap_feat_dim, qk_norm=qk_norm,
        )
        replace_linear_with_fp8(
            model, remapped, block_size=config.fp8_block_size,
            output_dtype=dtype,
        )

        remaining = {k: v for k, v in remapped.items()
                     if not k.endswith((".weight_scale", ".comfy_quant"))}
        model.load_state_dict(remaining, strict=False, assign=True)
        del remapped, remaining
        model = model.to(device)
    else:
        from .diffusion_model import load_diffusion_model
        model = load_diffusion_model(
            config.diffusion_model_path, device=device, dtype=dtype,
        )

    model.eval()

    # --- Precompute caches ---
    rope_cache, sigmas, latent_h, latent_w = prepare_latent_state(
        model, config.width, config.height, num_tokens, device, dtype,
        sampling_shift=config.sampling_shift, multiplier=config.multiplier,
        steps=config.steps,
    )

    # --- BTRM head ---
    btrm_head = BTRMHead(
        hidden_dim=model.dim,
        head_names=config.btrm_head_names,
        logit_cap=10.0,
    ).to(device=device, dtype=dtype)

    mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[setup] Complete. Peak GPU: {mem:.1f} GB")
    print(f"  Sigmas: {sigmas.shape} [{sigmas[0]:.4f}..{sigmas[-1]:.4f}]")

    return TrainingState(
        model=model,
        btrm_head=btrm_head,
        pos_cond=pos_cond,
        neg_cond=neg_cond,
        num_tokens=num_tokens,
        rope_cache=rope_cache,
        sigmas=sigmas,
        latent_h=latent_h,
        latent_w=latent_w,
        device=device,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# BTRM training
# ---------------------------------------------------------------------------

def train_btrm_epoch(
    state: TrainingState,
    config: TrainConfig,
    optimizer: torch.optim.Optimizer,
    pairs: list[tuple],
) -> list[dict]:
    """One epoch of BTRM dual-head training using checkpointed forward.

    Each pair is (x_pos, x_neg, sigma_val, head_idx) where:
      x_pos, x_neg are (1, 16, H, W) latent tensors
      sigma_val is the sigma at which these were captured
      head_idx is 0 for bit_quality, 1 for step_quality

    Returns list of per-step metric dicts.
    """
    log = []
    model = state.model
    btrm = state.btrm_head

    for i, (x_pos, x_neg, sigma_val, head_idx) in enumerate(pairs):
        t0 = time.time()

        x_pos = x_pos.to(device=state.device, dtype=state.dtype)
        x_neg = x_neg.to(device=state.device, dtype=state.dtype)

        sigma = torch.tensor([sigma_val], device=state.device, dtype=state.dtype)
        timestep = sigma * config.multiplier

        # Forward with gradient checkpointing
        _, hidden_pos = forward_checkpointed(
            model, x_pos, timestep, state.pos_cond,
            num_tokens=state.num_tokens, rope_cache=state.rope_cache,
        )
        _, hidden_neg = forward_checkpointed(
            model, x_neg, timestep, state.pos_cond,
            num_tokens=state.num_tokens, rope_cache=state.rope_cache,
        )

        # Score
        pos_scores = btrm(hidden_pos)  # (1, n_heads)
        neg_scores = btrm(hidden_neg)  # (1, n_heads)

        # Multi-head loss
        pos_head_indices = torch.tensor([head_idx], device=state.device)
        result = compute_multihead_loss(
            pos_scores,
            {"soft_neg": neg_scores},
            pos_head_indices,
            config.btrm_head_names,
            logsquare_weight=config.btrm_logsq_weight,
        )
        loss = result["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dt = time.time() - t0
        margin = (pos_scores[:, head_idx] - neg_scores[:, head_idx]).detach().item()

        entry = dict(
            step=i, sigma=sigma_val, loss=loss.item(),
            bt_loss=result["bt_loss"].item() if isinstance(result["bt_loss"], Tensor) else result["bt_loss"],
            margin=margin, head_idx=head_idx, time=dt,
        )
        log.append(entry)

        if i % 5 == 0:
            print(
                f"  btrm step {i:3d} | sigma={sigma_val:.4f} | "
                f"loss={loss.item():.4f} margin={margin:+.4f} | "
                f"head={config.btrm_head_names[head_idx]} | {dt:.1f}s"
            )

        del x_pos, x_neg, hidden_pos, hidden_neg, pos_scores, neg_scores
        torch.cuda.empty_cache()

    return log


# ---------------------------------------------------------------------------
# Rollout group
# ---------------------------------------------------------------------------

@dataclass
class RolloutResult:
    """Result of a single rollout."""
    seed: int
    final_latent: Tensor       # (1, 16, H, W) on CPU
    reward: float              # BTRM score
    denoised_steps: list[Tensor] | None = None  # per-step denoised if collected


def rollout_group(
    state: TrainingState,
    config: TrainConfig,
    base_seed: int,
    model_fn,
    collect_denoised: bool = False,
) -> list[RolloutResult]:
    """Run K rollouts with BTRM scoring.

    Uses the provided model_fn (which may include LoRA via CFG wrapper).

    Args:
        state: Current training state.
        config: Training config.
        base_seed: Base RNG seed (each rollout uses base_seed + k).
        model_fn: Callable(x, sigma) -> denoised.
        collect_denoised: If True, store per-step denoised for DRGRPO.

    Returns:
        List of K RolloutResults.
    """
    from .sampling import (
        const_inverse_noise_scaling,
        const_noise_scaling,
        sample_euler_train,
    )

    K = config.group_size
    results = []
    capture = HiddenCapture(state.model)
    capture.install()

    latent_h, latent_w = state.latent_h, state.latent_w

    for k in range(K):
        seed = base_seed + k
        generator = torch.Generator(device=state.device).manual_seed(seed)
        noise = torch.randn(
            1, 16, latent_h, latent_w,
            dtype=state.dtype, generator=generator, device=state.device,
        )
        latent = torch.zeros(1, 16, latent_h, latent_w,
                             device=state.device, dtype=state.dtype)
        x = const_noise_scaling(state.sigmas[0], noise, latent)

        with torch.inference_mode():
            x_out, checkpoints = sample_euler_train(
                model_fn, x, state.sigmas, s_churn=config.s_churn,
                return_denoised=collect_denoised,
            )
            x_out = const_inverse_noise_scaling(state.sigmas[-1], x_out)

        # Score with BTRM via forward hook
        # Use a mid-schedule sigma for scoring
        sigma_mid = state.sigmas[len(state.sigmas) // 2]
        timestep_mid = (sigma_mid * config.multiplier).unsqueeze(0)
        x_for_score = sigma_mid * noise  # noisy latent at mid sigma
        with torch.no_grad():
            state.model(
                x_for_score, timestep_mid, state.pos_cond,
                num_tokens=state.num_tokens, rope_cache=state.rope_cache,
            )
        hidden = capture.get()
        scores = state.btrm_head(hidden)
        reward = scores[:, 0].item()  # bit_quality

        # Handle denoised collection
        denoised_list = None
        if collect_denoised and isinstance(checkpoints, tuple):
            # sample_euler_train returns (x, checkpoints, denoised_list)
            _, _, denoised_list_raw = x_out, checkpoints[0], checkpoints[1]
            denoised_list = [d.detach().cpu() for d in denoised_list_raw]

        results.append(RolloutResult(
            seed=seed,
            final_latent=x_out.detach().cpu(),
            reward=reward,
            denoised_steps=denoised_list,
        ))

        del noise, latent, x, x_out, checkpoints, x_for_score, hidden
        torch.cuda.empty_cache()

    capture.remove()
    return results


# ---------------------------------------------------------------------------
# Policy step
# ---------------------------------------------------------------------------

def policy_step(
    state: TrainingState,
    config: TrainConfig,
    optimizer: torch.optim.Optimizer,
    ptheta_params: list[nn.Parameter],
) -> dict:
    """Single DRGRPO policy update with reference anchoring.

    Runs one forward pass through a random sigma with gradient checkpointing.
    Uses concurrent batch: batch[0] = LoRA (pi_theta), batch[1] = scale=0 (ref).

    Returns dict of metrics.
    """
    t0 = time.time()

    model = state.model
    btrm = state.btrm_head

    # Pick a random sigma from the schedule
    n_sigmas = len(state.sigmas) - 1
    sigma_idx = random.randint(0, n_sigmas - 1)
    sigma = state.sigmas[sigma_idx]
    timestep = (sigma * config.multiplier).unsqueeze(0)

    # Random noise
    noise = torch.randn(1, 16, state.latent_h, state.latent_w,
                         dtype=state.dtype, device=state.device)
    x_t = sigma * noise

    # Reference anchoring: batch[0] = pi (LoRA active), batch[1] = ref (LoRA off)
    set_lora_scale(model, torch.tensor([1.0, 0.0], device=state.device, dtype=state.dtype),
                   adapter_name="ptheta")

    x_batch = x_t.expand(2, -1, -1, -1)
    timestep_batch = timestep.expand(2)
    pos_batch = state.pos_cond.expand(2, -1, -1)

    model_output, last_hidden = forward_checkpointed(
        model, x_batch, timestep_batch, pos_batch,
        num_tokens=state.num_tokens, rope_cache=state.rope_cache,
    )

    pi_output, ref_output = model_output.chunk(2, dim=0)
    pi_hidden, ref_hidden = last_hidden.chunk(2, dim=0)

    # BTRM reward from policy hidden states
    pi_scores = btrm(pi_hidden)
    reward_bq = pi_scores[:, 0].mean()

    # Policy loss: maximize BTRM reward
    policy_loss = -reward_bq

    # Reference anchor
    anchor_loss = reference_anchor_loss(pi_output, ref_output)

    # Total loss
    loss = policy_loss + config.lambda_anchor * anchor_loss

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(ptheta_params, config.grad_clip)
    optimizer.step()

    # Restore LoRA scale to 1.0 for inference
    clear_lora_scale(model, adapter_name="ptheta")

    dt = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 1024**3

    n_with_grad = sum(1 for p in ptheta_params
                      if p.grad is not None and p.grad.abs().sum() > 0)

    entry = dict(
        iteration=state.iteration,
        sigma=sigma.item(),
        reward_bq=reward_bq.item(),
        policy_loss=policy_loss.item(),
        anchor_loss=anchor_loss.item(),
        loss=loss.item(),
        grad_norm=grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm,
        n_with_grad=n_with_grad,
        n_total=len(ptheta_params),
        mem_gb=mem,
        time=dt,
    )

    state.iteration += 1

    del noise, x_t, x_batch, model_output, last_hidden
    del pi_output, ref_output, pi_hidden, ref_hidden, pi_scores
    torch.cuda.empty_cache()

    return entry


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train_loop(state: TrainingState, config: TrainConfig) -> dict:
    """Full training: BTRM phase then policy phase.

    Returns dict with "btrm_log" and "policy_log".
    """
    from .attention import set_attention_backend
    from .sage_attention import configure_sage

    log_section("Training: BTRM Phase + Policy Phase")

    # ---------------------------------------------------------------
    # Phase 1: BTRM training (R_theta)
    # ---------------------------------------------------------------
    log_section("Phase 1: R_theta (BTRM Head + LoRA layers 28-29)")

    set_attention_backend("sdpa")

    # Inject rtheta LoRA on last 2 layers
    rtheta_injected = inject_lora(
        state.model, name="rtheta",
        rank=config.lora_rank, alpha=config.lora_alpha,
        layer_indices=config.rtheta_layer_indices,
    )
    rtheta_params = list(get_lora_params(state.model, adapter_name="rtheta"))
    n_rtheta = sum(p.numel() for p in rtheta_params)
    print(f"  rtheta LoRA: {len(rtheta_injected)} adapters, "
          f"{n_rtheta:,} params ({n_rtheta*2/1024**2:.1f} MB)")

    all_btrm_params = rtheta_params + list(state.btrm_head.parameters())
    btrm_optimizer = torch.optim.AdamW(all_btrm_params, lr=config.btrm_lr)

    # Generate probe pairs across sigma schedule
    n_sigmas = len(state.sigmas) - 1
    probe_indices = [int(i * (n_sigmas - 1) / 4) for i in range(5)]
    print(f"  Probe sigmas: {[(i, state.sigmas[i].item()) for i in probe_indices]}")

    btrm_log = []
    for epoch in range(config.btrm_epochs):
        pairs = []
        for sigma_idx in probe_indices:
            sigma_val = state.sigmas[sigma_idx].item()
            noise = torch.randn(1, 16, state.latent_h, state.latent_w,
                                dtype=state.dtype, device=state.device)
            x_t = state.sigmas[sigma_idx] * noise

            # Positive: SDPA hidden
            set_attention_backend("sdpa")
            with torch.no_grad():
                _, hidden_pos = forward_checkpointed(
                    state.model, x_t, (state.sigmas[sigma_idx] * config.multiplier).unsqueeze(0),
                    state.pos_cond, state.num_tokens, state.rope_cache,
                )
            # Negative: Sage hidden
            set_attention_backend("sage")
            configure_sage(smooth_k=config.sage_smooth_k,
                           qk_quant=config.sage_qk_quant,
                           pv_quant=config.sage_pv_quant)
            with torch.no_grad():
                _, hidden_neg = forward_checkpointed(
                    state.model, x_t, (state.sigmas[sigma_idx] * config.multiplier).unsqueeze(0),
                    state.pos_cond, state.num_tokens, state.rope_cache,
                )

            # Score and train
            pos_scores = state.btrm_head(hidden_pos)
            neg_scores = state.btrm_head(hidden_neg)
            bt_loss = bradley_terry_loss(pos_scores[:, 0], neg_scores[:, 0])
            logsq = logsquare_regularizer(pos_scores[:, 0])
            loss = bt_loss + config.btrm_logsq_weight * logsq

            btrm_optimizer.zero_grad()
            loss.backward()
            btrm_optimizer.step()

            margin = (pos_scores[:, 0] - neg_scores[:, 0]).detach().item()
            btrm_log.append(dict(
                epoch=epoch, sigma=sigma_val, loss=loss.item(),
                margin=margin,
            ))
            print(f"  ep{epoch} sigma={sigma_val:.4f} loss={loss.item():.4f} "
                  f"margin={margin:+.4f}")

            del noise, x_t, hidden_pos, hidden_neg
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Phase 2: Policy optimization (p_theta)
    # ---------------------------------------------------------------
    log_section("Phase 2: p_theta (DRGRPO Policy Optimization)")

    # Freeze rtheta
    freeze_adapter(state.model, "rtheta")
    set_lora_scale(state.model, torch.tensor([0.0], device=state.device, dtype=state.dtype),
                   adapter_name="rtheta")

    # Freeze BTRM head
    state.btrm_head.eval()
    for p in state.btrm_head.parameters():
        p.requires_grad_(False)

    # Stack ptheta on ALL layers
    ptheta_injected = inject_lora(
        state.model, name="ptheta",
        rank=config.lora_rank, alpha=config.lora_alpha,
    )
    ptheta_params = list(get_lora_params(state.model, adapter_name="ptheta"))
    n_ptheta = sum(p.numel() for p in ptheta_params)
    print(f"  ptheta LoRA: {len(ptheta_injected)} adapters, "
          f"{n_ptheta:,} params ({n_ptheta*2/1024**2:.1f} MB)")

    policy_optimizer = torch.optim.AdamW(ptheta_params, lr=config.policy_lr)

    # Switch to Sage for policy training
    set_attention_backend("sage")
    configure_sage(smooth_k=config.sage_smooth_k,
                   qk_quant=config.sage_qk_quant,
                   pv_quant=config.sage_pv_quant)

    policy_log = []
    state.iteration = 0

    for i in range(config.policy_iterations):
        entry = policy_step(state, config, policy_optimizer, ptheta_params)
        policy_log.append(entry)

        print(
            f"  iter {entry['iteration']:3d} | sigma={entry['sigma']:.4f} | "
            f"reward={entry['reward_bq']:+.4f} | loss={entry['loss']:.4f} | "
            f"anchor={entry['anchor_loss']:.4e} | "
            f"grad_norm={entry['grad_norm']:.3e} | "
            f"grads: {entry['n_with_grad']}/{entry['n_total']} | "
            f"mem={entry['mem_gb']:.1f}GB | {entry['time']:.1f}s"
        )

    # Save
    import os
    os.makedirs(config.save_dir, exist_ok=True)

    ptheta_sd = lora_state_dict(state.model, adapter_name="ptheta")
    ptheta_path = os.path.join(config.save_dir, "ptheta_lora.safetensors")
    from safetensors.torch import save_file
    save_file({k: v.cpu() for k, v in ptheta_sd.items()}, ptheta_path)
    print(f"\n  Saved ptheta ({len(ptheta_sd)} tensors) to {ptheta_path}")

    rtheta_sd = lora_state_dict(state.model, adapter_name="rtheta")
    rtheta_sd.update({f"btrm.{k}": v.cpu()
                      for k, v in state.btrm_head.state_dict().items()})
    rtheta_path = os.path.join(config.save_dir, "rtheta_weights.pt")
    torch.save(rtheta_sd, rtheta_path)
    print(f"  Saved rtheta+btrm ({len(rtheta_sd)} tensors) to {rtheta_path}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    log_section("Training Summary")

    print("BTRM margins:")
    for e in btrm_log:
        print(f"  ep{e['epoch']} sigma={e['sigma']:.4f} margin={e['margin']:+.4f}")

    print(f"\nPolicy rewards: {[f'{e['reward_bq']:+.4f}' for e in policy_log]}")

    all_ok = not any(math.isnan(e['loss']) for e in btrm_log + policy_log)
    print(f"\nResult: {'PASS (no NaN)' if all_ok else 'FAIL (NaN detected)'}")

    return {"btrm_log": btrm_log, "policy_log": policy_log, "ok": all_ok}

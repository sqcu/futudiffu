"""Shared training utilities extracted from smoke tests.

HiddenCapture, CFG model builder, conditioning prep, latent state prep,
gradient-checkpointed forward, and logging helpers.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_ckpt


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def log_param_stats(
    label: str,
    params: OrderedDict[str, nn.Parameter],
    prev_snapshot: dict[str, Tensor] | None = None,
) -> None:
    """Print gradient norms and weight deltas for a set of named parameters."""
    for pname, param in params.items():
        grad_norm = param.grad.norm().item() if param.grad is not None else 0.0
        w_norm = param.data.norm().item()
        w_max = param.data.abs().max().item()

        delta = 0.0
        if prev_snapshot is not None and pname in prev_snapshot:
            delta = (param.data - prev_snapshot[pname]).norm().item()

        flags = ""
        if torch.isnan(param.data).any():
            flags = " *** NaN WEIGHT ***"
        elif param.grad is not None and torch.isnan(param.grad).any():
            flags = " *** NaN GRAD ***"
        elif param.grad is not None and param.grad.abs().sum().item() == 0:
            flags = " [zero grad]"

        print(
            f"    {pname:55s} | grad={grad_norm:.3e} "
            f"w_norm={w_norm:.3e} w_max={w_max:.3e} delta={delta:.3e}{flags}"
        )


def snapshot_params(params: OrderedDict[str, nn.Parameter]) -> dict[str, Tensor]:
    """Clone param values for delta computation."""
    return {name: p.data.clone() for name, p in params.items()}


# ---------------------------------------------------------------------------
# Hidden state capture via forward hook
# ---------------------------------------------------------------------------

class HiddenCapture:
    """Hook on the last transformer block to capture hidden states."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.captured: Tensor | None = None
        self._handle = None

    def install(self) -> None:
        if self._handle is not None:
            return
        last_block = self.model.layers[-1]

        def hook(_module, _input, output):
            self.captured = output

        self._handle = last_block.register_forward_hook(hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def get(self) -> Tensor:
        """Return captured tensor (still in compute graph) and clear storage."""
        h = self.captured
        self.captured = None
        assert h is not None, "Hook did not fire -- check model.layers"
        return h


# ---------------------------------------------------------------------------
# Conditioning preparation
# ---------------------------------------------------------------------------

def prepare_conditioning(
    pos_cond: Tensor,
    neg_cond: Tensor,
) -> tuple[Tensor, Tensor, int]:
    """Pad positive and negative conditioning to same length for batched CFG.

    Args:
        pos_cond: (1, pos_len, dim) positive conditioning.
        neg_cond: (1, neg_len, dim) negative conditioning.

    Returns:
        (padded_pos, padded_neg, num_tokens) where both have shape (1, max_len, dim).
    """
    pos_len = pos_cond.shape[1]
    neg_len = neg_cond.shape[1]
    max_len = max(pos_len, neg_len)
    if pos_len < max_len:
        pos_cond = F.pad(pos_cond, (0, 0, 0, max_len - pos_len))
    if neg_len < max_len:
        neg_cond = F.pad(neg_cond, (0, 0, 0, max_len - neg_len))
    return pos_cond, neg_cond, max_len


# ---------------------------------------------------------------------------
# CFG model_fn builder
# ---------------------------------------------------------------------------

def build_cfg_model_fn(
    diff_model: nn.Module,
    pos_cond: Tensor,
    neg_cond: Tensor,
    rope_cache: dict,
    num_tokens: int,
    cfg: float,
    multiplier: float,
):
    """Build a batched CFG model_fn for the euler loop.

    Returns model_fn(x, sigma) -> denoised.
    """
    from .sampling import const_calculate_denoised

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


# ---------------------------------------------------------------------------
# Latent state preparation
# ---------------------------------------------------------------------------

def prepare_latent_state(
    model: nn.Module,
    width: int,
    height: int,
    num_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    sampling_shift: float = 1.0,
    multiplier: float = 1.0,
    steps: int = 30,
) -> tuple[dict, Tensor, int, int]:
    """Precompute RoPE cache and sigma schedule.

    Args:
        model: Diffusion model with patch_size and prepare_rope_cache.
        width: Image width.
        height: Image height.
        num_tokens: Number of text tokens.
        device: Target device.
        dtype: Target dtype.
        sampling_shift: Sigma schedule shift.
        multiplier: Sigma schedule multiplier.
        steps: Number of euler steps.

    Returns:
        (rope_cache, sigmas, latent_h, latent_w)
    """
    from .sampling import build_sigmas, simple_scheduler

    latent_h = height // 8
    latent_w = width // 8
    padded_h = latent_h + ((-latent_h) % model.patch_size)
    padded_w = latent_w + ((-latent_w) % model.patch_size)

    rope_cache = model.prepare_rope_cache(padded_h, padded_w, num_tokens, device)
    # Clone to escape any inference_mode context
    rope_cache = {k: v.clone() if isinstance(v, Tensor) else v
                  for k, v in rope_cache.items()}

    sigma_table = build_sigmas(shift=sampling_shift, multiplier=multiplier * 1000)
    sigmas = simple_scheduler(sigma_table, steps)
    sigmas = sigmas.to(device=device, dtype=dtype)

    return rope_cache, sigmas, latent_h, latent_w


# ---------------------------------------------------------------------------
# Gradient-checkpointed model forward
# ---------------------------------------------------------------------------

def forward_checkpointed(
    model: nn.Module,
    x: Tensor,
    timesteps: Tensor,
    context: Tensor,
    num_tokens: int,
    rope_cache: dict,
) -> tuple[Tensor, Tensor]:
    """Model forward with per-block gradient checkpointing on the 30 main layers.

    Embedding + refiners run in no_grad (no trainable params there).
    Each main layer is individually checkpointed.

    Returns:
        (-img, last_hidden) where last_hidden is the output of model.layers[-1]
        before final_layer. Both retain gradient connections.
    """
    from .diffusion_model import pad_to_patch_size, pad_zimage

    # --- Phase 1: Embedding + refiners (no grad, cheap) ---
    with torch.no_grad():
        t = 1 - timesteps
        bs, c, h, w = x.shape
        x_padded = pad_to_patch_size(x, (model.patch_size, model.patch_size))

        t_emb = model.t_embedder(t * model.time_scale, dtype=x.dtype)
        adaln_input = t_emb

        bsz = x_padded.shape[0]
        pH = pW = model.patch_size

        cap_feats_embedded = model.cap_embedder(context)
        if model.pad_tokens_multiple is not None:
            cap_feats_embedded, _ = pad_zimage(
                cap_feats_embedded, model.cap_pad_token, model.pad_tokens_multiple
            )

        B, C, H, W = x_padded.shape
        x_patches = model.x_embedder(
            x_padded.view(B, C, H // pH, pH, W // pW, pW)
            .permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2)
        )
        if model.pad_tokens_multiple is not None:
            x_patches, _ = pad_zimage(
                x_patches, model.x_pad_token, model.pad_tokens_multiple
            )

        img_len = x_patches.shape[1]

        cap_freqs_cis = rope_cache['cap_freqs_cis']
        x_freqs_cis = rope_cache['x_freqs_cis']
        freqs_cis = rope_cache['freqs_cis']
        if bsz > 1 and cap_freqs_cis.shape[0] == 1:
            cap_freqs_cis = cap_freqs_cis.expand(bsz, -1, -1, -1, -1, -1)
            x_freqs_cis = x_freqs_cis.expand(bsz, -1, -1, -1, -1, -1)
            freqs_cis = freqs_cis.expand(bsz, -1, -1, -1, -1, -1)

        for layer in model.context_refiner:
            cap_feats_embedded = layer(cap_feats_embedded, None, cap_freqs_cis)
        for layer in model.noise_refiner:
            x_patches = layer(x_patches, None, x_freqs_cis, adaln_input)

        embed = torch.cat([cap_feats_embedded, x_patches], dim=1)
        l_effective_cap_len = [embed.shape[1] - img_len] * bsz
        img_sizes = [(H, W)] * bsz

    # --- Phase 2: Detach and start autograd graph ---
    embed = embed.detach().clone().requires_grad_(True)
    adaln_input = adaln_input.detach().clone()
    freqs_cis = freqs_cis.detach().clone()

    # --- Phase 3: 30 main layers with per-block gradient checkpointing ---
    for layer in model.layers:
        embed = grad_ckpt(
            layer, embed, None, freqs_cis, adaln_input,
            use_reentrant=False,
        )

    last_hidden = embed

    # --- Phase 4: Final layer ---
    img = model.final_layer(embed, adaln_input)
    img = model.unpatchify(
        img, img_sizes, l_effective_cap_len, return_tensor=True
    )[:, :, :h, :w]

    return -img, last_hidden


def forward_no_grad(
    model: nn.Module,
    x: Tensor,
    timesteps: Tensor,
    context: Tensor,
    num_tokens: int,
    rope_cache: dict,
) -> Tensor:
    """Model forward under torch.no_grad() with no checkpointing.

    Same architecture as forward_checkpointed but optimized for the reference
    pass in REINFORCE: no gradient graph, no checkpoint storage, no
    last_hidden capture.

    Returns:
        -img output only.
    """
    from .diffusion_model import pad_to_patch_size, pad_zimage

    with torch.no_grad():
        t = 1 - timesteps
        bs, c, h, w = x.shape
        x_padded = pad_to_patch_size(x, (model.patch_size, model.patch_size))

        t_emb = model.t_embedder(t * model.time_scale, dtype=x.dtype)
        adaln_input = t_emb

        bsz = x_padded.shape[0]
        pH = pW = model.patch_size

        cap_feats_embedded = model.cap_embedder(context)
        if model.pad_tokens_multiple is not None:
            cap_feats_embedded, _ = pad_zimage(
                cap_feats_embedded, model.cap_pad_token, model.pad_tokens_multiple
            )

        B, C, H, W = x_padded.shape
        x_patches = model.x_embedder(
            x_padded.view(B, C, H // pH, pH, W // pW, pW)
            .permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2)
        )
        if model.pad_tokens_multiple is not None:
            x_patches, _ = pad_zimage(
                x_patches, model.x_pad_token, model.pad_tokens_multiple
            )

        img_len = x_patches.shape[1]

        cap_freqs_cis = rope_cache['cap_freqs_cis']
        x_freqs_cis = rope_cache['x_freqs_cis']
        freqs_cis = rope_cache['freqs_cis']
        if bsz > 1 and cap_freqs_cis.shape[0] == 1:
            cap_freqs_cis = cap_freqs_cis.expand(bsz, -1, -1, -1, -1, -1)
            x_freqs_cis = x_freqs_cis.expand(bsz, -1, -1, -1, -1, -1)
            freqs_cis = freqs_cis.expand(bsz, -1, -1, -1, -1, -1)

        for layer in model.context_refiner:
            cap_feats_embedded = layer(cap_feats_embedded, None, cap_freqs_cis)
        for layer in model.noise_refiner:
            x_patches = layer(x_patches, None, x_freqs_cis, adaln_input)

        embed = torch.cat([cap_feats_embedded, x_patches], dim=1)
        l_effective_cap_len = [embed.shape[1] - img_len] * bsz
        img_sizes = [(H, W)] * bsz

        # Main layers (no checkpointing, no grad)
        for layer in model.layers:
            embed = layer(embed, None, freqs_cis, adaln_input)

        # Final layer
        img = model.final_layer(embed, adaln_input)
        img = model.unpatchify(
            img, img_sizes, l_effective_cap_len, return_tensor=True
        )[:, :, :h, :w]

    return -img


# ---------------------------------------------------------------------------
# REINFORCE log-ratio computation (per-step)
# ---------------------------------------------------------------------------

def compute_reinforce_step(
    model: nn.Module,
    x_t: Tensor,
    sigma: Tensor,
    conditioning: Tensor,
    num_tokens: int,
    rope_cache: dict,
    multiplier: float,
    advantage: float,
    adapter_name: str,
) -> float:
    """Compute REINFORCE log-ratio for one step and backward into model params.

    Runs two separate B=1 passes instead of one concurrent B=2 batch:
      1. Reference forward: no_grad, scale=0 (no LoRA, no grad graph)
      2. Policy forward: checkpointed, scale=1.0 (full grad)

    This halves backward FLOPs and checkpoint memory vs the old B=2 approach
    since backward only recomputes 30 layers for B=1 (policy) instead of B=2.

    Args:
        model: Diffusion model with LoRA adapters.
        x_t: (1, C, H, W) checkpoint latent at this step.
        sigma: Scalar sigma value for this step.
        conditioning: (1, seq, dim) text conditioning (positive only).
        num_tokens: Number of text tokens.
        rope_cache: Pre-computed RoPE cache.
        multiplier: Timestep multiplier.
        advantage: Advantage weight for this rollout.
        adapter_name: Which LoRA adapter to differentiate.

    Returns:
        Log-ratio value (detached float) for this step.
    """
    from .lora import set_lora_scale
    from .sampling import const_calculate_denoised

    device = x_t.device
    dtype = x_t.dtype
    timestep = sigma * multiplier

    # 1. Reference forward: no_grad, scale=0 (no LoRA effect)
    set_lora_scale(
        model, torch.tensor([0.0], device=device, dtype=dtype),
        adapter_name=adapter_name,
    )
    ref_output = forward_no_grad(
        model, x_t, timestep.unsqueeze(0), conditioning, num_tokens, rope_cache,
    )
    ref_denoised = const_calculate_denoised(sigma, ref_output, x_t)

    # 2. Policy forward: checkpointed, scale=1.0
    set_lora_scale(
        model, torch.tensor([1.0], device=device, dtype=dtype),
        adapter_name=adapter_name,
    )
    pi_output, _ = forward_checkpointed(
        model, x_t, timestep.unsqueeze(0), conditioning, num_tokens, rope_cache,
    )
    pi_denoised = const_calculate_denoised(sigma, pi_output, x_t)

    # 3. Loss (ref_denoised has no grad from the no_grad block)
    diff = pi_denoised - ref_denoised
    mse = (diff * diff).sum()
    log_ratio = -mse / (2.0 * sigma * sigma + 1e-10)

    step_loss = -advantage * log_ratio
    step_loss.backward()

    return log_ratio.detach().item()


# ---------------------------------------------------------------------------
# Server-side training orchestration (extracted from server.py handlers)
# ---------------------------------------------------------------------------

def run_backbone_hidden(diff_model, latent, sigma, conditioning, device, dtype,
                        multiplier=1.0, requires_grad=False):
    """Run frozen backbone and return hidden states from last transformer block.

    Args:
        diff_model: Raw (uncompiled) diffusion model.
        latent: (B, 16, H, W) noisy latent.
        sigma: (B,) sigma values.
        conditioning: (B, seq, dim) text conditioning.
        device: CUDA device.
        dtype: Working dtype.
        multiplier: Timestep multiplier.
        requires_grad: If True, use no_grad (allows downstream grad graphs).
            If False, use inference_mode for max performance.

    Returns:
        Hidden states (B, N_tokens, hidden_dim) on GPU.
    """
    from .sampling import make_rope_cache

    latent = latent.to(device=device, dtype=dtype)
    sigma = sigma.to(device=device, dtype=dtype)
    conditioning = conditioning.to(device=device, dtype=dtype)

    timestep = sigma * multiplier
    num_tokens = conditioning.shape[1]

    B, C, H, W = latent.shape
    rope_cache = make_rope_cache(diff_model, H, W, num_tokens, device)

    capture = HiddenCapture(diff_model)
    capture.install()
    try:
        ctx = torch.no_grad() if requires_grad else torch.inference_mode()
        with ctx:
            diff_model(
                latent, timestep, conditioning,
                num_tokens=num_tokens, rope_cache=rope_cache,
            )
    finally:
        capture.remove()

    return capture.get()


def train_btrm_step(diff_model, btrm_head, btrm_optimizer, device, dtype,
                    params, tensors):
    """Complete BTRM training step: forward N examples, BT loss, backward, step.

    Args:
        diff_model: Raw diffusion model (for backbone hidden capture).
        btrm_head: BTRMHead module (trainable).
        btrm_optimizer: Optimizer for btrm_head parameters.
        device: CUDA device.
        dtype: Working dtype.
        params: RPC params (labels, logsquare_weight, multiplier).
        tensors: RPC tensors (latent_N, sigma_N, conditioning_N).

    Returns:
        Metadata dict with loss, bt_loss, logsq_loss, per_head_accuracy, etc.
    """
    from .btrm import compute_labeled_btrm_loss

    labels = params["labels"]
    logsq_weight = params.get("logsquare_weight", 0.1)
    multiplier = params.get("multiplier", 1.0)
    n_examples = len(labels)

    all_scores = []
    for i in range(n_examples):
        hidden = run_backbone_hidden(
            diff_model,
            tensors[f"latent_{i}"], tensors[f"sigma_{i}"],
            tensors[f"conditioning_{i}"],
            device, dtype,
            multiplier=multiplier, requires_grad=True,
        )
        scores = btrm_head(hidden)  # (1, N_heads)
        all_scores.append(scores.squeeze(0))  # (N_heads,)

    all_scores = torch.stack(all_scores)  # (N, N_heads)

    loss_result = compute_labeled_btrm_loss(
        all_scores, labels, btrm_head.head_names,
        logsquare_weight=logsq_weight,
    )
    loss = loss_result["loss"]

    btrm_optimizer.zero_grad()
    loss.backward()
    btrm_optimizer.step()

    return {
        "loss": loss.item(),
        "bt_loss": loss_result["bt_loss"].item(),
        "logsq_loss": loss_result["logsq_loss"].item(),
        "per_head_accuracy": loss_result["per_head_accuracy"],
        "n_examples": n_examples,
        "active_heads": loss_result["active_heads"],
    }


def accumulate_policy_gradients(diff_model, device, dtype, params, tensors):
    """Accumulate REINFORCE gradients for a LoRA adapter.

    Runs two separate B=1 forwards (ref under no_grad, pi with checkpointing)
    at each sparse step, computes MSE-based log-ratio, and backwards into LoRA
    params. Scale management is per-pass inside compute_reinforce_step.

    Args:
        diff_model: Raw diffusion model with LoRA adapters.
        device: CUDA device.
        dtype: Working dtype.
        params: RPC params (adapter_name, sparse_steps, advantage, multiplier).
        tensors: RPC tensors (checkpoint_N, sigmas, conditioning).

    Returns:
        Metadata dict with total_log_ratio and n_steps.
    """
    from .lora import clear_lora_scale, get_lora_params
    from .sampling import make_rope_cache

    adapter_name = params["adapter_name"]
    sparse_steps = params["sparse_steps"]
    advantage = params["advantage"]
    multiplier = params.get("multiplier", 1.0)

    conditioning = tensors["conditioning"].to(device=device, dtype=dtype)
    sigmas = tensors["sigmas"].to(device=device, dtype=dtype)

    # Enable gradients on target LoRA params (don't zero -- accumulate)
    lora_params = list(get_lora_params(diff_model, adapter_name=adapter_name))
    for p in lora_params:
        p.requires_grad_(True)

    # RoPE cache from first checkpoint
    first_key = f"checkpoint_{sparse_steps[0]}"
    sample_latent = tensors[first_key]
    B, C, H, W = sample_latent.shape
    num_tokens = conditioning.shape[1]
    rope_cache = make_rope_cache(diff_model, H, W, num_tokens, device)

    total_log_ratio = 0.0
    for step_idx in sparse_steps:
        x_t = tensors[f"checkpoint_{step_idx}"].to(device=device, dtype=dtype)
        step_lr = compute_reinforce_step(
            diff_model, x_t, sigmas[step_idx], conditioning,
            num_tokens, rope_cache, multiplier, advantage,
            adapter_name=adapter_name,
        )
        total_log_ratio += step_lr

    # Reset scale to broadcast 1.0
    clear_lora_scale(diff_model, adapter_name=adapter_name)
    torch.cuda.empty_cache()

    return {
        "total_log_ratio": total_log_ratio,
        "n_steps": len(sparse_steps),
    }


def policy_optimizer_step(diff_model, policy_optimizers, device, dtype, params):
    """Clip gradients, step policy optimizer, zero gradients.

    Lazy-initializes the optimizer on first call for each adapter.

    Args:
        diff_model: Raw diffusion model with LoRA adapters.
        policy_optimizers: Dict mapping adapter_name -> optimizer (mutated).
        device: CUDA device.
        dtype: Working dtype.
        params: RPC params (adapter_name, max_grad_norm, lr).

    Returns:
        Metadata dict with grad_norm and n_params.
    """
    from .lora import get_lora_params

    adapter_name = params["adapter_name"]
    max_grad_norm = params.get("max_grad_norm", 1.0)
    lr = params.get("lr", 1e-4)

    lora_params = list(get_lora_params(diff_model, adapter_name=adapter_name))

    # Lazy-init optimizer on first call
    if adapter_name not in policy_optimizers:
        policy_optimizers[adapter_name] = torch.optim.AdamW(lora_params, lr=lr)

    optimizer = policy_optimizers[adapter_name]

    grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, max_grad_norm)
    optimizer.step()

    for p in lora_params:
        if p.grad is not None:
            p.grad = None

    grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    return {
        "grad_norm": grad_norm_val,
        "n_params": len(lora_params),
    }

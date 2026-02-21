"""Packed multi-image forward path with uint8 SageAttention block masks.

Orchestrates block_mask.py and attention.py to provide a clean packed
forward interface for the NextDiT diffusion model. Callers use this
module instead of constructing masks themselves.

Three things happen here that do NOT happen in the unpacked path:
  1. A uint8 block-diagonal mask is constructed from packing segment info
  2. The attention backend is configured for masked Sage dispatch
  3. Multiple images are processed in a single transformer forward pass

The model's forward_packed() method already threads block_mask through
all transformer layers to sdpa_attention(), which routes uint8 masks to
the masked SageAttention kernel when the backend is "sage" or "auto".
This module constructs the correct mask type and ensures the backend is
configured, then delegates to the model.

Relationship to the other two modules:
  - block_mask.py constructs the uint8 mask tensor from segment lengths
  - attention.py provides the dispatch function (used directly when
    overriding attention, or indirectly via the model's built-in dispatch)
  - forward_packed.py (this) orchestrates both for the full packed path

Lifecycle axes (from user_dataflow_and_lifecycle_rollup.md):
  - Axis 6 (activation checkpointing): Supported -- the model's forward_packed
    is a normal autograd-compatible forward pass. Gradient checkpointing can
    wrap individual layer calls.
  - Axis 7 (rollout-training coupling): Supported -- the uint8 mask and
    sage_attn_masked_op have register_autograd, so gradients flow through
    attention in training mode.
  - Axis 10 (sequence packing): This IS the packing implementation.

Import constraints:
  - IMPORTS from futudiffu.attention: set_attention_backend (backend config)
  - IMPORTS from src_ii.block_mask: build_block_mask_from_packing_info
  - IMPORTS from src_ii.attention: select_backend (for backend selection logic)
  - Does NOT import model classes (model is passed as argument)
  - Does NOT import sampling, training, or server modules
"""

from __future__ import annotations

from typing import Any

import torch

from src_ii.block_mask import build_block_mask, build_block_mask_from_packing_info
from src_ii.attention import AttentionBackend, select_backend


def prepare_packed_forward(
    model,
    context_list: list[torch.Tensor],
    img_sizes: list[tuple[int, int]],
    cap_lens: list[int],
    device: torch.device,
    force_sdpa: bool = False,
) -> dict[str, Any]:
    """Prepare all constant state for packed multi-image generation.

    This is the packed equivalent of prepare_rope_cache() for single images.
    Runs the model's prepare_packed_state() (cap_embedder + context_refiner),
    then constructs a uint8 block mask from the resulting packing layout.

    The returned dict contains everything needed to call packed_forward()
    repeatedly across euler steps. All values are constant across steps.

    Args:
        model: NextDiT model instance (raw or compiled). Must have
            prepare_packed_state() method.
        context_list: N raw text conditionings, each (B, seq_i, cap_feat_dim).
            B=1 for unconditional, B=2 for CFG (pos+neg stacked).
        img_sizes: (H, W) per image AFTER pad_to_patch_size (pixel dimensions).
        cap_lens: Original caption lengths per image (before embedding/padding).
        device: Target device.
        force_sdpa: If True, use SDPA fallback instead of SageAttention.
            Useful for CPU testing or environments without Triton.

    Returns:
        Dict with keys:
            'refined_caps': List of N refined+padded caption embeddings.
            'packing_info': PackingInfo describing the packed sequence layout.
            'packed_rope': RoPE frequencies for the packed sequence.
            'block_mask': uint8 tensor (n_q_blocks, n_kv_blocks).
            'backend': AttentionBackend string for this configuration.
    """
    # Use the model's prepare_packed_state to compute refined caps,
    # packing layout, and RoPE. These are all invariant across euler steps.
    refined_caps, packing_info, packed_rope = model.prepare_packed_state(
        context_list, img_sizes, cap_lens, device,
    )

    # Construct the uint8 block mask from packing info
    block_mask = build_block_mask_from_packing_info(packing_info, device=device)

    # Determine the attention backend
    is_packed = packing_info.n_images > 1
    backend = select_backend(is_packed=is_packed, force_sdpa=force_sdpa)

    return {
        'refined_caps': refined_caps,
        'packing_info': packing_info,
        'packed_rope': packed_rope,
        'block_mask': block_mask,
        'backend': backend,
    }


def packed_forward(
    model,
    x_list: list[torch.Tensor],
    timesteps: torch.Tensor,
    refined_caps: list[torch.Tensor],
    packing_info,
    block_mask: torch.Tensor,
    packed_rope: torch.Tensor,
    ensure_sage_backend: bool = True,
) -> list[torch.Tensor]:
    """Execute a packed multi-image forward pass through the diffusion model.

    Thin wrapper around model.forward_packed() that ensures:
      1. The block_mask is a uint8 tensor (not a callable)
      2. The attention backend is configured for masked Sage dispatch

    The model's forward_packed() threads block_mask through all transformer
    layers to sdpa_attention(), which routes uint8 masks to the masked
    SageAttention kernel. This function ensures the backend is set correctly
    before calling.

    Args:
        model: NextDiT model instance (raw or compiled).
        x_list: N noisy latent images, each (B, C, H_i, W_i).
        timesteps: (B,) sigma values (shared across all images).
        refined_caps: N pre-refined caption embeddings from
            prepare_packed_forward()['refined_caps'].
        packing_info: PackingInfo from prepare_packed_forward()['packing_info'].
        block_mask: uint8 tensor from prepare_packed_forward()['block_mask'].
        packed_rope: RoPE frequencies from prepare_packed_forward()['packed_rope'].
        ensure_sage_backend: If True (default), set the attention backend in
            src/futudiffu/attention.py to "sage" before calling. This ensures
            the model's internal sdpa_attention() dispatches uint8 masks to
            the masked Sage kernel. Set to False if the backend is already
            configured or if you want to use a different backend.

    Returns:
        List of N output tensors, each (B, C, H_i, W_i), NEGATED.
    """
    if ensure_sage_backend:
        from futudiffu.attention import set_attention_backend
        set_attention_backend("sage")

    return model.forward_packed(
        x_list, timesteps, refined_caps,
        packing_info, block_mask, packed_rope,
    )


def make_packed_model_fn(
    model,
    refined_caps: list[torch.Tensor],
    packing_info,
    block_mask: torch.Tensor,
    packed_rope: torch.Tensor,
    cfg: float,
    multiplier: float,
    ensure_sage_backend: bool = True,
):
    """Create a packed model function for use with sample_euler_packed().

    Returns a callable compatible with sample_euler_packed()'s
    packed_forward_fn signature:
        fn(x_cfg_list, t_batch, refined_caps, packing_info,
           block_mask, packed_rope) -> list[Tensor]

    This function captures the backend configuration and ensures Sage
    attention is active before each call.

    Args:
        model: NextDiT model instance (compiled or raw).
        refined_caps: Pre-refined caption embeddings.
        packing_info: PackingInfo from prepare_packed_forward().
        block_mask: uint8 block mask from prepare_packed_forward().
        packed_rope: Packed RoPE from prepare_packed_forward().
        cfg: CFG scale (unused directly -- CFG is applied in the euler loop).
        multiplier: Timestep multiplier (unused directly -- applied in euler loop).
        ensure_sage_backend: Whether to set sage backend before each call.

    Returns:
        Callable matching sample_euler_packed()'s packed_forward_fn signature.
    """
    if ensure_sage_backend:
        from futudiffu.attention import set_attention_backend
        set_attention_backend("sage")

    def fn(x_cfg_list, t_batch, caps, info, mask, rope):
        return model.forward_packed(
            x_cfg_list, t_batch, caps, info, mask, rope,
        )

    return fn


def prepare_and_build_mask(
    segment_lengths: list[int],
    total_len: int | None = None,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Build a uint8 block mask from raw segment lengths.

    Convenience re-export of build_block_mask for callers that don't have
    a PackingInfo object (e.g., test code, standalone packing experiments).

    Args:
        segment_lengths: Per-image token counts in the packed sequence.
        total_len: Total packed sequence length (computed if None).
        device: Target device.

    Returns:
        uint8 tensor of shape (n_q_blocks, n_kv_blocks).
    """
    return build_block_mask(segment_lengths, total_len=total_len, device=device)

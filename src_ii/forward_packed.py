"""Packed multi-image forward path with uint8 block masks.

Builds a block-diagonal mask from packing segment info, then delegates
to the model. The model's forward() IS the packed forward.

ALL plans are padded to REFERENCE_TOTAL_LEN so there is exactly one
compiled graph regardless of content resolution. The block mask zeros
out padding tiles — they are never computed by the attention kernel.

Import constraints:
  - IMPORTS from src_ii.block_mask: build_block_mask, build_block_mask_from_packing_info
  - IMPORTS from src_ii.bin_packer: REFERENCE_TOTAL_LEN
  - Does NOT import model classes (model is passed as argument)
  - Does NOT import sampling, training, or server modules
"""

from __future__ import annotations

from typing import Any

import torch

from src_ii.block_mask import build_block_mask, build_block_mask_from_packing_info
from src_ii.bin_packer import REFERENCE_TOTAL_LEN


def _pad_plan_to_fixed_len(
    packing_info,
    packed_rope: torch.Tensor,
    target_len: int = REFERENCE_TOTAL_LEN,
) -> tuple:
    """Pad packing_info and packed_rope to a fixed total_len.

    Ensures every forward call sees the same sequence length, so
    torch.compile traces one graph and never recompiles.

    Args:
        packing_info: PackingInfo from build_packed_sequence.
        packed_rope: (1, n_axes, natural_len, 1, dim, 2) RoPE tensor.
        target_len: Fixed sequence length to pad to.

    Returns:
        (packing_info, packed_rope) with total_len == target_len.
    """
    natural_len = packing_info.total_len
    if natural_len > target_len:
        raise ValueError(
            f"Packed sequence length {natural_len} exceeds "
            f"REFERENCE_TOTAL_LEN {target_len}. Content doesn't fit."
        )
    if natural_len == target_len:
        return packing_info, packed_rope

    pad_count = target_len - natural_len

    # Extend document_id with -1 (padding tokens)
    pad_ids = torch.full(
        (pad_count,), -1,
        dtype=packing_info.document_id.dtype,
        device=packing_info.document_id.device,
    )
    packing_info.document_id = torch.cat([packing_info.document_id, pad_ids])
    packing_info.total_len = target_len

    # Extend packed_rope with zeros along the sequence dimension (dim=1).
    # After movedim(1,2) in build_packed_rope, shape is (B, seq, 1, n_pairs, 2, 2).
    rope_pad = torch.zeros(
        packed_rope.shape[0], pad_count, *packed_rope.shape[2:],
        dtype=packed_rope.dtype, device=packed_rope.device,
    )
    packed_rope = torch.cat([packed_rope, rope_pad], dim=1)

    return packing_info, packed_rope


def prepare_packed_forward(
    model,
    context_list: list[torch.Tensor],
    img_sizes: list[tuple[int, int]],
    cap_lens: list[int],
    device: torch.device,
    target_len: int = REFERENCE_TOTAL_LEN,
) -> dict[str, Any]:
    """Prepare constant state for packed multi-image forward.

    Runs model.prepare_packed_state() then pads everything to
    target_len and builds the block mask at that fixed size.
    Every call at the same target_len hits one compiled graph.
    """
    refined_caps, packing_info, packed_rope = model.prepare_packed_state(
        context_list, img_sizes, cap_lens, device,
    )

    # Pad to fixed length BEFORE building block mask
    packing_info, packed_rope = _pad_plan_to_fixed_len(
        packing_info, packed_rope, target_len,
    )

    block_mask = build_block_mask_from_packing_info(packing_info, device=device)

    return {
        'refined_caps': refined_caps,
        'packing_info': packing_info,
        'packed_rope': packed_rope,
        'block_mask': block_mask,
    }


def packed_forward(
    model,
    x_list: list[torch.Tensor],
    timesteps: torch.Tensor | list[torch.Tensor],
    refined_caps: list[torch.Tensor],
    packing_info,
    block_mask: torch.Tensor,
    packed_rope: torch.Tensor,
    adapter_scales: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Execute a packed multi-image forward pass.

    Resolves per-image timesteps then calls model().
    Returns (diffusion_fields, scores).
    """
    if isinstance(timesteps, list):
        timesteps_list = timesteps
    else:
        timesteps_list = [timesteps] * len(x_list)

    return model(
        x_list, timesteps_list, refined_caps,
        packing_info, block_mask, packed_rope,
        adapter_scales=adapter_scales,
    )


def make_packed_model_fn(
    model,
    refined_caps: list[torch.Tensor],
    packing_info,
    block_mask: torch.Tensor,
    packed_rope: torch.Tensor,
    cfg: float,
    multiplier: float,
    adapter_scales: torch.Tensor | None = None,
):
    """Create a packed model callable for Euler sampling.

    Returns fn(x_cfg_list, t_batch, caps, info, mask, rope) -> (list[Tensor], Tensor).
    """
    def fn(x_cfg_list, t_batch, caps, info, mask, rope):
        timesteps_list = [t_batch] * len(x_cfg_list)
        return model(
            x_cfg_list, timesteps_list, caps, info, mask, rope,
            adapter_scales=adapter_scales,
        )

    return fn


def prepare_and_build_mask(
    segment_lengths: list[int],
    total_len: int | None = None,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Build a uint8 block mask from raw segment lengths."""
    if total_len is None:
        total_len = REFERENCE_TOTAL_LEN
    return build_block_mask(segment_lengths, total_len=total_len, device=device)

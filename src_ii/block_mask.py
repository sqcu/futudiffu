"""uint8 block mask construction for packed multi-image sequences.

Constructs block-diagonal attention masks from packing segment information.
Each image's Q tokens can only attend to that same image's KV tokens.
The mask tensor is uint8 (0 = skip, 1 = compute) at the tile granularity
of the SageAttention Triton kernels: BLOCK_M=128 for Q, BLOCK_N=64 for KV.

This module constructs attention masks as direct uint8 tensors, bypassing
the callable mask_mod approach entirely:
  1. torch.compile compatible (no Python callbacks, no graph breaks)
  2. Type-correct for Sage kernels (uint8 tensor, not a callable)
  3. Broadcast-ready (2D mask shared across all batch*head dimensions)

Relationship to the other two modules:
  - block_mask.py (this) constructs the mask
  - attention.py consumes the mask (routes to sage_attn_masked_op)
  - forward_packed.py orchestrates both (calls block_mask, passes to attention)

Import constraints:
  - torch only. No futudiffu imports.
"""

import torch


# Sage kernel tile sizes (must match sage_kernels.py constants)
BLOCK_M: int = 128  # Q tile size
BLOCK_N: int = 64   # KV tile size


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def build_block_mask(
    segment_lengths: list[int],
    total_len: int | None = None,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Build a uint8 block-diagonal attention mask from segment lengths.

    Each segment corresponds to one packed image's tokens (text + image patches,
    already padded to pad_tokens_multiple). Segments are contiguous and laid out
    in order: [seg_0, seg_1, ..., seg_{N-1}].

    The mask has shape (n_q_blocks, n_kv_blocks) where:
        n_q_blocks  = ceil(total_len / BLOCK_M)
        n_kv_blocks = ceil(total_len / BLOCK_N)

    mask[qi, kj] = 1 if Q block qi and KV block kj belong to the same image.

    The construction uses pure tensor ops (no Python loops over sequence
    positions) to ensure torch.compile compatibility.

    Args:
        segment_lengths: Per-image token counts in the packed sequence.
            Sum must equal total_len. Each value should already include
            text + image padding.
        total_len: Total packed sequence length. If None, computed as
            sum(segment_lengths).
        device: Target device for the mask tensor.

    Returns:
        uint8 tensor of shape (n_q_blocks, n_kv_blocks).
    """
    if total_len is None:
        total_len = sum(segment_lengths)

    n_q_blocks = _ceildiv(total_len, BLOCK_M)
    n_kv_blocks = _ceildiv(total_len, BLOCK_N)

    # Build per-token document_id: token_idx -> image_index
    # This is a 1D tensor of length total_len, constructed without
    # Python-level per-token iteration.
    doc_ids = torch.zeros(total_len, dtype=torch.int32, device=device)
    offset = 0
    for img_idx, seg_len in enumerate(segment_lengths):
        doc_ids[offset:offset + seg_len] = img_idx
        offset += seg_len

    # For each Q block, take the document_id of its first token.
    # Since segments are contiguous and block-aligned (pad_tokens_multiple=32
    # divides BLOCK_M=128 and BLOCK_N=64), the first token of a block
    # determines the entire block's ownership. If a block spans two images
    # (which should not happen with proper padding), the first token's
    # document_id is used -- conservative for safety.
    q_block_starts = torch.arange(n_q_blocks, device=device) * BLOCK_M
    q_block_starts = q_block_starts.clamp(max=total_len - 1)
    q_doc = doc_ids[q_block_starts]  # (n_q_blocks,)

    kv_block_starts = torch.arange(n_kv_blocks, device=device) * BLOCK_N
    kv_block_starts = kv_block_starts.clamp(max=total_len - 1)
    kv_doc = doc_ids[kv_block_starts]  # (n_kv_blocks,)

    # Block mask: outer product equality check
    # mask[i, j] = 1 if q_doc[i] == kv_doc[j]
    mask = (q_doc.unsqueeze(1) == kv_doc.unsqueeze(0)).to(torch.uint8)

    return mask


def build_block_mask_from_packing_info(
    packing_info,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Build a uint8 block mask from a PackingInfo dataclass.

    Convenience wrapper that extracts segment lengths from packing_info
    and delegates to build_block_mask().

    Args:
        packing_info: A PackingInfo instance (from diffusion_model.py).
            Must have 'segments' (list of (text_start, text_len, img_start,
            img_len) tuples) and 'total_len' attributes.
        device: Target device for the mask tensor.

    Returns:
        uint8 tensor of shape (n_q_blocks, n_kv_blocks).
    """
    # Each image's total token span = text_len + img_len (contiguous)
    segment_lengths = [
        text_len + img_len
        for (text_start, text_len, img_start, img_len) in packing_info.segments
    ]
    return build_block_mask(
        segment_lengths,
        total_len=packing_info.total_len,
        device=device,
    )

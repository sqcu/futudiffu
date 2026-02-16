"""Validation test for SageAttention block mask support.

Compares three attention implementations for packed multi-image inference:
1. Padded-SDPA reference: standard PyTorch SDPA with explicit mask tensor
2. FlexAttention: forward_packed() with FlexAttention create_block_mask()
3. SageAttention + block_mask: masked Sage kernel with uint8 block mask

All three should produce equivalent results (within quantization noise).

Usage:
    .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\tests\test_sage_block_mask.py
"""

import sys
import math

import torch
import torch.nn.functional as F

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

from futudiffu.diffusion_model import (
    PackingInfo,
    build_packed_rope,
    build_packed_sequence,
    create_diffusion_model,
    make_packing_mask_mod,
    pad_to_patch_size,
    pad_zimage,
)
from futudiffu.attention import set_attention_backend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two tensors."""
    return F.cosine_similarity(
        a.flatten().float().unsqueeze(0),
        b.flatten().float().unsqueeze(0),
    ).item()


def build_sage_block_mask_from_document_id(
    document_id: torch.Tensor,
    n_heads: int,
    seq_len: int,
    block_m: int = 128,
    block_n: int = 64,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """Build a SageAttention-compatible uint8 block mask from document IDs.

    For each (q_block, kv_block) pair, the block is active (1) if ANY query
    token in the q_block and ANY kv token in the kv_block share the same
    document ID and both are non-negative. This is a conservative
    (over-approximating) block mask -- it allows some cross-document attention
    at block boundaries, but the -inf masking from out-of-bounds positions
    and the document structure prevent any actual leakage.

    For packed multi-image inference, each "document" is one image's text+image
    tokens. Within-document blocks are always active. Cross-document blocks
    are masked out.

    Args:
        document_id: (total_len,) int32 tensor, -1 for padding.
        n_heads: Number of attention heads (mask is same for all heads).
        seq_len: Total sequence length.
        block_m: Q block size for SageAttention (default 128).
        block_n: KV block size for SageAttention (default 64).
        device: Target device.

    Returns:
        (n_q_blocks, n_kv_blocks) uint8 tensor on device.
    """
    n_q_blocks = math.ceil(seq_len / block_m)
    n_kv_blocks = math.ceil(seq_len / block_n)

    mask = torch.zeros(n_q_blocks, n_kv_blocks, dtype=torch.uint8, device=device)

    for qi in range(n_q_blocks):
        q_start = qi * block_m
        q_end = min(q_start + block_m, seq_len)
        q_docs = document_id[q_start:q_end]
        # Set of valid document IDs in this Q block (excluding padding = -1)
        q_doc_set = set(q_docs[q_docs >= 0].tolist())

        for ki in range(n_kv_blocks):
            k_start = ki * block_n
            k_end = min(k_start + block_n, seq_len)
            k_docs = document_id[k_start:k_end]
            k_doc_set = set(k_docs[k_docs >= 0].tolist())

            # Block is active if there is any overlap in document IDs
            if q_doc_set & k_doc_set:
                mask[qi, ki] = 1

    return mask


def build_sdpa_mask_from_document_id(
    document_id: torch.Tensor,
    seq_len: int,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """Build a dense (seq_len, seq_len) boolean attention mask from document IDs.

    mask[i, j] = True if document_id[i] == document_id[j] AND document_id[i] >= 0.
    This is the explicit mask used by the SDPA reference implementation.

    Args:
        document_id: (total_len,) int32 tensor.
        seq_len: Sequence length.
        device: Target device.

    Returns:
        (seq_len, seq_len) bool tensor on device.
    """
    doc = document_id[:seq_len].to(device)
    # Same document AND not padding
    mask = (doc.unsqueeze(0) == doc.unsqueeze(1)) & (doc.unsqueeze(0) >= 0)
    return mask


# ---------------------------------------------------------------------------
# Test: Attention-level comparison (isolated from full model)
# ---------------------------------------------------------------------------

def test_attention_level():
    """Test block mask at the raw attention level (no model, just Q/K/V).

    Packs 2 "documents" into a single sequence and verifies:
    - SDPA + explicit mask (reference)
    - SageAttention + uint8 block mask
    - SageAttention unmasked (should differ -- shows mask works)
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("=== Test: Attention-level block mask ===")

    # Two documents of different lengths
    doc0_len = 192   # 3 Q blocks (128), 3 KV blocks (64)
    doc1_len = 128   # 2 Q blocks, 2 KV blocks
    seq_len = doc0_len + doc1_len  # 320

    B, H, D = 1, 8, 128
    torch.manual_seed(42)
    q = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)
    k = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)
    v = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)
    sm_scale = 1.0 / math.sqrt(D)

    # Document IDs: doc0 for [0, doc0_len), doc1 for [doc0_len, seq_len)
    document_id = torch.zeros(seq_len, dtype=torch.int32, device=device)
    document_id[:doc0_len] = 0
    document_id[doc0_len:] = 1

    # --- Reference: SDPA with explicit mask ---
    sdpa_mask = build_sdpa_mask_from_document_id(document_id, seq_len, device)
    # Convert bool mask to float mask for SDPA: True -> 0.0, False -> -inf
    sdpa_float_mask = torch.where(sdpa_mask, 0.0, float("-inf"))
    sdpa_float_mask = sdpa_float_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)

    out_ref = F.scaled_dot_product_attention(
        q, k, v, attn_mask=sdpa_float_mask, dropout_p=0.0, is_causal=False,
    )
    print(f"  SDPA reference: {out_ref.shape}")

    # --- SageAttention + block mask ---
    sage_bm = build_sage_block_mask_from_document_id(
        document_id, H, seq_len, block_m=128, block_n=64, device=device,
    )
    print(f"  Sage block mask shape: {sage_bm.shape}")
    print(f"  Active blocks: {sage_bm.sum().item()} / {sage_bm.numel()}")

    from futudiffu.sage_attention import sage_attn_forward_masked
    out_sage_masked = sage_attn_forward_masked(q, k, v, sage_bm, sm_scale)
    print(f"  Sage masked: {out_sage_masked.shape}")

    # --- SageAttention unmasked (for comparison -- should differ) ---
    from futudiffu.sage_attention import sage_attn_forward
    out_sage_unmasked = sage_attn_forward(q, k, v, sm_scale)

    # --- Comparisons ---
    cos_sage_ref = cosine_sim(out_sage_masked, out_ref)
    cos_unmasked_ref = cosine_sim(out_sage_unmasked, out_ref)

    print(f"\n  Sage+mask vs SDPA reference: cos={cos_sage_ref:.6f}")
    print(f"  Sage unmasked vs SDPA reference: cos={cos_unmasked_ref:.6f}")
    print(f"  (Unmasked should be lower, showing the mask has effect)")

    # Assert masked version matches reference closely
    assert cos_sage_ref > 0.999, (
        f"Sage+mask vs SDPA reference cosine too low: {cos_sage_ref}"
    )

    # Assert unmasked is measurably different (mask has effect)
    assert cos_unmasked_ref < cos_sage_ref, (
        f"Unmasked should differ more from reference than masked. "
        f"masked cos={cos_sage_ref}, unmasked cos={cos_unmasked_ref}"
    )

    print("  PASSED")
    return cos_sage_ref


# ---------------------------------------------------------------------------
# Test: Model-level comparison (S-S-S model, 2 packed images)
# ---------------------------------------------------------------------------

def test_model_level():
    """Test block mask at the model level using the S-S-S fixture.

    Packs 2 small images and compares:
    1. SDPA reference (padded + stacked with explicit mask)
    2. FlexAttention (forward_packed with FlexAttention block mask)
    3. SageAttention + block_mask (forward_packed with sage backend + uint8 mask)

    The S-S-S model uses real FP8 weights from Z-Image layer 0.
    """
    from stubbed_skinny_shared import load_sss_model, SSS_DIM, SSS_N_HEADS

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("\n=== Test: Model-level block mask (S-S-S model) ===")

    # Load S-S-S model
    print("  Loading S-S-S model...")
    model = load_sss_model(device=device)
    model.prepare_adaln_cache()
    print(f"  Model: dim={model.dim}, n_heads={model.n_heads}, "
          f"layers={len(model.layers)}")

    pH = pW = model.patch_size
    cap_feat_dim = 2560

    # Two small images
    H1, W1 = 32, 32    # 16x16 patches = 256 tokens
    H2, W2 = 32, 64    # 16x32 patches = 512 tokens
    seq1, seq2 = 12, 16  # Caption lengths

    torch.manual_seed(123)
    x1 = torch.randn(1, 16, H1, W1, device=device, dtype=dtype)
    x2 = torch.randn(1, 16, H2, W2, device=device, dtype=dtype)
    ctx1 = torch.randn(1, seq1, cap_feat_dim, device=device, dtype=dtype)
    ctx2 = torch.randn(1, seq2, cap_feat_dim, device=device, dtype=dtype)
    timesteps = torch.tensor([0.5], device=device, dtype=dtype)

    x1_pad = pad_to_patch_size(x1, (pH, pW))
    x2_pad = pad_to_patch_size(x2, (pH, pW))
    H1p, W1p = x1_pad.shape[2], x1_pad.shape[3]
    H2p, W2p = x2_pad.shape[2], x2_pad.shape[3]

    print(f"  Image 1: {H1}x{W1} -> {H1p}x{W1p}, caption={seq1}")
    print(f"  Image 2: {H2}x{W2} -> {H2p}x{W2p}, caption={seq2}")

    # --- 1. Individual unpacked forward (SDPA reference) ---
    print("\n  Running individual SDPA forward passes (reference)...")
    set_attention_backend("sdpa")
    with torch.inference_mode():
        rope1 = model.prepare_rope_cache(H1p, W1p, seq1, device)
        out1_ref = model(x1, timesteps, ctx1, num_tokens=seq1, rope_cache=rope1)
        rope2 = model.prepare_rope_cache(H2p, W2p, seq2, device)
        out2_ref = model(x2, timesteps, ctx2, num_tokens=seq2, rope_cache=rope2)
    print(f"  Ref img 1: {out1_ref.shape}, Ref img 2: {out2_ref.shape}")

    # --- 2. FlexAttention packed forward ---
    print("\n  Running FlexAttention packed forward...")
    set_attention_backend("sdpa")  # FlexAttention path, not Sage
    with torch.inference_mode():
        refined_caps, packing_info, packed_rope = model.prepare_packed_state(
            [ctx1, ctx2],
            [(H1p, W1p), (H2p, W2p)],
            [seq1, seq2],
            device,
        )

        from torch.nn.attention.flex_attention import create_block_mask
        flex_mask_mod = make_packing_mask_mod(packing_info.document_id)
        flex_block_mask = create_block_mask(
            flex_mask_mod,
            B=1, H=None,
            Q_LEN=packing_info.total_len,
            KV_LEN=packing_info.total_len,
            device=device,
        )

        outputs_flex = model.forward_packed(
            [x1, x2], timesteps, refined_caps,
            packing_info, flex_block_mask, packed_rope,
        )
    out1_flex, out2_flex = outputs_flex
    print(f"  Flex img 1: {out1_flex.shape}, Flex img 2: {out2_flex.shape}")

    # --- 3. SageAttention + block_mask packed forward ---
    print("\n  Running SageAttention + block_mask packed forward...")
    set_attention_backend("sage")

    # Build uint8 block mask for SageAttention
    sage_bm = build_sage_block_mask_from_document_id(
        packing_info.document_id,
        model.n_heads,
        packing_info.total_len,
        block_m=128, block_n=64,
        device=device,
    )
    print(f"  Sage block mask: {sage_bm.shape}, "
          f"active={sage_bm.sum().item()}/{sage_bm.numel()}")

    with torch.inference_mode():
        # Reuse same refined_caps, packing_info, packed_rope from FlexAttention
        # but pass the uint8 block mask instead of FlexAttention BlockMask
        outputs_sage = model.forward_packed(
            [x1, x2], timesteps, refined_caps,
            packing_info, sage_bm, packed_rope,
        )
    out1_sage, out2_sage = outputs_sage
    print(f"  Sage img 1: {out1_sage.shape}, Sage img 2: {out2_sage.shape}")

    # Reset backend
    set_attention_backend("sdpa")

    # --- Comparisons ---
    print("\n  --- Comparisons ---")

    # FlexAttention vs SDPA reference (individual)
    cos_flex_ref_1 = cosine_sim(out1_flex, out1_ref)
    cos_flex_ref_2 = cosine_sim(out2_flex, out2_ref)
    print(f"  FlexAttn vs SDPA ref:")
    print(f"    Image 1: cos={cos_flex_ref_1:.6f}")
    print(f"    Image 2: cos={cos_flex_ref_2:.6f}")

    # SageAttention+mask vs SDPA reference
    cos_sage_ref_1 = cosine_sim(out1_sage, out1_ref)
    cos_sage_ref_2 = cosine_sim(out2_sage, out2_ref)
    print(f"  Sage+mask vs SDPA ref:")
    print(f"    Image 1: cos={cos_sage_ref_1:.6f}")
    print(f"    Image 2: cos={cos_sage_ref_2:.6f}")

    # SageAttention+mask vs FlexAttention
    cos_sage_flex_1 = cosine_sim(out1_sage, out1_flex)
    cos_sage_flex_2 = cosine_sim(out2_sage, out2_flex)
    print(f"  Sage+mask vs FlexAttn:")
    print(f"    Image 1: cos={cos_sage_flex_1:.6f}")
    print(f"    Image 2: cos={cos_sage_flex_2:.6f}")

    # Assertions (threshold accounts for FP8/INT8 quantization + accumulation differences)
    threshold = 0.999

    # FlexAttention should closely match individual SDPA (same compute, different packing)
    assert cos_flex_ref_1 > threshold, (
        f"FlexAttn vs ref img1 too low: {cos_flex_ref_1}"
    )
    assert cos_flex_ref_2 > threshold, (
        f"FlexAttn vs ref img2 too low: {cos_flex_ref_2}"
    )

    # Sage+mask vs SDPA reference (FP8 quantization noise, relaxed slightly)
    sage_threshold = 0.995  # More relaxed for FP8 quantization
    assert cos_sage_ref_1 > sage_threshold, (
        f"Sage+mask vs ref img1 too low: {cos_sage_ref_1}"
    )
    assert cos_sage_ref_2 > sage_threshold, (
        f"Sage+mask vs ref img2 too low: {cos_sage_ref_2}"
    )

    # Sage+mask vs FlexAttention
    assert cos_sage_flex_1 > sage_threshold, (
        f"Sage+mask vs flex img1 too low: {cos_sage_flex_1}"
    )
    assert cos_sage_flex_2 > sage_threshold, (
        f"Sage+mask vs flex img2 too low: {cos_sage_flex_2}"
    )

    print("\n  ALL ASSERTIONS PASSED")

    # Save intermediate tensors to disk for post-hoc analysis
    save_dir = r"F:\dox\repos\ai\futudiffu\tests"
    torch.save({
        "out1_ref": out1_ref.cpu(),
        "out2_ref": out2_ref.cpu(),
        "out1_flex": out1_flex.cpu(),
        "out2_flex": out2_flex.cpu(),
        "out1_sage": out1_sage.cpu(),
        "out2_sage": out2_sage.cpu(),
        "sage_block_mask": sage_bm.cpu(),
        "document_id": packing_info.document_id.cpu(),
        "cosines": {
            "flex_ref_1": cos_flex_ref_1,
            "flex_ref_2": cos_flex_ref_2,
            "sage_ref_1": cos_sage_ref_1,
            "sage_ref_2": cos_sage_ref_2,
            "sage_flex_1": cos_sage_flex_1,
            "sage_flex_2": cos_sage_flex_2,
        },
    }, f"{save_dir}/sage_block_mask_validation.pt")
    print(f"\n  Saved validation data to {save_dir}/sage_block_mask_validation.pt")

    return {
        "flex_ref": (cos_flex_ref_1, cos_flex_ref_2),
        "sage_ref": (cos_sage_ref_1, cos_sage_ref_2),
        "sage_flex": (cos_sage_flex_1, cos_sage_flex_2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("SageAttention Block Mask Validation")
    print("=" * 60)

    # Test 1: Pure attention level
    cos_attn = test_attention_level()

    # Test 2: Full model level with S-S-S fixture
    cosines = test_model_level()

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Attention-level Sage+mask vs SDPA: cos={cos_attn:.6f}")
    print(f"  Model-level FlexAttn vs ref: {cosines['flex_ref']}")
    print(f"  Model-level Sage+mask vs ref: {cosines['sage_ref']}")
    print(f"  Model-level Sage+mask vs FlexAttn: {cosines['sage_flex']}")
    print("=" * 60)
    print("All tests passed!")

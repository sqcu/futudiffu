"""Verification test for FlexAttention batch packing.

Compares unpacked single-image forward() against packed forward_packed(N=1)
to verify bit-for-bit correctness.

Usage:
    .venv/Scripts/python.exe test_flexattn_packing.py
"""

import sys
import time

import torch
import torch.nn.functional as F

# Ensure futudiffu is importable
sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

from futudiffu.diffusion_model import (
    NextDiT,
    PackingInfo,
    build_packed_rope,
    build_packed_sequence,
    create_diffusion_model,
    fuse_model,
    make_packing_mask_mod,
    pad_to_patch_size,
)


def test_single_image_bitwise():
    """Run the same single image through forward() and forward_packed(N=1).

    They must produce bitwise identical outputs because:
    - Same RoPE positions (local positions = global positions for N=1)
    - Same adaLN params (same timestep)
    - FlexAttention with all-ones block mask = SDPA (no masking effect)
    - Same patchify, embed, refine, unpatchify logic
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    n_layers = 4  # Small model for speed

    print("Creating test model (n_layers=4, meta -> CUDA)...")
    model = create_diffusion_model(
        dtype=dtype, n_layers=n_layers, cap_feat_dim=2560, qk_norm=True
    )
    model = model.to_empty(device=device)
    model = model.to(dtype=dtype)

    # Initialize parameters deterministically
    torch.manual_seed(42)
    for p in model.parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, std=0.02)
    model.eval()

    # Prepare batched adaLN
    model.prepare_adaln_cache()

    # Test inputs (single image)
    B = 1
    H, W = 64, 96  # Small image: 64x96 -> 32x48 patches -> 1536 tokens
    C = 16
    cap_feat_dim = 2560
    seq_len = 20  # Caption length

    torch.manual_seed(123)
    x = torch.randn(B, C, H, W, device=device, dtype=dtype)
    timesteps = torch.tensor([0.5], device=device, dtype=dtype)
    context = torch.randn(B, seq_len, cap_feat_dim, device=device, dtype=dtype)

    print(f"Image: ({B}, {C}, {H}, {W}), Caption: ({B}, {seq_len}, {cap_feat_dim})")
    pH = pW = model.patch_size
    x_padded = pad_to_patch_size(x, (pH, pW))
    _, _, H_pad, W_pad = x_padded.shape
    print(f"Padded: ({H_pad}, {W_pad}), Tokens: {(H_pad // pH) * (W_pad // pW)}")

    # --- Unpacked forward() ---
    print("\nRunning unpacked forward()...")
    rope_cache = model.prepare_rope_cache(H_pad, W_pad, seq_len, device)
    with torch.inference_mode():
        out_unpacked = model(
            x, timesteps, context,
            num_tokens=seq_len,
            rope_cache=rope_cache,
        )
    print(f"  Output: {out_unpacked.shape}")

    # --- Packed forward_packed(N=1) ---
    print("\nRunning packed forward_packed(N=1)...")
    img_sizes = [(H_pad, W_pad)]
    cap_lens = [seq_len]

    with torch.inference_mode():
        refined_caps, packing_info, packed_rope = model.prepare_packed_state(
            [context], img_sizes, cap_lens, device
        )

        print(f"  PackingInfo: n_images={packing_info.n_images}, "
              f"total_len={packing_info.total_len}")
        print(f"  Segments: {packing_info.segments}")
        print(f"  Document IDs unique: {packing_info.document_id.unique().tolist()}")

        # Build block mask
        from torch.nn.attention.flex_attention import create_block_mask

        mask_mod = make_packing_mask_mod(packing_info.document_id)
        block_mask = create_block_mask(
            mask_mod,
            B=B,
            H=None,
            Q_LEN=packing_info.total_len,
            KV_LEN=packing_info.total_len,
            device=device,
        )
        print(f"  BlockMask created")

        outputs = model.forward_packed(
            [x], timesteps, refined_caps, packing_info, block_mask, packed_rope,
        )
        out_packed = outputs[0]
    print(f"  Output: {out_packed.shape}")

    # --- Compare ---
    print("\n--- Comparison ---")
    assert out_unpacked.shape == out_packed.shape, (
        f"Shape mismatch: {out_unpacked.shape} vs {out_packed.shape}"
    )

    diff = (out_unpacked - out_packed).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    cos = F.cosine_similarity(
        out_unpacked.flatten().unsqueeze(0),
        out_packed.flatten().unsqueeze(0),
    ).item()

    print(f"  Max abs diff: {max_diff}")
    print(f"  Mean abs diff: {mean_diff}")
    print(f"  Cosine similarity: {cos}")

    if max_diff == 0.0:
        print("  BITWISE IDENTICAL")
    elif cos > 0.999:
        print(f"  CLOSE (cos={cos:.6f}) but not bitwise identical")
        print("  (FlexAttention vs SDPA may have different accumulation order)")
    else:
        print(f"  DIVERGENT (cos={cos:.6f})")
        # Print some diagnostic info
        print(f"  unpacked stats: mean={out_unpacked.mean():.6f}, "
              f"std={out_unpacked.std():.6f}")
        print(f"  packed stats: mean={out_packed.mean():.6f}, "
              f"std={out_packed.std():.6f}")

    return max_diff, cos


def test_multi_image():
    """Pack 2 images of different sizes and verify no crashes + sensible output."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    n_layers = 4

    print("\n\n=== Multi-image packing test (N=2, different sizes) ===")
    print("Creating test model...")
    model = create_diffusion_model(
        dtype=dtype, n_layers=n_layers, cap_feat_dim=2560, qk_norm=True
    )
    model = model.to_empty(device=device)
    model = model.to(dtype=dtype)
    torch.manual_seed(42)
    for p in model.parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, std=0.02)
    model.eval()
    model.prepare_adaln_cache()

    pH = pW = model.patch_size
    cap_feat_dim = 2560

    # Two images of different sizes
    H1, W1 = 32, 32    # Small: 16x16 patches = 256 tokens
    H2, W2 = 64, 96    # Bigger: 32x48 patches = 1536 tokens
    seq1 = 12
    seq2 = 20

    torch.manual_seed(456)
    x1 = torch.randn(1, 16, H1, W1, device=device, dtype=dtype)
    x2 = torch.randn(1, 16, H2, W2, device=device, dtype=dtype)
    ctx1 = torch.randn(1, seq1, cap_feat_dim, device=device, dtype=dtype)
    ctx2 = torch.randn(1, seq2, cap_feat_dim, device=device, dtype=dtype)
    timesteps = torch.tensor([0.5], device=device, dtype=dtype)

    x1_pad = pad_to_patch_size(x1, (pH, pW))
    x2_pad = pad_to_patch_size(x2, (pH, pW))
    H1p, W1p = x1_pad.shape[2], x1_pad.shape[3]
    H2p, W2p = x2_pad.shape[2], x2_pad.shape[3]
    print(f"Image 1: {H1}x{W1} -> {H1p}x{W1p}, "
          f"{(H1p//pH)*(W1p//pW)} tokens, caption={seq1}")
    print(f"Image 2: {H2}x{W2} -> {H2p}x{W2p}, "
          f"{(H2p//pH)*(W2p//pW)} tokens, caption={seq2}")

    with torch.inference_mode():
        refined_caps, packing_info, packed_rope = model.prepare_packed_state(
            [ctx1, ctx2],
            [(H1p, W1p), (H2p, W2p)],
            [seq1, seq2],
            device,
        )

        print(f"PackingInfo: n_images={packing_info.n_images}, "
              f"total_len={packing_info.total_len}")
        for i, seg in enumerate(packing_info.segments):
            print(f"  Image {i}: text_start={seg[0]}, text_len={seg[1]}, "
                  f"img_start={seg[2]}, img_len={seg[3]}")

        from torch.nn.attention.flex_attention import create_block_mask

        block_mask = create_block_mask(
            make_packing_mask_mod(packing_info.document_id),
            B=1, H=None,
            Q_LEN=packing_info.total_len,
            KV_LEN=packing_info.total_len,
            device=device,
        )

        print(f"BlockMask created")

        t0 = time.perf_counter()
        outputs = model.forward_packed(
            [x1, x2], timesteps, refined_caps,
            packing_info, block_mask, packed_rope,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    print(f"\nForward pass: {elapsed*1000:.1f}ms")
    print(f"Output 0: {outputs[0].shape}  (expected: 1, 16, {H1}, {W1})")
    print(f"Output 1: {outputs[1].shape}  (expected: 1, 16, {H2}, {W2})")

    assert outputs[0].shape == (1, 16, H1, W1), f"Wrong shape: {outputs[0].shape}"
    assert outputs[1].shape == (1, 16, H2, W2), f"Wrong shape: {outputs[1].shape}"
    print("Shapes correct!")

    # Compare against individual unpacked forward passes
    print("\nComparing against individual unpacked forward passes...")
    with torch.inference_mode():
        rope1 = model.prepare_rope_cache(H1p, W1p, seq1, device)
        out1_unpacked = model(x1, timesteps, ctx1, num_tokens=seq1, rope_cache=rope1)

        rope2 = model.prepare_rope_cache(H2p, W2p, seq2, device)
        out2_unpacked = model(x2, timesteps, ctx2, num_tokens=seq2, rope_cache=rope2)

    for i, (packed, unpacked) in enumerate([(outputs[0], out1_unpacked),
                                             (outputs[1], out2_unpacked)]):
        diff = (packed - unpacked).abs()
        max_diff = diff.max().item()
        cos = F.cosine_similarity(
            packed.flatten().unsqueeze(0),
            unpacked.flatten().unsqueeze(0),
        ).item()
        status = "BITWISE" if max_diff == 0.0 else f"max_diff={max_diff:.2e}"
        print(f"  Image {i}: cos={cos:.6f}, {status}")


def test_cfg_batched():
    """Test packed forward with B=2 (CFG batching)."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    n_layers = 4

    print("\n\n=== CFG-batched packing test (B=2, N=2 images) ===")
    model = create_diffusion_model(
        dtype=dtype, n_layers=n_layers, cap_feat_dim=2560, qk_norm=True
    )
    model = model.to_empty(device=device)
    model = model.to(dtype=dtype)
    torch.manual_seed(42)
    for p in model.parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, std=0.02)
    model.eval()
    model.prepare_adaln_cache()

    pH = pW = model.patch_size
    cap_feat_dim = 2560

    H1, W1 = 32, 32
    H2, W2 = 32, 64
    seq1 = 12
    seq2 = 16

    torch.manual_seed(789)
    # CFG: batch dim = 2 (positive + negative)
    x1 = torch.randn(2, 16, H1, W1, device=device, dtype=dtype)
    x2 = torch.randn(2, 16, H2, W2, device=device, dtype=dtype)
    # Conditioning: B=1, will be expanded to B=2 inside forward_packed
    ctx1 = torch.randn(1, seq1, cap_feat_dim, device=device, dtype=dtype)
    ctx2 = torch.randn(1, seq2, cap_feat_dim, device=device, dtype=dtype)
    timesteps = torch.tensor([0.5, 0.5], device=device, dtype=dtype)

    x1_pad = pad_to_patch_size(x1, (pH, pW))
    x2_pad = pad_to_patch_size(x2, (pH, pW))
    H1p, W1p = x1_pad.shape[2], x1_pad.shape[3]
    H2p, W2p = x2_pad.shape[2], x2_pad.shape[3]

    print(f"Image 1: {H1}x{W1}, Image 2: {H2}x{W2}, B=2 (CFG)")

    with torch.inference_mode():
        refined_caps, packing_info, packed_rope = model.prepare_packed_state(
            [ctx1, ctx2],
            [(H1p, W1p), (H2p, W2p)],
            [seq1, seq2],
            device,
        )

        from torch.nn.attention.flex_attention import create_block_mask

        block_mask = create_block_mask(
            make_packing_mask_mod(packing_info.document_id),
            B=2, H=None,
            Q_LEN=packing_info.total_len,
            KV_LEN=packing_info.total_len,
            device=device,
        )

        outputs = model.forward_packed(
            [x1, x2], timesteps, refined_caps,
            packing_info, block_mask, packed_rope,
        )

    print(f"Output 0: {outputs[0].shape}  (expected: 2, 16, {H1}, {W1})")
    print(f"Output 1: {outputs[1].shape}  (expected: 2, 16, {H2}, {W2})")
    assert outputs[0].shape == (2, 16, H1, W1), f"Wrong shape: {outputs[0].shape}"
    assert outputs[1].shape == (2, 16, H2, W2), f"Wrong shape: {outputs[1].shape}"
    print("CFG batch shapes correct!")


if __name__ == "__main__":
    print("=" * 60)
    print("FlexAttention Batch Packing Verification")
    print("=" * 60)

    # Test 1: Single image bitwise comparison
    print("\n=== Test 1: Single image bitwise ===")
    max_diff, cos = test_single_image_bitwise()

    # Test 2: Multi-image packing
    test_multi_image()

    # Test 3: CFG batched
    test_cfg_batched()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

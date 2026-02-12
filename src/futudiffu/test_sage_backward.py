"""Test SageAttention2 backward pass against PyTorch SDPA reference.

Usage:
    .venv/Scripts/python.exe -c "import sys; sys.path.insert(0, r'F:\\dox\\repos\\ai\\futudiffu\\src'); from futudiffu.test_sage_backward import run_tests; run_tests()"

Or via bootstrap:
    .venv/Scripts/python.exe bootstrap.py run "from futudiffu.test_sage_backward import run_tests; run_tests()"
"""

import math
import sys

import torch
import torch.nn.functional as F


def test_forward_lse_consistency():
    """Verify that forward-with-LSE produces the same output as forward-without-LSE."""
    from .sage_attention import sage_attn_forward, sage_attn_forward_with_lse

    B, H, N, D = 1, 4, 512, 128
    q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")

    out_no_lse = sage_attn_forward(q, k, v)
    out_lse, lse = sage_attn_forward_with_lse(q, k, v)

    # Should be bitwise identical (same kernel logic, just extra LSE store)
    max_diff = (out_no_lse.float() - out_lse.float()).abs().max().item()
    cos = F.cosine_similarity(
        out_no_lse.float().flatten().unsqueeze(0),
        out_lse.float().flatten().unsqueeze(0),
    ).item()
    print(f"  forward LSE consistency: max_diff={max_diff:.2e}, cosine={cos:.6f}")
    print(f"  LSE shape: {lse.shape}, dtype: {lse.dtype}, range: [{lse.min().item():.2f}, {lse.max().item():.2f}]")
    assert max_diff < 1e-6, f"Forward outputs diverged: max_diff={max_diff}"
    assert lse.shape == (B * H, N), f"LSE shape mismatch: {lse.shape}"
    assert lse.dtype == torch.float32, f"LSE dtype mismatch: {lse.dtype}"
    print("  PASS")


def test_backward_small():
    """Compare backward gradients against PyTorch SDPA on small shapes."""
    from .sage_attention import sage_attn_op

    B, H, N, D = 1, 4, 512, 128
    sm_scale = 1.0 / math.sqrt(D)

    # Create inputs with requires_grad
    q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)

    # Sage forward + backward
    out_sage, _ = sage_attn_op(q, k, v, sm_scale)
    loss_sage = out_sage.sum()
    loss_sage.backward()
    dq_sage = q.grad.clone()
    dk_sage = k.grad.clone()
    dv_sage = v.grad.clone()

    # Reference: PyTorch SDPA forward + backward
    q.grad, k.grad, v.grad = None, None, None
    out_ref = F.scaled_dot_product_attention(q, k, v)
    loss_ref = out_ref.sum()
    loss_ref.backward()
    dq_ref = q.grad.clone()
    dk_ref = k.grad.clone()
    dv_ref = v.grad.clone()

    # Forward comparison
    fwd_cos = F.cosine_similarity(
        out_sage.float().flatten().unsqueeze(0),
        out_ref.float().flatten().unsqueeze(0),
    ).item()
    print(f"  forward cosine: {fwd_cos:.6f}")

    # Backward comparison
    results = {}
    for name, sage, ref in [("dQ", dq_sage, dq_ref), ("dK", dk_sage, dk_ref), ("dV", dv_sage, dv_ref)]:
        cos = F.cosine_similarity(
            sage.float().flatten().unsqueeze(0),
            ref.float().flatten().unsqueeze(0),
        ).item()
        max_err = (sage.float() - ref.float()).abs().max().item()
        rel_err = max_err / (ref.float().abs().max().item() + 1e-12)
        print(f"  {name}: cosine={cos:.6f}, max_abs_err={max_err:.4e}, rel_err={rel_err:.4e}")
        results[name] = cos

    # Target: cosine > 0.95 for all (allowing for FP8 forward divergence)
    for name, cos in results.items():
        assert cos > 0.95, f"{name} cosine similarity {cos:.6f} is below threshold 0.95"
    print("  PASS")


def test_backward_medium():
    """Test backward on medium shapes matching the diffusion model."""
    from .sage_attention import sage_attn_op

    B, H, N, D = 2, 30, 4288, 128
    sm_scale = 1.0 / math.sqrt(D)

    q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)

    # Just verify it runs without error or OOM
    out, lse = sage_attn_op(q, k, v, sm_scale)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None, "dQ is None"
    assert k.grad is not None, "dK is None"
    assert v.grad is not None, "dV is None"
    assert q.grad.shape == q.shape, f"dQ shape mismatch: {q.grad.shape}"
    assert not torch.isnan(q.grad).any(), "dQ contains NaN"
    assert not torch.isnan(k.grad).any(), "dK contains NaN"
    assert not torch.isnan(v.grad).any(), "dV contains NaN"

    print(f"  shapes: dQ={q.grad.shape}, dK={k.grad.shape}, dV={v.grad.shape}")
    print(f"  no NaN, no OOM")
    print("  PASS")


def test_d_prepass():
    """Validate the D pre-pass kernel: D_i = rowsum(dO_i * O_i)."""
    import triton
    from .sage_kernels import _sage_attn_d_prepass

    BH, N, D = 4, 512, 128
    stride_z = N * D

    dO = torch.randn(BH, N, D, dtype=torch.bfloat16, device="cuda")
    O = torch.randn(BH, N, D, dtype=torch.bfloat16, device="cuda")
    delta = torch.empty(BH, N, dtype=torch.float32, device="cuda")

    BLOCK_M = 128
    grid = (triton.cdiv(N, BLOCK_M), BH)
    _sage_attn_d_prepass[grid](
        dO, O, delta,
        stride_z, N,
        D_HEAD=D, BLOCK_M=BLOCK_M,
        num_warps=4, num_stages=1,
    )

    # Reference
    ref = (dO.float() * O.float()).sum(dim=-1)  # (BH, N)

    max_err = (delta - ref).abs().max().item()
    cos = F.cosine_similarity(
        delta.flatten().unsqueeze(0),
        ref.flatten().unsqueeze(0),
    ).item()
    print(f"  D prepass: max_err={max_err:.4e}, cosine={cos:.6f}")
    assert cos > 0.999, f"D prepass cosine {cos:.6f} below threshold"
    print("  PASS")


def run_tests():
    """Run all SageAttention backward tests."""
    print("=" * 60)
    print("SageAttention2 Backward Pass Tests")
    print("=" * 60)

    tests = [
        ("Forward LSE consistency", test_forward_lse_consistency),
        ("D pre-pass kernel", test_d_prepass),
        ("Backward (small: B=1 H=4 N=512)", test_backward_small),
        ("Backward (medium: B=2 H=30 N=4288)", test_backward_medium),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n[TEST] {name}")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()

"""Integration test: verify multi_lora_op produces identical output to the
cat-based path in LoRALinear, then verify the Triton kernel works as the
sole implementation inside LoRALinear.forward.

Tests:
  1. Triton kernel vs cat-based reference (same weights, same scales)
  2. Scale routing: one adapter on, one off
  3. Registered buffer scale mechanism through LoRALinear.forward
  4. Backward through multi_lora_op (gradient flow for training)
  5. Single-adapter fast path still works
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from futudiffu.lora import (
    LoRAAdapter,
    LoRALinear,
    inject_lora,
    set_lora_scale,
    clear_lora_scale,
)
from futudiffu.lora_kernels import multi_lora_forward, multi_lora_forward_ref, multi_lora_op


DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


def test_triton_vs_ref():
    """Test 1: Triton kernel matches PyTorch reference."""
    print("Test 1: Triton kernel vs PyTorch reference")
    B, S, IN, OUT = 2, 128, 3840, 3840
    N, R = 3, 8

    x = torch.randn(B, S, IN, dtype=DTYPE, device=DEVICE)
    A_all = torch.randn(N, R, IN, dtype=DTYPE, device=DEVICE) * 0.01
    B_all = torch.randn(N, OUT, R, dtype=DTYPE, device=DEVICE) * 0.01
    # scale includes alpha/rank pre-folded
    scale_all = torch.tensor([
        [2.0, 0.0, 1.5],  # batch 0: adapter 0 on, 1 off, 2 half
        [0.0, 2.0, 0.0],  # batch 1: only adapter 1
    ], dtype=torch.float32, device=DEVICE)

    out_triton = multi_lora_forward(x, A_all, B_all, scale_all)
    out_ref = multi_lora_forward_ref(x, A_all, B_all, scale_all)

    cos = F.cosine_similarity(out_triton.flatten().float(), out_ref.flatten().float(), dim=0)
    max_diff = (out_triton.float() - out_ref.float()).abs().max().item()
    print(f"  cos={cos:.6f}, max_diff={max_diff:.6f}")
    assert cos > 0.999, f"cos too low: {cos}"
    print("  PASS")


def test_triton_vs_lora_linear():
    """Test 2: Triton kernel matches LoRALinear.forward for N>1."""
    print("\nTest 2: Triton kernel matches LoRALinear.forward")
    IN, OUT = 256, 512
    B, S = 2, 64
    R = 8

    base = nn.Linear(IN, OUT, bias=False).to(DTYPE).to(DEVICE)
    ll = LoRALinear(base)
    a1 = ll.add_adapter("a1", rank=R, alpha=16.0)
    a2 = ll.add_adapter("a2", rank=R, alpha=16.0)

    x = torch.randn(B, S, IN, dtype=DTYPE, device=DEVICE)

    # Get LoRALinear output (uses the Triton kernel now)
    out_ll = ll(x)

    # Manually compute: base + triton_delta
    base_out = base(x)
    A_all = torch.stack([a1.lora_A.data, a2.lora_A.data])  # (2, R, IN)
    B_all = torch.stack([a1.lora_B.data, a2.lora_B.data])  # (2, OUT, R)
    # Default scale: broadcast 1.0, alpha/rank pre-folded
    alpha_rank = a1.scale  # alpha/rank = 16/8 = 2.0
    scale_all = torch.full((B, 2), alpha_rank, dtype=torch.float32, device=DEVICE)
    delta = multi_lora_forward(x, A_all, B_all, scale_all)
    out_manual = base_out + delta

    cos = F.cosine_similarity(out_ll.flatten().float(), out_manual.flatten().float(), dim=0)
    print(f"  cos={cos:.6f}")
    assert cos > 0.999, f"cos too low: {cos}"
    print("  PASS")


def test_scale_routing():
    """Test 3: Per-batch scale routing through LoRALinear."""
    print("\nTest 3: Scale routing through registered buffers")
    IN, OUT = 256, 512
    B, S = 2, 32
    R = 8

    base = nn.Linear(IN, OUT, bias=False).to(DTYPE).to(DEVICE)
    ll = LoRALinear(base)
    a1 = ll.add_adapter("a1", rank=R, alpha=16.0)
    a2 = ll.add_adapter("a2", rank=R, alpha=16.0)
    # Make B non-zero for observable effect
    nn.init.normal_(a1.lora_B, std=0.1)
    nn.init.normal_(a2.lora_B, std=0.1)

    x = torch.randn(B, S, IN, dtype=DTYPE, device=DEVICE)

    # Baseline: both adapters on (default scale=1.0)
    out_both = ll(x).clone()

    # Set a1 to [1, 0] (on for batch 0, off for batch 1)
    set_lora_scale(ll, torch.tensor([1.0, 0.0], dtype=DTYPE, device=DEVICE), "a1")
    # Set a2 to [0, 1] (off for batch 0, on for batch 1)
    set_lora_scale(ll, torch.tensor([0.0, 1.0], dtype=DTYPE, device=DEVICE), "a2")
    out_routed = ll(x).clone()

    # Batch 0 should only have a1's contribution
    # Batch 1 should only have a2's contribution
    # Neither should match the "both on" case
    base_out = base(x)

    # Check batch 0: only a1
    a1_delta = (x[0:1] @ a1.lora_A.t()) @ a1.lora_B.t() * a1.scale
    expected_b0 = base_out[0:1] + a1_delta
    cos_b0 = F.cosine_similarity(
        out_routed[0].flatten().float(), expected_b0[0].flatten().float(), dim=0)

    # Check batch 1: only a2
    a2_delta = (x[1:2] @ a2.lora_A.t()) @ a2.lora_B.t() * a2.scale
    expected_b1 = base_out[1:2] + a2_delta
    cos_b1 = F.cosine_similarity(
        out_routed[1].flatten().float(), expected_b1[0].flatten().float(), dim=0)

    print(f"  batch 0 (a1 only): cos={cos_b0:.6f}")
    print(f"  batch 1 (a2 only): cos={cos_b1:.6f}")
    assert cos_b0 > 0.999, f"batch 0 cos too low: {cos_b0}"
    assert cos_b1 > 0.999, f"batch 1 cos too low: {cos_b1}"

    # Verify scale=0 actually kills the adapter
    set_lora_scale(ll, torch.tensor([0.0], dtype=DTYPE, device=DEVICE), "a1")
    set_lora_scale(ll, torch.tensor([0.0], dtype=DTYPE, device=DEVICE), "a2")
    out_off = ll(x)
    cos_base = F.cosine_similarity(
        out_off.flatten().float(), base_out.flatten().float(), dim=0)
    print(f"  both off vs base: cos={cos_base:.6f}")
    assert cos_base > 0.9999, f"scale=0 didn't kill adapters: cos={cos_base}"

    clear_lora_scale(ll, "a1")
    clear_lora_scale(ll, "a2")
    print("  PASS")


def test_backward():
    """Test 4: Backward through multi_lora_op matches reference autograd.

    Compare gradients from the custom_op backward against PyTorch's autograd
    through multi_lora_forward_ref (which uses standard ops).
    """
    print("\nTest 4: Backward through multi_lora_op vs reference autograd")
    IN, OUT = 128, 256
    B, S = 2, 16
    N, R = 2, 8
    alpha = 16.0

    x = torch.randn(B, S, IN, dtype=DTYPE, device=DEVICE)

    # Shared weight init
    A_init = torch.randn(N, R, IN, dtype=DTYPE, device=DEVICE) * 0.01
    B_init = torch.randn(N, OUT, R, dtype=DTYPE, device=DEVICE) * 0.01
    scale_data = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32, device=DEVICE)

    # --- Path 1: custom_op backward ---
    A_all_op = A_init.clone().requires_grad_(True)
    B_all_op = B_init.clone().requires_grad_(True)
    delta_op = multi_lora_op(x, A_all_op, B_all_op, scale_data, N, R)
    loss_op = delta_op.float().pow(2).sum()
    loss_op.backward()

    # --- Path 2: reference autograd ---
    A_all_ref = A_init.clone().requires_grad_(True)
    B_all_ref = B_init.clone().requires_grad_(True)
    delta_ref = multi_lora_forward_ref(x, A_all_ref, B_all_ref, scale_data)
    loss_ref = delta_ref.float().pow(2).sum()
    loss_ref.backward()

    # Compare
    for name, g_op, g_ref in [
        ("A_all", A_all_op.grad, A_all_ref.grad),
        ("B_all", B_all_op.grad, B_all_ref.grad),
    ]:
        cos = F.cosine_similarity(g_op.flatten().float(), g_ref.flatten().float(), dim=0)
        rel_err = (g_op.float() - g_ref.float()).abs().max() / (g_ref.float().abs().max() + 1e-8)
        print(f"  {name}: cos={cos:.6f}, rel_err={rel_err:.6f}")
        assert cos > 0.99, f"{name} gradient cos too low: {cos}"

    # Also test through LoRALinear
    base = nn.Linear(IN, OUT, bias=False).to(DTYPE).to(DEVICE)
    ll = LoRALinear(base)
    a1 = ll.add_adapter("a1", rank=R, alpha=alpha, init_b_std=0.01)
    a2 = ll.add_adapter("a2", rank=R, alpha=alpha, init_b_std=0.01)
    set_lora_scale(ll, torch.tensor([1.0, 0.0], dtype=DTYPE, device=DEVICE), "a1")
    set_lora_scale(ll, torch.tensor([0.0, 1.0], dtype=DTYPE, device=DEVICE), "a2")

    a1.lora_A.requires_grad_(True)
    a1.lora_B.requires_grad_(True)
    a2.lora_A.requires_grad_(True)
    a2.lora_B.requires_grad_(True)

    out = ll(x)
    loss = out.float().pow(2).sum()
    loss.backward()

    n_with_grad = sum(1 for p in [a1.lora_A, a1.lora_B, a2.lora_A, a2.lora_B]
                      if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  LoRALinear backward: {n_with_grad}/4 params with non-zero grad")
    assert n_with_grad == 4, f"Expected 4, got {n_with_grad}"
    print("  PASS")


def test_single_adapter_fast_path():
    """Test 5: Single adapter fast path (n=1) still works."""
    print("\nTest 5: Single adapter fast path")
    IN, OUT = 256, 512
    B, S = 2, 32
    R = 8

    base = nn.Linear(IN, OUT, bias=False).to(DTYPE).to(DEVICE)
    ll = LoRALinear(base)
    a1 = ll.add_adapter("a1", rank=R, alpha=16.0, init_b_std=0.01)

    x = torch.randn(B, S, IN, dtype=DTYPE, device=DEVICE)

    # With default scale (1.0)
    out = ll(x)
    base_out = base(x)
    delta = (x.to(DTYPE) @ a1.lora_A.t()) @ a1.lora_B.t() * a1.scale
    expected = base_out + delta.to(base_out.dtype)

    # Scale the single adapter
    # Use lora_scale buffer directly
    a1.lora_scale.fill_(0.0)
    out_off = ll(x)
    cos_off = F.cosine_similarity(
        out_off.flatten().float(), base_out.flatten().float(), dim=0)
    print(f"  scale=0 vs base: cos={cos_off:.6f}")
    assert cos_off > 0.9999, f"single adapter scale=0 broken: cos={cos_off}"

    a1.lora_scale.fill_(1.0)
    out_on = ll(x)
    cos_on = F.cosine_similarity(
        out_on.flatten().float(), expected.flatten().float(), dim=0)
    print(f"  scale=1 vs expected: cos={cos_on:.6f}")
    assert cos_on > 0.999, f"single adapter output wrong: cos={cos_on}"

    # Backward
    a1.lora_A.requires_grad_(True)
    a1.lora_B.requires_grad_(True)
    out2 = ll(x)
    out2.float().pow(2).sum().backward()
    a_grad = a1.lora_A.grad is not None and a1.lora_A.grad.abs().sum() > 0
    b_grad = a1.lora_B.grad is not None and a1.lora_B.grad.abs().sum() > 0
    print(f"  backward: A has grad={a_grad}, B has grad={b_grad}")
    assert a_grad and b_grad
    print("  PASS")


def test_inject_and_forward():
    """Test 6: inject_lora + forward on a small model with multiple adapters."""
    print("\nTest 6: inject_lora on small model")

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    "attention": nn.ModuleDict({
                        "qkv": nn.Linear(256, 768, bias=False),
                        "out": nn.Linear(256, 256, bias=False),
                    }),
                    "feed_forward": nn.ModuleDict({
                        "w1": nn.Linear(256, 512, bias=False),
                        "w2": nn.Linear(512, 256, bias=False),
                        "w3": nn.Linear(256, 512, bias=False),
                    }),
                })
                for _ in range(2)
            ])

        def forward(self, x):
            for layer in self.layers:
                q = layer["attention"]["qkv"](x)
                x = x + layer["attention"]["out"](q[:, :, :256])
                h = F.silu(layer["feed_forward"]["w1"](x)) * layer["feed_forward"]["w3"](x)
                x = x + layer["feed_forward"]["w2"](h)
            return x

    model = TinyModel().to(DTYPE).to(DEVICE)

    # Inject two adapters
    injected_r = inject_lora(model, name="rtheta", rank=8, alpha=16.0,
                              layer_indices={1})
    injected_p = inject_lora(model, name="ptheta", rank=8, alpha=16.0,
                              init_b_std=0.01)

    print(f"  rtheta: {len(injected_r)} modules")
    print(f"  ptheta: {len(injected_p)} modules")

    x = torch.randn(2, 16, 256, dtype=DTYPE, device=DEVICE)

    # Forward works
    out = model(x)
    print(f"  forward output shape: {out.shape}")

    # Set per-batch routing: batch[0]=ptheta on, batch[1]=ptheta off
    set_lora_scale(model, torch.tensor([1.0, 0.0], dtype=DTYPE, device=DEVICE),
                   adapter_name="ptheta")
    out_routed = model(x)

    # Should be different from uniform scale
    cos = F.cosine_similarity(out.flatten().float(), out_routed.flatten().float(), dim=0)
    print(f"  uniform vs routed: cos={cos:.6f}")
    # They should differ because batch[1] has ptheta off in routed case
    assert cos < 0.9999, f"Routing had no effect: cos={cos}"

    # Backward with routing active
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad = None
    out3 = model(x)
    out3.float().pow(2).sum().backward()
    n_grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  backward: {n_grads} params with non-zero grad")
    assert n_grads > 0
    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("  Multi-LoRA Kernel Integration Test")
    print("=" * 60)
    print()

    test_triton_vs_ref()
    test_triton_vs_lora_linear()
    test_scale_routing()
    test_backward()
    test_single_adapter_fast_path()
    test_inject_and_forward()

    print()
    print("=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)

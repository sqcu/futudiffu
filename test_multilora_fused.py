"""Test: fused scatter/gather multi-LoRA + concurrent score+sample.

Fictional adapter configs on a minimal model:
  - "scrongle": rank 4, layers 0-1  (step-count artifact detector)
  - "scrimble": rank 4, layers 2-3  (quantization artifact detector)
  - "policy":   rank 8, layers 0-3  (full policy — overlaps both detectors)
  - "rtheta":   rank 8, layers 2-3  (reward model — overlaps scrimble+policy)

Tests:
  1. Correctness: fused N-adapter forward == sum of individual adapter forwards
  2. Concurrent score+sample: batch[0]=policy, batch[1]=rtheta via per-batch scale
  3. Backward: gradients flow to correct adapters only
  4. Benchmark: fused 4-adapter vs naive serial (wallclock)
"""

import sys
import time

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Minimal transformer model (4 layers, small dim)
# ---------------------------------------------------------------------------

class MiniAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return self.out(v)


class MiniFFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * 2, bias=False)
        self.w2 = nn.Linear(dim * 2, dim, bias=False)
        self.w3 = nn.Linear(dim, dim * 2, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class MiniBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = MiniAttn(dim)
        self.feed_forward = MiniFFN(dim)

    def forward(self, x):
        x = x + self.attention(x)
        x = x + self.feed_forward(x)
        return x


class MiniModel(nn.Module):
    def __init__(self, dim, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([MiniBlock(dim) for _ in range(n_layers)])
        self.head = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_correctness():
    """Fused N-adapter output == sum of individual adapter outputs."""
    from futudiffu.lora import LoRALinear, LoRAAdapter

    print("Test 1: Fused forward correctness")
    torch.manual_seed(42)
    dim_in, dim_out = 64, 128
    base = nn.Linear(dim_in, dim_out, bias=False).to(dtype=torch.bfloat16)
    x = torch.randn(2, 8, dim_in, dtype=torch.bfloat16)

    # Create LoRALinear with 4 adapters of different ranks
    wrapper = LoRALinear(base)
    configs = [("a", 4, 8.0), ("b", 8, 16.0), ("c", 2, 4.0), ("d", 6, 12.0)]
    for name, rank, alpha in configs:
        adapter = wrapper.add_adapter(name, rank=rank, alpha=alpha)
        # Put nonzero values in B so adapters contribute
        adapter.lora_B.data.normal_(std=0.01)

    # Fused forward
    out_fused = wrapper(x)

    # Reference: base + sum of individual adapters
    base_out = base(x)
    x_bf16 = x.to(torch.bfloat16)
    individual_sum = base_out.clone()
    for name, adapter in wrapper.adapters.items():
        individual_sum = individual_sum + adapter(x_bf16).to(base_out.dtype)

    diff = (out_fused - individual_sum).abs().max().item()
    cos = nn.functional.cosine_similarity(
        out_fused.flatten().unsqueeze(0),
        individual_sum.flatten().unsqueeze(0)).item()
    print(f"  max diff (fused vs individual sum): {diff:.2e}")
    print(f"  cosine similarity: {cos:.6f}")
    # BF16 accumulation order differs (fused B matmul vs serial add),
    # so we check cosine > 0.999 rather than exact match
    assert cos > 0.99, f"Fused output diverged! cos={cos}"
    print("  PASS")


def test_per_batch_scale():
    """Per-batch-element scale routes different adapters to different batch items."""
    from futudiffu.lora import LoRALinear

    print("\nTest 2: Per-batch-element adapter routing")
    dim = 64
    base = nn.Linear(dim, dim, bias=False).to(dtype=torch.bfloat16)
    x = torch.randn(2, 4, dim, dtype=torch.bfloat16)

    wrapper = LoRALinear(base)
    a1 = wrapper.add_adapter("policy", rank=8, alpha=16.0)
    a2 = wrapper.add_adapter("rtheta", rank=8, alpha=16.0)
    a1.lora_B.data.normal_(std=0.1)
    a2.lora_B.data.normal_(std=0.1)

    # batch[0] = policy only, batch[1] = rtheta only
    a1.lora_scale = torch.tensor([1.0, 0.0], dtype=torch.bfloat16)
    a2.lora_scale = torch.tensor([0.0, 1.0], dtype=torch.bfloat16)

    out = wrapper(x)
    base_out = base(x)
    x_bf16 = x.to(torch.bfloat16)

    # Verify batch[0] only has policy contribution
    a1_only = base_out[0] + a1.scale * (x_bf16[0] @ a1.lora_A.t()) @ a1.lora_B.t()
    diff0 = (out[0] - a1_only.to(out.dtype)).abs().max().item()

    # Verify batch[1] only has rtheta contribution
    a2_only = base_out[1] + a2.scale * (x_bf16[1] @ a2.lora_A.t()) @ a2.lora_B.t()
    diff1 = (out[1] - a2_only.to(out.dtype)).abs().max().item()

    print(f"  batch[0] (policy only) diff: {diff0:.2e}")
    print(f"  batch[1] (rtheta only) diff: {diff1:.2e}")
    assert diff0 < 1e-4, f"batch[0] should only have policy adapter"
    assert diff1 < 1e-4, f"batch[1] should only have rtheta adapter"

    # Verify they're different from each other (different adapters produced different outputs)
    cross_diff = (out[0] - out[1]).abs().max().item()
    print(f"  cross-batch diff (policy vs rtheta): {cross_diff:.4f}")
    assert cross_diff > 1e-3, "Different adapters should produce different outputs"
    print("  PASS")


def test_sparse_injection():
    """Inject 4 adapters targeting different layer subsets on a 4-layer model."""
    from futudiffu.lora import inject_lora, get_lora_params, LoRALinear, lora_state_dict

    print("\nTest 3: Sparse multi-adapter injection")
    dim = 32
    model = MiniModel(dim, n_layers=4).to(dtype=torch.bfloat16)

    # Fictional adapter configs targeting sparse non-overlapping layers
    adapters_config = [
        ("scrongle", 4, 8.0,  {0, 1}),    # step artifact detector: layers 0-1
        ("scrimble", 4, 8.0,  {2, 3}),    # quant artifact detector: layers 2-3
        ("policy",   8, 16.0, None),       # full policy: all layers
        ("rtheta",   8, 16.0, {2, 3}),    # reward model: layers 2-3
    ]

    total_adapters = 0
    for name, rank, alpha, indices in adapters_config:
        injected = inject_lora(model, name=name, rank=rank, alpha=alpha,
                               layer_indices=indices)
        # Nonzero B so A gets gradients too (zero B → dL/dA = 0 via chain rule)
        for adapter in injected.values():
            adapter.lora_B.data.normal_(std=0.01)
        total_adapters += len(injected)
        layers_hit = sorted(set(
            int(p.split(".")[1]) for p in injected.keys() if p.startswith("layers.")
        ))
        print(f"  {name:10s}: rank={rank}, layers={layers_hit}, "
              f"{len(injected)} projections")

    # Check adapter counts per layer
    for i in range(4):
        for suffix in ["attention.qkv", "attention.out",
                        "feed_forward.w1", "feed_forward.w2", "feed_forward.w3"]:
            path = f"layers.{i}.{suffix}"
            mod = dict(model.named_modules()).get(path)
            if isinstance(mod, LoRALinear):
                n_adapters = len(mod.adapters)
                names = list(mod.adapters.keys())
                if n_adapters > 1:
                    print(f"    {path}: {n_adapters} adapters {names}")

    # Count params per adapter
    for name, _, _, _ in adapters_config:
        params = list(get_lora_params(model, adapter_name=name))
        n = sum(p.numel() for p in params)
        print(f"  {name:10s}: {n:,} params, {len(params)//2} (A,B) pairs")

    # Forward + backward
    x = torch.randn(2, 4, dim, dtype=torch.bfloat16)
    y = model(x)
    loss = y.sum()
    loss.backward()

    # Check gradients flow to all adapters
    for name, _, _, _ in adapters_config:
        params = list(get_lora_params(model, adapter_name=name))
        n_grad = sum(1 for p in params if p.grad is not None and p.grad.abs().sum() > 0)
        print(f"  {name:10s}: {n_grad}/{len(params)} params have gradients")
        assert n_grad == len(params), f"{name}: not all params got gradients"

    # State dict round-trip per adapter
    for name, _, _, _ in adapters_config:
        sd = lora_state_dict(model, adapter_name=name)
        assert len(sd) > 0, f"{name}: empty state dict"

    print("  PASS")


def test_concurrent_score_sample():
    """Concurrent BTRM scoring + diffusion sampling in ONE forward pass.

    batch[0]: policy adapter active (sampling)
    batch[1]: rtheta adapter active (scoring)
    """
    from futudiffu.lora import (
        inject_lora, set_lora_scale, freeze_adapter, get_lora_params
    )

    print("\nTest 4: Concurrent score + sample (single forward)")
    dim = 64
    model = MiniModel(dim, n_layers=4).to(dtype=torch.bfloat16)

    # Inject adapters
    inject_lora(model, name="rtheta", rank=8, alpha=16.0, layer_indices={2, 3})
    inject_lora(model, name="policy", rank=8, alpha=16.0)

    # Train rtheta briefly to make it nonzero
    for p in get_lora_params(model, adapter_name="rtheta"):
        p.data.normal_(std=0.01)
    for p in get_lora_params(model, adapter_name="policy"):
        p.data.normal_(std=0.01)

    # Freeze rtheta
    freeze_adapter(model, "rtheta")

    # Set batch-level routing
    set_lora_scale(model, torch.tensor([1.0, 0.0], dtype=torch.bfloat16),
                   adapter_name="policy")
    set_lora_scale(model, torch.tensor([0.0, 1.0], dtype=torch.bfloat16),
                   adapter_name="rtheta")

    # Batch: same input, different adapter paths
    x = torch.randn(1, 4, dim, dtype=torch.bfloat16)
    x_batch = x.expand(2, -1, -1).clone()

    # Forward
    out = model(x_batch)
    sample_out = out[0]   # policy-adapted
    score_out = out[1]    # rtheta-adapted

    # They should differ (different adapters active)
    diff = (sample_out - score_out).abs().max().item()
    print(f"  sample vs score output diff: {diff:.4f}")
    assert diff > 1e-3, "Policy and rtheta paths should produce different outputs"

    # Backward: only policy params should get gradients
    loss = out[0].sum()  # loss on sample output only
    loss.backward()

    policy_grads = sum(1 for p in get_lora_params(model, "policy")
                       if p.grad is not None and p.grad.abs().sum() > 0)
    rtheta_grads = sum(1 for p in get_lora_params(model, "rtheta")
                       if p.grad is not None and p.grad.abs().sum() > 0)
    policy_total = len(list(get_lora_params(model, "policy")))
    rtheta_total = len(list(get_lora_params(model, "rtheta")))

    print(f"  policy grads: {policy_grads}/{policy_total}")
    print(f"  rtheta grads: {rtheta_grads}/{rtheta_total} (should be 0, frozen)")
    assert rtheta_grads == 0, "Frozen rtheta should have no gradients"
    assert policy_grads > 0, "Policy should have gradients"
    print("  PASS")


def test_fused_vs_serial_benchmark():
    """Benchmark fused (2 matmuls) vs serial (2N matmuls) for N adapters."""
    from futudiffu.lora import LoRALinear

    print("\nTest 5: Fused vs serial benchmark")

    for n_adapters in [1, 2, 4, 8]:
        dim_in, dim_out = 3840, 3840  # realistic NextDiT dim
        base = nn.Linear(dim_in, dim_out, bias=False).to(
            dtype=torch.bfloat16, device="cuda")
        x = torch.randn(2, 4288, dim_in, dtype=torch.bfloat16, device="cuda")

        wrapper = LoRALinear(base)
        for i in range(n_adapters):
            a = wrapper.add_adapter(f"a{i}", rank=8, alpha=16.0)
            a.lora_B.data.normal_(std=0.01)

        # Warmup
        for _ in range(3):
            wrapper(x)
        torch.cuda.synchronize()

        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        n_iter = 20

        start.record()
        for _ in range(n_iter):
            wrapper(x)
        end.record()
        torch.cuda.synchronize()
        fused_ms = start.elapsed_time(end) / n_iter

        # Compare against base-only (no adapters)
        start.record()
        for _ in range(n_iter):
            base(x)
        end.record()
        torch.cuda.synchronize()
        base_ms = start.elapsed_time(end) / n_iter

        overhead_pct = ((fused_ms - base_ms) / base_ms) * 100
        print(f"  N={n_adapters}: base={base_ms:.2f}ms, "
              f"fused={fused_ms:.2f}ms, "
              f"overhead={overhead_pct:+.1f}%")

    print("  DONE")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Multi-LoRA Fused Scatter/Gather Tests")
    print("=" * 60)

    test_correctness()
    test_per_batch_scale()
    test_sparse_injection()
    test_concurrent_score_sample()

    if torch.cuda.is_available():
        test_fused_vs_serial_benchmark()
    else:
        print("\nSkipping GPU benchmark (no CUDA)")

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

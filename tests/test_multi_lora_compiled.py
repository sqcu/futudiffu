"""Integration test: MultiLoRA on torch.compile'd models.

Validates the full lifecycle that _run_policy_intervention exercises:
  1. Compile model (layers become OptimizedModule with _orig_mod paths)
  2. install_multi_lora finds targets through _orig_mod
  3. load_adapter matches checkpoint keys to canonical (non-_orig_mod) names
  4. Second install_multi_lora adds adapter to existing wrappers (no re-wrap)
  5. save_adapter produces canonical keys (no _orig_mod in output)
  6. get_adapter_params keys are canonical
"""

import gc
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

from src_ii.multi_lora import (
    MultiLoRALinear,
    install_multi_lora,
    load_adapter,
    save_adapter,
    get_adapter_params,
    _strip_orig_mod,
)

# Project-local scratch dir (Windows tempdir + safetensors mmap = PermissionError)
SCRATCH = Path(__file__).resolve().parent.parent / "_test_scratch_lora_compiled"


def setup_scratch():
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)


def teardown_scratch():
    gc.collect()
    shutil.rmtree(SCRATCH, ignore_errors=True)


# ---------------------------------------------------------------------------
# Minimal model that mirrors ZImageRLAIF structure after compile
# ---------------------------------------------------------------------------

class FakeAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.out(self.qkv(x)[..., :x.shape[-1]])


class FakeFeedForward(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.w2(x)


class FakeBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attention = FakeAttention(dim)
        self.feed_forward = FakeFeedForward(dim)

    def forward(self, x):
        return x + self.attention(x) + self.feed_forward(x)


class FakeModel(nn.Module):
    def __init__(self, dim: int = 32, n_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([FakeBlock(dim) for _ in range(n_layers)])
        # Excluded modules (should NOT get LoRA)
        self.score_proj = nn.Linear(dim, 2, bias=False)
        self.final_layer = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def compile_model(model: FakeModel) -> FakeModel:
    """Mimic ZImageRLAIF.compile_for_execution()."""
    for i in range(len(model.layers)):
        model.layers[i] = torch.compile(model.layers[i], mode="default")
    return model


def make_fake_checkpoint(model: FakeModel, adapter_name: str, out_dir: Path) -> Path:
    """Create a fake adapter checkpoint in old-format keys."""
    path = out_dir / f"{adapter_name}_adapter.safetensors"
    tensors = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear,)):
            canonical = _strip_orig_mod(name)
            if not canonical.startswith("layers."):
                continue
            if any(ex in canonical for ex in ("score_proj", "final_layer")):
                continue
            in_f = module.in_features
            out_f = module.out_features
            rank = 4
            # Old format: *.adapters.{name}.lora_{A,B}
            tensors[f"{canonical}.adapters.{adapter_name}.lora_A"] = torch.randn(rank, in_f)
            tensors[f"{canonical}.adapters.{adapter_name}.lora_B"] = torch.randn(out_f, rank)
    save_file(tensors, str(path))
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_strip_orig_mod():
    assert _strip_orig_mod("layers.0._orig_mod.attention.qkv") == "layers.0.attention.qkv"
    assert _strip_orig_mod("layers.0.attention.qkv") == "layers.0.attention.qkv"
    assert _strip_orig_mod("a._orig_mod.b._orig_mod.c") == "a.b.c"
    print("[PASS] _strip_orig_mod")


def test_install_on_compiled_model():
    """install_multi_lora should find targets through _orig_mod wrappers."""
    model = FakeModel()
    model = compile_model(model)

    configs = [{"name": "rtheta", "rank": 4, "alpha": 8.0}]
    wrappers = install_multi_lora(model, configs)

    assert len(wrappers) > 0, "No wrappers installed"

    # Verify wrappers are on the right modules (3 layers * 3 linears = 9)
    n_expected = 3 * 3  # layers * (qkv, out, w2)
    assert len(wrappers) == n_expected, f"Expected {n_expected} wrappers, got {len(wrappers)}"

    # Verify excluded modules were NOT wrapped
    assert not isinstance(model.score_proj, MultiLoRALinear)
    assert not isinstance(model.final_layer, MultiLoRALinear)

    # Verify each wrapper has the adapter
    for name, wrapper in wrappers.items():
        assert "rtheta" in wrapper.lora_A, f"Missing rtheta in {name}"
        assert wrapper.n_adapters == 1

    print(f"[PASS] install_on_compiled_model: {len(wrappers)} wrappers")


def test_load_adapter_compiled():
    """load_adapter should match canonical keys on a compiled model."""
    model = FakeModel()

    # Create checkpoint BEFORE compile (uses canonical names)
    ckpt_path = make_fake_checkpoint(model, "rtheta", SCRATCH)

    # Now compile and install
    model = compile_model(model)
    configs = [{"name": "rtheta", "rank": 4, "alpha": 8.0}]
    install_multi_lora(model, configs)

    n_loaded = load_adapter(model, "rtheta", str(ckpt_path))
    assert n_loaded > 0, f"Loaded 0 tensors"
    # 3 layers * 3 linears * 2 (A+B) = 18
    assert n_loaded == 18, f"Expected 18, got {n_loaded}"

    print(f"[PASS] load_adapter_compiled: {n_loaded} tensors loaded")


def test_second_install_adds_adapter():
    """Second install_multi_lora should add adapter to existing wrappers."""
    model = FakeModel()
    model = compile_model(model)

    # First install
    configs1 = [{"name": "rtheta", "rank": 4, "alpha": 8.0}]
    wrappers1 = install_multi_lora(model, configs1)

    # Second install with different adapter
    configs2 = [{"name": "policy_pinkify", "rank": 4, "alpha": 8.0}]
    wrappers2 = install_multi_lora(model, configs2)

    assert len(wrappers2) == len(wrappers1), "Different wrapper count on second install"

    # Each wrapper should now have both adapters
    for name, wrapper in wrappers2.items():
        assert "rtheta" in wrapper.lora_A, f"Missing rtheta in {name}"
        assert "policy_pinkify" in wrapper.lora_A, f"Missing policy_pinkify in {name}"
        assert wrapper.n_adapters == 2, f"Expected 2 adapters, got {wrapper.n_adapters}"

    print(f"[PASS] second_install_adds_adapter: {len(wrappers2)} wrappers, 2 adapters each")


def test_save_adapter_canonical_keys():
    """save_adapter should produce keys without _orig_mod."""
    model = FakeModel()
    model = compile_model(model)

    configs = [{"name": "rtheta", "rank": 4, "alpha": 8.0}]
    install_multi_lora(model, configs)

    path = str(SCRATCH / "test_save_adapter.safetensors")
    save_adapter(model, "rtheta", path)

    sd = load_file(path)
    for key in sd:
        assert "_orig_mod" not in key, f"_orig_mod in saved key: {key}"
        assert ".lora_A.rtheta" in key or ".lora_B.rtheta" in key, f"Bad key format: {key}"

    assert len(sd) == 18, f"Expected 18 tensors, got {len(sd)}"
    del sd
    gc.collect()

    print(f"[PASS] save_adapter_canonical_keys: 18 tensors, all canonical")


def test_get_adapter_params_canonical():
    """get_adapter_params should produce canonical keys."""
    model = FakeModel()
    model = compile_model(model)

    configs = [{"name": "rtheta", "rank": 4, "alpha": 8.0}]
    install_multi_lora(model, configs)

    params = get_adapter_params(model, "rtheta")
    for key in params:
        assert "_orig_mod" not in key, f"_orig_mod in param key: {key}"

    assert len(params) == 18, f"Expected 18 params, got {len(params)}"
    print(f"[PASS] get_adapter_params_canonical: {len(params)} params, all canonical")


def test_roundtrip_save_load_compiled():
    """Full roundtrip: install -> save -> new model -> compile -> install -> load."""
    # Model A: install adapter, modify weights, save
    model_a = FakeModel()
    model_a = compile_model(model_a)
    configs = [{"name": "rtheta", "rank": 4, "alpha": 8.0}]
    install_multi_lora(model_a, configs)

    # Set A matrices to known values
    for module in model_a.modules():
        if isinstance(module, MultiLoRALinear):
            module.lora_A["rtheta"].data.fill_(1.0)
            module.lora_B["rtheta"].data.fill_(2.0)

    path = str(SCRATCH / "roundtrip.safetensors")
    save_adapter(model_a, "rtheta", path)
    del model_a
    gc.collect()

    # Model B: fresh compile, install, load
    model_b = FakeModel()
    model_b = compile_model(model_b)
    install_multi_lora(model_b, configs)
    n_loaded = load_adapter(model_b, "rtheta", path)

    assert n_loaded == 18

    # Verify values match
    for module in model_b.modules():
        if isinstance(module, MultiLoRALinear):
            assert torch.all(module.lora_A["rtheta"] == 1.0)
            assert torch.all(module.lora_B["rtheta"] == 2.0)

    del model_b
    gc.collect()

    print(f"[PASS] roundtrip_save_load_compiled")


if __name__ == "__main__":
    setup_scratch()
    try:
        test_strip_orig_mod()
        test_install_on_compiled_model()
        test_load_adapter_compiled()
        test_second_install_adds_adapter()
        test_save_adapter_canonical_keys()
        test_get_adapter_params_canonical()
        test_roundtrip_save_load_compiled()
        print("\n=== ALL TESTS PASSED ===")
    finally:
        teardown_scratch()

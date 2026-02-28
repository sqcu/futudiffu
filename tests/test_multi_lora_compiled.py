"""Integration test: MultiLoRA on torch.compile'd models.

Validates the full lifecycle that _run_policy_intervention exercises:
  1. Compile model (layers become OptimizedModule with _orig_mod paths)
  2. install_multi_lora pre-allocates capacity through _orig_mod
  3. assign_adapter populates slots in pre-allocated wrappers
  4. load_adapter matches checkpoint keys to canonical (non-_orig_mod) names
  5. save_adapter produces canonical keys (no _orig_mod in output)
  6. get_adapter_params keys are canonical
"""

import gc
import os
import sys
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

from src_ii.multi_lora import (
    MultiLoRALinear,
    install_multi_lora,
    assign_adapter,
    load_adapter,
    save_adapter,
    get_adapter_params,
    _strip_orig_mod,
    slot_index_for,
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

    wrappers = install_multi_lora(model, max_adapters=4, max_rank=8)
    assign_adapter(model, 0, "rtheta", 4, 8.0)

    assert len(wrappers) > 0, "No wrappers installed"

    # Verify wrappers are on the right modules (3 layers * 3 linears = 9)
    n_expected = 3 * 3  # layers * (qkv, out, w2)
    assert len(wrappers) == n_expected, f"Expected {n_expected} wrappers, got {len(wrappers)}"

    # Verify excluded modules were NOT wrapped
    assert not isinstance(model.score_proj, MultiLoRALinear)
    assert not isinstance(model.final_layer, MultiLoRALinear)

    # Verify each wrapper has the adapter in slot 0
    for name, wrapper in wrappers.items():
        assert wrapper._slot_names[0] == "rtheta", f"Missing rtheta in slot 0 of {name}"
        assert wrapper.max_adapters == 4

    print(f"[PASS] install_on_compiled_model: {len(wrappers)} wrappers")


def test_load_adapter_compiled():
    """load_adapter should match canonical keys on a compiled model."""
    model = FakeModel()

    # Create checkpoint BEFORE compile (uses canonical names)
    ckpt_path = make_fake_checkpoint(model, "rtheta", SCRATCH)

    # Now compile, install capacity, and assign adapter
    model = compile_model(model)
    install_multi_lora(model, max_adapters=4, max_rank=8)
    assign_adapter(model, 0, "rtheta", 4, 8.0)

    n_loaded = load_adapter(model, "rtheta", str(ckpt_path))
    assert n_loaded > 0, f"Loaded 0 tensors"
    # 3 layers * 3 linears * 2 (A+B) = 18
    assert n_loaded == 18, f"Expected 18, got {n_loaded}"

    print(f"[PASS] load_adapter_compiled: {n_loaded} tensors loaded")


def test_assign_adapter_fills_slots():
    """assign_adapter populates named slots in pre-allocated wrappers."""
    model = FakeModel()
    model = compile_model(model)

    wrappers = install_multi_lora(model, max_adapters=4, max_rank=8)

    # Assign two adapters to different slots
    assign_adapter(model, 0, "rtheta", 4, 8.0)
    assign_adapter(model, 1, "policy_pinkify", 4, 8.0)

    # Verify each wrapper has both adapters in the right slots
    for name, wrapper in wrappers.items():
        assert wrapper._slot_names[0] == "rtheta", f"Missing rtheta in slot 0 of {name}"
        assert wrapper._slot_names[1] == "policy_pinkify", f"Missing policy_pinkify in slot 1 of {name}"
        assert wrapper._slot_names[2] is None, f"Slot 2 should be empty in {name}"
        assert wrapper._slot_names[3] is None, f"Slot 3 should be empty in {name}"
        assert wrapper.max_adapters == 4

    # Verify slot lookup
    assert slot_index_for(model, "rtheta") == 0
    assert slot_index_for(model, "policy_pinkify") == 1

    print(f"[PASS] assign_adapter_fills_slots: {len(wrappers)} wrappers, 2 adapters each")


def test_save_adapter_canonical_keys():
    """save_adapter should produce keys without _orig_mod."""
    model = FakeModel()
    model = compile_model(model)

    install_multi_lora(model, max_adapters=4, max_rank=8)
    assign_adapter(model, 0, "rtheta", 4, 8.0)

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

    install_multi_lora(model, max_adapters=4, max_rank=8)
    assign_adapter(model, 0, "rtheta", 4, 8.0)

    params = get_adapter_params(model, "rtheta")
    for key in params:
        assert "_orig_mod" not in key, f"_orig_mod in param key: {key}"

    assert len(params) == 18, f"Expected 18 params, got {len(params)}"
    print(f"[PASS] get_adapter_params_canonical: {len(params)} params, all canonical")


def test_roundtrip_save_load_compiled():
    """Full roundtrip: install -> save -> new model -> compile -> install -> load."""
    # Model A: install capacity, assign adapter, modify weights, save
    model_a = FakeModel()
    model_a = compile_model(model_a)
    install_multi_lora(model_a, max_adapters=4, max_rank=8)
    assign_adapter(model_a, 0, "rtheta", 4, 8.0)

    # Set A and B matrices to known values (slot 0, first `rank` rows/cols)
    idx = slot_index_for(model_a, "rtheta")
    for module in model_a.modules():
        if isinstance(module, MultiLoRALinear):
            module.lora_A[idx].data[:4, :].fill_(1.0)  # rank=4 rows
            module.lora_B[idx].data[:, :4].fill_(2.0)  # rank=4 cols

    path = str(SCRATCH / "roundtrip.safetensors")
    save_adapter(model_a, "rtheta", path)
    del model_a
    gc.collect()

    # Model B: fresh compile, install capacity, assign adapter, load
    model_b = FakeModel()
    model_b = compile_model(model_b)
    install_multi_lora(model_b, max_adapters=4, max_rank=8)
    assign_adapter(model_b, 0, "rtheta", 4, 8.0)
    n_loaded = load_adapter(model_b, "rtheta", path)

    assert n_loaded == 18

    # Verify values match (active rank region only)
    idx_b = slot_index_for(model_b, "rtheta")
    for module in model_b.modules():
        if isinstance(module, MultiLoRALinear):
            assert torch.all(module.lora_A[idx_b][:4, :] == 1.0), "A matrix mismatch"
            assert torch.all(module.lora_B[idx_b][:, :4] == 2.0), "B matrix mismatch"

    del model_b
    gc.collect()

    print(f"[PASS] roundtrip_save_load_compiled")


if __name__ == "__main__":
    setup_scratch()
    try:
        test_strip_orig_mod()
        test_install_on_compiled_model()
        test_load_adapter_compiled()
        test_assign_adapter_fills_slots()
        test_save_adapter_canonical_keys()
        test_get_adapter_params_canonical()
        test_roundtrip_save_load_compiled()
        print("\n=== ALL TESTS PASSED ===")
    finally:
        teardown_scratch()

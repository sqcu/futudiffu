"""LoRA (Low-Rank Adaptation) — horizontal multi-adapter via Triton sparse kernel.

Each LoRALinear holds N named adapter slots.  All adapters read the same
input x, and their contributions are summed into the base output::

    output = base(x) + Σ_i scale_i * (α_i/r_i) * (x @ A_i^T) @ B_i^T

Multi-adapter forward dispatches to the Triton sparse kernel in
lora_kernels.py which:
  - Takes scale_all (B, N_ADAPTERS) as an explicit tensor operand
  - Skips zero-scaled adapters entirely (zero warp divergence)
  - Is registered as a custom_op with register_fake + register_autograd

Per-adapter per-batch-element scale is a registered buffer on LoRAAdapter,
which torch.compile treats as a dynamic graph input.
"""

from __future__ import annotations

import math
import re
from typing import Iterator, Optional

import torch
import torch.nn as nn

from .fp8 import FP8Linear
from .lora_kernels import multi_lora_op


DEFAULT_TARGET_SUFFIXES = (
    "attention.qkv",
    "attention.out",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
)


class LoRAAdapter(nn.Module):
    """Single rank-r adapter: A (down-project), B (up-project), scale.

    lora_scale is a registered buffer — torch.compile treats it as a
    dynamic graph input whose value can change between calls without
    recompilation (same shape) or with one recompilation (shape change).

    Shape (1,) broadcasts to any batch size (default = 1.0 = active).
    Shape (B,) gives per-batch-element routing (e.g. [1, 0] for policy ON
    in batch[0], OFF in batch[1]).
    """

    def __init__(
        self,
        rank: int,
        alpha: float,
        in_features: int,
        out_features: int,
        device: torch.device,
        init_b_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features, dtype=torch.bfloat16, device=device))
        if init_b_std > 0:
            self.lora_B = nn.Parameter(
                torch.empty(out_features, rank, dtype=torch.bfloat16, device=device))
            nn.init.normal_(self.lora_B, mean=0.0, std=init_b_std)
        else:
            self.lora_B = nn.Parameter(
                torch.zeros(out_features, rank, dtype=torch.bfloat16, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Registered buffer: torch.compile sees this as a graph input operand.
        # Shape (1,) = broadcast scalar. Shape (B,) = per-batch routing.
        self.register_buffer(
            "lora_scale",
            torch.ones(1, dtype=torch.bfloat16, device=device),
        )

    def forward(self, x_bf16: torch.Tensor) -> torch.Tensor:
        """Standalone forward for single-adapter fast path."""
        mid = x_bf16 @ self.lora_A.t()
        lora_out = mid @ self.lora_B.t()
        # lora_scale: (1,) broadcasts to any B, (B,) for per-batch routing.
        n_extra = lora_out.ndim - 1
        shape = (-1,) + (1,) * n_extra
        return self.lora_scale.view(shape) * self.scale * lora_out


class LoRALinear(nn.Module):
    """Linear layer with N named low-rank adapters (fused via Triton kernel).

    All adapter counts (N>=1) dispatch to multi_lora_op, which is registered
    as a custom_op opaque to torch.compile/inductor.  This keeps graph
    topology constant regardless of adapter count — compile time does not
    scale with number of adapters.
    """

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.adapters = nn.ModuleDict()

        # Freeze base
        for p in self.base.parameters():
            p.requires_grad = False
        if isinstance(self.base, FP8Linear):
            for buf in self.base.buffers():
                buf.requires_grad = False

        try:
            self._device = next(base.parameters()).device
        except StopIteration:
            self._device = next(base.buffers()).device

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return getattr(self.base, 'bias', None)

    def __getattr__(self, name: str):
        # Forward attribute lookups to the base module for FP8-specific attrs
        # (block_size, weight_scale, output_dtype, _transposed, etc.)
        # that the fused FFN chain accesses.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)

    def add_adapter(
        self, name: str, rank: int = 8, alpha: float = 16.0,
        init_b_std: float = 0.0,
    ) -> LoRAAdapter:
        if name in self.adapters:
            return self.adapters[name]  # idempotent: re-injection is a no-op
        adapter = LoRAAdapter(
            rank, alpha, self.in_features, self.out_features, self._device,
            init_b_std=init_b_std,
        )
        self.adapters[name] = adapter
        return adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        n = len(self.adapters)
        if n == 0:
            return base_out

        x_bf16 = x.to(torch.bfloat16)
        adapters = list(self.adapters.values())

        # --- All adapter counts: dispatch to Triton sparse kernel ---
        # multi_lora_op is a custom_op, opaque to inductor. Using it for
        # N=1 replaces 2 visible matmul graph nodes with 1 opaque node,
        # eliminating inductor GEMM analysis for LoRA paths entirely.
        B = x.shape[0]
        # x must be 3D (B, S, IN) for the kernel; flatten middle dims if needed
        orig_shape = x_bf16.shape
        if x_bf16.ndim > 3:
            x_3d = x_bf16.view(B, -1, x_bf16.shape[-1])
        else:
            x_3d = x_bf16.contiguous()

        A_all = torch.stack([a.lora_A for a in adapters])  # (N, R, IN)
        B_all = torch.stack([a.lora_B for a in adapters])  # (N, OUT, R)

        # Build scale_all: (B, N) float32 with alpha/rank pre-folded.
        # Each a.lora_scale is a registered buffer: (1,) or (B,).
        scale_parts = []
        for a in adapters:
            scale_parts.append(a.lora_scale.expand(B).float() * a.scale)
        scale_all = torch.stack(scale_parts, dim=-1).contiguous()  # (B, N)

        R = adapters[0].rank
        delta = multi_lora_op(x_3d, A_all, B_all, scale_all, n, R)

        # Restore shape if we flattened
        if x_bf16.ndim > 3:
            delta = delta.view(*orig_shape[:-1], delta.shape[-1])

        return base_out + delta.to(base_out.dtype)


# ---------------------------------------------------------------------------
# Injection — split into graph-mutating allocation vs weight-only init
# ---------------------------------------------------------------------------

def _find_lora_targets(
    model: nn.Module,
    target_modules: Optional[list[str]] = None,
    layer_indices: Optional[set[int]] = None,
) -> list[tuple[str, str, str]]:
    """Find candidate modules for LoRA injection.

    Returns list of (full_path, parent_path, attr_name) tuples.
    """
    suffixes = tuple(target_modules) if target_modules else DEFAULT_TARGET_SUFFIXES

    candidates: list[tuple[str, str, str]] = []
    for full_path, module in model.named_modules():
        if not any(full_path.endswith(s) for s in suffixes):
            continue
        if not isinstance(module, (nn.Linear, FP8Linear, LoRALinear)):
            continue
        if target_modules is None:
            if not any(bp in full_path for bp in
                       ("layers.", "noise_refiner.", "context_refiner.")):
                continue
        if layer_indices is not None:
            m = re.match(r"layers\.(\d+)\.", full_path)
            if not m or int(m.group(1)) not in layer_indices:
                continue
        parts = full_path.rsplit(".", 1)
        parent_path, attr_name = (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])
        candidates.append((full_path, parent_path, attr_name))

    return candidates


def allocate_adapter(
    model: nn.Module,
    name: str = "default",
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[list[str]] = None,
    layer_indices: Optional[set[int]] = None,
) -> dict[str, LoRAAdapter]:
    """Allocate adapter slots — GRAPH MUTATING. Must happen before torch.compile.

    Wraps Linear/FP8Linear → LoRALinear if not already wrapped, then creates
    adapter with zero-init B and scale=0.0 (silent by default). Does NOT
    initialize weights for training — call init_adapter_weights() for that.

    This is the expensive, compile-invalidating operation. After all adapters
    are allocated, compile once, then use init_adapter_weights() and
    set_lora_scale() freely without recompilation.

    Returns:
        Dict mapping full module path -> LoRAAdapter instance.
    """
    candidates = _find_lora_targets(model, target_modules, layer_indices)
    module_map = dict(model.named_modules())
    injected: dict[str, LoRAAdapter] = {}

    for full_path, parent_path, attr_name in candidates:
        parent = module_map[parent_path] if parent_path else model
        child = getattr(parent, attr_name)

        if not isinstance(child, LoRALinear):
            wrapper = LoRALinear(child)
            setattr(parent, attr_name, wrapper)
            module_map = dict(model.named_modules())
            child = wrapper

        # Zero-init B, scale=0.0 — silent until explicitly activated
        adapter = child.add_adapter(name, rank=rank, alpha=alpha, init_b_std=0.0)
        injected[full_path] = adapter

    # Default to silent (scale=0) — caller activates via set_lora_scale
    set_lora_scale(
        model,
        torch.zeros(1, dtype=torch.bfloat16, device=next(model.parameters()).device),
        adapter_name=name,
    )
    return injected


def init_adapter_weights(
    model: nn.Module,
    name: str,
    init_b_std: float = 0.0,
    scale: float = 1.0,
) -> int:
    """(Re-)initialize adapter weights. Graph-INVARIANT — safe after compile.

    Reinitializes lora_A (kaiming uniform) and lora_B (zeros or normal).
    Sets lora_scale to the given value. Does not add/remove modules or
    change the compute graph.

    Args:
        name: Adapter name to initialize.
        init_b_std: If > 0, N(0, std) for B. If 0, zeros (standard LoRA).
        scale: Initial scale value (1.0 = active, 0.0 = silent).

    Returns:
        Number of adapter modules initialized.
    """
    n = 0
    for module in model.modules():
        if not isinstance(module, LoRALinear):
            continue
        if name not in module.adapters:
            continue
        adapter = module.adapters[name]
        nn.init.kaiming_uniform_(adapter.lora_A, a=math.sqrt(5))
        if init_b_std > 0:
            nn.init.normal_(adapter.lora_B, mean=0.0, std=init_b_std)
        else:
            nn.init.zeros_(adapter.lora_B)
        n += 1

    # Set scale
    device = next(model.parameters()).device
    scale_t = torch.tensor([scale], dtype=torch.bfloat16, device=device)
    set_lora_scale(model, scale_t, adapter_name=name)
    return n


def inject_lora(
    model: nn.Module,
    name: str = "default",
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[list[str]] = None,
    layer_indices: Optional[set[int]] = None,
    init_b_std: float = 0.0,
) -> dict[str, LoRAAdapter]:
    """Add a named adapter to matching linear layers (legacy convenience wrapper).

    Combines allocate_adapter() + weight initialization in one call.
    Prefer allocate_adapter() + init_adapter_weights() for new code to
    separate graph-mutating allocation from weight-only initialization.

    If a target is already LoRALinear, adds a new adapter slot.
    If a target is Linear/FP8Linear, wraps it first.
    """
    candidates = _find_lora_targets(model, target_modules, layer_indices)
    module_map = dict(model.named_modules())
    injected: dict[str, LoRAAdapter] = {}

    for full_path, parent_path, attr_name in candidates:
        parent = module_map[parent_path] if parent_path else model
        child = getattr(parent, attr_name)

        if not isinstance(child, LoRALinear):
            wrapper = LoRALinear(child)
            setattr(parent, attr_name, wrapper)
            module_map = dict(model.named_modules())
            child = wrapper

        adapter = child.add_adapter(name, rank=rank, alpha=alpha, init_b_std=init_b_std)
        injected[full_path] = adapter

    return injected


# ---------------------------------------------------------------------------
# Per-adapter utilities
# ---------------------------------------------------------------------------

def get_lora_params(
    model: nn.Module,
    adapter_name: Optional[str] = None,
) -> Iterator[nn.Parameter]:
    """Yield LoRA A and B parameters, optionally filtered by adapter name."""
    for module in model.modules():
        if not isinstance(module, LoRALinear):
            continue
        for aname, adapter in module.adapters.items():
            if adapter_name is not None and aname != adapter_name:
                continue
            yield adapter.lora_A
            yield adapter.lora_B


def set_lora_scale(
    model: nn.Module,
    scale: torch.Tensor,
    adapter_name: Optional[str] = None,
) -> None:
    """Set per-batch-element scale tensor on adapter buffers.

    scale shape (1,): broadcast scalar, same value for all batch elements.
    scale shape (B,): per-batch-element routing.

    Same-shape updates use in-place copy_ (no recompilation under
    torch.compile).  Shape changes reassign the buffer (one recompilation).
    """
    for module in model.modules():
        if not isinstance(module, LoRALinear):
            continue
        for aname, adapter in module.adapters.items():
            if adapter_name is not None and aname != adapter_name:
                continue
            buf = adapter.lora_scale
            if buf.shape == scale.shape:
                buf.copy_(scale)
            else:
                adapter.lora_scale = scale.to(
                    dtype=buf.dtype, device=buf.device)


def clear_lora_scale(
    model: nn.Module,
    adapter_name: Optional[str] = None,
) -> None:
    """Reset scale to broadcast 1.0 (all adapters active, uniform)."""
    for module in model.modules():
        if not isinstance(module, LoRALinear):
            continue
        for aname, adapter in module.adapters.items():
            if adapter_name is not None and aname != adapter_name:
                continue
            buf = adapter.lora_scale
            if buf.shape == (1,):
                buf.fill_(1.0)
            else:
                adapter.lora_scale = torch.ones(
                    1, dtype=buf.dtype, device=buf.device)


def freeze_adapter(model: nn.Module, adapter_name: str) -> int:
    """Freeze adapter params (requires_grad=False). Returns count frozen."""
    n = 0
    for module in model.modules():
        if not isinstance(module, LoRALinear):
            continue
        if adapter_name in module.adapters:
            a = module.adapters[adapter_name]
            a.lora_A.requires_grad_(False)
            a.lora_B.requires_grad_(False)
            n += 1
    return n


def unfreeze_adapter(model: nn.Module, adapter_name: str) -> int:
    """Unfreeze adapter params (requires_grad=True). Returns count unfrozen."""
    n = 0
    for module in model.modules():
        if not isinstance(module, LoRALinear):
            continue
        if adapter_name in module.adapters:
            a = module.adapters[adapter_name]
            a.lora_A.requires_grad_(True)
            a.lora_B.requires_grad_(True)
            n += 1
    return n


# ---------------------------------------------------------------------------
# State dict
# ---------------------------------------------------------------------------

def enumerate_adapters(model: nn.Module) -> dict[str, dict]:
    """Return metadata for every adapter in the model.

    Returns:
        Dict mapping adapter_name -> {
            "n_modules": int,
            "rank": int,
            "alpha": float,
            "scale": float,  # current lora_scale buffer value (first element)
        }
    """
    info: dict[str, dict] = {}
    for module in model.modules():
        if not isinstance(module, LoRALinear):
            continue
        for aname, adapter in module.adapters.items():
            if aname not in info:
                info[aname] = {
                    "n_modules": 0,
                    "rank": adapter.rank,
                    "alpha": adapter.alpha,
                    "scale": adapter.lora_scale[0].item(),
                }
            info[aname]["n_modules"] += 1
    return info


def lora_state_dict(
    model: nn.Module,
    adapter_name: Optional[str] = None,
) -> dict[str, torch.Tensor]:
    """Extract LoRA weights. Keys: "path.adapters.name.lora_A" etc."""
    sd = {}
    for path, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        for aname, adapter in module.adapters.items():
            if adapter_name is not None and aname != adapter_name:
                continue
            prefix = f"{path}.adapters.{aname}"
            sd[f"{prefix}.lora_A"] = adapter.lora_A.data.clone()
            sd[f"{prefix}.lora_B"] = adapter.lora_B.data.clone()
    return sd


def load_lora_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Load LoRA weights. Model must already have matching adapters."""
    adapter_map: dict[str, LoRAAdapter] = {}
    for path, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        for aname, adapter in module.adapters.items():
            adapter_map[f"{path}.adapters.{aname}"] = adapter

    loaded = set()
    for key, tensor in state_dict.items():
        if key.endswith(".lora_A"):
            prefix, param_name = key[:-len(".lora_A")], "lora_A"
        elif key.endswith(".lora_B"):
            prefix, param_name = key[:-len(".lora_B")], "lora_B"
        else:
            raise ValueError(f"Unexpected key: {key}")
        if prefix not in adapter_map:
            raise ValueError(f"No adapter at '{prefix}'. Available: {sorted(adapter_map)}")
        param = getattr(adapter_map[prefix], param_name)
        if param.shape != tensor.shape:
            raise ValueError(f"Shape mismatch {key}: {param.shape} vs {tensor.shape}")
        param.data.copy_(tensor.to(param.dtype))
        loaded.add(key)

    unloaded = set(state_dict) - loaded
    if unloaded:
        raise ValueError(f"Unrecognized keys: {sorted(unloaded)}")


def count_lora_params(
    model: nn.Module,
    adapter_name: Optional[str] = None,
) -> tuple[int, int]:
    """Count LoRA params (optionally by name) and total model params."""
    lora_count = sum(p.numel() for p in get_lora_params(model, adapter_name))
    total_count = sum(p.numel() for p in model.parameters())
    return lora_count, total_count


# ---------------------------------------------------------------------------
# Emergency dump
# ---------------------------------------------------------------------------

def dump_all_loras(
    model: nn.Module,
    output_dir,
    btrm_head: Optional[nn.Module] = None,
    btrm_config: Optional[dict] = None,
) -> dict:
    """Dump all LoRA adapters (and optionally BTRM head) to disk.

    Writes each adapter as a separate safetensors file plus a JSON manifest.

    Args:
        model: Model with LoRA adapters.
        output_dir: Path or str for output directory.
        btrm_head: Optional BTRM head to include in the dump.
        btrm_config: Optional config dict for the BTRM head.

    Returns:
        Dict with keys: files (list), manifest (str path), btrm_head (dict or None).
    """
    import json as _json
    import time
    from pathlib import Path

    from safetensors.torch import save_file as st_save

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapters = enumerate_adapters(model)
    if not adapters:
        return {"files": [], "manifest": None, "btrm_head": None,
                "note": "no adapters found"}

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    files = []
    for adapter_name, info in adapters.items():
        sd = lora_state_dict(model, adapter_name=adapter_name)
        sd_cpu = {k: v.cpu() for k, v in sd.items()}
        fname = f"{adapter_name}_{timestamp}.safetensors"
        fpath = output_dir / fname
        st_save(sd_cpu, str(fpath))
        files.append({
            "adapter": adapter_name,
            "path": str(fpath),
            "n_tensors": len(sd_cpu),
            "rank": info["rank"],
            "alpha": info["alpha"],
            "scale": info["scale"],
        })
        print(f"  [dump_all_loras] {adapter_name}: "
              f"{len(sd_cpu)} tensors -> {fpath}")

    btrm_file = None
    if btrm_head is not None:
        btrm_sd = {k: v.cpu() for k, v in btrm_head.state_dict().items()}
        btrm_fname = f"btrm_head_{timestamp}.safetensors"
        btrm_fpath = output_dir / btrm_fname
        st_save(btrm_sd, str(btrm_fpath))
        btrm_file = {
            "path": str(btrm_fpath),
            "n_tensors": len(btrm_sd),
            "config": btrm_config,
        }
        print(f"  [dump_all_loras] BTRM head: {len(btrm_sd)} tensors -> {btrm_fpath}")

    manifest_path = output_dir / f"dump_manifest_{timestamp}.json"
    manifest = {"timestamp": timestamp, "adapters": files}
    if btrm_file is not None:
        manifest["btrm_head"] = btrm_file
    with open(manifest_path, "w") as f:
        _json.dump(manifest, f, indent=2)

    print(f"  [dump_all_loras] {len(files)} adapter(s) dumped to {output_dir}")
    return {"files": files, "manifest": str(manifest_path), "btrm_head": btrm_file}

"""LoRA (Low-Rank Adaptation) for NextDiT Z-Image diffusion model.

Horizontal multi-adapter design: each LoRALinear holds N named adapter
slots.  All adapters read the same input x, compute independently, and
sum their contributions to the base output::

    output = base(x) + Σ_i adapter_i(x)
    adapter_i(x) = scale_i * (α_i/r_i) * (x @ A_i^T) @ B_i^T

Each adapter has its own rank, alpha, A/B parameters, and per-batch-element
scale tensor.  scale=0 disables an adapter (zero contribution, no graph
mutation, compile-safe).  This is fine-grained MoE with explicit routing:
each adapter is a rank-r micro-expert, the scale tensor is the router.

Designed for QAT with separate R_theta (reward) and p_theta (policy)
adapters on the same layers.
"""

from __future__ import annotations

import math
import re
from typing import Iterator, Optional, Sequence

import torch
import torch.nn as nn

from .fp8 import FP8Linear


# Default target module suffixes for NextDiT Z-Image.
DEFAULT_TARGET_SUFFIXES = (
    "attention.qkv",
    "attention.out",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
)


class LoRAAdapter(nn.Module):
    """A single low-rank adapter (A, B, scale).

    B is zero-initialized so the adapter starts as identity.
    """

    def __init__(
        self,
        rank: int,
        alpha: float,
        in_features: int,
        out_features: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features, dtype=torch.bfloat16, device=device))
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, dtype=torch.bfloat16, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Per-batch-element scale. None = uniform 1.0.
        # Shape (B,) when set — broadcast to (B, 1, ..., 1) in forward.
        self._lora_scale: Optional[torch.Tensor] = None

    @property
    def lora_scale(self) -> Optional[torch.Tensor]:
        return self._lora_scale

    @lora_scale.setter
    def lora_scale(self, value: Optional[torch.Tensor]) -> None:
        self._lora_scale = value

    def forward(self, x_bf16: torch.Tensor) -> torch.Tensor:
        """Compute rank-r residual from input (already bf16)."""
        mid = x_bf16 @ self.lora_A.t()    # (..., rank)
        lora_out = mid @ self.lora_B.t()   # (..., out_features)

        if self._lora_scale is not None:
            n_extra = lora_out.ndim - 1
            shape = (-1,) + (1,) * n_extra
            return self._lora_scale.view(shape) * self.scale * lora_out
        return self.scale * lora_out


class LoRALinear(nn.Module):
    """Linear layer with N named low-rank adapters (horizontal multi-LoRA).

    output = base(x) + Σ adapter_i(x)

    All adapters read the same input x — no serial dependencies between
    them.  Each adapter has independent rank, alpha, scale, and parameters.

    Args:
        base: Frozen base layer (nn.Linear or FP8Linear).
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

        # Cache device
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
        return self.base.bias

    def add_adapter(self, name: str, rank: int = 8, alpha: float = 16.0) -> LoRAAdapter:
        """Add a named adapter slot. Returns the new adapter."""
        if name in self.adapters:
            raise ValueError(f"Adapter '{name}' already exists")
        adapter = LoRAAdapter(
            rank, alpha, self.in_features, self.out_features, self._device)
        self.adapters[name] = adapter
        return adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.adapters:
            return out
        x_bf16 = x.to(torch.bfloat16)
        for adapter in self.adapters.values():
            out = out + adapter(x_bf16).to(out.dtype)
        return out


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

def inject_lora(
    model: nn.Module,
    name: str = "default",
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[list[str]] = None,
    layer_indices: Optional[set[int]] = None,
) -> dict[str, LoRAAdapter]:
    """Add a named adapter to matching linear layers.

    If a target is already a LoRALinear, adds a new adapter slot.
    If a target is Linear/FP8Linear, wraps it in LoRALinear first.

    Args:
        model: Model to inject into.
        name: Adapter name (e.g. "rtheta", "ptheta").
        rank: Low-rank dimension.
        alpha: LoRA scaling factor.
        target_modules: Suffixes to target. None = DEFAULT_TARGET_SUFFIXES.
        layer_indices: If set, only target layers.N with N in set.
            Excludes refiners when specified.

    Returns:
        Dict mapping full module path -> LoRAAdapter instance.
    """
    suffixes = tuple(target_modules) if target_modules else DEFAULT_TARGET_SUFFIXES

    # Phase 1: identify targets
    candidates: list[tuple[str, str, str]] = []  # (full_path, parent_path, attr_name)
    for full_path, module in model.named_modules():
        if not any(full_path.endswith(s) for s in suffixes):
            continue
        if not isinstance(module, (nn.Linear, FP8Linear, LoRALinear)):
            continue
        # Block prefix filter (default targets only)
        if target_modules is None:
            if not any(bp in full_path for bp in
                       ("layers.", "noise_refiner.", "context_refiner.")):
                continue
        # Layer index filter
        if layer_indices is not None:
            m = re.match(r"layers\.(\d+)\.", full_path)
            if not m or int(m.group(1)) not in layer_indices:
                continue
        parts = full_path.rsplit(".", 1)
        parent_path, attr_name = (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])
        candidates.append((full_path, parent_path, attr_name))

    # Phase 2: wrap if needed, add adapter
    module_map = dict(model.named_modules())
    injected: dict[str, LoRAAdapter] = {}

    for full_path, parent_path, attr_name in candidates:
        parent = module_map[parent_path] if parent_path else model
        child = getattr(parent, attr_name)

        if not isinstance(child, LoRALinear):
            # Wrap in LoRALinear first
            wrapper = LoRALinear(child)
            setattr(parent, attr_name, wrapper)
            # Update module map for subsequent lookups
            module_map = dict(model.named_modules())
            child = wrapper

        adapter = child.add_adapter(name, rank=rank, alpha=alpha)
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
    """Set per-batch-element scale on adapters.

    Args:
        scale: (B,) tensor.
        adapter_name: If set, only affects that adapter. None = all.
    """
    for module in model.modules():
        if not isinstance(module, LoRALinear):
            continue
        for aname, adapter in module.adapters.items():
            if adapter_name is not None and aname != adapter_name:
                continue
            adapter.lora_scale = scale


def clear_lora_scale(
    model: nn.Module,
    adapter_name: Optional[str] = None,
) -> None:
    """Clear scale (revert to uniform 1.0)."""
    for module in model.modules():
        if not isinstance(module, LoRALinear):
            continue
        for aname, adapter in module.adapters.items():
            if adapter_name is not None and aname != adapter_name:
                continue
            adapter.lora_scale = None


def set_lora_enabled(model: nn.Module, enabled: bool) -> None:
    """Enable/disable ALL adapters by setting scale to 0 or clearing it."""
    if not enabled:
        zero = torch.tensor([0.0], dtype=torch.bfloat16)
        set_lora_scale(model, zero)
    else:
        clear_lora_scale(model)


def freeze_adapter(model: nn.Module, adapter_name: str) -> int:
    """Freeze an adapter's parameters (requires_grad=False). Returns count."""
    n = 0
    for module in model.modules():
        if not isinstance(module, LoRALinear):
            continue
        if adapter_name in module.adapters:
            adapter = module.adapters[adapter_name]
            adapter.lora_A.requires_grad_(False)
            adapter.lora_B.requires_grad_(False)
            n += 1
    return n


# ---------------------------------------------------------------------------
# State dict
# ---------------------------------------------------------------------------

def lora_state_dict(
    model: nn.Module,
    adapter_name: Optional[str] = None,
) -> dict[str, torch.Tensor]:
    """Extract LoRA weights, optionally filtered by adapter name.

    Keys: "layers.0.attention.qkv.adapters.rtheta.lora_A" etc.
    """
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
    # Build lookup: "path.adapters.name" -> adapter
    adapter_map: dict[str, LoRAAdapter] = {}
    for path, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        for aname, adapter in module.adapters.items():
            adapter_map[f"{path}.adapters.{aname}"] = adapter

    loaded = set()
    for key, tensor in state_dict.items():
        if key.endswith(".lora_A"):
            prefix = key[:-len(".lora_A")]
            param_name = "lora_A"
        elif key.endswith(".lora_B"):
            prefix = key[:-len(".lora_B")]
            param_name = "lora_B"
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

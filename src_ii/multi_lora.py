"""Multi-tenant batched LoRA with per-image sparse adapter routing.

Pre-allocated adapter capacity: MultiLoRALinear is created with fixed
max_adapters and max_rank. Adapters are assigned to explicit slot indices
by the caller. adapter_scales is always (n_images, max_adapters) wide.

Lifecycle:
  1. install_multi_lora(model, max_adapters, max_rank) -- pre-allocate empty slots
  2. assign_adapter(model, slot_index, name, rank, alpha) -- populate a slot
  3. model.forward(..., adapter_scales=..., token_to_image=...) -- routing
  4. save_adapter / load_adapter -- persist individual adapters by name
  5. release_adapter(model, slot_index) -- free a slot

No dynamic growth. No recompilation on assign/release. adapter_scales is
always (n_images, max_adapters) wide, regardless of how many slots are active.

Implementation: Python loop over active-flagged slots with masked
gather/scatter for token routing. Triton kernel optimization deferred.

Import constraints:
  - torch, nn, safetensors
  - No futudiffu imports (pure module)
  - No src_ii imports
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file


class MultiLoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear that supports multiple LoRA adapters.

    Wraps a base linear layer (nn.Linear or FP8Linear) and adds N pre-allocated
    adapter slots, each with its own A (down-projection) and B (up-projection)
    matrices. The adapter output is:

        sum_j[ (x @ A_j^T @ B_j^T) * (alpha_j / rank_j) * scale_j ]

    where scale_j comes from adapter_scales and can be per-image in a
    packed batch via token_to_image routing.

    Slots are pre-allocated at construction time with fixed max_adapters and
    max_rank. Adapters are assigned to slots via assign_adapter(). Empty slots
    (scale=0, all-zero weights) contribute nothing and cost negligible FLOPs.

    When no adapter_scales are passed, returns base output only.
    """

    def __init__(
        self,
        base_linear: nn.Module,
        max_adapters: int,
        max_rank: int,
        adapter_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | None = None,
    ):
        """Initialize MultiLoRALinear wrapping an existing linear layer.

        Args:
            base_linear: The original nn.Linear or FP8Linear to wrap.
                NOT removed from the parent -- the caller (install_multi_lora)
                handles module replacement.
            max_adapters: Number of pre-allocated adapter slots.
            max_rank: Maximum LoRA rank across all slots.
            adapter_dtype: dtype for adapter parameters (default bfloat16).
            device: Device for adapter parameters. If None, inferred from
                base_linear.weight.device.
        """
        super().__init__()
        self.base = base_linear

        # Infer dimensions from base linear
        if hasattr(base_linear, "in_features"):
            self.in_features = base_linear.in_features
            self.out_features = base_linear.out_features
        elif hasattr(base_linear, "weight"):
            self.out_features, self.in_features = base_linear.weight.shape
        else:
            raise ValueError(f"Cannot infer dimensions from {type(base_linear)}")

        # Infer device from base linear if not specified
        if device is None and hasattr(base_linear, "weight") and base_linear.weight is not None:
            device = base_linear.weight.device

        self.max_adapters = max_adapters
        self.max_rank = max_rank

        # Pre-allocated per-slot parameters via ParameterList.
        # Fixed length -> no recompilation on assign/release.
        # Constructed on-device directly (no post-hoc migration).
        self.lora_A = nn.ParameterList([
            nn.Parameter(
                torch.zeros(max_rank, self.in_features, dtype=adapter_dtype, device=device),
                requires_grad=False,
            )
            for _ in range(max_adapters)
        ])
        self.lora_B = nn.ParameterList([
            nn.Parameter(
                torch.zeros(self.out_features, max_rank, dtype=adapter_dtype, device=device),
                requires_grad=False,
            )
            for _ in range(max_adapters)
        ])

        # Metadata (not parameters, not compiled)
        self._slot_names: list[str | None] = [None] * max_adapters
        self._slot_ranks: list[int] = [0] * max_adapters
        self._slot_alphas: list[float] = [0.0] * max_adapters

        self.register_buffer(
            "_lora_scalings",
            torch.zeros(max_adapters, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "_slot_active",
            torch.zeros(max_adapters, dtype=torch.float32, device=device),
        )

    def assign_adapter(
        self,
        slot_index: int,
        name: str,
        rank: int,
        alpha: float,
        init_a: str = "kaiming",
    ) -> None:
        """Assign adapter to pre-allocated slot. No recompilation.

        Args:
            slot_index: Which slot to assign to (0..max_adapters-1).
            name: Unique adapter identifier.
            rank: LoRA rank (<= max_rank). Extra rows stay zero.
            alpha: LoRA scaling factor.
            init_a: Initialization for A matrix ("kaiming" or "zeros").
        """
        if slot_index < 0 or slot_index >= self.max_adapters:
            raise IndexError(
                f"slot_index {slot_index} out of range [0, {self.max_adapters})"
            )
        if rank > self.max_rank:
            raise ValueError(
                f"rank {rank} exceeds max_rank {self.max_rank}"
            )
        if self._slot_names[slot_index] is not None:
            raise ValueError(
                f"Slot {slot_index} already occupied by {self._slot_names[slot_index]!r}"
            )
        # Check name uniqueness
        if name in self._slot_names:
            raise ValueError(
                f"Adapter {name!r} already assigned to slot {self._slot_names.index(name)}"
            )

        self._slot_names[slot_index] = name
        self._slot_ranks[slot_index] = rank
        self._slot_alphas[slot_index] = alpha

        # Init A: kaiming on first `rank` rows, rest stays zero
        self.lora_A[slot_index].data.zero_()
        if init_a == "kaiming" and rank > 0:
            nn.init.kaiming_uniform_(
                self.lora_A[slot_index].data[:rank, :], a=math.sqrt(5)
            )

        # B stays zero (silent until explicitly initialized)
        self.lora_B[slot_index].data.zero_()

        self._lora_scalings[slot_index] = alpha / rank if rank > 0 else 0.0
        self._slot_active[slot_index] = 1.0

    def release_adapter(self, slot_index: int) -> None:
        """Release slot, zero weights + metadata.

        Args:
            slot_index: Which slot to release.
        """
        if slot_index < 0 or slot_index >= self.max_adapters:
            raise IndexError(
                f"slot_index {slot_index} out of range [0, {self.max_adapters})"
            )
        self._slot_names[slot_index] = None
        self._slot_ranks[slot_index] = 0
        self._slot_alphas[slot_index] = 0.0
        self.lora_A[slot_index].data.zero_()
        self.lora_B[slot_index].data.zero_()
        self.lora_A[slot_index].requires_grad = False
        self.lora_B[slot_index].requires_grad = False
        self._lora_scalings[slot_index] = 0.0
        self._slot_active[slot_index] = 0.0

    def __getattr__(self, name: str):
        """Proxy unknown attributes to the base linear.

        FP8Linear has attributes (block_size, weight_scale, etc.) that
        the fused FeedForward chain accesses directly on self.w2. When
        w2 is wrapped by MultiLoRALinear, these lookups must reach the
        base. Standard nn.Module.__getattr__ checks _parameters, _buffers,
        _modules; if not found there, we delegate to the base linear.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)

    @property
    def out_features_val(self) -> int:
        return self.out_features

    def init_adapter_b(self, slot_index: int, std: float = 0.01) -> None:
        """Initialize adapter B matrix with small random values.

        Call this to give the adapter nonzero signal at scale=1.0,
        which is needed to produce nonzero MSE gradients at training start.

        Args:
            slot_index: Which slot to initialize.
            std: Standard deviation for normal initialization.
        """
        if slot_index < 0 or slot_index >= self.max_adapters:
            raise IndexError(
                f"slot_index {slot_index} out of range [0, {self.max_adapters})"
            )
        nn.init.normal_(self.lora_B[slot_index], std=std)

    def _compute_adapter_delta(
        self,
        x_2d: torch.Tensor,
        adapter_scales: torch.Tensor,
        token_to_image: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute adapter contribution in flattened 2D space.

        Shared implementation for forward() and adapter_delta().
        Branchless: empty slots (_slot_active=0) and zero-scale adapters
        produce zero contribution via arithmetic, not conditionals.
        The loop over max_adapters is a compile-time constant unrolled by
        torch.compile; no data-dependent branches remain.

        Args:
            x_2d: (n_tokens, in_features) flattened input.
            adapter_scales: (n_images, max_adapters) or (max_adapters,) scales.
            token_to_image: (seq_len,) routing or None.

        Returns:
            (n_tokens, out_features) adapter contribution.
        """
        n_tokens = x_2d.shape[0]

        # Normalize to (n_images, max_adapters) -- shape guard, not data-dependent
        if adapter_scales.dim() == 1:
            scales = adapter_scales.unsqueeze(0)  # (1, max_adapters)
        else:
            scales = adapter_scales  # (n_images, max_adapters)

        # Build per-token routing. No token_to_image → all tokens map to image 0
        if token_to_image is not None:
            routing = token_to_image
            if n_tokens > routing.shape[0]:
                routing = routing.repeat(n_tokens // routing.shape[0])
        else:
            routing = torch.zeros(n_tokens, dtype=torch.long, device=x_2d.device)

        # Padding sentinels get routing=-1; clamp for safe gather, zero via valid mask
        valid = routing >= 0
        safe_routing = routing.clamp(min=0)

        # Per-token scales: (n_tokens, max_adapters)
        token_scales = scales[safe_routing]
        token_scales = token_scales * valid.unsqueeze(1).to(token_scales.dtype)

        # Fold in slot_active and lora_scalings (both zero for empty slots)
        # effective: (n_tokens, max_adapters)
        effective = token_scales * (self._lora_scalings * self._slot_active).unsqueeze(0)

        adapter_out = x_2d.new_zeros(n_tokens, self.out_features)

        for j in range(self.max_adapters):
            # No data-dependent branch. Empty/zero slots: effective[:,j]=0 → contribution*0=0
            A = self.lora_A[j]   # (max_rank, in_features)
            B = self.lora_B[j]   # (out_features, max_rank)
            s = effective[:, j]  # (n_tokens,)

            h = x_2d @ A.t()                         # (n_tokens, max_rank)
            contribution = (h @ B.t()) * s.unsqueeze(1)  # (n_tokens, out_features)
            adapter_out = adapter_out + contribution.to(adapter_out.dtype)

        return adapter_out

    def forward(
        self,
        x: torch.Tensor,
        adapter_scales: torch.Tensor | None = None,
        token_to_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass: base linear + adapter contributions.

        Args:
            x: (..., in_features) input tensor.
            adapter_scales: (n_images, max_adapters) or (max_adapters,) scale tensor.
                If None, adapters are skipped (base output only).
            token_to_image: (total_len,) int32 tensor mapping each token position
                to an image index for per-image routing.

        Returns:
            (..., out_features) output tensor.
        """
        out = self.base(x)

        if adapter_scales is None:
            return out

        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        out_2d = out.reshape(-1, self.out_features)

        adapter_out = self._compute_adapter_delta(x_2d, adapter_scales, token_to_image)

        result = out_2d + adapter_out
        return result.reshape(orig_shape[:-1] + (self.out_features,))

    def adapter_delta(
        self,
        x: torch.Tensor,
        adapter_scales: torch.Tensor | None = None,
        token_to_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ONLY the adapter contribution, without the base linear.

        Used when the base linear is computed by a fused kernel chain
        (e.g., FP8 SiLU+gate+requant -> FP8 GEMM) that bypasses forward().
        The caller runs the fused chain for the base output, then adds
        this delta for the LoRA contribution.

        Args:
            x: (..., in_features) input tensor (BF16, pre-activation).
            adapter_scales: (n_images, max_adapters) or (max_adapters,) scales.
                If None, returns zeros.
            token_to_image: (total_len,) routing or None.

        Returns:
            (..., out_features) adapter-only contribution.
        """
        if adapter_scales is None:
            return x.new_zeros(*x.shape[:-1], self.out_features)

        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)

        adapter_out = self._compute_adapter_delta(x_2d, adapter_scales, token_to_image)

        return adapter_out.reshape(orig_shape[:-1] + (self.out_features,))


def _strip_orig_mod(name: str) -> str:
    """Remove ``._orig_mod`` segments inserted by torch.compile wrappers."""
    return name.replace("._orig_mod", "")


def _is_target_module(name: str, target_patterns: list[str] | None) -> bool:
    """Check if a module name matches any target pattern.

    Matching rules:
      - If target_patterns is None, match all Linear layers in main blocks
      - Patterns are matched against the full dotted module path
      - "layers." prefix matches main transformer blocks
      - Exact substring match (e.g., "attention.qkv" matches "layers.5.attention.qkv")
      - torch.compile ``._orig_mod`` segments are stripped before matching.
    """
    canonical = _strip_orig_mod(name)
    if target_patterns is None:
        return canonical.startswith("layers.")
    return any(pattern in canonical for pattern in target_patterns)


def _find_target_linears(
    model: nn.Module,
    target_patterns: list[str] | None,
    exclude_patterns: list[str] | None = None,
) -> dict[str, nn.Module]:
    """Find all Linear/FP8Linear modules matching target patterns.

    Args:
        model: The model to search.
        target_patterns: Patterns to match (None = all linears in layers.*).
        exclude_patterns: Patterns to exclude (e.g., ["score_proj", "final_layer"]).

    Returns:
        Dict mapping full dotted name -> module.
    """
    exclude = exclude_patterns or [
        "score_proj", "score_norm", "final_layer",
        "noise_refiner", "context_refiner",
        "t_embedder", "cap_embedder",
        "x_embedder", "rope_embedder",
        "adaLN_modulation",  # timestep->modulation, not a sequence-token linear
    ]

    targets = {}
    for name, module in model.named_modules():
        if not _is_target_module(name, target_patterns):
            continue
        canonical = _strip_orig_mod(name)
        if any(ex in canonical for ex in exclude):
            continue
        # Skip inner .base of an existing MultiLoRALinear (it's a child module
        # visible to named_modules but should not be independently targeted)
        if ".base" in canonical:
            continue
        # Already-wrapped MultiLoRALinear -- skip (install only replaces unwrapped linears)
        if isinstance(module, MultiLoRALinear):
            continue
        # Check if it's a Linear-like layer
        if isinstance(module, nn.Linear):
            targets[name] = module
        elif hasattr(module, "weight") and hasattr(module, "__call__"):
            # FP8Linear or similar custom linear
            if hasattr(module, "in_features") or (
                hasattr(module, "weight") and module.weight.dim() == 2
            ):
                targets[name] = module

    return targets


def install_multi_lora(
    model: nn.Module,
    max_adapters: int,
    max_rank: int,
    adapter_dtype: torch.dtype = torch.bfloat16,
    target_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> dict[str, MultiLoRALinear]:
    """Replace target Linear modules with pre-allocated MultiLoRALinear wrappers.

    Creates EMPTY slots. Call assign_adapter() to populate.

    Args:
        model: The model to modify in-place.
        max_adapters: Number of adapter slots per wrapper.
        max_rank: Maximum LoRA rank across all adapters.
        adapter_dtype: dtype for adapter parameters (default bfloat16).
        target_patterns: Override target patterns (None = default).
        exclude_patterns: Override exclude patterns (None = default).

    Returns:
        Dict mapping full module name -> MultiLoRALinear wrapper.
    """
    targets = _find_target_linears(model, target_patterns, exclude_patterns)

    wrappers: dict[str, MultiLoRALinear] = {}

    for name, module in targets.items():
        # Split name into parent path and attribute name
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model
            attr_name = parts[0]

        # Create wrapper (device inferred from module.weight automatically)
        wrapper = MultiLoRALinear(module, max_adapters, max_rank, adapter_dtype)

        # Replace in parent
        setattr(parent, attr_name, wrapper)
        wrappers[name] = wrapper

    return wrappers


def assign_adapter(
    model: nn.Module,
    slot_index: int,
    name: str,
    rank: int,
    alpha: float | None = None,
) -> int:
    """Assign named adapter to slot_index across all wrappers.

    Args:
        model: Model with MultiLoRALinear wrappers installed.
        slot_index: Which slot to assign to.
        name: Unique adapter identifier.
        rank: LoRA rank.
        alpha: LoRA alpha (defaults to float(rank)).

    Returns:
        Number of wrappers the adapter was assigned to.
    """
    if alpha is None:
        alpha = float(rank)
    count = 0
    for module in model.modules():
        if isinstance(module, MultiLoRALinear):
            module.assign_adapter(slot_index, name, rank, alpha)
            count += 1
    return count


def release_adapter(model: nn.Module, slot_index: int) -> int:
    """Release slot across all wrappers.

    Args:
        model: Model with MultiLoRALinear wrappers installed.
        slot_index: Which slot to release.

    Returns:
        Number of wrappers the slot was released from.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, MultiLoRALinear):
            module.release_adapter(slot_index)
            count += 1
    return count


def adapter_capacity(model: nn.Module) -> dict:
    """Query capacity from the first MultiLoRALinear wrapper found.

    Returns:
        Dict with keys:
            "max_adapters": int
            "max_rank": int
            "slots": list of dicts, each with:
                "index": int
                "name": str | None
                "rank": int
                "alpha": float
                "active": bool
    """
    for module in model.modules():
        if isinstance(module, MultiLoRALinear):
            slots = []
            for j in range(module.max_adapters):
                slots.append({
                    "index": j,
                    "name": module._slot_names[j],
                    "rank": module._slot_ranks[j],
                    "alpha": module._slot_alphas[j],
                    "active": module._slot_active[j].item() > 0.5,
                })
            return {
                "max_adapters": module.max_adapters,
                "max_rank": module.max_rank,
                "slots": slots,
            }
    return {"max_adapters": 0, "max_rank": 0, "slots": []}


def slot_index_for(model: nn.Module, adapter_name: str) -> int:
    """Look up slot index for a named adapter.

    Args:
        model: Model with MultiLoRALinear wrappers installed.
        adapter_name: Name of the adapter to find.

    Returns:
        Slot index.

    Raises:
        KeyError: If adapter_name is not assigned to any slot.
    """
    for module in model.modules():
        if isinstance(module, MultiLoRALinear):
            if adapter_name in module._slot_names:
                return module._slot_names.index(adapter_name)
    raise KeyError(f"Adapter {adapter_name!r} not assigned to any slot")


def set_adapter_trainable(
    model: nn.Module,
    adapter_name: str,
    trainable: bool = True,
) -> int:
    """Set requires_grad on all A/B params for a named adapter.

    Args:
        model: Model with MultiLoRALinear wrappers installed.
        adapter_name: Which adapter to make trainable/frozen.
        trainable: Whether to enable gradients.

    Returns:
        Number of parameters affected.
    """
    idx = slot_index_for(model, adapter_name)
    count = 0
    for module in model.modules():
        if isinstance(module, MultiLoRALinear):
            module.lora_A[idx].requires_grad = trainable
            module.lora_B[idx].requires_grad = trainable
            count += 2
    return count


def init_adapter_b_weights(
    model: nn.Module,
    adapter_name: str,
    std: float = 0.01,
) -> int:
    """Initialize B matrices for a named adapter across all MultiLoRALinear modules.

    Must be called AFTER install_multi_lora and AFTER torch.compile (if used),
    because compile captures the zero-init B matrices. Reinitializing after
    compile gives the adapter nonzero signal for training.

    Args:
        model: Model with MultiLoRALinear modules installed.
        adapter_name: Which adapter to initialize.
        std: Standard deviation for B matrix initialization.

    Returns:
        Number of modules initialized.
    """
    idx = slot_index_for(model, adapter_name)
    count = 0
    for module in model.modules():
        if isinstance(module, MultiLoRALinear):
            module.init_adapter_b(idx, std=std)
            count += 1
    return count


def freeze_base_params(model: nn.Module) -> int:
    """Freeze ALL parameters (including all LoRA slots).

    After calling this, NOTHING has requires_grad=True.
    Callers explicitly unfreeze what they need via
    set_adapter_trainable(model, name, True).

    Args:
        model: Model with MultiLoRALinear modules installed.

    Returns:
        Number of frozen parameters.
    """
    n_frozen = 0
    for _name, param in model.named_parameters():
        param.requires_grad = False
        n_frozen += 1
    return n_frozen


def unfreeze_score_head(model: nn.Module) -> int:
    """Unfreeze score_proj and score_norm parameters for joint training.

    Call after freeze_base_params() to enable training of the score head
    alongside LoRA adapters.

    Args:
        model: Model with score_proj and score_norm attributes.

    Returns:
        Number of parameters unfrozen.
    """
    count = 0
    for name, param in model.named_parameters():
        if "score_proj" in name or "score_norm" in name:
            param.requires_grad = True
            count += 1
    return count


def get_adapter_params(
    model: nn.Module,
    adapter_name: str,
) -> dict[str, nn.Parameter]:
    """Collect all A, B parameters for a named adapter.

    Args:
        model: Model with MultiLoRALinear modules installed.
        adapter_name: Which adapter's parameters to collect.

    Returns:
        Dict mapping parameter names to nn.Parameter objects.
        Keys are like: "layers.0.attention.qkv.lora_A.rtheta",
        "layers.0.attention.qkv.lora_B.rtheta", etc.
    """
    idx = slot_index_for(model, adapter_name)
    params = {}
    for name, module in model.named_modules():
        if isinstance(module, MultiLoRALinear):
            canonical = _strip_orig_mod(name)
            params[f"{canonical}.lora_A.{adapter_name}"] = module.lora_A[idx]
            params[f"{canonical}.lora_B.{adapter_name}"] = module.lora_B[idx]
    return params


def save_adapter(
    model: nn.Module,
    adapter_name: str,
    path: str,
) -> None:
    """Save one adapter's A, B weights to safetensors.

    Saves actual-rank-sized tensors (not max_rank padded) for backward
    compatibility and smaller checkpoints.

    Args:
        model: Model with MultiLoRALinear modules installed.
        adapter_name: Which adapter to save.
        path: Output safetensors file path.
    """
    idx = slot_index_for(model, adapter_name)
    tensors = {}
    for name, module in model.named_modules():
        if isinstance(module, MultiLoRALinear):
            canonical = _strip_orig_mod(name)
            rank = module._slot_ranks[idx]
            # Save only the active rank rows/cols, not the full max_rank
            tensors[f"{canonical}.lora_A.{adapter_name}"] = (
                module.lora_A[idx].data[:rank, :].contiguous()
            )
            tensors[f"{canonical}.lora_B.{adapter_name}"] = (
                module.lora_B[idx].data[:, :rank].contiguous()
            )

    if not tensors:
        raise ValueError(f"No adapter named {adapter_name!r} found in model")

    save_file(tensors, path)


def _remap_old_adapter_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap old-format adapter keys to new format.

    Old format: ``layers.0.attention.qkv.adapters.rtheta.lora_A``
    New format: ``layers.0.attention.qkv.lora_A.rtheta``

    Returns a new dict with remapped keys. Keys already in new format
    are passed through unchanged.
    """
    remapped = {}
    for key, tensor in state.items():
        if ".adapters." in key:
            parts = key.split(".")
            try:
                adapters_idx = parts.index("adapters")
                adapter_name = parts[adapters_idx + 1]
                lora_part = parts[adapters_idx + 2]
                prefix = ".".join(parts[:adapters_idx])
                new_key = f"{prefix}.{lora_part}.{adapter_name}"
                remapped[new_key] = tensor
            except (IndexError, ValueError):
                remapped[key] = tensor
        else:
            remapped[key] = tensor
    return remapped


def load_adapter(
    model: nn.Module,
    adapter_name: str,
    path: str,
) -> int:
    """Load one adapter's A, B weights from safetensors.

    Supports both old-format keys (``*.adapters.{name}.lora_{A,B}``) and
    new-format keys (``*.lora_{A,B}.{name}``). Old-format keys are
    remapped transparently.

    Checkpoint tensors may have different rank than max_rank. They are
    loaded into the active rank region, with zero-padding to max_rank.

    Args:
        model: Model with MultiLoRALinear wrappers installed (adapter must
            already be assigned via assign_adapter).
        adapter_name: Which adapter to load into.
        path: Input safetensors file path.

    Returns:
        Number of parameter tensors loaded.
    """
    import logging
    log = logging.getLogger(__name__)

    idx = slot_index_for(model, adapter_name)

    raw_state = load_file(path)
    state = _remap_old_adapter_keys(raw_state)

    matched_keys: set[str] = set()
    loaded = 0
    for name, module in model.named_modules():
        if isinstance(module, MultiLoRALinear):
            canonical = _strip_orig_mod(name)
            a_key = f"{canonical}.lora_A.{adapter_name}"
            b_key = f"{canonical}.lora_B.{adapter_name}"

            if a_key in state:
                ckpt_a = state[a_key]
                ckpt_rank = ckpt_a.shape[0]
                # Zero the slot then copy checkpoint data into active rows
                module.lora_A[idx].data.zero_()
                module.lora_A[idx].data[:ckpt_rank, :].copy_(ckpt_a)
                loaded += 1
                matched_keys.add(a_key)
            if b_key in state:
                ckpt_b = state[b_key]
                ckpt_rank = ckpt_b.shape[1]
                module.lora_B[idx].data.zero_()
                module.lora_B[idx].data[:, :ckpt_rank].copy_(ckpt_b)
                loaded += 1
                matched_keys.add(b_key)

    # Warn about unmatched checkpoint keys (expected for refiner tensors)
    unmatched = set(state.keys()) - matched_keys
    if unmatched:
        log.warning(
            f"[load_adapter] {len(unmatched)} checkpoint keys unmatched "
            f"(expected for refiner/excluded layers): "
            f"{sorted(unmatched)[:5]}{'...' if len(unmatched) > 5 else ''}"
        )

    if loaded == 0:
        file_keys = sorted(raw_state.keys())[:10]
        raise RuntimeError(
            f"load_adapter loaded 0 tensors for adapter {adapter_name!r} "
            f"from {path!r}. Checkpoint keys (first 10): {file_keys}. "
            f"Ensure the model has MultiLoRALinear wrappers installed with "
            f"adapter name {adapter_name!r} assigned."
        )

    return loaded


def adapter_summary(model: nn.Module) -> dict:
    """Summary statistics for all installed adapters.

    Returns:
        Dict with keys:
            "n_wrapped_layers": number of MultiLoRALinear modules
            "adapters": dict per adapter name with:
                "rank", "alpha", "n_params", "trainable", "slot_index"
    """
    n_wrapped = 0
    adapter_info: dict[str, dict] = {}

    for module in model.modules():
        if isinstance(module, MultiLoRALinear):
            n_wrapped += 1
            for j in range(module.max_adapters):
                name = module._slot_names[j]
                if name is None:
                    continue
                if name not in adapter_info:
                    adapter_info[name] = {
                        "rank": module._slot_ranks[j],
                        "alpha": module._slot_alphas[j],
                        "slot_index": j,
                        "n_params": 0,
                        "trainable": 0,
                    }
                A = module.lora_A[j]
                B = module.lora_B[j]
                n_a = A.numel()
                n_b = B.numel()
                adapter_info[name]["n_params"] += n_a + n_b
                if A.requires_grad:
                    adapter_info[name]["trainable"] += n_a
                if B.requires_grad:
                    adapter_info[name]["trainable"] += n_b

    return {
        "n_wrapped_layers": n_wrapped,
        "adapters": adapter_info,
    }

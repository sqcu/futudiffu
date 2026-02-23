"""Multi-tenant batched LoRA with per-image sparse adapter routing.

Multiple named LoRA adapters coexist on the same model. Each adapter has
independent A, B weight matrices on every target linear layer. A scale
tensor (n_images, n_adapters) controls which adapters are active for which
images in a packed FlexAttention batch. When scale=0 for an adapter on an
image, that adapter's computation is skipped for that image's tokens.

Lifecycle:
  1. install_multi_lora(model, configs) -- replace target linears with MultiLoRALinear
  2. model.forward(..., adapter_scales=..., token_to_image=...) -- routing
  3. save_adapter / load_adapter -- persist individual adapters

Implementation: Python loop over nonzero-scale adapters with masked
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

    Wraps a base linear layer (nn.Linear or FP8Linear) and adds N named
    adapters, each with its own A (down-projection) and B (up-projection)
    matrices. The adapter output is:

        sum_j[ (x @ A_j^T @ B_j^T) * (alpha_j / rank_j) * scale_j ]

    where scale_j comes from adapter_scales and can be per-image in a
    packed batch via token_to_image routing.

    When no adapter_scales are passed, returns base output only.
    When all scales are zero, adapter contribution is zero (B is zero-init
    by default, so even with scale=1 a fresh adapter has no effect).
    """

    def __init__(
        self,
        base_linear: nn.Module,
        adapter_configs: list[dict],
    ):
        """Initialize MultiLoRALinear wrapping an existing linear layer.

        Args:
            base_linear: The original nn.Linear or FP8Linear to wrap.
                NOT removed from the parent -- the caller (install_multi_lora)
                handles module replacement.
            adapter_configs: List of dicts, each with:
                - "name" (str): Unique adapter identifier
                - "rank" (int): LoRA rank (bottleneck dimension)
                - "alpha" (float): LoRA scaling factor
        """
        super().__init__()
        self.base = base_linear

        # Infer dimensions from base linear
        if hasattr(base_linear, "in_features"):
            self.in_features = base_linear.in_features
            self.out_features = base_linear.out_features
        elif hasattr(base_linear, "weight"):
            # FP8Linear: weight is (out_features, in_features) even if quantized
            self.out_features, self.in_features = base_linear.weight.shape
        else:
            raise ValueError(f"Cannot infer dimensions from {type(base_linear)}")

        self.n_adapters = len(adapter_configs)
        self.adapter_names: list[str] = []
        self.adapter_ranks: list[int] = []
        self.adapter_alphas: list[float] = []

        # Create A, B parameter pairs for each adapter as named sub-modules
        # so they appear in state_dict with structured names
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()

        for cfg in adapter_configs:
            name = cfg["name"]
            rank = cfg["rank"]
            alpha = cfg.get("alpha", float(rank))

            self.adapter_names.append(name)
            self.adapter_ranks.append(rank)
            self.adapter_alphas.append(alpha)

            # A: (rank, in_features) -- Kaiming uniform init
            A = nn.Parameter(torch.empty(rank, self.in_features))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))

            # B: (out_features, rank) -- zero init (adapter starts silent)
            B = nn.Parameter(torch.zeros(self.out_features, rank))

            self.lora_A[name] = A
            self.lora_B[name] = B

        # Precompute scaling factors: alpha / rank for each adapter
        self.register_buffer(
            "_lora_scalings",
            torch.tensor(
                [a / r for a, r in zip(self.adapter_alphas, self.adapter_ranks)],
                dtype=torch.float32,
            ),
        )

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
            # Proxy to base linear (FP8Linear attributes, etc.)
            return getattr(self.base, name)

    @property
    def out_features_val(self) -> int:
        return self.out_features

    def init_adapter_b(self, adapter_name: str, std: float = 0.01) -> None:
        """Initialize adapter B matrix with small random values.

        Call this to give the adapter nonzero signal at scale=1.0,
        which is needed to produce nonzero MSE gradients at training start.

        Args:
            adapter_name: Which adapter to initialize.
            std: Standard deviation for normal initialization.
        """
        if adapter_name not in self.lora_B:
            raise KeyError(f"No adapter named {adapter_name!r}")
        nn.init.normal_(self.lora_B[adapter_name], std=std)

    def _compute_adapter_delta(
        self,
        x_2d: torch.Tensor,
        adapter_scales: torch.Tensor,
        token_to_image: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute adapter contribution in flattened 2D space.

        Shared implementation for forward() and adapter_delta().

        Args:
            x_2d: (n_tokens, in_features) flattened input.
            adapter_scales: (n_images, n_adapters) or (n_adapters,) scales.
            token_to_image: (seq_len,) routing or None.

        Returns:
            (n_tokens, out_features) adapter contribution.
        """
        n_tokens = x_2d.shape[0]

        # Normalize scales shape
        if adapter_scales.dim() == 1:
            scales = adapter_scales.unsqueeze(0).expand(1, -1)
            per_token = False
        else:
            scales = adapter_scales
            per_token = token_to_image is not None

        adapter_out = x_2d.new_zeros(n_tokens, self.out_features)

        for j in range(self.n_adapters):
            name = self.adapter_names[j]
            A = self.lora_A[name]  # (rank, in_features)
            B = self.lora_B[name]  # (out_features, rank)
            scaling = self._lora_scalings[j]

            if per_token:
                per_image_scale = scales[:, j]

                if (per_image_scale.abs() < 1e-8).all():
                    continue

                routing = token_to_image
                if n_tokens > routing.shape[0]:
                    B_expand = n_tokens // routing.shape[0]
                    routing = routing.repeat(B_expand)

                safe_routing = routing.clamp(min=0)
                token_scale = per_image_scale[safe_routing]

                active_mask = (token_scale.abs() > 1e-8) & (routing >= 0)
                if not active_mask.any():
                    continue

                active_idx = active_mask.nonzero(as_tuple=True)[0]
                x_active = x_2d[active_idx]
                scale_active = token_scale[active_idx]

                h = x_active @ A.t()
                contribution = h @ B.t()
                contribution = contribution * (scaling * scale_active.unsqueeze(1))

                adapter_out[active_idx] += contribution.to(adapter_out.dtype)
            else:
                scale_val = scales[0, j]
                if scale_val.abs() < 1e-8:
                    continue

                h = x_2d @ A.t()
                contribution = h @ B.t()
                contribution = contribution * (scaling * scale_val)
                adapter_out += contribution.to(adapter_out.dtype)

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
            adapter_scales: (n_images, n_adapters) or (n_adapters,) scale tensor.
                If None, adapters are skipped (base output only).
            token_to_image: (total_len,) int32 tensor mapping each token position
                to an image index for per-image routing.

        Returns:
            (..., out_features) output tensor.
        """
        out = self.base(x)

        if adapter_scales is None or self.n_adapters == 0:
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
            adapter_scales: (n_images, n_adapters) or (n_adapters,) scales.
                If None, returns zeros.
            token_to_image: (total_len,) routing or None.

        Returns:
            (..., out_features) adapter-only contribution.
        """
        if adapter_scales is None or self.n_adapters == 0:
            return x.new_zeros(*x.shape[:-1], self.out_features)

        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)

        adapter_out = self._compute_adapter_delta(x_2d, adapter_scales, token_to_image)

        return adapter_out.reshape(orig_shape[:-1] + (self.out_features,))


def _is_target_module(name: str, target_patterns: list[str] | None) -> bool:
    """Check if a module name matches any target pattern.

    Matching rules:
      - If target_patterns is None, match all Linear layers in main blocks
      - Patterns are matched against the full dotted module path
      - "layers." prefix matches main transformer blocks
      - Exact substring match (e.g., "attention.qkv" matches "layers.5.attention.qkv")
    """
    if target_patterns is None:
        # Default: target all linears in main transformer blocks
        return name.startswith("layers.")
    return any(pattern in name for pattern in target_patterns)


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
        "adaLN_modulation",  # timestep→modulation, not a sequence-token linear
    ]

    targets = {}
    for name, module in model.named_modules():
        if not _is_target_module(name, target_patterns):
            continue
        if any(ex in name for ex in exclude):
            continue
        # Check if it's a Linear-like layer
        if isinstance(module, nn.Linear):
            targets[name] = module
        elif hasattr(module, "weight") and hasattr(module, "__call__"):
            # FP8Linear or similar custom linear
            # Verify it has the right shape attributes
            if hasattr(module, "in_features") or (
                hasattr(module, "weight") and module.weight.dim() == 2
            ):
                targets[name] = module

    return targets


def install_multi_lora(
    model: nn.Module,
    adapter_configs: list[dict],
    target_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> dict[str, MultiLoRALinear]:
    """Replace target Linear modules with MultiLoRALinear wrappers.

    Walks the model tree, finds Linear/FP8Linear modules matching the
    target patterns, and replaces each with a MultiLoRALinear that wraps
    the original and adds LoRA adapter parameters.

    Default targeting: all Linear layers in the 30 main transformer blocks
    (self.layers[*]). Does NOT target: embedders, refiners, final_layer,
    score_proj, score_norm.

    Args:
        model: The model to modify in-place.
        adapter_configs: List of adapter configs, each with:
            - "name" (str): Unique adapter name
            - "rank" (int): LoRA rank
            - "alpha" (float): LoRA alpha (defaults to rank)
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

        # Create wrapper
        wrapper = MultiLoRALinear(module, adapter_configs)

        # Move wrapper parameters to same device/dtype as base
        if hasattr(module, "weight"):
            device = module.weight.device
            # Use bf16 for adapter weights regardless of base (FP8 base, bf16 adapters)
            dtype = torch.bfloat16
            for param_name, param in wrapper.lora_A.items():
                wrapper.lora_A[param_name] = nn.Parameter(
                    param.to(device=device, dtype=dtype)
                )
            for param_name, param in wrapper.lora_B.items():
                wrapper.lora_B[param_name] = nn.Parameter(
                    param.to(device=device, dtype=dtype)
                )
            wrapper._lora_scalings = wrapper._lora_scalings.to(device=device)

        # Replace in parent
        setattr(parent, attr_name, wrapper)
        wrappers[name] = wrapper

    return wrappers


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
    count = 0
    for module in model.modules():
        if isinstance(module, MultiLoRALinear):
            if adapter_name in module.lora_B:
                module.init_adapter_b(adapter_name, std=std)
                count += 1
    return count


def freeze_base_params(model: nn.Module) -> tuple[int, int]:
    """Freeze all base model parameters, leave LoRA adapter params trainable.

    After calling this, only lora_A and lora_B parameters have
    requires_grad=True. Everything else (base weights, norms, embedders,
    score_proj, etc.) is frozen.

    Args:
        model: Model with MultiLoRALinear modules installed.

    Returns:
        (n_frozen, n_trainable) parameter counts.
    """
    n_frozen = 0
    n_trainable = 0

    for name, param in model.named_parameters():
        if "lora_A." in name or "lora_B." in name:
            param.requires_grad = True
            n_trainable += 1
        else:
            param.requires_grad = False
            n_frozen += 1

    return n_frozen, n_trainable


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
) -> dict[str, torch.Tensor]:
    """Collect all A, B parameters for a named adapter.

    Args:
        model: Model with MultiLoRALinear modules installed.
        adapter_name: Which adapter's parameters to collect.

    Returns:
        Dict mapping parameter names to tensors. Keys are like:
        "layers.0.attention.qkv.lora_A.rtheta",
        "layers.0.attention.qkv.lora_B.rtheta", etc.
    """
    params = {}
    for name, module in model.named_modules():
        if isinstance(module, MultiLoRALinear):
            if adapter_name in module.lora_A:
                params[f"{name}.lora_A.{adapter_name}"] = module.lora_A[adapter_name]
            if adapter_name in module.lora_B:
                params[f"{name}.lora_B.{adapter_name}"] = module.lora_B[adapter_name]
    return params


def save_adapter(
    model: nn.Module,
    adapter_name: str,
    path: str,
) -> None:
    """Save one adapter's A, B weights to safetensors.

    Args:
        model: Model with MultiLoRALinear modules installed.
        adapter_name: Which adapter to save.
        path: Output safetensors file path.
    """
    tensors = {}
    for name, module in model.named_modules():
        if isinstance(module, MultiLoRALinear):
            if adapter_name in module.lora_A:
                tensors[f"{name}.lora_A.{adapter_name}"] = (
                    module.lora_A[adapter_name].data.contiguous()
                )
            if adapter_name in module.lora_B:
                tensors[f"{name}.lora_B.{adapter_name}"] = (
                    module.lora_B[adapter_name].data.contiguous()
                )

    if not tensors:
        raise ValueError(f"No adapter named {adapter_name!r} found in model")

    save_file(tensors, path)


def load_adapter(
    model: nn.Module,
    adapter_name: str,
    path: str,
) -> int:
    """Load one adapter's A, B weights from safetensors.

    Args:
        model: Model with MultiLoRALinear modules installed (adapter must
            already exist from install_multi_lora).
        adapter_name: Which adapter to load into.
        path: Input safetensors file path.

    Returns:
        Number of parameter tensors loaded.
    """
    state = load_file(path)

    loaded = 0
    for name, module in model.named_modules():
        if isinstance(module, MultiLoRALinear):
            a_key = f"{name}.lora_A.{adapter_name}"
            b_key = f"{name}.lora_B.{adapter_name}"

            if a_key in state and adapter_name in module.lora_A:
                module.lora_A[adapter_name].data.copy_(state[a_key])
                loaded += 1
            if b_key in state and adapter_name in module.lora_B:
                module.lora_B[adapter_name].data.copy_(state[b_key])
                loaded += 1

    return loaded


def adapter_summary(model: nn.Module) -> dict:
    """Summary statistics for all installed adapters.

    Returns:
        Dict with keys:
            "n_wrapped_layers": number of MultiLoRALinear modules
            "adapters": dict per adapter name with:
                "rank", "alpha", "n_params", "trainable"
    """
    n_wrapped = 0
    adapter_info: dict[str, dict] = {}

    for module in model.modules():
        if isinstance(module, MultiLoRALinear):
            n_wrapped += 1
            for i, name in enumerate(module.adapter_names):
                if name not in adapter_info:
                    adapter_info[name] = {
                        "rank": module.adapter_ranks[i],
                        "alpha": module.adapter_alphas[i],
                        "n_params": 0,
                        "trainable": 0,
                    }
                A = module.lora_A[name]
                B = module.lora_B[name]
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

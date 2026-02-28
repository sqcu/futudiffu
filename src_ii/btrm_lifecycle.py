"""BTRM training lifecycle utilities for ZImageRLAIF.

Replaces the BTRMCompoundModel coordinator class with simple free functions
for model setup, optimizer construction, persist/load, and packed scoring.

The model IS a ZImageRLAIF. LoRA adapters are installed via multi_lora.py.
The score head is built into the model. This module provides the glue
between those components and the training loop.

Import constraints:
  - IMPORTS from src_ii.zimage_model: ZImageRLAIF, load_zimage_rlaif
  - IMPORTS from src_ii.multi_lora: assign_adapter, freeze_base_params,
    unfreeze_score_head, set_adapter_trainable, init_adapter_b_weights,
    get_adapter_params, save_adapter, load_adapter, adapter_summary
  - IMPORTS from src_ii.forward_packed: prepare_packed_forward, packed_forward
  - No futudiffu imports (all model access through ZImageRLAIF interface)
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from src_ii.multi_lora import (
    MultiLoRALinear,
    install_multi_lora,
    assign_adapter,
    freeze_base_params,
    unfreeze_score_head,
    set_adapter_trainable,
    init_adapter_b_weights,
    get_adapter_params,
    save_adapter,
    load_adapter,
)


DEFAULT_RTHETA_RANK = 8
DEFAULT_RTHETA_ALPHA = 16.0
DEFAULT_RTHETA_INIT_B_STD = 0.01


def setup_btrm_training(
    model: nn.Module,
    adapter_name: str = "rtheta",
    adapter_slot: int = 0,
    adapter_rank: int = DEFAULT_RTHETA_RANK,
    adapter_alpha: float = DEFAULT_RTHETA_ALPHA,
    adapter_init_b_std: float = DEFAULT_RTHETA_INIT_B_STD,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
    optimizer_type: str = "adam",
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    gradient_checkpointing: bool = True,
) -> torch.optim.Optimizer:
    """Set up a ZImageRLAIF model for BTRM training.

    Installs adapter capacity if not already present, assigns LoRA adapter
    to a slot, freezes base, unfreezes score head + adapter, initializes
    adapter B matrices, enables gradient checkpointing, and creates optimizer.

    Args:
        model: ZImageRLAIF model (capacity installed automatically if needed).
        adapter_name: LoRA adapter name.
        adapter_slot: Which pre-allocated slot to assign to.
        adapter_rank: LoRA rank.
        adapter_alpha: LoRA alpha.
        adapter_init_b_std: Std for B matrix init (nonzero = nonzero gradient).
        lr: Learning rate.
        weight_decay: Weight decay.
        betas: Adam betas.
        optimizer_type: "adam" or "muon".
        muon_lr: Muon learning rate (LoRA params).
        muon_momentum: Muon momentum.
        gradient_checkpointing: Whether to enable gradient checkpointing.

    Returns:
        Optimizer over adapter + score head params.
    """
    # Ensure adapter capacity is installed (idempotent: skips existing wrappers)
    has_wrappers = any(isinstance(m, MultiLoRALinear) for m in model.modules())
    if not has_wrappers:
        install_multi_lora(model, max_adapters=4, max_rank=max(16, adapter_rank))

    # Assign adapter to pre-allocated slot
    n_wrappers = assign_adapter(model, adapter_slot, adapter_name, adapter_rank, adapter_alpha)
    print(f"[btrm_lifecycle] Assigned adapter {adapter_name!r} to slot {adapter_slot} "
          f"across {n_wrappers} wrappers (rank={adapter_rank}, alpha={adapter_alpha})")

    # Freeze everything, then unfreeze score head + this adapter
    n_frozen = freeze_base_params(model)
    n_head_unfrozen = unfreeze_score_head(model)
    n_adapter_unfrozen = set_adapter_trainable(model, adapter_name, True)

    # Init B matrices for nonzero gradient signal
    n_init = init_adapter_b_weights(model, adapter_name, std=adapter_init_b_std)
    print(f"[btrm_lifecycle] Frozen={n_frozen}, head_unfrozen={n_head_unfrozen}, "
          f"adapter_unfrozen={n_adapter_unfrozen}, B_init={n_init}")

    # Gradient checkpointing
    model.gradient_checkpointing = gradient_checkpointing
    model.train()

    # Create optimizer
    optimizer = make_training_optimizer(
        model, adapter_name, lr=lr,
        weight_decay=weight_decay, betas=betas,
        optimizer_type=optimizer_type,
        muon_lr=muon_lr, muon_momentum=muon_momentum,
    )

    return optimizer


def make_training_optimizer(
    model: nn.Module,
    adapter_name: str,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
    optimizer_type: str = "adam",
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
) -> torch.optim.Optimizer:
    """Create optimizer over adapter params + score head params.

    Replaces BTRMCompoundModel.optimizer(). Structurally includes both
    adapter and score head params, preventing Defect 24.

    Args:
        model: ZImageRLAIF with MultiLoRALinear wrappers installed.
        adapter_name: Which adapter's params to include.
        lr: Learning rate.
        weight_decay: Weight decay.
        betas: Adam betas.
        optimizer_type: "adam" or "muon".
        muon_lr: Muon learning rate for LoRA params.
        muon_momentum: Muon momentum.

    Returns:
        Optimizer.
    """
    adapter_ps = list(get_adapter_params(model, adapter_name).values())
    head_ps = list(model.score_proj.parameters()) + list(model.score_norm.parameters())

    n_adapter = sum(p.numel() for p in adapter_ps)
    n_head = sum(p.numel() for p in head_ps)

    if optimizer_type == "muon":
        from torch.optim import Muon
        print(f"[btrm_lifecycle] Muon optimizer: {n_adapter} adapter params "
              f"(lr={muon_lr}) + {n_head} head params (lr={lr})")
        return Muon(
            muon_params=adapter_ps,
            lr=muon_lr,
            momentum=muon_momentum,
            adamw_params=head_ps,
            adamw_lr=lr,
            adamw_betas=betas,
            adamw_wd=weight_decay,
        )
    else:
        print(f"[btrm_lifecycle] AdamW: {n_adapter} adapter + "
              f"{n_head} head = {n_adapter + n_head} params")
        return torch.optim.AdamW(
            adapter_ps + head_ps, lr=lr,
            weight_decay=weight_decay, betas=betas,
        )


def get_all_trainable_params(model: nn.Module, adapter_name: str) -> list[nn.Parameter]:
    """Return all trainable params: adapter + score head.

    Replaces BTRMCompoundModel.all_trainable_params().
    """
    adapter_ps = list(get_adapter_params(model, adapter_name).values())
    head_ps = list(model.score_proj.parameters()) + list(model.score_norm.parameters())
    return adapter_ps + head_ps


def score_packed(
    model: nn.Module,
    images: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]],
    gradient_checkpointing: bool = True,
    executor=None,
) -> torch.Tensor:
    """Score N images via scatter-gather through BatchExecutor.

    Images are submitted as queries with identity fork specs. The executor
    bins them into REFERENCE_TOTAL_LEN-sized launches via FFD, executes
    each bin as a packed forward, and returns tagged scores. This function
    collects scores in submission order.

    BTRM training: adapter_scales is intentionally NOT passed. The backward
    flows only through score_norm + score_proj (the score head), not through
    the full backbone. LoRA adapters are trained during policy optimization,
    not during BTRM reward model training.

    Args:
        model: ZImageRLAIF model (compiled or raw).
        images: List of (latent, timestep, conditioning, num_tokens) tuples.
            latent: (1, C, H, W) noisy latent.
            timestep: (1,) sigma value.
            conditioning: (1, seq, cap_feat_dim) text conditioning.
            num_tokens: Number of text tokens.
        gradient_checkpointing: Whether to use gradient checkpointing.
        executor: Optional pre-built executor (BatchExecutor or AcceleratorPool).
            If None, creates a throwaway BatchExecutor (today's behavior).

    Returns:
        (N, n_score_heads) score tensor with grad_fn.
    """
    device = images[0][0].device

    # Build queries: each image is a single-fork identity query
    queries = []
    for i, (latent, timestep, conditioning, num_tokens) in enumerate(images):
        queries.append({
            "query_id": f"score_{i}",
            "base_latent": latent,
            "base_cond": conditioning,
            "base_cap_len": num_tokens,
            "base_resolution": (latent.shape[3] * 8, latent.shape[2] * 8),
            "sigma": float(timestep),
            "forks": [{"entry_id": "e0"}],
        })

    old_gc = getattr(model, 'gradient_checkpointing', False)
    model.gradient_checkpointing = gradient_checkpointing

    if executor is None:
        from src_ii.batch_executor import BatchExecutor
        executor = BatchExecutor(model, device=device)

    results = executor.execute(queries)

    model.gradient_checkpointing = old_gc

    # Collect scores in submission order
    score_map = {}
    for r in results:
        idx = int(r["query_id"].split("_")[1])
        score_map[idx] = r["scores"].to(device)

    return torch.stack([score_map[i] for i in range(len(images))])


def score_serial(
    model: nn.Module,
    latent: torch.Tensor,
    timestep: torch.Tensor,
    conditioning: torch.Tensor,
    num_tokens: int,
    gradient_checkpointing: bool = True,
) -> torch.Tensor:
    """Score a single image (serial path).

    Replaces BTRMCompoundModel.score_differentiable() for the single-image case.
    Wraps the image as a packed batch of size 1.

    Args:
        model: ZImageRLAIF model.
        latent: (1, C, H, W) noisy latent.
        timestep: (1,) sigma.
        conditioning: (1, seq, cap_feat_dim).
        num_tokens: Caption token count.
        gradient_checkpointing: Whether to use gradient checkpointing.

    Returns:
        (1, n_score_heads) score tensor with grad_fn.
    """
    return score_packed(
        model,
        [(latent, timestep, conditioning, num_tokens)],
        gradient_checkpointing=gradient_checkpointing,
    )


def persist_btrm(
    model: nn.Module,
    adapter_name: str,
    output_dir: str | Path,
    head_names: Sequence[str],
) -> dict:
    """Save adapter + score head + config.

    Replaces BTRMCompoundModel.persist().

    Creates:
        output_dir/rtheta_adapter.safetensors
        output_dir/btrm_head.safetensors
        output_dir/btrm_compound_config.json

    Args:
        model: ZImageRLAIF model with score head and adapters.
        adapter_name: LoRA adapter name to persist.
        output_dir: Directory to write checkpoint files into.
        head_names: Ordered list of score head names. Required — no defaults.

    Returns:
        Manifest dict with file paths and param counts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save adapter
    adapter_path = output_dir / "rtheta_adapter.safetensors"
    save_adapter(model, adapter_name, str(adapter_path))

    # Save score head (score_proj + score_norm)
    head_sd = {}
    for name, param in model.named_parameters():
        if "score_proj" in name or "score_norm" in name:
            head_sd[name] = param.data.cpu().contiguous()
    head_path = output_dir / "btrm_head.safetensors"
    save_file(head_sd, str(head_path))

    # Save config
    config = {
        "adapter_name": adapter_name,
        "n_score_heads": model.n_score_heads,
        "score_cap": model.score_cap,
        "head_names": list(head_names),
    }
    config_path = output_dir / "btrm_compound_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    n_adapter = sum(
        p.numel() for p in get_adapter_params(model, adapter_name).values()
    )
    n_head = sum(v.numel() for v in head_sd.values())

    manifest = {
        "adapter_path": str(adapter_path),
        "head_path": str(head_path),
        "config_path": str(config_path),
        "n_adapter_params": n_adapter,
        "n_head_params": n_head,
    }
    print(f"[btrm_lifecycle] Persisted: {n_adapter} adapter + {n_head} head params to {output_dir}")
    return manifest


def load_btrm(
    model: nn.Module,
    adapter_name: str,
    input_dir: str | Path,
) -> None:
    """Load adapter + score head weights from a persist_btrm() directory.

    Replaces BTRMCompoundModel.load(). Assumes the model already has
    MultiLoRALinear wrappers installed with the right adapter name.

    Args:
        model: ZImageRLAIF with MultiLoRALinear wrappers.
        adapter_name: Which adapter to load into.
        input_dir: Directory containing rtheta_adapter.safetensors and
            btrm_head.safetensors.
    """
    input_dir = Path(input_dir)

    adapter_path = input_dir / "rtheta_adapter.safetensors"
    head_path = input_dir / "btrm_head.safetensors"

    if not adapter_path.exists():
        raise FileNotFoundError(f"Missing adapter: {adapter_path}")
    if not head_path.exists():
        raise FileNotFoundError(f"Missing head: {head_path}")

    # Load adapter (raises RuntimeError if 0 tensors loaded)
    n_loaded = load_adapter(model, adapter_name, str(adapter_path))
    print(f"[btrm_lifecycle] Loaded {n_loaded} adapter tensors from {adapter_path}")

    # Load score head with dual-format key support
    raw_head_sd = load_file(str(head_path))

    # Remap old key format: norm.weight -> score_norm.weight,
    # proj.weight -> score_proj.weight
    _HEAD_KEY_REMAP = {
        "norm.weight": "score_norm.weight",
        "proj.weight": "score_proj.weight",
    }
    head_sd = {}
    for name, tensor in raw_head_sd.items():
        new_name = _HEAD_KEY_REMAP.get(name, name)
        head_sd[new_name] = tensor

    head_loaded = 0
    for name, tensor in head_sd.items():
        # Navigate to the parameter
        parts = name.split(".")
        obj = model
        try:
            for part in parts[:-1]:
                obj = getattr(obj, part)
            param = getattr(obj, parts[-1])
            param.data.copy_(tensor.to(param.device))
            head_loaded += 1
        except AttributeError:
            print(f"[btrm_lifecycle] WARNING: head key {name!r} not found on model")

    if head_loaded == 0:
        raise RuntimeError(
            f"load_btrm loaded 0 head tensors from {head_path!r}. "
            f"Checkpoint keys: {sorted(raw_head_sd.keys())}. "
            f"Expected keys like 'score_norm.weight', 'score_proj.weight'."
        )

    print(f"[btrm_lifecycle] Loaded {head_loaded} head tensors from {head_path}")

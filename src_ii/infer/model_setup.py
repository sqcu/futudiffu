"""Model loading and adapter setup for inference scripts.

Wraps the full model lifecycle — load ZImageRLAIF, install LoRA adapters,
load BTRM weights, optionally load policy adapters with key remapping,
patch sage attention for torch.compile, and compile — into two callable
functions that inference scripts treat as single operations.

Primary consumers: demonstrate_rtheta_policy.py, validate_policy_intervention.py

Key contract: the returned compiled model is ready for eval-mode forward
passes. The raw model is returned alongside so callers can inspect weights,
install additional adapters, or change adapter_scales without recompiling.

Import constraints:
  - Top-level: json, pathlib.Path, torch, torch.nn
  - Deferred inside functions: src_ii.zimage_model, src_ii.attention_srcii,
    src_ii.multi_lora, src_ii.btrm_lifecycle, safetensors.torch
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn


def load_and_prepare_model(
    fp8_path: str,
    adapter_assignments: list[dict],
    btrm_dir: str | Path,
    btrm_adapter_name: str,
    max_adapters: int = 4,
    max_rank: int = 16,
    extra_adapter_loads: list[dict] | None = None,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[nn.Module, nn.Module, list[str]]:
    """Load ZImageRLAIF, install pre-allocated LoRA capacity, load weights, compile.

    Executes the full model construction pipeline in one call:
      1. Load ZImageRLAIF from FP8 checkpoint (fused, sage attention enabled).
      2. Read btrm_compound_config.json for head_names.
      3. Install MultiLoRALinear wrappers with pre-allocated capacity.
      4. Assign adapters to explicit slots.
      5. Load BTRM adapter weights and score head from btrm_dir.
      6. Load any extra policy adapters via load_policy_adapter().
      7. Set eval mode and torch.compile(mode="default").

    Args:
        fp8_path: Path to the FP8 blockwise safetensors checkpoint.
        adapter_assignments: List of adapter assignment specs, each a dict:
            "slot" (int): Pre-allocated slot index.
            "name" (str): Unique adapter name.
            "rank" (int): LoRA rank.
            "alpha" (float): LoRA alpha.
            Example: [{"slot": 0, "name": "rtheta", "rank": 8, "alpha": 16.0}]
        btrm_dir: Directory containing btrm_compound_config.json,
            rtheta_adapter.safetensors, and btrm_head.safetensors.
        btrm_adapter_name: Which assigned adapter to load BTRM weights into.
        max_adapters: Number of pre-allocated adapter slots.
        max_rank: Maximum LoRA rank across all adapters.
        extra_adapter_loads: Optional list of policy adapter load specs,
            each a dict with keys:
              "target_name" (str): adapter name already assigned,
              "path" (str): path to safetensors checkpoint,
              "source_name" (str, optional): adapter name embedded in checkpoint
                  keys (default "policy_pinkify").
        device: Target device.
        dtype: Base dtype for non-FP8 parameters (typically bfloat16).

    Returns:
        (compiled_model, raw_model, head_names) where:
          compiled_model: torch.compile'd model ready for eval forward passes.
          raw_model: Uncompiled model; use for weight inspection or adapter
              manipulation without triggering recompilation.
          head_names: List of score head names from btrm_compound_config.json,
              e.g. ["pinkify", "thisnotthat"].
    """
    from src_ii.zimage_model import load_zimage_rlaif
    from src_ii.multi_lora import install_multi_lora, assign_adapter
    from src_ii.btrm_lifecycle import load_btrm

    t0 = time.perf_counter()

    # Step 1: load model
    raw_model = load_zimage_rlaif(
        fp8_path,
        device=device,
        dtype=dtype,
        compile_model=False,
        fuse=True,
        use_sage=True,
    )
    print(
        f"[model_setup] Model loaded+fused: {time.perf_counter() - t0:.1f}s  "
        f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB",
        flush=True,
    )

    # Step 2: read btrm config for head_names
    btrm_dir = Path(btrm_dir)
    config_path = btrm_dir / "btrm_compound_config.json"
    with open(config_path) as f:
        btrm_config = json.load(f)
    head_names: list[str] = btrm_config["head_names"]
    print(
        f"[model_setup] BTRM config: heads={head_names}  "
        f"rank={btrm_config.get('adapter_rank')}  alpha={btrm_config.get('adapter_alpha')}",
        flush=True,
    )

    # Step 3: install pre-allocated capacity
    wrappers = install_multi_lora(raw_model, max_adapters=max_adapters, max_rank=max_rank)
    print(
        f"[model_setup] Installed {len(wrappers)} MultiLoRALinear wrappers "
        f"(max_adapters={max_adapters}, max_rank={max_rank})",
        flush=True,
    )

    # Step 4: assign adapters to explicit slots
    for spec in adapter_assignments:
        assign_adapter(raw_model, spec["slot"], spec["name"], spec["rank"], spec["alpha"])
        print(
            f"[model_setup] Assigned adapter {spec['name']!r} to slot {spec['slot']} "
            f"(rank={spec['rank']}, alpha={spec['alpha']})",
            flush=True,
        )

    # Step 5: load BTRM adapter + score head
    load_btrm(raw_model, btrm_adapter_name, btrm_dir)
    print(
        f"[model_setup] BTRM weights loaded (adapter={btrm_adapter_name!r})  "
        f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB",
        flush=True,
    )

    # Step 6: load extra policy adapters if any
    if extra_adapter_loads:
        for spec in extra_adapter_loads:
            target_name = spec["target_name"]
            path = spec["path"]
            source_name = spec.get("source_name", "policy_pinkify")
            n = load_policy_adapter(raw_model, target_name, path, source_name)
            print(
                f"[model_setup] Policy adapter loaded: {target_name!r} "
                f"({n} tensors from {Path(path).name}, source={source_name!r})",
                flush=True,
            )

    # Step 7: eval + compile
    raw_model.eval()
    compiled = torch.compile(raw_model, mode="default")
    print(
        f"[model_setup] torch.compile done  "
        f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB  "
        f"total: {time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    return compiled, raw_model, head_names


def load_policy_adapter(
    model: nn.Module,
    target_name: str,
    ckpt_path: str | Path,
    source_name: str = "policy_pinkify",
) -> int:
    """Load a DDGRPO policy adapter with _orig_mod key remapping.

    DDGRPO policy adapters are saved from torch.compiled models, so their
    state dict keys contain "._orig_mod." (the compiled model's internal
    attribute prefix). Loading into an uncompiled model requires stripping
    those prefixes. Additionally, the adapter name embedded in the checkpoint
    keys (source_name) may differ from the adapter slot already assigned in
    the model (target_name).

    Key transformation applied to every key in the checkpoint:
      1. Replace every occurrence of "._orig_mod." with ".".
      2. Replace every occurrence of source_name with target_name in key paths.

    Args:
        model: Model with MultiLoRALinear wrappers installed and target_name
            assigned to a slot (via assign_adapter).
        target_name: Name of the adapter to load into.
        ckpt_path: Path to safetensors checkpoint containing adapter weights.
        source_name: Adapter name embedded in the checkpoint keys. Defaults
            to "policy_pinkify" which is what DDGRPO training saves.

    Returns:
        Number of adapter parameter tensors loaded (lora_A + lora_B combined).

    Raises:
        RuntimeError: If zero tensors are loaded (mismatched checkpoint,
            wrong source_name, or adapter not assigned in model).
    """
    from safetensors.torch import load_file
    from src_ii.multi_lora import MultiLoRALinear, _strip_orig_mod, slot_index_for

    ckpt_path = Path(ckpt_path)
    raw_state = load_file(str(ckpt_path))
    print(
        f"[model_setup] load_policy_adapter: loaded {len(raw_state)} raw keys "
        f"from {ckpt_path.name}",
        flush=True,
    )

    # Remap keys: strip _orig_mod prefix, rename source adapter to target
    remapped: dict[str, torch.Tensor] = {}
    for key, tensor in raw_state.items():
        new_key = key.replace("._orig_mod.", ".")
        if source_name != target_name:
            new_key = new_key.replace(source_name, target_name)
        remapped[new_key] = tensor

    # Resolve slot index for target adapter
    idx = slot_index_for(model, target_name)

    # Copy matching tensors into model's MultiLoRALinear modules
    loaded = 0
    for name, module in model.named_modules():
        if not isinstance(module, MultiLoRALinear):
            continue
        canonical = _strip_orig_mod(name)
        a_key = f"{canonical}.lora_A.{target_name}"
        b_key = f"{canonical}.lora_B.{target_name}"

        if a_key in remapped:
            ckpt_a = remapped[a_key]
            ckpt_rank = ckpt_a.shape[0]
            module.lora_A[idx].data.zero_()
            module.lora_A[idx].data[:ckpt_rank, :].copy_(ckpt_a)
            loaded += 1
        if b_key in remapped:
            ckpt_b = remapped[b_key]
            ckpt_rank = ckpt_b.shape[1]
            module.lora_B[idx].data.zero_()
            module.lora_B[idx].data[:, :ckpt_rank].copy_(ckpt_b)
            loaded += 1

    if loaded == 0:
        sample_keys = sorted(remapped.keys())[:8]
        raise RuntimeError(
            f"load_policy_adapter loaded 0 tensors for target={target_name!r}, "
            f"source={source_name!r} from {ckpt_path}. "
            f"Remapped keys (first 8): {sample_keys}. "
            f"Check that assign_adapter was called with name={target_name!r} "
            f"and that source_name matches the adapter name in the checkpoint."
        )

    return loaded

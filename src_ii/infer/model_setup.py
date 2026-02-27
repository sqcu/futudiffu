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
    adapter_configs: list[dict],
    btrm_dir: str | Path,
    btrm_adapter_name: str,
    extra_adapter_loads: list[dict] | None = None,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[nn.Module, nn.Module, list[str]]:
    """Load ZImageRLAIF, install LoRA adapters, load weights, and compile.

    Executes the full model construction pipeline in one call:
      1. Load ZImageRLAIF from FP8 checkpoint (fused, sage attention enabled).
      2. Read btrm_compound_config.json for head_names.
      3. Install MultiLoRALinear wrappers for each adapter_config entry.
      4. Load BTRM adapter weights and score head from btrm_dir.
      5. Load any extra policy adapters via load_policy_adapter().
      6. Patch sage attention custom ops for torch.compile AOT compatibility.
      7. Set eval mode and torch.compile(mode="default").

    Args:
        fp8_path: Path to the FP8 blockwise safetensors checkpoint.
        adapter_configs: List of adapter specs, each a dict with keys:
            "name" (str), "rank" (int), "alpha" (float).
            Example: [{"name": "rtheta", "rank": 8, "alpha": 16.0}]
        btrm_dir: Directory containing btrm_compound_config.json,
            rtheta_adapter.safetensors, and btrm_head.safetensors.
        btrm_adapter_name: Which installed adapter to load BTRM weights into.
            Must match a "name" entry in adapter_configs.
        extra_adapter_loads: Optional list of policy adapter load specs,
            each a dict with keys:
              "target_name" (str): adapter name already installed in adapter_configs,
              "path" (str): path to safetensors checkpoint,
              "source_name" (str, optional): adapter name embedded in checkpoint
                  keys (default "policy_pinkify").
            Checkpoints saved from torch.compiled models have "._orig_mod." in
            their keys; load_policy_adapter strips those automatically.
        device: Target device.
        dtype: Base dtype for non-FP8 parameters (typically bfloat16).

    Returns:
        (compiled_model, raw_model, head_names) where:
          compiled_model: torch.compile'd model ready for eval forward passes.
          raw_model: Uncompiled model; use for weight inspection or adapter
              manipulation without triggering recompilation.
          head_names: List of score head names from btrm_compound_config.json,
              e.g. ["bit_quality", "step_quality"].
    """
    from src_ii.zimage_model import load_zimage_rlaif
    from src_ii.multi_lora import install_multi_lora
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

    # Step 3: install LoRA wrappers
    wrappers = install_multi_lora(raw_model, adapter_configs)
    print(
        f"[model_setup] Installed {len(wrappers)} MultiLoRALinear wrappers "
        f"for adapters: {[c['name'] for c in adapter_configs]}",
        flush=True,
    )

    # Step 4: load BTRM adapter + score head
    load_btrm(raw_model, btrm_adapter_name, btrm_dir)
    print(
        f"[model_setup] BTRM weights loaded (adapter={btrm_adapter_name!r})  "
        f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB",
        flush=True,
    )

    # Step 5: load extra policy adapters if any
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

    # Step 6: eval + compile
    # Note: sage attention ops are compile-compatible at import time via
    # register_fake in attention_srcii.py — no patching step needed.
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
    keys (source_name) may differ from the adapter slot already installed in
    the model (target_name).

    Key transformation applied to every key in the checkpoint:
      1. Replace every occurrence of "._orig_mod." with ".".
      2. Replace every occurrence of source_name with target_name in key paths.
         This is a substring replace on the full key, so it only fires where
         the adapter name appears in a lora_A/lora_B key segment.

    Args:
        model: Model with MultiLoRALinear wrappers already installed for
            target_name (via install_multi_lora).
        target_name: Name of the adapter slot to load into.
        ckpt_path: Path to safetensors checkpoint containing adapter weights.
        source_name: Adapter name embedded in the checkpoint keys. Defaults
            to "policy_pinkify" which is what DDGRPO training saves.

    Returns:
        Number of adapter parameter tensors loaded (lora_A + lora_B combined).
        Each MultiLoRALinear contributes 2 tensors (A and B).

    Raises:
        RuntimeError: If zero tensors are loaded (mismatched checkpoint,
            wrong source_name, or adapter not installed in model).
    """
    from safetensors.torch import load_file
    from src_ii.multi_lora import MultiLoRALinear

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

    # Copy matching tensors into model's MultiLoRALinear modules
    from src_ii.multi_lora import _strip_orig_mod
    loaded = 0
    for name, module in model.named_modules():
        if not isinstance(module, MultiLoRALinear):
            continue
        canonical = _strip_orig_mod(name)
        a_key = f"{canonical}.lora_A.{target_name}"
        b_key = f"{canonical}.lora_B.{target_name}"

        if a_key in remapped and target_name in module.lora_A:
            module.lora_A[target_name].data.copy_(remapped[a_key])
            loaded += 1
        if b_key in remapped and target_name in module.lora_B:
            module.lora_B[target_name].data.copy_(remapped[b_key])
            loaded += 1

    if loaded == 0:
        sample_keys = sorted(remapped.keys())[:8]
        raise RuntimeError(
            f"load_policy_adapter loaded 0 tensors for target={target_name!r}, "
            f"source={source_name!r} from {ckpt_path}. "
            f"Remapped keys (first 8): {sample_keys}. "
            f"Check that install_multi_lora was called with target_name={target_name!r} "
            f"and that source_name matches the adapter name in the checkpoint."
        )

    return loaded

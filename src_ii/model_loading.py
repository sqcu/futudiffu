"""Model loading without ModelManager.

Extracts the FP8 diffusion model loading sequence from ModelManager.ensure_diffusion()
into a standalone function. Uses futudiffu.diffusion_model for architecture and
futudiffu.fp8 for FP8 weight injection, but not model_manager.py itself.

Import constraints:
  - IMPORTS from futudiffu: diffusion_model (architecture), fp8 (FP8Linear),
    lora (adapter allocation/init), sage_attention (optional)
  - DOES NOT import: model_manager, server, client, training_utils
"""

import time

import torch
from safetensors.torch import load_file


def load_fp8_diffusion_model(
    fp8_safetensors_path: str,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
    fp8_block_size: int = 128,
    compile_model: bool = True,
    fuse: bool = True,
) -> tuple:
    """Load the FP8 NextDiT diffusion model from safetensors.

    Replicates the loading sequence from ModelManager.ensure_diffusion():
      1. Load safetensors state dict
      2. Create meta-device model skeleton
      3. Replace nn.Linear with FP8Linear where state_dict has FP8 weights
      4. Load remaining (non-FP8) weights
      5. Move to device, eval mode
      6. Fuse model (w1w3, FP8 chain, elementwise, QKV, adaLN batching)
      7. Optionally torch.compile

    Args:
        fp8_safetensors_path: Path to the FP8 safetensors checkpoint.
        device: Target CUDA device.
        dtype: Working dtype (bfloat16).
        fp8_block_size: FP8 blockwise quantization block size (128).
        compile_model: Whether to torch.compile the model.
        fuse: Whether to apply model fusions.

    Returns:
        If compile_model:
            (diff_compiled, diff_model) where diff_compiled wraps forward().
        Else:
            (diff_model, diff_model) -- same object twice for uniform API.
    """
    from futudiffu.diffusion_model import (
        _detect_cap_feat_dim,
        _detect_n_layers,
        _detect_qk_norm,
        _strip_diffusion_prefix,
        create_diffusion_model,
        fuse_model,
    )
    from futudiffu.fp8 import replace_linear_with_fp8

    t0 = time.perf_counter()
    print(f"[model_loading] Loading FP8 weights from {fp8_safetensors_path}")

    diff_sd = load_file(fp8_safetensors_path, device=str(device))
    remapped = _strip_diffusion_prefix(diff_sd)
    del diff_sd

    n_layers = _detect_n_layers(remapped.keys())
    cap_feat_dim = _detect_cap_feat_dim(remapped)
    qk_norm = _detect_qk_norm(remapped.keys())
    print(f"[model_loading] Detected: n_layers={n_layers}, cap_feat_dim={cap_feat_dim}, qk_norm={qk_norm}")

    model = create_diffusion_model(
        dtype=dtype, n_layers=n_layers,
        cap_feat_dim=cap_feat_dim, qk_norm=qk_norm,
    )

    replace_linear_with_fp8(
        model, remapped, block_size=fp8_block_size,
        output_dtype=dtype,
    )

    # Load remaining non-FP8 weights (filter out scale/quant metadata)
    remaining = {k: v for k, v in remapped.items()
                 if not k.endswith((".weight_scale", ".comfy_quant"))}
    model.load_state_dict(remaining, strict=False, assign=True)
    del remapped, remaining

    model = model.to(device)
    model.eval()

    if fuse:
        fuse_model(model)

    elapsed_load = time.perf_counter() - t0
    print(f"[model_loading] Model loaded and fused in {elapsed_load:.1f}s")

    if compile_model:
        t1 = time.perf_counter()
        diff_compiled = torch.compile(model, mode="default")
        elapsed_compile = time.perf_counter() - t1
        print(f"[model_loading] torch.compile wrapper created in {elapsed_compile:.1f}s")
        return diff_compiled, model
    else:
        return model, model


def configure_sage_attention(qk_quant: str = "int8", pv_quant: str = "bf16"):
    """Configure SageAttention if available.

    Args:
        qk_quant: QK quantization type ("int8" or "fp8").
        pv_quant: PV quantization type ("bf16").
    """
    try:
        from futudiffu.sage_attention import configure_sage
        configure_sage(smooth_k=True, qk_quant=qk_quant, pv_quant=pv_quant)
        print(f"[model_loading] SageAttention configured: qk={qk_quant}, pv={pv_quant}")
    except ImportError:
        print("[model_loading] SageAttention not available, using SDPA")

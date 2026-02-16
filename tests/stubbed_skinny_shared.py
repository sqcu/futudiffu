"""Stubbed-Skinny-Shared (S-S-S) Z-Image test config.

Builds a sub-gigabyte diffusion model that uses REAL FP8 tensor data sliced
from the full Z-Image model, exercising real tensor core FP8 paths.

Design:
- Extract ONE real layer from z_image_fp8_blockwise.safetensors
- Use 1/4 of the attention heads: n_heads=8 -> dim=1024 (head_dim=128 preserved)
- Loop that single layer 4 times (shared weights) to form a 4-"layer" model
- Total VRAM: ~200-400 MB (vs 5.8 GB full model)

The model works with existing forward() and forward_packed() paths.
"""

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

from futudiffu.diffusion_model import (
    NextDiT,
    JointTransformerBlock,
    JointAttention,
    FeedForward,
    FinalLayer,
    RMSNormModule,
    TimestepEmbedder,
    EmbedND,
)

# S-S-S config constants
SSS_DIM = 1024           # 8 heads * 128 head_dim
SSS_N_HEADS = 8          # 1/4 of 30 (rounded)
SSS_N_KV_HEADS = 8
SSS_HEAD_DIM = 128       # Same as real model
SSS_N_LAYERS = 4         # Shared-weight virtual layers
SSS_N_REFINER = 2        # Same as real model
SSS_CAP_FEAT_DIM = 2560  # Same as real model
SSS_PATCH_SIZE = 2
SSS_IN_CHANNELS = 16
SSS_PAD_TOKENS_MULTIPLE = 32
SSS_AXES_DIMS = [32, 48, 48]  # Sum = 128 = head_dim
SSS_AXES_LENS = [1536, 512, 512]
SSS_ROPE_THETA = 256.0

# FP8 model path (Windows)
FP8_MODEL_PATH = os.environ.get(
    "FUTUDIFFU_FP8_PATH",
    r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors",
)


def _compute_ffn_hidden_dim(dim: int, ffn_dim_multiplier: float = 8.0 / 3.0,
                            multiple_of: int = 256) -> int:
    """Compute FFN hidden dim matching NextDiT's FeedForward.__init__."""
    hidden_dim = int(ffn_dim_multiplier * dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


def _extract_layer_slice(
    state_dict: dict[str, torch.Tensor],
    src_prefix: str,
    full_dim: int,
    full_n_heads: int,
    full_n_kv_heads: int,
    skinny_dim: int,
    skinny_n_heads: int,
    skinny_n_kv_heads: int,
    full_ffn_hidden: int,
    skinny_ffn_hidden: int,
) -> dict[str, torch.Tensor]:
    """Slice a JointTransformerBlock's weights from full to skinny dimensions.

    For FP8 weights (dtype=float8_e4m3fn), slices both the weight and its
    blockwise scale tensor. For BF16/FP32 weights (norms, biases), slices
    directly.

    Returns a dict of sliced tensors with the src_prefix stripped.
    """
    head_dim = full_dim // full_n_heads
    result = {}

    for key, tensor in state_dict.items():
        if not key.startswith(src_prefix):
            continue
        local_key = key[len(src_prefix):]

        # Skip .comfy_quant metadata
        if ".comfy_quant" in local_key:
            continue

        # QKV linear: input dim sliced, output dim sliced per head count
        if local_key == "attention.qkv.weight":
            # Full: ((n_h + n_kv + n_kv) * head_dim, dim) = (11520, 3840)
            # Skinny: ((8 + 8 + 8) * 128, 1024) = (3072, 1024)
            full_q = full_n_heads * head_dim
            full_k = full_n_kv_heads * head_dim
            full_v = full_n_kv_heads * head_dim
            skinny_q = skinny_n_heads * head_dim
            skinny_k = skinny_n_kv_heads * head_dim
            skinny_v = skinny_n_kv_heads * head_dim

            # Slice output dim (rows): take first skinny_q from Q, first skinny_k from K, etc
            q_part = tensor[:skinny_q, :skinny_dim]
            k_part = tensor[full_q:full_q + skinny_k, :skinny_dim]
            v_part = tensor[full_q + full_k:full_q + full_k + skinny_v, :skinny_dim]
            result[local_key] = torch.cat([q_part, k_part, v_part], dim=0).contiguous()
            continue

        if local_key == "attention.qkv.weight_scale":
            # Scale shape: (out_blocks, in_blocks) where block_size=128
            bs = 128
            full_q = full_n_heads * head_dim
            full_k = full_n_kv_heads * head_dim
            skinny_q_blocks = (skinny_n_heads * head_dim) // bs
            skinny_k_blocks = (skinny_n_kv_heads * head_dim) // bs
            skinny_v_blocks = skinny_k_blocks
            skinny_in_blocks = skinny_dim // bs

            full_q_blocks = full_q // bs
            full_k_blocks = full_n_kv_heads * head_dim // bs

            q_s = tensor[:skinny_q_blocks, :skinny_in_blocks]
            k_s = tensor[full_q_blocks:full_q_blocks + skinny_k_blocks, :skinny_in_blocks]
            v_s = tensor[full_q_blocks + full_k_blocks:full_q_blocks + full_k_blocks + skinny_v_blocks, :skinny_in_blocks]
            result[local_key] = torch.cat([q_s, k_s, v_s], dim=0).contiguous()
            continue

        # Attention output linear: (dim, n_heads * head_dim)
        if local_key == "attention.out.weight":
            result[local_key] = tensor[:skinny_dim, :skinny_n_heads * head_dim].contiguous()
            continue
        if local_key == "attention.out.weight_scale":
            bs = 128
            result[local_key] = tensor[:skinny_dim // bs, :(skinny_n_heads * head_dim) // bs].contiguous()
            continue

        # QK norm weights: (head_dim,) -- no slicing needed (head_dim unchanged)
        if local_key in ("attention.q_norm.weight", "attention.k_norm.weight"):
            result[local_key] = tensor.clone()
            continue

        # RMSNorm weights: (dim,) -> slice to skinny_dim
        if local_key in ("attention_norm1.weight", "attention_norm2.weight",
                         "ffn_norm1.weight", "ffn_norm2.weight"):
            result[local_key] = tensor[:skinny_dim].contiguous()
            continue

        # FFN w1: (ffn_hidden, dim)
        if local_key == "feed_forward.w1.weight":
            result[local_key] = tensor[:skinny_ffn_hidden, :skinny_dim].contiguous()
            continue
        if local_key == "feed_forward.w1.weight_scale":
            bs = 128
            result[local_key] = tensor[:skinny_ffn_hidden // bs, :skinny_dim // bs].contiguous()
            continue

        # FFN w3: same shape as w1
        if local_key == "feed_forward.w3.weight":
            result[local_key] = tensor[:skinny_ffn_hidden, :skinny_dim].contiguous()
            continue
        if local_key == "feed_forward.w3.weight_scale":
            bs = 128
            result[local_key] = tensor[:skinny_ffn_hidden // bs, :skinny_dim // bs].contiguous()
            continue

        # FFN w2: (dim, ffn_hidden)
        if local_key == "feed_forward.w2.weight":
            result[local_key] = tensor[:skinny_dim, :skinny_ffn_hidden].contiguous()
            continue
        if local_key == "feed_forward.w2.weight_scale":
            bs = 128
            result[local_key] = tensor[:skinny_dim // bs, :skinny_ffn_hidden // bs].contiguous()
            continue

        # adaLN_modulation: nn.Sequential(nn.Linear(256, 4*dim))
        # Input dim=256 (min(dim,256) for z_image_modulation=True)
        if local_key == "adaLN_modulation.0.weight":
            # Output: 4*dim, Input: min(dim, 256) = 256
            # For skinny: 4*1024=4096, input still 256
            result[local_key] = tensor[:4 * skinny_dim, :].contiguous()
            continue
        if local_key == "adaLN_modulation.0.weight_scale":
            bs = 128
            result[local_key] = tensor[:(4 * skinny_dim) // bs, :].contiguous()
            continue
        if local_key == "adaLN_modulation.0.bias":
            result[local_key] = tensor[:4 * skinny_dim].contiguous()
            continue

    return result


def load_sss_model(
    device: torch.device | str = "cuda",
    fp8_path: str = FP8_MODEL_PATH,
) -> NextDiT:
    """Load a Stubbed-Skinny-Shared Z-Image model for fast GPU testing.

    Extracts real FP8 weights from layer 0 of the full model, slices to
    1/4 head count, and shares that single layer across 4 virtual layers.

    Args:
        device: Target device.
        fp8_path: Path to z_image_fp8_blockwise.safetensors (Windows path).

    Returns:
        NextDiT model with S-S-S config, ready for forward()/forward_packed().
    """
    from safetensors.torch import load_file
    from futudiffu.fp8 import replace_linear_with_fp8, FP8Linear
    from futudiffu.diffusion_model import (
        _strip_diffusion_prefix,
        create_diffusion_model,
    )

    device = torch.device(device) if isinstance(device, str) else device

    # Load full FP8 state dict
    full_sd = load_file(fp8_path, device="cpu")
    full_sd = _strip_diffusion_prefix(full_sd)

    # Full model dimensions
    full_dim = 3840
    full_n_heads = 30
    full_n_kv_heads = 30
    full_ffn_hidden = _compute_ffn_hidden_dim(full_dim)  # 10240

    # Skinny dimensions
    skinny_ffn_hidden = _compute_ffn_hidden_dim(SSS_DIM)  # 2816 (or similar)

    # Create the skinny model skeleton on meta device
    model = NextDiT(
        patch_size=SSS_PATCH_SIZE,
        in_channels=SSS_IN_CHANNELS,
        dim=SSS_DIM,
        n_layers=SSS_N_LAYERS,
        n_refiner_layers=SSS_N_REFINER,
        n_heads=SSS_N_HEADS,
        n_kv_heads=SSS_N_KV_HEADS,
        multiple_of=256,
        ffn_dim_multiplier=8.0 / 3.0,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=SSS_CAP_FEAT_DIM,
        axes_dims=SSS_AXES_DIMS,
        axes_lens=SSS_AXES_LENS,
        rope_theta=SSS_ROPE_THETA,
        z_image_modulation=True,
        time_scale=1000.0,
        pad_tokens_multiple=SSS_PAD_TOKENS_MULTIPLE,
        device="meta",
        dtype=torch.bfloat16,
    )

    # --- Extract and slice real weights from layer 0 ---
    layer0_slice = _extract_layer_slice(
        full_sd, "layers.0.",
        full_dim, full_n_heads, full_n_kv_heads,
        SSS_DIM, SSS_N_HEADS, SSS_N_KV_HEADS,
        full_ffn_hidden, skinny_ffn_hidden,
    )

    # Build state dict for the skinny model.
    # All 4 main layers share the same sliced weights from layer 0.
    skinny_sd = {}
    for i in range(SSS_N_LAYERS):
        for k, v in layer0_slice.items():
            skinny_sd[f"layers.{i}.{k}"] = v

    # Extract noise_refiner weights from noise_refiner.0
    nr_slice = _extract_layer_slice(
        full_sd, "noise_refiner.0.",
        full_dim, full_n_heads, full_n_kv_heads,
        SSS_DIM, SSS_N_HEADS, SSS_N_KV_HEADS,
        full_ffn_hidden, skinny_ffn_hidden,
    )
    for i in range(SSS_N_REFINER):
        for k, v in nr_slice.items():
            skinny_sd[f"noise_refiner.{i}.{k}"] = v

    # Extract context_refiner weights from context_refiner.0
    cr_slice = _extract_layer_slice(
        full_sd, "context_refiner.0.",
        full_dim, full_n_heads, full_n_kv_heads,
        SSS_DIM, SSS_N_HEADS, SSS_N_KV_HEADS,
        full_ffn_hidden, skinny_ffn_hidden,
    )
    for i in range(SSS_N_REFINER):
        for k, v in cr_slice.items():
            skinny_sd[f"context_refiner.{i}.{k}"] = v

    # x_embedder: (dim, patch_size^2 * in_channels) = (3840, 64) -> (1024, 64)
    skinny_sd["x_embedder.weight"] = full_sd["x_embedder.weight"][:SSS_DIM, :].contiguous()
    skinny_sd["x_embedder.bias"] = full_sd["x_embedder.bias"][:SSS_DIM].contiguous()

    # cap_embedder: RMSNorm(2560) + Linear(2560, dim)
    # RMSNorm weight: (2560,) -- no change
    skinny_sd["cap_embedder.0.weight"] = full_sd["cap_embedder.0.weight"].clone()
    # Linear: (dim, 2560) -> (1024, 2560)
    if "cap_embedder.1.weight" in full_sd:
        w = full_sd["cap_embedder.1.weight"]
        if w.dtype == torch.float8_e4m3fn:
            skinny_sd["cap_embedder.1.weight"] = w[:SSS_DIM, :].contiguous()
            scale_key = "cap_embedder.1.weight_scale"
            if scale_key in full_sd:
                bs = 128
                skinny_sd[scale_key] = full_sd[scale_key][:SSS_DIM // bs, :].contiguous()
        else:
            skinny_sd["cap_embedder.1.weight"] = w[:SSS_DIM, :].contiguous()
    if "cap_embedder.1.bias" in full_sd:
        skinny_sd["cap_embedder.1.bias"] = full_sd["cap_embedder.1.bias"][:SSS_DIM].contiguous()

    # t_embedder: TimestepEmbedder(min(dim,1024), output_size=256)
    # Full: hidden=min(3840,1024)=1024, output=256
    # Skinny: hidden=min(1024,1024)=1024, output=256
    # mlp.0: Linear(256, 1024), mlp.2: Linear(1024, 256) -- same shapes
    # Copy weights and scales directly (no slicing needed)
    for k in ("t_embedder.mlp.0.weight", "t_embedder.mlp.0.weight_scale",
              "t_embedder.mlp.0.bias",
              "t_embedder.mlp.2.weight", "t_embedder.mlp.2.weight_scale",
              "t_embedder.mlp.2.bias"):
        if k in full_sd:
            skinny_sd[k] = full_sd[k].clone()

    # final_layer: LayerNorm + Linear(dim, p*p*C) + adaLN(SiLU + Linear(256, dim))
    if "final_layer.linear.weight" in full_sd:
        w = full_sd["final_layer.linear.weight"]
        if w.dtype == torch.float8_e4m3fn:
            # (p*p*C, dim) -> (64, 1024) -- output dim unchanged, input sliced
            out_dim = SSS_PATCH_SIZE * SSS_PATCH_SIZE * SSS_IN_CHANNELS  # 64
            skinny_sd["final_layer.linear.weight"] = w[:out_dim, :SSS_DIM].contiguous()
            sk = "final_layer.linear.weight_scale"
            if sk in full_sd:
                bs = 128
                skinny_sd[sk] = full_sd[sk][:max(1, out_dim // bs), :SSS_DIM // bs].contiguous()
        else:
            out_dim = SSS_PATCH_SIZE * SSS_PATCH_SIZE * SSS_IN_CHANNELS
            skinny_sd["final_layer.linear.weight"] = w[:out_dim, :SSS_DIM].contiguous()
    if "final_layer.linear.bias" in full_sd:
        out_dim = SSS_PATCH_SIZE * SSS_PATCH_SIZE * SSS_IN_CHANNELS
        skinny_sd["final_layer.linear.bias"] = full_sd["final_layer.linear.bias"][:out_dim].contiguous()

    # final_layer.adaLN_modulation: SiLU + Linear(256, dim)
    if "final_layer.adaLN_modulation.1.weight" in full_sd:
        w = full_sd["final_layer.adaLN_modulation.1.weight"]
        if w.dtype == torch.float8_e4m3fn:
            skinny_sd["final_layer.adaLN_modulation.1.weight"] = w[:SSS_DIM, :].contiguous()
            sk = "final_layer.adaLN_modulation.1.weight_scale"
            if sk in full_sd:
                bs = 128
                skinny_sd[sk] = full_sd[sk][:SSS_DIM // bs, :].contiguous()
        else:
            skinny_sd["final_layer.adaLN_modulation.1.weight"] = w[:SSS_DIM, :].contiguous()
    if "final_layer.adaLN_modulation.1.bias" in full_sd:
        skinny_sd["final_layer.adaLN_modulation.1.bias"] = full_sd["final_layer.adaLN_modulation.1.bias"][:SSS_DIM].contiguous()

    # Pad tokens: (1, dim)
    if "x_pad_token" in full_sd:
        skinny_sd["x_pad_token"] = full_sd["x_pad_token"][:, :SSS_DIM].contiguous()
    if "cap_pad_token" in full_sd:
        skinny_sd["cap_pad_token"] = full_sd["cap_pad_token"][:, :SSS_DIM].contiguous()

    # Free full state dict
    del full_sd

    # Replace nn.Linear with FP8Linear where FP8 weights exist
    replace_linear_with_fp8(
        model, skinny_sd, block_size=128, output_dtype=torch.bfloat16,
    )

    # Load remaining (non-FP8) weights
    remaining = {k: v for k, v in skinny_sd.items()
                 if not k.endswith((".weight_scale", ".comfy_quant"))}
    model.load_state_dict(remaining, strict=False, assign=True)
    del skinny_sd, remaining

    model = model.to(device)

    # Cast non-FP8 params to BF16 without clobbering FP8 weights/scales.
    # model.to(torch.bfloat16) would destroy FP8Linear's float8_e4m3fn weights
    # and float32 scales (known assign=True footgun, see MEMORY.md).
    from futudiffu.fp8 import FP8Linear
    fp8_params = set()
    for name, mod in model.named_modules():
        if isinstance(mod, FP8Linear):
            fp8_params.add(id(mod.weight))
            if hasattr(mod, "weight_scale"):
                fp8_params.add(id(mod.weight_scale))
    for p in model.parameters():
        if id(p) not in fp8_params and p.dtype != torch.bfloat16:
            p.data = p.data.to(torch.bfloat16)
    for b in model.buffers():
        if id(b) not in fp8_params and b.dtype not in (torch.bfloat16, torch.long, torch.int):
            b.data = b.data.to(torch.bfloat16)

    model.eval()

    return model


def make_random_conditioning(
    batch_size: int = 1,
    seq_len: int = 20,
    cap_feat_dim: int = SSS_CAP_FEAT_DIM,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Create random text conditioning matching the real shape.

    Args:
        batch_size: Batch size.
        seq_len: Number of caption tokens.
        cap_feat_dim: Caption feature dimension (2560 for Z-Image).
        device: Target device.
        dtype: Data type.

    Returns:
        (batch_size, seq_len, cap_feat_dim) conditioning tensor.
    """
    return torch.randn(batch_size, seq_len, cap_feat_dim, device=device, dtype=dtype)


if __name__ == "__main__":
    print("Loading S-S-S model...")
    model = load_sss_model(device="cuda")

    # Print model summary
    n_params = sum(p.numel() for p in model.parameters())
    n_buffers = sum(b.numel() for b in model.buffers())
    total_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) + sum(
        b.numel() * b.element_size() for b in model.buffers()
    )
    print(f"Parameters: {n_params:,}")
    print(f"Buffers: {n_buffers:,}")
    print(f"Total size: {total_bytes / 1e6:.1f} MB")
    print(f"dim={model.dim}, n_heads={model.n_heads}")
    print(f"Main layers: {len(model.layers)}")
    print(f"Refiner layers: {len(model.noise_refiner)}")

    # Quick forward test
    print("\nRunning forward pass...")
    B, C, H, W = 1, 16, 64, 64
    x = torch.randn(B, C, H, W, device="cuda", dtype=torch.bfloat16)
    t = torch.tensor([0.5], device="cuda", dtype=torch.bfloat16)
    ctx = make_random_conditioning(B, 20, device="cuda")

    with torch.inference_mode():
        out = model(x, t, ctx, num_tokens=20)
    print(f"Output shape: {out.shape}")
    print(f"Output stats: mean={out.mean():.4f}, std={out.std():.4f}")
    print("S-S-S model loaded and forward pass works!")

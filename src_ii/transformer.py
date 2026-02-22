"""Branchless transformer components for ZImageRLAIF.

Reimplementation of architecture components from futudiffu.diffusion_model.
All flag-guarded branching removed: one code path per class, always fused
kernels, always block-masked attention.

Attention backend (sage vs sdpa/flex) is a module-level global set once
before torch.compile. torch.compile specializes on its value at trace time;
the dead branch is eliminated. No recompiles as long as the backend doesn't
change between calls.

Kernel imports from frozen modules (Triton kernels registered as custom ops,
correct, no bugs):
  - futudiffu.fused_kernels: fused_rms_norm_modulate, fused_rms_norm_gate_residual,
    fused_qkv_postprocess
  - futudiffu.sage_attention: sage_attn_masked_op
  - futudiffu.fp8: FP8Linear, dequantize_fp8_blockwise
  - futudiffu.fp8_kernels: fp8_silu_gate_quant_op, fp8_gemm_blockwise_op,
    fp8_gemm_v1t_op, _DTYPE_TO_CODE_K

The bugs were in the Python dispatch layer above the kernels (sdpa_attention
branching, fused/unfused flag checks, import-time capture of local
sdpa_attention copy). This module eliminates that layer.
"""

from __future__ import annotations

import torch._dynamo
torch._dynamo.config.suppress_errors = False

import torch._inductor.config
torch._inductor.config.triton.persistent_reductions = False  # torch 2.10.0 sympy Infinity bug

import ctypes
import itertools
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

# --- Kernel imports (all custom ops, all correct) ---

from futudiffu.fused_kernels import (
    fused_rms_norm_modulate,
    fused_rms_norm_gate_residual,
    fused_qkv_postprocess,
)
from futudiffu.sage_attention import sage_attn_masked_op
from futudiffu.fp8 import FP8Linear, dequantize_fp8_blockwise
from futudiffu.fp8_kernels import (
    fp8_silu_gate_quant_op,
    fp8_gemm_blockwise_op,
    fp8_gemm_v1t_op,
    _DTYPE_TO_CODE_K,
)
from torch.nn.attention.flex_attention import flex_attention


# --- Attention backend ---
# Set once before torch.compile. torch.compile specializes on this value
# at trace time — the dead branch is eliminated, no graph breaks, no
# recompiles as long as this doesn't change between forward calls.

_USE_SAGE: bool = True


def set_attention_backend(backend: str) -> None:
    """Set the attention backend for JointAttention.

    Args:
        backend: "sage" (SageAttention INT8 QK, default) or "sdpa" (FlexAttention).
            Must be called before torch.compile. Changing after compilation
            triggers a recompile.
    """
    global _USE_SAGE
    if backend not in ("sage", "sdpa"):
        raise ValueError(f"Unknown attention backend: {backend!r}")
    _USE_SAGE = (backend == "sage")


# =====================================================================
# Category 1: Small utilities (vendored verbatim)
# =====================================================================

@dataclass
class PackingInfo:
    """Describes how N images are packed into a single token sequence."""
    n_images: int
    segments: list[tuple[int, int, int, int]]
    total_len: int
    document_id: torch.Tensor
    img_grid_sizes: list[tuple[int, int]]
    cap_lens: list[int]


def pad_to_patch_size(img: torch.Tensor, patch_size: tuple[int, int] = (2, 2)) -> torch.Tensor:
    """Pad image to be divisible by patch_size."""
    pad = ()
    for i in range(img.ndim - 2):
        pad = (0, (patch_size[i] - img.shape[i + 2] % patch_size[i]) % patch_size[i]) + pad
    return F.pad(img, pad, mode="circular")


def pad_zimage(feats: torch.Tensor, pad_token: torch.Tensor, pad_tokens_multiple: int) -> tuple[torch.Tensor, int]:
    pad_extra = (-feats.shape[1]) % pad_tokens_multiple
    return torch.cat((
        feats,
        pad_token.to(device=feats.device, dtype=feats.dtype, copy=True).unsqueeze(0).repeat(feats.shape[0], pad_extra, 1),
    ), dim=1), pad_extra


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def modulate(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation."""
    return x * (1 + scale.unsqueeze(1))


def build_packed_sequence(
    cap_feats_list: list[torch.Tensor],
    img_patches_list: list[torch.Tensor],
    img_grid_sizes: list[tuple[int, int]],
    cap_lens: list[int],
    pad_tokens_multiple: int,
    cap_pad_token: torch.Tensor,
    x_pad_token: torch.Tensor,
) -> tuple[torch.Tensor, PackingInfo]:
    """Pack N (caption, image) pairs into a single token sequence."""
    n_images = len(cap_feats_list)

    segments: list[tuple[int, int, int, int]] = []
    all_tokens: list[torch.Tensor] = []
    doc_ids: list[int] = []
    offset = 0

    for i in range(n_images):
        cap_padded, _ = pad_zimage(cap_feats_list[i], cap_pad_token, pad_tokens_multiple)
        cap_padded_len = cap_padded.shape[1]

        img_padded, _ = pad_zimage(img_patches_list[i], x_pad_token, pad_tokens_multiple)
        img_padded_len = img_padded.shape[1]

        text_start = offset
        text_len = cap_padded_len
        img_start = offset + cap_padded_len
        img_len = img_padded_len

        segments.append((text_start, text_len, img_start, img_len))

        all_tokens.append(cap_padded)
        all_tokens.append(img_padded)

        doc_ids.extend([i] * (cap_padded_len + img_padded_len))

        offset += cap_padded_len + img_padded_len

    total_len = offset
    packed = torch.cat(all_tokens, dim=1)
    document_id = torch.tensor(doc_ids, dtype=torch.int32, device=cap_feats_list[0].device)

    packing_info = PackingInfo(
        n_images=n_images,
        segments=segments,
        total_len=total_len,
        document_id=document_id,
        img_grid_sizes=img_grid_sizes,
        cap_lens=cap_lens,
    )
    return packed, packing_info


def build_packed_rope(
    packing_info: PackingInfo,
    rope_embedder: "EmbedND",
    pad_tokens_multiple: int,
    device: torch.device,
) -> torch.Tensor:
    """Build RoPE frequencies for a packed multi-image sequence."""
    pos_ids = torch.zeros(1, packing_info.total_len, 3, dtype=torch.float32, device=device)

    for i in range(packing_info.n_images):
        text_start, text_len, img_start, img_len = packing_info.segments[i]
        H_t, W_t = packing_info.img_grid_sizes[i]

        cap_padded_len = text_len

        pos_ids[0, text_start:text_start + text_len, 0] = (
            torch.arange(cap_padded_len, dtype=torch.float32, device=device) + 1.0
        )

        n_real_img = H_t * W_t
        pos_ids[0, img_start:img_start + n_real_img, 0] = cap_padded_len + 1
        pos_ids[0, img_start:img_start + n_real_img, 1] = (
            torch.arange(H_t, dtype=torch.float32, device=device)
            .view(-1, 1).repeat(1, W_t).flatten()
        )
        pos_ids[0, img_start:img_start + n_real_img, 2] = (
            torch.arange(W_t, dtype=torch.float32, device=device)
            .view(1, -1).repeat(H_t, 1).flatten()
        )

    freqs_cis = rope_embedder(pos_ids).movedim(1, 2)
    return freqs_cis


def unpack_and_unpatchify(
    packed_output: torch.Tensor,
    packing_info: PackingInfo,
    patch_size: int,
    out_channels: int,
) -> list[torch.Tensor]:
    """Extract per-image tokens from packed output and reshape to spatial tensors."""
    pH = pW = patch_size
    results = []

    for i in range(packing_info.n_images):
        _text_start, _text_len, img_start, _img_len = packing_info.segments[i]
        H_t, W_t = packing_info.img_grid_sizes[i]
        n_real_img = H_t * W_t

        img_tokens = packed_output[:, img_start:img_start + n_real_img, :]

        img = (
            img_tokens
            .view(-1, H_t, W_t, pH, pW, out_channels)
            .permute(0, 5, 1, 3, 2, 4)
            .flatten(4, 5)
            .flatten(2, 3)
        )
        results.append(img)

    return results


def _detect_n_layers(state_dict_keys) -> int:
    """Detect n_layers from state dict keys."""
    n_layers = 0
    for k in state_dict_keys:
        if k.startswith("layers."):
            layer_idx = int(k.split(".")[1])
            n_layers = max(n_layers, layer_idx + 1)
    return n_layers


def _detect_cap_feat_dim(state_dict: dict[str, torch.Tensor]) -> int:
    """Detect cap_feat_dim from cap_embedder weights. Defaults to 2560."""
    key = "cap_embedder.1.weight"
    if key in state_dict:
        return state_dict[key].shape[1]
    return 2560


def _detect_qk_norm(state_dict_keys) -> bool:
    """Detect whether the model uses qk_norm from state dict keys."""
    return any(k.endswith(".q_norm.weight") for k in state_dict_keys)


def _strip_diffusion_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip 'model.diffusion_model.' or 'diffusion_model.' prefix from keys."""
    remapped = {}
    for k, v in state_dict.items():
        new_key = k
        if new_key.startswith("model.diffusion_model."):
            new_key = new_key[len("model.diffusion_model."):]
        elif new_key.startswith("diffusion_model."):
            new_key = new_key[len("diffusion_model."):]
        remapped[new_key] = v
    return remapped


# =====================================================================
# Category 2: Simple nn.Module classes + attention utilities
# =====================================================================

def rms_norm(x: Tensor, weight: Tensor | None = None, eps: float = 1e-6) -> Tensor:
    """RMSNorm. No try/except. Crash if torch too old."""
    if weight is None:
        return F.rms_norm(x, (x.shape[-1],), eps=eps)
    return F.rms_norm(x, weight.shape, weight=weight.to(dtype=x.dtype, device=x.device), eps=eps)


def rope_embed(pos: Tensor, dim: int, theta: float) -> Tensor:
    """Generate RoPE rotation matrices (Flux/Lumina style)."""
    assert dim % 2 == 0
    device = pos.device
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=device)


def _apply_rope1_flux(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """Apply RoPE to a single tensor (Flux/Lumina style)."""
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    x_out = freqs_cis[..., 0] * x_[..., 0]
    x_out.addcmul_(freqs_cis[..., 1], x_[..., 1])
    return x_out.reshape(*x.shape).type_as(x)


def apply_rope_flux(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    """Apply RoPE (Flux/Lumina style 2x2 rotation matrices)."""
    return _apply_rope1_flux(xq, freqs_cis), _apply_rope1_flux(xk, freqs_cis)


class RMSNormModule(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


class LayerNormModule(nn.Module):
    """LayerNorm without elementwise affine, for FinalLayer."""
    def __init__(self, dim: int, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self.dim,), eps=self.eps)


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: float, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope_embed(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256,
                 output_size: int | None = None, device=None, dtype=None):
        super().__init__()
        if output_size is None:
            output_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden_size, output_size, bias=True, device=device, dtype=dtype),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        t_freq = timestep_embedding(t, self.frequency_embedding_size).to(dtype)
        return self.mlp(t_freq)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int,
                 z_image_modulation: bool = False, device=None, dtype=None):
        super().__init__()
        self.norm_final = LayerNormModule(hidden_size, eps=1e-6, device=device, dtype=dtype)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels,
                                bias=True, device=device, dtype=dtype)
        min_mod = 256 if z_image_modulation else 1024
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, min_mod), hidden_size, bias=True, device=device, dtype=dtype),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale = self.adaLN_modulation(c)
        x = modulate(self.norm_final(x), scale)
        x = self.linear(x)
        return x


# =====================================================================
# Category 3: Branchless core
# =====================================================================

def _crash() -> None:
    """Kill the interpreter. Not raise, not sys.exit — SIGSEGV."""
    ctypes.string_at(0)


def build_trivial_mask(seq_len: int, device: torch.device):
    """Build an all-ones block mask for a single-image sequence.

    Returns the right type for the active backend:
    - Sage: uint8 tensor (n_q_blocks, n_kv_blocks), all ones
    - SDPA: FlexAttention BlockMask with identity mask_mod

    Called once per refiner invocation. The mask is small (a few tiles)
    and cheap to construct.
    """
    from src_ii.block_mask import build_block_mask, BLOCK_M, BLOCK_N, _ceildiv
    if _USE_SAGE:
        return build_block_mask([seq_len], total_len=seq_len, device=device)
    else:
        from torch.nn.attention.flex_attention import create_block_mask
        def _all_true(b, h, q_idx, kv_idx):
            return True
        n_heads = 1  # broadcast across heads
        return create_block_mask(_all_true, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)


class JointAttention(nn.Module):
    """Branchless attention. Always fused QKV. Block-masked.

    block_mask is REQUIRED. Crash if None.
    - Sage backend (_USE_SAGE=True): block_mask is uint8 Tensor.
    - SDPA backend (_USE_SAGE=False): block_mask is FlexAttention BlockMask.
    Backend is a module-level global set once before torch.compile.
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, qk_norm: bool,
                 out_bias: bool = False, device=None, dtype=None):
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_local_heads = n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, (n_heads + self.n_kv_heads + self.n_kv_heads) * self.head_dim,
                              bias=False, device=device, dtype=dtype)
        self.out = nn.Linear(n_heads * self.head_dim, dim, bias=out_bias, device=device, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNormModule(self.head_dim, elementwise_affine=True, device=device, dtype=dtype)
            self.k_norm = RMSNormModule(self.head_dim, elementwise_affine=True, device=device, dtype=dtype)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None,
                freqs_cis: torch.Tensor, block_mask=None) -> torch.Tensor:
        if block_mask is None:
            _crash()

        b, seqlen, _ = x.shape

        qkv_out = self.qkv(x)

        xq, xk, xv = fused_qkv_postprocess(
            qkv_out, self.q_norm.weight, self.k_norm.weight,
            freqs_cis, self.n_local_heads, self.head_dim,
            self.q_norm.eps,
        )

        if _USE_SAGE:
            sm_scale = 1.0 / (self.head_dim ** 0.5)
            out, _lse = sage_attn_masked_op(xq, xk, xv, block_mask, sm_scale)
        else:
            out = flex_attention(xq, xk, xv, block_mask=block_mask)

        return self.out(out.transpose(1, 2).reshape(b, -1, self.n_local_heads * self.head_dim))


class FeedForward(nn.Module):
    """FeedForward with pre-fusion and post-fusion paths.

    Pre-fusion (before fuse_model): w2(silu(w1(x)) * w3(x))
    Post-fusion: fused w1w3 GEMM -> fp8_silu_gate_quant_op -> fp8_gemm_*_op
    No eager Triton fallback. Always custom_op path after fusion.
    """

    def __init__(self, dim: int, hidden_dim: int, multiple_of: int,
                 ffn_dim_multiplier: float, device=None, dtype=None):
        super().__init__()
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False, device=device, dtype=dtype)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False, device=device, dtype=dtype)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False, device=device, dtype=dtype)
        self.hidden_dim = hidden_dim
        self._fused = False

    def fuse_w1w3(self):
        """Horizontally fuse w1 and w3 into a single GEMM."""
        if self._fused:
            return

        w1, w3 = self.w1, self.w3
        if isinstance(w1, FP8Linear) and isinstance(w3, FP8Linear):
            fused_weight = torch.cat([w1.weight, w3.weight], dim=0)
            fused_scale = torch.cat([w1.weight_scale, w3.weight_scale], dim=0)
            self.w1w3 = FP8Linear(
                fused_weight, fused_scale, bias=None,
                block_size=w1.block_size, output_dtype=w1.output_dtype,
            )
        elif isinstance(w1, nn.Linear) and isinstance(w3, nn.Linear):
            fused = nn.Linear(
                w1.in_features, w1.out_features + w3.out_features,
                bias=False, device=w1.weight.device, dtype=w1.weight.dtype,
            )
            fused.weight = nn.Parameter(torch.cat([w1.weight, w3.weight], dim=0))
            self.w1w3 = fused
        else:
            return

        del self.w1
        del self.w3
        self._fused = True

    def enable_fused_chain(self):
        """Enable FP8 chain fusion: SiLU+gate -> FP8 requant -> w2 GEMM."""
        if not self._fused:
            return
        if isinstance(self.w2, FP8Linear):
            self._fused_chain = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fused:
            if getattr(self, '_fused_chain', False):
                # Full FP8 chain: w1w3 -> SiLU+gate+requant -> w2 (custom ops only)
                w1w3_out = self.w1w3(x)
                gated_fp8, gated_scale = fp8_silu_gate_quant_op(
                    w1w3_out.contiguous(), self.w2.block_size,
                )
                dtype_code = _DTYPE_TO_CODE_K[self.w2.output_dtype]
                if getattr(self.w2, '_transposed', False):
                    return fp8_gemm_v1t_op(
                        gated_fp8, gated_scale,
                        self.w2.weight, self.w2.weight_scale,
                        self.w2.block_size, dtype_code,
                    )
                return fp8_gemm_blockwise_op(
                    gated_fp8, gated_scale,
                    self.w2.weight, self.w2.weight_scale,
                    self.w2.block_size, dtype_code,
                )
            # Fused w1w3 but no chain (e.g. BF16 weights)
            w1w3_out = self.w1w3(x)
            w1_out, w3_out = w1w3_out.split(self.hidden_dim, dim=-1)
            return self.w2(F.silu(w1_out) * w3_out)
        # Pre-fusion
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class JointTransformerBlock(nn.Module):
    """Branchless transformer block.

    Two structural paths (resolved at __init__ time, torch.compile traces both):
    - Modulated (noise_refiner + main layers): Always fused Triton kernels.
    - Unmodulated (context_refiner): Simple norm + attention + residual.
    """

    def __init__(
        self, layer_id: int, dim: int, n_heads: int, n_kv_heads: int,
        multiple_of: int, ffn_dim_multiplier: float, norm_eps: float,
        qk_norm: bool, modulation: bool = True, z_image_modulation: bool = False,
        attn_out_bias: bool = False, device=None, dtype=None,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = JointAttention(dim, n_heads, n_kv_heads, qk_norm,
                                        out_bias=attn_out_bias, device=device, dtype=dtype)
        self.feed_forward = FeedForward(dim, dim, multiple_of, ffn_dim_multiplier,
                                        device=device, dtype=dtype)
        self.layer_id = layer_id
        self.attention_norm1 = RMSNormModule(dim, eps=norm_eps, elementwise_affine=True, device=device, dtype=dtype)
        self.ffn_norm1 = RMSNormModule(dim, eps=norm_eps, elementwise_affine=True, device=device, dtype=dtype)
        self.attention_norm2 = RMSNormModule(dim, eps=norm_eps, elementwise_affine=True, device=device, dtype=dtype)
        self.ffn_norm2 = RMSNormModule(dim, eps=norm_eps, elementwise_affine=True, device=device, dtype=dtype)

        self.modulation = modulation
        if modulation:
            if z_image_modulation:
                self.adaLN_modulation = nn.Sequential(
                    nn.Linear(min(dim, 256), 4 * dim, bias=True, device=device, dtype=dtype),
                )
            else:
                self.adaLN_modulation = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(min(dim, 1024), 4 * dim, bias=True, device=device, dtype=dtype),
                )

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor | None,
        freqs_cis: torch.Tensor, adaln_input: torch.Tensor | None = None,
        precomputed_adaln: tuple[torch.Tensor, ...] | None = None,
        block_mask=None,
    ) -> torch.Tensor:
        if self.modulation:
            if precomputed_adaln is not None:
                scale_msa, gate_msa, scale_mlp, gate_mlp = precomputed_adaln
            else:
                assert adaln_input is not None
                scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)

            # chunk() returns non-contiguous views; fused kernels need contiguous
            scale_msa = scale_msa.contiguous()
            gate_msa = gate_msa.contiguous()
            scale_mlp = scale_mlp.contiguous()
            gate_mlp = gate_mlp.contiguous()

            # Per-token modulation (3D: B, total_len, dim) vs per-batch (2D: B, dim).
            # The fused Triton kernels (fused_rms_norm_modulate, fused_rms_norm_gate_residual)
            # expect (B, dim) scale/gate and broadcast across the sequence dimension via
            # unsqueeze(1). For per-token adaLN (packed multi-image with different sigmas),
            # scale/gate are (B, total_len, dim) and we use unfused F.rms_norm + manual
            # modulation. This branch is resolved at torch.compile trace time — the dead
            # path is eliminated, no graph break.
            if scale_msa.dim() == 3:
                # --- Per-token path (unfused) ---
                # Matches fused kernel semantics:
                #   fused_rms_norm_modulate: rms_norm(x) * (1 + scale)
                #   fused_rms_norm_gate_residual: rms_norm(y) * tanh(gate) + residual
                # Weight is cast to input dtype to match the rms_norm() helper.
                d = (x.shape[-1],)
                w1 = self.attention_norm1.weight.to(dtype=x.dtype)
                attn_in = F.rms_norm(x, d, w1, self.attention_norm1.eps) * (1 + scale_msa)
                attn_out = self.attention(attn_in, x_mask, freqs_cis,
                                          block_mask=block_mask)
                w2 = self.attention_norm2.weight.to(dtype=x.dtype)
                x = F.rms_norm(attn_out, d, w2, self.attention_norm2.eps) * gate_msa.tanh() + x
                w3 = self.ffn_norm1.weight.to(dtype=x.dtype)
                ffn_in = F.rms_norm(x, d, w3, self.ffn_norm1.eps) * (1 + scale_mlp)
                ffn_out = self.feed_forward(ffn_in)
                w4 = self.ffn_norm2.weight.to(dtype=x.dtype)
                x = F.rms_norm(ffn_out, d, w4, self.ffn_norm2.eps) * gate_mlp.tanh() + x
            else:
                # --- Per-batch path (fused Triton kernels) ---
                attn_in = fused_rms_norm_modulate(
                    x, self.attention_norm1.weight, scale_msa,
                    self.attention_norm1.eps,
                )
                attn_out = self.attention(attn_in, x_mask, freqs_cis,
                                          block_mask=block_mask)
                x = fused_rms_norm_gate_residual(
                    attn_out, self.attention_norm2.weight, gate_msa, x,
                    self.attention_norm2.eps,
                )
                ffn_in = fused_rms_norm_modulate(
                    x, self.ffn_norm1.weight, scale_mlp,
                    self.ffn_norm1.eps,
                )
                ffn_out = self.feed_forward(ffn_in)
                x = fused_rms_norm_gate_residual(
                    ffn_out, self.ffn_norm2.weight, gate_mlp, x,
                    self.ffn_norm2.eps,
                )
        else:
            # Unmodulated (context_refiner): no adaLN, no fused kernels needed
            assert adaln_input is None
            x = x + self.attention_norm2(
                self.attention(self.attention_norm1(x), x_mask, freqs_cis,
                               block_mask=block_mask)
            )
            x = x + self.ffn_norm2(
                self.feed_forward(self.ffn_norm1(x))
            )
        return x


# =====================================================================
# fuse_model
# =====================================================================

def fuse_model(model: nn.Module) -> None:
    """Walk model tree and apply all safe horizontal fusions.

    Fusions applied:
    1. FeedForward w1+w3 -> single GEMM
    2. FP8 chain: SiLU+gate -> FP8 requant -> w2 GEMM
    3. Batched adaLN precomputation

    No flag-setting (_use_fused_elementwise, _use_fused_qkv). Those flags
    don't exist in the branchless version -- the fused path is the ONLY path.
    """
    import logging
    log = logging.getLogger(__name__)

    n_ffn = 0
    n_chain = 0

    for module in model.modules():
        if isinstance(module, FeedForward):
            module.fuse_w1w3()
            if module._fused:
                n_ffn += 1
                module.enable_fused_chain()
                if getattr(module, '_fused_chain', False):
                    n_chain += 1

    # Pre-batch adaLN weights for batched GEMM
    n_adaln = 0
    if hasattr(model, 'prepare_adaln_cache'):
        model.prepare_adaln_cache()
        n_adaln = getattr(model, '_adaln_n_blocks', 0)

    if n_ffn > 0:
        log.info(f"Fused {n_ffn} FeedForward w1+w3 layers")
    if n_chain > 0:
        log.info(f"Fused {n_chain} FP8 FFN chains (SiLU+gate+requant)")
    if n_adaln > 0:
        log.info(f"Pre-batched adaLN: {n_adaln} blocks")

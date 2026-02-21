"""NextDiT (Z-Image variant) diffusion model ported from ComfyUI.

Sources:
- comfy/ldm/lumina/model.py (NextDiT, JointTransformerBlock, JointAttention, FeedForward, FinalLayer)
- comfy/ldm/flux/layers.py (EmbedND, timestep_embedding)
- comfy/ldm/modules/diffusionmodules/mmdit.py (TimestepEmbedder)
- comfy/ldm/flux/math.py (rope, apply_rope)
- comfy/ldm/common_dit.py (pad_to_patch_size)

Z-Image config (from model_detection.py:438-456):
  dim=3840, n_heads=30, n_kv_heads=30, axes_dims=[32,48,48],
  axes_lens=[1536,512,512], rope_theta=256.0, ffn_dim_multiplier=8/3,
  z_image_modulation=True, time_scale=1000.0, in_channels=16, patch_size=2,
  cap_feat_dim=2560, pad_tokens_multiple=32
"""

import itertools
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from .attention import (
    apply_rope_flux,
    rms_norm,
    rope_embed,
    sdpa_attention,
)

try:
    from .fused_kernels import fused_rms_norm_modulate, fused_rms_norm_gate_residual
    _HAS_FUSED_KERNELS = True
except ImportError:
    _HAS_FUSED_KERNELS = False

try:
    from .fused_kernels import fused_qkv_postprocess
    _HAS_FUSED_QKV = True
except ImportError:
    _HAS_FUSED_QKV = False


# --- Packing infrastructure for multi-image FlexAttention ---

@dataclass
class PackingInfo:
    """Describes how N images are packed into a single token sequence."""
    n_images: int
    # Per-image: (text_start, text_len, img_start, img_len) in the packed sequence
    segments: list[tuple[int, int, int, int]]
    total_len: int
    # document_id[token_idx] -> image_index, -1 for padding
    document_id: torch.Tensor  # (total_len,) int32
    # Per-image spatial grid dims for RoPE (H_tokens, W_tokens)
    img_grid_sizes: list[tuple[int, int]]
    # Per-image caption lengths (before padding to multiple)
    cap_lens: list[int]


# --- Utilities ---

def pad_to_patch_size(img: torch.Tensor, patch_size: tuple[int, int] = (2, 2)) -> torch.Tensor:
    """Pad image to be divisible by patch_size."""
    pad = ()
    for i in range(img.ndim - 2):
        pad = (0, (patch_size[i] - img.shape[i + 2] % patch_size[i]) % patch_size[i]) + pad
    return F.pad(img, pad, mode="circular")


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embedding (from diffusionmodules/util.py)."""
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
    """AdaLN modulation (no timestep_zero_index variant)."""
    return x * (1 + scale.unsqueeze(1))


def apply_gate(gate: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Gate application (no timestep_zero_index variant)."""
    return gate * x


def pad_zimage(feats: torch.Tensor, pad_token: torch.Tensor, pad_tokens_multiple: int) -> tuple[torch.Tensor, int]:
    pad_extra = (-feats.shape[1]) % pad_tokens_multiple
    return torch.cat((
        feats,
        pad_token.to(device=feats.device, dtype=feats.dtype, copy=True).unsqueeze(0).repeat(feats.shape[0], pad_extra, 1),
    ), dim=1), pad_extra


# --- Packing functions for multi-image FlexAttention ---

def build_packed_sequence(
    cap_feats_list: list[torch.Tensor],
    img_patches_list: list[torch.Tensor],
    img_grid_sizes: list[tuple[int, int]],
    cap_lens: list[int],
    pad_tokens_multiple: int,
    cap_pad_token: torch.Tensor,
    x_pad_token: torch.Tensor,
) -> tuple[torch.Tensor, PackingInfo]:
    """Pack N (caption, image) pairs into a single token sequence.

    Produces a packed sequence [text_0, img_0, text_1, img_1, ...] with
    document_id tracking which image each token belongs to (-1 for padding).

    Args:
        cap_feats_list: N tensors, each (B, cap_len_i, dim) -- already embedded.
        img_patches_list: N tensors, each (B, n_img_i, dim) -- already embedded.
        img_grid_sizes: (H_tokens, W_tokens) per image.
        cap_lens: Original caption lengths before padding.
        pad_tokens_multiple: Pad each segment to this multiple.
        cap_pad_token: (1, dim) pad token for captions.
        x_pad_token: (1, dim) pad token for image patches.

    Returns:
        packed: (B, total_len, dim) packed token sequence.
        packing_info: PackingInfo describing the layout.
    """
    n_images = len(cap_feats_list)
    B = cap_feats_list[0].shape[0]
    device = cap_feats_list[0].device

    segments: list[tuple[int, int, int, int]] = []
    all_tokens: list[torch.Tensor] = []
    doc_ids: list[int] = []
    offset = 0

    for i in range(n_images):
        # Pad caption
        cap_padded, cap_pad_extra = pad_zimage(cap_feats_list[i], cap_pad_token, pad_tokens_multiple)
        cap_padded_len = cap_padded.shape[1]

        # Pad image patches
        img_padded, img_pad_extra = pad_zimage(img_patches_list[i], x_pad_token, pad_tokens_multiple)
        img_padded_len = img_padded.shape[1]

        text_start = offset
        text_len = cap_padded_len
        img_start = offset + cap_padded_len
        img_len = img_padded_len

        segments.append((text_start, text_len, img_start, img_len))

        all_tokens.append(cap_padded)
        all_tokens.append(img_padded)

        # document_id: ALL tokens in this image's segment (including padding)
        # get this image's index. This matches the unpacked case where padding
        # tokens participate in attention (mask=None) and affect the softmax
        # denominator. Only inter-segment padding gets -1.
        doc_ids.extend([i] * (cap_padded_len + img_padded_len))

        offset += cap_padded_len + img_padded_len

    total_len = offset
    packed = torch.cat(all_tokens, dim=1)  # (B, total_len, dim)
    document_id = torch.tensor(doc_ids, dtype=torch.int32, device=device)

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
    """Build RoPE frequencies for a packed multi-image sequence.

    Each image's tokens get LOCAL positions identical to what
    prepare_rope_cache() would compute for that image alone:
    - Text tokens: axis0 = 1, 2, ..., cap_padded_len; axis1 = 0; axis2 = 0
    - Image tokens: axis0 = cap_padded_len + 1; axis1 = row; axis2 = col
    - Padding tokens: all zeros (masked out by block mask)

    Args:
        packing_info: Layout descriptor from build_packed_sequence().
        rope_embedder: The EmbedND module for computing RoPE frequencies.
        pad_tokens_multiple: Token padding multiple (for computing padded cap len).
        device: Target device.

    Returns:
        freqs_cis: RoPE frequencies for the packed sequence, shape
            (1, n_rope_axes, total_len, 1, rope_dim_per_axis, 2).
    """
    # Build position IDs for the entire packed sequence
    pos_ids = torch.zeros(1, packing_info.total_len, 3, dtype=torch.float32, device=device)

    for i in range(packing_info.n_images):
        text_start, text_len, img_start, img_len = packing_info.segments[i]
        H_t, W_t = packing_info.img_grid_sizes[i]

        # Caption padded length (same as text_len since pad_zimage was already applied)
        cap_padded_len = text_len

        # Text tokens: axis0 = 1, 2, ..., cap_padded_len (local positions)
        pos_ids[0, text_start:text_start + text_len, 0] = (
            torch.arange(cap_padded_len, dtype=torch.float32, device=device) + 1.0
        )
        # axis1, axis2 stay 0 for text tokens

        # Image tokens: axis0 = cap_padded_len + 1, axis1 = row, axis2 = col
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
        # Remaining img_len - n_real_img padding tokens stay at zero

    # Compute RoPE frequencies via the model's rope_embedder
    freqs_cis = rope_embedder(pos_ids).movedim(1, 2)  # (1, n_axes, total_len, 1, dim, 2)
    return freqs_cis


def make_packing_mask_mod(document_id: torch.Tensor):
    """Create FlexAttention mask_mod for document-level attention isolation.

    Tokens can only attend to other tokens belonging to the same image.
    Padding tokens (document_id == -1) are masked out entirely.

    Args:
        document_id: (total_len,) int32 tensor mapping token index to image index.

    Returns:
        mask_mod function compatible with create_block_mask().
    """
    def mask_mod(b, h, q_idx, kv_idx):
        q_doc = document_id[q_idx]
        kv_doc = document_id[kv_idx]
        return (q_doc == kv_doc) & (q_doc >= 0)
    return mask_mod


def unpack_and_unpatchify(
    packed_output: torch.Tensor,
    packing_info: PackingInfo,
    patch_size: int,
    out_channels: int,
) -> list[torch.Tensor]:
    """Extract per-image tokens from packed output and reshape to spatial tensors.

    Args:
        packed_output: (B, total_len, patch_size^2 * out_channels) final layer output.
        packing_info: Layout descriptor from build_packed_sequence().
        patch_size: Spatial patch size (2 for Z-Image).
        out_channels: Number of output channels (16 for Z-Image).

    Returns:
        List of N tensors, each (B, out_channels, H_pixels, W_pixels).
    """
    pH = pW = patch_size
    results = []

    for i in range(packing_info.n_images):
        _text_start, _text_len, img_start, _img_len = packing_info.segments[i]
        H_t, W_t = packing_info.img_grid_sizes[i]
        n_real_img = H_t * W_t

        # Extract real image tokens (skip padding)
        img_tokens = packed_output[:, img_start:img_start + n_real_img, :]

        # Unpatchify: (B, H_t*W_t, pH*pW*C) -> (B, C, H_t*pH, W_t*pW)
        img = (
            img_tokens
            .view(-1, H_t, W_t, pH, pW, out_channels)
            .permute(0, 5, 1, 3, 2, 4)  # (B, C, H_t, pH, W_t, pW)
            .flatten(4, 5)  # (B, C, H_t, pH, H_pixels_w)
            .flatten(2, 3)  # (B, C, H_pixels_h, H_pixels_w)
        )
        results.append(img)

    return results


# --- EmbedND (from flux/layers.py) ---

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


# --- TimestepEmbedder (from mmdit.py) ---

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


# --- RMSNorm (nn.Module version) ---

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


# --- JointAttention ---

class JointAttention(nn.Module):
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
        bsz, seqlen, _ = x.shape

        qkv_out = self.qkv(x)

        # Fused path: split + qk_norm + rope + transpose in one kernel
        if getattr(self, '_use_fused_qkv', False) and _HAS_FUSED_QKV:
            xq, xk, xv = fused_qkv_postprocess(
                qkv_out, self.q_norm.weight, self.k_norm.weight,
                freqs_cis, self.n_local_heads, self.head_dim,
                self.q_norm.eps,
            )
            # Output is already (B, heads, seq, head_dim) — SDPA-ready
            output = sdpa_attention(
                xq, xk, xv,
                self.n_local_heads, x_mask, skip_reshape=True,
                block_mask=block_mask,
            )
            return self.out(output)

        # Unfused path
        xq, xk, xv = torch.split(
            qkv_out,
            [
                self.n_local_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq, xk = apply_rope_flux(xq, xk, freqs_cis)

        # GQA repeat
        n_rep = self.n_local_heads // self.n_local_kv_heads
        if n_rep >= 1:
            xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        output = sdpa_attention(
            xq.movedim(1, 2), xk.movedim(1, 2), xv.movedim(1, 2),
            self.n_local_heads, x_mask, skip_reshape=True,
            block_mask=block_mask,
        )
        return self.out(output)


# --- FeedForward ---

class FeedForward(nn.Module):
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
        """Horizontally fuse w1 and w3 into a single GEMM.

        After fusion, forward() does one act_quant + one GEMM (instead of two
        each), then splits the output for SiLU gating. Bit-for-bit identical
        to the unfused path because GEMM output for any given (m, n) element
        depends only on the corresponding weight rows, which are unchanged.
        """
        if self._fused:
            return

        from .fp8 import FP8Linear

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
        """Enable FP8 chain fusion: SiLU+gate → FP8 requant → w2 GEMM.

        After enable, the fused forward skips the BF16 intermediate between
        SiLU+gate output and w2 input quantization. Requires w1w3 fusion first.
        """
        if not self._fused:
            return
        from .fp8 import FP8Linear
        if isinstance(self.w2, FP8Linear):
            self._fused_chain = True

    def _forward_fused_chain(self, x: torch.Tensor) -> torch.Tensor:
        """Full FP8 chain: w1w3 → SiLU+gate+requant → w2 (no BF16 intermediate).

        Uses custom_op-wrapped Triton kernels for torch.compile compatibility.
        All internal calls (w1w3 via FP8Linear, fp8_silu_gate_quant_op,
        fp8_gemm_blockwise_op) are registered as custom_ops, so this method
        has NO graph breaks and can be captured into CUDA graphs.
        """
        from .fp8_kernels import (
            fp8_silu_gate_quant, fp8_silu_gate_quant_op,
            fp8_gemm_blockwise, fp8_gemm_blockwise_op,
            fp8_gemm_v1t, fp8_gemm_v1t_op,
            _USE_FP8_KERNEL_CUSTOM_OP, _DTYPE_TO_CODE_K,
        )
        w1w3_out = self.w1w3(x)  # FP8Linear → BF16 (B, seq, 2*hidden)

        if _USE_FP8_KERNEL_CUSTOM_OP:
            # Custom op path: torch.compile-friendly, no graph breaks
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
        else:
            # Eager fallback
            gated_fp8, gated_scale = fp8_silu_gate_quant(
                w1w3_out.contiguous(), block_size=self.w2.block_size,
            )
            if getattr(self.w2, '_transposed', False):
                return fp8_gemm_v1t(
                    gated_fp8, gated_scale,
                    self.w2.weight, self.w2.weight_scale,
                    input_block_size=self.w2.block_size,
                    output_dtype=self.w2.output_dtype,
                )
            return fp8_gemm_blockwise(
                gated_fp8, gated_scale,
                self.w2.weight, self.w2.weight_scale,
                input_block_size=self.w2.block_size,
                output_dtype=self.w2.output_dtype,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fused:
            if getattr(self, '_fused_chain', False):
                return self._forward_fused_chain(x)
            w1w3_out = self.w1w3(x)
            w1_out, w3_out = w1w3_out.split(self.hidden_dim, dim=-1)
            return self.w2(F.silu(w1_out) * w3_out)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# --- JointTransformerBlock ---

class JointTransformerBlock(nn.Module):
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

            if getattr(self, '_use_fused_elementwise', False):
                # Fused path: 4 kernel launches instead of ~12
                # chunk() returns non-contiguous views; kernels need contiguous
                scale_msa = scale_msa.contiguous()
                gate_msa = gate_msa.contiguous()
                scale_mlp = scale_mlp.contiguous()
                gate_mlp = gate_mlp.contiguous()

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
                x = x + apply_gate(gate_msa.unsqueeze(1).tanh(), self.attention_norm2(
                    self.attention(
                        modulate(self.attention_norm1(x), scale_msa),
                        x_mask, freqs_cis,
                        block_mask=block_mask,
                    )
                ))
                x = x + apply_gate(gate_mlp.unsqueeze(1).tanh(), self.ffn_norm2(
                    self.feed_forward(
                        modulate(self.ffn_norm1(x), scale_mlp),
                    )
                ))
        else:
            assert adaln_input is None
            x = x + self.attention_norm2(
                self.attention(self.attention_norm1(x), x_mask, freqs_cis,
                               block_mask=block_mask)
            )
            x = x + self.ffn_norm2(
                self.feed_forward(self.ffn_norm1(x))
            )
        return x


# --- FinalLayer ---

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


# --- NextDiT (Z-Image variant) ---

class NextDiT(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        dim: int = 3840,
        n_layers: int = 32,
        n_refiner_layers: int = 2,
        n_heads: int = 30,
        n_kv_heads: int = 30,
        multiple_of: int = 256,
        ffn_dim_multiplier: float = 8.0 / 3.0,
        norm_eps: float = 1e-5,
        qk_norm: bool = False,
        cap_feat_dim: int = 2560,
        axes_dims: list[int] = (32, 48, 48),
        axes_lens: list[int] = (1536, 512, 512),
        rope_theta: float = 256.0,
        z_image_modulation: bool = True,
        time_scale: float = 1000.0,
        pad_tokens_multiple: int | None = 32,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.dtype = dtype
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.time_scale = time_scale
        self.pad_tokens_multiple = pad_tokens_multiple

        self.x_embedder = nn.Linear(patch_size * patch_size * in_channels, dim,
                                    bias=True, device=device, dtype=dtype)

        self.noise_refiner = nn.ModuleList([
            JointTransformerBlock(
                layer_id, dim, n_heads, n_kv_heads, multiple_of, ffn_dim_multiplier,
                norm_eps, qk_norm, modulation=True, z_image_modulation=z_image_modulation,
                device=device, dtype=dtype,
            )
            for layer_id in range(n_refiner_layers)
        ])

        self.context_refiner = nn.ModuleList([
            JointTransformerBlock(
                layer_id, dim, n_heads, n_kv_heads, multiple_of, ffn_dim_multiplier,
                norm_eps, qk_norm, modulation=False, device=device, dtype=dtype,
            )
            for layer_id in range(n_refiner_layers)
        ])

        self.t_embedder = TimestepEmbedder(
            min(dim, 1024), output_size=256 if z_image_modulation else None,
            device=device, dtype=dtype,
        )

        self.cap_embedder = nn.Sequential(
            RMSNormModule(cap_feat_dim, eps=norm_eps, elementwise_affine=True, device=device, dtype=dtype),
            nn.Linear(cap_feat_dim, dim, bias=True, device=device, dtype=dtype),
        )

        self.layers = nn.ModuleList([
            JointTransformerBlock(
                layer_id, dim, n_heads, n_kv_heads, multiple_of, ffn_dim_multiplier,
                norm_eps, qk_norm, z_image_modulation=z_image_modulation,
                attn_out_bias=False, device=device, dtype=dtype,
            )
            for layer_id in range(n_layers)
        ])

        self.final_layer = FinalLayer(dim, patch_size, self.out_channels,
                                      z_image_modulation=z_image_modulation,
                                      device=device, dtype=dtype)

        if self.pad_tokens_multiple is not None:
            self.x_pad_token = nn.Parameter(torch.empty((1, dim), device=device, dtype=dtype))
            self.cap_pad_token = nn.Parameter(torch.empty((1, dim), device=device, dtype=dtype))

        assert (dim // n_heads) == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.rope_embedder = EmbedND(dim=dim // n_heads, theta=rope_theta, axes_dim=list(axes_dims))
        self.dim = dim
        self.n_heads = n_heads

    def prepare_adaln_cache(self) -> None:
        """Pre-stack adaLN weights from all modulated blocks for batched computation.

        Concatenates the Linear weight matrices from noise_refiner (2 blocks) and
        main layers (30 blocks) into a single (32*15360, 256) weight matrix and
        (32*15360,) bias vector. This allows computing all 32 adaLN projections
        in one GEMM instead of 32 separate kernel launches.

        The FinalLayer's adaLN is excluded (different architecture: SiLU + different
        output dim).
        """
        from .fp8 import FP8Linear, dequantize_fp8_blockwise
        weights = []
        biases = []
        for block in itertools.chain(self.noise_refiner, self.layers):
            if block.modulation:
                # z_image_modulation=True -> adaLN is nn.Sequential(nn.Linear(256, 15360))
                linear = block.adaLN_modulation[0]  # The nn.Linear or FP8Linear
                if isinstance(linear, FP8Linear):
                    # Dequantize FP8 weight to BF16 for batched F.linear
                    w_bf16 = dequantize_fp8_blockwise(
                        linear.weight, linear.weight_scale,
                        block_size=linear.block_size,
                        output_dtype=torch.bfloat16,
                    )
                    weights.append(w_bf16)
                    biases.append(linear.bias)
                else:
                    weights.append(linear.weight)
                    biases.append(linear.bias)

        # Stack into one big GEMM target (all BF16)
        self.register_buffer('_adaln_W', torch.cat(weights, dim=0))   # (n_blocks*4*dim, 256)
        self.register_buffer('_adaln_B', torch.cat(biases, dim=0))     # (n_blocks*4*dim,)
        self._adaln_n_blocks = len(weights)
        self._adaln_output_dim = weights[0].shape[0]  # 4*dim = 15360

    def _compute_adaln_params(self, adaln_input: torch.Tensor) -> list[tuple[torch.Tensor, ...]] | None:
        """Compute all adaLN params in one batched GEMM.

        Args:
            adaln_input: (B, 256) timestep embedding.

        Returns:
            List of N tuples: (scale_msa, gate_msa, scale_mlp, gate_mlp),
            each containing (B, dim) tensors. Returns None if cache not prepared.
        """
        if not hasattr(self, '_adaln_W'):
            return None

        # One GEMM: (B, 256) @ (256, N*4*dim) + bias -> (B, N*4*dim)
        all_params = F.linear(adaln_input, self._adaln_W, self._adaln_B)

        # Split into N blocks of (B, 4*dim), then chunk each into 4 x (B, dim)
        per_block = all_params.split(self._adaln_output_dim, dim=-1)
        return [p.chunk(4, dim=-1) for p in per_block]

    def prepare_rope_cache(self, h: int, w: int, cap_len: int, device: torch.device) -> dict:
        """Precompute RoPE embeddings for reuse across sampling steps.

        Args:
            h: Image height (before patching, after pad_to_patch_size).
            w: Image width (before patching, after pad_to_patch_size).
            cap_len: Caption token count after cap_embedder (before padding).
            device: Target device.

        Returns:
            Dict with 'cap_freqs_cis', 'x_freqs_cis', 'freqs_cis' tensors.
        """
        pH = pW = self.patch_size

        # Caption position IDs and RoPE
        cap_len_padded = cap_len + ((-cap_len) % self.pad_tokens_multiple) if self.pad_tokens_multiple else cap_len
        cap_pos_ids = torch.zeros(1, cap_len_padded, 3, dtype=torch.float32, device=device)
        cap_pos_ids[:, :, 0] = torch.arange(cap_len_padded, dtype=torch.float32, device=device) + 1.0
        cap_freqs_cis = self.rope_embedder(cap_pos_ids).movedim(1, 2)

        # Image patch position IDs and RoPE
        H_tokens, W_tokens = h // pH, w // pW
        n_img_tokens = H_tokens * W_tokens
        n_img_padded = n_img_tokens + ((-n_img_tokens) % self.pad_tokens_multiple) if self.pad_tokens_multiple else n_img_tokens

        x_pos_ids = torch.zeros((1, n_img_padded, 3), dtype=torch.float32, device=device)
        x_pos_ids[:, :n_img_tokens, 0] = cap_len_padded + 1
        x_pos_ids[:, :n_img_tokens, 1] = torch.arange(H_tokens, dtype=torch.float32, device=device).view(-1, 1).repeat(1, W_tokens).flatten()
        x_pos_ids[:, :n_img_tokens, 2] = torch.arange(W_tokens, dtype=torch.float32, device=device).view(1, -1).repeat(H_tokens, 1).flatten()
        x_freqs_cis = self.rope_embedder(x_pos_ids).movedim(1, 2)

        # Concatenated for main transformer layers
        freqs_cis = torch.cat([cap_freqs_cis, x_freqs_cis], dim=1)

        return {
            'cap_freqs_cis': cap_freqs_cis,
            'x_freqs_cis': x_freqs_cis,
            'freqs_cis': freqs_cis,
        }

    def unpatchify(self, x: torch.Tensor, img_size: list[tuple[int, int]],
                   cap_size: list[int], return_tensor: bool = False) -> list[torch.Tensor] | torch.Tensor:
        pH = pW = self.patch_size
        imgs = []
        for i in range(x.size(0)):
            H, W = img_size[i]
            begin = cap_size[i]
            end = begin + (H // pH) * (W // pW)
            imgs.append(
                x[i][begin:end]
                .view(H // pH, W // pW, pH, pW, self.out_channels)
                .permute(4, 0, 2, 1, 3)
                .flatten(3, 4)
                .flatten(1, 2)
            )
        if return_tensor:
            imgs = torch.stack(imgs, dim=0)
        return imgs

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor,
                context: torch.Tensor, num_tokens: int,
                attention_mask: torch.Tensor | None = None,
                rope_cache: dict | None = None) -> torch.Tensor:
        """Forward pass of NextDiT.

        Args:
            x: (B, C, H, W) noisy latent.
            timesteps: (B,) sigma values.
            context: (B, seq, cap_feat_dim) text encoder hidden states.
            num_tokens: Number of text tokens.
            attention_mask: Optional attention mask for text.
            rope_cache: Optional precomputed RoPE from prepare_rope_cache().

        Returns:
            (B, C, H, W) model output (NEGATED, as per ComfyUI).
        """
        t = 1 - timesteps
        cap_feats = context
        bs, c, h, w = x.shape
        x = pad_to_patch_size(x, (self.patch_size, self.patch_size))

        t_emb = self.t_embedder(t * self.time_scale, dtype=x.dtype)
        adaln_input = t_emb

        # Pre-compute all adaLN params in one batched GEMM (if cache prepared)
        adaln_params = self._compute_adaln_params(adaln_input)

        bsz = x.shape[0]
        pH = pW = self.patch_size
        device = x.device

        # Embed caption
        cap_feats_embedded = self.cap_embedder(cap_feats)
        if self.pad_tokens_multiple is not None:
            cap_feats_embedded, _ = pad_zimage(cap_feats_embedded, self.cap_pad_token, self.pad_tokens_multiple)

        # Embed image patches
        B, C, H, W = x.shape
        x_patches = self.x_embedder(
            x.view(B, C, H // pH, pH, W // pW, pW).permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2)
        )

        if self.pad_tokens_multiple is not None:
            x_patches, _ = pad_zimage(x_patches, self.x_pad_token, self.pad_tokens_multiple)

        img_len = x_patches.shape[1]

        if rope_cache is not None:
            cap_freqs_cis = rope_cache['cap_freqs_cis']
            x_freqs_cis = rope_cache['x_freqs_cis']
            freqs_cis = rope_cache['freqs_cis']
            # Expand batch dim if batched (CFG batching)
            if bsz > 1 and cap_freqs_cis.shape[0] == 1:
                cap_freqs_cis = cap_freqs_cis.expand(bsz, -1, -1, -1, -1, -1)
                x_freqs_cis = x_freqs_cis.expand(bsz, -1, -1, -1, -1, -1)
                freqs_cis = freqs_cis.expand(bsz, -1, -1, -1, -1, -1)
        else:
            # Compute RoPE on the fly (backwards compatible)
            cap_pos_ids = torch.zeros(bsz, cap_feats_embedded.shape[1], 3, dtype=torch.float32, device=device)
            cap_pos_ids[:, :, 0] = torch.arange(cap_feats_embedded.shape[1], dtype=torch.float32, device=device) + 1.0
            cap_freqs_cis = self.rope_embedder(cap_pos_ids).movedim(1, 2)

            H_tokens, W_tokens = H // pH, W // pW
            x_pos_ids = torch.zeros((bsz, H_tokens * W_tokens, 3), dtype=torch.float32, device=device)
            cap_feats_len_total = cap_feats_embedded.shape[1]
            x_pos_ids[:, :, 0] = cap_feats_len_total + 1
            x_pos_ids[:, :, 1] = torch.arange(H_tokens, dtype=torch.float32, device=device).view(-1, 1).repeat(1, W_tokens).flatten()
            x_pos_ids[:, :, 2] = torch.arange(W_tokens, dtype=torch.float32, device=device).view(1, -1).repeat(H_tokens, 1).flatten()
            if self.pad_tokens_multiple is not None:
                pad_extra = (-H_tokens * W_tokens) % self.pad_tokens_multiple
                x_pos_ids = F.pad(x_pos_ids, (0, 0, 0, pad_extra))
            x_freqs_cis = self.rope_embedder(x_pos_ids).movedim(1, 2)
            freqs_cis = torch.cat([cap_freqs_cis, x_freqs_cis], dim=1)

        # Refine context
        for layer in self.context_refiner:
            cap_feats_embedded = layer(cap_feats_embedded, None, cap_freqs_cis)

        # Refine noise (first N_refiner blocks from adaln_params)
        param_idx = 0
        for layer in self.noise_refiner:
            precomputed = adaln_params[param_idx] if adaln_params is not None else None
            x_patches = layer(x_patches, None, x_freqs_cis,
                              adaln_input=adaln_input,
                              precomputed_adaln=precomputed)
            param_idx += 1

        # Concatenate and run main layers
        padded_full_embed = torch.cat([cap_feats_embedded, x_patches], dim=1)

        l_effective_cap_len = [padded_full_embed.shape[1] - img_len] * bsz
        img_sizes = [(H, W)] * bsz

        for layer in self.layers:
            precomputed = adaln_params[param_idx] if adaln_params is not None else None
            padded_full_embed = layer(padded_full_embed, None, freqs_cis,
                                      adaln_input=adaln_input,
                                      precomputed_adaln=precomputed)
            param_idx += 1

        img = self.final_layer(padded_full_embed, adaln_input)
        img = self.unpatchify(img, img_sizes, l_effective_cap_len, return_tensor=True)[:, :, :h, :w]

        return -img

    def prepare_packed_state(
        self,
        context_list: list[torch.Tensor],
        img_sizes: list[tuple[int, int]],
        cap_lens: list[int],
        device: torch.device,
    ) -> tuple[list[torch.Tensor], PackingInfo, torch.Tensor]:
        """Pre-compute constant state for packed multi-image generation.

        Runs cap_embedder + context_refiner for each image, computes packing
        layout and RoPE frequencies. These are all constant across euler steps.

        Args:
            context_list: N raw text conditionings, each (1, seq_i, cap_feat_dim).
            img_sizes: (H, W) per image AFTER pad_to_patch_size (pixel dimensions).
            cap_lens: Original caption lengths per image (before embedding/padding).
            device: Target device.

        Returns:
            refined_cap_list: N tensors of refined+padded caption embeddings,
                each (1, cap_padded_len_i, dim).
            packing_info: PackingInfo describing the packed sequence layout.
            packed_rope: RoPE frequencies for the packed sequence.
        """
        pH = pW = self.patch_size

        refined_cap_list = []
        dummy_img_patches_list = []
        img_grid_sizes = []
        embedded_cap_lens = []

        for i in range(len(context_list)):
            # Embed caption
            cap_embedded = self.cap_embedder(context_list[i])  # (1, seq_i, dim)
            cap_embedded_len = cap_embedded.shape[1]

            # Pad caption
            if self.pad_tokens_multiple is not None:
                cap_embedded, _ = pad_zimage(cap_embedded, self.cap_pad_token, self.pad_tokens_multiple)

            # Compute per-image RoPE for context_refiner
            H, W = img_sizes[i]
            H_t, W_t = H // pH, W // pW
            img_grid_sizes.append((H_t, W_t))

            rope_cache = self.prepare_rope_cache(H, W, cap_embedded_len, device)
            cap_freqs_cis = rope_cache['cap_freqs_cis']

            # Expand RoPE for CFG batch dimension (B=2 for pos+neg)
            bsz = cap_embedded.shape[0]
            if bsz > 1 and cap_freqs_cis.shape[0] == 1:
                cap_freqs_cis = cap_freqs_cis.expand(bsz, -1, -1, -1, -1, -1)

            # Run context_refiner (no adaLN, no timestep dependency)
            for layer in self.context_refiner:
                cap_embedded = layer(cap_embedded, None, cap_freqs_cis)

            refined_cap_list.append(cap_embedded)
            embedded_cap_lens.append(cap_embedded_len)

            # Create a dummy image patch tensor to determine layout
            # (only need shape, not data -- data changes each step)
            n_img_tokens = H_t * W_t
            if self.pad_tokens_multiple is not None:
                n_img_padded = n_img_tokens + ((-n_img_tokens) % self.pad_tokens_multiple)
            else:
                n_img_padded = n_img_tokens
            dummy_img_patches_list.append(
                torch.zeros(1, n_img_padded, self.dim, device=device, dtype=cap_embedded.dtype)
            )

        # Build packing layout from refined caps + dummy img patches.
        # Use batch-1 slices for layout computation — the segment layout
        # (offsets, document_id) is B-independent.  This allows the caller
        # to pass (B, seq, cap_feat_dim) conditionings for CFG batching
        # where B=2 and batch[0]=pos, batch[1]=neg.
        layout_caps = [cap[:1] for cap in refined_cap_list]
        _, packing_info = build_packed_sequence(
            layout_caps,
            dummy_img_patches_list,
            img_grid_sizes,
            embedded_cap_lens,
            self.pad_tokens_multiple or 1,
            self.cap_pad_token,
            self.x_pad_token,
        )

        # Build packed RoPE
        packed_rope = build_packed_rope(
            packing_info,
            self.rope_embedder,
            self.pad_tokens_multiple or 1,
            device,
        )

        return refined_cap_list, packing_info, packed_rope

    def forward_packed(
        self,
        x_list: list[torch.Tensor],
        timesteps: torch.Tensor,
        refined_cap_list: list[torch.Tensor],
        packing_info: PackingInfo,
        block_mask,
        packed_rope: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Packed multi-image forward pass through the diffusion model.

        Processes N images of varying resolutions in a single forward pass
        through the 30 main transformer layers, using FlexAttention block
        masks for document-level attention isolation.

        Refiners run per-image (NOT packed): context_refiner outputs are
        pre-computed (passed in as refined_cap_list), noise_refiner runs
        per-image inside this method.

        Args:
            x_list: N noisy latent images, each (B, C, H_i, W_i).
            timesteps: (B,) sigma values (shared across all images).
            refined_cap_list: N pre-refined caption embeddings from
                prepare_packed_state(), each (1, cap_padded_len_i, dim).
            packing_info: Pre-built PackingInfo (constant across euler steps).
            block_mask: Pre-built FlexAttention BlockMask (constant across steps).
            packed_rope: Pre-built packed RoPE frequencies (constant across steps).

        Returns:
            List of N output tensors, each (B, C, H_i, W_i), NEGATED.
        """
        n_images = len(x_list)
        B = x_list[0].shape[0]
        pH = pW = self.patch_size
        device = x_list[0].device

        # Compute timestep embedding and adaLN input (same for all images)
        t = 1 - timesteps
        t_emb = self.t_embedder(t * self.time_scale, dtype=x_list[0].dtype)
        adaln_input = t_emb

        # Pre-compute all adaLN params in one batched GEMM
        adaln_params = self._compute_adaln_params(adaln_input)

        # Process each image through patchify + noise_refiner
        img_patches_list = []
        original_hw = []  # (h, w) before pad_to_patch_size, for final crop

        for i in range(n_images):
            x_i = x_list[i]
            bs_i, c_i, h_i, w_i = x_i.shape
            original_hw.append((h_i, w_i))

            # Pad to patch size
            x_i = pad_to_patch_size(x_i, (pH, pW))
            B_i, C_i, H_i, W_i = x_i.shape
            H_t, W_t = H_i // pH, W_i // pW

            # Patchify + embed
            patches = self.x_embedder(
                x_i.view(B_i, C_i, H_t, pH, W_t, pW)
                .permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2)
            )  # (B, H_t*W_t, dim)

            # Pad image patches
            if self.pad_tokens_multiple is not None:
                patches, _ = pad_zimage(patches, self.x_pad_token, self.pad_tokens_multiple)

            # Compute per-image RoPE for noise_refiner
            # packing_info.cap_lens[i] = embedded cap length before padding,
            # which is what prepare_rope_cache expects (it pads internally)
            rope_cache_i = self.prepare_rope_cache(
                H_i, W_i,
                packing_info.cap_lens[i],
                device,
            )
            x_freqs_cis = rope_cache_i['x_freqs_cis']

            # Expand for CFG batch
            if B > 1 and x_freqs_cis.shape[0] == 1:
                x_freqs_cis = x_freqs_cis.expand(B, -1, -1, -1, -1, -1)

            # Run noise_refiner per-image (first N_refiner adaln params)
            param_idx = 0
            for layer in self.noise_refiner:
                precomputed = adaln_params[param_idx] if adaln_params is not None else None
                patches = layer(patches, None, x_freqs_cis,
                                adaln_input=adaln_input,
                                precomputed_adaln=precomputed)
                param_idx += 1

            img_patches_list.append(patches)

        # Build packed sequence from refined caps + noise-refined patches
        # The packing_info layout was built with the same cap/img sizes, so
        # we can directly concatenate in the same order.
        all_tokens = []
        for i in range(n_images):
            # Expand refined caps for CFG batch if needed
            cap_i = refined_cap_list[i]
            if B > 1 and cap_i.shape[0] == 1:
                cap_i = cap_i.expand(B, -1, -1)
            all_tokens.append(cap_i)
            all_tokens.append(img_patches_list[i])
        packed = torch.cat(all_tokens, dim=1)  # (B, total_len, dim)

        # Expand packed RoPE for CFG batch
        rope = packed_rope
        if B > 1 and rope.shape[0] == 1:
            rope = rope.expand(B, -1, -1, -1, -1, -1)

        # Run 30 main transformer layers on packed sequence with block_mask
        # adaLN param_idx continues from where noise_refiner left off
        n_refiner = len(self.noise_refiner)
        param_idx = n_refiner
        for layer in self.layers:
            precomputed = adaln_params[param_idx] if adaln_params is not None else None
            packed = layer(packed, None, rope,
                           adaln_input=adaln_input,
                           precomputed_adaln=precomputed,
                           block_mask=block_mask)
            param_idx += 1

        # Final layer (uses raw adaln_input, not precomputed params)
        packed = self.final_layer(packed, adaln_input)

        # Unpack and unpatchify
        results = unpack_and_unpatchify(
            packed, packing_info, self.patch_size, self.out_channels,
        )

        # Crop to original sizes and negate (matching forward())
        outputs = []
        for i in range(n_images):
            h_orig, w_orig = original_hw[i]
            outputs.append(-results[i][:, :, :h_orig, :w_orig])

        return outputs


def fuse_model(model: nn.Module) -> None:
    """Walk model tree and apply all safe horizontal fusions.

    Fusions applied:
    1. FeedForward w1+w3 → single GEMM (saves 1 act_quant + 1 GEMM per layer)
    2. FP8 chain: SiLU+gate → FP8 requant → w2 GEMM (eliminates BF16 intermediate)
    3. Fused elementwise: RMSNorm+modulate and RMSNorm+gate+residual Triton kernels
       (saves ~10 kernel launches per modulated layer)

    Call after model loading but before torch.compile.
    """
    n_ffn = 0
    n_chain = 0
    n_elem = 0
    n_qkv = 0

    for module in model.modules():
        if isinstance(module, FeedForward):
            module.fuse_w1w3()
            if module._fused:
                n_ffn += 1
                module.enable_fused_chain()
                if getattr(module, '_fused_chain', False):
                    n_chain += 1

        if isinstance(module, JointTransformerBlock):
            if module.modulation and _HAS_FUSED_KERNELS:
                module._use_fused_elementwise = True
                n_elem += 1

        if isinstance(module, JointAttention):
            if _HAS_FUSED_QKV and hasattr(module, 'q_norm') and not isinstance(module.q_norm, nn.Identity):
                module._use_fused_qkv = True
                n_qkv += 1

    # Pre-batch adaLN weights for batched GEMM
    n_adaln = 0
    if isinstance(model, NextDiT):
        model.prepare_adaln_cache()
        n_adaln = model._adaln_n_blocks

    import logging
    log = logging.getLogger(__name__)
    if n_ffn > 0:
        log.info(f"Fused {n_ffn} FeedForward w1+w3 layers")
    if n_chain > 0:
        log.info(f"Fused {n_chain} FP8 FFN chains (SiLU+gate+requant)")
    if n_elem > 0:
        log.info(f"Fused {n_elem} elementwise kernel chains")
    if n_qkv > 0:
        log.info(f"Fused {n_qkv} QKV post-process kernels (norm+rope+transpose)")
    if n_adaln > 0:
        log.info(f"Pre-batched adaLN: {n_adaln} blocks")


def _detect_n_layers(state_dict_keys) -> int:
    """Detect n_layers from state dict keys (looks for 'layers.N.' pattern)."""
    n_layers = 0
    for k in state_dict_keys:
        if k.startswith("layers."):
            layer_idx = int(k.split(".")[1])
            n_layers = max(n_layers, layer_idx + 1)
    return n_layers


def _detect_cap_feat_dim(state_dict: dict[str, torch.Tensor]) -> int:
    """Detect cap_feat_dim from cap_embedder weights. Defaults to 2560 (Z-Image)."""
    key = "cap_embedder.1.weight"
    if key in state_dict:
        return state_dict[key].shape[1]
    return 2560


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


def _detect_qk_norm(state_dict_keys) -> bool:
    """Detect whether the model uses qk_norm from state dict keys."""
    return any(k.endswith(".q_norm.weight") for k in state_dict_keys)


def create_diffusion_model(
    dtype: torch.dtype = torch.bfloat16,
    n_layers: int = 32,
    cap_feat_dim: int = 2560,
    qk_norm: bool = True,
) -> NextDiT:
    """Create the NextDiT Z-Image model architecture on the 'meta' device.

    Returns an uninitialized model skeleton (all parameters on meta device).
    Use this when you need the architecture without loading BF16 weights,
    e.g. for FP8 blockwise weight injection via replace_linear_with_fp8().
    """
    model = NextDiT(
        patch_size=2,
        in_channels=16,
        dim=3840,
        n_layers=n_layers,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        multiple_of=256,
        ffn_dim_multiplier=8.0 / 3.0,
        norm_eps=1e-5,
        qk_norm=qk_norm,
        cap_feat_dim=cap_feat_dim,
        axes_dims=[32, 48, 48],
        axes_lens=[1536, 512, 512],
        rope_theta=256.0,
        z_image_modulation=True,
        time_scale=1000.0,
        pad_tokens_multiple=32,
        device="meta",
        dtype=dtype,
    )
    return model


def load_diffusion_model(
    safetensors_path: str,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
    n_layers: int | None = None,
) -> NextDiT:
    """Load NextDiT Z-Image model from safetensors.

    Auto-detects n_layers from the state dict if not provided.
    For FP8 blockwise models, use create_diffusion_model() instead and
    apply weights via replace_linear_with_fp8().
    """
    state_dict = load_file(safetensors_path, device=str(device))

    remapped = _strip_diffusion_prefix(state_dict)

    if n_layers is None:
        n_layers = _detect_n_layers(remapped.keys())
    cap_feat_dim = _detect_cap_feat_dim(remapped)
    qk_norm = _detect_qk_norm(remapped.keys())

    model = create_diffusion_model(dtype=dtype, n_layers=n_layers, cap_feat_dim=cap_feat_dim, qk_norm=qk_norm)

    missing, unexpected = model.load_state_dict(remapped, strict=False, assign=True)
    model = model.to(device)
    model.eval()
    return model


def test_batched_adaln():
    """Test that batched adaLN produces bitwise identical output to per-layer computation."""
    import time

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Create a small NextDiT (n_layers=4 + 2 noise_refiner = 6 modulated blocks)
    model = create_diffusion_model(dtype=dtype, n_layers=4, cap_feat_dim=2560, qk_norm=True)
    # Meta device -> real device: to_empty() then initialize weights
    model = model.to_empty(device=device)
    model = model.to(dtype=dtype)
    # Initialize all parameters with random data (they were uninitialized from meta)
    for p in model.parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p)
    model.eval()

    # Prepare the batched adaLN cache
    model.prepare_adaln_cache()
    print(f"Prepared adaLN cache: {model._adaln_n_blocks} blocks, "
          f"output_dim={model._adaln_output_dim}")
    print(f"  _adaln_W shape: {model._adaln_W.shape}")
    print(f"  _adaln_B shape: {model._adaln_B.shape}")

    # Input: (B=2, 256) — simulates CFG batching
    adaln_input = torch.randn(2, 256, device=device, dtype=dtype)

    # OLD: per-layer sequential computation
    old_params = []
    for block in itertools.chain(model.noise_refiner, model.layers):
        if block.modulation:
            params = block.adaLN_modulation(adaln_input).chunk(4, dim=-1)
            old_params.append(params)

    # NEW: batched computation
    new_params = model._compute_adaln_params(adaln_input)

    assert new_params is not None, "Cache not prepared"
    assert len(old_params) == len(new_params), (
        f"Block count mismatch: old={len(old_params)} vs new={len(new_params)}"
    )

    # Compare: must be bitwise identical
    param_names = ["scale_msa", "gate_msa", "scale_mlp", "gate_mlp"]
    for i, (old, new) in enumerate(zip(old_params, new_params)):
        for j, (o, n) in enumerate(zip(old, new)):
            diff = (o - n).abs().max().item()
            assert diff == 0.0, (
                f"Block {i} {param_names[j]}: max diff {diff} (expected 0.0)"
            )

    print(f"All {len(old_params)} blocks match (bitwise identical)")

    # Timing comparison
    n_iters = 200
    torch.cuda.synchronize()

    # Warm up
    for _ in range(10):
        for block in itertools.chain(model.noise_refiner, model.layers):
            if block.modulation:
                block.adaLN_modulation(adaln_input).chunk(4, dim=-1)
    torch.cuda.synchronize()

    # Time old path
    t0 = time.perf_counter()
    for _ in range(n_iters):
        for block in itertools.chain(model.noise_refiner, model.layers):
            if block.modulation:
                block.adaLN_modulation(adaln_input).chunk(4, dim=-1)
    torch.cuda.synchronize()
    old_time = time.perf_counter() - t0

    # Warm up new path
    for _ in range(10):
        model._compute_adaln_params(adaln_input)
    torch.cuda.synchronize()

    # Time new path
    t0 = time.perf_counter()
    for _ in range(n_iters):
        model._compute_adaln_params(adaln_input)
    torch.cuda.synchronize()
    new_time = time.perf_counter() - t0

    print(f"Timing ({n_iters} iterations, {model._adaln_n_blocks} blocks):")
    print(f"  Sequential: {old_time*1000:.2f} ms total, {old_time/n_iters*1000:.3f} ms/iter")
    print(f"  Batched:    {new_time*1000:.2f} ms total, {new_time/n_iters*1000:.3f} ms/iter")
    print(f"  Speedup:    {old_time/new_time:.2f}x")
    print("Test passed!")

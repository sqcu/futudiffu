"""Attention utilities ported from ComfyUI.

Sources:
- comfy/ldm/modules/attention.py (attention_pytorch / SDPA path)
- comfy/ldm/flux/math.py (rope, apply_rope for NextDiT)
- comfy/text_encoders/llama.py (apply_rope for Qwen3)
"""

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


# --- Attention backend dispatch ---

_ATTENTION_BACKEND = "sdpa"  # "sdpa" | "sage" | "auto"


def set_attention_backend(backend: str) -> None:
    """Set the attention backend for sdpa_attention().

    Args:
        backend: One of "sdpa" (PyTorch SDPA), "sage" (SageAttention FP8 QK),
                 or "auto" (use sage when possible, fall back to SDPA).
    """
    global _ATTENTION_BACKEND
    if backend not in ("sdpa", "sage", "auto"):
        raise ValueError(f"Unknown attention backend: {backend!r}")
    _ATTENTION_BACKEND = backend


# --- SDPA attention (port of attention_pytorch) ---

def sdpa_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    heads: int,
    mask: Tensor | None = None,
    skip_reshape: bool = False,
) -> Tensor:
    """Scaled dot-product attention via PyTorch SDPA.

    Args:
        q, k, v: Query/Key/Value tensors.
        heads: Number of attention heads.
        mask: Optional attention mask.
        skip_reshape: If True, input is already (B, heads, seq, dim).

    Returns:
        Output tensor (B, seq, heads*dim).
    """
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))

    # Dispatch to SageAttention when mask=None and backend is sage/auto.
    # Text encoder always passes mask != None, so it always uses PyTorch SDPA.
    # Diffusion model always passes mask=None, so it uses Sage when enabled.
    if mask is None and _ATTENTION_BACKEND in ("sage", "auto"):
        try:
            from .sage_attention import sage_attn_op
            sm_scale = 1.0 / (dim_head ** 0.5)
            out, _lse = sage_attn_op(q, k, v, sm_scale)
            out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
            return out
        except Exception:
            if _ATTENTION_BACKEND == "sage":
                raise
            # "auto" mode: fall through to SDPA on failure

    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out


# --- RoPE for NextDiT (Lumina / Flux style) ---

def rope_embed(pos: Tensor, dim: int, theta: float) -> Tensor:
    """Generate RoPE rotation matrices.

    Port of comfy/ldm/flux/math.py:rope().
    """
    assert dim % 2 == 0
    device = pos.device
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=device)


def apply_rope_flux(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    """Apply RoPE (Flux/Lumina style 2x2 rotation matrices).

    Port of comfy/ldm/flux/math.py:_apply_rope().
    """
    return _apply_rope1_flux(xq, freqs_cis), _apply_rope1_flux(xk, freqs_cis)


def _apply_rope1_flux(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """Apply RoPE to a single tensor (Flux/Lumina style).

    Port of comfy/ldm/flux/math.py:_apply_rope1().
    """
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    x_out = freqs_cis[..., 0] * x_[..., 0]
    x_out.addcmul_(freqs_cis[..., 1], x_[..., 1])
    return x_out.reshape(*x.shape).type_as(x)


# --- RoPE for Qwen3/Llama (standard half-rotate style) ---

def rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dims."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def precompute_freqs_cis_llama(
    head_dim: int,
    position_ids: Tensor,
    theta: float = 1000000.0,
    device=None,
) -> tuple[Tensor, Tensor]:
    """Precompute cos/sin for Llama/Qwen3 RoPE.

    Port of comfy/text_encoders/llama.py:precompute_freqs_cis() (single-theta path).
    """
    theta_numerator = torch.arange(0, head_dim, 2, device=device).float()
    inv_freq = 1.0 / (theta ** (theta_numerator / head_dim))

    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(1)
    sin = emb.sin().unsqueeze(1)
    return cos, sin


def apply_rope_llama(xq: Tensor, xk: Tensor, freqs_cis: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    """Apply standard Llama/Qwen3 RoPE.

    Port of comfy/text_encoders/llama.py:apply_rope().
    """
    org_dtype = xq.dtype
    cos, sin = freqs_cis
    q_embed = (xq * cos) + (rotate_half(xq) * sin)
    k_embed = (xk * cos) + (rotate_half(xk) * sin)
    return q_embed.to(org_dtype), k_embed.to(org_dtype)


# --- RMSNorm ---

def rms_norm(x: Tensor, weight: Tensor | None = None, eps: float = 1e-6) -> Tensor:
    """RMSNorm matching comfy/rmsnorm.py behavior."""
    try:
        if weight is None:
            return torch.nn.functional.rms_norm(x, (x.shape[-1],), eps=eps)
        else:
            return torch.nn.functional.rms_norm(x, weight.shape, weight=weight.to(dtype=x.dtype, device=x.device), eps=eps)
    except AttributeError:
        r = x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        if weight is None:
            return r
        return r * weight.to(dtype=x.dtype, device=x.device)

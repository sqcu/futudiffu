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

import math
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
                freqs_cis: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq, xk, xv = torch.split(
            self.qkv(x),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    ) -> torch.Tensor:
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)

            x = x + apply_gate(gate_msa.unsqueeze(1).tanh(), self.attention_norm2(
                self.attention(
                    modulate(self.attention_norm1(x), scale_msa),
                    x_mask, freqs_cis,
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
                self.attention(self.attention_norm1(x), x_mask, freqs_cis)
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

        # Refine noise
        for layer in self.noise_refiner:
            x_patches = layer(x_patches, None, x_freqs_cis, adaln_input)

        # Concatenate and run main layers
        padded_full_embed = torch.cat([cap_feats_embedded, x_patches], dim=1)

        l_effective_cap_len = [padded_full_embed.shape[1] - img_len] * bsz
        img_sizes = [(H, W)] * bsz

        for layer in self.layers:
            padded_full_embed = layer(padded_full_embed, None, freqs_cis, adaln_input)

        img = self.final_layer(padded_full_embed, adaln_input)
        img = self.unpatchify(img, img_sizes, l_effective_cap_len, return_tensor=True)[:, :, :h, :w]

        return -img


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

"""Qwen3-4B text encoder ported from ComfyUI.

Sources:
- comfy/text_encoders/llama.py (Qwen3_4BConfig, Llama2_, TransformerBlock, Attention, MLP, RMSNorm)
- comfy/text_encoders/z_image.py (chat template, tokenizer wrapping)

Architecture (Qwen3_4BConfig):
  hidden_size=2560, num_hidden_layers=36, num_attention_heads=32,
  num_key_value_heads=8, intermediate_size=9728, vocab_size=151936,
  head_dim=128, rope_theta=1e6, rms_norm_eps=1e-6, q_norm="gemma3", k_norm="gemma3"
"""

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from .attention import (
    apply_rope_llama,
    precompute_freqs_cis_llama,
    rms_norm,
    sdpa_attention,
)

CHAT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
PAD_TOKEN = 151643


@dataclass
class Qwen3_4BConfig:
    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    head_dim: int = 128
    mlp_activation: str = "silu"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


class Attention(nn.Module):
    def __init__(self, config: Qwen3_4BConfig, device=None, dtype=None):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.inner_size = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(config.hidden_size, self.inner_size, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(self.inner_size, config.hidden_size, bias=False, device=device, dtype=dtype)

        # Qwen3 uses "gemma3" style q/k norms
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape

        xq = self.q_proj(hidden_states)
        xk = self.k_proj(hidden_states)
        xv = self.v_proj(hidden_states)

        xq = xq.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq, xk = apply_rope_llama(xq, xk, freqs_cis)

        # GQA: repeat KV heads
        xk = xk.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        xv = xv.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        output = sdpa_attention(xq, xk, xv, self.num_heads, mask=attention_mask, skip_reshape=True)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, config: Qwen3_4BConfig, device=None, dtype=None):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: Qwen3_4BConfig, device=None, dtype=None):
        super().__init__()
        self.self_attn = Attention(config, device=device, dtype=dtype)
        self.mlp = MLP(config, device=device, dtype=dtype)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, attention_mask, freqs_cis)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class Qwen3TextEncoder(nn.Module):
    """Qwen3-4B transformer, matching ComfyUI's Llama2_ with Qwen3_4BConfig."""

    def __init__(self, config: Qwen3_4BConfig | None = None, device=None, dtype=None):
        super().__init__()
        if config is None:
            config = Qwen3_4BConfig()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(config, device=device, dtype=dtype)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_idx: int = -2,
    ) -> torch.Tensor:
        """Forward pass returning intermediate layer output.

        Args:
            input_ids: (B, seq_len) token IDs.
            attention_mask: (B, seq_len) binary mask (1=attend, 0=pad).
            layer_idx: Which layer's output to return. -2 = second-to-last.

        Returns:
            Hidden states from the specified layer, without final norm.
        """
        x = self.embed_tokens(input_ids)
        seq_len = x.shape[1]

        # Cache RoPE embeddings keyed by seq_len
        if not hasattr(self, '_rope_cache') or self._rope_cache_len != seq_len:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
            self._rope_cache = precompute_freqs_cis_llama(
                self.config.head_dim,
                position_ids,
                self.config.rope_theta,
                device=x.device,
            )
            self._rope_cache_len = seq_len
        freqs_cis = self._rope_cache

        # Build attention mask
        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape(
                attention_mask.shape[0], 1, -1, attention_mask.shape[-1]
            ).expand(attention_mask.shape[0], 1, seq_len, attention_mask.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), torch.finfo(x.dtype).min / 4)

        # Causal mask
        if seq_len > 1:
            causal_mask = torch.empty(seq_len, seq_len, dtype=x.dtype, device=x.device).fill_(
                torch.finfo(x.dtype).min / 4
            ).triu_(1)
            if mask is not None:
                mask = mask + causal_mask
            else:
                mask = causal_mask

        # Resolve negative layer_idx
        target_layer = layer_idx
        if target_layer < 0:
            target_layer = len(self.layers) + target_layer

        # Run layers, early-exit after target layer
        for i, layer in enumerate(self.layers):
            x = layer(x, mask, freqs_cis)
            if i == target_layer:
                return x

        # Fallback: return final output (with norm)
        return self.norm(x)


def load_text_encoder(
    safetensors_path: str,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
) -> Qwen3TextEncoder:
    """Load Qwen3-4B text encoder from a safetensors file."""
    config = Qwen3_4BConfig()
    model = Qwen3TextEncoder(config, device="meta", dtype=dtype)

    state_dict = load_file(safetensors_path, device=str(device))

    # Remap keys: ComfyUI stores as "transformer.{layer}.{param}"
    # The safetensors may use various prefixes; try direct load first
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)

    if len(missing) > 0:
        # Try with "model." prefix stripping
        remapped = {}
        for k, v in state_dict.items():
            new_key = k
            if new_key.startswith("model."):
                new_key = new_key[len("model."):]
            if new_key.startswith("transformer."):
                new_key = new_key[len("transformer."):]
            remapped[new_key] = v
        missing, unexpected = model.load_state_dict(remapped, strict=False, assign=True)

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


def create_tokenizer(tokenizer_path: str | None = None):
    """Create Qwen2Tokenizer from the bundled tokenizer data.

    Args:
        tokenizer_path: Path to directory containing vocab.json, merges.txt, tokenizer_config.json.
                        If None, uses bundled tokenizer under this package.
    """
    from transformers import Qwen2Tokenizer

    if tokenizer_path is None:
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tokenizer")

    tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer


def tokenize_prompt(
    tokenizer,
    text: str,
    device: torch.device = torch.device("cuda"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize text with Z-Image chat template.

    Returns:
        (input_ids, attention_mask) tensors on device.
    """
    templated = CHAT_TEMPLATE.format(text)
    encoded = tokenizer(templated, return_tensors="pt", padding=False, add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    return input_ids, attention_mask


def encode_prompt(
    model: Qwen3TextEncoder,
    tokenizer,
    text: str,
    device: torch.device = torch.device("cuda"),
    layer_idx: int = -2,
) -> torch.Tensor:
    """Full encode pipeline: template -> tokenize -> encode -> hidden states.

    Returns:
        Hidden states tensor (1, seq_len, 2560).
    """
    input_ids, attention_mask = tokenize_prompt(tokenizer, text, device)
    with torch.inference_mode():
        hidden_states = model(input_ids, attention_mask=attention_mask, layer_idx=layer_idx)
    return hidden_states

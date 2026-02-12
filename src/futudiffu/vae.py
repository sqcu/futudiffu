"""VAE decoder ported from ComfyUI.

Sources:
- comfy/ldm/models/autoencoder.py (AutoencoderKL, AutoencodingEngineLegacy)
- comfy/ldm/modules/diffusionmodules/model.py (Decoder, ResnetBlock, AttnBlock, Upsample)
- comfy/sd.py (VAE class, config detection)
- comfy/diffusers_convert.py (diffusers-to-SD key conversion)

The Z-Image VAE uses the Flux latent format (16 channels, 8x downscale)
with AutoencoderKL-style architecture.
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

log = logging.getLogger(__name__)


# --- Diffusers-to-SD state dict conversion ---
# Ported faithfully from comfy/diffusers_convert.py

def _build_vae_conversion_maps():
    """Build the VAE key conversion maps (diffusers -> SD format).

    Returns (vae_conversion_map, vae_conversion_map_attn) where each entry
    is (sd_part, hf_part) -- replacements go from hf_part to sd_part.
    """
    vae_conversion_map = [
        # (stable-diffusion, HF Diffusers)
        ("nin_shortcut", "conv_shortcut"),
        ("norm_out", "conv_norm_out"),
        ("mid.attn_1.", "mid_block.attentions.0."),
    ]

    for i in range(4):
        # down_blocks have two resnets
        for j in range(2):
            hf_down_prefix = f"encoder.down_blocks.{i}.resnets.{j}."
            sd_down_prefix = f"encoder.down.{i}.block.{j}."
            vae_conversion_map.append((sd_down_prefix, hf_down_prefix))

        if i < 3:
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0."
            sd_downsample_prefix = f"down.{i}.downsample."
            vae_conversion_map.append((sd_downsample_prefix, hf_downsample_prefix))

            hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
            sd_upsample_prefix = f"up.{3 - i}.upsample."
            vae_conversion_map.append((sd_upsample_prefix, hf_upsample_prefix))

        # up_blocks have three resnets
        # also, up blocks in hf are numbered in reverse from sd
        for j in range(3):
            hf_up_prefix = f"decoder.up_blocks.{i}.resnets.{j}."
            sd_up_prefix = f"decoder.up.{3 - i}.block.{j}."
            vae_conversion_map.append((sd_up_prefix, hf_up_prefix))

    # mid blocks in both encoder and decoder
    for i in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{i}."
        sd_mid_res_prefix = f"mid.block_{i + 1}."
        vae_conversion_map.append((sd_mid_res_prefix, hf_mid_res_prefix))

    vae_conversion_map_attn = [
        # (stable-diffusion, HF Diffusers)
        ("norm.", "group_norm."),
        ("q.", "query."),
        ("k.", "key."),
        ("v.", "value."),
        ("q.", "to_q."),
        ("k.", "to_k."),
        ("v.", "to_v."),
        ("proj_out.", "to_out.0."),
        ("proj_out.", "proj_attn."),
    ]

    return vae_conversion_map, vae_conversion_map_attn


_VAE_CONVERSION_MAP, _VAE_CONVERSION_MAP_ATTN = _build_vae_conversion_maps()


def _reshape_weight_for_sd(w):
    """Convert HF linear weights to SD conv2d weights by adding spatial dims."""
    return w.reshape(*w.shape, 1, 1)


def convert_vae_state_dict(vae_state_dict):
    """Convert a diffusers-format VAE state dict to SD format.

    Ported from comfy/diffusers_convert.py:convert_vae_state_dict.
    """
    mapping = {k: k for k in vae_state_dict.keys()}

    # Apply general key replacements
    for k, v in mapping.items():
        for sd_part, hf_part in _VAE_CONVERSION_MAP:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v

    # Apply attention-specific replacements (only for keys that had "attentions")
    for k, v in mapping.items():
        if "attentions" in k:
            for sd_part, hf_part in _VAE_CONVERSION_MAP_ATTN:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v

    new_state_dict = {v: vae_state_dict[k] for k, v in mapping.items()}

    # Reshape attention weights from 2D linear to 4D conv2d
    weights_to_convert = ["q", "k", "v", "proj_out"]
    for k, v in new_state_dict.items():
        for weight_name in weights_to_convert:
            if f"mid.attn_1.{weight_name}.weight" in k:
                log.debug("Reshaping %s for SD format", k)
                new_state_dict[k] = _reshape_weight_for_sd(v)

    return new_state_dict


def is_diffusers_vae(state_dict):
    """Detect whether a state dict uses diffusers-format keys."""
    return "decoder.up_blocks.0.resnets.0.norm1.weight" in state_dict


# --- Basic blocks ---

def Normalize(in_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: int | None = None,
                 conv_shortcut: bool = False, dropout: float = 0.0,
                 temb_channels: int = 512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, temb: torch.Tensor | None = None) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Reshape for attention
        b, c, h, w = q.shape
        q = q.reshape(b, 1, c, h * w).transpose(2, 3)  # (B, 1, HW, C)
        k = k.reshape(b, 1, c, h * w).transpose(2, 3)
        v = v.reshape(b, 1, c, h * w).transpose(2, 3)

        h_ = F.scaled_dot_product_attention(q, k, v)
        h_ = h_.transpose(2, 3).reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_


# --- Decoder ---

class Decoder(nn.Module):
    def __init__(self, *, ch: int, out_ch: int, ch_mult: tuple[int, ...],
                 num_res_blocks: int, attn_resolutions: list[int],
                 dropout: float = 0.0, resamp_with_conv: bool = True,
                 in_channels: int, resolution: int, z_channels: int,
                 tanh_out: bool = False, **kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.tanh_out = tanh_out

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                         temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        temb = None

        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        if self.tanh_out:
            h = torch.tanh(h)
        return h


# --- AutoencoderKL ---

class AutoencoderKL(nn.Module):
    """AutoencoderKL with only the decode path needed.

    When post_quant_conv weights are missing from the state dict (as with the
    Z-Image VAE), the conv is initialized as an identity 1x1 convolution so
    decode() still passes through correctly.  If the state dict DOES contain
    post_quant_conv weights, load_state_dict will overwrite the identity init.
    """

    def __init__(self, ddconfig: dict, embed_dim: int):
        super().__init__()
        self.decoder = Decoder(**ddconfig)
        z_channels = ddconfig["z_channels"]
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1)
        self.embed_dim = embed_dim

        # Initialize post_quant_conv as identity when embed_dim == z_channels.
        # This makes the conv a no-op if the state dict has no post_quant_conv
        # weights (loaded with strict=False, so missing keys are simply skipped).
        if embed_dim == z_channels:
            with torch.no_grad():
                nn.init.zeros_(self.post_quant_conv.bias)
                # Set weight to identity: (out, in, 1, 1) with eye on (out, in)
                nn.init.zeros_(self.post_quant_conv.weight)
                for i in range(z_channels):
                    self.post_quant_conv.weight[i, i, 0, 0] = 1.0

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        return self.decoder(z)


def detect_vae_config(state_dict: dict) -> tuple[dict, int]:
    """Detect VAE configuration from state dict keys.

    Expects SD-format keys (run convert_vae_state_dict first if diffusers).

    Returns (ddconfig, embed_dim).
    """
    # Detect z_channels from decoder.conv_in.weight
    z_channels = state_dict["decoder.conv_in.weight"].shape[1]

    # Detect embed_dim from post_quant_conv if present, else assume == z_channels.
    # The Z-Image VAE (16 latent channels) has no post_quant_conv in the safetensors
    # file. ComfyUI handles this by using AutoencodingEngine (no post_quant_conv)
    # when the key is absent. We handle it by setting embed_dim = z_channels and
    # later initializing post_quant_conv as identity if needed.
    if "post_quant_conv.weight" in state_dict:
        embed_dim = state_dict["post_quant_conv.weight"].shape[1]
    else:
        embed_dim = z_channels
        log.debug("post_quant_conv not in state dict, inferring embed_dim=%d from z_channels", embed_dim)

    # Detect ch_mult from decoder blocks
    ch_mult = [1, 2, 4, 4]  # default
    # Count up levels
    max_level = 0
    for k in state_dict.keys():
        if k.startswith("decoder.up."):
            level = int(k.split(".")[2])
            max_level = max(max_level, level)

    # Adjust ch_mult length if the model has fewer levels (e.g. x4 upscaler VAE)
    if max_level < 3 and "decoder.up.3.upsample.conv.weight" not in state_dict:
        ch_mult = [1, 2, 4]

    # Detect ch from decoder.mid.block_1
    block_in = state_dict["decoder.mid.block_1.conv1.weight"].shape[0]
    # ch = block_in / ch_mult[-1]
    ch = block_in // ch_mult[-1]

    # Detect num_res_blocks: find max block index at level 0 (highest resolution)
    num_res_blocks = 0
    for k in state_dict.keys():
        if k.startswith("decoder.up.0.block.") and k.endswith(".conv1.weight"):
            idx = int(k.split(".")[4])
            num_res_blocks = max(num_res_blocks, idx)
    # Block indices go from 0 to num_res_blocks. The Decoder __init__ creates
    # num_res_blocks+1 blocks, so the config value is the max index.
    # (e.g. indices 0,1,2 means num_res_blocks=2 in config)

    # Detect out_ch
    out_ch = state_dict["decoder.conv_out.weight"].shape[0]

    ddconfig = {
        "double_z": True,
        "z_channels": z_channels,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": out_ch,
        "ch": ch,
        "ch_mult": ch_mult,
        "num_res_blocks": num_res_blocks,
        "attn_resolutions": [],
        "dropout": 0.0,
    }
    return ddconfig, embed_dim


def load_vae(
    safetensors_path: str,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
) -> AutoencoderKL:
    """Load VAE from safetensors file.

    Handles both SD-format and diffusers-format state dicts. Diffusers keys
    are detected and converted automatically (matching ComfyUI's sd.py:428).
    """
    state_dict = load_file(safetensors_path, device=str(device))

    # Strip any prefix (e.g. "first_stage_model.")
    remapped = {}
    for k, v in state_dict.items():
        new_key = k
        if new_key.startswith("first_stage_model."):
            new_key = new_key[len("first_stage_model."):]
        remapped[new_key] = v

    # Detect and convert diffusers format -> SD format
    # (ComfyUI does this at sd.py:428 before any config detection)
    if is_diffusers_vae(remapped):
        log.info("Detected diffusers-format VAE keys, converting to SD format")
        remapped = convert_vae_state_dict(remapped)

    # Only keep decoder + post_quant_conv keys (drop encoder, quant_conv, etc.)
    decode_keys = {k: v for k, v in remapped.items()
                   if k.startswith("decoder.") or k.startswith("post_quant_conv.")}

    ddconfig, embed_dim = detect_vae_config(decode_keys)

    model = AutoencoderKL(ddconfig, embed_dim)
    missing, unexpected = model.load_state_dict(decode_keys, strict=False, assign=True)

    if missing:
        log.debug("VAE load missing keys: %s", missing)
    if unexpected:
        log.debug("VAE load unexpected keys: %s", unexpected)

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


def vae_decode(model: AutoencoderKL, latent: torch.Tensor) -> torch.Tensor:
    """Decode latent to image.

    Applies Flux process_out before decode, and SD process_output after.

    Args:
        model: Loaded AutoencoderKL.
        latent: (B, 16, H, W) latent tensor.

    Returns:
        (B, 3, H*8, W*8) image tensor in [0, 1] range.
    """
    from .sampling import flux_process_out

    # Apply Flux latent format process_out
    z = flux_process_out(latent)

    with torch.inference_mode():
        decoded = model.decode(z)

    # SD process_output: clamp((image + 1) / 2, 0, 1)
    image = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)
    return image

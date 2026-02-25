"""Centralized model weight paths.

All scripts import paths from here instead of hardcoding Windows paths.
The COMFYUI_ROOT environment variable can override the default location.
"""

from __future__ import annotations

import os
from pathlib import Path

COMFYUI_ROOT = Path(os.environ.get(
    "COMFYUI_ROOT",
    r"F:\dox\ai\comfyui\ComfyUI",
))

FP8_PATH = str(COMFYUI_ROOT / "models" / "diffusion_models" / "z_image_fp8_blockwise.safetensors")
TE_PATH = str(COMFYUI_ROOT / "models" / "text_encoders" / "qwen_3_4b.safetensors")
VAE_PATH = str(COMFYUI_ROOT / "models" / "vae" / "zimage.safetensors")

REPO_ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_PATH = str(REPO_ROOT / "src" / "futudiffu" / "tokenizer")

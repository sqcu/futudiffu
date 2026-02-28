"""Pydantic request/response models for the yeetums BFF.

No torch, no safetensors. Pure JSON interface between browser and BFF.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Generation request/response
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    """Single image generation request from the browser."""
    prompt: str
    negative_prompt: str = ""
    seed: int = -1  # -1 = random
    n_steps: int = 30
    cfg: float = 4.0
    attention_backend: str = "sage"
    sampling_shift: float = 1.0
    multiplier: float = 1.0
    denoise: float = 1.0
    source_id: str | None = None  # For i2i: references uploaded source

    # Resolution: anchor pixels + aspect ratio
    anchor_pixels: int = 1048576  # 1024sq default
    aspect_ratio: float = 1.5385  # ~1280x832


class GenerateResponse(BaseModel):
    """Response after a generation completes."""
    gallery_id: str
    seed: int
    width: int
    height: int
    image_url: str
    elapsed_s: float
    prompt: str


class BatchGenerateRequest(BaseModel):
    """Batch generation from a distributional config. k is inside config."""
    config: dict[str, Any]


class BatchGenerateResponse(BaseModel):
    """Response after batch generation is enqueued.

    Each job in jobs[] has its own stream_url. No multiplexed stream.
    """
    batch_id: str
    k: int
    jobs: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

class ResolutionRequest(BaseModel):
    """Compute (W, H) from anchor + aspect."""
    anchor_pixels: int = 1048576
    aspect_ratio: float = 1.0


class ResolutionResponse(BaseModel):
    """Resolved (W, H) pair."""
    width: int
    height: int
    actual_pixels: int
    anchor_label: str


# ---------------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------------

class GalleryEntry(BaseModel):
    """One entry in the gallery."""
    id: str
    prompt: str
    seed: int
    width: int
    height: int
    n_steps: int
    cfg: float
    attention_backend: str
    elapsed_s: float
    timestamp: float
    image_url: str
    denoise: float = 1.0
    source_id: str | None = None
    batch_id: str | None = None
    batch_index: int | None = None
    resolved_config: dict[str, Any] | None = None


class GalleryListResponse(BaseModel):
    """List of gallery entries."""
    entries: list[GalleryEntry]
    total: int


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

class ServerStatusResponse(BaseModel):
    """Proxied server status + BFF metadata."""
    connected: bool = False
    loaded_models: list[str] = Field(default_factory=list)
    vram_allocated_gb: float = 0.0
    vram_reserved_gb: float = 0.0
    vram_total_gb: float = 0.0
    phase: str | None = None
    server_version: str = ""
    bff_version: str = "yeetums-v1"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class DefaultConfigResponse(BaseModel):
    """Default generation config with distribution annotations."""
    config: dict[str, Any]


class ConfigVolumesRequest(BaseModel):
    """Request for config space volume computation."""
    config: dict[str, Any]
    k: int = 1


class ConfigVolumesResponse(BaseModel):
    """Per-field volume info for distributional config fields."""
    volumes: list[dict[str, Any]]
    total_log_volume: float


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TrainingRunRequest(BaseModel):
    """Request to start a training run."""
    config: dict[str, Any]
    run_type: str = "btrm"  # "btrm" | "ddgrpo"


class TrainingStatusResponse(BaseModel):
    """Current training run status."""
    run_id: str | None = None
    active: bool = False
    step: int = 0
    n_steps: int = 0
    phase: str = "idle"
    loss: float | None = None
    accuracy: dict[str, float] | None = None
    elapsed_s: float = 0.0


class ValidationRequest(BaseModel):
    """Request for on-demand validation."""
    challenge_type: str = "pinkify"  # "pinkify" | "tnt" | "decorrelation"


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

class AdapterAllocateRequest(BaseModel):
    """Allocate a LoRA adapter."""
    name: str
    rank: int = 8
    alpha: float = 16.0
    layer_indices: list[int] | None = None


class AdapterInitRequest(BaseModel):
    """Initialize adapter weights."""
    name: str
    init_b_std: float = 0.0
    scale: float = 1.0


class AdapterConfigRequest(BaseModel):
    """Set adapter config."""
    name: str
    scale: float | None = None
    frozen: bool | None = None


class BTRMHeadRequest(BaseModel):
    """Inject BTRM scoring head."""
    head_names: list[str]
    logit_cap: float = 10.0
    lr: float | None = None
    hidden_dim: int = 3840

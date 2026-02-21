"""Pydantic models for FastAPI server request/response types.

All RPC endpoints use JSON request bodies and JSON responses.
Tensor payloads are transferred as base64-encoded safetensors bytes
in the JSON body, or via multipart form uploads for large payloads.

Import constraints:
  - pydantic for validation
  - Does NOT import torch, futudiffu, or any GPU code
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Common response envelope
# ---------------------------------------------------------------------------

class ServerResponse(BaseModel):
    """Standard response envelope for all endpoints."""
    status: str = "ok"
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    elapsed_s: float | None = None


# ---------------------------------------------------------------------------
# Status / lifecycle
# ---------------------------------------------------------------------------

class StatusResponse(BaseModel):
    """Response from /status."""
    status: str = "ok"
    loaded_models: list[str] = Field(default_factory=list)
    phase: str | None = None
    vram_allocated_gb: float = 0.0
    vram_reserved_gb: float = 0.0
    vram_total_gb: float = 0.0
    sage_configured: bool = False
    server_version: str = "fastapi-v1"


class FreeRequest(BaseModel):
    """Request to free model(s) from VRAM."""
    model: str = "all"  # "all", "te", "diffusion", "vae"


# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------

class EncodePromptRequest(BaseModel):
    """Request to encode a text prompt."""
    prompt: str
    layer_idx: int = -2


# ---------------------------------------------------------------------------
# Sampling (trajectory generation)
# ---------------------------------------------------------------------------

class SampleTrajectoryRequest(BaseModel):
    """Request for a single diffusion sampling trajectory."""
    seed: int
    n_steps: int
    cfg: float = 4.0
    width: int = 1280
    height: int = 832
    attention_backend: str = "sdpa"
    sampling_shift: float = 1.0
    multiplier: float = 1.0
    save_steps: list[int] | None = None
    denoise: float = 1.0
    score_at_step: int | None = None


class SampleTrajectoryPackedRequest(BaseModel):
    """Request for N packed diffusion trajectories."""
    n_images: int
    seeds: list[int]
    n_steps: int
    cfg: float = 4.0
    width: int | None = None
    height: int | None = None
    widths: list[int] | None = None
    heights: list[int] | None = None
    attention_backend: str = "sdpa"
    sampling_shift: float | None = None
    sampling_shifts: list[float] | None = None
    multiplier: float = 1.0
    save_steps: list[int] | None = None
    denoise: float = 1.0


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

class WarmupRequest(BaseModel):
    """Request for model warmup."""
    attention_backend: str = "sdpa"
    width: int = 1280
    height: int = 832


class WarmupPackedRequest(BaseModel):
    """Request for packed forward warmup."""
    n_images: int = 2


# ---------------------------------------------------------------------------
# LoRA adapter management
# ---------------------------------------------------------------------------

class AllocateAdapterRequest(BaseModel):
    """Allocate adapter slots (graph-mutating, no recompile)."""
    adapter_name: str
    rank: int = 8
    alpha: float = 16.0
    layer_indices: list[int] | None = None


class InitAdapterWeightsRequest(BaseModel):
    """Initialize adapter weights (graph-invariant, safe after compile)."""
    adapter_name: str
    init_b_std: float = 0.0
    scale: float = 1.0


class InjectLoraRequest(BaseModel):
    """Legacy: allocate + init + recompile in one call."""
    adapter_name: str
    rank: int = 8
    alpha: float = 16.0
    layer_indices: list[int] | None = None
    init_b_std: float = 0.0


class SetAdapterConfigRequest(BaseModel):
    """Set adapter scale and/or freeze state."""
    adapter_name: str
    scale: float | list[float] | None = None
    frozen: bool | None = None
    clear_scale: bool = False


class GetLoraStateDictRequest(BaseModel):
    """Get current LoRA weights."""
    adapter_name: str | None = None


class DumpAllLorasRequest(BaseModel):
    """Emergency dump all LoRA adapters to disk."""
    output_dir: str = "lora_dumps"


# ---------------------------------------------------------------------------
# BTRM (reward model)
# ---------------------------------------------------------------------------

class InjectBTRMHeadRequest(BaseModel):
    """Create BTRM scoring head on server."""
    hidden_dim: int = 3840
    head_names: list[str] = Field(default=["bit_quality", "step_quality"])
    logit_cap: float = 10.0
    lr: float | None = None
    weight_decay: float = 0.0


class ScoreBTRMRequest(BaseModel):
    """Score latents via backbone + BTRM head."""
    attention_backend: str = "sdpa"
    multiplier: float = 1.0


class TrainBTRMStepRequest(BaseModel):
    """One BTRM optimizer step from labeled examples."""
    labels: list[dict[str, Any]]
    logsquare_weight: float = 0.1
    attention_backend: str = "sdpa"
    multiplier: float = 1.0


# ---------------------------------------------------------------------------
# Policy optimization
# ---------------------------------------------------------------------------

class AccumulatePolicyGradientsRequest(BaseModel):
    """Accumulate REINFORCE gradients on server-side LoRA params."""
    adapter_name: str
    sparse_steps: list[int]
    advantage: float
    multiplier: float = 1.0


class PolicyOptimizerStepRequest(BaseModel):
    """Clip gradients, step policy optimizer, zero gradients."""
    adapter_name: str
    max_grad_norm: float = 1.0
    lr: float = 1e-4

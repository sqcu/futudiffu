"""FastAPI inference server: boring HTTP transport over src_ii/ modules.

Replaces the ZMQ server (src/futudiffu/server.py) with standard HTTP.
All model lifecycle, inference, and training logic lives in src_ii/ modules.
This file handles HTTP routing, serialization, timeouts, and error handling.

Design principles:
  - Boring transport: JSON in, JSON out. Tensors via safetensors multipart.
  - No silent deadlocks: every request has a timeout; timeouts produce HTTP
    errors, not stuck sockets. The server is killable and restartable without
    client-side state corruption.
  - Model lifecycle is not the server's job: it calls src_ii/ modules for
    everything algorithmic.
  - Testable without GPU: all routes can be tested with a mock ModelBackend.

Lifecycle axes (from user_dataflow_and_lifecycle_rollup.md):
  1. Outer loop topology: server exposes endpoints for both denoising rollouts
     and BTRM training, but does not own the outer loop. Clients orchestrate.
  2. Trajectory persistence: rollout endpoints return tensors; clients persist.
     The server also saves intermediates to disk on request (dump_all_loras).
  3. Side-channel observability: hidden state extraction via score_btrm.
  4. Weight mutability: LoRA inject/update/config RPCs mutate weights in-place.
     BTRM head injection + training mutate head + adapter weights.
  5. Optimizer state residency: BTRM optimizer + policy optimizers live on
     GPU, co-resident with weights. Exposed via train_btrm_step /
     policy_optimizer_step / accumulate_policy_gradients.
  6. Activation checkpointing: model.gradient_checkpointing attribute. Server
     passes gradient_checkpointing params through.
  7. Rollout-training coupling: the SAME server process handles both rollout
     generation (sample_trajectory) and training (train_btrm_step,
     accumulate_policy_gradients). Shared model state.
  8. Kernel compilation lifecycle: warmup endpoints trigger torch.compile
     + initial kernel compilation. Shape changes trigger recompilation.
     Server does NOT own compile logic; ModelBackend does.
  9. Multi-instance coherence: single-GPU server. Multi-GPU via separate
     server processes on different ports (as with ZMQ multi_gpu_client).
  10. Sequence packing: sample_trajectory_packed uses FlexAttention batch
      packing via src_ii/bin_packer.py.

Import constraints:
  - FastAPI, Starlette for HTTP
  - Pydantic for request/response models
  - src_ii.server_models for shared types
  - DOES NOT import torch at module level (lazy import for GPU operations)
  - DOES NOT import from src.futudiffu (the frozen v1 codebase)
"""

from __future__ import annotations

import base64
import functools
import io
import logging
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, Protocol, runtime_checkable

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse

from src_ii.server_models import (
    AccumulatePolicyGradientsRequest,
    AllocateAdapterRequest,
    DumpAllLorasRequest,
    EncodePromptRequest,
    FreeRequest,
    GetLoraStateDictRequest,
    InitAdapterWeightsRequest,
    InjectBTRMHeadRequest,
    InjectLoraRequest,
    PolicyOptimizerStepRequest,
    SampleTrajectoryPackedRequest,
    SampleTrajectoryRequest,
    ScoreBTRMRequest,
    ServerResponse,
    SetAdapterConfigRequest,
    StatusResponse,
    TrainBTRMStepRequest,
    WarmupPackedRequest,
    WarmupRequest,
)

logger = logging.getLogger("futudiffu.server")


# ---------------------------------------------------------------------------
# Model backend protocol (for dependency injection / testing)
# ---------------------------------------------------------------------------

@runtime_checkable
class ModelBackend(Protocol):
    """Protocol that any model backend must implement.

    The real backend (GPUModelBackend) wraps src_ii zimage_model, rollout,
    btrm_lifecycle, etc. A mock backend can be injected for testing.

    Methods correspond 1:1 to the server's RPC handlers.
    """

    def get_status(self) -> dict[str, Any]: ...
    def free(self, model: str) -> None: ...

    # Text encoding
    def encode_prompt(self, prompt: str, layer_idx: int) -> dict: ...

    # Sampling
    def sample_trajectory_packed(self, params: dict, tensors: dict) -> tuple[dict, dict]: ...

    # VAE
    def vae_encode(self, image_bytes: bytes) -> dict: ...
    def vae_decode(self, latent_bytes: bytes) -> dict: ...
    def vae_decode_png(self, latent_bytes: bytes) -> bytes: ...
    def vae_encode_png(self, png_bytes: bytes) -> bytes: ...

    # Warmup
    def warmup(self, attention_backend: str, width: int, height: int) -> None: ...
    def warmup_packed(self, n_images: int) -> None: ...

    # LoRA
    def allocate_adapter(self, params: dict) -> dict: ...
    def init_adapter_weights(self, params: dict) -> dict: ...
    def inject_lora(self, params: dict) -> dict: ...
    def update_lora_weights(self, tensor_bytes: bytes) -> dict: ...
    def set_adapter_config(self, params: dict) -> dict: ...
    def get_lora_state_dict(self, adapter_name: str | None) -> dict: ...
    def dump_all_loras(self, output_dir: str) -> dict: ...

    # BTRM
    def inject_btrm_head(self, params: dict) -> dict: ...
    def score_btrm(self, params: dict, tensor_bytes: bytes) -> dict: ...
    def train_btrm_step(self, params: dict, tensor_bytes: bytes) -> dict: ...

    # Policy
    def accumulate_policy_gradients(self, params: dict, tensor_bytes: bytes) -> dict: ...
    def policy_optimizer_step(self, params: dict) -> dict: ...

    # Batch forward (fork-and-mutate)
    def batch_forward(self, params: dict, tensor_bytes: bytes) -> tuple[dict, dict]: ...


# ---------------------------------------------------------------------------
# Tensor serialization helpers
# ---------------------------------------------------------------------------

def _tensors_to_safetensors_bytes(tensors: dict) -> bytes:
    """Serialize a dict of torch tensors to safetensors bytes."""
    from safetensors.torch import save as st_save
    return st_save(tensors)


def _safetensors_bytes_to_tensors(data: bytes) -> dict:
    """Deserialize safetensors bytes to a dict of torch tensors."""
    from safetensors.torch import load as st_load
    return st_load(data)


def _tensors_to_b64(tensors: dict) -> dict[str, str]:
    """Encode tensor dict as {name: base64_safetensors_string}.

    For JSON transport, each tensor is individually safetensors-encoded
    then base64-encoded. This avoids the need for multipart transport
    for small tensors (conditioning, sigmas, etc).
    """
    result = {}
    for name, tensor in tensors.items():
        st_bytes = _tensors_to_safetensors_bytes({name: tensor})
        result[name] = base64.b64encode(st_bytes).decode("ascii")
    return result


def _b64_to_tensors(encoded: dict[str, str]) -> dict:
    """Decode {name: base64_safetensors_string} to tensor dict."""
    import torch
    result = {}
    for name, b64_str in encoded.items():
        st_bytes = base64.b64decode(b64_str)
        loaded = _safetensors_bytes_to_tensors(st_bytes)
        # The tensor was stored under its own name
        if name in loaded:
            result[name] = loaded[name]
        else:
            # Fall back to first tensor in the safetensors
            for k, v in loaded.items():
                result[name] = v
                break
    return result


# ---------------------------------------------------------------------------
# Request timeout middleware
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT_S = 600  # 10 minutes -- generous for torch.compile warmups


async def timeout_middleware(request: Request, call_next):
    """Add server-side timeout to all requests.

    If a request takes longer than the configured timeout, the response
    will be a 504 Gateway Timeout. The handler coroutine is cancelled.
    This prevents silent hangs.
    """
    import asyncio
    timeout_s = request.app.state.request_timeout_s

    try:
        response = await asyncio.wait_for(
            call_next(request),
            timeout=timeout_s,
        )
        return response
    except asyncio.TimeoutError:
        method = request.url.path
        logger.error(f"Request timed out after {timeout_s}s: {method}")
        return JSONResponse(
            status_code=504,
            content={
                "status": "error",
                "error": f"Request timed out after {timeout_s}s",
                "metadata": {},
            },
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

async def error_handler(request: Request, exc: Exception):
    """Global exception handler: all unhandled exceptions become 500 JSON."""
    tb = traceback.format_exc()
    logger.error(f"Unhandled exception on {request.url.path}: {tb}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": str(exc),
            "traceback": tb,
            "metadata": {},
        },
    )


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app(
    backend: ModelBackend,
    request_timeout_s: float = DEFAULT_TIMEOUT_S,
    enable_queue: bool = True,
    batch_window_ms: float = 100,
    max_batch: int = 16,
) -> FastAPI:
    """Create the FastAPI application with all routes.

    Args:
        backend: Model backend (real GPU or mock).
        request_timeout_s: Per-request timeout in seconds.
        enable_queue: Whether to start the inference queue worker.
        batch_window_ms: Queue batch accumulation window in ms.
        max_batch: Maximum jobs per batch.

    Returns:
        Configured FastAPI application.
    """
    inference_queue = None

    from src_ii.training_orchestrator import TrainingOrchestrator
    training_orchestrator = TrainingOrchestrator()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal inference_queue
        logger.info("futudiffu FastAPI server starting")

        if enable_queue and type(backend).__name__ == "GPUModelBackend":
            from src_ii.inference_queue import InferenceQueue
            inference_queue = InferenceQueue(
                backend,
                batch_window_ms=batch_window_ms,
                max_batch=max_batch,
            )
            await inference_queue.start()
            app.state.inference_queue = inference_queue

        app.state.training_orchestrator = training_orchestrator

        yield

        if inference_queue is not None:
            await inference_queue.stop()

        logger.info("futudiffu FastAPI server shutting down")
        try:
            backend.free("all")
        except Exception:
            pass

    app = FastAPI(
        title="futudiffu inference server",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.backend = backend
    app.state.request_timeout_s = request_timeout_s

    # Middleware
    app.middleware("http")(timeout_middleware)

    # Global error handler
    app.add_exception_handler(Exception, error_handler)

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    def _timed(fn_name: str):
        """Decorator factory for timing + logging RPC handlers."""
        def decorator(fn):
            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    result = await fn(*args, **kwargs)
                    elapsed = time.perf_counter() - t0
                    logger.info(f"[{fn_name}] {elapsed:.2f}s")
                    return result
                except HTTPException:
                    raise
                except Exception as e:
                    elapsed = time.perf_counter() - t0
                    tb = traceback.format_exc()
                    logger.error(f"[{fn_name}] failed after {elapsed:.2f}s: {e}")
                    return JSONResponse(
                        status_code=500,
                        content={
                            "status": "error",
                            "error": str(e),
                            "traceback": tb,
                            "metadata": {},
                        },
                    )
            return wrapper
        return decorator

    # --- Status / lifecycle ---

    @app.get("/status", response_model=StatusResponse)
    @_timed("status")
    async def get_status():
        info = backend.get_status()
        return StatusResponse(
            loaded_models=info.get("loaded_models", []),
            phase=info.get("phase"),
            vram_allocated_gb=info.get("vram_allocated_gb", 0.0),
            vram_reserved_gb=info.get("vram_reserved_gb", 0.0),
            vram_total_gb=info.get("vram_total_gb", 0.0),
            sage_configured=info.get("sage_configured", False),
        )

    @app.post("/free", response_model=ServerResponse)
    @_timed("free")
    async def free_model(req: FreeRequest):
        backend.free(req.model)
        return ServerResponse(metadata={"freed": req.model})

    # --- Text encoding ---

    @app.post("/encode_prompt")
    @_timed("encode_prompt")
    async def encode_prompt(req: EncodePromptRequest):
        result = backend.encode_prompt(req.prompt, req.layer_idx)
        # Return conditioning tensor as safetensors bytes
        st_bytes = _tensors_to_safetensors_bytes(result)
        return StreamingResponse(
            io.BytesIO(st_bytes),
            media_type="application/octet-stream",
            headers={"X-Tensor-Format": "safetensors"},
        )

    # --- Sampling ---

    @app.post("/sample_trajectory")
    @_timed("sample_trajectory")
    async def sample_trajectory(request: Request):
        """Run a diffusion sampling trajectory.

        Accepts multipart form: JSON params + safetensors tensor payloads.
        Returns safetensors bytes with all result tensors.
        """
        content_type = request.headers.get("content-type", "")

        if "multipart" in content_type:
            form = await request.form()
            params = await _parse_json_field(form.get("params", "{}"))
            tensor_bytes = await _read_upload(form.get("tensors"))
            tensors = _safetensors_bytes_to_tensors(tensor_bytes) if tensor_bytes else {}
        else:
            body = await request.json()
            params = body.get("params", {})
            tensors = _b64_to_tensors(body.get("tensors", {}))

        result_tensors, metadata = backend.sample_trajectory(params, tensors)
        st_bytes = _tensors_to_safetensors_bytes(result_tensors)

        return StreamingResponse(
            io.BytesIO(st_bytes),
            media_type="application/octet-stream",
            headers={
                "X-Tensor-Format": "safetensors",
                "X-Metadata": _json_encode(metadata),
            },
        )

    @app.post("/sample_trajectory_relay")
    @_timed("sample_trajectory_relay")
    async def sample_trajectory_relay(request: Request):
        """Relay endpoint for torch-free clients (yeetums BFF).

        Accepts separate safetensors files for pos_cond, neg_cond, and
        optionally clean_latent. Merges them server-side and delegates
        to sample_trajectory.

        Multipart form fields:
          - params: JSON sampling parameters
          - pos_cond_st: safetensors with "conditioning" key (positive)
          - neg_cond_st: safetensors with "conditioning" key (negative)
          - clean_latent_st: optional safetensors with "latent" key (i2i)
        """
        form = await request.form()
        params = await _parse_json_field(form.get("params", "{}"))

        pos_bytes = await _read_upload(form.get("pos_cond_st"))
        neg_bytes = await _read_upload(form.get("neg_cond_st"))
        clean_bytes = await _read_upload(form.get("clean_latent_st"))

        pos_tensors = _safetensors_bytes_to_tensors(pos_bytes) if pos_bytes else {}
        neg_tensors = _safetensors_bytes_to_tensors(neg_bytes) if neg_bytes else {}

        tensors = {
            "pos_cond": pos_tensors.get("conditioning", next(iter(pos_tensors.values()))),
            "neg_cond": neg_tensors.get("conditioning", next(iter(neg_tensors.values()))),
        }

        if clean_bytes:
            clean_tensors = _safetensors_bytes_to_tensors(clean_bytes)
            tensors["clean_latent"] = clean_tensors.get(
                "latent", next(iter(clean_tensors.values()))
            )

        result_tensors, metadata = backend.sample_trajectory(params, tensors)

        # Return only the final latent as safetensors
        final = result_tensors.get("final")
        if final is not None:
            out = {"latent": final}
        else:
            out = result_tensors

        st_bytes = _tensors_to_safetensors_bytes(out)
        return StreamingResponse(
            io.BytesIO(st_bytes),
            media_type="application/octet-stream",
            headers={
                "X-Tensor-Format": "safetensors",
                "X-Metadata": _json_encode(metadata),
            },
        )

    @app.post("/sample_trajectory_packed")
    @_timed("sample_trajectory_packed")
    async def sample_trajectory_packed(request: Request):
        """Run N packed diffusion trajectories.

        Same multipart/JSON interface as sample_trajectory.
        """
        content_type = request.headers.get("content-type", "")

        if "multipart" in content_type:
            form = await request.form()
            params = await _parse_json_field(form.get("params", "{}"))
            tensor_bytes = await _read_upload(form.get("tensors"))
            tensors = _safetensors_bytes_to_tensors(tensor_bytes) if tensor_bytes else {}
        else:
            body = await request.json()
            params = body.get("params", {})
            tensors = _b64_to_tensors(body.get("tensors", {}))

        result_tensors, metadata = backend.sample_trajectory_packed(params, tensors)
        st_bytes = _tensors_to_safetensors_bytes(result_tensors)

        return StreamingResponse(
            io.BytesIO(st_bytes),
            media_type="application/octet-stream",
            headers={
                "X-Tensor-Format": "safetensors",
                "X-Metadata": _json_encode(metadata),
            },
        )

    # --- VAE ---

    @app.post("/vae_encode")
    @_timed("vae_encode")
    async def vae_encode(request: Request):
        """Encode image tensor to latent. Accepts safetensors body."""
        body = await request.body()
        result = backend.vae_encode(body)
        st_bytes = _tensors_to_safetensors_bytes(result)
        return StreamingResponse(
            io.BytesIO(st_bytes),
            media_type="application/octet-stream",
            headers={"X-Tensor-Format": "safetensors"},
        )

    @app.post("/vae_decode")
    @_timed("vae_decode")
    async def vae_decode(request: Request):
        """Decode latent to image tensor. Accepts safetensors body."""
        body = await request.body()
        result = backend.vae_decode(body)
        st_bytes = _tensors_to_safetensors_bytes(result)
        return StreamingResponse(
            io.BytesIO(st_bytes),
            media_type="application/octet-stream",
            headers={"X-Tensor-Format": "safetensors"},
        )

    @app.post("/vae_decode_png")
    @_timed("vae_decode_png")
    async def vae_decode_png(request: Request):
        """Decode latent to PNG. Accepts safetensors body, returns PNG bytes."""
        body = await request.body()
        png_bytes = backend.vae_decode_png(body)
        return StreamingResponse(
            io.BytesIO(png_bytes),
            media_type="image/png",
        )

    @app.post("/vae_encode_png")
    @_timed("vae_encode_png")
    async def vae_encode_png(request: Request):
        """Encode PNG to latent. Accepts PNG bytes, returns safetensors."""
        body = await request.body()
        st_bytes = backend.vae_encode_png(body)
        return StreamingResponse(
            io.BytesIO(st_bytes),
            media_type="application/octet-stream",
            headers={"X-Tensor-Format": "safetensors"},
        )

    # --- Warmup ---

    @app.post("/warmup", response_model=ServerResponse)
    @_timed("warmup")
    async def warmup(req: WarmupRequest):
        backend.warmup(req.attention_backend, req.width, req.height)
        return ServerResponse(metadata={"attention_backend": req.attention_backend})

    @app.post("/warmup_packed", response_model=ServerResponse)
    @_timed("warmup_packed")
    async def warmup_packed(req: WarmupPackedRequest):
        backend.warmup_packed(req.n_images)
        return ServerResponse(metadata={"n_images": req.n_images})

    # --- LoRA adapter management ---

    @app.post("/allocate_adapter", response_model=ServerResponse)
    @_timed("allocate_adapter")
    async def allocate_adapter(req: AllocateAdapterRequest):
        result = backend.allocate_adapter(req.model_dump())
        return ServerResponse(metadata=result)

    @app.post("/init_adapter_weights", response_model=ServerResponse)
    @_timed("init_adapter_weights")
    async def init_adapter_weights(req: InitAdapterWeightsRequest):
        result = backend.init_adapter_weights(req.model_dump())
        return ServerResponse(metadata=result)

    @app.post("/inject_lora", response_model=ServerResponse)
    @_timed("inject_lora")
    async def inject_lora(req: InjectLoraRequest):
        result = backend.inject_lora(req.model_dump())
        return ServerResponse(metadata=result)

    @app.post("/update_lora_weights")
    @_timed("update_lora_weights")
    async def update_lora_weights(request: Request):
        """Update LoRA weights in-place. Accepts safetensors body."""
        body = await request.body()
        result = backend.update_lora_weights(body)
        return ServerResponse(metadata=result)

    @app.post("/set_adapter_config", response_model=ServerResponse)
    @_timed("set_adapter_config")
    async def set_adapter_config(req: SetAdapterConfigRequest):
        result = backend.set_adapter_config(req.model_dump())
        return ServerResponse(metadata=result)

    @app.post("/get_lora_state_dict")
    @_timed("get_lora_state_dict")
    async def get_lora_state_dict(req: GetLoraStateDictRequest):
        """Get current LoRA weights as safetensors bytes."""
        result = backend.get_lora_state_dict(req.adapter_name)
        st_bytes = _tensors_to_safetensors_bytes(result)
        return StreamingResponse(
            io.BytesIO(st_bytes),
            media_type="application/octet-stream",
            headers={"X-Tensor-Format": "safetensors"},
        )

    @app.post("/dump_all_loras", response_model=ServerResponse)
    @_timed("dump_all_loras")
    async def dump_all_loras(req: DumpAllLorasRequest):
        result = backend.dump_all_loras(req.output_dir)
        return ServerResponse(metadata=result)

    # --- BTRM ---

    @app.post("/inject_btrm_head", response_model=ServerResponse)
    @_timed("inject_btrm_head")
    async def inject_btrm_head(req: InjectBTRMHeadRequest):
        result = backend.inject_btrm_head(req.model_dump())
        return ServerResponse(metadata=result)

    @app.post("/score_btrm")
    @_timed("score_btrm")
    async def score_btrm(request: Request):
        """Score latents via BTRM head. Multipart or JSON+b64 tensors."""
        content_type = request.headers.get("content-type", "")
        if "multipart" in content_type:
            form = await request.form()
            params = await _parse_json_field(form.get("params", "{}"))
            tensor_bytes = await _read_upload(form.get("tensors"))
        else:
            body = await request.json()
            params = body.get("params", {})
            tensor_bytes = base64.b64decode(body["tensors"]) if "tensors" in body else b""
        result = backend.score_btrm(params, tensor_bytes)
        return JSONResponse(content={"status": "ok", "metadata": result})

    @app.post("/train_btrm_step")
    @_timed("train_btrm_step")
    async def train_btrm_step(request: Request):
        """One BTRM optimizer step. Multipart or JSON+b64 tensors."""
        content_type = request.headers.get("content-type", "")
        if "multipart" in content_type:
            form = await request.form()
            params = await _parse_json_field(form.get("params", "{}"))
            tensor_bytes = await _read_upload(form.get("tensors"))
        else:
            body = await request.json()
            params = body.get("params", {})
            tensor_bytes = base64.b64decode(body["tensors"]) if "tensors" in body else b""
        result = backend.train_btrm_step(params, tensor_bytes)
        return JSONResponse(content={"status": "ok", "metadata": result})

    # --- Policy ---

    @app.post("/accumulate_policy_gradients")
    @_timed("accumulate_policy_gradients")
    async def accumulate_policy_gradients(request: Request):
        """Accumulate REINFORCE gradients. Multipart or JSON+b64 tensors."""
        content_type = request.headers.get("content-type", "")
        if "multipart" in content_type:
            form = await request.form()
            params = await _parse_json_field(form.get("params", "{}"))
            tensor_bytes = await _read_upload(form.get("tensors"))
        else:
            body = await request.json()
            params = body.get("params", {})
            tensor_bytes = base64.b64decode(body["tensors"]) if "tensors" in body else b""
        result = backend.accumulate_policy_gradients(params, tensor_bytes)
        return JSONResponse(content={"status": "ok", "metadata": result})

    @app.post("/policy_optimizer_step", response_model=ServerResponse)
    @_timed("policy_optimizer_step")
    async def policy_optimizer_step(req: PolicyOptimizerStepRequest):
        result = backend.policy_optimizer_step(req.model_dump())
        return ServerResponse(metadata=result)

    # --- Batch forward (fork-and-mutate, SCATTER->PACKSOLVE->EXECUTE->DENOISE) ---

    @app.post("/batch_forward")
    @_timed("batch_forward")
    async def batch_forward(request: Request):
        """Execute N fork specs through the model, return tagged (field, score) results.

        Accepts multipart or JSON+b64 encoding.
        params: {queries: [...]}  -- see BatchExecutor.execute() for query schema
        tensors: safetensors blob with all literal tensors keyed by name

        Response: safetensors bytes with flattened result tensors + X-Metadata header
        with entry tags [{query_id, entry_id, denoised_key, score_key}, ...].
        """
        content_type = request.headers.get("content-type", "")

        if "multipart" in content_type:
            form = await request.form()
            params = await _parse_json_field(form.get("params", "{}"))
            tensor_bytes = await _read_upload(form.get("tensors"))
            tensors = _safetensors_bytes_to_tensors(tensor_bytes) if tensor_bytes else {}
        else:
            body = await request.json()
            params = body.get("params", {})
            tensors = _b64_to_tensors(body.get("tensors", {}))

        result_tensors, metadata = backend.batch_forward(params, tensors)
        st_bytes = _tensors_to_safetensors_bytes(result_tensors)

        return StreamingResponse(
            io.BytesIO(st_bytes),
            media_type="application/octet-stream",
            headers={
                "X-Tensor-Format": "safetensors",
                "X-Metadata": _json_encode(metadata),
            },
        )

    # --- Training orchestration ---

    @app.post("/training/start")
    @_timed("training_start")
    async def training_start(request: Request):
        """Start a background training run. Returns {run_id, stream_url}."""
        body = await request.json()
        run_type = body.pop("run_type", "btrm")

        if run_type == "btrm":
            result = training_orchestrator.start_btrm_run(body, backend)
        elif run_type == "ddgrpo":
            result = training_orchestrator.start_ddgrpo_run(body, backend)
        elif run_type == "policy_intervention":
            result = training_orchestrator.start_policy_intervention_run(body, backend)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown run_type: {run_type!r}. Valid: btrm, ddgrpo, policy_intervention",
            )
        return result

    @app.get("/training/status")
    async def training_status():
        """Get current training run status."""
        return training_orchestrator.get_status()

    @app.post("/training/stop")
    @_timed("training_stop")
    async def training_stop(request: Request):
        """Stop the active training run."""
        training_orchestrator.stop()
        return {"stopped": True}

    @app.get("/training/stream/{run_id}")
    async def training_stream(run_id: str):
        """SSE event stream for a training run."""
        if training_orchestrator._run_id != run_id:
            raise HTTPException(status_code=404, detail="Run not found")

        queue = training_orchestrator.subscribe()

        import asyncio as _asyncio
        import json as _json

        async def event_gen():
            while True:
                try:
                    event = await _asyncio.wait_for(queue.get(), timeout=30.0)
                except _asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue

                event_type = event.get("type", "message")
                data = _json.dumps(event.get("data", {}))
                yield f"event: {event_type}\ndata: {data}\n\n"

                if event_type in ("complete", "error"):
                    break

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/training/artifacts/{run_id}/{path:path}")
    async def training_artifacts(run_id: str, path: str):
        """Serve training artifacts (charts, metrics, checkpoints)."""
        artifact_path = training_orchestrator.get_artifact_path(run_id, path)
        if artifact_path is None:
            raise HTTPException(status_code=404, detail="Artifact not found")

        # Infer content type from extension
        suffix = artifact_path.suffix.lower()
        content_types = {
            ".png": "image/png",
            ".json": "application/json",
            ".jsonl": "application/x-ndjson",
            ".md": "text/markdown",
            ".safetensors": "application/octet-stream",
        }
        ct = content_types.get(suffix, "application/octet-stream")

        from fastapi.responses import FileResponse
        return FileResponse(str(artifact_path), media_type=ct)

    @app.post("/training/validate")
    @_timed("training_validate")
    async def training_validate(request: Request):
        """Run an on-demand validation challenge against current model state."""
        body = await request.json()
        challenge_type = body.get("challenge_type", "pinkify")
        result = training_orchestrator.run_validation(challenge_type, backend)
        return result

    # --- Inference queue (enqueue + SSE stream) ---

    @app.post("/enqueue")
    @_timed("enqueue")
    async def enqueue_job(request: Request):
        """Enqueue a generation job. Returns job_id immediately."""
        if not hasattr(app.state, 'inference_queue') or app.state.inference_queue is None:
            raise HTTPException(
                status_code=503,
                detail="Inference queue not enabled",
            )
        body = await request.json()

        from src_ii.inference_queue import InferenceJob
        job = InferenceJob(
            prompt=body.get("prompt", ""),
            negative_prompt=body.get("negative_prompt", ""),
            seed=body.get("seed", 42),
            n_steps=body.get("n_steps", 30),
            cfg=body.get("cfg", 4.0),
            width=body.get("width", 1280),
            height=body.get("height", 832),
            sampling_shift=body.get("sampling_shift", 1.0),
            multiplier=body.get("multiplier", 1.0),
            denoise=body.get("denoise", 1.0),
            attention_backend=body.get("attention_backend", "sage"),
        )
        job_id = await app.state.inference_queue.enqueue(job)
        return {"job_id": job_id}

    @app.get("/stream/{job_id}")
    async def stream_job(job_id: str):
        """SSE event stream for a job."""
        if not hasattr(app.state, 'inference_queue') or app.state.inference_queue is None:
            raise HTTPException(status_code=503, detail="Queue not enabled")

        queue = app.state.inference_queue.subscribe(job_id)
        if queue is None:
            raise HTTPException(status_code=404, detail="Job not found")

        import asyncio as _asyncio
        import json as _json

        async def event_gen():
            while True:
                try:
                    event = await _asyncio.wait_for(queue.get(), timeout=30.0)
                except _asyncio.TimeoutError:
                    # Send keepalive comment
                    yield ": keepalive\n\n"
                    continue

                event_type = event.get("type", "message")
                data = _json.dumps(event.get("data", {}))
                yield f"event: {event_type}\ndata: {data}\n\n"

                if event_type in ("complete", "error"):
                    break

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/result/{job_id}")
    async def get_result(job_id: str):
        """Get the PNG result of a completed job."""
        if not hasattr(app.state, 'inference_queue') or app.state.inference_queue is None:
            raise HTTPException(status_code=503, detail="Queue not enabled")

        job = app.state.inference_queue.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != "complete":
            raise HTTPException(
                status_code=409,
                detail=f"Job not complete (status={job.status})",
            )
        if job.result_png is None:
            raise HTTPException(status_code=500, detail="No PNG result")

        return StreamingResponse(
            io.BytesIO(job.result_png),
            media_type="image/png",
            headers={
                "X-Job-Id": job.job_id,
                "X-Metadata": _json_encode(job.metadata or {}),
            },
        )

    @app.get("/queue_status")
    async def queue_status():
        """Get queue statistics."""
        if not hasattr(app.state, 'inference_queue') or app.state.inference_queue is None:
            return {"pending": 0, "processing": 0, "completed_last_min": 0, "enabled": False}
        stats = app.state.inference_queue.queue_status()
        stats["enabled"] = True
        return stats

    # --- Health check ---

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

import json as _json


def _json_encode(obj: Any) -> str:
    """JSON-encode for HTTP headers (no newlines)."""
    return _json.dumps(obj, separators=(",", ":"))


async def _parse_json_field(field) -> dict:
    """Parse a JSON string from a form field (str or UploadFile)."""
    if isinstance(field, str):
        return _json.loads(field)
    if hasattr(field, "read"):
        data = await field.read()
        if data:
            return _json.loads(data)
    return {}


async def _read_upload(field) -> bytes | None:
    """Read bytes from an UploadFile or return None."""
    if field is None:
        return None
    if hasattr(field, "read"):
        return await field.read()
    if isinstance(field, bytes):
        return field
    return None


# ---------------------------------------------------------------------------
# GPU Model Backend (the real implementation)
# ---------------------------------------------------------------------------

class GPUModelBackend:
    """Real GPU backend: wraps src_ii modules for model lifecycle + inference.

    This is the production backend. It imports torch and src_ii modules
    lazily to support the server module being importable without a GPU.
    """

    def __init__(
        self,
        fp8_diff_path: str,
        te_path: str,
        vae_path: str,
        tokenizer_path: str | None = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        fp8_block_size: int = 128,
    ):
        import torch
        self._device = torch.device(device)
        self._dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]

        # Store paths for lazy loading
        self._fp8_diff_path = fp8_diff_path
        self._te_path = te_path
        self._vae_path = vae_path
        self._tokenizer_path = tokenizer_path
        self._fp8_block_size = fp8_block_size

        # Model state (lazy-loaded)
        self._te_model = None
        self._tokenizer = None
        self._diff_model = None
        self._diff_compiled = None
        self._vae_model = None
        self._phase = None
        self._sage_configured = False

        # LoRA state
        self._lora_configs: list[dict] = []
        self._lora_weights: dict[str, dict] = {}
        self._lora_scales: dict[str, Any] = {}

        # BTRM state
        self._btrm_head = None
        self._btrm_optimizer = None
        self._btrm_config = None

        # Policy state
        self._policy_optimizers: dict = {}

        # Batch executor (lazy-initialized after diffusion model loads)
        self._batch_executor = None

    # --- Status ---

    def get_status(self) -> dict[str, Any]:
        import torch
        vram_allocated = torch.cuda.memory_allocated(self._device) / (1024**3)
        vram_reserved = torch.cuda.memory_reserved(self._device) / (1024**3)
        vram_total = torch.cuda.get_device_properties(
            self._device).total_memory / (1024**3)

        loaded = []
        if self._te_model is not None:
            loaded.append("te")
        if self._diff_model is not None:
            loaded.append("diffusion")
        if self._vae_model is not None:
            loaded.append("vae")

        return {
            "loaded_models": loaded,
            "phase": self._phase,
            "vram_allocated_gb": round(vram_allocated, 2),
            "vram_reserved_gb": round(vram_reserved, 2),
            "vram_total_gb": round(vram_total, 2),
            "sage_configured": self._sage_configured,
        }

    # --- Free ---

    def free(self, model: str) -> None:
        import torch
        if model == "all":
            self._free_te()
            self._free_diffusion()
            self._free_vae()
            self._phase = None
            torch.cuda.empty_cache()
        elif model == "te":
            self._free_te()
        elif model == "diffusion":
            self._free_diffusion()
        elif model == "vae":
            self._free_vae()
        else:
            raise ValueError(f"Unknown model: {model!r}. Valid: all, te, diffusion, vae")

    def _free_te(self):
        import torch
        if self._te_model is not None:
            del self._te_model
            self._te_model = None
            self._tokenizer = None
            torch.cuda.empty_cache()

    def _free_diffusion(self):
        import torch
        if self._diff_model is not None:
            # Snapshot LoRA weights before freeing
            self._snapshot_lora_weights()
            del self._diff_model, self._diff_compiled
            self._diff_model = None
            self._diff_compiled = None
            torch.cuda.empty_cache()

    def _free_vae(self):
        import torch
        if self._vae_model is not None:
            del self._vae_model
            self._vae_model = None
            torch.cuda.empty_cache()

    def _snapshot_lora_weights(self):
        if self._diff_model is not None and self._lora_configs:
            from futudiffu.lora import lora_state_dict
            for cfg in self._lora_configs:
                name = cfg["adapter_name"]
                sd = lora_state_dict(self._diff_model, adapter_name=name)
                self._lora_weights[name] = {
                    k: v.detach().cpu() for k, v in sd.items()
                }

    # --- Ensure (lazy load) ---

    def _ensure_te(self):
        if self._te_model is not None:
            return
        # Free diffusion to make room
        if self._diff_model is not None:
            self._snapshot_lora_weights()
            self._free_diffusion()

        from futudiffu.text_encoder import create_tokenizer, load_text_encoder
        import torch

        self._tokenizer = create_tokenizer(self._tokenizer_path)
        self._te_model = load_text_encoder(
            self._te_path, device=self._device, dtype=self._dtype
        )
        self._te_model = torch.compile(self._te_model, mode="default")
        self._phase = "te"
        logger.info(f"TE loaded ({self._dtype})")

    def _ensure_diffusion(self):
        if self._diff_model is not None:
            return
        self._free_te()

        from src_ii.zimage_model import load_zimage_rlaif
        import torch

        diff_model = load_zimage_rlaif(
            self._fp8_diff_path,
            device=self._device,
            dtype=self._dtype,
            fp8_block_size=self._fp8_block_size,
            compile_model=True,
            fuse=True,
            use_sage=True,
        )
        self._diff_model = diff_model
        self._diff_compiled = diff_model

        # Replay LoRA injections
        if self._lora_configs:
            from futudiffu.lora import (
                allocate_adapter, init_adapter_weights,
                load_lora_state_dict, set_lora_scale,
            )
            for cfg in self._lora_configs:
                allocate_adapter(
                    diff_model,
                    name=cfg["adapter_name"],
                    rank=cfg["rank"],
                    alpha=cfg["alpha"],
                    layer_indices=cfg.get("layer_indices"),
                )
            for adapter_name, weights in self._lora_weights.items():
                if weights:
                    sd = {k: v.to(dtype=self._dtype) for k, v in weights.items()}
                    load_lora_state_dict(diff_model, sd)
            for adapter_name, scale in self._lora_scales.items():
                if isinstance(scale, (int, float)):
                    scale_t = torch.tensor([scale], device=self._device, dtype=self._dtype)
                else:
                    scale_t = torch.tensor(scale, device=self._device, dtype=self._dtype)
                set_lora_scale(diff_model, scale_t, adapter_name=adapter_name)
            logger.info(f"Replayed {len(self._lora_configs)} LoRA injection(s)")

        self._phase = "diffusion"

    def _ensure_vae(self):
        if self._vae_model is not None:
            return
        from src_ii.vae_utils import load_vae
        self._vae_model = load_vae(
            self._vae_path, device=self._device, dtype=self._dtype
        )
        logger.info("VAE loaded")

    def _configure_sage_if_needed(self, attention_backend: str):
        if not self._sage_configured and attention_backend != "sdpa":
            try:
                from futudiffu.sage_attention import configure_sage
                configure_sage(smooth_k=True, qk_quant="int8", pv_quant="bf16")
            except ImportError:
                pass
            self._sage_configured = True

    # --- Text encoding ---

    def encode_prompt(self, prompt: str, layer_idx: int) -> dict:
        import torch
        self._ensure_te()
        from futudiffu.text_encoder import encode_prompt as _encode_prompt
        with torch.inference_mode():
            cond = _encode_prompt(
                self._te_model, self._tokenizer, prompt,
                device=self._device, layer_idx=layer_idx,
            )
        return {"conditioning": cond.cpu()}

    # --- Sampling ---

    def sample_trajectory(self, params: dict, tensors: dict) -> tuple[dict, dict]:
        """Single-image sampling -- normalizes to packed contract."""
        p = dict(params)
        t = dict(tensors)
        if "n_images" not in p:
            p["n_images"] = 1
        if "seeds" not in p and "seed" in p:
            p["seeds"] = [p.pop("seed")]
        if "pos_cond" in t and "pos_cond_0" not in t:
            t["pos_cond_0"] = t.pop("pos_cond")
        if "clean_latent" in t and "clean_latent_0" not in t:
            t["clean_latent_0"] = t.pop("clean_latent")

        result_tensors, metadata = self.sample_trajectory_packed(p, t)

        # Remap output: callers expect "final" not "final_0"
        if "final_0" in result_tensors and "final" not in result_tensors:
            result_tensors["final"] = result_tensors["final_0"]
        # Remap step intermediates: "step_XX_0" -> "step_XX"
        for k in list(result_tensors.keys()):
            if k.startswith("step_") and k.endswith("_0"):
                result_tensors[k[:-2]] = result_tensors[k]

        return result_tensors, metadata

    def sample_trajectory_packed(
        self, params: dict, tensors: dict, callback=None,
    ) -> tuple[dict, dict]:
        import torch
        self._ensure_diffusion()
        self._ensure_batch_executor()
        attn = params.get("attention_backend", "sdpa")
        self._configure_sage_if_needed(attn)

        from src_ii.inference_sampling import run_trajectory_packed

        with torch.inference_mode():
            result_tensors, metadata = run_trajectory_packed(
                self._diff_model,
                self._device, self._dtype, params, tensors,
                callback=callback,
                batch_executor=self._batch_executor,
            )
        return result_tensors, metadata

    # --- VAE ---

    def vae_encode(self, image_bytes: bytes) -> dict:
        import torch
        self._ensure_vae()
        from futudiffu.vae import vae_encode
        tensors = _safetensors_bytes_to_tensors(image_bytes)
        image = tensors["image"].to(device=self._device, dtype=self._dtype)
        with torch.inference_mode():
            latent = vae_encode(self._vae_model, image)
        return {"latent": latent.cpu()}

    def vae_decode(self, latent_bytes: bytes) -> dict:
        import torch
        self._ensure_vae()
        from futudiffu.vae import vae_decode
        tensors = _safetensors_bytes_to_tensors(latent_bytes)
        latent = tensors["latent"].to(device=self._device, dtype=self._dtype)
        with torch.inference_mode():
            image = vae_decode(self._vae_model, latent)
        return {"image": image.cpu()}

    def vae_decode_png(self, latent_bytes: bytes) -> bytes:
        """Decode latent to PNG bytes. Returns raw PNG."""
        result = self.vae_decode(latent_bytes)
        from src_ii.rendering import tensor_to_pil
        pil_img = tensor_to_pil(result["image"])
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()

    def vae_encode_png(self, png_bytes: bytes) -> bytes:
        """Encode PNG bytes to safetensors latent."""
        import torch
        import numpy as np
        from PIL import Image
        pil_img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        arr = np.array(pil_img, dtype="float32") / 255.0
        image = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
        st_bytes = _tensors_to_safetensors_bytes({"image": image})
        result = self.vae_encode(st_bytes)
        return _tensors_to_safetensors_bytes(result)

    # --- Warmup ---

    def warmup(self, attention_backend: str, width: int, height: int) -> None:
        self._ensure_diffusion()
        self._configure_sage_if_needed(attention_backend)

        from futudiffu.sampling import warmup_diffusion
        warmup_diffusion(
            self._diff_compiled, self._diff_model,
            self._device, self._dtype,
            width=width, height=height,
        )

    def warmup_packed(self, n_images: int) -> None:
        self._ensure_diffusion()
        from futudiffu.sampling import warmup_packed
        warmup_packed(
            self._diff_compiled, self._diff_model,
            self._device, self._dtype, n_images=n_images,
        )

    # --- LoRA ---

    def allocate_adapter(self, params: dict) -> dict:
        self._ensure_diffusion()
        from futudiffu.lora import allocate_adapter

        adapter_name = params["adapter_name"]
        rank = params.get("rank", 8)
        alpha = params.get("alpha", 16.0)
        layer_indices = params.get("layer_indices")
        if layer_indices is not None:
            layer_indices = set(layer_indices)

        injected = allocate_adapter(
            self._diff_model,
            name=adapter_name,
            rank=rank,
            alpha=alpha,
            layer_indices=layer_indices,
        )
        self._lora_configs.append({
            "adapter_name": adapter_name,
            "rank": rank,
            "alpha": alpha,
            "layer_indices": layer_indices,
            "init_b_std": 0.0,
        })
        n_params = sum(a.lora_A.numel() + a.lora_B.numel() for a in injected.values())
        return {
            "adapter_name": adapter_name,
            "n_adapters": len(injected),
            "n_params": n_params,
            "graph_mutated": True,
        }

    def init_adapter_weights(self, params: dict) -> dict:
        self._ensure_diffusion()
        from futudiffu.lora import init_adapter_weights

        n = init_adapter_weights(
            self._diff_model,
            name=params["adapter_name"],
            init_b_std=params.get("init_b_std", 0.0),
            scale=params.get("scale", 1.0),
        )
        return {
            "adapter_name": params["adapter_name"],
            "n_modules_initialized": n,
        }

    def inject_lora(self, params: dict) -> dict:
        import torch
        self._ensure_diffusion()
        from futudiffu.lora import inject_lora

        adapter_name = params["adapter_name"]
        rank = params.get("rank", 8)
        alpha = params.get("alpha", 16.0)
        layer_indices = params.get("layer_indices")
        init_b_std = params.get("init_b_std", 0.0)
        if layer_indices is not None:
            layer_indices = set(layer_indices)

        injected = inject_lora(
            self._diff_model,
            name=adapter_name,
            rank=rank,
            alpha=alpha,
            layer_indices=layer_indices,
            init_b_std=init_b_std,
        )
        self._lora_configs.append({
            "adapter_name": adapter_name,
            "rank": rank,
            "alpha": alpha,
            "layer_indices": layer_indices,
            "init_b_std": init_b_std,
        })

        torch._dynamo.reset()
        self._diff_model.compile_for_execution()
        self._diff_compiled = self._diff_model

        n_params = sum(a.lora_A.numel() + a.lora_B.numel() for a in injected.values())
        return {
            "adapter_name": adapter_name,
            "n_adapters": len(injected),
            "n_params": n_params,
        }

    def update_lora_weights(self, tensor_bytes: bytes) -> dict:
        self._ensure_diffusion()
        from futudiffu.lora import load_lora_state_dict

        tensors = _safetensors_bytes_to_tensors(tensor_bytes)
        sd = {k: v.to(dtype=self._dtype) for k, v in tensors.items()}
        load_lora_state_dict(self._diff_model, sd)

        for key, val in sd.items():
            parts = key.split(".adapters.")
            if len(parts) == 2:
                adapter_name = parts[1].split(".")[0]
                if adapter_name not in self._lora_weights:
                    self._lora_weights[adapter_name] = {}
                self._lora_weights[adapter_name][key] = val.detach().cpu()

        return {"n_tensors": len(sd)}

    def set_adapter_config(self, params: dict) -> dict:
        import torch
        self._ensure_diffusion()
        from futudiffu.lora import (
            clear_lora_scale, freeze_adapter, set_lora_scale, unfreeze_adapter,
        )

        adapter_name = params["adapter_name"]
        scale = params.get("scale")
        frozen = params.get("frozen")
        clear_scale = params.get("clear_scale", False)

        if scale is not None:
            if isinstance(scale, (int, float)):
                scale_t = torch.tensor([scale], device=self._device, dtype=self._dtype)
            else:
                scale_t = torch.tensor(scale, device=self._device, dtype=self._dtype)
            set_lora_scale(self._diff_model, scale_t, adapter_name=adapter_name)
            self._lora_scales[adapter_name] = scale
        elif clear_scale:
            clear_lora_scale(self._diff_model, adapter_name=adapter_name)
            self._lora_scales.pop(adapter_name, None)

        n_frozen = 0
        if frozen is True:
            n_frozen = freeze_adapter(self._diff_model, adapter_name)
        elif frozen is False:
            n_frozen = unfreeze_adapter(self._diff_model, adapter_name)

        return {"adapter_name": adapter_name, "n_frozen": n_frozen}

    def get_lora_state_dict(self, adapter_name: str | None) -> dict:
        self._ensure_diffusion()
        from futudiffu.lora import lora_state_dict
        sd = lora_state_dict(self._diff_model, adapter_name=adapter_name)
        return {k: v.cpu() for k, v in sd.items()}

    def dump_all_loras(self, output_dir: str) -> dict:
        from futudiffu.lora import dump_all_loras
        if self._diff_model is None:
            return {"files": [], "note": "no diffusion model loaded"}
        return dump_all_loras(
            self._diff_model, output_dir,
            btrm_head=self._btrm_head, btrm_config=self._btrm_config,
        )

    # --- BTRM ---

    def inject_btrm_head(self, params: dict) -> dict:
        import torch
        from futudiffu.btrm import ScoreUnembedder
        from futudiffu.lora import get_lora_params

        hidden_dim = params.get("hidden_dim", 3840)
        head_names = params.get("head_names", ["bit_quality", "step_quality"])
        logit_cap = params.get("logit_cap", 10.0)
        lr = params.get("lr")
        weight_decay = params.get("weight_decay", 0.0)

        self._btrm_head = ScoreUnembedder(
            hidden_dim=hidden_dim,
            head_names=head_names,
            logit_cap=logit_cap,
        ).to(device=self._device, dtype=self._dtype)
        self._btrm_head.train()

        self._btrm_config = {
            "hidden_dim": hidden_dim,
            "head_names": list(head_names),
            "logit_cap": logit_cap,
        }

        if lr is not None:
            rtheta_params = list(get_lora_params(self._diff_model, adapter_name="rtheta")) \
                if self._diff_model is not None else []
            param_groups = [
                {"params": list(self._btrm_head.parameters()), "lr": lr},
            ]
            if rtheta_params:
                param_groups.append({"params": rtheta_params, "lr": lr})
            self._btrm_optimizer = torch.optim.AdamW(
                param_groups, weight_decay=weight_decay,
            )

        n_params = sum(p.numel() for p in self._btrm_head.parameters())
        return {
            "n_heads": len(head_names),
            "n_params": n_params,
            "has_optimizer": lr is not None,
        }

    def score_btrm(self, params: dict, tensor_bytes: bytes) -> dict:
        import torch
        self._ensure_diffusion()
        assert self._btrm_head is not None, "BTRM head not injected"

        attn = params.get("attention_backend", "sdpa")
        self._configure_sage_if_needed(attn)
        from futudiffu.training_utils import run_backbone_hidden
        tensors = _safetensors_bytes_to_tensors(tensor_bytes)
        hidden = run_backbone_hidden(
            self._diff_model,
            tensors["latent"], tensors["sigma"], tensors["conditioning"],
            self._device, self._dtype,
            multiplier=params.get("multiplier", 1.0),
        )
        with torch.no_grad():
            scores = self._btrm_head(hidden)
        return {"scores": scores.detach().cpu().tolist()}

    def train_btrm_step(self, params: dict, tensor_bytes: bytes) -> dict:
        self._ensure_diffusion()
        assert self._btrm_head is not None, "BTRM head not injected"
        assert self._btrm_optimizer is not None, "BTRM optimizer not created"

        attn = params.get("attention_backend", "sdpa")
        self._configure_sage_if_needed(attn)
        from futudiffu.training_utils import train_btrm_step

        tensors = _safetensors_bytes_to_tensors(tensor_bytes)
        metadata = train_btrm_step(
            self._diff_model, self._btrm_head, self._btrm_optimizer,
            self._device, self._dtype, params, tensors,
        )
        return metadata

    # --- Policy ---

    def accumulate_policy_gradients(self, params: dict, tensor_bytes: bytes) -> dict:
        self._ensure_diffusion()
        from src_ii.policy_step import accumulate_reinforce_gradients

        tensors = _safetensors_bytes_to_tensors(tensor_bytes)
        return accumulate_reinforce_gradients(
            self._diff_model, self._device, self._dtype, params, tensors,
        )

    def policy_optimizer_step(self, params: dict) -> dict:
        self._ensure_diffusion()
        from src_ii.policy_step import policy_optimizer_step

        return policy_optimizer_step(
            self._diff_model, self._policy_optimizers,
            self._device, self._dtype, params,
        )

    # --- Batch forward ---

    def _ensure_batch_executor(self):
        if self._batch_executor is not None:
            return
        from src_ii.batch_executor import BatchExecutor
        self._batch_executor = BatchExecutor(
            self._diff_model, device=self._device,
        )

    def batch_forward(self, params: dict, tensors: dict) -> tuple[dict, dict]:
        """Execute fork specs: SCATTER -> PACKSOLVE -> EXECUTE -> DENOISE.

        params["queries"]: list of query dicts. Literal tensors are embedded
        directly — client pre-resolves tensor names to actual tensors before
        calling this method. (Over HTTP the endpoint does the lookup from
        the safetensors blob using tensor_key fields in the query.)

        Returns (result_tensors, metadata) where result_tensors is a flat dict
        keyed "denoised_{query_id}_{entry_id}" and metadata["tags"] lists
        [{query_id, entry_id, denoised_key, score_key}, ...].
        """
        import torch
        self._ensure_diffusion()
        self._ensure_batch_executor()

        queries_raw = params.get("queries", [])
        # Resolve tensor references: query fields ending in "_key" are
        # looked up in the tensors dict; plain tensor values pass through.
        queries = []
        for q in queries_raw:
            resolved = dict(q)
            # base_latent and base_cond may arrive as tensor_key strings
            for field in ("base_latent", "base_cond"):
                key_field = field + "_key"
                if key_field in q and q[key_field] in tensors:
                    resolved[field] = tensors[q[key_field]].to(self._device)
            # forks: resolve cond tensors
            resolved_forks = []
            for fork in q.get("forks", []):
                rf = dict(fork)
                cond_key = fork.get("cond_key")
                if cond_key and cond_key in tensors:
                    rf["cond"] = tensors[cond_key].to(self._device)
                resolved_forks.append(rf)
            resolved["forks"] = resolved_forks
            # adapter_scales
            as_key = q.get("adapter_scales_key")
            if as_key and as_key in tensors:
                resolved["adapter_scales"] = tensors[as_key].to(self._device)
            queries.append(resolved)

        results = self._batch_executor.execute(queries)

        # Flatten to safetensors-serializable dict + tag metadata
        result_tensors = {}
        tags = []
        for r in results:
            qid, eid = r["query_id"], r["entry_id"]
            d_key = f"denoised_{qid}_{eid}"
            result_tensors[d_key] = r["denoised"]
            tag = {"query_id": qid, "entry_id": eid, "denoised_key": d_key}
            if r["scores"] is not None:
                s_key = f"scores_{qid}_{eid}"
                result_tensors[s_key] = r["scores"]
                tag["score_key"] = s_key
            tags.append(tag)

        return result_tensors, {"tags": tags, "n_entries": len(results)}

"""FastAPI BFF for client_yeetums: routes, static file serving, generation.

Torch-free. The only dependencies are FastAPI, httpx, Pillow, and the
bridge/gallery/models modules in this package.
"""

from __future__ import annotations

import json
import logging
import random
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src_ii.client_yeetums.bridge import InferenceBridge
from src_ii.client_yeetums.gallery import Gallery
from src_ii.client_yeetums.models import (
    AdapterAllocateRequest,
    AdapterConfigRequest,
    AdapterInitRequest,
    BTRMHeadRequest,
    BatchGenerateRequest,
    BatchGenerateResponse,
    ConfigVolumesRequest,
    ConfigVolumesResponse,
    DefaultConfigResponse,
    GalleryEntry,
    GalleryListResponse,
    GenerateRequest,
    ResolutionRequest,
    ResolutionResponse,
    ServerStatusResponse,
    TrainingRunRequest,
    TrainingStatusResponse,
    ValidationRequest,
)
from src_ii.config_distributions import compute_config_volumes, resolve_generation_config

logger = logging.getLogger("yeetums.app")

STATIC_DIR = Path(__file__).parent / "static"

# Default generation config (the config JSON panel starts with this)
DEFAULT_CONFIG: dict[str, Any] = {
    "prompt": "",
    "negative_prompt": "",
    "seed": {"min": 0, "max": 4294967295},
    "n_steps": 30,
    "cfg": 4.0,
    "attention_backend": "sage",
    "sampling_shift": 1.0,
    "multiplier": 1.0,
    "denoise": 1.0,
    "k": 1,
    "resolution": {
        "megapixels": 1048576,
        "aspect_ratio": 1.5385,
        "quantize": 32,
    },
}


# ---------------------------------------------------------------------------
# Uploaded source storage (in-memory, keyed by source_id)
# ---------------------------------------------------------------------------

_source_store: dict[str, bytes] = {}

# Resolved configs from distributional draws (keyed by job_id, ephemeral)
_resolved_configs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app(
    inference_url: str = "http://localhost:8000",
    gallery_dir: str = "yeetums_gallery",
    timeout_s: float = 600.0,
) -> FastAPI:
    """Create the yeetums BFF FastAPI application.

    Args:
        inference_url: URL of the GPU inference server.
        gallery_dir: Directory for gallery images.
        timeout_s: Request timeout for inference server calls.

    Returns:
        Configured FastAPI app.
    """
    bridge = InferenceBridge(inference_url, timeout_s=timeout_s)
    gallery = Gallery(gallery_dir)

    app = FastAPI(
        title="client_yeetums",
        description="Diegetic web UI for futudiffu inference",
        version="0.2.0",
    )

    # ---------------------------------------------------------------
    # Static files + index
    # ---------------------------------------------------------------

    @app.get("/")
    async def index():
        """Serve the main page."""
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ---------------------------------------------------------------
    # Status
    # ---------------------------------------------------------------

    @app.get("/api/status")
    async def api_status():
        """Proxy inference server status."""
        status = bridge.get_status()
        if not status:
            return ServerStatusResponse(connected=False).model_dump()

        return ServerStatusResponse(
            connected=True,
            loaded_models=status.get("loaded_models", []),
            vram_allocated_gb=status.get("vram_allocated_gb", 0.0),
            vram_reserved_gb=status.get("vram_reserved_gb", 0.0),
            vram_total_gb=status.get("vram_total_gb", 0.0),
            phase=status.get("phase"),
            server_version=status.get("server_version", ""),
        ).model_dump()

    # ---------------------------------------------------------------
    # Resolution computation (pure Python, no torch)
    # ---------------------------------------------------------------

    @app.post("/api/resolution")
    async def api_resolution(req: ResolutionRequest):
        """Compute (W, H) from anchor pixels + aspect ratio."""
        from src_ii.resolution_sampling import (
            sample_resolution,
            ANCHOR_LABELS,
            assign_budget_tier,
        )

        w, h = sample_resolution(req.anchor_pixels, req.aspect_ratio)
        anchor = assign_budget_tier(w, h)
        label = ANCHOR_LABELS.get(anchor, f"{anchor}px")

        return ResolutionResponse(
            width=w,
            height=h,
            actual_pixels=w * h,
            anchor_label=label,
        ).model_dump()

    # ---------------------------------------------------------------
    # Config
    # ---------------------------------------------------------------

    @app.get("/api/config/default")
    async def api_config_default():
        """Return the default generation config."""
        return DefaultConfigResponse(config=DEFAULT_CONFIG).model_dump()

    @app.get("/api/config/presets")
    async def api_config_presets():
        """List available config presets from the presets/ directory."""
        presets_dir = Path(__file__).parent / "presets"
        presets = []
        if presets_dir.is_dir():
            for p in sorted(presets_dir.glob("*.json")):
                with open(p) as f:
                    presets.append({"name": p.stem, "config": json.load(f)})
        return {"presets": presets}

    @app.post("/api/config/volumes")
    async def api_config_volumes(req: ConfigVolumesRequest):
        """Compute per-field volume info for distributional config fields."""
        k = max(1, min(16, req.k))
        volumes = compute_config_volumes(req.config, k=k)
        total_log = sum(v["log_volume"] for v in volumes)
        return ConfigVolumesResponse(
            volumes=volumes,
            total_log_volume=total_log,
        ).model_dump()

    # ---------------------------------------------------------------
    # Generation (queue-based with SSE streaming)
    # ---------------------------------------------------------------

    @app.post("/api/batch_generate")
    async def api_batch_generate(req: BatchGenerateRequest):
        """Resolve distributional config k times, enqueue k scalar jobs.

        Pure data transformation: distributional config → k scalar configs →
        k enqueue calls. No scheduling, no stream management. Each job gets
        its own stream_url through the ONE streaming path (/api/stream/{job_id}).
        """
        config = req.config

        # Training runs route through the same batch_generate endpoint.
        # The frontend gets a single-job batch with a training stream URL.
        config_type = config.get("type", "inference")
        if config_type in ("btrm", "ddgrpo", "validate", "policy_intervention"):
            try:
                train_config = dict(config)
                train_config["run_type"] = config_type
                result = bridge.start_training_run(train_config)
            except Exception as e:
                raise HTTPException(status_code=502, detail=str(e))

            batch_id = uuid.uuid4().hex[:12]
            return BatchGenerateResponse(
                batch_id=batch_id,
                k=1,
                jobs=[{
                    "job_id": result["run_id"],
                    "batch_index": 0,
                    "seed": 0,
                    "width": 0,
                    "height": 0,
                    "stream_url": f"/api/train/stream/{result['run_id']}?batch_id={batch_id}",
                    "batch_id": batch_id,
                    "resolved_config": config,
                }],
            ).model_dump()

        k = max(1, min(16, int(config.get("k", 1))))
        batch_id = uuid.uuid4().hex[:12]
        base_seed = random.randint(0, 2**32 - 1)

        from src_ii.resolution_sampling import sample_resolution

        jobs = []
        for i in range(k):
            rng = random.Random(base_seed + i)
            resolved = resolve_generation_config(config, rng)

            # Derive (w, h) from resolved megapixels + aspect_ratio.
            # resolve_generation_config preserves input shape, so resolution
            # is always {megapixels, aspect_ratio, quantize} — never {width, height}.
            res = resolved.get("resolution", {})
            w, h = sample_resolution(
                res.get("megapixels", 1048576),
                float(res.get("aspect_ratio", 1.0)),
                step=res.get("quantize", 32),
            )

            seed = resolved.get("seed", rng.randint(0, 2**32 - 1))
            if isinstance(seed, float):
                seed = int(seed)

            try:
                job_id = bridge.enqueue_generation(
                    prompt=resolved.get("prompt", ""),
                    negative_prompt=resolved.get("negative_prompt", ""),
                    seed=seed,
                    n_steps=int(resolved.get("n_steps", 30)),
                    cfg=float(resolved.get("cfg", 4.0)),
                    width=w,
                    height=h,
                    attention_backend=resolved.get("attention_backend", "sage"),
                    sampling_shift=float(resolved.get("sampling_shift", 1.0)),
                    multiplier=float(resolved.get("multiplier", 1.0)),
                    denoise=float(resolved.get("denoise", 1.0)),
                )
            except Exception as e:
                logger.error(f"Enqueue failed for batch {batch_id}[{i}]: {e}")
                raise HTTPException(status_code=502, detail=str(e))

            # Build resolved config: a complete, valid input config with all
            # distributions collapsed to their sampled scalars. Pasting this
            # back into the input config panel and hitting generate reproduces
            # the exact same image.
            resolved_scalar = dict(resolved)
            resolved_scalar["seed"] = seed
            resolved_scalar["k"] = 1
            _resolved_configs[job_id] = resolved_scalar

            jobs.append({
                "job_id": job_id,
                "batch_index": i,
                "seed": seed,
                "width": w,
                "height": h,
                "stream_url": f"/api/stream/{job_id}",
                "batch_id": batch_id,
                "resolved_config": resolved_scalar,
            })

        return BatchGenerateResponse(
            batch_id=batch_id,
            k=k,
            jobs=jobs,
        ).model_dump()

    @app.get("/api/stream/{job_id}")
    async def api_stream(job_id: str, batch_id: str = "", batch_index: int = -1):
        """Proxy SSE events from the inference server to the browser.

        ONE streaming path for all jobs. Batch metadata (batch_id, batch_index)
        passed as query params, injected into gallery_ready so the frontend
        can reconstruct batch groups.
        """
        async def event_gen():
            url = f"{bridge._base_url}/stream/{job_id}"
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(600.0, connect=30.0)
                ) as client:
                    async with client.stream("GET", url) as resp:
                        if resp.status_code != 200:
                            yield _sse_event("error", {
                                "error": f"Stream connect failed: {resp.status_code}"
                            })
                            return

                        event_type = "message"
                        data_buf = ""
                        async for line in resp.aiter_lines():
                            if line.startswith("event: "):
                                event_type = line[7:].strip()
                            elif line.startswith("data: "):
                                data_buf = line[6:]
                            elif line == "" and data_buf:
                                # Forward event to browser
                                yield f"event: {event_type}\ndata: {data_buf}\n\n"

                                if event_type == "complete":
                                    # Fetch PNG, save to gallery, emit gallery_ready
                                    try:
                                        png_bytes, metadata = bridge.get_result_png(job_id)
                                        if batch_id:
                                            metadata["batch_id"] = batch_id
                                            metadata["batch_index"] = batch_index
                                        resolved_cfg = _resolved_configs.pop(job_id, None)
                                        if resolved_cfg is not None:
                                            metadata["resolved_config"] = resolved_cfg
                                        entry = gallery.add(png_bytes, metadata)
                                        gallery_data = json.dumps({
                                            "gallery_id": entry["id"],
                                            "image_url": entry["image_url"],
                                            "seed": metadata.get("seed"),
                                            "width": metadata.get("width"),
                                            "height": metadata.get("height"),
                                            "elapsed_s": metadata.get("elapsed_s"),
                                            "prompt": metadata.get("prompt", ""),
                                            "batch_id": batch_id or None,
                                            "batch_index": batch_index if batch_index >= 0 else None,
                                            "resolved_config": resolved_cfg,
                                        })
                                        yield f"event: gallery_ready\ndata: {gallery_data}\n\n"
                                    except Exception as e:
                                        logger.error(f"Gallery save failed: {e}")
                                        yield _sse_event("gallery_error", {
                                            "error": str(e)
                                        })
                                    yield _sse_event("done", {"reason": "complete"})
                                    return

                                if event_type == "error":
                                    yield _sse_event("done", {"reason": "error"})
                                    return

                                event_type = "message"
                                data_buf = ""

            except Exception as e:
                logger.error(f"SSE proxy error: {e}")
                yield _sse_event("error", {"error": str(e)})

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/generate_sync")
    async def api_generate_sync(req: GenerateRequest):
        """Blocking generation endpoint for curl testing.

        Enqueues a job, polls until complete, saves to gallery, returns result.
        """
        import asyncio
        from src_ii.resolution_sampling import sample_resolution

        seed = req.seed if req.seed >= 0 else random.randint(0, 2**32 - 1)
        w, h = sample_resolution(req.anchor_pixels, req.aspect_ratio)

        try:
            job_id = bridge.enqueue_generation(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                seed=seed,
                n_steps=req.n_steps,
                cfg=req.cfg,
                width=w,
                height=h,
                attention_backend=req.attention_backend,
                sampling_shift=req.sampling_shift,
                multiplier=req.multiplier,
                denoise=req.denoise,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

        # Poll for completion by consuming SSE events
        metadata = {}
        try:
            for event in bridge.stream_job_events(job_id):
                if event["type"] == "complete":
                    metadata = event.get("data", {})
                    break
                elif event["type"] == "error":
                    raise HTTPException(
                        status_code=502,
                        detail=event.get("data", {}).get("error", "Generation failed"),
                    )
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e))

        # Fetch PNG and save to gallery
        try:
            png_bytes, result_meta = bridge.get_result_png(job_id)
            entry = gallery.add(png_bytes, result_meta)
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

        return GenerateResponse(
            gallery_id=entry["id"],
            seed=seed,
            width=w,
            height=h,
            image_url=entry["image_url"],
            elapsed_s=result_meta.get("elapsed_s", 0.0),
            prompt=req.prompt,
        ).model_dump()

    # ---------------------------------------------------------------
    # Queue status
    # ---------------------------------------------------------------

    @app.get("/api/queue_status")
    async def api_queue_status():
        """Proxy queue status from the inference server."""
        return bridge.get_queue_status()

    # ---------------------------------------------------------------
    # i2i source upload
    # ---------------------------------------------------------------

    @app.post("/api/upload_source")
    async def api_upload_source(file: UploadFile = File(...)):
        """Upload an i2i source image. Returns a source_id for use in /generate."""
        png_bytes = await file.read()
        try:
            latent_bytes = bridge.vae_encode_png(png_bytes)
        except Exception as e:
            logger.error(f"VAE encode failed: {e}")
            raise HTTPException(status_code=502, detail=str(e))

        source_id = f"src_{random.randint(0, 2**32 - 1):08x}"
        _source_store[source_id] = latent_bytes

        return {"source_id": source_id}

    # ---------------------------------------------------------------
    # Gallery
    # ---------------------------------------------------------------

    @app.get("/api/gallery")
    async def api_gallery(limit: int = 50, offset: int = 0):
        """List recent gallery entries."""
        entries = gallery.list_entries(limit=limit, offset=offset)
        return GalleryListResponse(
            entries=[GalleryEntry(**e) for e in entries],
            total=gallery.total,
        ).model_dump()

    @app.get("/api/gallery/{entry_id}/image")
    async def api_gallery_image(entry_id: str):
        """Serve a gallery image as PNG."""
        path = gallery.get_image_path(entry_id)
        if path is None:
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(str(path), media_type="image/png")

    @app.get("/api/gallery/{entry_id}/meta")
    async def api_gallery_meta(entry_id: str):
        """Get gallery entry metadata."""
        entry = gallery.get(entry_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Entry not found")
        return entry

    # ---------------------------------------------------------------
    # Warmup
    # ---------------------------------------------------------------

    @app.post("/api/warmup")
    async def api_warmup():
        """Trigger inference server warmup."""
        result = bridge.warmup()
        return result

    # ---------------------------------------------------------------
    # Training lifecycle (proxy to server via bridge)
    # ---------------------------------------------------------------

    @app.post("/api/train/btrm/start")
    async def api_train_btrm_start(req: TrainingRunRequest):
        """Start a BTRM training run on the inference server."""
        config = dict(req.config)
        config["run_type"] = "btrm"
        try:
            result = bridge.start_training_run(config)
            return result
        except Exception as e:
            logger.error(f"Training start failed: {e}")
            raise HTTPException(status_code=502, detail=str(e))

    @app.post("/api/train/ddgrpo/start")
    async def api_train_ddgrpo_start(req: TrainingRunRequest):
        """Start a DDGRPO policy optimization run."""
        config = dict(req.config)
        config["run_type"] = "ddgrpo"
        try:
            result = bridge.start_training_run(config)
            return result
        except Exception as e:
            logger.error(f"DDGRPO start failed: {e}")
            raise HTTPException(status_code=502, detail=str(e))

    @app.get("/api/train/status")
    async def api_train_status():
        """Get current training run status."""
        try:
            result = bridge.get_training_status()
            return TrainingStatusResponse(**result).model_dump()
        except Exception as e:
            return TrainingStatusResponse().model_dump()

    @app.post("/api/train/stop")
    async def api_train_stop():
        """Stop the active training run."""
        try:
            result = bridge.stop_training_run("")
            return result
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @app.get("/api/train/stream/{run_id}")
    async def api_train_stream(run_id: str, batch_id: str = ""):
        """Proxy SSE events from a training run.

        Parses SSE events from the server. Intercepts ``artifact_ready``
        events: fetches the PNG from the server, saves it to the gallery,
        and emits ``gallery_ready`` so the frontend treats training charts
        the same as inference images. Emits ``done`` on completion/error.
        """
        async def event_gen():
            url = f"{bridge._base_url}/training/stream/{run_id}"
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(3600.0, connect=30.0)
                ) as client:
                    async with client.stream("GET", url) as resp:
                        if resp.status_code != 200:
                            yield _sse_event("error", {
                                "error": f"Stream connect failed: {resp.status_code}"
                            })
                            yield _sse_event("done", {"reason": "error"})
                            return

                        event_type = "message"
                        data_buf = ""
                        artifact_index = 0

                        async for line in resp.aiter_lines():
                            if line.startswith("event: "):
                                event_type = line[7:].strip()
                            elif line.startswith("data: "):
                                data_buf = line[6:]
                            elif line == "" and data_buf:
                                # Forward non-intercepted events to the browser
                                if event_type not in ("artifact_ready", "complete", "error"):
                                    yield f"event: {event_type}\ndata: {data_buf}\n\n"

                                if event_type == "artifact_ready":
                                    try:
                                        payload = json.loads(data_buf)
                                        artifact_path = payload.get("path", "")
                                        png_url = f"{bridge._base_url}/training/artifacts/{run_id}/{artifact_path}"
                                        png_resp = await client.get(png_url)
                                        if png_resp.status_code == 200:
                                            metadata = payload.get("metadata", {})
                                            if batch_id:
                                                metadata["batch_id"] = batch_id
                                            metadata["batch_index"] = artifact_index
                                            metadata["artifact_type"] = payload.get("type", "chart")
                                            metadata["label"] = payload.get("label", artifact_path)
                                            metadata["run_id"] = run_id
                                            entry = gallery.add(png_resp.content, metadata)
                                            gallery_data = json.dumps({
                                                "gallery_id": entry["id"],
                                                "image_url": entry["image_url"],
                                                "batch_id": batch_id or None,
                                                "batch_index": artifact_index,
                                                "artifact_type": metadata["artifact_type"],
                                                "label": metadata["label"],
                                                "run_id": run_id,
                                            })
                                            yield f"event: gallery_ready\ndata: {gallery_data}\n\n"
                                            artifact_index += 1
                                        else:
                                            logger.warning(
                                                f"Artifact fetch failed ({png_resp.status_code}): {artifact_path}"
                                            )
                                    except Exception as e:
                                        logger.error(f"Artifact interception failed: {e}")

                                elif event_type == "complete":
                                    yield f"event: {event_type}\ndata: {data_buf}\n\n"
                                    yield _sse_event("done", {
                                        "reason": "complete",
                                        "n_artifacts": artifact_index,
                                    })
                                    return

                                elif event_type == "error":
                                    yield f"event: {event_type}\ndata: {data_buf}\n\n"
                                    yield _sse_event("done", {"reason": "error"})
                                    return

                                event_type = "message"
                                data_buf = ""

            except Exception as e:
                logger.error(f"Training SSE proxy error: {e}")
                yield _sse_event("error", {"error": str(e)})
                yield _sse_event("done", {"reason": "error"})

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ---------------------------------------------------------------
    # Training artifacts
    # ---------------------------------------------------------------

    @app.get("/api/train/metrics/{run_id}")
    async def api_train_metrics(run_id: str):
        """Fetch training metrics JSONL."""
        try:
            data = bridge.get_training_artifacts(run_id, "training_metrics.jsonl")
            return StreamingResponse(
                iter([data]),
                media_type="application/x-ndjson",
            )
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/api/train/charts/{run_id}/{chart_name}")
    async def api_train_chart(run_id: str, chart_name: str):
        """Fetch a training chart PNG."""
        try:
            data = bridge.get_training_artifacts(run_id, chart_name)
            return StreamingResponse(
                iter([data]),
                media_type="image/png",
            )
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/api/train/analysis/{run_id}")
    async def api_train_analysis(run_id: str):
        """Fetch training analysis markdown."""
        try:
            data = bridge.get_training_artifacts(run_id, "training_analysis.md")
            return StreamingResponse(
                iter([data]),
                media_type="text/markdown",
            )
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/api/train/summary/{run_id}")
    async def api_train_summary(run_id: str):
        """Fetch run summary JSON."""
        try:
            data = bridge.get_training_artifacts(run_id, "run_summary.json")
            return JSONResponse(content=json.loads(data))
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/api/train/validation/{run_id}/{val_type}")
    async def api_train_validation(run_id: str, val_type: str):
        """Fetch validation log (pinkify, tnt, decorrelation)."""
        filename = f"{val_type}_validation_log.jsonl"
        try:
            data = bridge.get_training_artifacts(run_id, filename)
            return StreamingResponse(
                iter([data]),
                media_type="application/x-ndjson",
            )
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/api/train/checkpoint/{run_id}/{step}")
    async def api_train_checkpoint(run_id: str, step: int):
        """Fetch a checkpoint's adapter weights."""
        path = f"checkpoint_step{step:03d}/rtheta_adapter.safetensors"
        try:
            data = bridge.get_training_artifacts(run_id, path)
            return StreamingResponse(
                iter([data]),
                media_type="application/octet-stream",
            )
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

    # ---------------------------------------------------------------
    # On-demand validation
    # ---------------------------------------------------------------

    @app.post("/api/validate/pinkify")
    async def api_validate_pinkify():
        """Run pinkify validation against current model state."""
        try:
            config = {"challenge_type": "pinkify", "run_type": "validate"}
            # Use the training endpoint for validation
            resp = bridge._client.post(
                "/training/validate",
                json={"challenge_type": "pinkify"},
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=resp.text[:500])
            return resp.json()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @app.post("/api/validate/tnt")
    async def api_validate_tnt():
        """Run TNT validation against current model state."""
        try:
            resp = bridge._client.post(
                "/training/validate",
                json={"challenge_type": "tnt"},
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=resp.text[:500])
            return resp.json()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @app.post("/api/validate/decorrelation")
    async def api_validate_decorrelation():
        """Run cross-head decorrelation check."""
        try:
            resp = bridge._client.post(
                "/training/validate",
                json={"challenge_type": "decorrelation"},
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=resp.text[:500])
            return resp.json()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    # ---------------------------------------------------------------
    # Model management (relay to server)
    # ---------------------------------------------------------------

    @app.post("/api/model/adapter/allocate")
    async def api_adapter_allocate(req: AdapterAllocateRequest):
        """Allocate a LoRA adapter on the inference server."""
        try:
            return bridge.allocate_adapter(
                req.name, req.rank, req.alpha, req.layer_indices,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @app.post("/api/model/adapter/init")
    async def api_adapter_init(req: AdapterInitRequest):
        """Initialize adapter weights."""
        try:
            return bridge.init_adapter_weights(
                req.name, req.init_b_std, req.scale,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @app.post("/api/model/adapter/config")
    async def api_adapter_config(req: AdapterConfigRequest):
        """Set adapter scale and/or freeze state."""
        try:
            return bridge.set_adapter_config(
                req.name, req.scale, req.frozen,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @app.get("/api/model/adapter/state")
    async def api_adapter_state(name: str | None = None):
        """Get current LoRA weights as safetensors bytes."""
        try:
            data = bridge.get_lora_state_dict(name)
            return StreamingResponse(
                iter([data]),
                media_type="application/octet-stream",
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @app.post("/api/model/btrm/inject")
    async def api_btrm_inject(req: BTRMHeadRequest):
        """Inject BTRM scoring head."""
        try:
            return bridge.inject_btrm_head(
                req.head_names, req.logit_cap, req.lr, req.hidden_dim,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    return app


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def _sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

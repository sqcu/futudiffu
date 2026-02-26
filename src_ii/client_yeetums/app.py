"""FastAPI BFF for client_yeetums: routes, static file serving, generation.

Torch-free. The only dependencies are FastAPI, httpx, Pillow, and the
bridge/gallery/models modules in this package.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src_ii.client_yeetums.bridge import InferenceBridge
from src_ii.client_yeetums.gallery import Gallery
from src_ii.client_yeetums.models import (
    BatchGenerateRequest,
    DefaultConfigResponse,
    GalleryEntry,
    GalleryListResponse,
    GenerateRequest,
    GenerateResponse,
    ResolutionRequest,
    ResolutionResponse,
    ServerStatusResponse,
)

logger = logging.getLogger("yeetums.app")

STATIC_DIR = Path(__file__).parent / "static"

# Default generation config (the config JSON panel starts with this)
DEFAULT_CONFIG: dict[str, Any] = {
    "prompt": "",
    "negative_prompt": "",
    "seed": -1,
    "n_steps": 30,
    "cfg": 4.0,
    "attention_backend": "sage",
    "sampling_shift": 1.0,
    "multiplier": 1.0,
    "denoise": 1.0,
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

    # ---------------------------------------------------------------
    # Generation (queue-based with SSE streaming)
    # ---------------------------------------------------------------

    @app.post("/api/generate")
    async def api_generate(req: GenerateRequest):
        """Enqueue a generation job. Returns job_id + stream URL immediately.

        The actual GPU work happens asynchronously on the inference server.
        Use /api/stream/{job_id} to get progress events.
        """
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
            logger.error(f"Enqueue failed: {e}")
            raise HTTPException(status_code=502, detail=str(e))

        return {
            "job_id": job_id,
            "stream_url": f"/api/stream/{job_id}",
            "seed": seed,
            "width": w,
            "height": h,
        }

    @app.get("/api/stream/{job_id}")
    async def api_stream(job_id: str):
        """Proxy SSE events from the inference server to the browser.

        Connects to the inference server's /stream/{job_id} SSE endpoint and
        forwards events. On 'complete', fetches the PNG result, saves to
        gallery, and emits a 'gallery_ready' event with the image URL.
        """
        import httpx

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
                                        entry = gallery.add(png_bytes, metadata)
                                        gallery_data = json.dumps({
                                            "gallery_id": entry["id"],
                                            "image_url": entry["image_url"],
                                            "seed": metadata.get("seed"),
                                            "width": metadata.get("width"),
                                            "height": metadata.get("height"),
                                            "elapsed_s": metadata.get("elapsed_s"),
                                            "prompt": metadata.get("prompt", ""),
                                        })
                                        yield f"event: gallery_ready\ndata: {gallery_data}\n\n"
                                    except Exception as e:
                                        logger.error(f"Gallery save failed: {e}")
                                        yield _sse_event("gallery_error", {
                                            "error": str(e)
                                        })
                                    return

                                if event_type == "error":
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

    return app


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def _sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

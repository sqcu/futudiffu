"""Inference work queue: batched generation with SSE progress.

Thin batching shim over the existing server primitives. Accumulates
generation requests over a configurable window, groups by n_steps,
packs them into the params/tensors format that
GPUModelBackend.sample_trajectory_packed already expects, and
distributes results back to subscribers via SSE events.

The queue does NOT reimplement the GPU pipeline. It calls:
  - backend.encode_prompt() for TE
  - backend.sample_trajectory_packed() for diffusion
  - backend.vae_decode_png() for VAE
These are the same methods the existing RPC endpoints call. The queue
is one client among many — it has no special status over GPU access.

Import constraints:
  - asyncio, dataclasses, uuid for queue machinery
  - json for SSE serialization
  - Does NOT import torch at module level
  - Does NOT import from src.futudiffu (frozen)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("futudiffu.inference_queue")


# ---------------------------------------------------------------------------
# Job dataclass
# ---------------------------------------------------------------------------

@dataclass
class InferenceJob:
    """A single generation request in the queue."""

    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    prompt: str = ""
    negative_prompt: str = ""
    seed: int = 42
    n_steps: int = 30
    cfg: float = 4.0
    width: int = 1280
    height: int = 832
    sampling_shift: float = 1.0
    multiplier: float = 1.0
    denoise: float = 1.0
    attention_backend: str = "sage"
    source_latent_bytes: bytes | None = None

    # Filled by the worker
    status: str = "queued"
    step: int = 0
    total_steps: int = 0
    result_png: bytes | None = None
    metadata: dict | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    completed_at: float | None = None

    # Async plumbing
    _event: asyncio.Event = field(default_factory=asyncio.Event)
    _subscribers: list[asyncio.Queue] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SSE event helpers
# ---------------------------------------------------------------------------

def _make_event(event_type: str, data: dict) -> dict:
    return {"type": event_type, "data": data}


def _notify(job: InferenceJob, event_type: str, extra: dict | None = None):
    """Push an SSE event to all subscribers of a job."""
    data = {"job_id": job.job_id, "status": job.status}
    if extra:
        data.update(extra)
    event = _make_event(event_type, data)
    for q in job._subscribers:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass


def _notify_all(jobs: list[InferenceJob], status: str, extra: dict | None = None):
    """Set status and notify all jobs in a batch."""
    for job in jobs:
        job.status = status
        _notify(job, status, extra)


# ---------------------------------------------------------------------------
# InferenceQueue
# ---------------------------------------------------------------------------

class InferenceQueue:
    """Batching shim over the existing server primitives.

    Accumulates jobs, groups by n_steps, packs into the params/tensors
    format that backend.sample_trajectory_packed() expects, calls the
    existing backend methods, distributes results.

    The queue is one client among many. It submits work to the backend
    through the same methods any other caller uses. The backend handles
    serialization naturally (single-threaded on GPU per process).

    Usage:
        queue = InferenceQueue(backend)
        await queue.start()
        job_id = await queue.enqueue(job)
        await queue.stop()
    """

    def __init__(
        self,
        backend,
        batch_window_ms: float = 100,
        max_batch: int = 16,
    ):
        self._backend = backend
        self._batch_window_ms = batch_window_ms
        self._max_batch = max_batch

        self._pending: asyncio.Queue[InferenceJob] = asyncio.Queue()
        self._jobs: dict[str, InferenceJob] = {}
        self._worker_task: asyncio.Task | None = None
        self._running = False

        # Stats
        self._completed_times: list[float] = []

    async def enqueue(self, job: InferenceJob) -> str:
        """Add a job to the queue. Returns job_id."""
        self._jobs[job.job_id] = job
        await self._pending.put(job)
        logger.info(
            f"Enqueued job {job.job_id}: "
            f"{job.width}x{job.height} {job.n_steps}steps "
            f"prompt={job.prompt[:40]!r}"
        )
        return job.job_id

    def get_job(self, job_id: str) -> InferenceJob | None:
        return self._jobs.get(job_id)

    def subscribe(self, job_id: str) -> asyncio.Queue | None:
        """Get an SSE event queue for a job. Returns None if job not found."""
        job = self._jobs.get(job_id)
        if job is None:
            return None
        q: asyncio.Queue = asyncio.Queue(maxsize=256)

        if job.status in ("complete", "error"):
            terminal_data = {"job_id": job.job_id, "status": job.status}
            if job.error:
                terminal_data["error"] = job.error
            q.put_nowait(_make_event(job.status, terminal_data))
        else:
            q.put_nowait(_make_event("status", {
                "job_id": job.job_id, "status": job.status,
                "step": job.step, "total_steps": job.total_steps,
            }))

        job._subscribers.append(q)
        return q

    def queue_status(self) -> dict[str, Any]:
        now = time.monotonic()
        cutoff = now - 60
        recent = [t for t in self._completed_times if t > cutoff]
        self._completed_times = recent

        pending = sum(1 for j in self._jobs.values() if j.status == "queued")
        processing = sum(
            1 for j in self._jobs.values()
            if j.status in ("encoding", "sampling", "decoding")
        )
        return {
            "pending": pending,
            "processing": processing,
            "completed_last_min": len(recent),
            "total_tracked": len(self._jobs),
        }

    async def start(self):
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("InferenceQueue worker started")

    async def stop(self):
        self._running = False
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("InferenceQueue worker stopped")

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    async def _worker_loop(self):
        """Drain -> group by n_steps -> execute -> resolve."""
        while self._running:
            try:
                batch = await self._drain_batch()
                if not batch:
                    continue

                # Group by n_steps (packed euler requires same step count)
                groups: dict[int, list[InferenceJob]] = {}
                for job in batch:
                    groups.setdefault(job.n_steps, []).append(job)

                for n_steps, group_jobs in groups.items():
                    try:
                        await self._execute_batch(group_jobs)
                    except Exception as e:
                        logger.error(f"Batch failed: {e}", exc_info=True)
                        for job in group_jobs:
                            job.status = "error"
                            job.error = str(e)
                            _notify(job, "error", {"error": str(e)})
                            job._event.set()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker loop error: {e}", exc_info=True)
                await asyncio.sleep(0.1)

    async def _drain_batch(self) -> list[InferenceJob]:
        """Block until at least one job, then drain up to max_batch."""
        try:
            first = await asyncio.wait_for(self._pending.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return []

        batch = [first]
        deadline = time.monotonic() + self._batch_window_ms / 1000.0
        while len(batch) < self._max_batch:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                job = await asyncio.wait_for(self._pending.get(), timeout=remaining)
                batch.append(job)
            except asyncio.TimeoutError:
                break

        logger.info(f"Drained batch of {len(batch)} jobs")
        return batch

    # ------------------------------------------------------------------
    # Batch execution: calls existing backend methods
    # ------------------------------------------------------------------

    async def _execute_batch(self, jobs: list[InferenceJob]):
        """Execute a batch through the existing backend primitives.

        1. encode_prompt() for each unique prompt (TE phase)
        2. sample_trajectory_packed() with all jobs (diffusion phase)
        3. vae_decode_png() for each final latent (VAE phase)
        """
        t0 = time.monotonic()
        n_images = len(jobs)

        # Phase 1: TE encode — call backend.encode_prompt() per unique prompt
        _notify_all(jobs, "encoding")
        cond_cache = await asyncio.to_thread(self._encode_prompts, jobs)

        # Phase 2: Diffusion — pack into params/tensors, call backend
        _notify_all(jobs, "sampling")
        for job in jobs:
            job.total_steps = job.n_steps

        result_tensors, metadata = await asyncio.to_thread(
            self._run_packed_trajectory, jobs, cond_cache,
        )

        # Phase 3: VAE decode — call backend.vae_decode_png() per final latent
        _notify_all(jobs, "decoding")
        png_results = await asyncio.to_thread(
            self._decode_finals, jobs, result_tensors,
        )

        # Resolve
        elapsed = time.monotonic() - t0
        now = time.monotonic()
        for i, job in enumerate(jobs):
            job.result_png = png_results[i]
            job.status = "complete"
            job.completed_at = now
            job.metadata = {
                "width": job.width,
                "height": job.height,
                "seed": job.seed,
                "n_steps": job.n_steps,
                "cfg": job.cfg,
                "elapsed_s": round(elapsed, 2),
                "prompt": job.prompt,
                "attention_backend": job.attention_backend,
                "batch_size": n_images,
            }
            _notify(job, "complete", job.metadata)
            job._event.set()
            self._completed_times.append(now)

        logger.info(f"Batch of {n_images} completed in {elapsed:.2f}s")

    # ------------------------------------------------------------------
    # Phase implementations (run in thread via asyncio.to_thread)
    # ------------------------------------------------------------------

    def _encode_prompts(self, jobs: list[InferenceJob]) -> dict[str, Any]:
        """Encode unique prompts via backend.encode_prompt(). Returns cache."""
        cache: dict[str, Any] = {}
        for job in jobs:
            for prompt in (job.prompt, job.negative_prompt):
                if prompt not in cache:
                    result = self._backend.encode_prompt(prompt, layer_idx=-2)
                    cache[prompt] = result["conditioning"]
        return cache

    def _run_packed_trajectory(
        self,
        jobs: list[InferenceJob],
        cond_cache: dict[str, Any],
    ) -> tuple[dict, dict]:
        """Pack jobs into the params/tensors format and call
        backend.sample_trajectory_packed().
        """
        n_images = len(jobs)

        # Build params dict (same contract as run_trajectory_packed)
        params = {
            "n_images": n_images,
            "seeds": [job.seed for job in jobs],
            "n_steps": jobs[0].n_steps,  # grouped by n_steps, so uniform
            "cfg": jobs[0].cfg,
            "multiplier": jobs[0].multiplier,
            "denoise": jobs[0].denoise,
            "widths": [job.width for job in jobs],
            "heights": [job.height for job in jobs],
            "sampling_shifts": [job.sampling_shift for job in jobs],
            "attention_backend": jobs[0].attention_backend,
            "save_steps": [],  # No intermediates for inference
        }

        # Build tensors dict
        tensors = {
            "neg_cond": cond_cache[jobs[0].negative_prompt],
        }
        for i, job in enumerate(jobs):
            tensors[f"pos_cond_{i}"] = cond_cache[job.prompt]

        # Step callback for SSE progress
        def step_callback(info):
            step_i = info.get("i", 0)
            n_steps = info.get("n_steps", jobs[0].n_steps)
            for job in jobs:
                job.step = step_i + 1
                _notify(job, "progress", {
                    "step": step_i + 1,
                    "total_steps": n_steps,
                })

        return self._backend.sample_trajectory_packed(
            params, tensors, callback=step_callback,
        )

    def _decode_finals(
        self,
        jobs: list[InferenceJob],
        result_tensors: dict,
    ) -> list[bytes]:
        """Decode each final_I latent to PNG via backend.vae_decode_png()."""
        from safetensors.torch import save as st_save

        png_list = []
        for i, job in enumerate(jobs):
            latent = result_tensors[f"final_{i}"]
            # vae_decode_png expects safetensors bytes with "latent" key
            st_bytes = st_save({"latent": latent})
            png_bytes = self._backend.vae_decode_png(st_bytes)
            png_list.append(png_bytes)
        return png_list

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def prune_old_jobs(self, max_age_s: float = 300):
        """Remove completed/errored jobs older than max_age_s."""
        now = time.monotonic()
        to_remove = [
            jid for jid, job in self._jobs.items()
            if job.status in ("complete", "error")
            and job.completed_at is not None
            and (now - job.completed_at) > max_age_s
        ]
        for jid in to_remove:
            del self._jobs[jid]

"""Torch-free inference bridge: connects the BFF to the GPU inference server.

Uses httpx for HTTP only. No torch, no safetensors parsing. Tensor blobs
are treated as opaque bytes — received from the inference server and
forwarded back without deserialization.

The bridge handles:
  - Text encoding (prompt -> opaque conditioning bytes)
  - Sampling trajectory via /sample_trajectory_relay (separate cond files)
  - VAE decode to PNG (latent bytes -> PNG bytes via /vae_decode_png)
  - VAE encode from PNG (PNG bytes -> latent bytes via /vae_encode_png)
  - Status polling and warmup
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

logger = logging.getLogger("yeetums.bridge")


class InferenceBridge:
    """Torch-free bridge to the futudiffu inference server.

    All tensor data flows through as opaque bytes. The BFF never parses
    safetensors, never imports torch.
    """

    def __init__(
        self,
        inference_url: str = "http://localhost:8000",
        timeout_s: float = 600.0,
    ):
        self._base_url = inference_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_s, connect=30.0),
        )

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def _raise_server_error(self, endpoint: str, resp: httpx.Response) -> None:
        """Extract and raise a meaningful error from a failed server response."""
        try:
            detail = resp.json()
            msg = detail.get("error", detail.get("detail", resp.text[:500]))
            tb = detail.get("traceback", "")
            if tb:
                logger.error(f"Server traceback for /{endpoint}:\n{tb}")
        except Exception:
            msg = resp.text[:500]
        raise RuntimeError(f"/{endpoint} failed ({resp.status_code}): {msg}")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Get inference server status. Returns empty dict on connection failure."""
        try:
            resp = self._client.get("/status")
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, httpx.ConnectError):
            return {}

    def is_connected(self) -> bool:
        """Quick connectivity check."""
        try:
            resp = self._client.get("/status")
            return resp.status_code == 200
        except (httpx.HTTPError, httpx.ConnectError):
            return False

    # ------------------------------------------------------------------
    # Text encoding (returns opaque safetensors bytes)
    # ------------------------------------------------------------------

    def encode_prompt(self, prompt: str, layer_idx: int = -2) -> bytes:
        """Encode a text prompt. Returns raw safetensors bytes (opaque)."""
        resp = self._client.post(
            "/encode_prompt",
            json={"prompt": prompt, "layer_idx": layer_idx},
        )
        if resp.status_code != 200:
            self._raise_server_error("encode_prompt", resp)
        return resp.content

    # ------------------------------------------------------------------
    # VAE PNG endpoints (torch-free image I/O)
    # ------------------------------------------------------------------

    def vae_decode_png(self, latent_bytes: bytes) -> bytes:
        """Decode latent safetensors to PNG bytes."""
        resp = self._client.post(
            "/vae_decode_png",
            content=latent_bytes,
            headers={"Content-Type": "application/octet-stream"},
        )
        if resp.status_code != 200:
            self._raise_server_error("vae_decode_png", resp)
        return resp.content

    def vae_encode_png(self, png_bytes: bytes) -> bytes:
        """Encode PNG bytes to latent safetensors bytes (opaque)."""
        resp = self._client.post(
            "/vae_encode_png",
            content=png_bytes,
            headers={"Content-Type": "image/png"},
        )
        if resp.status_code != 200:
            self._raise_server_error("vae_encode_png", resp)
        return resp.content

    # ------------------------------------------------------------------
    # Full generation pipeline (torch-free)
    # ------------------------------------------------------------------

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: int = 42,
        n_steps: int = 30,
        cfg: float = 4.0,
        width: int = 1280,
        height: int = 832,
        attention_backend: str = "sage",
        sampling_shift: float = 1.0,
        multiplier: float = 1.0,
        denoise: float = 1.0,
        source_latent_bytes: bytes | None = None,
    ) -> tuple[bytes, dict[str, Any]]:
        """Full t2i/i2i pipeline. Returns (png_bytes, metadata).

        Orchestrates: encode_prompt -> sample_trajectory_relay -> vae_decode_png.
        All tensor data flows as opaque bytes. No torch needed.
        """
        t0 = time.monotonic()

        pos_cond = self.encode_prompt(prompt)
        neg_cond = self.encode_prompt(negative_prompt)

        params = {
            "seed": seed,
            "n_steps": n_steps,
            "cfg": cfg,
            "width": width,
            "height": height,
            "attention_backend": attention_backend,
            "sampling_shift": sampling_shift,
            "multiplier": multiplier,
            "denoise": denoise,
        }

        # Use the relay endpoint that accepts separate conditioning files
        latent_bytes = self._relay_sample_trajectory(
            pos_cond, neg_cond, params, source_latent_bytes
        )

        png_bytes = self.vae_decode_png(latent_bytes)

        elapsed = time.monotonic() - t0
        metadata = {
            "seed": seed,
            "width": width,
            "height": height,
            "n_steps": n_steps,
            "cfg": cfg,
            "attention_backend": attention_backend,
            "elapsed_s": round(elapsed, 2),
            "prompt": prompt,
            "denoise": denoise,
        }

        return png_bytes, metadata

    def _relay_sample_trajectory(
        self,
        pos_cond_bytes: bytes,
        neg_cond_bytes: bytes,
        params: dict[str, Any],
        clean_latent_bytes: bytes | None = None,
    ) -> bytes:
        """Send trajectory request via /sample_trajectory_relay.

        Uses separate multipart files for each conditioning tensor,
        avoiding the need to merge safetensors blobs (which requires torch).
        Returns raw safetensors bytes containing the final latent.
        """
        files: dict[str, tuple] = {
            "params": ("params.json", json.dumps(params), "application/json"),
            "pos_cond_st": ("pos_cond.st", pos_cond_bytes, "application/octet-stream"),
            "neg_cond_st": ("neg_cond.st", neg_cond_bytes, "application/octet-stream"),
        }
        if clean_latent_bytes is not None:
            files["clean_latent_st"] = (
                "clean_latent.st", clean_latent_bytes, "application/octet-stream"
            )

        resp = self._client.post("/sample_trajectory_relay", files=files)
        if resp.status_code != 200:
            self._raise_server_error("sample_trajectory_relay", resp)
        return resp.content

    # ------------------------------------------------------------------
    # Queue-based generation (enqueue + SSE stream)
    # ------------------------------------------------------------------

    def enqueue_generation(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: int = -1,
        n_steps: int = 30,
        cfg: float = 4.0,
        width: int = 1280,
        height: int = 832,
        attention_backend: str = "sage",
        sampling_shift: float = 1.0,
        multiplier: float = 1.0,
        denoise: float = 1.0,
    ) -> str:
        """Enqueue a generation job. Returns job_id."""
        resp = self._client.post("/enqueue", json={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "n_steps": n_steps,
            "cfg": cfg,
            "width": width,
            "height": height,
            "attention_backend": attention_backend,
            "sampling_shift": sampling_shift,
            "multiplier": multiplier,
            "denoise": denoise,
        })
        if resp.status_code != 200:
            self._raise_server_error("enqueue", resp)
        return resp.json()["job_id"]

    def stream_job_events(self, job_id: str):
        """Yield SSE events from /stream/{job_id}.

        Yields dicts with 'type' and 'data' keys. Terminal events have
        type 'complete' or 'error'.
        """
        import json

        url = f"{self._base_url}/stream/{job_id}"
        with httpx.stream("GET", url, timeout=self._client.timeout) as resp:
            if resp.status_code != 200:
                raise RuntimeError(
                    f"/stream/{job_id} failed ({resp.status_code})"
                )
            event_type = "message"
            data_buf = ""
            for line in resp.iter_lines():
                if line.startswith("event: "):
                    event_type = line[7:].strip()
                elif line.startswith("data: "):
                    data_buf = line[6:]
                elif line == "" and data_buf:
                    try:
                        parsed = json.loads(data_buf)
                    except json.JSONDecodeError:
                        parsed = {"raw": data_buf}
                    yield {"type": event_type, "data": parsed}
                    if event_type in ("complete", "error"):
                        return
                    event_type = "message"
                    data_buf = ""

    def get_result_png(self, job_id: str) -> tuple[bytes, dict]:
        """Fetch the PNG result of a completed job.

        Returns (png_bytes, metadata).
        """
        import json

        resp = self._client.get(f"/result/{job_id}")
        if resp.status_code != 200:
            self._raise_server_error(f"result/{job_id}", resp)
        metadata_str = resp.headers.get("X-Metadata", "{}")
        try:
            metadata = json.loads(metadata_str)
        except json.JSONDecodeError:
            metadata = {}
        return resp.content, metadata

    def get_queue_status(self) -> dict:
        """Get queue statistics from the inference server."""
        try:
            resp = self._client.get("/queue_status")
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, httpx.ConnectError):
            return {"enabled": False}

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def warmup(
        self,
        attention_backend: str = "sage",
        width: int = 1280,
        height: int = 832,
    ) -> dict[str, Any]:
        """Trigger model warmup on the inference server."""
        try:
            resp = self._client.post("/warmup", json={
                "attention_backend": attention_backend,
                "width": width,
                "height": height,
            })
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, httpx.ConnectError) as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # LoRA lifecycle (JSON relay, torch-free)
    # ------------------------------------------------------------------

    def allocate_adapter(
        self,
        name: str,
        rank: int = 8,
        alpha: float = 16.0,
        layer_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Allocate a LoRA adapter on the inference server."""
        params: dict[str, Any] = {
            "adapter_name": name,
            "rank": rank,
            "alpha": alpha,
        }
        if layer_indices is not None:
            params["layer_indices"] = layer_indices
        resp = self._client.post("/allocate_adapter", json=params)
        if resp.status_code != 200:
            self._raise_server_error("allocate_adapter", resp)
        return resp.json()

    def init_adapter_weights(
        self, name: str, init_b_std: float = 0.0, scale: float = 1.0,
    ) -> dict[str, Any]:
        """(Re-)initialize adapter weights on the server."""
        resp = self._client.post("/init_adapter_weights", json={
            "adapter_name": name,
            "init_b_std": init_b_std,
            "scale": scale,
        })
        if resp.status_code != 200:
            self._raise_server_error("init_adapter_weights", resp)
        return resp.json()

    def set_adapter_config(
        self, name: str, scale: float | None = None, frozen: bool | None = None,
    ) -> dict[str, Any]:
        """Set adapter scale and/or freeze state."""
        params: dict[str, Any] = {"adapter_name": name}
        if scale is not None:
            params["scale"] = scale
        if frozen is not None:
            params["frozen"] = frozen
        resp = self._client.post("/set_adapter_config", json=params)
        if resp.status_code != 200:
            self._raise_server_error("set_adapter_config", resp)
        return resp.json()

    def update_lora_weights(self, safetensors_bytes: bytes) -> dict[str, Any]:
        """Push updated LoRA weights (opaque safetensors blob)."""
        resp = self._client.post(
            "/update_lora_weights",
            content=safetensors_bytes,
            headers={"Content-Type": "application/octet-stream"},
        )
        if resp.status_code != 200:
            self._raise_server_error("update_lora_weights", resp)
        return resp.json()

    def get_lora_state_dict(self, name: str | None = None) -> bytes:
        """Get current LoRA weights as opaque safetensors bytes."""
        resp = self._client.post(
            "/get_lora_state_dict",
            json={"adapter_name": name},
        )
        if resp.status_code != 200:
            self._raise_server_error("get_lora_state_dict", resp)
        return resp.content

    # ------------------------------------------------------------------
    # BTRM lifecycle (JSON relay)
    # ------------------------------------------------------------------

    def inject_btrm_head(
        self,
        head_names: list[str],
        logit_cap: float = 10.0,
        lr: float | None = None,
        hidden_dim: int = 3840,
    ) -> dict[str, Any]:
        """Create a BTRM scoring head on the inference server."""
        params: dict[str, Any] = {
            "head_names": head_names,
            "logit_cap": logit_cap,
            "hidden_dim": hidden_dim,
        }
        if lr is not None:
            params["lr"] = lr
        resp = self._client.post("/inject_btrm_head", json=params)
        if resp.status_code != 200:
            self._raise_server_error("inject_btrm_head", resp)
        return resp.json()

    # ------------------------------------------------------------------
    # Training orchestration (server-side background tasks)
    # ------------------------------------------------------------------

    def start_training_run(self, config: dict[str, Any]) -> dict[str, Any]:
        """Start a training run on the server. Returns {run_id, stream_url}."""
        resp = self._client.post("/training/start", json=config)
        if resp.status_code != 200:
            self._raise_server_error("training/start", resp)
        return resp.json()

    def stop_training_run(self, run_id: str) -> dict[str, Any]:
        """Stop a running training run."""
        resp = self._client.post("/training/stop", json={"run_id": run_id})
        if resp.status_code != 200:
            self._raise_server_error("training/stop", resp)
        return resp.json()

    def get_training_status(self, run_id: str | None = None) -> dict[str, Any]:
        """Get training run status."""
        params = {}
        if run_id is not None:
            params["run_id"] = run_id
        resp = self._client.get("/training/status", params=params)
        if resp.status_code != 200:
            self._raise_server_error("training/status", resp)
        return resp.json()

    def stream_training_events(self, run_id: str):
        """Yield SSE events from a training run. Same protocol as inference SSE."""
        url = f"{self._base_url}/training/stream/{run_id}"
        with httpx.stream("GET", url, timeout=self._client.timeout) as resp:
            if resp.status_code != 200:
                raise RuntimeError(
                    f"/training/stream/{run_id} failed ({resp.status_code})"
                )
            event_type = "message"
            data_buf = ""
            for line in resp.iter_lines():
                if line.startswith("event: "):
                    event_type = line[7:].strip()
                elif line.startswith("data: "):
                    data_buf = line[6:]
                elif line == "" and data_buf:
                    try:
                        parsed = json.loads(data_buf)
                    except json.JSONDecodeError:
                        parsed = {"raw": data_buf}
                    yield {"type": event_type, "data": parsed}
                    if event_type in ("complete", "error"):
                        return
                    event_type = "message"
                    data_buf = ""

    def get_training_artifacts(
        self, run_id: str, path: str,
    ) -> bytes:
        """Fetch a training artifact (PNG chart, JSONL, JSON, etc.)."""
        resp = self._client.get(f"/training/artifacts/{run_id}/{path}")
        if resp.status_code != 200:
            self._raise_server_error(f"training/artifacts/{run_id}/{path}", resp)
        return resp.content

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._client.close()

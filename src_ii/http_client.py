"""HTTP inference client: drop-in replacement for futudiffu.client.InferenceClient.

Uses httpx to talk to the FastAPI server instead of ZMQ.
Same public API as InferenceClient so scripts can switch with minimal changes.

Key differences from the ZMQ client:
  - No socket state machine. No REQ/REP poisoning. No _reset_socket().
  - Timeouts produce httpx.TimeoutException (catchable) not stuck sockets.
  - Tensor transport via safetensors bytes over HTTP body, not ZMQ frames.
  - Server is killable and restartable without client-side state corruption.

Import constraints:
  - httpx for HTTP (module-level)
  - torch and safetensors imported LAZILY (only when tensor ops are needed)
  - This allows the client to be imported and constructed on headless/remote
    machines without a torch installation (e.g., for status/free/warmup RPCs)
  - DOES NOT import from src.futudiffu
"""

from __future__ import annotations

import io
import json as _json
from typing import Any, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    import torch


def _tensor_to_st_bytes(tensors: dict[str, torch.Tensor]) -> bytes:
    """Serialize tensor dict to safetensors bytes."""
    from safetensors.torch import save as st_save
    buf = io.BytesIO()
    st_save(tensors, buf)
    return buf.getvalue()


def _st_bytes_to_tensors(data: bytes) -> dict[str, torch.Tensor]:
    """Deserialize safetensors bytes to tensor dict."""
    from safetensors.torch import load as st_load
    return st_load(data)


class HTTPInferenceClient:
    """HTTP client for the futudiffu FastAPI inference server.

    Drop-in replacement for InferenceClient. All methods have the same
    signatures and return types.

    Constructor does NOT import torch. Only methods that handle tensors
    (encode_prompt, sample_trajectory, vae_encode/decode, etc.) trigger
    lazy torch imports. Status/free/warmup/lifecycle RPCs work without torch.

    Args:
        base_url: Server URL, e.g. "http://localhost:8000".
        timeout_s: Request timeout in seconds. Default 600 (10 min,
            generous for torch.compile warmups).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout_s: float = 600.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_s, connect=30.0),
        )

    def _post_json(self, path: str, json_body: dict | None = None) -> dict:
        """POST JSON, return parsed JSON response."""
        resp = self._client.post(path, json=json_body or {})
        resp.raise_for_status()
        return resp.json()

    def _post_st(self, path: str, tensors: dict[str, Any]) -> bytes:
        """POST safetensors bytes, return raw response bytes."""
        data = _tensor_to_st_bytes(tensors)
        resp = self._client.post(
            path,
            content=data,
            headers={"Content-Type": "application/octet-stream"},
        )
        resp.raise_for_status()
        return resp.content

    def _post_mixed(
        self,
        path: str,
        params: dict,
        tensors: dict[str, Any] | None = None,
    ) -> tuple[dict, dict]:
        """POST JSON params + safetensors tensors via multipart.

        Returns (result_tensors, metadata).
        """
        files = {"params": ("params.json", _json.dumps(params), "application/json")}
        if tensors:
            st_bytes = _tensor_to_st_bytes(tensors)
            files["tensors"] = ("tensors.st", st_bytes, "application/octet-stream")

        resp = self._client.post(path, files=files)
        resp.raise_for_status()

        # Parse response: either safetensors bytes with metadata header,
        # or JSON with metadata
        content_type = resp.headers.get("content-type", "")
        if "octet-stream" in content_type:
            result_tensors = _st_bytes_to_tensors(resp.content)
            metadata_str = resp.headers.get("X-Metadata", "{}")
            metadata = _json.loads(metadata_str)
            return result_tensors, metadata
        else:
            body = resp.json()
            return {}, body.get("metadata", {})

    # ------------------------------------------------------------------
    # Status / lifecycle (no torch required)
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Get server status."""
        resp = self._client.get("/status")
        resp.raise_for_status()
        return resp.json()

    def free(self, model: str = "all") -> None:
        """Free specified model(s) on server.

        Args:
            model: "te", "diffusion", "vae", or "all".
        """
        self._post_json("/free", {"model": model})

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def encode_prompt(self, prompt: str, layer_idx: int = -2) -> torch.Tensor:
        """Encode a text prompt to conditioning tensor.

        Args:
            prompt: Text to encode (empty string for negative).
            layer_idx: Hidden layer index, default -2.

        Returns:
            Conditioning tensor (1, seq_len, 2560) on CPU.
        """
        resp = self._client.post(
            "/encode_prompt",
            json={"prompt": prompt, "layer_idx": layer_idx},
        )
        resp.raise_for_status()
        tensors = _st_bytes_to_tensors(resp.content)
        return tensors["conditioning"]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_trajectory(
        self,
        pos_cond: torch.Tensor,
        neg_cond: torch.Tensor,
        seed: int,
        n_steps: int,
        cfg: float = 4.0,
        width: int = 1280,
        height: int = 832,
        attention_backend: str = "sdpa",
        sampling_shift: float = 1.0,
        multiplier: float = 1.0,
        save_steps: list[int] | None = None,
        denoise: float = 1.0,
        clean_latent: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        score_at_step: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run a diffusion sampling trajectory with optional inline BTRM scoring.

        Args:
            pos_cond: Positive conditioning (1, seq, 2560).
            neg_cond: Negative conditioning (1, seq, 2560).
            seed: RNG seed (ignored if noise is provided).
            n_steps: Number of euler steps.
            cfg: CFG scale.
            width: Image width.
            height: Image height.
            attention_backend: "sdpa" or "sage".
            sampling_shift: Default 1.0.
            multiplier: Default 1.0.
            save_steps: Steps to save intermediates. None = all.
            denoise: Denoise strength for i2i (0-1). Default 1.0 (t2i).
            clean_latent: For i2i, the encoded source image latent.
            noise: Pre-generated noise tensor (1, 16, H/8, W/8).
            score_at_step: If set, score this step's latent inline via BTRM.

        Returns:
            Dict of {name: tensor} with "final" and optionally "step_NN" keys.
            If score_at_step was set, also contains "_btrm_scores".
        """
        req_tensors = {"pos_cond": pos_cond, "neg_cond": neg_cond}
        if clean_latent is not None:
            req_tensors["clean_latent"] = clean_latent
        if noise is not None:
            req_tensors["noise"] = noise

        params = {
            "seed": seed,
            "n_steps": n_steps,
            "cfg": cfg,
            "width": width,
            "height": height,
            "attention_backend": attention_backend,
            "sampling_shift": sampling_shift,
            "multiplier": multiplier,
            "save_steps": save_steps,
            "denoise": denoise,
        }
        if score_at_step is not None:
            params["score_at_step"] = score_at_step

        result_tensors, metadata = self._post_mixed(
            "/sample_trajectory", params, req_tensors,
        )
        if "btrm_scores" in metadata:
            result_tensors["_btrm_scores"] = metadata["btrm_scores"]
        return result_tensors

    def sample_trajectory_packed(
        self,
        pos_conds: list[torch.Tensor],
        neg_cond: torch.Tensor,
        seeds: list[int],
        n_steps: int,
        cfg: float = 4.0,
        width: int | None = None,
        height: int | None = None,
        widths: list[int] | None = None,
        heights: list[int] | None = None,
        attention_backend: str = "sdpa",
        sampling_shift: float | None = None,
        sampling_shifts: list[float] | None = None,
        multiplier: float = 1.0,
        save_steps: list[int] | None = None,
        denoise: float = 1.0,
        clean_latents: list[torch.Tensor] | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """Run N packed diffusion trajectories via FlexAttention.

        Supports both uniform resolution (width/height) and mixed resolution
        (widths/heights lists). Each image can have its own resolution and
        corresponding sigma schedule shift (SD3 Eq.23).

        Args:
            pos_conds: N positive conditionings, each (1, seq_i, 2560).
            neg_cond: Shared negative conditioning (1, seq, 2560).
            seeds: N RNG seeds.
            n_steps: Number of euler steps (shared by all images).
            cfg: CFG scale.
            width: Uniform image width (used if widths is None).
            height: Uniform image height (used if heights is None).
            widths: Per-image widths. Takes precedence over width.
            heights: Per-image heights. Takes precedence over height.
            attention_backend: "sdpa" or "sage".
            sampling_shift: Uniform sigma shift override.
            sampling_shifts: Per-image sigma shifts.
            multiplier: Timestep multiplier.
            save_steps: Steps to save intermediates.
            denoise: Denoise strength (0-1). Default 1.0 (t2i).
            clean_latents: N optional source image latents for i2i.

        Returns:
            List of N dicts, each {name: tensor} with "final" and "step_NN" keys.
        """
        n_images = len(pos_conds)
        req_tensors: dict[str, Any] = {"neg_cond": neg_cond}
        for i, pc in enumerate(pos_conds):
            req_tensors[f"pos_cond_{i}"] = pc
        if clean_latents is not None:
            for i, cl in enumerate(clean_latents):
                if cl is not None:
                    req_tensors[f"clean_latent_{i}"] = cl

        params: dict = {
            "n_images": n_images,
            "seeds": seeds,
            "n_steps": n_steps,
            "cfg": cfg,
            "attention_backend": attention_backend,
            "multiplier": multiplier,
            "save_steps": save_steps,
            "denoise": denoise,
        }

        # Resolution: per-image lists take precedence over scalar
        if widths is not None and heights is not None:
            params["widths"] = widths
            params["heights"] = heights
        elif width is not None and height is not None:
            params["width"] = width
            params["height"] = height
        else:
            # Default to reference resolution
            params["width"] = 1280
            params["height"] = 832

        # Sigma shift: per-image list takes precedence
        if sampling_shifts is not None:
            params["sampling_shifts"] = sampling_shifts
        elif sampling_shift is not None:
            params["sampling_shift"] = sampling_shift
        # else: server auto-computes from resolution

        result_tensors, metadata = self._post_mixed(
            "/sample_trajectory_packed", params, req_tensors,
        )

        # Unpack per-image results
        results: list[dict] = [{} for _ in range(n_images)]
        for key, tensor in result_tensors.items():
            if key.startswith("final_"):
                img_idx = int(key.split("_")[1])
                results[img_idx]["final"] = tensor
            elif key.startswith("step_"):
                # step_NN_I format
                parts = key.split("_")
                step_num = parts[1]
                img_idx = int(parts[2])
                results[img_idx][f"step_{step_num}"] = tensor
        return results

    # ------------------------------------------------------------------
    # VAE
    # ------------------------------------------------------------------

    def vae_encode(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latent.

        Args:
            image: (1, 3, H, W) in [0, 1] range.

        Returns:
            Latent tensor (1, 16, H/8, W/8) on CPU.
        """
        result_bytes = self._post_st("/vae_encode", {"image": image})
        tensors = _st_bytes_to_tensors(result_bytes)
        return tensors["latent"]

    def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to image.

        Args:
            latent: (1, 16, H, W).

        Returns:
            Image tensor (1, 3, H*8, W*8) in [0, 1] on CPU.
        """
        result_bytes = self._post_st("/vae_decode", {"latent": latent})
        tensors = _st_bytes_to_tensors(result_bytes)
        return tensors["image"]

    def vae_decode_png(self, latent: torch.Tensor) -> bytes:
        """Decode latent to PNG bytes.

        Args:
            latent: (1, 16, H, W).

        Returns:
            Raw PNG bytes.
        """
        data = _tensor_to_st_bytes({"latent": latent})
        resp = self._client.post(
            "/vae_decode_png",
            content=data,
            headers={"Content-Type": "application/octet-stream"},
        )
        resp.raise_for_status()
        return resp.content

    def vae_encode_png(self, png_bytes: bytes) -> torch.Tensor:
        """Encode a PNG image to latent.

        Args:
            png_bytes: Raw PNG file bytes.

        Returns:
            Latent tensor (1, 16, H/8, W/8) on CPU.
        """
        resp = self._client.post(
            "/vae_encode_png",
            content=png_bytes,
            headers={"Content-Type": "image/png"},
        )
        resp.raise_for_status()
        tensors = _st_bytes_to_tensors(resp.content)
        return tensors["latent"]

    # ------------------------------------------------------------------
    # Warmup (no torch required)
    # ------------------------------------------------------------------

    def warmup(self, attention_backend: str = "sdpa",
               width: int = 1280, height: int = 832) -> None:
        """Warmup the diffusion model for a given attention backend and resolution."""
        self._post_json("/warmup", {
            "attention_backend": attention_backend,
            "width": width,
            "height": height,
        })

    def warmup_packed(self, n_images: int = 2) -> None:
        """Warmup the packed forward path (FlexAttention + torch.compile)."""
        self._post_json("/warmup_packed", {"n_images": n_images})

    # ------------------------------------------------------------------
    # LoRA management
    # ------------------------------------------------------------------

    def allocate_adapter(
        self,
        adapter_name: str,
        rank: int = 8,
        alpha: float = 16.0,
        layer_indices: list[int] | None = None,
    ) -> int:
        """Allocate adapter slots -- graph-mutating, NO recompile.

        Call for all adapters BEFORE warmup(). Adapter starts silent
        (scale=0, zero weights). Use set_adapter_config() to activate
        and init_adapter_weights() to prepare for training.

        Returns:
            Number of adapter slots allocated.
        """
        params: dict = {
            "adapter_name": adapter_name,
            "rank": rank,
            "alpha": alpha,
        }
        if layer_indices is not None:
            params["layer_indices"] = layer_indices
        result = self._post_json("/allocate_adapter", params)
        return result.get("metadata", {}).get("n_adapters", 0)

    def init_adapter_weights(
        self,
        adapter_name: str,
        init_b_std: float = 0.0,
        scale: float = 1.0,
    ) -> int:
        """(Re-)initialize adapter weights -- graph-invariant, safe after compile.

        Returns:
            Number of modules initialized.
        """
        result = self._post_json("/init_adapter_weights", {
            "adapter_name": adapter_name,
            "init_b_std": init_b_std,
            "scale": scale,
        })
        return result.get("metadata", {}).get("n_modules_initialized", 0)

    def inject_lora(
        self,
        adapter_name: str,
        rank: int = 8,
        alpha: float = 16.0,
        layer_indices: list[int] | None = None,
        init_b_std: float = 0.0,
    ) -> int:
        """Legacy: allocate + init + recompile in one call.

        Prefer allocate_adapter() + init_adapter_weights() for new code.
        """
        params: dict = {
            "adapter_name": adapter_name,
            "rank": rank,
            "alpha": alpha,
        }
        if layer_indices is not None:
            params["layer_indices"] = layer_indices
        if init_b_std > 0:
            params["init_b_std"] = init_b_std
        result = self._post_json("/inject_lora", params)
        return result.get("metadata", {}).get("n_adapters", 0)

    def update_lora_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Push updated LoRA weights to the server (hot path).

        Uses .data.copy_() on the server side -- no recompilation needed.

        Args:
            state_dict: Dict with keys like "path.adapters.name.lora_A".
        """
        data = _tensor_to_st_bytes(state_dict)
        resp = self._client.post(
            "/update_lora_weights",
            content=data,
            headers={"Content-Type": "application/octet-stream"},
        )
        resp.raise_for_status()

    def set_adapter_config(
        self,
        adapter_name: str,
        scale: float | list[float] | None = None,
        frozen: bool | None = None,
        clear_scale: bool = False,
    ) -> None:
        """Set adapter scale and/or freeze state on the server.

        Args:
            adapter_name: Which adapter to configure.
            scale: Per-batch scale value(s). None = don't change.
            frozen: If True, freeze the adapter. If False, unfreeze.
            clear_scale: If True, explicitly clear the scale (reset to 1.0).
        """
        params: dict = {"adapter_name": adapter_name}
        if scale is not None:
            params["scale"] = scale
        elif clear_scale:
            params["clear_scale"] = True
        if frozen is not None:
            params["frozen"] = frozen
        self._post_json("/set_adapter_config", params)

    # ------------------------------------------------------------------
    # Training support
    # ------------------------------------------------------------------

    def get_lora_state_dict(
        self,
        adapter_name: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Get current LoRA weights from the server.

        Args:
            adapter_name: Filter by adapter name. None = all.

        Returns:
            Dict of LoRA weight tensors on CPU.
        """
        resp = self._client.post(
            "/get_lora_state_dict",
            json={"adapter_name": adapter_name},
        )
        resp.raise_for_status()
        return _st_bytes_to_tensors(resp.content)

    def inject_btrm_head(
        self,
        head_names: list[str] | tuple[str, ...] = ("bit_quality", "step_quality"),
        logit_cap: float = 10.0,
        lr: float | None = None,
        weight_decay: float = 0.0,
        hidden_dim: int = 3840,
    ) -> dict:
        """Create a BTRM scoring head on the server.

        The head is ~30KB and stays on GPU permanently.

        Args:
            head_names: Names for each scoring head.
            logit_cap: Soft tanh cap magnitude.
            lr: If set, create an Adam optimizer for BTRM training.
            weight_decay: Weight decay for Adam.
            hidden_dim: Backbone hidden dimension.

        Returns:
            Metadata dict with n_heads, n_params.
        """
        params: dict = {
            "hidden_dim": hidden_dim,
            "head_names": list(head_names),
            "logit_cap": logit_cap,
        }
        if lr is not None:
            params["lr"] = lr
            params["weight_decay"] = weight_decay
        result = self._post_json("/inject_btrm_head", params)
        return result.get("metadata", {})

    def score_btrm(
        self,
        latent: torch.Tensor,
        sigma: torch.Tensor,
        conditioning: torch.Tensor,
        attention_backend: str = "sdpa",
        multiplier: float = 1.0,
    ) -> list[list[float]]:
        """Score latents via backbone + BTRM head on server.

        Args:
            latent: (B, 16, H, W) noisy latent.
            sigma: (B,) sigma values.
            conditioning: (B, seq, dim) text conditioning.
            attention_backend: "sdpa" or "sage".
            multiplier: Default 1.0.

        Returns:
            List of per-example score lists: [[head0, head1], ...].
        """
        st_bytes = _tensor_to_st_bytes({
            "latent": latent,
            "sigma": sigma,
            "conditioning": conditioning,
        })
        files = {
            "params": ("params.json", _json.dumps({
                "attention_backend": attention_backend,
                "multiplier": multiplier,
            }), "application/json"),
            "tensors": ("tensors.st", st_bytes, "application/octet-stream"),
        }
        resp = self._client.post("/score_btrm", files=files)
        resp.raise_for_status()
        body = resp.json()
        return body["metadata"]["scores"]

    def train_btrm_step(
        self,
        examples: list[dict],
        logsquare_weight: float = 0.1,
        attention_backend: str = "sdpa",
        multiplier: float = 1.0,
    ) -> dict:
        """One BTRM optimizer step from labeled examples.

        The server does backbone forward + BTRM scoring + BT loss + backward + step.

        Args:
            examples: List of dicts, each with:
                latent: (1, 16, H, W) noisy latent.
                sigma: (1,) or scalar sigma value.
                conditioning: (1, seq, dim) text conditioning.
                head_idx: Which head this example trains.
                is_positive: Whether positive.
            logsquare_weight: Logsquare regularization weight.
            attention_backend: "sdpa" or "sage".
            multiplier: Default 1.0.

        Returns:
            Metadata dict with loss, bt_loss, logsq_loss, per_head_accuracy, etc.
        """
        import torch
        labels = []
        req_tensors: dict[str, Any] = {}
        for i, ex in enumerate(examples):
            req_tensors[f"latent_{i}"] = ex["latent"]
            sigma = ex["sigma"]
            if not isinstance(sigma, torch.Tensor):
                sigma = torch.tensor([sigma])
            req_tensors[f"sigma_{i}"] = sigma
            req_tensors[f"conditioning_{i}"] = ex["conditioning"]
            labels.append({
                "head_idx": ex["head_idx"],
                "is_positive": ex["is_positive"],
            })

        st_bytes = _tensor_to_st_bytes(req_tensors)
        params = {
            "labels": labels,
            "logsquare_weight": logsquare_weight,
            "attention_backend": attention_backend,
            "multiplier": multiplier,
        }
        files = {
            "params": ("params.json", _json.dumps(params), "application/json"),
            "tensors": ("tensors.st", st_bytes, "application/octet-stream"),
        }
        resp = self._client.post("/train_btrm_step", files=files)
        resp.raise_for_status()
        return resp.json().get("metadata", {})

    def accumulate_policy_gradients(
        self,
        checkpoints: dict[int, torch.Tensor],
        sigmas: torch.Tensor,
        conditioning: torch.Tensor,
        adapter_name: str,
        advantage: float,
        multiplier: float = 1.0,
        eta_used: list[float] | None = None,
        beta: float = 0.04,
    ) -> dict:
        """Accumulate LoRA gradients on the server for a rollout.

        Gradients stay on server (no tensor output). Call this K times
        (once per rollout), then call policy_optimizer_step once.

        Args:
            checkpoints: {step_idx: x_t tensor} at sparse steps.
            sigmas: (n_steps+1,) full sigma schedule.
            conditioning: (1, seq, dim) text conditioning (positive only).
            adapter_name: Which LoRA adapter to differentiate.
            advantage: Advantage weight for this rollout.
            multiplier: Default 1.0.
            eta_used: Per-step noise scales from SDE generation. Required
                for correct η_t² denominator in flow-matching log-prob.
            beta: KL penalty coefficient for DDGRPO (default 0.04).

        Returns:
            Metadata dict with total_log_ratio, total_kl, n_steps.
        """
        req_tensors: dict[str, Any] = {
            "sigmas": sigmas,
            "conditioning": conditioning,
        }
        sparse_steps = sorted(checkpoints.keys())
        for step_idx, tensor in checkpoints.items():
            req_tensors[f"checkpoint_{step_idx}"] = tensor

        st_bytes = _tensor_to_st_bytes(req_tensors)
        params: dict[str, Any] = {
            "adapter_name": adapter_name,
            "sparse_steps": sparse_steps,
            "advantage": advantage,
            "multiplier": multiplier,
            "beta": beta,
        }
        if eta_used is not None:
            params["eta_used"] = eta_used
        files = {
            "params": ("params.json", _json.dumps(params), "application/json"),
            "tensors": ("tensors.st", st_bytes, "application/octet-stream"),
        }
        resp = self._client.post("/accumulate_policy_gradients", files=files)
        resp.raise_for_status()
        return resp.json().get("metadata", {})

    def policy_optimizer_step(
        self,
        adapter_name: str,
        max_grad_norm: float = 1.0,
        lr: float = 1e-4,
    ) -> dict:
        """Clip gradients, step optimizer, zero gradients on server.

        First call creates the optimizer lazily from LoRA params.

        Args:
            adapter_name: Which LoRA adapter to step.
            max_grad_norm: Gradient clipping norm.
            lr: Learning rate (used only on first call to create optimizer).

        Returns:
            Metadata dict with grad_norm, n_params.
        """
        result = self._post_json("/policy_optimizer_step", {
            "adapter_name": adapter_name,
            "max_grad_norm": max_grad_norm,
            "lr": lr,
        })
        return result.get("metadata", {})

    def dump_all_loras(self, output_dir: str = "lora_dumps") -> dict:
        """Emergency dump: save all LoRA adapters on the server to disk.

        The server writes safetensors files + a JSON manifest.

        Args:
            output_dir: Directory for dump files (server-side path).

        Returns:
            Metadata dict with "files" list and "manifest" path.
        """
        result = self._post_json("/dump_all_loras", {"output_dir": output_dir})
        return result.get("metadata", {})

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Close the HTTP connection."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

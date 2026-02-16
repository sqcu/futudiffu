"""Inference client: thin wrapper over ZeroMQ + protocol for talking to the server.

Usage:
    client = InferenceClient("tcp://localhost:5555")
    cond = client.encode_prompt("a laser shark")
    neg = client.encode_prompt("")
    result = client.sample_trajectory(cond, neg, seed=42, n_steps=30, ...)
    image = client.vae_decode(result["final"])
"""

import torch
import zmq

from .protocol import pack_request, unpack_response


class InferenceClient:
    """Client for the futudiffu inference server."""

    def __init__(self, endpoint: str = "tcp://localhost:5555", timeout_ms: int = 0):
        """Connect to an inference server.

        Args:
            endpoint: ZMQ endpoint (e.g. "tcp://localhost:5555").
            timeout_ms: Receive timeout in ms. 0 = infinite (block forever).
        """
        self._ctx = zmq.Context()
        self._endpoint = endpoint
        self._timeout_ms = timeout_ms
        self._socket = self._make_socket()

    def _make_socket(self) -> zmq.Socket:
        """Create and connect a fresh REQ socket."""
        sock = self._ctx.socket(zmq.REQ)
        if self._timeout_ms > 0:
            sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        sock.connect(self._endpoint)
        return sock

    def _reset_socket(self):
        """Destroy the poisoned socket and create a fresh one.

        ZMQ REQ sockets get stuck in a broken state if recv times out
        after send (the socket is waiting for a reply that will never come).
        The only recovery is to close and reconnect.
        """
        self._socket.close(linger=0)
        self._socket = self._make_socket()

    def _call(self, method: str, params: dict | None = None,
              tensors: dict[str, torch.Tensor] | None = None) -> tuple[dict[str, torch.Tensor], dict]:
        """Send a request and wait for the response.

        Returns:
            (tensors, metadata) on success.

        Raises:
            RuntimeError: If server returns an error.
            zmq.Again: If timeout_ms > 0 and server doesn't respond in time.
                The socket is automatically reset so the next call can proceed.
        """
        frames = pack_request(method, params, tensors)
        self._socket.send_multipart(frames)
        try:
            response_frames = self._socket.recv_multipart()
        except zmq.Again:
            self._reset_socket()
            raise
        status, resp_tensors, metadata = unpack_response(response_frames)
        if status != "ok":
            raise RuntimeError(f"Server error: {metadata.get('error', 'unknown')}")
        return resp_tensors, metadata

    def encode_prompt(self, prompt: str, layer_idx: int = -2) -> torch.Tensor:
        """Encode a text prompt to conditioning tensor.

        Args:
            prompt: Text to encode (empty string for negative).
            layer_idx: Hidden layer index, default -2.

        Returns:
            Conditioning tensor (1, seq_len, 2560) on CPU.
        """
        tensors, _ = self._call("encode_prompt", {
            "prompt": prompt,
            "layer_idx": layer_idx,
        })
        return tensors["conditioning"]

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
            noise: Pre-generated noise tensor (1, 16, H/8, W/8). If provided,
                bypasses torch.randn entirely — use for cross-version
                reproducibility with a canonical noise tensor.
            score_at_step: If set (and BTRM head is injected on server),
                score this step's latent inline during the trajectory. The
                result dict will contain a "_btrm_scores" key with
                [[head0, head1, ...], ...] scores.

        Returns:
            Dict of {name: tensor} with "final" and optionally "step_NN" keys.
            If score_at_step was set and BTRM head is available, also contains
            "_btrm_scores" (list, not tensor). All tensors on CPU.
        """
        req_tensors = {"pos_cond": pos_cond, "neg_cond": neg_cond}
        if clean_latent is not None:
            req_tensors["clean_latent"] = clean_latent
        if noise is not None:
            req_tensors["noise"] = noise

        req_params = {
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
            req_params["score_at_step"] = score_at_step

        tensors, metadata = self._call("sample_trajectory", req_params, req_tensors)
        if "btrm_scores" in metadata:
            tensors["_btrm_scores"] = metadata["btrm_scores"]
        return tensors

    def sample_trajectory_packed(
        self,
        pos_conds: list[torch.Tensor],
        neg_cond: torch.Tensor,
        seeds: list[int],
        n_steps: int,
        cfg: float = 4.0,
        width: int = 1280,
        height: int = 832,
        attention_backend: str = "sdpa",
        sampling_shift: float = 1.0,
        multiplier: float = 1.0,
        save_steps: list[int] | None = None,
        denoise: float = 1.0,
        clean_latents: list[torch.Tensor] | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """Run N diffusion trajectories packed via FlexAttention.

        All trajectories share the same schedule (n_steps, denoise, cfg, size).
        Each has its own prompt (pos_cond) and seed.

        Args:
            pos_conds: N positive conditionings, each (1, seq_i, 2560).
            neg_cond: Shared negative conditioning (1, seq, 2560).
            seeds: N RNG seeds.
            n_steps, cfg, width, height, sampling_shift, multiplier, denoise:
                Shared trajectory parameters.
            attention_backend: "sdpa" or "sage".
            save_steps: Steps to save intermediates. None = all.
            clean_latents: N optional source image latents for i2i.

        Returns:
            List of N dicts, each {name: tensor} with "final" and "step_NN" keys.
        """
        n_images = len(pos_conds)
        req_tensors = {"neg_cond": neg_cond}
        for i, pc in enumerate(pos_conds):
            req_tensors[f"pos_cond_{i}"] = pc
        if clean_latents is not None:
            for i, cl in enumerate(clean_latents):
                if cl is not None:
                    req_tensors[f"clean_latent_{i}"] = cl

        resp_tensors, metadata = self._call("sample_trajectory_packed", {
            "n_images": n_images,
            "seeds": seeds,
            "n_steps": n_steps,
            "cfg": cfg,
            "width": width,
            "height": height,
            "attention_backend": attention_backend,
            "sampling_shift": sampling_shift,
            "multiplier": multiplier,
            "save_steps": save_steps,
            "denoise": denoise,
        }, req_tensors)

        # Unpack per-image results
        results = [{} for _ in range(n_images)]
        for key, tensor in resp_tensors.items():
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

    def vae_encode(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latent.

        Args:
            image: (1, 3, H, W) in [0, 1] range.

        Returns:
            Latent tensor (1, 16, H/8, W/8) on CPU.
        """
        tensors, _ = self._call("vae_encode", tensors={"image": image})
        return tensors["latent"]

    def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to image.

        Args:
            latent: (1, 16, H, W).

        Returns:
            Image tensor (1, 3, H*8, W*8) in [0, 1] on CPU.
        """
        tensors, _ = self._call("vae_decode", tensors={"latent": latent})
        return tensors["image"]

    def warmup(self, attention_backend: str = "sdpa",
               width: int = 1280, height: int = 832) -> None:
        """Warmup the diffusion model for a given attention backend and resolution."""
        params: dict = {"attention_backend": attention_backend}
        if width != 1280:
            params["width"] = width
        if height != 832:
            params["height"] = height
        self._call("warmup", params)

    def warmup_packed(self, n_images: int = 2) -> None:
        """Warmup the packed forward path (FlexAttention + torch.compile)."""
        self._call("warmup_packed", {"n_images": n_images})

    def status(self) -> dict:
        """Get server status."""
        _, metadata = self._call("status")
        return metadata

    def free(self, model: str = "all") -> None:
        """Free specified model(s) on server.

        Args:
            model: "te", "diffusion", "vae", or "all".
        """
        self._call("free", {"model": model})

    # ------------------------------------------------------------------
    # LoRA management
    # ------------------------------------------------------------------

    def inject_lora(
        self,
        adapter_name: str,
        rank: int = 8,
        alpha: float = 16.0,
        layer_indices: list[int] | None = None,
        init_b_std: float = 0.0,
    ) -> int:
        """Inject a named LoRA adapter into the server's diffusion model.

        After injection, the client must call warmup() to recompile.

        Args:
            adapter_name: Name for the adapter (e.g. "ptheta").
            rank: LoRA rank.
            alpha: LoRA alpha.
            layer_indices: If set, only inject on these layer indices.
            init_b_std: If > 0, initialize lora_B with N(0, init_b_std)
                instead of zeros. Needed for policy gradient.

        Returns:
            Number of adapters injected.
        """
        params = {
            "adapter_name": adapter_name,
            "rank": rank,
            "alpha": alpha,
        }
        if layer_indices is not None:
            params["layer_indices"] = layer_indices
        if init_b_std > 0:
            params["init_b_std"] = init_b_std
        _, metadata = self._call("inject_lora", params)
        return metadata.get("n_adapters", 0)

    def update_lora_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Push updated LoRA weights to the server (hot path).

        Uses .data.copy_() on the server side -- no recompilation needed.

        Args:
            state_dict: Dict with keys like "path.adapters.name.lora_A".
        """
        self._call("update_lora_weights", tensors=state_dict)

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
            params["scale"] = None
        if frozen is not None:
            params["frozen"] = frozen
        self._call("set_adapter_config", params)

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
        params = {}
        if adapter_name is not None:
            params["adapter_name"] = adapter_name
        tensors, _ = self._call("get_lora_state_dict", params)
        return tensors

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
        _, metadata = self._call("inject_btrm_head", params)
        return metadata

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
        _, metadata = self._call("score_btrm", {
            "attention_backend": attention_backend,
            "multiplier": multiplier,
        }, {
            "latent": latent,
            "sigma": sigma,
            "conditioning": conditioning,
        })
        return metadata["scores"]

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
        labels = []
        req_tensors: dict[str, torch.Tensor] = {}
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

        _, metadata = self._call("train_btrm_step", {
            "labels": labels,
            "logsquare_weight": logsquare_weight,
            "attention_backend": attention_backend,
            "multiplier": multiplier,
        }, req_tensors)
        return metadata

    def accumulate_policy_gradients(
        self,
        checkpoints: dict[int, torch.Tensor],
        sigmas: torch.Tensor,
        conditioning: torch.Tensor,
        adapter_name: str,
        advantage: float,
        multiplier: float = 1.0,
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

        Returns:
            Metadata dict with total_log_ratio, n_steps.
        """
        req_tensors: dict[str, torch.Tensor] = {
            "sigmas": sigmas,
            "conditioning": conditioning,
        }
        sparse_steps = sorted(checkpoints.keys())
        for step_idx, tensor in checkpoints.items():
            req_tensors[f"checkpoint_{step_idx}"] = tensor

        _, metadata = self._call("accumulate_policy_gradients", {
            "adapter_name": adapter_name,
            "sparse_steps": sparse_steps,
            "advantage": advantage,
            "multiplier": multiplier,
        }, req_tensors)
        return metadata

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
        _, metadata = self._call("policy_optimizer_step", {
            "adapter_name": adapter_name,
            "max_grad_norm": max_grad_norm,
            "lr": lr,
        })
        return metadata

    def dump_all_loras(self, output_dir: str = "lora_dumps") -> dict:
        """Emergency dump: save all LoRA adapters on the server to disk.

        The server writes safetensors files + a JSON manifest.

        Args:
            output_dir: Directory for dump files (server-side path).

        Returns:
            Metadata dict with "files" list and "manifest" path.
        """
        _, metadata = self._call("dump_all_loras", {"output_dir": output_dir})
        return metadata

    def close(self):
        """Close the connection."""
        self._socket.close()
        self._ctx.term()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

"""Multi-GPU client: dispatches inference across N InferenceServer instances.

Round-robin dispatch for generation/inference RPCs. Training/mutation RPCs
route to the primary (first) server only. ThreadPoolExecutor for concurrent
trajectory generation.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from .client import InferenceClient


class MultiGPUClient:
    """Wraps N InferenceClient instances for data-parallel trajectory generation."""

    def __init__(
        self,
        endpoints: list[str] | list[int] | list[tuple[str, int]],
        timeout_ms: int = 0,
    ):
        """Connect to N inference servers.

        Args:
            endpoints: One of:
                - List of full ZMQ endpoints: ["tcp://host:5555", ...]
                - List of ports (assumes localhost): [5555, 5556, ...]
                - List of (host, port) tuples: [("host1", 5555), ...]
            timeout_ms: Per-socket receive timeout (0 = infinite).
        """
        resolved = []
        for ep in endpoints:
            if isinstance(ep, str):
                resolved.append(ep)
            elif isinstance(ep, int):
                resolved.append(f"tcp://localhost:{ep}")
            elif isinstance(ep, (tuple, list)) and len(ep) == 2:
                resolved.append(f"tcp://{ep[0]}:{ep[1]}")
            else:
                raise ValueError(f"Bad endpoint spec: {ep!r}")

        self.clients = [InferenceClient(e, timeout_ms=timeout_ms) for e in resolved]
        self._robin = 0
        self._pool = ThreadPoolExecutor(max_workers=len(self.clients))

    @property
    def primary(self) -> InferenceClient:
        """Server 0 -- handles all training/mutation RPCs."""
        return self.clients[0]

    @property
    def n_servers(self) -> int:
        return len(self.clients)

    def _next(self) -> InferenceClient:
        """Round-robin select the next client."""
        c = self.clients[self._robin % len(self.clients)]
        self._robin += 1
        return c

    # ------------------------------------------------------------------
    # Parallel inference dispatch (round-robin)
    # ------------------------------------------------------------------

    def encode_prompt(self, prompt: str, layer_idx: int = -2) -> torch.Tensor:
        """Encode a prompt on the next server (round-robin)."""
        return self._next().encode_prompt(prompt, layer_idx=layer_idx)

    def sample_trajectory(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Sample one trajectory on the next server (round-robin)."""
        return self._next().sample_trajectory(*args, **kwargs)

    def sample_trajectory_packed(self, *args, **kwargs) -> list[dict[str, torch.Tensor]]:
        """Sample packed trajectories on the next server (round-robin)."""
        return self._next().sample_trajectory_packed(*args, **kwargs)

    def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a latent on the next server (round-robin)."""
        return self._next().vae_decode(latent)

    def vae_encode(self, image: torch.Tensor) -> torch.Tensor:
        """Encode an image on the next server (round-robin)."""
        return self._next().vae_encode(image)

    # ------------------------------------------------------------------
    # Parallel batch operations
    # ------------------------------------------------------------------

    def generate_trajectories(
        self,
        jobs: list[dict],
        **shared_kwargs,
    ) -> list[dict[str, torch.Tensor]]:
        """Generate M trajectories across N servers in parallel.

        Each job is a dict of per-trajectory kwargs (pos_cond, neg_cond, seed,
        etc.). shared_kwargs are merged into every job (n_steps, cfg, width,
        height, attention_backend, etc.).

        Args:
            jobs: List of per-trajectory kwarg dicts. Each must contain at
                least pos_cond, neg_cond, and seed.
            **shared_kwargs: Common kwargs applied to all jobs.

        Returns:
            List of result dicts in the same order as jobs.

        Raises:
            RuntimeError: If any job failed. All futures are collected before
                raising, so partial results are not leaked.
        """
        results: list[dict[str, torch.Tensor] | None] = [None] * len(jobs)
        futures = {}
        for i, job_kwargs in enumerate(jobs):
            merged = {**shared_kwargs, **job_kwargs}
            client = self._next()
            fut = self._pool.submit(client.sample_trajectory, **merged)
            futures[fut] = i

        first_error = None
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                if first_error is None:
                    first_error = (idx, e)

        if first_error is not None:
            idx, err = first_error
            raise RuntimeError(f"Job {idx} failed: {err}") from err

        return results  # type: ignore[return-value]

    def warmup(self, attention_backend: str = "sdpa") -> None:
        """Warmup primary server (drop-in compat with InferenceClient)."""
        self.primary.warmup(attention_backend)

    def warmup_all(self, attention_backend: str = "sdpa") -> None:
        """Warmup all servers in parallel."""
        futs = [self._pool.submit(c.warmup, attention_backend) for c in self.clients]
        for f in futs:
            f.result()

    def warmup_packed_all(self, n_images: int = 2) -> None:
        """Warmup packed forward on all servers in parallel."""
        futs = [self._pool.submit(c.warmup_packed, n_images) for c in self.clients]
        for f in futs:
            f.result()

    def status(self) -> dict:
        """Get status from primary server (for drop-in compat with InferenceClient)."""
        return self.primary.status()

    def status_all(self) -> list[dict]:
        """Get status from all servers."""
        futs = [self._pool.submit(c.status) for c in self.clients]
        return [f.result() for f in futs]

    # ------------------------------------------------------------------
    # Primary-only: training / mutation RPCs
    # ------------------------------------------------------------------

    def inject_lora(self, *args, **kwargs) -> int:
        return self.primary.inject_lora(*args, **kwargs)

    def inject_btrm_head(self, *args, **kwargs) -> dict:
        return self.primary.inject_btrm_head(*args, **kwargs)

    def train_btrm_step(self, *args, **kwargs) -> dict:
        return self.primary.train_btrm_step(*args, **kwargs)

    def score_btrm(self, *args, **kwargs) -> list[list[float]]:
        return self.primary.score_btrm(*args, **kwargs)

    def accumulate_policy_gradients(self, *args, **kwargs) -> dict:
        return self.primary.accumulate_policy_gradients(*args, **kwargs)

    def policy_optimizer_step(self, *args, **kwargs) -> dict:
        return self.primary.policy_optimizer_step(*args, **kwargs)

    def dump_all_loras(self, *args, **kwargs) -> dict:
        return self.primary.dump_all_loras(*args, **kwargs)

    def set_adapter_config(self, *args, **kwargs) -> None:
        return self.primary.set_adapter_config(*args, **kwargs)

    def update_lora_weights(self, *args, **kwargs) -> None:
        return self.primary.update_lora_weights(*args, **kwargs)

    def get_lora_state_dict(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        return self.primary.get_lora_state_dict(*args, **kwargs)

    def free(self, *args, **kwargs) -> None:
        return self.primary.free(*args, **kwargs)

    # ------------------------------------------------------------------
    # Weight synchronization
    # ------------------------------------------------------------------

    def sync_lora_to_all(self, adapter_name: str | None = None) -> None:
        """Pull LoRA weights from primary, push to all workers."""
        sd = self.primary.get_lora_state_dict(adapter_name)
        futs = [
            self._pool.submit(c.update_lora_weights, sd)
            for c in self.clients[1:]
        ]
        for f in futs:
            f.result()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Close all connections and shut down the thread pool."""
        self._pool.shutdown(wait=True, cancel_futures=True)
        for c in self.clients:
            c.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

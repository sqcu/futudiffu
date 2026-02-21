"""BTRM trajectory dataset generation: schedule-driven, server-backed.

Generates diffusion trajectories via an inference server (InferenceClient)
and writes them to V2 format (DatasetWriter). Supports mixed-resolution
generation, attention backend variation, provenance tracking, and bin-packed
FlexAttention batches.

Replaces the monolithic scripts/generate_btrm_dataset.py with a composable
library module. The CLI wrapper lives in scripts_ii/generate_btrm_dataset.py.

Outer specification axes affected (from user_dataflow_and_lifecycle_rollup.md):
  2. Trajectory Persistence: V2 parquet + safetensors blobs (sealed, no WIP on disk)
  3. Side-Channel Observability: sampling state hashes (model, adapter, trajectory)
  10. Sequence Packing: bin-packed FlexAttention batches for mixed-resolution

Import constraints:
  - IMPORTS from src_ii: bin_packer, sampling_identity
  - IMPORTS from futudiffu: client (InferenceClient), dataset_v2 (DatasetWriter)
  - DOES NOT import: server, model_manager, diffusion_model, training code
"""

from __future__ import annotations

import json
import random
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .bin_packer import (
    REFERENCE_SEQ_LEN,
    RESOLUTION_TIERS,
    BinPackScheduler,
    build_generation_plan,
    compute_seq_len,
)
from .sigma_schedule import resolution_shift
from .sampling_identity import (
    compute_adapter_set_hash,
    compute_model_state_hash,
    compute_trajectory_hash,
    serialize_active_adapters,
)


# ---------------------------------------------------------------------------
# Graceful interruption
# ---------------------------------------------------------------------------

_interrupted = False


def _install_signal_handler():
    """Install a SIGINT handler that sets the interrupted flag on first press."""
    global _interrupted

    def _handler(signum, frame):
        global _interrupted
        if _interrupted:
            print("\nForce quit.")
            sys.exit(1)
        _interrupted = True
        print("\nInterrupt received. Finishing current trajectory then saving...")

    signal.signal(signal.SIGINT, _handler)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DatasetGenerationConfig:
    """Configuration for a generation run.

    Attributes:
        prompts: List of prompt strings to generate from.
        seeds: List of PRNG seeds. None = generate random seeds.
        resolution_tiers: Which resolution tiers to generate ("full", "medium", "small").
        attention_backends: Which attention backends to use ("sdpa", "sage").
        n_steps: Number of Euler diffusion steps.
        cfg: Classifier-free guidance scale.
        output_dir: Path to write V2 dataset.
        run_name: Human-readable name for this generation run (stored in provenance).
        base_model_hash: Identifier for the base model weights. Placeholder until
            we can hash the actual weights at generation time.
        sparse_steps: Which step indices to save latents for.
        sampling_shift: Sigma schedule shift parameter. None = auto-compute
            per-item from resolution_shift(width, height). Explicit float
            overrides the automatic computation for all items.
        multiplier: Timestep multiplier.
        server_endpoint: ZeroMQ endpoint for the inference server.
        flush_interval: Seal the current blob every N trajectories (limits
            data loss on crash).
        render_count: How many trajectories to VAE-decode and save as PNGs.
        active_adapters: List of adapter descriptors currently active on the
            model. Each dict has "name", "strength", "param_hash". Empty list
            means base model only.
        source_device: Human-readable device identifier (e.g. "rtx4090_0").
    """
    prompts: list[str]
    seeds: list[int] | None = None
    resolution_tiers: list[str] = field(default_factory=lambda: ["full"])
    attention_backends: list[str] = field(default_factory=lambda: ["sdpa", "sage"])
    n_steps: int = 30
    cfg: float = 4.0
    output_dir: str = "btrm_dataset_v2"
    run_name: str = "generation"
    base_model_hash: str = "z_image_v1"
    sparse_steps: list[int] = field(default_factory=lambda: [0, 4, 9, 14, 19, 24, 29])
    sampling_shift: float | None = None
    multiplier: float = 1.0
    server_endpoint: str = "tcp://localhost:5555"
    flush_interval: int = 50
    render_count: int = 6
    active_adapters: list[dict[str, Any]] = field(default_factory=list)
    source_device: str = "unknown"


# ---------------------------------------------------------------------------
# DatasetGenerator
# ---------------------------------------------------------------------------

class DatasetGenerator:
    """Generates BTRM training trajectories and writes to V2 format.

    Lifecycle:
      1. __init__: accept config and client
      2. generate_all(): run the full generation plan
         - Phase 1: Build generation plan (Cartesian product of params)
         - Phase 2: Bin-pack plan items into FlexAttention batches
         - Phase 3: Encode all unique prompts via server
         - Phase 4: Warmup diffusion model for each (backend, resolution) combo
         - Phase 5: Execute bins sequentially, writing V2 after each trajectory
         - Phase 6: Render a sample of trajectories
         - Phase 7: Print throughput profile

    The generator does NOT import torch at class level or call any model code.
    All model interaction happens through InferenceClient RPCs.
    """

    def __init__(self, config: DatasetGenerationConfig, client):
        """Initialize the generator.

        Args:
            config: Generation configuration.
            client: An InferenceClient instance (or compatible interface).
                Must support: encode_prompt, sample_trajectory, free, warmup,
                warmup_packed, status, vae_decode.
        """
        self.config = config
        self.client = client

        # Compute identity hashes once (they don't change during a run)
        adapter_set_hash = compute_adapter_set_hash(config.active_adapters)
        self._model_state_hash = compute_model_state_hash(
            config.base_model_hash, adapter_set_hash,
        )
        self._adapter_set_hash = adapter_set_hash
        self._active_adapters_json = serialize_active_adapters(config.active_adapters)

    def generate_all(self) -> dict[str, Any]:
        """Run the full generation pipeline.

        Returns:
            Summary dict with timing, counts, and efficiency stats.
        """
        global _interrupted
        _install_signal_handler()

        cfg = self.config
        output_dir = Path(cfg.output_dir)

        # ---------------------------------------------------------------
        # Phase 1: Build generation plan
        # ---------------------------------------------------------------
        seeds = cfg.seeds
        if seeds is None:
            rng = random.Random(42)
            # Generate enough seeds for the worst case: each prompt * each tier's
            # rollouts_per_prompt * each backend
            max_rollouts = sum(
                RESOLUTION_TIERS[t]["rollouts_per_prompt"]
                for t in cfg.resolution_tiers
            )
            n_seeds_needed = len(cfg.prompts) * max_rollouts * len(cfg.attention_backends)
            seeds = [rng.randint(0, 2**32 - 1) for _ in range(n_seeds_needed)]

        plan = build_generation_plan(
            prompts=cfg.prompts,
            seeds=seeds,
            resolution_tiers=cfg.resolution_tiers,
            attention_backends=cfg.attention_backends,
            n_steps=cfg.n_steps,
            cfg=cfg.cfg,
            sparse_steps=cfg.sparse_steps,
        )

        print(f"Generation plan: {len(plan)} trajectories")
        print(f"  Prompts: {len(cfg.prompts)}")
        print(f"  Resolution tiers: {cfg.resolution_tiers}")
        print(f"  Attention backends: {cfg.attention_backends}")
        print(f"  Steps: {cfg.n_steps}, CFG: {cfg.cfg}")
        print(f"  Output: {output_dir}")

        # ---------------------------------------------------------------
        # Phase 2: Bin-pack
        # ---------------------------------------------------------------
        scheduler = BinPackScheduler(max_seq_len=REFERENCE_SEQ_LEN)
        bins = scheduler.pack_generation_plan(plan)
        efficiency = scheduler.estimate_efficiency(bins)

        print(f"\nBin packing: {efficiency['n_items']} items -> {efficiency['n_bins']} bins")
        print(f"  Utilization: {efficiency['utilization']:.1%}")
        print(f"  Wasted seq_len: {efficiency['total_wasted']}")

        # ---------------------------------------------------------------
        # Phase 3: Encode prompts
        # ---------------------------------------------------------------
        unique_prompts = sorted(set(item["prompt"] for item in plan))
        print(f"\nEncoding {len(unique_prompts)} unique prompts + 1 negative...")

        timing = {
            "te_encode": 0.0,
            "warmup": 0.0,
            "diffusion": 0.0,
            "vae_decode": 0.0,
            "overhead": 0.0,
        }
        wall_start = time.perf_counter()

        t0 = time.perf_counter()
        neg_cond = self.client.encode_prompt("")
        te_cache: dict[str, torch.Tensor] = {}
        for prompt in unique_prompts:
            te_cache[prompt] = self.client.encode_prompt(prompt)
        timing["te_encode"] = time.perf_counter() - t0
        print(f"  Done ({timing['te_encode']:.1f}s)")

        # Free TE on server (diffusion model will be loaded on next call)
        self.client.free("te")

        # ---------------------------------------------------------------
        # Phase 4: Warmup
        # ---------------------------------------------------------------
        # Collect unique (backend, width, height) combinations
        warmup_configs: set[tuple[str, int, int]] = set()
        for item in plan:
            warmup_configs.add((item["attention_backend"], item["width"], item["height"]))

        for backend, w, h in sorted(warmup_configs):
            print(f"  Warming up {backend} @ {w}x{h}...")
            t0 = time.perf_counter()
            self.client.warmup(attention_backend=backend, width=w, height=h)
            timing["warmup"] += time.perf_counter() - t0

        # Warmup packed forward for each unique bin size
        bin_sizes = sorted(set(len(b) for b in bins))
        packed_sizes = [s for s in bin_sizes if s > 1]
        for ps in packed_sizes:
            print(f"  Warming up packed forward (n_images={ps})...")
            t0 = time.perf_counter()
            self.client.warmup_packed(n_images=ps)
            timing["warmup"] += time.perf_counter() - t0

        # ---------------------------------------------------------------
        # Phase 5: Generate trajectories
        # ---------------------------------------------------------------
        from futudiffu.dataset_v2 import DatasetWriter

        render_queue: list[tuple[torch.Tensor, Path]] = []
        generated_count = 0
        renders_done = 0

        with DatasetWriter(output_dir) as writer:
            for bin_idx, bin_items in enumerate(bins):
                if _interrupted:
                    break

                if len(bin_items) == 1:
                    # Single-image path: use sample_trajectory
                    item = bin_items[0]
                    generated_count += self._generate_single(
                        item, te_cache, neg_cond, writer, timing,
                    )
                else:
                    # Packed path: all items in this bin share a FlexAttention
                    # forward pass. They MUST share n_steps (FlexAttention packing
                    # constraint: all items step in lockstep).
                    #
                    # Group by n_steps within the bin (should be uniform for our
                    # plans, but handle gracefully if not).
                    by_steps: dict[int, list[dict]] = {}
                    for item in bin_items:
                        by_steps.setdefault(item["n_steps"], []).append(item)

                    for n_steps_group, group_items in by_steps.items():
                        if _interrupted:
                            break
                        generated_count += self._generate_packed(
                            group_items, te_cache, neg_cond, writer, timing,
                        )

                # Periodic flush
                if generated_count > 0 and generated_count % cfg.flush_interval == 0:
                    writer.flush()
                    print(f"  [flush] {generated_count} trajectories sealed")

                # Collect renders from the last trajectory in this bin
                # (up to render_count total)
                if renders_done < cfg.render_count and bin_items:
                    last_item = bin_items[-1]
                    # We don't have the tensor in hand after writing to V2.
                    # The render queue is populated from within the generate methods.

            # Final summary
            print(f"\nGenerated {generated_count} trajectories")
            print(f"Total in dataset: {writer.n_trajectories}")

        # ---------------------------------------------------------------
        # Phase 6: Render sample (skipped if no renders queued)
        # ---------------------------------------------------------------
        if render_queue and not _interrupted:
            render_dir = output_dir / "renders"
            render_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nRendering {len(render_queue)} trajectories...")
            for final_tensor, render_path in render_queue:
                t0 = time.perf_counter()
                self._render_latent(final_tensor, render_path)
                timing["vae_decode"] += time.perf_counter() - t0

        # ---------------------------------------------------------------
        # Phase 7: Throughput profile
        # ---------------------------------------------------------------
        wall_total = time.perf_counter() - wall_start
        timing["overhead"] = wall_total - sum(timing.values())

        summary = self._print_throughput_profile(timing, wall_total, generated_count)
        summary["n_bins"] = len(bins)
        summary["packing_efficiency"] = efficiency
        return summary

    # -------------------------------------------------------------------
    # Internal generation methods
    # -------------------------------------------------------------------

    def _generate_single(
        self,
        item: dict,
        te_cache: dict[str, torch.Tensor],
        neg_cond: torch.Tensor,
        writer,
        timing: dict,
    ) -> int:
        """Generate a single trajectory (no packing).

        Returns:
            1 if generated, 0 if interrupted.
        """
        if _interrupted:
            return 0

        cfg = self.config
        prompt = item["prompt"]
        w, h = item["width"], item["height"]

        # Resolve per-item sampling_shift: explicit override or auto from resolution
        if cfg.sampling_shift is not None:
            item_shift = cfg.sampling_shift
        else:
            item_shift = resolution_shift(w, h)

        t0 = time.perf_counter()
        result = self.client.sample_trajectory(
            pos_cond=te_cache[prompt],
            neg_cond=neg_cond,
            seed=item["seed"],
            n_steps=item["n_steps"],
            cfg=item["cfg"],
            width=w,
            height=h,
            attention_backend=item["attention_backend"],
            sampling_shift=item_shift,
            multiplier=cfg.multiplier,
            save_steps=item["sparse_steps"],
        )
        dt = time.perf_counter() - t0
        timing["diffusion"] += dt

        # Compute trajectory hash
        traj_hash = compute_trajectory_hash(
            self._model_state_hash,
            prompt=prompt,
            seed=item["seed"],
            cfg=item["cfg"],
            n_steps=item["n_steps"],
            width=w,
            height=h,
        )

        metadata = self._build_metadata(item, packed=False, timing_seconds=dt,
                                         trajectory_hash=traj_hash,
                                         sampling_shift=item_shift)
        writer.add_trajectory(tensors=result, metadata=metadata)

        label = f"{w}x{h} {item['attention_backend']}"
        print(f"  traj {writer.n_trajectories - 1:06d}: {label} ({dt:.1f}s)")

        return 1

    def _generate_packed(
        self,
        items: list[dict],
        te_cache: dict[str, torch.Tensor],
        neg_cond: torch.Tensor,
        writer,
        timing: dict,
    ) -> int:
        """Generate a packed batch of trajectories via FlexAttention.

        All items must share the same n_steps. They may differ in resolution,
        prompt, seed, and attention_backend. Mixed-resolution bins use
        per-image widths/heights and per-image sigma schedule shifts.

        Returns:
            Number of trajectories generated.
        """
        if _interrupted or not items:
            return 0

        n_steps = items[0]["n_steps"]

        # Compute per-item shifts
        per_item_shifts = []
        for item in items:
            if self.config.sampling_shift is not None:
                per_item_shifts.append(self.config.sampling_shift)
            else:
                per_item_shifts.append(resolution_shift(item["width"], item["height"]))

        pos_conds = [te_cache[item["prompt"]] for item in items]
        seeds = [item["seed"] for item in items]
        widths = [item["width"] for item in items]
        heights = [item["height"] for item in items]

        # Use the first item's attention_backend for the batch
        # (packed forward currently uses a single backend for all items)
        backend = items[0]["attention_backend"]

        t0 = time.perf_counter()
        results = self.client.sample_trajectory_packed(
            pos_conds=pos_conds,
            neg_cond=neg_cond,
            seeds=seeds,
            n_steps=n_steps,
            cfg=items[0]["cfg"],
            widths=widths,
            heights=heights,
            attention_backend=backend,
            sampling_shifts=per_item_shifts,
            multiplier=self.config.multiplier,
            save_steps=items[0]["sparse_steps"],
        )
        dt = time.perf_counter() - t0
        timing["diffusion"] += dt

        dt_per_image = dt / len(items)

        for i, (item, result) in enumerate(zip(items, results)):
            traj_hash = compute_trajectory_hash(
                self._model_state_hash,
                prompt=item["prompt"],
                seed=item["seed"],
                cfg=item["cfg"],
                n_steps=item["n_steps"],
                width=item["width"],
                height=item["height"],
            )

            metadata = self._build_metadata(
                item, packed=True, timing_seconds=dt_per_image,
                trajectory_hash=traj_hash,
                sampling_shift=per_item_shifts[i],
            )
            writer.add_trajectory(tensors=result, metadata=metadata)

        idxs_start = writer.n_trajectories - len(items)
        unique_res = set(zip(widths, heights))
        if len(unique_res) == 1:
            w0, h0 = next(iter(unique_res))
            res_label = f"{w0}x{h0}"
        else:
            res_label = "mixed-res"
        label = f"{len(items)}x {res_label} {backend}"
        print(f"  traj {idxs_start:06d}..{writer.n_trajectories - 1:06d}: "
              f"packed {label} ({dt:.1f}s, {dt_per_image:.1f}s/img)")

        return len(items)

    def _build_metadata(
        self,
        item: dict,
        packed: bool,
        timing_seconds: float,
        trajectory_hash: str,
        sampling_shift: float = 1.0,
    ) -> dict[str, Any]:
        """Build V2 metadata dict for one trajectory.

        Args:
            item: Generation plan item dict.
            packed: Whether this was part of a packed batch.
            timing_seconds: Wall time for this trajectory.
            trajectory_hash: Full trajectory identity hash.
            sampling_shift: The sigma schedule shift used for this trajectory.

        Returns:
            Metadata dict compatible with DatasetWriter.add_trajectory().
        """
        cfg = self.config
        return {
            "prompt": item["prompt"],
            "prompt_idx": item.get("prompt_idx", -1),
            "seed": item["seed"],
            "cfg": item["cfg"],
            "width": item["width"],
            "height": item["height"],
            "n_steps": item["n_steps"],
            "attention_backend": item["attention_backend"],
            "batch_type": item.get("batch_type", "t2i"),
            "denoise": item.get("denoise"),
            "image_file": item.get("image_file"),
            "is_gold": (
                item["attention_backend"] == "sdpa"
                and item["n_steps"] == 30
            ),
            "batch_idx": 0,
            "packed": packed,
            "timing_seconds": timing_seconds,
            "sampling_shift": sampling_shift,
            # Provenance / identity hashes
            "model_state_hash": self._model_state_hash,
            "base_model_hash": cfg.base_model_hash,
            "adapter_set_hash": self._adapter_set_hash,
            "trajectory_hash": trajectory_hash,
            "active_adapters": self._active_adapters_json,
        }

    def _render_latent(self, latent: torch.Tensor, output_path: Path) -> None:
        """VAE-decode a latent and save as PNG.

        Args:
            latent: (1, 16, H, W) latent tensor on CPU.
            output_path: Path for the output PNG file.
        """
        from src_ii.rendering import save_tensor_as_png

        output_path.parent.mkdir(parents=True, exist_ok=True)
        # client.vae_decode returns (1, 3, H*8, W*8) in [0, 1]
        image = self.client.vae_decode(latent)
        save_tensor_as_png(image, output_path)

    def _print_throughput_profile(
        self,
        timing: dict[str, float],
        wall_total: float,
        generated_count: int,
    ) -> dict[str, Any]:
        """Print throughput summary and return summary dict."""
        print(f"\n{'=' * 60}")
        print(f"  THROUGHPUT PROFILE ({generated_count} trajectories)")
        print(f"{'=' * 60}")
        print(f"  {'Phase':<20} {'Time (s)':>10} {'%':>6}")
        print(f"  {'-' * 38}")
        for phase, t in sorted(timing.items(), key=lambda x: -x[1]):
            if t > 0.01:
                print(f"  {phase:<20} {t:>10.1f} {100 * t / wall_total:>5.1f}%")
        print(f"  {'-' * 38}")
        print(f"  {'TOTAL':<20} {wall_total:>10.1f}")

        summary: dict[str, Any] = {
            "generated_count": generated_count,
            "wall_total": wall_total,
            "timing": dict(timing),
        }

        if generated_count > 0:
            avg = timing["diffusion"] / generated_count
            imgs_per_min = 60.0 / (wall_total / generated_count)
            gpu_active = timing["diffusion"] + timing["vae_decode"] + timing["te_encode"]
            print(f"\n  Avg diffusion: {avg:.1f}s/trajectory")
            print(f"  Throughput:    {imgs_per_min:.2f} images/min")
            print(f"  GPU util:      {100 * gpu_active / wall_total:.1f}%")
            summary["avg_diffusion_s"] = avg
            summary["throughput_imgs_per_min"] = imgs_per_min
            summary["gpu_utilization"] = gpu_active / wall_total

        print(f"{'=' * 60}")
        return summary

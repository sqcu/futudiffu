"""Server-side training orchestration for BTRM and DDGRPO runs.

Wraps existing training functions (btrm_training, btrm_lifecycle, policy_step,
ddreinforce) into a lifecycle manager with SSE event streaming. The orchestrator
runs on the GPU server process and accesses the model directly — no tensor
round-tripping through HTTP.

Lifecycle: configure → start → stream metrics → stop/complete → artifacts.
One active run at a time (single-GPU constraint).

Import constraints:
  - torch imported LAZILY (only inside methods that need it)
  - src_ii training modules imported LAZILY (inside start methods)
  - This allows the module to be imported by server.py without triggering
    GPU initialization at import time
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger("futudiffu.training_orchestrator")


class TrainingOrchestrator:
    """Manages background training runs on the GPU server.

    One active run at a time. SSE events are published to subscriber queues.
    """

    def __init__(self):
        self._run_id: str | None = None
        self._task: asyncio.Task | None = None
        self._subscribers: list[asyncio.Queue] = []
        self._status: dict[str, Any] = {"active": False, "phase": "idle"}
        self._stop_requested = False
        self._output_dir: Path | None = None

    # ------------------------------------------------------------------
    # SSE event publishing
    # ------------------------------------------------------------------

    def _publish(self, event_type: str, data: dict[str, Any]) -> None:
        """Push an SSE event to all subscribers."""
        event = {"type": event_type, "data": data}
        dead = []
        for i, q in enumerate(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(i)
        for i in reversed(dead):
            self._subscribers.pop(i)

    def subscribe(self) -> asyncio.Queue:
        """Create a new subscriber queue for SSE events."""
        q: asyncio.Queue = asyncio.Queue(maxsize=1024)
        self._subscribers.append(q)
        return q

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        return {
            "run_id": self._run_id,
            "active": self._task is not None and not self._task.done(),
            **self._status,
        }

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Request graceful stop of the current run."""
        self._stop_requested = True

    # ------------------------------------------------------------------
    # BTRM training run
    # ------------------------------------------------------------------

    def start_btrm_run(
        self, config: dict[str, Any], backend,
    ) -> dict[str, Any]:
        """Launch BTRM training as a background asyncio task.

        Config fields (all have defaults):
          dataset_path, n_steps, lr, head_names, pref_keys,
          gradient_checkpointing, max_grad_norm, warmup_steps,
          lr_schedule, macrobatch_budget, megapixel_flops_fraction,
          checkpoint_steps, adapter_name, adapter_rank, adapter_alpha,
          output_dir

        Returns:
            {run_id, stream_url}
        """
        if self._task is not None and not self._task.done():
            raise RuntimeError("A training run is already active")

        self._run_id = uuid.uuid4().hex[:12]
        self._stop_requested = False
        self._subscribers.clear()

        output_dir = config.get("output_dir", f"training_output/run_{self._run_id}")
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Save config for reproducibility
        with open(self._output_dir / "run_config.json", "w") as f:
            json.dump(config, f, indent=2)

        self._status = {
            "active": True,
            "phase": "initializing",
            "step": 0,
            "n_steps": config.get("n_steps", 100),
            "loss": None,
            "accuracy": None,
            "elapsed_s": 0.0,
        }

        loop = asyncio.get_event_loop()
        self._task = loop.create_task(
            self._run_btrm(config, backend),
        )

        return {
            "run_id": self._run_id,
            "stream_url": f"/training/stream/{self._run_id}",
            "output_dir": str(self._output_dir),
        }

    async def _run_btrm(self, config: dict[str, Any], backend) -> None:
        """Background coroutine for BTRM training."""
        import torch
        from src_ii.btrm_training import train_btrm_differentiable
        from src_ii.btrm_lifecycle import (
            setup_btrm_training, persist_btrm,
        )
        from src_ii.pair_sampler import BTRMPairSampler, build_positions_from_v2
        from src_ii.dataset_io import (
            make_load_latent_fn,
            make_reward_manifest_preference_fn,
        )
        from src_ii.training_setup import encode_training_prompts
        from src_ii.training_artifacts import TrainingArtifacts
        from src_ii.incremental_save import TrainingCurveWriter

        run_id = self._run_id
        output_dir = self._output_dir
        t_start = time.monotonic()

        try:
            # --- Extract config with defaults ---
            dataset_path = config.get("dataset_path", "multi_res_trajectories")
            n_steps = config.get("n_steps", 100)
            lr = config.get("lr", 3e-4)
            head_names = tuple(config.get("head_names", ["pinkify", "thisnotthat"]))
            pref_keys = tuple(config.get("pref_keys", ["pinkify_pref", "thisnotthat_pref"]))
            gradient_checkpointing = config.get("gradient_checkpointing", True)
            max_grad_norm = config.get("max_grad_norm", 0.1)
            warmup_steps = config.get("warmup_steps", 5)
            lr_schedule = config.get("lr_schedule", "warmup_cosine")
            macrobatch_budget = config.get("macrobatch_budget", 3.0)
            megapixel_flops_fraction = config.get("megapixel_flops_fraction", 0.33)
            checkpoint_steps = config.get("checkpoint_steps", [25, 50, 75, 100, 125])
            adapter_name = config.get("adapter_name", "rtheta")
            adapter_rank = config.get("adapter_rank", 8)
            adapter_alpha = config.get("adapter_alpha", 16.0)
            clean_fraction = config.get("clean_fraction", 0.8)

            self._status["phase"] = "loading_dataset"
            self._publish("status", {"phase": "loading_dataset"})

            # --- Build pair sampler ---
            positions = build_positions_from_v2(dataset_path)
            pair_sampler = BTRMPairSampler(
                positions,
                clean_fraction=clean_fraction,
            )

            # --- Build preference function ---
            preference_fn = make_reward_manifest_preference_fn(
                dataset_path, pref_keys=pref_keys,
            )

            # --- Build load_latent_fn ---
            load_latent_fn = make_load_latent_fn(dataset_path)

            self._status["phase"] = "encoding_prompts"
            self._publish("status", {"phase": "encoding_prompts"})

            # --- Encode prompts (needs TE loaded) ---
            # The backend manages model lifecycle; we call it for text encoding
            unique_prompts = list(pair_sampler.unique_prompts)
            prompt_cache = {}
            for p in unique_prompts:
                result = backend.encode_prompt(p, layer_idx=-2)
                prompt_cache[p] = result["conditioning"]

            self._status["phase"] = "loading_model"
            self._publish("status", {"phase": "loading_model"})

            # --- Set up model for training ---
            backend._ensure_diffusion()
            model = backend._diff_model
            optimizer = setup_btrm_training(
                model,
                adapter_name=adapter_name,
                adapter_rank=adapter_rank,
                adapter_alpha=adapter_alpha,
                lr=lr,
                gradient_checkpointing=gradient_checkpointing,
            )

            # --- Artifacts + curve writer ---
            artifacts = TrainingArtifacts(
                str(output_dir), head_names=head_names,
            )
            curve_writer = TrainingCurveWriter(str(output_dir / "training_curve.jsonl"))

            # --- Step callback (publishes SSE events) ---
            orch = self

            def step_callback(step: int, metrics: dict) -> None:
                orch._status.update({
                    "step": step,
                    "loss": metrics.get("loss"),
                    "accuracy": metrics.get("per_head_accuracy"),
                    "elapsed_s": round(time.monotonic() - t_start, 1),
                })
                orch._publish("step", {
                    "step": step,
                    "n_steps": n_steps,
                    **metrics,
                })

            # --- Checkpoint callback ---
            def checkpoint_cb(step: int, _model) -> None:
                persist_btrm(model, adapter_name, str(output_dir / f"checkpoint_step{step:03d}"))
                self._publish("checkpoint", {"step": step})

            self._status["phase"] = "training"
            self._publish("status", {"phase": "training"})

            # --- Run training ---
            # train_btrm_differentiable is synchronous and GPU-bound.
            # We run it in the event loop's executor to avoid blocking.
            loop = asyncio.get_event_loop()
            step_metrics = await loop.run_in_executor(
                None,
                lambda: train_btrm_differentiable(
                    model=model,
                    pair_sampler=pair_sampler,
                    preference_fn=preference_fn,
                    load_latent_fn=load_latent_fn,
                    n_steps=n_steps,
                    lr=lr,
                    head_names=head_names,
                    pref_keys=pref_keys,
                    gradient_checkpointing=gradient_checkpointing,
                    max_grad_norm=max_grad_norm,
                    warmup_steps=warmup_steps,
                    lr_schedule=lr_schedule,
                    callback=step_callback,
                    checkpoint_fn=checkpoint_cb,
                    checkpoint_steps=checkpoint_steps,
                    output_dir=str(output_dir),
                    artifacts=artifacts,
                    macrobatch_budget=macrobatch_budget,
                    megapixel_flops_fraction=megapixel_flops_fraction,
                    curve_writer=curve_writer,
                    adapter_name=adapter_name,
                ),
            )

            # --- Persist final ---
            persist_btrm(model, adapter_name, str(output_dir))

            # --- Generate analysis ---
            if hasattr(artifacts, "generate_analysis"):
                artifacts.generate_analysis()

            # --- Emit chart artifacts for gallery ---
            charts_dir = output_dir / "charts"
            if charts_dir.exists():
                for png_path in sorted(charts_dir.glob("*.png")):
                    label = png_path.stem.replace("_", " ").lstrip("0123456789 ")
                    self._publish("artifact_ready", {
                        "path": f"charts/{png_path.name}",
                        "type": "chart",
                        "label": label or png_path.stem,
                        "metadata": {
                            "artifact_type": "chart",
                            "label": label or png_path.stem,
                            "run_id": run_id,
                        },
                    })

            elapsed = round(time.monotonic() - t_start, 1)
            self._status.update({
                "phase": "complete",
                "active": False,
                "elapsed_s": elapsed,
            })
            self._publish("complete", {
                "run_id": run_id,
                "output_dir": str(output_dir),
                "elapsed_s": elapsed,
                "n_steps": n_steps,
            })

        except Exception as e:
            logger.error(f"BTRM training run {run_id} failed: {e}\n{traceback.format_exc()}")
            self._status.update({
                "phase": "error",
                "active": False,
                "error": str(e),
            })
            self._publish("error", {
                "run_id": run_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    # ------------------------------------------------------------------
    # DDGRPO policy optimization run
    # ------------------------------------------------------------------

    def start_ddgrpo_run(
        self, config: dict[str, Any], backend,
    ) -> dict[str, Any]:
        """Launch DDGRPO policy optimization as a background task.

        Config fields:
          btrm_checkpoint, n_iters, prompts, n_rollouts_per_prompt,
          rollout_steps, advantage_threshold, policy_lr, max_grad_norm,
          kl_coeff, adapter_name, adapter_rank, output_dir

        Returns:
            {run_id, stream_url}
        """
        if self._task is not None and not self._task.done():
            raise RuntimeError("A training run is already active")

        self._run_id = uuid.uuid4().hex[:12]
        self._stop_requested = False
        self._subscribers.clear()

        output_dir = config.get("output_dir", f"training_output/ddgrpo_{self._run_id}")
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        with open(self._output_dir / "run_config.json", "w") as f:
            json.dump(config, f, indent=2)

        n_iters = config.get("n_iters", 100)
        self._status = {
            "active": True,
            "phase": "initializing",
            "step": 0,
            "n_steps": n_iters,
            "loss": None,
            "elapsed_s": 0.0,
        }

        loop = asyncio.get_event_loop()
        self._task = loop.create_task(
            self._run_ddgrpo(config, backend),
        )

        return {
            "run_id": self._run_id,
            "stream_url": f"/training/stream/{self._run_id}",
            "output_dir": str(self._output_dir),
        }

    async def _run_ddgrpo(self, config: dict[str, Any], backend) -> None:
        """Background coroutine for DDGRPO policy optimization.

        This is structurally parallel to _run_btrm but uses the REINFORCE +
        sparse-step policy gradient pipeline instead of supervised BT loss.
        """
        import torch
        from src_ii.btrm_lifecycle import setup_btrm_training, load_btrm
        from src_ii.multi_lora import (
            install_multi_lora, init_adapter_b_weights,
            freeze_base_params, save_adapter,
        )
        from src_ii.ddreinforce import compute_eta_schedule, group_advantages
        from src_ii.policy_step import (
            accumulate_reinforce_gradients, policy_optimizer_step,
        )
        from src_ii.sigma_schedule import build_sigma_schedule, resolution_shift
        from src_ii.resolution_sampling import sample_random_resolution
        from src_ii.incremental_save import TrainingCurveWriter, atomic_json_save

        run_id = self._run_id
        output_dir = self._output_dir
        t_start = time.monotonic()

        try:
            # --- Config ---
            btrm_checkpoint = config.get("btrm_checkpoint", "training_output/reward_function_run")
            n_iters = config.get("n_iters", 100)
            rollout_steps = config.get("rollout_steps", 20)
            eta_scale = config.get("eta_scale", 0.1)
            policy_lr = config.get("policy_lr", 2e-5)
            max_grad_norm = config.get("max_grad_norm", 0.1)
            btrm_adapter = config.get("btrm_adapter_name", "rtheta")
            policy_adapter = config.get("policy_adapter_name", "policy_pinkify")
            adapter_rank = config.get("adapter_rank", 8)
            adapter_alpha = config.get("adapter_alpha", 16.0)
            init_b_std = config.get("init_b_std", 0.01)
            prompts = config.get("prompts", ["a beautiful landscape"])
            n_rollouts = config.get("n_rollouts_per_prompt", 2)
            advantage_threshold = config.get("advantage_threshold", 0.01)
            resolution_budgets = config.get("resolution_budgets", [65536, 102400, 147456, 262144])

            self._status["phase"] = "loading_model"
            self._publish("status", {"phase": "loading_model"})

            # --- Encode prompts ---
            prompt_conds = {}
            for p in prompts:
                result = backend.encode_prompt(p, layer_idx=-2)
                prompt_conds[p] = result["conditioning"]
            neg_result = backend.encode_prompt("", layer_idx=-2)
            neg_cond = neg_result["conditioning"]

            # --- Load model + adapters ---
            backend._ensure_diffusion()
            model = backend._diff_model

            # Install BTRM adapter + load checkpoint
            btrm_configs = [{"name": btrm_adapter, "rank": adapter_rank, "alpha": adapter_alpha}]
            install_multi_lora(model, btrm_configs)
            load_btrm(model, btrm_adapter, btrm_checkpoint)

            # Install policy adapter
            policy_configs = [{"name": policy_adapter, "rank": adapter_rank, "alpha": adapter_alpha}]
            install_multi_lora(model, policy_configs)
            freeze_base_params(model)
            init_adapter_b_weights(model, policy_adapter, std=init_b_std)

            # Policy optimizer
            from src_ii.multi_lora import get_adapter_params
            policy_params = list(get_adapter_params(model, policy_adapter).values())
            optimizer = torch.optim.AdamW(policy_params, lr=policy_lr)

            curve_writer = TrainingCurveWriter(str(output_dir / "training_curve.jsonl"))

            self._status["phase"] = "training"
            self._publish("status", {"phase": "training"})

            # --- Training loop ---
            import random
            rng = random.Random(42)

            for iteration in range(n_iters):
                if self._stop_requested:
                    break

                iter_t0 = time.monotonic()

                # Sample prompts + resolutions
                batch_prompts = [rng.choice(prompts) for _ in range(len(prompts))]
                budget = rng.choice(resolution_budgets)
                w, h = sample_random_resolution(budget, rng=rng)

                # Build sigma schedule
                shift = resolution_shift(w, h)
                sigmas = build_sigma_schedule(rollout_steps, shift=shift)
                eta_schedule = compute_eta_schedule(rollout_steps, scale=eta_scale)

                # Generate rollouts and accumulate gradients
                all_rewards = []
                for prompt in batch_prompts:
                    cond = prompt_conds[prompt].to(model.score_proj.weight.device)
                    nc = neg_cond.to(model.score_proj.weight.device)

                    group_rollouts = []
                    for k in range(n_rollouts):
                        seed = rng.randint(0, 2**32 - 1)
                        # Use backend for rollout generation
                        result = backend.sample_trajectory(
                            params={
                                "seed": seed, "n_steps": rollout_steps,
                                "cfg": 4.0, "width": w, "height": h,
                                "attention_backend": "sage",
                                "sampling_shift": shift,
                                "save_steps": list(range(rollout_steps)),
                            },
                            tensors={
                                "pos_cond": cond.cpu(),
                                "neg_cond": nc.cpu(),
                            },
                        )
                        group_rollouts.append(result)

                    # Score finals via BTRM
                    from src_ii.btrm_lifecycle import score_serial
                    rewards = []
                    for rollout in group_rollouts:
                        final = rollout[0]["final"].to(cond.device)
                        scores = score_serial(
                            model, final.unsqueeze(0),
                            torch.zeros(1, device=cond.device),
                            cond.unsqueeze(0), cond.shape[-2],
                        )
                        rewards.append(scores[0, 0].item())
                    all_rewards.extend(rewards)

                    # Compute advantages
                    advantages = group_advantages(torch.tensor(rewards))

                    # Accumulate gradients
                    for k, (rollout, adv) in enumerate(zip(group_rollouts, advantages)):
                        if abs(adv.item()) < advantage_threshold:
                            continue
                        accumulate_reinforce_gradients(
                            model=model,
                            checkpoints=rollout[0],
                            sigmas=sigmas,
                            conditioning=cond,
                            adapter_name=policy_adapter,
                            advantage=adv.item(),
                            eta_used=eta_schedule,
                        )

                # Optimizer step
                step_info = policy_optimizer_step(
                    model, policy_adapter, optimizer,
                    max_grad_norm=max_grad_norm,
                )

                iter_elapsed = time.monotonic() - iter_t0
                metrics = {
                    "iteration": iteration,
                    "mean_reward": sum(all_rewards) / max(len(all_rewards), 1),
                    "grad_norm": step_info.get("grad_norm", 0.0),
                    "elapsed_s": round(iter_elapsed, 2),
                }

                self._status.update({
                    "step": iteration + 1,
                    "loss": metrics["mean_reward"],
                    "elapsed_s": round(time.monotonic() - t_start, 1),
                })
                self._publish("step", {"step": iteration, "n_steps": n_iters, **metrics})
                curve_writer.write_step(metrics)

                # Checkpoint every iteration
                save_adapter(model, policy_adapter, str(output_dir / f"policy_iter{iteration:04d}.safetensors"))

            # --- Emit chart artifacts for gallery ---
            charts_dir = output_dir / "charts"
            if charts_dir.exists():
                for png_path in sorted(charts_dir.glob("*.png")):
                    label = png_path.stem.replace("_", " ").lstrip("0123456789 ")
                    self._publish("artifact_ready", {
                        "path": f"charts/{png_path.name}",
                        "type": "chart",
                        "label": label or png_path.stem,
                        "metadata": {
                            "artifact_type": "chart",
                            "label": label or png_path.stem,
                            "run_id": run_id,
                        },
                    })

            # --- Complete ---
            elapsed = round(time.monotonic() - t_start, 1)
            self._status.update({"phase": "complete", "active": False, "elapsed_s": elapsed})
            self._publish("complete", {
                "run_id": run_id, "output_dir": str(output_dir), "elapsed_s": elapsed,
            })

        except Exception as e:
            logger.error(f"DDGRPO run {run_id} failed: {e}\n{traceback.format_exc()}")
            self._status.update({"phase": "error", "active": False, "error": str(e)})
            self._publish("error", {
                "run_id": run_id, "error": str(e), "traceback": traceback.format_exc(),
            })

    # ------------------------------------------------------------------
    # Policy intervention diff run
    # ------------------------------------------------------------------

    def start_policy_intervention_run(
        self, config: dict[str, Any], backend,
    ) -> dict[str, Any]:
        """Launch a policy intervention A/B comparison as a background task.

        Generates reference (adapter_scales=0) vs policy (adapter_scales=1)
        trajectories for each (prompt, seed), VAE decodes, builds false-color
        diffs and comparison composites, and emits artifact_ready events.

        Config fields (all have defaults except prompts):
          prompts, seeds, width, height, n_steps, btrm_dir, adapter_name,
          policy_checkpoint, policy_adapter_name, diff_scale, output_dir

        Returns:
            {run_id, stream_url, output_dir}
        """
        if self._task is not None and not self._task.done():
            raise RuntimeError("A training run is already active")

        self._run_id = uuid.uuid4().hex[:12]
        self._stop_requested = False
        self._subscribers.clear()

        output_dir = config.get("output_dir", f"training_output/policy_intervention_{self._run_id}")
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        with open(self._output_dir / "run_config.json", "w") as f:
            json.dump(config, f, indent=2)

        prompts = config.get("prompts", [])
        seeds = config.get("seeds", [42])
        self._status = {
            "active": True,
            "phase": "initializing",
            "step": 0,
            "n_steps": len(prompts) * len(seeds),
            "elapsed_s": 0.0,
        }

        loop = asyncio.get_event_loop()
        self._task = loop.create_task(
            self._run_policy_intervention(config, backend),
        )

        return {
            "run_id": self._run_id,
            "stream_url": f"/training/stream/{self._run_id}",
            "output_dir": str(self._output_dir),
        }

    async def _run_policy_intervention(self, config: dict[str, Any], backend) -> None:
        """Background coroutine for policy intervention A/B generation."""
        import gc
        import torch
        from src_ii.btrm_lifecycle import setup_btrm_training, load_btrm
        from src_ii.multi_lora import install_multi_lora
        from src_ii.inference_sampling import run_trajectory_packed
        from src_ii.infer.charts import draw_score_chart
        from src_ii.infer.composites import build_comparison_composite
        from src_ii.infer.diff_analysis import compute_pixel_diff_stats, make_false_color_diff
        from src_ii.sigma_schedule import sigma_to_logsnr
        from src_ii.vae_utils import load_vae, decode_latent_to_pil
        from src_ii.model_paths import VAE_PATH

        run_id = self._run_id
        output_dir = self._output_dir
        t_start = time.monotonic()

        try:
            # --- Config ---
            prompts = config.get("prompts", ["a beautiful landscape"])
            seeds = config.get("seeds", [42])
            width = config.get("width", 1280)
            height = config.get("height", 832)
            n_steps = config.get("n_steps", 20)
            cfg = config.get("cfg", 4.0)
            negative_prompt = config.get("negative_prompt", "")
            btrm_dir = config.get("btrm_dir", "training_output/reward_function_run_tnt_v2")
            adapter_name = config.get("adapter_name", "rtheta")
            policy_checkpoint = config.get("policy_checkpoint")
            policy_adapter_name = config.get("policy_adapter_name", "policy_pinkify")
            diff_scale = config.get("diff_scale", 10.0)
            adapter_rank = config.get("adapter_rank", 8)
            adapter_alpha = config.get("adapter_alpha", 16.0)

            device = torch.device("cuda")
            dtype = torch.bfloat16

            # --- Phase 1: Encode prompts ---
            self._status["phase"] = "encoding"
            self._publish("step", {"phase": "encoding", "detail": f"{len(prompts)} prompts"})

            prompt_conds = {}
            for p in prompts:
                result = backend.encode_prompt(p, layer_idx=-2)
                prompt_conds[p] = result["conditioning"]

            neg_result = backend.encode_prompt(negative_prompt, layer_idx=-2)
            neg_cond = neg_result["conditioning"]

            # --- Phase 2: Prepare model ---
            self._status["phase"] = "model_ready"
            self._publish("step", {"phase": "model_ready"})

            backend._ensure_diffusion()
            model = backend._diff_model

            # Read BTRM config for head names and adapter params
            btrm_config_path = Path(btrm_dir) / "btrm_compound_config.json"
            head_names = ["pinkify", "thisnotthat"]
            if btrm_config_path.exists():
                with open(btrm_config_path) as f:
                    btrm_cfg = json.load(f)
                head_names = btrm_cfg.get("head_names", head_names)
                adapter_rank = btrm_cfg.get("adapter_rank", adapter_rank)
                adapter_alpha = btrm_cfg.get("adapter_alpha", adapter_alpha)

            # Install BTRM adapter + load weights
            adapter_configs = [{"name": adapter_name, "rank": adapter_rank, "alpha": adapter_alpha}]
            install_multi_lora(model, adapter_configs)
            load_btrm(model, adapter_name, btrm_dir)

            # Optional: install and load policy adapter
            if policy_checkpoint:
                from src_ii.infer.model_setup import load_policy_adapter
                policy_configs = [{"name": policy_adapter_name, "rank": adapter_rank, "alpha": adapter_alpha}]
                install_multi_lora(model, policy_configs)
                load_policy_adapter(model, policy_adapter_name, policy_checkpoint)

            # --- Phase 3: Generate trajectories ---
            self._status["phase"] = "sampling"

            # Build adapter scale tensors matching installed adapter count.
            # With policy checkpoint: 2 adapters [rtheta, policy_pinkify]
            #   ref  = [1.0, 0.0] (reward adapter on, policy off)
            #   policy = [1.0, 1.0] (both on — diff shows policy effect)
            # Without policy checkpoint: 1 adapter [rtheta]
            #   ref  = [0.0] (adapter off)
            #   policy = [1.0] (adapter on — diff shows reward adapter effect)
            if policy_checkpoint:
                scales_ref = torch.tensor([[1.0, 0.0]], device=device)
                scales_policy = torch.tensor([[1.0, 1.0]], device=device)
            else:
                scales_ref = torch.tensor([[0.0]], device=device)
                scales_policy = torch.tensor([[1.0]], device=device)

            loop = asyncio.get_event_loop()
            all_results: dict[str, dict] = {}
            pair_idx = 0

            for prompt in prompts:
                for seed in seeds:
                    if self._stop_requested:
                        break
                    slug = f"p{prompts.index(prompt)}_s{seed}"

                    def _sample_pair(prompt=prompt, seed=seed, slug=slug):
                        cond = prompt_conds[prompt].to(device)
                        neg = neg_cond.to(device)
                        params = {
                            "n_images": 1, "seeds": [seed], "n_steps": n_steps,
                            "cfg": cfg, "multiplier": 1.0, "denoise": 1.0,
                            "width": width, "height": height,
                        }
                        tensors = {"neg_cond": neg, "pos_cond_0": cond}

                        result_ref, meta_ref = run_trajectory_packed(
                            model, device, dtype, params, tensors,
                            adapter_scales=scales_ref,
                        )
                        result_pol, meta_pol = run_trajectory_packed(
                            model, device, dtype, params, tensors,
                            adapter_scales=scales_policy,
                        )
                        return {
                            "latent_ref": result_ref["final_0"],
                            "latent_policy": result_pol["final_0"],
                            "recs_ref": meta_ref.get("step_scores", []),
                            "recs_policy": meta_pol.get("step_scores", []),
                            "prompt": prompt,
                            "seed": seed,
                        }

                    result = await loop.run_in_executor(None, _sample_pair)
                    all_results[slug] = result
                    pair_idx += 1
                    self._status["step"] = pair_idx
                    self._publish("step", {
                        "phase": "sampling", "detail": slug, "prompt_idx": pair_idx,
                        "total": len(prompts) * len(seeds),
                    })

            if self._stop_requested:
                self._status.update({"phase": "stopped", "active": False})
                self._publish("complete", {"run_id": run_id, "stopped": True})
                return

            # --- Phase 4: VAE decode + composites ---
            self._status["phase"] = "decoding"
            self._publish("step", {"phase": "decoding", "detail": f"{len(all_results)} pairs"})

            def _decode_and_composite():
                vae = load_vae(VAE_PATH, device=device, dtype=dtype)
                images = {}
                for slug, res in all_results.items():
                    img_ref = decode_latent_to_pil(vae, res["latent_ref"], device=device, dtype=dtype)
                    img_pol = decode_latent_to_pil(vae, res["latent_policy"], device=device, dtype=dtype)
                    images[slug] = {"ref": img_ref, "policy": img_pol}
                del vae
                gc.collect()
                torch.cuda.empty_cache()

                scores_log = []
                diff_stats_all = {}

                for slug, res in all_results.items():
                    img_ref = images[slug]["ref"]
                    img_pol = images[slug]["policy"]
                    recs_ref = res["recs_ref"]
                    recs_pol = res["recs_policy"]

                    # Save individual images
                    img_ref.save(str(output_dir / f"{slug}_ref.png"))
                    img_pol.save(str(output_dir / f"{slug}_policy.png"))

                    # False-color diff
                    diff_img = make_false_color_diff(img_ref, img_pol, scale=diff_scale)
                    diff_img.save(str(output_dir / f"{slug}_diff.png"))

                    # Score charts
                    logsnrs = [sigma_to_logsnr(r["sigma"]) for r in recs_ref]
                    n_heads = len(recs_ref[0]["scores"]) if recs_ref else 0
                    charts = []
                    for head_idx in range(n_heads):
                        head_name = head_names[head_idx] if head_idx < len(head_names) else f"head_{head_idx}"
                        named_series = {
                            "ref": {"values": [r["scores"][head_idx] for r in recs_ref], "color": (50, 50, 200)},
                            "policy": {"values": [r["scores"][head_idx] for r in recs_pol], "color": (200, 50, 50)},
                        }
                        charts.append(draw_score_chart(logsnrs, named_series, head_name))

                    # Comparison composite
                    composite = build_comparison_composite(
                        image_panels=[img_ref, img_pol, diff_img],
                        panel_labels=["ref (scale=0)", "policy (scale=1)", f"diff (x{diff_scale:.0f})"],
                        charts=charts,
                        title=f"{slug} — ref vs policy",
                    )
                    composite.save(str(output_dir / f"{slug}_composite.png"))

                    # Emit artifact_ready events (thread-safe via Queue.put_nowait)
                    self._publish("artifact_ready", {
                        "path": f"{slug}_composite.png",
                        "type": "diff",
                        "label": f"{slug}: ref vs policy",
                        "metadata": {
                            "artifact_type": "diff",
                            "label": f"{slug}: ref vs policy",
                            "run_id": run_id,
                        },
                    })
                    self._publish("artifact_ready", {
                        "path": f"{slug}_diff.png",
                        "type": "diff",
                        "label": f"{slug}: false-color diff",
                        "metadata": {
                            "artifact_type": "diff",
                            "label": f"{slug}: false-color diff",
                            "run_id": run_id,
                        },
                    })

                    # Diff stats
                    diff_stats_all[slug] = compute_pixel_diff_stats(img_ref, img_pol)

                    # Score log
                    for step_i, (r_ref, r_pol) in enumerate(zip(recs_ref, recs_pol)):
                        sigma = r_ref["sigma"]
                        scores_log.append({
                            "slug": slug, "step": step_i,
                            "sigma": sigma, "logsnr": sigma_to_logsnr(sigma),
                            "scores_ref": r_ref["scores"], "scores_policy": r_pol["scores"],
                        })

                # Write scores JSONL
                with open(output_dir / "scores_per_step.jsonl", "w") as f:
                    for entry in scores_log:
                        f.write(json.dumps(entry) + "\n")

                # Write manifest
                from datetime import datetime, timezone
                manifest = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "prompts": prompts,
                    "seeds": seeds,
                    "resolution": f"{width}x{height}",
                    "n_steps": n_steps,
                    "adapter_name": adapter_name,
                    "btrm_dir": btrm_dir,
                    "policy_checkpoint": policy_checkpoint,
                    "head_names": head_names,
                    "diff_scale": diff_scale,
                    "n_pairs": len(all_results),
                    "output_files": sorted(str(p.name) for p in output_dir.glob("*.png")),
                }
                with open(output_dir / "manifest.json", "w") as f:
                    json.dump(manifest, f, indent=2)

                return diff_stats_all

            diff_stats = await loop.run_in_executor(None, _decode_and_composite)

            # --- Phase 5: Complete ---
            elapsed = round(time.monotonic() - t_start, 1)
            self._status.update({
                "phase": "complete",
                "active": False,
                "elapsed_s": elapsed,
            })
            self._publish("complete", {
                "run_id": run_id,
                "output_dir": str(output_dir),
                "elapsed_s": elapsed,
                "n_pairs": len(all_results),
                "diff_stats": diff_stats,
            })

        except Exception as e:
            logger.error(f"Policy intervention run {run_id} failed: {e}\n{traceback.format_exc()}")
            self._status.update({
                "phase": "error",
                "active": False,
                "error": str(e),
            })
            self._publish("error", {
                "run_id": run_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    # ------------------------------------------------------------------
    # Validation endpoints (on-demand, synchronous)
    # ------------------------------------------------------------------

    def run_validation(
        self, challenge_type: str, backend,
    ) -> dict[str, Any]:
        """Run a validation challenge against the current model state.

        Returns results dict. Runs synchronously (fast — few forward passes).
        """
        backend._ensure_diffusion()
        model = backend._diff_model

        if challenge_type == "pinkify":
            from src_ii.pinkify_validation import validate_btrm_pinkify_ranking
            return validate_btrm_pinkify_ranking(model)

        elif challenge_type == "tnt":
            from src_ii.tnt_validation import validate_tnt_ranking
            return validate_tnt_ranking(model)

        elif challenge_type == "decorrelation":
            from src_ii.cross_head_decorrelation import measure_cross_head_decorrelation
            return measure_cross_head_decorrelation(model)

        else:
            raise ValueError(f"Unknown challenge type: {challenge_type!r}")

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def get_artifact_path(self, run_id: str, path: str) -> Path | None:
        """Resolve an artifact path within a training output directory."""
        if self._run_id != run_id or self._output_dir is None:
            return None
        full = self._output_dir / path
        # Security: ensure path doesn't escape output_dir
        try:
            full.resolve().relative_to(self._output_dir.resolve())
        except ValueError:
            return None
        if full.exists():
            return full
        return None

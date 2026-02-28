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
        self._task: asyncio.Future | None = None
        self._subscribers: list[asyncio.Queue] = []
        self._status: dict[str, Any] = {"active": False, "phase": "idle"}
        self._stop_requested = False
        self._output_dir: Path | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------
    # SSE event publishing
    # ------------------------------------------------------------------

    def _publish(self, event_type: str, data: dict[str, Any]) -> None:
        """Push an SSE event to all subscribers. Thread-safe.

        Training runs execute in a worker thread (via run_in_executor).
        Queue operations must be scheduled on the event loop thread via
        call_soon_threadsafe so the async event loop stays responsive
        while GPU work proceeds uninterrupted.
        """
        event = {"type": event_type, "data": data}
        loop = self._event_loop
        dead = []
        for i, q in enumerate(self._subscribers):
            try:
                if loop is not None and loop.is_running():
                    loop.call_soon_threadsafe(q.put_nowait, event)
                else:
                    q.put_nowait(event)
            except (asyncio.QueueFull, RuntimeError):
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
        self._event_loop = loop
        self._task = loop.run_in_executor(
            None, self._run_btrm, config, backend,
        )

        return {
            "run_id": self._run_id,
            "stream_url": f"/training/stream/{self._run_id}",
            "output_dir": str(self._output_dir),
        }

    def _run_btrm(self, config: dict[str, Any], backend) -> None:
        """Background thread for BTRM training."""
        run_id = self._run_id
        output_dir = self._output_dir
        t_start = time.monotonic()

        try:
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
            # --- Extract config with defaults ---
            dataset_path = config.get("dataset_path", "multi_res_trajectories")
            n_steps = config.get("n_steps", 100)
            lr = config.get("lr", 3e-4)
            head_names = tuple(config["head_names"])
            pref_keys = tuple(config["pref_keys"])
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

            # --- Open dataset reader ---
            from futudiffu.dataset_v2 import DatasetReader
            reader = DatasetReader(str(dataset_path))
            n_available = len(reader)
            traj_ids = list(range(n_available))
            logger.info(f"Dataset: {dataset_path}, {n_available} trajectories")

            # --- Build pair sampler ---
            positions = build_positions_from_v2(reader, traj_ids=traj_ids)
            pair_sampler = BTRMPairSampler(
                positions,
                clean_fraction=clean_fraction,
            )

            self._status["phase"] = "encoding_prompts"
            self._publish("status", {"phase": "encoding_prompts"})

            # --- Encode prompts (needs TE loaded) ---
            # Collect unique prompts from dataset metadata
            unique_prompts = set()
            for tid in traj_ids:
                meta, _ = reader[tid]
                p = meta.get("prompt", "")
                if p:
                    unique_prompts.add(p)
            unique_prompts = sorted(unique_prompts)
            logger.info(f"Encoding {len(unique_prompts)} unique prompts")

            prompt_cache = {}
            for p in unique_prompts:
                result = backend.encode_prompt(p, layer_idx=-2)
                prompt_cache[p] = result["conditioning"]

            # --- Build load_latent_fn ---
            load_latent_fn = make_load_latent_fn(
                reader, prompt_cache, device="cuda",
            )

            # --- Build preference function (sigma-based: cleaner wins) ---
            def preference_fn(pair: dict) -> dict:
                prefs = {}
                for pref_key in pref_keys:
                    sigma_a = pair.get("sigma_a", 0.5)
                    sigma_b = pair.get("sigma_b", 0.5)
                    if sigma_a < sigma_b - 0.001:
                        prefs[pref_key] = 1
                    elif sigma_b < sigma_a - 0.001:
                        prefs[pref_key] = -1
                    else:
                        prefs[pref_key] = 0
                return prefs

            self._status["phase"] = "loading_model"
            self._publish("status", {"phase": "loading_model"})

            # --- Set up model for training ---
            backend._ensure_diffusion()
            model = backend._diff_model
            optimizer = setup_btrm_training(
                model,
                adapter_name=adapter_name,
                adapter_slot=0,
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
                persist_btrm(model, adapter_name, str(output_dir / f"checkpoint_step{step:03d}"), head_names=head_names)
                self._publish("checkpoint", {"step": step})

            self._status["phase"] = "training"
            self._publish("status", {"phase": "training"})

            # --- Run training ---
            # Already in a worker thread (run_in_executor from start_btrm_run),
            # so call directly. The event loop stays responsive for status/SSE.
            step_metrics = train_btrm_differentiable(
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
            )

            # --- Persist final ---
            persist_btrm(model, adapter_name, str(output_dir), head_names=head_names)

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
        self._event_loop = loop
        self._task = loop.run_in_executor(
            None, self._run_ddgrpo, config, backend,
        )

        return {
            "run_id": self._run_id,
            "stream_url": f"/training/stream/{self._run_id}",
            "output_dir": str(self._output_dir),
        }

    def _run_ddgrpo(self, config: dict[str, Any], backend) -> None:
        """Background thread for DDGRPO policy optimization.

        This is structurally parallel to _run_btrm but uses the REINFORCE +
        sparse-step policy gradient pipeline instead of supervised BT loss.
        """
        run_id = self._run_id
        output_dir = self._output_dir
        t_start = time.monotonic()

        try:
            import torch
            from src_ii.btrm_lifecycle import load_btrm
            from src_ii.multi_lora import (
                assign_adapter, init_adapter_b_weights,
                freeze_base_params, set_adapter_trainable,
                save_adapter, adapter_capacity,
            )
            from src_ii.ddreinforce import compute_eta_schedule, group_advantages
            from src_ii.policy_step import (
                accumulate_reinforce_gradients, policy_optimizer_step,
            )
            from src_ii.sigma_schedule import build_sigma_schedule, resolution_shift
            from src_ii.resolution_sampling import sample_random_resolution
            from src_ii.incremental_save import TrainingCurveWriter, atomic_json_save
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
            device = model.score_proj.weight.device

            # Assign adapters to explicit pre-allocated slots.
            # Capacity was pre-allocated by server's _ensure_diffusion().
            BTRM_SLOT = 0
            POLICY_SLOT = 1
            assign_adapter(model, BTRM_SLOT, btrm_adapter, adapter_rank, adapter_alpha)
            load_btrm(model, btrm_adapter, btrm_checkpoint)

            assign_adapter(model, POLICY_SLOT, policy_adapter, adapter_rank, adapter_alpha)
            freeze_base_params(model)
            set_adapter_trainable(model, policy_adapter, True)
            init_adapter_b_weights(model, policy_adapter, std=init_b_std)

            # --- Adapter scale mapping ---
            # Scales are always max_adapters wide (pre-allocated capacity).
            # Explicit slot indices: BTRM_SLOT=0, POLICY_SLOT=1.
            #   scales_policy: policy adapter active. Valid diffusion field.
            #   scales_reward: reward adapter active. Valid BTRM scores.
            cap = adapter_capacity(model)
            n_slots = cap["max_adapters"]
            scales_policy = torch.zeros(1, n_slots, device=device, dtype=torch.bfloat16)
            scales_policy[0, POLICY_SLOT] = 1.0
            scales_reward = torch.zeros(1, n_slots, device=device, dtype=torch.bfloat16)
            scales_reward[0, BTRM_SLOT] = 1.0

            adapter_mapping = {
                "slots": {btrm_adapter: BTRM_SLOT, policy_adapter: POLICY_SLOT},
                "max_adapters": n_slots,
            }

            # Persist adapter mapping for debugging / checkpoint interpretation
            from src_ii.incremental_save import atomic_json_save
            atomic_json_save(adapter_mapping, str(output_dir / "adapter_mapping.json"))

            # Policy optimizer — only policy adapter params receive gradients.
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
                sigmas = build_sigma_schedule(
                    rollout_steps, sampling_shift=shift,
                    device=device, dtype=torch.bfloat16,
                )
                eta_schedule = compute_eta_schedule(sigmas, eta_scale=eta_scale)

                # Zero gradients before accumulation
                optimizer.zero_grad()

                # Generate rollouts, score, compute advantages, accumulate grads
                #
                # Memory lifecycle discipline (2 compilation contexts):
                #  - Rollouts + Scoring: inference_mode. No autograd graph.
                #    Rollouts run the full Euler chain; scoring runs one forward
                #    per clean final. Both are queries to the same BatchExecutor.
                #    Scoring uses adapter_scales=scales_reward (BTRM on, policy off)
                #    so the score head sees hidden states shaped by the reward
                #    adapter, producing semantically valid BTRM predictions.
                #  - Gradient accumulation: grad-enabled + gradient_checkpointing=True.
                #    Only the micro-batch backward retains activations (~500MB with gc).
                #    adapter_scales=scales_policy gates BTRM gradient to zero.
                from src_ii.batch_executor import BatchExecutor, ExecutorAdapter
                from src_ii.inference_sampling import run_trajectory_packed
                from src_ii.triumphant_future_reduction_ops import cfg2

                # gradient_checkpointing=True for grad accumulation.
                # During inference_mode (rollouts, scoring), checkpointing is
                # irrelevant since autograd is disabled. Stays True throughout.
                model.gradient_checkpointing = True

                # One BatchExecutor for the entire iteration. Plan cache is
                # reused within rollouts+scoring (both inference_mode), then
                # invalidated before gradient accumulation (needs autograd).
                iter_be = BatchExecutor(model, device)

                all_rewards = []
                all_rollout_data = []  # Collect across prompts for scoring

                for prompt in batch_prompts:
                    cond = prompt_conds[prompt].to(device)
                    nc = neg_cond.to(device)

                    group_rollouts = []
                    for k in range(n_rollouts):
                        seed = rng.randint(0, 2**32 - 1)
                        # Rollouts via run_trajectory_packed directly:
                        # inference_mode wrapping, adapter_scales=scales_policy
                        # (policy adapter active for on-policy sampling).
                        params = {
                            "n_images": 1, "seeds": [seed],
                            "n_steps": rollout_steps,
                            "cfg": 4.0, "width": w, "height": h,
                            "save_steps": list(range(rollout_steps)),
                        }
                        tensors_in = {
                            "neg_cond": nc, "pos_cond_0": cond,
                        }
                        with torch.inference_mode():
                            result_tensors, metadata = run_trajectory_packed(
                                model, device, torch.bfloat16,
                                params, tensors_in,
                                adapter_scales=scales_policy,
                                batch_executor=iter_be,
                            )
                        group_rollouts.append(result_tensors)

                    # --- Score finals via BatchExecutor (NOT score_serial) ---
                    # Build scoring queries: clean final latents at sigma=0
                    # with adapter_scales=scales_reward (BTRM adapter active).
                    # All queries submitted at once — executor bin-packs them.
                    scoring_queries = []
                    for k, result_tensors in enumerate(group_rollouts):
                        final = result_tensors["final_0"].to(device)
                        scoring_queries.append({
                            "query_id": f"score_{k}",
                            "base_latent": final,
                            "base_cond": cond,
                            "base_cap_len": cond.shape[1],
                            "base_resolution": (w, h),
                            "sigma": 0.0,
                            "forks": [{"entry_id": "e0"}],
                            "adapter_scales": scales_reward,
                        })

                    with torch.inference_mode():
                        score_results = iter_be.execute(scoring_queries)

                    rewards = []
                    for r in score_results:
                        # scores: (n_heads,) per entry. Head 0 is the reward signal.
                        rewards.append(r["scores"][0].item())
                    all_rewards.extend(rewards)

                    # Compute group advantages
                    advantages = group_advantages(torch.tensor(rewards))

                    # Build trajectory dicts + specs for high-advantage rollouts
                    grad_trajectories = []
                    grad_specs = []
                    grad_sigmas = []
                    grad_advantages = []

                    for k, (result_tensors, adv) in enumerate(zip(group_rollouts, advantages)):
                        if abs(adv.item()) < advantage_threshold:
                            continue
                        traj_dict = {"eta_used": eta_schedule}
                        for step_i in range(rollout_steps):
                            step_key = f"step_{step_i:02d}_0"
                            if step_key in result_tensors:
                                # .clone() exits inference_mode tensor — required
                                # because accumulate_reinforce_gradients runs a
                                # differentiable forward from these checkpoints.
                                traj_dict[f"checkpoint_{step_i}"] = result_tensors[step_key].clone()
                        traj_dict[f"checkpoint_{rollout_steps}"] = result_tensors["final_0"].clone()

                        grad_trajectories.append(traj_dict)
                        grad_specs.append(cfg2(cond, nc, (w, h), 4.0))
                        grad_sigmas.append(sigmas)
                        grad_advantages.append(adv.item())

                    if grad_trajectories:
                        step_stride = max(1, rollout_steps // 5)
                        gradient_steps = list(range(0, rollout_steps, step_stride))

                        # Invalidate plan cache: rollout/scoring ran under
                        # inference_mode, so cached tensors (refined_caps,
                        # packed_rope, block_mask) are inference tensors.
                        # Gradient accumulation needs autograd-compatible
                        # tensors — force rebuild on next execute().
                        iter_be.invalidate_plan_cache()

                        grad_executor = ExecutorAdapter(iter_be, grad_sigmas, device)

                        # adapter_scales=scales_policy: policy adapter active,
                        # BTRM adapter scale=0 → zero gradient via chain rule.
                        # Only policy params receive gradient signal.
                        accumulate_reinforce_gradients(
                            executor=grad_executor,
                            specs=grad_specs,
                            query_sigmas=grad_sigmas,
                            trajectories=grad_trajectories,
                            gradient_steps=gradient_steps,
                            advantages=grad_advantages,
                            adapter_scales=scales_policy,
                        )

                # Optimizer step
                step_info = policy_optimizer_step(
                    optimizer,
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
        self._event_loop = loop
        self._task = loop.run_in_executor(
            None, self._run_policy_intervention, config, backend,
        )

        return {
            "run_id": self._run_id,
            "stream_url": f"/training/stream/{self._run_id}",
            "output_dir": str(self._output_dir),
        }

    def _run_policy_intervention(self, config: dict[str, Any], backend) -> None:
        """Background coroutine for policy intervention A/B generation."""
        run_id = self._run_id
        output_dir = self._output_dir
        t_start = time.monotonic()

        try:
            import gc
            import torch
            from src_ii.btrm_lifecycle import load_btrm
            from src_ii.multi_lora import assign_adapter, adapter_capacity
            from src_ii.inference_sampling import run_trajectory_packed
            from src_ii.infer.charts import draw_score_chart
            from src_ii.infer.composites import build_comparison_composite
            from src_ii.infer.diff_analysis import compute_pixel_diff_stats, make_false_color_diff
            from src_ii.sigma_schedule import sigma_to_logsnr
            from src_ii.vae_utils import load_vae, decode_latent_to_pil
            from src_ii.model_paths import VAE_PATH
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
            if not btrm_config_path.exists():
                raise FileNotFoundError(
                    f"BTRM config not found: {btrm_config_path}. "
                    f"Cannot determine head_names without a persisted config."
                )
            with open(btrm_config_path) as f:
                btrm_cfg = json.load(f)
            if "head_names" not in btrm_cfg:
                raise KeyError(
                    f"'head_names' missing from {btrm_config_path}. "
                    f"Re-persist the BTRM checkpoint with head_names."
                )
            head_names = btrm_cfg["head_names"]
            adapter_rank = btrm_cfg.get("adapter_rank", adapter_rank)
            adapter_alpha = btrm_cfg.get("adapter_alpha", adapter_alpha)

            # Assign BTRM adapter to slot 0 + load weights
            BTRM_SLOT = 0
            POLICY_SLOT = 1
            assign_adapter(model, BTRM_SLOT, adapter_name, adapter_rank, adapter_alpha)
            load_btrm(model, adapter_name, btrm_dir)

            # Optional: assign and load policy adapter to slot 1
            if policy_checkpoint:
                from src_ii.infer.model_setup import load_policy_adapter
                assign_adapter(model, POLICY_SLOT, policy_adapter_name, adapter_rank, adapter_alpha)
                load_policy_adapter(model, policy_adapter_name, policy_checkpoint)

            # --- Phase 3: Generate trajectories ---
            self._status["phase"] = "sampling"

            # Build adapter scale tensors — always max_adapters wide.
            cap = adapter_capacity(model)
            n_slots = cap["max_adapters"]
            if policy_checkpoint:
                scales_ref = torch.zeros(1, n_slots, device=device)
                scales_ref[0, BTRM_SLOT] = 1.0
                scales_policy = torch.zeros(1, n_slots, device=device)
                scales_policy[0, BTRM_SLOT] = 1.0
                scales_policy[0, POLICY_SLOT] = 1.0
            else:
                scales_ref = torch.zeros(1, n_slots, device=device)
                scales_policy = torch.zeros(1, n_slots, device=device)
                scales_policy[0, BTRM_SLOT] = 1.0

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

                    result = _sample_pair()
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

            diff_stats = _decode_and_composite()

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

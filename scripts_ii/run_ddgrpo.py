r"""DDGRPO policy optimization via BTRM reward environment.

Usage:
    .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\run_ddgrpo.py ^
        --env pinkify --adapter policy_pinkify ^
        --btrm-dir training_output/reward_function_run ^
        --output training_output/pinkify_p_theta/ --n-iters 100

Concurrent policies: run multiple instances with different --env/--adapter.
All share the same server via HTTP multi-LoRA routing.

The training loop per iteration:
  1. Sample prompt from pool
  2. Generate K SDE trajectories via batch_rollout_sde
     (executor wraps client.batch_forward)
  3. Score K final latents via client.score_btrm()
  4. Compute rewards + advantages via env.compute_advantages(scores)
  5. For each trajectory k with non-trivial advantage:
     client.accumulate_policy_gradients(...)
  6. client.policy_optimizer_step(adapter_name, max_grad_norm, lr)
  7. Write JSONL metrics, checkpoint adapter periodically
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from src_ii.http_client import HTTPInferenceClient
from src_ii.reward_env import PolicyConfig, make_standard_envs
from src_ii.ktuple_sampling import batch_rollout_sde
from src_ii.sigma_schedule import build_sigma_schedule, resolution_shift



BTRM_HEAD_NAMES = ("pinkify", "thisnotthat")

PROMPTS = [
    "a cat sitting on a windowsill watching the rain",
    "an astronaut floating in space above earth",
    "a medieval castle on a hilltop at sunset",
    "a portrait of a woman with flowers in her hair",
    "a cozy cabin in the woods during winter",
    "a futuristic city skyline at night",
    "a field of sunflowers under a blue sky",
    "a dragon flying over a mountain range",
    "a still life with fruit and wine on a table",
    "a lighthouse on a rocky coast during a storm",
    "a japanese garden with cherry blossoms",
    "a steampunk airship above the clouds",
]

CHECKPOINT_INTERVAL = 25
LOG_INTERVAL = 1
ADVANTAGE_THRESHOLD = 0.01  # skip trivial advantages



def make_client_executor(client: HTTPInferenceClient, n_steps: int, device, dtype):
    """Build an executor for ktuple_sampling that wraps client.batch_forward().

    The executor protocol: executor(x_bases, specs, step_i, adapter_scales)
    Returns: (denoised_per_query, scores)

    Each query has exactly 1-2 forks (cfg1 or cfg2). We build batch_forward
    query dicts from the ktuple spec structure and parse results back.
    """
    def executor_fn(x_bases, specs, step_i, adapter_scales=None):
        queries = []
        for k, (x_base, spec) in enumerate(zip(x_bases, specs)):
            base_cond, base_res, _ = spec[0]

            w, h = base_res
            sigma = float(_sigma_for_step(w, h, step_i, n_steps))

            forks = []
            for j, (cond, res, scale) in enumerate(spec):
                fork = {"entry_id": f"e{j}"}
                if j > 0:
                    fork["cond"] = cond
                if res != base_res:
                    fork["resolution"] = res
                forks.append(fork)

            query = {
                "query_id": f"q{k}",
                "base_latent": x_base,
                "base_cond": base_cond,
                "base_cap_len": base_cond.shape[1],
                "base_resolution": base_res,
                "sigma": sigma,
                "forks": forks,
            }
            if adapter_scales is not None:
                query["adapter_scales"] = adapter_scales
            queries.append(query)

        result_tensors, metadata = client._post_mixed(
            "/batch_forward",
            {"queries": _serialize_queries(queries)},
            _collect_tensors(queries),
        )

        tags = metadata.get("tags", [])
        denoised_per_query = _parse_batch_results(
            result_tensors, tags, specs, device)

        scores = None
        for tag in tags:
            if "score_key" in tag and tag["score_key"] in result_tensors:
                if scores is None:
                    scores = result_tensors[tag["score_key"]].unsqueeze(0)
                else:
                    scores = torch.cat([
                        scores,
                        result_tensors[tag["score_key"]].unsqueeze(0)
                    ], dim=0)

        return denoised_per_query, scores

    return executor_fn


def _sigma_for_step(w, h, step_i, n_steps_hint):
    """Reconstruct sigma for a given step (helper for executor)."""
    alpha = resolution_shift(w, h)
    sigmas = build_sigma_schedule(
        n_steps_hint, sampling_shift=alpha,
        device=torch.device("cpu"), dtype=torch.float32)
    if step_i < len(sigmas):
        return float(sigmas[step_i])
    return 0.0


def _serialize_queries(queries):
    """Convert query dicts to JSON-serializable form for HTTP transport."""
    out = []
    tensor_idx = 0
    for q in queries:
        sq = {
            "query_id": q["query_id"],
            "base_latent_key": f"base_latent_{q['query_id']}",
            "base_cond_key": f"base_cond_{q['query_id']}",
            "base_cap_len": q["base_cap_len"],
            "base_resolution": q["base_resolution"],
            "sigma": q["sigma"],
            "forks": [],
        }
        if "adapter_scales" in q:
            sq["adapter_scales_key"] = f"adapter_scales_{q['query_id']}"

        for fork in q["forks"]:
            sf = {"entry_id": fork["entry_id"]}
            if "cond" in fork:
                sf["cond_key"] = f"cond_{q['query_id']}_{fork['entry_id']}"
            if "resolution" in fork:
                sf["resolution"] = fork["resolution"]
            sq["forks"].append(sf)
        out.append(sq)
    return out


def _collect_tensors(queries):
    """Collect all tensors from queries into a flat dict for transport."""
    tensors = {}
    for q in queries:
        qid = q["query_id"]
        tensors[f"base_latent_{qid}"] = q["base_latent"]
        tensors[f"base_cond_{qid}"] = q["base_cond"]
        if "adapter_scales" in q:
            tensors[f"adapter_scales_{qid}"] = q["adapter_scales"]
        for fork in q["forks"]:
            if "cond" in fork:
                tensors[f"cond_{qid}_{fork['entry_id']}"] = fork["cond"]
    return tensors


def _parse_batch_results(result_tensors, tags, specs, device):
    """Parse batch_forward results back into per-query denoised lists."""
    K = len(specs)
    buckets: dict[int, list] = {k: [] for k in range(K)}
    for tag in tags:
        k = int(tag["query_id"][1:])
        j = int(tag["entry_id"][1:])
        d = result_tensors[tag["denoised_key"]].to(device)
        buckets[k].append((j, d))

    denoised_per_query = []
    for k in range(K):
        items = sorted(buckets[k], key=lambda t: t[0])
        denoised_per_query.append([d for _, d in items])
    return denoised_per_query



def run_ddgrpo(args):
    print(f"[DDGRPO] Starting: env={args.env}, adapter={args.adapter}")
    print(f"[DDGRPO] Server: {args.server_url}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = HTTPInferenceClient(args.server_url, timeout_s=600.0)

    status = client.status()
    print(f"[DDGRPO] Server status: {status.get('loaded_models', [])}")

    envs = make_standard_envs(BTRM_HEAD_NAMES)
    if args.env not in envs:
        print(f"[ERROR] Unknown env: {args.env}. Valid: {list(envs.keys())}")
        sys.exit(1)
    env = envs[args.env]

    config = PolicyConfig(
        adapter_name=args.adapter,
        env=env,
        lr=args.lr,
        beta=args.beta,
        eta_scale=args.eta_scale,
        K=args.K,
        n_steps=args.n_steps,
        max_grad_norm=args.max_grad_norm,
        adapter_rank=args.rank,
        adapter_alpha=args.alpha,
        init_b_std=args.init_b_std,
        width=args.width,
        height=args.height,
        cfg=args.cfg,
    )

    print(f"[DDGRPO] Allocating adapter '{config.adapter_name}' "
          f"(rank={config.adapter_rank})")
    client.allocate_adapter(
        config.adapter_name,
        rank=config.adapter_rank,
        alpha=config.adapter_alpha,
    )
    client.init_adapter_weights(
        config.adapter_name,
        init_b_std=config.init_b_std,
    )

    print(f"[DDGRPO] Encoding {len(PROMPTS)} prompts...")
    pos_conds = []
    neg_cond = None
    for prompt in PROMPTS:
        cond = client.encode_prompt(prompt, layer_idx=-2)
        pos_conds.append(cond)
    neg_cond = client.encode_prompt("", layer_idx=-2)

    device = torch.device("cpu")  # Client-side is CPU; server is GPU
    dtype = torch.bfloat16

    run_config = {
        "env": args.env,
        "adapter": config.adapter_name,
        "K": config.K,
        "n_steps": config.n_steps,
        "eta_scale": config.eta_scale,
        "beta": config.beta,
        "lr": config.lr,
        "cfg": config.cfg,
        "width": config.width,
        "height": config.height,
        "n_iters": args.n_iters,
        "env_metadata": env.reward_metadata(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    metrics_path = output_dir / "training_metrics.jsonl"
    metrics_file = open(metrics_path, "a")

    executor = make_client_executor(client, config.n_steps, device, dtype)

    for iteration in range(args.n_iters):
        t0 = time.perf_counter()

        prompt_idx = random.randint(0, len(PROMPTS) - 1)
        pos_cond = pos_conds[prompt_idx]

        seeds = [random.randint(0, 2**31 - 1) for _ in range(config.K)]
        resolutions = [(config.width, config.height)] * config.K
        cap_lens = [pos_cond.shape[1]] * config.K

        n = config.n_steps
        sparse_step_count = min(5, n)
        save_steps = set()
        for i in range(sparse_step_count):
            step_idx = (i * n) // sparse_step_count
            save_steps.add(step_idx)
        for s in list(save_steps):
            if s + 1 < n:
                save_steps.add(s + 1)

        trajectories, meta = batch_rollout_sde(
            executor,
            pos_conds=[pos_cond] * config.K,
            neg_conds=[neg_cond] * config.K,
            cap_lens=cap_lens,
            seeds=seeds,
            resolutions=resolutions,
            n_steps=config.n_steps,
            cfg=config.cfg,
            device=device,
            dtype=dtype,
            eta_scale=config.eta_scale,
            save_steps=save_steps,
        )

        sigma_zero = torch.zeros(1)
        cond_for_scoring = pos_cond

        all_scores = []
        for k in range(config.K):
            final_latent = trajectories[k]["final"]
            scores_k = client.score_btrm(
                final_latent, sigma_zero, cond_for_scoring,
                multiplier=1.0,
            )
            all_scores.append(scores_k[0])  # scores_k is [[h0, h1]]

        scores_tensor = torch.tensor(all_scores, dtype=torch.float32)

        rewards, advantages = env.compute_advantages(scores_tensor)

        total_log_ratio = 0.0
        total_kl = 0.0
        n_grad_steps = 0

        sigmas_tensor = torch.tensor(
            trajectories[0]["sigmas"], dtype=torch.float32)
        eta_used = trajectories[0]["eta_used"]

        for k in range(config.K):
            if abs(float(advantages[k])) < ADVANTAGE_THRESHOLD:
                continue

            checkpoints = {}
            sparse_steps_for_k = []
            for step_idx in sorted(save_steps):
                step_key = f"step_{step_idx:02d}"
                if step_key in trajectories[k]:
                    checkpoints[step_idx] = trajectories[k][step_key]["x"]
                    sparse_steps_for_k.append(step_idx)

            if not checkpoints:
                continue

            result = client.accumulate_policy_gradients(
                checkpoints=checkpoints,
                sigmas=sigmas_tensor,
                conditioning=pos_cond,
                adapter_name=config.adapter_name,
                advantage=float(advantages[k]),
                multiplier=1.0,
                eta_used=eta_used,
                beta=config.beta,
            )
            total_log_ratio += result.get("total_log_ratio", 0.0)
            total_kl += result.get("total_kl", 0.0)
            n_grad_steps += result.get("n_steps", 0)

        opt_result = client.policy_optimizer_step(
            adapter_name=config.adapter_name,
            max_grad_norm=config.max_grad_norm,
            lr=config.lr,
        )

        elapsed = time.perf_counter() - t0

        metrics = {
            "iteration": iteration,
            "prompt_idx": prompt_idx,
            "prompt": PROMPTS[prompt_idx],
            "mean_reward": float(rewards.mean()),
            "std_reward": float(rewards.std()),
            "rewards": rewards.tolist(),
            "advantages": advantages.tolist(),
            "scores": all_scores,
            "total_log_ratio": total_log_ratio,
            "total_kl": total_kl,
            "n_grad_steps": n_grad_steps,
            "grad_norm": opt_result.get("grad_norm", 0.0),
            "lr": opt_result.get("lr", config.lr),
            "elapsed_s": elapsed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        metrics_file.write(json.dumps(metrics) + "\n")
        metrics_file.flush()

        if iteration % LOG_INTERVAL == 0:
            print(f"[DDGRPO] iter={iteration:4d}  "
                  f"reward={float(rewards.mean()):+.4f}±{float(rewards.std()):.4f}  "
                  f"kl={total_kl:.4f}  "
                  f"grad={opt_result.get('grad_norm', 0.0):.4f}  "
                  f"t={elapsed:.1f}s")

        if (iteration + 1) % CHECKPOINT_INTERVAL == 0:
            ckpt_dir = output_dir / f"checkpoint_iter{iteration + 1:04d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            sd = client.get_lora_state_dict(config.adapter_name)
            from safetensors.torch import save_file
            save_file(sd, str(ckpt_dir / "adapter.safetensors"))
            with open(ckpt_dir / "config.json", "w") as f:
                json.dump({
                    "iteration": iteration + 1,
                    "adapter_name": config.adapter_name,
                    "env": args.env,
                    "mean_reward": float(rewards.mean()),
                }, f, indent=2)
            print(f"[DDGRPO] Checkpoint saved: {ckpt_dir}")

    final_sd = client.get_lora_state_dict(config.adapter_name)
    from safetensors.torch import save_file
    save_file(final_sd, str(output_dir / "adapter_final.safetensors"))

    metrics_file.close()

    print(f"[DDGRPO] Training complete. {args.n_iters} iterations.")
    print(f"[DDGRPO] Output: {output_dir}")



def main():
    parser = argparse.ArgumentParser(description="DDGRPO policy optimization")
    parser.add_argument("--env", type=str, required=True,
                        choices=["pinkify", "tnt", "pinkify_x_tnt"],
                        help="Reward environment")
    parser.add_argument("--adapter", type=str, required=True,
                        help="LoRA adapter name")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for checkpoints and metrics")
    parser.add_argument("--n-iters", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--K", type=int, default=4,
                        help="Trajectories per prompt group")
    parser.add_argument("--n-steps", type=int, default=30,
                        help="Denoising steps per trajectory")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.04,
                        help="KL penalty coefficient")
    parser.add_argument("--eta-scale", type=float, default=0.1,
                        help="SDE noise injection scale")
    parser.add_argument("--cfg", type=float, default=4.0,
                        help="CFG scale")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Gradient clipping norm")
    parser.add_argument("--rank", type=int, default=8,
                        help="LoRA adapter rank")
    parser.add_argument("--alpha", type=float, default=16.0,
                        help="LoRA alpha")
    parser.add_argument("--init-b-std", type=float, default=0.01,
                        help="LoRA B matrix init std")
    parser.add_argument("--width", type=int, default=1280,
                        help="Image width")
    parser.add_argument("--height", type=int, default=832,
                        help="Image height")
    parser.add_argument("--server-url", type=str,
                        default="http://localhost:8000",
                        help="Inference server URL")
    parser.add_argument("--btrm-dir", type=str, default=None,
                        help="BTRM checkpoint dir (for loading trained head)")
    args = parser.parse_args()
    run_ddgrpo(args)


if __name__ == "__main__":
    main()

r"""End-to-end DDGRPO policy optimization with full Z-Image weights.

Self-contained: loads model directly, no server needed. Tests the complete
policy gradient pipeline on the real model with trained BTRM heads.

Lifecycle:
  1. Load TE → encode prompts → free TE
  2. Load backbone (FP8), install multi-LoRA, compile
  3. Install multi-LoRA: "rtheta" (BTRM adapter) + "policy_pinkify"
  4. Load BTRM checkpoint (rtheta adapter + score head)
  5. Generate "before" exemplars (policy adapter at fresh init)
  6. Run N DDGRPO iterations with SDE rollout + BTRM scoring
  7. Generate "after" exemplars (same seeds)
  8. Load VAE → decode exemplars → save PNGs → free VAE
  9. Write metrics JSONL + analysis

Execution:
  PYTHONUNBUFFERED=1 .venv/Scripts/python.exe ^
      F:\dox\repos\ai\futudiffu\scripts_ii\validate_ddgrpo_e2e.py
"""

from __future__ import annotations

import json
import math
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


FP8_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
TE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"
VAE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\vae\zimage.safetensors"
TOKENIZER_PATH = str(REPO_ROOT / "src" / "futudiffu" / "tokenizer")

BTRM_DIR = REPO_ROOT / "training_output" / "reward_function_run_tnt_v2"
OUTPUT_DIR = REPO_ROOT / "training_output" / "ddgrpo_256sq_fast"

PROMPTS = [
    'ahem.\n*ting ting ting ting ting*\nthe query model for this is a LARGE LANGUAGE MODEL, specifically QWEN-3-4B, a GENERAL PURPOSE SEMANTIC PARSER which is able to WRITE SENTENCES AT A TIME when they are participating in dialogue. however, in this situation, they are being used as a hidden state generator to steer an *image generation model*, z-image.\n\nqwen-3-4b, draw me an "enormous laser shark for the sega saturn".',
    'qwen-3-4b, draw me a "gigantic laser shark breaching out of the ocean at sunset".',
    'qwen-3-4b, draw me a "laser shark swimming through a neon cyberpunk cityscape at night".',
    'qwen-3-4b, draw me a "laser shark made of chrome and glass, studio lighting, product photography".',
    'A fluffy orange cat curled up on a stack of leather-bound books in a sunlit library.',
    'Extreme macro photograph of a blue morpho butterfly wing showing iridescent scales.',
]

N_ITERS = 30
K = 4                   # trajectories per prompt group
N_STEPS = 30            # denoising steps
ETA_SCALE = 0.1         # SDE noise injection
LR = 1e-4
MAX_GRAD_NORM = 1.0
CFG = 4.0
POLICY_ADAPTER_NAME = "policy_pinkify"
BTRM_ADAPTER_NAME = "rtheta"
ADAPTER_RANK = 8
ADAPTER_ALPHA = 16.0
INIT_B_STD = 0.01

RESOLUTION_TIERS = [
    (320, 208),   # landscape ~66560px
    (208, 320),   # portrait
    (256, 256),   # square
    (288, 224),   # mild landscape ~64512px
    (224, 288),   # mild portrait
]

EXEMPLAR_SEEDS = [42, 137, 256, 999]
EXEMPLAR_PROMPT_IDX = 0  # canonical laser shark
EXEMPLAR_RESOLUTION = (320, 208)

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


def _log(msg):
    print(f"[DDGRPO] {msg}", flush=True)



def encode_all_prompts(prompts, device, dtype):
    """Load TE, encode all prompts + negative, free TE."""
    _log("Phase 1: Encoding prompts...")
    from futudiffu.text_encoder import create_tokenizer, load_text_encoder

    tokenizer = create_tokenizer(TOKENIZER_PATH)
    te_model = load_text_encoder(TE_PATH, device=device, dtype=dtype)
    te_compiled = torch.compile(te_model, mode="default")

    from futudiffu.text_encoder import encode_prompt

    pos_conds = []
    cap_lens = []
    with torch.inference_mode():
        for i, prompt in enumerate(prompts):
            cond = encode_prompt(te_compiled, tokenizer, prompt, device=device, layer_idx=-2)
            pos_conds.append(cond.cpu())
            cap_lens.append(cond.shape[1])
            _log(f"  [{i}] {prompt[:60]}... → {cond.shape}")

        neg_cond = encode_prompt(te_compiled, tokenizer, "", device=device, layer_idx=-2)
        neg_cond = neg_cond.cpu()
        _log(f"  [neg] → {neg_cond.shape}")

    del te_model, te_compiled, tokenizer
    torch.cuda.empty_cache()
    _log(f"  TE freed. {len(pos_conds)} prompts encoded.")
    return pos_conds, neg_cond, cap_lens



def load_model_and_setup(device, dtype):
    """Load backbone, install adapters, load BTRM checkpoint."""
    _log("Phase 2: Loading backbone...")
    from src_ii.zimage_model import load_zimage_rlaif
    from src_ii.multi_lora import install_multi_lora, init_adapter_b_weights
    from src_ii.btrm_lifecycle import load_btrm

    model = load_zimage_rlaif(
        FP8_PATH, device=device, dtype=dtype,
        compile_model=False, fuse=True, use_sage=True,
    )
    _log(f"  Backbone loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")

    adapter_configs = [
        {"name": BTRM_ADAPTER_NAME, "rank": ADAPTER_RANK, "alpha": ADAPTER_ALPHA},
        {"name": POLICY_ADAPTER_NAME, "rank": ADAPTER_RANK, "alpha": ADAPTER_ALPHA},
    ]
    install_multi_lora(model, adapter_configs)
    _log(f"  Multi-LoRA installed: {[c['name'] for c in adapter_configs]}")

    load_btrm(model, BTRM_ADAPTER_NAME, str(BTRM_DIR))
    _log(f"  BTRM loaded from {BTRM_DIR.name}")

    init_adapter_b_weights(model, POLICY_ADAPTER_NAME, std=INIT_B_STD)
    _log(f"  Policy adapter '{POLICY_ADAPTER_NAME}' initialized (B std={INIT_B_STD})")

    from src_ii.multi_lora import freeze_base_params
    n_frozen, n_total = freeze_base_params(model)
    _log(f"  Base frozen: {n_frozen}/{n_total} params")

    model.compile_for_execution()
    _log("  compile_for_execution done")

    return model



def make_local_executor(model, device, cap_lens, resolutions, pool=None):
    """Build a local executor that wraps BatchExecutor for ktuple_sampling.

    Sizes the bin capacity from the actual workload: max single-entry seq_len
    across all prompts × resolutions. Every entry must fit in one bin.

    executor_fn is called as: executor(x_bases, specs, step_i, adapter_scales)
    The caller must set executor_fn.query_sigmas before the solve loop.

    Args:
        pool: Optional AcceleratorPool. If provided, used as the batch
            executor instead of creating a new BatchExecutor.
    """
    from src_ii.inference_packing import compute_entry_seq_len

    max_entry_len = max(
        compute_entry_seq_len(h // 8, w // 8, cap_len)
        for cap_len in cap_lens
        for w, h in resolutions
    )

    if pool is None:
        from src_ii.accelerator_pool import AcceleratorPool
        pool = AcceleratorPool(
            model_factory=lambda d, _m=model: _m,
            devices=[device],
            max_total_len=max_entry_len,
        )

    batch_exec = pool
    _log(f"  AcceleratorPool ({len(pool.devices)} devices) "
         f"max_total_len={pool.max_total_len} "
         f"(workload max_entry_len={max_entry_len})")

    class ExecutorWithSigmas:
        """Executor that closes over mutable query_sigmas reference."""
        def __init__(self):
            self.query_sigmas = None
            self.batch_exec = batch_exec

        def __call__(self, x_bases, specs, step_i, adapter_scales=None):
            queries = []
            for k, (x_base, spec) in enumerate(zip(x_bases, specs)):
                base_cond, base_res, _ = spec[0]
                sigma = float(self.query_sigmas[k][step_i])

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

            results = batch_exec.execute(queries)

            buckets = {k: [] for k in range(len(specs))}
            score_list = []
            for r in results:
                k = int(r["query_id"][1:])
                j = int(r["entry_id"][1:])
                buckets[k].append((j, r["denoised"].to(device)))
                if r["scores"] is not None:
                    score_list.append(r["scores"])

            denoised_per_query = []
            for k in range(len(specs)):
                entries_sorted = sorted(buckets[k], key=lambda t: t[0])
                denoised_per_query.append([d for _, d in entries_sorted])

            scores = torch.stack(score_list) if score_list else None
            return denoised_per_query, scores

    return ExecutorWithSigmas()



def generate_sde_trajectories(
    executor, pos_cond, neg_cond, cap_len, seeds, resolution,
    n_steps, cfg, eta_scale, device, dtype,
    adapter_scales=None, save_steps=None,
):
    """Generate K SDE trajectories for one prompt. Returns trajectories."""
    from src_ii.ktuple_sampling import solve_sde
    from src_ii.sigma_schedule import build_sigma_schedule, resolution_shift
    from src_ii.triumphant_future_reduction_ops import cfg2
    from src_ii.ddreinforce import compute_eta_schedule

    K = len(seeds)
    w, h = resolution

    specs = [cfg2(pos_cond.to(device, dtype), neg_cond.to(device, dtype),
                  resolution, cfg) for _ in range(K)]

    alpha = resolution_shift(w, h)
    query_sigmas = [
        build_sigma_schedule(n_steps, sampling_shift=alpha, device=device, dtype=dtype)
        for _ in range(K)
    ]

    eta_schedule = compute_eta_schedule(query_sigmas[0], eta_scale=eta_scale)

    x_bases = []
    for k in range(K):
        gen = torch.Generator(device=device).manual_seed(seeds[k])
        x_bases.append(
            query_sigmas[k][0] * torch.randn(
                1, 16, h // 8, w // 8, dtype=dtype, device=device, generator=gen))

    executor.query_sigmas = query_sigmas

    trajectories = [{} for _ in range(K)]

    def save_fn(i, x_pres, guided_list):
        if save_steps and i in save_steps:
            for k in range(K):
                trajectories[k][f"step_{i:02d}"] = {
                    "x": x_pres[k].detach().cpu(),
                    "guided_denoised": guided_list[k].detach().cpu(),
                    "sigma": float(query_sigmas[k][i]),
                }

    x_finals, scores_all, mu_all, eta_used = solve_sde(
        executor, x_bases, specs, query_sigmas, n_steps, eta_schedule,
        adapter_scales=adapter_scales,
        save_fn=save_fn if save_steps else None,
    )

    from src_ii.sigma_schedule import const_inverse_noise_scaling
    for k in range(K):
        x_k = const_inverse_noise_scaling(query_sigmas[k][-1:], x_finals[k])
        trajectories[k].update({
            "final": x_k.detach().cpu(),
            "seed": seeds[k],
            "resolution": resolution,
            "sigmas": [float(s) for s in query_sigmas[k]],
            "mu_trajectory": [mu_all[t][k] for t in range(n_steps)],
            "eta_used": eta_used,
        })

    return trajectories


def score_finals(executor, trajectories, pos_cond, resolution, device, dtype):
    """Score K final latents via the existing BatchExecutor. Returns (K, n_heads) tensor."""
    cond = pos_cond.to(device, dtype)
    queries = []
    for k, traj in enumerate(trajectories):
        final = traj["final"].to(device, dtype)
        queries.append({
            "query_id": f"score_{k}",
            "base_latent": final,
            "base_cond": cond,
            "base_cap_len": cond.shape[1],
            "base_resolution": resolution,
            "sigma": 0.0,
            "forks": [{"entry_id": "e0"}],
        })

    with torch.no_grad():
        results = executor.batch_exec.execute(queries)

    score_map = {}
    for r in results:
        idx = int(r["query_id"].split("_")[1])
        score_map[idx] = r["scores"].to(device)
    return torch.stack([score_map[i] for i in range(len(trajectories))]).cpu()



def accumulate_and_step(
    executor, model, trajectories, advantages,
    pos_cond, neg_cond, resolution, cfg,
    optimizer, max_grad_norm, device, dtype,
    gather_fn=None,
):
    """Accumulate REINFORCE gradients for all K trajectories, then step.

    Caller owns optimizer creation (at setup) and zero_grad (before this call).
    This function: accumulate gradients → clip → step.
    """
    from src_ii.policy_step import accumulate_reinforce_gradients, policy_optimizer_step
    from src_ii.triumphant_future_reduction_ops import cfg2

    # Build spec -- identical to what generate_sde_trajectories uses
    spec = cfg2(pos_cond.to(device, dtype), neg_cond.to(device, dtype),
                resolution, cfg)

    # Build query_sigmas from the recorded trajectory
    query_sigmas = torch.tensor(
        trajectories[0]["sigmas"], dtype=torch.float32, device=device)

    # Adapter scales: index 0 = BTRM, index 1 = policy
    policy_scales = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
    ref_scales = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)

    # Enable gradient checkpointing for the model during gradient accumulation
    old_gc = getattr(model, 'gradient_checkpointing', False)
    model.gradient_checkpointing = True

    total_log_prob = 0.0
    total_drift_mse = 0.0
    n_grad_steps = 0
    all_per_step: list[dict] = []

    for k, traj in enumerate(trajectories):
        adv_k = float(advantages[k])
        if abs(adv_k) < 0.01:
            continue

        # Extract checkpoints from step_NN entries into flat dict
        grad_traj = {"eta_used": traj.get("eta_used")}
        for key in sorted(traj.keys()):
            if key.startswith("step_"):
                step_idx = int(key.split("_")[1])
                grad_traj[f"checkpoint_{step_idx}"] = traj[key]["x"]

        sparse_steps = sorted(
            int(key.split("_")[1]) for key in traj.keys()
            if key.startswith("step_"))
        # Only keep steps where we also have the next checkpoint
        sparse_steps = [s for s in sparse_steps
                        if f"checkpoint_{s + 1}" in grad_traj]

        if not sparse_steps:
            continue

        result = accumulate_reinforce_gradients(
            executor, [spec], [query_sigmas], [grad_traj],
            sparse_steps, [adv_k],
            adapter_scales=policy_scales,
            ref_adapter_scales=ref_scales,
            gather_fn=gather_fn,
        )

        total_log_prob += result["total_log_prob"]
        total_drift_mse += result["total_drift_mse"]
        n_grad_steps += result["n_steps"]
        all_per_step.extend(result.get("per_step", []))

    model.gradient_checkpointing = old_gc

    opt_result = policy_optimizer_step(optimizer, max_grad_norm)

    return {
        "total_log_prob": total_log_prob,
        "total_drift_mse": total_drift_mse,
        "n_grad_steps": n_grad_steps,
        "grad_norm": opt_result.get("grad_norm", 0.0),
        "n_params_with_grad": opt_result.get("n_params_with_grad", 0),
        "module_grad_norms": opt_result.get("module_grad_norms"),
        "per_step_diag": all_per_step,
        "lr": float(optimizer.param_groups[0]["lr"]),
    }



def generate_exemplars(executor, pos_cond, neg_cond, cap_len, seeds,
                       resolution, n_steps, cfg, device, dtype, adapter_scales):
    """Generate exemplar latents for before/after comparison."""
    trajs = generate_sde_trajectories(
        executor, pos_cond, neg_cond, cap_len, seeds, resolution,
        n_steps, cfg, eta_scale=0.0,  # deterministic ODE for exemplars
        device=device, dtype=dtype, adapter_scales=adapter_scales,
    )
    return [t["final"] for t in trajs]


def decode_and_save(latents, output_dir, prefix, device, dtype):
    """VAE-decode latents and save as PNGs."""
    _log(f"  VAE decoding {len(latents)} exemplars...")
    from src_ii.vae_utils import load_vae, decode_latent_to_pil

    vae = load_vae(VAE_PATH, device=device, dtype=dtype)

    for i, latent in enumerate(latents):
        pil = decode_latent_to_pil(vae, latent, device=device, dtype=dtype)
        path = output_dir / f"{prefix}_{i}.png"
        pil.save(str(path))
        _log(f"    Saved {path.name}")

    del vae
    torch.cuda.empty_cache()



def main():
    _log("=" * 60)
    _log("DDGRPO E2E VALIDATION — FULL Z-IMAGE WEIGHTS")
    _log("=" * 60)
    t_start = time.perf_counter()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)

    pos_conds, neg_cond, cap_lens = encode_all_prompts(PROMPTS, DEVICE, DTYPE)

    model = load_model_and_setup(DEVICE, DTYPE)

    all_cap_lens = cap_lens + [neg_cond.shape[1]]
    executor = make_local_executor(
        model, DEVICE, all_cap_lens, RESOLUTION_TIERS)

    _log("Phase 3: Warmup forward...")
    warmup_trajs = generate_sde_trajectories(
        executor,
        pos_conds[0], neg_cond, cap_lens[0],
        seeds=[12345], resolution=RESOLUTION_TIERS[0],
        n_steps=2, cfg=CFG, eta_scale=0.0,
        device=DEVICE, dtype=DTYPE,
    )
    _log(f"  Warmup done (final shape: {warmup_trajs[0]['final'].shape})")

    _log("Phase 4: Generating 'before' exemplars (fresh policy adapter)...")
    adapter_scales_both = torch.tensor(
        [[1.0, 1.0]], device=DEVICE, dtype=DTYPE)
    before_latents = generate_exemplars(
        executor, pos_conds[EXEMPLAR_PROMPT_IDX], neg_cond,
        cap_lens[EXEMPLAR_PROMPT_IDX], EXEMPLAR_SEEDS,
        EXEMPLAR_RESOLUTION, N_STEPS, CFG, DEVICE, DTYPE,
        adapter_scales=adapter_scales_both,
    )
    _log(f"  {len(before_latents)} before exemplars generated")

    _log(f"\nPhase 5: DDGRPO training ({N_ITERS} iterations, K={K})...")
    from src_ii.reward_env import make_standard_envs
    envs = make_standard_envs(("pinkify", "thisnotthat"))
    env = envs["pinkify"]

    from src_ii.multi_lora import save_adapter, load_adapter
    from src_ii.incremental_save import atomic_json_save, load_training_curve_jsonl, TrainingCurveWriter

    from src_ii.multi_lora import get_adapter_params
    policy_params = list(get_adapter_params(model, POLICY_ADAPTER_NAME).values())
    policy_optimizer = torch.optim.AdamW(policy_params, lr=LR)
    _log(f"  Policy optimizer: AdamW, {len(policy_params)} param groups, lr={LR}")

    metrics_path = OUTPUT_DIR / "training_metrics.jsonl"
    resume_path = OUTPUT_DIR / "resume_state.json"
    checkpoint_dir = OUTPUT_DIR / "checkpoints"

    start_iteration = 0
    if resume_path.exists():
        with open(str(resume_path), "r") as f:
            resume_state = json.load(f)
        last_completed = resume_state["iteration"]
        start_iteration = last_completed + 1

        adapter_ckpt = checkpoint_dir / f"policy_step_{last_completed:04d}.safetensors"
        if adapter_ckpt.exists():
            load_adapter(model, POLICY_ADAPTER_NAME, str(adapter_ckpt))
            _log(f"  Resumed adapter from {adapter_ckpt.name}")

        opt_ckpt = checkpoint_dir / "optimizer_state.pt"
        if opt_ckpt.exists():
            opt_state = torch.load(str(opt_ckpt), weights_only=True)
            policy_optimizer.load_state_dict(opt_state)
            _log(f"  Optimizer state restored from {opt_ckpt.name}")

        _log(f"  Resuming from iteration {start_iteration} (last completed: {last_completed})")

    # Load existing rewards from JSONL for reward tracking continuity
    all_rewards = []
    existing_metrics = load_training_curve_jsonl(metrics_path)
    for m in existing_metrics:
        if "mean_reward" in m:
            all_rewards.append(m["mean_reward"])

    metrics_writer = TrainingCurveWriter(metrics_path, append=(start_iteration > 0))

    sparse_step_count = 5
    save_steps = set()
    for i in range(sparse_step_count):
        s = (i * N_STEPS) // sparse_step_count
        save_steps.add(s)
        if s + 1 < N_STEPS:
            save_steps.add(s + 1)
    _log(f"  Save steps for gradient: {sorted(save_steps)}")

    for iteration in range(start_iteration, N_ITERS):
        t0 = time.perf_counter()

        prompt_idx = random.randint(0, len(PROMPTS) - 1)
        resolution = random.choice(RESOLUTION_TIERS)
        seeds = [random.randint(0, 2**31 - 1) for _ in range(K)]

        trajs = generate_sde_trajectories(
            executor,
            pos_conds[prompt_idx], neg_cond, cap_lens[prompt_idx], seeds,
            resolution, N_STEPS, CFG, ETA_SCALE,
            DEVICE, DTYPE,
            adapter_scales=adapter_scales_both,
            save_steps=save_steps,
        )

        scores = score_finals(
            executor, trajs, pos_conds[prompt_idx], resolution, DEVICE, DTYPE)

        rewards, advantages = env.compute_advantages(scores)
        all_rewards.append(float(rewards.mean()))

        policy_optimizer.zero_grad()
        grad_result = accumulate_and_step(
            executor, model, trajs, advantages,
            pos_conds[prompt_idx], neg_cond, resolution, CFG,
            policy_optimizer, MAX_GRAD_NORM,
            DEVICE, DTYPE,
        )

        elapsed = time.perf_counter() - t0

        metrics = {
            "iteration": iteration,
            "prompt_idx": prompt_idx,
            "resolution": list(resolution),
            "seeds": seeds,
            "mean_reward": float(rewards.mean()),
            "std_reward": float(rewards.std()),
            "rewards": rewards.tolist(),
            "advantages": advantages.tolist(),
            "scores": scores.tolist(),
            "total_log_prob": grad_result["total_log_prob"],
            "total_drift_mse": grad_result["total_drift_mse"],
            "n_grad_steps": grad_result["n_grad_steps"],
            "grad_norm": grad_result["grad_norm"],
            "n_params_with_grad": grad_result.get("n_params_with_grad", 0),
            "module_grad_norms": grad_result.get("module_grad_norms"),
            "per_step_diag": grad_result.get("per_step_diag"),
            "elapsed_s": elapsed,
        }
        metrics_writer.write_step(metrics)

        head_names = ("pinkify", "tnt")
        score_str = " | ".join(
            f"{h}={float(scores[:, i].mean()):+.3f}" for i, h in enumerate(head_names))
        _log(f"  iter={iteration:3d}  reward={float(rewards.mean()):+.4f}"
             f"±{float(rewards.std()):.3f}  [{score_str}]"
             f"  drift={grad_result['total_drift_mse']:.4f}"
             f"  grad={grad_result['grad_norm']:.4f}"
             f"  {resolution[0]}x{resolution[1]}  {elapsed:.1f}s")

        # -- Per-iteration persistence (crash loses at most 1 iteration) --
        adapter_ckpt_path = checkpoint_dir / f"policy_step_{iteration:04d}.safetensors"
        save_adapter(model, POLICY_ADAPTER_NAME, str(adapter_ckpt_path))

        torch.save(
            policy_optimizer.state_dict(),
            str(checkpoint_dir / "optimizer_state.pt"),
        )

        atomic_json_save({"iteration": iteration, "completed": True}, resume_path)

        del trajs, scores, rewards, advantages
        torch.cuda.empty_cache()

    metrics_writer.close()

    _log(f"\nPhase 6: Generating 'after' exemplars (trained policy)...")
    after_latents = generate_exemplars(
        executor, pos_conds[EXEMPLAR_PROMPT_IDX], neg_cond,
        cap_lens[EXEMPLAR_PROMPT_IDX], EXEMPLAR_SEEDS,
        EXEMPLAR_RESOLUTION, N_STEPS, CFG, DEVICE, DTYPE,
        adapter_scales=adapter_scales_both,
    )
    _log(f"  {len(after_latents)} after exemplars generated")

    adapter_path = OUTPUT_DIR / "policy_adapter_final.safetensors"
    save_adapter(model, POLICY_ADAPTER_NAME, str(adapter_path))
    _log(f"  Policy adapter saved: {adapter_path}")

    del model
    torch.cuda.empty_cache()

    _log("\nPhase 7: Decoding exemplars...")
    exemplar_dir = OUTPUT_DIR / "exemplars"
    exemplar_dir.mkdir(parents=True, exist_ok=True)
    decode_and_save(before_latents, exemplar_dir, "before", DEVICE, DTYPE)
    decode_and_save(after_latents, exemplar_dir, "after", DEVICE, DTYPE)

    _log("\nPhase 8: Analysis...")
    mean_first5 = sum(all_rewards[:5]) / max(len(all_rewards[:5]), 1)
    mean_last5 = sum(all_rewards[-5:]) / max(len(all_rewards[-5:]), 1)
    _log(f"  Mean reward (first 5): {mean_first5:+.4f}")
    _log(f"  Mean reward (last 5):  {mean_last5:+.4f}")
    _log(f"  Reward delta:          {mean_last5 - mean_first5:+.4f}")

    analysis = {
        "n_iters": N_ITERS,
        "K": K,
        "n_steps": N_STEPS,
        "eta_scale": ETA_SCALE,
        "lr": LR,
        "env": "pinkify",
        "mean_reward_first5": mean_first5,
        "mean_reward_last5": mean_last5,
        "reward_delta": mean_last5 - mean_first5,
        "all_rewards": all_rewards,
        "btrm_dir": str(BTRM_DIR),
        "elapsed_total_s": time.perf_counter() - t_start,
    }
    with open(OUTPUT_DIR / "run_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    _log(f"\nTotal elapsed: {time.perf_counter() - t_start:.0f}s")
    _log(f"Output: {OUTPUT_DIR}")
    _log("=" * 60)
    _log("DDGRPO E2E VALIDATION COMPLETE")
    _log("=" * 60)


if __name__ == "__main__":
    main()

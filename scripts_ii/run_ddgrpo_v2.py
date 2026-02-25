r"""DDGRPO v2: All-step gradients, funfetti batching, corrected hyperparams.

Fixes three defects from v1 (training_output/ddgrpo_256sq_fast/):
  1. Sparse step subsampling (5/30) → ALL denoising steps get gradients
  2. Underutilized hardware (B=1, K=4) → B=4 prompts × K=4 = 16 rollouts
  3. Hyperparameter mismatch → LR=2e-5, MAX_GRAD_NORM=0.1

Self-contained: loads model directly, no server needed. Full lifecycle:
  1. Load TE → encode prompts → free TE
  2. Load backbone (FP8), install multi-LoRA, compile
  3. Load BTRM checkpoint (rtheta adapter + score head)
  4. Generate "before" exemplars
  5. Run N DDGRPO iterations with full-step SDE rollout + BTRM scoring
  6. Generate "after" exemplars
  7. Load VAE → decode exemplars → save PNGs → free VAE
  8. Write metrics JSONL + analysis

See docs/essay_ddgrpo_nonconvergence_diagnosis.md for the v1 failure analysis.

Execution:
  PYTHONUNBUFFERED=1 .venv/Scripts/python.exe ^
      F:\dox\repos\ai\futudiffu\scripts_ii\run_ddgrpo_v2.py
"""

from __future__ import annotations

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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

from src_ii.model_paths import FP8_PATH, TE_PATH, VAE_PATH, TOKENIZER_PATH

BTRM_DIR = REPO_ROOT / "training_output" / "reward_function_run_tnt_v2"
OUTPUT_DIR = REPO_ROOT / "training_output" / "ddgrpo_v2"

# ---------------------------------------------------------------------------
# Corrected hyperparams (v1 → v2 changes annotated)
# ---------------------------------------------------------------------------

N_ITERS = 100          # v1=30, more iterations with smaller steps
N_STEPS = 20           # v2b: 30→20, faster rollouts
ETA_SCALE = 0.1        # unchanged
LR = 2e-5              # v1=1e-4, 5× reduction
MAX_GRAD_NORM = 0.1    # v1=1.0, match BTRM training
CFG = 4.0              # unchanged
ADAPTER_RANK = 8       # unchanged
ADAPTER_ALPHA = 16.0   # unchanged
INIT_B_STD = 0.01      # unchanged

# Batching: B prompts × K trajectories per prompt = B×K rollouts
B = 2                  # v1=1, v2=4, v2b=2
K = 2                  # v1=4, v2=4, v2b=2
# Total: 4 rollouts per iteration

# Funfetti resolution budgets (pixel counts for sampling)
# Exclude 704² and 1024² — too expensive for K=4 each at 24GB
RESOLUTION_BUDGETS = [65536, 102400, 147456, 262144]  # 256² → 512²

BTRM_ADAPTER_NAME = "rtheta"
POLICY_ADAPTER_NAME = "policy_pinkify"

ADVANTAGE_THRESHOLD = 0.01
GRAD_MICROBATCH = 1        # trajectories per backward(). 1 = lowest memory.

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

EXEMPLAR_SEEDS = [42, 137, 256, 999]
EXEMPLAR_PROMPT_IDX = 0
EXEMPLAR_RESOLUTION = (320, 208)

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16

# ---------------------------------------------------------------------------
# Logging: tee to stdout + persistent log file (survives kill/crash)
# ---------------------------------------------------------------------------

_LOG_FILE = None


def _init_log():
    """Open the persistent log file. Call once at script start."""
    global _LOG_FILE
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = open(str(OUTPUT_DIR / "run.log"), "a")
    _log("--- log opened ---")


def _log(msg):
    line = f"[DDGRPOv2] [{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _LOG_FILE is not None:
        _LOG_FILE.write(line + "\n")
        _LOG_FILE.flush()


# ---------------------------------------------------------------------------
# Phase 1: Encode prompts
# ---------------------------------------------------------------------------

def encode_all_prompts(prompts, device, dtype):
    """Load TE, encode all prompts + negative, free TE."""
    _log("Phase 1: Encoding prompts...")
    from futudiffu.text_encoder import create_tokenizer, load_text_encoder, encode_prompt

    tokenizer = create_tokenizer(TOKENIZER_PATH)
    te_model = load_text_encoder(TE_PATH, device=device, dtype=dtype)
    te_compiled = torch.compile(te_model, mode="default")

    pos_conds = []
    cap_lens = []
    with torch.inference_mode():
        for i, prompt in enumerate(prompts):
            cond = encode_prompt(te_compiled, tokenizer, prompt, device=device, layer_idx=-2)
            pos_conds.append(cond.cpu())
            cap_lens.append(cond.shape[1])
            _log(f"  [{i}] {prompt[:60]}... -> {cond.shape}")

        neg_cond = encode_prompt(te_compiled, tokenizer, "", device=device, layer_idx=-2)
        neg_cond = neg_cond.cpu()
        _log(f"  [neg] -> {neg_cond.shape}")

    del te_model, te_compiled, tokenizer
    torch.cuda.empty_cache()
    _log(f"  TE freed. {len(pos_conds)} prompts encoded.")
    return pos_conds, neg_cond, cap_lens


# ---------------------------------------------------------------------------
# Phase 2: Load model + adapters
# ---------------------------------------------------------------------------

def load_model_and_setup(device, dtype):
    """Load backbone, install adapters, load BTRM checkpoint."""
    _log("Phase 2: Loading backbone...")
    from src_ii.zimage_model import load_zimage_rlaif
    from src_ii.multi_lora import install_multi_lora, init_adapter_b_weights, freeze_base_params
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

    n_frozen, n_total = freeze_base_params(model)
    _log(f"  Base frozen: {n_frozen}/{n_total} params")

    model.compile_for_execution()
    _log("  compile_for_execution done")

    return model


# ---------------------------------------------------------------------------
# Executor (local, wraps BatchExecutor — same as validate_ddgrpo_e2e.py)
# ---------------------------------------------------------------------------

def make_local_executor(model, device):
    """Build a local executor wrapping BatchExecutor for ktuple_sampling.

    executor_fn is called as: executor(x_bases, specs, step_i, adapter_scales)
    The caller must set executor_fn.query_sigmas before the solve loop.

    Uses REFERENCE_TOTAL_LEN — the fixed bin capacity sized for 1280x832 +
    p90 text overhead. One compiled graph, zero recompiles, and a cfg2 pair
    at 512² fits in a single bin with room to spare.
    """
    from src_ii.batch_executor import BatchExecutor
    from src_ii.bin_packer import REFERENCE_TOTAL_LEN

    _log(f"  BatchExecutor max_total_len={REFERENCE_TOTAL_LEN}")

    batch_exec = BatchExecutor(model, device=device, max_total_len=REFERENCE_TOTAL_LEN)

    class ExecutorWithSigmas:
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
                k_idx = int(r["query_id"][1:])
                j_idx = int(r["entry_id"][1:])
                buckets[k_idx].append((j_idx, r["denoised"].to(device)))
                if r["scores"] is not None:
                    score_list.append(r["scores"])

            denoised_per_query = []
            for k_idx in range(len(specs)):
                entries_sorted = sorted(buckets[k_idx], key=lambda t: t[0])
                denoised_per_query.append([d for _, d in entries_sorted])

            scores = torch.stack(score_list) if score_list else None
            return denoised_per_query, scores

    return ExecutorWithSigmas()


# ---------------------------------------------------------------------------
# Rollout generation (SDE trajectories)
# ---------------------------------------------------------------------------

def generate_sde_trajectories(
    executor, pos_cond, neg_cond, cap_len, seeds, resolution,
    n_steps, cfg, eta_scale, device, dtype,
    adapter_scales=None, save_steps=None,
):
    """Generate K SDE trajectories for one prompt+resolution group."""
    from src_ii.ktuple_sampling import solve_sde
    from src_ii.sigma_schedule import build_sigma_schedule, resolution_shift, const_inverse_noise_scaling
    from src_ii.triumphant_future_reduction_ops import cfg2
    from src_ii.ddreinforce import compute_eta_schedule

    K_local = len(seeds)
    w, h = resolution

    specs = [cfg2(pos_cond.to(device, dtype), neg_cond.to(device, dtype),
                  resolution, cfg) for _ in range(K_local)]

    alpha = resolution_shift(w, h)
    query_sigmas = [
        build_sigma_schedule(n_steps, sampling_shift=alpha, device=device, dtype=dtype)
        for _ in range(K_local)
    ]

    eta_schedule = compute_eta_schedule(query_sigmas[0], eta_scale=eta_scale)

    x_bases = []
    for k in range(K_local):
        gen = torch.Generator(device=device).manual_seed(seeds[k])
        x_bases.append(
            query_sigmas[k][0] * torch.randn(
                1, 16, h // 8, w // 8, dtype=dtype, device=device, generator=gen))

    executor.query_sigmas = query_sigmas

    trajectories = [{} for _ in range(K_local)]

    def save_fn(i, x_pres, guided_list):
        if save_steps and i in save_steps:
            for k in range(K_local):
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

    for k in range(K_local):
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


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_finals(executor, trajectories, pos_conds_by_group, resolutions_by_group,
                 device, dtype):
    """Score all final latents in one batch. Returns (total_K, n_heads) tensor.

    Args:
        trajectories: flat list of trajectory dicts (B×K total).
        pos_conds_by_group: list of (pos_cond, count) tuples mapping each
            trajectory to its prompt conditioning.
        resolutions_by_group: flat list of (w,h) per trajectory.
    """
    queries = []
    for k, traj in enumerate(trajectories):
        cond, _ = pos_conds_by_group[k]
        cond = cond.to(device, dtype)
        final = traj["final"].to(device, dtype)
        res = resolutions_by_group[k]
        queries.append({
            "query_id": f"score_{k}",
            "base_latent": final,
            "base_cond": cond,
            "base_cap_len": cond.shape[1],
            "base_resolution": res,
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


# ---------------------------------------------------------------------------
# Gradient accumulation + optimizer step (all steps, not sparse)
# ---------------------------------------------------------------------------

def accumulate_and_step(
    executor, model, groups, all_trajectories, all_advantages,
    neg_cond, cfg, optimizer, max_grad_norm, device, dtype,
    sign_tracker=None,
):
    """Accumulate REINFORCE gradients for all B×K trajectories, then step.

    Caller owns optimizer creation (at setup) and zero_grad (before this call).
    This function: accumulate gradients → clip → step.
    """
    from src_ii.policy_step import accumulate_reinforce_gradients, policy_optimizer_step
    from src_ii.triumphant_future_reduction_ops import cfg2

    policy_scales = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
    ref_scales = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)

    old_gc = getattr(model, 'gradient_checkpointing', False)
    model.gradient_checkpointing = True

    # Build per-trajectory inputs, filtering by advantage threshold
    specs, query_sigmas_list, grad_trajs, advantages_list = [], [], [], []
    n_skipped = 0

    for k, (traj, adv_k) in enumerate(zip(all_trajectories, all_advantages)):
        adv_val = float(adv_k)
        if abs(adv_val) < ADVANTAGE_THRESHOLD:
            n_skipped += 1
            continue

        group = groups[k // K]
        spec = cfg2(group["pos_cond"].to(device, dtype),
                    neg_cond.to(device, dtype),
                    group["resolution"], cfg)
        specs.append(spec)

        query_sigmas_list.append(
            torch.tensor(traj["sigmas"], dtype=torch.float32, device=device))

        grad_traj = {"eta_used": traj.get("eta_used")}
        for key in sorted(traj.keys()):
            if key.startswith("step_"):
                step_idx = int(key.split("_")[1])
                grad_traj[f"checkpoint_{step_idx}"] = traj[key]["x"]
        grad_trajs.append(grad_traj)

        advantages_list.append(adv_val)

    if not grad_trajs:
        model.gradient_checkpointing = old_gc
        opt_result = policy_optimizer_step(optimizer, max_grad_norm,
                                           sign_tracker=sign_tracker)
        return {
            "total_log_prob": 0.0,
            "total_drift_mse": 0.0,
            "n_grad_steps": 0,
            "n_skipped": len(all_trajectories),
            "grad_norm": opt_result["grad_norm"],
            "n_params_with_grad": opt_result["n_params_with_grad"],
            "per_step_diag": [],
            "lr": opt_result["lr"],
            "sign_agreement": opt_result.get("sign_agreement"),
        }

    # Gradient steps from first active trajectory (all share same step set)
    all_steps = sorted(
        int(k.split("_")[1]) for k in grad_trajs[0].keys()
        if k.startswith("checkpoint_"))
    gradient_steps = [s for s in all_steps
                      if f"checkpoint_{s + 1}" in grad_trajs[0]]

    result = accumulate_reinforce_gradients(
        executor, specs, query_sigmas_list, grad_trajs,
        gradient_steps, advantages_list,
        adapter_scales=policy_scales,
        ref_adapter_scales=ref_scales,
        microbatch_size=GRAD_MICROBATCH,
    )

    model.gradient_checkpointing = old_gc

    opt_result = policy_optimizer_step(optimizer, max_grad_norm,
                                       sign_tracker=sign_tracker)

    return {
        "total_log_prob": result["total_log_prob"],
        "total_drift_mse": result["total_drift_mse"],
        "n_grad_steps": result["n_steps"],
        "n_skipped": n_skipped,
        "grad_norm": opt_result["grad_norm"],
        "n_params_with_grad": opt_result["n_params_with_grad"],
        "per_step_diag": result.get("per_step", []),
        "lr": opt_result["lr"],
        "sign_agreement": opt_result.get("sign_agreement"),
    }


# ---------------------------------------------------------------------------
# Exemplar generation + decode
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Per-step diagnostic summary
# ---------------------------------------------------------------------------

def summarize_per_step(per_step_diag):
    """Aggregate per-step diagnostics into a summary dict."""
    from collections import defaultdict
    by_step = defaultdict(list)
    for d in per_step_diag:
        if d.get("skipped"):
            continue
        by_step[d["step_idx"]].append(d)

    mean_log_prob_by_step = {}
    mean_inv_eta_sq_by_step = {}
    for step_idx, entries in sorted(by_step.items()):
        mean_log_prob_by_step[step_idx] = (
            sum(e["log_prob"] for e in entries) / len(entries))
        mean_inv_eta_sq_by_step[step_idx] = (
            sum(e["inv_eta_sq"] for e in entries) / len(entries))

    return {
        "mean_log_prob_by_step": mean_log_prob_by_step,
        "mean_inv_eta_sq_by_step": mean_inv_eta_sq_by_step,
        "n_steps_with_data": len(by_step),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    _init_log()
    _log("=" * 60)
    _log("DDGRPO v2 — ALL-STEP GRADIENTS, FUNFETTI BATCHING")
    _log(f"B={B} prompts x K={K} trajectories = {B*K} rollouts/iter")
    _log(f"LR={LR}, MAX_GRAD_NORM={MAX_GRAD_NORM}, N_ITERS={N_ITERS}")
    _log("=" * 60)
    t_start = time.perf_counter()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = OUTPUT_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: Encode prompts ----
    pos_conds, neg_cond, cap_lens = encode_all_prompts(PROMPTS, DEVICE, DTYPE)

    # ---- Phase 2: Load model ----
    model = load_model_and_setup(DEVICE, DTYPE)

    # ---- Build executor ----
    executor = make_local_executor(model, DEVICE)

    # ---- Phase 3: Warmup ----
    _log("Phase 3: Warmup forward...")
    warmup_trajs = generate_sde_trajectories(
        executor,
        pos_conds[0], neg_cond, cap_lens[0],
        seeds=[12345], resolution=EXEMPLAR_RESOLUTION,
        n_steps=2, cfg=CFG, eta_scale=0.0,
        device=DEVICE, dtype=DTYPE,
    )
    _log(f"  Warmup done (final shape: {warmup_trajs[0]['final'].shape})")

    # ---- Phase 4: Before exemplars ----
    _log("Phase 4: Generating 'before' exemplars...")
    adapter_scales_both = torch.tensor([[1.0, 1.0]], device=DEVICE, dtype=DTYPE)
    before_latents = generate_exemplars(
        executor, pos_conds[EXEMPLAR_PROMPT_IDX], neg_cond,
        cap_lens[EXEMPLAR_PROMPT_IDX], EXEMPLAR_SEEDS,
        EXEMPLAR_RESOLUTION, N_STEPS, CFG, DEVICE, DTYPE,
        adapter_scales=adapter_scales_both,
    )
    _log(f"  {len(before_latents)} before exemplars generated")

    # ---- Phase 5: Training loop ----
    _log(f"\nPhase 5: DDGRPO v2 training ({N_ITERS} iterations)...")

    from src_ii.reward_env import make_standard_envs
    from src_ii.multi_lora import save_adapter, load_adapter, get_adapter_params
    from src_ii.incremental_save import atomic_json_save, load_training_curve_jsonl, TrainingCurveWriter
    from src_ii.resolution_sampling import sample_random_resolution as _sample_res
    from src_ii.ddreinforce import group_advantages
    from src_ii.policy_step import SignAgreementTracker

    envs = make_standard_envs(("pinkify", "thisnotthat"))
    env = envs["pinkify"]

    # Create optimizer at setup — not lazily inside the step function
    policy_params = list(get_adapter_params(model, POLICY_ADAPTER_NAME).values())
    policy_optimizer = torch.optim.AdamW(policy_params, lr=LR)
    _log(f"  Policy optimizer: AdamW, {len(policy_params)} param groups, lr={LR}")

    sign_tracker = SignAgreementTracker(policy_optimizer)
    n_tracked = sum(p.numel() for g in policy_optimizer.param_groups for p in g["params"])
    _log(f"  SignAgreementTracker: {n_tracked} params tracked")

    metrics_path = OUTPUT_DIR / "training_metrics.jsonl"
    resume_path = OUTPUT_DIR / "resume_state.json"

    # ALL steps saved for gradient computation
    save_steps = set(range(N_STEPS))
    _log(f"  Save steps: ALL {len(save_steps)} steps (v1 used 5)")

    # Resume logic
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
            try:
                policy_optimizer.load_state_dict(
                    torch.load(str(opt_ckpt), weights_only=True))
                _log(f"  Optimizer state restored")
            except Exception as e:
                _log(f"  Warning: could not restore optimizer state: {e}")

        _log(f"  Resuming from iteration {start_iteration}")

    all_rewards = []
    existing_metrics = load_training_curve_jsonl(metrics_path)
    for m in existing_metrics:
        if "mean_reward" in m:
            all_rewards.append(m["mean_reward"])

    metrics_writer = TrainingCurveWriter(metrics_path, append=(start_iteration > 0))
    training_rng = random.Random(42 + start_iteration)

    # Write run config
    run_config = {
        "version": "ddgrpo_v2",
        "B": B, "K": K, "N_ITERS": N_ITERS, "N_STEPS": N_STEPS,
        "LR": LR, "MAX_GRAD_NORM": MAX_GRAD_NORM, "ETA_SCALE": ETA_SCALE,
        "CFG": CFG, "ADAPTER_RANK": ADAPTER_RANK, "ADAPTER_ALPHA": ADAPTER_ALPHA,
        "INIT_B_STD": INIT_B_STD, "RESOLUTION_BUDGETS": RESOLUTION_BUDGETS,
        "GRAD_MICROBATCH": GRAD_MICROBATCH,
        "gradient_steps": "all",
        "env": "pinkify",
        "btrm_dir": str(BTRM_DIR),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(OUTPUT_DIR / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    for iteration in range(start_iteration, N_ITERS):
        t0 = time.perf_counter()

        # ---- Step 1: Sample B prompt-resolution groups ----
        groups = []
        for b in range(B):
            prompt_idx = training_rng.randint(0, len(PROMPTS) - 1)
            budget = training_rng.choice(RESOLUTION_BUDGETS)
            resolution = _sample_res(budget, training_rng)
            seeds = [training_rng.randint(0, 2**31 - 1) for _ in range(K)]
            groups.append({
                "prompt_idx": prompt_idx,
                "pos_cond": pos_conds[prompt_idx],
                "cap_len": cap_lens[prompt_idx],
                "resolution": resolution,
                "budget": budget,
                "seeds": seeds,
            })

        # ---- Step 2: Generate B×K rollouts ----
        # Each group generates K trajectories sequentially
        # (groups could be parallelized on multi-GPU, but sequential on 1 GPU)
        all_trajectories = []
        t_rollout = time.perf_counter()
        for b, group in enumerate(groups):
            trajs = generate_sde_trajectories(
                executor,
                group["pos_cond"], neg_cond, group["cap_len"],
                group["seeds"], group["resolution"],
                N_STEPS, CFG, ETA_SCALE,
                DEVICE, DTYPE,
                adapter_scales=adapter_scales_both,
                save_steps=save_steps,
            )
            all_trajectories.extend(trajs)
        t_rollout = time.perf_counter() - t_rollout

        # ---- Step 3: Score all B×K finals ----
        t_score = time.perf_counter()
        # Build per-trajectory cond/resolution mapping
        cond_map = []
        res_map = []
        for b, group in enumerate(groups):
            for _ in range(K):
                cond_map.append((group["pos_cond"], group["cap_len"]))
                res_map.append(group["resolution"])

        all_scores = score_finals(
            executor, all_trajectories, cond_map, res_map, DEVICE, DTYPE)
        t_score = time.perf_counter() - t_score

        # ---- Step 4: Per-group advantages ----
        all_rewards_iter = []
        all_advantages = []
        group_metrics = []
        for b, group in enumerate(groups):
            start = b * K
            end = start + K
            group_scores = all_scores[start:end]

            rewards, advantages = env.compute_advantages(group_scores)
            all_rewards_iter.extend(rewards.tolist())
            all_advantages.extend(advantages.tolist())

            group_metrics.append({
                "prompt_idx": group["prompt_idx"],
                "resolution": list(group["resolution"]),
                "budget": group["budget"],
                "seeds": group["seeds"],
                "rewards": rewards.tolist(),
                "advantages": advantages.tolist(),
                "scores": group_scores.tolist(),
            })

        advantages_tensor = torch.tensor(all_advantages, dtype=torch.float32)
        mean_reward = sum(all_rewards_iter) / len(all_rewards_iter)
        all_rewards.append(mean_reward)

        # ---- Step 5: Accumulate REINFORCE gradients (all steps) ----
        t_grad = time.perf_counter()
        policy_optimizer.zero_grad()
        grad_result = accumulate_and_step(
            executor, model, groups, all_trajectories, advantages_tensor,
            neg_cond, CFG, policy_optimizer, MAX_GRAD_NORM,
            DEVICE, DTYPE,
            sign_tracker=sign_tracker,
        )
        t_grad = time.perf_counter() - t_grad

        elapsed = time.perf_counter() - t0

        # ---- Step 6: Persist ----
        per_step_summary = summarize_per_step(grad_result.get("per_step_diag", []))

        sign_agree = grad_result.get("sign_agreement")

        metrics = {
            "iteration": iteration,
            "groups": group_metrics,
            "mean_reward": mean_reward,
            "std_reward": float(torch.tensor(all_rewards_iter).std()),
            "total_log_prob": grad_result["total_log_prob"],
            "total_drift_mse": grad_result["total_drift_mse"],
            "n_grad_steps": grad_result["n_grad_steps"],
            "n_skipped": grad_result["n_skipped"],
            "grad_norm": grad_result["grad_norm"],
            "n_params_with_grad": grad_result.get("n_params_with_grad", 0),
            "per_step_summary": per_step_summary,
            "elapsed_s": elapsed,
            "t_rollout_s": t_rollout,
            "t_score_s": t_score,
            "t_grad_s": t_grad,
            "lr": grad_result.get("lr", LR),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if sign_agree is not None:
            metrics["sign_agreement"] = sign_agree
        metrics_writer.write_step(metrics)

        # Log
        res_summary = ", ".join(
            f"{g['resolution'][0]}x{g['resolution'][1]}" for g in groups)
        sign_str = (f"  sign={sign_agree['sign_agreement']:.3f}"
                    if sign_agree else "")
        _log(f"  iter={iteration:3d}  reward={mean_reward:+.4f}"
             f"  grad={grad_result['grad_norm']:.4f}"
             f"  steps={grad_result['n_grad_steps']}"
             f"  skip={grad_result['n_skipped']}"
             f"{sign_str}"
             f"  [{res_summary}]"
             f"  {elapsed:.1f}s (roll={t_rollout:.1f} score={t_score:.1f} grad={t_grad:.1f})")

        # Checkpoint every iteration (adapters are tiny, ~29KB)
        adapter_ckpt_path = checkpoint_dir / f"policy_step_{iteration:04d}.safetensors"
        save_adapter(model, POLICY_ADAPTER_NAME, str(adapter_ckpt_path))

        torch.save(
            policy_optimizer.state_dict(),
            str(checkpoint_dir / "optimizer_state.pt"),
        )

        atomic_json_save({"iteration": iteration, "completed": True}, resume_path)

        # Free trajectory storage
        del all_trajectories, all_scores
        torch.cuda.empty_cache()

    metrics_writer.close()

    # ---- Phase 6: After exemplars ----
    _log(f"\nPhase 6: Generating 'after' exemplars (trained policy)...")
    after_latents = generate_exemplars(
        executor, pos_conds[EXEMPLAR_PROMPT_IDX], neg_cond,
        cap_lens[EXEMPLAR_PROMPT_IDX], EXEMPLAR_SEEDS,
        EXEMPLAR_RESOLUTION, N_STEPS, CFG, DEVICE, DTYPE,
        adapter_scales=adapter_scales_both,
    )
    _log(f"  {len(after_latents)} after exemplars generated")

    # Save final adapter
    adapter_path = OUTPUT_DIR / "policy_adapter_final.safetensors"
    save_adapter(model, POLICY_ADAPTER_NAME, str(adapter_path))
    _log(f"  Policy adapter saved: {adapter_path}")

    del model
    torch.cuda.empty_cache()

    # ---- Phase 7: Decode exemplars ----
    _log("\nPhase 7: Decoding exemplars...")
    exemplar_dir = OUTPUT_DIR / "exemplars"
    exemplar_dir.mkdir(parents=True, exist_ok=True)
    decode_and_save(before_latents, exemplar_dir, "before", DEVICE, DTYPE)
    decode_and_save(after_latents, exemplar_dir, "after", DEVICE, DTYPE)

    # ---- Phase 8: Analysis ----
    _log("\nPhase 8: Analysis...")
    n_completed = len(all_rewards)
    if n_completed >= 10:
        mean_first5 = sum(all_rewards[:5]) / 5
        mean_last5 = sum(all_rewards[-5:]) / 5
    elif n_completed >= 2:
        half = max(1, n_completed // 2)
        mean_first5 = sum(all_rewards[:half]) / half
        mean_last5 = sum(all_rewards[-half:]) / half
    else:
        mean_first5 = all_rewards[0] if all_rewards else 0.0
        mean_last5 = mean_first5

    _log(f"  Mean reward (first 5):  {mean_first5:+.4f}")
    _log(f"  Mean reward (last 5):   {mean_last5:+.4f}")
    _log(f"  Reward delta:           {mean_last5 - mean_first5:+.4f}")

    analysis = {
        "version": "ddgrpo_v2",
        "n_iters_completed": n_completed,
        "n_iters_target": N_ITERS,
        "B": B, "K": K, "total_rollouts_per_iter": B * K,
        "gradient_steps_per_trajectory": "all (N_STEPS-1)",
        "lr": LR, "max_grad_norm": MAX_GRAD_NORM,
        "eta_scale": ETA_SCALE,
        "env": "pinkify",
        "mean_reward_first5": mean_first5,
        "mean_reward_last5": mean_last5,
        "reward_delta": mean_last5 - mean_first5,
        "all_rewards": all_rewards,
        "btrm_dir": str(BTRM_DIR),
        "elapsed_total_s": time.perf_counter() - t_start,
        "v1_comparison": {
            "v1_sparse_steps": 5,
            "v2_gradient_steps": "all (29)",
            "v1_lr": 1e-4,
            "v2_lr": LR,
            "v1_max_grad_norm": 1.0,
            "v2_max_grad_norm": MAX_GRAD_NORM,
            "v1_B": 1,
            "v2_B": B,
            "v1_total_rollouts": 4,
            "v2_total_rollouts": B * K,
        },
    }
    with open(OUTPUT_DIR / "run_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    _log(f"\nTotal elapsed: {time.perf_counter() - t_start:.0f}s")
    _log(f"Output: {OUTPUT_DIR}")
    _log("=" * 60)
    _log("DDGRPO v2 COMPLETE")
    _log("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        tb = traceback.format_exc()
        _log(f"FATAL EXCEPTION:\n{tb}")
        raise

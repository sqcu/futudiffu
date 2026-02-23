r"""Binary search over learning rate for r_theta BTRM adapter.

This is the CORRECT approach: each training step runs a full forward+backward
through the 6B parameter diffusion model on CUDA with gradients flowing
through the adapter's LoRA matrices. The hidden states are NOT detached
from the computation graph.

v2 fixes (from VRAM audit):
  - torch.compile(model, mode="default") ENABLED -- was compile_model=False
  - Adapter allocated ONCE before compile; between probes only re-init weights
    (allocate_adapter is idempotent, init_adapter_weights is graph-invariant)
  - NO remove_all_adapters between same-config probes (graph-mutating = invalidates compile)
  - VRAM instrumentation at every major phase boundary
  - Properly frees text encoder before diffusion model load

Phase 1: 5 LR probes x 100 steps each

Execution:
  PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe ^
      F:\dox\repos\ai\futudiffu\scripts_ii\sweep_rtheta_lr.py

Output:
  rtheta_sweep_output_v2/
    lr_{lr}/training_curve.json, final_metrics.json, adapter+head
    sweep_summary.json
"""

from __future__ import annotations

import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.nn as nn

def vram_report(phase: str) -> dict:
    """Print and return VRAM stats at a named phase boundary."""
    alloc_gb = torch.cuda.memory_allocated() / (1024**3)
    max_alloc_gb = torch.cuda.max_memory_allocated() / (1024**3)
    reserved_gb = torch.cuda.memory_reserved() / (1024**3)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
    report = {
        "phase": phase,
        "allocated_gb": round(alloc_gb, 3),
        "max_allocated_gb": round(max_alloc_gb, 3),
        "reserved_gb": round(reserved_gb, 3),
        "total_gb": round(total_gb, 3),
    }
    print(f"  [VRAM] {phase}: {alloc_gb:.3f} GB allocated, "
          f"{max_alloc_gb:.3f} GB peak, {reserved_gb:.3f} GB reserved "
          f"(of {total_gb:.1f} GB total)")
    return report


FP8_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
TE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"
TOKENIZER_PATH = str(REPO_ROOT / "src" / "futudiffu" / "tokenizer")
OUTPUT_DIR = REPO_ROOT / "rtheta_sweep_output_v2"
LABELS_PATH = REPO_ROOT / "pinkify_thisnotthat_output" / "preference_labels.json"
SCORES_PATH = REPO_ROOT / "pinkify_thisnotthat_output" / "per_image_scores.json"
MANIFEST_PATH = REPO_ROOT / "btrm_dataset" / "manifest.json"

HEAD_NAMES = ("pinkify", "thisnotthat")
PREF_KEYS = ("pinkify_pref", "thisnotthat_pref")
N_TRAJECTORIES = 10  # default; overridable via --n-trajectories
LOGSQUARE_WEIGHT = 0.05

PROBE_LRS = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
PROBE_STEPS = 100
WINNER_STEPS = 1000

RANK_SWEEP = [8, 32, 64]
LAYER_SUBSETS = {
    "all_30": None,
    "last_15": set(range(15, 30)),
    "last_8": set(range(22, 30)),
}
INIT_B_STD_SWEEP = [0.01, 0.1]


@dataclass
class ProbeResult:
    name: str
    lr: float
    n_steps: int
    rank: int
    init_b_std: float
    layer_subset: str
    final_loss: float
    min_loss: float
    final_acc_pinkify: float
    final_acc_thisnotthat: float
    best_acc_pinkify: float
    best_acc_thisnotthat: float
    mean_acc_pinkify: float
    mean_acc_thisnotthat: float
    ema_loss_final: float
    loss_std_last_20: float
    mean_step_time_s: float
    total_time_s: float
    n_adapter_params: int
    mean_grad_norm: float


def load_training_data() -> tuple[list[dict], list[dict], dict]:
    """Load preference labels, per-image scores, and manifest."""
    with open(LABELS_PATH) as f:
        label_data = json.load(f)
    labels = label_data["labels"]

    with open(SCORES_PATH) as f:
        per_image_scores = json.load(f)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    return labels, per_image_scores, manifest


def build_training_pairs(
    labels: list[dict],
    per_image_scores: list[dict],
) -> list[dict]:
    """Build training pairs with index references."""
    image_index = {}
    for i, entry in enumerate(per_image_scores):
        key = (entry["traj_idx"], entry["step_key"])
        image_index[key] = i

    training_pairs = []
    for label in labels:
        traj_idx = label["traj_idx"]
        key_a = (traj_idx, label["step_a"])
        key_b = (traj_idx, label["step_b"])

        idx_a = image_index.get(key_a)
        idx_b = image_index.get(key_b)
        if idx_a is None or idx_b is None:
            continue

        training_pairs.append({
            "idx_a": idx_a,
            "idx_b": idx_b,
            "pinkify_pref": label["pinkify_preference"],
            "thisnotthat_pref": label["thisnotthat_preference"],
        })
    return training_pairs


def encode_all_prompts(records, device, dtype, trajectory_indices=None):
    """Encode all unique prompts with the TE, then free TE from VRAM.

    Args:
        records: Manifest records list.
        device: CUDA device.
        dtype: Model dtype.
        trajectory_indices: Set/list of trajectory indices to encode prompts for.
            Defaults to range(N_TRAJECTORIES) for backward compat.
    """
    from futudiffu.text_encoder import create_tokenizer, encode_prompt, load_text_encoder

    if trajectory_indices is None:
        trajectory_indices = range(N_TRAJECTORIES)

    tokenizer = create_tokenizer(TOKENIZER_PATH)
    te_model = load_text_encoder(TE_PATH, device=device, dtype=dtype)

    unique_prompts = {}
    for traj_idx in trajectory_indices:
        prompt = records[traj_idx]["prompt"]
        if prompt not in unique_prompts:
            cond = encode_prompt(te_model, tokenizer, prompt, device=device)
            unique_prompts[prompt] = cond.cpu()
            print(f"  Encoded: '{prompt[:60]}...' -> {cond.shape}")

    del te_model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  TE freed. {len(unique_prompts)} unique prompts encoded.")
    return unique_prompts


def build_latent_loader(
    per_image_scores: list[dict],
    records: list[dict],
    unique_prompts: dict[str, torch.Tensor],
    diff_model: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
):
    """Build a closure that loads (latent, timestep, conditioning, num_tokens, rope_cache)
    for a given image index. Caches conditioning and rope_cache per trajectory."""
    from src_ii.stats import sigma_for_step
    from futudiffu.sampling import make_rope_cache

    cache = {}

    def load_latent(image_idx: int):
        if image_idx in cache:
            return cache[image_idx]

        entry = per_image_scores[image_idx]
        traj_idx = entry["traj_idx"]
        step_key = entry["step_key"]
        prompt = records[traj_idx]["prompt"]

        traj_dir = REPO_ROOT / "btrm_dataset" / "latents" / f"traj_{traj_idx:06d}"
        pt_path = traj_dir / f"{step_key}.pt"
        latent = torch.load(str(pt_path), weights_only=True).to(device=device, dtype=dtype)

        conditioning = unique_prompts[prompt].to(device=device, dtype=dtype)
        num_tokens = conditioning.shape[1]

        n_steps = records[traj_idx]["n_steps"]
        sigma_val = sigma_for_step(step_key, n_steps, device=device, dtype=dtype)
        timestep = sigma_val.unsqueeze(0)

        B, C, H, W = latent.shape
        rope_cache = make_rope_cache(diff_model, H, W, num_tokens, device)

        result = (latent, timestep, conditioning, num_tokens, rope_cache)
        cache[image_idx] = result
        return result

    return load_latent


def build_sigma_lookup(
    per_image_scores: list[dict],
    records: list[dict],
) -> dict[int, float]:
    """Build a dict mapping image_idx -> sigma (float) for log-SNR loss weighting.

    Pre-computes all sigma values on CPU so the lookup in the training loop is
    just a dict access with no torch overhead.
    """
    from src_ii.stats import sigma_for_step

    sigma_map = {}
    for image_idx, entry in enumerate(per_image_scores):
        traj_idx = entry["traj_idx"]
        step_key = entry["step_key"]
        n_steps = records[traj_idx]["n_steps"]
        sigma_val = sigma_for_step(step_key, n_steps, device="cpu", dtype=torch.float32)
        sigma_map[image_idx] = float(sigma_val.item())
    return sigma_map


def parse_trajectory_range(range_str: str) -> list[int]:
    """Parse a trajectory range string like '0-9' or '0-9,40-49' into indices.

    Supports:
        '10'        -> [10]       (single trajectory index)
        '0-9'       -> [0,1,...,9]
        '0-9,40-49' -> [0,1,...,9,40,41,...,49]
        '5,10,15'   -> [5,10,15]

    Returns sorted, deduplicated list of indices.
    """
    indices = set()
    for part in range_str.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.update(range(int(lo), int(hi) + 1))
        else:
            indices.add(int(part))
    return sorted(indices)


def remove_all_adapters(model: nn.Module):
    """Remove all LoRA wrappers, restoring base modules."""
    from futudiffu.lora import LoRALinear
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            parts = name.rsplit(".", 1)
            parent_path = parts[0] if len(parts) == 2 else ""
            attr_name = parts[-1]
            replacements.append((parent_path, attr_name, module.base))

    module_map = dict(model.named_modules())
    for parent_path, attr_name, base in replacements:
        parent = module_map[parent_path] if parent_path else model
        setattr(parent, attr_name, base)


def _make_streaming_callback(jsonl_path: Path):
    """Create a callback that streams per-step metrics to JSONL + tracks VRAM peak.

    The callback fires after every training step (via train_btrm_differentiable's
    callback parameter). It:
      1. Records torch.cuda.max_memory_allocated() as vram_peak_gb
      2. Records torch.cuda.memory_allocated() as vram_allocated_gb
      3. Resets peak stats so next step's peak is isolated
      4. Appends the entry as one JSON line (crash-safe: flushed immediately)
    """
    fh = open(jsonl_path, "w")

    def callback(step: int, entry: dict):
        entry["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / (1024**3), 3)
        entry["vram_allocated_gb"] = round(torch.cuda.memory_allocated() / (1024**3), 3)
        torch.cuda.reset_peak_memory_stats()
        fh.write(json.dumps(entry) + "\n")
        fh.flush()

    callback._fh = fh  # prevent GC closing the file
    return callback


def run_probe(
    diff_model: nn.Module,
    training_pairs: list[dict],
    load_latent_fn,
    device: torch.device,
    *,
    name: str,
    lr: float,
    n_steps: int,
    rank: int = 8,
    init_b_std: float = 0.01,
    layer_indices: set[int] | None = None,
    layer_subset_name: str = "all_30",
    output_dir: Path,
    warmup_steps: int = 40,
    vram_timeline: list | None = None,
    use_sigma_weighting: bool = True,
    sigma_lookup_fn=None,
    optimizer_type: str = "adam",
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    grad_accum_steps: int = 1,
    logsnr_threshold: float = 10.0,
    logsnr_interval: float = 5.0,
    logsnr_decay_rate: float = 0.5,
) -> ProbeResult:
    """Run a single probe: re-init adapter weights, create head, train, persist.

    IMPORTANT: Adapter structure is pre-allocated on diff_model before compile.
    install_multi_lora is IDEMPOTENT (same name = no-op, no graph mutation).
    Between probes we do NOT remove adapters -- that would strip MultiLoRALinear
    wrappers and invalidate the compiled graph.
    """
    from src_ii.btrm_lifecycle import setup_btrm_training, persist_btrm
    from src_ii.multi_lora import get_adapter_params
    from src_ii.btrm_training import train_btrm_differentiable

    probe_dir = output_dir / name
    probe_dir.mkdir(parents=True, exist_ok=True)

    optimizer = setup_btrm_training(
        diff_model,
        adapter_name="rtheta",
        adapter_rank=rank,
        adapter_init_b_std=init_b_std,
    )

    n_adapter_params = sum(p.numel() for p in get_adapter_params(diff_model, "rtheta").values())

    streaming_cb = _make_streaming_callback(probe_dir / "training_curve.jsonl")
    torch.cuda.reset_peak_memory_stats()  # baseline before first step

    t0 = time.perf_counter()
    training_curve = train_btrm_differentiable(
        model=diff_model,
        training_pairs=training_pairs,
        load_latent_fn=load_latent_fn,
        n_steps=n_steps,
        lr=lr,
        logsquare_weight=LOGSQUARE_WEIGHT,
        head_names=HEAD_NAMES,
        pref_keys=PREF_KEYS,
        gradient_checkpointing=True,
        max_grad_norm=0.1,  # v2 sweep: 0.01 clip-saturated all probes (pre-clip norms 15-750x above)
        log_interval=max(1, n_steps // 20),
        warmup_steps=warmup_steps,
        callback=streaming_cb,
        use_sigma_weighting=use_sigma_weighting,
        sigma_lookup_fn=sigma_lookup_fn,
        optimizer_type=optimizer_type,
        muon_lr=muon_lr,
        muon_momentum=muon_momentum,
        grad_accum_steps=grad_accum_steps,
        logsnr_threshold=logsnr_threshold,
        logsnr_interval=logsnr_interval,
        logsnr_decay_rate=logsnr_decay_rate,
    )
    total_time = time.perf_counter() - t0

    streaming_cb._fh.close()

    if vram_timeline is not None:
        vram_timeline.append(vram_report(f"probe_{name}_done"))

    with open(probe_dir / "training_curve.json", "w") as f:
        json.dump(training_curve, f, indent=2)

    persist_btrm(diff_model, "rtheta", str(probe_dir))

    final = training_curve[-1]
    min_loss = min(e["loss"] for e in training_curve)
    best_pink = max(e["accuracy_pinkify"] for e in training_curve)
    best_tnt = max(e["accuracy_thisnotthat"] for e in training_curve)
    mean_step_time = sum(e["time_s"] for e in training_curve) / len(training_curve)
    mean_grad_norm = sum(e["grad_norm"] for e in training_curve) / len(training_curve)

    n_tc = len(training_curve)
    mean_acc_pink = sum(e["accuracy_pinkify"] for e in training_curve) / n_tc
    mean_acc_tnt = sum(e["accuracy_thisnotthat"] for e in training_curve) / n_tc

    ema_alpha = 0.1
    ema_loss = training_curve[0]["loss"]
    for e in training_curve[1:]:
        ema_loss = ema_alpha * e["loss"] + (1.0 - ema_alpha) * ema_loss

    tail = training_curve[-20:] if n_tc >= 20 else training_curve
    tail_losses = [e["loss"] for e in tail]
    tail_mean = sum(tail_losses) / len(tail_losses)
    loss_std_last_20 = (sum((x - tail_mean) ** 2 for x in tail_losses) / len(tail_losses)) ** 0.5

    result = ProbeResult(
        name=name,
        lr=lr,
        n_steps=n_steps,
        rank=rank,
        init_b_std=init_b_std,
        layer_subset=layer_subset_name,
        final_loss=final["loss"],
        min_loss=min_loss,
        final_acc_pinkify=final["accuracy_pinkify"],
        final_acc_thisnotthat=final["accuracy_thisnotthat"],
        best_acc_pinkify=best_pink,
        best_acc_thisnotthat=best_tnt,
        mean_acc_pinkify=mean_acc_pink,
        mean_acc_thisnotthat=mean_acc_tnt,
        ema_loss_final=ema_loss,
        loss_std_last_20=loss_std_last_20,
        mean_step_time_s=mean_step_time,
        total_time_s=total_time,
        n_adapter_params=n_adapter_params,
        mean_grad_norm=mean_grad_norm,
    )

    with open(probe_dir / "final_metrics.json", "w") as f:
        json.dump(asdict(result), f, indent=2)

    torch.cuda.empty_cache()

    return result


def _collect_trajectory_metadata(
    records: list[dict],
    trajectory_indices: list[int],
) -> dict:
    """Collect metadata about the trajectories used in this sweep.

    Returns a dict suitable for inclusion in sweep_summary.json with:
      - n_trajectories: count
      - trajectory_indices: the indices used
      - resolutions: set of (W, H) pixel sizes
      - step_counts: set of n_steps values
      - types: set of t2i/i2i
      - precisions: set of sdpa/sage
      - per_trajectory: list of per-traj metadata dicts
    """
    resolutions = set()
    step_counts = set()
    types = set()
    precisions = set()
    per_traj = []

    for traj_idx in trajectory_indices:
        if traj_idx >= len(records):
            continue
        rec = records[traj_idx]
        traj_type = rec.get("type", "t2i")
        types.add(traj_type)
        precisions.add(rec.get("precision", "unknown"))
        step_counts.add(rec["n_steps"])

        if traj_type == "i2i":
            w = rec.get("output_width", 1280)
            h = rec.get("output_height", 832)
        else:
            w, h = 1280, 832
        resolutions.add(f"{w}x{h}")

        per_traj.append({
            "traj_idx": traj_idx,
            "type": traj_type,
            "resolution": f"{w}x{h}",
            "n_steps": rec["n_steps"],
            "precision": rec.get("precision", "unknown"),
        })

    return {
        "n_trajectories": len(trajectory_indices),
        "trajectory_indices": trajectory_indices,
        "resolutions": sorted(resolutions),
        "step_counts": sorted(step_counts),
        "types": sorted(types),
        "precisions": sorted(precisions),
        "per_trajectory": per_traj,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: rtheta_sweep_output_v2)")
    parser.add_argument("--n-trajectories", type=int, default=N_TRAJECTORIES,
                        help="Number of trajectories from index 0 (default: 10)")
    parser.add_argument("--trajectory-range", type=str, default=None,
                        help="Explicit trajectory range, e.g. '0-9' or '0-9,40-49'. "
                             "Overrides --n-trajectories when set.")
    parser.add_argument("--include-multires", action="store_true",
                        help="Additionally include trajectories 40-49 (mixed resolution "
                             "i2i) alongside base trajectories.")
    parser.add_argument("--no-sigma-weighting", action="store_true",
                        help="Disable log-SNR-based sigma sampling weighting "
                             "(use uniform pair sampling instead).")
    parser.add_argument("--optimizer-type", type=str, default="adam",
                        choices=["adam", "muon"],
                        help="Optimizer type: 'adam' (default) or 'muon'. "
                             "When 'muon', uses Muon for LoRA params and AdamW "
                             "for ScoreUnembedder params.")
    parser.add_argument("--muon-lr", type=float, default=0.02,
                        help="Learning rate for Muon (default: 0.02).")
    parser.add_argument("--muon-momentum", type=float, default=0.95,
                        help="Momentum for Muon (default: 0.95).")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Gradient accumulation steps per optimizer step (default: 1).")
    parser.add_argument("--include-rollouts", action="store_true",
                        help="Include policy rollout trajectories in training data. "
                             "By default, policy rollouts (run_name in "
                             "('policy_rollout_v1', '2xh100_20260216')) are excluded "
                             "from BTRM reward model training because their latents "
                             "were generated with LoRA adapters active, conflating "
                             "model quality with adapter effects. This flag overrides "
                             "that exclusion. Only applies when loading from V2 format.")
    parser.add_argument("--logsnr-threshold", type=float, default=10.0,
                        help="logSNR value where geometric decay begins (default: 10.0). "
                             "Sigma weighting is flat at 1.0 for logSNR >= threshold.")
    parser.add_argument("--logsnr-interval", type=float, default=5.0,
                        help="logSNR nats per decay step (default: 5.0). "
                             "E.g. 3.0 for faster decay per logSNR unit.")
    parser.add_argument("--logsnr-decay-rate", type=float, default=0.5,
                        help="Multiplicative factor per interval (default: 0.5). "
                             "E.g. 0.75 for gentler decay.")
    args = parser.parse_args()

    global OUTPUT_DIR
    if args.output_dir is not None:
        OUTPUT_DIR = Path(args.output_dir)

    if args.trajectory_range is not None:
        trajectory_indices = parse_trajectory_range(args.trajectory_range)
    else:
        trajectory_indices = list(range(args.n_trajectories))
    if args.include_multires:
        multires_indices = list(range(40, 50))
        trajectory_indices = sorted(set(trajectory_indices) | set(multires_indices))

    use_sigma_weighting = not args.no_sigma_weighting

    dataset_filter_overrides = None
    if args.include_rollouts:
        dataset_filter_overrides = {"exclude_policy_rollouts": False}

    t_total = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    import io
    class TeeStream:
        def __init__(self, original, logfile):
            self.original = original
            self.logfile = logfile
        def write(self, data):
            self.original.write(data)
            self.original.flush()
            self.logfile.write(data)
            self.logfile.flush()
        def flush(self):
            self.original.flush()
            self.logfile.flush()
    _log_fh = open(OUTPUT_DIR / "sweep_log.txt", "w")
    sys.stdout = TeeStream(sys.__stdout__, _log_fh)
    sys.stderr = TeeStream(sys.__stderr__, _log_fh)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("=" * 70)
    print("  r_theta LR BINARY SEARCH: Full Forward Differentiable Training")
    print("=" * 70)
    print(f"  Trajectory indices: {trajectory_indices}")
    print(f"  Sigma weighting: {use_sigma_weighting}")
    print(f"  logSNR schedule: threshold={args.logsnr_threshold}, "
          f"interval={args.logsnr_interval}, decay_rate={args.logsnr_decay_rate}")

    print("\n=== Loading training data ===")
    labels, per_image_scores, manifest = load_training_data()
    records = manifest["records"]

    traj_meta = _collect_trajectory_metadata(records, trajectory_indices)
    print(f"  Trajectory metadata: {len(trajectory_indices)} trajectories, "
          f"resolutions={sorted(traj_meta['resolutions'])}, "
          f"step_counts={sorted(traj_meta['step_counts'])}")

    training_pairs = build_training_pairs(labels, per_image_scores)
    print(f"  {len(training_pairs)} training pairs from {len(per_image_scores)} images")

    sigma_lookup_map = build_sigma_lookup(per_image_scores, records)
    def sigma_lookup_fn(image_idx: int) -> float:
        return sigma_lookup_map[image_idx]

    if use_sigma_weighting:
        from src_ii.btrm_training import log_snr_weight
        weight_dist = {}
        for idx, sigma in sigma_lookup_map.items():
            w = log_snr_weight(
                sigma,
                threshold=args.logsnr_threshold,
                interval=args.logsnr_interval,
                decay_rate=args.logsnr_decay_rate,
            )
            w_str = f"{w:.4f}"
            weight_dist[w_str] = weight_dist.get(w_str, 0) + 1
        print(f"  logSNR schedule: threshold={args.logsnr_threshold}, "
              f"interval={args.logsnr_interval}, decay_rate={args.logsnr_decay_rate}")
        print(f"  Sigma weight distribution: {weight_dist}")

    print("\n=== Encoding prompts ===")
    unique_prompts = encode_all_prompts(records, device, dtype,
                                        trajectory_indices=trajectory_indices)

    vram_timeline = []
    vram_timeline.append(vram_report("00_before_model_load"))

    print("\n=== Loading diffusion model ===")
    from src_ii.zimage_model import load_zimage_rlaif

    diff_model = load_zimage_rlaif(
        FP8_PATH, device=device, dtype=dtype,
        compile_model=False, fuse=True,
    )
    vram_timeline.append(vram_report("01_model_loaded_uncompiled"))

    print("\n=== Allocating adapter (before compile) ===")
    from futudiffu.lora import allocate_adapter, init_adapter_weights
    allocate_adapter(
        diff_model,
        name="rtheta",
        rank=8,
        alpha=16.0,
        layer_indices=None,  # all layers
    )
    vram_timeline.append(vram_report("02_adapter_allocated"))

    print("\n=== Compiling model (torch.compile, mode='default') ===")
    t_compile = time.perf_counter()
    diff_compiled = torch.compile(diff_model, mode="default")
    compile_time = time.perf_counter() - t_compile
    print(f"  torch.compile wrapper created in {compile_time:.1f}s")
    vram_timeline.append(vram_report("03_compiled"))

    print("\n=== Building latent loader ===")
    load_latent_fn = build_latent_loader(
        per_image_scores, records, unique_prompts,
        diff_model, device, dtype,
    )

    print("  Pre-warming latent cache...")
    t_cache = time.perf_counter()
    unique_indices = set()
    for pair in training_pairs:
        unique_indices.add(pair["idx_a"])
        unique_indices.add(pair["idx_b"])
    for idx in sorted(unique_indices):
        load_latent_fn(idx)
    print(f"  Cached {len(unique_indices)} latents in {time.perf_counter() - t_cache:.1f}s")
    vram_timeline.append(vram_report("04_latents_cached"))

    all_results: list[ProbeResult] = []

    print("\n" + "=" * 70)
    print("  PHASE 1: LR Binary Search (5 probes x 100 steps)")
    print("=" * 70)

    for lr in PROBE_LRS:
        name = f"lr_{lr:.0e}"
        print(f"\n  --- Probe: {name} ---")
        result = run_probe(
            diff_model, training_pairs, load_latent_fn, device,
            name=name, lr=lr, n_steps=PROBE_STEPS,
            rank=8, init_b_std=0.01,
            output_dir=OUTPUT_DIR,
            vram_timeline=vram_timeline,
            use_sigma_weighting=use_sigma_weighting,
            sigma_lookup_fn=sigma_lookup_fn,
            optimizer_type=args.optimizer_type,
            muon_lr=args.muon_lr,
            muon_momentum=args.muon_momentum,
            grad_accum_steps=args.grad_accum_steps,
            logsnr_threshold=args.logsnr_threshold,
            logsnr_interval=args.logsnr_interval,
            logsnr_decay_rate=args.logsnr_decay_rate,
        )
        all_results.append(result)
        print(f"    Result: loss={result.final_loss:.4f}, min_loss={result.min_loss:.4f}, "
              f"ema_loss={result.ema_loss_final:.4f}, "
              f"loss_std20={result.loss_std_last_20:.4f}")
        print(f"            mean_pink={result.mean_acc_pinkify:.3f}, "
              f"mean_tnt={result.mean_acc_thisnotthat:.3f}, "
              f"gnorm={result.mean_grad_norm:.4f}, "
              f"time={result.total_time_s:.0f}s ({result.mean_step_time_s:.1f}s/step)")

    phase1_results = [r for r in all_results if r.name.startswith("lr_")]
    winner = min(phase1_results, key=lambda r: r.ema_loss_final)
    winning_lr = winner.lr

    print(f"\n  Phase 1 Winner: {winner.name}  (criterion: lowest ema_loss_final)")
    print(f"    LR={winning_lr}, ema_loss_final={winner.ema_loss_final:.4f}, "
          f"final_loss={winner.final_loss:.4f}, min_loss={winner.min_loss:.4f}")
    print(f"    mean_pink={winner.mean_acc_pinkify:.3f}, "
          f"mean_tnt={winner.mean_acc_thisnotthat:.3f}, "
          f"loss_std_last_20={winner.loss_std_last_20:.4f}")


    print("\n" + "=" * 70)
    print("  SWEEP SUMMARY")
    print("=" * 70)

    vram_timeline.append(vram_report("99_sweep_complete"))

    summary = {
        "total_time_s": time.perf_counter() - t_total,
        "compile_time_s": compile_time,
        "compiled": True,
        "winning_lr": winning_lr,
        "use_sigma_weighting": use_sigma_weighting,
        "logsnr_schedule": {
            "threshold": args.logsnr_threshold,
            "interval": args.logsnr_interval,
            "decay_rate": args.logsnr_decay_rate,
        },
        "trajectory_metadata": traj_meta,
        "vram_timeline": vram_timeline,
        "all_results": [asdict(r) for r in all_results],
    }
    with open(OUTPUT_DIR / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'Name':<20s} {'LR':>10s} {'Steps':>6s} {'Loss':>8s} {'MinLoss':>8s} "
          f"{'EMALoss':>8s} {'MnPink':>7s} {'MnTNT':>7s} {'LStd20':>7s} "
          f"{'GNorm':>7s} {'Time':>7s} {'Params':>8s}")
    print("-" * 115)

    for r in sorted(all_results, key=lambda x: x.ema_loss_final):
        print(f"{r.name:<20s} {r.lr:10.0e} {r.n_steps:6d} "
              f"{r.final_loss:8.4f} {r.min_loss:8.4f} "
              f"{r.ema_loss_final:8.4f} "
              f"{r.mean_acc_pinkify:7.3f} {r.mean_acc_thisnotthat:7.3f} "
              f"{r.loss_std_last_20:7.4f} "
              f"{r.mean_grad_norm:7.4f} {r.total_time_s:7.0f}s "
              f"{r.n_adapter_params:8d}")

    elapsed = time.perf_counter() - t_total
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Output: {OUTPUT_DIR}")


def run_attention_diff(
    diff_model: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    unique_prompts: dict,
    records: list[dict],
    model_dir: Path,
):
    """Run attention diff capture: scale=0 vs scale=1 for the trained adapter."""
    from src_ii.btrm_lifecycle import load_btrm
    from src_ii.multi_lora import install_multi_lora, MultiLoRALinear
    from src_ii.stats import sigma_for_step
    from futudiffu.sampling import make_rope_cache
    from futudiffu.attention import set_attention_backend
    import futudiffu.attention as attn_mod
    import futudiffu.diffusion_model as dm_mod

    attn_dir = model_dir / "attention_diffs"
    attn_dir.mkdir(parents=True, exist_ok=True)

    LATENT_SELECTIONS = [
        {"name": "high_pinkify", "traj_idx": 5, "step_key": "step_09"},
        {"name": "low_pinkify", "traj_idx": 3, "step_key": "final"},
        {"name": "high_thisnotthat", "traj_idx": 2, "step_key": "final"},
        {"name": "low_thisnotthat", "traj_idx": 7, "step_key": "final"},
    ]

    config_path = model_dir / "btrm_compound_config.json"
    if not config_path.exists():
        print(f"  No compound config at {config_path}, skipping attention diff")
        return

    install_multi_lora(diff_model, [{"name": "rtheta", "rank": 8, "alpha": 16.0}])
    load_btrm(diff_model, "rtheta", str(model_dir))

    def _set_global_adapter_scale(scale: float):
        scale_t = torch.tensor([scale], device=device)
        for m in diff_model.modules():
            if isinstance(m, MultiLoRALinear):
                m._adapter_scales = scale_t

    set_attention_backend("sdpa")
    capture = AttentionCapture()
    capture.install()
    dm_mod.sdpa_attention = attn_mod.sdpa_attention

    per_latent_stats = {}

    for sel in LATENT_SELECTIONS:
        name = sel["name"]
        traj_idx = sel["traj_idx"]
        step_key = sel["step_key"]
        prompt = records[traj_idx]["prompt"]

        traj_dir = REPO_ROOT / "btrm_dataset" / "latents" / f"traj_{traj_idx:06d}"
        latent = torch.load(str(traj_dir / f"{step_key}.pt"), weights_only=True)
        latent = latent.to(device=device, dtype=dtype)

        conditioning = unique_prompts[prompt].to(device=device, dtype=dtype)
        num_tokens = conditioning.shape[1]

        n_steps = records[traj_idx]["n_steps"]
        sigma_val = sigma_for_step(step_key, n_steps, device=device, dtype=dtype)
        timestep = sigma_val.unsqueeze(0)

        B, C, H, W = latent.shape
        rope_cache = make_rope_cache(diff_model, H, W, num_tokens, device)

        _set_global_adapter_scale(0.0)
        stats_a = capture.capture_forward(
            diff_model, latent, timestep, conditioning, num_tokens, rope_cache,
        )

        _set_global_adapter_scale(1.0)
        stats_b = capture.capture_forward(
            diff_model, latent, timestep, conditioning, num_tokens, rope_cache,
        )

        from collections import Counter
        dom_a = Counter(v["seq_len"] for v in stats_a.values()).most_common(1)[0][0]
        dom_b = Counter(v["seq_len"] for v in stats_b.values()).most_common(1)[0][0]
        main_a = [i for i in sorted(stats_a.keys()) if stats_a[i]["seq_len"] == dom_a]
        main_b = [i for i in sorted(stats_b.keys()) if stats_b[i]["seq_len"] == dom_b]
        n_main = min(len(main_a), len(main_b))

        total_abs_diff = 0.0
        max_abs_diff = 0.0
        for di in range(n_main):
            sa = stats_a[main_a[di]]
            sb = stats_b[main_b[di]]
            min_seq = min(sa["seq_len"], sb["seq_len"])
            recv_diff = (
                sb["attn_received"][:, :min_seq] - sa["attn_received"][:, :min_seq]
            ).float()
            total_abs_diff += recv_diff.abs().mean().item()
            max_abs_diff = max(max_abs_diff, recv_diff.abs().max().item())

        mean_abs = total_abs_diff / max(n_main, 1)
        per_latent_stats[name] = {
            "mean_abs_received_diff": mean_abs,
            "max_abs_received_diff": max_abs_diff,
            "n_main_layers": n_main,
        }
        print(f"    {name}: mean|delta|={mean_abs:.6f}, max|delta|={max_abs_diff:.4f}")

    attn_manifest = {
        "per_latent_stats": per_latent_stats,
        "overall_mean": sum(
            s["mean_abs_received_diff"] for s in per_latent_stats.values()
        ) / max(len(per_latent_stats), 1),
    }
    with open(attn_dir / "attention_diff_manifest.json", "w") as f:
        json.dump(attn_manifest, f, indent=2)

    _set_global_adapter_scale(1.0)
    capture.remove()


if __name__ == "__main__":
    main()

r"""Hyperparameter sweep for stronger r_theta BTRM adapter.

Three-phase sweep strategy:
  Phase 1: Fix rank=8, init_b_std=0.01, all layers. Sweep LR x epochs. (20 configs)
  Phase 2: Best LR/epoch from Phase 1. Sweep rank x init_b_std. (12 configs)
  Phase 3: Best overall config. Sweep layer subsets. (3 configs)

Hidden states are extracted ONCE per unique (rank, layer_subset) combination
and reused across all LR/epoch combos for that architecture.

Execution:
  PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe ^
      F:\dox\repos\ai\futudiffu\scripts_ii\sweep_rtheta_hparams.py

Output:
  pinkify_thisnotthat_output/sweep/{config_name}/
    training_curve.json
    final_metrics.json
    rtheta_adapter.safetensors
    btrm_head.safetensors
    btrm_compound_config.json
"""

from __future__ import annotations

import json
import sys
import time
import gc
from pathlib import Path
from dataclasses import dataclass, asdict

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.nn as nn

# --- Configuration ---
FP8_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
TE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"
TOKENIZER_PATH = str(REPO_ROOT / "src" / "futudiffu" / "tokenizer")
OUTPUT_DIR = REPO_ROOT / "pinkify_thisnotthat_output"
SWEEP_DIR = OUTPUT_DIR / "sweep"
LABELS_PATH = OUTPUT_DIR / "preference_labels.json"
SCORES_PATH = OUTPUT_DIR / "per_image_scores.json"
MANIFEST_PATH = REPO_ROOT / "btrm_dataset" / "manifest.json"

HEAD_NAMES = ("pinkify", "thisnotthat")
PREF_KEYS = ("pinkify_pref", "thisnotthat_pref")
N_TRAJECTORIES = 10
LOGSQUARE_WEIGHT = 0.05
BATCH_SIZE = 16

# Sweep grids
PHASE1_LRS = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
PHASE1_EPOCHS = [40, 100, 200, 400]

PHASE2_RANKS = [8, 16, 32, 64]
PHASE2_INIT_B_STDS = [0.01, 0.05, 0.1]

PHASE3_LAYER_SUBSETS = {
    "all_30": None,  # all layers (default)
    "last_15": set(range(15, 30)),
    "last_8": set(range(22, 30)),
}


@dataclass
class SweepConfig:
    name: str
    lr: float
    n_epochs: int
    rank: int
    init_b_std: float
    layer_subset_name: str
    layer_indices: list[int] | None  # None = all layers
    phase: int


@dataclass
class SweepResult:
    config_name: str
    final_loss: float
    best_loss: float
    final_acc_pinkify: float
    final_acc_thisnotthat: float
    best_acc_pinkify: float
    best_acc_thisnotthat: float
    n_adapter_params: int
    training_time_s: float


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


def encode_all_prompts(records, device, dtype):
    """Phase 1: Encode unique prompts, then free TE."""
    from futudiffu.text_encoder import create_tokenizer, load_text_encoder, encode_prompt

    tokenizer = create_tokenizer(TOKENIZER_PATH)
    te_model = load_text_encoder(TE_PATH, device=device, dtype=dtype)

    unique_prompts = {}
    for traj_idx in range(N_TRAJECTORIES):
        prompt = records[traj_idx]["prompt"]
        if prompt not in unique_prompts:
            cond = encode_prompt(te_model, tokenizer, prompt, device=device)
            unique_prompts[prompt] = cond.cpu()
            print(f"  Encoded: '{prompt[:60]}...' -> {cond.shape}")

    del te_model, tokenizer
    torch.cuda.empty_cache()
    print(f"  TE freed. {len(unique_prompts)} unique prompts encoded.")
    return unique_prompts


def extract_hidden_states(
    diff_model: nn.Module,
    per_image_scores: list[dict],
    records: list[dict],
    unique_prompts: dict,
    rank: int,
    init_b_std: float,
    layer_indices: set[int] | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[torch.Tensor], "BTRMCompoundModel"]:
    """Extract hidden states for a given adapter architecture.

    Creates a fresh BTRMCompoundModel with the specified rank/layer_indices,
    runs all images through it, returns CPU hidden states and the compound model.
    """
    from src_ii.btrm_model import BTRMCompoundModel
    from src_ii.stats import sigma_for_step
    from futudiffu.sampling import make_rope_cache

    btrm = BTRMCompoundModel(
        diff_model,
        head_names=HEAD_NAMES,
        hidden_dim=3840,
        logit_cap=10.0,
        adapter_rank=rank,
        adapter_init_b_std=init_b_std,
        adapter_layer_indices=layer_indices,
        device=device,
    )

    hidden_states_cpu = []
    n_images = len(per_image_scores)
    multiplier = 1.0

    for img_entry in per_image_scores:
        traj_idx = img_entry["traj_idx"]
        step_key = img_entry["step_key"]
        prompt = records[traj_idx]["prompt"]
        conditioning = unique_prompts[prompt].to(device=device, dtype=dtype)

        traj_dir = REPO_ROOT / "btrm_dataset" / "latents" / f"traj_{traj_idx:06d}"
        pt_path = traj_dir / f"{step_key}.pt"
        latent = torch.load(str(pt_path), weights_only=True).to(device=device, dtype=dtype)

        n_steps = records[traj_idx]["n_steps"]
        sigma_val = sigma_for_step(step_key, n_steps, device=device, dtype=dtype)
        timestep = sigma_val * multiplier
        num_tokens = conditioning.shape[1]

        B, C, H, W = latent.shape
        rope_cache = make_rope_cache(diff_model, H, W, num_tokens, device)

        hidden = btrm.extract_hidden(
            latent, timestep.unsqueeze(0), conditioning, num_tokens, rope_cache,
        )
        hidden_states_cpu.append(hidden.cpu())

        if len(hidden_states_cpu) % 20 == 0:
            print(f"    Extracted {len(hidden_states_cpu)}/{n_images} hidden states")

    print(f"    All {len(hidden_states_cpu)} hidden states extracted")
    return hidden_states_cpu, btrm


def remove_all_adapters(model: nn.Module):
    """Remove all LoRA wrappers from the model, restoring base modules."""
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


def train_single_config(
    config: SweepConfig,
    hidden_states_cpu: list[torch.Tensor],
    training_pairs: list[dict],
    diff_model: nn.Module,
    device: torch.device,
) -> SweepResult:
    """Train a single config using pre-extracted hidden states.

    Creates a fresh BTRMCompoundModel for each config (but reuses hidden states).
    This is necessary because each train needs fresh head + adapter weights.
    """
    from src_ii.btrm_model import BTRMCompoundModel
    from src_ii.btrm_training import train_btrm

    layer_indices = set(config.layer_indices) if config.layer_indices else None

    # Create fresh compound model with fresh weights
    btrm = BTRMCompoundModel(
        diff_model,
        head_names=HEAD_NAMES,
        hidden_dim=3840,
        logit_cap=10.0,
        adapter_rank=config.rank,
        adapter_init_b_std=config.init_b_std,
        adapter_layer_indices=layer_indices,
        device=device,
    )

    t0 = time.perf_counter()
    training_curve = train_btrm(
        btrm_model=btrm,
        training_pairs=training_pairs,
        hidden_states_cpu=hidden_states_cpu,
        n_epochs=config.n_epochs,
        lr=config.lr,
        logsquare_weight=LOGSQUARE_WEIGHT,
        batch_size=BATCH_SIZE,
        head_names=HEAD_NAMES,
        pref_keys=PREF_KEYS,
        device=device,
    )
    training_time = time.perf_counter() - t0

    # Compute metrics
    final = training_curve[-1]
    best_loss = min(e["loss"] for e in training_curve)
    best_pinkify = max(e["accuracy_pinkify"] for e in training_curve)
    best_thisnotthat = max(e["accuracy_thisnotthat"] for e in training_curve)

    n_adapter_params = sum(p.numel() for p in btrm.adapter_params())

    result = SweepResult(
        config_name=config.name,
        final_loss=final["loss"],
        best_loss=best_loss,
        final_acc_pinkify=final["accuracy_pinkify"],
        final_acc_thisnotthat=final["accuracy_thisnotthat"],
        best_acc_pinkify=best_pinkify,
        best_acc_thisnotthat=best_thisnotthat,
        n_adapter_params=n_adapter_params,
        training_time_s=training_time,
    )

    # Save outputs
    config_dir = SWEEP_DIR / config.name
    config_dir.mkdir(parents=True, exist_ok=True)

    # Save training curve
    with open(config_dir / "training_curve.json", "w") as f:
        json.dump(training_curve, f, indent=2)

    # Save final metrics
    with open(config_dir / "final_metrics.json", "w") as f:
        json.dump(asdict(result), f, indent=2)

    # Persist adapter + head
    btrm.persist(str(config_dir))

    # Clean up compound model
    btrm.cleanup()
    remove_all_adapters(diff_model)

    return result


def main():
    t_total = time.perf_counter()
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("=" * 70)
    print("  HYPERPARAMETER SWEEP: Stronger r_theta BTRM Adapter")
    print("=" * 70)

    # --- Load training data ---
    print("\n=== Loading training data ===")
    labels, per_image_scores, manifest = load_training_data()
    records = manifest["records"]
    training_pairs = build_training_pairs(labels, per_image_scores)
    print(f"  {len(training_pairs)} training pairs from {len(per_image_scores)} images")

    # --- Phase 0: Encode prompts ---
    print("\n=== Phase 0: Encoding prompts ===")
    unique_prompts = encode_all_prompts(records, device, dtype)

    # --- Load diffusion model (once, kept in VRAM throughout) ---
    print("\n=== Loading diffusion model ===")
    from src_ii.model_loading import load_fp8_diffusion_model

    diff_model, _ = load_fp8_diffusion_model(
        FP8_PATH, device=device, dtype=dtype,
        compile_model=False, fuse=True,
    )

    all_results: list[SweepResult] = []

    # =====================================================================
    # PHASE 1: LR x Epochs (rank=8, init_b_std=0.01, all layers)
    # =====================================================================
    print("\n" + "=" * 70)
    print("  PHASE 1: LR x Epochs sweep (rank=8, init_b_std=0.01, all layers)")
    print("=" * 70)

    # Extract hidden states ONCE for Phase 1 architecture
    print("\n  Extracting hidden states for Phase 1 (rank=8, all layers)...")
    t_extract = time.perf_counter()
    hidden_states_p1, btrm_p1 = extract_hidden_states(
        diff_model, per_image_scores, records, unique_prompts,
        rank=8, init_b_std=0.01, layer_indices=None,
        device=device, dtype=dtype,
    )
    print(f"  Hidden extraction: {time.perf_counter() - t_extract:.1f}s")

    # Clean up extraction compound model (we only needed it for hidden states)
    btrm_p1.cleanup()
    remove_all_adapters(diff_model)

    phase1_configs = []
    for lr in PHASE1_LRS:
        for n_epochs in PHASE1_EPOCHS:
            name = f"p1_lr{lr:.0e}_ep{n_epochs}"
            phase1_configs.append(SweepConfig(
                name=name, lr=lr, n_epochs=n_epochs, rank=8,
                init_b_std=0.01, layer_subset_name="all_30",
                layer_indices=None, phase=1,
            ))

    print(f"\n  Running {len(phase1_configs)} Phase 1 configs...")
    for i, config in enumerate(phase1_configs):
        print(f"\n  --- Phase 1 [{i+1}/{len(phase1_configs)}]: {config.name} ---")
        result = train_single_config(
            config, hidden_states_p1, training_pairs, diff_model, device,
        )
        all_results.append(result)
        print(f"    Final: loss={result.final_loss:.4f}, "
              f"pinkify={result.final_acc_pinkify:.3f}, "
              f"thisnotthat={result.final_acc_thisnotthat:.3f}, "
              f"best_tnt={result.best_acc_thisnotthat:.3f}, "
              f"time={result.training_time_s:.1f}s")

    # Find best Phase 1 config by best thisnotthat accuracy
    phase1_results = [r for r in all_results if r.config_name.startswith("p1_")]
    best_p1 = max(phase1_results, key=lambda r: r.best_acc_thisnotthat)
    # Extract lr and epochs from best config
    best_p1_config = [c for c in phase1_configs if c.name == best_p1.config_name][0]
    best_lr = best_p1_config.lr
    best_epochs = best_p1_config.n_epochs

    print(f"\n  Phase 1 winner: {best_p1.config_name}")
    print(f"    LR={best_lr}, epochs={best_epochs}")
    print(f"    best_acc_thisnotthat={best_p1.best_acc_thisnotthat:.3f}")
    print(f"    best_acc_pinkify={best_p1.best_acc_pinkify:.3f}")
    print(f"    final_loss={best_p1.final_loss:.4f}")

    del hidden_states_p1
    gc.collect()
    torch.cuda.empty_cache()

    # =====================================================================
    # PHASE 2: Rank x init_b_std (best LR/epochs from Phase 1, all layers)
    # =====================================================================
    print("\n" + "=" * 70)
    print(f"  PHASE 2: Rank x init_b_std sweep (lr={best_lr}, epochs={best_epochs})")
    print("=" * 70)

    # Group by rank: extract hidden states once per rank
    phase2_configs_by_rank: dict[int, list[SweepConfig]] = {}
    for rank in PHASE2_RANKS:
        for init_b_std in PHASE2_INIT_B_STDS:
            name = f"p2_r{rank}_b{init_b_std}"
            config = SweepConfig(
                name=name, lr=best_lr, n_epochs=best_epochs, rank=rank,
                init_b_std=init_b_std, layer_subset_name="all_30",
                layer_indices=None, phase=2,
            )
            if rank not in phase2_configs_by_rank:
                phase2_configs_by_rank[rank] = []
            phase2_configs_by_rank[rank].append(config)

    for rank, configs in phase2_configs_by_rank.items():
        print(f"\n  --- Phase 2: Extracting hidden states for rank={rank} ---")
        t_extract = time.perf_counter()
        # Use a representative init_b_std for extraction (the adapter affects
        # hidden states via its init, but the effect is small at init_b_std=0.01)
        # We use the first config's init_b_std
        hidden_states, btrm_tmp = extract_hidden_states(
            diff_model, per_image_scores, records, unique_prompts,
            rank=rank, init_b_std=configs[0].init_b_std, layer_indices=None,
            device=device, dtype=dtype,
        )
        print(f"    Extraction: {time.perf_counter() - t_extract:.1f}s")
        btrm_tmp.cleanup()
        remove_all_adapters(diff_model)

        for j, config in enumerate(configs):
            print(f"\n    Phase 2 [{j+1}/{len(configs)}]: {config.name}")
            result = train_single_config(
                config, hidden_states, training_pairs, diff_model, device,
            )
            all_results.append(result)
            print(f"      Final: loss={result.final_loss:.4f}, "
                  f"pinkify={result.final_acc_pinkify:.3f}, "
                  f"thisnotthat={result.final_acc_thisnotthat:.3f}, "
                  f"best_tnt={result.best_acc_thisnotthat:.3f}, "
                  f"time={result.training_time_s:.1f}s")

        del hidden_states
        gc.collect()
        torch.cuda.empty_cache()

    # Find best Phase 2 config
    phase2_results = [r for r in all_results if r.config_name.startswith("p2_")]
    best_p2 = max(phase2_results, key=lambda r: r.best_acc_thisnotthat)
    best_p2_config = [c for configs in phase2_configs_by_rank.values()
                      for c in configs if c.name == best_p2.config_name][0]
    best_rank = best_p2_config.rank
    best_init_b_std = best_p2_config.init_b_std

    print(f"\n  Phase 2 winner: {best_p2.config_name}")
    print(f"    rank={best_rank}, init_b_std={best_init_b_std}")
    print(f"    best_acc_thisnotthat={best_p2.best_acc_thisnotthat:.3f}")
    print(f"    best_acc_pinkify={best_p2.best_acc_pinkify:.3f}")
    print(f"    final_loss={best_p2.final_loss:.4f}")

    # =====================================================================
    # PHASE 3: Layer subsets (best LR/epochs/rank/init_b_std)
    # =====================================================================
    print("\n" + "=" * 70)
    print(f"  PHASE 3: Layer subset sweep (lr={best_lr}, epochs={best_epochs}, "
          f"rank={best_rank}, init_b_std={best_init_b_std})")
    print("=" * 70)

    for subset_name, layer_set in PHASE3_LAYER_SUBSETS.items():
        print(f"\n  --- Phase 3: {subset_name} ---")

        layer_list = sorted(layer_set) if layer_set else None
        config = SweepConfig(
            name=f"p3_{subset_name}",
            lr=best_lr, n_epochs=best_epochs, rank=best_rank,
            init_b_std=best_init_b_std, layer_subset_name=subset_name,
            layer_indices=layer_list, phase=3,
        )

        print(f"    Extracting hidden states for {subset_name}...")
        t_extract = time.perf_counter()
        hidden_states, btrm_tmp = extract_hidden_states(
            diff_model, per_image_scores, records, unique_prompts,
            rank=best_rank, init_b_std=best_init_b_std,
            layer_indices=layer_set,
            device=device, dtype=dtype,
        )
        print(f"    Extraction: {time.perf_counter() - t_extract:.1f}s")
        btrm_tmp.cleanup()
        remove_all_adapters(diff_model)

        result = train_single_config(
            config, hidden_states, training_pairs, diff_model, device,
        )
        all_results.append(result)
        print(f"    Final: loss={result.final_loss:.4f}, "
              f"pinkify={result.final_acc_pinkify:.3f}, "
              f"thisnotthat={result.final_acc_thisnotthat:.3f}, "
              f"best_tnt={result.best_acc_thisnotthat:.3f}, "
              f"time={result.training_time_s:.1f}s")

        del hidden_states
        gc.collect()
        torch.cuda.empty_cache()

    # Find best Phase 3 config
    phase3_results = [r for r in all_results if r.config_name.startswith("p3_")]
    best_p3 = max(phase3_results, key=lambda r: r.best_acc_thisnotthat)

    print(f"\n  Phase 3 winner: {best_p3.config_name}")
    print(f"    best_acc_thisnotthat={best_p3.best_acc_thisnotthat:.3f}")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 70)
    print("  SWEEP SUMMARY")
    print("=" * 70)

    # Save all results
    summary = {
        "sweep_time_s": time.perf_counter() - t_total,
        "phase1_winner": best_p1.config_name,
        "phase2_winner": best_p2.config_name,
        "phase3_winner": best_p3.config_name,
        "best_config": {
            "lr": best_lr,
            "n_epochs": best_epochs,
            "rank": best_rank,
            "init_b_std": best_init_b_std,
            "layer_subset": best_p3.config_name.replace("p3_", ""),
        },
        "all_results": [asdict(r) for r in all_results],
    }

    with open(SWEEP_DIR / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print table
    print(f"\n{'Config':<30s} {'Loss':>8s} {'BestLoss':>8s} "
          f"{'Pink':>6s} {'BPink':>6s} {'TNT':>6s} {'BTNT':>6s} "
          f"{'Params':>10s} {'Time':>7s}")
    print("-" * 100)

    for r in sorted(all_results, key=lambda x: -x.best_acc_thisnotthat):
        print(f"{r.config_name:<30s} {r.final_loss:8.4f} {r.best_loss:8.4f} "
              f"{r.final_acc_pinkify:6.3f} {r.best_acc_pinkify:6.3f} "
              f"{r.final_acc_thisnotthat:6.3f} {r.best_acc_thisnotthat:6.3f} "
              f"{r.n_adapter_params:10d} {r.training_time_s:7.1f}s")

    # Top 3 by thisnotthat accuracy
    top3 = sorted(all_results, key=lambda x: -x.best_acc_thisnotthat)[:3]
    print(f"\n  TOP 3 configs (by best thisnotthat accuracy):")
    for i, r in enumerate(top3):
        print(f"    {i+1}. {r.config_name}: best_tnt={r.best_acc_thisnotthat:.3f}, "
              f"best_pink={r.best_acc_pinkify:.3f}, loss={r.final_loss:.4f}")

    # Save top 3 list for attention diff phase
    top3_names = [r.config_name for r in top3]
    with open(SWEEP_DIR / "top3_configs.json", "w") as f:
        json.dump(top3_names, f, indent=2)

    # =====================================================================
    # ATTENTION DIFF for top 3 (if time permits)
    # =====================================================================
    print("\n" + "=" * 70)
    print("  ATTENTION DIFF for Top 3 Configs")
    print("=" * 70)

    # Free diffusion model first, then reload for attention diff
    del diff_model
    gc.collect()
    torch.cuda.empty_cache()

    run_attention_diffs_for_top3(top3, device, dtype, unique_prompts, records)

    elapsed_total = time.perf_counter() - t_total
    print(f"\n{'=' * 70}")
    print(f"  SWEEP COMPLETE in {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(f"  Output: {SWEEP_DIR}")
    print(f"{'=' * 70}")


def run_attention_diffs_for_top3(
    top3: list[SweepResult],
    device: torch.device,
    dtype: torch.dtype,
    unique_prompts: dict,
    records: list[dict],
):
    """Run attention adapter diff capture for the top 3 configs."""
    from src_ii.model_loading import load_fp8_diffusion_model
    from src_ii.btrm_model import BTRMCompoundModel
    from src_ii.attention_capture import AttentionCapture
    from src_ii.stats import sigma_for_step
    from futudiffu.sampling import make_rope_cache
    from futudiffu.attention import set_attention_backend
    import futudiffu.attention as attn_mod
    import futudiffu.diffusion_model as dm_mod

    # The 4 representative latents from the original attention study
    LATENT_SELECTIONS = [
        {"name": "high_pinkify", "traj_idx": 5, "step_key": "step_09"},
        {"name": "low_pinkify", "traj_idx": 3, "step_key": "final"},
        {"name": "high_thisnotthat", "traj_idx": 2, "step_key": "final"},
        {"name": "low_thisnotthat", "traj_idx": 7, "step_key": "final"},
    ]

    # Also load baseline adapter diffs for comparison
    baseline_manifest_path = OUTPUT_DIR / "adapter_attention_diffs" / "run_manifest.json"
    baseline_diffs = None
    if baseline_manifest_path.exists():
        with open(baseline_manifest_path) as f:
            baseline_diffs = json.load(f).get("results", {})
        print(f"  Loaded baseline attention diff manifest for comparison")

    for rank_idx, top_result in enumerate(top3):
        config_name = top_result.config_name
        config_dir = SWEEP_DIR / config_name
        attn_dir = config_dir / "attention_diffs"
        attn_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  --- Attention diff for #{rank_idx+1}: {config_name} ---")

        # Load this config's compound model config
        with open(config_dir / "btrm_compound_config.json") as f:
            compound_config = json.load(f)

        # Reload diffusion model fresh
        set_attention_backend("sdpa")
        diff_model, _ = load_fp8_diffusion_model(
            FP8_PATH, device=device, dtype=dtype,
            compile_model=False, fuse=True,
        )
        diff_model.eval()
        for p in diff_model.parameters():
            p.requires_grad_(False)
        dm_mod.sdpa_attention = attn_mod.sdpa_attention

        # Load compound BTRM model with this config's weights
        compound = BTRMCompoundModel.load(str(config_dir), diff_model, device=device)

        # Install attention capture
        capture = AttentionCapture()
        capture.install()
        dm_mod.sdpa_attention = attn_mod.sdpa_attention

        per_latent_stats = {}

        for sel in LATENT_SELECTIONS:
            name = sel["name"]
            traj_idx = sel["traj_idx"]
            step_key = sel["step_key"]
            prompt = records[traj_idx]["prompt"]

            # Load latent
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

            # Forward A: scale=0 (unadapted)
            compound.set_adapter_scale(0.0)
            stats_a = capture.capture_forward(
                diff_model, latent, timestep, conditioning, num_tokens, rope_cache,
            )

            # Forward B: scale=1 (adapter active)
            compound.set_adapter_scale(1.0)
            stats_b = capture.capture_forward(
                diff_model, latent, timestep, conditioning, num_tokens, rope_cache,
            )

            # Compute mean absolute diff across main layers
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

        # Compare to baseline
        comparison = {}
        if baseline_diffs:
            for name in per_latent_stats:
                if name in baseline_diffs:
                    baseline_mean = baseline_diffs[name].get("mean_abs_received_diff", 0)
                    sweep_mean = per_latent_stats[name]["mean_abs_received_diff"]
                    if baseline_mean > 0:
                        ratio = sweep_mean / baseline_mean
                        comparison[name] = {
                            "baseline_mean": baseline_mean,
                            "sweep_mean": sweep_mean,
                            "ratio": ratio,
                        }
                        print(f"    {name}: {ratio:.1f}x baseline")

        # Save attention diff results
        attn_manifest = {
            "config_name": config_name,
            "per_latent_stats": per_latent_stats,
            "baseline_comparison": comparison,
            "overall_mean": sum(
                s["mean_abs_received_diff"] for s in per_latent_stats.values()
            ) / len(per_latent_stats),
        }
        with open(attn_dir / "attention_diff_manifest.json", "w") as f:
            json.dump(attn_manifest, f, indent=2)

        # Clean up
        compound.set_adapter_scale(1.0)
        capture.remove()
        compound.cleanup()
        remove_all_adapters(diff_model)
        del compound, diff_model, capture
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

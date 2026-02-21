r"""Generate BTRM training dataset with mixed-resolution bin packing (V2 format).

Thin CLI wrapper around src_ii.dataset_generator.DatasetGenerator. All
generation logic lives in the library module; this script owns:
  - CLI argument parsing
  - Prompt loading (file or inline)
  - InferenceClient lifecycle
  - Dry-run mode (plan + pack, no server)

Usage:
  # Generate from prompt file with default settings:
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\generate_btrm_dataset.py \
      --prompts-file F:\dox\repos\ai\futudiffu\prompts.txt

  # Mixed-resolution, multiple backends:
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\generate_btrm_dataset.py \
      --prompts-file F:\dox\repos\ai\futudiffu\prompts.txt \
      --resolution-tiers full medium small \
      --attention-backends sdpa sage \
      --output-dir F:\dox\repos\ai\futudiffu\btrm_dataset_v2_mixed

  # Dry run (show plan + packing stats, no server needed):
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\generate_btrm_dataset.py \
      --prompts-file prompts.txt --dry-run

  # Use built-in 24 prompt templates:
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\generate_btrm_dataset.py \
      --use-builtin-prompts --resolution-tiers full medium
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src_ii.bin_packer import (
    REFERENCE_SEQ_LEN,
    RESOLUTION_TIERS,
    BinPackScheduler,
    build_generation_plan,
)


def _load_prompts_from_file(path: str) -> list[str]:
    """Load prompts from a text file (one per line, blank lines ignored)."""
    p = Path(path)
    if not p.exists():
        print(f"Error: prompts file not found: {path}", file=sys.stderr)
        sys.exit(1)
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    return [line.strip() for line in lines if line.strip()]


def _load_builtin_prompts() -> list[str]:
    """Load the 24 built-in prompt templates from futudiffu.btrm_dataset."""
    from futudiffu.btrm_dataset import PROMPT_TEMPLATES
    return list(PROMPT_TEMPLATES)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate BTRM training dataset with mixed-resolution bin packing (V2 format)")

    # --- Prompt sources (mutually exclusive) ---
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompts-file", type=str,
        help="Path to a text file with one prompt per line.")
    prompt_group.add_argument(
        "--prompts", type=str, nargs="+",
        help="Inline prompt strings.")
    prompt_group.add_argument(
        "--use-builtin-prompts", action="store_true",
        help="Use the 24 built-in prompt templates from futudiffu.btrm_dataset.")

    # --- Generation parameters ---
    parser.add_argument(
        "--resolution-tiers", type=str, nargs="+",
        default=["full"],
        choices=["full", "medium", "small"],
        help="Which resolution tiers to generate. Default: full.")
    parser.add_argument(
        "--attention-backends", type=str, nargs="+",
        default=["sdpa", "sage"],
        choices=["sdpa", "sage"],
        help="Attention backends to generate with. Default: sdpa sage.")
    parser.add_argument("--n-steps", type=int, default=30,
                        help="Number of Euler diffusion steps. Default: 30.")
    parser.add_argument("--cfg", type=float, default=4.0,
                        help="CFG guidance scale. Default: 4.0.")
    parser.add_argument(
        "--sparse-steps", type=int, nargs="+",
        default=[0, 4, 9, 14, 19, 24, 29],
        help="Step indices to save latents for. Default: 0 4 9 14 19 24 29.")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Explicit PRNG seeds. Default: auto-generate deterministically.")
    parser.add_argument(
        "--sampling-shift", type=float, default=None,
        help="Sigma schedule shift. Default: auto-compute per resolution "
             "(SD3 Eq.23). Explicit value overrides auto for all items.")
    parser.add_argument("--multiplier", type=float, default=1.0)

    # --- Output ---
    parser.add_argument(
        "--output-dir", type=str,
        default=str(REPO_ROOT / "btrm_dataset_v2"),
        help="Output directory for V2 dataset.")
    parser.add_argument("--run-name", type=str, default="generation",
                        help="Human-readable name for this run.")
    parser.add_argument("--render-count", type=int, default=6,
                        help="Number of trajectories to VAE-decode as PNGs.")
    parser.add_argument("--flush-interval", type=int, default=50,
                        help="Seal blob every N trajectories.")

    # --- Server ---
    parser.add_argument(
        "--server", type=str, default="tcp://localhost:5555",
        help="Inference server ZeroMQ endpoint.")
    parser.add_argument("--timeout-ms", type=int, default=0,
                        help="ZeroMQ receive timeout in ms. 0 = infinite.")

    # --- Identity ---
    parser.add_argument("--base-model-hash", type=str, default="z_image_v1",
                        help="Base model identifier for provenance tracking.")
    parser.add_argument("--source-device", type=str, default="unknown",
                        help="Device identifier for provenance (e.g. rtx4090_0).")

    # --- Modes ---
    parser.add_argument("--dry-run", action="store_true",
                        help="Show generation plan and packing stats, then exit.")

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Load prompts
    # ---------------------------------------------------------------
    if args.prompts_file:
        prompts = _load_prompts_from_file(args.prompts_file)
    elif args.use_builtin_prompts:
        prompts = _load_builtin_prompts()
    else:
        prompts = args.prompts

    if not prompts:
        print("Error: no prompts provided.", file=sys.stderr)
        return 1

    print(f"Loaded {len(prompts)} prompts")

    # ---------------------------------------------------------------
    # Dry run: build plan, pack, report, exit
    # ---------------------------------------------------------------
    if args.dry_run:
        import random as _rng

        seeds = args.seeds
        if seeds is None:
            rng = _rng.Random(42)
            max_rollouts = sum(
                RESOLUTION_TIERS[t]["rollouts_per_prompt"]
                for t in args.resolution_tiers
            )
            n_seeds = len(prompts) * max_rollouts * len(args.attention_backends)
            seeds = [rng.randint(0, 2**32 - 1) for _ in range(n_seeds)]

        plan = build_generation_plan(
            prompts=prompts,
            seeds=seeds,
            resolution_tiers=args.resolution_tiers,
            attention_backends=args.attention_backends,
            n_steps=args.n_steps,
            cfg=args.cfg,
            sparse_steps=args.sparse_steps,
        )

        scheduler = BinPackScheduler(max_seq_len=REFERENCE_SEQ_LEN)
        bins = scheduler.pack_generation_plan(plan)
        efficiency = scheduler.estimate_efficiency(bins)

        print(f"\n--- Generation Plan ---")
        print(f"  Total trajectories: {len(plan)}")
        print(f"  Resolution tiers: {args.resolution_tiers}")
        print(f"  Attention backends: {args.attention_backends}")
        print(f"  Steps: {args.n_steps}, CFG: {args.cfg}")

        # Per-tier breakdown
        from collections import Counter
        tier_counts = Counter(item.get("resolution_tier", "?") for item in plan)
        res_counts = Counter(f"{item['width']}x{item['height']}" for item in plan)
        backend_counts = Counter(item["attention_backend"] for item in plan)

        print(f"\n  By tier: {dict(tier_counts)}")
        print(f"  By resolution: {dict(res_counts)}")
        print(f"  By backend: {dict(backend_counts)}")

        print(f"\n--- Bin Packing ---")
        print(f"  Bins: {efficiency['n_bins']}")
        print(f"  Utilization: {efficiency['utilization']:.1%}")
        print(f"  Total seq_len capacity: {efficiency['total_capacity']}")
        print(f"  Total seq_len used: {efficiency['total_used']}")
        print(f"  Total seq_len wasted: {efficiency['total_wasted']}")

        # Histogram of bin sizes
        size_counts = Counter(len(b) for b in bins)
        print(f"\n  Bin size distribution:")
        for size in sorted(size_counts):
            print(f"    {size} items/bin: {size_counts[size]} bins")

        # Show first few bins in detail
        n_show = min(5, len(bins))
        print(f"\n  First {n_show} bins:")
        for i, b in enumerate(bins[:n_show]):
            items_desc = ", ".join(
                f"{item['width']}x{item['height']}({item['seq_len']})"
                for item in b
            )
            used = sum(item["seq_len"] for item in b)
            print(f"    bin {i}: [{items_desc}] = {used}/{REFERENCE_SEQ_LEN} "
                  f"({used / REFERENCE_SEQ_LEN:.0%})")

        return 0

    # ---------------------------------------------------------------
    # Live generation
    # ---------------------------------------------------------------
    from futudiffu.client import InferenceClient
    from src_ii.dataset_generator import DatasetGenerationConfig, DatasetGenerator

    config = DatasetGenerationConfig(
        prompts=prompts,
        seeds=args.seeds,
        resolution_tiers=args.resolution_tiers,
        attention_backends=args.attention_backends,
        n_steps=args.n_steps,
        cfg=args.cfg,
        output_dir=args.output_dir,
        run_name=args.run_name,
        base_model_hash=args.base_model_hash,
        sparse_steps=args.sparse_steps,
        sampling_shift=args.sampling_shift,
        multiplier=args.multiplier,
        server_endpoint=args.server,
        flush_interval=args.flush_interval,
        render_count=args.render_count,
        source_device=args.source_device,
    )

    with InferenceClient(config.server_endpoint, timeout_ms=args.timeout_ms) as client:
        try:
            status = client.status()
            print(f"Connected to server: {status.get('loaded_models', [])} loaded, "
                  f"VRAM {status.get('vram_allocated_gb', '?')}GB allocated")
        except Exception as e:
            print(f"Cannot connect to inference server at {config.server_endpoint}: {e}")
            print("Start the server first: python -m futudiffu.server ...")
            return 1

        generator = DatasetGenerator(config, client)
        summary = generator.generate_all()

        # Save summary to output dir
        summary_path = Path(config.output_dir) / "generation_summary.json"
        # Convert non-serializable values
        summary_serializable = json.loads(json.dumps(summary, default=str))
        summary_path.write_text(json.dumps(summary_serializable, indent=2))
        print(f"\nSummary saved to: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

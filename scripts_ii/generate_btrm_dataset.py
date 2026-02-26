r"""Generate BTRM training dataset with mixed-resolution bin packing (V2 format).

Thin CLI wrapper around src_ii.dataset_generator.DatasetGenerator. All
generation logic lives in the library module; this script owns:
  - CLI argument parsing
  - Prompt loading (file or inline)
  - Plan loading (JSON generation plans)
  - InferenceClient lifecycle
  - Dry-run mode (plan + pack, no server)

Usage:
  # From a plan file (recommended):
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\generate_btrm_dataset.py \
      --plan F:\dox\repos\ai\futudiffu\plans\scrimblo_only.json

  # Multi-node with seed partitioning:
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\generate_btrm_dataset.py \
      --plan plans\2x2xh100_joint.json --node-rank 0

  # Dry-run to inspect expanded plan:
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\generate_btrm_dataset.py \
      --plan plans\joint_scrimblo_scrongle.json --dry-run

  # Legacy inline prompts (still supported):
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\generate_btrm_dataset.py \
      --prompts-file F:\dox\repos\ai\futudiffu\prompts.txt

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

    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--plan", type=str,
        help="Path to a JSON generation plan file. Overrides all other "
             "prompt/config args (CLI flags like --output-dir, --server "
             "still apply as overlays).")
    prompt_group.add_argument(
        "--prompts-file", type=str,
        help="Path to a text file with one prompt per line.")
    prompt_group.add_argument(
        "--prompts", type=str, nargs="+",
        help="Inline prompt strings.")
    prompt_group.add_argument(
        "--use-builtin-prompts", action="store_true",
        help="Use the 24 built-in prompt templates from futudiffu.btrm_dataset.")

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

    parser.add_argument(
        "--server", type=str, default="tcp://localhost:5555",
        help="Inference server ZeroMQ endpoint.")
    parser.add_argument("--timeout-ms", type=int, default=0,
                        help="ZeroMQ receive timeout in ms. 0 = infinite.")

    parser.add_argument("--base-model-hash", type=str, default="z_image_v1",
                        help="Base model identifier for provenance tracking.")
    parser.add_argument("--source-device", type=str, default="unknown",
                        help="Device identifier for provenance (e.g. rtx4090_0).")

    parser.add_argument("--dry-run", action="store_true",
                        help="Show generation plan and packing stats, then exit.")
    parser.add_argument("--node-rank", type=int, default=0,
                        help="Node rank for seed partitioning in multi-node plans. "
                             "Offsets seeds by rank * seeds_per_node. Default: 0.")

    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Resolve configs: either from --plan or from individual CLI args
    # -----------------------------------------------------------------
    if args.plan:
        from src_ii.dataset_generator import load_plan
        configs = load_plan(args.plan, node_rank=args.node_rank)
        # CLI args override plan fields when explicitly provided
        for cfg in configs:
            if args.output_dir != str(REPO_ROOT / "btrm_dataset_v2"):
                cfg.output_dir = args.output_dir
            if args.server != "tcp://localhost:5555":
                cfg.server_endpoint = args.server
            if args.run_name != "generation":
                cfg.run_name = args.run_name
            if args.source_device != "unknown":
                cfg.source_device = args.source_device
            if args.base_model_hash != "z_image_v1":
                cfg.base_model_hash = args.base_model_hash
        print(f"Loaded plan: {args.plan}")
        print(f"  Phases: {len(configs)}")
        if args.node_rank > 0:
            print(f"  Node rank: {args.node_rank}")
        for i, cfg in enumerate(configs):
            if cfg.plan_spec is not None:
                from src_ii.dataset_generator import distribution_plan_summary
                spec = cfg.plan_spec
                summ = distribution_plan_summary(
                    spec, spec["_prompts"], spec["_base_seed"],
                    spec["_seeds_per_combo"], spec["_node_rank"],
                )
                print(f"  Phase {i}: {len(cfg.prompts)} prompts, "
                      f"{summ['total_items']} items (streaming tiles), "
                      f"steps={sorted(summ['step_counts'].keys())}, "
                      f"backends={sorted(summ['backend_counts'].keys())}")
            else:
                print(f"  Phase {i}: {len(cfg.prompts)} prompts, "
                      f"tiers={cfg.resolution_tiers}, "
                      f"backends={cfg.attention_backends}, "
                      f"steps={cfg.n_steps}")
    else:
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

        from src_ii.dataset_generator import DatasetGenerationConfig
        configs = [DatasetGenerationConfig(
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
        )]

    # -----------------------------------------------------------------
    # Dry run: show expanded plan for each phase, then exit
    # -----------------------------------------------------------------
    if args.dry_run:
        import random as _rng
        from collections import Counter

        for phase_idx, cfg in enumerate(configs):
            if len(configs) > 1:
                print(f"\n{'=' * 50}")
                print(f"  PHASE {phase_idx}")
                print(f"{'=' * 50}")

            if cfg.plan_spec is not None:
                # --- Distribution-valued plan: streaming summary ---
                from src_ii.dataset_generator import (
                    distribution_plan_summary,
                    _iter_distribution_tiles,
                    _pack_tile,
                )
                spec = cfg.plan_spec
                summ = distribution_plan_summary(
                    spec, spec["_prompts"], spec["_base_seed"],
                    spec["_seeds_per_combo"], spec["_node_rank"],
                )

                print(f"\n--- Generation Plan ---")
                print(f"  Total trajectories: {summ['total_items']}")
                print(f"  Tiles: {summ['n_tiles']} ({summ['tile_size']} items/tile)")

                print(f"\n  --- Distribution Summary ---")

                step_counts = summ["step_counts"]
                if len(step_counts) > 1:
                    print(f"  n_steps [enumeration]: {sorted(step_counts.keys())} "
                          f"({len(step_counts)} values)")
                else:
                    print(f"  n_steps [fixed]: {list(step_counts.keys())[0]}")

                backend_counts = summ["backend_counts"]
                if len(backend_counts) > 1:
                    print(f"  attention_backends [enumeration]: {sorted(backend_counts.keys())} "
                          f"({len(backend_counts)} values)")
                else:
                    print(f"  attention_backends [fixed]: {list(backend_counts.keys())[0]}")

                cfg_counts = summ["cfg_counts"]
                if len(cfg_counts) > 1:
                    print(f"  cfg [sampled]: {len(cfg_counts)} unique values, "
                          f"range [{min(cfg_counts.keys()):.2f}, {max(cfg_counts.keys()):.2f}]")
                else:
                    print(f"  cfg [fixed]: {list(cfg_counts.keys())[0]}")

                print(f"  resolution [sampled]: {summ['n_unique_resolutions']} unique (W,H) pairs")

                print(f"  Resolution anchors hit:")
                for label, count in summ["anchor_counts"].items():
                    print(f"    {label}: {count} items")

                lo, hi = summ["seed_range"]
                print(f"  Seeds: {lo}..{hi} ({summ['n_unique_seeds']} unique)")
                print(f"  Prompts: {len(cfg.prompts)}")

                n_enum_steps = len(step_counts)
                n_backends = len(backend_counts)
                print(f"\n  Trajectory count: {summ['n_unique_seeds']} seeds x "
                      f"{len(cfg.prompts)} prompts x {n_enum_steps} step counts x "
                      f"{n_backends} backends = {summ['total_items']}")

                # Pair invariant: guaranteed by construction (all items in a tile
                # share the same (W, H), sampled from (seed, prompt_idx)).
                print(f"\n  --- Pair Invariants ---")
                print(f"  Scrimblo (same res across backends): "
                      f"PASS (by construction — tile shares (W,H))")
                print(f"  Scrongle (same res across step counts): "
                      f"PASS (by construction — tile shares (W,H))")

                # Cross-tile packing stats
                from src_ii.dataset_generator import TilePacker
                tile_iter = _iter_distribution_tiles(
                    spec, spec["_prompts"], spec["_base_seed"],
                    spec["_seeds_per_combo"], spec["_node_rank"],
                )
                packer = TilePacker()
                all_bins = []
                for tile in tile_iter:
                    all_bins.extend(packer.add_tile(tile))
                all_bins.extend(packer.flush())

                total_bin_items = sum(len(b) for b in all_bins)
                size_counts = Counter(len(b) for b in all_bins)

                print(f"\n--- Cross-Tile Bin Packing ---")
                print(f"  Total bins: {len(all_bins)}")
                print(f"  Total items: {total_bin_items}")
                print(f"  Bin size distribution:")
                for size in sorted(size_counts):
                    print(f"    {size} items/bin: {size_counts[size]} bins")

                # Show a few example bins
                n_show = min(5, len(all_bins))
                print(f"\n  First {n_show} bins:")
                for bi, b in enumerate(all_bins[:n_show]):
                    w, h = b[0]["width"], b[0]["height"]
                    ns = sorted(set(item["n_steps"] for item in b))
                    seeds = sorted(set(item["seed"] for item in b))
                    prompts = sorted(set(item["prompt_idx"] for item in b))
                    print(f"    bin {bi}: {len(b)}x {w}x{h}, n_steps={ns}, "
                          f"seeds={seeds}, prompts={prompts}")
            else:
                # --- Legacy plan: materialize and global bin pack ---
                seeds = cfg.seeds
                if seeds is None:
                    rng = _rng.Random(42)
                    max_rollouts = sum(
                        RESOLUTION_TIERS[t]["rollouts_per_prompt"]
                        for t in cfg.resolution_tiers
                    )
                    n_seeds = len(cfg.prompts) * max_rollouts * len(cfg.attention_backends)
                    seeds = [rng.randint(0, 2**32 - 1) for _ in range(n_seeds)]

                plan = build_generation_plan(
                    prompts=cfg.prompts,
                    seeds=seeds,
                    resolution_tiers=cfg.resolution_tiers,
                    attention_backends=cfg.attention_backends,
                    n_steps=cfg.n_steps,
                    cfg=cfg.cfg,
                    sparse_steps=cfg.sparse_steps,
                )

                scheduler = BinPackScheduler(max_seq_len=REFERENCE_SEQ_LEN)
                bins = scheduler.pack_generation_plan(plan)
                efficiency = scheduler.estimate_efficiency(bins)

                print(f"\n--- Generation Plan ---")
                print(f"  Total trajectories: {len(plan)}")
                print(f"  Resolution tiers: {cfg.resolution_tiers}")
                print(f"  Attention backends: {cfg.attention_backends}")
                print(f"  Steps: {cfg.n_steps}, CFG: {cfg.cfg}")
                if cfg.seeds:
                    print(f"  Seed range: {min(cfg.seeds)}..{max(cfg.seeds)} ({len(cfg.seeds)} seeds)")

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

                size_counts = Counter(len(b) for b in bins)
                print(f"\n  Bin size distribution:")
                for size in sorted(size_counts):
                    print(f"    {size} items/bin: {size_counts[size]} bins")

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

    # -----------------------------------------------------------------
    # Live generation: iterate phases sequentially
    # -----------------------------------------------------------------
    from futudiffu.client import InferenceClient
    from src_ii.dataset_generator import DatasetGenerator

    # All phases write to the same output dir (first config's)
    server_endpoint = configs[0].server_endpoint

    with InferenceClient(server_endpoint, timeout_ms=args.timeout_ms) as client:
        try:
            status = client.status()
            print(f"Connected to server: {status.get('loaded_models', [])} loaded, "
                  f"VRAM {status.get('vram_allocated_gb', '?')}GB allocated")
        except Exception as e:
            print(f"Cannot connect to inference server at {server_endpoint}: {e}")
            print("Start the server first: python -m futudiffu.server ...")
            return 1

        all_summaries = []
        for phase_idx, cfg in enumerate(configs):
            if len(configs) > 1:
                print(f"\n{'=' * 50}")
                print(f"  PHASE {phase_idx}: steps={cfg.n_steps}, "
                      f"backends={cfg.attention_backends}")
                print(f"{'=' * 50}")

            generator = DatasetGenerator(cfg, client)
            summary = generator.generate_all()
            all_summaries.append(summary)

        # Save combined summary
        output_dir = Path(configs[0].output_dir)
        summary_path = output_dir / "generation_summary.json"
        if len(all_summaries) == 1:
            save_data = all_summaries[0]
        else:
            save_data = {"phases": all_summaries}
        summary_serializable = json.loads(json.dumps(save_data, default=str))
        summary_path.write_text(json.dumps(summary_serializable, indent=2))
        print(f"\nSummary saved to: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

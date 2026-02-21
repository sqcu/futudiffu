r"""Terminal inspector for V2 dataset catalog.

Structured text output designed for both humans and AI agents (Claude,
DeepSeek, Gemini, etc.). No ncurses, no interactive TUI -- just
well-formatted terminal output with clear sections that any LLM can
parse and reason about.

Commands:
  list      -- List all registered datasets with summary
  inspect   -- Deep-dive into one dataset's index structure
  register  -- Register a new V2 dataset
  verify    -- Verify integrity of all registered datasets
  split     -- Define a train/val split on a dataset
  unregister -- Remove a dataset from the catalog

Usage:
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\inspect_datasets.py list
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\inspect_datasets.py inspect <dataset_id>
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\inspect_datasets.py register <path> [--name NAME] [--tags tag1,tag2]
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\inspect_datasets.py verify
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\inspect_datasets.py split <dataset_id> <split_name> --range <start>-<end>
  .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\inspect_datasets.py unregister <dataset_id>
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# Ensure repo root is on path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src_ii.dataset_catalog import (
    DatasetCatalog,
    DatasetIntegrityError,
    DatasetNotFoundError,
    register_dataset,
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_DOUBLE_LINE = "\u2550"  # box drawing double horizontal
_SINGLE_LINE = "\u2500"  # box drawing light horizontal

def _header(title: str, width: int = 62) -> str:
    """Format a section header with double-line box drawing."""
    border = _DOUBLE_LINE * width
    return f"{border}\n {title}\n{border}"


def _subheader(title: str, width: int = 62) -> str:
    """Format a sub-section header with single-line + label."""
    pad = width - len(title) - 4
    return f"\n{_SINGLE_LINE}{_SINGLE_LINE} {title} {_SINGLE_LINE * max(pad, 2)}"


def _status_str(ok: bool, msg: str) -> str:
    """Format an integrity status string."""
    if ok:
        return f"INTACT (hash verified)"
    else:
        return f"FAILED: {msg}"


def _format_distribution(dist: dict[str, int], max_entries: int = 12) -> str:
    """Format a value distribution as a compact string."""
    if not dist:
        return "{}"

    items = sorted(dist.items(), key=lambda kv: -kv[1])

    if len(items) <= max_entries:
        parts = [f"{k}:{v}" for k, v in items]
    else:
        parts = [f"{k}:{v}" for k, v in items[:max_entries]]
        remaining = len(items) - max_entries
        parts.append(f"...+{remaining} more")

    return "{" + ", ".join(parts) + "}"


def _format_range(rng: list) -> str:
    """Format a [min, max] range."""
    if len(rng) != 2:
        return str(rng)
    lo, hi = rng
    if isinstance(lo, float):
        return f"[{lo:.2f}, {hi:.2f}]"
    return f"[{lo}, {hi}]"


def _format_size(mb: float) -> str:
    """Format a size in MB with appropriate units."""
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.1f} MB"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_list(catalog: DatasetCatalog) -> None:
    """List all registered datasets."""
    dataset_ids = catalog.list_datasets()

    if not dataset_ids:
        print(_header("DATASET CATALOG (empty)"))
        print("\nNo datasets registered. Use 'register' to add one.")
        return

    print(_header(f"DATASET CATALOG ({len(dataset_ids)} datasets registered)"))

    for i, did in enumerate(dataset_ids, 1):
        entry = catalog.get(did)
        summary = entry.get("index_summary", {})
        unique_counts = summary.get("unique_counts", {})
        splits = entry.get("splits", {})

        # Compute quick metrics
        n_prompts = unique_counts.get("prompt_idx", "?")
        n_widths = unique_counts.get("width", 0)
        n_heights = unique_counts.get("height", 0)
        n_resolutions = max(n_widths, n_heights)
        n_backends = unique_counts.get("attention_backend", "?")

        # Verify integrity
        ok, msg = catalog.verify(did)

        # Format splits
        split_str = "none"
        if splits:
            split_parts = [f"{name} ({len(ids)})" for name, ids in splits.items()]
            split_str = ", ".join(split_parts)

        # Format tags
        tags = entry.get("tags", [])
        tag_str = ", ".join(tags) if tags else "none"

        print(f"\n[{i}] {did}")
        print(f"    Path: {entry['path']}")
        print(f"    Trajectories: {entry['n_trajectories']}  |  "
              f"Blobs: {entry['blob_count']}  |  "
              f"Size: {_format_size(entry['total_size_mb'])}")
        print(f"    Prompts: {n_prompts}  |  Resolutions: {n_resolutions}  |  "
              f"Backends: {n_backends}")
        print(f"    Tags: {tag_str}")
        print(f"    Splits: {split_str}")
        print(f"    Status: {_status_str(ok, msg)}")

    print()


def cmd_inspect(catalog: DatasetCatalog, dataset_id: str) -> None:
    """Deep-dive inspection of a single dataset."""
    entry = catalog.get(dataset_id)
    summary = entry.get("index_summary", {})
    unique_counts = summary.get("unique_counts", {})
    value_ranges = summary.get("value_ranges", {})
    value_distributions = summary.get("value_distributions", {})
    columns = summary.get("columns", [])
    splits = entry.get("splits", {})

    # Verify
    ok, msg = catalog.verify(dataset_id)

    print(_header(f"DATASET: {dataset_id}"))
    print(f"\nPath: {entry['path']}")
    print(f"Created: {entry['created']}")
    print(f"Index hash: {entry['index_hash'][:16]}... "
          f"({'VERIFIED' if ok else 'FAILED: ' + msg})")
    print(f"Format: {entry['format_version']}  |  "
          f"Trajectories: {entry['n_trajectories']}  |  "
          f"Blobs: {entry['blob_count']}  |  "
          f"Size: {_format_size(entry['total_size_mb'])}")

    # --- Variation analysis table ---
    print(_subheader("Index Columns (variation analysis)"))
    print()

    # Determine column widths
    col_w = max(len(c) for c in columns) + 1 if columns else 20
    col_w = max(col_w, 20)

    header_fmt = f"{'Column':<{col_w}} {'Unique':>7}  Range/Values"
    sep_fmt = f"{_SINGLE_LINE * col_w} {_SINGLE_LINE * 7} {_SINGLE_LINE * 32}"
    print(header_fmt)
    print(sep_fmt)

    constant_cols: list[tuple[str, str]] = []
    max_variation_cols: list[tuple[str, int, int]] = []
    n_traj = entry["n_trajectories"]

    for col_name in columns:
        if col_name == "step_indices":
            continue

        n_unique = unique_counts.get(col_name, 0)

        # Determine range/values string
        if col_name in value_distributions and n_unique <= _LOW_CARDINALITY_THRESHOLD:
            dist = value_distributions[col_name]
            if n_unique == 1:
                val = list(dist.keys())[0]
                rv_str = f"{{{val}}}  <-- NO VARIATION"
                constant_cols.append((col_name, val))
            else:
                rv_str = _format_distribution(dist)
        elif col_name in value_ranges:
            rv_str = _format_range(value_ranges[col_name])
        else:
            rv_str = ""

        # Track max-variation columns
        if n_unique > 0 and n_unique == n_traj:
            max_variation_cols.append((col_name, n_unique, n_traj))

        print(f"{col_name:<{col_w}} {n_unique:>7}  {rv_str}")

    # --- Constant columns ---
    if constant_cols:
        print(_subheader("Columns with NO variation (constant)"))
        for col_name, val in constant_cols:
            print(f"{col_name} = {val} (all trajectories)")

    # --- Max variation columns ---
    if max_variation_cols:
        print(_subheader("Columns with MAXIMUM variation (all unique)"))
        parts = [f"{c} ({n}/{t})" for c, n, t in max_variation_cols]
        print(", ".join(parts))

    # --- Resolution distribution ---
    width_dist = value_distributions.get("width", {})
    height_dist = value_distributions.get("height", {})

    if width_dist or height_dist:
        print(_subheader("Resolution Distribution"))

        # Build (width, height) pairs from the parquet if possible
        # We can reconstruct from the distributions of width and height
        # For a more accurate view, we look at the combined width+height data
        # But since we only have marginal distributions, show them separately
        # unless we can read the actual parquet
        try:
            import pyarrow.parquet as pq
            dataset_dir = Path(entry["path"])
            index_path = dataset_dir / "index.parquet"
            if index_path.exists():
                table = pq.read_table(str(index_path), columns=["width", "height"])
                res_counter: dict[str, int] = {}
                for i in range(len(table)):
                    w = table.column("width")[i].as_py()
                    h = table.column("height")[i].as_py()
                    key = f"{w}x{h}"
                    res_counter[key] = res_counter.get(key, 0) + 1

                # Group by approximate megapixel tier
                tier_groups: dict[str, list[tuple[str, int]]] = {}
                for res_str, count in sorted(res_counter.items()):
                    w, h = res_str.split("x")
                    pixels = int(w) * int(h)
                    # Assign to nearest square-root tier
                    side = int(math.sqrt(pixels))
                    # Round to nearest 128
                    tier_side = max(128, round(side / 128) * 128)
                    tier_label = f"{tier_side}sq"
                    tier_groups.setdefault(tier_label, []).append((res_str, count))

                for tier_label in sorted(tier_groups.keys(),
                                         key=lambda t: int(t.replace("sq", ""))):
                    entries = tier_groups[tier_label]
                    total = sum(c for _, c in entries)
                    pct = total / n_traj * 100
                    res_list = ", ".join(r for r, _ in entries[:6])
                    if len(entries) > 6:
                        res_list += ", ..."
                    print(f"{tier_label:>8s}: {total:>3d} ({pct:>5.1f}%)  | {res_list}")
        except Exception:
            # Fallback: show width and height distributions separately
            if width_dist:
                print(f"  Width distribution: {_format_distribution(width_dist)}")
            if height_dist:
                print(f"  Height distribution: {_format_distribution(height_dist)}")

    # --- Splits ---
    if splits:
        print(_subheader("Splits"))
        for split_name, traj_ids in splits.items():
            if traj_ids:
                lo, hi = min(traj_ids), max(traj_ids)
                print(f"{split_name}: {len(traj_ids)} trajectories "
                      f"(traj_id {lo}-{hi})")
            else:
                print(f"{split_name}: 0 trajectories")
    else:
        print(_subheader("Splits"))
        print("No splits defined. Use 'split' command to create train/val splits.")

    print()


def cmd_register(
    catalog: DatasetCatalog,
    path: str,
    name: str | None,
    tags: list[str] | None,
) -> None:
    """Register a new dataset."""
    print(f"Registering dataset at: {path}")

    try:
        dataset_id = catalog.register(path, name=name, tags=tags)
        entry = catalog.get(dataset_id)

        print(f"\nRegistered: {dataset_id}")
        print(f"  Trajectories: {entry['n_trajectories']}")
        print(f"  Blobs: {entry['blob_count']}")
        print(f"  Size: {_format_size(entry['total_size_mb'])}")
        print(f"  Hash: {entry['index_hash'][:16]}...")
        print(f"  Tags: {', '.join(entry.get('tags', [])) or 'none'}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_verify(catalog: DatasetCatalog) -> None:
    """Verify integrity of all registered datasets."""
    results = catalog.verify_all()

    if not results:
        print("No datasets registered.")
        return

    print(_header(f"INTEGRITY VERIFICATION ({len(results)} datasets)"))

    all_ok = True
    for did, (ok, msg) in results.items():
        status = _status_str(ok, msg)
        print(f"\n  {did}")
        print(f"    {status}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print(f"All {len(results)} datasets INTACT.")
    else:
        n_failed = sum(1 for ok, _ in results.values() if not ok)
        print(f"WARNING: {n_failed}/{len(results)} datasets FAILED integrity check.")


def cmd_split(
    catalog: DatasetCatalog,
    dataset_id: str,
    split_name: str,
    traj_range: str | None,
    traj_ids_str: str | None,
) -> None:
    """Define a split on a dataset."""
    if traj_range:
        parts = traj_range.split("-")
        if len(parts) != 2:
            print(f"ERROR: --range must be start-end (e.g., 0-47)", file=sys.stderr)
            sys.exit(1)
        start, end = int(parts[0]), int(parts[1])
        traj_ids = list(range(start, end + 1))
    elif traj_ids_str:
        traj_ids = [int(x.strip()) for x in traj_ids_str.split(",")]
    else:
        print("ERROR: must specify --range or --ids", file=sys.stderr)
        sys.exit(1)

    result = catalog.define_split(dataset_id, split_name, traj_ids=traj_ids)
    print(f"Defined split '{split_name}' on {dataset_id}: {len(result)} trajectories")
    if result:
        print(f"  traj_id range: {min(result)} - {max(result)}")


def cmd_unregister(catalog: DatasetCatalog, dataset_id: str) -> None:
    """Unregister a dataset (does not delete files)."""
    try:
        catalog.unregister(dataset_id)
        print(f"Unregistered: {dataset_id}")
        print(f"(Files are not deleted, only the catalog entry is removed.)")
    except DatasetNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


# Constant used in inspect for NO VARIATION flagging
_LOW_CARDINALITY_THRESHOLD = 50


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Terminal inspector for V2 dataset catalog.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--catalog",
        type=str,
        default="catalog.json",
        help="Path to catalog.json (default: catalog.json in cwd)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # list
    subparsers.add_parser("list", help="List all registered datasets")

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="Inspect a dataset")
    p_inspect.add_argument("dataset_id", help="Dataset ID to inspect")

    # register
    p_register = subparsers.add_parser("register", help="Register a V2 dataset")
    p_register.add_argument("path", help="Path to the dataset directory")
    p_register.add_argument("--name", type=str, default=None, help="Human-readable name")
    p_register.add_argument("--tags", type=str, default=None,
                            help="Comma-separated tags")

    # verify
    subparsers.add_parser("verify", help="Verify integrity of all datasets")

    # split
    p_split = subparsers.add_parser("split", help="Define a named split")
    p_split.add_argument("dataset_id", help="Dataset ID")
    p_split.add_argument("split_name", help="Split name (e.g., train, val)")
    p_split.add_argument("--range", type=str, dest="traj_range",
                         help="Trajectory ID range (e.g., 0-47)")
    p_split.add_argument("--ids", type=str, dest="traj_ids",
                         help="Comma-separated trajectory IDs")

    # unregister
    p_unreg = subparsers.add_parser("unregister",
                                     help="Remove a dataset from catalog")
    p_unreg.add_argument("dataset_id", help="Dataset ID to remove")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    catalog = DatasetCatalog(args.catalog)

    if args.command == "list":
        cmd_list(catalog)
    elif args.command == "inspect":
        cmd_inspect(catalog, args.dataset_id)
    elif args.command == "register":
        tags = args.tags.split(",") if args.tags else None
        cmd_register(catalog, args.path, args.name, tags)
    elif args.command == "verify":
        cmd_verify(catalog)
    elif args.command == "split":
        cmd_split(catalog, args.dataset_id, args.split_name,
                  args.traj_range, args.traj_ids)
    elif args.command == "unregister":
        cmd_unregister(catalog, args.dataset_id)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

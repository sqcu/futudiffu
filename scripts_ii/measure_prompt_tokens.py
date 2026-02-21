r"""Measure actual prompt token lengths from the V2 dataset.

Tokenizes all unique prompts using the Qwen3 tokenizer with the Z-Image
chat template, reports statistics, and saves results to disk.

Usage:
    .venv/Scripts/python.exe F:\dox\repos\ai\futudiffu\scripts_ii\measure_prompt_tokens.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Paths (Windows for the Python interpreter)
# ---------------------------------------------------------------------------
REPO_ROOT_WIN = r"F:\dox\repos\ai\futudiffu"
REPO_ROOT_WSL = os.path.join(os.path.dirname(__file__), "..")
TOKENIZER_PATH = os.path.join(REPO_ROOT_WIN, "src", "futudiffu", "tokenizer")
PARQUET_PATH = os.path.join(REPO_ROOT_WIN, "btrm_dataset_v2", "index.parquet")
OUTPUT_DIR = os.path.join(REPO_ROOT_WSL, "bin_packing_audit")

# Add src to path for imports
sys.path.insert(0, os.path.join(REPO_ROOT_WIN, "src"))

CHAT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


def pad_to_multiple(n: int, multiple: int) -> int:
    """Round up n to the next multiple."""
    return n + ((-n) % multiple)


def main():
    # -----------------------------------------------------------------------
    # 1. Read unique prompts from V2 parquet
    # -----------------------------------------------------------------------
    import pyarrow.parquet as pq

    table = pq.read_table(PARQUET_PATH)
    all_prompts = table.column("prompt").to_pylist()
    unique_prompts = sorted(set(all_prompts))  # sorted for determinism
    print(f"V2 dataset: {table.num_rows} rows, {len(unique_prompts)} unique prompts")

    # -----------------------------------------------------------------------
    # 2. Tokenize each with the chat template
    # -----------------------------------------------------------------------
    from transformers import Qwen2Tokenizer

    tokenizer = Qwen2Tokenizer.from_pretrained(TOKENIZER_PATH)

    results: list[dict] = []
    token_counts: list[int] = []

    for prompt in unique_prompts:
        templated = CHAT_TEMPLATE.format(prompt)
        encoded = tokenizer(templated, return_tensors=None, padding=False, add_special_tokens=False)
        n_tokens = len(encoded["input_ids"])
        token_counts.append(n_tokens)
        results.append({
            "prompt": prompt[:200],  # truncate for readability
            "raw_token_count": n_tokens,
            "padded_to_32": pad_to_multiple(n_tokens, 32),
        })

    # -----------------------------------------------------------------------
    # 3. Compute statistics
    # -----------------------------------------------------------------------
    token_counts_sorted = sorted(token_counts)
    n = len(token_counts_sorted)

    def percentile(data: list[int], p: float) -> int:
        """Compute percentile (nearest-rank method)."""
        k = int(p / 100.0 * (len(data) - 1) + 0.5)
        return data[min(k, len(data) - 1)]

    stats = {
        "n_unique_prompts": n,
        "n_total_rows": table.num_rows,
        "min": min(token_counts),
        "max": max(token_counts),
        "mean": round(statistics.mean(token_counts), 2),
        "median": int(statistics.median(token_counts)),
        "p90": percentile(token_counts_sorted, 90),
        "p95": percentile(token_counts_sorted, 95),
        "p99": percentile(token_counts_sorted, 99),
    }

    padded_stats = {
        "min_padded": pad_to_multiple(stats["min"], 32),
        "max_padded": pad_to_multiple(stats["max"], 32),
        "mean_padded": pad_to_multiple(round(stats["mean"]), 32),
        "median_padded": pad_to_multiple(stats["median"], 32),
        "p90_padded": pad_to_multiple(stats["p90"], 32),
        "p95_padded": pad_to_multiple(stats["p95"], 32),
        "p99_padded": pad_to_multiple(stats["p99"], 32),
    }

    # -----------------------------------------------------------------------
    # 4. Print report
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("PROMPT TOKEN LENGTH STATISTICS")
    print("=" * 60)
    print(f"  Unique prompts:  {stats['n_unique_prompts']}")
    print(f"  Total dataset rows: {stats['n_total_rows']}")
    print()
    print("  Raw token counts:")
    print(f"    min:    {stats['min']}")
    print(f"    max:    {stats['max']}")
    print(f"    mean:   {stats['mean']}")
    print(f"    median: {stats['median']}")
    print(f"    p90:    {stats['p90']}")
    print(f"    p95:    {stats['p95']}")
    print(f"    p99:    {stats['p99']}")
    print()
    print("  Padded to 32:")
    print(f"    min:    {padded_stats['min_padded']}")
    print(f"    max:    {padded_stats['max_padded']}")
    print(f"    mean:   {padded_stats['mean_padded']}")
    print(f"    median: {padded_stats['median_padded']}")
    print(f"    p90:    {padded_stats['p90_padded']}")
    print(f"    p95:    {padded_stats['p95_padded']}")
    print(f"    p99:    {padded_stats['p99_padded']}")

    print()
    print("Per-prompt breakdown:")
    print(f"  {'Prompt (truncated)':60s}  {'Raw':>5s}  {'Pad32':>5s}")
    for r in sorted(results, key=lambda x: x["raw_token_count"]):
        label = r["prompt"][:58]
        print(f"  {label:60s}  {r['raw_token_count']:5d}  {r['padded_to_32']:5d}")

    # -----------------------------------------------------------------------
    # 5. Distribution histogram (text-based)
    # -----------------------------------------------------------------------
    print()
    print("Token count distribution (bucket size = 32):")
    bucket_size = 32
    max_tok = max(token_counts)
    buckets: dict[int, int] = {}
    for tc in token_counts:
        bucket = (tc // bucket_size) * bucket_size
        buckets[bucket] = buckets.get(bucket, 0) + 1

    max_count = max(buckets.values())
    for b in sorted(buckets.keys()):
        bar_len = int(40 * buckets[b] / max_count)
        bar = "#" * bar_len
        print(f"  [{b:4d}-{b + bucket_size - 1:4d}]: {buckets[b]:3d} {bar}")

    # -----------------------------------------------------------------------
    # 6. Save to disk
    # -----------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "prompt_token_stats.json")

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chat_template": CHAT_TEMPLATE,
        "tokenizer": "Qwen2Tokenizer (Qwen3-4B)",
        "pad_multiple": 32,
        "stats": stats,
        "padded_stats": padded_stats,
        "per_prompt": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")

    # Return p90 for use by bin_packer update
    print(f"\n{'=' * 60}")
    print(f"KEY RESULT: p90 raw token count = {stats['p90']}")
    print(f"KEY RESULT: p90 padded to 32    = {padded_stats['p90_padded']}")
    print(f"{'=' * 60}")

    return stats, padded_stats


if __name__ == "__main__":
    main()

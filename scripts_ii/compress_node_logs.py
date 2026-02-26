"""Post-hoc (or mid-run) log compression for pulled remote log files.

Parses multiple log formats (server, training, lifecycle), merges into
a unified timeline, and compresses healthy spans while expanding anomalous
regions with context.

Usage:
    python scripts_ii/compress_node_logs.py --logs-dir pulled_logs/
    python scripts_ii/compress_node_logs.py --logs-dir pulled_logs/ --output timeline.md
    python scripts_ii/compress_node_logs.py --file remote_validation/server_gpu0.log
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class LogEntry:
    timestamp: datetime | None
    source: str  # filename
    line_number: int
    raw: str
    entry_type: str = "info"  # info, lifecycle, step, error, oom
    parsed: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

# Server log patterns
RE_SERVER_ENDPOINT = re.compile(
    r"\[(\w+)\]\s+.*?(\d+\.\d+)s$"
)
RE_LIFECYCLE = re.compile(
    r"\[lifecycle\]\s+(.+?)(?:\s+in\s+(\d+\.\d+)s)?$"
)
RE_INFERENCE_SERVER = re.compile(
    r"Inference server listening on (.+)"
)
RE_CUDA_OOM = re.compile(
    r"CUDA out of memory|torch\.OutOfMemoryError|CUDA OOM"
)

# Training log patterns
RE_TRAINING_STEP = re.compile(
    r"[Ss]tep\s+(\d+)(?:/(\d+))?.*?loss[=:]\s*([\d.eE+-]+|nan|NaN|inf|Inf)"
)

# ISO timestamp
RE_ISO_TS = re.compile(
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)?)"
)

# Generic timestamp in brackets: [2026-02-25 14:30:00]
RE_BRACKET_TS = re.compile(
    r"\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]"
)


def _try_parse_timestamp(line: str) -> datetime | None:
    """Extract a timestamp from a log line, if present."""
    m = RE_ISO_TS.search(line)
    if m:
        ts_str = m.group(1)
        try:
            return datetime.fromisoformat(ts_str)
        except ValueError:
            pass

    m = RE_BRACKET_TS.search(line)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass

    return None


def parse_log_entries(path: Path) -> list[LogEntry]:
    """Read a log file and extract structured entries."""
    source = path.name
    entries: list[LogEntry] = []

    try:
        text = path.read_text(errors="replace")
    except OSError as e:
        print(f"  [warn] cannot read {path}: {e}", file=sys.stderr)
        return []

    for line_num, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if not stripped:
            continue

        ts = _try_parse_timestamp(stripped)
        entry_type = "info"
        parsed: dict = {}

        # Check for OOM
        if RE_CUDA_OOM.search(stripped):
            entry_type = "oom"

        # Check for training step
        m = RE_TRAINING_STEP.search(stripped)
        if m:
            entry_type = "step"
            parsed["step"] = int(m.group(1))
            if m.group(2):
                parsed["total_steps"] = int(m.group(2))
            loss_str = m.group(3)
            try:
                parsed["loss"] = float(loss_str)
            except ValueError:
                parsed["loss"] = float("nan")

        # Check for lifecycle event
        m = RE_LIFECYCLE.search(stripped)
        if m:
            entry_type = "lifecycle"
            parsed["event"] = m.group(1)
            if m.group(2):
                parsed["duration_s"] = float(m.group(2))

        # Check for server endpoint
        m = RE_SERVER_ENDPOINT.search(stripped)
        if m and entry_type == "info":
            entry_type = "endpoint"
            parsed["endpoint"] = m.group(1)
            parsed["duration_s"] = float(m.group(2))

        # Check for server startup
        if RE_INFERENCE_SERVER.search(stripped):
            entry_type = "lifecycle"
            parsed["event"] = "server_start"

        # Error detection
        if entry_type == "info":
            lower = stripped.lower()
            if any(kw in lower for kw in ["error", "exception", "traceback", "failed"]):
                entry_type = "error"

        entries.append(LogEntry(
            timestamp=ts,
            source=source,
            line_number=line_num,
            raw=stripped,
            entry_type=entry_type,
            parsed=parsed,
        ))

    return entries


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_timelines(entries_by_source: dict[str, list[LogEntry]]) -> list[LogEntry]:
    """Interleave entries from multiple files, sorted by timestamp.

    Entries without timestamps retain their source-relative order but are
    interleaved at the position of the nearest preceding timestamped entry.
    """
    all_entries: list[LogEntry] = []
    for source_entries in entries_by_source.values():
        all_entries.extend(source_entries)

    # Stable sort: entries with timestamps first, then by timestamp
    # Entries without timestamps keep their original order
    def sort_key(e: LogEntry):
        if e.timestamp is not None:
            return (0, e.timestamp, e.source, e.line_number)
        # Put un-timestamped entries at the end, ordered by source + line
        return (1, datetime.min, e.source, e.line_number)

    all_entries.sort(key=sort_key)
    return all_entries


# ---------------------------------------------------------------------------
# Compress
# ---------------------------------------------------------------------------

ANOMALOUS_TYPES = {"oom", "error"}


def _is_anomalous(entry: LogEntry) -> bool:
    """Entry types that should be expanded rather than compressed."""
    if entry.entry_type in ANOMALOUS_TYPES:
        return True
    if entry.entry_type == "step":
        loss = entry.parsed.get("loss", 0)
        if loss != loss:  # NaN check
            return True
    return False


@dataclass
class CompressedSpan:
    """A run of consecutive healthy entries collapsed into a summary."""
    start_idx: int
    end_idx: int
    count: int
    start_ts: datetime | None
    end_ts: datetime | None
    sources: set[str]
    step_range: tuple[int, int] | None = None  # (first, last) step if training
    entry_types: dict[str, int] = field(default_factory=dict)


def compress(merged: list[LogEntry], context_lines: int = 5) -> list[str]:
    """Compress healthy spans, expand anomalous regions with context.

    Returns lines of markdown.
    """
    if not merged:
        return ["_(empty log)_"]

    # Identify anomalous indices
    anomalous_indices: set[int] = set()
    for i, entry in enumerate(merged):
        if _is_anomalous(entry):
            # Mark this entry plus context_lines before and after
            for j in range(max(0, i - context_lines), min(len(merged), i + context_lines + 1)):
                anomalous_indices.add(j)

    output: list[str] = []
    i = 0
    while i < len(merged):
        if i in anomalous_indices:
            # Emit individual lines (expanded region)
            entry = merged[i]
            prefix = ""
            if entry.timestamp:
                prefix = entry.timestamp.strftime("%H:%M:%S") + " "
            marker = ""
            if _is_anomalous(entry):
                marker = " **<<<**"
            output.append(f"> {prefix}[{entry.source}:{entry.line_number}] {entry.raw}{marker}")
            i += 1
        else:
            # Start a compressed span
            span_start = i
            sources: set[str] = set()
            type_counts: dict[str, int] = {}
            first_ts = None
            last_ts = None
            first_step = None
            last_step = None

            while i < len(merged) and i not in anomalous_indices:
                e = merged[i]
                sources.add(e.source)
                type_counts[e.entry_type] = type_counts.get(e.entry_type, 0) + 1
                if e.timestamp:
                    if first_ts is None:
                        first_ts = e.timestamp
                    last_ts = e.timestamp
                if e.entry_type == "step":
                    step = e.parsed.get("step")
                    if step is not None:
                        if first_step is None:
                            first_step = step
                        last_step = step
                i += 1

            count = i - span_start

            # Format the span summary
            parts: list[str] = []
            if first_ts and last_ts and first_ts != last_ts:
                parts.append(f"{first_ts.strftime('%H:%M:%S')}–{last_ts.strftime('%H:%M:%S')}")
            elif first_ts:
                parts.append(first_ts.strftime("%H:%M:%S"))

            parts.append(f"{count} entries")

            if first_step is not None and last_step is not None:
                n_steps = type_counts.get("step", 0)
                if first_step != last_step:
                    parts.append(f"steps {first_step}–{last_step} ({n_steps} logged)")
                else:
                    parts.append(f"step {first_step}")

            type_summary = ", ".join(f"{v} {k}" for k, v in sorted(type_counts.items()) if k != "info")
            if type_summary:
                parts.append(type_summary)

            src_str = ", ".join(sorted(sources))
            parts.append(f"from {src_str}")

            output.append(f"_{' — '.join(parts)}, healthy_")

    return output


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_timeline(
    compressed_lines: list[str],
    entries_by_source: dict[str, list[LogEntry]],
) -> str:
    """Render a markdown timeline document."""
    lines: list[str] = []
    lines.append("# Compressed Log Timeline")
    lines.append("")

    # Source summary
    lines.append("## Sources")
    lines.append("")
    for src, entries in sorted(entries_by_source.items()):
        n_entries = len(entries)
        n_errors = sum(1 for e in entries if e.entry_type in ANOMALOUS_TYPES)
        err_str = f" ({n_errors} anomalous)" if n_errors else ""
        lines.append(f"- **{src}**: {n_entries} entries{err_str}")

    lines.append("")
    lines.append("## Timeline")
    lines.append("")
    lines.extend(compressed_lines)
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compress remote log files into a readable timeline",
    )
    parser.add_argument("--logs-dir", type=str, default=None,
                        help="Directory containing log files to process")
    parser.add_argument("--file", type=str, default=None,
                        help="Single log file to process")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (default: stdout)")
    parser.add_argument("--context", type=int, default=5,
                        help="Lines of context around anomalous entries (default: 5)")
    args = parser.parse_args()

    if not args.logs_dir and not args.file:
        print("error: specify --logs-dir or --file", file=sys.stderr)
        sys.exit(1)

    entries_by_source: dict[str, list[LogEntry]] = {}

    if args.file:
        path = Path(args.file)
        entries_by_source[path.name] = parse_log_entries(path)
    else:
        logs_dir = Path(args.logs_dir)
        if not logs_dir.is_dir():
            print(f"error: {logs_dir} is not a directory", file=sys.stderr)
            sys.exit(1)
        for path in sorted(logs_dir.iterdir()):
            if path.is_file() and path.suffix in (".log", ".jsonl", ".txt"):
                entries_by_source[path.name] = parse_log_entries(path)

    if not entries_by_source:
        print("warning: no log entries found", file=sys.stderr)

    total = sum(len(v) for v in entries_by_source.values())
    print(f"[compress] {len(entries_by_source)} source(s), {total} entries", file=sys.stderr)

    merged = merge_timelines(entries_by_source)
    compressed = compress(merged, context_lines=args.context)
    doc = render_timeline(compressed, entries_by_source)

    if args.output:
        Path(args.output).write_text(doc)
        print(f"[compress] wrote {args.output}", file=sys.stderr)
    else:
        print(doc)


if __name__ == "__main__":
    main()

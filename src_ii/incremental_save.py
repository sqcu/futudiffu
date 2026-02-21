"""Incremental persistence for BTRM training artifacts.

Provides crash-safe, append-only persistence for training metrics so that
intermediate results survive process termination. The core principle: every
computed result should be durable within seconds of computation, not hours.

Two components:
  1. TrainingCurveWriter: Append-only JSONL writer for per-step metrics.
     Each step's dict is written as one JSON line, flushed immediately.
     The file is valid (parseable) at all times because JSONL lines are
     independent.

  2. PeriodicSaver: Wraps a save callable and gates it on step interval.
     Used to auto-save ValidationMetrics every N steps without littering
     the training loop with interval logic.

Import constraints:
  - json, os, tempfile for file operations
  - No torch, no numpy (pure Python persistence utilities)
  - No futudiffu imports (standalone)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# TrainingCurveWriter: append-only JSONL persistence
# ---------------------------------------------------------------------------

class TrainingCurveWriter:
    """Append-only JSONL writer for per-step training metrics.

    Each call to write_step() appends one JSON object as a single line,
    then flushes to disk. The file is always valid: partial writes of
    the last line are the only possible corruption mode, and JSONL
    readers skip malformed trailing lines.

    Usage:
        writer = TrainingCurveWriter("output/training_curve.jsonl")

        for step in range(n_steps):
            entry = {"step": step, "loss": 0.5, ...}
            writer.write_step(entry)

        writer.close()

        # Later, load back:
        curve = load_training_curve_jsonl("output/training_curve.jsonl")
    """

    def __init__(self, path: str | Path, append: bool = False):
        """Open a JSONL file for writing.

        Args:
            path: Path to the JSONL output file. Parent directories are
                created if they do not exist.
            append: If True, append to an existing file (for resuming
                training). If False, truncate and start fresh.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if append else "w"
        self._file = open(str(self.path), mode)
        self._n_written = 0

    def write_step(self, entry: dict[str, Any]) -> None:
        """Write one step's metrics as a JSONL line.

        The entry is serialized as compact JSON (no indentation) on a
        single line, followed by a newline. The file is flushed after
        each write so the data is durable on the OS buffer level.

        Args:
            entry: Dict of metrics for one training step.
        """
        line = json.dumps(entry, default=str)
        self._file.write(line + "\n")
        self._file.flush()
        self._n_written += 1

    @property
    def n_written(self) -> int:
        """Number of steps written so far."""
        return self._n_written

    def close(self) -> None:
        """Close the underlying file."""
        if self._file and not self._file.closed:
            self._file.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def load_training_curve_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL training curve file back as a list of dicts.

    Skips blank lines and malformed lines (e.g., a partial write from a
    crash). This makes it safe to read a file that was being written to
    when the process terminated.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of dicts, one per successfully parsed line.
    """
    path = Path(path)
    if not path.exists():
        return []

    rows = []
    with open(str(path), "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines (e.g., partial write from crash).
                # This is expected for the last line if the process was
                # interrupted mid-write.
                pass
    return rows


# ---------------------------------------------------------------------------
# PeriodicSaver: interval-gated save trigger
# ---------------------------------------------------------------------------

class PeriodicSaver:
    """Gates a save callable on a step-count interval.

    Wraps any save function and calls it only every N steps. The save
    function receives the current step number as its argument. Saves
    are also triggered on explicit flush() calls (e.g., at end of
    training or at checkpoint steps).

    Usage:
        def save_metrics(step: int):
            tracker.save_json("validation_metrics.json")

        saver = PeriodicSaver(save_fn=save_metrics, interval=10)

        for step in range(n_steps):
            # ... training ...
            saver.maybe_save(step)  # saves at step 0, 10, 20, ...

        saver.flush(n_steps - 1)  # force final save
    """

    def __init__(
        self,
        save_fn: Callable[[int], None],
        interval: int = 10,
    ):
        """Create a periodic saver.

        Args:
            save_fn: Callable(step) that performs the actual save.
            interval: Save every N steps. Default 10.
        """
        self.save_fn = save_fn
        self.interval = interval
        self._last_saved_step: int | None = None

    def maybe_save(self, step: int) -> bool:
        """Conditionally trigger a save if the interval has elapsed.

        Args:
            step: Current training step number.

        Returns:
            True if a save was triggered, False otherwise.
        """
        if step % self.interval == 0:
            self.save_fn(step)
            self._last_saved_step = step
            return True
        return False

    def flush(self, step: int) -> None:
        """Force a save regardless of interval.

        Avoids redundant saves if the step was already saved by
        maybe_save() in the same step.

        Args:
            step: Current training step number.
        """
        if self._last_saved_step != step:
            self.save_fn(step)
            self._last_saved_step = step


# ---------------------------------------------------------------------------
# Atomic JSON save helper
# ---------------------------------------------------------------------------

def atomic_json_save(data: Any, path: str | Path, indent: int = 2) -> None:
    """Write a JSON file atomically (write to temp, then rename).

    This prevents corruption if the process is interrupted during the
    write. The reader always sees either the old complete file or the
    new complete file, never a partial write.

    Args:
        data: JSON-serializable data.
        path: Destination file path.
        indent: JSON indentation level. Default 2.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dir_name = str(path.parent)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json.tmp", dir=dir_name, delete=False
    ) as f:
        tmp_path = f.name
        json.dump(data, f, indent=indent, default=str)

    os.replace(tmp_path, str(path))

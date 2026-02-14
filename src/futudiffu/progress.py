"""Progress tracker with throughput estimation and logfile output.

Provides a terminal-friendly progress display with:
  - Running throughput (trajectories/sec, steps/sec, images/sec)
  - Exponentially smoothed ETA
  - Periodic milestone reports at configurable intervals
  - Concurrent logfile for post-hoc analysis
  - Unicode bar visualization

Usage:
    tracker = ProgressTracker(total=2304, report_interval=0.01)
    for i, entry in enumerate(plan):
        with tracker.trajectory(entry) as t:
            # ... run trajectory ...
            t.record_steps(30)
        # tracker auto-prints progress line
    tracker.finish()
"""

from __future__ import annotations

import io
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TextIO


# Box-drawing bar characters (8 levels per cell)
_BAR_CHARS = " " + "".join(chr(0x2588 - i) for i in range(7, -1, -1))
# Result: " ▏▎▍▌▋▊▉█"


def _render_bar(fraction: float, width: int = 30) -> str:
    """Render a Unicode progress bar."""
    fraction = max(0.0, min(1.0, fraction))
    full_blocks = int(fraction * width)
    remainder = (fraction * width) - full_blocks

    bar = _BAR_CHARS[8] * full_blocks
    if full_blocks < width:
        partial_idx = int(remainder * 8)
        bar += _BAR_CHARS[partial_idx]
        bar += " " * (width - full_blocks - 1)
    return bar


def _format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 0:
        return "??:??"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    return f"{h}h{m:02d}m"


def _format_rate(rate: float, unit: str) -> str:
    """Format a rate with adaptive precision."""
    if rate < 0.01:
        return f"{rate:.4f} {unit}"
    if rate < 1.0:
        return f"{rate:.3f} {unit}"
    if rate < 100:
        return f"{rate:.2f} {unit}"
    return f"{rate:.0f} {unit}"


@dataclass
class _TrajectoryContext:
    """Context manager for timing a single trajectory."""
    tracker: ProgressTracker
    entry: dict
    t_start: float = 0.0
    steps_recorded: int = 0
    rendered: bool = False

    def __enter__(self):
        self.t_start = time.monotonic()
        return self

    def record_steps(self, n: int):
        self.steps_recorded = n

    def mark_rendered(self):
        self.rendered = True

    def __exit__(self, *exc):
        elapsed = time.monotonic() - self.t_start
        self.tracker._on_trajectory_done(self, elapsed)
        return False


@dataclass
class ProgressTracker:
    """Track BTRM dataset generation progress with throughput estimation."""

    total: int
    report_interval: float = 0.01  # Report every N fraction of total (0.01 = 1%)
    logfile: Optional[Path] = None
    stream: TextIO = field(default_factory=lambda: sys.stderr)

    # Internal state
    _completed: int = field(default=0, init=False)
    _total_steps: int = field(default=0, init=False)
    _total_renders: int = field(default=0, init=False)
    _t_start: float = field(default=0.0, init=False)
    _t_last_report: float = field(default=0.0, init=False)
    _last_milestone: int = field(default=-1, init=False)
    _ema_secs_per_traj: float = field(default=0.0, init=False)
    _ema_alpha: float = field(default=0.15, init=False)
    _log_handle: Optional[TextIO] = field(default=None, init=False)
    _traj_times: list[float] = field(default_factory=list, init=False)

    def __post_init__(self):
        self._t_start = time.monotonic()
        self._t_last_report = self._t_start
        if self.logfile:
            self.logfile.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = open(self.logfile, "a", buffering=1)
            self._log(f"=== BTRM dataset generation started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
            self._log(f"Total trajectories: {self.total}, report interval: {self.report_interval:.4f}")

    def trajectory(self, entry: dict):
        """Context manager that times a trajectory and updates progress."""
        return _TrajectoryContext(tracker=self, entry=entry)

    def _on_trajectory_done(self, ctx: _TrajectoryContext, elapsed: float):
        """Called when a trajectory completes."""
        self._completed += 1
        self._total_steps += ctx.steps_recorded
        if ctx.rendered:
            self._total_renders += 1
        self._traj_times.append(elapsed)

        # EMA smoothing for per-trajectory time
        if self._ema_secs_per_traj == 0:
            self._ema_secs_per_traj = elapsed
        else:
            self._ema_secs_per_traj = (
                self._ema_alpha * elapsed
                + (1 - self._ema_alpha) * self._ema_secs_per_traj)

        # Check if we hit a milestone
        if self.total > 0:
            step_size = max(1, int(self.total * self.report_interval))
            current_milestone = self._completed // step_size

            if current_milestone > self._last_milestone:
                self._last_milestone = current_milestone
                self._print_milestone(ctx, elapsed)
            else:
                self._print_inline(ctx, elapsed)

    def _compute_stats(self) -> dict:
        """Compute current throughput statistics."""
        wall = time.monotonic() - self._t_start
        remaining = self.total - self._completed

        # Raw throughput
        traj_per_sec = self._completed / wall if wall > 0 else 0
        steps_per_sec = self._total_steps / wall if wall > 0 else 0

        # ETA from EMA
        eta_ema = remaining * self._ema_secs_per_traj if self._ema_secs_per_traj > 0 else 0

        # ETA from wall clock average (more stable)
        eta_wall = remaining / traj_per_sec if traj_per_sec > 0 else 0

        # Blend: 70% EMA (responsive) + 30% wall (stable)
        eta = 0.7 * eta_ema + 0.3 * eta_wall if self._completed > 3 else eta_wall

        return {
            "completed": self._completed,
            "total": self.total,
            "fraction": self._completed / self.total if self.total > 0 else 0,
            "wall_secs": wall,
            "traj_per_sec": traj_per_sec,
            "steps_per_sec": steps_per_sec,
            "renders": self._total_renders,
            "eta_secs": eta,
            "ema_secs_per_traj": self._ema_secs_per_traj,
        }

    def _print_inline(self, ctx: _TrajectoryContext, elapsed: float):
        """Print compact inline progress (overwrites current line)."""
        s = self._compute_stats()
        frac = s["fraction"]
        bar = _render_bar(frac, 20)

        ttype = ctx.entry.get("type", "t2i")
        prec = ctx.entry.get("precision", "?")[:3]

        line = (
            f"\r  {bar} {s['completed']:>5d}/{s['total']} "
            f"({frac*100:5.1f}%) "
            f"{_format_rate(s['traj_per_sec'], 'traj/s')} "
            f"{_format_rate(s['steps_per_sec'], 'step/s')} "
            f"ETA {_format_duration(s['eta_secs'])} "
            f"[{ttype} {prec} {elapsed:.1f}s]"
        )
        # Pad to overwrite previous longer lines
        self.stream.write(f"{line:<120}")
        self.stream.flush()

    def _print_milestone(self, ctx: _TrajectoryContext, elapsed: float):
        """Print a full milestone report line (newline, not overwritten)."""
        s = self._compute_stats()
        frac = s["fraction"]
        bar = _render_bar(frac, 20)

        # Compute percentile trajectory times
        if len(self._traj_times) >= 5:
            sorted_times = sorted(self._traj_times)
            p50 = sorted_times[len(sorted_times) // 2]
            p95 = sorted_times[int(len(sorted_times) * 0.95)]
            timing = f"p50={p50:.1f}s p95={p95:.1f}s"
        else:
            timing = f"avg={s['ema_secs_per_traj']:.1f}s"

        line = (
            f"\n  {bar} {s['completed']:>5d}/{s['total']} "
            f"({frac*100:5.1f}%) "
            f"{_format_rate(s['traj_per_sec'], 'traj/s')} "
            f"{_format_rate(s['steps_per_sec'], 'step/s')} "
            f"| {timing} "
            f"| renders: {s['renders']} "
            f"| wall: {_format_duration(s['wall_secs'])} "
            f"ETA: {_format_duration(s['eta_secs'])}"
        )
        self.stream.write(line)
        self.stream.flush()

        self._log(
            f"[{frac*100:5.1f}%] {s['completed']}/{s['total']} "
            f"traj/s={s['traj_per_sec']:.3f} step/s={s['steps_per_sec']:.1f} "
            f"renders={s['renders']} wall={s['wall_secs']:.1f}s "
            f"eta={s['eta_secs']:.0f}s ema={s['ema_secs_per_traj']:.2f}s/traj")

    def _log(self, msg: str):
        """Write to logfile if open."""
        if self._log_handle:
            self._log_handle.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

    def finish(self):
        """Print final summary."""
        s = self._compute_stats()
        wall = s["wall_secs"]

        # Clear the inline progress line
        self.stream.write("\r" + " " * 120 + "\r")

        summary = [
            f"\n  {'='*60}",
            f"  BTRM Dataset Generation Complete",
            f"  {'='*60}",
            f"  Trajectories: {s['completed']}/{s['total']}",
            f"  Total steps:  {self._total_steps:,}",
            f"  Renders:      {s['renders']}",
            f"  Wall time:    {_format_duration(wall)}",
            f"  Throughput:   {_format_rate(s['traj_per_sec'], 'traj/s')}",
            f"                {_format_rate(s['steps_per_sec'], 'step/s')}",
        ]

        if len(self._traj_times) >= 5:
            sorted_t = sorted(self._traj_times)
            summary.append(
                f"  Latency:      p50={sorted_t[len(sorted_t)//2]:.2f}s "
                f"p95={sorted_t[int(len(sorted_t)*0.95)]:.2f}s "
                f"max={sorted_t[-1]:.2f}s")

        summary.append(f"  {'='*60}")

        text = "\n".join(summary)
        self.stream.write(text + "\n")
        self.stream.flush()

        self._log(text)
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None

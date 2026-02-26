"""Remote-resident node observer for futudiffu spot instances.

Runs as a tmux session (fd_observer) on each remote node. Pure local
operations — no SSH, no outbound network. Reads local files and localhost
HTTP endpoints. Writes observations to ~/.futudiffu_observations/.

Survives SSH outages. Observations accumulate whether or not anyone is
connected.

Usage:
    python scripts_ii/node_heartbeat.py \
        --interval 60 \
        --expected-sessions server_0,server_1,train \
        --ports 5555,5556 \
        --training-metrics ~/futudiffu/training_output/current_run/training_metrics.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import urllib.request
import urllib.error
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

PREFIX = "fd_"
OBS_DIR_DEFAULT = Path.home() / ".futudiffu_observations"
LOG_DIR = Path.home() / ".futudiffu_logs"

ANOMALY_CRITICAL = "CRITICAL"
ANOMALY_MAJOR = "MAJOR"
ANOMALY_MINOR = "MINOR"


@dataclass
class SessionCheck:
    name: str
    exists: bool
    last_log_lines: list[str] = field(default_factory=list)


@dataclass
class ServerCheck:
    port: int
    healthy: bool
    loaded_models: list[str] = field(default_factory=list)
    vram_allocated_gb: float = 0.0
    vram_total_gb: float = 0.0
    phase: str = ""
    error: str = ""


@dataclass
class TrainingCheck:
    available: bool = False
    step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    accuracy_pinkify: float = 0.0
    accuracy_thisnotthat: float = 0.0
    step_time: float = 0.0
    grad_norm: float = 0.0
    timestamp: str = ""
    error: str = ""


@dataclass
class Anomaly:
    severity: str  # CRITICAL, MAJOR, MINOR
    source: str  # e.g. "server_0", "train"
    pattern: str  # e.g. "session_dead", "nan_loss"
    message: str
    detail: str = ""


# ---------------------------------------------------------------------------
# Session checking
# ---------------------------------------------------------------------------

def check_sessions(expected: list[str]) -> list[SessionCheck]:
    """Check which expected tmux sessions exist and tail their logs."""
    # Get list of active tmux sessions
    try:
        result = subprocess.run(
            ["tmux", "list-sessions", "-F", "#{session_name}"],
            capture_output=True, text=True, timeout=5,
        )
        active = set(result.stdout.strip().splitlines()) if result.returncode == 0 else set()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        active = set()

    checks = []
    for name in expected:
        full_name = f"{PREFIX}{name}" if not name.startswith(PREFIX) else name
        exists = full_name in active

        # Tail log file
        log_lines: list[str] = []
        log_path = LOG_DIR / f"{name}.log"
        if log_path.exists():
            try:
                text = log_path.read_text(errors="replace")
                log_lines = text.strip().splitlines()[-30:]
            except OSError:
                pass

        checks.append(SessionCheck(name=name, exists=exists, last_log_lines=log_lines))
    return checks


# ---------------------------------------------------------------------------
# Server health
# ---------------------------------------------------------------------------

def check_server(port: int, timeout_s: float = 5.0) -> ServerCheck:
    """Hit localhost:{port}/status and parse the response."""
    url = f"http://127.0.0.1:{port}/status"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode())
        return ServerCheck(
            port=port,
            healthy=data.get("status") == "ok",
            loaded_models=data.get("loaded_models", []),
            vram_allocated_gb=data.get("vram_allocated_gb", 0.0),
            vram_total_gb=data.get("vram_total_gb", 0.0),
            phase=data.get("phase", ""),
        )
    except Exception as e:
        return ServerCheck(port=port, healthy=False, error=str(e))


# ---------------------------------------------------------------------------
# Training metrics
# ---------------------------------------------------------------------------

def check_training(metrics_path: Path | None) -> TrainingCheck:
    """Read the last entry from training_metrics.jsonl."""
    if metrics_path is None or not metrics_path.exists():
        return TrainingCheck(available=False)

    try:
        # Read last 10 lines (binary tail for efficiency)
        text = metrics_path.read_text(errors="replace")
        lines = [l for l in text.strip().splitlines() if l.strip()]
        if not lines:
            return TrainingCheck(available=False)

        last = json.loads(lines[-1])
        return TrainingCheck(
            available=True,
            step=last.get("step", 0),
            total_steps=last.get("total_steps", 0),
            loss=last.get("loss", 0.0),
            accuracy_pinkify=last.get("accuracy_pinkify", 0.0),
            accuracy_thisnotthat=last.get("accuracy_thisnotthat", 0.0),
            step_time=last.get("time_s", 0.0),
            grad_norm=last.get("pre_clip_grad_norm", 0.0),
            timestamp=last.get("timestamp", ""),
        )
    except Exception as e:
        return TrainingCheck(available=False, error=str(e))


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """Stateful anomaly detector. Maintains rolling windows for thresholds."""

    def __init__(self):
        self._step_times: deque[float] = deque(maxlen=30)
        self._grad_norms: deque[float] = deque(maxlen=30)
        self._vram: dict[int, deque[float]] = {}  # port -> recent vram
        self._last_step: int | None = None
        self._last_step_time: float | None = None
        self._same_accuracy_count: int = 0
        self._last_accuracy: tuple[float, float] | None = None

    def detect(
        self,
        session_checks: list[SessionCheck],
        server_checks: list[ServerCheck],
        training: TrainingCheck,
    ) -> list[Anomaly]:
        anomalies: list[Anomaly] = []

        # --- Session-level ---
        for sc in session_checks:
            if not sc.exists:
                last_lines = "\n".join(sc.last_log_lines[-5:]) if sc.last_log_lines else "(no log)"
                anomalies.append(Anomaly(
                    severity=ANOMALY_CRITICAL,
                    source=sc.name,
                    pattern="session_dead",
                    message=f"Session fd_{sc.name} not found in tmux",
                    detail=f"Last log lines:\n{last_lines}",
                ))

        # --- Server-level ---
        for sv in server_checks:
            session_name = f"server_{sv.port % 100}"  # approximate
            if not sv.healthy:
                anomalies.append(Anomaly(
                    severity=ANOMALY_CRITICAL,
                    source=f"port_{sv.port}",
                    pattern="server_unreachable",
                    message=f"Server on port {sv.port} unreachable or unhealthy",
                    detail=sv.error,
                ))
                continue

            # VRAM > 95%
            if sv.vram_total_gb > 0:
                vram_pct = sv.vram_allocated_gb / sv.vram_total_gb
                if vram_pct > 0.95:
                    anomalies.append(Anomaly(
                        severity=ANOMALY_MAJOR,
                        source=f"port_{sv.port}",
                        pattern="vram_critical",
                        message=f"VRAM {vram_pct:.0%} on port {sv.port}",
                        detail=f"{sv.vram_allocated_gb:.1f}/{sv.vram_total_gb:.1f} GB",
                    ))

                # VRAM growing
                if sv.port not in self._vram:
                    self._vram[sv.port] = deque(maxlen=10)
                self._vram[sv.port].append(sv.vram_allocated_gb)
                vq = self._vram[sv.port]
                if len(vq) >= 5:
                    growth = vq[-1] - vq[0]
                    if vq[0] > 0 and growth / vq[0] > 0.10:
                        anomalies.append(Anomaly(
                            severity=ANOMALY_MINOR,
                            source=f"port_{sv.port}",
                            pattern="vram_growing",
                            message=f"VRAM grew {growth:.1f} GB over {len(vq)} observations",
                        ))

        # --- Training-level ---
        if training.available:
            # NaN / Inf loss
            if math.isnan(training.loss) or math.isinf(training.loss):
                anomalies.append(Anomaly(
                    severity=ANOMALY_CRITICAL,
                    source="train",
                    pattern="nan_loss",
                    message=f"Loss is {training.loss} at step {training.step}",
                ))

            # Gradient explosion
            if training.grad_norm > 0:
                self._grad_norms.append(training.grad_norm)
                if len(self._grad_norms) >= 5:
                    median_gn = sorted(self._grad_norms)[len(self._grad_norms) // 2]
                    if median_gn > 0 and training.grad_norm > 100 * median_gn:
                        anomalies.append(Anomaly(
                            severity=ANOMALY_MAJOR,
                            source="train",
                            pattern="gradient_explosion",
                            message=f"grad_norm {training.grad_norm:.2f} vs median {median_gn:.2f}",
                        ))

            # Step time spike
            if training.step_time > 0:
                self._step_times.append(training.step_time)
                if len(self._step_times) >= 5:
                    median_st = sorted(self._step_times)[len(self._step_times) // 2]
                    if median_st > 0 and training.step_time > 3 * median_st:
                        anomalies.append(Anomaly(
                            severity=ANOMALY_MINOR,
                            source="train",
                            pattern="step_time_spike",
                            message=f"Step time {training.step_time:.1f}s vs median {median_st:.1f}s",
                        ))

            # Training stall (same step for too long)
            now = time.monotonic()
            if self._last_step is not None and training.step == self._last_step:
                if self._last_step_time is not None:
                    stall_s = now - self._last_step_time
                    median_st = (
                        sorted(self._step_times)[len(self._step_times) // 2]
                        if len(self._step_times) >= 3 else 60.0
                    )
                    if stall_s > 3 * max(median_st, 10.0):
                        anomalies.append(Anomaly(
                            severity=ANOMALY_MAJOR,
                            source="train",
                            pattern="training_stalled",
                            message=f"Step {training.step} for {stall_s:.0f}s (median step time: {median_st:.1f}s)",
                        ))
            else:
                self._last_step_time = now
            self._last_step = training.step

            # Accuracy plateau
            acc = (training.accuracy_pinkify, training.accuracy_thisnotthat)
            if self._last_accuracy is not None and acc == self._last_accuracy:
                self._same_accuracy_count += 1
            else:
                self._same_accuracy_count = 0
            self._last_accuracy = acc
            if self._same_accuracy_count >= 20:
                anomalies.append(Anomaly(
                    severity=ANOMALY_MINOR,
                    source="train",
                    pattern="accuracy_plateau",
                    message=f"Same accuracy for {self._same_accuracy_count} consecutive observations",
                ))

        # --- Log staleness (server sessions) ---
        for sc in session_checks:
            if sc.exists and "server" in sc.name and sc.last_log_lines:
                # Check if last log line has a timestamp we can parse
                log_path = LOG_DIR / f"{sc.name}.log"
                if log_path.exists():
                    try:
                        mtime = log_path.stat().st_mtime
                        age = time.time() - mtime
                        if age > 60:
                            anomalies.append(Anomaly(
                                severity=ANOMALY_MAJOR,
                                source=sc.name,
                                pattern="log_stale",
                                message=f"Log file unchanged for {age:.0f}s",
                            ))
                    except OSError:
                        pass

        return anomalies


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_observation(
    obs_dir: Path,
    cycle: int,
    session_checks: list[SessionCheck],
    server_checks: list[ServerCheck],
    training: TrainingCheck,
    anomalies: list[Anomaly],
) -> None:
    """Write heartbeat.jsonl, summary.md, and optionally anomalies.jsonl."""
    obs_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)

    # -- heartbeat.jsonl --
    heartbeat = {
        "timestamp": now.isoformat(),
        "cycle": cycle,
        "sessions": {sc.name: sc.exists for sc in session_checks},
        "servers": {
            str(sv.port): {
                "healthy": sv.healthy,
                "vram_allocated_gb": sv.vram_allocated_gb,
                "vram_total_gb": sv.vram_total_gb,
            }
            for sv in server_checks
        },
        "training": asdict(training) if training.available else None,
        "n_anomalies": len(anomalies),
    }
    with open(obs_dir / "heartbeat.jsonl", "a") as f:
        f.write(json.dumps(heartbeat) + "\n")

    # -- anomalies.jsonl --
    if anomalies:
        with open(obs_dir / "anomalies.jsonl", "a") as f:
            for a in anomalies:
                entry = {
                    "timestamp": now.isoformat(),
                    "cycle": cycle,
                    **asdict(a),
                }
                f.write(json.dumps(entry) + "\n")

    # -- summary.md --
    write_summary(obs_dir, cycle, now, session_checks, server_checks, training, anomalies)


def write_summary(
    obs_dir: Path,
    cycle: int,
    now: datetime,
    session_checks: list[SessionCheck],
    server_checks: list[ServerCheck],
    training: TrainingCheck,
    anomalies: list[Anomaly],
) -> None:
    """Overwrite summary.md with current node status."""
    n_sessions = len(session_checks)
    n_servers = len(server_checks)
    n_anomalies = len(anomalies)
    critical = [a for a in anomalies if a.severity == ANOMALY_CRITICAL]

    lines: list[str] = []
    ts = now.strftime("%Y-%m-%d %H:%M:%S UTC")
    lines.append(f"# Node Observation @ {ts} (cycle {cycle})")

    if n_anomalies == 0:
        lines.append(f"## ALL HEALTHY — {n_sessions} sessions, {n_servers} server ports")
    else:
        lines.append(f"## {n_anomalies} ANOMAL{'Y' if n_anomalies == 1 else 'IES'}"
                      f" — {n_sessions} sessions, {n_servers} server ports")

    lines.append("")
    lines.append("| Source | Status | Detail |")
    lines.append("|--------|--------|--------|")

    # Sessions
    for sc in session_checks:
        status = "ok" if sc.exists else "**DEAD**"
        detail = ""
        if not sc.exists and sc.last_log_lines:
            detail = sc.last_log_lines[-1][:80]
        lines.append(f"| {sc.name} | {status} | {detail} |")

    # Servers
    for sv in server_checks:
        if sv.healthy:
            vram = (f"{sv.vram_allocated_gb:.1f}/{sv.vram_total_gb:.1f}GB"
                    if sv.vram_total_gb > 0 else "")
            models = f"{len(sv.loaded_models)} models"
            detail = f"VRAM {vram}, {models}"
            lines.append(f"| port:{sv.port} | ok | {detail} |")
        else:
            lines.append(f"| port:{sv.port} | **DOWN** | {sv.error[:80]} |")

    # Training
    if training.available:
        step_info = f"step {training.step}"
        if training.total_steps > 0:
            step_info += f"/{training.total_steps}"
        loss_str = f"loss={training.loss:.4f}" if not math.isnan(training.loss) else "loss=NaN"
        detail = f"{step_info}, {loss_str}, time={training.step_time:.1f}s"
        lines.append(f"| train | ok | {detail} |")

    # Anomaly details
    if anomalies:
        lines.append("")
        for a in anomalies:
            severity_marker = {"CRITICAL": "###", "MAJOR": "####", "MINOR": "#####"}
            lines.append(f"{severity_marker.get(a.severity, '####')} {a.severity}: {a.source} — {a.pattern}")
            lines.append(a.message)
            if a.detail:
                lines.append(f"```\n{a.detail}\n```")
            lines.append("")

    lines.append(f"\n_Next check in {{interval}}s._")

    (obs_dir / "summary.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Node-resident observer for futudiffu remote instances",
    )
    parser.add_argument("--interval", type=int, default=60,
                        help="Observation interval in seconds (default: 60)")
    parser.add_argument("--obs-dir", type=str,
                        default=str(OBS_DIR_DEFAULT),
                        help="Output directory for observations")
    parser.add_argument("--expected-sessions", type=str,
                        default="server_0,server_1,train",
                        help="Comma-separated expected tmux session names")
    parser.add_argument("--ports", type=str, default="",
                        help="Comma-separated server ports to check (e.g. 5555,5556)")
    parser.add_argument("--training-metrics", type=str, default=None,
                        help="Path to training_metrics.jsonl")
    parser.add_argument("--cycles", type=int, default=0,
                        help="Number of cycles (0 = forever)")
    args = parser.parse_args()

    obs_dir = Path(args.obs_dir)
    expected = [s.strip() for s in args.expected_sessions.split(",") if s.strip()]
    ports = [int(p.strip()) for p in args.ports.split(",") if p.strip()]
    metrics_path = Path(args.training_metrics) if args.training_metrics else None

    detector = AnomalyDetector()
    cycle = 0

    print(f"[observer] starting — interval={args.interval}s, obs_dir={obs_dir}")
    print(f"[observer] expected sessions: {expected}")
    print(f"[observer] server ports: {ports}")
    print(f"[observer] training metrics: {metrics_path}")

    while True:
        cycle += 1
        t0 = time.monotonic()

        session_checks = check_sessions(expected)
        server_checks = [check_server(p) for p in ports]
        training = check_training(metrics_path)
        anomalies = detector.detect(session_checks, server_checks, training)

        write_observation(obs_dir, cycle, session_checks, server_checks, training, anomalies)

        elapsed = time.monotonic() - t0
        n_anomalies = len(anomalies)
        status = "HEALTHY" if n_anomalies == 0 else f"{n_anomalies} ANOMALIES"
        print(f"[observer] cycle {cycle}: {status} ({elapsed:.2f}s)")

        for a in anomalies:
            print(f"  [{a.severity}] {a.source}: {a.message}")

        if args.cycles > 0 and cycle >= args.cycles:
            print(f"[observer] completed {cycle} cycles, exiting")
            break

        sleep_time = max(0, args.interval - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()

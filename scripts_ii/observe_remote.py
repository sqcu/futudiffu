"""Local aggregator for futudiffu remote observer protocol.

Runs on the operator's machine. Pulls observation files from all nodes
via rsync, merges into a unified multi-node summary. The thing a human
or agent actually reads.

Falls back to direct SSH observation when the remote observer isn't
running (graceful degradation).

Usage:
    python scripts_ii/observe_remote.py --config remote_target.json
    python scripts_ii/observe_remote.py --config remote_target.json --interval 300 --cycles 0
    python scripts_ii/observe_remote.py --config remote_target.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PREFIX = "fd_"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class NodeConfig:
    """Config for a single remote node."""
    host: str
    ssh_key: str
    remote_dir: str
    base_port: int
    n_gpus: int
    training_output_dir: str
    expected_sessions: list[str]
    node_idx: int = 0


def load_nodes(config_path: Path) -> list[NodeConfig]:
    """Load node configs from remote_target.json.

    Supports both single-node (object) and multi-node (array) formats.
    """
    with open(config_path) as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        raw = [raw]

    nodes = []
    for idx, entry in enumerate(raw):
        n_gpus = entry.get("n_gpus", 2)
        remote_dir = entry["remote_dir"]
        default_sessions = [f"server_{i}" for i in range(n_gpus)] + ["train"]
        nodes.append(NodeConfig(
            host=entry["host"],
            ssh_key=os.path.expanduser(entry["ssh_key"]),
            remote_dir=remote_dir,
            base_port=entry.get("base_port", 5555),
            n_gpus=n_gpus,
            training_output_dir=entry.get(
                "training_output_dir",
                f"{remote_dir}/training_output",
            ),
            expected_sessions=entry.get("expected_sessions", default_sessions),
            node_idx=idx,
        ))
    return nodes


# ---------------------------------------------------------------------------
# rsync pull
# ---------------------------------------------------------------------------

def pull_observations(nodes: list[NodeConfig], local_dir: Path) -> dict[int, bool]:
    """rsync ~/.futudiffu_observations/ from each node. Returns {idx: success}."""
    results = {}
    for node in nodes:
        node_dir = local_dir / f"node_{node.node_idx}"
        node_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "rsync", "-avz", "--partial", "--timeout=15",
            "-e", f"ssh -i {node.ssh_key} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10",
            f"{node.host}:~/.futudiffu_observations/",
            str(node_dir) + "/",
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=30)
            results[node.node_idx] = True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"  [warn] rsync from node {node.node_idx} ({node.host}) failed: {e}")
            results[node.node_idx] = False
    return results


# ---------------------------------------------------------------------------
# Fallback: direct SSH observation
# ---------------------------------------------------------------------------

def fallback_direct_ssh(node: NodeConfig) -> dict:
    """SSH in directly and capture tmux state. Used when observer isn't running."""
    ssh_base = [
        "ssh", "-i", node.ssh_key,
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
        node.host,
    ]

    info: dict = {"node_idx": node.node_idx, "method": "direct_ssh", "sessions": {}}

    # List tmux sessions
    try:
        result = subprocess.run(
            ssh_base + ["tmux list-sessions -F '#{session_name}' 2>/dev/null || true"],
            capture_output=True, text=True, timeout=15,
        )
        active = set()
        for line in result.stdout.strip().splitlines():
            if line.startswith(PREFIX):
                active.add(line[len(PREFIX):])
        for name in node.expected_sessions:
            info["sessions"][name] = name in active
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        info["error"] = "SSH connection failed"

    # Check server ports
    info["servers"] = {}
    for i in range(node.n_gpus):
        port = node.base_port + i
        try:
            result = subprocess.run(
                ssh_base + [
                    f"curl -s --connect-timeout 3 http://127.0.0.1:{port}/status 2>/dev/null || echo FAIL"
                ],
                capture_output=True, text=True, timeout=15,
            )
            if result.stdout.strip() != "FAIL":
                data = json.loads(result.stdout)
                info["servers"][port] = {"healthy": True, "status": data}
            else:
                info["servers"][port] = {"healthy": False}
        except Exception:
            info["servers"][port] = {"healthy": False}

    return info


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

@dataclass
class NodeSummary:
    node_idx: int
    host: str
    observer_running: bool
    heartbeat_age_s: float  # seconds since last heartbeat
    sessions: dict  # name -> exists
    servers: dict  # port -> {healthy, vram, ...}
    training: dict | None
    anomalies: list[dict]
    method: str = "observer"  # "observer" or "direct_ssh"


def read_node_observations(node_dir: Path, node: NodeConfig) -> NodeSummary | None:
    """Read a single node's observation files."""
    heartbeat_path = node_dir / "heartbeat.jsonl"
    summary_path = node_dir / "summary.md"

    if not heartbeat_path.exists():
        return None

    # Read last heartbeat
    try:
        lines = heartbeat_path.read_text(errors="replace").strip().splitlines()
        if not lines:
            return None
        last = json.loads(lines[-1])
    except (json.JSONDecodeError, OSError):
        return None

    ts = last.get("timestamp", "")
    try:
        hb_time = datetime.fromisoformat(ts)
        age = (datetime.now(timezone.utc) - hb_time).total_seconds()
    except (ValueError, TypeError):
        age = float("inf")

    # Read recent anomalies
    anomalies: list[dict] = []
    anomaly_path = node_dir / "anomalies.jsonl"
    if anomaly_path.exists():
        try:
            alines = anomaly_path.read_text(errors="replace").strip().splitlines()
            # Last 10 anomalies
            for line in alines[-10:]:
                anomalies.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            pass

    return NodeSummary(
        node_idx=node.node_idx,
        host=node.host,
        observer_running=age < 300,  # stale if >5 min
        heartbeat_age_s=age,
        sessions=last.get("sessions", {}),
        servers=last.get("servers", {}),
        training=last.get("training"),
        anomalies=anomalies,
        method="observer",
    )


def merge_summaries(
    local_dir: Path,
    nodes: list[NodeConfig],
    pull_results: dict[int, bool],
) -> list[NodeSummary]:
    """Read each node's observations and build unified list."""
    summaries: list[NodeSummary] = []

    for node in nodes:
        node_dir = local_dir / f"node_{node.node_idx}"
        summary = read_node_observations(node_dir, node)

        if summary is not None and summary.observer_running:
            summaries.append(summary)
        else:
            # Fallback to direct SSH
            if pull_results.get(node.node_idx, False) and summary is not None:
                # We have stale data — report it but flag
                summary.observer_running = False
                summaries.append(summary)
            else:
                # Try direct SSH
                print(f"  [info] node {node.node_idx}: observer not running, trying direct SSH...")
                direct = fallback_direct_ssh(node)
                summaries.append(NodeSummary(
                    node_idx=node.node_idx,
                    host=node.host,
                    observer_running=False,
                    heartbeat_age_s=float("inf"),
                    sessions=direct.get("sessions", {}),
                    servers=direct.get("servers", {}),
                    training=None,
                    anomalies=[],
                    method="direct_ssh",
                ))

    # Cross-node anomaly: training step drift
    training_steps = {}
    for s in summaries:
        if s.training and s.training.get("available"):
            training_steps[s.node_idx] = s.training.get("step", 0)
    if len(training_steps) > 1:
        steps = list(training_steps.values())
        drift = max(steps) - min(steps)
        if drift > 10:
            for s in summaries:
                s.anomalies.append({
                    "severity": "MAJOR",
                    "source": "cross_node",
                    "pattern": "inter_node_drift",
                    "message": f"Training step drift: {drift} steps across nodes",
                    "detail": json.dumps(training_steps),
                })

    return summaries


# ---------------------------------------------------------------------------
# Unified output
# ---------------------------------------------------------------------------

def write_unified(output_dir: Path, summaries: list[NodeSummary], cycle: int) -> None:
    """Write unified summary.md and heartbeat.jsonl."""
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%d %H:%M:%S UTC")

    # Collect all anomalies
    all_anomalies: list[dict] = []
    for s in summaries:
        all_anomalies.extend(s.anomalies)

    n_nodes = len(summaries)
    total_sessions = sum(len(s.sessions) for s in summaries)
    total_gpus = sum(len(s.servers) for s in summaries)
    n_anomalies = len(all_anomalies)

    lines: list[str] = []
    lines.append(f"# Observation @ {ts} (cycle {cycle})")

    if n_anomalies == 0:
        lines.append(f"## ALL HEALTHY — {n_nodes} node(s), {total_gpus} GPU(s), {total_sessions} sessions")
    else:
        lines.append(f"## {n_anomalies} ANOMAL{'Y' if n_anomalies == 1 else 'IES'}"
                      f" — {n_nodes} node(s), {total_gpus} GPU(s), {total_sessions} sessions")

    lines.append("")
    lines.append("| Node | Source | Status | Detail |")
    lines.append("|------|--------|--------|--------|")

    for s in summaries:
        method_tag = "" if s.observer_running else " (ssh fallback)"

        # Sessions
        for name, exists in s.sessions.items():
            status = "ok" if exists else "**DEAD**"
            lines.append(f"| {s.node_idx} | {name} | {status} | {method_tag} |")

        # Servers
        for port_str, info in s.servers.items():
            if isinstance(info, dict) and info.get("healthy"):
                vram_a = info.get("vram_allocated_gb", 0)
                vram_t = info.get("vram_total_gb", 0)
                vram_str = f"VRAM {vram_a:.1f}/{vram_t:.1f}GB" if vram_t > 0 else ""
                lines.append(f"| {s.node_idx} | port:{port_str} | ok | {vram_str} |")
            else:
                lines.append(f"| {s.node_idx} | port:{port_str} | **DOWN** | |")

        # Training
        if s.training and s.training.get("available"):
            t = s.training
            step = t.get("step", "?")
            total = t.get("total_steps", 0)
            step_str = f"step {step}" + (f"/{total}" if total else "")
            loss = t.get("loss", 0)
            lines.append(f"| {s.node_idx} | train | ok | {step_str}, loss={loss:.4f} |")

        # Heartbeat staleness
        if not s.observer_running and s.heartbeat_age_s < float("inf"):
            lines.append(f"| {s.node_idx} | observer | **STALE** | {s.heartbeat_age_s:.0f}s since last heartbeat |")

    # Anomaly details
    if all_anomalies:
        lines.append("")
        lines.append("---")
        lines.append("## Anomalies")
        lines.append("")
        for a in all_anomalies:
            sev = a.get("severity", "?")
            src = a.get("source", "?")
            msg = a.get("message", "")
            detail = a.get("detail", "")
            lines.append(f"### {sev}: {src} — {a.get('pattern', '?')}")
            lines.append(msg)
            if detail:
                lines.append(f"```\n{detail}\n```")
            lines.append("")

    lines.append(f"\n_Cycle {cycle}. Generated {ts}._")

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")

    # -- heartbeat.jsonl --
    heartbeat = {
        "timestamp": now.isoformat(),
        "cycle": cycle,
        "n_nodes": n_nodes,
        "n_anomalies": n_anomalies,
        "nodes": {
            s.node_idx: {
                "observer_running": s.observer_running,
                "heartbeat_age_s": s.heartbeat_age_s if s.heartbeat_age_s < float("inf") else None,
                "method": s.method,
            }
            for s in summaries
        },
    }
    with open(output_dir / "heartbeat.jsonl", "a") as f:
        f.write(json.dumps(heartbeat) + "\n")

    # Print anomalies to stdout
    if all_anomalies:
        print(f"\n  === {n_anomalies} ANOMALIES ===")
        for a in all_anomalies:
            print(f"  [{a.get('severity')}] {a.get('source')}: {a.get('message')}")
    else:
        print(f"  ALL HEALTHY — {n_nodes} node(s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local aggregator for futudiffu remote observer protocol",
    )
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "remote_target.json"),
                        help="Path to remote_target.json")
    parser.add_argument("--interval", type=int, default=300,
                        help="Aggregation interval in seconds (default: 300)")
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "observations"),
                        help="Local output directory")
    parser.add_argument("--cycles", type=int, default=0,
                        help="Number of cycles (0 = forever)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip rsync, produce empty report")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"error: {config_path} not found", file=sys.stderr)
        sys.exit(1)

    nodes = load_nodes(config_path)
    output_dir = Path(args.output_dir)
    local_obs_dir = output_dir / "pulled"

    print(f"[aggregator] {len(nodes)} node(s), interval={args.interval}s, output={output_dir}")
    for n in nodes:
        print(f"  node {n.node_idx}: {n.host} ({n.n_gpus} GPUs)")

    cycle = 0
    while True:
        cycle += 1
        t0 = time.monotonic()
        print(f"\n[aggregator] cycle {cycle} starting...")

        if args.dry_run:
            pull_results = {n.node_idx: False for n in nodes}
        else:
            pull_results = pull_observations(nodes, local_obs_dir)

        summaries = merge_summaries(local_obs_dir, nodes, pull_results)
        write_unified(output_dir, summaries, cycle)

        elapsed = time.monotonic() - t0
        print(f"[aggregator] cycle {cycle} complete ({elapsed:.1f}s)")

        if args.cycles > 0 and cycle >= args.cycles:
            print(f"[aggregator] completed {cycle} cycles, exiting")
            break

        sleep_time = max(0, args.interval - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()

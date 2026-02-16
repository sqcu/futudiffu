"""Remote SSH orchestrator for futudiffu spot instances.

Local-side CLI that wraps SSH + rsync + remote tmux to manage the full
lifecycle of a rented GPU instance. Never runs on the remote machine itself.

All persistent remote execution goes through tmux sessions dispatched via
short-lived SSH commands. No long-lived SSH pipes.

Config: reads remote_target.json (gitignored) for host/key/paths.
See remote_target.json.example for the schema.

Usage:
    python scripts/remote.py provision
    python scripts/remote.py bootstrap
    python scripts/remote.py models --hf
    python scripts/remote.py validate
    python scripts/remote.py launch
    python scripts/remote.py status
    python scripts/remote.py logs server_0
    python scripts/remote.py patch
    python scripts/remote.py pull
    python scripts/remote.py kill server_0
    python scripts/remote.py teardown
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = REPO_ROOT / "remote_target.json"

PREFIX = "fd_"


@dataclass
class RemoteConfig:
    host: str
    ssh_key: str
    remote_dir: str
    model_dir: str
    base_port: int
    n_gpus: int

    @classmethod
    def load(cls, path: Path = CONFIG_FILE) -> "RemoteConfig":
        if not path.exists():
            print(f"error: {path} not found", file=sys.stderr)
            print(f"  Copy remote_target.json.example and fill in your instance details.",
                  file=sys.stderr)
            sys.exit(1)
        with open(path) as f:
            data = json.load(f)
        return cls(
            host=data["host"],
            ssh_key=os.path.expanduser(data["ssh_key"]),
            remote_dir=data["remote_dir"],
            model_dir=data.get("model_dir", data["remote_dir"] + "/models"),
            base_port=data.get("base_port", 5555),
            n_gpus=data.get("n_gpus", 2),
        )


# ---------------------------------------------------------------------------
# Rsync patterns
# ---------------------------------------------------------------------------

RSYNC_PUSH_EXCLUDE = [
    ".venv/",
    "__pycache__/",
    "*.pyc",
    ".git/",
    "*.safetensors",
    "btrm_dataset/",
    "stream_comfyui/",
    "stream_futudiffu_bf16/",
    "stream_futudiffu_f16te/",
    "stream_compat_bf16/",
    "lora_dumps_test*/",
    "throughput_study*/",
    "validation_renders/",
    "heattest_renders/",
    "bench_renders/",
    "remote_target.json",
    "d20_mcp_debug.log",
    "*.pt",
    "btrm_dataset_v2_heattest/",
    "packed_dataset/",
    "test_split_piref_output/",
    "remote_validation/",
]

RSYNC_PULL_PATTERNS = [
    "remote_validation/",
    "training_output/",
    "*.jsonl",
    "*.json",
    "*.png",
]


# ---------------------------------------------------------------------------
# Transport primitives
# ---------------------------------------------------------------------------

def _ssh_args(config: RemoteConfig) -> list[str]:
    """Base SSH command fragments with key + common options."""
    return [
        "ssh",
        "-i", config.ssh_key,
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
        config.host,
    ]


def _ssh(
    config: RemoteConfig,
    cmd: str,
    timeout: int = 120,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a command on the remote via SSH. Short-lived, not interactive."""
    full_cmd = _ssh_args(config) + [cmd]
    result = subprocess.run(
        full_cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        print(f"  SSH command failed (rc={result.returncode}): {cmd}", file=sys.stderr)
        if result.stderr.strip():
            for line in result.stderr.strip().splitlines()[-5:]:
                print(f"    {line}", file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd,
                                            result.stdout, result.stderr)
    return result


def _rsync(
    config: RemoteConfig,
    src: str,
    dst: str,
    exclude: list[str] | None = None,
    include: list[str] | None = None,
    delete: bool = False,
    timeout: int = 600,
) -> subprocess.CompletedProcess:
    """rsync with SSH key config. src/dst use rsync host: prefix syntax for remote."""
    cmd = [
        "rsync", "-avz", "--progress",
        "-e", f"ssh -i {config.ssh_key} -o StrictHostKeyChecking=accept-new",
    ]
    if delete:
        cmd.append("--delete")
    for pat in (exclude or []):
        cmd.extend(["--exclude", pat])
    for pat in (include or []):
        cmd.extend(["--include", pat])
    cmd.extend([src, dst])

    print(f"  rsync: {src} -> {dst}")
    result = subprocess.run(cmd, timeout=timeout)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, " ".join(cmd))
    return result


# ---------------------------------------------------------------------------
# Remote tmux helpers
# ---------------------------------------------------------------------------

def _remote_tmux_launch(config: RemoteConfig, session_name: str, command: str) -> None:
    """Create a detached tmux session on the remote running `command`.

    Sets up pipe-pane logging to ~/.futudiffu_logs/<session>.log.
    """
    full_name = f"{PREFIX}{session_name}" if not session_name.startswith(PREFIX) else session_name
    log_path = f"~/.futudiffu_logs/{session_name}.log"

    # Ensure log dir exists, kill stale session if any, launch new one
    script = (
        f"mkdir -p ~/.futudiffu_logs && "
        f"tmux kill-session -t {full_name} 2>/dev/null; "
        f"tmux new-session -d -s {full_name} '{command}' && "
        f"tmux pipe-pane -t {full_name} -o 'cat >> {log_path}'"
    )
    _ssh(config, script, timeout=30)
    print(f"  launched remote session '{session_name}'")
    print(f"    command: {command}")
    print(f"    log: {log_path}")


def _remote_tmux_list(config: RemoteConfig) -> list[tuple[str, str]]:
    """List fd_* tmux sessions on the remote. Returns [(name, status)]."""
    try:
        result = _ssh(
            config,
            "tmux list-sessions -F '#{session_name}\t#{session_activity}' 2>/dev/null || true",
            check=False,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        return []

    sessions = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t", 1)
        if parts and parts[0].startswith(PREFIX):
            name = parts[0][len(PREFIX):]
            activity = parts[1] if len(parts) > 1 else "?"
            sessions.append((name, activity))
    return sessions


def _remote_tmux_kill(config: RemoteConfig, session_name: str) -> None:
    """Kill a remote tmux session."""
    full_name = f"{PREFIX}{session_name}"
    _ssh(config, f"tmux kill-session -t {full_name}", timeout=15)
    print(f"  killed remote session '{session_name}'")


def _remote_tmux_log_tail(config: RemoteConfig, session_name: str, lines: int = 50) -> str:
    """Read the last N lines of a remote tmux session's pipe-pane log."""
    log_path = f"~/.futudiffu_logs/{session_name}.log"
    result = _ssh(config, f"tail -n {lines} {log_path} 2>/dev/null || echo '(no log yet)'",
                  check=False, timeout=15)
    return result.stdout


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_provision(args: argparse.Namespace) -> None:
    """Git clone (or pull) on remote, rsync reference data."""
    config = RemoteConfig.load()

    # Check if repo already exists on remote
    result = _ssh(config, f"test -d {config.remote_dir}/.git && echo yes || echo no",
                  check=False, timeout=15)
    repo_exists = result.stdout.strip() == "yes"

    if repo_exists:
        print("  Remote repo exists, pulling latest...")
        _ssh(config, f"cd {config.remote_dir} && git pull --ff-only", timeout=60)
    else:
        print("  Cloning repo on remote...")
        # Get the remote URL from local git
        local_remote = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        if local_remote.returncode != 0:
            print("  error: cannot determine git remote URL", file=sys.stderr)
            sys.exit(1)
        remote_url = local_remote.stdout.strip()
        _ssh(config, f"git clone {remote_url} {config.remote_dir}", timeout=300)

    # Rsync code (covers uncommitted changes)
    print("\n  Syncing code...")
    _rsync(
        config,
        src=str(REPO_ROOT) + "/",
        dst=f"{config.host}:{config.remote_dir}/",
        exclude=RSYNC_PUSH_EXCLUDE,
    )

    # Rsync reference data
    ref_dir = REPO_ROOT / "stream_futudiffu"
    if ref_dir.exists():
        print("\n  Syncing reference trajectory (stream_futudiffu/)...")
        _rsync(
            config,
            src=str(ref_dir) + "/",
            dst=f"{config.host}:{config.remote_dir}/stream_futudiffu/",
        )

    btrm_dir = REPO_ROOT / "btrm_dataset"
    if btrm_dir.exists():
        print("\n  Syncing BTRM dataset...")
        _rsync(
            config,
            src=str(btrm_dir) + "/",
            dst=f"{config.host}:{config.remote_dir}/btrm_dataset/",
        )

    print("\n  Provision complete.")


def cmd_sync(args: argparse.Namespace) -> None:
    """Rsync code changes from local to remote (fast delta)."""
    config = RemoteConfig.load()
    print("  Syncing code to remote...")
    _rsync(
        config,
        src=str(REPO_ROOT) + "/",
        dst=f"{config.host}:{config.remote_dir}/",
        exclude=RSYNC_PUSH_EXCLUDE,
    )
    print("  Sync complete.")


def cmd_bootstrap(args: argparse.Namespace) -> None:
    """Rsync code + run remote_uv_bootstrap.py on remote."""
    config = RemoteConfig.load()

    # Sync code first (bootstrap script needs to be there)
    print("  Syncing code...")
    _rsync(
        config,
        src=str(REPO_ROOT) + "/",
        dst=f"{config.host}:{config.remote_dir}/",
        exclude=RSYNC_PUSH_EXCLUDE,
    )

    # Find python on remote
    python = _find_remote_python(config)

    print(f"\n  Running bootstrap on remote (python: {python})...")
    result = _ssh(
        config,
        f"cd {config.remote_dir} && {python} scripts/remote_uv_bootstrap.py",
        timeout=600,
        check=False,
    )
    print(result.stdout)
    if result.stderr.strip():
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"  Bootstrap FAILED (rc={result.returncode})", file=sys.stderr)
        sys.exit(1)

    print("  Bootstrap complete.")


def _find_remote_python(config: RemoteConfig) -> str:
    """Find a working Python on the remote. Prefers venv, falls back to system."""
    candidates = [
        f"{config.remote_dir}/.venv/bin/python",
        "python3",
        "python",
    ]
    for candidate in candidates:
        result = _ssh(
            config,
            f"{candidate} --version 2>/dev/null && echo __ok__",
            check=False, timeout=10,
        )
        if "__ok__" in result.stdout:
            return candidate
    # Default fallback
    return "python3"


def cmd_models(args: argparse.Namespace) -> None:
    """Download/transfer models to remote."""
    config = RemoteConfig.load()

    if args.rsync:
        # Push pre-quantized models from local
        print("  Pushing models via rsync...")
        # Find local model files
        local_models = REPO_ROOT / "models"
        if not local_models.exists():
            # Try paths4claude for model locations
            print("  error: no local models/ directory found", file=sys.stderr)
            print("  Use --hf to download on remote instead, or create models/ with symlinks.",
                  file=sys.stderr)
            sys.exit(1)

        _ssh(config, f"mkdir -p {config.model_dir}", timeout=15)
        _rsync(
            config,
            src=str(local_models) + "/",
            dst=f"{config.host}:{config.model_dir}/",
            timeout=3600,  # large files
        )
        print("  Model push complete.")

    elif args.hf:
        # Run remote_node_bootstrap.py on remote (HF download + quantize)
        python = _find_remote_python(config)
        cmd = (
            f"cd {config.remote_dir} && "
            f"PYTHONPATH={config.remote_dir}/src "
            f"{python} scripts/remote_node_bootstrap.py "
            f"--output-dir {config.model_dir}"
        )

        print(f"  Launching model download on remote via tmux...")
        _remote_tmux_launch(config, "model_download", cmd)
        print("  Monitor with: python scripts/remote.py logs model_download")

        if args.wait:
            print("  Waiting for download to complete...")
            _wait_for_session_exit(config, "model_download", timeout=3600)
    else:
        print("  error: specify --hf (download on remote) or --rsync (push from local)",
              file=sys.stderr)
        sys.exit(1)


def _wait_for_session_exit(
    config: RemoteConfig,
    session_name: str,
    timeout: int = 3600,
    poll_interval: float = 10.0,
) -> bool:
    """Poll until a remote tmux session no longer exists (i.e. command finished)."""
    full_name = f"{PREFIX}{session_name}"
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        result = _ssh(
            config,
            f"tmux has-session -t {full_name} 2>/dev/null && echo alive || echo done",
            check=False, timeout=15,
        )
        if "done" in result.stdout:
            # Print final log lines
            log = _remote_tmux_log_tail(config, session_name, lines=20)
            if log.strip():
                print(log)
            return True
        time.sleep(poll_interval)

    print(f"  Timeout waiting for session '{session_name}' ({timeout}s)", file=sys.stderr)
    return False


def cmd_validate(args: argparse.Namespace) -> None:
    """Run launch_remote.py on remote via tmux, optionally poll for completion."""
    config = RemoteConfig.load()
    python = _find_remote_python(config)

    extra_args = ""
    if args.quick:
        extra_args += " --quick"
    if args.skip_download:
        extra_args += " --skip-download"

    cmd = (
        f"cd {config.remote_dir} && "
        f"PYTHONPATH={config.remote_dir}/src "
        f"{python} scripts/launch_remote.py "
        f"--model-dir {config.model_dir} "
        f"--n-gpus {config.n_gpus} "
        f"--base-port {config.base_port}"
        f"{extra_args}"
    )

    print("  Launching validation on remote...")
    _remote_tmux_launch(config, "validate", cmd)
    print("  Monitor with: python scripts/remote.py logs validate")

    if args.wait:
        print("  Waiting for validation to complete...")
        _wait_for_session_exit(config, "validate", timeout=1800)


def cmd_launch(args: argparse.Namespace) -> None:
    """Launch inference server(s) on remote via tmux."""
    config = RemoteConfig.load()
    python = _find_remote_python(config)

    for i in range(config.n_gpus):
        port = config.base_port + i
        session_name = f"server_{i}"

        cmd = (
            f"cd {config.remote_dir} && "
            f"CUDA_VISIBLE_DEVICES={i} "
            f"PYTHONPATH={config.remote_dir}/src "
            f"{python} -m futudiffu.server "
            f"--port {port} "
            f"--fp8-diff {config.model_dir}/z_image_fp8_blockwise.safetensors "
            f"--te {config.model_dir}/qwen_3_4b.safetensors "
            f"--vae {config.model_dir}/ae.safetensors"
        )
        _remote_tmux_launch(config, session_name, cmd)

    # Wait for ports
    if args.wait:
        print("\n  Waiting for servers to become ready...")
        for i in range(config.n_gpus):
            port = config.base_port + i
            print(f"  Checking port {port} on remote...", end="", flush=True)
            ok = _wait_for_remote_port(config, port, timeout=180)
            print(" ready" if ok else " TIMEOUT")
            if not ok:
                print(f"  Check logs: python scripts/remote.py logs server_{i}",
                      file=sys.stderr)

    print(f"\n  {config.n_gpus} server(s) launched.")
    print(f"  Monitor with: python scripts/remote.py status")


def _wait_for_remote_port(
    config: RemoteConfig,
    port: int,
    timeout: float = 180.0,
    poll_interval: float = 3.0,
) -> bool:
    """Poll until a port is open on the remote machine."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = _ssh(
            config,
            f"python3 -c \"import socket; s=socket.socket(); s.settimeout(1); "
            f"s.connect(('127.0.0.1', {port})); s.close(); print('open')\" "
            f"2>/dev/null || echo closed",
            check=False, timeout=15,
        )
        if "open" in result.stdout:
            return True
        time.sleep(poll_interval)
    return False


def cmd_patch(args: argparse.Namespace) -> None:
    """Rsync code delta only (no git operations). For hot-fixing defects."""
    config = RemoteConfig.load()
    print("  Patching remote code (rsync delta)...")
    _rsync(
        config,
        src=str(REPO_ROOT) + "/",
        dst=f"{config.host}:{config.remote_dir}/",
        exclude=RSYNC_PUSH_EXCLUDE,
    )
    print("  Patch complete. Restart affected sessions to pick up changes.")


def cmd_status(args: argparse.Namespace) -> None:
    """List remote tmux sessions + tail recent log lines."""
    config = RemoteConfig.load()
    sessions = _remote_tmux_list(config)

    if not sessions:
        print("  No active futudiffu sessions on remote.")
        return

    print(f"  {'Session':<25} {'Last Activity'}")
    print(f"  {'-' * 50}")
    for name, activity in sessions:
        print(f"  {name:<25} {activity}")

    # Tail last few lines of each session's log
    if args.verbose:
        for name, _ in sessions:
            print(f"\n  --- {name} (last 10 lines) ---")
            log = _remote_tmux_log_tail(config, name, lines=10)
            for line in log.strip().splitlines():
                print(f"    {line}")


def cmd_logs(args: argparse.Namespace) -> None:
    """Print or tail a specific remote tmux session's log."""
    config = RemoteConfig.load()
    lines = args.lines or 50

    if args.follow:
        # Stream via ssh tail -f (this IS a long-lived SSH pipe, but it's user-initiated)
        log_path = f"~/.futudiffu_logs/{args.session_name}.log"
        ssh_cmd = _ssh_args(config) + [f"tail -f -n {lines} {log_path}"]
        print(f"  Following {args.session_name} (Ctrl+C to stop)...")
        try:
            subprocess.run(ssh_cmd)
        except KeyboardInterrupt:
            print()
    else:
        log = _remote_tmux_log_tail(config, args.session_name, lines=lines)
        print(log)


def cmd_pull(args: argparse.Namespace) -> None:
    """Rsync training outputs / renders from remote to local."""
    config = RemoteConfig.load()
    local_dest = REPO_ROOT / "remote_output"
    local_dest.mkdir(parents=True, exist_ok=True)

    # Pull everything interesting (no --delete: append-only)
    include_args = []
    for pat in RSYNC_PULL_PATTERNS:
        include_args.extend(["--include", pat])

    # Include directories needed for recursion, exclude the rest
    cmd = [
        "rsync", "-avz", "--progress",
        "-e", f"ssh -i {config.ssh_key} -o StrictHostKeyChecking=accept-new",
        "--include", "*/",  # recurse into dirs
    ]
    for pat in RSYNC_PULL_PATTERNS:
        cmd.extend(["--include", pat])
    cmd.extend([
        "--exclude", "*",  # exclude everything else
        f"{config.host}:{config.remote_dir}/",
        str(local_dest) + "/",
    ])

    print(f"  Pulling outputs to {local_dest}/")
    subprocess.run(cmd, timeout=3600)
    print("  Pull complete.")


def cmd_kill(args: argparse.Namespace) -> None:
    """Kill a specific remote tmux session."""
    config = RemoteConfig.load()

    if not args.force:
        ans = input(f"  Kill remote session '{args.session_name}'? [y/N] ")
        if ans.strip().lower() != "y":
            print("  Aborted.")
            return

    try:
        _remote_tmux_kill(config, args.session_name)
    except subprocess.CalledProcessError:
        print(f"  Session '{args.session_name}' does not exist or already dead.")


def cmd_upload(args: argparse.Namespace) -> None:
    """Launch upload_to_hf.py in watch mode on remote (tee to HF)."""
    config = RemoteConfig.load()
    python = _find_remote_python(config)

    source_dir = args.source_dir or f"{config.remote_dir}/training_output"
    repo_id = args.repo_id or "SQCU/futudiffu-btrm"
    interval = args.interval or 60

    cmd = (
        f"cd {config.remote_dir} && "
        f"PYTHONPATH={config.remote_dir}/src "
        f"{python} scripts/upload_to_hf.py "
        f"--repo-id {repo_id} "
        f"--source-dir {source_dir} "
        f"--watch --interval {interval}"
    )
    _remote_tmux_launch(config, "hf_upload", cmd)


def cmd_teardown(args: argparse.Namespace) -> None:
    """Kill all fd_ sessions, optionally rsync final outputs back."""
    config = RemoteConfig.load()

    sessions = _remote_tmux_list(config)
    if not sessions:
        print("  No active sessions to tear down.")
    else:
        if not args.force:
            print(f"  Active sessions: {', '.join(name for name, _ in sessions)}")
            ans = input(f"  Kill all {len(sessions)} session(s)? [y/N] ")
            if ans.strip().lower() != "y":
                print("  Aborted.")
                return

        for name, _ in sessions:
            try:
                _remote_tmux_kill(config, name)
            except subprocess.CalledProcessError:
                print(f"  warning: failed to kill '{name}'")

    if args.pull:
        print("\n  Pulling final outputs...")
        cmd_pull(argparse.Namespace())

    print("  Teardown complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remote SSH orchestrator for futudiffu spot instances",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # provision
    p = sub.add_parser("provision", help="Git clone + rsync reference data to remote")
    p.set_defaults(func=cmd_provision)

    # sync
    p = sub.add_parser("sync", help="Rsync code changes to remote (fast delta)")
    p.set_defaults(func=cmd_sync)

    # bootstrap
    p = sub.add_parser("bootstrap", help="Install uv + deps on remote")
    p.set_defaults(func=cmd_bootstrap)

    # models
    p = sub.add_parser("models", help="Transfer models to remote")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--hf", action="store_true",
                   help="Download from HuggingFace on remote (+ quantize)")
    g.add_argument("--rsync", action="store_true",
                   help="Push pre-quantized models from local via rsync")
    p.add_argument("--wait", action="store_true",
                   help="Wait for download/transfer to complete")
    p.set_defaults(func=cmd_models)

    # validate
    p = sub.add_parser("validate", help="Run launch_remote.py validation on remote")
    p.add_argument("--quick", action="store_true",
                   help="Pass --quick to launch_remote.py")
    p.add_argument("--skip-download", action="store_true",
                   help="Pass --skip-download to launch_remote.py")
    p.add_argument("--wait", action="store_true",
                   help="Wait for validation to complete")
    p.set_defaults(func=cmd_validate)

    # launch
    p = sub.add_parser("launch", help="Launch inference server(s) on remote")
    p.add_argument("--wait", action="store_true",
                   help="Wait for servers to become ready")
    p.set_defaults(func=cmd_launch)

    # patch
    p = sub.add_parser("patch", help="Rsync code delta only (hot-fix)")
    p.set_defaults(func=cmd_patch)

    # status
    p = sub.add_parser("status", help="List remote tmux sessions")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Also tail recent log lines for each session")
    p.set_defaults(func=cmd_status)

    # logs
    p = sub.add_parser("logs", help="View a remote session's log")
    p.add_argument("session_name", help="Session name (without fd_ prefix)")
    p.add_argument("-n", "--lines", type=int, default=50,
                   help="Number of lines to show (default: 50)")
    p.add_argument("-f", "--follow", action="store_true",
                   help="Follow log output (like tail -f)")
    p.set_defaults(func=cmd_logs)

    # upload
    p = sub.add_parser("upload", help="Launch upload_to_hf.py in watch mode on remote")
    p.add_argument("--repo-id", type=str, default=None,
                   help="HuggingFace repo ID (default: SQCU/futudiffu-btrm)")
    p.add_argument("--source-dir", type=str, default=None,
                   help="Remote directory to watch (default: {remote_dir}/training_output)")
    p.add_argument("--interval", type=int, default=None,
                   help="Poll interval in seconds (default: 60)")
    p.set_defaults(func=cmd_upload)

    # pull
    p = sub.add_parser("pull", help="Rsync training outputs from remote to local")
    p.set_defaults(func=cmd_pull)

    # kill
    p = sub.add_parser("kill", help="Kill a remote tmux session")
    p.add_argument("session_name", help="Session name (without fd_ prefix)")
    p.add_argument("-f", "--force", action="store_true",
                   help="Skip confirmation")
    p.set_defaults(func=cmd_kill)

    # teardown
    p = sub.add_parser("teardown", help="Kill all sessions, optionally pull outputs")
    p.add_argument("-f", "--force", action="store_true",
                   help="Skip confirmation")
    p.add_argument("--pull", action="store_true",
                   help="Pull final outputs before teardown")
    p.set_defaults(func=cmd_teardown)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

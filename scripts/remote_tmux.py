#!/usr/bin/env python3
"""Tmux session management for persistent remote training runs.

Manages named tmux sessions with automatic logging, read-only viewing,
and a convenience launcher for the standard server/train/sync trio.

All session names are prefixed with fd_ to avoid collision with other
tmux sessions on the same machine.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PREFIX = "fd_"
LOG_DIR = Path.home() / ".futudiffu_logs"


def _prefixed(name: str) -> str:
    return f"{PREFIX}{name}" if not name.startswith(PREFIX) else name


def _unprefixed(name: str) -> str:
    return name[len(PREFIX):] if name.startswith(PREFIX) else name


def _check_tmux():
    if shutil.which("tmux") is None:
        print("error: tmux is not installed or not on PATH", file=sys.stderr)
        sys.exit(1)


def _session_exists(session: str) -> bool:
    r = subprocess.run(
        ["tmux", "has-session", "-t", session],
        capture_output=True,
    )
    return r.returncode == 0


def _list_sessions():
    """Return list of (session_name, created_epoch, command) for fd_ sessions."""
    r = subprocess.run(
        ["tmux", "list-sessions", "-F",
         "#{session_name}\t#{session_created}\t#{pane_current_command}"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return []
    sessions = []
    for line in r.stdout.strip().splitlines():
        parts = line.split("\t", 2)
        if len(parts) >= 2 and parts[0].startswith(PREFIX):
            name = parts[0]
            created = int(parts[1]) if parts[1].isdigit() else 0
            cmd = parts[2] if len(parts) > 2 else "?"
            sessions.append((name, created, cmd))
    return sessions


def _print_sessions(sessions):
    if not sessions:
        print("No active futudiffu sessions.")
        return
    now = time.time()
    print(f"{'Name':<20} {'Uptime':<14} {'Command'}")
    print("-" * 60)
    for name, created, cmd in sessions:
        elapsed = int(now - created)
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        uptime = f"{h}h{m:02d}m{s:02d}s"
        print(f"{_unprefixed(name):<20} {uptime:<14} {cmd}")


def _log_path(session: str) -> Path:
    return LOG_DIR / f"{_unprefixed(session)}.log"


def cmd_launch(args):
    _check_tmux()
    session = _prefixed(args.session_name)
    if _session_exists(session):
        print(f"warning: session '{_unprefixed(session)}' already exists, not clobbering", file=sys.stderr)
        sys.exit(1)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = _log_path(session)

    command = " ".join(args.command)
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, command],
        check=True,
    )
    subprocess.run(
        ["tmux", "pipe-pane", "-t", session, "-o", f"cat >> {log}"],
        check=True,
    )
    print(f"launched session '{_unprefixed(session)}'")
    print(f"  command: {command}")
    print(f"  logfile: {log}")


def cmd_attach(args):
    _check_tmux()
    session = _prefixed(args.session_name)
    if not _session_exists(session):
        print(f"error: session '{_unprefixed(session)}' does not exist", file=sys.stderr)
        _print_sessions(_list_sessions())
        sys.exit(1)
    os.execvp("tmux", ["tmux", "attach-session", "-t", session])


def cmd_watch(args):
    _check_tmux()
    session = _prefixed(args.session_name)
    if not _session_exists(session):
        print(f"error: session '{_unprefixed(session)}' does not exist", file=sys.stderr)
        _print_sessions(_list_sessions())
        sys.exit(1)
    os.execvp("tmux", ["tmux", "attach-session", "-r", "-t", session])


def cmd_list(_args):
    _check_tmux()
    _print_sessions(_list_sessions())


def cmd_logs(args):
    log = _log_path(_prefixed(args.session_name))
    if not log.exists():
        print(f"error: no logfile at {log}", file=sys.stderr)
        sys.exit(1)
    os.execvp("tail", ["tail", "-f", str(log)])


def cmd_kill(args):
    _check_tmux()
    session = _prefixed(args.session_name)
    if not _session_exists(session):
        print(f"error: session '{_unprefixed(session)}' does not exist", file=sys.stderr)
        sys.exit(1)
    if not args.force:
        ans = input(f"Kill session '{_unprefixed(session)}'? [y/N] ")
        if ans.strip().lower() != "y":
            print("aborted")
            return
    subprocess.run(["tmux", "kill-session", "-t", session], check=True)
    print(f"killed session '{_unprefixed(session)}'")


def _wait_for_port(port: int, timeout: float = 60.0) -> bool:
    """Poll until a TCP port is accepting connections."""
    import socket
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(1)
    return False


def cmd_setup(args):
    _check_tmux()

    # Check no sessions already running
    for name in ["server", "train", "sync"]:
        if _session_exists(_prefixed(name)):
            print(f"error: session '{name}' already exists. kill it first or use attach/watch.", file=sys.stderr)
            sys.exit(1)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Launch server
    print(f"launching server: {args.server_args}")
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", _prefixed("server"), args.server_args],
        check=True,
    )
    subprocess.run(
        ["tmux", "pipe-pane", "-t", _prefixed("server"), "-o",
         f"cat >> {_log_path(_prefixed('server'))}"],
        check=True,
    )

    # Wait for server port
    port = args.port
    print(f"waiting for server on port {port}...", end="", flush=True)
    if not _wait_for_port(port, timeout=args.timeout):
        print(" TIMEOUT")
        print(f"error: server did not open port {port} within {args.timeout}s", file=sys.stderr)
        print("check logs: remote_tmux.py logs server", file=sys.stderr)
        sys.exit(1)
    print(" ready")

    # Launch train
    print(f"launching train: {args.train_args}")
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", _prefixed("train"), args.train_args],
        check=True,
    )
    subprocess.run(
        ["tmux", "pipe-pane", "-t", _prefixed("train"), "-o",
         f"cat >> {_log_path(_prefixed('train'))}"],
        check=True,
    )

    # Launch sync
    if args.sync_cmd:
        print(f"launching sync: {args.sync_cmd}")
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", _prefixed("sync"), args.sync_cmd],
            check=True,
        )
        subprocess.run(
            ["tmux", "pipe-pane", "-t", _prefixed("sync"), "-o",
             f"cat >> {_log_path(_prefixed('sync'))}"],
            check=True,
        )

    print()
    _print_sessions(_list_sessions())


def main():
    parser = argparse.ArgumentParser(
        description="Tmux session manager for futudiffu remote training runs",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # launch
    p = sub.add_parser("launch", help="Create a new session running a command")
    p.add_argument("session_name")
    p.add_argument("command", nargs="+")
    p.set_defaults(func=cmd_launch)

    # attach
    p = sub.add_parser("attach", help="Attach to a session (full control)")
    p.add_argument("session_name")
    p.set_defaults(func=cmd_attach)

    # watch
    p = sub.add_parser("watch", help="Attach read-only (observers / Claudes)")
    p.add_argument("session_name")
    p.set_defaults(func=cmd_watch)

    # list
    p = sub.add_parser("list", help="List active futudiffu sessions")
    p.set_defaults(func=cmd_list)

    # logs
    p = sub.add_parser("logs", help="Tail the logfile for a session")
    p.add_argument("session_name")
    p.set_defaults(func=cmd_logs)

    # kill
    p = sub.add_parser("kill", help="Kill a session")
    p.add_argument("session_name")
    p.add_argument("--force", "-f", action="store_true", help="Skip confirmation")
    p.set_defaults(func=cmd_kill)

    # setup
    p = sub.add_parser("setup", help="Launch server + train + sync in one go")
    p.add_argument("--server-args", required=True,
                   help="Full command for the server session")
    p.add_argument("--train-args", required=True,
                   help="Full command for the train session")
    p.add_argument("--sync-cmd", default=None,
                   help="Full command for the sync session (optional)")
    p.add_argument("--port", type=int, default=5555,
                   help="TCP port to poll for server readiness (default: 5555)")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="Seconds to wait for server port (default: 120)")
    p.set_defaults(func=cmd_setup)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

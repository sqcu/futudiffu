"""Kill existing tensor-server, launch a new one, verify health.

Atomic restart script. Always does ALL three steps:
  1. Kill any existing server on the target port
  2. Launch a new server with stdout/stderr captured to a log file
  3. Poll /status until the server responds (or timeout)

Log files are written to training_output/server.log (rotated on restart).

Usage:
    python restart_server.py                     # restart on port 8000
    python restart_server.py --port 8000         # explicit port
    python restart_server.py --timeout 120       # wait up to 120s for health
    python restart_server.py --skip-health       # launch only, don't wait
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_EXE = os.path.join(REPO_ROOT, ".venv", "Scripts", "python.exe")
LAUNCH_SCRIPT = os.path.join(REPO_ROOT, "scripts_ii", "launch_server.py")
LOG_DIR = os.path.join(REPO_ROOT, "training_output")
LOG_FILE = os.path.join(LOG_DIR, "server.log")
LOG_PREV = os.path.join(LOG_DIR, "server.prev.log")


def find_server_pids(port: int) -> list[int]:
    """Find PIDs of processes listening on the given port.

    Works in WSL2 by querying Windows netstat via interop.
    Falls back to Linux ss/lsof if Windows interop is unavailable.
    """
    pids = set()

    # Try Windows netstat (works in WSL2 with interop)
    try:
        out = subprocess.check_output(
            ["netstat.exe", "-ano"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        for line in out.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                if parts:
                    try:
                        pids.add(int(parts[-1]))
                    except ValueError:
                        pass
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        pass

    # Also try Linux ss (for native Linux or if interop doesn't find it)
    try:
        out = subprocess.check_output(
            ["ss", "-tlnp", f"sport = :{port}"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        for line in out.splitlines():
            if "pid=" in line:
                import re
                for m in re.finditer(r"pid=(\d+)", line):
                    pids.add(int(m.group(1)))
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        pass

    return sorted(pids)


def kill_server(port: int) -> bool:
    """Kill any server process on the given port. Returns True if something was killed."""
    pids = find_server_pids(port)
    if not pids:
        print(f"  No process found on port {port}")
        return False

    for pid in pids:
        print(f"  Killing PID {pid} on port {port}")
        # Try Windows taskkill first (for Windows Python processes seen from WSL)
        try:
            subprocess.run(
                ["taskkill.exe", "/F", "/PID", str(pid)],
                capture_output=True, timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Fall back to Linux kill
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass

    # Wait for port to be released
    for _ in range(20):
        if not find_server_pids(port):
            return True
        time.sleep(0.5)

    remaining = find_server_pids(port)
    if remaining:
        print(f"  WARNING: PIDs {remaining} still on port {port} after kill", file=sys.stderr)
        return False
    return True


def launch_server(port: int) -> subprocess.Popen:
    """Launch server as a background process with log capture."""
    os.makedirs(LOG_DIR, exist_ok=True)

    # Rotate previous log
    if os.path.exists(LOG_FILE):
        shutil.move(LOG_FILE, LOG_PREV)

    log_fh = open(LOG_FILE, "w")
    win_script = LAUNCH_SCRIPT.replace("/mnt/f/", "F:\\").replace("/", "\\")

    proc = subprocess.Popen(
        [PYTHON_EXE, win_script, "--port", str(port)],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        cwd=REPO_ROOT,
    )
    print(f"  Launched server PID={proc.pid}, log={LOG_FILE}")
    return proc


def check_health(port: int, timeout: float) -> bool:
    """Poll /status until server responds or timeout."""
    import urllib.request
    import urllib.error

    url = f"http://localhost:{port}/status"
    t0 = time.monotonic()
    attempt = 0

    while time.monotonic() - t0 < timeout:
        attempt += 1
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    elapsed = time.monotonic() - t0
                    print(f"  Server healthy after {elapsed:.1f}s ({attempt} attempts)")
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, OSError, TimeoutError):
            pass
        time.sleep(2)

    print(f"  TIMEOUT: server not healthy after {timeout:.0f}s", file=sys.stderr)
    # Dump last 20 lines of log for diagnostics
    if os.path.exists(LOG_FILE):
        print("  --- Last 20 lines of server.log ---", file=sys.stderr)
        with open(LOG_FILE) as f:
            lines = f.readlines()
            for line in lines[-20:]:
                print(f"  | {line.rstrip()}", file=sys.stderr)
    return False


def main():
    parser = argparse.ArgumentParser(description="Restart tensor-server atomically")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default 8000)")
    parser.add_argument("--timeout", type=float, default=90, help="Health check timeout in seconds")
    parser.add_argument("--skip-health", action="store_true", help="Skip health check")
    args = parser.parse_args()

    print(f"[1/3] Killing existing server on port {args.port}...")
    kill_server(args.port)

    print(f"[2/3] Launching new server on port {args.port}...")
    proc = launch_server(args.port)

    if args.skip_health:
        print(f"[3/3] Skipped health check (--skip-health)")
        print(f"\nServer PID={proc.pid}, log={LOG_FILE}")
        return

    print(f"[3/3] Waiting for server health (timeout={args.timeout:.0f}s)...")
    time.sleep(3)  # Brief delay for uvicorn startup
    if check_health(args.port, args.timeout):
        print(f"\nServer ready on http://localhost:{args.port}")
    else:
        print(f"\nServer failed to start. Check {LOG_FILE}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

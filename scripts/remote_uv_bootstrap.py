"""Bootstrap uv and install dependencies on a remote node.

This script runs ON THE REMOTE MACHINE. It is rsynced there by remote.py.
It installs uv (if needed), syncs dependencies, and validates the environment.

Usage (called by remote.py, not directly):
    python scripts/remote_uv_bootstrap.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], label: str, check: bool = True) -> subprocess.CompletedProcess:
    print(f"  [{label}] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  FAIL: {label}", file=sys.stderr)
        if result.stdout.strip():
            print(result.stdout[-1000:], file=sys.stderr)
        if result.stderr.strip():
            print(result.stderr[-1000:], file=sys.stderr)
        sys.exit(1)
    return result


def install_uv() -> str:
    """Ensure uv is available. Returns path to uv binary."""
    uv_path = shutil.which("uv")
    if uv_path:
        print(f"  uv found: {uv_path}")
        return uv_path

    print("  uv not found, installing via pip...")
    _run([sys.executable, "-m", "pip", "install", "uv"], "pip install uv")

    uv_path = shutil.which("uv")
    if uv_path:
        print(f"  uv installed: {uv_path}")
        return uv_path

    # pip may have installed to a user bin not on PATH
    user_bin = Path.home() / ".local" / "bin" / "uv"
    if user_bin.exists():
        print(f"  uv installed at: {user_bin}")
        return str(user_bin)

    print("  FAIL: uv not found after install", file=sys.stderr)
    sys.exit(1)


def sync_deps(uv: str) -> None:
    """Run uv sync to install all dependencies."""
    pyproject = REPO_ROOT / "pyproject.toml"
    if not pyproject.exists():
        print(f"  FAIL: pyproject.toml not found at {pyproject}", file=sys.stderr)
        sys.exit(1)

    # Detect platform for triton extras
    import platform
    extra = "triton-linux" if platform.system() == "Linux" else "triton-windows"

    _run(
        [uv, "sync", "--extra", extra],
        f"uv sync --extra {extra}",
    )


def validate_imports() -> None:
    """Verify critical imports work."""
    critical = ["torch", "triton", "safetensors", "zmq", "pyarrow"]
    failed = []

    for mod in critical:
        try:
            __import__(mod)
            print(f"  {mod}: OK")
        except ImportError as e:
            print(f"  {mod}: FAIL ({e})")
            failed.append(mod)

    if failed:
        print(f"\n  FAIL: Missing imports: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)


def print_cuda_info() -> None:
    """Print GPU info for verification."""
    import torch

    if not torch.cuda.is_available():
        print("  CUDA: not available")
        return

    n = torch.cuda.device_count()
    print(f"  CUDA devices: {n}")

    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"    GPU {i}: {props.name} (SM {props.major}.{props.minor}, {vram_gb:.1f} GB)")

    print(f"  torch: {torch.__version__}")
    print(f"  CUDA runtime: {torch.version.cuda}")


def main() -> int:
    print("=" * 50)
    print("futudiffu remote bootstrap")
    print("=" * 50)
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Repo root: {REPO_ROOT}")
    print()

    # Step 1: uv
    print("--- Step 1: Install uv ---")
    uv = install_uv()
    print()

    # Step 2: sync
    print("--- Step 2: Sync dependencies ---")
    os.chdir(REPO_ROOT)
    sync_deps(uv)
    print()

    # Step 3: validate imports
    print("--- Step 3: Validate imports ---")
    validate_imports()
    print()

    # Step 4: CUDA info
    print("--- Step 4: CUDA info ---")
    print_cuda_info()
    print()

    print("Bootstrap complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

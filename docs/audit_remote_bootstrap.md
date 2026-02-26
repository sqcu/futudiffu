# Audit: Remote Bootstrap and Environment Setup

**Date:** 2026-02-25
**Scope:** Remote environment initialization, dependency installation, model transfer.
**Files examined:** `scripts/remote_uv_bootstrap.py`, `scripts/remote_node_bootstrap.py`, `scripts/remote.py`, `scripts/launch_remote.py`, `docs/remote_deployment.md`, `docs/case_study_2xh100_20260216.md`, `remote_target.json.example`

---

## Executive Summary

The bootstrap is split across two scripts with confusingly similar names: `remote_uv_bootstrap.py` (installs uv and syncs Python dependencies) and `remote_node_bootstrap.py` (downloads model weights from HuggingFace and optionally quantizes them). The uv bootstrap is reasonably well-structured but has no error recovery for partial installations. The node bootstrap has skip-if-present logic for downloads but unconditionally re-quantizes existing FP8 models. Neither bootstrap verifies CUDA driver compatibility, sets up system-level dependencies (like `tmux` itself), or handles the case where the remote machine has a different Linux distribution or Python version than expected. The `remote.py` bootstrap command runs `remote_uv_bootstrap.py` via SSH with a 600-second timeout, which may be insufficient for initial `uv sync` on a cold VM.

**Critical Issues: 1 | Major Issues: 4 | Minor Issues: 3**

---

## Detailed Findings

### 1. No system-level dependency verification (Critical)

`remote_uv_bootstrap.py` installs Python dependencies via `uv sync` but does not verify or install system-level prerequisites:

- `tmux` (required by all session management)
- CUDA drivers and toolkit (required for GPU operation)
- `rsync` (required for file synchronization)
- `git` (required for provisioning)
- system build tools (may be needed for Triton compilation)

The script validates imports (`torch`, `triton`, `safetensors`, `zmq`, `pyarrow`) after installation, and prints CUDA info, but if `tmux` is not installed, the entire remote workflow breaks silently (the first SSH command that tries to create a tmux session fails).

`remote_tmux.py` has a `_check_tmux()` function that verifies tmux is on PATH, but `remote.py` dispatches tmux commands directly via SSH without any such check.

### 2. Bootstrap is not fully idempotent (Major)

`remote_uv_bootstrap.py` is mostly idempotent -- `install_uv()` checks `shutil.which("uv")` before installing, and `uv sync` is inherently idempotent. However:

- `sync_deps()` calls `os.chdir(REPO_ROOT)` (line 128), which is a global side effect
- `validate_imports()` imports modules into the current process, which can have side effects (e.g., torch initializing CUDA on import)
- If `uv sync` partially succeeds and then fails (e.g., a package fails to compile), re-running the script will attempt the full sync again, but `uv`'s lockfile may be in an inconsistent state

`remote_node_bootstrap.py` has a more significant idempotency issue: if the FP8 model already exists, it re-quantizes anyway:

> ```python
> # scripts/remote_node_bootstrap.py lines 489-493
> if fp8_path.exists():
>     size_gb = fp8_path.stat().st_size / (1024 ** 3)
>     print(f"  FP8 model already exists ({size_gb:.2f} GB), re-quantizing...")
> n_quant = quantize_diffusion_model(bf16_path, fp8_path, block_size=args.block_size)
> ```

Re-quantization is a GPU-intensive operation that takes several minutes. This wastes time and money on spot instances for no reason.

In contrast, `launch_remote.py`'s Phase 1 correctly skips quantization if the FP8 model exists:

> ```python
> # scripts/launch_remote.py lines 405-407
> elif fp8_path.exists():
>     size_gb = fp8_path.stat().st_size / (1024**3)
>     print(f"  FP8 model already exists ({size_gb:.2f} GB)")
> ```

### 3. Hardcoded paths and assumptions about remote filesystem (Major)

`remote_target.json.example` assumes `~/futudiffu` as the remote directory:

> ```json
> {
>     "remote_dir": "~/futudiffu",
>     "model_dir": "~/futudiffu/models"
> }
> ```

The `~` expansion happens on the remote side (via SSH), which works for typical Linux VMs. However:

- `remote_uv_bootstrap.py` uses `Path(__file__).resolve().parent.parent` for REPO_ROOT. This assumes the script is located at `{repo}/scripts/remote_uv_bootstrap.py`. If the repo is cloned to a non-standard location or if the script is symlinked, this breaks.
- `remote_node_bootstrap.py` line 95 inserts a hardcoded relative path: `sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))`. This inserts `{scripts_dir}/src`, not `{repo_root}/src`, because the Path resolution goes from the script's parent (scripts/) not the repo root.

The `cmd_bootstrap` function in `remote.py` (lines 320-351) runs the bootstrap script on the remote with the remote Python:

> ```python
> result = _ssh(
>     config,
>     f"cd {config.remote_dir} && {python} scripts/remote_uv_bootstrap.py",
>     timeout=600,
> )
> ```

The 600-second timeout may be too short for a fresh `uv sync` on a cold VM that needs to download and compile packages (especially Triton, which can take 5+ minutes to build from source).

### 4. uv installation via pip is fragile (Major)

`remote_uv_bootstrap.py` installs uv via pip if not found:

> ```python
> def install_uv() -> str:
>     uv_path = shutil.which("uv")
>     if uv_path:
>         return uv_path
>     print("  uv not found, installing via pip...")
>     _run([sys.executable, "-m", "pip", "install", "uv"], "pip install uv")
> ```

Problems:
- Uses `sys.executable` which is whatever Python was used to invoke the script. On a cloud VM, this might be system Python (3.8, 3.9) which may not support the uv version needed.
- `pip install uv` may install to a user site-packages not on PATH, which the script handles by checking `~/.local/bin/uv`. But if pip itself is not installed or is too old, this fails.
- The official uv installation method is `curl -LsSf https://astral.sh/uv/install.sh | sh`, which handles all platform quirks. The pip method is a less reliable fallback.
- No version pinning for uv -- different versions may produce different lockfile behavior.

### 5. ZMQ still in critical imports (Major)

`remote_uv_bootstrap.py` validates `zmq` as a critical import (line 76):

> ```python
> critical = ["torch", "triton", "safetensors", "zmq", "pyarrow"]
> ```

And `launch_remote.py` also checks for `zmq` (line 276):

> ```python
> for mod in ["safetensors", "zmq", "triton", "pyarrow"]:
> ```

But ZMQ is documented as dead architecture in CLAUDE.md. The new FastAPI server does not use ZMQ. Including `zmq` in the critical imports means the bootstrap can fail on a machine where `zmq` fails to compile (which is common on minimal cloud VMs without `libzmq-dev`), even though `zmq` is not actually needed for the current codebase.

### 6. No CUDA driver version verification (Minor)

`remote_uv_bootstrap.py`'s `print_cuda_info()` prints GPU info but does not verify compatibility:

> ```python
> def print_cuda_info() -> None:
>     import torch
>     if not torch.cuda.is_available():
>         print("  CUDA: not available")
>         return
>     # ... prints device info ...
> ```

There is no check that the CUDA toolkit version matches what PyTorch was compiled against, or that the driver version supports the CUDA runtime version. A mismatch here causes subtle failures during Triton kernel compilation that are diagnosed only when the first kernel is JIT-compiled.

### 7. Model download lacks integrity verification (Minor)

`remote_node_bootstrap.py`'s `download_models()` uses `hf_hub_download` which handles HTTP-level retries and cache management. However, there is no checksum verification after download:

> ```python
> downloaded = hf_hub_download(
>     repo_id=HF_REPO, filename=hf_path,
>     revision=HF_REVISION, local_dir=str(output_dir),
>     local_dir_use_symlinks=False,
> )
> ```

HuggingFace Hub does verify ETags during download, so this is partially mitigated by the library. But the `skip-if-present` logic (line 50-53) only checks file existence, not file integrity:

> ```python
> if dest.exists():
>     size_gb = dest.stat().st_size / (1024 ** 3)
>     print(f"  [skip] {local_name} already exists ({size_gb:.2f} GB)")
> ```

A truncated or corrupted file will be skipped, causing mysterious failures later during model loading.

### 8. rsync exclude list duplicated and not in sync (Minor)

`RSYNC_PUSH_EXCLUDE` in `remote.py` is a 22-element list of glob patterns for files to exclude during provisioning. This list is defined in `remote.py` only and is not shared with any other code. The case study documents (Part 4) that the exclude list was insufficient:

> ```
> remote.py provision rsynced the entire repository to the remote server, including:
> - bench_renders/ (79MB of local SM89 renders)
> - btrm_data/ (48MB of local trajectory data)
> ...
> None of these were needed on the remote.
> ```

The exclude list has been expanded since the case study (now includes more patterns), but the fundamental design of "exclude what we don't want" rather than "include only what we need" means every new directory or file type added to the repo must be manually excluded.

---

## Summary Table

| # | Issue | Severity |
|---|-------|----------|
| 1 | No system-level dependency verification (tmux, etc.) | Critical |
| 2 | Bootstrap not fully idempotent (re-quantizes, chdir) | Major |
| 3 | Hardcoded paths and remote filesystem assumptions | Major |
| 4 | uv installation via pip is fragile | Major |
| 5 | ZMQ still listed as critical import (dead architecture) | Major |
| 6 | No CUDA driver version verification | Minor |
| 7 | Model download lacks integrity verification | Minor |
| 8 | rsync exclude list not an include whitelist | Minor |
